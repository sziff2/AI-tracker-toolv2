"""
Backfill document titles + doc_types for rows ingested before the IR
scraper fixes landed (commit d5c477c, 2026-04-20).

Target: Documents where `source = 'ir_scrape'` whose title is a generic
CMS anchor ("Open", "Download", "View", ...) OR whose document_type
disagrees with what the new classifier would say from the filename.

Dry-run by default — prints the proposed changes. Pass --apply to
commit the updates.

Safe by construction:
  - Only touches source='ir_scrape' rows (skips manual uploads, EDGAR,
    Investegate, etc.)
  - Never changes period_label, file_path, checksum, company_id, source
  - Never deletes
  - Reports every proposed change line by line so you can eyeball before
    committing

Usage (Windows, against production Railway PG):
  python scripts/backfill_document_titles.py                     # all tickers, dry-run
  python scripts/backfill_document_titles.py --ticker "NWC CN"   # one ticker, dry-run
  python scripts/backfill_document_titles.py --ticker "NWC CN" --apply   # commit
"""

import argparse
import asyncio
import sys
from urllib.parse import urlparse, unquote
import re


def _slug_from_url(url: str) -> str:
    """Extract a human-readable filename slug from a URL. Strips the
    .pdf, url-decodes, replaces separator punctuation with spaces."""
    if not url:
        return ""
    try:
        path = urlparse(url).path
    except Exception:
        return ""
    raw = path.split("/")[-1]
    raw = unquote(raw)
    raw = re.sub(r"\.pdf$", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("-", " ").replace("_", " ")
    return raw.strip()


async def _main(args) -> int:
    from sqlalchemy import select, update
    from apps.api.database import AsyncSessionLocal
    from apps.api.models import Document, Company
    from services.harvester.sources.ir_scraper import (
        _GENERIC_ANCHOR_TEXTS, _classify_doc_type, _fix_mojibake,
    )

    proposed = 0
    title_only = 0
    type_only = 0
    both = 0
    unchanged = 0

    async with AsyncSessionLocal() as db:
        q = select(Document, Company.ticker).join(
            Company, Company.id == Document.company_id,
        ).where(Document.source == "ir_scrape")
        if args.ticker:
            q = q.where(Company.ticker == args.ticker.upper())
        q = q.order_by(Company.ticker, Document.period_label)

        result = await db.execute(q)
        rows = result.all()

        print(f"\nInspecting {len(rows)} ir_scrape documents"
              + (f" for {args.ticker.upper()}" if args.ticker else "")
              + " ...\n")

        updates: list[tuple] = []  # (doc_id, new_title, new_doc_type)

        for doc, ticker in rows:
            old_title = (doc.title or "").strip()
            old_type = doc.document_type or ""

            slug = _slug_from_url(doc.source_url or "")
            slug = _fix_mojibake(slug)
            url_path = ""
            try:
                url_path = urlparse(doc.source_url or "").path
            except Exception:
                pass

            # Propose a new title when the current one is a generic CMS
            # anchor OR when it contains mojibake. Otherwise keep.
            title_is_generic = old_title.lower() in _GENERIC_ANCHOR_TEXTS
            has_mojibake = any(x in old_title for x in (
                "\u00e2\u0080", "\u00c3\u00a9", "\u00c3\u00a8",
            ))
            new_title = old_title
            if (title_is_generic or has_mojibake) and slug:
                new_title = slug.title()

            # Re-classify using the new narrower function
            new_type = _classify_doc_type(
                slug=slug.lower(),
                headline=new_title,
                url_path=url_path,
            )

            title_changed = new_title != old_title
            # Only change doc_type if the new value is more specific.
            # Don't downgrade away from a useful classification TO 'other'.
            type_changed = (
                new_type != old_type
                and new_type != "other"
                and old_type in ("presentation", "other", "earnings_release", "")
            )

            if title_changed or type_changed:
                proposed += 1
                if title_changed and type_changed:
                    both += 1
                elif title_changed:
                    title_only += 1
                else:
                    type_only += 1
                updates.append((doc.id, new_title, new_type if type_changed else old_type))
                # Print the proposed change
                def _show(label, old, new):
                    if old != new:
                        return f"    {label:<12} {old!r:<40} → {new!r}"
                    return f"    {label:<12} {old!r} (unchanged)"
                print(f"  {ticker} | {doc.period_label:<12} | doc_id={str(doc.id)[:8]}")
                print(_show("title",    old_title, new_title))
                print(_show("doc_type", old_type,  new_type if type_changed else old_type))
                print(f"    source_url: {doc.source_url}")
                print()
            else:
                unchanged += 1

        print("="*70)
        print(f"Summary: {proposed} proposed changes / {len(rows)} inspected")
        print(f"  title only:      {title_only}")
        print(f"  doc_type only:   {type_only}")
        print(f"  both:            {both}")
        print(f"  unchanged:       {unchanged}")
        print("="*70)

        if not args.apply:
            print("\nDRY-RUN — no database changes made.")
            print("Re-run with --apply to commit these updates.")
            return 0

        if not updates:
            print("\nNothing to update.")
            return 0

        print(f"\nApplying {len(updates)} updates ...")
        for doc_id, new_title, new_type in updates:
            await db.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(title=new_title, document_type=new_type)
            )
        await db.commit()
        print(f"Committed {len(updates)} updates.")
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker",
        help="Restrict to one ticker (e.g. 'NWC CN'). Default: all.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit changes. Without this flag runs as dry-run.",
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
