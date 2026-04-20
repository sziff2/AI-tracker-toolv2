"""
Ad-hoc inspector — dumps everything we have on one company.

Usage (Windows, against production Railway PG):
  DATABASE_URL="<prod_async_url>" python scripts/inspect_company.py NWG LN

Prints (in order):
  1. Matching companies (fuzzy — ticker and name)
  2. Documents ingested (doc_type, period, parsing_status, metric count)
  3. Harvester source config
  4. Recent harvested candidates
  5. Document Triage decisions (last 30)
  6. Coverage Monitor gap check for this ticker

All read-only. Safe to run against production. Closes the session
cleanly on exit.
"""

import asyncio
import sys
from datetime import datetime


async def _main(search_term: str) -> None:
    from sqlalchemy import select, text, desc
    from apps.api.database import AsyncSessionLocal
    from apps.api.models import (
        Company, Document, HarvesterSource,
        HarvestedDocument, IngestionTriage,
    )

    async with AsyncSessionLocal() as db:
        # ── 1. Find matching companies ───────────────────────────
        pattern = f"%{search_term}%"
        q = await db.execute(text("""
            SELECT id, ticker, name, sector, country, coverage_status
            FROM companies
            WHERE name ILIKE :p OR ticker ILIKE :p
            ORDER BY ticker
        """), {"p": pattern})
        matches = q.all()

        print(f"\n{'='*70}")
        print(f"Companies matching '{search_term}' — {len(matches)} hit(s)")
        print("="*70)
        for m in matches:
            print(f"  {m.ticker:<12} | {m.name:<45} | "
                  f"{m.sector or '-':<20} | {m.country or '-':<4} | "
                  f"coverage={m.coverage_status}")
        if not matches:
            print("  (no matches — try a different search term)")
            return

        for m in matches:
            cid = m.id
            print(f"\n\n{'#'*70}")
            print(f"# {m.ticker} — {m.name}")
            print("#"*70)

            # ── 2. Documents ingested ────────────────────────────
            q = await db.execute(text("""
                SELECT
                  d.document_type, d.period_label, d.parsing_status,
                  d.published_at, d.source,
                  LEFT(d.title, 80) AS title,
                  (SELECT COUNT(*) FROM extracted_metrics em
                   WHERE em.document_id = d.id) AS metric_count,
                  (SELECT COUNT(*) FROM document_sections ds
                   WHERE ds.document_id = d.id) AS section_count
                FROM documents d
                WHERE d.company_id = :cid
                ORDER BY d.published_at DESC NULLS LAST, d.period_label DESC
                LIMIT 50
            """), {"cid": cid})
            docs = q.all()
            print(f"\n── Documents ingested ({len(docs)}):")
            if not docs:
                print("  (none)")
            else:
                for d in docs:
                    pub = d.published_at.date().isoformat() if d.published_at else "-"
                    print(f"  {pub:<10} | {d.period_label or '-':<10} | "
                          f"{d.document_type or '-':<18} | "
                          f"{d.parsing_status or '-':<10} | "
                          f"metrics={d.metric_count:>4} sections={d.section_count:>3} | "
                          f"{d.source or '-':<12} | {d.title or ''}")

            # ── 3. Harvester source config ───────────────────────
            q = await db.execute(text("""
                SELECT ir_docs_url, ir_url, override, last_checked_at, notes
                FROM harvester_sources
                WHERE company_id = :cid
            """), {"cid": cid})
            src = q.one_or_none()
            print("\n── Harvester source config:")
            if not src:
                print("  (no HarvesterSource row — nothing configured)")
            else:
                print(f"  ir_docs_url:     {src.ir_docs_url or '(none)'}")
                print(f"  ir_url:          {src.ir_url or '(none)'}")
                print(f"  override (lock): {src.override}")
                print(f"  last_checked_at: {src.last_checked_at}")
                print(f"  notes:           {(src.notes or '').strip()}")

            # ── 4. Recent harvested candidates ───────────────────
            q = await db.execute(text("""
                SELECT
                  source, source_url, LEFT(headline, 90) AS headline,
                  period_label, discovered_at, ingested, error
                FROM harvested_documents
                WHERE company_id = :cid
                ORDER BY discovered_at DESC
                LIMIT 20
            """), {"cid": cid})
            hd = q.all()
            print(f"\n── Recent harvested candidates ({len(hd)}):")
            if not hd:
                print("  (none)")
            else:
                for h in hd:
                    dt = h.discovered_at.date().isoformat() if h.discovered_at else "-"
                    ing = "✓" if h.ingested else "✗"
                    err = f" err={h.error[:80]}" if h.error else ""
                    print(f"  {dt:<10} | {ing} | {h.source or '-':<14} | "
                          f"{h.period_label or '-':<10} | {h.headline or ''}{err}")

            # ── 5. Triage decisions ──────────────────────────────
            q = await db.execute(text("""
                SELECT
                  document_type, period_label, priority, relevance_score,
                  auto_ingest, needs_review, was_ingested,
                  LEFT(rationale, 200) AS rationale, created_at
                FROM ingestion_triage
                WHERE company_id = :cid
                ORDER BY created_at DESC
                LIMIT 30
            """), {"cid": cid})
            tri = q.all()
            print(f"\n── Document Triage decisions ({len(tri)}):")
            if not tri:
                print("  (no triage rows — company may be new or pre-Sprint1)")
            else:
                for t in tri:
                    dt = t.created_at.date().isoformat() if t.created_at else "-"
                    flags = []
                    if t.auto_ingest: flags.append("auto")
                    if t.needs_review: flags.append("review")
                    if t.was_ingested: flags.append("ingested")
                    tag = ",".join(flags)
                    print(f"  {dt:<10} | {t.priority or '-':<9} | score={t.relevance_score:>3} | "
                          f"{t.document_type or '-':<15} | {t.period_label or '-':<10} | "
                          f"[{tag}] | {t.rationale or ''}")

            # ── 6. Coverage Monitor check ────────────────────────
            print("\n── Coverage Monitor (live snapshot):")
            try:
                from services.harvester.coverage_advanced import (
                    analyze_company_cadence, find_overdue_gaps,
                )
                cadence = await analyze_company_cadence(db, cid, m.ticker)
                print(f"  Cadence: frequency={cadence.frequency}  "
                      f"report_lag={cadence.report_lag_days}d  "
                      f"transcript_lag={cadence.transcript_lag_days}  "
                      f"sample_size={cadence.sample_size}  "
                      f"last_earnings_period={cadence.last_earnings_period}  "
                      f"last_earnings_release={cadence.last_earnings_release}")
                print(f"  Doc types observed: {sorted(cadence.doc_types_observed)}")

                # Gap check across all active companies, then filter to this one
                gaps = await find_overdue_gaps(db)
                my_gaps = [g for g in gaps if g.ticker == m.ticker]
                print(f"  Current gaps for {m.ticker}: {len(my_gaps)}")
                for g in my_gaps:
                    print(f"    [{g.severity:<14}] {g.doc_type:<18} "
                          f"expected {g.expected_period} by {g.expected_by} "
                          f"({g.days_overdue}d overdue) — {g.reason[:80]}")
            except Exception as exc:
                print(f"  (coverage check failed: {exc})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_company.py <ticker-or-name-substring>")
        print("Example: python scripts/inspect_company.py NWG")
        print("         python scripts/inspect_company.py \"North West\"")
        sys.exit(1)
    search = " ".join(sys.argv[1:])
    asyncio.run(_main(search))


if __name__ == "__main__":
    main()
