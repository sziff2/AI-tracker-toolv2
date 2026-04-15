"""
Dump key PostgreSQL tables to a dated JSON file under DB Backup/.

Usage (one-off):
    DATABASE_URL="postgresql+asyncpg://user:pass@host:port/dbname" \
        python scripts/backup_db.py

Usage (automated, Windows Task Scheduler daily):
    create a .bat that sets DATABASE_URL and runs this script — see
    end of file for an example.

Output:
    DB Backup/db_backup_YYYY-MM-DD.json

Matches the format of earlier manual backups (one top-level key per
table, each mapped to a list of row dicts). Values are JSON-serialised
with UUIDs → str and datetimes → ISO 8601 strings.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Tables to back up — matches the structure of the 2026-04-11 reference file.
# Add new tables here as the schema grows.
TABLES: list[str] = [
    "companies",
    "document_sections",
    "documents",
    "extracted_metrics",
    "extraction_profiles",
    "harvested_documents",
    "harvester_sources",
    "kpi_actuals",
    "llm_usage_log",
    "portfolio_holdings",
    "portfolios",
    "price_records",
    "processing_jobs",
    "research_outputs",
    "review_queue",
    "scenario_snapshots",
    "valuation_scenarios",
]


def _json_default(obj):
    if isinstance(obj, (UUID,)):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


async def dump_table(conn, table_name: str) -> list[dict]:
    result = await conn.execute(text(f'SELECT * FROM "{table_name}"'))
    rows = result.mappings().all()
    return [dict(row) for row in rows]


async def run() -> Path:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL env var not set", file=sys.stderr)
        sys.exit(1)

    # SQLAlchemy async needs the asyncpg driver
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    backup_dir = Path(__file__).parent.parent / "DB Backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    out_path = backup_dir / f"db_backup_{today}.json"

    engine = create_async_engine(database_url, pool_pre_ping=True)
    dump: dict[str, list[dict]] = {}

    try:
        async with engine.connect() as conn:
            for table in TABLES:
                try:
                    rows = await dump_table(conn, table)
                    dump[table] = rows
                    print(f"  {table}: {len(rows)} rows")
                except Exception as exc:
                    print(f"  {table}: SKIPPED ({exc.__class__.__name__}: {exc})", file=sys.stderr)
                    dump[table] = []
    finally:
        await engine.dispose()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, default=_json_default, ensure_ascii=False)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")
    return out_path


if __name__ == "__main__":
    asyncio.run(run())


# ─── Automation examples ────────────────────────────────────────────────
#
# Windows Task Scheduler:
#   1. Create a new Basic Task, "AI Tracker DB Backup", trigger Daily
#   2. Action: Start a Program
#        Program: C:\path\to\python.exe
#        Arguments: scripts\backup_db.py
#        Start in: C:\Users\sam\OneDrive - Oldfield Partners\Reading\AI agent\AI-tracker-toolv2
#   3. Set DATABASE_URL as a user env var (System Properties → Environment
#      Variables) so the scheduled task inherits it.
#
# Cron (WSL / macOS / Linux):
#   0 7 * * * cd /path/to/AI-tracker-toolv2 && \
#             DATABASE_URL="..." /usr/bin/python3 scripts/backup_db.py \
#             >> "DB Backup/backup.log" 2>&1
