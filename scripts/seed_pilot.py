"""
Seed script — sets up the pilot company (Heineken / HEIA) and an initial thesis.

Usage:
    python scripts/seed_pilot.py
"""

import asyncio
import uuid
from datetime import date, datetime, timezone

from apps.api.database import AsyncSessionLocal, async_engine, Base
from apps.api.models import Company, ThesisVersion


async def seed():
    # Create tables if they don't exist
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        # ── Company ──────────────────────────────────────────────
        company = Company(
            id=uuid.uuid4(),
            ticker="HEIA",
            name="Heineken N.V.",
            sector="Consumer Staples",
            industry="Beverages — Brewers",
            country="Netherlands",
            coverage_status="active",
            primary_analyst="Research Team",
        )
        db.add(company)

        # ── Initial Thesis ───────────────────────────────────────
        thesis = ThesisVersion(
            id=uuid.uuid4(),
            company_id=company.id,
            thesis_date=date(2025, 12, 1),
            core_thesis=(
                "Heineken is a premium global brewer with pricing power and "
                "volume recovery tailwinds in emerging markets. The EverGreen "
                "strategy should drive margin expansion through premiumisation "
                "and cost discipline. The stock is undervalued relative to "
                "the sector on a FCF yield basis."
            ),
            variant_perception=(
                "The market underestimates the margin improvement trajectory "
                "in Africa and Asia-Pacific, and overweights near-term input "
                "cost headwinds."
            ),
            key_risks=(
                "1. Prolonged consumer downtrading in Europe\n"
                "2. FX headwinds in Africa\n"
                "3. Regulatory risk (alcohol taxation)\n"
                "4. Integration risk in recent acquisitions"
            ),
            debate_points=(
                "1. Sustainability of premium mix shift\n"
                "2. Africa volume growth vs currency translation\n"
                "3. Capital allocation: M&A vs buybacks\n"
                "4. Input cost outlook (barley, aluminium, energy)"
            ),
            capital_allocation_view=(
                "Preference for organic growth reinvestment and selective bolt-on "
                "acquisitions. Dividend payout ~30-35% of net profit. "
                "Share buybacks only if FCF materially exceeds investment needs."
            ),
            valuation_framework=(
                "DCF with 7.5% WACC, 2% terminal growth. Cross-check with "
                "EV/EBITDA (target 10-11x forward) and FCF yield (target 6%+)."
            ),
            active=True,
        )
        db.add(thesis)

        await db.commit()
        print(f"✓ Seeded company: {company.ticker} ({company.name})")
        print(f"✓ Seeded thesis dated {thesis.thesis_date}")


if __name__ == "__main__":
    asyncio.run(seed())
