"""
Sector-specific KPI dictionaries and context injection.

Maps sector/industry to the KPIs that matter most for extraction.
These are injected into extraction prompts to ensure the LLM prioritises
sector-relevant metrics. Sourced from the subsector agent plan.

Usage:
    from services.sector_kpi_config import get_sector_context, get_sector_kpis

    context = get_sector_context("Financials", "Banks", "United Kingdom")
    # Returns a string block to inject into extraction prompts
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SUBSECTOR KPI DICTIONARIES
# Aligned with the agent plan's subsector agent focus areas
# ═══════════════════════════════════════════════════════════════════

SUBSECTOR_KPIS: dict[str, dict] = {
    # ── Financials ────────────────────────────────────────────────
    "banks": {
        "display_name": "Banks",
        "match_patterns": [r"bank", r"banking", r"lend"],
        "priority_kpis": [
            "NII (Net Interest Income)", "NIM (Net Interest Margin)", "Loan Growth",
            "Deposit Beta", "Cost of Risk", "NPL Ratio", "CET1 Ratio",
            "RWA Density", "Fee Income Mix", "Cost/Income Ratio",
            "Provision Charge", "Loan-to-Deposit Ratio", "ROTE",
            "Stage 3 Loans", "Coverage Ratio",
        ],
        "reporting_standards": "IFRS 9 / CECL impairment, Basel III/IV capital",
        "notes": "Distinguish between reported and underlying NIM. Watch for RWA inflation from Basel IV.",
        "kpi_validation": {
            # Ranges expressed in the metric's native extracted unit.
            # Bank NIM is almost always reported as a percentage (e.g.
            # 3.45 not 0.0345). Keep bounds generous to accommodate
            # digital-only banks (~1.8%) through to credit-card-heavy
            # specialists (~8%).
            "nim": {"reasonable_range": (1.5, 8.0), "unit": "%",
                    "valid_denominators": ["average_earning_assets", "average_interest_earning_assets"]},
            "cet1 ratio": {"reasonable_range": (4.0, 25.0), "unit": "%",
                           "valid_denominators": ["risk_weighted_assets"]},
            "charge_off_rate": {"reasonable_range": (0.0, 5.0), "unit": "%",
                                "valid_denominators": ["average_loans", "average_loans_outstanding", "average_loans_and_leases"],
                                "invalid_denominators": ["revenue", "total_revenue", "total_assets", "deposits", "net_income"]},
            "npl ratio": {"reasonable_range": (0.0, 15.0), "unit": "%",
                          "valid_denominators": ["total_loans", "gross_loans"]},
            "cost/income ratio": {"reasonable_range": (30.0, 80.0), "unit": "%"},
            "rote": {"reasonable_range": (0.0, 30.0), "unit": "%"},
            "loan-to-deposit ratio": {"reasonable_range": (50.0, 140.0), "unit": "%"},
            "coverage ratio": {"reasonable_range": (20.0, 200.0), "unit": "%"},
        },
    },
    "insurance": {
        "display_name": "Insurance",
        "match_patterns": [r"insurance", r"insur", r"underwrite", r"reinsur"],
        "priority_kpis": [
            "Combined Ratio", "Loss Ratio", "Expense Ratio",
            "Investment Income Yield", "Solvency Ratio (SCR Coverage)",
            "Reserve Development", "Cat Losses", "GWP (Gross Written Premiums)",
            "Net Earned Premiums", "Underwriting Result",
            "Embedded Value", "New Business Margin", "VNB (Value of New Business)",
        ],
        "reporting_standards": "Solvency II, IFRS 17",
        "notes": "Separate life vs P&C. Prior-year reserve releases distort combined ratio.",
        "kpi_validation": {
            "combined ratio": {"reasonable_range": (70.0, 120.0), "unit": "%",
                               "notes": ">100 means underwriting loss"},
            "loss ratio": {"reasonable_range": (40.0, 90.0), "unit": "%"},
            "expense ratio": {"reasonable_range": (15.0, 45.0), "unit": "%"},
            "solvency ratio": {"reasonable_range": (120.0, 300.0), "unit": "%",
                               "notes": "Regulator-defined; below 100 is critical"},
        },
    },
    "asset_management": {
        "display_name": "Asset Management",
        "match_patterns": [r"asset.?manage", r"fund.?manage", r"investment.?manage"],
        "priority_kpis": [
            "AUM", "Net New Money (Flows)", "Management Fee Margin",
            "Performance Fees", "Cost/Income Ratio",
            "Investment Performance vs Benchmark", "Organic Growth Rate",
            "AUM Mix (equity/fixed income/alternatives)",
        ],
        "reporting_standards": "IFRS, fee revenue recognition",
        "notes": "Distinguish between market effect and net flows on AUM changes.",
    },
    "payments_fintech": {
        "display_name": "Payments / Fintech",
        "match_patterns": [r"payment", r"fintech", r"merchant.?acquir"],
        "priority_kpis": [
            "TPV (Total Payment Volume)", "Take Rate", "Transactions Processed",
            "Merchant Count", "Cross-Border Mix", "BNPL Delinquency Rates",
            "Revenue per Transaction", "Active Users/Accounts",
        ],
        "reporting_standards": "IFRS / US GAAP",
        "notes": "Watch for FX impact on cross-border volumes.",
    },
    "reits": {
        "display_name": "REITs / Real Estate",
        "match_patterns": [r"reit", r"real.?estate", r"property.?trust"],
        "priority_kpis": [
            "FFO (Funds From Operations)", "AFFO", "NAV",
            "Occupancy Rate", "Like-for-Like Rental Growth",
            "WAULT (Weighted Average Unexpired Lease Term)",
            "Development Pipeline", "LTV (Loan-to-Value)", "Cost of Debt",
            "Reversionary Yield", "ERV (Estimated Rental Value)",
        ],
        "reporting_standards": "EPRA metrics, IFRS fair value",
        "notes": "NAV vs share price discount/premium is key. Distinguish EPRA NTA from IFRS NAV.",
        "kpi_validation": {
            "occupancy rate": {"reasonable_range": (70.0, 100.0), "unit": "%"},
            "ltv": {"reasonable_range": (10.0, 70.0), "unit": "%"},
            "wault": {"reasonable_range": (2.0, 20.0), "unit": "years"},
            "cost of debt": {"reasonable_range": (1.0, 10.0), "unit": "%"},
        },
    },

    # ── Consumer ──────────────────────────────────────────────────
    "beverages": {
        "display_name": "Beverages",
        "match_patterns": [r"beverage", r"brew", r"beer", r"spirits", r"wine", r"soft.?drink"],
        "priority_kpis": [
            "Organic Revenue Growth", "Volume Growth", "Price/Mix",
            "EBITDA Margin", "Operating Margin Expansion (bps)",
            "Market Share", "Premiumisation Mix", "Input Cost Inflation",
            "A&P Spend (% of Revenue)", "FCF Conversion",
            "Net Debt/EBITDA", "DPS Growth",
        ],
        "reporting_standards": "IFRS, organic growth definitions vary by company",
        "notes": "Separate volume vs price/mix. Watch FX translation in EM-heavy brewers.",
    },
    "food_staples": {
        "display_name": "Food & Staples",
        "match_patterns": [r"food", r"staple", r"consumer.?staple", r"fmcg", r"cpg", r"household", r"personal.?care"],
        "priority_kpis": [
            "Organic Revenue Growth", "Volume Growth", "Price/Mix",
            "Gross Margin", "EBITDA Margin", "Market Share",
            "Private Label Threat", "Input Cost Basket",
            "Innovation Pipeline (% of Revenue from New Products)",
            "Trade Spend", "Distribution Gains/Losses",
        ],
        "reporting_standards": "IFRS / US GAAP, organic growth ex-M&A and FX",
        "notes": "Volume vs price decomposition critical. Watch for trade-down signals.",
    },
    "tobacco": {
        "display_name": "Tobacco",
        "match_patterns": [r"tobacco", r"cigarette", r"nicotine", r"smoke.?free"],
        "priority_kpis": [
            "Combustible Volume Decline Rate", "Pricing Power (net of tax)",
            "Smoke-Free Product Revenue", "IQOS/HnB Users",
            "Operating Margin", "FCF Yield", "Dividend Cover",
            "Excise Tax Trajectory", "Illicit Trade Share",
        ],
        "reporting_standards": "IFRS / US GAAP, excise accounting varies",
        "notes": "Net revenue (ex-excise) is the relevant topline. RRP adoption curves vary by market.",
    },
    "retail": {
        "display_name": "Retail",
        "match_patterns": [r"retail", r"grocer", r"supermarket", r"store", r"e.?commerce"],
        "priority_kpis": [
            "Like-for-Like Sales Growth", "Basket Size", "Footfall/Traffic",
            "Online Penetration", "Gross Margin", "Operating Margin",
            "Space Growth (sq ft)", "Sales per Square Foot",
            "Inventory Turnover", "Working Capital Days",
            "Store Count (opens/closes)", "Customer Count",
        ],
        "reporting_standards": "IFRS / US GAAP",
        "notes": "Calendar shift effects. Distinguish food vs non-food LFL. Watch Clubcard/loyalty data.",
        "kpi_validation": {
            "like-for-like sales growth": {"reasonable_range": (-20.0, 30.0), "unit": "%"},
            "gross margin": {"reasonable_range": (15.0, 55.0), "unit": "%"},
            "operating margin": {"reasonable_range": (-5.0, 15.0), "unit": "%"},
        },
    },
    "luxury_apparel": {
        "display_name": "Luxury / Apparel",
        "match_patterns": [r"luxury", r"apparel", r"fashion", r"watch", r"jewel"],
        "priority_kpis": [
            "Organic Revenue Growth (by region)", "Comparable Store Sales",
            "Gross Margin", "EBIT Margin", "ASP (Average Selling Price)",
            "DTC vs Wholesale Mix", "China/Asia Exposure",
            "Brand Heat Indicators", "Inventory Freshness",
        ],
        "reporting_standards": "IFRS, constant currency growth",
        "notes": "Geographic mix drives margin. Watch China tourist spending channel shifts.",
    },

    # ── Industrials ───────────────────────────────────────────────
    "capital_goods": {
        "display_name": "Capital Goods / Distribution",
        "match_patterns": [r"capital.?good", r"distribut", r"industrial.*service", r"conglomerate"],
        "priority_kpis": [
            "Organic Revenue Growth", "Operating Margin",
            "Order Intake / Book-to-Bill", "Backlog",
            "Aftermarket Revenue %", "Acquisition Revenue Contribution",
            "ROIC", "FCF Conversion", "Net Debt/EBITDA",
            "Bolt-on M&A Spend", "Working Capital/Revenue",
        ],
        "reporting_standards": "IFRS / US GAAP",
        "notes": "Distinguish organic vs M&A growth. Aftermarket = recurring revenue quality indicator.",
    },
    "aerospace_defence": {
        "display_name": "Aerospace & Defence",
        "match_patterns": [r"aerospace", r"defence", r"defense", r"aircraft", r"missile", r"avionics"],
        "priority_kpis": [
            "Order Intake", "Backlog / Book-to-Bill",
            "Revenue by Civil vs Defence", "EBIT Margin",
            "R&D as % of Revenue", "Program Milestones",
            "Aftermarket / MRO Revenue", "Cash Conversion",
        ],
        "reporting_standards": "IFRS 15 / ASC 606 over-time recognition",
        "notes": "Long-cycle contracts. Watch for EAC (Estimate at Completion) adjustments.",
    },
    "transport_logistics": {
        "display_name": "Transport & Logistics",
        "match_patterns": [r"transport", r"logistic", r"airline", r"shipping", r"freight", r"travel"],
        "priority_kpis": [
            "RASK (Revenue per ASK)", "CASK (Cost per ASK ex-fuel)",
            "Load Factor", "Yield", "Revenue Passenger Kilometres",
            "Ancillary Revenue per Pax", "Fuel Cost per Litre",
            "RevPAR (hotels)", "Occupancy Rate (hotels)",
            "Like-for-Like Sales (pubs/restaurants)",
        ],
        "reporting_standards": "IFRS / US GAAP, IFRS 16 lease capitalisation",
        "notes": "Airlines: fuel hedging book affects comparability. Hotels: RevPAR = occupancy × ADR.",
    },

    # ── Technology ────────────────────────────────────────────────
    "software": {
        "display_name": "Software / SaaS",
        "match_patterns": [r"software", r"saas", r"cloud.*platform", r"subscription.*software"],
        "priority_kpis": [
            "ARR (Annual Recurring Revenue)", "Net Revenue Retention",
            "Gross Margin", "Rule of 40 (growth + margin)",
            "RPO (Remaining Performance Obligations)",
            "CAC Payback Period", "LTV/CAC",
            "Churn Rate", "Free Cash Flow Margin",
            "Product Attach Rate",
        ],
        "reporting_standards": "US GAAP / IFRS, ASC 606 subscription revenue",
        "notes": "Separate subscription from licence + services. Watch for RPO duration shifts.",
        "kpi_validation": {
            "net revenue retention": {"reasonable_range": (90.0, 135.0), "unit": "%"},
            "gross margin": {"reasonable_range": (60.0, 90.0), "unit": "%"},
            "churn rate": {"reasonable_range": (0.0, 20.0), "unit": "%"},
            "free cash flow margin": {"reasonable_range": (-20.0, 50.0), "unit": "%"},
        },
    },
    "semiconductors": {
        "display_name": "Semiconductors",
        "match_patterns": [r"semiconductor", r"chip", r"wafer", r"fabless", r"foundry"],
        "priority_kpis": [
            "Wafer Starts", "ASP by Node",
            "Utilisation Rate", "Inventory Days (channel + own)",
            "Design Wins", "HBM / AI Accelerator Mix",
            "Gross Margin", "R&D as % of Revenue",
            "Capex Intensity", "Book-to-Bill",
        ],
        "reporting_standards": "US GAAP / IFRS / K-GAAP / J-GAAP",
        "notes": "Inventory correction cycles are the key risk. Distinguish leading-edge vs mature nodes.",
    },
    "it_services": {
        "display_name": "IT Services / Consulting",
        "match_patterns": [r"it.?service", r"consult", r"outsourc", r"system.?integrat"],
        "priority_kpis": [
            "Book-to-Bill", "TCV of Large Deals",
            "Attrition Rate", "Utilisation",
            "Offshore Mix", "Pricing Trends",
            "GenAI Cannibalisation Risk", "Revenue per Employee",
            "Digital Revenue %",
        ],
        "reporting_standards": "IFRS / US GAAP / Ind AS",
        "notes": "Watch for deal deferrals in macro slowdowns. Offshore mix drives margin.",
    },

    # ── Healthcare ────────────────────────────────────────────────
    "pharma": {
        "display_name": "Pharmaceuticals",
        "match_patterns": [r"pharma", r"drug", r"therapeutic", r"biotech"],
        "priority_kpis": [
            "Pipeline Readouts (Phase Transitions)", "LOE Exposure",
            "Pricing/Rebate Trends", "Geographic Mix",
            "Biosimilar Erosion Curves", "R&D Productivity (Phase I→Approval)",
            "Revenue by Franchise", "Core Operating Margin",
            "Peak Sales Estimates", "Patent Cliff Dates",
        ],
        "reporting_standards": "IFRS / US GAAP, core vs reported earnings distinction",
        "notes": "Exclude restructuring/impairment for core earnings. Pipeline probability-weight adjustments.",
        "kpi_validation": {
            "core operating margin": {"reasonable_range": (15.0, 50.0), "unit": "%"},
            "r&d as % of revenue": {"reasonable_range": (8.0, 30.0), "unit": "%"},
        },
    },
    "medtech": {
        "display_name": "Medical Devices / Medtech",
        "match_patterns": [r"med.?tech", r"medical.?device", r"diagnostic", r"surgical"],
        "priority_kpis": [
            "Organic Revenue Growth", "Procedure Volumes",
            "Installed Base Growth", "Consumables Attach Rate",
            "Gross Margin", "R&D as % of Revenue",
            "Regulatory Approvals (510k, PMA, CE Mark)",
            "Recurring Revenue %",
        ],
        "reporting_standards": "IFRS / US GAAP",
        "notes": "Procedure volumes recover post-COVID. Consumables = recurring revenue moat.",
    },

    # ── Energy ────────────────────────────────────────────────────
    "oil_gas": {
        "display_name": "Oil & Gas",
        "match_patterns": [r"oil", r"gas", r"petroleum", r"upstream", r"downstream", r"refin"],
        "priority_kpis": [
            "Production (boe/d)", "Realised Price",
            "Lifting Cost / Opex per boe", "Finding Cost",
            "Reserve Replacement Ratio", "Refining Margin",
            "Capex (growth vs maintenance)", "FCF Yield",
            "Net Debt/EBITDAX", "Shareholder Distributions",
            "Breakeven Oil Price",
        ],
        "reporting_standards": "IFRS / US GAAP, SEC reserve rules",
        "notes": "Distinguish organic vs inorganic production growth. Watch for impairment triggers.",
        "kpi_validation": {
            # Ranges cover all cycle regimes including stress
            "reserve replacement ratio": {"reasonable_range": (0.0, 300.0), "unit": "%"},
            "net debt/ebitdax": {"reasonable_range": (0.0, 6.0), "unit": "x"},
            "breakeven oil price": {"reasonable_range": (15.0, 90.0), "unit": "USD/bbl"},
        },
    },
    "oilfield_services": {
        "display_name": "Oilfield Services",
        "match_patterns": [r"oilfield", r"drilling", r"well.?service", r"rig.?count"],
        "priority_kpis": [
            "Revenue by Geography", "Rig Count Exposure",
            "Operating Margin by Segment", "Backlog",
            "Technology Adoption Rate", "Day Rates",
            "Market Share", "International vs North America Mix",
        ],
        "reporting_standards": "US GAAP / IFRS",
        "notes": "Rig count is leading indicator. International cycle lags NAM by 6-12 months.",
    },

    # ── Materials ─────────────────────────────────────────────────
    "mining_metals": {
        "display_name": "Mining & Metals",
        "match_patterns": [r"mining", r"metal", r"steel", r"iron.?ore", r"copper", r"alumin"],
        "priority_kpis": [
            "Production Volume (by commodity)", "Realised Price vs Benchmark",
            "C1 Cash Cost", "All-In Sustaining Cost (AISC)",
            "EBITDA Margin", "Capex (sustaining vs growth)",
            "Net Debt/EBITDA", "Reserve/Resource Base",
            "Grade Decline / Recovery Rate",
        ],
        "reporting_standards": "IFRS, JORC/NI 43-101 reserves",
        "notes": "Cost curve position is the moat. Watch for grade decline in mature assets.",
    },
    "chemicals_packaging": {
        "display_name": "Chemicals / Packaging",
        "match_patterns": [r"chemical", r"packaging", r"polymer", r"coating", r"adhesive", r"specialty.?chem"],
        "priority_kpis": [
            "Volume Growth", "Price/Raw Material Spread",
            "EBITDA Margin", "Capacity Utilisation",
            "Innovation Revenue (%)", "Sustainability Mix (%)",
            "Working Capital / Revenue",
        ],
        "reporting_standards": "IFRS / US GAAP",
        "notes": "Spread management is the key metric. Distinguish commodity vs specialty.",
    },

    # ── Telecoms & Media ──────────────────────────────────────────
    "telecoms": {
        "display_name": "Telecoms",
        "match_patterns": [r"telecom", r"mobile.*operator", r"broadband.*provider"],
        "priority_kpis": [
            "Service Revenue Growth", "ARPU",
            "Subscriber Net Adds/Churn", "EBITDA Margin (ex-leases)",
            "Capex/Revenue", "FTTH Penetration",
            "Convergence Rate", "Net Debt/EBITDA",
            "FCF after Leases", "Spectrum Costs",
        ],
        "reporting_standards": "IFRS 16, adjusted EBITDA definitions vary",
        "notes": "IFRS 16 lease treatment distorts EBITDA comparisons. Use EBITDAaL.",
    },
    "media_entertainment": {
        "display_name": "Media & Entertainment",
        "match_patterns": [r"media", r"entertainment", r"streaming", r"broadcast", r"content", r"advertising"],
        "priority_kpis": [
            "Subscribers / DAU / MAU", "ARPU",
            "Content Spend", "Ad Revenue Growth",
            "Engagement Time", "Churn Rate",
            "Operating Margin", "DTC Profitability",
        ],
        "reporting_standards": "US GAAP / IFRS",
        "notes": "Content amortisation policies vary wildly. Watch for sports rights inflation.",
    },
}


# ═══════════════════════════════════════════════════════════════════
# REPORTING STANDARD CONTEXT — injected based on country
# ═══════════════════════════════════════════════════════════════════

COUNTRY_CONTEXT: dict[str, dict] = {
    "United States": {
        "gaap": "US GAAP",
        "currency": "USD",
        "fiscal_year_note": "Calendar year typical. Retailers often end Jan/Feb.",
        "regulatory_filings": "10-K (annual), 10-Q (quarterly), 8-K (events)",
    },
    "United Kingdom": {
        "gaap": "IFRS (UK-adopted)",
        "currency": "GBP",
        "fiscal_year_note": "Varies widely. Retailers often end Feb/Mar.",
        "regulatory_filings": "Annual Report, Interim Results, RNS announcements",
    },
    "Netherlands": {
        "gaap": "IFRS (EU-adopted)",
        "currency": "EUR",
        "fiscal_year_note": "Calendar year typical.",
        "regulatory_filings": "Annual Report, Half-Year, AFM filings",
    },
    "Germany": {
        "gaap": "IFRS (EU-adopted)",
        "currency": "EUR",
        "fiscal_year_note": "Calendar year typical.",
        "regulatory_filings": "Annual Report, BaFin filings",
    },
    "France": {
        "gaap": "IFRS (EU-adopted)",
        "currency": "EUR",
        "fiscal_year_note": "Calendar year typical.",
        "regulatory_filings": "Document d'Enregistrement Universel, AMF filings",
    },
    "Japan": {
        "gaap": "J-GAAP or IFRS",
        "currency": "JPY",
        "fiscal_year_note": "Usually ends March 31. Some use December.",
        "regulatory_filings": "Yuho (annual), Tanshin (quarterly flash)",
    },
    "South Korea": {
        "gaap": "K-IFRS",
        "currency": "KRW",
        "fiscal_year_note": "Calendar year.",
        "regulatory_filings": "DART filings, Earnings Flash",
    },
    "Canada": {
        "gaap": "IFRS",
        "currency": "CAD",
        "fiscal_year_note": "Calendar year typical. Some retailers differ.",
        "regulatory_filings": "Annual Information Form, MD&A, SEDAR",
    },
    "Italy": {
        "gaap": "IFRS (EU-adopted)",
        "currency": "EUR",
        "fiscal_year_note": "Calendar year typical.",
        "regulatory_filings": "Annual Report, CONSOB filings",
    },
    "Switzerland": {
        "gaap": "IFRS or Swiss GAAP FER",
        "currency": "CHF",
        "fiscal_year_note": "Calendar year typical.",
        "regulatory_filings": "Annual Report, SIX filings",
    },
    "Sweden": {
        "gaap": "IFRS (EU-adopted)",
        "currency": "SEK",
        "fiscal_year_note": "Calendar year typical.",
        "regulatory_filings": "Annual Report, FI filings",
    },
    "Hong Kong": {
        "gaap": "HKFRS (converged with IFRS)",
        "currency": "HKD",
        "fiscal_year_note": "December typical. Some use March.",
        "regulatory_filings": "Annual Report, HKEx announcements",
    },
}


# ═══════════════════════════════════════════════════════════════════
# LOOKUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _match_subsector(sector: str, industry: str = "") -> Optional[dict]:
    """
    Find the best-matching subsector config.
    Tries industry first (more specific), then sector.
    """
    combined = f"{sector} {industry}".lower()

    best_match = None
    best_score = 0

    for key, config in SUBSECTOR_KPIS.items():
        for pattern in config["match_patterns"]:
            if re.search(pattern, combined, re.IGNORECASE):
                # Prefer matches in the industry field (more specific)
                industry_match = re.search(pattern, industry.lower()) if industry else None
                score = 2 if industry_match else 1
                if score > best_score:
                    best_score = score
                    best_match = config

    return best_match


def get_sector_kpis(sector: str, industry: str = "") -> list[str]:
    """Return the priority KPI list for a sector/industry combo."""
    config = _match_subsector(sector, industry)
    if config:
        return config["priority_kpis"]
    return []


def get_sector_context(
    sector: str,
    industry: str = "",
    country: str = "",
) -> str:
    """
    Build a context block to inject into extraction prompts.
    Returns a formatted string with sector KPIs, reporting standards, and country context.
    Returns empty string if no specific context is found.
    """
    subsector = _match_subsector(sector, industry)
    country_ctx = COUNTRY_CONTEXT.get(country, {})

    if not subsector and not country_ctx:
        return ""

    parts = []

    if subsector:
        parts.append(f"SECTOR CONTEXT: {subsector['display_name']}")
        parts.append(f"Priority KPIs to extract: {', '.join(subsector['priority_kpis'])}")
        if subsector.get("reporting_standards"):
            parts.append(f"Reporting standards: {subsector['reporting_standards']}")
        if subsector.get("notes"):
            parts.append(f"Extraction notes: {subsector['notes']}")

    if country_ctx:
        parts.append(f"Country: {country} | GAAP: {country_ctx.get('gaap', 'Unknown')}")
        parts.append(f"Currency: {country_ctx.get('currency', 'Unknown')}")
        if country_ctx.get("fiscal_year_note"):
            parts.append(f"Fiscal year: {country_ctx['fiscal_year_note']}")

    return "\n".join(parts)


def get_sector_normalisation_overrides(sector: str, industry: str = "") -> dict[str, str]:
    """
    Return sector-specific metric name normalisations.
    E.g. for banks, "Net interest income" → "NII".
    """
    subsector = _match_subsector(sector, industry)
    if not subsector:
        return {}

    # Build a mapping from common variations to canonical names
    overrides = {}
    for kpi in subsector.get("priority_kpis", []):
        # If the KPI has an acronym in parens, map both ways
        # e.g. "NII (Net Interest Income)" → {"net interest income": "NII"}
        if "(" in kpi and ")" in kpi:
            acronym = kpi.split("(")[0].strip()
            full_name = kpi.split("(")[1].rstrip(")").strip()
            overrides[full_name.lower()] = acronym
            overrides[acronym.lower()] = acronym
        else:
            overrides[kpi.lower()] = kpi

    return overrides
