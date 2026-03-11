"""
Streamlit MVP dashboard (§11).

Run with:
    streamlit run apps/ui/dashboard.py
"""

import json
from pathlib import Path

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"


# ── Helpers ──────────────────────────────────────────────────────
def api(method: str, path: str, **kwargs):
    url = f"{API_BASE}{path}"
    resp = getattr(requests, method)(url, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ──────────────────────────────────────────────────────
st.set_page_config(page_title="Research CoWork Agent", layout="wide")
st.sidebar.title("Research CoWork Agent")
page = st.sidebar.radio("Navigate", [
    "Portfolio View",
    "Company Page",
    "Upload Document",
    "Review Queue",
])


# ═══════════════════════════════════════════════════════════════════
# Portfolio View
# ═══════════════════════════════════════════════════════════════════
if page == "Portfolio View":
    st.header("Portfolio Overview")
    try:
        companies = api("get", "/companies")
        if not companies:
            st.info("No companies added yet.")
        else:
            for c in companies:
                col1, col2, col3 = st.columns([2, 2, 1])
                col1.write(f"**{c['ticker']}** — {c['name']}")
                col2.write(f"Sector: {c.get('sector', '—')} | Analyst: {c.get('primary_analyst', '—')}")
                col3.write(f"Status: {c.get('coverage_status', '—')}")
    except Exception as e:
        st.error(f"Could not load companies: {e}")

    st.divider()
    st.subheader("Add Company")
    with st.form("add_company"):
        ticker = st.text_input("Ticker")
        name = st.text_input("Company Name")
        sector = st.text_input("Sector")
        analyst = st.text_input("Primary Analyst")
        submitted = st.form_submit_button("Add")
        if submitted and ticker and name:
            try:
                api("post", "/companies", json={
                    "ticker": ticker, "name": name,
                    "sector": sector, "primary_analyst": analyst,
                })
                st.success(f"Added {ticker}")
                st.rerun()
            except Exception as e:
                st.error(str(e))


# ═══════════════════════════════════════════════════════════════════
# Company Page
# ═══════════════════════════════════════════════════════════════════
elif page == "Company Page":
    st.header("Company Page")
    ticker = st.text_input("Enter ticker", value="HEIA").upper()
    if ticker:
        try:
            company = api("get", f"/companies/{ticker}")
            st.subheader(f"{company['name']} ({company['ticker']})")

            # Documents
            st.markdown("### Documents")
            docs = api("get", f"/companies/{ticker}/documents")
            if docs:
                for d in docs:
                    st.write(f"- **{d['title']}** ({d['document_type']}) — {d['period_label']} — Status: {d['parsing_status']}")
            else:
                st.info("No documents yet.")

            # Outputs
            st.markdown("### Research Outputs")
            outputs = api("get", f"/companies/{ticker}/outputs")
            if outputs:
                for o in outputs:
                    st.write(f"- {o['output_type']} — {o['period_label']} — {o['review_status']}")
            else:
                st.info("No outputs generated yet.")

            # Generate outputs
            st.markdown("### Generate Outputs")
            period = st.text_input("Period label", value="2026_Q1")
            col1, col2, col3 = st.columns(3)
            if col1.button("Briefing"):
                r = api("post", f"/companies/{ticker}/generate-briefing", params={"period_label": period})
                st.json(r)
            if col2.button("IR Questions"):
                r = api("post", f"/companies/{ticker}/generate-ir-questions", params={"period_label": period})
                st.json(r)
            if col3.button("Thesis Drift"):
                r = api("post", f"/companies/{ticker}/generate-thesis-drift", params={"period_label": period})
                st.json(r)

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                st.warning(f"Company {ticker} not found. Add it first in Portfolio View.")
            else:
                st.error(str(e))


# ═══════════════════════════════════════════════════════════════════
# Upload Document
# ═══════════════════════════════════════════════════════════════════
elif page == "Upload Document":
    st.header("Upload Document")
    ticker = st.text_input("Company ticker", value="HEIA").upper()
    period = st.text_input("Period label", value="2026_Q1")
    doc_type = st.selectbox("Document type", [
        "earnings_release", "transcript", "presentation", "10-Q", "10-K", "annual_report", "other",
    ])
    title = st.text_input("Title (optional)")
    file = st.file_uploader("Choose file", type=["pdf", "docx", "txt"])

    if st.button("Upload & Process") and file and ticker:
        try:
            # Upload
            resp = requests.post(
                f"{API_BASE}/companies/{ticker}/documents/upload",
                files={"file": (file.name, file.getvalue())},
                data={"document_type": doc_type, "period_label": period, "title": title or file.name},
            )
            resp.raise_for_status()
            doc = resp.json()
            st.success(f"Uploaded: {doc['title']} (id: {doc['id']})")

            # Process
            with st.spinner("Processing document …"):
                r = api("post", f"/documents/{doc['id']}/process")
                st.write("Parse result:", r)

            # Extract
            with st.spinner("Extracting metrics …"):
                r = api("post", f"/documents/{doc['id']}/extract")
                st.write("Extraction result:", r)

            # Compare
            with st.spinner("Comparing thesis …"):
                try:
                    r = api("post", f"/documents/{doc['id']}/compare")
                    st.write("Thesis comparison:", r)
                except Exception:
                    st.info("Thesis comparison skipped (no active thesis).")

        except Exception as e:
            st.error(str(e))


# ═══════════════════════════════════════════════════════════════════
# Review Queue
# ═══════════════════════════════════════════════════════════════════
elif page == "Review Queue":
    st.header("Review Queue")
    status_filter = st.selectbox("Status", ["open", "approved", "rejected", "edited"])
    try:
        items = api("get", "/review-queue", params={"status": status_filter})
        if not items:
            st.info("No items in queue.")
        for item in items:
            with st.expander(f"[{item['priority']}] {item['entity_type']} — {item['queue_reason']}"):
                st.write(f"Entity ID: `{item['entity_id']}`")
                st.write(f"Created: {item['created_at']}")
                col1, col2, col3 = st.columns(3)
                if col1.button("Approve", key=f"a-{item['id']}"):
                    api("post", f"/review-queue/{item['id']}/approve")
                    st.rerun()
                if col2.button("Reject", key=f"r-{item['id']}"):
                    api("post", f"/review-queue/{item['id']}/reject")
                    st.rerun()
                if col3.button("Edit", key=f"e-{item['id']}"):
                    api("post", f"/review-queue/{item['id']}/edit")
                    st.rerun()
    except Exception as e:
        st.error(str(e))
