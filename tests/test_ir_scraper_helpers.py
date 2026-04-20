"""Tests for pure helpers in services/harvester/sources/ir_scraper.py.
Focused on the three fixes that landed after the NWC CN diagnostic:
mojibake URL repair, narrower doc-type classification, and filename
fallback when link text is uninformative."""

from services.harvester.sources.ir_scraper import (
    _GENERIC_ANCHOR_TEXTS,
    _classify_doc_type,
    _fix_mojibake,
    _make_candidate,
)


class TestFixMojibake:
    def test_em_dash_triad_restored(self):
        # U+2013 (en-dash) UTF-8 bytes = E2 80 93; misinterpreted as
        # Latin-1 = â€™-ish; re-encoded as UTF-8 = c3 a2 c2 80 c2 93.
        # The literal triple that came out of NWC's IR URL.
        broken = "Press release \u00e2\u0080\u0093 Q1 2025.pdf"
        assert _fix_mojibake(broken) == "Press release \u2013 Q1 2025.pdf"

    def test_em_dash_variant(self):
        # Em-dash U+2014 (long dash)
        broken = "Title \u00e2\u0080\u0094 subtitle"
        assert _fix_mojibake(broken) == "Title \u2014 subtitle"

    def test_smart_quotes_restored(self):
        broken = "\u00e2\u0080\u009cQuoted\u00e2\u0080\u009d value"
        assert _fix_mojibake(broken) == "\u201cQuoted\u201d value"

    def test_clean_string_untouched(self):
        assert _fix_mojibake("plain ascii title") == "plain ascii title"
        assert _fix_mojibake("Title – Q1 2025.pdf") == "Title – Q1 2025.pdf"

    def test_empty_and_none(self):
        assert _fix_mojibake("") == ""
        assert _fix_mojibake(None) is None


class TestClassifyDocType:
    def test_press_release_beats_presentation(self):
        """Key fix: 'Press Release - MAR23-22.pdf' sitting on an IR
        page titled 'Investor Presentations' must NOT be classified as
        presentation. Previously page context polluted the classifier."""
        assert _classify_doc_type(
            slug="press release mar23 22",
            headline="Press Release - MAR23-22.pdf",
            url_path="/uploads/documents/press-release-mar23-22.pdf",
        ) == "earnings_release"

    def test_explicit_presentation_still_works(self):
        assert _classify_doc_type(
            slug="investor presentation q4 2024",
            headline="Q4 2024 Investor Presentation",
            url_path="/presentations/q4-2024.pdf",
        ) == "presentation"

    def test_transcript(self):
        assert _classify_doc_type(
            slug="earnings call transcript q3 2024",
            headline="Q3 2024 Earnings Call Transcript",
            url_path="/transcripts/q3-2024.pdf",
        ) == "transcript"

    def test_annual_report(self):
        assert _classify_doc_type(
            slug="annual report 2024",
            headline="2024 Annual Report",
            url_path="/annual-reports/2024.pdf",
        ) == "annual_report"

    def test_proxy(self):
        assert _classify_doc_type(
            slug="proxy statement 2024",
            headline="2024 Proxy Statement",
            url_path="/proxy/2024.pdf",
        ) == "proxy_statement"

    def test_bare_results_falls_through_to_earnings_release(self):
        assert _classify_doc_type(
            slug="q1 2024 results",
            headline="Q1 2024 Results",
            url_path="/results/q1-2024.pdf",
        ) == "earnings_release"

    def test_presentation_and_press_release_both_present_press_wins(self):
        """If the link text explicitly says press release, that should
        win even if the slug says 'presentation' for some reason."""
        assert _classify_doc_type(
            slug="press release",
            headline="Press release — Q1 2024 presentation attached",
            url_path="/news/press-release-q1-2024.pdf",
        ) == "earnings_release"

    def test_unknown_returns_other(self):
        assert _classify_doc_type(
            slug="some document",
            headline="Some Document",
            url_path="/docs/some-document.pdf",
        ) == "other"

    def test_only_url_path_provided(self):
        """Even with empty slug/headline, URL path alone should classify."""
        assert _classify_doc_type(
            slug="",
            headline="",
            url_path="/press-release/q1-2025.pdf",
        ) == "earnings_release"


class TestMakeCandidate:
    def test_uninformative_open_falls_back_to_slug(self):
        """NWC CN case: every IR link's visible text is 'Open'. The
        headline should come from the filename, not the anchor text."""
        c = _make_candidate(
            ticker="NWC CN",
            pdf_url="https://www.northwest.ca/uploads/documents/JUN3-20.pdf",
            link_text="Open",
            context="",
        )
        assert c["headline"].lower() != "open"
        assert "jun3" in c["headline"].lower()

    def test_download_also_falls_back(self):
        c = _make_candidate(
            ticker="TEST",
            pdf_url="https://example.com/docs/annual-report-2024.pdf",
            link_text="Download",
            context="",
        )
        assert c["headline"].lower() != "download"
        assert "annual" in c["headline"].lower()

    def test_case_insensitive_generic(self):
        c = _make_candidate(
            ticker="TEST",
            pdf_url="https://example.com/docs/q1-results.pdf",
            link_text="CLICK HERE",
            context="",
        )
        assert "click here" not in c["headline"].lower()

    def test_real_link_text_preserved(self):
        c = _make_candidate(
            ticker="TEST",
            pdf_url="https://example.com/docs/2024-annual.pdf",
            link_text="2024 Annual Report — Full",
            context="",
        )
        # Real anchor text should win
        assert "2024 Annual Report" in c["headline"]

    def test_mojibake_url_fixed(self):
        """Em-dash in filename that arrived as double-UTF-8 mojibake
        should be corrected to the real unicode character before being
        stored as the source_url. Then httpx will URL-encode it properly."""
        broken_url = (
            "https://www.northwest.ca/uploads/documents/"
            "Press release \u00e2\u0080\u0093 Q1 2025.pdf"
        )
        c = _make_candidate(
            ticker="NWC CN",
            pdf_url=broken_url,
            link_text="Open",
            context="",
        )
        # The stored URL should have the en-dash as a single character,
        # not the three-byte Latin-1 triad.
        assert "\u00e2\u0080\u0093" not in c["source_url"]
        assert "\u2013" in c["source_url"]
        # And the doc_type should classify as earnings_release (filename
        # contains "press release")
        assert c["document_type"] == "earnings_release"

    def test_mojibake_preserved_through_candidate_fields(self):
        """Slug + headline also go through _fix_mojibake."""
        c = _make_candidate(
            ticker="TEST",
            pdf_url="https://example.com/docs/Report\u00e2\u0080\u0093Q4.pdf",
            link_text="",
            context="",
        )
        assert "\u00e2\u0080\u0093" not in c["headline"]


class TestGenericAnchorTextsSet:
    def test_set_contains_common_variants(self):
        for text in ("open", "download", "pdf", "view", "click here"):
            assert text in _GENERIC_ANCHOR_TEXTS
