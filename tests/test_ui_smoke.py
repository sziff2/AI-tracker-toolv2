"""
Browser smoke tests — would have caught the stress-test MutationObserver
infinite loop that crashed Analytics on 2026-04-22.

Loads each main tab in a real Chromium browser, asserts the page renders
the expected heading, and asserts NO uncaught console errors. If the JS
throws or hangs, these tests fail.

How to run (locally):

    pip install -r requirements-dev.txt
    playwright install chromium
    APP_PASSWORD=oldfield UI_BASE=https://ai-tracker-tool-production.up.railway.app \
        pytest tests/test_ui_smoke.py -v

Skipped automatically when APP_PASSWORD env var is unset, so this file
is safe to live alongside the rest of the suite — it never runs in
the regular `pytest tests/` path used in CI today.

Add `--ui` (or set RUN_UI_TESTS=1) to opt-in explicitly when you want
to gate the suite on the smoke result before deploy.
"""
from __future__ import annotations

import os
import re
import time

import pytest

APP_PASSWORD = os.environ.get("APP_PASSWORD")
UI_BASE = os.environ.get("UI_BASE", "https://ai-tracker-tool-production.up.railway.app").rstrip("/")
RUN_UI = bool(APP_PASSWORD) and (
    os.environ.get("RUN_UI_TESTS") == "1"
    or "--ui" in os.environ.get("PYTEST_ADDOPTS", "")
    or os.environ.get("FORCE_UI_SMOKE")
)

pytestmark = pytest.mark.skipif(
    not RUN_UI,
    reason="UI smoke disabled — set APP_PASSWORD + RUN_UI_TESTS=1 to enable",
)

# Console messages we tolerate (fonts blocked by CSP, missing favicon, etc.)
TOLERATED_CONSOLE_PATTERNS = [
    re.compile(r"fonts\.googleapis\.com", re.IGNORECASE),
    re.compile(r"favicon", re.IGNORECASE),
    re.compile(r"Failed to load resource.*404", re.IGNORECASE),
]


def _is_tolerated(msg: str) -> bool:
    return any(p.search(msg) for p in TOLERATED_CONSOLE_PATTERNS)


# ─────────────────────────────────────────────────────────────────
# Fixtures — login once per browser context so individual tabs are fast.
# ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def authed_context():
    """Yield a Playwright BrowserContext with a valid auth cookie set."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto(f"{UI_BASE}/login", timeout=30_000)
        page.fill("input[type=password]", APP_PASSWORD)
        page.click("button[type=submit], button:has-text('Sign in')")
        # Wait for cookie + redirect; main app shows the workspace shell
        page.wait_for_selector("text=WORKSPACE", timeout=15_000)
        page.close()
        yield ctx
        browser.close()


def _collect_console_errors(page):
    """Attach a console listener and return the list it accumulates."""
    errors = []
    page.on("console", lambda m: errors.append((m.type, m.text)) if m.type in ("error",) else None)
    page.on("pageerror", lambda exc: errors.append(("pageerror", str(exc))))
    return errors


def _click_sidebar(page, label: str):
    """Click a sidebar nav item by visible text."""
    page.locator(f"text={label}").first.click()


# ─────────────────────────────────────────────────────────────────
# Per-tab smoke tests
# ─────────────────────────────────────────────────────────────────
TABS = [
    ("Portfolio",            "h1:has-text('Portfolio')"),
    ("Analytics & Risk",     "h1:has-text('Analytics & Risk')"),
    ("Portfolio Optimisation", "h1:has-text('Portfolio Optimisation')"),
    ("Macro View",           "h1:has-text('Macro')"),
    ("Data Hub",             "h1:has-text('Data Hub')"),
]


@pytest.mark.parametrize("label,heading_selector", TABS, ids=[t[0] for t in TABS])
def test_tab_renders_without_console_errors(authed_context, label, heading_selector):
    """Each main tab loads, shows its heading, and produces no JS errors
    within 8 seconds. Catches JS exceptions, missing functions, infinite
    loops (via wait timeout), and unhandled promise rejections."""
    page = authed_context.new_page()
    errors = _collect_console_errors(page)
    try:
        page.goto(UI_BASE, timeout=30_000)
        page.wait_for_selector("text=WORKSPACE", timeout=10_000)
        _click_sidebar(page, label)
        # Heading proves render completed
        page.wait_for_selector(heading_selector, timeout=10_000)
        # Brief settle so any async chip-loaders / chart-painters fire
        time.sleep(2.5)
        # Must not be stuck in an infinite loop — sentinel: page can run JS
        assert page.evaluate("() => true") is True, f"{label}: JS thread unresponsive"
        non_tolerated = [(t, m) for (t, m) in errors if not _is_tolerated(m)]
        assert not non_tolerated, (
            f"{label}: console produced {len(non_tolerated)} unexpected error(s):\n"
            + "\n".join(f"  [{t}] {m[:200]}" for (t, m) in non_tolerated[:10])
        )
    finally:
        page.close()


def test_analytics_stress_chips_render(authed_context):
    """Specific regression check for the MutationObserver-loop bug:
    Analytics tab must show stress chips within 6 seconds of rendering,
    not stay stuck on 'Loading presets…'."""
    page = authed_context.new_page()
    errors = _collect_console_errors(page)
    try:
        page.goto(UI_BASE, timeout=30_000)
        page.wait_for_selector("text=WORKSPACE", timeout=10_000)
        _click_sidebar(page, "Analytics & Risk")
        page.wait_for_selector("text=Historical Stress Tests", timeout=10_000)
        # Stress preset chips include button labels with episode names.
        page.wait_for_selector(
            "#stress-presets button:has-text('Global Financial Crisis')",
            timeout=8_000,
        )
        chip_count = page.locator("#stress-presets button").count()
        assert chip_count >= 5, f"Expected ≥5 stress chips, got {chip_count}"
        non_tolerated = [(t, m) for (t, m) in errors if not _is_tolerated(m)]
        assert not non_tolerated, (
            "Analytics threw console errors after stress card rendered:\n"
            + "\n".join(f"  [{t}] {m[:200]}" for (t, m) in non_tolerated[:10])
        )
    finally:
        page.close()
