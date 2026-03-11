"""
Unit tests for service-level logic (no DB or LLM required).
"""

from services.thesis_comparator import _previous_period


def test_previous_period_q2():
    assert _previous_period("2026_Q2") == "2026_Q1"


def test_previous_period_q1():
    assert _previous_period("2026_Q1") == "2025_Q4"


def test_previous_period_q4():
    assert _previous_period("2025_Q4") == "2025_Q3"


def test_previous_period_invalid():
    assert _previous_period("invalid") == ""
