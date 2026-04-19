"""Tests for agents/ingestion/coverage_monitor.py — verifies the auto-
rescan gating policy (24h throttle, 3-attempt cap, source_broken skip)
and per-ticker batching. DB and Orchestrator are mocked."""

import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.ingestion.coverage_monitor import CoverageMonitor
from services.harvester.coverage_advanced import CoverageGap


def _make_gap(ticker: str, doc_type: str, severity: str, *,
              company_id: str | None = None,
              expected_period: str = "2026_Q1",
              days_overdue: int = 3) -> CoverageGap:
    """Build a CoverageGap for tests. Deterministic UUID per ticker so
    we can match the expected orchestrator calls."""
    return CoverageGap(
        company_id=company_id or str(uuid.uuid4()),
        ticker=ticker,
        name=ticker,
        doc_type=doc_type,
        expected_period=expected_period,
        expected_by=date(2026, 4, 17),
        days_overdue=days_overdue,
        severity=severity,
        reason="test",
        cadence_frequency="quarterly",
        cadence_sample_size=8,
        sources_to_retry=["ir_scrape", "ir_llm"],
    )


def _session_cm(*, recent_count: int = 0, total_auto: int = 0):
    """Build an async context manager that yields a mock DB session
    which records its `add()` calls. Supports sequential execute() so
    the first query returns `recent_count` rows, the second `total_auto`.
    Not used for the main gap lookup path — that's patched separately."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm, session


@pytest.mark.asyncio
class TestAutoTriggerDisabled:
    async def test_returns_gaps_without_calling_orchestrator(self):
        gaps = [_make_gap("ASML", "transcript", "critical")]
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        mock_orch_instance.run_targeted_scan = AsyncMock()
        mock_orch_cls.return_value = mock_orch_instance

        with patch("agents.ingestion.coverage_monitor.find_overdue_gaps",
                   new=AsyncMock(return_value=gaps)), \
             patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=_session_cm()[0]), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            result = await CoverageMonitor().run_daily_check(auto_trigger=False)

        assert result.gaps_found == 1
        assert result.rescans_triggered == 0
        mock_orch_instance.run_targeted_scan.assert_not_called()


@pytest.mark.asyncio
class TestRescanGating:
    async def test_source_broken_severity_not_auto_rescanned(self):
        gaps = [_make_gap("BUNZL LN", "earnings_release", "source_broken",
                          days_overdue=150)]
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        mock_orch_instance.run_targeted_scan = AsyncMock()
        mock_orch_cls.return_value = mock_orch_instance

        session_cm, _ = _session_cm()
        with patch("agents.ingestion.coverage_monitor.find_overdue_gaps",
                   new=AsyncMock(return_value=gaps)), \
             patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=session_cm), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            result = await CoverageMonitor().run_daily_check()

        assert result.gaps_found == 1
        assert result.rescans_triggered == 0
        mock_orch_instance.run_targeted_scan.assert_not_called()

    async def test_warning_severity_not_auto_rescanned(self):
        """`warning` = not-yet-overdue. Don't pre-emptively rescan."""
        gaps = [_make_gap("ASML", "transcript", "warning", days_overdue=-2)]
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        mock_orch_instance.run_targeted_scan = AsyncMock()
        mock_orch_cls.return_value = mock_orch_instance

        session_cm, _ = _session_cm()
        with patch("agents.ingestion.coverage_monitor.find_overdue_gaps",
                   new=AsyncMock(return_value=gaps)), \
             patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=session_cm), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            result = await CoverageMonitor().run_daily_check()

        assert result.rescans_triggered == 0
        mock_orch_instance.run_targeted_scan.assert_not_called()

    async def test_recent_rescan_skipped(self):
        """24h throttle: if the gap was rescanned in the last 24h,
        skip it and bump rescans_skipped_recent."""
        gaps = [_make_gap("ASML", "transcript", "critical")]
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        mock_orch_instance.run_targeted_scan = AsyncMock()
        mock_orch_cls.return_value = mock_orch_instance

        session_cm, _ = _session_cm()
        with patch("agents.ingestion.coverage_monitor.find_overdue_gaps",
                   new=AsyncMock(return_value=gaps)), \
             patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=session_cm), \
             patch("agents.ingestion.coverage_monitor._recent_rescan_count",
                   new=AsyncMock(return_value=1)), \
             patch("agents.ingestion.coverage_monitor._total_auto_rescans_for_gap",
                   new=AsyncMock(return_value=1)), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            result = await CoverageMonitor().run_daily_check()

        assert result.rescans_triggered == 0
        assert result.rescans_skipped_recent == 1
        mock_orch_instance.run_targeted_scan.assert_not_called()

    async def test_max_attempts_exhausted_skipped(self):
        """3-attempt cap: after 3 auto-rescans for the same gap, stop.
        The analyst has to trigger manually if they want another try."""
        gaps = [_make_gap("ASML", "transcript", "critical")]
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        mock_orch_instance.run_targeted_scan = AsyncMock()
        mock_orch_cls.return_value = mock_orch_instance

        session_cm, _ = _session_cm()
        with patch("agents.ingestion.coverage_monitor.find_overdue_gaps",
                   new=AsyncMock(return_value=gaps)), \
             patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=session_cm), \
             patch("agents.ingestion.coverage_monitor._recent_rescan_count",
                   new=AsyncMock(return_value=0)), \
             patch("agents.ingestion.coverage_monitor._total_auto_rescans_for_gap",
                   new=AsyncMock(return_value=3)), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            result = await CoverageMonitor().run_daily_check()

        assert result.rescans_triggered == 0
        assert result.rescans_skipped_exhausted == 1
        mock_orch_instance.run_targeted_scan.assert_not_called()


@pytest.mark.asyncio
class TestSuccessfulRescan:
    async def test_eligible_gap_triggers_orchestrator(self):
        gaps = [_make_gap("ASML", "transcript", "overdue")]
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        mock_orch_instance.run_targeted_scan = AsyncMock(
            return_value={"new": 1, "skipped": 0, "failed": 0}
        )
        mock_orch_cls.return_value = mock_orch_instance

        session_cm, _ = _session_cm()
        with patch("agents.ingestion.coverage_monitor.find_overdue_gaps",
                   new=AsyncMock(return_value=gaps)), \
             patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=session_cm), \
             patch("agents.ingestion.coverage_monitor._recent_rescan_count",
                   new=AsyncMock(return_value=0)), \
             patch("agents.ingestion.coverage_monitor._total_auto_rescans_for_gap",
                   new=AsyncMock(return_value=0)), \
             patch("agents.ingestion.coverage_monitor._record_rescan",
                   new=AsyncMock()), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            result = await CoverageMonitor().run_daily_check()

        assert result.rescans_triggered == 1
        assert result.rescan_successes == 1
        assert "ASML" in result.triggered_tickers
        mock_orch_instance.run_targeted_scan.assert_awaited_once()

    async def test_multiple_gaps_same_ticker_single_scan(self):
        """One company with two missing doc types should trigger ONE
        orchestrator call, not two. Both rescans should be counted in
        the summary though."""
        shared_company = str(uuid.uuid4())
        gaps = [
            _make_gap("ASML", "transcript",   "critical", company_id=shared_company),
            _make_gap("ASML", "presentation", "overdue",  company_id=shared_company),
        ]
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        mock_orch_instance.run_targeted_scan = AsyncMock(
            return_value={"new": 2, "skipped": 0, "failed": 0}
        )
        mock_orch_cls.return_value = mock_orch_instance

        session_cm, _ = _session_cm()
        with patch("agents.ingestion.coverage_monitor.find_overdue_gaps",
                   new=AsyncMock(return_value=gaps)), \
             patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=session_cm), \
             patch("agents.ingestion.coverage_monitor._recent_rescan_count",
                   new=AsyncMock(return_value=0)), \
             patch("agents.ingestion.coverage_monitor._total_auto_rescans_for_gap",
                   new=AsyncMock(return_value=0)), \
             patch("agents.ingestion.coverage_monitor._record_rescan",
                   new=AsyncMock()), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            result = await CoverageMonitor().run_daily_check()

        # Two gaps counted, but only one orchestrator call
        assert result.rescans_triggered == 2
        assert mock_orch_instance.run_targeted_scan.await_count == 1
        assert result.triggered_tickers == ["ASML"]

    async def test_orchestrator_error_recorded_and_continues(self):
        """If the orchestrator raises, every gap for that ticker counts
        as an error — but the Monitor continues processing other tickers."""
        gaps = [
            _make_gap("ASML", "transcript", "critical"),
            _make_gap("LKQ US", "transcript", "overdue"),
        ]
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        # First call raises, second succeeds
        mock_orch_instance.run_targeted_scan = AsyncMock(
            side_effect=[Exception("network down"),
                         {"new": 0, "skipped": 0, "failed": 0}]
        )
        mock_orch_cls.return_value = mock_orch_instance

        session_cm, _ = _session_cm()
        with patch("agents.ingestion.coverage_monitor.find_overdue_gaps",
                   new=AsyncMock(return_value=gaps)), \
             patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=session_cm), \
             patch("agents.ingestion.coverage_monitor._recent_rescan_count",
                   new=AsyncMock(return_value=0)), \
             patch("agents.ingestion.coverage_monitor._total_auto_rescans_for_gap",
                   new=AsyncMock(return_value=0)), \
             patch("agents.ingestion.coverage_monitor._record_rescan",
                   new=AsyncMock()), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            result = await CoverageMonitor().run_daily_check()

        assert result.rescan_errors == 1
        assert result.rescan_no_new == 1
        assert mock_orch_instance.run_targeted_scan.await_count == 2


@pytest.mark.asyncio
class TestRescanOneGap:
    async def test_manual_rescan_bypasses_throttle(self):
        """Manual rescan ALWAYS runs — doesn't consult _recent_rescan_count."""
        mock_orch_cls = MagicMock()
        mock_orch_instance = MagicMock()
        mock_orch_instance.run_targeted_scan = AsyncMock(
            return_value={"new": 1, "skipped": 0, "failed": 0}
        )
        mock_orch_cls.return_value = mock_orch_instance

        # Patch AsyncSessionLocal — rescan_one_gap looks up the Company
        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock()
        session_cm.__aexit__ = AsyncMock(return_value=None)
        # Company lookup returns a mock with .id attribute
        mock_company = MagicMock()
        mock_company.id = uuid.uuid4()
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none = MagicMock(return_value=mock_company)
        session.execute = AsyncMock(return_value=scalar_result)
        session_cm.__aenter__.return_value = session

        with patch("agents.ingestion.coverage_monitor.AsyncSessionLocal",
                   return_value=session_cm), \
             patch("agents.ingestion.coverage_monitor._recent_rescan_count",
                   new=AsyncMock(return_value=99)), \
             patch("agents.ingestion.orchestrator.IngestionOrchestrator",
                   new=mock_orch_cls):
            summary = await CoverageMonitor().rescan_one_gap(
                "ASML", doc_type="transcript", expected_period="2026_Q1",
            )

        # Orchestrator was called regardless of high recent_rescan_count
        mock_orch_instance.run_targeted_scan.assert_awaited_once()
        assert summary["new"] == 1
