"""Tests for agents/ingestion/orchestrator.py — tier filtering and
explicit ticker override paths. Uses patch to avoid touching the DB
or the real run_harvest."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from agents.ingestion.orchestrator import IngestionOrchestrator, _TIER_TO_COVERAGE_STATUS


class TestTierMap:
    def test_portfolio_maps_to_active(self):
        assert _TIER_TO_COVERAGE_STATUS["portfolio"] == ["active"]

    def test_watchlist_maps_to_watchlist(self):
        assert _TIER_TO_COVERAGE_STATUS["watchlist"] == ["watchlist"]

    def test_all_maps_to_both(self):
        assert set(_TIER_TO_COVERAGE_STATUS["all"]) == {"active", "watchlist"}


@pytest.mark.asyncio
class TestRunScheduledScan:
    async def test_unknown_tier_raises(self):
        orch = IngestionOrchestrator()
        with pytest.raises(ValueError, match="Unknown tier"):
            await orch.run_scheduled_scan(tier="bogus")

    async def test_empty_tier_short_circuits(self):
        """If no companies match the tier, run_harvest must NOT be called —
        orchestrator should return an empty summary immediately."""
        orch = IngestionOrchestrator()
        with patch.object(orch, "_tickers_for_tier",
                          new=AsyncMock(return_value=[])), \
             patch("services.harvester.run_harvest",
                   new=AsyncMock()) as mock_harvest:
            result = await orch.run_scheduled_scan(tier="watchlist")
        mock_harvest.assert_not_called()
        assert result["new"] == 0
        assert result["skipped"] == 0
        assert result["failed"] == 0
        assert result["tickers"] == []

    async def test_tier_passes_tickers_to_run_harvest(self):
        orch = IngestionOrchestrator()
        with patch.object(orch, "_tickers_for_tier",
                          new=AsyncMock(return_value=["LKQ US", "ASML NA"])), \
             patch("services.harvester.run_harvest",
                   new=AsyncMock(return_value={"new": 2, "skipped": 0, "failed": 0,
                                               "tickers": ["LKQ US", "ASML NA"],
                                               "details": []})) as mock_harvest:
            result = await orch.run_scheduled_scan(tier="portfolio", skip_llm=True)
        mock_harvest.assert_awaited_once()
        kwargs = mock_harvest.await_args.kwargs
        assert kwargs["tickers"] == ["LKQ US", "ASML NA"]
        assert kwargs["skip_llm"] is True
        assert result["tier"] == "portfolio"

    async def test_explicit_tickers_bypass_tier_filter(self):
        """When ``tickers=[...]`` is passed explicitly, tier is ignored —
        the Orchestrator must NOT query the DB for tier-based ticker
        selection. Used by the Event Scanner and manual rescans."""
        orch = IngestionOrchestrator()
        with patch.object(orch, "_tickers_for_tier",
                          new=AsyncMock()) as mock_tier_lookup, \
             patch("services.harvester.run_harvest",
                   new=AsyncMock(return_value={"new": 1, "skipped": 0, "failed": 0,
                                               "tickers": ["ASML NA"], "details": []})):
            result = await orch.run_scheduled_scan(
                tier="portfolio",   # should be ignored
                tickers=["ASML NA"],
            )
        mock_tier_lookup.assert_not_called()
        assert result["tier"] == "explicit"

    async def test_run_targeted_scan_always_uses_all_tier_default(self):
        """run_targeted_scan is the Event-Scanner hook — skip_llm defaults
        to False so single-company scans use the best available source."""
        orch = IngestionOrchestrator()
        with patch.object(orch, "run_scheduled_scan",
                          new=AsyncMock(return_value={})) as mock_scan:
            await orch.run_targeted_scan(["LKQ US"])
        kwargs = mock_scan.await_args.kwargs
        assert kwargs["skip_llm"] is False
        assert kwargs["tickers"] == ["LKQ US"]
