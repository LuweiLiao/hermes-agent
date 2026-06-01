"""Tests for the background work cost ledger + budget gate."""

import importlib

import pytest


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    """Fresh ledger module pointed at an isolated HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import agent.background_ledger as bl

    importlib.reload(bl)
    return bl


def test_record_and_daily_sum(ledger):
    ledger.record_background_spend(ledger.ORIGIN_REVIEW, cost_usd=0.01, total_tokens=100)
    ledger.record_background_spend(ledger.ORIGIN_CURATOR, cost_usd=0.02, total_tokens=200)
    assert ledger.get_daily_spend() == pytest.approx(0.03)
    assert ledger.get_daily_spend(origin=ledger.ORIGIN_REVIEW) == pytest.approx(0.01)


def test_session_spend_attribution(ledger):
    ledger.record_background_spend(
        ledger.ORIGIN_REVIEW, cost_usd=0.05, parent_session_id="sess-A"
    )
    ledger.record_background_spend(
        ledger.ORIGIN_REVIEW, cost_usd=0.07, parent_session_id="sess-B"
    )
    assert ledger.get_session_spend("sess-A") == pytest.approx(0.05)
    assert ledger.get_session_spend("sess-B") == pytest.approx(0.07)
    assert ledger.get_session_spend("nope") == 0.0


def test_today_breakdown(ledger):
    ledger.record_background_spend(ledger.ORIGIN_REVIEW, cost_usd=0.01, total_tokens=10)
    ledger.record_background_spend(ledger.ORIGIN_REVIEW, cost_usd=0.02, total_tokens=20)
    ledger.record_background_spend(ledger.ORIGIN_CURATOR, cost_usd=0.10, total_tokens=5)
    bd = ledger.get_today_breakdown()
    assert bd[ledger.ORIGIN_REVIEW]["runs"] == 2
    assert bd[ledger.ORIGIN_REVIEW]["cost_usd"] == pytest.approx(0.03)
    assert bd[ledger.ORIGIN_REVIEW]["total_tokens"] == 30
    assert bd[ledger.ORIGIN_CURATOR]["runs"] == 1


def test_record_from_result_dict(ledger):
    conv_result = {
        "estimated_cost_usd": 0.042,
        "total_tokens": 321,
        "api_calls": 3,
        "model": "test-model",
        "provider": "test",
        "session_id": "child-1",
        "cost_status": "estimated",
    }
    ledger.record_from_result(ledger.ORIGIN_REVIEW, conv_result, parent_session_id="P")
    assert ledger.get_daily_spend() == pytest.approx(0.042)
    assert ledger.get_session_spend("P") == pytest.approx(0.042)


def test_record_from_result_ignores_non_dict(ledger):
    ledger.record_from_result(ledger.ORIGIN_REVIEW, None)
    ledger.record_from_result(ledger.ORIGIN_REVIEW, "not a dict")
    assert ledger.get_daily_spend() == 0.0


def test_default_config_allows_unlimited(ledger, monkeypatch):
    monkeypatch.setattr(
        ledger,
        "get_background_config",
        lambda: {"enabled": True, "daily_cost_limit_usd": None, "session_cost_limit_usd": None},
    )
    ledger.record_background_spend(ledger.ORIGIN_REVIEW, cost_usd=9999.0)
    allowed, reason = ledger.is_background_allowed()
    assert allowed is True
    assert reason == ""


def test_disabled_blocks(ledger, monkeypatch):
    monkeypatch.setattr(
        ledger,
        "get_background_config",
        lambda: {"enabled": False, "daily_cost_limit_usd": None, "session_cost_limit_usd": None},
    )
    allowed, reason = ledger.is_background_allowed()
    assert allowed is False
    assert "disabled" in reason


def test_daily_cap_blocks(ledger, monkeypatch):
    monkeypatch.setattr(
        ledger,
        "get_background_config",
        lambda: {"enabled": True, "daily_cost_limit_usd": 0.05, "session_cost_limit_usd": None},
    )
    # Under the cap: allowed.
    ledger.record_background_spend(ledger.ORIGIN_REVIEW, cost_usd=0.02)
    assert ledger.is_background_allowed()[0] is True
    # Push over the cap: blocked.
    ledger.record_background_spend(ledger.ORIGIN_REVIEW, cost_usd=0.04)
    allowed, reason = ledger.is_background_allowed()
    assert allowed is False
    assert "daily" in reason


def test_session_cap_blocks(ledger, monkeypatch):
    monkeypatch.setattr(
        ledger,
        "get_background_config",
        lambda: {"enabled": True, "daily_cost_limit_usd": None, "session_cost_limit_usd": 0.10},
    )
    ledger.record_background_spend(ledger.ORIGIN_REVIEW, cost_usd=0.12, parent_session_id="S")
    # Blocked for the over-budget session...
    assert ledger.is_background_allowed(parent_session_id="S")[0] is False
    # ...but a different session is unaffected.
    assert ledger.is_background_allowed(parent_session_id="other")[0] is True


def test_budget_fails_open_on_error(ledger, monkeypatch):
    def boom():
        raise RuntimeError("ledger down")

    monkeypatch.setattr(ledger, "get_background_config", boom)
    allowed, reason = ledger.is_background_allowed()
    assert allowed is True  # fail open


def test_zero_limit_treated_as_unlimited(ledger, monkeypatch):
    def fake_load_config():
        return {"background": {"enabled": True, "daily_cost_limit_usd": 0, "session_cost_limit_usd": -1}}

    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config", fake_load_config)
    cfg = ledger.get_background_config()
    assert cfg["daily_cost_limit_usd"] is None
    assert cfg["session_cost_limit_usd"] is None
    assert cfg["enabled"] is True


def test_budget_status_snapshot(ledger, monkeypatch):
    monkeypatch.setattr(
        ledger,
        "get_background_config",
        lambda: {"enabled": True, "daily_cost_limit_usd": 1.0, "session_cost_limit_usd": None},
    )
    ledger.record_background_spend(ledger.ORIGIN_CURATOR, cost_usd=0.25)
    status = ledger.budget_status()
    assert status["allowed"] is True
    assert status["daily_spend_usd"] == pytest.approx(0.25)
    assert status["daily_limit_usd"] == 1.0
