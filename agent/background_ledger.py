"""Cost ledger + budget caps for autonomous *background* work.

Hermes does two kinds of work that the user never explicitly asked for and never
directly sees: the **background review** (memory/skill self-improvement that forks
a child agent every N turns) and the **Curator** (periodic skill consolidation).
Both call an LLM, both cost money, and historically neither was metered separately
or capped — so spend accrued silently and failures were swallowed.

This module gives that work its own accounting:

  * ``record_background_spend()`` — append one row per background LLM pass.
  * ``get_daily_spend()`` / ``get_session_spend()`` — read accrued cost.
  * ``is_background_allowed()`` — the budget gate callers check *before* spawning.

Budgets are **opt-in**: with no config the limits are ``None`` and behaviour is
unchanged (everything is allowed, just now also recorded). Set
``background.daily_cost_limit_usd`` / ``background.session_cost_limit_usd`` in
``~/.hermes/config.yaml`` to cap it, or ``background.enabled: false`` to turn
autonomous background work off entirely.

Storage is a dedicated SQLite file (``~/.hermes/background_ledger.db``) so we never
contend with or pollute the main ``state.db`` session rows. Every public function
is best-effort: a ledger error must never break the agent, so they log at debug
and degrade to a permissive default.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Recognised background work origins. Kept as constants so callers and tests
# don't hardcode strings.
ORIGIN_REVIEW = "background_review"
ORIGIN_CURATOR = "curator"

# Serialise writes from background daemon threads in *this* process. SQLite's
# own locking handles cross-process safety (WAL + busy_timeout); this just keeps
# our short-lived connections from tripping over each other.
_LOCK = threading.Lock()


def _db_path() -> Path:
    return get_hermes_home() / "background_ledger.db"


def _connect() -> sqlite3.Connection:
    """Open a short-lived connection with the schema ensured.

    Short-lived (open → use → close) connections are the simplest correct story
    for code that runs from daemon threads across multiple processes (CLI and
    gateway can both be live). WAL + a busy timeout absorbs the contention.
    """
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS background_spend (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                REAL    NOT NULL,
                day               TEXT    NOT NULL,
                origin            TEXT    NOT NULL,
                cost_usd          REAL    NOT NULL DEFAULT 0,
                total_tokens      INTEGER NOT NULL DEFAULT 0,
                api_calls         INTEGER NOT NULL DEFAULT 0,
                model             TEXT,
                provider          TEXT,
                session_id        TEXT,
                parent_session_id TEXT,
                cost_status       TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bg_day ON background_spend(day)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bg_parent "
            "ON background_spend(parent_session_id)"
        )
        conn.commit()
    except Exception:
        conn.close()
        raise
    return conn


def _today() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_background_config() -> Dict[str, Any]:
    """Read the ``background`` config block with safe defaults.

    Returns a dict with ``enabled`` (bool), ``daily_cost_limit_usd`` and
    ``session_cost_limit_usd`` (float or None = unlimited).
    """
    cfg: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        cfg = (load_config().get("background") or {})
    except Exception as e:  # pragma: no cover - config is best-effort
        logger.debug("background config load failed: %s", e)
        cfg = {}

    def _limit(key: str) -> Optional[float]:
        val = cfg.get(key)
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        # Treat 0 / negative as "no limit" so a stray 0 never silently
        # disables all background work.
        return f if f > 0 else None

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "daily_cost_limit_usd": _limit("daily_cost_limit_usd"),
        "session_cost_limit_usd": _limit("session_cost_limit_usd"),
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def record_background_spend(
    origin: str,
    *,
    cost_usd: float = 0.0,
    total_tokens: int = 0,
    api_calls: int = 0,
    model: str = "",
    provider: str = "",
    session_id: str = "",
    parent_session_id: str = "",
    cost_status: str = "",
) -> None:
    """Append one background spend row. Never raises."""
    try:
        cost = float(cost_usd or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    try:
        with _LOCK:
            conn = _connect()
            try:
                conn.execute(
                    """
                    INSERT INTO background_spend
                        (ts, day, origin, cost_usd, total_tokens, api_calls,
                         model, provider, session_id, parent_session_id,
                         cost_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(),
                        _today(),
                        str(origin or "unknown"),
                        cost,
                        int(total_tokens or 0),
                        int(api_calls or 0),
                        str(model or ""),
                        str(provider or ""),
                        str(session_id or ""),
                        str(parent_session_id or ""),
                        str(cost_status or ""),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:  # pragma: no cover - ledger must not break the agent
        logger.debug("record_background_spend failed: %s", e)


def record_from_result(
    origin: str,
    conv_result: Any,
    *,
    parent_session_id: str = "",
) -> None:
    """Record spend from a ``run_conversation`` return dict (or agent attrs).

    ``conv_result`` is the dict returned by ``AIAgent.run_conversation``; we read
    the standard ``estimated_cost_usd`` / ``total_tokens`` / ``api_calls`` fields.
    """
    if not isinstance(conv_result, dict):
        return
    record_background_spend(
        origin,
        cost_usd=conv_result.get("estimated_cost_usd", 0.0) or 0.0,
        total_tokens=conv_result.get("total_tokens", 0) or 0,
        api_calls=conv_result.get("api_calls", 0) or 0,
        model=conv_result.get("model", "") or "",
        provider=conv_result.get("provider", "") or "",
        session_id=conv_result.get("session_id", "") or "",
        parent_session_id=parent_session_id or "",
        cost_status=conv_result.get("cost_status", "") or "",
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_daily_spend(day: Optional[str] = None, origin: Optional[str] = None) -> float:
    """Total background cost (USD) for a day (default: today)."""
    day = day or _today()
    try:
        conn = _connect()
        try:
            if origin:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM background_spend "
                    "WHERE day = ? AND origin = ?",
                    (day, origin),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM background_spend "
                    "WHERE day = ?",
                    (day,),
                ).fetchone()
            return float(row[0] or 0.0)
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover
        logger.debug("get_daily_spend failed: %s", e)
        return 0.0


def get_session_spend(parent_session_id: str) -> float:
    """Total background cost (USD) attributed to a parent session."""
    if not parent_session_id:
        return 0.0
    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM background_spend "
                "WHERE parent_session_id = ?",
                (parent_session_id,),
            ).fetchone()
            return float(row[0] or 0.0)
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover
        logger.debug("get_session_spend failed: %s", e)
        return 0.0


def get_today_breakdown() -> Dict[str, Dict[str, float]]:
    """Per-origin totals for today: ``{origin: {cost_usd, total_tokens, runs}}``."""
    out: Dict[str, Dict[str, float]] = {}
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT origin, COALESCE(SUM(cost_usd), 0), "
                "COALESCE(SUM(total_tokens), 0), COUNT(*) "
                "FROM background_spend WHERE day = ? GROUP BY origin",
                (_today(),),
            ).fetchall()
            for origin, cost, tokens, runs in rows:
                out[str(origin)] = {
                    "cost_usd": float(cost or 0.0),
                    "total_tokens": int(tokens or 0),
                    "runs": int(runs or 0),
                }
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover
        logger.debug("get_today_breakdown failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# Budget gate
# ---------------------------------------------------------------------------

def budget_status(*, parent_session_id: Optional[str] = None) -> Dict[str, Any]:
    """Full budget snapshot for callers that want to display it."""
    cfg = get_background_config()
    daily_spend = get_daily_spend()
    session_spend = (
        get_session_spend(parent_session_id) if parent_session_id else 0.0
    )
    allowed, reason = _evaluate(cfg, daily_spend, session_spend)
    return {
        "allowed": allowed,
        "reason": reason,
        "enabled": cfg["enabled"],
        "daily_spend_usd": daily_spend,
        "daily_limit_usd": cfg["daily_cost_limit_usd"],
        "session_spend_usd": session_spend,
        "session_limit_usd": cfg["session_cost_limit_usd"],
    }


def _evaluate(
    cfg: Dict[str, Any], daily_spend: float, session_spend: float
) -> Tuple[bool, str]:
    if not cfg["enabled"]:
        return False, "background work disabled (background.enabled=false)"
    daily_limit = cfg["daily_cost_limit_usd"]
    if daily_limit is not None and daily_spend >= daily_limit:
        return (
            False,
            f"daily background budget reached "
            f"(${daily_spend:.4f} ≥ ${daily_limit:.2f})",
        )
    session_limit = cfg["session_cost_limit_usd"]
    if session_limit is not None and session_spend >= session_limit:
        return (
            False,
            f"session background budget reached "
            f"(${session_spend:.4f} ≥ ${session_limit:.2f})",
        )
    return True, ""


def is_background_allowed(
    *, parent_session_id: Optional[str] = None
) -> Tuple[bool, str]:
    """Budget gate: ``(allowed, reason)``.

    Returns ``(True, "")`` when background work may proceed. On any internal
    error this fails *open* (allowed) — a broken ledger should degrade to the
    pre-existing behaviour, not silently disable self-improvement.
    """
    try:
        cfg = get_background_config()
        daily_spend = get_daily_spend()
        session_spend = (
            get_session_spend(parent_session_id) if parent_session_id else 0.0
        )
        return _evaluate(cfg, daily_spend, session_spend)
    except Exception as e:  # pragma: no cover
        logger.debug("is_background_allowed failed open: %s", e)
        return True, ""
