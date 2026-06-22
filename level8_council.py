
try:
    from debug_tools import (
        module_debug,
        module_exception,
        debug_every,
        debug_timer,
    )
except Exception:
    def module_debug(*args, **kwargs):
        pass
    def module_exception(*args, **kwargs):
        pass
    def debug_every(*args, **kwargs):
        pass
    class debug_timer:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

MODULE_NAME = __name__.split(".")[-1]
module_debug(
    MODULE_NAME,
    "module_loaded",
    data={"file": __file__},
    level="DEBUG",
    also_overall=False,
)
"""Level 8 trading council capital allocation and risk guidance."""

import csv
import json
import math
import os
import sqlite3
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")
MISSED_OPPORTUNITIES_CSV = os.path.join(BASE_DIR, "missed_opportunities.csv")
AGENT_PERFORMANCE_CSV = os.path.join(BASE_DIR, "agent_performance.csv")
BACKTEST_AGENT_PRIORS_CSV = os.path.join(BASE_DIR, "backtest_agent_priors.csv")
COUNCIL_OBSERVATION_OUTCOMES_CSV = os.path.join(BASE_DIR, "council_observation_outcomes.csv")
AGENT_ADJUSTMENTS_CSV = os.path.join(BASE_DIR, "agent_adjustments.csv")
ADAPTIVE_THRESHOLDS_CSV = os.path.join(BASE_DIR, "adaptive_thresholds.csv")
SHADOW_TRADES_CSV = os.path.join(BASE_DIR, "shadow_trades.csv")
AGENT_LEADERBOARD_CSV = os.path.join(BASE_DIR, "agent_leaderboard.csv")
AGENT_ABLATION_CSV = os.path.join(BASE_DIR, "agent_ablation.csv")
AGENT_TRADE_POLICY_CSV = os.path.join(BASE_DIR, "agent_trade_policy.csv")
AGENT_SIDE_RATINGS_CSV = os.path.join(BASE_DIR, "agent_side_ratings.csv")
FOUR_PASS_AGENT_CONTEXT_RATINGS_CSV = os.path.join(BASE_DIR, "four_pass_agent_context_ratings.csv")
AGENT_DECISION_INFLUENCE_CSV = os.path.join(BASE_DIR, "agent_decision_influence.csv")
PRODUCT_AGENT_INFLUENCE_CSV = os.path.join(BASE_DIR, "product_agent_influence.csv")
LEVEL8_EVENTS_DB = os.path.join(BASE_DIR, "level8_events.sqlite3")

BUY_LEAD_ONLY_MODE = False
BUY_LEAD_AGENT_NAMES = set()
BUY_LEAD_AGENT_FALLBACK_WEIGHT_PCT = {}
BUY_LEAD_AGENT_MIN_ROWS = 25

LEVEL8_CSV_TAIL_LIMITS = {
    "agent_performance.csv": 50000,
    "council_observation_outcomes.csv": 50000,
    "backtest_agent_priors.csv": 50000,
    "missed_opportunities.csv": 5000,
    "trades.csv": 5000,
    "agent_ablation.csv": 5000,
    "agent_side_ratings.csv": 5000,
    "four_pass_agent_context_ratings.csv": 20000,
    "agent_decision_influence.csv": 5000,
    "product_agent_influence.csv": 10000,
}

LEVEL8_CSV_USECOLS = {
    "agent_performance.csv": ["ts", "product_id", "agent", "strategy", "setup_tag", "market_regime", "execution_state", "outcome_source", "source", "outcome_move_bps", "move_bps", "outcome_success", "success", "agent_credit_score", "weighted_agent_credit_score"],
    "council_observation_outcomes.csv": ["ts", "product_id", "agent", "decision_strategy", "strategy", "setup_tag", "market_regime", "execution_state", "would_have_won", "success", "move_bps"],
    "backtest_agent_priors.csv": ["ts", "product_id", "agent", "strategy", "setup_tag", "market_regime", "execution_state", "outcome_move_bps", "move_bps", "outcome_success", "success", "outcome_source"],
    "missed_opportunities.csv": ["ts", "product_id", "move_bps", "decision_action", "decision_strategy"],
    "trades.csv": ["ts", "event", "product_id", "side", "entry_price", "exit_price", "price", "move_bps", "net_pnl_usd", "pnl", "agent"],
    "agent_side_ratings.csv": ["ts", "agent", "buy_rows", "buy_accuracy", "buy_score", "buy_weight_pct", "buy_weight_multiplier", "sell_rows", "sell_accuracy", "sell_score", "sell_weight_pct", "sell_weight_multiplier"],
    "four_pass_agent_context_ratings.csv": [
        "agent", "side", "product_id", "market_regime",
        "selected_count", "smoothed_win_rate", "ev_bps",
        "score", "weight_pct", "profitability_mode",
    ],
    "agent_decision_influence.csv": [
        "agent", "side", "selected_count", "frequency_per_day",
        "smoothed_win_rate", "ev_bps", "decision_weight_pct", "role",
    ],
    "product_agent_influence.csv": [
        "product_id", "market_regime", "agent", "side",
        "selected_count", "frequency_per_day",
        "smoothed_win_rate", "ev_bps", "decision_weight_pct", "role",
    ],
}


def _read_csv_tail_direct(path: str, max_lines: int, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            header = f.readline()
            tail_lines = deque(f, maxlen=max(1, int(max_lines)))
        if not header:
            return pd.DataFrame()
        text = header + "".join(tail_lines)
        if usecols:
            allowed = set(usecols)
            return pd.read_csv(StringIO(text), usecols=lambda c: c in allowed)
        return pd.read_csv(StringIO(text))
    except Exception:
        return pd.DataFrame()

# Initial council reliability priors.
# These are starting weights only. Live outcomes, sell outcomes, SHADOW outcomes,
# and backtest priors still adapt these over time.
INITIAL_AGENT_RELIABILITY_PRIORS = {
    # Core truth / economic authorities.
    "truth": 1.24,
    "exit_truth": 1.24,
    "utility_leader": 1.24,
    "sell_utility_leader": 1.22,
    "setup_performance_agent": 1.14,
    "order_book_liquidity_agent": 1.14,
    # Execution / risk authorities.
    "product_health": 1.10,
    "execution": 1.10,
    "risk": 1.12,
    "fee_recovery": 1.16,
    "drawdown_exit": 1.18,
    # Primary market map.
    "volume_profile_leader": 1.35,
    "volume_profile_leader_exit": 1.30,
    "volume_profile_agent": 1.05,
    "volume_profile_harvest": 1.06,
    # Institutional/session memory.
    "previous_session_volume_profile_agent": 1.22,
    "previous_session_profile_exit": 1.18,
    "previous_session_profile_agent": 1.08,
    # Quant/statistical reality check.
    # Starts useful, but below volume until it proves live edge.
    "quant_boundary_agent": 1.08,
    "quant_boundary_exit": 1.08,
    "ai_outcome": 0.92,
    # Session / structure / liquidity confirmation.
    "market_structure_agent": 1.10,
    "validated_liquidity_agent": 1.08,
    "candle_context_agent": 1.04,
    "candle_sequence_agent": 0.98,
    "candle_exhaustion_sell": 1.12,
    "fresh_zone_retest_agent": 1.00,
    "fair_value_gap_agent": 1.00,
    "fvg_reclaim_rejection_exit": 1.04,
    # Cross-asset context.
    "smt_divergence_agent": 0.88,
    "smt_divergence_exit": 0.92,
    # Exit mechanics.
    "profit_capture": 1.18,
    "peak_capture": 1.20,
    "momentum_fade": 1.12,
    "continuation_hold": 1.10,
    "harvest_sizing": 1.12,
    "spike_profit_protection": 1.22,
    # Older generic modes remain low until proven.
    "trend": 0.96,
    "mean_reversion": 0.92,
    "breakout": 0.92,
    "exploration": 0.28,
}

INITIAL_AGENT_RELIABILITY_PRIORS.update({
    "setup_pattern_edge_agent": 1.38,
    "market_structure_reclaim_agent": 1.34,
    "score_band_anti_chase_agent": 1.22,
    "product_edge_governor_agent": 1.18,
    "clean_path_analog_agent": 1.36,
    "bad_setup_veto_agent": 1.30,
    "volume_chop_veto_agent": 1.24,
    "quant_regime_veto_agent": 1.20,
    "execution_quality_gate_agent": 1.18,
    "profit_pullback_capture_agent": 1.35,
    "higher_low_wave_stop_agent": 1.32,
    "failed_entry_escape_agent": 1.28,
    "hard_stop_prevention_agent": 1.26,
    "max_hold_decay_agent": 1.16,
    "volume_profile_leader": 0.45,
    "volume_profile_agent": 0.50,
    "quant_boundary_agent": 0.55,
    "fresh_zone_retest_agent": 0.45,
    "fair_value_gap_agent": 0.62,
    "trend": 0.45,
    "mean_reversion": 0.45,
    "breakout": 0.45,
    "execution": 0.55,
    "order_book_liquidity_agent": 0.55,
    "exploration": 0.10,
})


INITIAL_AGENT_RELIABILITY_PRIORS.update({
    "bayesian_setup_pattern_edge_agent": 1.45,
    "calibrated_logistic_meta_agent": 1.42,
    "tree_regime_agent": 1.24,
    "market_structure_reclaim_agent": 1.34,
    "validated_liquidity_confirmer_agent": 1.26,
    "score_band_anti_chase_agent": 1.22,
    "product_edge_governor_agent": 1.24,
    "clean_path_analog_gate_agent": 1.38,
    "volume_chop_veto_agent": 1.24,
    "quant_regime_veto_agent": 1.20,
    "execution_cost_gate_agent": 1.26,
    "profit_pullback_capture_agent": 1.40,
    "higher_low_wave_stop_agent": 1.34,
    "failed_entry_hazard_escape_agent": 1.32,
    "hard_stop_prevention_agent": 1.30,
    "max_hold_decay_agent": 1.16,
    "volume_profile_leader": 0.35, "volume_profile_agent": 0.40, "quant_boundary_agent": 0.45,
    "fresh_zone_retest_agent": 0.45, "fair_value_gap_agent": 0.50, "trend": 0.40,
    "mean_reversion": 0.40, "breakout": 0.40, "execution": 0.50,
    "order_book_liquidity_agent": 0.50, "exploration": 0.05,
})


INITIAL_AGENT_RELIABILITY_PRIORS.update({
    "volume_profile_leader": 0.10,
    "volume_profile_agent": 0.10,
    "quant_boundary_agent": 0.10,
    "fresh_zone_retest_agent": 0.05,
    "fair_value_gap_agent": 0.05,
    "trend": 0.05,
    "mean_reversion": 0.05,
    "breakout": 0.05,
    "execution": 0.05,
    "order_book_liquidity_agent": 0.05,
    "exploration": 0.00,
    "reclaimed_value_low_reversal_agent": 1.40,
    "inside_fair_fvg_retest_agent": 1.40,
    "poc_compression_release_agent": 1.32,
    "high_volume_absorption_agent": 1.24,
    "liquidity_sweep_reclaim_agent": 1.34,
    "chart_analog_similarity_agent": 1.45,
    "bad_intersection_veto_agent": 1.50,
    "execution_cost_gate_agent": 1.20,
    "profit_pullback_wave_agent": 1.45,
    "higher_low_wave_stop_agent": 1.38,
    "wick_exhaustion_sell_agent": 1.25,
    "liquidity_target_hit_agent": 1.30,
    "failed_run_escape_agent": 1.35,
    "analog_sell_path_agent": 1.40,
})
# ============================================================
# AGENT PRIORITY / REDUNDANCY POLICY
# ============================================================
AGENT_SHRINKAGE_TARGET_N: float = 80.0
AGENT_STRONG_SAMPLE_N: float = 150.0
AGENT_UNPROVEN_MAX_DIRECTIONAL_ADJ: float = 0.07
AGENT_PROVEN_MAX_DIRECTIONAL_ADJ: float = 0.24

BUY_REDUNDANCY_GROUP_CAPS = {
    "institutional_alpha": 0.48,
    "institutional_veto": 0.36,
    "economics": 0.30,
    "execution": 0.24,
    "legacy_context": 0.12,
    "risk": 0.24,
    "learning": 0.18,
    "other": 0.10,
}
SELL_REDUNDANCY_GROUP_CAPS = {
    "institutional_sell_alpha": 0.50,
    "profit_capture": 0.38,
    "risk_exit": 0.34,
    "execution": 0.20,
    "legacy_context": 0.12,
    "learning": 0.14,
    "other": 0.10,
}


def agent_redundancy_group(agent: str) -> str:
    """Return the evidence family for redundancy control."""
    text = str(agent or "").lower()
    if text in {"bayesian_setup_pattern_edge_agent", "calibrated_logistic_meta_agent", "tree_regime_agent", "market_structure_reclaim_agent", "validated_liquidity_confirmer_agent", "score_band_anti_chase_agent", "product_edge_governor_agent", "clean_path_analog_gate_agent"}:
        return "institutional_alpha"
    if text in {"volume_chop_veto_agent", "quant_regime_veto_agent", "execution_cost_gate_agent"}:
        return "institutional_veto"
    if text in {"profit_pullback_capture_agent", "higher_low_wave_stop_agent", "failed_entry_hazard_escape_agent", "hard_stop_prevention_agent", "max_hold_decay_agent"}:
        return "institutional_sell_alpha"
    if text in {"volume_profile_leader", "volume_profile_agent", "quant_boundary_agent", "fresh_zone_retest_agent", "fair_value_gap_agent", "trend", "mean_reversion", "breakout", "execution", "order_book_liquidity_agent"}:
        return "legacy_context"
    if text in {"setup_pattern_edge_agent", "market_structure_reclaim_agent", "score_band_anti_chase_agent", "product_edge_governor_agent", "clean_path_analog_agent"}:
        return "buy_alpha"
    if text in {"bad_setup_veto_agent", "volume_chop_veto_agent", "quant_regime_veto_agent", "execution_quality_gate_agent"}:
        return "buy_veto"
    if text in {"profit_pullback_capture_agent", "higher_low_wave_stop_agent", "failed_entry_escape_agent", "hard_stop_prevention_agent", "max_hold_decay_agent"}:
        return "sell_alpha"
    if text in {"truth", "exit_truth", "utility_leader", "sell_utility_leader"}:
        return "economics"
    if "volume_profile" in text or text in {"volume_profile_agent", "volume_profile_harvest"}:
        return "volume"
    if "previous_session" in text or "prior_session" in text:
        return "previous_session"
    if "quant" in text or "stationarity" in text or "forecast" in text:
        return "quant"
    if "order_book" in text or text == "order_book_liquidity_agent":
        return "risk_execution"
    if (
        "validated_liquidity" in text
        or "candle" in text
        or "structure" in text
        or "fresh_zone" in text
        or "fair_value_gap" in text
        or "fvg" in text
    ):
        return "price_action"
    if "session" in text or "sweep" in text or "breakout_continuation" in text:
        return "session_liquidity"
    if "smt" in text or "peer" in text or "relative" in text:
        return "cross_asset"
    if text in {"risk", "execution", "product_health", "fee_recovery", "drawdown_exit"}:
        return "risk_execution"
    if text in {"profit_capture", "peak_capture", "momentum_fade", "continuation_hold", "harvest_sizing", "spike_profit_protection"}:
        return "profit_capture"
    if "setup_performance" in text or "ai" in text or "learning" in text:
        return "learning"
    return "other"


def dominant_vote_direction(vote: Dict[str, Any]) -> str:
    values = {
        "buy": float(vote.get("buy", vote.get("raw_buy_score", 0.0)) or 0.0),
        "sell": float(vote.get("sell", vote.get("raw_sell_score", 0.0)) or 0.0),
        "hold": float(vote.get("hold", vote.get("raw_hold_score", 0.0)) or 0.0),
        "wait": float(vote.get("wait", vote.get("raw_wait_score", 0.0)) or 0.0),
    }
    return max(values, key=values.get)


def initial_agent_reliability_prior(agent: str) -> float:
    agent_text = str(agent or "")
    if agent_text in INITIAL_AGENT_RELIABILITY_PRIORS:
        return float(INITIAL_AGENT_RELIABILITY_PRIORS[agent_text])
    if agent_text.endswith("_sweep_reversal"):
        return 1.04
    if agent_text.endswith("_breakout_continuation"):
        return 0.98
    if agent_text.endswith("_liquidity_harvest"):
        return 0.88
    if agent_text.endswith("_sell_context"):
        return 1.06
    return 1.00


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` to the inclusive range bounded by minimum and maximum."""
    return max(minimum, min(maximum, value))


def utc_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def utc_dt(ts: Optional[float] = None) -> str:
    value = float(ts if ts is not None else utc_ts())
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def append_sqlite_event(
    *,
    event_type: str,
    source_path: str,
    row: Dict[str, Any],
) -> None:
    """
    Durable Level 8 event mirror.

    CSV remains the viewer-friendly format.
    SQLite becomes the safer long-term learning/event ledger.
    """
    try:
        payload = json.dumps(row, default=str)

        conn = sqlite3.connect(LEVEL8_EVENTS_DB)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS level8_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    dt_utc TEXT,
                    event_type TEXT,
                    source_path TEXT,
                    decision_id TEXT,
                    product_id TEXT,
                    agent TEXT,
                    strategy TEXT,
                    payload_json TEXT
                )
                """
            )

            conn.execute(
                """
                INSERT INTO level8_events (
                    ts, dt_utc, event_type, source_path, decision_id,
                    product_id, agent, strategy, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    float(row.get("ts", utc_ts()) or utc_ts()),
                    str(row.get("dt_utc", utc_dt())),
                    event_type,
                    os.path.basename(source_path),
                    str(row.get("decision_id", "")),
                    str(row.get("product_id", "")),
                    str(row.get("agent", "")),
                    str(row.get("strategy", "")),
                    payload,
                ),
            )

            conn.commit()
        finally:
            conn.close()

    except Exception:
        pass


def append_csv_row(path: str, columns: list[str], row: Dict[str, Any]) -> None:
    """
    Append a CSV row and mirror only high-value Level 8 events to SQLite.

    Do NOT mirror high-frequency agent_adjustments / agent_leaderboard rows to
    SQLite on every vote. Those rows are useful in CSV for the viewer, but they
    can overwhelm the bot if every agent vote opens SQLite and writes a row.
    """
    exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(columns)

        writer.writerow([row.get(column, "") for column in columns])

    basename = os.path.basename(path)

    # Keep SQLite for higher-value event ledgers.
    # Skip the very noisy per-vote adjustment/leaderboard telemetry.
    noisy_sqlite_skip = {
        "agent_adjustments.csv",
        "agent_leaderboard.csv",
    }

    if basename in noisy_sqlite_skip:
        return

    append_sqlite_event(
        event_type=os.path.splitext(basename)[0],
        source_path=path,
        row=row,
    )


@dataclass
class AgentVote:
    """Normalized council vote, including its outcome-based adjustments."""

    agent: str
    buy: float
    sell: float
    hold: float
    wait: float
    confidence: float
    reliability: float = 0.80

    adjusted_buy_score: float = 0.0
    adjusted_sell_score: float = 0.0
    adjusted_hold_score: float = 0.0
    adjusted_wait_score: float = 0.0

    product_adjustment: float = 0.0
    strategy_adjustment: float = 0.0
    recent_performance_adjustment: float = 0.0
    weight: float = 0.0
    leaderboard_rank: float = 999.0
    leaderboard_score: float = 0.5
    leader_bonus: float = 0.0
    leader_penalty: float = 0.0
    reason: str = ""
    product_id: str = ""
    market_regime: str = "unknown"
    momentum_15_bps: float = 0.0
    momentum_30_bps: float = 0.0
    volatility_bps: float = 0.0
    atr_bps: float = 0.0
    range_bps: float = 0.0


class Level8Council:
    """Outcome-adaptive council with an 80% maximum portfolio deployment."""

    def __init__(self) -> None:
        # High-conviction buying:
        # fewer live buys, stronger proof required.
        self.base_buy_threshold = 0.38
        self.base_sell_threshold = 0.44

        self.min_buy_threshold = 0.30
        self.max_buy_threshold = 0.82
        self.min_sell_threshold = 0.30
        self.max_sell_threshold = 0.76

        self.max_agent_adjustment = 0.32
        self.min_agent_reliability = 0.20
        self.max_agent_reliability = 1.65

        self.min_truth_to_trade = 0.55
        self.min_truth_to_core_trade = 0.72

        # Portfolio allocation model.
        self.reserve_bucket_pct = 0.00
        self.max_single_asset_pct = 1.00
        self.max_total_exposure_pct = 1.00

        # Council-controlled high-conviction sizing.
        # Weak setups should be SHADOW, not tiny live tests.
        # Approved setups start at 50%.
        # Approved trades start at 50%.
        # High-confidence CORE trades start at 80% and can scale to 100%.
        self.test_bucket_trade_pct = 0.50
        self.min_core_trade_pct = 0.80
        self.max_core_trade_pct = 1.00

        # These are descriptive only now; they do not hard-block spending.
        self.test_bucket_pct = 0.10
        self.core_bucket_pct = 0.70

        self.last_summary: Dict[str, Any] = {}
        self._agent_leaderboard_cache: Dict[str, Dict[str, float]] = {}
        self._agent_leaderboard_cache_ts: float = 0.0
        self._agent_leaderboard_cache_sec: float = 60.0
        self._agent_side_ratings_cache: Dict[str, Dict[str, float]] = {}
        self._agent_side_ratings_cache_ts: float = 0.0
        self._agent_side_ratings_cache_sec: float = 60.0
        self._agent_context_ratings_cache: Dict[str, Dict[str, Any]] = {}
        self._agent_context_ratings_cache_ts: float = 0.0
        self._agent_context_ratings_cache_sec: float = 60.0
        self._agent_decision_influence_cache: Dict[str, Dict[str, Any]] = {}
        self._agent_decision_influence_cache_ts: float = 0.0
        self._agent_decision_influence_cache_sec: float = 60.0
        self._product_agent_influence_cache: Dict[str, Dict[str, Any]] = {}
        self._product_agent_influence_cache_ts: float = 0.0
        self._product_agent_influence_cache_sec: float = 60.0
        self._agent_ablation_cache: Dict[str, Dict[str, float]] = {}
        self._agent_ablation_cache_ts: float = 0.0
        # Lightweight in-process caches prevent every agent vote from re-reading
        # the same large CSV files over and over.
        self._csv_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._outcome_stats_cache: Dict[Tuple[str, str, str], Tuple[float, Dict[str, float]]] = {}
        self._csv_cache_sec: float = 60.0
        self._outcome_stats_cache_sec: float = 60.0

    def _neutral_stats(self, reason: str = "no_matching_data") -> Dict[str, float]:
        return {
            "n": 0.0,
            "win_rate": 0.5,
            "avg_move": 0.0,
            "avg_adverse": 0.0,
            "avg_credit": 0.5,
            "weighted_credit": 0.5,
            "real_trade_n": 0.0,
            "observation_n": 0.0,
            "reason": reason,
        }

    def _read_csv_cached(self, path: str, *, ttl_sec: Optional[float] = None) -> pd.DataFrame:
        """
        Read runtime learning CSVs with a short TTL cache and tail limits.
        This prevents Level 8 from repeatedly loading huge CSVs into memory.
        """
        try:
            ttl = float(ttl_sec if ttl_sec is not None else self._csv_cache_sec)
            now_value = utc_ts()
            basename = os.path.basename(path)
            cached = self._csv_cache.get(path)

            if cached is not None:
                cached_ts, cached_frame = cached
                if now_value - float(cached_ts) <= ttl:
                    return cached_frame.copy(deep=False)

            if not os.path.exists(path):
                frame = pd.DataFrame()
            else:
                max_lines = int(LEVEL8_CSV_TAIL_LIMITS.get(basename, 20000))
                usecols = LEVEL8_CSV_USECOLS.get(basename)
                try:
                    size_bytes = os.path.getsize(path)
                except Exception:
                    size_bytes = 0

                if size_bytes > 5_000_000 or basename in LEVEL8_CSV_TAIL_LIMITS:
                    frame = _read_csv_tail_direct(path, max_lines=max_lines, usecols=usecols)
                else:
                    if usecols:
                        allowed = set(usecols)
                        frame = pd.read_csv(path, usecols=lambda c: c in allowed)
                    else:
                        frame = pd.read_csv(path)

            self._csv_cache[path] = (now_value, frame)
            return frame.copy(deep=False)

        except Exception:
            return pd.DataFrame()

    def _clear_level8_memory_cache(self) -> None:
        """Clear short-lived caches after heavy learning updates if needed."""
        self._csv_cache = {}
        self._outcome_stats_cache = {}

    def _missed_opportunity_relief(self, product_id: str) -> float:
        """Reduce strictness after repeated WAIT/SHADOW decisions missed jumps."""
        try:
            if not os.path.exists(MISSED_OPPORTUNITIES_CSV):
                return 0.0
            frame = self._read_csv_cached(MISSED_OPPORTUNITIES_CSV, ttl_sec=20.0)
            if frame.empty or "product_id" not in frame.columns:
                return 0.0
            frame = frame[
                frame["product_id"].astype(str) == str(product_id)
            ].copy()
            if frame.empty:
                return 0.0
            frame["move_bps"] = pd.to_numeric(
                frame["move_bps"], errors="coerce"
            ).fillna(0.0)
            recent = frame.tail(20)
            big_misses = int((recent["move_bps"] >= 120.0).sum())
            huge_misses = int((recent["move_bps"] >= 250.0).sum())
            # Missed jumps should strongly teach the council that it was too strict.
            relief = big_misses * 0.018 + huge_misses * 0.030
            return clamp(relief, 0.0, 0.20)
        except Exception:
            return 0.0

    def _recent_trades(self, lookback_rows: int = 80) -> pd.DataFrame:
        """Return recent trades, tolerating absent or malformed history."""
        trades = self._read_csv_cached(TRADES_CSV, ttl_sec=20.0)

        if trades.empty:
            return pd.DataFrame()

        try:
            if "ts" in trades.columns:
                trades["ts"] = pd.to_numeric(trades["ts"], errors="coerce")
                trades = trades.sort_values("ts")

            return trades.tail(lookback_rows).copy()

        except Exception:
            return pd.DataFrame()

    def session_health(self) -> Dict[str, Any]:
        """Summarize session outcomes without imposing a hard pause mode."""
        trades = self._recent_trades(80)
        if trades.empty or "net_pnl_usd" not in trades.columns:
            summary = {
                "risk_mode": "NORMAL",
                "session_net": 0.0,
                "closed_count": 0,
                "loss_streak": 0,
                "reason": "no_recent_trade_data",
            }
            self.last_summary = summary
            return summary

        trades["net_pnl_usd"] = pd.to_numeric(
            trades["net_pnl_usd"], errors="coerce"
        ).fillna(0.0)
        if "event" in trades.columns:
            sells = trades[
                trades["event"].astype(str).str.upper() == "SELL"
            ].copy()
        else:
            sells = pd.DataFrame(columns=trades.columns)

        session_net = float(trades["net_pnl_usd"].sum())
        closed_count = int(len(sells))
        loss_streak = 0
        if not sells.empty:
            if "ts" in sells.columns:
                sells = sells.sort_values("ts", ascending=False)
            for _, row in sells.iterrows():
                if float(row.get("net_pnl_usd", 0.0)) < 0:
                    loss_streak += 1
                else:
                    break

        if loss_streak >= 4 or session_net <= -2.00:
            risk_mode = "DEFENSIVE"
        elif loss_streak >= 2 or session_net <= -1.00:
            risk_mode = "CAUTIOUS"
        elif session_net >= 0.75 and loss_streak == 0:
            risk_mode = "AGGRESSIVE"
        else:
            risk_mode = "NORMAL"

        summary = {
            "risk_mode": risk_mode,
            "session_net": session_net,
            "closed_count": closed_count,
            "loss_streak": loss_streak,
            "reason": (
                f"session_net={session_net:.4f};loss_streak={loss_streak};"
                f"closed={closed_count}"
            ),
        }
        self.last_summary = summary
        return summary

    def risk_agent(
        self,
        risk_mode: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Return risk-agent votes that influence rather than veto the council."""
        mode = str(
            risk_mode or self.session_health().get("risk_mode", "NORMAL")
        ).upper()

        if mode == "DEFENSIVE":
            buy, sell, hold, wait = 0.38, 0.58, 0.48, 0.62
            conf = 0.80
        elif mode == "CAUTIOUS":
            buy, sell, hold, wait = 0.44, 0.52, 0.50, 0.56
            conf = 0.65
        elif mode == "AGGRESSIVE":
            buy, sell, hold, wait = 0.72, 0.35, 0.62, 0.25
            conf = 0.65
        else:
            buy, sell, hold, wait = 0.55, 0.42, 0.55, 0.40
            conf = 0.50

        return {
            "agent": "risk",
            "risk_mode": mode,
            "buy": buy,
            "sell": sell,
            "hold": hold,
            "wait": wait,
            "confidence": conf,
        }

    def _outcome_stats(
        self,
        *,
        agent: Optional[str] = None,
        product_id: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Summarize real trades, chart-only outcomes, and agent-specific credit.

        Important behavior:
        - Real filled trade outcomes are weighted more heavily than observations.
        - Heartbeat/observation outcomes still teach, but they do not overwhelm fills.
        - Missing agent/product/strategy data returns neutral stats instead of silently
          falling back to broad unrelated data.
        """
        cache_key = (
            str(agent) if agent is not None else "",
            str(product_id) if product_id is not None else "",
            str(strategy) if strategy is not None else "",
        )

        now_value = utc_ts()
        cached_stats = self._outcome_stats_cache.get(cache_key)

        if cached_stats is not None:
            cached_ts, cached_result = cached_stats
            if now_value - float(cached_ts) <= float(self._outcome_stats_cache_sec):
                return dict(cached_result)

        frames = []

        try:
            if os.path.exists(AGENT_PERFORMANCE_CSV):
                perf = self._read_csv_cached(AGENT_PERFORMANCE_CSV, ttl_sec=20.0)

                if not perf.empty:
                    perf = perf.rename(columns={
                        "outcome_move_bps": "move_bps",
                        "outcome_success": "success",
                    })

                    if "outcome_source" in perf.columns:
                        perf["source"] = perf["outcome_source"].astype(str)
                    elif "source" not in perf.columns:
                        perf["source"] = "agent_performance"

                    frames.append(perf)
        except Exception:
            pass

        try:
            if os.path.exists(BACKTEST_AGENT_PRIORS_CSV):
                priors = self._read_csv_cached(BACKTEST_AGENT_PRIORS_CSV, ttl_sec=20.0)

                if not priors.empty:
                    priors = priors.rename(columns={
                        "outcome_move_bps": "move_bps",
                        "outcome_success": "success",
                    })

                    if "outcome_source" in priors.columns:
                        priors["source"] = priors["outcome_source"].astype(str)
                    elif "source" not in priors.columns:
                        priors["source"] = "backtest_profit_replay"

                    frames.append(priors)
        except Exception:
            pass

        try:
            if os.path.exists(COUNCIL_OBSERVATION_OUTCOMES_CSV):
                obs = self._read_csv_cached(COUNCIL_OBSERVATION_OUTCOMES_CSV, ttl_sec=20.0)

                if not obs.empty:
                    obs = obs.rename(columns={
                        "decision_strategy": "strategy",
                        "would_have_won": "success",
                    })
                    obs["agent"] = obs.get("agent", "council_observation")
                    obs["source"] = "observation_outcome"
                    frames.append(obs)
        except Exception:
            pass

        try:
            trades = self._recent_trades(240)

            if not trades.empty:
                # Do not treat dollar P&L as basis points. Use true move_bps when
                # available, derive it from entry/exit prices when possible, and
                # reserve dollar P&L for win/loss classification.
                if "move_bps" in trades.columns:
                    trades["move_bps"] = pd.to_numeric(
                        trades["move_bps"], errors="coerce"
                    ).fillna(0.0)
                elif "entry_price" in trades.columns and "exit_price" in trades.columns:
                    entry_px = pd.to_numeric(trades["entry_price"], errors="coerce")
                    exit_px = pd.to_numeric(trades["exit_price"], errors="coerce")
                    trades["move_bps"] = (((exit_px / entry_px) - 1.0) * 10000.0).replace(
                        [pd.NA, pd.NaT, float("inf"), float("-inf")],
                        0.0,
                    ).fillna(0.0)
                else:
                    trades["move_bps"] = 0.0

                if "net_pnl_usd" in trades.columns:
                    pnl = pd.to_numeric(trades["net_pnl_usd"], errors="coerce").fillna(0.0)
                    trades["success"] = (pnl > 0.0).astype(int)
                else:
                    trades["success"] = (
                        pd.to_numeric(trades["move_bps"], errors="coerce").fillna(0.0) > 0.0
                    ).astype(int)

                trades["source"] = "real_trade"
                frames.append(trades)
        except Exception:
            pass

        if not frames:
            return self._neutral_stats("no_frames")

        data = pd.concat(frames, ignore_index=True, sort=False)

        try:
            if "ts" in data.columns:
                data["ts_num"] = pd.to_numeric(data.get("ts"), errors="coerce")
                data = data.sort_values("ts_num").tail(50000)
        except Exception:
            data = data.tail(50000)

        if data.empty:
            return self._neutral_stats("empty_data")

        for column, value in (
            ("agent", agent),
            ("product_id", product_id),
            ("strategy", strategy),
        ):
            if value is not None:
                if column not in data.columns:
                    return self._neutral_stats(f"missing_column:{column}")

                rows = data[data[column].astype(str) == str(value)].copy()

                if rows.empty:
                    return self._neutral_stats(f"no_match:{column}={value}")

                data = rows

        if data.empty:
            return self._neutral_stats("empty_after_filters")

        move_source = data["move_bps"] if "move_bps" in data.columns else pd.Series(0.0, index=data.index)
        move = pd.to_numeric(move_source, errors="coerce").fillna(0.0)

        if "weighted_agent_credit_score" in data.columns:
            credit = pd.to_numeric(data["weighted_agent_credit_score"], errors="coerce").fillna(0.5)
            success = (credit >= 0.5).astype(int)
        elif "agent_credit_score" in data.columns:
            credit = pd.to_numeric(data["agent_credit_score"], errors="coerce").fillna(0.5)
            success = (credit >= 0.5).astype(int)
        elif "success" in data.columns:
            success = pd.to_numeric(data["success"], errors="coerce").fillna((move > 0).astype(int))
            credit = success.astype(float)
        else:
            success = (move > 0).astype(int)
            credit = success.astype(float)

        adverse_col = next(
            (
                c for c in (
                    "adverse_move_bps",
                    "max_adverse_bps",
                    "avg_adverse",
                    "adverse",
                )
                if c in data.columns
            ),
            None,
        )

        if adverse_col:
            adverse = pd.to_numeric(data[adverse_col], errors="coerce").fillna(0.0).abs()
        else:
            adverse = pd.Series(0.0, index=data.index)

        source = data["source"].astype(str) if "source" in data.columns else pd.Series("unknown", index=data.index)
        source_weight = source.map({
            "real_trade": 1.35,
            "trade_outcome": 1.35,
            "sell_outcome": 1.30,
            "backtest_profit_replay": 0.90,
            "agent_performance": 0.80,
            "level8_observation": 0.45,
            "observation_outcome": 0.35,
            "unknown": 0.35,
        }).fillna(0.35).astype(float)

        if source_weight.sum() > 0:
            weighted_credit = float((credit * source_weight).sum() / source_weight.sum())
            weighted_success = float((success * source_weight).sum() / source_weight.sum())
            weighted_move = float((move * source_weight).sum() / source_weight.sum())
            weighted_adverse = float((adverse * source_weight).sum() / source_weight.sum())
        else:
            weighted_credit = 0.5
            weighted_success = 0.5
            weighted_move = 0.0
            weighted_adverse = 0.0

        real_trade_n = float(source.isin(["real_trade", "trade_outcome", "sell_outcome"]).sum())
        backtest_prior_n = float(source.isin(["backtest_profit_replay"]).sum())
        observation_n = float(source.isin(["level8_observation", "observation_outcome"]).sum())

        result = {
            "n": float(len(data)),
            "win_rate": weighted_success,
            "avg_move": weighted_move,
            "avg_adverse": weighted_adverse,
            "avg_credit": float(credit.mean()),
            "weighted_credit": weighted_credit,
            "real_trade_n": real_trade_n,
            "backtest_prior_n": backtest_prior_n,
            "observation_n": observation_n,
            "reason": "weighted_stats_live_outcomes_above_priors",
        }

        self._outcome_stats_cache[cache_key] = (now_value, dict(result))
        return result

    def _leaderboard_bonus_penalty(
        self,
        *,
        rank: float,
        score: float,
        sample_size: float,
    ) -> Tuple[float, float]:
        """Convert leaderboard rank into a bounded influence adjustment."""
        leader_bonus = 0.0
        leader_penalty = 0.0

        if sample_size >= 25:
            if rank == 1 and score > 0.58:
                leader_bonus = 0.060
            elif rank <= 3 and score > 0.55:
                leader_bonus = 0.035
            elif score < 0.45:
                leader_penalty = 0.050
            elif score < 0.49:
                leader_penalty = 0.025

        return leader_bonus, leader_penalty

    def _refresh_agent_leaderboard_cache(self, *, force: bool = False) -> None:
        """Rebuild and log the competitive leaderboard at most once per cache window."""
        now_value = utc_ts()

        if (
            not force
            and self._agent_leaderboard_cache
            and now_value - float(self._agent_leaderboard_cache_ts) < float(self._agent_leaderboard_cache_sec)
        ):
            return

        self._agent_leaderboard_cache_ts = now_value
        self._agent_leaderboard_cache = {}

        try:
            if not os.path.exists(AGENT_PERFORMANCE_CSV):
                return

            frame = self._read_csv_cached(AGENT_PERFORMANCE_CSV, ttl_sec=30.0)

            if frame.empty or "agent" not in frame.columns:
                return

            credit_col = (
                "weighted_agent_credit_score"
                if "weighted_agent_credit_score" in frame.columns
                else "agent_credit_score"
            )

            if credit_col not in frame.columns:
                return

            frame[credit_col] = pd.to_numeric(frame[credit_col], errors="coerce").fillna(0.5)

            if "outcome_source" in frame.columns:
                source = frame["outcome_source"].astype(str)
            else:
                source = pd.Series("unknown", index=frame.index)

            frame["_source_weight"] = source.map({
                "trade_outcome": 1.00,
                "real_trade": 1.00,
                "sell_outcome": 1.00,
                "agent_performance": 0.80,
                "observation_outcome": 0.40,
                "level8_observation": 0.40,
                "unknown": 0.35,
            }).fillna(0.35).astype(float)

            rows = []

            for name, group in frame.groupby(frame["agent"].astype(str)):
                sample_size = float(len(group))

                if sample_size <= 0:
                    continue

                weighted_credit = float(
                    (group[credit_col] * group["_source_weight"]).sum()
                    / max(group["_source_weight"].sum(), 1e-9)
                )

                recent = group.tail(50)
                recent_credit = (
                    float(recent[credit_col].mean())
                    if not recent.empty
                    else weighted_credit
                )

                sample_factor = clamp(sample_size / 50.0, 0.0, 1.0)

                leaderboard_score = clamp(
                    weighted_credit * 0.70
                    + recent_credit * 0.20
                    + sample_factor * 0.10,
                    0.0,
                    1.0,
                )

                rows.append({
                    "agent": str(name),
                    "sample_size": sample_size,
                    "weighted_credit": weighted_credit,
                    "recent_credit": recent_credit,
                    "leaderboard_score": leaderboard_score,
                })

            if not rows:
                return

            board = (
                pd.DataFrame(rows)
                .sort_values("leaderboard_score", ascending=False)
                .reset_index(drop=True)
            )
            board["leaderboard_rank"] = board.index + 1

            for _, row in board.iterrows():
                agent_name = str(row["agent"])
                rank = float(row["leaderboard_rank"])
                score = float(row["leaderboard_score"])
                sample_size = float(row["sample_size"])
                leader_bonus, leader_penalty = self._leaderboard_bonus_penalty(
                    rank=rank,
                    score=score,
                    sample_size=sample_size,
                )

                record = {
                    "leaderboard_rank": rank,
                    "leaderboard_score": score,
                    "leader_bonus": leader_bonus,
                    "leader_penalty": leader_penalty,
                    "sample_size": sample_size,
                    "weighted_credit": float(row["weighted_credit"]),
                    "recent_credit": float(row["recent_credit"]),
                }

                self._agent_leaderboard_cache[agent_name] = record

                try:
                    append_csv_row(
                        AGENT_LEADERBOARD_CSV,
                        [
                            "ts", "dt_utc", "agent", "leaderboard_rank",
                            "leaderboard_score", "weighted_credit", "recent_credit",
                            "sample_size", "leader_bonus", "leader_penalty", "reason",
                        ],
                        {
                            "ts": f"{now_value:.6f}",
                            "dt_utc": utc_dt(now_value),
                            "agent": agent_name,
                            "leaderboard_rank": f"{rank:.0f}",
                            "leaderboard_score": f"{score:.6f}",
                            "weighted_credit": f"{float(row['weighted_credit']):.6f}",
                            "recent_credit": f"{float(row['recent_credit']):.6f}",
                            "sample_size": f"{sample_size:.0f}",
                            "leader_bonus": f"{leader_bonus:.6f}",
                            "leader_penalty": f"{leader_penalty:.6f}",
                            "reason": (
                                f"competitive_agent_goal_cached;"
                                f"rank={rank:.0f};score={score:.3f};"
                                f"bonus={leader_bonus:.3f};penalty={leader_penalty:.3f}"
                            ),
                        },
                    )
                except Exception:
                    pass

        except Exception:
            self._agent_leaderboard_cache = {}

    def _agent_competition_score(self, agent: str) -> Dict[str, float]:
        """Return the current competitive score for an agent."""
        neutral = {
            "leaderboard_rank": 999.0,
            "leaderboard_score": 0.5,
            "leader_bonus": 0.0,
            "leader_penalty": 0.0,
            "sample_size": 0.0,
            "weighted_credit": 0.5,
            "recent_credit": 0.5,
        }

        try:
            self._refresh_agent_leaderboard_cache(force=False)
            return dict(self._agent_leaderboard_cache.get(str(agent), neutral))
        except Exception:
            return neutral

    def _agent_ablation_score(self, agent: str) -> Dict[str, float]:
        neutral = {
            "ablation_score": 0.0,
            "support_edge_bps": 0.0,
            "sample_count": 0.0,
            "weight_adjust": 0.0,
        }
        try:
            now_value = utc_ts()
            if (
                self._agent_ablation_cache
                and now_value - float(self._agent_ablation_cache_ts) < 300.0
            ):
                return dict(self._agent_ablation_cache.get(str(agent), neutral))
            self._agent_ablation_cache = {}
            self._agent_ablation_cache_ts = now_value
            if not os.path.exists(AGENT_ABLATION_CSV):
                return neutral
            frame = pd.read_csv(AGENT_ABLATION_CSV)
            if frame.empty or "agent" not in frame.columns:
                return neutral
            frame["ablation_score"] = pd.to_numeric(frame.get("ablation_score", 0.0), errors="coerce").fillna(0.0)
            frame["support_edge_bps"] = pd.to_numeric(frame.get("support_edge_bps", 0.0), errors="coerce").fillna(0.0)
            frame["sample_count"] = pd.to_numeric(frame.get("sample_count", 0.0), errors="coerce").fillna(0.0)
            for name, group in frame.groupby("agent"):
                recent = group.tail(3)
                score = float(recent["ablation_score"].mean())
                edge = float(recent["support_edge_bps"].mean())
                samples = float(recent["sample_count"].max())
                confidence = clamp(samples / 80.0, 0.0, 1.0)
                weight_adjust = clamp((score / 200.0) * confidence, -0.10, 0.10)
                self._agent_ablation_cache[str(name)] = {
                    "ablation_score": score,
                    "support_edge_bps": edge,
                    "sample_count": samples,
                    "weight_adjust": weight_adjust,
                }
            return dict(self._agent_ablation_cache.get(str(agent), neutral))
        except Exception:
            return neutral

    def _agent_adjustments(
        self,
        agent: str,
        product_id: str,
        strategy: str,
    ) -> Dict[str, float]:
        """Calculate bounded adjustments from outcomes and agent competition."""
        agent_stats = self._outcome_stats(agent=agent)
        product_stats = self._outcome_stats(agent=agent, product_id=product_id)
        strategy_stats = self._outcome_stats(agent=agent, strategy=strategy)

        recent = self._recent_trades(20)
        if "agent" in recent.columns:
            recent = recent[recent["agent"].astype(str) == str(agent)]
        if "event" in recent.columns:
            recent = recent[recent["event"].astype(str).str.upper() == "SELL"]
        pnl_column = next((column for column in ("net_pnl_usd", "pnl", "move_bps") if column in recent), None)
        recent_win_rate = 0.5
        if not recent.empty and pnl_column:
            pnl = pd.to_numeric(recent[pnl_column], errors="coerce").fillna(0.0)
            recent_win_rate = float((pnl > 0.0).mean())

        n = float(agent_stats.get("n", 0.0))
        # Bayesian-style shrinkage: tiny samples barely adjust weights, while
        # large samples can earn real authority.
        sample_factor = clamp(n / float(AGENT_SHRINKAGE_TARGET_N), 0.0, 1.0)
        strong_sample_factor = clamp(n / float(AGENT_STRONG_SAMPLE_N), 0.0, 1.0)
        agent_credit = float(agent_stats.get("weighted_credit", agent_stats.get("avg_credit", 0.5)))
        product_win_rate = float(product_stats.get("win_rate", 0.5))
        strategy_win_rate = float(strategy_stats.get("win_rate", 0.5))
        product_adj = (product_win_rate - 0.5) * 0.65 * sample_factor
        strategy_adj = (strategy_win_rate - 0.5) * 0.60 * sample_factor
        recent_adj = (recent_win_rate - 0.5) * 0.85 * sample_factor
        competition = self._agent_competition_score(agent)
        ablation = self._agent_ablation_score(agent)
        ablation_adjust = float(ablation.get("weight_adjust", 0.0))
        leader_bonus = float(competition.get("leader_bonus", 0.0))
        leader_penalty = float(competition.get("leader_penalty", 0.0))
        product_adj = clamp(product_adj, -self.max_agent_adjustment, self.max_agent_adjustment)
        strategy_adj = clamp(strategy_adj, -self.max_agent_adjustment, self.max_agent_adjustment)
        recent_adj = clamp(recent_adj, -self.max_agent_adjustment, self.max_agent_adjustment)
        raw_directional = product_adj + strategy_adj + recent_adj + leader_bonus - leader_penalty
        dynamic_directional_cap = (
            float(AGENT_UNPROVEN_MAX_DIRECTIONAL_ADJ) * (1.0 - strong_sample_factor)
            + float(AGENT_PROVEN_MAX_DIRECTIONAL_ADJ) * strong_sample_factor
        )
        directional = clamp(raw_directional + ablation_adjust, -dynamic_directional_cap, dynamic_directional_cap)
        initial_prior = initial_agent_reliability_prior(agent)
        # Reliability starts near the prior, but outcome credit is shrunk toward
        # neutral until the agent has enough samples.
        base_reliability = (
            0.86 * float(initial_prior)
            + (agent_credit - 0.5) * 1.05 * sample_factor
        )
        reliability = clamp(
            base_reliability + leader_bonus - leader_penalty + ablation_adjust,
            self.min_agent_reliability,
            self.max_agent_reliability,
        )

        try:
            ts = utc_ts()
            append_csv_row(
                AGENT_ADJUSTMENTS_CSV,
                [
                    "ts", "dt_utc", "agent", "product_id", "strategy",
                    "initial_prior", "base_reliability", "product_adjustment", "strategy_adjustment",
                    "recent_performance_adjustment", "directional_adjustment",
                    "final_reliability", "sample_size", "agent_credit",
                    "product_win_rate", "strategy_win_rate", "recent_win_rate",
                    "leaderboard_rank", "leaderboard_score", "leader_bonus",
                    "leader_penalty", "ablation_adjustment", "reason",
                ],
                {
                    "ts": f"{ts:.6f}", "dt_utc": utc_dt(ts), "agent": agent,
                    "product_id": product_id, "strategy": strategy,
                    "initial_prior": f"{initial_prior:.6f}",
                    "base_reliability": f"{base_reliability:.6f}",
                    "product_adjustment": f"{product_adj:.6f}",
                    "strategy_adjustment": f"{strategy_adj:.6f}",
                    "recent_performance_adjustment": f"{recent_adj:.6f}",
                    "directional_adjustment": f"{directional:.6f}",
                    "final_reliability": f"{reliability:.6f}",
                    "sample_size": f"{n:.0f}",
                    "agent_credit": f"{agent_credit:.6f}",
                    "product_win_rate": f"{product_win_rate:.6f}",
                    "strategy_win_rate": f"{strategy_win_rate:.6f}",
                    "recent_win_rate": f"{recent_win_rate:.6f}",
                    "leaderboard_rank": f"{float(competition.get('leaderboard_rank', 999.0)):.0f}",
                    "leaderboard_score": f"{float(competition.get('leaderboard_score', 0.5)):.6f}",
                    "leader_bonus": f"{leader_bonus:.6f}",
                    "leader_penalty": f"{leader_penalty:.6f}",
                    "ablation_adjustment": f"{ablation_adjust:.6f}",
                    "reason": (
                        f"agent={agent};competitive_goal=profit_weighted_shrunk_reliability;"
                        f"initial_prior={initial_prior:.3f};credit={agent_credit:.3f};"
                        f"n={n:.0f};sample_factor={sample_factor:.3f};"
                        f"directional_cap={dynamic_directional_cap:.3f};"
                        f"leader_rank={float(competition.get('leaderboard_rank', 999.0)):.0f};"
                        f"leader_score={float(competition.get('leaderboard_score', 0.5)):.3f};"
                        f"bonus={leader_bonus:.3f};penalty={leader_penalty:.3f};"
                        f"ablation_adjust={ablation_adjust:.3f};"
                    ),
                },
            )
        except Exception:
            pass
        return {
            "product": product_adj, "strategy": strategy_adj, "recent": recent_adj,
            "directional": directional, "reliability": reliability,
            "leaderboard_rank": float(competition.get("leaderboard_rank", 999.0)),
            "leaderboard_score": float(competition.get("leaderboard_score", 0.5)),
            "leader_bonus": leader_bonus, "leader_penalty": leader_penalty,
            "ablation_adjustment": ablation_adjust,
        }

    def _adjust_vote(
        self,
        vote: Dict[str, Any],
        product_id: str,
        strategy: str,
    ) -> AgentVote:
        """Apply outcome-derived direction and reliability to a raw vote."""
        agent_name = str(vote.get("agent", "unknown"))

        adjustments = self._agent_adjustments(
            agent_name,
            product_id,
            strategy,
        )

        directional_adj = float(adjustments["directional"])

        raw_buy = float(vote.get("buy", 0.0) or 0.0)
        raw_sell = float(vote.get("sell", 0.0) or 0.0)
        raw_hold = float(vote.get("hold", 0.0) or 0.0)
        raw_wait = float(vote.get("wait", 0.0) or 0.0)

        # Direction-specific adjustment: boost the agent's dominant direction
        # instead of raising buy and sell together.
        dominant = dominant_vote_direction({
            "buy": raw_buy, "sell": raw_sell, "hold": raw_hold, "wait": raw_wait,
        })
        buy = raw_buy
        sell = raw_sell
        hold = raw_hold
        wait = raw_wait
        if directional_adj >= 0:
            if dominant == "buy":
                buy += directional_adj
                sell -= directional_adj * 0.35
                wait -= directional_adj * 0.25
            elif dominant == "sell":
                sell += directional_adj
                buy -= directional_adj * 0.35
                hold -= directional_adj * 0.15
            elif dominant == "hold":
                hold += directional_adj * 0.70
                sell -= directional_adj * 0.20
                buy -= directional_adj * 0.10
            else:
                wait += directional_adj * 0.65
                buy -= directional_adj * 0.20
                sell -= directional_adj * 0.20
        else:
            penalty = abs(directional_adj)
            if dominant == "buy":
                buy -= penalty
                wait += penalty * 0.45
            elif dominant == "sell":
                sell -= penalty
                hold += penalty * 0.35
                wait += penalty * 0.20
            elif dominant == "hold":
                hold -= penalty * 0.75
                wait += penalty * 0.35
            else:
                wait -= penalty * 0.60
        buy = clamp(buy, 0.0, 1.0)
        sell = clamp(sell, 0.0, 1.0)
        hold = clamp(hold, 0.0, 1.0)
        wait = clamp(wait, 0.0, 1.0)

        confidence = clamp(float(vote.get("confidence", 0.5) or 0.5), 0.0, 1.0)
        reliability = float(adjustments["reliability"])
        weight = max(0.0, confidence * reliability)

        return AgentVote(
            agent=agent_name,
            buy=raw_buy,
            sell=raw_sell,
            hold=raw_hold,
            wait=raw_wait,
            confidence=confidence,
            reliability=reliability,
            adjusted_buy_score=buy,
            adjusted_sell_score=sell,
            adjusted_hold_score=hold,
            adjusted_wait_score=wait,
            product_adjustment=float(adjustments["product"]),
            strategy_adjustment=float(adjustments["strategy"]),
            recent_performance_adjustment=float(adjustments["recent"]),
            weight=weight,
            leaderboard_rank=float(adjustments.get("leaderboard_rank", 999.0)),
            leaderboard_score=float(adjustments.get("leaderboard_score", 0.5)),
            leader_bonus=float(adjustments.get("leader_bonus", 0.0)),
            leader_penalty=float(adjustments.get("leader_penalty", 0.0)),
            reason=str(vote.get("reason", "")),
            product_id=str(vote.get("product_id", product_id) or product_id),
            market_regime=str(vote.get("market_regime", "unknown") or "unknown"),
            momentum_15_bps=float(vote.get("momentum_15_bps", 0.0) or 0.0),
            momentum_30_bps=float(vote.get("momentum_30_bps", 0.0) or 0.0),
            volatility_bps=float(vote.get("volatility_bps", 0.0) or 0.0),
            atr_bps=float(vote.get("atr_bps", 0.0) or 0.0),
            range_bps=float(vote.get("range_bps", 0.0) or 0.0),
        )

    def adaptive_thresholds(self, product_id: str, strategy: str) -> Dict[str, Any]:
        """Return stable thresholds that adapt only after meaningful samples."""
        health = self.session_health()
        risk_mode = health.get("risk_mode", "NORMAL")
        product_stats = self._outcome_stats(product_id=product_id)
        buy = self.base_buy_threshold
        sell = self.base_sell_threshold

        risk_mode_u = str(risk_mode).upper()

        if risk_mode_u == "DEFENSIVE":
            buy += 0.020
            sell += 0.015
        elif risk_mode_u == "CAUTIOUS":
            buy += 0.010
            sell += 0.008
        elif risk_mode_u == "AGGRESSIVE":
            buy -= 0.065
            sell -= 0.020

        n = float(product_stats.get("n", 0.0))
        wr = float(product_stats.get("win_rate", 0.5))
        avg = float(product_stats.get("avg_move", 0.0))
        adverse = float(product_stats.get("avg_adverse", 0.0))

        if n >= 8:
            if wr < 0.35 or avg < -60:
                buy += 0.025
                sell += 0.015
            elif wr > 0.55 and avg > 15:
                buy -= 0.075
                sell -= 0.020

        if n >= 20:
            if adverse > 120:
                buy += 0.04
            elif adverse < 45 and avg > 20:
                buy -= 0.03

        if strategy == "BREAKOUT_CONTINUATION":
            buy += 0.02
        elif strategy == "MEAN_REVERSION_BOUNCE":
            buy += 0.01
        elif strategy == "PULLBACK_CONTINUATION":
            buy -= 0.01
        elif strategy == "STAND_ASIDE":
            buy += 0.18

        missed_relief = self._missed_opportunity_relief(product_id)
        buy -= missed_relief
        buy = clamp(buy, self.min_buy_threshold, self.max_buy_threshold)
        sell = clamp(sell, self.min_sell_threshold, self.max_sell_threshold)

        try:
            ts = utc_ts()

            append_csv_row(
                ADAPTIVE_THRESHOLDS_CSV,
                [
                    "ts", "dt_utc", "scope", "product_id", "strategy",
                    "buy_threshold", "sell_threshold", "risk_mode",
                    "sample_size", "win_rate", "avg_move", "avg_adverse",
                    "missed_opportunity_relief", "reason",
                ],
                {
                    "ts": f"{ts:.6f}",
                    "dt_utc": utc_dt(ts),
                    "scope": "product_strategy",
                    "product_id": product_id,
                    "strategy": strategy,
                    "buy_threshold": f"{buy:.6f}",
                    "sell_threshold": f"{sell:.6f}",
                    "risk_mode": risk_mode_u,
                    "sample_size": f"{n:.0f}",
                    "win_rate": f"{wr:.6f}",
                    "avg_move": f"{avg:.6f}",
                    "avg_adverse": f"{adverse:.6f}",
                    "missed_opportunity_relief": f"{missed_relief:.6f}",
                    "reason": (
                        f"risk={risk_mode_u};n={n:.0f};wr={wr:.3f};"
                        f"avg={avg:.2f};adverse={adverse:.2f};"
                        f"missed_relief={missed_relief:.3f}"
                    ),
                },
            )
        except Exception:
            pass

        return {
            "buy_threshold": buy,
            "sell_threshold": sell,
            "risk_mode": risk_mode_u,
            "product_stats": product_stats,
            "missed_opportunity_relief": missed_relief,
        }


    def _latest_agent_trade_policy(self) -> Dict[str, Dict[str, Any]]:
        try:
            frame = _read_csv_tail_direct(AGENT_TRADE_POLICY_CSV, 5000)
            if frame.empty or "agent" not in frame.columns:
                return {}
            frame = frame.sort_values("ts").groupby("agent", as_index=False).tail(1)
            out: Dict[str, Dict[str, Any]] = {}
            for _, r in frame.iterrows():
                agent = str(r.get("agent") or "")
                if not agent:
                    continue
                out[agent] = {
                    "recommended_role": str(r.get("recommended_role") or "neutral"),
                    "entry_weight_multiplier": float(r.get("entry_weight_multiplier", 1.0) or 1.0),
                    "veto_weight_multiplier": float(r.get("veto_weight_multiplier", 1.0) or 1.0),
                }
            return out
        except Exception:
            return {}

    def _latest_agent_side_ratings(self) -> Dict[str, Dict[str, Any]]:
        """Latest separate buy/sell analyst authority."""
        try:
            now_value = utc_ts()
            if self._agent_side_ratings_cache and now_value - float(self._agent_side_ratings_cache_ts) < float(self._agent_side_ratings_cache_sec):
                return dict(self._agent_side_ratings_cache)
            self._agent_side_ratings_cache = {}
            self._agent_side_ratings_cache_ts = now_value
            frame = _read_csv_tail_direct(AGENT_SIDE_RATINGS_CSV, 5000, usecols=LEVEL8_CSV_USECOLS.get("agent_side_ratings.csv"))
            if frame.empty or "agent" not in frame.columns:
                return {}
            if "ts" in frame.columns:
                frame["ts_num"] = pd.to_numeric(frame["ts"], errors="coerce")
                frame = frame.sort_values("ts_num")
            latest = frame.groupby(frame["agent"].astype(str), as_index=False).tail(1)
            for _, row in latest.iterrows():
                agent = str(row.get("agent") or "")
                if not agent:
                    continue
                self._agent_side_ratings_cache[agent] = {
                    "buy_rows": float(row.get("buy_rows", 0.0) or 0.0),
                    "buy_accuracy": float(row.get("buy_accuracy", 0.5) or 0.5),
                    "buy_score": float(row.get("buy_score", 0.5) or 0.5),
                    "buy_weight_pct": float(row.get("buy_weight_pct", 0.0) or 0.0),
                    "buy_weight_multiplier": float(row.get("buy_weight_multiplier", 1.0) or 1.0),
                    "sell_rows": float(row.get("sell_rows", 0.0) or 0.0),
                    "sell_accuracy": float(row.get("sell_accuracy", 0.5) or 0.5),
                    "sell_score": float(row.get("sell_score", 0.5) or 0.5),
                    "sell_weight_pct": float(row.get("sell_weight_pct", 0.0) or 0.0),
                    "sell_weight_multiplier": float(row.get("sell_weight_multiplier", 1.0) or 1.0),
                }
            return dict(self._agent_side_ratings_cache)
        except Exception:
            return {}


    def _latest_agent_context_ratings(self) -> Dict[str, Dict[str, Any]]:
        """Load product/regime-specific context ratings keyed by SIDE|PRODUCT_ID|MARKET_REGIME|AGENT."""
        try:
            now_value = utc_ts()
            if self._agent_context_ratings_cache and now_value - float(self._agent_context_ratings_cache_ts) < float(self._agent_context_ratings_cache_sec):
                return dict(self._agent_context_ratings_cache)
            self._agent_context_ratings_cache = {}
            self._agent_context_ratings_cache_ts = now_value
            frame = _read_csv_tail_direct(
                FOUR_PASS_AGENT_CONTEXT_RATINGS_CSV,
                20000,
                usecols=LEVEL8_CSV_USECOLS.get("four_pass_agent_context_ratings.csv"),
            )
            if frame.empty:
                return {}
            for _, row in frame.iterrows():
                side = str(row.get("side") or "").upper()
                product_id = str(row.get("product_id") or "")
                market_regime = str(row.get("market_regime") or "unknown")
                agent = str(row.get("agent") or "")
                if not side or not product_id or not agent:
                    continue
                key = f"{side}|{product_id}|{market_regime}|{agent}"
                self._agent_context_ratings_cache[key] = {
                    "selected_count": float(row.get("selected_count", 0.0) or 0.0),
                    "smoothed_win_rate": float(row.get("smoothed_win_rate", 0.5) or 0.5),
                    "ev_bps": float(row.get("ev_bps", 0.0) or 0.0),
                    "score": float(row.get("score", 0.5) or 0.5),
                    "weight_pct": float(row.get("weight_pct", 0.0) or 0.0),
                    "profitability_mode": str(row.get("profitability_mode") or "unknown"),
                }
            return dict(self._agent_context_ratings_cache)
        except Exception:
            return {}


    def _latest_agent_decision_influence(self) -> Dict[str, Dict[str, Any]]:
        try:
            now_value = utc_ts()
            if self._agent_decision_influence_cache and now_value - self._agent_decision_influence_cache_ts < self._agent_decision_influence_cache_sec:
                return dict(self._agent_decision_influence_cache)
            self._agent_decision_influence_cache = {}
            self._agent_decision_influence_cache_ts = now_value
            frame = _read_csv_tail_direct(AGENT_DECISION_INFLUENCE_CSV, 5000, usecols=LEVEL8_CSV_USECOLS.get("agent_decision_influence.csv"))
            if frame.empty:
                return {}
            for _, row in frame.iterrows():
                key = f"{str(row.get('side') or '').upper()}|{str(row.get('agent') or '')}"
                self._agent_decision_influence_cache[key] = {
                    "selected_count": float(row.get("selected_count", 0.0) or 0.0),
                    "frequency_per_day": float(row.get("frequency_per_day", 0.0) or 0.0),
                    "smoothed_win_rate": float(row.get("smoothed_win_rate", 0.5) or 0.5),
                    "ev_bps": float(row.get("ev_bps", 0.0) or 0.0),
                    "decision_weight_pct": float(row.get("decision_weight_pct", 0.0) or 0.0),
                    "role": str(row.get("role") or ""),
                }
            return dict(self._agent_decision_influence_cache)
        except Exception:
            return {}

    def _latest_product_agent_influence(self) -> Dict[str, Dict[str, Any]]:
        try:
            now_value = utc_ts()
            if self._product_agent_influence_cache and now_value - self._product_agent_influence_cache_ts < self._product_agent_influence_cache_sec:
                return dict(self._product_agent_influence_cache)
            self._product_agent_influence_cache = {}
            self._product_agent_influence_cache_ts = now_value
            frame = _read_csv_tail_direct(PRODUCT_AGENT_INFLUENCE_CSV, 10000, usecols=LEVEL8_CSV_USECOLS.get("product_agent_influence.csv"))
            if frame.empty:
                return {}
            for _, row in frame.iterrows():
                key = f"{str(row.get('side') or '').upper()}|{str(row.get('product_id') or '')}|{str(row.get('market_regime') or 'unknown')}|{str(row.get('agent') or '')}"
                self._product_agent_influence_cache[key] = {
                    "selected_count": float(row.get("selected_count", 0.0) or 0.0),
                    "frequency_per_day": float(row.get("frequency_per_day", 0.0) or 0.0),
                    "smoothed_win_rate": float(row.get("smoothed_win_rate", 0.5) or 0.5),
                    "ev_bps": float(row.get("ev_bps", 0.0) or 0.0),
                    "decision_weight_pct": float(row.get("decision_weight_pct", 0.0) or 0.0),
                    "role": str(row.get("role") or ""),
                }
            return dict(self._product_agent_influence_cache)
        except Exception:
            return {}

    def _infer_live_market_regime_for_votes(self, adjusted_votes: list[AgentVote]) -> str:
        """Lightweight live regime inference from vote metadata."""
        try:
            explicit_regimes = []
            for vote in adjusted_votes:
                try:
                    regime = str(getattr(vote, "market_regime", "") or "")
                    if regime and regime != "unknown":
                        explicit_regimes.append(regime)
                except Exception:
                    pass
            if explicit_regimes:
                return max(set(explicit_regimes), key=explicit_regimes.count)

            momentum_values = []
            volatility_values = []
            for vote in adjusted_votes:
                data = asdict(vote)
                for key in ["momentum_15_bps", "momentum_30_bps", "trend_bps", "macro_momentum_bps"]:
                    if key in data:
                        try:
                            momentum_values.append(float(data.get(key) or 0.0))
                        except Exception:
                            pass
                for key in ["volatility_bps", "atr_bps", "range_bps", "macro_volatility_bps"]:
                    if key in data:
                        try:
                            volatility_values.append(float(data.get(key) or 0.0))
                        except Exception:
                            pass
            momentum = sum(momentum_values) / max(1, len(momentum_values))
            volatility = sum(volatility_values) / max(1, len(volatility_values))
            if momentum >= 20.0 and volatility >= 80.0:
                return "trend_high_vol"
            if momentum >= 20.0 and volatility < 80.0:
                return "trend_low_vol"
            if momentum <= -20.0 and volatility >= 80.0:
                return "downtrend_high_vol"
            if momentum <= -20.0 and volatility < 80.0:
                return "downtrend_low_vol"
            if abs(momentum) < 20.0 and volatility >= 80.0:
                return "range_high_vol"
            return "range_low_vol"
        except Exception:
            return "unknown"

    def _weighted_vote_pairs(
        self,
        adjusted_votes: list[AgentVote],
        *,
        decision_side: str,
        product_id: str = "",
        market_regime: str = "",
    ) -> list[tuple[AgentVote, float]]:
        """Return vote weights after redundancy-group caps."""
        raw_pairs: list[tuple[AgentVote, float, str]] = []
        agent_policy = self._latest_agent_trade_policy() if hasattr(self, "_latest_agent_trade_policy") else {}
        side_ratings = self._latest_agent_side_ratings() if hasattr(self, "_latest_agent_side_ratings") else {}
        context_ratings = self._latest_agent_context_ratings() if hasattr(self, "_latest_agent_context_ratings") else {}
        decision_influence = self._latest_agent_decision_influence() if hasattr(self, "_latest_agent_decision_influence") else {}
        product_influence = self._latest_product_agent_influence() if hasattr(self, "_latest_product_agent_influence") else {}
        if not market_regime:
            market_regime = self._infer_live_market_regime_for_votes(adjusted_votes)
        for vote in adjusted_votes:
            # BUY_LEAD_ONLY_MODE is intentionally disabled.
            # All analysts remain eligible for BUY voting.
            # The four-pass model determines each analyst's actual buy weight.

            raw_weight = max(0.0, float(vote.confidence) * float(vote.reliability))
            side_rating = side_ratings.get(str(vote.agent), {})
            side = str(decision_side).upper()
            context_key = f"{side}|{str(product_id)}|{str(market_regime)}|{str(vote.agent)}"
            context_rating = context_ratings.get(context_key, {})
            if context_rating:
                context_rows = float(context_rating.get("selected_count", 0.0) or 0.0)
                context_weight_pct = float(context_rating.get("weight_pct", 0.0) or 0.0)
                context_ev = float(context_rating.get("ev_bps", 0.0) or 0.0)
                if context_rows >= 20 and context_weight_pct > 0.0 and context_ev > 0.0:
                    equal_context_weight = 100.0 / max(1.0, len(adjusted_votes))
                    context_multiplier = max(0.25, min(5.0, context_weight_pct / max(1e-9, equal_context_weight)))
                    raw_weight *= context_multiplier
            product_influence_key = f"{side}|{str(product_id)}|{str(market_regime)}|{str(vote.agent)}"
            global_influence_key = f"{side}|{str(vote.agent)}"
            influence = product_influence.get(product_influence_key) or decision_influence.get(global_influence_key) or {}
            if influence:
                samples = float(influence.get("selected_count", 0.0) or 0.0)
                ev = float(influence.get("ev_bps", 0.0) or 0.0)
                pct = float(influence.get("decision_weight_pct", 0.0) or 0.0)
                if samples >= 20 and pct > 0.0 and ev > 0.0:
                    equal_pct = 100.0 / max(1.0, len(adjusted_votes))
                    influence_multiplier = max(0.15, min(8.0, pct / max(1e-9, equal_pct)))
                    raw_weight *= influence_multiplier
            if side == "BUY":
                buy_rows = float(side_rating.get("buy_rows", 0.0) or 0.0)
                if buy_rows >= 10:
                    raw_weight *= float(side_rating.get("buy_weight_multiplier", 1.0) or 1.0)
            elif side == "SELL":
                sell_rows = float(side_rating.get("sell_rows", 0.0) or 0.0)
                if sell_rows >= 5:
                    raw_weight *= float(side_rating.get("sell_weight_multiplier", 1.0) or 1.0)
            direction = dominant_vote_direction(asdict(vote)).upper()
            policy = agent_policy.get(str(vote.agent), {})
            role = str(policy.get("recommended_role", "neutral"))
            entry_mult = float(policy.get("entry_weight_multiplier", 1.0) or 1.0)
            veto_mult = float(policy.get("veto_weight_multiplier", 1.0) or 1.0)
            if direction == "BUY":
                raw_weight *= entry_mult
                if role in {"avoidance_veto_filter", "buy_signal_penalty_veto_filter"}:
                    raw_weight *= 0.05
                if role == "primary_sell_alpha":
                    raw_weight *= 0.10
            if direction in {"WAIT", "HOLD"} and role in {"avoidance_veto_filter", "buy_signal_penalty_veto_filter"}:
                raw_weight *= veto_mult
            if direction == "BUY" and role in {"primary_buy_alpha", "secondary_buy_confirmer"}:
                raw_weight *= max(1.0, entry_mult)
            if str(decision_side).upper() == "SELL" and role in {"primary_sell_alpha", "secondary_sell_confirmer"}:
                raw_weight *= max(1.0, float(policy.get("sell_weight_multiplier", 1.0) or 1.0))
            group = agent_redundancy_group(vote.agent)
            raw_pairs.append((vote, raw_weight, group))

        total_weight = sum(weight for _, weight, _ in raw_pairs) or 1.0
        group_totals: Dict[str, float] = {}
        for _, weight, group in raw_pairs:
            group_totals[group] = group_totals.get(group, 0.0) + weight

        caps = BUY_REDUNDANCY_GROUP_CAPS if str(decision_side).upper() == "BUY" else SELL_REDUNDANCY_GROUP_CAPS
        final_pairs: list[tuple[AgentVote, float]] = []
        for vote, weight, group in raw_pairs:
            group_cap_fraction = float(caps.get(group, caps.get("other", 0.16)))
            group_cap_abs = group_cap_fraction * total_weight
            group_total = max(group_totals.get(group, 0.0), 1e-12)
            scale = min(1.0, group_cap_abs / group_total)
            final_pairs.append((vote, weight * scale))
        return final_pairs

    def decide_buy(
        self,
        product_id: str,
        strategy: str,
        votes: list[Dict[str, Any]],
        truth_vote: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine adjusted votes into a buy, shadow, or wait decision."""
        decision_id = f"l8buy-{product_id}-{int(utc_ts())}-{uuid.uuid4().hex[:8]}"
        adjusted = [self._adjust_vote(vote, product_id, strategy) for vote in votes]
        adjusted_truth = self._adjust_vote(truth_vote, product_id, strategy)
        weighted = self._weighted_vote_pairs(adjusted, decision_side="BUY", product_id=product_id)
        weight_total = sum(weight for _, weight in weighted) or 1.0
        combined = {
            "adj_buy": sum(v.adjusted_buy_score * w for v, w in weighted) / weight_total,
            "adj_sell": sum(v.adjusted_sell_score * w for v, w in weighted) / weight_total,
            "confidence": sum(v.confidence * w for v, w in weighted) / weight_total,
        }

        raw_learning_scores = []

        for vote in votes:
            try:
                raw_learning_scores.append(float(vote.get("learning_score", 0.0) or 0.0))
            except Exception:
                pass

        learning_score = clamp(
            sum(raw_learning_scores) / len(raw_learning_scores) if raw_learning_scores else 0.0,
            0.0,
            1.0,
        )

        experience_n = float(self._outcome_stats(product_id=product_id).get("n", 0.0))
        exploration_decay = clamp(experience_n / 160.0, 0.0, 1.0)

        # Keep SHADOW learning active while preventing exploration from pushing
        # weak overnight setups into live money before enough outcomes exist.
        exploration_weight = 0.14 * (1.0 - exploration_decay) + 0.05 * exploration_decay

        truth_score = clamp(
            (
                adjusted_truth.adjusted_buy_score * 0.55
                + adjusted_truth.confidence * 0.25
                + adjusted_truth.reliability * 0.20
            ),
            0.0,
            1.0,
        )
        # Truth modulates the score, but learning mode should not let low early sample
        # quality completely suppress all buys.
        base_final_buy = clamp(
            combined["adj_buy"] * (0.85 + truth_score * 0.15),
            0.0,
            1.0,
        )

        final_sell = clamp(
            combined["adj_sell"] * (0.65 + truth_score * 0.35),
            0.0,
            1.0,
        )

        contradiction_penalty = 0.0

        if final_sell > base_final_buy:
            contradiction_penalty += min(0.18, (final_sell - base_final_buy) * 0.40)

        if truth_score < 0.58:
            contradiction_penalty += min(0.10, (0.58 - truth_score) * 0.25)

        final_buy = clamp(
            base_final_buy * (1.0 - exploration_weight)
            + learning_score * exploration_weight
            - contradiction_penalty,
            0.0,
            1.0,
        )
        thresholds = self.adaptive_thresholds(product_id, strategy)
        buy_threshold = thresholds["buy_threshold"]
        bucket, position_pct, sizing_reason = self._position_pct_from_decision(
            final_buy_score=final_buy,
            threshold=buy_threshold,
            truth_score=truth_score,
            risk_mode=thresholds["risk_mode"],
        )

        if bucket in ("APPROVED", "TEST", "CORE"):
            action = "ALLOW_BUY"
        elif final_buy >= buy_threshold and bucket == "SHADOW":
            action = "SHADOW"
        else:
            action = "WAIT"

        if action == "SHADOW":
            try:
                ts = utc_ts()

                append_csv_row(
                    SHADOW_TRADES_CSV,
                    [
                        "ts", "dt_utc", "decision_id", "product_id", "strategy",
                        "shadow_action", "council_buy_score", "buy_threshold",
                        "truth_score", "recommended_position_pct", "reason",
                    ],
                    {
                        "ts": f"{ts:.6f}",
                        "dt_utc": utc_dt(ts),
                        "decision_id": decision_id,
                        "product_id": product_id,
                        "strategy": strategy,
                        "shadow_action": "BUY",
                        "council_buy_score": f"{final_buy:.6f}",
                        "buy_threshold": f"{buy_threshold:.6f}",
                        "truth_score": f"{truth_score:.6f}",
                        "recommended_position_pct": f"{position_pct:.6f}",
                        "reason": sizing_reason,
                    },
                )
            except Exception:
                pass

        debug_every(
            MODULE_NAME,
            f"decide_buy:{product_id}",
            15.0,
            "level8_decide_buy",
            data={
                "decision_id": decision_id,
                "product_id": product_id,
                "strategy": strategy,
                "action": action,
                "final_buy": final_buy,
                "final_sell": final_sell,
                "truth_score": truth_score,
                "buy_threshold": buy_threshold,
                "bucket": bucket,
                "position_pct": position_pct,
                "vote_count": len(votes),
                "adjusted_vote_count": len(adjusted),
                "weight_total": weight_total,
                "contradiction_penalty": contradiction_penalty,
                "exploration_weight": exploration_weight,
            },
            level="DEBUG",
            also_overall=False,
        )

        return {
            "decision_id": decision_id,
            "action": action,
            "final_buy": final_buy,
            "final_sell": final_sell,
            "truth_score": truth_score,
            "bucket": bucket,
            "position_pct": position_pct,
            "sizing_reason": sizing_reason,
            "learning_score": learning_score,
            "exploration_weight": exploration_weight,
            "contradiction_penalty": contradiction_penalty,
            "base_final_buy": base_final_buy,
            "confidence": combined["confidence"],
            **thresholds,
            "votes": [asdict(vote) for vote in adjusted],
            "truth_vote": asdict(adjusted_truth),
        }

    def decide_exit(
        self,
        product_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Level 8 sell council.

        Selling answers three questions:
        1. Should we sell now?
        2. If yes, should we sell part or all of the position?
        3. Are we capturing profit near a wave peak, or should we hold for continuation?
        """
        decision_id = f"l8exit-{product_id}-{int(utc_ts())}-{uuid.uuid4().hex[:8]}"

        unrealized_bps = float(context.get("unrealized_bps", 0.0) or 0.0)
        spread_bps = float(context.get("spread_bps", 0.0) or 0.0)
        cost_bps = float(context.get("cost_bps", 0.0) or 0.0)
        hold_seconds = float(context.get("hold_seconds", 0.0) or 0.0)
        hold_minutes = hold_seconds / 60.0
        max_hold_minutes = float(context.get("adaptive_max_hold_minutes", context.get("max_hold_minutes", 120.0)) or 120.0)

        min_hold_elapsed = bool(context.get("min_hold_elapsed", False))
        target_hold_elapsed = bool(context.get("target_hold_elapsed", False))
        max_hold_elapsed = bool(context.get("max_hold_elapsed", False))
        hard_stop_hit = bool(context.get("hard_stop_hit", False))
        early_profit_ok = bool(context.get("early_profit_ok", False))

        net_after_exit_bps = float(context.get("net_after_exit_bps", 0.0) or 0.0)
        min_net_after_exit_bps = float(context.get("min_net_after_exit_bps", 0.0) or 0.0)

        peak_unrealized_bps = float(context.get("peak_unrealized_bps", unrealized_bps) or unrealized_bps)
        pullback_from_peak_bps = float(context.get("pullback_from_peak_bps", 0.0) or 0.0)

        momentum_1_bps = float(context.get("momentum_1_bps", 0.0) or 0.0)
        momentum_3_bps = float(context.get("momentum_3_bps", 0.0) or 0.0)
        momentum_5_bps = float(context.get("momentum_5_bps", 0.0) or 0.0)
        green_candles = int(float(context.get("green_candles", 0) or 0))

        session_liquidity = dict(context.get("session_liquidity", {}) or {})
        session_sell_score = float(session_liquidity.get("best_sell_score", 0.0) or 0.0)
        session_buy_score = float(session_liquidity.get("best_buy_score", 0.0) or 0.0)
        session_hold_score = float(session_liquidity.get("best_hold_score", 0.50) or 0.50)
        session_confidence = float(session_liquidity.get("confidence", 0.0) or 0.0)
        session_agent = str(session_liquidity.get("strongest_agent", "session_liquidity"))
        session_reason = str(session_liquidity.get("reason", "no_session_liquidity_sell_context"))

        price_action_context = dict(context.get("price_action_context", {}) or {})
        pa_sell_score = float(price_action_context.get("candle_context_sell_score", 0.0) or 0.0)
        pa_hold_score = float(price_action_context.get("candle_context_hold_score", 0.50) or 0.50)
        pa_confidence = float(price_action_context.get("candle_context_confidence", 0.0) or 0.0)
        candle_exhaustion_score = float(price_action_context.get("candle_exhaustion_score", 0.0) or 0.0)
        candle_continuation_score = float(price_action_context.get("candle_continuation_score", 0.0) or 0.0)
        volume_profile_sell_score = float(price_action_context.get("volume_profile_sell_score", 0.0) or 0.0)
        volume_profile_hold_score = float(price_action_context.get("volume_profile_hold_score", 0.50) or 0.50)
        volume_profile_leader_sell_score = float(price_action_context.get("volume_profile_leader_sell_score", volume_profile_sell_score) or volume_profile_sell_score)
        volume_profile_leader_hold_score = float(price_action_context.get("volume_profile_leader_hold_score", volume_profile_hold_score) or volume_profile_hold_score)
        volume_profile_leader_wait_score = float(price_action_context.get("volume_profile_leader_wait_score", 0.50) or 0.50)
        value_acceptance_state = str(price_action_context.get("value_acceptance_state", ""))
        volume_node_state = str(price_action_context.get("volume_node_state", ""))
        fvg_sell_score = float(price_action_context.get("fvg_sell_score", 0.0) or 0.0)
        value_area_state = str(price_action_context.get("value_area_state", ""))
        fvg_state = str(price_action_context.get("fvg_state", ""))
        pa_reason = str(price_action_context.get("reason", "no_price_action_sell_context"))

        smt_context = dict(context.get("smt_divergence", {}) or {})
        smt_sell_score = float(smt_context.get("sell", 0.0) or 0.0)
        previous_session_profile = dict(context.get("previous_session_profile", {}) or {})
        previous_session_sell_score = float(previous_session_profile.get("sell_score", 0.0) or 0.0)
        previous_session_hold_score = float(previous_session_profile.get("hold_score", 0.50) or 0.50)
        previous_session_wait_score = float(previous_session_profile.get("wait_score", 0.50) or 0.50)
        previous_session_reaction = str(previous_session_profile.get("reaction_state", ""))
        previous_session_reason = str(previous_session_profile.get("reason", ""))
        quant_context = dict(context.get("quant_context", {}) or {})
        quant_sell_score = float(quant_context.get("quant_sell_score", 0.0) or 0.0)
        quant_hold_score = float(quant_context.get("quant_hold_score", 0.50) or 0.50)
        quant_wait_score = float(quant_context.get("quant_wait_score", 0.50) or 0.50)
        quant_boundary_state = str(quant_context.get("boundary_state", ""))
        quant_vol_state = str(quant_context.get("volatility_cluster_state", ""))
        quant_reason = str(quant_context.get("reason", ""))
        smt_confidence = float(smt_context.get("confidence", 0.0) or 0.0)
        smt_reason = str(smt_context.get("reason", "no_smt_sell_context"))

        sell_utility_context = dict(context.get("sell_side_expected_utility", {}) or {})
        hold_utility_bps = float(sell_utility_context.get("hold_utility_bps", 0.0) or 0.0)
        sell_utility_bps = float(sell_utility_context.get("sell_utility_bps", 0.0) or 0.0)
        sell_minus_hold_utility_bps = float(sell_utility_context.get("sell_minus_hold_utility_bps", 0.0) or 0.0)
        sell_utility_reason = str(sell_utility_context.get("reason", "no_sell_utility_context"))
        sell_utility_fraction = float(sell_utility_context.get("sell_utility_suggested_fraction", 0.0) or 0.0)

        spike_context = dict(context.get("spike_profit_protection", {}) or {})
        spike_armed = bool(spike_context.get("armed", False))
        spike_allow_partial = bool(spike_context.get("allow_partial", False))
        spike_immediate_partial = bool(spike_context.get("allow_immediate_partial", False))
        spike_suggested_fraction = float(spike_context.get("suggested_fraction", 0.0) or 0.0)
        spike_reason = str(spike_context.get("reason", "no_spike_profit_protection"))

        sell_quality_context = dict(context.get("sell_quality_context", {}) or {})
        sell_quality_penalty = float(sell_quality_context.get("sell_quality_penalty", 0.0) or 0.0)
        sell_quality_reason = str(sell_quality_context.get("reason", ""))

        adaptive_exit = dict(context.get("adaptive_exit_profile", {}) or {})
        adaptive_enabled = bool(adaptive_exit.get("enabled", False))
        adaptive_expected_favorable_bps = float(adaptive_exit.get("expected_favorable_bps", 0.0) or 0.0)
        adaptive_progress_to_expected = float(adaptive_exit.get("progress_to_expected", 0.0) or 0.0)
        adaptive_protection_armed = bool(adaptive_exit.get("protection_armed", False))
        adaptive_floor_exit_confirmed = bool(adaptive_exit.get("floor_exit_confirmed", False))
        adaptive_partial_harvest = bool(adaptive_exit.get("adaptive_partial_harvest", False))
        adaptive_full_exit = bool(adaptive_exit.get("adaptive_full_exit", False))
        adaptive_harvest_fraction = float(adaptive_exit.get("adaptive_harvest_fraction", 0.0) or 0.0)
        adaptive_strong_continuation = bool(adaptive_exit.get("strong_continuation", False))
        adaptive_dynamic_pullback_bps = float(adaptive_exit.get("dynamic_pullback_bps", 0.0) or 0.0)
        adaptive_dynamic_strong_pullback_bps = float(adaptive_exit.get("dynamic_strong_pullback_bps", 0.0) or 0.0)
        adaptive_dynamic_full_exit_pullback_bps = float(adaptive_exit.get("dynamic_full_exit_pullback_bps", 0.0) or 0.0)
        adaptive_reason = str(adaptive_exit.get("reason", "no_adaptive_exit_context"))

        min_partial_fraction = float(context.get("min_partial_sell_fraction", 0.25) or 0.25)
        max_partial_fraction = float(context.get("max_partial_sell_fraction", 1.0) or 1.0)
        peak_capture_trigger_bps = float(context.get("peak_capture_trigger_bps", 45.0) or 45.0)
        strong_pullback_bps = float(context.get("peak_capture_strong_pullback_bps", 120.0) or 120.0)
        full_exit_pullback_bps = float(context.get("peak_capture_full_exit_pullback_bps", 240.0) or 240.0)
        if adaptive_enabled:
            if adaptive_dynamic_pullback_bps > 0:
                peak_capture_trigger_bps = adaptive_dynamic_pullback_bps
            if adaptive_dynamic_strong_pullback_bps > 0:
                strong_pullback_bps = adaptive_dynamic_strong_pullback_bps
            if adaptive_dynamic_full_exit_pullback_bps > 0:
                full_exit_pullback_bps = adaptive_dynamic_full_exit_pullback_bps
        strong_continuation_mom3_bps = float(context.get("strong_continuation_mom3_bps", 10.0) or 10.0)
        strong_continuation_pullback_max_bps = float(context.get("strong_continuation_pullback_max_bps", 35.0) or 35.0)

        profit_capture = clamp(0.18 + max(0.0, net_after_exit_bps) / 190.0, 0.0, 1.0)
        loss_exit = clamp((0.90 if hard_stop_hit else 0.10) + max(0.0, -unrealized_bps - 220.0) / 420.0, 0.0, 1.0)
        continuation_quality = clamp(
            0.48 + max(-80.0, min(120.0, momentum_3_bps)) / 180.0 + max(-120.0, min(180.0, momentum_5_bps)) / 320.0
            + min(3, max(0, green_candles)) * 0.035 + max(0.0, net_after_exit_bps) / 850.0 - pullback_from_peak_bps / 150.0
            - (0.60 if hard_stop_hit else 0.0)
            + previous_session_hold_score * 0.10
            + quant_hold_score * 0.08,
            0.0, 1.0,
        )
        continuation_hold = clamp(continuation_quality + (0.12 if not min_hold_elapsed else 0.0) + (0.08 if not target_hold_elapsed else 0.0) - (0.18 if max_hold_elapsed else 0.0), 0.0, 1.0)
        peak_capture = clamp(0.12 + max(0.0, peak_unrealized_bps - min_net_after_exit_bps) / 500.0 + max(0.0, pullback_from_peak_bps - peak_capture_trigger_bps) / 150.0 - max(0.0, momentum_3_bps) / 260.0, 0.0, 1.0)
        momentum_fade_sell = clamp(0.18 + max(0.0, -momentum_1_bps) / 120.0 + max(0.0, -momentum_3_bps) / 180.0 + max(0.0, pullback_from_peak_bps - peak_capture_trigger_bps) / 180.0, 0.0, 1.0)
        execution_sell_quality = clamp(1.0 - max(0.0, spread_bps) / 120.0, 0.0, 1.0)
        fee_recovery = clamp(0.20 + max(0.0, net_after_exit_bps) / 230.0, 0.0, 1.0)
        price_action_harvest_pressure = clamp(
            candle_exhaustion_score * 0.35
            + pa_sell_score * pa_confidence * 0.25
            + volume_profile_sell_score * 0.08
            + volume_profile_leader_sell_score * 0.24
            + fvg_sell_score * 0.12
            + smt_sell_score * smt_confidence * 0.10
            + previous_session_sell_score * 0.16
            + quant_sell_score * 0.14,
            0.0,
            1.0,
        )

        harvest_pressure = clamp(
            profit_capture * 0.24
            + peak_capture * 0.25
            + momentum_fade_sell * 0.18
            + loss_exit * 0.18
            + price_action_harvest_pressure * 0.15,
            0.0,
            1.0,
        )

        adaptive_wave_pressure = clamp(
            (0.22 if adaptive_protection_armed else 0.0)
            + adaptive_progress_to_expected * 0.28
            + (0.30 if adaptive_partial_harvest else 0.0)
            + (0.55 if adaptive_full_exit else 0.0)
            + (0.80 if adaptive_floor_exit_confirmed else 0.0)
            - (0.25 if adaptive_strong_continuation else 0.0),
            0.0,
            1.0,
        )

        votes = [
            {"agent": "profit_capture", "buy": 0.0, "sell": profit_capture, "hold": 1.0 - profit_capture * 0.55, "wait": 0.20, "confidence": 0.70, "reason": "sell when net profit is actually available"},
            {"agent": "drawdown_exit", "buy": 0.0, "sell": loss_exit, "hold": 1.0 - loss_exit * 0.70, "wait": 0.25, "confidence": 0.72, "reason": "sell hard stops but avoid tiny fee-loss churn"},
            {"agent": "continuation_hold", "buy": 0.0, "sell": 1.0 - continuation_hold, "hold": continuation_hold, "wait": 0.20, "confidence": 0.68, "reason": "hold if the wave still appears to be continuing"},
            {"agent": "peak_capture", "buy": 0.0, "sell": peak_capture, "hold": 1.0 - peak_capture * 0.70, "wait": 0.25, "confidence": 0.72, "reason": "sell when price pulls back from a profitable local peak"},
            {"agent": "momentum_fade", "buy": 0.0, "sell": momentum_fade_sell, "hold": 1.0 - momentum_fade_sell * 0.65, "wait": 0.30, "confidence": 0.66, "reason": "sell more when short-term momentum fades"},
            {
                "agent": "higher_low_wave_stop_agent",
                "buy": 0.0,
                "sell": 1.0 if bool(context.get("wave_stop_exit_confirmed", False)) else 0.15,
                "hold": 0.15 if bool(context.get("wave_stop_exit_confirmed", False)) else 0.85,
                "wait": 0.20,
                "confidence": 0.86 if bool(context.get("wave_stop_exit_confirmed", False)) else 0.55,
                "reason": "ride wave until confirmed higher-low stop breaks",
            },
            {"agent": "profit_pullback_capture_agent", "buy": 0.0, "sell": clamp(profit_capture * 0.70 + peak_capture * 0.30, 0.0, 1.0), "hold": clamp(1.0 - profit_capture * 0.50 - peak_capture * 0.25, 0.0, 1.0), "wait": 0.20, "confidence": 0.86, "reason": "new_sell_alpha_profit_pullback_capture"},
            {"agent": "failed_entry_escape_agent", "buy": 0.0, "sell": clamp(loss_exit * 0.55 + max(0.0, -net_after_exit_bps) / 120.0 + max(0.0, -momentum_1_bps) / 160.0, 0.0, 1.0), "hold": clamp(0.75 - loss_exit * 0.55, 0.0, 1.0), "wait": 0.25, "confidence": 0.78, "reason": "new_sell_alpha_failed_entry_escape"},
            {"agent": "hard_stop_prevention_agent", "buy": 0.0, "sell": clamp(loss_exit * 0.70 + max(0.0, -net_after_exit_bps) / 160.0, 0.0, 1.0), "hold": clamp(0.80 - loss_exit * 0.65, 0.0, 1.0), "wait": 0.25, "confidence": 0.76, "reason": "new_sell_alpha_hard_stop_prevention"},
            {"agent": "max_hold_decay_agent", "buy": 0.0, "sell": clamp(max(0.0, hold_minutes - max_hold_minutes) / max(max_hold_minutes, 1.0), 0.0, 1.0), "hold": clamp(1.0 - max(0.0, hold_minutes - max_hold_minutes) / max(max_hold_minutes, 1.0), 0.0, 1.0), "wait": 0.20, "confidence": 0.62, "reason": "new_sell_alpha_max_hold_decay"},
            {"agent": "execution", "buy": 0.0, "sell": execution_sell_quality, "hold": 0.40, "wait": 1.0 - execution_sell_quality, "confidence": 0.65, "reason": "avoid selling into poor spread conditions unless necessary"},
            {"agent": "fee_recovery", "buy": 0.0, "sell": fee_recovery, "hold": clamp(0.85 - fee_recovery * 0.35, 0.0, 1.0), "wait": 0.30, "confidence": 0.67, "reason": "protect all-in fee-adjusted breakeven"},
            {"agent": "harvest_sizing", "buy": 0.0, "sell": harvest_pressure, "hold": 1.0 - harvest_pressure * 0.55, "wait": 0.25, "confidence": 0.64, "reason": "choose partial vs full sell pressure"},
            {
                "agent": f"{session_agent}_sell_context",
                "buy": 0.0,
                "sell": clamp(session_sell_score * session_confidence, 0.0, 1.0),
                "hold": clamp(
                    session_hold_score * 0.60
                    + session_buy_score * session_confidence * 0.25
                    + 0.15,
                    0.0,
                    1.0,
                ),
                "wait": clamp(0.55 - session_sell_score * session_confidence * 0.25, 0.0, 1.0),
                "confidence": clamp(0.25 + session_confidence * 0.55, 0.15, 0.85),
                "reason": f"session_liquidity_sell_context;{session_reason}",
            },
            {
                "agent": "candle_exhaustion_sell",
                "buy": 0.0,
                "sell": clamp(candle_exhaustion_score * pa_confidence, 0.0, 1.0),
                "hold": clamp(0.30 + candle_continuation_score * 0.55 - candle_exhaustion_score * 0.20, 0.0, 1.0),
                "wait": 0.25,
                "confidence": clamp(0.25 + pa_confidence * 0.55, 0.15, 0.85),
                "reason": f"candle_exhaustion_sell;{pa_reason}",
            },
            {
                "agent": "volume_profile_harvest",
                "buy": 0.0,
                "sell": clamp(volume_profile_sell_score, 0.0, 1.0),
                "hold": clamp(volume_profile_hold_score, 0.0, 1.0),
                "wait": 0.30,
                "confidence": clamp(0.25 + pa_confidence * 0.45, 0.15, 0.80),
                "reason": f"volume_profile_harvest;value_area_state={value_area_state};{pa_reason}",
            },
            {
                "agent": "volume_profile_leader_exit",
                "buy": 0.0,
                "sell": clamp(volume_profile_leader_sell_score, 0.0, 1.0),
                "hold": clamp(volume_profile_leader_hold_score, 0.0, 1.0),
                "wait": clamp(volume_profile_leader_wait_score, 0.0, 1.0),
                "confidence": 0.78,
                "reason": (
                    f"volume_profile_leader_exit;"
                    f"value_acceptance_state={value_acceptance_state};"
                    f"volume_node_state={volume_node_state};"
                    f"sell={volume_profile_leader_sell_score:.3f};"
                    f"hold={volume_profile_leader_hold_score:.3f};"
                    f"wait={volume_profile_leader_wait_score:.3f};"
                    f"{pa_reason}"
                ),
            },
            {
                "agent": "previous_session_profile_exit",
                "buy": 0.0,
                "sell": clamp(previous_session_sell_score, 0.0, 1.0),
                "hold": clamp(previous_session_hold_score, 0.0, 1.0),
                "wait": clamp(previous_session_wait_score, 0.0, 1.0),
                "confidence": clamp(float(previous_session_profile.get("confidence", 0.10) or 0.10), 0.10, 0.90),
                "reason": f"previous_session_profile_exit;reaction={previous_session_reaction};{previous_session_reason}",
            },
            {
                "agent": "quant_boundary_exit",
                "buy": 0.0,
                "sell": clamp(quant_sell_score, 0.0, 1.0),
                "hold": clamp(quant_hold_score, 0.0, 1.0),
                "wait": clamp(quant_wait_score, 0.0, 1.0),
                "confidence": clamp(float(quant_context.get("confidence", 0.10) or 0.10), 0.10, 0.90),
                "reason": f"quant_boundary_exit;boundary={quant_boundary_state};vol={quant_vol_state};{quant_reason}",
            },
            {
                "agent": "fvg_reclaim_rejection_exit",
                "buy": 0.0,
                "sell": clamp(fvg_sell_score, 0.0, 1.0),
                "hold": clamp(0.55 - fvg_sell_score * 0.25, 0.0, 1.0),
                "wait": 0.32,
                "confidence": clamp(0.25 + pa_confidence * 0.40, 0.15, 0.78),
                "reason": f"fvg_exit_context;fvg_state={fvg_state};{pa_reason}",
            },
            {
                "agent": "smt_divergence_exit",
                "buy": 0.0,
                "sell": clamp(smt_sell_score * smt_confidence, 0.0, 1.0),
                "hold": clamp(0.52 - smt_sell_score * smt_confidence * 0.25, 0.0, 1.0),
                "wait": 0.35,
                "confidence": clamp(0.20 + smt_confidence * 0.50, 0.10, 0.75),
                "reason": f"smt_divergence_exit;{smt_reason}",
            },
            {
                "agent": "sell_utility_leader",
                "buy": 0.0,
                "sell": clamp(0.45 + sell_minus_hold_utility_bps / 180.0, 0.0, 1.0),
                "hold": clamp(0.45 + hold_utility_bps / 180.0, 0.0, 1.0),
                "wait": 0.20,
                "confidence": clamp(0.35 + abs(sell_minus_hold_utility_bps) / 260.0, 0.20, 0.88),
                "reason": f"sell_utility_leader;{sell_utility_reason}",
            },
            {
                "agent": "spike_profit_protection",
                "buy": 0.0,
                "sell": clamp(
                    0.20
                    + (0.46 if spike_armed else 0.0)
                    + (0.28 if spike_allow_partial else 0.0)
                    + (0.18 if spike_immediate_partial else 0.0),
                    0.0,
                    1.0,
                ),
                "hold": clamp(
                    0.55
                    - (0.25 if spike_allow_partial else 0.0)
                    - (0.18 if spike_immediate_partial else 0.0),
                    0.0,
                    1.0,
                ),
                "wait": 0.20,
                "confidence": clamp(
                    0.30
                    + (0.28 if spike_armed else 0.0)
                    + (0.22 if spike_allow_partial else 0.0),
                    0.15,
                    0.88,
                ),
                "reason": f"spike_profit_protection_vote;{spike_reason}",
            },
            {
                "agent": "adaptive_wave_capture",
                "buy": 0.0,
                "sell": adaptive_wave_pressure,
                "hold": clamp(0.62 + (0.20 if adaptive_strong_continuation else 0.0) - adaptive_wave_pressure * 0.45, 0.0, 1.0),
                "wait": 0.20,
                "confidence": 0.82 if adaptive_enabled else 0.15,
                "reason": f"adaptive_wave_capture;{adaptive_reason}",
            },
        ]

        intersection_sell_votes = [
            {"agent": "profit_pullback_wave_agent", "buy": 0.0, "sell": clamp(profit_capture * 0.45 + peak_capture * 0.35 + max(0.0, pullback_from_peak_bps) / 240.0, 0.0, 1.0), "hold": clamp(1.0 - peak_capture * 0.55, 0.0, 1.0), "wait": 0.20, "confidence": 0.86, "reason": "sell_path_intersection_profit_pullback_wave"},
            {"agent": "higher_low_wave_stop_agent", "buy": 0.0, "sell": 1.0 if bool(context.get("wave_stop_exit_confirmed", False)) else 0.15, "hold": 0.15 if bool(context.get("wave_stop_exit_confirmed", False)) else 0.85, "wait": 0.20, "confidence": 0.86 if bool(context.get("wave_stop_exit_confirmed", False)) else 0.55, "reason": "sell_path_intersection_higher_low_stop"},
            {"agent": "wick_exhaustion_sell_agent", "buy": 0.0, "sell": clamp(candle_exhaustion_score * pa_confidence + max(0.0, net_after_exit_bps) / 260.0, 0.0, 1.0), "hold": clamp(0.70 - candle_exhaustion_score * 0.35, 0.0, 1.0), "wait": 0.25, "confidence": clamp(0.25 + pa_confidence * 0.55, 0.15, 0.85), "reason": f"sell_path_intersection_wick_exhaustion;{pa_reason}"},
            {"agent": "liquidity_target_hit_agent", "buy": 0.0, "sell": clamp(session_sell_score * session_confidence + volume_profile_leader_sell_score * 0.25, 0.0, 1.0), "hold": clamp(session_hold_score, 0.0, 1.0), "wait": 0.25, "confidence": clamp(0.30 + session_confidence * 0.50, 0.15, 0.85), "reason": f"sell_path_intersection_liquidity_target;{session_reason}"},
            {"agent": "failed_run_escape_agent", "buy": 0.0, "sell": clamp(loss_exit * 0.50 + max(0.0, -net_after_exit_bps) / 120.0 + max(0.0, -momentum_1_bps) / 160.0, 0.0, 1.0), "hold": clamp(0.75 - loss_exit * 0.55, 0.0, 1.0), "wait": 0.25, "confidence": 0.78, "reason": "sell_path_intersection_failed_run_escape"},
            {"agent": "analog_sell_path_agent", "buy": 0.0, "sell": clamp(adaptive_wave_pressure * 0.35 + peak_capture * 0.25 + momentum_fade_sell * 0.20 + loss_exit * 0.20, 0.0, 1.0), "hold": clamp(continuation_hold * 0.65, 0.0, 1.0), "wait": 0.20, "confidence": 0.72 if adaptive_enabled else 0.35, "reason": f"sell_path_intersection_analog;{adaptive_reason}"},
        ]
        votes = intersection_sell_votes

        truth_vote = {
            "agent": "exit_truth",
            "buy": 0.0,
            "sell": clamp(
                profit_capture * 0.15
                + loss_exit * 0.17
                + peak_capture * 0.17
                + momentum_fade_sell * 0.13
                + execution_sell_quality * 0.09
                + fee_recovery * 0.10
                + harvest_pressure * 0.09
                + price_action_harvest_pressure * 0.08
                + adaptive_wave_pressure * 0.12,
                0.0,
                1.0,
            ),
            "hold": clamp(
                continuation_hold * 0.70
                + candle_continuation_score * pa_confidence * 0.18
                + volume_profile_hold_score * 0.06
                + volume_profile_leader_hold_score * 0.18
                + previous_session_hold_score * 0.10
                + quant_hold_score * 0.08,
                0.0,
                1.0,
            ),
            "wait": 1.0 - execution_sell_quality,
            "confidence": 0.72,
            "reason": "exit truth weighs sell-path intersections: realized net bps, giveback, wick exhaustion, liquidity target, higher-low stop, failed-run escape, and max-hold decay",
        }

        adjusted = [self._adjust_vote(vote, product_id, "EXIT_REVIEW") for vote in votes]
        adjusted_truth = self._adjust_vote(truth_vote, product_id, "EXIT_REVIEW")
        weighted = self._weighted_vote_pairs(adjusted, decision_side="SELL", product_id=product_id)
        weight_total = sum(weight for _, weight in weighted) or 1.0
        final_sell = clamp(
            sum(v.adjusted_sell_score * w for v, w in weighted) / weight_total
            - sell_quality_penalty,
            0.0,
            1.0,
        )
        final_hold = clamp(sum(v.adjusted_hold_score * w for v, w in weighted) / weight_total, 0.0, 1.0)
        truth_score = clamp(adjusted_truth.adjusted_sell_score * 0.55 + adjusted_truth.confidence * 0.25 + adjusted_truth.reliability * 0.20, 0.0, 1.0)

        thresholds = self.adaptive_thresholds(product_id, "EXIT_REVIEW")
        sell_threshold = float(thresholds["sell_threshold"])
        raw_strong_continuation = bool(
            continuation_quality >= 0.68
            and momentum_3_bps >= strong_continuation_mom3_bps
            and pullback_from_peak_bps <= strong_continuation_pullback_max_bps
            and not max_hold_elapsed
            and not hard_stop_hit
        )

        continuation_override_by_harvest = bool(
            price_action_harvest_pressure >= 0.68
            or candle_exhaustion_score >= 0.72
            or volume_profile_sell_score >= 0.72
            or volume_profile_leader_sell_score >= 0.66
            or fvg_sell_score >= 0.70
            or (session_sell_score * session_confidence) >= 0.55
            or (smt_sell_score * smt_confidence) >= 0.55
            or spike_allow_partial
            or spike_immediate_partial
        )

        strong_continuation = bool(
            (raw_strong_continuation or adaptive_strong_continuation)
            and not continuation_override_by_harvest
            and not adaptive_partial_harvest
            and not adaptive_full_exit
            and not adaptive_floor_exit_confirmed
        )

        if hard_stop_hit:
            recommended_sell_fraction = 1.0
            sell_fraction_reason = "hard_stop_full_exit"
        elif adaptive_floor_exit_confirmed or adaptive_full_exit:
            recommended_sell_fraction = 1.0
            sell_fraction_reason = f"adaptive_full_exit;{adaptive_reason}"
        elif net_after_exit_bps < min_net_after_exit_bps:
            recommended_sell_fraction = 0.0
            sell_fraction_reason = f"not_net_profitable_enough;net_after_exit_bps={net_after_exit_bps:.2f};min={min_net_after_exit_bps:.2f}"
        elif strong_continuation:
            recommended_sell_fraction = 0.0
            sell_fraction_reason = f"strong_continuation_hold;mom3={momentum_3_bps:.2f};pullback={pullback_from_peak_bps:.2f};continuation_quality={continuation_quality:.3f}"
        else:
            fraction = min_partial_fraction
            fraction += clamp((net_after_exit_bps - min_net_after_exit_bps) / 260.0, 0.0, 0.25)
            fraction += clamp((pullback_from_peak_bps - peak_capture_trigger_bps) / 170.0, 0.0, 0.25)
            fraction += clamp((final_sell - sell_threshold) / 0.35, 0.0, 0.20)
            fraction += clamp((harvest_pressure - 0.50) / 1.50, 0.0, 0.15)
            if (
                continuation_quality >= 0.64
                and momentum_3_bps > 0
                and pullback_from_peak_bps < strong_pullback_bps
                and not continuation_override_by_harvest
            ):
                fraction -= 0.18
            if continuation_override_by_harvest and net_after_exit_bps >= min_net_after_exit_bps:
                fraction = max(fraction, 0.35)

            if sell_minus_hold_utility_bps >= 20.0:
                fraction = max(fraction, 0.35)

            if sell_utility_fraction > 0.0:
                fraction = max(fraction, sell_utility_fraction)
            if spike_allow_partial and spike_suggested_fraction > 0.0:
                fraction = max(fraction, spike_suggested_fraction)
            if adaptive_partial_harvest and adaptive_harvest_fraction > 0.0:
                fraction = max(fraction, adaptive_harvest_fraction)
            if adaptive_strong_continuation and not adaptive_partial_harvest and not adaptive_full_exit:
                fraction = max(0.0, fraction - 0.20)
            if max_hold_elapsed:
                fraction = max(fraction, 0.50)
            if pullback_from_peak_bps >= strong_pullback_bps and peak_unrealized_bps > min_net_after_exit_bps:
                fraction = max(fraction, 0.65)
            if pullback_from_peak_bps >= full_exit_pullback_bps:
                fraction = 1.0
            recommended_sell_fraction = clamp(fraction, min_partial_fraction, max_partial_fraction)
            sell_fraction_reason = f"fraction_from_profit_peak_momentum;net_after_exit_bps={net_after_exit_bps:.2f};peak_unrealized_bps={peak_unrealized_bps:.2f};pullback_from_peak_bps={pullback_from_peak_bps:.2f};mom1={momentum_1_bps:.2f};mom3={momentum_3_bps:.2f};mom5={momentum_5_bps:.2f};harvest_pressure={harvest_pressure:.3f};continuation_quality={continuation_quality:.3f}"

        harvest_confirmation = bool(
            pullback_from_peak_bps >= peak_capture_trigger_bps
            or momentum_1_bps < 0.0
            or momentum_3_bps < 0.0
            or max_hold_elapsed
            or target_hold_elapsed
            or peak_capture >= 0.62
            or momentum_fade_sell >= 0.62
            or (session_sell_score * session_confidence) >= 0.38
            or candle_exhaustion_score >= 0.58
            or price_action_harvest_pressure >= 0.55
            or volume_profile_sell_score >= 0.58
            or volume_profile_leader_sell_score >= 0.58
            or value_acceptance_state in {"rejected_above_value", "accepted_below_value"}
            or previous_session_reaction in {"rejected_prior_vah", "accepted_below_prior_val", "rejected_prior_poc"}
            or quant_boundary_state == "above_upper_boundary_stretched"
            or fvg_sell_score >= 0.58
            or (smt_sell_score * smt_confidence) >= 0.38
            or sell_minus_hold_utility_bps >= 20.0
            or spike_allow_partial
            or spike_immediate_partial
        )

        strong_profit_exception = bool(
            net_after_exit_bps >= min_net_after_exit_bps + 140.0
            and (
                final_sell >= sell_threshold + 0.08
                or spike_immediate_partial
                or spike_allow_partial
                or peak_capture >= 0.72
                or volume_profile_leader_sell_score >= 0.70
                or previous_session_reaction in {"rejected_prior_vah", "rejected_prior_poc"}
                or quant_boundary_state == "above_upper_boundary_stretched"
            )
            and recommended_sell_fraction > 0.0
        )

        if hard_stop_hit or adaptive_floor_exit_confirmed or adaptive_full_exit:
            action = "ALLOW_SELL"

        elif (
            strong_profit_exception
            and recommended_sell_fraction > 0.0
        ):
            action = "ALLOW_SELL"

        elif strong_continuation:
            action = "HOLD"

        elif (
            early_profit_ok
            and harvest_confirmation
            and final_sell >= sell_threshold
            and net_after_exit_bps >= min_net_after_exit_bps
            and recommended_sell_fraction > 0.0
        ):
            action = "ALLOW_SELL"

        elif (
            max_hold_elapsed
            and final_sell >= sell_threshold
            and net_after_exit_bps >= min_net_after_exit_bps
            and recommended_sell_fraction > 0.0
        ):
            action = "ALLOW_SELL"

        else:
            action = "HOLD"

        debug_every(
            MODULE_NAME,
            f"decide_exit:{product_id}",
            10.0,
            "level8_decide_exit",
            data={
                "decision_id": decision_id,
                "product_id": product_id,
                "action": action,
                "final_sell": final_sell,
                "final_hold": final_hold,
                "truth_score": truth_score,
                "sell_threshold": sell_threshold,
                "recommended_sell_fraction": recommended_sell_fraction,
                "net_after_exit_bps": net_after_exit_bps,
                "peak_unrealized_bps": peak_unrealized_bps,
                "pullback_from_peak_bps": pullback_from_peak_bps,
                "spike_armed": spike_armed,
                "spike_allow_partial": spike_allow_partial,
                "sell_quality_penalty": sell_quality_penalty,
                "vote_count": len(votes),
                "adjusted_vote_count": len(adjusted),
                "weight_total": weight_total,
            },
            level="DEBUG",
            also_overall=False,
        )

        return {
            "decision_id": decision_id,
            "action": action,
            "final_sell": final_sell,
            "final_hold": final_hold,
            "truth_score": truth_score,
            "sell_threshold": sell_threshold,
            "buy_threshold": thresholds["buy_threshold"],
            "risk_mode": thresholds["risk_mode"],
            "recommended_sell_fraction": float(recommended_sell_fraction),
            "sell_fraction_reason": sell_fraction_reason,
            "votes": [asdict(vote) for vote in adjusted],
            "truth_vote": asdict(adjusted_truth),
            "reason": (
                f"exit_council;unrealized_bps={unrealized_bps:.2f};spread_bps={spread_bps:.2f};cost_bps={cost_bps:.2f};"
                f"final_sell={final_sell:.3f};final_hold={final_hold:.3f};threshold={sell_threshold:.3f};truth={truth_score:.3f};"
                f"hold_seconds={hold_seconds:.1f};min_hold_elapsed={min_hold_elapsed};target_hold_elapsed={target_hold_elapsed};max_hold_elapsed={max_hold_elapsed};"
                f"net_after_exit_bps={net_after_exit_bps:.2f};min_net_after_exit_bps={min_net_after_exit_bps:.2f};peak_unrealized_bps={peak_unrealized_bps:.2f};"
                f"pullback_from_peak_bps={pullback_from_peak_bps:.2f};mom1={momentum_1_bps:.2f};mom3={momentum_3_bps:.2f};mom5={momentum_5_bps:.2f};"
                f"green={green_candles};"
                f"raw_strong_continuation={raw_strong_continuation};"
                f"strong_continuation={strong_continuation};"
                f"continuation_override_by_harvest={continuation_override_by_harvest};"
                f"harvest_confirmation={harvest_confirmation};"
                f"strong_profit_exception={strong_profit_exception};"
                f"sell_quality_penalty={sell_quality_penalty:.3f};"
                f"sell_quality_reason={sell_quality_reason};"
                f"session_sell_score={session_sell_score:.3f};"
                f"session_buy_score={session_buy_score:.3f};"
                f"session_confidence={session_confidence:.3f};"
                f"session_agent={session_agent};"
                f"candle_exhaustion_score={candle_exhaustion_score:.3f};"
                f"candle_continuation_score={candle_continuation_score:.3f};"
                f"price_action_harvest_pressure={price_action_harvest_pressure:.3f};"
                f"value_area_state={value_area_state};"
                f"value_acceptance_state={value_acceptance_state};"
                f"volume_node_state={volume_node_state};"
                f"volume_profile_leader_sell={volume_profile_leader_sell_score:.3f};"
                f"volume_profile_leader_hold={volume_profile_leader_hold_score:.3f};"
                f"previous_session_reaction={previous_session_reaction};"
                f"previous_session_sell={previous_session_sell_score:.3f};"
                f"quant_boundary={quant_boundary_state};"
                f"quant_sell={quant_sell_score:.3f};"
                f"fvg_state={fvg_state};"
                f"smt_reason={smt_reason};"
                f"recommended_sell_fraction={recommended_sell_fraction:.3f};"
                f"sell_fraction_reason={sell_fraction_reason};"
                f"sell_utility_hold={hold_utility_bps:.2f};"
                f"sell_utility_sell={sell_utility_bps:.2f};"
                f"sell_minus_hold_utility={sell_minus_hold_utility_bps:.2f};"
                f"spike_profit_armed={spike_armed};"
                f"spike_profit_partial={spike_allow_partial};"
                f"spike_profit_immediate={spike_immediate_partial};"
                f"spike_profit_fraction={spike_suggested_fraction:.3f};"
                f"spike_profit_reason={spike_reason};"
                f"adaptive_enabled={adaptive_enabled};"
                f"adaptive_expected_favorable_bps={adaptive_expected_favorable_bps:.2f};"
                f"adaptive_progress_to_expected={adaptive_progress_to_expected:.3f};"
                f"adaptive_protection_armed={adaptive_protection_armed};"
                f"adaptive_floor_exit_confirmed={adaptive_floor_exit_confirmed};"
                f"adaptive_partial_harvest={adaptive_partial_harvest};"
                f"adaptive_full_exit={adaptive_full_exit};"
                f"adaptive_harvest_fraction={adaptive_harvest_fraction:.3f};"
                f"adaptive_strong_continuation={adaptive_strong_continuation};"
                f"adaptive_wave_pressure={adaptive_wave_pressure:.3f};"
                f"adaptive_reason={adaptive_reason};"
                f"hard_stop_hit={hard_stop_hit}"
            ),
        }

    def _position_pct_from_decision(
        self,
        *,
        final_buy_score: float,
        threshold: float,
        truth_score: float,
        risk_mode: str,
    ) -> Tuple[str, float, str]:
        """
        High-conviction adaptive sizing.

        Rules:
        - below threshold = SHADOW
        - below truth floor = SHADOW
        - approved live trade = never below 50%
        - core/high-conviction trade = 80%-100%
        """
        margin = float(final_buy_score) - float(threshold)

        if margin < 0:
            return "SHADOW", 0.0, (
                f"below_threshold_shadow_only;"
                f"margin={margin:.3f};truth={truth_score:.3f}"
            )

        if truth_score < self.min_truth_to_trade:
            return "SHADOW", 0.0, (
                f"truth_below_live_trade_min;"
                f"margin={margin:.3f};truth={truth_score:.3f};"
                f"min_truth={self.min_truth_to_trade:.3f}"
            )

        risk_mode_u = str(risk_mode).upper()

        conviction = clamp(
            margin * 2.10
            + max(0.0, truth_score - self.min_truth_to_trade) * 1.65,
            0.0,
            1.0,
        )

        bucket = "APPROVED"
        base_pct = 0.50
        ceiling_pct = 0.80

        if truth_score >= self.min_truth_to_core_trade or margin >= 0.10:
            bucket = "CORE"
            base_pct = self.min_core_trade_pct
            ceiling_pct = self.max_core_trade_pct

        pct = base_pct + conviction * (ceiling_pct - base_pct)

        if risk_mode_u == "DEFENSIVE":
            pct *= 0.90
        elif risk_mode_u == "CAUTIOUS":
            pct *= 0.95
        elif risk_mode_u == "AGGRESSIVE":
            pct *= 1.05

        pct = clamp(pct, 0.50, self.max_single_asset_pct)

        return bucket, pct, (
            f"high_conviction_{bucket.lower()}_bucket;"
            f"margin={margin:.3f};truth={truth_score:.3f};"
            f"risk={risk_mode_u};conviction={conviction:.3f};"
            f"pct={pct:.3f};sizing_floor=50pct"
        )
