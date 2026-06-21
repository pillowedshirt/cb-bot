import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import pickle
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd



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
BACKTEST_RECOMMENDATIONS_COLUMNS: List[str] = [
    "ts", "dt_utc", "product_id", "sample_count", "accepted_count",
    "sample_confidence", "replay_quality_score", "live_gate_bias_bps",
    "recommended_min_score", "recommended_min_probability",
    "recommended_min_expected_value_bps", "recommended_min_projected_net_bps",
    "recommended_forward_window_minutes", "expected_win_rate", "expected_net_bps",
    "expected_adverse_bps", "objective_score",
    "dominant_session_agent", "dominant_session_setup",
    "dominant_structure_state", "dominant_value_area_state",
    "dominant_fvg_state", "dominant_smt_state",
    "source", "reason",
]


BACKTEST_SELL_RECOMMENDATIONS_COLUMNS: List[str] = [
    "ts", "dt_utc", "product_id", "sell_samples", "too_early_rate",
    "good_exit_rate", "avg_move_after_sell_bps", "recommended_peak_capture_trigger_bps",
    "recommended_strong_pullback_bps", "recommended_full_exit_pullback_bps",
    "sell_objective_score", "source", "reason",
]

BACKTEST_AGENT_PRIOR_COLUMNS: List[str] = [
    "ts", "dt_utc", "decision_id", "product_id", "strategy", "agent",
    "agent_buy_score", "agent_sell_score", "agent_hold_score", "agent_wait_score",
    "confidence", "reliability", "weight", "leaderboard_rank", "leaderboard_score",
    "leader_bonus", "leader_penalty", "outcome_source", "outcome_weight",
    "review_minutes", "outcome_move_bps", "outcome_kind", "agent_credit_score",
    "weighted_agent_credit_score", "outcome_success", "reason",
]

BACKTEST_SUMMARY_COLUMNS: List[str] = ["ts", "dt_utc", "metric", "value", "reason"]

BACKTEST_SETUP_PERFORMANCE_COLUMNS: List[str] = [
    "ts", "dt_utc", "product_id", "setup_key", "sample_count",
    "win_rate", "avg_net_bps", "avg_adverse_bps", "objective_score", "reason",
]

WALK_FORWARD_VALIDATION_COLUMNS: List[str] = [
    "ts",
    "dt_utc",
    "product_id",
    "fold",
    "train_rows",
    "test_rows",
    "train_win_rate",
    "test_win_rate",
    "train_avg_net_bps",
    "test_avg_net_bps",
    "generalization_gap",
    "walk_forward_score",
    "reason",
]

AGENT_ABLATION_COLUMNS: List[str] = [
    "ts",
    "dt_utc",
    "agent",
    "sample_count",
    "avg_outcome_all_bps",
    "avg_when_supportive_bps",
    "avg_when_not_supportive_bps",
    "support_edge_bps",
    "win_rate_when_supportive",
    "win_rate_when_not_supportive",
    "ablation_score",
    "reason",
]

FOUR_PASS_AGENT_BUY_COLUMNS: List[str] = [
    "ts", "dt_utc", "pass_name", "agent", "source_column",
    "sample_count", "selected_count", "threshold",
    "win_rate", "avg_net_bps", "median_net_bps", "avg_adverse_bps",
    "score", "raw_authority", "buy_weight_pct", "reason",
]

FOUR_PASS_COUNCIL_BUY_COLUMNS: List[str] = [
    "ts", "dt_utc", "pass_name", "product_id", "sample_count",
    "selected_count", "threshold", "win_rate", "avg_net_bps",
    "median_net_bps", "avg_adverse_bps", "portfolio_return_pct_100_ref",
    "score", "profitability_mode", "reason",
]

FOUR_PASS_AGENT_SELL_COLUMNS: List[str] = [
    "ts", "dt_utc", "pass_name", "agent", "source_column",
    "sample_count", "selected_count", "threshold",
    "good_exit_rate", "too_early_rate", "avg_move_after_sell_bps",
    "avg_realized_net_bps", "score", "raw_authority",
    "sell_weight_pct", "reason",
]

FOUR_PASS_COUNCIL_SELL_COLUMNS: List[str] = [
    "ts", "dt_utc", "pass_name", "product_id", "sample_count",
    "selected_count", "threshold", "good_exit_rate", "too_early_rate",
    "avg_move_after_sell_bps", "avg_realized_net_bps",
    "portfolio_return_pct_100_ref", "score", "profitability_mode", "reason",
]

FOUR_PASS_FINAL_AGENT_RATINGS_COLUMNS: List[str] = [
    "ts", "dt_utc", "agent",
    "buy_rows", "buy_accuracy", "buy_avg_net_bps", "buy_score", "buy_weight_pct",
    "sell_rows", "sell_accuracy", "sell_avg_net_bps", "sell_score", "sell_weight_pct",
    "profitability_mode", "reason",
]

FOUR_PASS_PROFITABILITY_SUMMARY_COLUMNS: List[str] = [
    "ts", "dt_utc",
    "buy_agent_rows", "buy_council_rows", "sell_agent_rows", "sell_council_rows",
    "buy_profitability_mode", "sell_profitability_mode",
    "buy_council_reference_return_pct", "sell_council_reference_return_pct",
    "final_reference_return_pct",
    "buy_council_positive_products", "sell_council_positive_products",
    "verdict", "reason",
]

FOUR_PASS_AGENT_CONTEXT_RATINGS_COLUMNS: List[str] = [
    "ts", "dt_utc",
    "agent", "side", "product_id", "market_regime",
    "source_column", "sample_count", "selected_count", "threshold",
    "raw_win_rate", "smoothed_win_rate",
    "avg_win_bps", "avg_loss_bps", "ev_bps",
    "avg_net_bps", "median_net_bps", "avg_adverse_bps",
    "score", "raw_authority", "weight_pct",
    "profitability_mode", "reason",
]

FOUR_PASS_SELL_PATH_REPLAY_COLUMNS = [
    "ts", "dt_utc",
    "product_id", "entry_ts", "exit_ts",
    "entry_price", "exit_price",
    "held_minutes", "exit_agent", "exit_score",
    "exit_reason", "realized_net_bps",
    "max_favorable_bps", "max_adverse_bps",
    "profitability_mode", "reason",
]

FOUR_PASS_PURGED_WALK_FORWARD_COLUMNS = [
    "ts", "dt_utc",
    "fold_id", "side", "train_start_ts", "train_end_ts",
    "embargo_start_ts", "embargo_end_ts",
    "validation_start_ts", "validation_end_ts",
    "train_rows", "validation_rows",
    "train_win_rate", "validation_win_rate",
    "train_avg_net_bps", "validation_avg_net_bps",
    "train_median_net_bps", "validation_median_net_bps",
    "validation_reference_return_pct",
    "verdict", "reason",
]

FOUR_PASS_PRODUCT_LIVE_GATE_COLUMNS = [
    "ts", "dt_utc",
    "product_id",
    "buy_selected_count", "buy_win_rate", "buy_avg_net_bps", "buy_median_net_bps",
    "buy_score", "buy_profitability_mode",
    "walk_forward_folds", "walk_forward_positive_folds",
    "walk_forward_avg_validation_net_bps",
    "walk_forward_avg_validation_return_pct",
    "sell_path_rows", "sell_path_avg_realized_net_bps",
    "sell_path_total_reference_return_pct",
    "approved_for_live_buy",
    "cooldown_minutes",
    "gate_reason",
]

PRODUCT_COOLDOWN_COLUMNS = [
    "ts", "dt_utc",
    "product_id", "cooldown_until_ts", "cooldown_minutes",
    "cooldown_type",
    "can_escape_early",
    "reason",
]

AGENT_DECISION_INFLUENCE_COLUMNS = [
    "ts", "dt_utc",
    "agent", "side",
    "sample_count", "selected_count",
    "frequency_per_day",
    "raw_win_rate", "smoothed_win_rate",
    "avg_win_bps", "avg_loss_bps",
    "avg_net_bps", "median_net_bps",
    "ev_bps", "avg_adverse_bps",
    "reliability_score", "frequency_score",
    "edge_score", "decision_influence_score",
    "decision_weight_pct",
    "role",
    "reason",
]

PRODUCT_AGENT_INFLUENCE_COLUMNS = [
    "ts", "dt_utc",
    "product_id", "market_regime",
    "agent", "side",
    "selected_count", "frequency_per_day",
    "smoothed_win_rate", "ev_bps",
    "avg_net_bps", "median_net_bps",
    "decision_influence_score",
    "decision_weight_pct",
    "role",
    "reason",
]

TRADE_FREQUENCY_ESTIMATE_COLUMNS = [
    "ts", "dt_utc",
    "scope", "product_id",
    "dedupe_minutes",
    "candidate_days",
    "raw_selected_count",
    "deduped_trade_count",
    "estimated_trades_per_day",
    "win_rate",
    "avg_net_bps", "median_net_bps",
    "avg_win_bps", "avg_loss_bps",
    "expected_net_per_trade_bps",
    "expected_daily_net_bps_if_all_traded",
    "reason",
]

FIFTH_PASS_LIVE_STYLE_REPLAY_COLUMNS = [
    "ts", "dt_utc",
    "product_id",
    "entry_ts",
    "exit_ts",
    "held_minutes",
    "entry_score",
    "entry_threshold",
    "market_eligible",
    "market_eligibility_reason",
    "realized_or_proxy_net_bps",
    "win",
    "source_mode",
    "reason",
]

FIFTH_PASS_LIVE_STYLE_SUMMARY_COLUMNS = [
    "ts", "dt_utc",
    "scope",
    "product_id",
    "replay_days",
    "raw_candidate_count",
    "deduped_trade_count",
    "estimated_trades_per_day",
    "win_rate",
    "avg_net_bps",
    "median_net_bps",
    "avg_win_bps",
    "avg_loss_bps",
    "reference_return_pct_5pct_size",
    "verdict",
    "reason",
]

FIFTH_PASS_PRODUCT_CONTRIBUTION_COLUMNS = [
    "ts", "dt_utc",
    "product_id",
    "trade_count",
    "estimated_trades_per_day",
    "win_rate",
    "avg_net_bps",
    "median_net_bps",
    "reference_return_pct_5pct_size",
    "contribution_rank",
    "reason",
]

FIFTH_PASS_BLOCKER_COLUMNS = [
    "ts", "dt_utc",
    "product_id",
    "blocker",
    "count",
    "reason",
]

APPROVED_BUT_SHADOWED_COLUMNS = [
    "ts", "dt_mst",
    "product_id", "symbol", "quote_asset",
    "product_gate_approved",
    "council_action",
    "expected_utility_bps",
    "candidate_notional_usd",
    "top_of_book_age_sec",
    "block_reasons",
    "next_best_action",
]

FEATURE_STORE_SUMMARY_COLUMNS = [
    "ts", "dt_utc",
    "feature_store_path", "row_count", "column_count",
    "source_files", "cache_mode", "reason",
]

BUY_AGENT_SCORE_COLUMNS = {
    "price_action": "price_action_buy_score",
    "market_structure_agent": "market_structure_buy_score",
    "validated_liquidity_agent": "validated_liquidity_buy_score",
    "fresh_zone_retest_agent": "fresh_zone_buy_score",
    "fair_value_gap_agent": "fvg_buy_score",
    "volume_profile_agent": "volume_profile_buy_score",
    "volume_profile_leader": "volume_profile_leader_buy_score",
    "previous_session_volume_profile_agent": "previous_session_profile_buy_score",
    "trend": "trend_buy_score",
    "quant_boundary_agent": "quant_buy_score",
    "candle_sequence_agent": "candle_sequence_score",
    "candle_continuation_agent": "candle_continuation_score",
}

SELL_AGENT_SCORE_COLUMNS = {
    "price_action": "price_action_sell_score",
    "volume_profile_leader": "volume_profile_leader_sell_score",
    "previous_session_volume_profile_agent": "previous_session_profile_sell_score",
    "quant_boundary_agent": "quant_sell_score",
    "candle_exhaustion_agent": "candle_exhaustion_score",
    "exit_truth": "exit_truth_sell_score",
    "sell_utility_leader": "sell_utility_score",
    "drawdown_exit": "drawdown_exit_score",
    "fee_recovery": "fee_recovery_score",
    "market_structure_agent": "market_structure_sell_score",
    "volume_profile_harvest": "volume_profile_sell_score",
}

AGENT_CANONICAL_ALIASES = {
    "validated_liquidity": "validated_liquidity_agent",
    "fresh_zone": "fresh_zone_retest_agent",
    "fvg": "fair_value_gap_agent",
    "fair_value_gap": "fair_value_gap_agent",
    "volume_profile": "volume_profile_agent",
    "volume_profile_leader_exit": "volume_profile_leader",
    "market_structure": "market_structure_agent",
    "previous_session_profile": "previous_session_volume_profile_agent",
    "previous_session_volume_profile": "previous_session_volume_profile_agent",
    "quant": "quant_boundary_agent",
    "quant_boundary": "quant_boundary_agent",
    "candle_exhaustion_sell": "candle_exhaustion_agent",
    "price_action_exit": "price_action",
    "volume_profile_harvest": "volume_profile_agent",
}


def _canonical_agent_name(agent: Any) -> str:
    text = str(agent or "").strip()
    if not text:
        return ""
    return AGENT_CANONICAL_ALIASES.get(text, text)


def _canonicalize_score_columns(mapping: Dict[str, str], frame: pd.DataFrame) -> Dict[str, str]:
    """Remove duplicate aliases so the same score column cannot become two analysts."""
    out: Dict[str, str] = {}
    used_columns = set()
    for agent, col in mapping.items():
        if col not in frame.columns:
            continue
        canonical = _canonical_agent_name(agent)
        if not canonical or col in used_columns or canonical in out:
            continue
        out[canonical] = col
        used_columns.add(col)
    return out


def _utc_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def _utc_dt(ts: Optional[float] = None) -> str:
    value = float(ts if ts is not None else _utc_ts())
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "win", "won"}


def _read_csv(path: str) -> pd.DataFrame:
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
            return pd.DataFrame()
        return pd.read_csv(path, on_bad_lines="skip", engine="python")
    except Exception:
        return pd.DataFrame()


def _write_rows(path: str, columns: List[str], rows: List[List[Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(tmp, path)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].map(_safe_bool)


def _mode_text(frame: pd.DataFrame, column: str, default: str = "") -> str:
    try:
        if column not in frame.columns or frame.empty:
            return default
        values = frame[column].astype(str).replace({"nan": "", "None": ""})
        values = values[values.str.len() > 0]
        if values.empty:
            return default
        return str(values.mode().iloc[0])
    except Exception:
        return default


def _candidate_rows(base_dir: str, *, min_product_rows: int) -> Tuple[List[List[Any]], Dict[str, Dict[str, Any]]]:
    """
    Build product-level live-buy recommendations from candidate_replay.csv.

    This version scores candidates by profit usefulness instead of only movement:
    - net peak after cost,
    - win rate after cost,
    - adverse movement,
    - time to profit,
    - post-profit continuation,
    - opportunity frequency,
    - sample confidence.

    The output is intentionally conservative for weak products and only gives
    small relief to strong products.
    """
    frame = _read_csv(os.path.join(base_dir, "candidate_replay.csv"))
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    rows: List[List[Any]] = []
    recs: Dict[str, Dict[str, Any]] = {}

    if frame.empty or "product_id" not in frame.columns:
        return rows, recs

    # Support both older and newer column names.
    if "probability" not in frame.columns and "estimated_prob_up" in frame.columns:
        frame["probability"] = frame["estimated_prob_up"]

    required_numeric = [
        "score", "probability", "expected_net_edge_bps", "cost_bps",
        "max_favorable_bps", "max_adverse_bps", "adverse_before_profit_bps",
        "time_to_min_profit_minutes", "forward_window_minutes", "post_profit_extra_gain_bps",
    ]

    for column in required_numeric:
        frame[column] = _numeric(frame, column, 0.0)

    frame["reached_min_profit_bool"] = _bool_series(frame, "reached_min_profit")
    frame["survived_to_profit_bool"] = _bool_series(frame, "survived_to_profit")

    frame["net_peak_bps"] = frame["max_favorable_bps"] - frame["cost_bps"]
    frame["net_success"] = (
        frame["survived_to_profit_bool"]
        | frame["reached_min_profit_bool"]
        | (frame["net_peak_bps"] >= 45.0)
    )

    for product_id, group in frame.groupby(frame["product_id"].astype(str)):
        group = group.copy()
        sample_count = int(len(group))

        if sample_count < int(min_product_rows):
            continue

        score_values = group["score"].dropna()
        prob_values = group["probability"].dropna()
        ev_values = group["expected_net_edge_bps"].dropna()

        if score_values.empty or prob_values.empty or ev_values.empty:
            continue

        best: Optional[Dict[str, Any]] = None

        for sq in [0.45, 0.55, 0.65, 0.75, 0.85, 0.90]:
            score_cut = float(score_values.quantile(sq))

            for pq in [0.40, 0.50, 0.60, 0.70, 0.80, 0.88]:
                prob_cut = float(prob_values.quantile(pq))

                for eq in [0.35, 0.50, 0.65, 0.80, 0.90]:
                    ev_cut = float(ev_values.quantile(eq))

                    accepted = group[
                        (group["score"] >= score_cut)
                        & (group["probability"] >= prob_cut)
                        & (group["expected_net_edge_bps"] >= ev_cut)
                    ].copy()

                    accepted_count = int(len(accepted))
                    if accepted_count < max(8, int(min_product_rows * 0.04)):
                        continue

                    opportunity_rate = accepted_count / max(1, sample_count)
                    win_rate = float(accepted["net_success"].mean())
                    avg_net = float(accepted["net_peak_bps"].mean())
                    median_net = float(accepted["net_peak_bps"].median())
                    avg_adverse = float(accepted["adverse_before_profit_bps"].abs().mean())
                    avg_post_extra = float(accepted["post_profit_extra_gain_bps"].mean())

                    clean_time = accepted["time_to_min_profit_minutes"].replace([np.inf, -np.inf], np.nan).dropna()
                    median_time = float(clean_time.median()) if not clean_time.empty else 240.0

                    sample_confidence = min(
                        1.0,
                        math.sqrt(accepted_count / 60.0) * math.sqrt(sample_count / 300.0),
                    )

                    expected_trade_value_bps = (
                        win_rate * max(0.0, avg_net)
                        - (1.0 - win_rate) * avg_adverse * 0.55
                    )

                    monthly_compound_proxy = (
                        expected_trade_value_bps
                        * min(1.0, opportunity_rate * 12.0)
                        * sample_confidence
                    )

                    objective = (
                        win_rate * 115.0
                        + avg_net * 0.45
                        + median_net * 0.22
                        + avg_post_extra * 0.16
                        + monthly_compound_proxy * 0.35
                        - avg_adverse * 0.30
                        - max(0.0, median_time - 180.0) * 0.040
                        + sample_confidence * 18.0
                    )

                    candidate = {
                        "score_cut": score_cut, "prob_cut": prob_cut, "ev_cut": ev_cut,
                        "accepted_count": accepted_count, "opportunity_rate": opportunity_rate,
                        "win_rate": win_rate, "avg_net": avg_net, "median_net": median_net,
                        "avg_adverse": avg_adverse, "avg_post_extra": avg_post_extra,
                        "median_time": median_time, "sample_confidence": sample_confidence,
                        "expected_trade_value_bps": expected_trade_value_bps,
                        "monthly_compound_proxy": monthly_compound_proxy, "objective": objective,
                    }

                    if best is None or objective > float(best["objective"]):
                        best = candidate

        if best is None:
            continue

        sample_confidence = float(best["sample_confidence"])
        win_rate = float(best["win_rate"])
        avg_net = float(best["avg_net"])
        avg_adverse = float(best["avg_adverse"])
        objective = float(best["objective"])

        replay_quality_score = max(
            0.0,
            min(
                1.0,
                0.50
                + (win_rate - 0.50) * 1.20
                + max(-150.0, min(250.0, avg_net)) / 600.0
                - max(0.0, avg_adverse - 120.0) / 600.0
                + (sample_confidence - 0.50) * 0.35,
            ),
        )

        if win_rate < 0.48 or avg_net < 0.0 or replay_quality_score < 0.45:
            live_gate_bias_bps = min(
                80.0,
                25.0 + max(0.0, 0.48 - win_rate) * 120.0
                + max(0.0, -avg_net) * 0.10
                + max(0.0, 0.45 - replay_quality_score) * 80.0,
            )
        elif win_rate >= 0.58 and avg_net >= 70.0 and replay_quality_score >= 0.58 and sample_confidence >= 0.60:
            live_gate_bias_bps = -min(
                25.0,
                6.0 + max(0.0, win_rate - 0.58) * 80.0 + max(0.0, avg_net - 70.0) * 0.04,
            )
        else:
            live_gate_bias_bps = min(
                35.0,
                max(0.0, 0.52 - win_rate) * 80.0 + max(0.0, 35.0 - avg_net) * 0.15,
            )

        base_projected = 45.0 + max(0.0, avg_net) * 0.35 + max(0.0, avg_adverse) * 0.08
        recommended_min_projected_net_bps = max(
            45.0,
            min(220.0, base_projected + max(0.0, live_gate_bias_bps)),
        )

        if live_gate_bias_bps < 0:
            recommended_min_projected_net_bps = max(
                55.0,
                recommended_min_projected_net_bps + live_gate_bias_bps,
            )

        recommended_forward_window_minutes = max(
            15.0,
            min(360.0, float(best["median_time"]) if math.isfinite(float(best["median_time"])) else 240.0),
        )

        reason = (
            f"candidate_replay_profit_grid;sample_count={sample_count};"
            f"accepted_count={int(best['accepted_count'])};sample_confidence={sample_confidence:.4f};"
            f"win_rate={win_rate:.4f};avg_net={avg_net:.2f};avg_adverse={avg_adverse:.2f};"
            f"replay_quality={replay_quality_score:.4f};live_gate_bias_bps={live_gate_bias_bps:.2f};"
            f"objective={objective:.4f}"
        )

        accepted_for_mode = group[
            (group["score"] >= float(best["score_cut"]))
            & (group["probability"] >= float(best["prob_cut"]))
            & (group["expected_net_edge_bps"] >= float(best["ev_cut"]))
        ].copy()

        dominant_session_agent = _mode_text(accepted_for_mode, "session_liquidity_agent")
        dominant_session_setup = _mode_text(accepted_for_mode, "session_liquidity_setup")
        dominant_structure_state = _mode_text(accepted_for_mode, "structure_state")
        dominant_value_area_state = _mode_text(accepted_for_mode, "value_area_state")
        dominant_fvg_state = _mode_text(accepted_for_mode, "fvg_state")
        dominant_smt_state = _mode_text(accepted_for_mode, "smt_state")

        rows.append([
            f"{ts_value:.6f}", dt_value, product_id, sample_count, int(best["accepted_count"]),
            f"{sample_confidence:.6f}", f"{replay_quality_score:.6f}", f"{live_gate_bias_bps:.6f}",
            f"{float(best['score_cut']):.6f}", f"{float(best['prob_cut']):.6f}", f"{float(best['ev_cut']):.6f}",
            f"{float(recommended_min_projected_net_bps):.6f}", f"{float(recommended_forward_window_minutes):.6f}",
            f"{win_rate:.6f}", f"{avg_net:.6f}", f"{avg_adverse:.6f}", f"{objective:.6f}",
            dominant_session_agent, dominant_session_setup, dominant_structure_state,
            dominant_value_area_state, dominant_fvg_state, dominant_smt_state,
            "candidate_replay.csv", reason,
        ])

        recs[product_id] = {
            "product_id": product_id, "sample_count": sample_count, "accepted_count": int(best["accepted_count"]),
            "sample_confidence": sample_confidence, "replay_quality_score": replay_quality_score,
            "live_gate_bias_bps": live_gate_bias_bps, "recommended_min_score": float(best["score_cut"]),
            "recommended_min_probability": float(best["prob_cut"]),
            "recommended_min_expected_value_bps": float(best["ev_cut"]),
            "recommended_min_projected_net_bps": float(recommended_min_projected_net_bps),
            "recommended_forward_window_minutes": float(recommended_forward_window_minutes),
            "expected_win_rate": win_rate, "expected_net_bps": avg_net,
            "expected_adverse_bps": avg_adverse, "objective_score": objective,
            "dominant_session_agent": dominant_session_agent,
            "dominant_session_setup": dominant_session_setup,
            "dominant_structure_state": dominant_structure_state,
            "dominant_value_area_state": dominant_value_area_state,
            "dominant_fvg_state": dominant_fvg_state,
            "dominant_smt_state": dominant_smt_state,
        }

    return rows, recs

def _sell_recommendation_rows(base_dir: str) -> List[List[Any]]:
    frame = _read_csv(os.path.join(base_dir, "sell_outcomes.csv")); ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); rows: List[List[Any]] = []
    if frame.empty or "product_id" not in frame.columns:
        return rows
    frame["move_after_sell_bps"] = _numeric(frame, "move_after_sell_bps", 0.0); frame["realized_net_pnl_bps"] = _numeric(frame, "realized_net_pnl_bps", 0.0); frame["earnings_quality_score"] = _numeric(frame, "earnings_quality_score", 0.5)
    for product_id, group in frame.groupby(frame["product_id"].astype(str)):
        sell_samples = int(len(group))
        if sell_samples < 5:
            continue
        avg_move_after = float(group["move_after_sell_bps"].mean()); too_early_rate = float((group["move_after_sell_bps"] >= 80.0).mean()); good_exit_rate = float((group["move_after_sell_bps"] <= 30.0).mean()); avg_earnings_quality = float(group["earnings_quality_score"].mean())
        trigger = 45.0; strong_pullback = 120.0; full_pullback = 240.0
        if too_early_rate >= 0.45:
            trigger += 20.0; strong_pullback += 35.0; full_pullback += 50.0
        if too_early_rate <= 0.20 and good_exit_rate >= 0.60:
            trigger = max(30.0, trigger - 10.0); strong_pullback = max(90.0, strong_pullback - 15.0)
        if avg_earnings_quality >= 0.62:
            trigger = max(30.0, trigger - 5.0)
        if avg_earnings_quality <= 0.45:
            trigger += 10.0; strong_pullback += 20.0
        sell_objective = good_exit_rate * 100.0 + avg_earnings_quality * 40.0 - too_early_rate * 55.0 - max(0.0, avg_move_after) * 0.08
        reason = f"sell_outcomes_replay;samples={sell_samples};too_early_rate={too_early_rate:.4f};good_exit_rate={good_exit_rate:.4f};avg_move_after_sell_bps={avg_move_after:.2f};avg_earnings_quality={avg_earnings_quality:.4f}"
        rows.append([f"{ts_value:.6f}", dt_value, product_id, sell_samples, f"{too_early_rate:.6f}", f"{good_exit_rate:.6f}", f"{avg_move_after:.6f}", f"{trigger:.6f}", f"{strong_pullback:.6f}", f"{full_pullback:.6f}", f"{sell_objective:.6f}", "sell_outcomes.csv", reason])
    return rows


def _agent_prior_rows(base_dir: str) -> List[List[Any]]:
    """
    Convert agent_performance.csv into profit-weighted priors.

    This creates useful Level 8 memory without allowing old replay rows to overpower
    new live trade evidence.
    """
    perf = _read_csv(os.path.join(base_dir, "agent_performance.csv"))
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    rows: List[List[Any]] = []

    if perf.empty or not {"product_id", "strategy", "agent"}.issubset(perf.columns):
        return rows

    perf["weighted_agent_credit_score"] = _numeric(perf, "weighted_agent_credit_score", 0.5)
    perf["agent_credit_score"] = _numeric(perf, "agent_credit_score", 0.5)
    perf["outcome_move_bps"] = _numeric(perf, "outcome_move_bps", 0.0)
    perf["confidence"] = _numeric(perf, "confidence", 0.6)
    perf["reliability"] = _numeric(perf, "reliability", 0.5)

    if "realized_net_pnl_usd" in perf.columns:
        perf["realized_net_pnl_usd"] = _numeric(perf, "realized_net_pnl_usd", 0.0)
    else:
        perf["realized_net_pnl_usd"] = 0.0

    if "realized_net_pnl_bps" in perf.columns:
        perf["realized_net_pnl_bps"] = _numeric(perf, "realized_net_pnl_bps", 0.0)
    else:
        perf["realized_net_pnl_bps"] = 0.0

    if "earnings_quality_score" in perf.columns:
        perf["earnings_quality_score"] = _numeric(perf, "earnings_quality_score", 0.5)
    else:
        perf["earnings_quality_score"] = 0.5

    if "outcome_source" not in perf.columns:
        perf["outcome_source"] = "agent_performance"

    source_weight = perf["outcome_source"].astype(str).map({
        "real_trade": 1.35, "trade_outcome": 1.35, "sell_outcome": 1.30,
        "agent_performance": 0.75, "level8_observation": 0.40,
        "observation_outcome": 0.30,
    }).fillna(0.45).astype(float)

    pnl_credit = (
        0.50
        + perf["realized_net_pnl_usd"].clip(-1.00, 1.00) * 0.16
        + perf["realized_net_pnl_bps"].clip(-250.0, 350.0) / 2500.0
        + (perf["earnings_quality_score"] - 0.50) * 0.25
    ).clip(0.0, 1.0)

    move_credit = ((perf["outcome_move_bps"].clip(-250.0, 350.0) + 250.0) / 600.0).clip(0.0, 1.0)

    perf["profit_replay_credit"] = (
        perf["weighted_agent_credit_score"] * source_weight
        + perf["agent_credit_score"] * 0.20
        + pnl_credit * 0.25
        + move_credit * 0.10
    ) / (source_weight + 0.55)

    perf["profit_replay_credit"] = perf["profit_replay_credit"].clip(0.0, 1.0)

    for (product_id, strategy, agent), group in perf.groupby(["product_id", "strategy", "agent"]):
        sample_count = int(len(group))
        if sample_count < 5:
            continue

        recent = group.tail(250).copy()
        recent_count = int(len(recent))
        sample_confidence = min(1.0, math.sqrt(recent_count / 80.0))

        credit = float(recent["profit_replay_credit"].mean())
        move = float(recent["outcome_move_bps"].mean())
        confidence = float(recent["confidence"].mean())
        reliability = float(recent["reliability"].mean())

        prior_credit = max(0.35, min(0.68, 0.50 + (credit - 0.50) * 0.75 * sample_confidence))
        prior_weight = max(0.45, min(1.05, 0.50 + sample_confidence * 0.45))

        success = 1 if prior_credit >= 0.50 else 0

        if str(strategy).upper() == "EXIT_REVIEW":
            buy_score = 0.0
            sell_score = prior_credit
            hold_score = 1.0 - max(0.0, prior_credit - 0.50) * 0.70
            wait_score = 0.25
        else:
            buy_score = prior_credit
            sell_score = 1.0 - prior_credit
            hold_score = 0.50
            wait_score = 0.35

        rows.append([
            f"{ts_value:.6f}", dt_value,
            f"backtest-prior-{product_id}-{strategy}-{agent}-{int(ts_value)}",
            str(product_id), str(strategy), str(agent), f"{buy_score:.6f}",
            f"{sell_score:.6f}", f"{hold_score:.6f}", f"{wait_score:.6f}",
            f"{max(0.50, min(0.85, confidence)):.6f}",
            f"{max(0.35, min(0.85, reliability)):.6f}", "1.000000", "", "",
            "0.000000", "0.000000", "backtest_profit_replay", f"{prior_weight:.6f}",
            "0", f"{move:.6f}", "profit_replay_prior", f"{prior_credit:.6f}",
            f"{prior_credit:.6f}", success,
            (
                f"backtest_profit_prior_capped;samples={sample_count};recent_samples={recent_count};"
                f"sample_confidence={sample_confidence:.4f};raw_credit={credit:.6f};"
                f"prior_credit={prior_credit:.6f};prior_weight={prior_weight:.6f};"
                f"avg_move_bps={move:.2f};source=agent_performance.csv"
            ),
        ])

    return rows


def _setup_performance_rows(base_dir: str) -> List[List[Any]]:
    frame = _read_csv(os.path.join(base_dir, "candidate_replay.csv"))
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    rows: List[List[Any]] = []

    if frame.empty or "product_id" not in frame.columns:
        return rows

    if "probability" not in frame.columns and "estimated_prob_up" in frame.columns:
        frame["probability"] = frame["estimated_prob_up"]

    frame["cost_bps"] = _numeric(frame, "cost_bps", 0.0)
    frame["max_favorable_bps"] = _numeric(frame, "max_favorable_bps", 0.0)
    frame["max_adverse_bps"] = _numeric(frame, "max_adverse_bps", 0.0)
    frame["net_peak_bps"] = frame["max_favorable_bps"] - frame["cost_bps"]
    frame["net_success"] = frame["net_peak_bps"] >= 45.0

    setup_columns = [
        "session_liquidity_agent",
        "session_liquidity_setup",
        "structure_state",
        "value_area_state",
        "value_acceptance_state",
        "volume_node_state",
        "fvg_state",
        "smt_state",
        "previous_session_profile_reaction_state",
        "previous_session_profile_bias",
        "quant_boundary_state",
        "quant_volatility_cluster_state",
        "quant_peer_state",
        "price_action_buy_score",
        "candle_exhaustion_score",
        "volume_profile_buy_score",
        "volume_profile_leader_buy_score",
        "unfair_trade_score",
        "expected_utility_bps",
        "buy_vs_wait_edge_bps",
        "validated_liquidity_buy_score",
        "fresh_zone_buy_score",
        "fvg_buy_score",
        "previous_session_profile_buy_score",
        "previous_session_profile_sell_score",
        "previous_session_profile_wait_score",
        "quant_buy_score",
        "quant_sell_score",
        "quant_wait_score",
        "quant_stationarity_score",
        "quant_forecast_return_bps",
    ]

    for col in setup_columns:
        if col not in frame.columns:
            frame[col] = ""

    for numeric_col in [
        "price_action_buy_score",
        "candle_exhaustion_score",
        "volume_profile_buy_score",
        "volume_profile_leader_buy_score",
        "unfair_trade_score",
        "expected_utility_bps",
        "buy_vs_wait_edge_bps",
        "validated_liquidity_buy_score",
        "fresh_zone_buy_score",
        "fvg_buy_score",
        "previous_session_profile_buy_score",
        "previous_session_profile_sell_score",
        "previous_session_profile_wait_score",
        "quant_buy_score",
        "quant_sell_score",
        "quant_wait_score",
        "quant_stationarity_score",
        "quant_forecast_return_bps",
    ]:
        frame[numeric_col] = _numeric(frame, numeric_col, 0.0)

    def _bucket(value: float) -> str:
        try:
            v = float(value)
        except Exception:
            v = 0.0
        if v >= 0.70:
            return "high"
        if v >= 0.45:
            return "medium"
        if v > 0.0:
            return "low"
        return "none"

    frame["setup_key"] = (
        frame["session_liquidity_agent"].astype(str)
        + "|session_setup=" + frame["session_liquidity_setup"].astype(str)
        + "|structure=" + frame["structure_state"].astype(str)
        + "|value=" + frame["value_area_state"].astype(str)
        + "|acceptance=" + frame["value_acceptance_state"].astype(str)
        + "|volume_node=" + frame["volume_node_state"].astype(str)
        + "|fvg=" + frame["fvg_state"].astype(str)
        + "|smt=" + frame["smt_state"].astype(str)
        + "|prev_session_reaction=" + frame["previous_session_profile_reaction_state"].astype(str)
        + "|prev_bias=" + frame["previous_session_profile_bias"].astype(str)
        + "|quant_boundary=" + frame["quant_boundary_state"].astype(str)
        + "|quant_vol=" + frame["quant_volatility_cluster_state"].astype(str)
        + "|quant_peer=" + frame["quant_peer_state"].astype(str)
        + "|pa=" + frame["price_action_buy_score"].map(_bucket).astype(str)
        + "|exhaust=" + frame["candle_exhaustion_score"].map(_bucket).astype(str)
        + "|volume=" + frame["volume_profile_buy_score"].map(_bucket).astype(str)
        + "|vp_leader=" + frame["volume_profile_leader_buy_score"].map(_bucket).astype(str)
        + "|unfair=" + frame["unfair_trade_score"].map(_bucket).astype(str)
        + "|utility=" + (frame["expected_utility_bps"] / 200.0).map(_bucket).astype(str)
        + "|buy_vs_wait=" + (frame["buy_vs_wait_edge_bps"] / 200.0).map(_bucket).astype(str)
        + "|validated_liq=" + frame["validated_liquidity_buy_score"].map(_bucket).astype(str)
        + "|fresh_zone=" + frame["fresh_zone_buy_score"].map(_bucket).astype(str)
        + "|fvg_score=" + frame["fvg_buy_score"].map(_bucket).astype(str)
        + "|prev_profile=" + frame["previous_session_profile_buy_score"].map(_bucket).astype(str)
        + "|prev_wait=" + frame["previous_session_profile_wait_score"].map(_bucket).astype(str)
        + "|quant=" + frame["quant_buy_score"].map(_bucket).astype(str)
        + "|quant_wait=" + frame["quant_wait_score"].map(_bucket).astype(str)
        + "|stationarity=" + frame["quant_stationarity_score"].map(_bucket).astype(str)
    )

    for (product_id, setup_key), group in frame.groupby(["product_id", "setup_key"]):
        group = group.copy()
        sample_count = int(len(group))
        # Full setup keys are intentionally detailed. Require stronger sample count
        # so thin combinations do not overfit.
        if sample_count < 20:
            continue

        win_rate = float(group["net_success"].mean())
        avg_net = float(group["net_peak_bps"].mean())
        avg_adverse = float(group["max_adverse_bps"].abs().mean())
        sample_confidence = min(1.0, sample_count / 80.0)
        raw_objective = win_rate * 100.0 + avg_net * 0.35 - avg_adverse * 0.20
        # Shrink toward neutral until enough rows exist.
        objective = raw_objective * sample_confidence + 50.0 * (1.0 - sample_confidence)

        rows.append([
            f"{ts_value:.6f}",
            dt_value,
            str(product_id),
            str(setup_key),
            sample_count,
            f"{win_rate:.6f}",
            f"{avg_net:.6f}",
            f"{avg_adverse:.6f}",
            f"{objective:.6f}",
            (
                f"product_session_setup_profit_replay;"
                f"sample_confidence={sample_confidence:.3f};"
                f"raw_objective={raw_objective:.3f};"
                f"shrunk_objective={objective:.3f}"
            ),
        ])

    return rows

def _compact_setup_key(frame: pd.DataFrame) -> pd.Series:
    """
    Compact setup key for validation.

    This intentionally uses fewer fields than the full setup key to reduce
    sparse-data overfitting during walk-forward validation.
    """
    for col in [
        "session_liquidity_setup",
        "value_acceptance_state",
        "volume_node_state",
        "previous_session_profile_reaction_state",
        "quant_boundary_state",
        "market_regime",
    ]:
        if col not in frame.columns:
            frame[col] = ""
    return (
        "session=" + frame["session_liquidity_setup"].astype(str)
        + "|value=" + frame["value_acceptance_state"].astype(str)
        + "|node=" + frame["volume_node_state"].astype(str)
        + "|prior=" + frame["previous_session_profile_reaction_state"].astype(str)
        + "|quant=" + frame["quant_boundary_state"].astype(str)
        + "|regime=" + frame["market_regime"].astype(str)
    )


def _walk_forward_validation_rows(base_dir: str) -> List[List[Any]]:
    frame = _read_csv(os.path.join(base_dir, "candidate_replay.csv"))
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    rows: List[List[Any]] = []
    if frame.empty or "product_id" not in frame.columns:
        module_debug(MODULE_NAME, "backtest_walk_forward_waiting_for_rows", data={"candidate_rows": int(len(frame)) if hasattr(frame, "__len__") else 0, "min_product_rows": 80, "reason": "not_enough_reviewed_outcomes_yet"}, level="INFO", also_overall=False)
        return rows
    if "ts" not in frame.columns:
        module_debug(MODULE_NAME, "backtest_walk_forward_waiting_for_rows", data={"candidate_rows": int(len(frame)), "min_product_rows": 80, "reason": "not_enough_reviewed_outcomes_yet"}, level="INFO", also_overall=False)
        return rows
    frame = frame.copy()
    frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
    frame = frame.dropna(subset=["ts"]).sort_values("ts")
    frame["cost_bps"] = _numeric(frame, "cost_bps", 0.0)
    frame["max_favorable_bps"] = _numeric(frame, "max_favorable_bps", 0.0)
    frame["max_adverse_bps"] = _numeric(frame, "max_adverse_bps", 0.0)
    frame["net_peak_bps"] = frame["max_favorable_bps"] - frame["cost_bps"]
    frame["net_success"] = frame["net_peak_bps"] >= 45.0
    frame["compact_setup_key"] = _compact_setup_key(frame)
    for product_id, product_frame in frame.groupby("product_id"):
        product_frame = product_frame.sort_values("ts").reset_index(drop=True)
        if len(product_frame) < 80:
            module_debug(MODULE_NAME, "backtest_walk_forward_waiting_for_rows", data={"candidate_rows": int(len(frame)), "product_id": str(product_id), "product_rows": int(len(product_frame)), "min_product_rows": 80, "reason": "not_enough_reviewed_outcomes_yet"}, level="INFO", also_overall=False)
            continue
        fold_count = 4
        fold_size = max(20, len(product_frame) // fold_count)
        for fold in range(1, fold_count):
            split_idx = fold * fold_size
            train = product_frame.iloc[:split_idx].copy()
            test = product_frame.iloc[split_idx: split_idx + fold_size].copy()
            if len(train) < 40 or len(test) < 15:
                continue
            train_wr = float(train["net_success"].mean())
            test_wr = float(test["net_success"].mean())
            train_avg = float(train["net_peak_bps"].mean())
            test_avg = float(test["net_peak_bps"].mean())
            # Generalization gap: large positive gap means train looked better
            # than later test behavior.
            generalization_gap = (train_avg - test_avg) + (train_wr - test_wr) * 100.0
            walk_score = (
                test_wr * 100.0
                + test_avg * 0.35
                - float(test["max_adverse_bps"].abs().mean()) * 0.20
                - max(0.0, generalization_gap) * 0.35
            )
            rows.append([
                f"{ts_value:.6f}", dt_value, str(product_id), int(fold), int(len(train)), int(len(test)),
                f"{train_wr:.6f}", f"{test_wr:.6f}", f"{train_avg:.6f}", f"{test_avg:.6f}",
                f"{generalization_gap:.6f}", f"{walk_score:.6f}",
                (
                    f"walk_forward_validation;"
                    f"train_wr={train_wr:.3f};test_wr={test_wr:.3f};"
                    f"train_avg={train_avg:.2f};test_avg={test_avg:.2f};"
                    f"gap={generalization_gap:.2f};score={walk_score:.2f}"
                ),
            ])
    return rows



BAYES_PRIOR_WINS = 25.0
BAYES_PRIOR_TOTAL = 50.0
MIN_CONTEXT_SAMPLES = 20
MIN_GLOBAL_SAMPLES = 20
FOUR_PASS_AGENT_SCAN_WORKERS = int(os.getenv("FOUR_PASS_AGENT_SCAN_WORKERS", "8"))
FOUR_PASS_FEATURE_CACHE_VERSION = "v2_bayesian_ev_context"


def _bayesian_win_rate(wins: float, total: float, *, prior_wins: float = BAYES_PRIOR_WINS, prior_total: float = BAYES_PRIOR_TOTAL) -> float:
    total = float(total or 0.0)
    wins = float(wins or 0.0)
    return float((wins + prior_wins) / max(1.0, total + prior_total))


def _ev_stats(values: pd.Series) -> Dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()

    if vals.empty:
        return {
            "raw_win_rate": 0.0,
            "smoothed_win_rate": _bayesian_win_rate(0.0, 0.0),
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "ev_bps": 0.0,
            "avg_net_bps": 0.0,
            "median_net_bps": 0.0,
        }

    wins = vals[vals > 0.0]
    losses = vals[vals <= 0.0]

    raw_win_rate = float(len(wins) / max(1, len(vals)))
    smoothed = _bayesian_win_rate(float(len(wins)), float(len(vals)))
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0
    ev = float((smoothed * avg_win) - ((1.0 - smoothed) * avg_loss))

    return {
        "raw_win_rate": raw_win_rate,
        "smoothed_win_rate": smoothed,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "ev_bps": ev,
        "avg_net_bps": float(vals.mean()),
        "median_net_bps": float(vals.median()),
    }




def _training_days_from_frame(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty:
        return 1.0
    time_col = None
    for col in ["replay_ts", "entry_ts", "ts"]:
        if col in frame.columns:
            time_col = col
            break
    if not time_col:
        return 1.0
    times = pd.to_numeric(frame[time_col], errors="coerce").dropna()
    if times.empty:
        return 1.0
    days = float((times.max() - times.min()) / 86400.0)
    return max(1.0, days)


def _frequency_score(frequency_per_day: float, *, target_per_day: float = 10.0) -> float:
    """
    Frequency matters because an agent with many correct decisions is more useful
    than an agent with rare correct decisions.

    This returns a wider 0.05..2.25 range so high-frequency positive-EV agents
    can separate from rare agents.
    """
    try:
        freq = max(0.0, float(frequency_per_day))
        target = max(1.0, float(target_per_day))
        score = math.sqrt(freq / target) if freq > 0 else 0.05
        return max(0.05, min(2.25, score))
    except Exception:
        return 0.05


def _sample_reliability(selected_count: int, *, half_confidence_count: int = 160) -> float:
    """
    Sample reliability should increase with more samples, but not max out too early.
    The old formula saturated too fast, flattening influence weights.
    """
    try:
        n = max(0.0, float(selected_count))
        h = max(1.0, float(half_confidence_count))
        return max(0.05, min(1.50, 1.50 * (1.0 - math.exp(-n / h))))
    except Exception:
        return 0.05


def _edge_score_from_stats(
    *,
    smoothed_win_rate: float,
    ev_bps: float,
    avg_net_bps: float,
    median_net_bps: float,
    avg_loss_bps: float,
) -> float:
    """
    Edge quality should dominate influence.

    High-frequency bad agents should not dominate.
    High-win-rate agents with tiny profits should not dominate.
    Agents with positive EV, positive median, and controlled losses should rise.
    """
    try:
        win = float(smoothed_win_rate)
        ev = float(ev_bps)
        avg_net = float(avg_net_bps)
        med = float(median_net_bps)
        avg_loss = float(avg_loss_bps)

        score = (
            0.45
            + (win - 0.50) * 1.35
            + max(-250.0, min(350.0, ev)) / 450.0
            + max(-200.0, min(300.0, avg_net)) / 700.0
            + max(-200.0, min(300.0, med)) / 900.0
            - max(0.0, avg_loss - 45.0) / 350.0
        )

        return max(0.02, min(2.50, float(score)))
    except Exception:
        return 0.02


def _decision_influence_score(
    *,
    selected_count: int,
    frequency_per_day: float,
    smoothed_win_rate: float,
    ev_bps: float,
    avg_net_bps: float,
    median_net_bps: float,
    avg_loss_bps: float,
) -> Dict[str, float]:
    reliability = _sample_reliability(selected_count)
    frequency = _frequency_score(frequency_per_day)
    edge = _edge_score_from_stats(
        smoothed_win_rate=smoothed_win_rate,
        ev_bps=ev_bps,
        avg_net_bps=avg_net_bps,
        median_net_bps=median_net_bps,
        avg_loss_bps=avg_loss_bps,
    )
    influence = max(0.0, edge) * max(0.05, reliability) * max(0.05, frequency)
    return {
        "reliability_score": float(reliability),
        "frequency_score": float(frequency),
        "edge_score": float(edge),
        "decision_influence_score": float(influence),
    }

def _normalize_influence_rows(rows: List[Dict[str, Any]], *, group_keys: List[str], score_key: str = "decision_influence_score", weight_key: str = "decision_weight_pct") -> List[Dict[str, Any]]:
    if not rows:
        return []
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for group_rows in grouped.values():
        total = sum(
            max(0.0, float(r.get(score_key, 0.0) or 0.0))
            for r in group_rows
            if float(r.get("ev_bps", 0.0) or 0.0) > 0.0
        )
        total = max(total, 1e-9)
        for row in group_rows:
            if float(row.get("ev_bps", 0.0) or 0.0) <= 0.0:
                row[weight_key] = 0.0
            else:
                row[weight_key] = max(0.0, float(row.get(score_key, 0.0) or 0.0)) / total * 100.0
            out.append(row)
    return out

def _infer_market_regime(frame: pd.DataFrame) -> pd.Series:
    """Creates a stable regime label using whatever columns are available."""
    if frame is None or frame.empty:
        return pd.Series([], dtype=str)

    out = pd.Series("unknown", index=frame.index, dtype="object")
    momentum = None
    volatility = None

    for col in ["momentum_15_bps", "momentum_30_bps", "trend_bps", "macro_momentum_bps"]:
        if col in frame.columns:
            momentum = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
            break

    for col in ["volatility_bps", "atr_bps", "range_bps", "macro_volatility_bps"]:
        if col in frame.columns:
            volatility = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
            break

    if momentum is None:
        momentum = pd.Series(0.0, index=frame.index)
    if volatility is None:
        volatility = pd.Series(0.0, index=frame.index)

    out[(momentum >= 20.0) & (volatility >= 80.0)] = "trend_high_vol"
    out[(momentum >= 20.0) & (volatility < 80.0)] = "trend_low_vol"
    out[(momentum <= -20.0) & (volatility >= 80.0)] = "downtrend_high_vol"
    out[(momentum <= -20.0) & (volatility < 80.0)] = "downtrend_low_vol"
    out[(momentum.abs() < 20.0) & (volatility >= 80.0)] = "range_high_vol"
    out[(momentum.abs() < 20.0) & (volatility < 80.0)] = "range_low_vol"
    return out.astype(str)


def _four_pass_cache_key(base_dir: str, filenames: List[str]) -> str:
    parts = [FOUR_PASS_FEATURE_CACHE_VERSION]
    for filename in filenames:
        path = os.path.join(base_dir, filename)
        try:
            stat = os.stat(path)
            parts.append(f"{filename}:{int(stat.st_mtime)}:{int(stat.st_size)}")
        except Exception:
            parts.append(f"{filename}:missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _four_pass_cache_path(base_dir: str, name: str) -> str:
    cache_dir = os.path.join(base_dir, "_four_pass_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}.pkl")


def _load_cached_four_pass_frame(base_dir: str, name: str, key: str) -> pd.DataFrame:
    path = _four_pass_cache_path(base_dir, name)
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return pd.DataFrame()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("key") != key:
            return pd.DataFrame()
        frame = payload.get("frame")
        if isinstance(frame, pd.DataFrame):
            return frame.copy()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _save_cached_four_pass_frame(base_dir: str, name: str, key: str, frame: pd.DataFrame) -> None:
    path = _four_pass_cache_path(base_dir, name)
    try:
        payload = {"key": key, "created_ts": time.time(), "frame": frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()}
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass





def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def _four_pass_parquet_path(base_dir: str, name: str) -> str:
    cache_dir = os.path.join(base_dir, "_four_pass_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}.parquet")


def _four_pass_key_path(base_dir: str, name: str) -> str:
    cache_dir = os.path.join(base_dir, "_four_pass_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}.key")


def _load_feature_frame_cached(base_dir: str, name: str, key: str) -> pd.DataFrame:
    """Load feature frames from Parquet when possible, with pickle fallback."""
    parquet_path = _four_pass_parquet_path(base_dir, name)
    key_path = _four_pass_key_path(base_dir, name)
    try:
        if _parquet_available() and os.path.exists(parquet_path) and os.path.exists(key_path):
            with open(key_path, "r", encoding="utf-8") as f:
                cached_key = f.read().strip()
            if cached_key == key:
                return pd.read_parquet(parquet_path)
    except Exception:
        pass
    return _load_cached_four_pass_frame(base_dir, name, key)


def _save_feature_frame_cached(base_dir: str, name: str, key: str, frame: pd.DataFrame) -> None:
    """Write Parquet when available and always keep the pickle fallback current."""
    if frame is None:
        frame = pd.DataFrame()
    parquet_path = _four_pass_parquet_path(base_dir, name)
    key_path = _four_pass_key_path(base_dir, name)
    try:
        if _parquet_available():
            frame.to_parquet(parquet_path, index=False)
            with open(key_path, "w", encoding="utf-8") as f:
                f.write(str(key))
    except Exception:
        pass
    _save_cached_four_pass_frame(base_dir, name, key, frame)


def _feature_store_summary_rows(base_dir: str) -> List[List[Any]]:
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    cache_dir = os.path.join(base_dir, "_four_pass_cache")
    rows: List[List[Any]] = []
    try:
        if not os.path.isdir(cache_dir):
            return rows
        total_rows = 0
        total_cols = 0
        files = []
        for name in os.listdir(cache_dir):
            if not (name.endswith(".pkl") or name.endswith(".parquet")):
                continue
            path = os.path.join(cache_dir, name)
            files.append(name)
            try:
                if name.endswith(".parquet"):
                    frame = pd.read_parquet(path)
                else:
                    with open(path, "rb") as f:
                        payload = pickle.load(f)
                    frame = payload.get("frame")
                if isinstance(frame, pd.DataFrame):
                    total_rows += int(len(frame))
                    total_cols = max(total_cols, int(len(frame.columns)))
            except Exception:
                pass
        rows.append([
            f"{ts_value:.6f}", dt_value, cache_dir, int(total_rows), int(total_cols),
            ",".join(files), "parquet_feature_frame_cache_with_pickle_fallback",
            "feature_store_summary_for_four_pass_cache",
        ])
    except Exception:
        pass
    return rows


def _four_pass_score_from_ev(*, smoothed_win_rate: float, ev_bps: float, avg_net_bps: float, median_net_bps: float, avg_adverse_bps: float, selected_count: int, sample_floor: int) -> float:
    sample_factor = min(1.0, math.sqrt(float(selected_count) / max(1.0, float(sample_floor))))
    score = (
        0.50
        + (float(smoothed_win_rate) - 0.50) * 0.65 * sample_factor
        + max(-250.0, min(350.0, float(ev_bps))) / 700.0
        + max(-200.0, min(300.0, float(avg_net_bps))) / 900.0
        + max(-200.0, min(300.0, float(median_net_bps))) / 1200.0
        - max(0.0, float(avg_adverse_bps) - 110.0) / 1000.0
    )
    return max(0.03, min(0.97, float(score)))


def _softmax_weights(scored_rows: List[Dict[str, Any]], *, score_key: str, raw_key: str, weight_key: str) -> List[Dict[str, Any]]:
    if not scored_rows:
        return []
    raw_total = 0.0
    for row in scored_rows:
        samples = float(row.get("selected_count", row.get("sample_count", 0)) or 0)
        if samples <= 0 and raw_key in row:
            raw = float(row.get(raw_key, 0.0) or 0.0)
        else:
            score = float(row.get(score_key, 0.50) or 0.50)
            sample_factor = max(0.05, min(1.0, samples / 80.0))
            raw = math.exp((score - 0.50) / 0.070) * sample_factor
        row[raw_key] = raw
        raw_total += raw
    raw_total = max(raw_total, 1e-9)
    for row in scored_rows:
        row[weight_key] = float(row[raw_key]) / raw_total * 100.0
    return scored_rows




def _softmax_weights_by_group(
    rows: List[Dict[str, Any]],
    *,
    group_keys: List[str],
    score_key: str,
    raw_key: str,
    weight_key: str,
) -> List[Dict[str, Any]]:
    """Normalize weights inside each context group."""
    if not rows:
        return []
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for group_rows in grouped.values():
        raw_total = 0.0
        for row in group_rows:
            score = float(row.get(score_key, 0.50) or 0.50)
            samples = float(row.get("selected_count", row.get("sample_count", 0)) or 0)
            sample_factor = max(0.05, min(1.0, samples / 80.0))
            raw = math.exp((score - 0.50) / 0.070) * sample_factor
            row[raw_key] = raw
            raw_total += raw
        raw_total = max(raw_total, 1e-9)
        for row in group_rows:
            row[weight_key] = float(row[raw_key]) / raw_total * 100.0
            out.append(row)
    return out

def _buy_agent_score_columns(frame: pd.DataFrame) -> Dict[str, str]:
    """Return every buy-capable analyst that has a score column in the replay frame."""
    discovered: Dict[str, str] = {}
    for agent, col in BUY_AGENT_SCORE_COLUMNS.items():
        if col in frame.columns:
            discovered[agent] = col
    for col in frame.columns:
        col_text = str(col)
        if col_text.endswith("_buy_score"):
            raw_agent = col_text.replace("_buy_score", "")
            discovered.setdefault(raw_agent, col_text)
    return _canonicalize_score_columns(discovered, frame)


def _sell_agent_score_columns(frame: pd.DataFrame) -> Dict[str, str]:
    """Return every sell-capable analyst that has a score column in the sell frame."""
    discovered: Dict[str, str] = {}
    for agent, col in SELL_AGENT_SCORE_COLUMNS.items():
        if col in frame.columns:
            discovered[agent] = col
    for col in frame.columns:
        col_text = str(col)
        if col_text.endswith("_sell_score"):
            raw_agent = col_text.replace("_sell_score", "")
            discovered.setdefault(raw_agent, col_text)
    return _canonicalize_score_columns(discovered, frame)

def _build_buy_training_frame(base_dir: str) -> pd.DataFrame:
    cache_key = _four_pass_cache_key(base_dir, ["candidate_replay.csv", "historical_shadow_replay.csv"])
    cached = _load_feature_frame_cached(base_dir, "buy_training_frame", cache_key)
    if not cached.empty:
        return cached

    frame = _read_csv(os.path.join(base_dir, "candidate_replay.csv"))
    if frame.empty:
        frame = _read_csv(os.path.join(base_dir, "historical_shadow_replay.csv"))
    if frame.empty or "product_id" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    for col in ["score", "probability", "expected_net_edge_bps", "cost_bps", "max_favorable_bps", "max_adverse_bps", "net_pnl_bps", "binance_taker_taker_net_pnl_bps", "synthetic_notional_usd"]:
        if col in frame.columns:
            frame[col] = _numeric(frame, col, 0.0)
    profitability_mode = "opportunity_proxy"

    if "buy_net_bps" not in frame.columns:
        if "binance_taker_taker_net_pnl_bps" in frame.columns:
            frame["buy_net_bps"] = _numeric(frame, "binance_taker_taker_net_pnl_bps", 0.0)
            profitability_mode = "realized_exit_replay"
        elif "net_pnl_bps" in frame.columns:
            frame["buy_net_bps"] = _numeric(frame, "net_pnl_bps", 0.0)
            profitability_mode = "realized_exit_replay"
        else:
            frame["buy_net_bps"] = _numeric(frame, "max_favorable_bps", 0.0) - _numeric(frame, "cost_bps", 0.0)
            profitability_mode = "opportunity_proxy"
    else:
        profitability_mode = "realized_or_precomputed_buy_net"

    frame["profitability_mode"] = profitability_mode
    # True BUY success must be cost-aware and directional.
    # Do not use generic reached_min_profit/survived_to_profit flags as an OR condition,
    # because those can inflate accuracy when the final net result was not actually profitable.
    frame["buy_success"] = frame["buy_net_bps"] > 0.0
    frame["buy_adverse_bps"] = _numeric(frame, "max_adverse_bps", 0.0).abs()
    if "market_regime" not in frame.columns:
        frame["market_regime"] = _infer_market_regime(frame)
    _save_feature_frame_cached(base_dir, "buy_training_frame", cache_key, frame)
    return frame


def _scan_buy_agent_thresholds(agent: str, col: str, frame: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    local = frame.copy()
    local[col] = _numeric(local, col, 0.0)
    valid = local.dropna(subset=[col]).copy()
    if valid.empty:
        return None, []
    if float(valid[col].max()) > 1.50:
        valid[col] = valid[col] / 100.0
    valid[col] = valid[col].clip(0.0, 1.0)
    best = None
    context_rows_for_weighting: List[Dict[str, Any]] = []
    for q in [0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.97]:
        threshold = float(valid[col].quantile(q))
        selected = valid[valid[col] >= threshold].copy()
        selected_count = int(len(selected))
        if selected_count < MIN_GLOBAL_SAMPLES:
            continue
        stats = _ev_stats(selected["buy_net_bps"])
        avg_adverse = float(pd.to_numeric(selected["buy_adverse_bps"], errors="coerce").fillna(0.0).mean())
        score = _four_pass_score_from_ev(smoothed_win_rate=stats["smoothed_win_rate"], ev_bps=stats["ev_bps"], avg_net_bps=stats["avg_net_bps"], median_net_bps=stats["median_net_bps"], avg_adverse_bps=avg_adverse, selected_count=selected_count, sample_floor=100)
        candidate = {"agent": agent, "source_column": col, "sample_count": int(len(valid)), "selected_count": selected_count, "threshold": threshold, "win_rate": stats["smoothed_win_rate"], "raw_win_rate": stats["raw_win_rate"], "avg_win_bps": stats["avg_win_bps"], "avg_loss_bps": stats["avg_loss_bps"], "ev_bps": stats["ev_bps"], "avg_net_bps": stats["avg_net_bps"], "median_net_bps": stats["median_net_bps"], "avg_adverse_bps": avg_adverse, "score": score, "profitability_mode": str(selected.get("profitability_mode", pd.Series(["unknown"])).iloc[0] if "profitability_mode" in selected.columns and not selected.empty else "unknown")}
        if best is None or score > float(best["score"]):
            best = candidate
    if "product_id" in valid.columns:
        if "market_regime" not in valid.columns:
            valid["market_regime"] = _infer_market_regime(valid)
        grouped = valid.groupby([valid["product_id"].astype(str), valid["market_regime"].astype(str)])
        for (product_id, regime), group in grouped:
            if len(group) < MIN_CONTEXT_SAMPLES:
                continue
            context_best = None
            for q in [0.55, 0.70, 0.85, 0.94]:
                threshold = float(group[col].quantile(q))
                selected = group[group[col] >= threshold].copy()
                selected_count = int(len(selected))
                if selected_count < MIN_CONTEXT_SAMPLES:
                    continue
                stats = _ev_stats(selected["buy_net_bps"])
                avg_adverse = float(pd.to_numeric(selected["buy_adverse_bps"], errors="coerce").fillna(0.0).mean())
                score = _four_pass_score_from_ev(smoothed_win_rate=stats["smoothed_win_rate"], ev_bps=stats["ev_bps"], avg_net_bps=stats["avg_net_bps"], median_net_bps=stats["median_net_bps"], avg_adverse_bps=avg_adverse, selected_count=selected_count, sample_floor=50)
                candidate = {"agent": agent, "side": "BUY", "product_id": str(product_id), "market_regime": str(regime), "source_column": col, "sample_count": int(len(group)), "selected_count": selected_count, "threshold": threshold, "raw_win_rate": stats["raw_win_rate"], "smoothed_win_rate": stats["smoothed_win_rate"], "avg_win_bps": stats["avg_win_bps"], "avg_loss_bps": stats["avg_loss_bps"], "ev_bps": stats["ev_bps"], "avg_net_bps": stats["avg_net_bps"], "median_net_bps": stats["median_net_bps"], "avg_adverse_bps": avg_adverse, "score": score, "profitability_mode": str(selected.get("profitability_mode", pd.Series(["unknown"])).iloc[0] if "profitability_mode" in selected.columns and not selected.empty else "unknown")}
                if context_best is None or score > float(context_best["score"]):
                    context_best = candidate
            if context_best is not None:
                context_rows_for_weighting.append(context_best)
    return best, context_rows_for_weighting

def _four_pass_buy_agent_rows(base_dir: str) -> Tuple[List[List[Any]], Dict[str, float], pd.DataFrame, List[List[Any]]]:
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    frame = _build_buy_training_frame(base_dir)

    if frame.empty:
        return [], {}, pd.DataFrame(), []

    score_cols = _buy_agent_score_columns(frame)
    if not score_cols:
        return [], {}, frame, []

    rows_for_weighting: List[Dict[str, Any]] = []
    context_rows_for_weighting: List[Dict[str, Any]] = []
    scan_items = list(score_cols.items())
    with ThreadPoolExecutor(max_workers=max(1, int(FOUR_PASS_AGENT_SCAN_WORKERS))) as executor:
        futures = [executor.submit(_scan_buy_agent_thresholds, agent, col, frame) for agent, col in scan_items]
        for future in as_completed(futures):
            try:
                best, context_rows = future.result()
                if best is not None:
                    rows_for_weighting.append(best)
                if context_rows:
                    context_rows_for_weighting.extend(context_rows)
            except Exception:
                pass

    rows_for_weighting = _softmax_weights(rows_for_weighting, score_key="score", raw_key="raw_authority", weight_key="buy_weight_pct")
    context_rows_for_weighting = _softmax_weights_by_group(context_rows_for_weighting, group_keys=["side", "product_id", "market_regime"], score_key="score", raw_key="raw_authority", weight_key="weight_pct")
    output_rows: List[List[Any]] = []
    weights: Dict[str, float] = {}
    for row in rows_for_weighting:
        agent = str(row["agent"])
        weights[agent] = float(row["buy_weight_pct"])
        output_rows.append([f"{ts_value:.6f}", dt_value, "buy_pass_1_agent_only_all_agents_bayesian_ev", agent, row["source_column"], int(row["sample_count"]), int(row["selected_count"]), f"{float(row['threshold']):.6f}", f"{float(row['win_rate']):.6f}", f"{float(row['avg_net_bps']):.6f}", f"{float(row['median_net_bps']):.6f}", f"{float(row['avg_adverse_bps']):.6f}", f"{float(row['score']):.6f}", f"{float(row['raw_authority']):.6f}", f"{float(row['buy_weight_pct']):.6f}", (f"buy_agent_pass_all_agents_bayesian_ev;agent={agent};source={row['source_column']};smoothed_win_rate={float(row['win_rate']):.4f};raw_win_rate={float(row['raw_win_rate']):.4f};ev={float(row['ev_bps']):.2f};avg_net={float(row['avg_net_bps']):.2f};median_net={float(row['median_net_bps']):.2f};weight={float(row['buy_weight_pct']):.2f}%")])

    context_output_rows: List[List[Any]] = []
    for row in context_rows_for_weighting:
        context_output_rows.append([f"{ts_value:.6f}", dt_value, row["agent"], row["side"], row["product_id"], row["market_regime"], row["source_column"], int(row["sample_count"]), int(row["selected_count"]), f"{float(row['threshold']):.6f}", f"{float(row['raw_win_rate']):.6f}", f"{float(row['smoothed_win_rate']):.6f}", f"{float(row['avg_win_bps']):.6f}", f"{float(row['avg_loss_bps']):.6f}", f"{float(row['ev_bps']):.6f}", f"{float(row['avg_net_bps']):.6f}", f"{float(row['median_net_bps']):.6f}", f"{float(row['avg_adverse_bps']):.6f}", f"{float(row['score']):.6f}", f"{float(row['raw_authority']):.6f}", f"{float(row['weight_pct']):.6f}", row.get("profitability_mode", "unknown"), "context_buy_rating;product_and_regime_specific"])
    return output_rows, weights, frame, context_output_rows

def _four_pass_council_buy_rows(base_dir: str, buy_weights: Dict[str, float], buy_frame: pd.DataFrame) -> Tuple[List[List[Any]], pd.DataFrame]:
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)

    if buy_frame is None or buy_frame.empty or not buy_weights:
        return [], pd.DataFrame()

    frame = buy_frame.copy()
    score_cols = _buy_agent_score_columns(frame)
    weighted_score = pd.Series(0.0, index=frame.index)
    total_weight = 0.0

    for agent, col in score_cols.items():
        weight = float(buy_weights.get(agent, 0.0) or 0.0)
        if weight <= 0:
            continue
        values = _numeric(frame, col, 0.0)
        if float(values.max()) > 1.50:
            values = values / 100.0
        values = values.clip(0.0, 1.0)
        weighted_score += values * weight
        total_weight += weight

    if total_weight <= 0:
        return [], pd.DataFrame()

    frame["four_pass_buy_council_score"] = weighted_score / total_weight
    rows: List[List[Any]] = []
    selected_all = pd.DataFrame()

    for product_id, group in frame.groupby(frame["product_id"].astype(str)):
        group = group.copy()
        best = None
        for q in [0.50, 0.60, 0.70, 0.80, 0.88, 0.94]:
            threshold = float(group["four_pass_buy_council_score"].quantile(q))
            selected = group[group["four_pass_buy_council_score"] >= threshold].copy()
            selected_count = int(len(selected))
            if selected_count < 10:
                continue
            win_rate = float(selected["buy_success"].mean())
            avg_net = float(selected["buy_net_bps"].mean())
            median_net = float(selected["buy_net_bps"].median())
            avg_adverse = float(selected["buy_adverse_bps"].mean())
            portfolio_return_pct = float(selected["buy_net_bps"].sum() * 5.0 / 10000.0)
            sample_factor = min(1.0, math.sqrt(selected_count / 80.0))
            score = max(0.03, min(0.97, 0.50 + (win_rate - 0.50) * 0.85 * sample_factor + max(-200.0, min(300.0, avg_net)) / 650.0 + max(-200.0, min(300.0, median_net)) / 900.0 - max(0.0, avg_adverse - 100.0) / 850.0))
            candidate = {"product_id": product_id, "sample_count": int(len(group)), "selected_count": selected_count, "threshold": threshold, "win_rate": win_rate, "avg_net_bps": avg_net, "median_net_bps": median_net, "avg_adverse_bps": avg_adverse, "portfolio_return_pct_100_ref": portfolio_return_pct, "score": score, "profitability_mode": str(selected.get("profitability_mode", pd.Series(["unknown"])).iloc[0] if "profitability_mode" in selected.columns and not selected.empty else "unknown"), "selected": selected}
            if best is None or score > float(best["score"]):
                best = candidate
        if best is None:
            continue
        selected = best.pop("selected")
        selected["four_pass_buy_selected"] = 1
        selected["four_pass_buy_product_threshold"] = float(best["threshold"])
        selected["four_pass_buy_product_score"] = float(best["score"])
        selected_all = pd.concat([selected_all, selected], ignore_index=True, sort=False)
        rows.append([f"{ts_value:.6f}", dt_value, "buy_pass_2_weighted_council_all_agents", best["product_id"], int(best["sample_count"]), int(best["selected_count"]), f"{float(best['threshold']):.6f}", f"{float(best['win_rate']):.6f}", f"{float(best['avg_net_bps']):.6f}", f"{float(best['median_net_bps']):.6f}", f"{float(best['avg_adverse_bps']):.6f}", f"{float(best['portfolio_return_pct_100_ref']):.6f}", f"{float(best['score']):.6f}", best.get("profitability_mode", "unknown"), f"weighted_buy_council_all_agents;mode={best.get('profitability_mode', 'unknown')};product={best['product_id']};threshold={float(best['threshold']):.4f};win_rate={float(best['win_rate']):.4f};avg_net={float(best['avg_net_bps']):.2f};median_net={float(best['median_net_bps']):.2f}"])

    return rows, selected_all


def _build_sell_training_frame(base_dir: str, council_buy_entries: pd.DataFrame) -> pd.DataFrame:
    """SELL training frame based on weighted council BUY entries from Pass 2."""
    buy_rows = 0 if council_buy_entries is None or council_buy_entries.empty else len(council_buy_entries)
    buy_fingerprint = "empty"
    try:
        if council_buy_entries is not None and not council_buy_entries.empty:
            fingerprint_cols = [
                c for c in [
                    "product_id", "replay_ts", "ts", "entry_ts",
                    "buy_net_bps", "four_pass_buy_council_score",
                    "four_pass_buy_product_threshold"
                ]
                if c in council_buy_entries.columns
            ]
            fingerprint_frame = council_buy_entries[fingerprint_cols].copy() if fingerprint_cols else council_buy_entries.head(500).copy()
            buy_fingerprint = hashlib.sha256(
                pd.util.hash_pandas_object(fingerprint_frame, index=True).values.tobytes()
            ).hexdigest()
    except Exception:
        buy_fingerprint = "fingerprint_failed"
    cache_key = _four_pass_cache_key(
        base_dir,
        ["sell_outcomes.csv", "shadow_sell_replay.csv", "candidate_replay.csv", "historical_shadow_replay.csv"],
    ) + f":buy_rows={buy_rows}:buy_fingerprint={buy_fingerprint}"
    cached = _load_feature_frame_cached(base_dir, "sell_training_frame", cache_key)
    if not cached.empty:
        return cached

    if council_buy_entries is None or council_buy_entries.empty:
        return pd.DataFrame()
    buys = council_buy_entries.copy()
    if "product_id" not in buys.columns:
        return pd.DataFrame()
    for col in ["replay_ts", "ts", "buy_net_bps", "net_pnl_bps", "max_favorable_bps", "max_adverse_bps"]:
        if col in buys.columns:
            buys[col] = pd.to_numeric(buys[col], errors="coerce")
    if "buy_net_bps" not in buys.columns:
        buys["buy_net_bps"] = _numeric(buys, "net_pnl_bps", 0.0) if "net_pnl_bps" in buys.columns else 0.0
    sell_frame = buys.copy()
    sell_frame["profitability_mode"] = "buy_entry_proxy_sell_labels"
    sell_frame["realized_net_pnl_bps"] = _numeric(sell_frame, "buy_net_bps", 0.0)
    if "move_after_sell_bps" not in sell_frame.columns:
        sell_frame["move_after_sell_bps"] = _numeric(sell_frame, "max_favorable_bps", 0.0) - _numeric(sell_frame, "buy_net_bps", 0.0).clip(lower=0.0)
    sell_frame["good_sell_success"] = (sell_frame["realized_net_pnl_bps"] > 0.0) & (sell_frame["move_after_sell_bps"] <= 80.0)
    sell_frame["too_early"] = sell_frame["move_after_sell_bps"] >= 120.0

    external_frames = []
    for path in [os.path.join(base_dir, "sell_outcomes.csv"), os.path.join(base_dir, "shadow_sell_replay.csv")]:
        ext = _read_csv(path)
        if not ext.empty and "product_id" in ext.columns:
            external_frames.append(ext.copy())
    if external_frames:
        external = pd.concat(external_frames, ignore_index=True, sort=False)
        for col in ["decision_ts", "entry_ts", "replay_ts", "ts", "move_after_sell_bps", "realized_net_pnl_bps"]:
            if col in external.columns:
                external[col] = pd.to_numeric(external[col], errors="coerce")
        if "realized_net_pnl_bps" not in external.columns:
            external["realized_net_pnl_bps"] = _numeric(external, "net_pnl_bps", 0.0) if "net_pnl_bps" in external.columns else 0.0
        if "move_after_sell_bps" not in external.columns:
            external["move_after_sell_bps"] = 0.0
        external["good_sell_success"] = (external["realized_net_pnl_bps"] > 0.0) & (external["move_after_sell_bps"] <= 80.0)
        external["too_early"] = external["move_after_sell_bps"] >= 120.0
        allowed_products = set(buys["product_id"].astype(str).unique())
        external = external[external["product_id"].astype(str).isin(allowed_products)].copy()
        if not external.empty:
            external["profitability_mode"] = "external_sell_outcome_replay"
            sell_frame = external
    if "market_regime" not in sell_frame.columns:
        sell_frame["market_regime"] = _infer_market_regime(sell_frame)
    _save_feature_frame_cached(base_dir, "sell_training_frame", cache_key, sell_frame)
    return sell_frame


def _four_pass_sell_agent_rows(base_dir: str, council_buy_entries: pd.DataFrame) -> Tuple[List[List[Any]], Dict[str, float], pd.DataFrame]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); frame = _build_sell_training_frame(base_dir, council_buy_entries)
    if frame.empty: return [], {}, frame
    score_cols = _sell_agent_score_columns(frame)
    if not score_cols: return [], {}, frame
    rows_for_weighting: List[Dict[str, Any]] = []
    for agent, col in score_cols.items():
        frame[col] = _numeric(frame, col, 0.0); valid = frame.dropna(subset=[col]).copy()
        if valid.empty: continue
        best = None
        for q in [0.50, 0.60, 0.70, 0.80, 0.88]:
            threshold = float(valid[col].quantile(q)); selected = valid[valid[col] >= threshold].copy(); selected_count = int(len(selected))
            if selected_count < 5: continue
            good_exit_rate = float(selected["good_sell_success"].mean()); too_early_rate = float(selected["too_early"].mean()); avg_move_after = float(selected["move_after_sell_bps"].mean()); avg_realized = float(selected["realized_net_pnl_bps"].mean())
            sample_factor = min(1.0, math.sqrt(selected_count / 40.0)); score = max(0.05, min(0.95, 0.50 + (good_exit_rate - 0.50) * 0.90 * sample_factor - too_early_rate * 0.35 - max(0.0, avg_move_after - 30.0) / 500.0 + max(-150.0, min(250.0, avg_realized)) / 900.0))
            candidate = {"agent": agent, "source_column": col, "sample_count": int(len(valid)), "selected_count": selected_count, "threshold": threshold, "good_exit_rate": good_exit_rate, "too_early_rate": too_early_rate, "avg_move_after_sell_bps": avg_move_after, "avg_realized_net_bps": avg_realized, "score": score}
            if best is None or score > float(best["score"]): best = candidate
        if best is not None: rows_for_weighting.append(best)
    rows_for_weighting = _softmax_weights(rows_for_weighting, score_key="score", raw_key="raw_authority", weight_key="sell_weight_pct")
    output_rows: List[List[Any]] = []; weights: Dict[str, float] = {}
    for row in rows_for_weighting:
        weights[str(row["agent"])] = float(row["sell_weight_pct"])
        output_rows.append([f"{ts_value:.6f}", dt_value, "sell_pass_1_agent_only", row["agent"], row["source_column"], int(row["sample_count"]), int(row["selected_count"]), f"{float(row['threshold']):.6f}", f"{float(row['good_exit_rate']):.6f}", f"{float(row['too_early_rate']):.6f}", f"{float(row['avg_move_after_sell_bps']):.6f}", f"{float(row['avg_realized_net_bps']):.6f}", f"{float(row['score']):.6f}", f"{float(row['raw_authority']):.6f}", f"{float(row['sell_weight_pct']):.6f}", f"sell_agent_pass;agent={row['agent']};source={row['source_column']};score={float(row['score']):.4f};weight={float(row['sell_weight_pct']):.2f}"])
    return output_rows, weights, frame


def _four_pass_council_sell_rows(base_dir: str, sell_weights: Dict[str, float], sell_frame: pd.DataFrame) -> List[List[Any]]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value)
    if sell_frame is None or sell_frame.empty or not sell_weights: return []
    frame = sell_frame.copy(); score_cols = _sell_agent_score_columns(frame); weighted_score = pd.Series(0.0, index=frame.index); total_weight = 0.0
    for agent, col in score_cols.items():
        weight = float(sell_weights.get(agent, 0.0) or 0.0)
        if weight <= 0: continue
        normalized = _numeric(frame, col, 0.0)
        if normalized.max() > 1.50: normalized = normalized / 100.0
        weighted_score += normalized.clip(0.0, 1.0) * weight; total_weight += weight
    if total_weight <= 0: return []
    frame["four_pass_sell_council_score"] = weighted_score / total_weight
    rows: List[List[Any]] = []
    for product_id, group in frame.groupby(frame["product_id"].astype(str)):
        group = group.copy(); best = None
        for q in [0.55, 0.65, 0.75, 0.85, 0.92]:
            threshold = float(group["four_pass_sell_council_score"].quantile(q)); selected = group[group["four_pass_sell_council_score"] >= threshold].copy(); selected_count = int(len(selected))
            if selected_count < 5: continue
            good_exit_rate = float(selected["good_sell_success"].mean()); too_early_rate = float(selected["too_early"].mean()); avg_move_after = float(selected["move_after_sell_bps"].mean()); avg_realized = float(selected["realized_net_pnl_bps"].mean()); portfolio_return_pct = float(avg_realized * selected_count * 5.0 / 10000.0)
            sample_factor = min(1.0, math.sqrt(selected_count / 40.0)); score = max(0.05, min(0.95, 0.50 + (good_exit_rate - 0.50) * 0.90 * sample_factor - too_early_rate * 0.35 - max(0.0, avg_move_after - 30.0) / 500.0 + max(-150.0, min(250.0, avg_realized)) / 900.0))
            candidate = {"product_id": product_id, "sample_count": int(len(group)), "selected_count": selected_count, "threshold": threshold, "good_exit_rate": good_exit_rate, "too_early_rate": too_early_rate, "avg_move_after_sell_bps": avg_move_after, "avg_realized_net_bps": avg_realized, "portfolio_return_pct_100_ref": portfolio_return_pct, "score": score, "profitability_mode": str(selected.get("profitability_mode", pd.Series(["unknown"])).iloc[0] if "profitability_mode" in selected.columns and not selected.empty else "unknown")}
            if best is None or score > float(best["score"]): best = candidate
        if best is None: continue
        rows.append([f"{ts_value:.6f}", dt_value, "sell_pass_2_weighted_council", best["product_id"], int(best["sample_count"]), int(best["selected_count"]), f"{float(best['threshold']):.6f}", f"{float(best['good_exit_rate']):.6f}", f"{float(best['too_early_rate']):.6f}", f"{float(best['avg_move_after_sell_bps']):.6f}", f"{float(best['avg_realized_net_bps']):.6f}", f"{float(best['portfolio_return_pct_100_ref']):.6f}", f"{float(best['score']):.6f}", best.get("profitability_mode", "unknown"), f"weighted_sell_council;mode={best.get('profitability_mode', 'unknown')};product={best['product_id']};good_exit_rate={float(best['good_exit_rate']):.2f};too_early={float(best['too_early_rate']):.2f}"])
    return rows


def _four_pass_final_agent_rating_rows(buy_agent_rows: List[List[Any]], sell_agent_rows: List[List[Any]]) -> List[List[Any]]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); buy_by_agent: Dict[str, Dict[str, Any]] = {}; sell_by_agent: Dict[str, Dict[str, Any]] = {}
    for row in buy_agent_rows:
        buy_by_agent[str(row[3])] = {"rows": int(row[6]), "accuracy": float(row[8]), "avg": float(row[9]), "score": float(row[12]), "weight": float(row[14])}
    for row in sell_agent_rows:
        sell_by_agent[str(row[3])] = {"rows": int(row[6]), "accuracy": float(row[8]), "avg": float(row[11]), "score": float(row[12]), "weight": float(row[14])}
    rows: List[List[Any]] = []
    for agent in sorted(set(buy_by_agent.keys()) | set(sell_by_agent.keys())):
        buy = buy_by_agent.get(agent, {}); sell = sell_by_agent.get(agent, {})
        rows.append([f"{ts_value:.6f}", dt_value, agent, int(buy.get("rows", 0)), f"{float(buy.get('accuracy', 0.50)):.6f}", f"{float(buy.get('avg', 0.0)):.6f}", f"{float(buy.get('score', 0.50)):.6f}", f"{float(buy.get('weight', 0.0)):.6f}", int(sell.get("rows", 0)), f"{float(sell.get('accuracy', 0.50)):.6f}", f"{float(sell.get('avg', 0.0)):.6f}", f"{float(sell.get('score', 0.50)):.6f}", f"{float(sell.get('weight', 0.0)):.6f}", "mixed_four_pass", "four_pass_final_agent_rating;buy_uses_all_available_agents;sell_is_based_on_weighted_council_buy_entries"])
    return rows



def _four_pass_sell_path_replay_rows(
    base_dir: str,
    council_buy_entries: pd.DataFrame,
    sell_weights: Dict[str, float],
    sell_frame: pd.DataFrame,
) -> List[List[Any]]:
    """
    Sell-path replay output.

    Priority:
    1. Use realized external sell rows when available.
    2. If realized external sell rows are not available, create clearly labeled
       proxy sell-path rows from weighted council buy entries / sell frame.

    This does not pretend proxy rows are final proof.
    It allows the viewer and profitability summary to see what the sell model
    would have done while still labeling it honestly.
    """
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)

    if sell_frame is None or sell_frame.empty:
        return []

    frame = sell_frame.copy()

    if "product_id" not in frame.columns:
        return []

    if council_buy_entries is not None and not council_buy_entries.empty and "product_id" in council_buy_entries.columns:
        allowed_products = set(council_buy_entries["product_id"].astype(str).unique())
        frame = frame[frame["product_id"].astype(str).isin(allowed_products)].copy()

    if frame.empty:
        return []

    if "realized_net_pnl_bps" not in frame.columns:
        if "net_pnl_bps" in frame.columns:
            frame["realized_net_pnl_bps"] = _numeric(frame, "net_pnl_bps", 0.0)
        elif "buy_net_bps" in frame.columns:
            frame["realized_net_pnl_bps"] = _numeric(frame, "buy_net_bps", 0.0)
        else:
            frame["realized_net_pnl_bps"] = 0.0
    else:
        frame["realized_net_pnl_bps"] = _numeric(frame, "realized_net_pnl_bps", 0.0)

    if "move_after_sell_bps" not in frame.columns:
        frame["move_after_sell_bps"] = 0.0

    frame["move_after_sell_bps"] = _numeric(frame, "move_after_sell_bps", 0.0)

    score_cols = _sell_agent_score_columns(frame)

    sort_col = "realized_net_pnl_bps"
    frame = frame.sort_values(sort_col, ascending=False).head(5000).copy()

    rows: List[List[Any]] = []

    for _, r in frame.iterrows():
        product_id = str(r.get("product_id") or "")

        best_agent = ""
        best_score = 0.0

        for agent, col in score_cols.items():
            if col not in r.index:
                continue
            try:
                score_val = float(r.get(col, 0.0) or 0.0)
                if score_val > best_score:
                    best_score = score_val
                    best_agent = agent
            except Exception:
                continue

        if not best_agent and sell_weights:
            try:
                best_agent = max(sell_weights.items(), key=lambda kv: float(kv[1]))[0]
                best_score = float(sell_weights.get(best_agent, 0.0) or 0.0)
            except Exception:
                best_agent = ""
                best_score = 0.0

        entry_ts = float(r.get("entry_ts", r.get("replay_ts", r.get("ts", 0.0))) or 0.0)
        exit_ts = float(r.get("exit_ts", r.get("decision_ts", r.get("ts", entry_ts))) or entry_ts)
        held_minutes = max(0.0, (exit_ts - entry_ts) / 60.0) if exit_ts and entry_ts else 0.0

        entry_price = float(r.get("entry_price", r.get("buy_price", 0.0)) or 0.0)
        exit_price = float(r.get("exit_price", r.get("sell_price", 0.0)) or 0.0)

        realized_net = float(r.get("realized_net_pnl_bps", 0.0) or 0.0)
        max_fav = float(r.get("max_favorable_bps", r.get("max_favorable_after_entry_bps", realized_net)) or 0.0)
        max_adv = float(r.get("max_adverse_bps", r.get("max_adverse_after_entry_bps", 0.0)) or 0.0)

        mode = str(r.get("profitability_mode", "") or "")
        if not mode:
            mode = "proxy_sell_path_from_buy_entries"

        if "proxy" not in mode.lower() and not entry_price and not exit_price:
            mode = "proxy_sell_path_from_buy_entries"

        rows.append([
            f"{ts_value:.6f}",
            dt_value,
            product_id,
            f"{entry_ts:.6f}",
            f"{exit_ts:.6f}",
            f"{entry_price:.12f}",
            f"{exit_price:.12f}",
            f"{held_minutes:.3f}",
            best_agent,
            f"{best_score:.6f}",
            str(r.get("exit_reason", r.get("reason", "sell_path_proxy"))),
            f"{realized_net:.6f}",
            f"{max_fav:.6f}",
            f"{max_adv:.6f}",
            mode,
            "sell_path_output;proxy_rows_are_not_final_live_proof",
        ])

    return rows

def _four_pass_product_live_gate_rows(
    council_buy_rows: List[List[Any]],
    purged_walk_forward_rows: List[List[Any]],
    sell_path_replay_rows: List[List[Any]],
) -> Tuple[List[List[Any]], List[List[Any]]]:
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    wf_by_side = {"BUY": [], "SELL": []}
    for row in purged_walk_forward_rows or []:
        try:
            side = str(row[3]).upper()
            wf_by_side.setdefault(side, []).append(row)
        except Exception:
            pass
    buy_wf = wf_by_side.get("BUY", [])
    positive_buy_folds = 0
    validation_net_values = []
    validation_return_values = []
    for row in buy_wf:
        try:
            verdict = str(row[19])
            validation_net = float(row[15])
            validation_return = float(row[18])
            validation_net_values.append(validation_net)
            validation_return_values.append(validation_return)
            if verdict != "not_validated" and validation_net > 0.0:
                positive_buy_folds += 1
        except Exception:
            pass
    wf_folds = len(buy_wf)
    wf_avg_net = sum(validation_net_values) / max(1, len(validation_net_values))
    wf_avg_return = sum(validation_return_values) / max(1, len(validation_return_values))
    sell_path_by_product: Dict[str, List[float]] = {}
    for row in sell_path_replay_rows or []:
        try:
            product_id = str(row[2])
            realized_net = float(row[11])
            sell_path_by_product.setdefault(product_id, []).append(realized_net)
        except Exception:
            pass
    gate_rows: List[List[Any]] = []
    cooldown_rows: List[List[Any]] = []
    for row in council_buy_rows or []:
        try:
            product_id = str(row[3])
            selected_count = int(float(row[5]))
            win_rate = float(row[7])
            avg_net = float(row[8])
            median_net = float(row[9])
            score = float(row[12])
            profitability_mode = str(row[13])
            sell_path_values = sell_path_by_product.get(product_id, [])
            sell_path_rows = len(sell_path_values)
            sell_path_avg = sum(sell_path_values) / max(1, sell_path_rows)
            sell_path_return = sum(v * 5.0 / 10000.0 for v in sell_path_values)
            proxy_mode = "proxy" in profitability_mode.lower()
            realized_mode = "realized" in profitability_mode.lower() or "exit" in profitability_mode.lower()
            buy_ok = selected_count >= 10 and win_rate >= 0.56 and avg_net > 0.0 and median_net > 0.0 and score >= 0.50
            if proxy_mode and not realized_mode:
                buy_ok = selected_count >= 50 and win_rate >= 0.64 and avg_net >= 35.0 and median_net >= 18.0 and score >= 0.58
            walk_forward_ok = wf_folds >= 2 and positive_buy_folds >= max(1, int(wf_folds * 0.50)) and wf_avg_net > 0.0
            sell_path_ok = True
            sell_path_note = "no_sell_path_rows_available"
            if sell_path_rows > 0:
                sell_path_ok = sell_path_avg > 0.0
                sell_path_note = "sell_path_positive" if sell_path_ok else "sell_path_negative"
            approved = bool(buy_ok and walk_forward_ok and sell_path_ok)
            cooldown_minutes = 0

            if approved:
                reason = f"approved;selected={selected_count};win={win_rate:.3f};avg={avg_net:.2f};median={median_net:.2f};wf_folds={wf_folds};wf_positive={positive_buy_folds};sell_path_rows={sell_path_rows};sell_path_avg={sell_path_avg:.2f};sell_path_note={sell_path_note}"
            else:
                if avg_net < 0 or median_net < 0:
                    market_state = "negative_ev_watch_only"
                elif not walk_forward_ok:
                    market_state = "validation_wait"
                else:
                    market_state = "near_miss"

                reason = (
                    f"not_live_eligible_now;market_state={market_state};"
                    f"buy_ok={buy_ok};walk_forward_ok={walk_forward_ok};sell_path_ok={sell_path_ok};"
                    f"selected={selected_count};win={win_rate:.3f};avg={avg_net:.2f};median={median_net:.2f};"
                    f"wf_folds={wf_folds};wf_positive={positive_buy_folds};"
                    f"sell_path_rows={sell_path_rows};sell_path_avg={sell_path_avg:.2f};sell_path_note={sell_path_note}"
                )

            gate_rows.append([f"{ts_value:.6f}", dt_value, product_id, int(selected_count), f"{win_rate:.6f}", f"{avg_net:.6f}", f"{median_net:.6f}", f"{score:.6f}", profitability_mode, int(wf_folds), int(positive_buy_folds), f"{wf_avg_net:.6f}", f"{wf_avg_return:.6f}", int(sell_path_rows), f"{sell_path_avg:.6f}", f"{sell_path_return:.6f}", int(1 if approved else 0), int(cooldown_minutes), reason])
        except Exception:
            continue
    return gate_rows, cooldown_rows


def _four_pass_profitability_summary_rows(
    buy_agent_rows: List[List[Any]],
    council_buy_rows: List[List[Any]],
    sell_agent_rows: List[List[Any]],
    council_sell_rows: List[List[Any]],
    sell_path_replay_rows: List[List[Any]] = None,
) -> List[List[Any]]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value)
    def _sum_return(rows: List[List[Any]], return_index: int) -> float:
        total = 0.0
        for row in rows:
            try: total += float(row[return_index])
            except Exception: pass
        return float(total)
    def _count_positive(rows: List[List[Any]], avg_index: int, median_index: Optional[int] = None) -> int:
        count = 0
        for row in rows:
            try:
                avg_ok = float(row[avg_index]) > 0.0
                median_ok = True if median_index is None else float(row[median_index]) > 0.0
                if avg_ok and median_ok: count += 1
            except Exception: pass
        return count
    buy_return = _sum_return(council_buy_rows, 11)
    sell_return = _sum_return(council_sell_rows, 11)
    true_sell_path_return = 0.0
    if sell_path_replay_rows:
        try:
            true_sell_path_return = sum(float(r[11]) * 5.0 / 10000.0 for r in sell_path_replay_rows)
        except Exception:
            true_sell_path_return = 0.0
    buy_modes = sorted(set(str(row[13]) for row in council_buy_rows if len(row) > 13)) if council_buy_rows else []
    sell_modes = sorted(set(str(row[13]) for row in council_sell_rows if len(row) > 13)) if council_sell_rows else []
    buy_positive_products = _count_positive(council_buy_rows, 8, 9)
    sell_positive_products = _count_positive(council_sell_rows, 10, None)
    final_return = true_sell_path_return if sell_path_replay_rows else (sell_return if council_sell_rows else buy_return)
    buy_mode_text = ",".join(buy_modes) if buy_modes else "none"
    sell_mode_text = ",".join(sell_modes) if sell_modes else "none"
    sell_is_real = (
        bool(sell_path_replay_rows)
        or "external_sell_outcome_replay" in sell_mode_text
        or "realized" in sell_mode_text
        or "exit" in sell_mode_text
    )
    sell_is_proxy = "proxy" in sell_mode_text and not bool(sell_path_replay_rows)
    sell_path_modes = []
    try:
        sell_path_modes = [str(r[14]).lower() for r in (sell_path_replay_rows or []) if len(r) > 14]
    except Exception:
        sell_path_modes = []

    sell_path_has_proxy = any("proxy" in m for m in sell_path_modes)
    sell_path_has_realized = any(("realized" in m or "external" in m or "exit" in m) and "proxy" not in m for m in sell_path_modes)

    if sell_path_replay_rows and true_sell_path_return > 0 and sell_path_has_realized and not sell_path_has_proxy:
        verdict = "positive_true_sell_path_replay"
    elif sell_path_replay_rows and true_sell_path_return > 0 and sell_path_has_proxy:
        verdict = "positive_proxy_sell_path_not_final_live_proof"
    elif council_sell_rows and sell_return > 0 and sell_is_real and not sell_is_proxy:
        verdict = "positive_realized_sell_replay"
    elif council_sell_rows and sell_return > 0 and sell_is_proxy:
        verdict = "positive_proxy_sell_model_not_final_live_proof"
    elif council_buy_rows and buy_return > 0:
        verdict = "positive_buy_timing_not_final_exit_proof"
    else:
        verdict = "not_profitable_yet"
    reason = (
        f"buy_return={buy_return:.4f};sell_return={sell_return:.4f};"
        f"true_sell_path_return={true_sell_path_return:.4f};"
        f"buy_positive_products={buy_positive_products};sell_positive_products={sell_positive_products};"
        f"buy_modes={buy_mode_text};sell_modes={sell_mode_text};"
        f"true_sell_path_rows={len(sell_path_replay_rows or [])}"
    )
    return [[f"{ts_value:.6f}", dt_value, int(len(buy_agent_rows)), int(len(council_buy_rows)), int(len(sell_agent_rows)), int(len(council_sell_rows)), buy_mode_text, sell_mode_text, f"{buy_return:.6f}", f"{sell_return:.6f}", f"{final_return:.6f}", int(buy_positive_products), int(sell_positive_products), verdict, reason]]


def _purged_walk_forward_rows(base_dir: str, frame: pd.DataFrame, *, side: str = "BUY") -> List[List[Any]]:
    """Purged walk-forward validation with a fixed embargo gap."""
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value)
    if frame is None or frame.empty:
        return []
    work = frame.copy()
    time_col = next((c for c in ["replay_ts", "entry_ts", "ts"] if c in work.columns), None)
    if not time_col:
        return []
    work["_time"] = pd.to_numeric(work[time_col], errors="coerce")
    work = work.dropna(subset=["_time"]).sort_values("_time").copy()
    if work.empty:
        return []
    if side.upper() == "BUY":
        if "buy_net_bps" not in work.columns:
            return []
        net_col = "buy_net_bps"
    else:
        if "realized_net_pnl_bps" in work.columns:
            net_col = "realized_net_pnl_bps"
        elif "net_pnl_bps" in work.columns:
            net_col = "net_pnl_bps"
        else:
            return []
    work["_net"] = pd.to_numeric(work[net_col], errors="coerce").fillna(0.0)
    min_t = float(work["_time"].min()); max_t = float(work["_time"].max())
    if max_t <= min_t:
        return []
    fold_count = 4; fold_span = (max_t - min_t) / float(fold_count + 1); embargo_seconds = 6 * 60 * 60
    rows: List[List[Any]] = []
    for fold_id in range(1, fold_count + 1):
        train_start = min_t; train_end = min_t + fold_span * fold_id
        embargo_start = train_end; embargo_end = embargo_start + embargo_seconds
        validation_start = embargo_end; validation_end = validation_start + fold_span
        train = work[(work["_time"] >= train_start) & (work["_time"] < train_end)].copy()
        validation = work[(work["_time"] >= validation_start) & (work["_time"] < validation_end)].copy()
        if len(train) < 50 or len(validation) < 20:
            continue
        train_stats = _ev_stats(train["_net"]); validation_stats = _ev_stats(validation["_net"])
        validation_reference_return_pct = float(validation["_net"].sum() * 5.0 / 10000.0)
        if validation_stats["avg_net_bps"] > 0.0 and validation_stats["median_net_bps"] > 0.0:
            verdict = "validated_positive"
        elif validation_stats["avg_net_bps"] > 0.0:
            verdict = "positive_average_weak_median"
        else:
            verdict = "not_validated"
        rows.append([
            f"{ts_value:.6f}", dt_value, int(fold_id), side.upper(), f"{train_start:.6f}", f"{train_end:.6f}",
            f"{embargo_start:.6f}", f"{embargo_end:.6f}", f"{validation_start:.6f}", f"{validation_end:.6f}",
            int(len(train)), int(len(validation)), f"{float(train_stats['smoothed_win_rate']):.6f}",
            f"{float(validation_stats['smoothed_win_rate']):.6f}", f"{float(train_stats['avg_net_bps']):.6f}",
            f"{float(validation_stats['avg_net_bps']):.6f}", f"{float(train_stats['median_net_bps']):.6f}",
            f"{float(validation_stats['median_net_bps']):.6f}", f"{validation_reference_return_pct:.6f}", verdict,
            f"purged_walk_forward;side={side.upper()};embargo_hours=6;fold={fold_id};train_avg={float(train_stats['avg_net_bps']):.2f};val_avg={float(validation_stats['avg_net_bps']):.2f}",
        ])
    return rows


def _agent_decision_influence_rows(buy_agent_rows: List[List[Any]], sell_agent_rows: List[List[Any]], buy_frame: pd.DataFrame, sell_frame: pd.DataFrame) -> List[List[Any]]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); rows: List[Dict[str, Any]] = []
    for source_rows, side, days in [(buy_agent_rows or [], "BUY", _training_days_from_frame(buy_frame)), (sell_agent_rows or [], "SELL", _training_days_from_frame(sell_frame))]:
        for row in source_rows:
            try:
                agent = str(row[3]); selected_count = int(float(row[6])); frequency_per_day = float(selected_count) / max(1.0, days)
                if side == "BUY":
                    smoothed = float(row[8]); avg_net = float(row[9]); median_net = float(row[10]); avg_adv = float(row[11]); reason = str(row[15]); raw = smoothed; ev = avg_net; role_prefix = "buy"
                else:
                    smoothed = float(row[8]); avg_net = float(row[11]); median_net = avg_net; avg_adv = 0.0; reason = str(row[15]); raw = smoothed; ev = avg_net; role_prefix = "sell"
                stats = _decision_influence_score(selected_count=selected_count, frequency_per_day=frequency_per_day, smoothed_win_rate=smoothed, ev_bps=ev, avg_net_bps=avg_net, median_net_bps=median_net, avg_loss_bps=0.0)
                role = f"{role_prefix}_leader" if stats["decision_influence_score"] >= 1.40 else (f"{role_prefix}_support" if stats["decision_influence_score"] >= 0.75 else f"{role_prefix}_context")
                rows.append({"agent": agent, "side": side, "sample_count": int(float(row[5])), "selected_count": selected_count, "frequency_per_day": frequency_per_day, "raw_win_rate": raw, "smoothed_win_rate": smoothed, "avg_win_bps": 0.0, "avg_loss_bps": 0.0, "avg_net_bps": avg_net, "median_net_bps": median_net, "ev_bps": ev, "avg_adverse_bps": avg_adv, "role": role, "reason": f"global_{role_prefix}_influence;{reason}", **stats})
            except Exception:
                continue
    rows = _normalize_influence_rows(rows, group_keys=["side"])
    return [[f"{ts_value:.6f}", dt_value, r["agent"], r["side"], int(r["sample_count"]), int(r["selected_count"]), f"{float(r['frequency_per_day']):.6f}", f"{float(r['raw_win_rate']):.6f}", f"{float(r['smoothed_win_rate']):.6f}", f"{float(r['avg_win_bps']):.6f}", f"{float(r['avg_loss_bps']):.6f}", f"{float(r['avg_net_bps']):.6f}", f"{float(r['median_net_bps']):.6f}", f"{float(r['ev_bps']):.6f}", f"{float(r['avg_adverse_bps']):.6f}", f"{float(r['reliability_score']):.6f}", f"{float(r['frequency_score']):.6f}", f"{float(r['edge_score']):.6f}", f"{float(r['decision_influence_score']):.6f}", f"{float(r['decision_weight_pct']):.6f}", r["role"], r["reason"]] for r in rows]


def _product_agent_influence_rows(context_rows: List[List[Any]], buy_frame: pd.DataFrame) -> List[List[Any]]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); days = _training_days_from_frame(buy_frame); rows: List[Dict[str, Any]] = []
    for row in context_rows or []:
        try:
            agent, side, product_id, market_regime = str(row[2]), str(row[3]), str(row[4]), str(row[5])
            selected_count = int(float(row[8])); freq = float(selected_count) / max(1.0, days); smoothed = float(row[11]); ev = float(row[14]); avg_net = float(row[15]); median_net = float(row[16])
            stats = _decision_influence_score(selected_count=selected_count, frequency_per_day=freq, smoothed_win_rate=smoothed, ev_bps=ev, avg_net_bps=avg_net, median_net_bps=median_net, avg_loss_bps=0.0)
            role = "product_leader" if stats["decision_influence_score"] >= 1.40 else ("product_support" if stats["decision_influence_score"] >= 0.75 else "product_context")
            rows.append({"product_id": product_id, "market_regime": market_regime, "agent": agent, "side": side, "selected_count": selected_count, "frequency_per_day": freq, "smoothed_win_rate": smoothed, "ev_bps": ev, "avg_net_bps": avg_net, "median_net_bps": median_net, "role": role, "reason": "product_context_influence;frequency_weighted", **stats})
        except Exception:
            continue
    rows = _normalize_influence_rows(rows, group_keys=["side", "product_id", "market_regime"])
    return [[f"{ts_value:.6f}", dt_value, r["product_id"], r["market_regime"], r["agent"], r["side"], int(r["selected_count"]), f"{float(r['frequency_per_day']):.6f}", f"{float(r['smoothed_win_rate']):.6f}", f"{float(r['ev_bps']):.6f}", f"{float(r['avg_net_bps']):.6f}", f"{float(r['median_net_bps']):.6f}", f"{float(r['decision_influence_score']):.6f}", f"{float(r['decision_weight_pct']):.6f}", r["role"], r["reason"]] for r in rows]


def _estimate_trade_frequency_rows(
    buy_frame: pd.DataFrame,
    council_buy_rows: List[List[Any]],
    product_live_gate_rows: List[List[Any]] = None,
) -> List[List[Any]]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value)
    if buy_frame is None or buy_frame.empty or not council_buy_rows or "product_id" not in buy_frame.columns or "buy_net_bps" not in buy_frame.columns or "four_pass_buy_council_score" not in buy_frame.columns:
        return []
    frame = buy_frame.copy(); time_col = next((c for c in ["replay_ts", "entry_ts", "ts"] if c in frame.columns), "ts")
    frame["_time"] = pd.to_numeric(frame[time_col], errors="coerce"); frame["_net"] = pd.to_numeric(frame["buy_net_bps"], errors="coerce").fillna(0.0); frame = frame.dropna(subset=["_time"]).copy(); days = _training_days_from_frame(frame)
    selected_parts = []
    for row in council_buy_rows:
        try:
            pid, threshold = str(row[3]), float(row[6])
            selected_parts.append(frame[frame["product_id"].astype(str).eq(pid) & (pd.to_numeric(frame["four_pass_buy_council_score"], errors="coerce").fillna(0.0) >= threshold)].copy())
        except Exception:
            pass
    if not selected_parts: return []
    selected_all = pd.concat(selected_parts, ignore_index=True, sort=False); rows=[]
    def build(scope, pid, selected, dedupe):
        if selected.empty: return
        vals=[]; last=-1e30; gap=float(dedupe)*60.0
        for _, rr in selected.sort_values("_time").iterrows():
            t=float(rr["_time"])
            if t-last >= gap: vals.append(float(rr["_net"])); last=t
        if not vals: return
        ser=pd.Series(vals); wins=ser[ser>0]; losses=ser[ser<=0]; avg=float(ser.mean()); trades=float(len(ser))/max(1.0,days)
        rows.append([f"{ts_value:.6f}", dt_value, scope, pid, int(dedupe), f"{days:.6f}", int(len(selected)), int(len(ser)), f"{trades:.6f}", f"{float((ser>0).mean()):.6f}", f"{avg:.6f}", f"{float(ser.median()):.6f}", f"{float(wins.mean()) if not wins.empty else 0.0:.6f}", f"{float(abs(losses.mean())) if not losses.empty else 0.0:.6f}", f"{avg:.6f}", f"{trades*avg:.6f}", f"deduped_frequency_estimate;dedupe_minutes={dedupe}"])
    for dedupe in [15,30,60,120]:
        build("all_products", "ALL", selected_all, dedupe)
        for pid, group in selected_all.groupby(selected_all["product_id"].astype(str)):
            build("product", str(pid), group, dedupe)

    approved_products = set()
    for gate_row in product_live_gate_rows or []:
        try:
            if str(gate_row[16]).strip() in {"1", "true", "True", "yes", "YES"}:
                approved_products.add(str(gate_row[2]))
        except Exception:
            pass

    if approved_products:
        approved_selected = selected_all[selected_all["product_id"].astype(str).isin(approved_products)].copy()
        for dedupe in [15, 30, 60, 120]:
            build("approved_products", "APPROVED", approved_selected, dedupe)

    return rows

def _fifth_pass_row_market_eligible(row: pd.Series, approved_products: set) -> Tuple[bool, str]:
    """
    Historical/backlog version of the live market eligibility gate.

    This is intentionally conservative. It lets the fifth pass simulate the way
    the live bot should behave after calibration:
    - approved products can trade if the setup is favorable enough,
    - non-approved products can only trade if the current setup is exceptional.
    """
    try:
        product_id = str(row.get("product_id") or "")

        def f(name: str, default: float = 0.0) -> float:
            try:
                return float(row.get(name, default) or default)
            except Exception:
                return float(default)

        approved = product_id in approved_products

        spread_bps = f("spread_bps", 0.0)
        expected_utility_bps = f("expected_utility_bps", f("expected_net_edge_bps", 0.0))
        maker_adjusted_ev_bps = f("maker_adjusted_expected_value_bps", expected_utility_bps)
        calibrated_p_win = f("calibrated_p_win", f("estimated_prob_up", 0.50))
        payoff_ratio = f("payoff_ratio", 1.0 if expected_utility_bps > 0 else 0.0)
        buy_vs_wait_edge_bps = f("buy_vs_wait_edge_bps", expected_utility_bps)
        council_score = f("four_pass_buy_council_score", 0.0)
        threshold = f("four_pass_buy_product_threshold", 0.0)

        value_state = str(row.get("value_acceptance_state", "") or "").lower()
        volume_node_state = str(row.get("volume_node_state", "") or "").lower()
        poc_distance_bps = f("poc_distance_bps", 9999.0)
        wait_score = f("volume_profile_leader_wait_score", 0.50)

        if spread_bps and spread_bps > 25.0:
            return False, f"spread_too_wide:{spread_bps:.2f}"
        if volume_node_state == "low_volume_node":
            return False, "low_volume_node"
        if value_state == "accepted_above_value":
            return False, "accepted_above_value_chase"
        if value_state in {"inside_fair_value", "inside_value_near_poc"} and poc_distance_bps <= 20.0 and wait_score >= 0.58:
            return False, "inside_value_or_poc_chop"

        if approved:
            if expected_utility_bps < 20.0:
                return False, f"approved_product_utility_too_low:{expected_utility_bps:.2f}"
            if calibrated_p_win < 0.56:
                return False, f"approved_product_probability_too_low:{calibrated_p_win:.3f}"
            if payoff_ratio < 0.75:
                return False, f"approved_product_payoff_too_low:{payoff_ratio:.3f}"
            return True, "approved_product_market_eligible"

        if expected_utility_bps < 85.0:
            return False, f"nonapproved_utility_too_low:{expected_utility_bps:.2f}"
        if maker_adjusted_ev_bps < 18.0:
            return False, f"nonapproved_maker_ev_too_low:{maker_adjusted_ev_bps:.2f}"
        if calibrated_p_win < 0.64:
            return False, f"nonapproved_probability_too_low:{calibrated_p_win:.3f}"
        if payoff_ratio < 0.95:
            return False, f"nonapproved_payoff_too_low:{payoff_ratio:.3f}"
        if buy_vs_wait_edge_bps < 18.0:
            return False, f"nonapproved_buy_vs_wait_too_low:{buy_vs_wait_edge_bps:.2f}"
        if threshold > 0 and (council_score - threshold) < 0.04:
            return False, f"nonapproved_council_margin_too_low:{(council_score - threshold):.3f}"

        return True, "nonapproved_exceptional_market_eligible"
    except Exception as exc:
        return False, f"market_eligibility_error:{exc}"


def _fifth_pass_live_style_replay_rows(
    council_buy_entries: pd.DataFrame,
    product_live_gate_rows: List[List[Any]],
    *,
    dedupe_minutes: int = 60,
) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]], List[List[Any]]]:
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    if council_buy_entries is None or council_buy_entries.empty:
        return [], [], [], []
    frame = council_buy_entries.copy()
    if "product_id" not in frame.columns:
        return [], [], [], []
    approved_products = set()
    for gate_row in product_live_gate_rows or []:
        try:
            if str(gate_row[16]).strip() in {"1", "true", "True", "yes", "YES"}:
                approved_products.add(str(gate_row[2]))
        except Exception:
            pass
    time_col = next((c for c in ["replay_ts", "entry_ts", "ts"] if c in frame.columns), None)
    if not time_col:
        return [], [], [], []
    frame["_time"] = pd.to_numeric(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=["_time"]).sort_values("_time").copy()
    if frame.empty:
        return [], [], [], []
    if "buy_net_bps" in frame.columns:
        frame["_net"] = pd.to_numeric(frame["buy_net_bps"], errors="coerce").fillna(0.0)
        source_mode = "buy_net_bps_live_style_proxy"
    elif "net_pnl_bps" in frame.columns:
        frame["_net"] = pd.to_numeric(frame["net_pnl_bps"], errors="coerce").fillna(0.0)
        source_mode = "net_pnl_bps_live_style_proxy"
    elif "binance_taker_taker_net_pnl_bps" in frame.columns:
        frame["_net"] = pd.to_numeric(frame["binance_taker_taker_net_pnl_bps"], errors="coerce").fillna(0.0)
        source_mode = "binance_taker_taker_net_pnl_bps"
    else:
        return [], [], [], []
    days = _training_days_from_frame(frame)
    gap_seconds = float(dedupe_minutes) * 60.0
    replay_rows: List[List[Any]] = []
    blocker_counts: Dict[str, int] = {}
    last_trade_time_by_product: Dict[str, float] = {}
    for _, row in frame.iterrows():
        product_id = str(row.get("product_id") or "")
        t = float(row["_time"])
        last_t = float(last_trade_time_by_product.get(product_id, -1e30))
        eligible, elig_reason = _fifth_pass_row_market_eligible(row, approved_products)
        if not eligible:
            blocker_counts[f"{product_id}|{elig_reason}"] = blocker_counts.get(f"{product_id}|{elig_reason}", 0) + 1
            continue
        if t - last_t < gap_seconds:
            blocker = f"dedupe_spacing_{dedupe_minutes}m"
            blocker_counts[f"{product_id}|{blocker}"] = blocker_counts.get(f"{product_id}|{blocker}", 0) + 1
            continue
        net = float(row.get("_net", 0.0) or 0.0)
        entry_score = float(row.get("four_pass_buy_council_score", 0.0) or 0.0)
        threshold = float(row.get("four_pass_buy_product_threshold", 0.0) or 0.0)
        entry_ts = t
        exit_ts = float(row.get("exit_ts", row.get("decision_ts", entry_ts)) or entry_ts)
        held_minutes = max(0.0, (exit_ts - entry_ts) / 60.0)
        last_trade_time_by_product[product_id] = t
        replay_rows.append([f"{ts_value:.6f}", dt_value, product_id, f"{entry_ts:.6f}", f"{exit_ts:.6f}", f"{held_minutes:.3f}", f"{entry_score:.6f}", f"{threshold:.6f}", 1, elig_reason, f"{net:.6f}", int(1 if net > 0 else 0), source_mode, f"fifth_pass_live_style;dedupe_minutes={dedupe_minutes}"])
    summary_rows: List[List[Any]] = []
    contribution_rows: List[List[Any]] = []
    blocker_rows: List[List[Any]] = []
    def summarize(scope: str, product_id: str, rows: List[List[Any]]) -> None:
        if not rows:
            return
        vals = pd.Series([float(r[10]) for r in rows])
        wins = vals[vals > 0]
        losses = vals[vals <= 0]
        trade_count = int(len(vals))
        trades_per_day = float(trade_count) / max(1.0, float(days))
        win_rate = float((vals > 0).mean())
        avg_net = float(vals.mean())
        median_net = float(vals.median())
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0
        reference_return = float(vals.sum() * 5.0 / 10000.0)
        if trade_count <= 0:
            verdict = "no_live_style_trades"
        elif avg_net > 0 and median_net > 0 and win_rate >= 0.55:
            verdict = "live_style_profitable"
        elif avg_net > 0:
            verdict = "positive_average_weak_median_or_win_rate"
        else:
            verdict = "not_live_style_profitable"
        summary_rows.append([f"{ts_value:.6f}", dt_value, scope, product_id, f"{float(days):.6f}", int(len(frame)), trade_count, f"{trades_per_day:.6f}", f"{win_rate:.6f}", f"{avg_net:.6f}", f"{median_net:.6f}", f"{avg_win:.6f}", f"{avg_loss:.6f}", f"{reference_return:.6f}", verdict, f"fifth_pass_live_style_summary;dedupe_minutes={dedupe_minutes};source_mode={source_mode}"])
    summarize("all_live_style", "ALL", replay_rows)
    grouped: Dict[str, List[List[Any]]] = {}
    for r in replay_rows:
        grouped.setdefault(str(r[2]), []).append(r)
    product_stats = []
    for pid, rows in grouped.items():
        vals = pd.Series([float(r[10]) for r in rows])
        trade_count = int(len(vals))
        trades_per_day = float(trade_count) / max(1.0, float(days))
        win_rate = float((vals > 0).mean())
        avg_net = float(vals.mean())
        median_net = float(vals.median())
        reference_return = float(vals.sum() * 5.0 / 10000.0)
        product_stats.append((pid, trade_count, trades_per_day, win_rate, avg_net, median_net, reference_return))
        summarize("product_live_style", pid, rows)
    product_stats.sort(key=lambda x: float(x[6]), reverse=True)
    for rank, item in enumerate(product_stats, start=1):
        pid, trade_count, trades_per_day, win_rate, avg_net, median_net, reference_return = item
        contribution_rows.append([f"{ts_value:.6f}", dt_value, pid, int(trade_count), f"{trades_per_day:.6f}", f"{win_rate:.6f}", f"{avg_net:.6f}", f"{median_net:.6f}", f"{reference_return:.6f}", int(rank), "fifth_pass_product_contribution;ranked_by_reference_return"])
    for key, count in sorted(blocker_counts.items(), key=lambda kv: kv[1], reverse=True):
        try:
            pid, blocker = key.split("|", 1)
        except Exception:
            pid, blocker = "UNKNOWN", key
        blocker_rows.append([f"{ts_value:.6f}", dt_value, pid, blocker, int(count), "fifth_pass_blocker_count"])
    return replay_rows, summary_rows, contribution_rows, blocker_rows


def _four_pass_backtest_outputs(base_dir: str) -> Dict[str, Any]:
    buy_agent_rows, buy_weights, buy_frame, buy_context_rows = _four_pass_buy_agent_rows(base_dir)
    walk_forward_buy_rows = _purged_walk_forward_rows(base_dir, buy_frame, side="BUY")
    council_buy_rows, council_buy_entries = _four_pass_council_buy_rows(base_dir, buy_weights, buy_frame)
    sell_agent_rows, sell_weights, sell_frame = _four_pass_sell_agent_rows(base_dir, council_buy_entries)
    walk_forward_sell_rows = _purged_walk_forward_rows(base_dir, sell_frame, side="SELL")
    council_sell_rows = _four_pass_council_sell_rows(base_dir, sell_weights, sell_frame)
    sell_path_replay_rows = _four_pass_sell_path_replay_rows(base_dir, council_buy_entries, sell_weights, sell_frame)
    product_live_gate_rows, product_cooldown_rows = _four_pass_product_live_gate_rows(
        council_buy_rows,
        walk_forward_buy_rows + walk_forward_sell_rows,
        sell_path_replay_rows,
    )
    final_rating_rows = _four_pass_final_agent_rating_rows(buy_agent_rows, sell_agent_rows)
    agent_decision_influence_rows = _agent_decision_influence_rows(buy_agent_rows, sell_agent_rows, buy_frame, sell_frame)
    product_agent_influence_rows = _product_agent_influence_rows(buy_context_rows, buy_frame)
    trade_frequency_estimate_rows = _estimate_trade_frequency_rows(council_buy_entries, council_buy_rows, product_live_gate_rows)

    fifth_pass_replay_rows, fifth_pass_summary_rows, fifth_pass_product_contribution_rows, fifth_pass_blocker_rows = _fifth_pass_live_style_replay_rows(
        council_buy_entries,
        product_live_gate_rows,
        dedupe_minutes=60,
    )

    profitability_summary_rows = _four_pass_profitability_summary_rows(
        buy_agent_rows,
        council_buy_rows,
        sell_agent_rows,
        council_sell_rows,
        sell_path_replay_rows,
    )

    feature_store_summary_rows = _feature_store_summary_rows(base_dir)

    return {
        "buy_agent_rows": buy_agent_rows,
        "buy_weights": buy_weights,
        "council_buy_rows": council_buy_rows,
        "council_buy_entries": council_buy_entries,
        "sell_agent_rows": sell_agent_rows,
        "sell_weights": sell_weights,
        "council_sell_rows": council_sell_rows,
        "sell_path_replay_rows": sell_path_replay_rows,
        "purged_walk_forward_rows": walk_forward_buy_rows + walk_forward_sell_rows,
        "final_rating_rows": final_rating_rows,
        "profitability_summary_rows": profitability_summary_rows,
        "context_rating_rows": buy_context_rows,
        "feature_store_summary_rows": feature_store_summary_rows,
        "product_live_gate_rows": product_live_gate_rows,
        "product_cooldown_rows": [],
        "agent_decision_influence_rows": agent_decision_influence_rows,
        "product_agent_influence_rows": product_agent_influence_rows,
        "trade_frequency_estimate_rows": trade_frequency_estimate_rows,
        "fifth_pass_replay_rows": fifth_pass_replay_rows,
        "fifth_pass_summary_rows": fifth_pass_summary_rows,
        "fifth_pass_product_contribution_rows": fifth_pass_product_contribution_rows,
        "fifth_pass_blocker_rows": fifth_pass_blocker_rows,
    }


def _agent_ablation_rows(base_dir: str) -> List[List[Any]]:
    votes = _read_csv(os.path.join(base_dir, "council_votes.csv"))
    audit = _read_csv(os.path.join(base_dir, "decision_audit.csv"))
    observations = _read_csv(os.path.join(base_dir, "council_observation_outcomes.csv"))
    ts_value = _utc_ts()
    dt_value = _utc_dt(ts_value)
    rows: List[List[Any]] = []
    if votes.empty or "agent" not in votes.columns or "decision_id" not in votes.columns:
        return rows
    outcome = pd.DataFrame()
    if not audit.empty and "decision_id" in audit.columns:
        audit = audit.copy()
        for candidate_col in ["move_bps", "max_favorable_bps", "realized_net_pnl_bps"]:
            if candidate_col in audit.columns:
                audit["outcome_bps"] = pd.to_numeric(audit[candidate_col], errors="coerce").fillna(0.0)
                break
        if "outcome_bps" in audit.columns:
            outcome = audit[["decision_id", "outcome_bps"]].copy()
    if outcome.empty and not observations.empty and "decision_id" in observations.columns and "move_bps" in observations.columns:
        observations = observations.copy()
        observations["outcome_bps"] = pd.to_numeric(observations["move_bps"], errors="coerce").fillna(0.0)
        outcome = observations[["decision_id", "outcome_bps"]].copy()
    if outcome.empty:
        module_debug(MODULE_NAME, "backtest_agent_ablation_waiting_for_outcomes", data={"vote_rows": int(len(votes)) if hasattr(votes, "__len__") else 0, "audit_rows": int(len(audit)) if hasattr(audit, "__len__") else 0, "observation_rows": int(len(observations)) if hasattr(observations, "__len__") else 0, "reason": "not_enough_reviewed_outcomes_yet"}, level="INFO", also_overall=False)
        return rows
    votes = votes.copy()
    for col in ["adjusted_buy_score", "adjusted_sell_score", "adjusted_hold_score", "adjusted_wait_score", "confidence", "weight"]:
        votes[col] = _numeric(votes, col, 0.0)
    merged = votes.merge(outcome, on="decision_id", how="inner")
    if merged.empty:
        return rows
    merged["support_strength"] = (
        merged["adjusted_buy_score"]
        + merged["adjusted_sell_score"]
        + merged["adjusted_hold_score"] * 0.35
        - merged["adjusted_wait_score"] * 0.25
    ) * merged["confidence"].clip(lower=0.0, upper=1.0)
    for agent, group in merged.groupby("agent"):
        if len(group) < 12:
            continue
        threshold = float(group["support_strength"].median())
        supportive = group[group["support_strength"] >= threshold]
        not_supportive = group[group["support_strength"] < threshold]
        if supportive.empty or not_supportive.empty:
            continue
        avg_all = float(group["outcome_bps"].mean())
        avg_support = float(supportive["outcome_bps"].mean())
        avg_not = float(not_supportive["outcome_bps"].mean())
        support_edge = avg_support - avg_not
        wr_support = float((supportive["outcome_bps"] > 0.0).mean())
        wr_not = float((not_supportive["outcome_bps"] > 0.0).mean())
        score = support_edge * 0.70 + (wr_support - wr_not) * 100.0 * 0.30
        rows.append([
            f"{ts_value:.6f}", dt_value, str(agent), int(len(group)),
            f"{avg_all:.6f}", f"{avg_support:.6f}", f"{avg_not:.6f}", f"{support_edge:.6f}",
            f"{wr_support:.6f}", f"{wr_not:.6f}", f"{score:.6f}",
            (
                f"agent_ablation;agent={agent};n={len(group)};"
                f"support_edge_bps={support_edge:.2f};"
                f"wr_support={wr_support:.3f};wr_not={wr_not:.3f};"
                f"score={score:.2f}"
            ),
        ])
    return rows

def run_backtest_intelligence(*, base_dir: str, log_fn: Optional[Callable[[str], None]] = None, min_product_rows: int = 80) -> Dict[str, Any]:
    """Run CSV replay intelligence and write backtest recommendation CSVs."""
    def log(message: str) -> None:
        if log_fn is not None:
            try:
                log_fn(message)
                return
            except Exception:
                pass
        print(message)

    started = time.time()
    module_debug(
        MODULE_NAME,
        "backtest_intelligence_started",
        data={
            "base_dir": base_dir,
            "min_product_rows": min_product_rows,
        },
        level="INFO",
        also_overall=False,
    )
    base_dir = os.path.abspath(base_dir)
    buy_rows, buy_recs = _candidate_rows(base_dir, min_product_rows=int(min_product_rows))
    sell_rows = _sell_recommendation_rows(base_dir)
    agent_rows = _agent_prior_rows(base_dir)
    setup_rows = _setup_performance_rows(base_dir)
    walk_forward_rows = _walk_forward_validation_rows(base_dir)
    ablation_rows = _agent_ablation_rows(base_dir)
    four_pass = _four_pass_backtest_outputs(base_dir)

    recommendations_path = os.path.join(base_dir, "backtest_recommendations.csv")
    sell_recommendations_path = os.path.join(base_dir, "backtest_sell_recommendations.csv")
    agent_priors_path = os.path.join(base_dir, "backtest_agent_priors.csv")
    setup_performance_path = os.path.join(base_dir, "backtest_setup_performance.csv")
    walk_forward_path = os.path.join(base_dir, "walk_forward_validation.csv")
    agent_ablation_path = os.path.join(base_dir, "agent_ablation.csv")
    four_pass_agent_buy_path = os.path.join(base_dir, "four_pass_agent_buy_timing.csv")
    four_pass_council_buy_path = os.path.join(base_dir, "four_pass_council_buy_timing.csv")
    four_pass_agent_sell_path = os.path.join(base_dir, "four_pass_agent_sell_timing.csv")
    four_pass_council_sell_path = os.path.join(base_dir, "four_pass_council_sell_timing.csv")
    four_pass_final_agent_ratings_path = os.path.join(base_dir, "four_pass_final_agent_ratings.csv")
    four_pass_profitability_summary_path = os.path.join(base_dir, "four_pass_profitability_summary.csv")
    four_pass_agent_context_ratings_path = os.path.join(base_dir, "four_pass_agent_context_ratings.csv")
    four_pass_sell_path_replay_path = os.path.join(base_dir, "four_pass_sell_path_replay.csv")
    four_pass_purged_walk_forward_path = os.path.join(base_dir, "four_pass_purged_walk_forward.csv")
    four_pass_product_live_gate_path = os.path.join(base_dir, "four_pass_product_live_gate.csv")
    product_cooldowns_path = os.path.join(base_dir, "product_cooldowns.csv")
    feature_store_summary_path = os.path.join(base_dir, "feature_store_summary.csv")
    agent_decision_influence_path = os.path.join(base_dir, "agent_decision_influence.csv")
    product_agent_influence_path = os.path.join(base_dir, "product_agent_influence.csv")
    trade_frequency_estimate_path = os.path.join(base_dir, "trade_frequency_estimate.csv")
    fifth_pass_replay_path = os.path.join(base_dir, "fifth_pass_live_style_replay.csv")
    fifth_pass_summary_path = os.path.join(base_dir, "fifth_pass_live_style_summary.csv")
    fifth_pass_product_contribution_path = os.path.join(base_dir, "fifth_pass_product_contribution.csv")
    fifth_pass_blockers_path = os.path.join(base_dir, "fifth_pass_blockers.csv")
    summary_path = os.path.join(base_dir, "backtest_summary.csv")

    _write_rows(recommendations_path, BACKTEST_RECOMMENDATIONS_COLUMNS, buy_rows)
    _write_rows(sell_recommendations_path, BACKTEST_SELL_RECOMMENDATIONS_COLUMNS, sell_rows)
    _write_rows(agent_priors_path, BACKTEST_AGENT_PRIOR_COLUMNS, agent_rows)
    _write_rows(setup_performance_path, BACKTEST_SETUP_PERFORMANCE_COLUMNS, setup_rows)
    _write_rows(walk_forward_path, WALK_FORWARD_VALIDATION_COLUMNS, walk_forward_rows)
    _write_rows(agent_ablation_path, AGENT_ABLATION_COLUMNS, ablation_rows)
    _write_rows(four_pass_agent_buy_path, FOUR_PASS_AGENT_BUY_COLUMNS, four_pass["buy_agent_rows"])
    _write_rows(four_pass_council_buy_path, FOUR_PASS_COUNCIL_BUY_COLUMNS, four_pass["council_buy_rows"])
    _write_rows(four_pass_agent_sell_path, FOUR_PASS_AGENT_SELL_COLUMNS, four_pass["sell_agent_rows"])
    _write_rows(four_pass_council_sell_path, FOUR_PASS_COUNCIL_SELL_COLUMNS, four_pass["council_sell_rows"])
    _write_rows(four_pass_final_agent_ratings_path, FOUR_PASS_FINAL_AGENT_RATINGS_COLUMNS, four_pass["final_rating_rows"])
    _write_rows(four_pass_profitability_summary_path, FOUR_PASS_PROFITABILITY_SUMMARY_COLUMNS, four_pass["profitability_summary_rows"])
    _write_rows(four_pass_agent_context_ratings_path, FOUR_PASS_AGENT_CONTEXT_RATINGS_COLUMNS, four_pass["context_rating_rows"])
    _write_rows(four_pass_sell_path_replay_path, FOUR_PASS_SELL_PATH_REPLAY_COLUMNS, four_pass["sell_path_replay_rows"])
    _write_rows(four_pass_purged_walk_forward_path, FOUR_PASS_PURGED_WALK_FORWARD_COLUMNS, four_pass["purged_walk_forward_rows"])
    _write_rows(four_pass_product_live_gate_path, FOUR_PASS_PRODUCT_LIVE_GATE_COLUMNS, four_pass["product_live_gate_rows"])
    # Product cooldown timers are retired. Keep the file/header only for compatibility.
    _write_rows(product_cooldowns_path, PRODUCT_COOLDOWN_COLUMNS, [])
    _write_rows(feature_store_summary_path, FEATURE_STORE_SUMMARY_COLUMNS, four_pass["feature_store_summary_rows"])
    _write_rows(agent_decision_influence_path, AGENT_DECISION_INFLUENCE_COLUMNS, four_pass["agent_decision_influence_rows"])
    _write_rows(product_agent_influence_path, PRODUCT_AGENT_INFLUENCE_COLUMNS, four_pass["product_agent_influence_rows"])
    _write_rows(trade_frequency_estimate_path, TRADE_FREQUENCY_ESTIMATE_COLUMNS, four_pass["trade_frequency_estimate_rows"])
    _write_rows(fifth_pass_replay_path, FIFTH_PASS_LIVE_STYLE_REPLAY_COLUMNS, four_pass["fifth_pass_replay_rows"])
    _write_rows(fifth_pass_summary_path, FIFTH_PASS_LIVE_STYLE_SUMMARY_COLUMNS, four_pass["fifth_pass_summary_rows"])
    _write_rows(fifth_pass_product_contribution_path, FIFTH_PASS_PRODUCT_CONTRIBUTION_COLUMNS, four_pass["fifth_pass_product_contribution_rows"])
    _write_rows(fifth_pass_blockers_path, FIFTH_PASS_BLOCKER_COLUMNS, four_pass["fifth_pass_blocker_rows"])

    ts_value = _utc_ts()
    summary_rows = [
        [f"{ts_value:.6f}", _utc_dt(ts_value), "buy_recommendation_rows", len(buy_rows), "product-level candidate replay recommendations"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "sell_recommendation_rows", len(sell_rows), "product-level sell replay recommendations"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "agent_prior_rows", len(agent_rows), "profit-weighted agent priors"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "setup_performance_rows", len(setup_rows), "product/session/setup-specific profit replay"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "walk_forward_validation_rows", len(walk_forward_rows), "out-of-sample validation rows" if walk_forward_rows else "not enough reviewed outcomes yet"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "agent_ablation_rows", len(ablation_rows), "agent marginal contribution report" if ablation_rows else "not enough reviewed outcomes yet"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_agent_buy_rows", len(four_pass["buy_agent_rows"]), "pass 1 buy timing by prior session and trend"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_council_buy_rows", len(four_pass["council_buy_rows"]), "pass 2 buy timing by weighted prior session and trend"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_agent_sell_rows", len(four_pass["sell_agent_rows"]), "pass 3 sell timing by individual sell agents"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_council_sell_rows", len(four_pass["council_sell_rows"]), "pass 4 sell timing by weighted sell council"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_profitability_summary_rows", len(four_pass["profitability_summary_rows"]), "four-pass profitability summary with mode labels"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_agent_context_rating_rows", len(four_pass["context_rating_rows"]), "product and regime specific agent ratings"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_sell_path_replay_rows", len(four_pass["sell_path_replay_rows"]), "true sell-path replay rows when realized sell path data exists"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_purged_walk_forward_rows", len(four_pass["purged_walk_forward_rows"]), "purged walk-forward validation rows"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_product_live_gate_rows", len(four_pass["product_live_gate_rows"]), "product-level live buy approval rows"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "product_cooldown_rows", 0, "timer-based product cooldowns retired; market eligibility is used instead"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "agent_decision_influence_rows", len(four_pass["agent_decision_influence_rows"]), "frequency weighted global agent decision influence"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "product_agent_influence_rows", len(four_pass["product_agent_influence_rows"]), "frequency weighted product agent decision influence"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "trade_frequency_estimate_rows", len(four_pass["trade_frequency_estimate_rows"]), "deduped trade frequency and avg win/loss estimates"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "fifth_pass_live_style_replay_rows", len(four_pass["fifth_pass_replay_rows"]), "final live-style replay rows using market eligibility instead of timer cooldowns"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "fifth_pass_live_style_summary_rows", len(four_pass["fifth_pass_summary_rows"]), "final live-style profitability, win rate, and trades/day"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "fifth_pass_product_contribution_rows", len(four_pass["fifth_pass_product_contribution_rows"]), "per-product contribution from final live-style replay"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "fifth_pass_blocker_rows", len(four_pass["fifth_pass_blocker_rows"]), "why candidates were excluded from final live-style replay"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "runtime_seconds", f"{time.time() - started:.3f}", "backtest intelligence runtime"],
    ]
    _write_rows(summary_path, BACKTEST_SUMMARY_COLUMNS, summary_rows)

    log(
        f"[backtest] completed buy_recs={len(buy_rows)} sell_recs={len(sell_rows)} "
        f"agent_priors={len(agent_rows)} setup_performance={len(setup_rows)} "
        f"walk_forward={len(walk_forward_rows)} ablation={len(ablation_rows)} seconds={time.time() - started:.2f}"
    )
    module_debug(
        MODULE_NAME,
        "backtest_intelligence_completed",
        data={
            "buy_recommendation_rows": len(buy_rows),
            "sell_recommendation_rows": len(sell_rows),
            "agent_prior_rows": len(agent_rows),
            "setup_performance_rows": len(setup_rows),
            "walk_forward_rows": len(walk_forward_rows),
            "agent_ablation_rows": len(ablation_rows),
            "runtime_sec": round(time.time() - started, 3),
        },
        level="INFO",
        also_overall=True,
    )
    return {
        "buy_recommendations": len(buy_rows),
        "sell_recommendations": len(sell_rows),
        "agent_priors": len(agent_rows),
        "setup_performance": len(setup_rows),
        "walk_forward_validation": len(walk_forward_rows),
        "agent_ablation": len(ablation_rows),
        "four_pass_product_live_gate": four_pass_product_live_gate_path,
        "product_cooldowns": product_cooldowns_path,
        "four_pass_agent_buy": len(four_pass["buy_agent_rows"]),
        "four_pass_council_buy": len(four_pass["council_buy_rows"]),
        "four_pass_agent_sell": len(four_pass["sell_agent_rows"]),
        "four_pass_council_sell": len(four_pass["council_sell_rows"]),
        "runtime_seconds": time.time() - started,
        "paths": {
            "backtest_recommendations": recommendations_path,
            "backtest_sell_recommendations": sell_recommendations_path,
            "backtest_agent_priors": agent_priors_path,
            "backtest_setup_performance": setup_performance_path,
            "walk_forward_validation": walk_forward_path,
            "agent_ablation": agent_ablation_path,
            "four_pass_agent_buy": four_pass_agent_buy_path,
            "four_pass_council_buy": four_pass_council_buy_path,
            "four_pass_agent_sell": four_pass_agent_sell_path,
            "four_pass_council_sell": four_pass_council_sell_path,
            "four_pass_final_agent_ratings": four_pass_final_agent_ratings_path,
            "four_pass_profitability_summary": four_pass_profitability_summary_path,
            "four_pass_agent_context_ratings": four_pass_agent_context_ratings_path,
            "four_pass_sell_path_replay": four_pass_sell_path_replay_path,
            "four_pass_purged_walk_forward": four_pass_purged_walk_forward_path,
            "four_pass_product_live_gate": four_pass_product_live_gate_path,
            "product_cooldowns": product_cooldowns_path,
            "feature_store_summary": feature_store_summary_path,
            "agent_decision_influence": agent_decision_influence_path,
            "product_agent_influence": product_agent_influence_path,
            "trade_frequency_estimate": trade_frequency_estimate_path,
            "fifth_pass_live_style_replay": fifth_pass_replay_path,
            "fifth_pass_live_style_summary": fifth_pass_summary_path,
            "fifth_pass_product_contribution": fifth_pass_product_contribution_path,
            "fifth_pass_blockers": fifth_pass_blockers_path,
            "backtest_summary": summary_path,
        },
        "recommendations": buy_recs,
    }

if __name__ == "__main__":
    run_backtest_intelligence(base_dir=os.path.dirname(os.path.abspath(__file__)))
