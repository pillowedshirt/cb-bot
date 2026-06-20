import csv
import math
import os
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
    "score", "reason",
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
    "portfolio_return_pct_100_ref", "score", "reason",
]

FOUR_PASS_FINAL_AGENT_RATINGS_COLUMNS: List[str] = [
    "ts", "dt_utc", "agent",
    "buy_rows", "buy_accuracy", "buy_avg_net_bps", "buy_score", "buy_weight_pct",
    "sell_rows", "sell_accuracy", "sell_avg_net_bps", "sell_score", "sell_weight_pct",
    "reason",
]


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
        return pd.read_csv(path)
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


def _softmax_weights(scored_rows: List[Dict[str, Any]], *, score_key: str, raw_key: str, weight_key: str) -> List[Dict[str, Any]]:
    if not scored_rows:
        return []
    raw_total = 0.0
    for row in scored_rows:
        score = float(row.get(score_key, 0.50) or 0.50)
        samples = float(row.get("selected_count", row.get("sample_count", 0)) or 0)
        sample_factor = max(0.05, min(1.0, samples / 80.0))
        raw = math.exp((score - 0.50) / 0.070) * sample_factor
        row[raw_key] = raw
        raw_total += raw
    raw_total = max(raw_total, 1e-9)
    for row in scored_rows:
        row[weight_key] = float(row[raw_key]) / raw_total * 100.0
    return scored_rows


def _buy_agent_score_columns(frame: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        "volume_profile_leader": "volume_profile_leader_buy_score", "volume_profile_agent": "volume_profile_buy_score",
        "price_action": "price_action_buy_score", "market_structure_agent": "market_structure_buy_score",
        "validated_liquidity_agent": "validated_liquidity_buy_score", "fresh_zone_retest_agent": "fresh_zone_buy_score",
        "fair_value_gap_agent": "fvg_buy_score", "previous_session_volume_profile_agent": "previous_session_profile_buy_score",
        "quant_boundary_agent": "quant_buy_score", "candle_exhaustion_agent": "candle_exhaustion_score",
    }
    return {agent: col for agent, col in candidates.items() if col in frame.columns}


def _sell_agent_score_columns(frame: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        "volume_profile_leader_exit": "volume_profile_leader_sell_score", "volume_profile_harvest": "volume_profile_sell_score",
        "price_action_exit": "price_action_sell_score", "previous_session_profile_exit": "previous_session_profile_sell_score",
        "quant_boundary_exit": "quant_sell_score", "candle_exhaustion_sell": "candle_exhaustion_score",
    }
    return {agent: col for agent, col in candidates.items() if col in frame.columns}


def _build_buy_training_frame(base_dir: str) -> pd.DataFrame:
    frame = _read_csv(os.path.join(base_dir, "candidate_replay.csv"))
    if frame.empty:
        frame = _read_csv(os.path.join(base_dir, "historical_shadow_replay.csv"))
    if frame.empty or "product_id" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    for col in ["score", "probability", "expected_net_edge_bps", "cost_bps", "max_favorable_bps", "max_adverse_bps", "net_pnl_bps", "binance_taker_taker_net_pnl_bps", "synthetic_notional_usd"]:
        if col in frame.columns:
            frame[col] = _numeric(frame, col, 0.0)
    if "buy_net_bps" not in frame.columns:
        if "net_pnl_bps" in frame.columns:
            frame["buy_net_bps"] = _numeric(frame, "net_pnl_bps", 0.0)
        elif "binance_taker_taker_net_pnl_bps" in frame.columns:
            frame["buy_net_bps"] = _numeric(frame, "binance_taker_taker_net_pnl_bps", 0.0)
        else:
            frame["buy_net_bps"] = _numeric(frame, "max_favorable_bps", 0.0) - _numeric(frame, "cost_bps", 0.0)
    frame["buy_success"] = ((frame["buy_net_bps"] > 0.0) | (_bool_series(frame, "survived_to_profit")) | (_bool_series(frame, "reached_min_profit")))
    frame["buy_adverse_bps"] = _numeric(frame, "max_adverse_bps", 0.0).abs()
    return frame


def _four_pass_buy_agent_rows(base_dir: str) -> Tuple[List[List[Any]], Dict[str, float], pd.DataFrame]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); frame = _build_buy_training_frame(base_dir)
    if frame.empty: return [], {}, pd.DataFrame()
    score_cols = _buy_agent_score_columns(frame)
    if not score_cols: return [], {}, frame
    rows_for_weighting: List[Dict[str, Any]] = []
    for agent, col in score_cols.items():
        frame[col] = _numeric(frame, col, 0.0); valid = frame.dropna(subset=[col]).copy()
        if valid.empty: continue
        best = None
        for q in [0.50, 0.60, 0.70, 0.80, 0.88]:
            threshold = float(valid[col].quantile(q)); selected = valid[valid[col] >= threshold].copy(); selected_count = int(len(selected))
            if selected_count < 10: continue
            win_rate = float(selected["buy_success"].mean()); avg_net = float(selected["buy_net_bps"].mean()); median_net = float(selected["buy_net_bps"].median()); avg_adverse = float(selected["buy_adverse_bps"].mean())
            sample_factor = min(1.0, math.sqrt(selected_count / 80.0))
            score = max(0.05, min(0.95, 0.50 + (win_rate - 0.50) * 0.90 * sample_factor + max(-150.0, min(250.0, avg_net)) / 700.0 + max(-150.0, min(250.0, median_net)) / 1000.0 - max(0.0, avg_adverse - 90.0) / 900.0))
            candidate = {"agent": agent, "source_column": col, "sample_count": int(len(valid)), "selected_count": selected_count, "threshold": threshold, "win_rate": win_rate, "avg_net_bps": avg_net, "median_net_bps": median_net, "avg_adverse_bps": avg_adverse, "score": score}
            if best is None or score > float(best["score"]): best = candidate
        if best is not None: rows_for_weighting.append(best)
    rows_for_weighting = _softmax_weights(rows_for_weighting, score_key="score", raw_key="raw_authority", weight_key="buy_weight_pct")
    output_rows: List[List[Any]] = []; weights: Dict[str, float] = {}
    for row in rows_for_weighting:
        weights[str(row["agent"])] = float(row["buy_weight_pct"])
        output_rows.append([f"{ts_value:.6f}", dt_value, "buy_pass_1_agent_only", row["agent"], row["source_column"], int(row["sample_count"]), int(row["selected_count"]), f"{float(row['threshold']):.6f}", f"{float(row['win_rate']):.6f}", f"{float(row['avg_net_bps']):.6f}", f"{float(row['median_net_bps']):.6f}", f"{float(row['avg_adverse_bps']):.6f}", f"{float(row['score']):.6f}", f"{float(row['raw_authority']):.6f}", f"{float(row['buy_weight_pct']):.6f}", f"buy_agent_pass;directional_success=buy_net_bps_gt_0;agent={row['agent']};source={row['source_column']};score={float(row['score']):.4f};weight={float(row['buy_weight_pct']):.2f}"])
    return output_rows, weights, frame


def _four_pass_council_buy_rows(base_dir: str, buy_weights: Dict[str, float], buy_frame: pd.DataFrame) -> Tuple[List[List[Any]], pd.DataFrame]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value)
    if buy_frame is None or buy_frame.empty or not buy_weights: return [], pd.DataFrame()
    frame = buy_frame.copy(); score_cols = _buy_agent_score_columns(frame); weighted_score = pd.Series(0.0, index=frame.index); total_weight = 0.0
    for agent, col in score_cols.items():
        weight = float(buy_weights.get(agent, 0.0) or 0.0)
        if weight <= 0: continue
        normalized = _numeric(frame, col, 0.0)
        if normalized.max() > 1.50: normalized = normalized / 100.0
        weighted_score += normalized.clip(0.0, 1.0) * weight; total_weight += weight
    if total_weight <= 0: return [], pd.DataFrame()
    frame["four_pass_buy_council_score"] = weighted_score / total_weight
    rows: List[List[Any]] = []; base_selected = pd.DataFrame()
    for product_id, group in frame.groupby(frame["product_id"].astype(str)):
        group = group.copy(); best = None
        for q in [0.55, 0.65, 0.75, 0.85, 0.92]:
            threshold = float(group["four_pass_buy_council_score"].quantile(q)); selected = group[group["four_pass_buy_council_score"] >= threshold].copy(); selected_count = int(len(selected))
            if selected_count < 8: continue
            win_rate = float(selected["buy_success"].mean()); avg_net = float(selected["buy_net_bps"].mean()); median_net = float(selected["buy_net_bps"].median()); avg_adverse = float(selected["buy_adverse_bps"].mean()); portfolio_return_pct = float(selected["buy_net_bps"].sum() * 5.0 / 10000.0)
            sample_factor = min(1.0, math.sqrt(selected_count / 80.0)); score = max(0.05, min(0.95, 0.50 + (win_rate - 0.50) * 0.85 * sample_factor + max(-150.0, min(250.0, avg_net)) / 700.0 + max(-150.0, min(250.0, median_net)) / 1000.0 - max(0.0, avg_adverse - 90.0) / 900.0))
            candidate = {"product_id": product_id, "sample_count": int(len(group)), "selected_count": selected_count, "threshold": threshold, "win_rate": win_rate, "avg_net_bps": avg_net, "median_net_bps": median_net, "avg_adverse_bps": avg_adverse, "portfolio_return_pct_100_ref": portfolio_return_pct, "score": score, "selected": selected}
            if best is None or score > float(best["score"]): best = candidate
        if best is None: continue
        selected = best.pop("selected"); base_selected = pd.concat([base_selected, selected], ignore_index=True, sort=False)
        rows.append([f"{ts_value:.6f}", dt_value, "buy_pass_2_weighted_council", best["product_id"], int(best["sample_count"]), int(best["selected_count"]), f"{float(best['threshold']):.6f}", f"{float(best['win_rate']):.6f}", f"{float(best['avg_net_bps']):.6f}", f"{float(best['median_net_bps']):.6f}", f"{float(best['avg_adverse_bps']):.6f}", f"{float(best['portfolio_return_pct_100_ref']):.6f}", f"{float(best['score']):.6f}", f"weighted_council_buy_pass;product={best['product_id']};threshold={float(best['threshold']):.4f};win_rate={float(best['win_rate']):.4f};avg_net={float(best['avg_net_bps']):.2f}"])
    return rows, base_selected


def _build_sell_training_frame(base_dir: str, council_buy_entries: pd.DataFrame) -> pd.DataFrame:
    sell_outcomes = _read_csv(os.path.join(base_dir, "sell_outcomes.csv"))
    if sell_outcomes.empty or "product_id" not in sell_outcomes.columns: return pd.DataFrame()
    sell_outcomes = sell_outcomes.copy(); sell_outcomes["move_after_sell_bps"] = _numeric(sell_outcomes, "move_after_sell_bps", 0.0); sell_outcomes["realized_net_pnl_bps"] = _numeric(sell_outcomes, "realized_net_pnl_bps", 0.0); sell_outcomes["earnings_quality_score"] = _numeric(sell_outcomes, "earnings_quality_score", 0.5)
    sell_outcomes["good_sell_success"] = ((sell_outcomes["move_after_sell_bps"] <= 30.0) | (sell_outcomes["realized_net_pnl_bps"] > 0.0) | (sell_outcomes["earnings_quality_score"] >= 0.55)); sell_outcomes["too_early"] = sell_outcomes["move_after_sell_bps"] >= 80.0
    return sell_outcomes


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
            candidate = {"product_id": product_id, "sample_count": int(len(group)), "selected_count": selected_count, "threshold": threshold, "good_exit_rate": good_exit_rate, "too_early_rate": too_early_rate, "avg_move_after_sell_bps": avg_move_after, "avg_realized_net_bps": avg_realized, "portfolio_return_pct_100_ref": portfolio_return_pct, "score": score}
            if best is None or score > float(best["score"]): best = candidate
        if best is None: continue
        rows.append([f"{ts_value:.6f}", dt_value, "sell_pass_2_weighted_council", best["product_id"], int(best["sample_count"]), int(best["selected_count"]), f"{float(best['threshold']):.6f}", f"{float(best['good_exit_rate']):.6f}", f"{float(best['too_early_rate']):.6f}", f"{float(best['avg_move_after_sell_bps']):.6f}", f"{float(best['avg_realized_net_bps']):.6f}", f"{float(best['portfolio_return_pct_100_ref']):.6f}", f"{float(best['score']):.6f}", f"weighted_council_sell_pass;product={best['product_id']};threshold={float(best['threshold']):.4f};good_exit_rate={float(best['good_exit_rate']):.4f};too_early={float(best['too_early_rate']):.4f}"])
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
        rows.append([f"{ts_value:.6f}", dt_value, agent, int(buy.get("rows", 0)), f"{float(buy.get('accuracy', 0.50)):.6f}", f"{float(buy.get('avg', 0.0)):.6f}", f"{float(buy.get('score', 0.50)):.6f}", f"{float(buy.get('weight', 0.0)):.6f}", int(sell.get("rows", 0)), f"{float(sell.get('accuracy', 0.50)):.6f}", f"{float(sell.get('avg', 0.0)):.6f}", f"{float(sell.get('score', 0.50)):.6f}", f"{float(sell.get('weight', 0.0)):.6f}", "four_pass_final_agent_rating;buy_and_sell_weights_are_side_specific"])
    return rows


def _four_pass_backtest_outputs(base_dir: str) -> Dict[str, Any]:
    buy_agent_rows, buy_weights, buy_frame = _four_pass_buy_agent_rows(base_dir)
    council_buy_rows, council_buy_entries = _four_pass_council_buy_rows(base_dir, buy_weights, buy_frame)
    sell_agent_rows, sell_weights, sell_frame = _four_pass_sell_agent_rows(base_dir, council_buy_entries)
    council_sell_rows = _four_pass_council_sell_rows(base_dir, sell_weights, sell_frame)
    final_rating_rows = _four_pass_final_agent_rating_rows(buy_agent_rows, sell_agent_rows)
    return {"buy_agent_rows": buy_agent_rows, "buy_weights": buy_weights, "council_buy_rows": council_buy_rows, "council_buy_entries": council_buy_entries, "sell_agent_rows": sell_agent_rows, "sell_weights": sell_weights, "council_sell_rows": council_sell_rows, "final_rating_rows": final_rating_rows}


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

    ts_value = _utc_ts()
    summary_rows = [
        [f"{ts_value:.6f}", _utc_dt(ts_value), "buy_recommendation_rows", len(buy_rows), "product-level candidate replay recommendations"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "sell_recommendation_rows", len(sell_rows), "product-level sell replay recommendations"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "agent_prior_rows", len(agent_rows), "profit-weighted agent priors"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "setup_performance_rows", len(setup_rows), "product/session/setup-specific profit replay"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "walk_forward_validation_rows", len(walk_forward_rows), "out-of-sample validation rows" if walk_forward_rows else "not enough reviewed outcomes yet"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "agent_ablation_rows", len(ablation_rows), "agent marginal contribution report" if ablation_rows else "not enough reviewed outcomes yet"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_agent_buy_rows", len(four_pass["buy_agent_rows"]), "pass 1 buy timing by individual agent"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_council_buy_rows", len(four_pass["council_buy_rows"]), "pass 2 buy timing by weighted council"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_agent_sell_rows", len(four_pass["sell_agent_rows"]), "pass 3 sell timing by individual agent"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "four_pass_council_sell_rows", len(four_pass["council_sell_rows"]), "pass 4 sell timing by weighted council"],
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
            "backtest_summary": summary_path,
        },
        "recommendations": buy_recs,
    }

if __name__ == "__main__":
    run_backtest_intelligence(base_dir=os.path.dirname(os.path.abspath(__file__)))
