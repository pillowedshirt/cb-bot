import csv
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
        if sample_count < 12:
            continue

        win_rate = float(group["net_success"].mean())
        avg_net = float(group["net_peak_bps"].mean())
        avg_adverse = float(group["max_adverse_bps"].abs().mean())
        objective = win_rate * 100.0 + avg_net * 0.35 - avg_adverse * 0.20

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
            "product_session_setup_profit_replay",
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
    base_dir = os.path.abspath(base_dir)
    buy_rows, buy_recs = _candidate_rows(base_dir, min_product_rows=int(min_product_rows))
    sell_rows = _sell_recommendation_rows(base_dir)
    agent_rows = _agent_prior_rows(base_dir)
    setup_rows = _setup_performance_rows(base_dir)

    recommendations_path = os.path.join(base_dir, "backtest_recommendations.csv")
    sell_recommendations_path = os.path.join(base_dir, "backtest_sell_recommendations.csv")
    agent_priors_path = os.path.join(base_dir, "backtest_agent_priors.csv")
    setup_performance_path = os.path.join(base_dir, "backtest_setup_performance.csv")
    summary_path = os.path.join(base_dir, "backtest_summary.csv")

    _write_rows(recommendations_path, BACKTEST_RECOMMENDATIONS_COLUMNS, buy_rows)
    _write_rows(sell_recommendations_path, BACKTEST_SELL_RECOMMENDATIONS_COLUMNS, sell_rows)
    _write_rows(agent_priors_path, BACKTEST_AGENT_PRIOR_COLUMNS, agent_rows)
    _write_rows(setup_performance_path, BACKTEST_SETUP_PERFORMANCE_COLUMNS, setup_rows)

    ts_value = _utc_ts()
    summary_rows = [
        [f"{ts_value:.6f}", _utc_dt(ts_value), "buy_recommendation_rows", len(buy_rows), "product-level candidate replay recommendations"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "sell_recommendation_rows", len(sell_rows), "product-level sell replay recommendations"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "agent_prior_rows", len(agent_rows), "profit-weighted agent priors"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "setup_performance_rows", len(setup_rows), "product/session/setup-specific profit replay"],
        [f"{ts_value:.6f}", _utc_dt(ts_value), "runtime_seconds", f"{time.time() - started:.3f}", "backtest intelligence runtime"],
    ]
    _write_rows(summary_path, BACKTEST_SUMMARY_COLUMNS, summary_rows)

    log(
        f"[backtest] completed buy_recs={len(buy_rows)} sell_recs={len(sell_rows)} "
        f"agent_priors={len(agent_rows)} setup_performance={len(setup_rows)} seconds={time.time() - started:.2f}"
    )
    return {
        "buy_recommendations": len(buy_rows),
        "sell_recommendations": len(sell_rows),
        "agent_priors": len(agent_rows),
        "setup_performance": len(setup_rows),
        "runtime_seconds": time.time() - started,
        "paths": {
            "backtest_recommendations": recommendations_path,
            "backtest_sell_recommendations": sell_recommendations_path,
            "backtest_agent_priors": agent_priors_path,
            "backtest_setup_performance": setup_performance_path,
            "backtest_summary": summary_path,
        },
        "recommendations": buy_recs,
    }

if __name__ == "__main__":
    run_backtest_intelligence(base_dir=os.path.dirname(os.path.abspath(__file__)))
