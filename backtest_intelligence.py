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
    "recommended_min_score", "recommended_min_probability",
    "recommended_min_expected_value_bps", "recommended_min_projected_net_bps",
    "recommended_forward_window_minutes", "expected_win_rate", "expected_net_bps",
    "expected_adverse_bps", "objective_score", "source", "reason",
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


def _candidate_rows(base_dir: str, *, min_product_rows: int) -> Tuple[List[List[Any]], Dict[str, Dict[str, Any]]]:
    frame = _read_csv(os.path.join(base_dir, "candidate_replay.csv"))
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value)
    rows: List[List[Any]] = []; recs: Dict[str, Dict[str, Any]] = {}
    if frame.empty or "product_id" not in frame.columns:
        return rows, recs
    required_numeric = ["score", "probability", "expected_net_edge_bps", "cost_bps", "max_favorable_bps", "max_adverse_bps", "adverse_before_profit_bps", "time_to_min_profit_minutes", "forward_window_minutes", "post_profit_extra_gain_bps"]
    for column in required_numeric:
        frame[column] = _numeric(frame, column, 0.0)
    frame["reached_min_profit_bool"] = _bool_series(frame, "reached_min_profit")
    frame["survived_to_profit_bool"] = _bool_series(frame, "survived_to_profit")
    frame["net_peak_bps"] = frame["max_favorable_bps"] - frame["cost_bps"]
    frame["net_success"] = frame["survived_to_profit_bool"] | frame["reached_min_profit_bool"] | (frame["net_peak_bps"] >= 45.0)
    for product_id, group in frame.groupby(frame["product_id"].astype(str)):
        group = group.copy(); sample_count = int(len(group))
        if sample_count < int(min_product_rows):
            continue
        score_values = group["score"].dropna(); prob_values = group["probability"].dropna(); ev_values = group["expected_net_edge_bps"].dropna()
        if score_values.empty or prob_values.empty:
            continue
        best: Optional[Dict[str, Any]] = None
        for sq in [0.45, 0.55, 0.65, 0.75, 0.85]:
            score_cut = float(score_values.quantile(sq))
            for pq in [0.40, 0.50, 0.60, 0.70, 0.80]:
                prob_cut = float(prob_values.quantile(pq))
                for eq in [0.35, 0.50, 0.65, 0.80]:
                    ev_cut = float(ev_values.quantile(eq))
                    accepted = group[(group["score"] >= score_cut) & (group["probability"] >= prob_cut) & (group["expected_net_edge_bps"] >= ev_cut)].copy()
                    accepted_count = int(len(accepted))
                    if accepted_count < max(8, int(min_product_rows * 0.04)):
                        continue
                    win_rate = float(accepted["net_success"].mean())
                    avg_net = float(accepted["net_peak_bps"].mean())
                    median_net = float(accepted["net_peak_bps"].median())
                    avg_adverse = float(accepted["adverse_before_profit_bps"].abs().mean())
                    avg_post_extra = float(accepted["post_profit_extra_gain_bps"].mean())
                    clean_time = accepted["time_to_min_profit_minutes"].replace([np.inf, -np.inf], np.nan).dropna()
                    median_time = float(clean_time.median()) if not clean_time.empty else 240.0
                    sample_quality = min(1.0, accepted_count / max(25.0, min_product_rows * 0.15))
                    objective = win_rate * 100.0 + avg_net * 0.42 + median_net * 0.20 + avg_post_extra * 0.16 - avg_adverse * 0.24 - max(0.0, median_time - 180.0) * 0.035 + sample_quality * 15.0
                    candidate = {"score_cut": score_cut, "prob_cut": prob_cut, "ev_cut": ev_cut, "accepted_count": accepted_count, "win_rate": win_rate, "avg_net": avg_net, "median_net": median_net, "avg_adverse": avg_adverse, "median_time": median_time, "objective": objective}
                    if best is None or objective > float(best["objective"]):
                        best = candidate
        if best is None:
            continue
        recommended_min_projected_net_bps = max(45.0, min(180.0, float(best["avg_net"]) * 0.35 + 45.0))
        recommended_forward_window_minutes = max(15.0, min(360.0, float(best["median_time"]) if math.isfinite(float(best["median_time"])) else 240.0))
        reason = f"candidate_replay_grid_search;sample_count={sample_count};accepted_count={int(best['accepted_count'])};win_rate={float(best['win_rate']):.4f};avg_net={float(best['avg_net']):.2f};avg_adverse={float(best['avg_adverse']):.2f};objective={float(best['objective']):.4f}"
        rows.append([f"{ts_value:.6f}", dt_value, product_id, sample_count, int(best["accepted_count"]), f"{float(best['score_cut']):.6f}", f"{float(best['prob_cut']):.6f}", f"{float(best['ev_cut']):.6f}", f"{float(recommended_min_projected_net_bps):.6f}", f"{float(recommended_forward_window_minutes):.6f}", f"{float(best['win_rate']):.6f}", f"{float(best['avg_net']):.6f}", f"{float(best['avg_adverse']):.6f}", f"{float(best['objective']):.6f}", "candidate_replay.csv", reason])
        recs[product_id] = {"product_id": product_id, "recommended_min_score": float(best["score_cut"]), "recommended_min_probability": float(best["prob_cut"]), "recommended_min_expected_value_bps": float(best["ev_cut"]), "recommended_min_projected_net_bps": float(recommended_min_projected_net_bps), "recommended_forward_window_minutes": float(recommended_forward_window_minutes), "expected_win_rate": float(best["win_rate"]), "expected_net_bps": float(best["avg_net"]), "expected_adverse_bps": float(best["avg_adverse"]), "objective_score": float(best["objective"])}
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
    perf = _read_csv(os.path.join(base_dir, "agent_performance.csv")); ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); rows: List[List[Any]] = []
    if perf.empty or not {"product_id", "strategy", "agent"}.issubset(perf.columns):
        return rows
    perf["weighted_agent_credit_score"] = _numeric(perf, "weighted_agent_credit_score", 0.5); perf["agent_credit_score"] = _numeric(perf, "agent_credit_score", 0.5); perf["outcome_move_bps"] = _numeric(perf, "outcome_move_bps", 0.0); perf["confidence"] = _numeric(perf, "confidence", 0.6); perf["reliability"] = _numeric(perf, "reliability", 0.5)
    if "outcome_source" not in perf.columns:
        perf["outcome_source"] = "agent_performance"
    source_weight = perf["outcome_source"].astype(str).map({"real_trade": 1.30, "trade_outcome": 1.30, "sell_outcome": 1.25, "agent_performance": 0.80, "level8_observation": 0.45, "observation_outcome": 0.35}).fillna(0.50).astype(float)
    perf["profit_replay_credit"] = ((perf["weighted_agent_credit_score"] * source_weight) + (perf["agent_credit_score"] * 0.25) + ((perf["outcome_move_bps"].clip(-250.0, 350.0) + 250.0) / 600.0 * 0.15)) / (source_weight + 0.40)
    perf["profit_replay_credit"] = perf["profit_replay_credit"].clip(0.0, 1.0)
    for (product_id, strategy, agent), group in perf.groupby(["product_id", "strategy", "agent"]):
        sample_count = int(len(group))
        if sample_count < 5:
            continue
        recent = group.tail(250).copy(); credit = float(recent["profit_replay_credit"].mean()); move = float(recent["outcome_move_bps"].mean()); confidence = float(recent["confidence"].mean()); reliability = float(recent["reliability"].mean()); success = 1 if credit >= 0.50 else 0
        if str(strategy).upper() == "EXIT_REVIEW":
            buy_score = 0.0; sell_score = credit; hold_score = 1.0 - max(0.0, credit - 0.50) * 0.75; wait_score = 0.25
        else:
            buy_score = credit; sell_score = 1.0 - credit; hold_score = 0.50; wait_score = 0.35
        rows.append([f"{ts_value:.6f}", dt_value, f"backtest-prior-{product_id}-{strategy}-{agent}-{int(ts_value)}", str(product_id), str(strategy), str(agent), f"{buy_score:.6f}", f"{sell_score:.6f}", f"{hold_score:.6f}", f"{wait_score:.6f}", f"{max(0.50, min(0.85, confidence)):.6f}", f"{max(0.35, min(0.85, reliability)):.6f}", "1.000000", "", "", "0.000000", "0.000000", "backtest_profit_replay", "1.250000", "0", f"{move:.6f}", "profit_replay_prior", f"{credit:.6f}", f"{credit:.6f}", success, f"backtest_profit_prior;samples={sample_count};recent_samples={len(recent)};credit={credit:.6f};avg_move_bps={move:.2f};source=agent_performance.csv"])
    return rows


def run_backtest_intelligence(*, base_dir: str, log_fn: Optional[Callable[[str], None]] = None, min_product_rows: int = 80) -> Dict[str, Any]:
    """Run CSV replay intelligence and write backtest recommendation CSVs."""
    def log(message: str) -> None:
        if log_fn is not None:
            try:
                log_fn(message); return
            except Exception:
                pass
        print(message)
    started = time.time(); base_dir = os.path.abspath(base_dir)
    buy_rows, buy_recs = _candidate_rows(base_dir, min_product_rows=int(min_product_rows)); sell_rows = _sell_recommendation_rows(base_dir); agent_rows = _agent_prior_rows(base_dir)
    recommendations_path = os.path.join(base_dir, "backtest_recommendations.csv"); sell_recommendations_path = os.path.join(base_dir, "backtest_sell_recommendations.csv"); agent_priors_path = os.path.join(base_dir, "backtest_agent_priors.csv"); summary_path = os.path.join(base_dir, "backtest_summary.csv")
    _write_rows(recommendations_path, BACKTEST_RECOMMENDATIONS_COLUMNS, buy_rows); _write_rows(sell_recommendations_path, BACKTEST_SELL_RECOMMENDATIONS_COLUMNS, sell_rows); _write_rows(agent_priors_path, BACKTEST_AGENT_PRIOR_COLUMNS, agent_rows)
    ts_value = _utc_ts()
    summary_rows = [[f"{ts_value:.6f}", _utc_dt(ts_value), "buy_recommendation_rows", len(buy_rows), "product-level candidate replay recommendations"], [f"{ts_value:.6f}", _utc_dt(ts_value), "sell_recommendation_rows", len(sell_rows), "product-level sell replay recommendations"], [f"{ts_value:.6f}", _utc_dt(ts_value), "agent_prior_rows", len(agent_rows), "profit-weighted agent priors"], [f"{ts_value:.6f}", _utc_dt(ts_value), "runtime_seconds", f"{time.time() - started:.3f}", "backtest intelligence runtime"]]
    _write_rows(summary_path, BACKTEST_SUMMARY_COLUMNS, summary_rows)
    log(f"[backtest] completed buy_recs={len(buy_rows)} sell_recs={len(sell_rows)} agent_priors={len(agent_rows)} seconds={time.time() - started:.2f}")
    return {"buy_recommendations": len(buy_rows), "sell_recommendations": len(sell_rows), "agent_priors": len(agent_rows), "runtime_seconds": time.time() - started, "paths": {"backtest_recommendations": recommendations_path, "backtest_sell_recommendations": sell_recommendations_path, "backtest_agent_priors": agent_priors_path, "backtest_summary": summary_path}, "recommendations": buy_recs}


if __name__ == "__main__":
    run_backtest_intelligence(base_dir=os.path.dirname(os.path.abspath(__file__)))
