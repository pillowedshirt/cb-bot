import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


RISK_EV_CONFIDENCE_COLUMNS = [
    "ts", "dt_utc",
    "scope", "product_id", "market_regime", "context_key",
    "sample_count",
    "raw_win_rate",
    "ev_mean_bps",
    "ev_median_bps",
    "ev_ci_low_bps",
    "ev_ci_high_bps",
    "prob_ev_positive",
    "avg_win_bps",
    "avg_loss_bps",
    "confidence_grade",
    "recommended_action",
    "size_multiplier",
    "reason",
]

RISK_MONTE_CARLO_COLUMNS = [
    "ts", "dt_utc", "scope", "product_id", "horizon_days", "trials",
    "sample_count", "position_size_pct", "median_return_pct", "p05_return_pct",
    "p95_return_pct", "prob_loss", "median_max_drawdown_pct",
    "p95_max_drawdown_pct", "prob_drawdown_gt_3pct", "prob_drawdown_gt_5pct",
    "risk_grade", "reason",
]

RISK_CONTEXT_PERFORMANCE_COLUMNS = [
    "ts", "dt_utc", "product_id", "market_regime", "context_key",
    "sample_count", "raw_win_rate", "ev_mean_bps", "ev_ci_low_bps",
    "prob_ev_positive", "avg_adverse_bps", "context_grade",
    "context_live_allowed", "context_size_multiplier", "reason",
]

RISK_LIVE_GATE_COLUMNS = [
    "ts", "dt_utc", "product_id", "sample_count", "ev_mean_bps",
    "ev_ci_low_bps", "prob_ev_positive", "p95_max_drawdown_pct_30d",
    "prob_loss_7d", "prob_drawdown_gt_3pct_30d", "risk_grade",
    "live_allowed", "size_multiplier", "reason",
]

SOURCE_QUALITY_WEIGHTS: Dict[str, float] = {
    "live_realized": 1.00,
    "fifth_pass_live_style_replay": 0.90,
    "council_observed": 0.80,
    "historical_shadow_replay": 0.75,
    "fixed_window_trade_outcome": 0.70,
    "candidate_replay_proxy": 0.45,
}


def _utc_ts() -> float:
    return float(time.time())


def _utc_dt(ts_value: Optional[float] = None) -> str:
    ts = _utc_ts() if ts_value is None else float(ts_value)
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


GENERATED_FILE_VERSION = "fast_startup_calc_v1_2026_06_21"


def _sidecar_meta_path(path: str) -> str:
    return f"{path}.meta.json"


def _write_generated_file_meta(path: str, *, reason: str = "") -> None:
    try:
        meta = {
            "generation_version": GENERATED_FILE_VERSION,
            "generated_at_ts": time.time(),
            "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            "reason": str(reason or ""),
        }
        with open(_sidecar_meta_path(path), "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2, sort_keys=True)
    except Exception:
        pass


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_rows(path: str, columns: List[str], rows: List[List[Any]]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)
    os.replace(tmp_path, path)
    _write_generated_file_meta(path, reason="risk_intelligence_regenerated")


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if frame is None or frame.empty or column not in frame.columns:
        return pd.Series([default] * (0 if frame is None else len(frame)), dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame is None or frame.empty or column not in frame.columns:
        return pd.Series([False] * (0 if frame is None else len(frame)), dtype="bool")
    values = frame[column].astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes", "y"})


def _normalize_context_part(value: Any, default: str) -> str:
    text = str(value if value not in (None, "") else default)
    text = text.strip().lower().replace(" ", "_")
    return text or default


def _regime_from_mapping(mapping: Dict[str, Any]) -> str:
    return (
        str(
            mapping.get("market_regime")
            or mapping.get("regime_tag")
            or mapping.get("quant_volatility_cluster_state")
            or mapping.get("volatility_cluster")
            or mapping.get("trend_regime")
            or "unknown_regime"
        )
        .strip()
        .lower()
        .replace(" ", "_")
    )


def context_key_from_mapping(mapping: Dict[str, Any]) -> str:
    regime = _normalize_context_part(_regime_from_mapping(mapping), "unknown_regime")
    session = _normalize_context_part(mapping.get("session_liquidity_setup"), "unknown_session")
    structure = _normalize_context_part(mapping.get("structure_state"), "unknown_structure")
    value_area = _normalize_context_part(
        mapping.get("value_area_state") or mapping.get("value_acceptance_state"),
        "unknown_value",
    )
    fvg = _normalize_context_part(mapping.get("fvg_state"), "unknown_fvg")
    volume_node = _normalize_context_part(mapping.get("volume_node_state"), "unknown_volume_node")
    quant = _normalize_context_part(mapping.get("quant_boundary_state"), "unknown_quant")

    return "|".join([regime, session, structure, value_area, fvg, volume_node, quant])


def context_lookup_keys_from_mapping(product_id: str, mapping: Dict[str, Any]) -> List[str]:
    full = context_key_from_mapping(mapping)
    parts = full.split("|")

    while len(parts) < 7:
        parts.append("unknown")

    regime, session, structure, _value_area, _fvg, _volume_node, quant = parts[:7]
    product_id = str(product_id or mapping.get("product_id", "")).strip()

    return [
        f"{product_id}||full||{full}",
        f"{product_id}||regime_session_structure||{regime}|{session}|{structure}",
        f"{product_id}||regime_quant||{regime}|{quant}",
        f"{product_id}||regime_only||{regime}",
        f"ALL||regime_only||{regime}",
    ]


def _infer_day_key(frame: pd.DataFrame) -> pd.Series:
    for col in ["entry_ts", "replay_ts", "ts"]:
        if col in frame.columns:
            ts = pd.to_numeric(frame[col], errors="coerce")
            return (ts // 86400).fillna(0).astype(int).astype(str)
    return pd.Series(["0"] * len(frame), index=frame.index)


def _extract_outcome_bps(frame: pd.DataFrame, source_name: str) -> pd.Series:
    """
    Return the best available realized/proxy outcome in bps.

    Important:
    - move_bps / net_pnl fields are outcomes.
    - ev_at_entry / expected_net_edge_bps are predictions.
    - Predictions may be used only for explicit proxy/replay sources, never as proof
      of realized live performance.
    """
    realized_cols = [
        "realized_or_proxy_net_bps",
        "binance_taker_taker_net_pnl_bps",
        "binance_maker_taker_net_pnl_bps",
        "net_pnl_bps",
        "buy_net_bps",
        "realized_net_pnl_bps",
        "primary_net_pnl_bps",
        "move_bps",
        "outcome_bps",
    ]

    for col in realized_cols:
        if col in frame.columns:
            return _numeric(frame, col, 0.0)

    if "max_favorable_bps" in frame.columns and "max_adverse_bps" in frame.columns:
        max_fav = _numeric(frame, "max_favorable_bps", 0.0)
        max_adv = _numeric(frame, "max_adverse_bps", 0.0).abs()
        cost = _numeric(frame, "cost_bps", 0.0)

        success = (
            _bool_series(frame, "survived_to_profit")
            | _bool_series(frame, "reached_min_profit")
            | ((max_fav - cost) > 0.0)
        )

        return pd.Series(
            np.where(success, max_fav - cost, -(max_adv + cost * 0.25)),
            index=frame.index,
        )

    # Do not use prediction fields like expected_net_edge_bps or ev_at_entry
    # as proof of outcome. If a row has no actual forward/result field, drop it.
    return pd.Series([np.nan] * len(frame), index=frame.index)


def _standardize_source_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if frame is None or frame.empty or "product_id" not in frame.columns:
        return pd.DataFrame()

    out = frame.copy()
    out["product_id"] = out["product_id"].astype(str)
    out["outcome_bps"] = _extract_outcome_bps(out, source_name)
    out["day_key"] = _infer_day_key(out)

    out["market_regime"] = out.apply(lambda row: _regime_from_mapping(row.to_dict()), axis=1)
    out["context_key"] = out.apply(lambda row: context_key_from_mapping(row.to_dict()), axis=1)
    out["source_name"] = source_name
    out["source_weight"] = float(SOURCE_QUALITY_WEIGHTS.get(source_name, 0.50))

    if "max_adverse_bps" in out.columns:
        out["max_adverse_bps"] = _numeric(out, "max_adverse_bps", 0.0).abs()
    else:
        out["max_adverse_bps"] = 0.0

    keep = [
        "product_id",
        "outcome_bps",
        "day_key",
        "market_regime",
        "context_key",
        "source_name",
        "source_weight",
        "max_adverse_bps",
    ]

    out = out[keep].copy()
    out["outcome_bps"] = pd.to_numeric(out["outcome_bps"], errors="coerce")
    out["source_weight"] = pd.to_numeric(out["source_weight"], errors="coerce").fillna(0.50)
    out = out.dropna(subset=["outcome_bps"])

    return out[np.isfinite(out["outcome_bps"])]


def _training_frame(base_dir: str) -> pd.DataFrame:
    frames = []
    sources = [
        ("fifth_pass_live_style_replay.csv", "fifth_pass_live_style_replay"),
        ("historical_shadow_replay.csv", "historical_shadow_replay"),
        ("candidate_replay.csv", "candidate_replay_proxy"),
        ("council_observation_outcomes.csv", "council_observed"),
        ("trade_outcomes.csv", "fixed_window_trade_outcome"),
    ]
    for filename, source_name in sources:
        frame = _standardize_source_frame(_read_csv(os.path.join(base_dir, filename)), source_name)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["outcome_bps"] = pd.to_numeric(combined["outcome_bps"], errors="coerce").fillna(0.0)
    combined["source_weight"] = pd.to_numeric(combined["source_weight"], errors="coerce").fillna(0.50)
    return combined[combined["product_id"].astype(str).str.len() > 0].copy()

def _bootstrap_ev_stats(values: pd.Series, *, weights: Optional[pd.Series] = None, trials: int = 2000, seed: int = 7) -> Dict[str, float]:
    local = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "weight": pd.to_numeric(weights, errors="coerce") if weights is not None else 1.0}).dropna()
    if local.empty:
        return {"sample_count": 0.0, "raw_win_rate": 0.0, "ev_mean_bps": 0.0, "ev_median_bps": 0.0, "ev_ci_low_bps": 0.0, "ev_ci_high_bps": 0.0, "prob_ev_positive": 0.0, "avg_win_bps": 0.0, "avg_loss_bps": 0.0}
    local = local[np.isfinite(local["value"])]
    vals = local["value"].to_numpy(dtype=float); w = local["weight"].clip(lower=0.05).to_numpy(dtype=float)
    if len(vals) <= 0:
        return {"sample_count": 0.0, "raw_win_rate": 0.0, "ev_mean_bps": 0.0, "ev_median_bps": 0.0, "ev_ci_low_bps": 0.0, "ev_ci_high_bps": 0.0, "prob_ev_positive": 0.0, "avg_win_bps": 0.0, "avg_loss_bps": 0.0}
    probability = w / max(float(w.sum()), 1e-12); rng = np.random.default_rng(seed); n = len(vals); means = np.empty(int(trials), dtype=float)
    for i in range(int(trials)):
        idx = rng.choice(np.arange(n), size=n, replace=True, p=probability)
        means[i] = float(np.average(vals[idx], weights=w[idx]))
    wins = vals[vals > 0.0]; losses = vals[vals <= 0.0]
    return {"sample_count": float(n), "raw_win_rate": float(np.mean(vals > 0.0)), "ev_mean_bps": float(np.average(vals, weights=w)), "ev_median_bps": float(np.median(vals)), "ev_ci_low_bps": float(np.percentile(means, 5)), "ev_ci_high_bps": float(np.percentile(means, 95)), "prob_ev_positive": float(np.mean(means > 0.0)), "avg_win_bps": float(np.mean(wins)) if len(wins) else 0.0, "avg_loss_bps": float(abs(np.mean(losses))) if len(losses) else 0.0}


def _monte_carlo_path_stats(frame: pd.DataFrame, *, horizon_days: int, trials: int, position_size_pct: float, seed: int) -> Dict[str, float]:
    if frame is None or frame.empty:
        return {"median_return_pct": 0.0, "p05_return_pct": 0.0, "p95_return_pct": 0.0, "prob_loss": 1.0, "median_max_drawdown_pct": 0.0, "p95_max_drawdown_pct": 0.0, "prob_drawdown_gt_3pct": 0.0, "prob_drawdown_gt_5pct": 0.0}
    local = frame.copy()
    local["outcome_bps"] = pd.to_numeric(local["outcome_bps"], errors="coerce").fillna(0.0)
    local["source_weight"] = pd.to_numeric(local.get("source_weight", 0.50), errors="coerce").fillna(0.50)
    daily_blocks = []; daily_weights = []
    for _, group in local.groupby("day_key"):
        if len(group) <= 0: continue
        daily_blocks.append(list(group["outcome_bps"].astype(float).values)); daily_weights.append(float(group["source_weight"].mean()))
    if not daily_blocks:
        daily_blocks = [list(local["outcome_bps"].astype(float).values)]; daily_weights = [float(local["source_weight"].mean())]
    daily_weights_np = np.clip(np.asarray(daily_weights, dtype=float), 0.05, None); daily_prob = daily_weights_np / max(float(daily_weights_np.sum()), 1e-12)
    rng = np.random.default_rng(seed); terminal_returns = []; max_drawdowns = []
    for _ in range(int(trials)):
        equity_pct = peak_pct = max_dd = 0.0
        for _day in range(int(horizon_days)):
            idx = int(rng.choice(np.arange(len(daily_blocks)), p=daily_prob)); block = daily_blocks[idx]
            day_return_pct = float(np.sum(block)) * float(position_size_pct) / 100.0
            equity_pct += day_return_pct; peak_pct = max(peak_pct, equity_pct); max_dd = max(max_dd, peak_pct - equity_pct)
        terminal_returns.append(equity_pct); max_drawdowns.append(max_dd)
    terminal = np.asarray(terminal_returns, dtype=float); drawdowns = np.asarray(max_drawdowns, dtype=float)
    return {"median_return_pct": float(np.percentile(terminal, 50)), "p05_return_pct": float(np.percentile(terminal, 5)), "p95_return_pct": float(np.percentile(terminal, 95)), "prob_loss": float(np.mean(terminal < 0.0)), "median_max_drawdown_pct": float(np.percentile(drawdowns, 50)), "p95_max_drawdown_pct": float(np.percentile(drawdowns, 95)), "prob_drawdown_gt_3pct": float(np.mean(drawdowns > 3.0)), "prob_drawdown_gt_5pct": float(np.mean(drawdowns > 5.0))}

def _confidence_grade(stats: Dict[str, float]) -> Tuple[str, str, float]:
    n = float(stats.get("sample_count", 0.0) or 0.0)
    ev = float(stats.get("ev_mean_bps", 0.0) or 0.0)
    ci_low = float(stats.get("ev_ci_low_bps", 0.0) or 0.0)
    prob_pos = float(stats.get("prob_ev_positive", 0.0) or 0.0)

    if n < 30:
        return "INSUFFICIENT_SAMPLE", "shadow_or_min_size", 0.50

    if ev > 0.0 and ci_low > 0.0 and prob_pos >= 0.75:
        return "STRONG_POSITIVE_EV", "allow_normal_or_larger_size", 1.10

    if ev > 0.0 and ci_low > -5.0 and prob_pos >= 0.62:
        return "WEAK_POSITIVE_EV", "allow_reduced_size", 0.75

    if ev > -3.0 and prob_pos >= 0.52:
        return "UNCERTAIN_EV", "allow_min_size_only", 0.50

    return "NEGATIVE_OR_UNRELIABLE_EV", "block_or_shadow", 0.0


def _risk_grade_from_mc(stats: Dict[str, float]) -> str:
    p_loss = float(stats.get("prob_loss", 1.0)); dd95 = float(stats.get("p95_max_drawdown_pct", 99.0)); p_dd3 = float(stats.get("prob_drawdown_gt_3pct", 1.0))
    if dd95 <= 1.5 and p_loss <= 0.35 and p_dd3 <= 0.10: return "LOW_PATH_RISK"
    if dd95 <= 3.0 and p_loss <= 0.45 and p_dd3 <= 0.30: return "MODERATE_PATH_RISK"
    if dd95 <= 5.0 and p_loss <= 0.55: return "ELEVATED_PATH_RISK"
    return "HIGH_PATH_RISK"


def _latest_product_mc(mc_rows: List[List[Any]], product_id: str, horizon_days: int) -> Optional[Dict[str, Any]]:
    for row in reversed(mc_rows):
        if str(row[3]) == str(product_id) and int(row[4]) == int(horizon_days):
            return {"p95_max_drawdown_pct": float(row[13]), "prob_loss": float(row[11]), "prob_drawdown_gt_3pct": float(row[14]), "risk_grade": str(row[16])}
    return None


def run_risk_intelligence(*, base_dir: str, log_fn=None, bootstrap_trials: int = 2000, monte_carlo_trials: int = 5000, position_size_pct: float = 0.10) -> Dict[str, Any]:
    def log(msg: str) -> None:
        if log_fn is not None:
            try: log_fn(msg); return
            except Exception: pass
        print(msg)
    base_dir = os.path.abspath(base_dir)
    frame = _training_frame(base_dir)
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value)
    ev_rows: List[List[Any]] = []; mc_rows: List[List[Any]] = []; context_rows: List[List[Any]] = []; live_gate_rows: List[List[Any]] = []
    if frame.empty:
        _write_rows(os.path.join(base_dir, "risk_ev_confidence.csv"), RISK_EV_CONFIDENCE_COLUMNS, [])
        _write_rows(os.path.join(base_dir, "risk_monte_carlo_summary.csv"), RISK_MONTE_CARLO_COLUMNS, [])
        _write_rows(os.path.join(base_dir, "risk_context_performance.csv"), RISK_CONTEXT_PERFORMANCE_COLUMNS, [])
        _write_rows(os.path.join(base_dir, "risk_live_gate.csv"), RISK_LIVE_GATE_COLUMNS, [])
        return {"rows": 0, "reason": "no_training_frame"}
    product_groups = [("ALL", frame)] + [(str(pid), group.copy()) for pid, group in frame.groupby("product_id")]
    product_ev_map: Dict[str, Dict[str, float]] = {}
    for product_id, group in product_groups:
        stats = _bootstrap_ev_stats(group["outcome_bps"], weights=group.get("source_weight"), trials=int(bootstrap_trials), seed=11)
        grade, action, size_mult = _confidence_grade(stats); product_ev_map[product_id] = stats
        ev_rows.append([f"{ts_value:.6f}", dt_value, "product" if product_id != "ALL" else "portfolio", product_id, "ALL", "ALL", int(stats["sample_count"]), f"{stats['raw_win_rate']:.6f}", f"{stats['ev_mean_bps']:.6f}", f"{stats['ev_median_bps']:.6f}", f"{stats['ev_ci_low_bps']:.6f}", f"{stats['ev_ci_high_bps']:.6f}", f"{stats['prob_ev_positive']:.6f}", f"{stats['avg_win_bps']:.6f}", f"{stats['avg_loss_bps']:.6f}", grade, action, f"{size_mult:.6f}", f"bootstrap_ev;product={product_id};n={int(stats['sample_count'])};ev={stats['ev_mean_bps']:.2f};ci_low={stats['ev_ci_low_bps']:.2f};p_ev_positive={stats['prob_ev_positive']:.3f}"])
        for horizon in [1, 7, 30]:
            mc = _monte_carlo_path_stats(group, horizon_days=horizon, trials=int(monte_carlo_trials), position_size_pct=float(position_size_pct), seed=100 + horizon)
            risk_grade = _risk_grade_from_mc(mc)
            mc_rows.append([f"{ts_value:.6f}", dt_value, "product" if product_id != "ALL" else "portfolio", product_id, int(horizon), int(monte_carlo_trials), int(len(group)), f"{float(position_size_pct):.6f}", f"{mc['median_return_pct']:.6f}", f"{mc['p05_return_pct']:.6f}", f"{mc['p95_return_pct']:.6f}", f"{mc['prob_loss']:.6f}", f"{mc['median_max_drawdown_pct']:.6f}", f"{mc['p95_max_drawdown_pct']:.6f}", f"{mc['prob_drawdown_gt_3pct']:.6f}", f"{mc['prob_drawdown_gt_5pct']:.6f}", risk_grade, f"monte_carlo_block_bootstrap;product={product_id};horizon_days={horizon};trials={monte_carlo_trials};position_size_pct={position_size_pct:.3f}"])
    for (product_id, regime, context_key), group in frame.groupby(["product_id", "market_regime", "context_key"]):
        if len(group) < 12: continue
        stats = _bootstrap_ev_stats(group["outcome_bps"], weights=group.get("source_weight"), trials=max(500, int(bootstrap_trials / 2)), seed=33)
        grade, action, size_mult = _confidence_grade(stats)
        avg_adverse = float(pd.to_numeric(group.get("max_adverse_bps", pd.Series([0.0] * len(group))), errors="coerce").fillna(0.0).abs().mean())
        context_live_allowed = not (int(stats["sample_count"]) >= 30 and (float(stats["ev_ci_low_bps"]) < -8.0 or float(stats["prob_ev_positive"]) < 0.45))
        context_rows.append([f"{ts_value:.6f}", dt_value, str(product_id), str(regime), str(context_key), int(stats["sample_count"]), f"{stats['raw_win_rate']:.6f}", f"{stats['ev_mean_bps']:.6f}", f"{stats['ev_ci_low_bps']:.6f}", f"{stats['prob_ev_positive']:.6f}", f"{avg_adverse:.6f}", grade, bool(context_live_allowed), f"{size_mult:.6f}", f"context_ev;product={product_id};regime={regime};n={int(stats['sample_count'])};ev={stats['ev_mean_bps']:.2f};ci_low={stats['ev_ci_low_bps']:.2f};p_ev_positive={stats['prob_ev_positive']:.3f};action={action}"])
    for product_id, group in frame.groupby("product_id"):
        stats = product_ev_map.get(str(product_id)) or _bootstrap_ev_stats(group["outcome_bps"], weights=group.get("source_weight"), trials=int(bootstrap_trials), seed=44)
        grade, action, size_mult = _confidence_grade(stats); mc7 = _latest_product_mc(mc_rows, str(product_id), 7) or {}; mc30 = _latest_product_mc(mc_rows, str(product_id), 30) or {}
        p95_dd30 = float(mc30.get("p95_max_drawdown_pct", 99.0)); prob_loss7 = float(mc7.get("prob_loss", 1.0)); prob_dd3_30 = float(mc30.get("prob_drawdown_gt_3pct", 1.0)); path_grade = str(mc30.get("risk_grade", "UNKNOWN_PATH_RISK"))
        live_allowed = True; block_reasons = []
        if int(stats["sample_count"]) >= 60:
            if float(stats["prob_ev_positive"]) < 0.52: live_allowed = False; block_reasons.append("prob_ev_positive_too_low")
            if float(stats["ev_ci_low_bps"]) < -12.0: live_allowed = False; block_reasons.append("ev_ci_low_too_negative")
            if p95_dd30 > 5.0 and prob_dd3_30 > 0.45: live_allowed = False; block_reasons.append("monte_carlo_drawdown_risk_too_high")
        if path_grade == "HIGH_PATH_RISK": size_mult = min(float(size_mult), 0.50)
        elif path_grade == "ELEVATED_PATH_RISK": size_mult = min(float(size_mult), 0.75)
        if not live_allowed: size_mult = 0.0
        live_gate_rows.append([f"{ts_value:.6f}", dt_value, str(product_id), int(stats["sample_count"]), f"{stats['ev_mean_bps']:.6f}", f"{stats['ev_ci_low_bps']:.6f}", f"{stats['prob_ev_positive']:.6f}", f"{p95_dd30:.6f}", f"{prob_loss7:.6f}", f"{prob_dd3_30:.6f}", f"{grade}|{path_grade}", bool(live_allowed), f"{float(size_mult):.6f}", f"risk_live_gate;product={product_id};action={action};ev={stats['ev_mean_bps']:.2f};ci_low={stats['ev_ci_low_bps']:.2f};p_ev_positive={stats['prob_ev_positive']:.3f};p95_dd30={p95_dd30:.2f};prob_loss7={prob_loss7:.3f};prob_dd3_30={prob_dd3_30:.3f};block_reasons={','.join(block_reasons) if block_reasons else 'none'}"])
    _write_rows(os.path.join(base_dir, "risk_ev_confidence.csv"), RISK_EV_CONFIDENCE_COLUMNS, ev_rows)
    _write_rows(os.path.join(base_dir, "risk_monte_carlo_summary.csv"), RISK_MONTE_CARLO_COLUMNS, mc_rows)
    _write_rows(os.path.join(base_dir, "risk_context_performance.csv"), RISK_CONTEXT_PERFORMANCE_COLUMNS, context_rows)
    _write_rows(os.path.join(base_dir, "risk_live_gate.csv"), RISK_LIVE_GATE_COLUMNS, live_gate_rows)
    log(f"[risk-intelligence] completed frame_rows={len(frame)} ev_rows={len(ev_rows)} mc_rows={len(mc_rows)} context_rows={len(context_rows)} live_gate_rows={len(live_gate_rows)}")
    return {"rows": int(len(frame)), "ev_rows": int(len(ev_rows)), "monte_carlo_rows": int(len(mc_rows)), "context_rows": int(len(context_rows)), "live_gate_rows": int(len(live_gate_rows)), "paths": {"risk_ev_confidence": os.path.join(base_dir, "risk_ev_confidence.csv"), "risk_monte_carlo_summary": os.path.join(base_dir, "risk_monte_carlo_summary.csv"), "risk_context_performance": os.path.join(base_dir, "risk_context_performance.csv"), "risk_live_gate": os.path.join(base_dir, "risk_live_gate.csv")}}


def _latest_by_key(frame: pd.DataFrame, key_col: str) -> Dict[str, Dict[str, Any]]:
    if frame is None or frame.empty or key_col not in frame.columns: return {}
    local = frame.copy()
    if "ts" in local.columns:
        local["ts"] = pd.to_numeric(local["ts"], errors="coerce").fillna(0.0); local = local.sort_values("ts")
    result: Dict[str, Dict[str, Any]] = {}
    for _, row in local.iterrows(): result[str(row.get(key_col, ""))] = row.to_dict()
    return result


def load_risk_live_gate_map(base_dir: str) -> Dict[str, Dict[str, Any]]:
    return _latest_by_key(_read_csv(os.path.join(base_dir, "risk_live_gate.csv")), "product_id")


def _split_context_key(context_key: str) -> Dict[str, str]:
    parts = str(context_key or "").split("|")
    while len(parts) < 7:
        parts.append("unknown")

    return {
        "regime": _normalize_context_part(parts[0], "unknown_regime"),
        "session": _normalize_context_part(parts[1], "unknown_session"),
        "structure": _normalize_context_part(parts[2], "unknown_structure"),
        "value_area": _normalize_context_part(parts[3], "unknown_value"),
        "fvg": _normalize_context_part(parts[4], "unknown_fvg"),
        "volume_node": _normalize_context_part(parts[5], "unknown_volume_node"),
        "quant": _normalize_context_part(parts[6], "unknown_quant"),
    }


def _weighted_summary_rows_to_context_map(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}

    if frame.empty:
        return output

    frame = frame.copy()
    frame["sample_count"] = pd.to_numeric(frame.get("sample_count", 0), errors="coerce").fillna(0.0)
    frame["ev_mean_bps"] = pd.to_numeric(frame.get("ev_mean_bps", 0.0), errors="coerce").fillna(0.0)
    frame["ev_ci_low_bps"] = pd.to_numeric(frame.get("ev_ci_low_bps", 0.0), errors="coerce").fillna(0.0)
    frame["prob_ev_positive"] = pd.to_numeric(frame.get("prob_ev_positive", 0.0), errors="coerce").fillna(0.0)
    frame["context_size_multiplier"] = pd.to_numeric(frame.get("context_size_multiplier", 1.0), errors="coerce").fillna(1.0)

    parsed = frame["context_key"].apply(_split_context_key)
    frame["_regime_key"] = parsed.apply(lambda x: x["regime"])
    frame["_session_key"] = parsed.apply(lambda x: x["session"])
    frame["_structure_key"] = parsed.apply(lambda x: x["structure"])
    frame["_quant_key"] = parsed.apply(lambda x: x["quant"])

    def add_row(key: str, rows: pd.DataFrame, tier: str) -> None:
        total_n = float(rows["sample_count"].sum())
        if total_n <= 0:
            return

        weights = rows["sample_count"].clip(lower=1.0)
        ev = float(np.average(rows["ev_mean_bps"], weights=weights))
        ci_low = float(np.average(rows["ev_ci_low_bps"], weights=weights))
        prob_pos = float(np.average(rows["prob_ev_positive"], weights=weights))
        size_mult = float(np.average(rows["context_size_multiplier"], weights=weights))

        allowed = not (total_n >= 30 and (ci_low < -8.0 or prob_pos < 0.45))

        output[key] = {
            "map_key": key,
            "context_tier": tier,
            "sample_count": total_n,
            "ev_mean_bps": ev,
            "ev_ci_low_bps": ci_low,
            "prob_ev_positive": prob_pos,
            "context_live_allowed": allowed,
            "context_size_multiplier": size_mult if allowed else 0.0,
            "reason": (
                f"context_fallback_tier={tier};n={total_n:.0f};"
                f"ev={ev:.2f};ci_low={ci_low:.2f};p_ev_positive={prob_pos:.3f}"
            ),
        }

    for _, row in frame.iterrows():
        product_id = str(row.get("product_id", ""))
        context_key = str(row.get("context_key", ""))
        if product_id and context_key:
            key = f"{product_id}||full||{context_key}"
            output[key] = row.to_dict()
            output[key]["map_key"] = key
            output[key]["context_tier"] = "full"

    for (product_id, regime, session, structure), rows in frame.groupby(
        ["product_id", "_regime_key", "_session_key", "_structure_key"]
    ):
        add_row(
            f"{product_id}||regime_session_structure||{regime}|{session}|{structure}",
            rows,
            "product_regime_session_structure",
        )

    for (product_id, regime, quant), rows in frame.groupby(
        ["product_id", "_regime_key", "_quant_key"]
    ):
        add_row(
            f"{product_id}||regime_quant||{regime}|{quant}",
            rows,
            "product_regime_quant",
        )

    for (product_id, regime), rows in frame.groupby(["product_id", "_regime_key"]):
        add_row(
            f"{product_id}||regime_only||{regime}",
            rows,
            "product_regime_only",
        )

    for regime, rows in frame.groupby("_regime_key"):
        add_row(
            f"ALL||regime_only||{regime}",
            rows,
            "portfolio_regime_only",
        )

    return output


def load_risk_context_map(base_dir: str) -> Dict[str, Dict[str, Any]]:
    frame = _read_csv(os.path.join(base_dir, "risk_context_performance.csv"))
    if frame.empty:
        return {}
    return _weighted_summary_rows_to_context_map(frame)


if __name__ == "__main__":
    run_risk_intelligence(base_dir=os.path.dirname(os.path.abspath(__file__)))
