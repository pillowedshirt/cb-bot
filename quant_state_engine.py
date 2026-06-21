import csv
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_OUTCOME_CORRELATION_COLUMNS = [
    "ts", "dt_utc", "scope", "product_id", "feature_name", "sample_count",
    "feature_mean", "feature_std", "outcome_corr", "abs_corr", "feature_weight",
    "reliability", "edge_direction", "live_enabled", "reason",
]

FEATURE_CORRELATION_MATRIX_COLUMNS = [
    "ts", "dt_utc", "scope", "product_id", "feature_a", "feature_b",
    "sample_count", "correlation", "redundant_pair", "reason",
]

MARKOV_TRANSITION_COLUMNS = [
    "ts", "dt_utc", "scope", "product_id", "from_regime", "to_regime",
    "transition_count", "transition_probability", "reason",
]

MARKOV_POLICY_COLUMNS = [
    "ts", "dt_utc", "scope", "product_id", "current_regime", "sample_count",
    "negative_next_probability", "high_vol_next_probability", "continuation_probability",
    "steady_state_negative_probability", "markov_grade", "live_allowed",
    "size_multiplier", "reason",
]

KALMAN_POLICY_COLUMNS = [
    "ts", "dt_utc", "scope", "product_id", "sample_count", "median_abs_return_bps",
    "measurement_noise_bps", "process_noise_bps", "kalman_enabled", "reason",
]

QUANT_STATE_SUMMARY_COLUMNS = [
    "ts", "dt_utc", "metric", "value", "reason",
]


FEATURE_CANDIDATES = [
    "score",
    "entry_score",
    "score_at_entry",
    "probability",
    "prob_at_entry",
    "estimated_prob_up",
    "calibrated_p_win",
    "expected_net_edge_bps",
    "ev_at_entry",
    "spread_at_entry",
    "move_bps", "projected_forward_gain_bps", "expected_utility_bps",
    "buy_vs_wait_edge_bps", "maker_adjusted_expected_value_bps", "payoff_ratio",
    "cost_bps", "spread_bps", "walk_forward_penalty_bps", "uncertainty_penalty_bps",
    "momentum_1_bps", "momentum_3_bps", "momentum_5_bps", "momentum_15_bps",
    "order_book_imbalance", "order_book_top_depth_usd", "spread_instability_bps",
    "liquidity_risk_score", "quant_forecast_return_bps", "quant_conditional_volatility_bps",
    "relative_volume", "atr_bps", "rsi", "max_favorable_bps", "max_adverse_bps",
]

NEGATIVE_REGIME_TERMS = ("down", "bear", "risk_off", "sell", "negative", "breakdown")
HIGH_VOL_REGIME_TERMS = ("high_vol", "high-vol", "volatile", "volatility", "panic")


def _utc_ts() -> float:
    return float(time.time())


def _utc_dt(ts_value: Optional[float] = None) -> str:
    ts = _utc_ts() if ts_value is None else float(ts_value)
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


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
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if frame is None or frame.empty or column not in frame.columns:
        return pd.Series([default] * (0 if frame is None else len(frame)), dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _extract_outcome_bps(frame: pd.DataFrame) -> pd.Series:
    for col in ["realized_or_proxy_net_bps", "binance_taker_taker_net_pnl_bps", "binance_maker_taker_net_pnl_bps", "net_pnl_bps", "buy_net_bps", "realized_net_pnl_bps", "primary_net_pnl_bps", "outcome_bps", "move_bps", "ev_at_entry"]:
        if col in frame.columns:
            return _num(frame, col, 0.0)
    if "max_favorable_bps" in frame.columns and "max_adverse_bps" in frame.columns:
        fav = _num(frame, "max_favorable_bps", 0.0)
        adv = _num(frame, "max_adverse_bps", 0.0).abs()
        cost = _num(frame, "cost_bps", 0.0)
        return fav - adv * 0.35 - cost
    if "expected_net_edge_bps" in frame.columns:
        return _num(frame, "expected_net_edge_bps", 0.0)
    return pd.Series([0.0] * len(frame), index=frame.index)


def _infer_regime(frame: pd.DataFrame) -> pd.Series:
    for col in [
        "market_regime",
        "regime_tag",
        "quant_volatility_cluster_state",
        "volatility_cluster",
        "trend_regime",
    ]:
        if col in frame.columns:
            return (
                frame[col]
                .astype(str)
                .fillna("unknown_regime")
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
            )
    return pd.Series(["unknown_regime"] * len(frame), index=frame.index)


def _infer_ts(frame: pd.DataFrame) -> pd.Series:
    for col in ["entry_ts", "replay_ts", "ts"]:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return pd.Series(np.arange(len(frame)), index=frame.index, dtype="float64")


def _source_frames(base_dir: str) -> pd.DataFrame:
    sources = [
        ("fifth_pass_live_style_replay.csv", "fifth_pass", 0.90),
        ("historical_shadow_replay.csv", "historical_shadow", 0.75),
        ("candidate_replay.csv", "candidate_proxy", 0.45),
        ("council_observation_outcomes.csv", "council_observed", 0.80),
        ("trade_outcomes.csv", "fixed_window_trade_outcome", 0.70),
        ("missed_opportunities.csv", "missed_opportunity", 0.60),
    ]
    frames: List[pd.DataFrame] = []
    for filename, source_name, source_weight in sources:
        raw = _read_csv(os.path.join(base_dir, filename))
        if raw.empty or "product_id" not in raw.columns:
            continue
        out = pd.DataFrame(index=raw.index)
        out["product_id"] = raw["product_id"].astype(str)
        out["outcome_bps"] = _extract_outcome_bps(raw)
        out["market_regime"] = _infer_regime(raw).astype(str)
        out["row_ts"] = _infer_ts(raw)
        out["source_name"] = source_name
        out["source_weight"] = float(source_weight)
        for feature in FEATURE_CANDIDATES:
            if feature in raw.columns:
                out[feature] = pd.to_numeric(raw[feature], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["product_id", "outcome_bps"])
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["product_id"].astype(str).str.len() > 0].copy()
    combined["outcome_bps"] = pd.to_numeric(combined["outcome_bps"], errors="coerce").fillna(0.0)
    return combined


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    local = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(local) < 20 or float(local["a"].std(ddof=0) or 0.0) <= 1e-12 or float(local["b"].std(ddof=0) or 0.0) <= 1e-12:
        return 0.0
    val = float(local["a"].corr(local["b"]))
    return val if math.isfinite(val) else 0.0


def _reliability(n: int, abs_corr: float) -> float:
    return float(min(1.0, max(0.0, (float(n) - 20.0) / 180.0)) * min(1.0, abs_corr / 0.35))


def _feature_rows(frame: pd.DataFrame, ts_value: float, dt_value: str) -> Tuple[List[List[Any]], List[List[Any]]]:
    feature_rows: List[List[Any]] = []
    matrix_rows: List[List[Any]] = []
    groups: List[Tuple[str, str, pd.DataFrame]] = [("portfolio", "ALL", frame)]
    groups.extend(("product", str(pid), group.copy()) for pid, group in frame.groupby("product_id"))
    for scope, product_id, group in groups:
        features = [feature for feature in FEATURE_CANDIDATES if feature in group.columns and pd.to_numeric(group[feature], errors="coerce").notna().sum() >= 20]
        for feature in features:
            local = group[[feature, "outcome_bps"]].copy(); local[feature] = pd.to_numeric(local[feature], errors="coerce"); local = local.dropna(); n = len(local)
            if n < 20: continue
            corr = _safe_corr(local[feature], local["outcome_bps"]); abs_corr = abs(corr); reliability = _reliability(n, abs_corr); feature_weight = corr * reliability
            direction = "positive_edge_when_high" if corr > 0 else "negative_edge_when_high" if corr < 0 else "neutral"
            live_enabled = bool(n >= 40 and abs_corr >= 0.05 and reliability >= 0.05)
            feature_rows.append([f"{ts_value:.6f}", dt_value, scope, product_id, feature, int(n), f"{float(local[feature].mean()):.8f}", f"{float(local[feature].std(ddof=0) or 0.0):.8f}", f"{corr:.8f}", f"{abs_corr:.8f}", f"{feature_weight:.8f}", f"{reliability:.8f}", direction, live_enabled, f"feature_outcome_corr;feature={feature};n={n};corr={corr:.4f};reliability={reliability:.3f}"])
        for i, feature_a in enumerate(features):
            for feature_b in features[i + 1:]:
                local = group[[feature_a, feature_b]].copy(); local[feature_a] = pd.to_numeric(local[feature_a], errors="coerce"); local[feature_b] = pd.to_numeric(local[feature_b], errors="coerce"); local = local.dropna()
                if len(local) < 30: continue
                corr = _safe_corr(local[feature_a], local[feature_b]); redundant = bool(abs(corr) >= 0.85)
                matrix_rows.append([f"{ts_value:.6f}", dt_value, scope, product_id, feature_a, feature_b, int(len(local)), f"{corr:.8f}", redundant, f"feature_pair_corr;abs_corr={abs(corr):.4f};redundant={redundant}"])
    return feature_rows, matrix_rows


def _is_negative_regime(regime: str) -> bool:
    return any(term in str(regime).lower() for term in NEGATIVE_REGIME_TERMS)


def _is_high_vol_regime(regime: str) -> bool:
    return any(term in str(regime).lower() for term in HIGH_VOL_REGIME_TERMS)


def _steady_state(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0: return np.array([])
    n = matrix.shape[0]; vec = np.ones(n, dtype=float) / max(n, 1)
    for _ in range(250):
        nxt = vec @ matrix
        if np.max(np.abs(nxt - vec)) < 1e-10: return nxt
        vec = nxt
    return vec


def _markov_rows(frame: pd.DataFrame, ts_value: float, dt_value: str) -> Tuple[List[List[Any]], List[List[Any]]]:
    transition_rows: List[List[Any]] = []; policy_rows: List[List[Any]] = []
    # Markov transitions must be built from one product's time series at a time.
    # Do not build portfolio-level transitions by sorting interleaved products,
    # because that creates fake transitions from one product's regime into another product's regime.
    groups: List[Tuple[str, str, pd.DataFrame]] = [
        ("product", str(pid), group.copy())
        for pid, group in frame.groupby("product_id")
    ]
    for scope, product_id, group in groups:
        local = group[["market_regime", "row_ts"]].copy().dropna()
        local["market_regime"] = (
            local["market_regime"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        local = local[
            (local["market_regime"] != "")
            & (local["market_regime"] != "unknown_regime")
            & (local["market_regime"] != "nan")
        ].copy()
        local = local.sort_values("row_ts")

        regimes = sorted([str(x) for x in local["market_regime"].dropna().unique()])
        if len(local) < 20 or len(regimes) < 2: continue
        idx = {regime: i for i, regime in enumerate(regimes)}; counts = np.zeros((len(regimes), len(regimes)), dtype=float); values = list(local["market_regime"].astype(str).values)
        for current_regime, next_regime in zip(values[:-1], values[1:]): counts[idx[current_regime], idx[next_regime]] += 1.0
        row_sums = counts.sum(axis=1); probs = np.zeros_like(counts)
        for i in range(len(regimes)):
            if row_sums[i] > 0: probs[i, :] = counts[i, :] / row_sums[i]
            else: probs[i, i] = 1.0
        steady = _steady_state(probs)
        for from_regime in regimes:
            i = idx[from_regime]
            for to_regime in regimes:
                j = idx[to_regime]
                if counts[i, j] <= 0: continue
                transition_rows.append([f"{ts_value:.6f}", dt_value, scope, product_id, from_regime, to_regime, int(counts[i, j]), f"{probs[i, j]:.8f}", f"markov_transition;from={from_regime};to={to_regime};p={probs[i, j]:.4f}"])
        for regime in regimes:
            i = idx[regime]
            neg = float(sum(probs[i, j] for j, r in enumerate(regimes) if _is_negative_regime(r)))
            high = float(sum(probs[i, j] for j, r in enumerate(regimes) if _is_high_vol_regime(r)))
            cont = float(probs[i, i]); steady_neg = float(sum(steady[j] for j, r in enumerate(regimes) if _is_negative_regime(r))) if steady.size else 0.0; sample_count = int(row_sums[i])
            if sample_count < 10: grade, live_allowed, size_multiplier = "INSUFFICIENT_REGIME_TRANSITIONS", True, 0.75
            elif neg >= 0.60 or (neg >= 0.45 and high >= 0.45): grade, live_allowed, size_multiplier = "NEGATIVE_TRANSITION_RISK", False, 0.0
            elif neg >= 0.35 or high >= 0.55: grade, live_allowed, size_multiplier = "ELEVATED_TRANSITION_RISK", True, 0.60
            elif steady_neg >= 0.45: grade, live_allowed, size_multiplier = "LONG_RUN_REGIME_RISK", True, 0.75
            else: grade, live_allowed, size_multiplier = "REGIME_TRANSITION_OK", True, 1.0
            policy_rows.append([f"{ts_value:.6f}", dt_value, scope, product_id, regime, sample_count, f"{neg:.8f}", f"{high:.8f}", f"{cont:.8f}", f"{steady_neg:.8f}", grade, live_allowed, f"{size_multiplier:.8f}", f"markov_policy;regime={regime};neg_next={neg:.3f};high_next={high:.3f};steady_neg={steady_neg:.3f};n={sample_count}"])
    return transition_rows, policy_rows


def _kalman_policy_rows(frame: pd.DataFrame, ts_value: float, dt_value: str) -> List[List[Any]]:
    rows: List[List[Any]] = []
    groups: List[Tuple[str, str, pd.DataFrame]] = [("portfolio", "ALL", frame)]
    groups.extend(("product", str(pid), group.copy()) for pid, group in frame.groupby("product_id"))
    for scope, product_id, group in groups:
        values = pd.to_numeric(group["outcome_bps"], errors="coerce").dropna().astype(float)
        if len(values) < 20: continue
        mad = float(np.median(np.abs(values - np.median(values))))
        meas = max(2.0, mad * 0.75); proc = max(0.25, mad * 0.10)
        rows.append([f"{ts_value:.6f}", dt_value, scope, product_id, int(len(values)), f"{mad:.8f}", f"{meas:.8f}", f"{proc:.8f}", True, f"kalman_policy;mad={mad:.3f};measurement_noise={meas:.3f};process_noise={proc:.3f}"])
    return rows


def run_quant_state_engine(*, base_dir: str, log_fn=None) -> Dict[str, Any]:
    def log(message: str) -> None:
        if log_fn is not None:
            try: log_fn(message); return
            except Exception: pass
        print(message)
    base_dir = os.path.abspath(base_dir); ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); frame = _source_frames(base_dir)
    if frame.empty:
        for path, columns in [("feature_outcome_correlation.csv", FEATURE_OUTCOME_CORRELATION_COLUMNS), ("feature_correlation_matrix.csv", FEATURE_CORRELATION_MATRIX_COLUMNS), ("markov_regime_transitions.csv", MARKOV_TRANSITION_COLUMNS), ("markov_regime_policy.csv", MARKOV_POLICY_COLUMNS), ("kalman_filter_policy.csv", KALMAN_POLICY_COLUMNS), ("quant_state_summary.csv", QUANT_STATE_SUMMARY_COLUMNS)]:
            _write_rows(os.path.join(base_dir, path), columns, [])
        return {"rows": 0, "reason": "no_quant_state_training_rows"}
    feature_rows, matrix_rows = _feature_rows(frame, ts_value, dt_value); transition_rows, policy_rows = _markov_rows(frame, ts_value, dt_value); kalman_rows = _kalman_policy_rows(frame, ts_value, dt_value)
    summary_rows = [[f"{ts_value:.6f}", dt_value, "training_rows", int(len(frame)), "quant_state_engine source rows"], [f"{ts_value:.6f}", dt_value, "feature_outcome_rows", int(len(feature_rows)), "active covariance/correlation feature policy rows"], [f"{ts_value:.6f}", dt_value, "feature_pair_rows", int(len(matrix_rows)), "feature covariance/correlation matrix rows"], [f"{ts_value:.6f}", dt_value, "markov_transition_rows", int(len(transition_rows)), "market-regime Markov transition rows"], [f"{ts_value:.6f}", dt_value, "markov_policy_rows", int(len(policy_rows)), "active Markov regime policy rows"], [f"{ts_value:.6f}", dt_value, "kalman_policy_rows", int(len(kalman_rows)), "active Kalman policy rows"]]
    _write_rows(os.path.join(base_dir, "feature_outcome_correlation.csv"), FEATURE_OUTCOME_CORRELATION_COLUMNS, feature_rows); _write_rows(os.path.join(base_dir, "feature_correlation_matrix.csv"), FEATURE_CORRELATION_MATRIX_COLUMNS, matrix_rows); _write_rows(os.path.join(base_dir, "markov_regime_transitions.csv"), MARKOV_TRANSITION_COLUMNS, transition_rows); _write_rows(os.path.join(base_dir, "markov_regime_policy.csv"), MARKOV_POLICY_COLUMNS, policy_rows); _write_rows(os.path.join(base_dir, "kalman_filter_policy.csv"), KALMAN_POLICY_COLUMNS, kalman_rows); _write_rows(os.path.join(base_dir, "quant_state_summary.csv"), QUANT_STATE_SUMMARY_COLUMNS, summary_rows)
    log(f"[quant-state] completed rows={len(frame)} feature={len(feature_rows)} pair={len(matrix_rows)} markov={len(policy_rows)} kalman={len(kalman_rows)}")
    return {"rows": int(len(frame)), "feature_rows": int(len(feature_rows)), "matrix_rows": int(len(matrix_rows)), "markov_transition_rows": int(len(transition_rows)), "markov_policy_rows": int(len(policy_rows)), "kalman_policy_rows": int(len(kalman_rows))}


def _latest_frame(path: str) -> pd.DataFrame:
    return _read_csv(path)


def load_feature_policy_map(base_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    frame = _latest_frame(os.path.join(base_dir, "feature_outcome_correlation.csv"))
    if frame.empty: return {}
    frame = frame.copy(); frame["live_enabled_bool"] = frame.get("live_enabled", True).astype(str).str.lower().isin({"true", "1", "yes", "y"}); frame = frame[frame["live_enabled_bool"]]
    output: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in frame.iterrows(): output.setdefault(str(row.get("product_id", "ALL")), []).append(row.to_dict())
    return output


def _normalize_regime_key(value: Any) -> str:
    return (
        str(value if value not in (None, "") else "unknown_regime")
        .strip()
        .lower()
        .replace(" ", "_")
    )


def load_markov_policy_map(base_dir: str) -> Dict[str, Dict[str, Any]]:
    frame = _latest_frame(os.path.join(base_dir, "markov_regime_policy.csv"))
    if frame.empty: return {}
    if "ts" in frame.columns: frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce").fillna(0.0); frame = frame.sort_values("ts")
    output: Dict[str, Dict[str, Any]] = {}
    for _, row in frame.iterrows(): output[f"{row.get('product_id', 'ALL')}||{_normalize_regime_key(row.get('current_regime', 'unknown_regime'))}"] = row.to_dict()
    return output


def load_kalman_policy_map(base_dir: str) -> Dict[str, Dict[str, Any]]:
    frame = _latest_frame(os.path.join(base_dir, "kalman_filter_policy.csv"))
    if frame.empty: return {}
    if "ts" in frame.columns: frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce").fillna(0.0); frame = frame.sort_values("ts")
    output: Dict[str, Dict[str, Any]] = {}
    for _, row in frame.iterrows(): output[str(row.get("product_id", "ALL"))] = row.to_dict()
    return output


def candidate_feature_value(candidate: Dict[str, Any], feature_name: str) -> Optional[float]:
    aliases = {
        "score": ["score", "entry_score", "score_at_entry"],
        "entry_score": ["score", "entry_score", "score_at_entry"],
        "score_at_entry": ["score", "entry_score", "score_at_entry"],

        "probability": ["probability", "estimated_prob_up", "prob_at_entry"],
        "estimated_prob_up": ["estimated_prob_up", "probability", "prob_at_entry"],
        "prob_at_entry": ["estimated_prob_up", "probability", "prob_at_entry"],

        "expected_net_edge_bps": ["expected_net_edge_bps", "ev_bps", "ev_at_entry"],
        "ev_at_entry": ["expected_net_edge_bps", "ev_bps", "ev_at_entry"],

        "spread_bps": ["spread_bps", "spread_at_entry"],
        "spread_at_entry": ["spread_bps", "spread_at_entry"],
    }
    for key in aliases.get(feature_name, [feature_name]):
        try:
            if key in candidate:
                value = float(candidate.get(key, 0.0) or 0.0)
                if math.isfinite(value): return value
        except Exception: pass
    return None


if __name__ == "__main__":
    run_quant_state_engine(base_dir=os.path.dirname(os.path.abspath(__file__)))
