import csv
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from runtime_paths import ensure_runtime_dirs, runtime_path, write_generated_file_meta

CONTINUOUS_RESEARCH_HISTORY_COLUMNS = ["ts","dt_utc","cycle_id","cycle_type","status","duration_sec","input_rows","output_rows","reason"]
MARKET_STATE_ANALOG_MATCH_COLUMNS = ["ts","dt_utc","cycle_id","product_id","match_rank","similarity_score","source_timeframe","source_row_ts","source_regime","outcome_bps","features_used","reason"]
MARKET_STATE_ANALOG_SUMMARY_COLUMNS = ["ts","dt_utc","cycle_id","product_id","analog_sample_count","analog_avg_outcome_bps","analog_median_outcome_bps","analog_win_rate","analog_p25_bps","analog_p75_bps","analog_best_similarity","analog_gate","size_multiplier","reason"]
RESEARCH_FILE_HEALTH_COLUMNS = ["ts", "dt_utc", "filename", "path", "exists", "rows", "products", "health", "priority", "reason"]
RESEARCH_BACKFILL_PLAN_COLUMNS = ["ts", "dt_utc", "task_id", "task_type", "product_id", "timeframe", "priority", "status", "reason"]

def _utc_ts() -> float: return float(time.time())
def _utc_dt(ts_value: Optional[float] = None) -> str:
    return datetime.fromtimestamp(_utc_ts() if ts_value is None else float(ts_value), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _read_csv(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 0: return pd.DataFrame()
        return pd.read_csv(path)
    except Exception: return pd.DataFrame()

def _write_rows(path: str, columns: List[str], rows: List[List[Any]], reason: str) -> None:
    tmp_path = path + ".tmp"; os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(columns); writer.writerows(rows)
    os.replace(tmp_path, path); write_generated_file_meta(path, reason=reason)

def _append_rows(path: str, columns: List[str], rows: List[List[Any]], reason: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) <= 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header: writer.writerow(columns)
        writer.writerows(rows)
    write_generated_file_meta(path, reason=reason)

def _file_health(filename: str, min_rows: int = 1, min_products: int = 0) -> dict:
    path = runtime_path(filename)
    result = {"filename": filename, "path": path, "exists": os.path.exists(path), "rows": 0, "products": 0, "health": "MISSING", "priority": 100, "reason": "missing"}
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return result
        if filename.endswith(".json"):
            result.update({"health": "OK", "priority": 20, "reason": "json_exists"})
            return result
        frame = _read_csv(path)
        if frame.empty:
            result.update({"health": "EMPTY", "priority": 90, "reason": "empty_or_header_only"})
            return result
        result["rows"] = int(len(frame))
        if "product_id" in frame.columns:
            result["products"] = int(frame["product_id"].astype(str).nunique())
        if result["rows"] < int(min_rows):
            result.update({"health": "LOW_ROWS", "priority": 80, "reason": f"rows={result['rows']};min_rows={min_rows}"})
        elif min_products > 0 and result["products"] < int(min_products):
            result.update({"health": "LOW_PRODUCT_COVERAGE", "priority": 75, "reason": f"products={result['products']};min_products={min_products}"})
        else:
            result.update({"health": "OK", "priority": 10, "reason": "usable"})
        return result
    except Exception as exc:
        result.update({"health": "READ_ERROR", "priority": 95, "reason": f"read_error:{exc}"})
        return result


def build_research_data_plan(products: list[str]) -> dict:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value)
    health_targets = [
        ("historical_replay_15m_90d.csv", 60 * max(1, len(products)), max(1, len(products))),
        ("historical_replay_1h_365d.csv", 20 * max(1, len(products)), max(1, len(products))),
        ("historical_shadow_replay.csv", 100, 1), ("candidate_replay.csv", 100, 1),
        ("risk_live_gate.csv", 1, 1), ("feature_outcome_correlation.csv", 1, 1),
        ("markov_regime_policy.csv", 1, 1), ("kalman_filter_policy.csv", 1, 1),
        ("market_state_analog_summary.csv", 1, 1),
    ]
    health_rows, plan_rows = [], []
    for filename, min_rows, min_products in health_targets:
        h = _file_health(filename, min_rows=min_rows, min_products=min_products)
        health_rows.append([f"{ts_value:.6f}", dt_value, h["filename"], h["path"], bool(h["exists"]), int(h["rows"]), int(h["products"]), h["health"], int(h["priority"]), h["reason"]])
    raw_15m = _read_csv(runtime_path("historical_replay_15m_90d.csv")); raw_1h = _read_csv(runtime_path("historical_replay_1h_365d.csv")); analog = _read_csv(runtime_path("market_state_analog_summary.csv"))
    for product_id in products:
        p = str(product_id)
        rows_15m = int(raw_15m[raw_15m["product_id"].astype(str).eq(p)].shape[0]) if not raw_15m.empty and "product_id" in raw_15m.columns else 0
        rows_1h = int(raw_1h[raw_1h["product_id"].astype(str).eq(p)].shape[0]) if not raw_1h.empty and "product_id" in raw_1h.columns else 0
        analog_n = 0
        if not analog.empty and "product_id" in analog.columns and "analog_sample_count" in analog.columns:
            sub = analog[analog["product_id"].astype(str).eq(p)]
            if not sub.empty:
                analog_n = int(float(sub.tail(1)["analog_sample_count"].iloc[0] or 0))
        if rows_15m < 300:
            plan_rows.append([f"{ts_value:.6f}", dt_value, f"{p}__expand_15m_history", "expand_historical_cache", p, "primary_15m_90d", 90, "planned", f"15m rows low: {rows_15m}"])
        if rows_1h < 120:
            plan_rows.append([f"{ts_value:.6f}", dt_value, f"{p}__expand_1h_history", "expand_historical_cache", p, "regime_1h_365d", 95, "planned", f"1h rows low or missing: {rows_1h}"])
        if analog_n < 20:
            plan_rows.append([f"{ts_value:.6f}", dt_value, f"{p}__analog_research", "market_state_analog_research", p, "current_state", 70, "planned", f"analog samples low: {analog_n}"])
    plan_rows.sort(key=lambda row: int(row[6]), reverse=True)
    _write_rows(runtime_path("research_file_health.csv"), RESEARCH_FILE_HEALTH_COLUMNS, health_rows, reason="research_file_health_scan")
    _write_rows(runtime_path("research_backfill_plan.csv"), RESEARCH_BACKFILL_PLAN_COLUMNS, plan_rows, reason="research_backfill_plan")
    return {"health_rows": len(health_rows), "plan_rows": len(plan_rows), "top_tasks": plan_rows[:10]}

def _latest_market_rows() -> pd.DataFrame:
    market = _read_csv(runtime_path("market.csv"))
    if market.empty or "product_id" not in market.columns: return pd.DataFrame()
    if "ts" in market.columns:
        market["ts"] = pd.to_numeric(market["ts"], errors="coerce").fillna(0.0); market = market.sort_values("ts")
    return market.groupby("product_id", as_index=False).tail(1)

def _historical_rows() -> pd.DataFrame:
    hist = _read_csv(runtime_path("historical_shadow_replay.csv"))
    if hist.empty or "product_id" not in hist.columns: return pd.DataFrame()
    if "outcome_bps" not in hist.columns:
        for col in ["realized_or_proxy_net_bps","binance_taker_taker_net_pnl_bps","binance_maker_taker_net_pnl_bps","net_pnl_bps","move_bps","expected_net_edge_bps"]:
            if col in hist.columns:
                hist["outcome_bps"] = pd.to_numeric(hist[col], errors="coerce").fillna(0.0); break
    if "outcome_bps" not in hist.columns: hist["outcome_bps"] = 0.0
    return hist

def _feature_columns(current: pd.DataFrame, historical: pd.DataFrame) -> List[str]:
    preferred = ["score","entry_score","probability","estimated_prob_up","expected_net_edge_bps","spread_bps","cost_bps","momentum_1_bps","momentum_3_bps","momentum_5_bps","momentum_15_bps","quant_forecast_return_bps","quant_conditional_volatility_bps","order_book_imbalance","liquidity_risk_score","rsi","atr_bps"]
    return [c for c in preferred if c in current.columns and c in historical.columns and pd.to_numeric(current[c], errors="coerce").notna().any() and pd.to_numeric(historical[c], errors="coerce").notna().any()]

def _analog_matches_for_product(product_id: str, current_row: pd.Series, hist: pd.DataFrame, max_matches: int = 50) -> Tuple[List[List[Any]], List[Any]]:
    ts_value = _utc_ts(); dt_value = _utc_dt(ts_value); cycle_id = str(int(ts_value))
    product_hist = hist[hist["product_id"].astype(str).eq(str(product_id))].copy()
    def summary_empty(gate, mult, reason):
        return [f"{ts_value:.6f}", dt_value, cycle_id, product_id, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gate, mult, reason]
    if product_hist.empty: return [], summary_empty("NO_ANALOGS", 0.75, "no historical rows for product")
    current_df = pd.DataFrame([current_row.to_dict()]); features = _feature_columns(current_df, product_hist)
    if not features: return [], summary_empty("NO_COMMON_FEATURES", 0.75, "no shared numeric features between current market and historical rows")
    local = product_hist.copy(); distances = np.zeros(len(local), dtype=float); used = []
    for feature in features:
        hist_values = pd.to_numeric(local[feature], errors="coerce"); current_value = pd.to_numeric(current_df[feature], errors="coerce").iloc[0]
        if not math.isfinite(float(current_value)): continue
        std = float(hist_values.std(ddof=0) or 0.0)
        if std <= 1e-12: continue
        mean = float(hist_values.mean()); distances += np.square(((hist_values.fillna(mean)-mean)/std).to_numpy(dtype=float) - ((float(current_value)-mean)/std)); used.append(feature)
    if not used: return [], summary_empty("NO_USABLE_FEATURES", 0.75, "shared features had unusable variance/current values")
    local["similarity_score"] = 1.0 / (1.0 + np.sqrt(distances / max(1, len(used))))
    local = local.sort_values("similarity_score", ascending=False).head(int(max_matches)); outcomes = pd.to_numeric(local["outcome_bps"], errors="coerce").fillna(0.0)
    match_rows = [[f"{ts_value:.6f}", dt_value, cycle_id, product_id, int(rank), f"{float(row.get('similarity_score',0.0) or 0.0):.8f}", str(row.get("timeframe", "")), row.get("row_ts", row.get("ts", "")), str(row.get("market_regime", row.get("regime_tag", ""))), f"{float(row.get('outcome_bps',0.0) or 0.0):.8f}", "|".join(used), "market_state_analog_match"] for rank, (_, row) in enumerate(local.iterrows(), start=1)]
    sample_count = int(len(outcomes)); avg = float(outcomes.mean()) if sample_count else 0.0; median = float(outcomes.median()) if sample_count else 0.0; win_rate = float((outcomes > 0).mean()) if sample_count else 0.0; p25 = float(outcomes.quantile(0.25)) if sample_count else 0.0; p75 = float(outcomes.quantile(0.75)) if sample_count else 0.0; best = float(local["similarity_score"].max()) if sample_count else 0.0
    if sample_count >= 20 and avg > 5.0 and win_rate >= 0.55: gate, mult = "ANALOG_POSITIVE", 1.00
    elif sample_count >= 20 and (avg < -5.0 or win_rate < 0.42): gate, mult = "ANALOG_NEGATIVE", 0.35
    elif sample_count >= 10: gate, mult = "ANALOG_NEUTRAL", 0.75
    else: gate, mult = "ANALOG_LOW_SAMPLE", 0.65
    return match_rows, [f"{ts_value:.6f}", dt_value, cycle_id, product_id, sample_count, f"{avg:.8f}", f"{median:.8f}", f"{win_rate:.8f}", f"{p25:.8f}", f"{p75:.8f}", f"{best:.8f}", gate, f"{mult:.8f}", f"features={','.join(used)};max_matches={max_matches}"]

def run_market_state_analog_research(max_matches: int = 50) -> Dict[str, Any]:
    started = time.time(); ensure_runtime_dirs(); current = _latest_market_rows(); hist = _historical_rows(); match_rows, summary_rows = [], []
    if current.empty or hist.empty: return {"status":"skipped","reason":f"current_empty={current.empty};historical_empty={hist.empty}","matches":0,"summary_rows":0,"duration_sec":time.time()-started}
    for _, current_row in current.iterrows():
        product_id = str(current_row.get("product_id", ""))
        if product_id:
            rows, summary = _analog_matches_for_product(product_id, current_row, hist, max_matches=max_matches); match_rows.extend(rows); summary_rows.append(summary)
    _write_rows(runtime_path("market_state_analog_matches.csv"), MARKET_STATE_ANALOG_MATCH_COLUMNS, match_rows, "continuous_market_state_analog_research")
    _write_rows(runtime_path("market_state_analog_summary.csv"), MARKET_STATE_ANALOG_SUMMARY_COLUMNS, summary_rows, "continuous_market_state_analog_research")
    return {"status":"ok","reason":"market_state_analog_research_completed","matches":len(match_rows),"summary_rows":len(summary_rows),"duration_sec":time.time()-started}

def run_continuous_research_cycle() -> Dict[str, Any]:
    started = time.time(); ts_value = _utc_ts(); cycle_id = str(int(ts_value))
    try:
        current = _latest_market_rows()
        products = []
        if current is not None and not current.empty and "product_id" in current.columns:
            products = current["product_id"].dropna().astype(str).unique().tolist()
        data_plan = build_research_data_plan(products)
        analog = run_market_state_analog_research(max_matches=50)
        status = {"ts":ts_value,"dt_utc":_utc_dt(ts_value),"cycle_id":cycle_id,"status":"ok","data_plan":data_plan,"analog":analog,"duration_sec":time.time()-started}
        with open(runtime_path("continuous_research_status.json"), "w", encoding="utf-8") as file: json.dump(status, file, indent=2, sort_keys=True)
        write_generated_file_meta(runtime_path("continuous_research_status.json"), reason="continuous_research_cycle")
        _append_rows(runtime_path("continuous_research_history.csv"), CONTINUOUS_RESEARCH_HISTORY_COLUMNS, [[f"{ts_value:.6f}", _utc_dt(ts_value), cycle_id, "market_state_analog_research", str(analog.get("status", "")), f"{float(analog.get('duration_sec',0.0) or 0.0):.6f}", 0, int(analog.get("summary_rows",0) or 0), str(analog.get("reason", ""))]], "continuous_research_history_append")
        return status
    except Exception as exc:
        status = {"ts":ts_value,"dt_utc":_utc_dt(ts_value),"cycle_id":cycle_id,"status":"error","error":str(exc),"traceback":traceback.format_exc(),"duration_sec":time.time()-started}
        try:
            with open(runtime_path("continuous_research_status.json"), "w", encoding="utf-8") as file: json.dump(status, file, indent=2, sort_keys=True)
        except Exception: pass
        return status
