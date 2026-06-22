from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

BUY_AGENT_WEIGHTS = {
    "cross_asset_analog_score": 0.17,
    "previous_session_profile_score": 0.15,
    "upside_room_score": 0.13,
    "low_volume_path_score": 0.11,
    "poc_fair_value_stretch_score": 0.10,
    "value_acceptance_score": 0.10,
    "volatility_expansion_score": 0.08,
    "range_discount_score": 0.07,
    "rsi_mean_reversion_score": 0.06,
    "fvg_fresh_zone_score": 0.03,
}

SELL_AGENT_WEIGHTS = {
    "low_stationarity_run_score": 0.17,
    "quant_forecast_strength_score": 0.13,
    "normal_volatility_cluster_score": 0.12,
    "upper_range_resistance_score": 0.11,
    "momentum_expansion_score": 0.10,
    "mean_extension_score": 0.10,
    "rsi_overbought_score": 0.09,
    "low_volume_path_exhaustion_score": 0.08,
    "volume_climax_score": 0.05,
    "post_profit_flattening_score": 0.05,
}

BUY_STRONG_THRESHOLD = 0.72
BUY_ACCEPTABLE_THRESHOLD = 0.60
BUY_SHADOW_THRESHOLD = 0.50
SELL_EXIT_THRESHOLD = 0.75
SELL_ARM_TRAIL_THRESHOLD = 0.62
SELL_TIGHTEN_THRESHOLD = 0.50
SIM_MIN_PROFIT_TARGET_BPS = 100.0
SIM_HARD_STOP_BPS = -100.0
DEFAULT_FEE_AND_SLIPPAGE_BPS = 20.0

def clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        x = float(value)
        if not math.isfinite(x):
            return lo
        return max(lo, min(hi, x))
    except Exception:
        return lo

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)

def bps_change(new_value: float, old_value: float) -> float:
    old = float(old_value or 0.0)
    if old <= 0.0:
        return 0.0
    return ((float(new_value) / old) - 1.0) * 10000.0

def add_basic_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ts" not in df.columns:
        raise ValueError("fixed_intersection_policy requires a ts column")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"]).copy().sort_values("ts").reset_index(drop=True)
    df["return_1_bps"] = df["close"].pct_change(1).fillna(0.0) * 10000.0
    df["return_3_bps"] = df["close"].pct_change(3).fillna(0.0) * 10000.0
    df["return_5_bps"] = df["close"].pct_change(5).fillna(0.0) * 10000.0
    df["return_15_bps"] = df["close"].pct_change(15).fillna(0.0) * 10000.0
    df["mean_20"] = df["close"].rolling(20, min_periods=5).mean()
    df["mean_60"] = df["close"].rolling(60, min_periods=10).mean()
    df["dist_mean_20_bps"] = (((df["close"] / df["mean_20"]) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 10000.0)
    df["dist_mean_60_bps"] = (((df["close"] / df["mean_60"]) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 10000.0)
    rolling_high = df["high"].rolling(96, min_periods=10).max(); rolling_low = df["low"].rolling(96, min_periods=10).min(); width = (rolling_high - rolling_low).replace(0.0, np.nan)
    df["range_position"] = ((df["close"] - rolling_low) / width).clip(0.0, 1.0).fillna(0.5)
    df["range_width_bps"] = (((rolling_high - rolling_low) / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 10000.0)
    df["volatility_20_bps"] = df["return_1_bps"].rolling(20, min_periods=5).std().fillna(0.0)
    df["volatility_60_bps"] = df["return_1_bps"].rolling(60, min_periods=10).std().fillna(0.0)
    df["volume_mean_60"] = df["volume"].rolling(60, min_periods=10).mean(); df["volume_std_60"] = df["volume"].rolling(60, min_periods=10).std().replace(0.0, np.nan)
    df["volume_z"] = (((df["volume"] - df["volume_mean_60"]) / df["volume_std_60"]).replace([np.inf, -np.inf], np.nan).fillna(0.0))
    df["relative_volume"] = (df["volume"] / df["volume_mean_60"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    delta = df["close"].diff(); gain = delta.clip(lower=0.0).rolling(14, min_periods=5).mean(); loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=5).mean().replace(0.0, np.nan)
    df["rsi"] = (100.0 - (100.0 / (1.0 + gain / loss))).replace([np.inf, -np.inf], np.nan).fillna(50.0)
    df["rolling_poc_proxy"] = df["close"].rolling(60, min_periods=10).mean()
    df["poc_distance_bps"] = (((df["close"] / df["rolling_poc_proxy"]) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 10000.0)
    df["upside_room_bps"] = (((rolling_high / df["close"]) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 10000.0)
    df["downside_room_bps"] = (((df["close"] / rolling_low) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 10000.0)
    return df

def score_buy_intersections(row: Dict[str, Any]) -> Dict[str, float]:
    range_pos = safe_float(row.get("range_position"), 0.5); rsi = safe_float(row.get("rsi"), 50.0); poc_distance_abs = abs(safe_float(row.get("poc_distance_bps"), 0.0)); upside_room = safe_float(row.get("upside_room_bps"), 0.0); downside_room = safe_float(row.get("downside_room_bps"), 0.0); vol = safe_float(row.get("volatility_60_bps"), 0.0); ret_5 = safe_float(row.get("return_5_bps"), 0.0); ret_15 = safe_float(row.get("return_15_bps"), 0.0); volume_z = safe_float(row.get("volume_z"), 0.0)
    scores = {}
    scores["cross_asset_analog_score"] = clamp(row.get("cross_asset_analog_score", row.get("chart_analog_similarity_buy_score", 0.50)))
    scores["previous_session_profile_score"] = clamp(row.get("previous_session_profile_score", 0.50))
    scores["upside_room_score"] = clamp(upside_room / 150.0)
    scores["low_volume_path_score"] = clamp((upside_room / 150.0) * 0.75 + (1.0 - clamp(downside_room / 180.0)) * 0.25)
    scores["poc_fair_value_stretch_score"] = clamp(poc_distance_abs / 120.0)
    scores["value_acceptance_score"] = clamp((1.0 - range_pos) * 0.60 + clamp((ret_5 + 20.0) / 50.0) * 0.40)
    scores["volatility_expansion_score"] = clamp(vol / 20.0) * (1.0 - clamp(max(0.0, vol - 45.0) / 45.0))
    scores["range_discount_score"] = clamp(1.0 - range_pos)
    scores["rsi_mean_reversion_score"] = clamp((55.0 - rsi) / 30.0)
    scores["fvg_fresh_zone_score"] = clamp(((-min(0.0, ret_15)) / 80.0) * 0.45 + clamp((ret_5 + 10.0) / 35.0) * 0.35 + (1.0 - clamp(volume_z / 4.0)) * 0.20)
    scores["buy_intersection_score"] = clamp(sum(float(w) * float(scores.get(n, 0.0)) for n, w in BUY_AGENT_WEIGHTS.items()))
    return scores

def score_sell_intersections(row: Dict[str, Any], entry_price: float, peak_price: float, profit_armed: bool) -> Dict[str, float]:
    close = safe_float(row.get("close", row.get("mid", 0.0)), 0.0); range_pos = safe_float(row.get("range_position"), 0.5); rsi = safe_float(row.get("rsi"), 50.0); ret_5 = safe_float(row.get("return_5_bps", row.get("momentum_5_bps", 0.0)), 0.0); ret_15 = safe_float(row.get("return_15_bps", row.get("momentum_15_bps", 0.0)), 0.0); dist20 = safe_float(row.get("dist_mean_20_bps"), 0.0); dist60 = safe_float(row.get("dist_mean_60_bps"), 0.0); vol = safe_float(row.get("volatility_60_bps", row.get("volatility_bps", 0.0)), 0.0); volume_z = safe_float(row.get("volume_z"), 0.0); upside_room = safe_float(row.get("upside_room_bps", row.get("room_to_target_bps", 0.0)), 0.0)
    move_from_entry_bps = bps_change(close, entry_price); giveback_from_peak_bps = bps_change(close, peak_price) if peak_price > 0 else 0.0
    scores = {}
    scores["low_stationarity_run_score"] = clamp((max(0.0, ret_15) / 80.0) * 0.60 + (vol / 30.0) * 0.40)
    scores["quant_forecast_strength_score"] = clamp(max(0.0, ret_5 + ret_15 * 0.5) / 80.0)
    scores["normal_volatility_cluster_score"] = clamp(vol / 18.0) * (1.0 - clamp(max(0.0, vol - 45.0) / 45.0))
    scores["upper_range_resistance_score"] = clamp(range_pos)
    scores["momentum_expansion_score"] = clamp((max(0.0, ret_5) / 45.0) * 0.45 + (max(0.0, ret_15) / 90.0) * 0.55)
    scores["mean_extension_score"] = clamp((max(0.0, dist20) / 70.0) * 0.45 + (max(0.0, dist60) / 100.0) * 0.55)
    scores["rsi_overbought_score"] = clamp((rsi - 60.0) / 25.0)
    scores["low_volume_path_exhaustion_score"] = clamp(1.0 - clamp(upside_room / 100.0))
    scores["volume_climax_score"] = clamp(volume_z / 3.0)
    scores["post_profit_flattening_score"] = clamp(abs(min(0.0, giveback_from_peak_bps)) / 35.0) if profit_armed else 0.0
    scores["sell_intersection_score"] = clamp(sum(float(w) * float(scores.get(n, 0.0)) for n, w in SELL_AGENT_WEIGHTS.items()))
    scores["move_from_entry_bps"] = move_from_entry_bps; scores["giveback_from_peak_bps"] = giveback_from_peak_bps
    return scores

def fixed_buy_decision(row: Dict[str, Any]) -> Dict[str, Any]:
    scores = score_buy_intersections(row); score = scores["buy_intersection_score"]
    decision = "STRONG_BUY" if score >= BUY_STRONG_THRESHOLD else "ALLOW_BUY" if score >= BUY_ACCEPTABLE_THRESHOLD else "WATCH" if score >= BUY_SHADOW_THRESHOLD else "NO_BUY"
    return {"decision": decision, "buy_intersection_score": score, **scores}

def fixed_sell_decision(row: Dict[str, Any], entry_price: float, peak_price: float, profit_armed: bool) -> Dict[str, Any]:
    scores = score_sell_intersections(row, entry_price=entry_price, peak_price=peak_price, profit_armed=profit_armed); score = scores["sell_intersection_score"]
    decision = "SELL" if score >= SELL_EXIT_THRESHOLD else "ARM_TRAIL" if score >= SELL_ARM_TRAIL_THRESHOLD else "TIGHTEN" if score >= SELL_TIGHTEN_THRESHOLD else "HOLD"
    return {"decision": decision, "sell_intersection_score": score, **scores}

def simulate_fixed_policy_on_candles(*, frame: pd.DataFrame, product_id: str, timeframe: str, fee_and_slippage_bps: float = DEFAULT_FEE_AND_SLIPPAGE_BPS, min_prefix_rows: int = 120, max_hold_bars: int = 96, stop_loss_bps: float = SIM_HARD_STOP_BPS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = add_basic_indicators(frame)
    if "product_id" in df.columns:
        df = df[df["product_id"].astype(str).eq(str(product_id))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    trades = []; snapshots = []; in_trade = False; entry_idx = -1; entry_ts = entry_price = peak_price = trough_price = entry_buy_score = 0.0; profit_armed = False
    for i in range(int(min_prefix_rows), len(df)):
        row = df.iloc[i].to_dict(); ts = safe_float(row.get("ts"), 0.0); close = safe_float(row.get("close"), 0.0); high = safe_float(row.get("high"), close); low = safe_float(row.get("low"), close)
        if close <= 0: continue
        if not in_trade:
            buy = fixed_buy_decision(row); snapshots.append({"ts": ts, "product_id": product_id, "timeframe": timeframe, "side": "BUY_SCAN", **buy})
            if buy["decision"] in {"STRONG_BUY", "ALLOW_BUY"}:
                in_trade = True; entry_idx = i; entry_ts = ts; entry_price = close; peak_price = close; trough_price = close; profit_armed = False; entry_buy_score = buy["buy_intersection_score"]
            continue
        peak_price = max(peak_price, high); trough_price = min(trough_price, low); favorable_bps = bps_change(peak_price, entry_price); adverse_bps = bps_change(trough_price, entry_price)
        if favorable_bps >= SIM_MIN_PROFIT_TARGET_BPS: profit_armed = True
        sell = fixed_sell_decision(row, entry_price=entry_price, peak_price=peak_price, profit_armed=profit_armed); snapshots.append({"ts": ts, "product_id": product_id, "timeframe": timeframe, "side": "SELL_SCAN", **sell})
        exit_reason = ""; exit_price = None
        if adverse_bps <= float(stop_loss_bps): exit_reason = "stop_loss"; exit_price = close
        elif profit_armed and sell["decision"] == "SELL": exit_reason = "sell_score_exit"; exit_price = close
        elif profit_armed and sell["decision"] == "ARM_TRAIL" and sell.get("giveback_from_peak_bps", 0.0) <= -25.0: exit_reason = "armed_trailing_giveback"; exit_price = close
        elif i - entry_idx >= int(max_hold_bars): exit_reason = "max_hold"; exit_price = close
        if exit_price is not None:
            gross_bps = bps_change(exit_price, entry_price); net_bps = gross_bps - float(fee_and_slippage_bps)
            trades.append({"entry_ts": entry_ts, "exit_ts": ts, "product_id": product_id, "timeframe": timeframe, "entry_price": entry_price, "exit_price": exit_price, "gross_bps": gross_bps, "net_bps": net_bps, "max_favorable_bps": favorable_bps, "max_adverse_bps": adverse_bps, "held_bars": i - entry_idx, "entry_buy_score": entry_buy_score, "exit_sell_score": sell["sell_intersection_score"], "exit_reason": exit_reason, "won": int(net_bps > 0)})
            in_trade = False
    return pd.DataFrame(trades), pd.DataFrame(snapshots)

def summarize_fixed_policy_trades(trades: pd.DataFrame, product_id: str, timeframe: str) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {"product_id": product_id, "timeframe": timeframe, "trade_count": 0, "win_rate": 0.0, "avg_net_bps": 0.0, "median_net_bps": 0.0, "total_net_bps": 0.0, "profit_factor": 0.0, "trades_per_day": 0.0, "approved_for_live": False, "reason": "no_simulated_trades"}
    t = trades.copy(); t["net_bps"] = pd.to_numeric(t["net_bps"], errors="coerce").fillna(0.0); wins = t[t["net_bps"] > 0.0]["net_bps"]; losses = t[t["net_bps"] <= 0.0]["net_bps"]
    gross_profit = float(wins.sum()); gross_loss = abs(float(losses.sum())); profit_factor = gross_profit / max(gross_loss, 1e-9)
    min_ts = float(pd.to_numeric(t["entry_ts"], errors="coerce").min()); max_ts = float(pd.to_numeric(t["exit_ts"], errors="coerce").max()); days = max(1.0 / 24.0, (max_ts - min_ts) / 86400.0)
    trade_count = int(len(t)); win_rate = float((t["net_bps"] > 0.0).mean()); avg_net_bps = float(t["net_bps"].mean()); median_net_bps = float(t["net_bps"].median()); total_net_bps = float(t["net_bps"].sum()); trades_per_day = float(trade_count / days)
    approved = bool(trade_count >= 20 and win_rate >= 0.52 and avg_net_bps > 0.0 and median_net_bps >= -5.0 and profit_factor >= 1.15 and trades_per_day >= 0.05)
    return {"product_id": product_id, "timeframe": timeframe, "trade_count": trade_count, "win_rate": win_rate, "avg_net_bps": avg_net_bps, "median_net_bps": median_net_bps, "total_net_bps": total_net_bps, "profit_factor": profit_factor, "trades_per_day": trades_per_day, "approved_for_live": approved, "reason": f"fixed_policy_summary;trades={trade_count};win_rate={win_rate:.3f};avg={avg_net_bps:.2f};median={median_net_bps:.2f};pf={profit_factor:.3f};tpd={trades_per_day:.3f}"}
