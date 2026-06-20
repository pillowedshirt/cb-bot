"""Worker-safe historical replay engine.

This module intentionally avoids TradingBot instance state so it can run inside
ProcessPoolExecutor workers.  The main bot prepares candle caches and serializes
a small payload; this module reads the cache and writes a worker output CSV.
"""

import csv
import os
import statistics
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from price_action_context import build_price_action_context
from session_liquidity import build_session_liquidity_signal

TZ = ZoneInfo("America/Phoenix")


@dataclass
class ReplayCandle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class ReplaySignal:
    score: float = 0.0
    estimated_prob_up: float = 0.0
    expected_net_edge_bps: float = 0.0
    target_bps: float = 0.0
    cost_bps: float = 0.0
    spread_bps: float = 0.0
    session_liquidity_setup: str = ""
    value_acceptance_state: str = ""
    volume_node_state: str = ""
    poc_distance_bps: float = 0.0
    volume_profile_leader_buy_score: float = 0.0
    volume_profile_leader_wait_score: float = 0.0
    price_action_buy_score: float = 0.0
    market_structure_buy_score: float = 0.0
    quant_buy_score: float = 0.0
    structure_state: str = ""
    fvg_state: str = ""


@dataclass
class ReplayEngineConfig:
    entry_fee_bps: float
    exit_fee_bps: float
    max_spread_bps: float
    est_slippage_bps: float
    est_adverse_fill_bps: float
    min_net_profit_bps: float
    max_position_loss_pct: float
    synthetic_notional_usd: float
    level8_min_hold_sec: float
    level8_max_hold_sec: float
    scalp_pullback_pct: float
    hist_replay_max_runtime_sec: float
    hist_replay_max_candidates_per_pass: int
    hist_replay_min_prefix_15m: int
    hist_replay_min_prefix_1h: int
    hist_replay_forward_bars_15m: List[int]
    hist_replay_forward_bars_1h: List[int]
    hist_replay_step_bars_15m: int
    hist_replay_step_bars_1h: int
    min_target_over_cost_bps: float
    min_target_to_cost_ratio: float
    min_score_for_calibration: float
    min_probability_for_calibration: float
    min_expected_edge_bps_for_calibration: float
    block_near_poc_chop: bool
    max_poc_distance_for_chop_bps: float
    resist_buffer_bps: float = 20.0
    micro_trend_lookback_min: int = 12
    micro_trend_down_bps: float = -25.0
    min_required_net_edge_bps: float = 15.0
    vwap_reclaim_buffer_bps: float = 3.0
    primary_fee_model: str = "binance_us"
    comparison_fee_model: str = "coinbase_legacy"
    comparison_entry_fee_bps: float = 0.0
    comparison_exit_fee_bps: float = 2.0
    enable_exchange_fee_comparison: bool = True
    enable_fee_scenario_matrix: bool = True
    fee_scenario_matrix: Optional[Dict[str, Dict[str, float]]] = None
    enable_strategy_variant_replay: bool = True
    strategy_variants: Optional[List[str]] = None
    high_win_min_target_to_cost_ratio: float = 2.20
    high_win_min_target_over_cost_bps: float = 80.0
    high_win_min_probability: float = 0.58
    high_win_min_score: float = 62.0
    high_win_max_spread_bps: float = 12.0
    high_win_require_positive_momentum_5: bool = True
    high_win_require_positive_momentum_15: bool = True
    high_win_min_momentum_5_bps: float = 4.0
    high_win_min_momentum_15_bps: float = 8.0
    high_win_block_inside_value_high_volume: bool = True
    high_win_stop_loss_pct: float = 0.006
    high_win_profit_pullback_pct: float = 0.0015
    high_win_min_profit_over_cost_bps: float = 25.0
    high_win_v2_min_target_to_cost_ratio: float = 1.35
    high_win_v2_min_target_over_cost_bps: float = 35.0
    high_win_v2_min_probability: float = 0.54
    high_win_v2_min_score: float = 54.0
    high_win_v2_max_spread_bps: float = 22.0
    high_win_v2_require_momentum_either: bool = True
    high_win_v2_min_momentum_either_bps: float = 0.0
    high_win_v2_block_low_room: bool = True
    high_win_v2_block_low_volume_above_value: bool = True
    high_win_v2_stop_loss_pct: float = 0.004
    high_win_v2_profit_pullback_pct: float = 0.0018
    coinbase_survival_min_target_to_cost_ratio: float = 2.25
    coinbase_survival_min_target_over_cost_bps: float = 140.0
    coinbase_survival_min_probability: float = 0.56
    coinbase_survival_min_score: float = 58.0
    coinbase_survival_max_spread_bps: float = 22.0
    coinbase_survival_stop_loss_pct: float = 0.0035
    coinbase_survival_profit_pullback_pct: float = 0.0015
    low_fee_scalp_min_target_to_cost_ratio: float = 1.15
    low_fee_scalp_min_target_over_cost_bps: float = 15.0
    low_fee_scalp_min_probability: float = 0.52
    low_fee_scalp_min_score: float = 50.0
    low_fee_scalp_max_spread_bps: float = 22.0
    low_fee_scalp_stop_loss_pct: float = 0.004
    low_fee_scalp_profit_pullback_pct: float = 0.0015
    enable_early_adverse_exit: bool = True
    early_adverse_exit_bps: float = 18.0
    early_adverse_min_age_bars: int = 2
    early_adverse_requires_negative_momentum: bool = True
    early_adverse_momentum_lookback: int = 5


def _dt_mst(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")


def _clip_score(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return float(max(lo, min(hi, float(value))))
    except Exception:
        return 0.0


def _score_from_bps(value_bps: float, *, center_bps: float, width_bps: float) -> float:
    width = max(float(width_bps), 1e-9)
    return _clip_score(50.0 + ((float(value_bps) - float(center_bps)) / width) * 50.0)


def _fee_usd(notional: float, fee_bps: float) -> float:
    return float(notional) * float(fee_bps) / 10000.0


def _same_timing_fee_result(
    *,
    notional: float,
    qty: float,
    exit_price: float,
    entry_fee_bps: float,
    exit_fee_bps: float,
) -> Dict[str, float]:
    entry_fee = _fee_usd(notional, entry_fee_bps)
    gross_proceeds = float(qty) * float(exit_price)
    exit_fee = _fee_usd(gross_proceeds, exit_fee_bps)
    net_pnl = gross_proceeds - float(notional) - entry_fee - exit_fee
    net_bps = (net_pnl / max(float(notional), 1e-12)) * 10000.0
    return {
        "entry_fee_usd": float(entry_fee),
        "exit_fee_usd": float(exit_fee),
        "net_pnl_usd": float(net_pnl),
        "net_pnl_bps": float(net_bps),
        "would_have_won": int(net_pnl > 0.0),
    }


def _required_exit_price_for_net_gain(*, effective_entry_price: float, exit_fee_bps: float, est_slippage_bps: float, est_adverse_fill_bps: float, min_net_gain_bps: float) -> float:
    total_bps = float(exit_fee_bps) + float(est_slippage_bps) + float(est_adverse_fill_bps) + float(min_net_gain_bps)
    return float(effective_entry_price) * (1.0 + total_bps / 10000.0)


def _read_replay_candles(path: str, product_id: str, min_ts: float = 0.0) -> List[ReplayCandle]:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return []
    frame = pd.read_csv(path)
    if frame.empty or "product_id" not in frame.columns:
        return []
    frame = frame[frame["product_id"].astype(str).eq(str(product_id))].copy()
    if frame.empty:
        return []
    frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
    frame = frame.dropna(subset=["ts"])
    frame = frame[frame["ts"] >= float(min_ts)].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).sort_values("ts")
    frame = frame.drop_duplicates(subset=["ts"], keep="last")
    return [ReplayCandle(ts=int(float(r["ts"])), open=float(r["open"]), high=float(r["high"]), low=float(r["low"]), close=float(r["close"]), volume=float(r.get("volume", 0.0) or 0.0)) for _, r in frame.iterrows()]


def _write_replay_rows(output_path: str, columns: List[str], rows: List[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return len(rows)


def _round_trip_cost_bps(config: ReplayEngineConfig, spread_bps: float) -> float:
    return float(config.entry_fee_bps) + float(config.exit_fee_bps) + float(spread_bps) + float(config.est_slippage_bps) + float(config.est_adverse_fill_bps)


def _recent_close_momentum_bps(candles: List[ReplayCandle], lookback: int) -> float:
    if len(candles) <= int(lookback):
        return 0.0
    first = float(candles[-int(lookback) - 1].close)
    last = float(candles[-1].close)
    return 0.0 if first <= 0 else ((last / first) - 1.0) * 10000.0


def _recent_range_position_score(candles: List[ReplayCandle], lookback: int = 20) -> float:
    recent = candles[-int(lookback):]
    if len(recent) < 3:
        return 50.0
    lo = min(float(c.low) for c in recent)
    hi = max(float(c.high) for c in recent)
    close = float(recent[-1].close)
    if hi <= lo:
        return 50.0
    pos = (close - lo) / (hi - lo)
    if pos < 0.20:
        return 35.0
    if pos > 0.90:
        return 45.0
    return _clip_score(45.0 + pos * 45.0)


def _simple_macro_levels(candles: List[ReplayCandle]) -> Dict[str, float]:
    if not candles:
        return {}
    recent = candles[-min(len(candles), 192):]
    highs = [float(c.high) for c in recent]
    lows = [float(c.low) for c in recent]
    closes = [float(c.close) for c in recent]
    return {"range_high": max(highs), "range_low": min(lows), "mid": closes[-1], "poc": statistics.median(closes) if closes else closes[-1]}


def _target_move_bps_from_room(*, mid: float, levels_day: Dict[str, float], levels_week: Dict[str, float], sigma_bps: Optional[float]) -> float:
    candidates = []
    for levels in [levels_day, levels_week]:
        high = float(levels.get("range_high", 0.0) or 0.0) if levels else 0.0
        if high > float(mid) > 0:
            candidates.append(((high / float(mid)) - 1.0) * 10000.0)
    room_bps = max(candidates) if candidates else 0.0
    if sigma_bps is not None and sigma_bps > 0:
        room_bps = max(room_bps, float(sigma_bps) * 2.0)
    return max(0.0, float(room_bps))


def _support_proximity_score(mid: float, levels_day: Dict[str, float], levels_week: Dict[str, float]) -> float:
    supports = []
    for levels in [levels_day, levels_week]:
        low = float(levels.get("range_low", 0.0) or 0.0)
        if 0 < low <= mid:
            supports.append(((mid / low) - 1.0) * 10000.0)
    if not supports:
        return 50.0
    nearest = min(supports)
    if nearest < 15:
        return 75.0
    if nearest < 50:
        return 65.0
    if nearest < 120:
        return 55.0
    return 45.0


def _room_score(mid: float, levels_day: Dict[str, float], levels_week: Dict[str, float], buffer_bps: float) -> Tuple[float, str]:
    target_bps = _target_move_bps_from_room(mid=mid, levels_day=levels_day, levels_week=levels_week, sigma_bps=None)
    if target_bps <= buffer_bps:
        return 35.0, "room_too_small"
    return _clip_score(45.0 + min(55.0, target_bps * 0.20)), f"room_bps={target_bps:.2f}"



def _to_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        if is_dataclass(value):
            return asdict(value)
    except Exception:
        pass
    try:
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
    except Exception:
        pass
    return {}


def _map_get(value: Any, key: str, default: Any = None) -> Any:
    mapping = _to_mapping(value)
    return mapping.get(key, default)

def _estimate_prob_up(score: float, spread_bps: float, expected_edge_bps: float) -> float:
    raw = 0.50 + (float(score) - 50.0) / 250.0 + float(expected_edge_bps) / 2000.0 - max(0.0, float(spread_bps) - 10.0) / 2000.0
    return max(0.01, min(0.99, raw))


def _build_worker_signal(*, product_id: str, candles: List[ReplayCandle], weekly_candles: Optional[List[ReplayCandle]], spread_bps: float, config: ReplayEngineConfig) -> ReplaySignal:
    mid = float(candles[-1].close)
    closes = [float(c.close) for c in candles if float(c.close) > 0]
    levels_day = _simple_macro_levels(candles)
    levels_week = _simple_macro_levels(weekly_candles or candles)
    sigma_bps = None
    if len(closes) >= 20:
        rets = [(closes[i] / closes[i - 1] - 1.0) * 10000.0 for i in range(1, len(closes)) if closes[i - 1] > 0]
        if rets:
            sigma_bps = float(np.std(rets[-60:])) if len(rets) >= 2 else 0.0
    target_bps = _target_move_bps_from_room(mid=mid, levels_day=levels_day, levels_week=levels_week, sigma_bps=sigma_bps)
    cost_bps = _round_trip_cost_bps(config, spread_bps)
    expected_net_edge_bps = float(target_bps - cost_bps)
    support_score = _support_proximity_score(mid, levels_day, levels_week)
    room_score, _ = _room_score(mid, levels_day, levels_week, config.resist_buffer_bps)
    momentum_score = _score_from_bps(_recent_close_momentum_bps(candles, 5), center_bps=0.0, width_bps=35.0) * 0.60 + _score_from_bps(_recent_close_momentum_bps(candles, 15), center_bps=0.0, width_bps=65.0) * 0.40
    range_position_score = _recent_range_position_score(candles, lookback=20)
    edge_score = _score_from_bps(expected_net_edge_bps, center_bps=0.0, width_bps=max(35.0, float(config.min_required_net_edge_bps)))
    candle_dicts = [{"ts": c.ts, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
    try:
        session_liquidity = _to_mapping(
            build_session_liquidity_signal(
                product_id=product_id,
                candles=candle_dicts,
                current_price=mid,
                spread_bps=float(spread_bps),
                cost_bps=float(cost_bps),
                projected_forward_gain_bps=float(target_bps),
            )
        )
    except Exception:
        session_liquidity = {}
    try:
        price_action_context = _to_mapping(
            build_price_action_context(
                product_id=product_id,
                candles=candle_dicts,
                current_price=mid,
                spread_bps=float(spread_bps),
                cost_bps=float(cost_bps),
                projected_forward_gain_bps=float(target_bps),
            )
        )
    except Exception:
        price_action_context = {}
    session_liquidity_score = float(session_liquidity.get("best_buy_score", 0.0) or 0.0)
    session_liquidity_confidence = float(session_liquidity.get("confidence", 0.0) or 0.0)
    price_action_buy_score = float(price_action_context.get("candle_context_buy_score", 0.0) or 0.0)
    market_structure_buy_score = float(price_action_context.get("market_structure_buy_score", 0.0) or 0.0)
    volume_profile_buy_score = float(price_action_context.get("volume_profile_buy_score", 0.0) or 0.0)
    volume_profile_leader_buy_score = float(price_action_context.get("volume_profile_leader_buy_score", volume_profile_buy_score) or 0.0)
    volume_profile_leader_wait_score = float(price_action_context.get("volume_profile_leader_wait_score", 0.50) or 0.50)
    price_action_confidence = float(price_action_context.get("candle_context_confidence", 0.0) or 0.0)
    combined_context_bonus = session_liquidity_score * session_liquidity_confidence * 10.0 + price_action_buy_score * price_action_confidence * 10.0 + market_structure_buy_score * 8.0 + volume_profile_buy_score * 8.0
    score = _clip_score(support_score * 0.16 + room_score * 0.18 + momentum_score * 0.20 + range_position_score * 0.12 + edge_score * 0.22 + combined_context_bonus - max(0.0, float(spread_bps) - 6.0) * 0.80 - max(0.0, float(cost_bps) - 50.0) * 0.10)
    return ReplaySignal(score=score, estimated_prob_up=_estimate_prob_up(score, spread_bps, expected_net_edge_bps), expected_net_edge_bps=expected_net_edge_bps, target_bps=target_bps, cost_bps=cost_bps, spread_bps=spread_bps, session_liquidity_setup=str(session_liquidity.get("strongest_setup", session_liquidity.get("best_setup", ""))), value_acceptance_state=str(price_action_context.get("value_acceptance_state", "")), volume_node_state=str(price_action_context.get("volume_node_state", "")), poc_distance_bps=float(price_action_context.get("poc_distance_bps", 0.0) or 0.0), volume_profile_leader_buy_score=volume_profile_leader_buy_score, volume_profile_leader_wait_score=volume_profile_leader_wait_score, price_action_buy_score=price_action_buy_score, market_structure_buy_score=market_structure_buy_score, quant_buy_score=float(edge_score / 100.0), structure_state=str(price_action_context.get("structure_state", "")), fvg_state=str(price_action_context.get("fvg_state", "")))


def _qualified_candidate(signal: ReplaySignal, config: ReplayEngineConfig) -> Tuple[bool, str, float]:
    score = float(signal.score); probability = float(signal.estimated_prob_up); expected_edge = float(signal.expected_net_edge_bps); target_bps = float(signal.target_bps); cost_bps = float(signal.cost_bps)
    target_over_cost = target_bps - cost_bps
    target_to_cost = target_bps / max(cost_bps, 1e-9)
    quality = max(0.0, min(25.0, (score - float(config.min_score_for_calibration)) * 100.0)) + max(0.0, min(25.0, (probability - float(config.min_probability_for_calibration)) * 100.0)) + max(-20.0, min(30.0, target_over_cost * 0.20)) + max(-20.0, min(20.0, expected_edge * 0.10))
    reasons = []
    if target_over_cost < float(config.min_target_over_cost_bps): reasons.append(f"target_over_cost_too_low target={target_bps:.2f};cost={cost_bps:.2f};over={target_over_cost:.2f}")
    if target_to_cost < float(config.min_target_to_cost_ratio): reasons.append(f"target_to_cost_ratio_too_low ratio={target_to_cost:.3f};min={float(config.min_target_to_cost_ratio):.3f}")
    if score < float(config.min_score_for_calibration): reasons.append(f"score_too_low score={score:.3f}")
    if probability < float(config.min_probability_for_calibration): reasons.append(f"probability_too_low probability={probability:.3f}")
    if expected_edge < float(config.min_expected_edge_bps_for_calibration): reasons.append(f"expected_edge_too_low expected_edge={expected_edge:.2f}")
    if bool(config.block_near_poc_chop) and "inside" in str(signal.value_acceptance_state).lower() and "high" in str(signal.volume_node_state).lower() and abs(float(signal.poc_distance_bps)) <= float(config.max_poc_distance_for_chop_bps):
        reasons.append(f"near_poc_high_volume_chop value={signal.value_acceptance_state};node={signal.volume_node_state};poc_distance={abs(float(signal.poc_distance_bps)):.2f}")
    if reasons:
        return False, ";".join(reasons), float(quality)
    return True, f"qualified target_over_cost={target_over_cost:.2f};target_to_cost={target_to_cost:.3f};score={score:.3f};probability={probability:.3f};expected_edge={expected_edge:.2f}", float(quality)



def _variant_entry_filter(*, variant: str, signal: ReplaySignal, candles: List[ReplayCandle], config: ReplayEngineConfig) -> Tuple[bool, str, Dict[str, float]]:
    momentum_5 = _recent_close_momentum_bps(candles, lookback=5)
    momentum_15 = _recent_close_momentum_bps(candles, lookback=15)
    target_to_cost = float(signal.target_bps) / max(float(signal.cost_bps), 1e-9)
    target_over_cost = float(signal.target_bps) - float(signal.cost_bps)
    metrics = {"target_to_cost_ratio": float(target_to_cost), "target_over_cost_bps": float(target_over_cost), "momentum_5_bps": float(momentum_5), "momentum_15_bps": float(momentum_15)}
    variant = str(variant or "baseline")
    if variant in {"baseline", "", "current"}:
        return True, "baseline_allowed", metrics

    def fail_reasons(*, min_ratio: float, min_over: float, min_prob: float, min_score: float, max_spread: float, require_momentum_either: bool, min_momentum_either: float) -> List[str]:
        reasons: List[str] = []
        if target_to_cost < float(min_ratio): reasons.append(f"target_to_cost_low {target_to_cost:.3f}")
        if target_over_cost < float(min_over): reasons.append(f"target_over_cost_low {target_over_cost:.2f}")
        if float(signal.estimated_prob_up) < float(min_prob): reasons.append(f"probability_low {float(signal.estimated_prob_up):.3f}")
        if float(signal.score) < float(min_score): reasons.append(f"score_low {float(signal.score):.2f}")
        if float(signal.spread_bps) > float(max_spread): reasons.append(f"spread_high {float(signal.spread_bps):.2f}")
        if require_momentum_either and max(momentum_5, momentum_15) < float(min_momentum_either): reasons.append(f"momentum_either_low m5={momentum_5:.2f};m15={momentum_15:.2f}")
        return reasons

    if variant == "high_win_rate_v1":
        reasons = fail_reasons(min_ratio=float(config.high_win_min_target_to_cost_ratio), min_over=float(config.high_win_min_target_over_cost_bps), min_prob=float(config.high_win_min_probability), min_score=float(config.high_win_min_score), max_spread=float(config.high_win_max_spread_bps), require_momentum_either=False, min_momentum_either=0.0)
        if bool(config.high_win_require_positive_momentum_5) and momentum_5 < float(config.high_win_min_momentum_5_bps): reasons.append(f"momentum_5_low {momentum_5:.2f}")
        if bool(config.high_win_require_positive_momentum_15) and momentum_15 < float(config.high_win_min_momentum_15_bps): reasons.append(f"momentum_15_low {momentum_15:.2f}")
    elif variant == "high_win_rate_v2":
        reasons = fail_reasons(min_ratio=float(config.high_win_v2_min_target_to_cost_ratio), min_over=float(config.high_win_v2_min_target_over_cost_bps), min_prob=float(config.high_win_v2_min_probability), min_score=float(config.high_win_v2_min_score), max_spread=float(config.high_win_v2_max_spread_bps), require_momentum_either=bool(config.high_win_v2_require_momentum_either), min_momentum_either=float(config.high_win_v2_min_momentum_either_bps))
        if bool(config.high_win_v2_block_low_room) and "low_room" in str(_setup_tag_from_signal(signal)[1]).lower(): reasons.append("low_room_blocked")
        if bool(config.high_win_v2_block_low_volume_above_value) and "above" in str(signal.value_acceptance_state).lower() and "low" in str(signal.volume_node_state).lower(): reasons.append("above_value_low_volume_blocked")
    elif variant == "high_fee_survival_v1":
        reasons = fail_reasons(min_ratio=float(config.coinbase_survival_min_target_to_cost_ratio), min_over=float(config.coinbase_survival_min_target_over_cost_bps), min_prob=float(config.coinbase_survival_min_probability), min_score=float(config.coinbase_survival_min_score), max_spread=float(config.coinbase_survival_max_spread_bps), require_momentum_either=True, min_momentum_either=0.0)
    elif variant == "low_fee_scalp_v1":
        reasons = fail_reasons(min_ratio=float(config.low_fee_scalp_min_target_to_cost_ratio), min_over=float(config.low_fee_scalp_min_target_over_cost_bps), min_prob=float(config.low_fee_scalp_min_probability), min_score=float(config.low_fee_scalp_min_score), max_spread=float(config.low_fee_scalp_max_spread_bps), require_momentum_either=True, min_momentum_either=0.0)
    else:
        reasons = [f"unknown_variant {variant}"]
    if "inside" in str(signal.value_acceptance_state).lower() and "high" in str(signal.volume_node_state).lower() and abs(float(signal.poc_distance_bps)) <= float(config.max_poc_distance_for_chop_bps):
        reasons.append("inside_high_volume_near_poc_chop")
    if reasons:
        return False, ";".join(reasons), metrics
    return True, f"{variant}_allowed", metrics


def _variant_sell_settings(variant: str, config: ReplayEngineConfig) -> Dict[str, float]:
    variant = str(variant or "baseline")
    if variant == "high_win_rate_v1":
        return {"stop_loss_pct": float(config.high_win_stop_loss_pct), "pullback_pct": float(config.high_win_profit_pullback_pct), "early_adverse_exit_bps": float(config.early_adverse_exit_bps)}
    if variant == "high_win_rate_v2":
        return {"stop_loss_pct": float(config.high_win_v2_stop_loss_pct), "pullback_pct": float(config.high_win_v2_profit_pullback_pct), "early_adverse_exit_bps": float(config.early_adverse_exit_bps)}
    if variant == "high_fee_survival_v1":
        return {"stop_loss_pct": float(config.coinbase_survival_stop_loss_pct), "pullback_pct": float(config.coinbase_survival_profit_pullback_pct), "early_adverse_exit_bps": float(config.early_adverse_exit_bps)}
    if variant == "low_fee_scalp_v1":
        return {"stop_loss_pct": float(config.low_fee_scalp_stop_loss_pct), "pullback_pct": float(config.low_fee_scalp_profit_pullback_pct), "early_adverse_exit_bps": float(config.early_adverse_exit_bps)}
    return {"stop_loss_pct": float(config.max_position_loss_pct), "pullback_pct": float(config.scalp_pullback_pct), "early_adverse_exit_bps": 0.0}

def _fee_scenario_matrix_results(*, notional: float, qty: float, exit_price: float, fee_scenario_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, fees in (fee_scenario_matrix or {}).items():
        entry_fee_bps = float(fees.get("entry_fee_bps", 0.0) or 0.0)
        exit_fee_bps = float(fees.get("exit_fee_bps", 0.0) or 0.0)
        result = _same_timing_fee_result(notional=notional, qty=qty, exit_price=exit_price, entry_fee_bps=entry_fee_bps, exit_fee_bps=exit_fee_bps)
        safe = str(name).lower().replace("-", "_").replace("/", "_")
        out[f"{safe}_entry_fee_bps"] = entry_fee_bps
        out[f"{safe}_exit_fee_bps"] = exit_fee_bps
        out[f"{safe}_net_pnl_bps"] = float(result["net_pnl_bps"])
        out[f"{safe}_net_pnl_usd"] = float(result["net_pnl_usd"])
        out[f"{safe}_would_have_won"] = int(result["would_have_won"])
    return out

def _setup_tag_from_signal(signal: ReplaySignal) -> Tuple[str, str]:
    parts = []
    value_state = str(signal.value_acceptance_state or ""); volume_node = str(signal.volume_node_state or "")
    if "inside" in value_state.lower(): parts.append("inside_value")
    if "above" in value_state.lower(): parts.append("above_value")
    if "below" in value_state.lower(): parts.append("below_value")
    if "low" in volume_node.lower(): parts.append("low_volume_node")
    if "high" in volume_node.lower(): parts.append("high_volume_node")
    if signal.structure_state: parts.append(str(signal.structure_state)[:24])
    if signal.fvg_state: parts.append(str(signal.fvg_state)[:24])
    if not parts: parts.append("generic")
    regime = "wide_spread" if float(signal.spread_bps) > 20 else "high_room" if float(signal.target_bps) > 150 else "low_room" if float(signal.target_bps) < 50 else "normal"
    return "|".join(parts[:4]), regime


def _simulate_candidate(*, product_id: str, timeframe: str, granularity: str, prefix: List[ReplayCandle], future: List[ReplayCandle], replay_source: str, config: ReplayEngineConfig, columns: List[str], variant: str = "baseline") -> Optional[Dict[str, Any]]:
    if not prefix or not future:
        return None
    entry = prefix[-1]; replay_ts = int(entry.ts); entry_price = float(entry.close)
    if entry_price <= 0:
        return None
    signal = _build_worker_signal(product_id=product_id, candles=prefix, weekly_candles=prefix, spread_bps=float(config.max_spread_bps), config=config)
    variant_allowed, variant_reason, variant_metrics = _variant_entry_filter(variant=variant, signal=signal, candles=prefix, config=config)
    if not variant_allowed:
        return None
    setup_tag, regime_tag = _setup_tag_from_signal(signal)
    qualified, qualification_reason, replay_quality = _qualified_candidate(signal, config)
    notional = float(config.synthetic_notional_usd)
    entry_fee = _fee_usd(notional, config.entry_fee_bps)
    qty = notional / max(entry_price, 1e-12)
    all_in_entry = (notional + entry_fee) / max(qty, 1e-12)
    min_exit = _required_exit_price_for_net_gain(effective_entry_price=all_in_entry, exit_fee_bps=config.exit_fee_bps, est_slippage_bps=config.est_slippage_bps, est_adverse_fill_bps=config.est_adverse_fill_bps, min_net_gain_bps=config.min_net_profit_bps)
    sell_settings = _variant_sell_settings(str(variant), config)
    stop_loss_pct = float(sell_settings["stop_loss_pct"])
    pullback_pct = float(sell_settings["pullback_pct"])
    early_adverse_exit_bps = float(sell_settings.get("early_adverse_exit_bps", 0.0) or 0.0)
    hard_stop = all_in_entry * (1.0 - float(stop_loss_pct))
    peak = entry_price; trough = entry_price; profit_armed = False; exit_ts = None; exit_price = None; exit_reason = ""
    bars_seen = 0
    for candle in future:
        age = float(candle.ts) - float(replay_ts); high = float(candle.high); low = float(candle.low); close = float(candle.close)
        bars_seen += 1
        if high > 0: peak = max(peak, high)
        if low > 0: trough = min(trough, low)
        adverse_bps = ((low / entry_price) - 1.0) * 10000.0 if low > 0 and entry_price > 0 else 0.0
        if bool(config.enable_early_adverse_exit) and early_adverse_exit_bps > 0 and not profit_armed and bars_seen >= int(config.early_adverse_min_age_bars) and adverse_bps <= -abs(float(early_adverse_exit_bps)):
            recent_momentum = _recent_close_momentum_bps(prefix + future[:bars_seen], lookback=int(config.early_adverse_momentum_lookback))
            if (not bool(config.early_adverse_requires_negative_momentum)) or recent_momentum < 0:
                exit_ts = float(candle.ts); exit_price = close if close > 0 else low; exit_reason = "historical_early_adverse_exit"; break
        if low > 0 and low <= hard_stop:
            exit_ts = float(candle.ts); exit_price = hard_stop; exit_reason = "historical_hard_stop"; break
        if age >= float(config.level8_min_hold_sec) and high >= min_exit:
            profit_armed = True
        if profit_armed and close > 0 and close <= float(peak) * (1.0 - float(pullback_pct)):
            exit_ts = float(candle.ts); exit_price = close; exit_reason = "historical_profit_pullback"; break
        if age >= float(config.level8_max_hold_sec):
            exit_ts = float(candle.ts); exit_price = close; exit_reason = "historical_max_hold_exit"; break
    if exit_ts is None:
        exit_ts = float(future[-1].ts); exit_price = float(future[-1].close); exit_reason = "historical_window_end"
    gross_proceeds = qty * float(exit_price)
    primary_fee_result = _same_timing_fee_result(
        notional=notional,
        qty=qty,
        exit_price=float(exit_price),
        entry_fee_bps=float(config.entry_fee_bps),
        exit_fee_bps=float(config.exit_fee_bps),
    )
    if bool(config.enable_exchange_fee_comparison):
        comparison_fee_result = _same_timing_fee_result(
            notional=notional,
            qty=qty,
            exit_price=float(exit_price),
            entry_fee_bps=float(config.comparison_entry_fee_bps),
            exit_fee_bps=float(config.comparison_exit_fee_bps),
        )
    else:
        comparison_fee_result = dict(primary_fee_result)
    entry_fee = float(primary_fee_result["entry_fee_usd"])
    exit_fee = float(primary_fee_result["exit_fee_usd"])
    net_pnl = float(primary_fee_result["net_pnl_usd"])
    net_bps = float(primary_fee_result["net_pnl_bps"])
    comparison_net_improvement_usd = float(comparison_fee_result["net_pnl_usd"]) - float(primary_fee_result["net_pnl_usd"])
    comparison_net_improvement_bps = float(comparison_fee_result["net_pnl_bps"]) - float(primary_fee_result["net_pnl_bps"])
    comparison_break_even_reduction_bps = float(config.entry_fee_bps) + float(config.exit_fee_bps) - float(config.comparison_entry_fee_bps) - float(config.comparison_exit_fee_bps)
    mfe = ((peak / entry_price) - 1.0) * 10000.0; mae = ((trough / entry_price) - 1.0) * 10000.0
    replay_key = f"{product_id}|{timeframe}|{str(variant)}|{replay_ts}|{int(float(signal.score) * 1000000)}"
    now = time.time()
    scenario_results = _fee_scenario_matrix_results(notional=notional, qty=qty, exit_price=float(exit_price), fee_scenario_matrix=getattr(config, "fee_scenario_matrix", {}) or {}) if bool(getattr(config, "enable_fee_scenario_matrix", False)) else {}
    row = {"ts": now, "dt_mst": _dt_mst(now), "replay_key": replay_key, "product_id": product_id, "timeframe": timeframe, "granularity": granularity, "replay_ts": replay_ts, "entry_price": entry_price, "entry_fee_bps": float(config.entry_fee_bps), "exit_fee_bps": float(config.exit_fee_bps), "synthetic_notional_usd": notional, "synthetic_qty": qty, "all_in_entry_price": all_in_entry, "min_profitable_exit_price": min_exit, "hard_stop_price": hard_stop, "exit_ts": float(exit_ts), "exit_price": float(exit_price), "exit_reason": exit_reason, "held_seconds": max(0.0, float(exit_ts) - float(replay_ts)), "max_favorable_bps": mfe, "max_adverse_bps": mae, "peak_price": peak, "trough_price": trough, "gross_pnl_usd": gross_proceeds - notional, "exit_fee_usd": exit_fee, "net_pnl_usd": net_pnl, "net_pnl_bps": net_bps, "primary_fee_model": str(config.primary_fee_model), "comparison_fee_model": str(config.comparison_fee_model), "primary_entry_fee_bps": float(config.entry_fee_bps), "primary_exit_fee_bps": float(config.exit_fee_bps), "primary_entry_fee_usd": float(primary_fee_result["entry_fee_usd"]), "primary_exit_fee_usd": float(primary_fee_result["exit_fee_usd"]), "primary_net_pnl_usd": float(primary_fee_result["net_pnl_usd"]), "primary_net_pnl_bps": float(primary_fee_result["net_pnl_bps"]), "primary_would_have_won": int(primary_fee_result["would_have_won"]), "comparison_entry_fee_bps": float(config.comparison_entry_fee_bps), "comparison_exit_fee_bps": float(config.comparison_exit_fee_bps), "comparison_entry_fee_usd": float(comparison_fee_result["entry_fee_usd"]), "comparison_exit_fee_usd": float(comparison_fee_result["exit_fee_usd"]), "comparison_net_pnl_usd": float(comparison_fee_result["net_pnl_usd"]), "comparison_net_pnl_bps": float(comparison_fee_result["net_pnl_bps"]), "comparison_would_have_won": int(comparison_fee_result["would_have_won"]), "comparison_net_improvement_usd": float(comparison_net_improvement_usd), "comparison_net_improvement_bps": float(comparison_net_improvement_bps), "comparison_break_even_reduction_bps": float(comparison_break_even_reduction_bps), "would_have_won": int(net_pnl > 0), "would_have_hit_stop": int(exit_reason == "historical_hard_stop"), "would_have_hit_min_profit": int(peak >= min_exit), "score": float(signal.score), "probability": float(signal.estimated_prob_up), "expected_net_edge_bps": float(signal.expected_net_edge_bps), "target_bps": float(signal.target_bps), "cost_bps": float(signal.cost_bps), "spread_bps": float(signal.spread_bps), "session_liquidity_setup": str(signal.session_liquidity_setup), "value_acceptance_state": str(signal.value_acceptance_state), "volume_node_state": str(signal.volume_node_state), "poc_distance_bps": float(signal.poc_distance_bps), "volume_profile_leader_buy_score": float(signal.volume_profile_leader_buy_score), "volume_profile_leader_wait_score": float(signal.volume_profile_leader_wait_score), "price_action_buy_score": float(signal.price_action_buy_score), "market_structure_buy_score": float(signal.market_structure_buy_score), "quant_buy_score": float(signal.quant_buy_score), "setup_tag": setup_tag, "regime_tag": regime_tag, "replay_candidate_qualified": int(bool(qualified)), "replay_candidate_quality": float(replay_quality), "replay_filter_reason": qualification_reason, "accepted_for_calibration": int(bool(qualified) and timeframe in {"primary_15m_90d", "regime_1h_365d"}), "replay_source": replay_source, "historical_source_exchange": "binance" if str(replay_source) == "binance_bulk" else "local_cache", "historical_source_symbol": product_id, "historical_source_note": "process_worker_replay", "strategy_variant": str(variant), "variant_entry_allowed": int(1), "variant_block_reason": str(variant_reason), "variant_target_to_cost_ratio": float(variant_metrics.get("target_to_cost_ratio", 0.0)), "variant_target_over_cost_bps": float(variant_metrics.get("target_over_cost_bps", 0.0)), "variant_momentum_5_bps": float(variant_metrics.get("momentum_5_bps", 0.0)), "variant_momentum_15_bps": float(variant_metrics.get("momentum_15_bps", 0.0)), "reason": f"process_worker_replay;exit={exit_reason};net_bps={net_bps:.2f};mfe={mfe:.2f};mae={mae:.2f};score={float(signal.score):.4f};prob={float(signal.estimated_prob_up):.4f};qualified={qualified};qualification={qualification_reason};setup={setup_tag};regime={regime_tag}"}
    row.update(scenario_results)
    return {col: row.get(col, "") for col in columns}


def run_replay_job_from_cache(payload: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    product_id = str(payload["product_id"]); timeframe = str(payload["timeframe"]); granularity = str(payload["granularity"])
    cache_path = str(payload["cache_path"]); output_path = str(payload["output_path"]); replay_source = str(payload.get("replay_source") or "local_cache")
    columns = list(payload["columns"]); config = ReplayEngineConfig(**payload["config"])
    if timeframe == "primary_15m_90d":
        min_prefix = int(config.hist_replay_min_prefix_15m); forward_windows = list(config.hist_replay_forward_bars_15m); step_bars = int(config.hist_replay_step_bars_15m)
    elif timeframe == "regime_1h_365d":
        min_prefix = int(config.hist_replay_min_prefix_1h); forward_windows = list(config.hist_replay_forward_bars_1h); step_bars = int(config.hist_replay_step_bars_1h)
    else:
        min_prefix = 60; forward_windows = [5, 10, 20]; step_bars = 10
    candles = _read_replay_candles(cache_path, product_id, min_ts=float(payload.get("min_ts") or 0.0))
    rows: List[Dict[str, Any]] = []; evaluated = 0; existing_prefixes = set(payload.get("existing_prefixes") or [])
    max_forward = max(forward_windows); eval_started = time.perf_counter()
    for i in range(min_prefix, len(candles) - max_forward, max(1, step_bars)):
        if time.perf_counter() - eval_started >= float(config.hist_replay_max_runtime_sec) or evaluated >= int(config.hist_replay_max_candidates_per_pass):
            break
        prefix = candles[:i]
        for forward_bars in forward_windows:
            future = candles[i:i + int(forward_bars)]
            if len(future) < int(forward_bars):
                continue
            if f"{product_id}|{timeframe}|{int(prefix[-1].ts)}|" in existing_prefixes:
                continue
            variants = list(config.strategy_variants or ["baseline"])
            for variant in variants:
                row = _simulate_candidate(product_id=product_id, timeframe=timeframe, granularity=granularity, prefix=prefix, future=future, replay_source=replay_source, config=config, columns=columns, variant=str(variant))
                if row:
                    rows.append(row)
            evaluated += 1
            break
    rows_written = _write_replay_rows(output_path, columns, rows)
    net_values = []
    qualified_rows = 0
    accepted_rows = 0
    for row in rows:
        try:
            net_values.append(float(row.get("net_pnl_bps") or 0.0))
            qualified_rows += int(int(float(row.get("replay_candidate_qualified") or 0)) == 1)
            accepted_rows += int(int(float(row.get("accepted_for_calibration") or 0)) == 1)
        except Exception:
            continue
    return {"ok": True, "worker_mode": "process", "product_id": product_id, "timeframe": timeframe, "granularity": granularity, "output_path": output_path, "cache_path": cache_path, "candles": len(candles), "evaluated": int(evaluated), "rows_written": int(rows_written), "qualified_rows": int(qualified_rows), "accepted_rows": int(accepted_rows), "avg_net_pnl_bps": float(np.mean(net_values)) if net_values else 0.0, "median_net_pnl_bps": float(np.median(net_values)) if net_values else 0.0, "elapsed_sec": round(time.perf_counter() - started, 3)}
