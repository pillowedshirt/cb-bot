from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class PreviousSessionVolumeProfileSignal:
    product_id: str
    session_key: str
    session_start_ts: float
    session_end_ts: float
    previous_session_high: float
    previous_session_low: float
    previous_session_open: float
    previous_session_close: float
    previous_session_poc: float
    previous_session_vah: float
    previous_session_val: float
    current_price: float
    distance_to_poc_bps: float
    distance_to_vah_bps: float
    distance_to_val_bps: float
    reaction_state: str
    higher_timeframe_bias: str
    bias_confidence: float
    buy_score: float
    sell_score: float
    hold_score: float
    wait_score: float
    confidence: float
    reason: str


def _get(c: Any, key: str, default: float = 0.0) -> float:
    try:
        if isinstance(c, dict):
            return float(c.get(key, default) or default)
        if hasattr(c, key):
            return float(getattr(c, key) or default)
        if key == "ts" and hasattr(c, "minute_start_ts"):
            return float(getattr(c, "minute_start_ts") or default)
        return float(default)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def _bps(price: float, ref: float) -> float:
    if ref <= 0:
        return 0.0
    return (float(price) / float(ref) - 1.0) * 10000.0


def _session_bounds_for_hour(ts: float, session_key: str) -> Tuple[float, float]:
    day = math.floor(float(ts) / 86400.0) * 86400.0
    windows = {
        "daily_utc": (day - 86400.0, day),
        "tokyo": (day + 0 * 3600.0, day + 6 * 3600.0),
        "london": (day + 7 * 3600.0, day + 12 * 3600.0),
        "new_york": (day + 13 * 3600.0, day + 20 * 3600.0),
        "london_new_york_overlap": (day + 13 * 3600.0, day + 16 * 3600.0),
    }
    if session_key not in windows:
        session_key = "daily_utc"
    start, end = windows[session_key]
    if float(ts) <= end:
        start -= 86400.0
        end -= 86400.0
    return float(start), float(end)


def _candles_in_bounds(candles: List[Any], start_ts: float, end_ts: float) -> List[Any]:
    out = []
    for c in candles or []:
        ts = _get(c, "ts", 0.0)
        if ts <= 0:
            ts = _get(c, "minute_start_ts", 0.0)
        if start_ts <= ts < end_ts:
            out.append(c)
    return out


def _fixed_range_volume_profile(candles: List[Any], bins: int = 36) -> Tuple[float, float, float, str]:
    if len(candles) < 12:
        return 0.0, 0.0, 0.0, "insufficient_session_candles"
    lows = [_get(c, "low", 0.0) for c in candles if _get(c, "low", 0.0) > 0]
    highs = [_get(c, "high", 0.0) for c in candles if _get(c, "high", 0.0) > 0]
    if not lows or not highs:
        return 0.0, 0.0, 0.0, "missing_high_low"
    lo = min(lows)
    hi = max(highs)
    if hi <= lo:
        return 0.0, 0.0, 0.0, "flat_session"
    bin_count = max(12, int(bins))
    step = (hi - lo) / bin_count
    volumes = [0.0 for _ in range(bin_count)]
    centers = [lo + (i + 0.5) * step for i in range(bin_count)]
    for c in candles:
        typical = (_get(c, "high", 0.0) + _get(c, "low", 0.0) + _get(c, "close", 0.0)) / 3.0
        idx = max(0, min(bin_count - 1, int((typical - lo) / step)))
        volumes[idx] += max(0.0, _get(c, "volume", 0.0))
    total_volume = sum(volumes)
    if total_volume <= 0:
        return 0.0, 0.0, 0.0, "missing_volume"
    poc_idx = max(range(bin_count), key=lambda i: volumes[i])
    poc = centers[poc_idx]
    included = {poc_idx}
    included_vol = volumes[poc_idx]
    left = poc_idx - 1
    right = poc_idx + 1
    while included_vol / total_volume < 0.70 and (left >= 0 or right < bin_count):
        left_vol = volumes[left] if left >= 0 else -1.0
        right_vol = volumes[right] if right < bin_count else -1.0
        if right_vol >= left_vol:
            included.add(right)
            included_vol += max(0.0, right_vol)
            right += 1
        else:
            included.add(left)
            included_vol += max(0.0, left_vol)
            left -= 1
    return float(poc), float(max(centers[i] for i in included)), float(min(centers[i] for i in included)), "ok"


def _higher_timeframe_bias(candles: List[Any]) -> Tuple[str, float, str]:
    closes = [_get(c, "close", 0.0) for c in candles if _get(c, "close", 0.0) > 0]
    if len(closes) < 24:
        return "neutral", 0.20, "bias_insufficient_candles"
    n = max(3, len(closes) // 5)
    first = sum(closes[:n]) / max(1, len(closes[:n]))
    last = sum(closes[-n:]) / max(1, len(closes[-n:]))
    slope_bps = _bps(last, first)
    highs = [_get(c, "high", 0.0) for c in candles[-24:]]
    lows = [_get(c, "low", 0.0) for c in candles[-24:]]
    range_pos = 0.50
    if max(highs) > min(lows):
        range_pos = (closes[-1] - min(lows)) / (max(highs) - min(lows))
    if slope_bps > 35 and range_pos >= 0.52:
        return "bullish", _clamp(0.45 + min(0.35, slope_bps / 300.0)), f"bias_bullish;slope_bps={slope_bps:.2f};range_pos={range_pos:.3f}"
    if slope_bps < -35 and range_pos <= 0.48:
        return "bearish", _clamp(0.45 + min(0.35, abs(slope_bps) / 300.0)), f"bias_bearish;slope_bps={slope_bps:.2f};range_pos={range_pos:.3f}"
    return "neutral", 0.38, f"bias_neutral;slope_bps={slope_bps:.2f};range_pos={range_pos:.3f}"


def build_previous_session_volume_profile_signal(*, product_id: str, candles: List[Any], current_price: float, session_key: str = "daily_utc", bins: int = 36) -> PreviousSessionVolumeProfileSignal:
    now = max([_get(c, "ts", 0.0) or _get(c, "minute_start_ts", 0.0) for c in candles] or [0.0])
    start_ts, end_ts = _session_bounds_for_hour(now, session_key)
    session_candles = _candles_in_bounds(candles, start_ts, end_ts)
    if len(session_candles) < 12:
        return PreviousSessionVolumeProfileSignal(product_id, session_key, start_ts, end_ts, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(current_price), 0.0, 0.0, 0.0, "unavailable", "neutral", 0.10, 0.0, 0.0, 0.50, 0.70, 0.10, f"previous_session_profile_unavailable;session={session_key};candles={len(session_candles)}")
    poc, vah, val, vp_reason = _fixed_range_volume_profile(session_candles, bins=bins)
    session_high = max(_get(c, "high", 0.0) for c in session_candles)
    session_low = min(_get(c, "low", 0.0) for c in session_candles)
    session_open = _get(session_candles[0], "open", _get(session_candles[0], "close", 0.0))
    session_close = _get(session_candles[-1], "close", 0.0)
    prior = candles[-2] if len(candles) >= 2 else candles[-1]
    last = candles[-1]
    last_close = _get(last, "close", current_price)
    prior_close = _get(prior, "close", last_close)
    last_high = _get(last, "high", last_close)
    bias, bias_conf, bias_reason = _higher_timeframe_bias(candles[-240:])
    d_poc, d_vah, d_val = _bps(current_price, poc), _bps(current_price, vah), _bps(current_price, val)
    near_poc, near_vah, near_val = abs(d_poc) <= 18, abs(d_vah) <= 22, abs(d_val) <= 22
    accepted_above_vah = current_price > vah and last_close > vah and prior_close > vah
    rejected_vah = last_high > vah and last_close < vah
    reclaimed_val = prior_close < val and last_close > val
    accepted_below_val = current_price < val and last_close < val and prior_close < val
    rejected_poc = near_poc and ((last_close < poc and prior_close > poc) or (last_close > poc and prior_close < poc))
    if accepted_above_vah:
        reaction_state = "accepted_above_prior_vah"
    elif rejected_vah:
        reaction_state = "rejected_prior_vah"
    elif reclaimed_val:
        reaction_state = "reclaimed_prior_val"
    elif accepted_below_val:
        reaction_state = "accepted_below_prior_val"
    elif rejected_poc:
        reaction_state = "rejected_prior_poc"
    elif near_poc:
        reaction_state = "near_prior_poc_chop"
    elif near_vah:
        reaction_state = "near_prior_vah"
    elif near_val:
        reaction_state = "near_prior_val"
    else:
        reaction_state = "between_or_away_from_prior_levels"
    buy_score = _clamp(0.18 + (0.48 if accepted_above_vah else 0.0) + (0.38 if reclaimed_val else 0.0) + (0.14 if bias == "bullish" else 0.0) - (0.36 if accepted_below_val else 0.0) - (0.28 if rejected_vah else 0.0) - (0.18 if near_poc else 0.0))
    sell_score = _clamp(0.18 + (0.48 if rejected_vah else 0.0) + (0.36 if accepted_below_val else 0.0) + (0.18 if near_poc else 0.0) + (0.14 if bias == "bearish" else 0.0) - (0.24 if accepted_above_vah else 0.0))
    hold_score = _clamp(0.36 + (0.32 if accepted_above_vah and bias != "bearish" else 0.0) + (0.20 if reclaimed_val and bias == "bullish" else 0.0) - (0.22 if rejected_vah or accepted_below_val else 0.0))
    wait_score = _clamp(0.35 + (0.32 if near_poc else 0.0) + (0.16 if reaction_state in {"near_prior_vah", "near_prior_val"} else 0.0) - (0.18 if accepted_above_vah or reclaimed_val else 0.0))
    confidence = _clamp(0.32 + min(0.20, len(session_candles) / 160.0) + (0.18 if reaction_state not in {"between_or_away_from_prior_levels", "unavailable"} else 0.0) + bias_conf * 0.20)
    return PreviousSessionVolumeProfileSignal(product_id, session_key, start_ts, end_ts, float(session_high), float(session_low), float(session_open), float(session_close), float(poc), float(vah), float(val), float(current_price), float(d_poc), float(d_vah), float(d_val), reaction_state, bias, float(bias_conf), float(buy_score), float(sell_score), float(hold_score), float(wait_score), float(confidence), f"previous_session_volume_profile;session={session_key};reaction={reaction_state};bias={bias};bias_conf={bias_conf:.3f};poc={poc:.10f};vah={vah:.10f};val={val:.10f};d_poc={d_poc:.2f};d_vah={d_vah:.2f};d_val={d_val:.2f};{vp_reason};{bias_reason}")


def signal_to_dict(signal: PreviousSessionVolumeProfileSignal) -> Dict[str, Any]:
    return asdict(signal)
