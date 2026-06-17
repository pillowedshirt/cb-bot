from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import math


@dataclass
class CandleAnatomy:
    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    direction: str
    range_bps: float
    body_bps: float
    upper_wick_bps: float
    lower_wick_bps: float
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    close_location: float
    is_full_body_bull: bool
    is_full_body_bear: bool
    is_doji: bool
    is_indecision: bool
    is_upper_rejection: bool
    is_lower_rejection: bool


@dataclass
class SwingPoint:
    ts: float
    price: float
    kind: str
    index: int
    broke_opposite: bool = False
    swept: bool = False
    stale: bool = False
    fresh: bool = True


@dataclass
class FreshZone:
    direction: str
    origin_ts: float
    zone_low: float
    zone_high: float
    caused_break_ts: float
    caused_break_price: float
    first_retest: bool
    fresh: bool
    swept_back_into_zone: bool
    reason: str


@dataclass
class PriceActionContext:
    product_id: str
    candle_context_buy_score: float
    candle_context_sell_score: float
    candle_context_hold_score: float
    candle_context_confidence: float
    candle_sequence_score: float
    candle_exhaustion_score: float
    candle_continuation_score: float
    market_structure_buy_score: float
    market_structure_sell_score: float
    market_structure_hold_score: float
    market_structure_confidence: float
    validated_liquidity_buy_score: float
    validated_liquidity_sell_score: float
    validated_liquidity_confidence: float
    fresh_zone_buy_score: float
    fresh_zone_sell_score: float
    fresh_zone_confidence: float
    volume_profile_buy_score: float
    volume_profile_sell_score: float
    volume_profile_hold_score: float
    volume_profile_confidence: float
    fvg_buy_score: float
    fvg_sell_score: float
    fvg_confidence: float
    trend_state: str
    structure_state: str
    last_swing_high: float
    last_swing_low: float
    validated_high: float
    validated_low: float
    validated_high_state: str
    validated_low_state: str
    liquidity_quality_score: float
    nearest_upside_liquidity: float
    nearest_downside_liquidity: float
    value_area_high: float
    value_area_low: float
    point_of_control: float
    value_area_state: str
    bullish_fvg_low: float
    bullish_fvg_high: float
    bearish_fvg_low: float
    bearish_fvg_high: float
    fvg_state: str
    fresh_zone_low: float
    fresh_zone_high: float
    fresh_zone_state: str
    reason: str

    # Volume Profile Leader framework.
    volume_profile_leader_buy_score: float = 0.0
    volume_profile_leader_sell_score: float = 0.0
    volume_profile_leader_hold_score: float = 0.50
    volume_profile_leader_wait_score: float = 0.50
    volume_profile_leader_confidence: float = 0.10
    value_acceptance_state: str = "unknown"
    volume_node_state: str = "unknown"
    nearest_high_volume_node: float = 0.0
    nearest_low_volume_node: float = 0.0
    low_volume_path_up_bps: float = 0.0
    low_volume_path_down_bps: float = 0.0
    poc_distance_bps: float = 0.0
    unfair_trade_score: float = 0.0
    volume_profile_leader_reason: str = ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


def _bps(new_price: float, base_price: float) -> float:
    base_price = float(base_price)
    if base_price <= 0:
        return 0.0
    return ((float(new_price) / base_price) - 1.0) * 10000.0


def _get(candle: Any, name: str, default: float = 0.0) -> float:
    if isinstance(candle, dict):
        return _safe_float(candle.get(name), default)
    return _safe_float(getattr(candle, name, default), default)


def candle_anatomy(candle: Any) -> CandleAnatomy:
    ts = _get(candle, "minute_start_ts", _get(candle, "ts", 0.0))
    o = _get(candle, "open", 0.0)
    h = _get(candle, "high", 0.0)
    l = _get(candle, "low", 0.0)
    c = _get(candle, "close", 0.0)
    v = _get(candle, "volume", 0.0)

    if h <= 0 or l <= 0 or c <= 0 or o <= 0 or h < l:
        return CandleAnatomy(
            ts=ts, open=o, high=h, low=l, close=c, volume=v,
            direction="flat", range_bps=0.0, body_bps=0.0,
            upper_wick_bps=0.0, lower_wick_bps=0.0,
            body_ratio=0.0, upper_wick_ratio=0.0, lower_wick_ratio=0.0,
            close_location=0.50, is_full_body_bull=False,
            is_full_body_bear=False, is_doji=False, is_indecision=True,
            is_upper_rejection=False, is_lower_rejection=False,
        )

    rng = max(1e-12, h - l)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    body_ratio = body / rng
    upper_ratio = max(0.0, upper_wick / rng)
    lower_ratio = max(0.0, lower_wick / rng)
    close_location = (c - l) / rng

    direction = "bull" if c > o else "bear" if c < o else "flat"

    range_bps = abs(_bps(h, l))
    body_bps = abs(_bps(c, o))
    upper_wick_bps = abs(_bps(h, max(o, c)))
    lower_wick_bps = abs(_bps(min(o, c), l))

    is_full_body_bull = bool(direction == "bull" and body_ratio >= 0.68 and close_location >= 0.72)
    is_full_body_bear = bool(direction == "bear" and body_ratio >= 0.68 and close_location <= 0.28)
    is_doji = bool(body_ratio <= 0.16 and range_bps >= 10.0)
    is_indecision = bool(body_ratio <= 0.28 and upper_ratio >= 0.20 and lower_ratio >= 0.20)
    is_upper_rejection = bool(upper_ratio >= 0.45 and close_location <= 0.55)
    is_lower_rejection = bool(lower_ratio >= 0.45 and close_location >= 0.45)

    return CandleAnatomy(ts, o, h, l, c, v, direction, range_bps, body_bps, upper_wick_bps, lower_wick_bps, body_ratio, upper_ratio, lower_ratio, close_location, is_full_body_bull, is_full_body_bear, is_doji, is_indecision, is_upper_rejection, is_lower_rejection)


def _recent_anatomy(candles: List[Any], lookback: int = 8) -> List[CandleAnatomy]:
    return [candle_anatomy(c) for c in list(candles)[-int(lookback):]]


def _swing_points(candles: List[Any], lookback: int = 160, radius: int = 2) -> List[SwingPoint]:
    source = list(candles)[-int(lookback):]
    out: List[SwingPoint] = []
    if len(source) < radius * 2 + 3:
        return out
    for i in range(radius, len(source) - radius):
        c = source[i]
        h = _get(c, "high", 0.0)
        l = _get(c, "low", 0.0)
        ts = _get(c, "minute_start_ts", _get(c, "ts", 0.0))
        left = source[i - radius:i]
        right = source[i + 1:i + 1 + radius]
        if h > 0 and all(h >= _get(x, "high", 0.0) for x in left + right):
            out.append(SwingPoint(ts=ts, price=h, kind="high", index=i))
        if l > 0 and all(l <= _get(x, "low", 0.0) for x in left + right):
            out.append(SwingPoint(ts=ts, price=l, kind="low", index=i))
    return out


def _mark_validated_liquidity(swings: List[SwingPoint], candles: List[Any]) -> Tuple[float, float, str, str, float, str]:
    """
    Validate highs/lows by checking whether price later broke the opposite swing.

    Also classify whether the level is:
    - unconfirmed
    - fresh
    - swept
    - stale

    This keeps liquidity as weighted evidence, not a hard rule.
    """
    if not swings or not candles:
        return 0.0, 0.0, "unconfirmed", "unconfirmed", 0.0, "no_swings"

    source = list(candles)[-160:]
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]

    validated_high = 0.0
    validated_low = 0.0
    validated_high_index = -1
    validated_low_index = -1

    for high in highs:
        prior_lows = [l for l in lows if l.index < high.index]
        later_candles = source[high.index + 1:]
        if not prior_lows or not later_candles:
            continue
        ref_low = prior_lows[-1].price
        if any(_get(c, "low", 0.0) < ref_low for c in later_candles):
            high.broke_opposite = True
            validated_high = high.price
            validated_high_index = high.index

    for low in lows:
        prior_highs = [h for h in highs if h.index < low.index]
        later_candles = source[low.index + 1:]
        if not prior_highs or not later_candles:
            continue
        ref_high = prior_highs[-1].price
        if any(_get(c, "high", 0.0) > ref_high for c in later_candles):
            low.broke_opposite = True
            validated_low = low.price
            validated_low_index = low.index

    def high_state(price: float, idx: int) -> str:
        if price <= 0 or idx < 0:
            return "unconfirmed"
        later = source[idx + 1:]
        if any(_get(c, "high", 0.0) > price and _get(c, "close", 0.0) < price for c in later):
            return "swept"
        if len(source) - idx > 90:
            return "stale"
        return "fresh"

    def low_state(price: float, idx: int) -> str:
        if price <= 0 or idx < 0:
            return "unconfirmed"
        later = source[idx + 1:]
        if any(_get(c, "low", 0.0) < price and _get(c, "close", 0.0) > price for c in later):
            return "swept"
        if len(source) - idx > 90:
            return "stale"
        return "fresh"

    high_state_value = high_state(validated_high, validated_high_index)
    low_state_value = low_state(validated_low, validated_low_index)

    quality = 0.0
    if validated_high > 0:
        quality += 0.25
        quality += 0.15 if high_state_value == "fresh" else 0.08 if high_state_value == "swept" else 0.0
    if validated_low > 0:
        quality += 0.25
        quality += 0.15 if low_state_value == "fresh" else 0.08 if low_state_value == "swept" else 0.0
    quality = _clamp(quality, 0.0, 1.0)

    reason = (
        f"validated_high={validated_high:.8f};validated_high_state={high_state_value};"
        f"validated_low={validated_low:.8f};validated_low_state={low_state_value};"
        f"liquidity_quality={quality:.3f}"
    )
    return float(validated_high), float(validated_low), high_state_value, low_state_value, float(quality), reason


def _structure_state(swings: List[SwingPoint]) -> Tuple[str, str]:
    highs = [s for s in swings if s.kind == "high"][-3:]
    lows = [s for s in swings if s.kind == "low"][-3:]
    if len(highs) >= 2 and len(lows) >= 2:
        higher_high = highs[-1].price > highs[-2].price
        higher_low = lows[-1].price > lows[-2].price
        lower_high = highs[-1].price < highs[-2].price
        lower_low = lows[-1].price < lows[-2].price
        if higher_high and higher_low:
            return "uptrend_structure", "higher_high_higher_low"
        if lower_high and lower_low:
            return "downtrend_structure", "lower_high_lower_low"
        if higher_high and lower_low:
            return "expanding_structure", "higher_high_lower_low"
        if lower_high and higher_low:
            return "compression_structure", "lower_high_higher_low"
    return "mixed_structure", "insufficient_clean_swings"


def _fresh_zone(candles: List[Any], swings: List[SwingPoint]) -> FreshZone:
    if len(candles) < 30 or len(swings) < 3:
        last = candle_anatomy(candles[-1]) if candles else None
        px = last.close if last else 0.0
        return FreshZone("none", 0.0, px, px, 0.0, px, False, False, False, "insufficient_structure")
    source = list(candles)[-160:]
    anatomies = [candle_anatomy(c) for c in source]
    current_close = anatomies[-1].close
    best_i = None
    best_strength = 0.0
    for i, a in enumerate(anatomies[:-3]):
        strength = a.body_ratio * a.body_bps if (a.is_full_body_bull or a.is_full_body_bear) else 0.0
        if strength > best_strength:
            best_strength = strength
            best_i = i
    if best_i is None:
        return FreshZone("none", 0.0, current_close, current_close, 0.0, current_close, False, False, False, "no_displacement_origin")
    origin = anatomies[best_i]
    direction = "bullish" if origin.close > origin.open else "bearish"
    zone_low = min(origin.open, origin.close)
    zone_high = max(origin.open, origin.close)
    later = anatomies[best_i + 1:]
    retests = [a for a in later if a.low <= zone_high and a.high >= zone_low]
    first_retest = bool(len(retests) == 1)
    fresh = bool(len(retests) == 0)
    swept_back = bool((direction == "bullish" and any(a.low < zone_low and a.close > zone_low for a in later[-8:])) or (direction == "bearish" and any(a.high > zone_high and a.close < zone_high for a in later[-8:])))
    reason = f"fresh_zone;direction={direction};zone_low={zone_low:.8f};zone_high={zone_high:.8f};fresh={fresh};first_retest={first_retest};swept_back_into_zone={swept_back}"
    return FreshZone(direction, origin.ts, zone_low, zone_high, origin.ts, origin.close, first_retest, fresh, swept_back, reason)


def _volume_profile(candles: List[Any], bins: int = 24, lookback: int = 180) -> Tuple[float, float, float, str]:
    source = list(candles)[-int(lookback):]
    if len(source) < 20:
        return 0.0, 0.0, 0.0, "volume_profile_unavailable"
    lows = [_get(c, "low", 0.0) for c in source if _get(c, "low", 0.0) > 0]
    highs = [_get(c, "high", 0.0) for c in source if _get(c, "high", 0.0) > 0]
    if not lows or not highs:
        return 0.0, 0.0, 0.0, "volume_profile_no_prices"
    lo = min(lows); hi = max(highs)
    if hi <= lo:
        return 0.0, 0.0, 0.0, "volume_profile_flat_range"
    bin_count = max(8, int(bins)); volumes = [0.0 for _ in range(bin_count)]; step = (hi - lo) / bin_count
    for c in source:
        price = (_get(c, "high", 0.0) + _get(c, "low", 0.0) + _get(c, "close", 0.0)) / 3.0
        idx = max(0, min(bin_count - 1, int((price - lo) / step)))
        volumes[idx] += max(0.0, _get(c, "volume", 0.0))
    total_vol = sum(volumes)
    if total_vol <= 0:
        return 0.0, 0.0, 0.0, "volume_profile_zero_volume"
    poc_idx = max(range(bin_count), key=lambda i: volumes[i]); target_vol = total_vol * 0.70
    included = {poc_idx}; cum_vol = volumes[poc_idx]; left = poc_idx - 1; right = poc_idx + 1
    while cum_vol < target_vol and (left >= 0 or right < bin_count):
        left_vol = volumes[left] if left >= 0 else -1.0; right_vol = volumes[right] if right < bin_count else -1.0
        if right_vol >= left_vol:
            included.add(right); cum_vol += max(0.0, right_vol); right += 1
        else:
            included.add(left); cum_vol += max(0.0, left_vol); left -= 1
    poc = lo + (poc_idx + 0.5) * step; val = lo + min(included) * step; vah = lo + (max(included) + 1.0) * step
    return float(val), float(vah), float(poc), f"volume_profile;poc={poc:.8f};val={val:.8f};vah={vah:.8f};total_vol={total_vol:.4f}"



def _volume_profile_leader_context(
    *,
    candles: List[Any],
    current_price: float,
    last: CandleAnatomy,
    prior: CandleAnatomy,
    value_area_low: float,
    value_area_high: float,
    point_of_control: float,
    bins: int = 32,
    lookback: int = 220,
) -> Dict[str, Any]:
    """
    Volume Profile Leader framework.

    Main model:
    - Inside value area = fair value / chop / lower edge.
    - POC = defended/choppy area, not automatic support/resistance.
    - HVN = high-volume node / slower motion / defended area.
    - LVN = low-volume node / fast-motion area.
    - Accepted above value = continuation / buy-supportive.
    - Rejected above value = harvest / sell-supportive.
    - Accepted below value = avoid-buy / sell-rally supportive.
    - Reclaim of value low = possible fakeout/reversal buy evidence.
    """
    source = list(candles or [])[-int(lookback):]
    out = {
        "volume_profile_leader_buy_score": 0.0,
        "volume_profile_leader_sell_score": 0.0,
        "volume_profile_leader_hold_score": 0.50,
        "volume_profile_leader_wait_score": 0.50,
        "volume_profile_leader_confidence": 0.10,
        "value_acceptance_state": "volume_profile_unavailable",
        "volume_node_state": "unknown",
        "nearest_high_volume_node": 0.0,
        "nearest_low_volume_node": 0.0,
        "low_volume_path_up_bps": 0.0,
        "low_volume_path_down_bps": 0.0,
        "poc_distance_bps": 0.0,
        "unfair_trade_score": 0.0,
        "volume_profile_leader_reason": "volume_profile_leader_unavailable",
    }

    try:
        if len(source) < 30 or current_price <= 0 or value_area_low <= 0 or value_area_high <= 0 or point_of_control <= 0:
            return out

        lows = [_get(c, "low", 0.0) for c in source if _get(c, "low", 0.0) > 0]
        highs = [_get(c, "high", 0.0) for c in source if _get(c, "high", 0.0) > 0]
        if not lows or not highs:
            return out

        lo = min(lows)
        hi = max(highs)
        if hi <= lo:
            return out

        bin_count = max(12, int(bins))
        step = (hi - lo) / bin_count
        volumes = [0.0 for _ in range(bin_count)]
        centers = [lo + (i + 0.5) * step for i in range(bin_count)]

        for c in source:
            typical = (_get(c, "high", 0.0) + _get(c, "low", 0.0) + _get(c, "close", 0.0)) / 3.0
            idx = max(0, min(bin_count - 1, int((typical - lo) / step)))
            volumes[idx] += max(0.0, _get(c, "volume", 0.0))

        nonzero = [v for v in volumes if v > 0]
        if not nonzero:
            return out

        sorted_vol = sorted(nonzero)
        high_cutoff = sorted_vol[int(max(0, min(len(sorted_vol) - 1, len(sorted_vol) * 0.72)))]
        low_cutoff = sorted_vol[int(max(0, min(len(sorted_vol) - 1, len(sorted_vol) * 0.25)))]

        current_idx = max(0, min(bin_count - 1, int((current_price - lo) / step)))
        current_bin_vol = volumes[current_idx]

        high_nodes = [centers[i] for i, v in enumerate(volumes) if v >= high_cutoff]
        low_nodes = [centers[i] for i, v in enumerate(volumes) if 0 < v <= low_cutoff]

        nearest_hvn = min(high_nodes, key=lambda pp: abs(pp - current_price)) if high_nodes else 0.0
        nearest_lvn = min(low_nodes, key=lambda pp: abs(pp - current_price)) if low_nodes else 0.0
        upside_lvns = [pp for pp in low_nodes if pp > current_price]
        downside_lvns = [pp for pp in low_nodes if pp < current_price]
        low_volume_path_up_bps = _bps(min(upside_lvns), current_price) if upside_lvns else 0.0
        low_volume_path_down_bps = abs(_bps(max(downside_lvns), current_price)) if downside_lvns else 0.0
        poc_distance_bps = abs(_bps(current_price, point_of_control))
        near_poc = bool(poc_distance_bps <= 22.0)
        inside_value = bool(value_area_low <= current_price <= value_area_high)
        above_value = bool(current_price > value_area_high)
        below_value = bool(current_price < value_area_low)
        accepted_above_value = bool(above_value and last.close > value_area_high and (prior.close > value_area_high or last.close_location >= 0.62) and not last.is_upper_rejection)
        rejected_above_value = bool(last.high > value_area_high and (last.close < value_area_high or last.is_upper_rejection))
        accepted_below_value = bool(below_value and last.close < value_area_low and (prior.close < value_area_low or last.close_location <= 0.38) and not last.is_lower_rejection)
        rejected_below_value = bool(last.low < value_area_low and (last.close > value_area_low or last.is_lower_rejection))
        reclaimed_value_low = bool(prior.close < value_area_low and last.close > value_area_low and last.close_location >= 0.50)

        if accepted_above_value:
            value_acceptance_state = "accepted_above_value"
        elif rejected_above_value:
            value_acceptance_state = "rejected_above_value"
        elif accepted_below_value:
            value_acceptance_state = "accepted_below_value"
        elif rejected_below_value or reclaimed_value_low:
            value_acceptance_state = "reclaimed_value_low"
        elif inside_value and near_poc:
            value_acceptance_state = "inside_value_near_poc"
        elif inside_value:
            value_acceptance_state = "inside_fair_value"
        elif above_value:
            value_acceptance_state = "above_value_unconfirmed"
        elif below_value:
            value_acceptance_state = "below_value_unconfirmed"
        else:
            value_acceptance_state = "unknown"

        if current_bin_vol >= high_cutoff:
            volume_node_state = "high_volume_node"
        elif 0 < current_bin_vol <= low_cutoff:
            volume_node_state = "low_volume_node"
        else:
            volume_node_state = "normal_volume_node"

        unfair_trade_score = _clamp((0.32 if not inside_value else 0.0) + (0.25 if volume_node_state == "low_volume_node" else 0.0) + min(0.25, max(low_volume_path_up_bps, low_volume_path_down_bps) / 240.0) - (0.18 if near_poc else 0.0) - (0.14 if volume_node_state == "high_volume_node" else 0.0), 0.0, 1.0)
        buy_score = _clamp(0.18 + (0.56 if accepted_above_value else 0.0) + (0.44 if reclaimed_value_low else 0.0) + (0.12 if above_value and volume_node_state == "low_volume_node" else 0.0) + min(0.12, low_volume_path_up_bps / 300.0) + unfair_trade_score * 0.12 - (0.24 if inside_value and near_poc else 0.0) - (0.38 if accepted_below_value else 0.0) - (0.34 if rejected_above_value else 0.0), 0.0, 1.0)
        sell_score = _clamp(0.18 + (0.56 if accepted_below_value else 0.0) + (0.48 if rejected_above_value else 0.0) + (0.18 if inside_value and near_poc else 0.0) + (0.12 if volume_node_state == "high_volume_node" else 0.0) + min(0.10, low_volume_path_down_bps / 320.0) - (0.28 if accepted_above_value else 0.0), 0.0, 1.0)
        hold_score = _clamp(0.38 + (0.32 if accepted_above_value else 0.0) + (0.12 if above_value and low_volume_path_up_bps > 40.0 else 0.0) - (0.26 if rejected_above_value or accepted_below_value else 0.0) - (0.12 if near_poc else 0.0), 0.0, 1.0)
        wait_score = _clamp(0.34 + (0.34 if inside_value else 0.0) + (0.26 if near_poc else 0.0) + (0.18 if volume_node_state == "high_volume_node" else 0.0) - (0.22 if accepted_above_value or reclaimed_value_low else 0.0), 0.0, 1.0)
        confidence = _clamp(0.30 + min(0.22, len(source) / 350.0) + (0.16 if not inside_value else 0.0) + (0.14 if volume_node_state in {"high_volume_node", "low_volume_node"} else 0.0) + unfair_trade_score * 0.18, 0.10, 0.94)

        out.update({
            "volume_profile_leader_buy_score": float(buy_score),
            "volume_profile_leader_sell_score": float(sell_score),
            "volume_profile_leader_hold_score": float(hold_score),
            "volume_profile_leader_wait_score": float(wait_score),
            "volume_profile_leader_confidence": float(confidence),
            "value_acceptance_state": value_acceptance_state,
            "volume_node_state": volume_node_state,
            "nearest_high_volume_node": float(nearest_hvn),
            "nearest_low_volume_node": float(nearest_lvn),
            "low_volume_path_up_bps": float(low_volume_path_up_bps),
            "low_volume_path_down_bps": float(low_volume_path_down_bps),
            "poc_distance_bps": float(poc_distance_bps),
            "unfair_trade_score": float(unfair_trade_score),
            "volume_profile_leader_reason": (f"volume_profile_leader;value_acceptance_state={value_acceptance_state};volume_node_state={volume_node_state};poc_distance_bps={poc_distance_bps:.2f};nearest_hvn={nearest_hvn:.8f};nearest_lvn={nearest_lvn:.8f};lvn_up_bps={low_volume_path_up_bps:.2f};lvn_down_bps={low_volume_path_down_bps:.2f};unfair_trade_score={unfair_trade_score:.3f};accepted_above={accepted_above_value};rejected_above={rejected_above_value};accepted_below={accepted_below_value};reclaimed_value_low={reclaimed_value_low};buy={buy_score:.3f};sell={sell_score:.3f};hold={hold_score:.3f};wait={wait_score:.3f};confidence={confidence:.3f}"),
        })
        return out
    except Exception as exc:
        out["volume_profile_leader_reason"] = f"volume_profile_leader_error:{exc}"
        return out


def _fvg_context(candles: List[Any]) -> Tuple[float, float, float, float, str]:
    source = list(candles)[-30:]
    bullish_low = bullish_high = bearish_low = bearish_high = 0.0
    state = "no_fvg"
    if len(source) < 3:
        return bullish_low, bullish_high, bearish_low, bearish_high, "fvg_insufficient_candles"
    current_close = _get(source[-1], "close", 0.0)
    for i in range(2, len(source)):
        a = source[i - 2]; c = source[i]
        a_high = _get(a, "high", 0.0); a_low = _get(a, "low", 0.0); c_high = _get(c, "high", 0.0); c_low = _get(c, "low", 0.0)
        if a_high > 0 and c_low > 0 and a_high < c_low:
            bullish_low = a_high; bullish_high = c_low; state = "bullish_fvg_open"
        if a_low > 0 and c_high > 0 and a_low > c_high:
            bearish_low = c_high; bearish_high = a_low; state = "bearish_fvg_open"
    if bullish_low > 0 and bullish_low <= current_close <= bullish_high:
        state = "bullish_fvg_retest"
    elif bearish_low > 0 and bearish_low <= current_close <= bearish_high:
        state = "bearish_fvg_retest"
    elif bullish_high > 0 and current_close < bullish_low:
        state = "bullish_fvg_inverted_ifvg"
    elif bearish_low > 0 and current_close > bearish_high:
        state = "bearish_fvg_inverted_ifvg"
    reason = f"fvg_state={state};bullish_fvg_low={bullish_low:.8f};bullish_fvg_high={bullish_high:.8f};bearish_fvg_low={bearish_low:.8f};bearish_fvg_high={bearish_high:.8f}"
    return bullish_low, bullish_high, bearish_low, bearish_high, reason


def build_price_action_context(*, product_id: str, candles: List[Any], current_price: float, spread_bps: float = 0.0, cost_bps: float = 0.0, projected_forward_gain_bps: float = 0.0) -> PriceActionContext:
    candles = list(candles or [])
    current_price = float(current_price or 0.0)
    if not candles or current_price <= 0:
        return PriceActionContext(
            product_id=product_id,
            candle_context_buy_score=0.0,
            candle_context_sell_score=0.0,
            candle_context_hold_score=0.50,
            candle_context_confidence=0.10,
            candle_sequence_score=0.0,
            candle_exhaustion_score=0.0,
            candle_continuation_score=0.0,
            market_structure_buy_score=0.0,
            market_structure_sell_score=0.0,
            market_structure_hold_score=0.50,
            market_structure_confidence=0.10,
            validated_liquidity_buy_score=0.0,
            validated_liquidity_sell_score=0.0,
            validated_liquidity_confidence=0.10,
            fresh_zone_buy_score=0.0,
            fresh_zone_sell_score=0.0,
            fresh_zone_confidence=0.10,
            volume_profile_buy_score=0.0,
            volume_profile_sell_score=0.0,
            volume_profile_hold_score=0.50,
            volume_profile_confidence=0.10,
            fvg_buy_score=0.0,
            fvg_sell_score=0.0,
            fvg_confidence=0.10,
            trend_state="unknown",
            structure_state="unknown",
            last_swing_high=0.0,
            last_swing_low=0.0,
            validated_high=0.0,
            validated_low=0.0,
            validated_high_state="unconfirmed",
            validated_low_state="unconfirmed",
            liquidity_quality_score=0.0,
            nearest_upside_liquidity=0.0,
            nearest_downside_liquidity=0.0,
            value_area_high=0.0,
            value_area_low=0.0,
            point_of_control=0.0,
            value_area_state="unknown",
            bullish_fvg_low=0.0,
            bullish_fvg_high=0.0,
            bearish_fvg_low=0.0,
            bearish_fvg_high=0.0,
            fvg_state="unknown",
            fresh_zone_low=0.0,
            fresh_zone_high=0.0,
            fresh_zone_state="unknown",
            reason="no_candles_or_price",
        )
    recent = _recent_anatomy(candles, 8); last = recent[-1]; prior = recent[-2] if len(recent) >= 2 else last; last3 = recent[-3:] if len(recent) >= 3 else recent
    full_body_bull_count = sum(1 for c in recent[-4:] if c.is_full_body_bull); full_body_bear_count = sum(1 for c in recent[-4:] if c.is_full_body_bear); indecision_count = sum(1 for c in recent[-4:] if c.is_indecision or c.is_doji); upper_rejection_count = sum(1 for c in recent[-4:] if c.is_upper_rejection); lower_rejection_count = sum(1 for c in recent[-4:] if c.is_lower_rejection)
    bodies = [c.body_ratio for c in last3]; upper_wicks = [c.upper_wick_ratio for c in last3]
    shrinking_bodies = bool(len(bodies) == 3 and bodies[0] > bodies[1] > bodies[2]); growing_upper_wicks = bool(len(upper_wicks) == 3 and upper_wicks[0] < upper_wicks[1] < upper_wicks[2]); advanced_block_exhaustion = bool(shrinking_bodies and growing_upper_wicks and last.close >= prior.close)
    candle_continuation_score = _clamp(0.22 + full_body_bull_count * 0.16 + max(0.0, last.close_location - 0.50) * 0.45 - upper_rejection_count * 0.08)
    candle_exhaustion_score = _clamp(0.10 + (0.25 if last.is_doji or last.is_indecision else 0.0) + upper_rejection_count * 0.16 + (0.28 if advanced_block_exhaustion else 0.0) + (0.12 if last.is_upper_rejection else 0.0) - full_body_bull_count * 0.06)
    lower_rejection_buy = _clamp(0.20 + lower_rejection_count * 0.17 + (0.15 if last.is_lower_rejection else 0.0) + max(0.0, last.close_location - 0.50) * 0.25)
    candle_context_buy_score = _clamp(lower_rejection_buy * 0.55 + candle_continuation_score * 0.35 + (0.10 if last.direction == "bull" else 0.0))
    candle_context_sell_score = _clamp(candle_exhaustion_score * 0.62 + upper_rejection_count * 0.10 + (0.08 if last.direction == "bear" else 0.0))
    candle_context_hold_score = _clamp(0.35 + candle_continuation_score * 0.45 - candle_exhaustion_score * 0.20)
    candle_sequence_score = _clamp(0.30 + (0.22 if lower_rejection_count >= 1 and last.direction == "bull" else 0.0) + (0.20 if full_body_bull_count >= 2 else 0.0) - (0.18 if indecision_count >= 2 else 0.0) - (0.18 if advanced_block_exhaustion else 0.0))
    swings = _swing_points(candles, lookback=180, radius=2); structure_state, structure_reason = _structure_state(swings)
    highs = [s for s in swings if s.kind == "high"]; lows = [s for s in swings if s.kind == "low"]; last_swing_high = highs[-1].price if highs else 0.0; last_swing_low = lows[-1].price if lows else 0.0; validated_high, validated_low, validated_high_state, validated_low_state, liquidity_quality_score, validation_reason = _mark_validated_liquidity(swings, candles)
    upside_candidates = [x for x in [last_swing_high, validated_high] if x > current_price]; downside_candidates = [x for x in [last_swing_low, validated_low] if 0 < x < current_price]
    nearest_upside = min(upside_candidates) if upside_candidates else 0.0; nearest_downside = max(downside_candidates) if downside_candidates else 0.0
    swept_validated_low = bool(validated_low > 0 and last.low < validated_low and last.close > validated_low); swept_validated_high = bool(validated_high > 0 and last.high > validated_high and last.close < validated_high)
    market_structure_buy_score = _clamp(0.32 + (0.20 if structure_state == "uptrend_structure" else 0.0) + (0.15 if structure_state == "compression_structure" else 0.0) + (0.20 if swept_validated_low else 0.0) + candle_sequence_score * 0.20)
    market_structure_sell_score = _clamp(0.25 + (0.22 if structure_state == "downtrend_structure" else 0.0) + (0.20 if swept_validated_high else 0.0) + candle_exhaustion_score * 0.25)
    market_structure_hold_score = _clamp(0.38 + (0.20 if structure_state == "uptrend_structure" else 0.0) + candle_continuation_score * 0.25 - candle_exhaustion_score * 0.12)
    validated_liquidity_buy_score = _clamp(0.20 + liquidity_quality_score * 0.18 + (0.35 if swept_validated_low else 0.0) + (0.12 if validated_low_state == "fresh" else 0.0) + (0.15 if last.is_lower_rejection else 0.0) + (0.10 if nearest_upside > current_price else 0.0))
    validated_liquidity_sell_score = _clamp(0.20 + liquidity_quality_score * 0.18 + (0.35 if swept_validated_high else 0.0) + (0.12 if validated_high_state == "fresh" else 0.0) + (0.15 if last.is_upper_rejection else 0.0) + (0.10 if nearest_downside > 0 else 0.0))
    zone = _fresh_zone(candles, swings)
    zone_width_bps = abs(_bps(zone.zone_high, zone.zone_low)) if zone.zone_high > 0 and zone.zone_low > 0 else 0.0
    zone_quality = _clamp(0.35 + (0.22 if zone.fresh else 0.0) + (0.16 if zone.first_retest else 0.0) + (0.18 if zone.swept_back_into_zone else 0.0) - min(0.20, zone_width_bps / 650.0), 0.0, 1.0)
    fresh_zone_buy_score = _clamp(0.18 + zone_quality * 0.24 + (0.28 if zone.direction == "bullish" and (zone.fresh or zone.first_retest) else 0.0) + (0.22 if zone.direction == "bullish" and zone.swept_back_into_zone else 0.0) + (0.10 if last.is_lower_rejection else 0.0))
    fresh_zone_sell_score = _clamp(0.18 + zone_quality * 0.24 + (0.28 if zone.direction == "bearish" and (zone.fresh or zone.first_retest) else 0.0) + (0.22 if zone.direction == "bearish" and zone.swept_back_into_zone else 0.0) + (0.10 if last.is_upper_rejection else 0.0))
    val, vah, poc, vp_reason = _volume_profile(candles)
    value_area_state = "above_value_area" if val > 0 and vah > 0 and current_price > vah else "below_value_area" if val > 0 and vah > 0 and current_price < val else "inside_value_area" if val > 0 and vah > 0 else "volume_profile_unavailable"
    value_area_width_bps = abs(_bps(vah, val)) if val > 0 and vah > 0 else 0.0
    value_area_available = bool(val > 0 and vah > 0 and poc > 0)
    volume_profile_quality = _clamp((0.30 if value_area_available else 0.0) + min(0.28, value_area_width_bps / 500.0) + (0.12 if value_area_state != "inside_value_area" else 0.0) + (0.10 if abs(_bps(current_price, poc)) >= 25.0 and poc > 0 else 0.0), 0.0, 1.0)
    accepted_above_value = bool(value_area_state == "above_value_area" and last.close_location >= 0.55 and not last.is_upper_rejection); rejected_above_value = bool(value_area_state == "above_value_area" and last.is_upper_rejection); accepted_below_value = bool(value_area_state == "below_value_area" and last.close_location <= 0.45 and not last.is_lower_rejection); reclaimed_value_low = bool(val > 0 and prior.close < val and last.close > val)
    volume_profile_buy_score = _clamp(0.28 + (0.25 if accepted_above_value else 0.0) + (0.25 if reclaimed_value_low else 0.0) + (0.08 if value_area_state != "inside_value_area" else -0.06))
    volume_profile_sell_score = _clamp(0.25 + (0.30 if rejected_above_value else 0.0) + (0.20 if accepted_below_value else 0.0) + (0.10 if last.is_upper_rejection else 0.0))
    volume_profile_hold_score = _clamp(0.42 + (0.18 if accepted_above_value else 0.0) - (0.20 if rejected_above_value else 0.0))
    vp_leader = _volume_profile_leader_context(
        candles=candles,
        current_price=current_price,
        last=last,
        prior=prior,
        value_area_low=val,
        value_area_high=vah,
        point_of_control=poc,
    )
    volume_profile_buy_score = _clamp(volume_profile_buy_score * 0.35 + float(vp_leader.get("volume_profile_leader_buy_score", 0.0)) * 0.65)
    volume_profile_sell_score = _clamp(volume_profile_sell_score * 0.35 + float(vp_leader.get("volume_profile_leader_sell_score", 0.0)) * 0.65)
    volume_profile_hold_score = _clamp(volume_profile_hold_score * 0.35 + float(vp_leader.get("volume_profile_leader_hold_score", 0.50)) * 0.65)
    volume_profile_confidence = _clamp(context_confidence * 0.35 + float(vp_leader.get("volume_profile_leader_confidence", 0.10)) * 0.65, 0.10, 0.94)
    bullish_fvg_low, bullish_fvg_high, bearish_fvg_low, bearish_fvg_high, fvg_reason = _fvg_context(candles); fvg_state = fvg_reason.split(";", 1)[0].replace("fvg_state=", "")
    fvg_buy_score = _clamp(0.24 + (0.25 if fvg_state in {"bullish_fvg_retest", "bearish_fvg_inverted_ifvg"} else 0.0) + (0.16 if swept_validated_low else 0.0) + (0.10 if last.direction == "bull" else 0.0))
    fvg_sell_score = _clamp(0.24 + (0.25 if fvg_state in {"bearish_fvg_retest", "bullish_fvg_inverted_ifvg"} else 0.0) + (0.16 if swept_validated_high else 0.0) + (0.10 if last.direction == "bear" else 0.0))
    context_confidence = _clamp(0.25 + min(0.25, len(candles) / 240.0) + min(0.20, last.range_bps / 160.0) + (0.12 if last_swing_high > 0 and last_swing_low > 0 else 0.0))
    reason = f"price_action;last_dir={last.direction};body_ratio={last.body_ratio:.3f};upper_wick_ratio={last.upper_wick_ratio:.3f};lower_wick_ratio={last.lower_wick_ratio:.3f};close_location={last.close_location:.3f};doji={last.is_doji};upper_rejection={last.is_upper_rejection};lower_rejection={last.is_lower_rejection};advanced_block={advanced_block_exhaustion};structure={structure_state};structure_reason={structure_reason};liquidity_quality={liquidity_quality_score:.3f};validated_high_state={validated_high_state};validated_low_state={validated_low_state};{validation_reason};fresh_zone={zone.reason};value_area_state={value_area_state};value_area_width_bps={value_area_width_bps:.2f};volume_profile_quality={volume_profile_quality:.3f};{vp_reason};{vp_leader.get('volume_profile_leader_reason', '')};{fvg_reason}"
    return PriceActionContext(product_id, float(candle_context_buy_score), float(candle_context_sell_score), float(candle_context_hold_score), float(context_confidence), float(candle_sequence_score), float(candle_exhaustion_score), float(candle_continuation_score), float(market_structure_buy_score), float(market_structure_sell_score), float(market_structure_hold_score), float(context_confidence), float(validated_liquidity_buy_score), float(validated_liquidity_sell_score), float(context_confidence), float(fresh_zone_buy_score), float(fresh_zone_sell_score), float(context_confidence), float(volume_profile_buy_score), float(volume_profile_sell_score), float(volume_profile_hold_score), float(volume_profile_confidence), float(fvg_buy_score), float(fvg_sell_score), float(context_confidence), str(structure_state), str(structure_reason), float(last_swing_high), float(last_swing_low), float(validated_high), float(validated_low), str(validated_high_state), str(validated_low_state), float(liquidity_quality_score), float(nearest_upside), float(nearest_downside), float(vah), float(val), float(poc), str(value_area_state), float(bullish_fvg_low), float(bullish_fvg_high), float(bearish_fvg_low), float(bearish_fvg_high), str(fvg_state), float(zone.zone_low), float(zone.zone_high), str(zone.reason), reason, float(vp_leader.get("volume_profile_leader_buy_score", 0.0)), float(vp_leader.get("volume_profile_leader_sell_score", 0.0)), float(vp_leader.get("volume_profile_leader_hold_score", 0.50)), float(vp_leader.get("volume_profile_leader_wait_score", 0.50)), float(vp_leader.get("volume_profile_leader_confidence", 0.10)), str(vp_leader.get("value_acceptance_state", "unknown")), str(vp_leader.get("volume_node_state", "unknown")), float(vp_leader.get("nearest_high_volume_node", 0.0)), float(vp_leader.get("nearest_low_volume_node", 0.0)), float(vp_leader.get("low_volume_path_up_bps", 0.0)), float(vp_leader.get("low_volume_path_down_bps", 0.0)), float(vp_leader.get("poc_distance_bps", 0.0)), float(vp_leader.get("unfair_trade_score", 0.0)), str(vp_leader.get("volume_profile_leader_reason", "")))


def context_to_dict(context: PriceActionContext) -> Dict[str, Any]:
    return asdict(context)
