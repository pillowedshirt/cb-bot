from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class SessionWindow:
    key: str
    label: str
    start_hour_utc: int
    end_hour_utc: int


@dataclass
class SessionLiquiditySignal:
    product_id: str
    active_session: str
    strongest_agent: str
    strongest_setup: str
    best_buy_score: float
    best_sell_score: float
    best_hold_score: float
    confidence: float
    session_high: float
    session_low: float
    swept_high: bool
    swept_low: bool
    reclaimed_low: bool
    rejected_high: bool
    breakout_hold_high: bool
    breakdown_hold_low: bool
    nearest_upside_liquidity: float
    nearest_downside_liquidity: float
    upside_target_bps: float
    downside_target_bps: float
    stop_distance_bps: float
    reason: str


GLOBAL_SESSION_WINDOWS: List[SessionWindow] = [
    SessionWindow("sydney", "Sydney / Oceania", 21, 0),
    SessionWindow("tokyo", "Tokyo / Asia", 0, 6),
    SessionWindow("hong_kong_singapore", "Hong Kong / Singapore", 1, 8),
    SessionWindow("london", "London / Europe", 7, 12),
    SessionWindow("new_york", "New York / US", 13, 20),
    SessionWindow("london_new_york_overlap", "London / New York Overlap", 13, 16),
    SessionWindow("daily_reset", "UTC Daily Reset", 0, 1),
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _utc_hour(ts: Optional[float] = None) -> int:
    if ts is None:
        ts = datetime.now(tz=timezone.utc).timestamp()
    return int(datetime.fromtimestamp(float(ts), tz=timezone.utc).hour)


def _session_contains_hour(window: SessionWindow, hour: int) -> bool:
    start = int(window.start_hour_utc)
    end = int(window.end_hour_utc)

    if start < end:
        return start <= hour < end

    # Handles sessions that cross midnight, such as Sydney 21 -> 0.
    return hour >= start or hour < end


def active_session_keys(ts: Optional[float] = None) -> List[str]:
    hour = _utc_hour(ts)
    keys: List[str] = []
    for window in GLOBAL_SESSION_WINDOWS:
        if _session_contains_hour(window, hour):
            keys.append(window.key)
    if not keys:
        keys.append("off_session")
    return keys


def _candles_in_window(candles: List[Any], window: SessionWindow, now_ts: float) -> List[Any]:
    out: List[Any] = []

    # Use the last 36 hours so overnight sessions can still be represented.
    min_ts = float(now_ts) - 36.0 * 3600.0

    for candle in candles:
        ts = _safe_float(getattr(candle, "minute_start_ts", 0.0), 0.0)
        if ts < min_ts:
            continue
        hour = _utc_hour(ts)
        if _session_contains_hour(window, hour):
            out.append(candle)

    return out


def _recent(candles: List[Any], count: int) -> List[Any]:
    if not candles:
        return []
    return candles[-int(count):]


def _highest(candles: List[Any]) -> float:
    if not candles:
        return 0.0
    return max(_safe_float(getattr(c, "high", 0.0), 0.0) for c in candles)


def _lowest(candles: List[Any]) -> float:
    if not candles:
        return 0.0
    lows = [_safe_float(getattr(c, "low", 0.0), 0.0) for c in candles]
    lows = [v for v in lows if v > 0]
    return min(lows) if lows else 0.0


def _bps_from_prices(new_price: float, base_price: float) -> float:
    if base_price <= 0:
        return 0.0
    return ((float(new_price) / float(base_price)) - 1.0) * 10000.0


def _nearby_swing_targets(candles: List[Any], current_price: float, lookback: int = 180) -> Tuple[float, float]:
    recent = _recent(candles, lookback)
    if not recent or current_price <= 0:
        return 0.0, 0.0

    highs = sorted(set(_safe_float(getattr(c, "high", 0.0), 0.0) for c in recent))
    lows = sorted(set(_safe_float(getattr(c, "low", 0.0), 0.0) for c in recent))

    upside = 0.0
    for value in highs:
        if value > current_price:
            upside = value
            break

    downside = 0.0
    for value in reversed(lows):
        if 0 < value < current_price:
            downside = value
            break

    return upside, downside


def build_session_liquidity_signal(
    *,
    product_id: str,
    candles: List[Any],
    current_price: float,
    spread_bps: float,
    cost_bps: float,
    projected_forward_gain_bps: float,
    ts: Optional[float] = None,
) -> SessionLiquiditySignal:
    """
    Build one combined session liquidity signal from global session windows.

    This intentionally returns scores, not hard pass/fail decisions.
    The Level 8 council weighs these scores against all other agents.
    """
    now_value = float(ts if ts is not None else datetime.now(tz=timezone.utc).timestamp())
    current_price = float(current_price)

    if not candles or current_price <= 0:
        return SessionLiquiditySignal(
            product_id=product_id,
            active_session="unknown",
            strongest_agent="session_liquidity_unavailable",
            strongest_setup="none",
            best_buy_score=0.0,
            best_sell_score=0.0,
            best_hold_score=0.50,
            confidence=0.10,
            session_high=0.0,
            session_low=0.0,
            swept_high=False,
            swept_low=False,
            reclaimed_low=False,
            rejected_high=False,
            breakout_hold_high=False,
            breakdown_hold_low=False,
            nearest_upside_liquidity=0.0,
            nearest_downside_liquidity=0.0,
            upside_target_bps=0.0,
            downside_target_bps=0.0,
            stop_distance_bps=0.0,
            reason="no_candles_or_price",
        )

    active_keys = active_session_keys(now_value)
    recent_8 = _recent(candles, 8)

    recent_low = _lowest(recent_8)
    recent_high = _highest(recent_8)
    recent_close = _safe_float(getattr(candles[-1], "close", current_price), current_price)

    upside_swing, downside_swing = _nearby_swing_targets(candles, current_price, lookback=240)

    best: Optional[Dict[str, Any]] = None

    for window in GLOBAL_SESSION_WINDOWS:
        window_candles = _candles_in_window(candles, window, now_value)
        if len(window_candles) < 8:
            continue

        session_high = _highest(window_candles)
        session_low = _lowest(window_candles)

        if session_high <= 0 or session_low <= 0:
            continue

        session_range_bps = abs(_bps_from_prices(session_high, session_low))
        if session_range_bps <= 0:
            continue

        swept_low = bool(recent_low > 0 and recent_low < session_low and recent_close > session_low)
        swept_high = bool(recent_high > session_high and recent_close < session_high)

        reclaimed_low = bool(swept_low and current_price > session_low)
        rejected_high = bool(swept_high and current_price < session_high)

        breakout_hold_high = bool(current_price > session_high and recent_low >= session_high)
        breakdown_hold_low = bool(current_price < session_low and recent_high <= session_low)

        nearest_upside = max(session_high, upside_swing)
        if nearest_upside <= current_price:
            nearest_upside = upside_swing if upside_swing > current_price else session_high

        nearest_downside = min(session_low, downside_swing) if downside_swing > 0 else session_low
        if nearest_downside >= current_price:
            nearest_downside = session_low

        upside_target_bps = _bps_from_prices(nearest_upside, current_price) if nearest_upside > current_price else 0.0
        downside_target_bps = abs(_bps_from_prices(nearest_downside, current_price)) if 0 < nearest_downside < current_price else 0.0

        stop_distance_bps = 0.0
        if reclaimed_low and recent_low > 0:
            stop_distance_bps = abs(_bps_from_prices(recent_low, current_price))
        elif rejected_high and recent_high > 0:
            stop_distance_bps = abs(_bps_from_prices(recent_high, current_price))

        economic_room = max(0.0, upside_target_bps - float(cost_bps))
        projected_room = max(0.0, float(projected_forward_gain_bps) - float(cost_bps))

        active_bonus = 0.08 if window.key in active_keys else 0.0
        overlap_bonus = 0.05 if window.key == "london_new_york_overlap" else 0.0

        reversal_buy_score = _clamp(
            0.40
            + (0.22 if reclaimed_low else 0.0)
            + min(0.20, economic_room / 420.0)
            + min(0.12, projected_room / 420.0)
            + active_bonus
            + overlap_bonus
            - min(0.14, float(spread_bps) / 280.0)
            - min(0.14, stop_distance_bps / 500.0),
            0.0,
            1.0,
        )

        breakout_buy_score = _clamp(
            0.38
            + (0.20 if breakout_hold_high else 0.0)
            + min(0.18, economic_room / 480.0)
            + active_bonus
            - min(0.12, float(spread_bps) / 300.0),
            0.0,
            1.0,
        )

        sell_score = _clamp(
            0.35
            + (0.24 if rejected_high else 0.0)
            + min(0.20, downside_target_bps / 500.0)
            + active_bonus
            + overlap_bonus
            - min(0.10, float(spread_bps) / 350.0),
            0.0,
            1.0,
        )

        if reversal_buy_score >= breakout_buy_score and reversal_buy_score >= sell_score:
            setup = "sweep_reclaim"
            agent = f"{window.key}_sweep_reversal"
            buy_score = reversal_buy_score
        elif breakout_buy_score >= sell_score:
            setup = "session_breakout_hold"
            agent = f"{window.key}_breakout_continuation"
            buy_score = breakout_buy_score
        else:
            setup = "sweep_reject_harvest"
            agent = f"{window.key}_liquidity_harvest"
            buy_score = max(reversal_buy_score, breakout_buy_score)

        confidence = _clamp(
            0.25
            + min(0.25, session_range_bps / 500.0)
            + min(0.20, max(upside_target_bps, downside_target_bps) / 500.0)
            + (0.15 if reclaimed_low or rejected_high or breakout_hold_high else 0.0)
            + active_bonus,
            0.10,
            0.90,
        )

        candidate = {
            "window": window,
            "agent": agent,
            "setup": setup,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "hold_score": _clamp(0.42 + buy_score * 0.24 - sell_score * 0.12, 0.0, 1.0),
            "confidence": confidence,
            "session_high": session_high,
            "session_low": session_low,
            "swept_high": swept_high,
            "swept_low": swept_low,
            "reclaimed_low": reclaimed_low,
            "rejected_high": rejected_high,
            "breakout_hold_high": breakout_hold_high,
            "breakdown_hold_low": breakdown_hold_low,
            "nearest_upside": nearest_upside,
            "nearest_downside": nearest_downside,
            "upside_target_bps": upside_target_bps,
            "downside_target_bps": downside_target_bps,
            "stop_distance_bps": stop_distance_bps,
            "session_range_bps": session_range_bps,
        }

        if best is None:
            best = candidate
        else:
            best_strength = max(float(best["buy_score"]), float(best["sell_score"])) * float(best["confidence"])
            candidate_strength = max(float(candidate["buy_score"]), float(candidate["sell_score"])) * float(candidate["confidence"])
            if candidate_strength > best_strength:
                best = candidate

    if best is None:
        return SessionLiquiditySignal(
            product_id=product_id,
            active_session=",".join(active_keys),
            strongest_agent="session_liquidity_no_range",
            strongest_setup="none",
            best_buy_score=0.40,
            best_sell_score=0.35,
            best_hold_score=0.55,
            confidence=0.15,
            session_high=0.0,
            session_low=0.0,
            swept_high=False,
            swept_low=False,
            reclaimed_low=False,
            rejected_high=False,
            breakout_hold_high=False,
            breakdown_hold_low=False,
            nearest_upside_liquidity=0.0,
            nearest_downside_liquidity=0.0,
            upside_target_bps=0.0,
            downside_target_bps=0.0,
            stop_distance_bps=0.0,
            reason=f"no_session_range_available;active={','.join(active_keys)}",
        )

    reason = (
        f"session_liquidity;"
        f"active={','.join(active_keys)};"
        f"agent={best['agent']};"
        f"setup={best['setup']};"
        f"session_high={float(best['session_high']):.8f};"
        f"session_low={float(best['session_low']):.8f};"
        f"swept_low={best['swept_low']};"
        f"reclaimed_low={best['reclaimed_low']};"
        f"swept_high={best['swept_high']};"
        f"rejected_high={best['rejected_high']};"
        f"breakout_hold_high={best['breakout_hold_high']};"
        f"upside_target_bps={float(best['upside_target_bps']):.2f};"
        f"downside_target_bps={float(best['downside_target_bps']):.2f};"
        f"stop_distance_bps={float(best['stop_distance_bps']):.2f};"
        f"range_bps={float(best['session_range_bps']):.2f}"
    )

    return SessionLiquiditySignal(
        product_id=product_id,
        active_session=",".join(active_keys),
        strongest_agent=str(best["agent"]),
        strongest_setup=str(best["setup"]),
        best_buy_score=float(best["buy_score"]),
        best_sell_score=float(best["sell_score"]),
        best_hold_score=float(best["hold_score"]),
        confidence=float(best["confidence"]),
        session_high=float(best["session_high"]),
        session_low=float(best["session_low"]),
        swept_high=bool(best["swept_high"]),
        swept_low=bool(best["swept_low"]),
        reclaimed_low=bool(best["reclaimed_low"]),
        rejected_high=bool(best["rejected_high"]),
        breakout_hold_high=bool(best["breakout_hold_high"]),
        breakdown_hold_low=bool(best["breakdown_hold_low"]),
        nearest_upside_liquidity=float(best["nearest_upside"]),
        nearest_downside_liquidity=float(best["nearest_downside"]),
        upside_target_bps=float(best["upside_target_bps"]),
        downside_target_bps=float(best["downside_target_bps"]),
        stop_distance_bps=float(best["stop_distance_bps"]),
        reason=reason,
    )


def signal_to_dict(signal: SessionLiquiditySignal) -> Dict[str, Any]:
    return asdict(signal)
