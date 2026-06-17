from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional



try:
    from debug_tools import (
        module_debug,
        module_exception,
        debug_every,
        debug_timer,
    )
except Exception:
    def module_debug(*args, **kwargs):
        pass
    def module_exception(*args, **kwargs):
        pass
    def debug_every(*args, **kwargs):
        pass
    class debug_timer:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

MODULE_NAME = __name__.split(".")[-1]
module_debug(
    MODULE_NAME,
    "module_loaded",
    data={"file": __file__},
    level="DEBUG",
    also_overall=False,
)
@dataclass
class QuantContextSignal:
    product_id: str
    log_return_mean_bps: float
    log_return_std_bps: float
    return_autocorr_1: float
    return_autocorr_3: float
    stationarity_score: float
    mean_drift_score: float
    variance_drift_score: float
    forecast_return_bps: float
    forecast_upper_bps: float
    forecast_lower_bps: float
    conditional_volatility_bps: float
    volatility_cluster_state: str
    boundary_state: str
    quant_buy_score: float
    quant_sell_score: float
    quant_hold_score: float
    quant_wait_score: float
    confidence: float
    peer_product: str
    peer_spread_z: float
    peer_state: str
    peer_reason: str
    reason: str


def _get(c: Any, key: str, default: float = 0.0) -> float:
    try:
        if isinstance(c, dict):
            return float(c.get(key, default) or default)
        if hasattr(c, key):
            return float(getattr(c, key) or default)
        return float(default)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / max(1, len(values) - 1))


def _corr(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    a = a[-n:]
    b = b[-n:]
    ma, mb, sa, sb = _mean(a), _mean(b), _std(a), _std(b)
    if sa <= 0 or sb <= 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / max(1e-12, (n - 1) * sa * sb)


def _closes(candles: List[Any]) -> List[float]:
    return [_get(c, "close", 0.0) for c in candles or [] if _get(c, "close", 0.0) > 0]


def _log_returns_from_closes(closes: List[float]) -> List[float]:
    return [math.log(b / a) * 10000.0 for a, b in zip(closes[:-1], closes[1:]) if a > 0 and b > 0]


def _autocorr(returns: List[float], lag: int) -> float:
    if len(returns) <= lag + 8:
        return 0.0
    return _corr(returns[:-lag], returns[lag:])


def _ewma_vol(returns: List[float], alpha: float = 0.12) -> float:
    if not returns:
        return 0.0
    var = returns[0] ** 2
    for r in returns[1:]:
        var = alpha * (r ** 2) + (1.0 - alpha) * var
    return math.sqrt(max(0.0, var))


def _stationarity_proxy(returns: List[float]) -> Dict[str, float]:
    if len(returns) < 40:
        return {"stationarity_score": 0.35, "mean_drift_score": 0.50, "variance_drift_score": 0.50}
    third = len(returns) // 3
    parts = [returns[:third], returns[third:2 * third], returns[2 * third:]]
    means = [_mean(p) for p in parts]
    stds = [_std(p) for p in parts]
    mean_drift = max(means) - min(means)
    avg_std = max(1e-9, _mean(stds))
    min_pos_std = min((s for s in stds if s > 0), default=avg_std)
    variance_drift = max(stds) / max(1e-9, min_pos_std)
    mean_drift_score = _clamp(1.0 - abs(mean_drift) / max(12.0, avg_std * 1.2))
    variance_drift_score = _clamp(1.0 - max(0.0, variance_drift - 1.0) / 2.0)
    stationarity_score = _clamp(mean_drift_score * 0.55 + variance_drift_score * 0.45)
    return {"stationarity_score": float(stationarity_score), "mean_drift_score": float(mean_drift_score), "variance_drift_score": float(variance_drift_score)}


def _peer_relative_context(*, product_id: str, returns: List[float], peer_returns_by_product: Optional[Dict[str, List[float]]]) -> Dict[str, Any]:
    if not peer_returns_by_product:
        return {"peer_product": "", "peer_spread_z": 0.0, "peer_state": "no_peer_context", "peer_reason": "peer_context_unavailable"}
    preferred = {"BTC-USD": ["ETH-USD"], "ETH-USD": ["BTC-USD"], "SOL-USD": ["AVAX-USD"], "AVAX-USD": ["SOL-USD"], "DOGE-USD": ["SHIB-USD"], "SHIB-USD": ["DOGE-USD"], "XLM-USD": ["XRP-USD"], "XRP-USD": ["XLM-USD"], "LTC-USD": ["BCH-USD"], "BCH-USD": ["LTC-USD"]}.get(product_id, [])
    peer_product, peer_returns = "", []
    for p in preferred:
        if p in peer_returns_by_product and len(peer_returns_by_product[p]) >= 30:
            peer_product, peer_returns = p, peer_returns_by_product[p]
            break
    if not peer_returns:
        for p, r in peer_returns_by_product.items():
            if p != product_id and len(r) >= 30:
                peer_product, peer_returns = p, r
                break
    n = min(len(returns), len(peer_returns))
    if n < 30:
        return {"peer_product": peer_product, "peer_spread_z": 0.0, "peer_state": "insufficient_peer_data", "peer_reason": f"peer={peer_product};n={n}"}
    spread = [a - b for a, b in zip(returns[-n:], peer_returns[-n:])]
    s = _std(spread[-60:])
    z = (spread[-1] - _mean(spread[-60:])) / s if s > 0 else 0.0
    if z <= -1.75:
        state = "product_cheap_vs_peer_snapback_possible"
    elif z >= 1.75:
        state = "product_rich_vs_peer_snapback_risk"
    else:
        state = "peer_spread_normal"
    return {"peer_product": peer_product, "peer_spread_z": float(z), "peer_state": state, "peer_reason": f"peer={peer_product};spread_z={z:.3f};state={state}"}


def build_quant_context_signal(*, product_id: str, candles: List[Any], peer_returns_by_product: Optional[Dict[str, List[float]]] = None, lookback: int = 240) -> QuantContextSignal:
    debug_every(
        MODULE_NAME,
        f"quant_context_start:{product_id}",
        30.0,
        "quant_context_start",
        data={
            "product_id": product_id,
            "candles_count": len(candles) if candles is not None else 0,
            "peer_count": len(peer_returns_by_product or {}),
            "lookback": lookback,
        },
        level="DEBUG",
        also_overall=False,
    )
    returns = _log_returns_from_closes(_closes(list(candles or [])[-int(lookback):]))
    if len(returns) < 30:
        return QuantContextSignal(product_id, 0.0, 0.0, 0.0, 0.0, 0.20, 0.50, 0.50, 0.0, 0.0, 0.0, 0.0, "insufficient_data", "no_boundary", 0.0, 0.0, 0.50, 0.75, 0.10, "", 0.0, "no_peer_context", "insufficient_returns", f"quant_context_unavailable;returns={len(returns)}")
    mean_ret, std_ret = _mean(returns[-80:]), _std(returns[-80:])
    ac1, ac3 = _autocorr(returns[-100:], 1), _autocorr(returns[-100:], 3)
    ewma = _ewma_vol(returns[-120:])
    stat = _stationarity_proxy(returns[-180:])
    stationarity_score = stat["stationarity_score"]
    last_ret = returns[-1]
    forecast = mean_ret + ac1 * (last_ret - mean_ret) * 0.55 + ac3 * mean_ret * 0.20
    upper = forecast + 1.96 * max(std_ret, ewma)
    lower = forecast - 1.96 * max(std_ret, ewma)
    vol_ratio = _std(returns[-20:]) / max(1e-9, _std(returns[-120:]))
    vol_state = "volatility_expansion_cluster" if vol_ratio >= 1.65 else "volatility_compression" if vol_ratio <= 0.70 else "normal_volatility"
    if last_ret > upper:
        boundary_state = "above_upper_boundary_stretched"
    elif last_ret < lower:
        boundary_state = "below_lower_boundary_stretched"
    elif forecast > 8:
        boundary_state = "positive_return_drift"
    elif forecast < -8:
        boundary_state = "negative_return_drift"
    else:
        boundary_state = "inside_quant_boundary"
    peer = _peer_relative_context(product_id=product_id, returns=returns, peer_returns_by_product=peer_returns_by_product)
    peer_state = str(peer.get("peer_state", ""))
    buy_score = _clamp(0.24 + (0.28 if forecast > 0 else 0.0) + (0.22 if boundary_state == "below_lower_boundary_stretched" else 0.0) + (0.16 if stationarity_score >= 0.55 else 0.0) + (0.18 if peer_state == "product_cheap_vs_peer_snapback_possible" else 0.0) - (0.24 if boundary_state == "above_upper_boundary_stretched" else 0.0) - (0.18 if vol_state == "volatility_expansion_cluster" and forecast < 0 else 0.0))
    sell_score = _clamp(0.22 + (0.28 if forecast < 0 else 0.0) + (0.22 if boundary_state == "above_upper_boundary_stretched" else 0.0) + (0.18 if peer_state == "product_rich_vs_peer_snapback_risk" else 0.0) + (0.14 if vol_state == "volatility_expansion_cluster" and forecast < 0 else 0.0) - (0.18 if boundary_state == "below_lower_boundary_stretched" else 0.0))
    hold_score = _clamp(0.40 + (0.20 if forecast > 0 and boundary_state != "above_upper_boundary_stretched" else 0.0) + (0.14 if vol_state == "normal_volatility" else 0.0) - (0.18 if sell_score >= 0.60 else 0.0))
    wait_score = _clamp(0.36 + (0.24 if stationarity_score < 0.42 else 0.0) + (0.18 if vol_state == "volatility_expansion_cluster" and abs(forecast) < 8 else 0.0) + (0.18 if boundary_state == "inside_quant_boundary" and abs(forecast) < 6 else 0.0))
    confidence = _clamp(0.28 + min(0.20, len(returns) / 400.0) + stationarity_score * 0.22 + min(0.16, abs(forecast) / 65.0) + (0.10 if peer_state not in {"no_peer_context", "insufficient_peer_data"} else 0.0))
    debug_every(
        MODULE_NAME,
        f"quant_context_result:{product_id}",
        30.0,
        "quant_context_result",
        data={
            "product_id": product_id,
            "boundary_state": boundary_state,
            "volatility_cluster_state": vol_state,
            "forecast_return_bps": forecast,
            "stationarity_score": stationarity_score,
            "peer_product": peer.get("peer_product", ""),
            "peer_state": peer_state,
            "peer_spread_z": peer.get("peer_spread_z", 0.0),
            "buy_score": buy_score,
            "sell_score": sell_score,
            "hold_score": hold_score,
            "wait_score": wait_score,
            "confidence": confidence,
        },
        level="DEBUG",
        also_overall=False,
    )
    return QuantContextSignal(product_id, float(mean_ret), float(std_ret), float(ac1), float(ac3), float(stationarity_score), float(stat["mean_drift_score"]), float(stat["variance_drift_score"]), float(forecast), float(upper), float(lower), float(ewma), vol_state, boundary_state, float(buy_score), float(sell_score), float(hold_score), float(wait_score), float(confidence), str(peer.get("peer_product", "")), float(peer.get("peer_spread_z", 0.0)), peer_state, str(peer.get("peer_reason", "")), f"quant_context;boundary={boundary_state};vol_state={vol_state};forecast={forecast:.2f};upper={upper:.2f};lower={lower:.2f};mean={mean_ret:.2f};std={std_ret:.2f};ewma_vol={ewma:.2f};ac1={ac1:.3f};ac3={ac3:.3f};stationarity={stationarity_score:.3f};{peer.get('peer_reason', '')}")


def signal_to_dict(signal: QuantContextSignal) -> Dict[str, Any]:
    return asdict(signal)
