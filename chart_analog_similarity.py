"""Chart analog similarity scoring for intersection-only buy policy."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math


def _to_float_sequence(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, dict):
        values = values.values()
    try:
        return [float(v or 0.0) for v in values]
    except Exception:
        return []


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n <= 1:
        return 0.0
    aa = list(a)[-n:]; bb = list(b)[-n:]
    ma = sum(aa) / n; mb = sum(bb) / n
    va = sum((x - ma) ** 2 for x in aa); vb = sum((y - mb) ** 2 for y in bb)
    if va <= 1e-12 or vb <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, sum((x - ma) * (y - mb) for x, y in zip(aa, bb)) / math.sqrt(va * vb)))


def _window_vector(window: Dict[str, Any]) -> List[float]:
    parts: List[float] = []
    for key in [
        "normalized_returns", "returns", "volume_sequence", "volume",
        "upper_wick_ratio", "lower_wick_ratio", "close_location", "poc_distance_sequence",
    ]:
        parts.extend(_to_float_sequence(window.get(key)))
    for key in ["liquidity_sweep", "liquidity_reclaim"]:
        parts.append(1.0 if window.get(key) else 0.0)
    return parts


def score_chart_analog_similarity(current_window: Dict[str, Any], historical_windows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a 0..1 analog score plus approval policy fields.

    Approval requires similarity >= 0.80, matched average net bps above +15,
    hard-stop rate below 10%, and win rate above 60%.
    """
    current_vec = _window_vector(current_window or {})
    matches: List[Tuple[float, Dict[str, Any]]] = []
    for hist in historical_windows or []:
        hist_vec = _window_vector(hist or {})
        if not current_vec or not hist_vec:
            continue
        similarity = (_corr(current_vec, hist_vec) + 1.0) / 2.0
        if str(current_window.get("volume_node_state", "")) == str(hist.get("volume_node_state", "")):
            similarity += 0.04
        if str(current_window.get("value_acceptance_state", "")) == str(hist.get("value_acceptance_state", "")):
            similarity += 0.04
        similarity = max(0.0, min(1.0, similarity))
        if similarity >= 0.60:
            matches.append((similarity, hist))
    matches.sort(key=lambda item: item[0], reverse=True)
    top = matches[:25]
    count = len(top)
    if count <= 0:
        return {"matched_count": 0, "avg_similarity": 0.0, "avg_net_bps": 0.0, "median_net_bps": 0.0, "win_rate": 0.0, "hard_stop_rate": 1.0, "early_adverse_rate": 1.0, "chart_analog_similarity_buy_score": 0.0, "approved": False}
    sims = [m[0] for m in top]
    nets = [float(m[1].get("net_bps", m[1].get("buy_net_bps", 0.0)) or 0.0) for m in top]
    hard = [1.0 if bool(m[1].get("hard_stop", False)) or float(m[1].get("max_adverse_bps", 0.0) or 0.0) <= -120.0 else 0.0 for m in top]
    early = [1.0 if bool(m[1].get("early_adverse", False)) else 0.0 for m in top]
    avg_similarity = sum(sims) / count
    avg_net = sum(nets) / count
    med_net = sorted(nets)[count // 2]
    win_rate = sum(1.0 for n in nets if n > 0.0) / count
    hard_stop_rate = sum(hard) / count
    early_adverse_rate = sum(early) / count
    score = max(0.0, min(1.0, avg_similarity * 0.55 + max(0.0, min(1.0, avg_net / 60.0)) * 0.25 + win_rate * 0.20 - hard_stop_rate * 0.30))
    approved = bool(avg_similarity >= 0.80 and avg_net > 15.0 and hard_stop_rate < 0.10 and win_rate > 0.60)
    return {"matched_count": count, "avg_similarity": avg_similarity, "avg_net_bps": avg_net, "median_net_bps": med_net, "win_rate": win_rate, "hard_stop_rate": hard_stop_rate, "early_adverse_rate": early_adverse_rate, "chart_analog_similarity_buy_score": score, "approved": approved}
