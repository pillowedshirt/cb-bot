import math
from typing import Dict, Iterable
import numpy as np
try:
    from scipy.stats import beta as beta_dist
except Exception:
    beta_dist = None

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try: return max(low, min(high, float(value)))
    except Exception: return low

def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0: return 0.0
    phat = wins / n; denom = 1.0 + z*z/n; center = phat + z*z/(2*n)
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n)
    return clamp((center - margin) / denom)

def beta_posterior_mean(wins: int, losses: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    return clamp((alpha + wins) / max(alpha + beta + wins + losses, 1e-9))

def beta_lower_quantile(wins: int, losses: int, alpha: float = 1.0, beta: float = 1.0, q: float = 0.05) -> float:
    if beta_dist is not None:
        try: return clamp(float(beta_dist.ppf(q, alpha + wins, beta + losses)))
        except Exception: pass
    return wilson_lower_bound(wins, wins + losses, z=1.645)

def breakeven_probability(avg_win_bps: float, avg_loss_bps: float) -> float:
    win = max(float(avg_win_bps), 1e-9); loss = max(abs(float(avg_loss_bps)), 1e-9)
    return clamp(loss / (win + loss))

def conservative_ev_bps(p_low: float, avg_win_bps: float, avg_loss_bps: float) -> float:
    p = clamp(p_low); return p * max(float(avg_win_bps), 0.0) - (1.0 - p) * max(abs(float(avg_loss_bps)), 0.0)

def payoff_ratio(avg_win_bps: float, avg_loss_bps: float) -> float:
    return max(float(avg_win_bps), 0.0) / max(abs(float(avg_loss_bps)), 1e-9)

def fractional_kelly(p: float, avg_win_bps: float, avg_loss_bps: float, fraction: float = 0.25) -> float:
    r = payoff_ratio(avg_win_bps, avg_loss_bps)
    return 0.0 if r <= 0 else clamp((float(p) - (1.0 - float(p)) / r) * float(fraction))

def brier_score(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    p = np.asarray(list(probabilities), dtype=float); y = np.asarray(list(outcomes), dtype=float)
    if len(p) == 0 or len(y) == 0 or len(p) != len(y): return 1.0
    return float(np.mean((p - y) ** 2))

def t_stat(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float); arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return 0.0
    std = float(np.std(arr, ddof=1))
    return 0.0 if std <= 1e-12 else float(np.mean(arr) / (std / math.sqrt(len(arr))))

def deflated_edge_score(*, avg_net_bps: float, t_value: float, trials: int, sample_count: int, skew_penalty: float = 0.0, kurtosis_penalty: float = 0.0) -> float:
    penalty = math.sqrt(max(1.0, math.log(max(2, int(trials))))) * math.sqrt(60.0 / max(60.0, float(sample_count))) + abs(skew_penalty)*0.10 + abs(kurtosis_penalty)*0.03
    return float(avg_net_bps) + float(t_value) * 2.0 - penalty * 3.0

def institutional_agent_score(*, wins: int, losses: int, avg_win_bps: float, avg_loss_bps: float, avg_net_bps: float, median_net_bps: float, hard_stop_rate: float, early_adverse_rate: float, brier: float, trials: int, t_value: float) -> Dict[str, float]:
    n = int(wins) + int(losses); p_mean = beta_posterior_mean(wins, losses); p_low = beta_lower_quantile(wins, losses); p_break = breakeven_probability(avg_win_bps, avg_loss_bps)
    ev_low = conservative_ev_bps(p_low, avg_win_bps, avg_loss_bps); r = payoff_ratio(avg_win_bps, avg_loss_bps); kelly = fractional_kelly(p_mean, avg_win_bps, avg_loss_bps)
    dscore = deflated_edge_score(avg_net_bps=avg_net_bps, t_value=t_value, trials=trials, sample_count=n)
    score = ev_low + max(0.0, avg_net_bps)*0.30 + max(0.0, median_net_bps)*0.20 + max(0.0, p_low-p_break)*80.0 + max(0.0, r-1.0)*6.0 + dscore*0.25 - hard_stop_rate*45.0 - early_adverse_rate*25.0 - max(0.0, brier-0.22)*45.0
    return {"n": float(n), "p_mean": p_mean, "p_low": p_low, "p_break_even": p_break, "ev_low_bps": ev_low, "payoff_ratio": r, "fractional_kelly": kelly, "deflated_edge_score": dscore, "institutional_score": float(score)}
