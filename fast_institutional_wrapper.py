import numpy as np
try:
    import fast_institutional_core as _core
except Exception:
    _core = None

def has_fast_core() -> bool: return _core is not None

def evaluate_agent_thresholds(scores, outcomes, adverse, thresholds):
    scores = np.asarray(scores, dtype=float); outcomes = np.asarray(outcomes, dtype=float); adverse = np.asarray(adverse, dtype=float); thresholds = np.asarray(thresholds, dtype=float)
    if _core is not None: return _core.evaluate_agent_thresholds(scores, outcomes, adverse, thresholds)
    rows = []
    for th in thresholds:
        mask = scores >= th; vals = outcomes[mask]; adv = adverse[mask]
        if len(vals) == 0: rows.append({"threshold": float(th), "selected": 0, "win_rate": 0.0, "avg_net": 0.0, "median": 0.0, "avg_adverse": 0.0}); continue
        rows.append({"threshold": float(th), "selected": int(len(vals)), "win_rate": float((vals > 0).mean()), "avg_net": float(vals.mean()), "median": float(np.median(vals)), "avg_adverse": float(np.mean(np.abs(adv))) if len(adv) else 0.0})
    return {"rows": rows}
