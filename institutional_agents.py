import os, pickle
from typing import Any, Dict, List, Tuple
import pandas as pd
from institutional_math import brier_score
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    ColumnTransformer = None

DECISION_TIME_NUMERIC_FEATURES = ["score","probability","estimated_prob_up","expected_net_edge_bps","projected_forward_gain_bps","target_bps","cost_bps","spread_bps","poc_distance_bps","low_volume_path_up_bps","low_volume_path_down_bps","market_structure_buy_score","validated_liquidity_buy_score","candle_sequence_score","candle_exhaustion_score","candle_continuation_score","previous_session_profile_buy_score","previous_session_profile_wait_score","volume_profile_leader_buy_score","volume_profile_leader_wait_score","quant_buy_score","quant_wait_score","quant_stationarity_score","quant_forecast_return_bps","quant_conditional_volatility_bps","quant_peer_spread_z","bayesian_setup_pattern_edge_buy_score"]
DECISION_TIME_CATEGORICAL_FEATURES = ["product_id","timeframe","strategy_variant","setup_tag","market_regime","structure_state","value_area_state","value_acceptance_state","volume_node_state","fvg_state","previous_session_profile_reaction_state","previous_session_profile_bias","quant_boundary_state","quant_volatility_cluster_state","quant_peer_state"]

def _label_column(frame: pd.DataFrame) -> str:
    for c in ["buy_net_bps","net_bps","outcome_bps","realized_net_pnl_bps"]:
        if c in frame.columns: return c
    return "buy_net_bps"

def make_buy_labels(frame: pd.DataFrame) -> pd.Series:
    c = _label_column(frame)
    if c not in frame.columns: return pd.Series([0] * len(frame), index=frame.index)
    return (pd.to_numeric(frame[c], errors="coerce").fillna(0.0) > 0.0).astype(int)

def make_feature_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric = [c for c in DECISION_TIME_NUMERIC_FEATURES if c in frame.columns]
    categorical = [c for c in DECISION_TIME_CATEGORICAL_FEATURES if c in frame.columns]
    X = frame[numeric + categorical].copy() if numeric or categorical else pd.DataFrame(index=frame.index)
    for c in numeric: X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical: X[c] = X[c].fillna("").astype(str)
    return X, numeric, categorical

def chronological_train_test_split(frame: pd.DataFrame, test_frac: float = 0.30):
    f = frame.copy()
    if "ts" in f.columns: f = f.sort_values("ts")
    split = max(1, int(len(f) * (1.0 - float(test_frac))))
    return f.iloc[:split].copy(), f.iloc[split:].copy()

def _build_preprocessor(numeric, categorical):
    transformers = []
    if numeric: transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical: transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical))
    return ColumnTransformer(transformers=transformers)

def _train(frame, kind):
    if ColumnTransformer is None or frame is None or len(frame) < (120 if kind == "tree" else 80): return {"ok": False, "reason": "sklearn_unavailable_or_too_few_rows"}
    train, test = chronological_train_test_split(frame); X_train,numeric,categorical = make_feature_frame(train); X_test,_,_ = make_feature_frame(test); y_train = make_buy_labels(train); y_test = make_buy_labels(test)
    if X_train.empty or y_train.nunique() < 2 or y_test.nunique() < 2: return {"ok": False, "reason": "one_class_labels"}
    if kind == "tree":
        base = RandomForestClassifier(n_estimators=240, max_depth=7, min_samples_leaf=25, class_weight="balanced_subsample", random_state=42, n_jobs=-1); method = "isotonic"
    else:
        base = LogisticRegression(max_iter=1000, C=0.35, class_weight="balanced", solver="lbfgs"); method = "sigmoid"
    model = CalibratedClassifierCV(Pipeline([("pre", _build_preprocessor(numeric, categorical)), ("model", base)]), method=method, cv=3)
    model.fit(X_train, y_train); p = model.predict_proba(X_test)[:,1]
    auc = float(roc_auc_score(y_test, p)) if len(set(y_test)) > 1 else 0.5; brier = brier_score(p, y_test)
    return {"ok": True, "model": model, "features_numeric": numeric, "features_categorical": categorical, "auc": auc, "brier": brier, "test_rows": int(len(test)), "reason": f"{kind}_agent_trained;auc={auc:.3f};brier={brier:.3f}"}

def train_logistic_meta_agent(frame): return _train(frame, "logistic_meta")
def train_tree_regime_agent(frame): return _train(frame, "tree")
def save_agent_model(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True); tmp = path + ".tmp"
    with open(tmp, "wb") as f: pickle.dump(payload, f)
    os.replace(tmp, path)
def load_agent_model(path: str) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except Exception as exc: return {"ok": False, "reason": f"load_failed:{exc}"}
def predict_meta_probability(model_payload: Dict[str, Any], row: Dict[str, Any]) -> float:
    if not model_payload or not model_payload.get("ok") or "model" not in model_payload: return 0.50
    try:
        X,_,_ = make_feature_frame(pd.DataFrame([row])); return float(max(0.0, min(1.0, model_payload["model"].predict_proba(X)[:,1][0])))
    except Exception: return 0.50
