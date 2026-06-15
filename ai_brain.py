import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SIGNAL_EVENTS_CSV = os.path.join(BASE_DIR, "signal_events.csv")
TRADE_OUTCOMES_CSV = os.path.join(BASE_DIR, "trade_outcomes.csv")
AI_MODEL_PATH = os.path.join(BASE_DIR, "ai_brain.joblib")
AI_PREDICTIONS_CSV = os.path.join(BASE_DIR, "ai_predictions.csv")

FEATURE_COLUMNS = [
    "score",
    "probability",
    "ev_bps",
    "projected_forward_bps",
    "cost_bps",
    "spread_bps",
    "momentum_1_bps",
    "momentum_3_bps",
    "momentum_5_bps",
    "momentum_15_bps",
    "green_candles",
    "rank_score",
    "buy_ready_count",
]


@dataclass
class AIDecision:
    product_id: str
    action: str
    confidence: float
    prob_up_5m: float
    prob_up_15m: float
    prob_up_30m: float
    expected_move_30m_bps: float
    expected_adverse_bps: float
    reason: str


class LocalAIBrain:
    def __init__(self, min_training_rows: int = 30) -> None:
        self.min_training_rows = max(2, int(min_training_rows))
        self.model_pack: Optional[Dict[str, Any]] = None
        self.load()

    def load(self) -> None:
        if not os.path.exists(AI_MODEL_PATH):
            return
        try:
            pack = joblib.load(AI_MODEL_PATH)
            if all(
                key in pack
                for key in ("classifier", "move_regressor", "adverse_regressor")
            ):
                self.model_pack = pack
        except Exception:
            self.model_pack = None

    def ready(self) -> bool:
        return bool(self.model_pack)

    @staticmethod
    def _safe_read(path: str) -> pd.DataFrame:
        try:
            if not os.path.exists(path):
                return pd.DataFrame()
            frame = pd.read_csv(path)
            return frame if not frame.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def build_training_frame(self) -> pd.DataFrame:
        outcomes = self._safe_read(TRADE_OUTCOMES_CSV)
        signals = self._safe_read(SIGNAL_EVENTS_CSV)
        if outcomes.empty or signals.empty:
            return pd.DataFrame()
        if "event_type" not in signals.columns:
            return pd.DataFrame()
        if "trade_id" not in signals.columns or "trade_id" not in outcomes.columns:
            return pd.DataFrame()

        entry = signals[
            signals["event_type"].astype(str).isin(
                [
                    "buy_attempt",
                    "inverted_buy_attempt",
                    "inverted_buy_fill",
                    "buy_fill",
                ]
            )
        ].copy()
        if entry.empty:
            return pd.DataFrame()

        # Keep one entry context per trade so duplicate attempt/fill events do not
        # overweight a trade during training.
        if "ts" in entry.columns:
            entry["_entry_ts"] = pd.to_numeric(entry["ts"], errors="coerce")
            entry = entry.sort_values("_entry_ts")
        entry = entry.drop_duplicates(subset=["trade_id"], keep="last")

        frame = outcomes.merge(
            entry,
            on="trade_id",
            how="inner",
            suffixes=("_outcome", "_entry"),
        )
        if "review_minutes" not in frame.columns:
            return pd.DataFrame()
        frame = frame[
            pd.to_numeric(frame["review_minutes"], errors="coerce") == 30
        ].copy()
        if frame.empty or "move_bps" not in frame.columns:
            return pd.DataFrame()

        for column in FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        frame["move_bps"] = pd.to_numeric(
            frame["move_bps"], errors="coerce"
        ).fillna(0.0)
        if "max_adverse_bps" not in frame.columns:
            frame["max_adverse_bps"] = 0.0
        frame["max_adverse_bps"] = pd.to_numeric(
            frame["max_adverse_bps"], errors="coerce"
        ).fillna(0.0)
        frame["y_up_30m"] = (frame["move_bps"] > 0.0).astype(int)
        frame["y_move_30m_bps"] = frame["move_bps"].astype(float)
        frame["y_adverse_bps"] = frame["max_adverse_bps"].abs().astype(float)
        return frame

    def train(self) -> Dict[str, Any]:
        frame = self.build_training_frame()
        if len(frame) < self.min_training_rows:
            return {
                "ok": False,
                "reason": (
                    f"not_enough_training_rows rows={len(frame)} "
                    f"required={self.min_training_rows}"
                ),
            }

        features = frame[FEATURE_COLUMNS].copy()
        classification_target = frame["y_up_30m"].astype(int)
        if classification_target.nunique() < 2:
            return {
                "ok": False,
                "reason": "not_enough_label_classes required=2",
            }

        classifier = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15
        )
        move_regressor = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15
        )
        adverse_regressor = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15
        )

        stratify = classification_target
        if classification_target.value_counts().min() < 2:
            stratify = None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            classification_target,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
        classifier.fit(x_train, y_train)
        move_regressor.fit(features, frame["y_move_30m_bps"].astype(float))
        adverse_regressor.fit(features, frame["y_adverse_bps"].astype(float))

        auc = None
        if y_test.nunique() > 1:
            try:
                probability = classifier.predict_proba(x_test)[:, 1]
                auc = float(roc_auc_score(y_test, probability))
            except (ValueError, IndexError):
                auc = None

        pack = {
            "feature_columns": FEATURE_COLUMNS,
            "classifier": classifier,
            "move_regressor": move_regressor,
            "adverse_regressor": adverse_regressor,
            "rows": int(len(frame)),
            "auc": auc,
        }
        joblib.dump(pack, AI_MODEL_PATH)
        self.model_pack = pack
        return {"ok": True, "rows": int(len(frame)), "auc": auc}

    @staticmethod
    def _row_to_features(context: Dict[str, Any]) -> pd.DataFrame:
        row: Dict[str, float] = {}
        for column in FEATURE_COLUMNS:
            try:
                row[column] = float(context.get(column, 0.0) or 0.0)
            except (TypeError, ValueError):
                row[column] = 0.0
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, product_id: str, context: Dict[str, Any]) -> AIDecision:
        if not self.ready():
            return AIDecision(
                product_id=product_id,
                action="NO_AI_MODEL",
                confidence=0.0,
                prob_up_5m=0.0,
                prob_up_15m=0.0,
                prob_up_30m=0.0,
                expected_move_30m_bps=0.0,
                expected_adverse_bps=0.0,
                reason="AI model not trained yet",
            )

        features = self._row_to_features(context)
        classifier = self.model_pack["classifier"]
        move_regressor = self.model_pack["move_regressor"]
        adverse_regressor = self.model_pack["adverse_regressor"]

        try:
            prob_up_30m = float(classifier.predict_proba(features)[0, 1])
        except (ValueError, IndexError):
            prob_up_30m = 0.5
        expected_move = float(move_regressor.predict(features)[0])
        expected_adverse = max(0.0, float(adverse_regressor.predict(features)[0]))

        if prob_up_30m >= 0.62 and expected_move > expected_adverse:
            action = "ALLOW_BUY"
        elif (
            prob_up_30m <= 0.42
            or expected_adverse > max(25.0, expected_move * 1.25)
        ):
            action = "BLOCK_BUY"
        else:
            action = "WAIT"

        decision = AIDecision(
            product_id=product_id,
            action=action,
            confidence=float(abs(prob_up_30m - 0.5) * 2.0),
            prob_up_5m=prob_up_30m,
            prob_up_15m=prob_up_30m,
            prob_up_30m=prob_up_30m,
            expected_move_30m_bps=expected_move,
            expected_adverse_bps=expected_adverse,
            reason=(
                f"prob_up_30m={prob_up_30m:.4f};"
                f"expected_move_30m_bps={expected_move:.2f};"
                f"expected_adverse_bps={expected_adverse:.2f}"
            ),
        )
        self.log_prediction(product_id, context, decision)
        return decision

    @staticmethod
    def log_prediction(
        product_id: str,
        context: Dict[str, Any],
        decision: AIDecision,
    ) -> None:
        exists = os.path.exists(AI_PREDICTIONS_CSV)
        row = {
            "ts": context.get("ts", 0.0),
            "product_id": product_id,
            "action": decision.action,
            "confidence": decision.confidence,
            "prob_up_30m": decision.prob_up_30m,
            "expected_move_30m_bps": decision.expected_move_30m_bps,
            "expected_adverse_bps": decision.expected_adverse_bps,
            "reason": decision.reason,
        }
        for column in FEATURE_COLUMNS:
            row[column] = context.get(column, 0.0)
        pd.DataFrame([row]).to_csv(
            AI_PREDICTIONS_CSV,
            mode="a",
            header=not exists,
            index=False,
        )
