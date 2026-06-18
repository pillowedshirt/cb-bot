import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SIGNAL_EVENTS_CSV = os.path.join(BASE_DIR, "signal_events.csv")
TRADE_OUTCOMES_CSV = os.path.join(BASE_DIR, "trade_outcomes.csv")
COUNCIL_OBSERVATION_OUTCOMES_CSV = os.path.join(BASE_DIR, "council_observation_outcomes.csv")
COUNCIL_DECISIONS_CSV = os.path.join(BASE_DIR, "council_decisions.csv")
DECISION_AUDIT_CSV = os.path.join(BASE_DIR, "decision_audit.csv")
SELL_QUALITY_REVIEWS_CSV = os.path.join(BASE_DIR, "sell_quality_reviews.csv")
AI_MODEL_PATH = os.path.join(BASE_DIR, "ai_brain.joblib")
AI_PREDICTIONS_CSV = os.path.join(BASE_DIR, "ai_predictions.csv")
AI_FEATURE_IMPORTANCE_CSV = os.path.join(BASE_DIR, "ai_feature_importance.csv")
AI_FEATURE_IMPORTANCE_CSV_PATH = AI_FEATURE_IMPORTANCE_CSV


def ensure_ai_feature_importance_file() -> None:
    columns = ["ts", "dt_utc", "feature", "importance", "rank", "model_ready", "reason"]
    if os.path.exists(AI_FEATURE_IMPORTANCE_CSV_PATH) and os.path.getsize(AI_FEATURE_IMPORTANCE_CSV_PATH) > 0:
        return
    tmp = AI_FEATURE_IMPORTANCE_CSV_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
    os.replace(tmp, AI_FEATURE_IMPORTANCE_CSV_PATH)


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
    "expected_utility_bps",
    "buy_vs_wait_edge_bps",
    "calibrated_p_win",
    "payoff_ratio",
    "maker_adjusted_expected_value_bps",
    "uncertainty_penalty_bps",
    "volume_profile_leader_buy_score",
    "volume_profile_leader_sell_score",
    "volume_profile_leader_hold_score",
    "volume_profile_leader_wait_score",
    "volume_profile_leader_confidence",
    "poc_distance_bps",
    "low_volume_path_up_bps",
    "low_volume_path_down_bps",
    "unfair_trade_score",
    "volume_profile_utility_adjust_bps",
    "candle_context_buy_score",
    "candle_context_sell_score",
    "candle_exhaustion_score",
    "candle_continuation_score",
    "market_structure_buy_score",
    "validated_liquidity_buy_score",
    "fresh_zone_buy_score",
    "fvg_buy_score",
    "smt_buy_score",
    "previous_session_profile_buy_score",
    "previous_session_profile_sell_score",
    "previous_session_profile_hold_score",
    "previous_session_profile_wait_score",
    "previous_session_profile_confidence",
    "previous_session_profile_poc",
    "previous_session_profile_vah",
    "previous_session_profile_val",
    "previous_session_profile_utility_adjust_bps",
    "quant_buy_score",
    "quant_sell_score",
    "quant_hold_score",
    "quant_wait_score",
    "quant_confidence",
    "quant_stationarity_score",
    "quant_forecast_return_bps",
    "quant_conditional_volatility_bps",
    "quant_peer_spread_z",
    "quant_context_utility_adjust_bps",
    "order_book_imbalance",
    "order_book_bid_depth_usd",
    "order_book_ask_depth_usd",
    "order_book_top_depth_usd",
    "spread_instability_bps",
    "liquidity_risk_score",
    "market_data_age_sec",
    "viewer_snapshot_age_sec",
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
        ensure_ai_feature_importance_file()
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
        """
        Train from both real trade outcomes and Level 8 chart-only observation outcomes.

        This lets the local AI learn from:
        - actual filled buys
        - missed opportunities
        - WAIT / SHADOW decisions that later moved
        - Level 8 council decisions reviewed at 5/15/30/60 minutes
        """
        signals = self._safe_read(SIGNAL_EVENTS_CSV)
        trade_outcomes = self._safe_read(TRADE_OUTCOMES_CSV)
        observation_outcomes = self._safe_read(COUNCIL_OBSERVATION_OUTCOMES_CSV)
        council_decisions = self._safe_read(COUNCIL_DECISIONS_CSV)
        decision_audit = self._safe_read(DECISION_AUDIT_CSV)
        sell_quality = self._safe_read(SELL_QUALITY_REVIEWS_CSV)

        frames = []

        if not trade_outcomes.empty and not signals.empty:
            if (
                "event_type" in signals.columns
                and "trade_id" in signals.columns
                and "trade_id" in trade_outcomes.columns
            ):
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

                if not entry.empty:
                    if "ts" in entry.columns:
                        entry["_entry_ts"] = pd.to_numeric(entry["ts"], errors="coerce")
                        entry = entry.sort_values("_entry_ts")

                    entry = entry.drop_duplicates(subset=["trade_id"], keep="last")

                    trade_frame = trade_outcomes.merge(
                        entry,
                        on="trade_id",
                        how="inner",
                        suffixes=("_outcome", "_entry"),
                    )

                    if not trade_frame.empty:
                        trade_frame["training_source"] = "trade_outcome"
                        frames.append(trade_frame)

        if not observation_outcomes.empty:
            obs = observation_outcomes.copy()

            if "review_minutes" in obs.columns:
                obs["review_minutes"] = pd.to_numeric(obs["review_minutes"], errors="coerce")

            if "move_bps" in obs.columns:
                obs["move_bps"] = pd.to_numeric(obs["move_bps"], errors="coerce").fillna(0.0)

            if (
                not signals.empty
                and "event_type" in signals.columns
                and "trade_id" in signals.columns
                and "decision_id" in obs.columns
            ):
                l8_events = signals[
                    signals["event_type"].astype(str).isin(
                        [
                            "level8_council_heartbeat",
                            "level8_council_decision",
                        ]
                    )
                ].copy()

                if not l8_events.empty:
                    obs = obs.merge(
                        l8_events,
                        left_on="decision_id",
                        right_on="trade_id",
                        how="left",
                        suffixes=("_outcome", "_entry"),
                    )

            obs["training_source"] = "level8_observation"
            frames.append(obs)

        if not council_decisions.empty:
            decisions = council_decisions.copy()
            if "review_minutes" not in decisions.columns:
                decisions["review_minutes"] = 30
            if "move_bps" not in decisions.columns:
                if "projected_forward_gain_bps" in decisions.columns:
                    decisions["move_bps"] = pd.to_numeric(decisions["projected_forward_gain_bps"], errors="coerce").fillna(0.0)
                else:
                    decisions["move_bps"] = 0.0
            decisions["training_source"] = "council_decision"
            frames.append(decisions)

        if not decision_audit.empty:
            audit = decision_audit.copy()
            if "review_minutes" not in audit.columns:
                audit["review_minutes"] = 30
            if "move_bps" not in audit.columns:
                for candidate_column in ["future_move_bps", "max_favorable_bps", "realized_net_pnl_bps"]:
                    if candidate_column in audit.columns:
                        audit["move_bps"] = pd.to_numeric(audit[candidate_column], errors="coerce").fillna(0.0)
                        break
                else:
                    audit["move_bps"] = 0.0
            audit["training_source"] = "decision_audit"
            frames.append(audit)

        if not sell_quality.empty:
            quality = sell_quality.copy()
            if "review_minutes" not in quality.columns:
                quality["review_minutes"] = 30
            if "move_bps" not in quality.columns:
                for candidate_column in ["move_after_sell_bps", "missed_upside_bps", "capture_quality_score"]:
                    if candidate_column in quality.columns:
                        quality["move_bps"] = pd.to_numeric(quality[candidate_column], errors="coerce").fillna(0.0)
                        break
                else:
                    quality["move_bps"] = 0.0
            quality["training_source"] = "sell_quality_review"
            frames.append(quality)

        if not frames:
            return pd.DataFrame()

        frame = pd.concat(frames, ignore_index=True, sort=False)

        if "review_minutes" not in frame.columns:
            return pd.DataFrame()

        frame["review_minutes"] = pd.to_numeric(frame["review_minutes"], errors="coerce")
        frame = frame[frame["review_minutes"].isin([15, 30, 60])].copy()

        if frame.empty or "move_bps" not in frame.columns:
            return pd.DataFrame()

        for column in FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0

            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        # Observation rows are useful, but only if they actually carried enough
        # feature context. If the merge failed and most features are zero, do not
        # let those sparse rows teach the AI model.
        feature_activity = (frame[FEATURE_COLUMNS].abs() > 1e-12).sum(axis=1)
        frame["feature_quality"] = feature_activity / max(float(len(FEATURE_COLUMNS)), 1.0)

        if "training_source" in frame.columns:
            keep_quality = (
                frame["training_source"].astype(str).ne("level8_observation")
                | frame["feature_quality"].ge(0.25)
            )
            frame = frame[keep_quality].copy()
        else:
            frame = frame[frame["feature_quality"].ge(0.25)].copy()

        if frame.empty:
            return pd.DataFrame()

        frame["move_bps"] = pd.to_numeric(frame["move_bps"], errors="coerce").fillna(0.0)

        if "max_adverse_bps" not in frame.columns:
            frame["max_adverse_bps"] = 0.0

        frame["max_adverse_bps"] = pd.to_numeric(
            frame["max_adverse_bps"], errors="coerce"
        ).fillna(0.0)

        frame["y_up_30m"] = (frame["move_bps"] > 0.0).astype(int)
        frame["y_move_30m_bps"] = frame["move_bps"].astype(float)
        frame["y_adverse_bps"] = frame["max_adverse_bps"].abs().astype(float)

        return frame

    def _write_feature_importance_report(self, frame: pd.DataFrame) -> None:
        """Write a lightweight feature-importance proxy for AI training inputs."""
        try:
            if frame.empty:
                return
            rows = []
            report_ts = float(pd.Timestamp.utcnow().timestamp())
            report_dt = pd.Timestamp.utcnow().isoformat()
            y_move = pd.to_numeric(frame["y_move_30m_bps"], errors="coerce").fillna(0.0)
            y_up = pd.to_numeric(frame["y_up_30m"], errors="coerce").fillna(0)
            winners = frame[y_up == 1]
            losers = frame[y_up == 0]
            for column in FEATURE_COLUMNS:
                x = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
                corr = 0.0
                try:
                    if x.std() > 0 and y_move.std() > 0:
                        corr = float(x.corr(y_move))
                except Exception:
                    corr = 0.0
                winner_mean = float(pd.to_numeric(winners[column], errors="coerce").fillna(0.0).mean()) if not winners.empty else 0.0
                loser_mean = float(pd.to_numeric(losers[column], errors="coerce").fillna(0.0).mean()) if not losers.empty else 0.0
                importance = abs(float(corr or 0.0))
                rows.append({
                    "ts": report_ts,
                    "dt_utc": report_dt,
                    "feature": column,
                    "importance": importance,
                    "rank": 0,
                    "model_ready": True,
                    "reason": "correlation_proxy_after_training",
                    "abs_corr_to_move": importance,
                    "corr_to_move": float(corr or 0.0),
                    "winner_mean": winner_mean,
                    "loser_mean": loser_mean,
                    "winner_minus_loser": winner_mean - loser_mean,
                    "sample_count": int(len(frame)),
                })
            out = pd.DataFrame(rows).sort_values(
                ["importance", "winner_minus_loser"],
                ascending=[False, False],
            )
            out["rank"] = list(range(1, len(out) + 1))
            base_cols = ["ts", "dt_utc", "feature", "importance", "rank", "model_ready", "reason"]
            extra_cols = [c for c in out.columns if c not in base_cols]
            out[base_cols + extra_cols].to_csv(AI_FEATURE_IMPORTANCE_CSV_PATH, index=False)
        except Exception:
            pass

    def train(self) -> Dict[str, Any]:
        frame = self.build_training_frame()
        module_debug(
            MODULE_NAME,
            "ai_train_frame_built",
            data={
                "rows": int(len(frame)),
                "columns": list(frame.columns)[:120] if not frame.empty else [],
                "min_training_rows": int(self.min_training_rows),
            },
            level="INFO",
            also_overall=False,
        )
        if len(frame) < self.min_training_rows:
            ensure_ai_feature_importance_file()
            module_debug(
                MODULE_NAME,
                "ai_train_skipped_not_enough_rows",
                data={
                    "rows": int(len(frame)),
                    "required": int(self.min_training_rows),
                    "state": "normal_early_learning",
                    "message": "AI training starts after enough reviewed rows exist.",
                },
                level="INFO",
                also_overall=False,
            )
            return {
                "ok": False,
                "sample_count": int(len(frame)),
                "feature_columns_used": list(FEATURE_COLUMNS),
                "auc_if_available": None,
                "model_ready": False,
                "reason": (
                    f"not_enough_training_rows rows={len(frame)} "
                    f"required={self.min_training_rows}"
                ),
            }

        features = frame[FEATURE_COLUMNS].copy()
        self._write_feature_importance_report(frame)
        classification_target = frame["y_up_30m"].astype(int)
        if classification_target.nunique() < 2:
            return {
                "ok": False,
                "sample_count": int(len(frame)),
                "feature_columns_used": list(FEATURE_COLUMNS),
                "auc_if_available": None,
                "model_ready": False,
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
        module_debug(
            MODULE_NAME,
            "ai_train_success",
            data={
                "sample_count": int(len(frame)),
                "auc_if_available": auc,
                "feature_count": len(FEATURE_COLUMNS),
                "model_path": AI_MODEL_PATH,
            },
            level="INFO",
            also_overall=True,
        )
        return {
            "ok": True,
            "sample_count": int(len(frame)),
            "feature_columns_used": list(FEATURE_COLUMNS),
            "auc_if_available": auc,
            "model_ready": True,
            "reason": "trained",
        }

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
        feature_nonzero_count = int((features.abs() > 1e-12).sum(axis=1).iloc[0])
        feature_coverage = feature_nonzero_count / max(float(len(FEATURE_COLUMNS)), 1.0)
        debug_every(
            MODULE_NAME,
            f"ai_predict:{product_id}",
            15.0,
            "ai_predict_feature_coverage",
            data={
                "product_id": product_id,
                "feature_nonzero_count": feature_nonzero_count,
                "feature_count": len(FEATURE_COLUMNS),
                "feature_coverage": feature_coverage,
            },
            level="DEBUG",
            also_overall=False,
        )
        classifier = self.model_pack["classifier"]
        move_regressor = self.model_pack["move_regressor"]
        adverse_regressor = self.model_pack["adverse_regressor"]

        try:
            prob_up_30m = float(classifier.predict_proba(features)[0, 1])
        except (ValueError, IndexError):
            prob_up_30m = 0.5
        expected_move = float(move_regressor.predict(features)[0])
        expected_adverse = max(0.0, float(adverse_regressor.predict(features)[0]))

        if feature_coverage < 0.35:
            action = "WAIT"
        elif prob_up_30m >= 0.62 and expected_move > expected_adverse:
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
                f"expected_adverse_bps={expected_adverse:.2f};"
                f"feature_coverage={feature_coverage:.3f};"
                f"nonzero_features={feature_nonzero_count};"
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
