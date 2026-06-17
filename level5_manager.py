import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd



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

TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")
MARKET_CSV = os.path.join(BASE_DIR, "market.csv")
TRADE_OUTCOMES_CSV = os.path.join(BASE_DIR, "trade_outcomes.csv")
AI_PREDICTIONS_CSV = os.path.join(BASE_DIR, "ai_predictions.csv")


@dataclass
class ManagerDecision:
    product_id: str
    action: str
    strategy: str
    risk_mode: str
    confidence: float
    max_position_pct: float
    reason: str


class Level5TradingManager:
    """High-level trade admission, strategy selection, and risk manager."""

    def __init__(self) -> None:
        self.last_summary: Dict[str, Any] = {}

    def _read_csv(self, path: str) -> pd.DataFrame:
        try:
            if not os.path.exists(path):
                return pd.DataFrame()
            df = pd.read_csv(path)
            return df if not df.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _recent_trades(self, lookback_rows: int = 50) -> pd.DataFrame:
        df = self._read_csv(TRADES_CSV)
        if df.empty:
            return df
        if "ts" in df.columns:
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
            df = df.sort_values("ts")
        return df.tail(lookback_rows).copy()

    def _recent_outcomes(self, product_id: Optional[str] = None) -> pd.DataFrame:
        df = self._read_csv(TRADE_OUTCOMES_CSV)
        if df.empty:
            return df
        if product_id and "product_id" in df.columns:
            df = df[df["product_id"].astype(str) == str(product_id)]
        if "ts" in df.columns:
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
            df = df.sort_values("ts")
        return df.tail(200).copy()

    def _latest_ai(self, product_id: str) -> Dict[str, Any]:
        df = self._read_csv(AI_PREDICTIONS_CSV)
        if df.empty or "product_id" not in df.columns:
            return {}
        rows = df[df["product_id"].astype(str) == str(product_id)].copy()
        if rows.empty:
            return {}
        if "ts" in rows.columns:
            rows["ts"] = pd.to_numeric(rows["ts"], errors="coerce")
            rows = rows.sort_values("ts")
        return rows.iloc[-1].to_dict()

    def _market_rows(self, product_id: str) -> pd.DataFrame:
        df = self._read_csv(MARKET_CSV)
        if df.empty or "product_id" not in df.columns:
            return pd.DataFrame()
        rows = df[df["product_id"].astype(str) == str(product_id)].copy()
        if "ts" in rows.columns:
            rows["ts"] = pd.to_numeric(rows["ts"], errors="coerce")
            rows = rows.sort_values("ts")
        return rows.tail(120).copy()

    def session_health(self) -> Dict[str, Any]:
        trades = self._recent_trades(80)
        if trades.empty or "net_pnl_usd" not in trades.columns:
            summary = {
                "risk_mode": "NORMAL",
                "session_net": 0.0,
                "closed_count": 0,
                "loss_streak": 0,
                "reason": "no_recent_trade_data",
            }
            self.last_summary = summary
            debug_every(
                MODULE_NAME,
                "session_health",
                30.0,
                "level5_session_health",
                data=summary,
                level="DEBUG",
                also_overall=False,
            )
            return summary

        trades["net_pnl_usd"] = pd.to_numeric(
            trades["net_pnl_usd"], errors="coerce"
        ).fillna(0.0)
        if "event" in trades.columns:
            sells = trades[trades["event"].astype(str).str.upper() == "SELL"].copy()
        else:
            sells = pd.DataFrame(columns=trades.columns)

        session_net = float(trades["net_pnl_usd"].sum())
        closed_count = int(len(sells))
        loss_streak = 0
        if not sells.empty:
            if "ts" in sells.columns:
                sells = sells.sort_values("ts", ascending=False)
            for _, row in sells.iterrows():
                if float(row.get("net_pnl_usd", 0.0)) < 0:
                    loss_streak += 1
                else:
                    break

        if loss_streak >= 4 or session_net <= -2.00:
            risk_mode = "PAUSE"
        elif loss_streak >= 2 or session_net <= -1.00:
            risk_mode = "DEFENSIVE"
        elif session_net > 0.75 and loss_streak == 0:
            risk_mode = "OPPORTUNISTIC"
        else:
            risk_mode = "NORMAL"

        summary = {
            "risk_mode": risk_mode,
            "session_net": session_net,
            "closed_count": closed_count,
            "loss_streak": loss_streak,
            "reason": (
                f"session_net={session_net:.4f};loss_streak={loss_streak};"
                f"closed={closed_count}"
            ),
        }
        self.last_summary = summary
        debug_every(
            MODULE_NAME,
            "session_health",
            30.0,
            "level5_session_health",
            data=summary,
            level="DEBUG",
            also_overall=False,
        )
        return summary

    def product_health(self, product_id: str) -> Dict[str, Any]:
        outcomes = self._recent_outcomes(product_id)
        if outcomes.empty or "move_bps" not in outcomes.columns:
            return {
                "product_ok": True,
                "product_score": 0.0,
                "reason": "no_product_outcomes",
            }

        rows = outcomes.copy()
        rows["move_bps"] = pd.to_numeric(
            rows["move_bps"], errors="coerce"
        ).fillna(0.0)
        if "review_minutes" in rows.columns:
            rows["review_minutes"] = pd.to_numeric(
                rows["review_minutes"], errors="coerce"
            )
            rows_30m = rows[rows["review_minutes"] == 30]
            if not rows_30m.empty:
                rows = rows_30m

        recent = rows.tail(20)
        avg_move = float(recent["move_bps"].mean()) if not recent.empty else 0.0
        win_rate = float((recent["move_bps"] > 0).mean()) if not recent.empty else 0.0
        product_score = avg_move + (win_rate - 0.5) * 100.0

        return {
            "product_ok": bool(product_score > -75.0),
            "product_score": product_score,
            "avg_move_bps": avg_move,
            "win_rate": win_rate,
            "reason": (
                f"avg30m={avg_move:.2f};wr={win_rate:.2f};"
                f"score={product_score:.2f}"
            ),
        }

    def detect_regime(self, product_id: str) -> Dict[str, Any]:
        rows = self._market_rows(product_id)
        if rows.empty or "bid" not in rows.columns:
            return {"regime": "UNKNOWN", "reason": "no_market_rows"}

        price_col = "mid" if "mid" in rows.columns else "bid"
        rows[price_col] = pd.to_numeric(rows[price_col], errors="coerce")
        prices = rows[price_col].dropna().to_numpy()
        if len(prices) < 21:
            return {
                "regime": "UNKNOWN",
                "reason": f"not_enough_market_rows n={len(prices)}",
            }

        ret_5 = ((prices[-1] / prices[-6]) - 1.0) * 10000.0
        ret_20 = ((prices[-1] / prices[-21]) - 1.0) * 10000.0
        recent = prices[-20:]
        volatility = float(
            np.std(np.diff(recent) / recent[:-1]) * 10000.0
        )

        if ret_20 > 45 and ret_5 > -10:
            regime = "UPTREND"
        elif ret_20 < -45 and ret_5 < 10:
            regime = "DOWNTREND"
        elif volatility > 35:
            regime = "VOLATILE_CHOP"
        else:
            regime = "RANGE"

        return {
            "regime": regime,
            "ret_5_bps": ret_5,
            "ret_20_bps": ret_20,
            "volatility_bps": volatility,
            "reason": (
                f"ret5={ret_5:.2f};ret20={ret_20:.2f};vol={volatility:.2f}"
            ),
        }

    def choose_strategy(
        self, *, product_id: str, candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        regime = self.detect_regime(product_id)
        regime_name = str(regime.get("regime", "UNKNOWN"))
        expected_edge = float(candidate.get("expected_net_edge_bps", 0.0) or 0.0)
        probability = float(candidate.get("estimated_prob_up", 0.0) or 0.0)
        spread = float(candidate.get("spread_bps", 0.0) or 0.0)

        if regime_name == "UPTREND":
            strategy = "PULLBACK_CONTINUATION"
        elif regime_name == "RANGE":
            strategy = "MEAN_REVERSION_BOUNCE"
        elif regime_name == "DOWNTREND":
            strategy = (
                "MEAN_REVERSION_BOUNCE"
                if expected_edge >= 120 and probability >= 0.62 and spread <= 5
                else "STAND_ASIDE"
            )
        else:
            strategy = "STAND_ASIDE"

        return {
            "strategy": strategy,
            "regime": regime_name,
            "reason": f"regime={regime_name};{regime.get('reason', '')}",
        }

    def decide(
        self,
        *,
        product_id: str,
        candidate: Dict[str, Any],
        has_open_position: bool,
    ) -> ManagerDecision:
        session = self.session_health()
        product = self.product_health(product_id)
        strategy_info = self.choose_strategy(
            product_id=product_id, candidate=candidate
        )
        ai = self._latest_ai(product_id)
        risk_mode = str(session.get("risk_mode", "NORMAL"))
        strategy = str(strategy_info.get("strategy", "STAND_ASIDE"))

        if risk_mode == "PAUSE":
            decision = ManagerDecision(
                product_id, "PAUSE", "STAND_ASIDE", risk_mode, 1.0, 0.0,
                f"session_pause;{session.get('reason')}",
            )
            debug_every(
                MODULE_NAME, f"level5_decide:{product_id}", 20.0, "level5_decision",
                data={"product_id": product_id, "action": decision.action if "decision" in locals() and hasattr(decision, "action") else "", "strategy": decision.strategy if "decision" in locals() and hasattr(decision, "strategy") else "", "risk_mode": decision.risk_mode if "decision" in locals() and hasattr(decision, "risk_mode") else "", "confidence": decision.confidence if "decision" in locals() and hasattr(decision, "confidence") else 0.0, "reason": decision.reason if "decision" in locals() and hasattr(decision, "reason") else ""},
                level="DEBUG", also_overall=False,
            )
            return decision
        if not bool(product.get("product_ok", True)):
            decision = ManagerDecision(
                product_id, "BLOCK", "STAND_ASIDE", risk_mode, 0.9, 0.0,
                f"product_underperforming;{product.get('reason')}",
            )
            debug_every(
                MODULE_NAME, f"level5_decide:{product_id}", 20.0, "level5_decision",
                data={"product_id": product_id, "action": decision.action if "decision" in locals() and hasattr(decision, "action") else "", "strategy": decision.strategy if "decision" in locals() and hasattr(decision, "strategy") else "", "risk_mode": decision.risk_mode if "decision" in locals() and hasattr(decision, "risk_mode") else "", "confidence": decision.confidence if "decision" in locals() and hasattr(decision, "confidence") else 0.0, "reason": decision.reason if "decision" in locals() and hasattr(decision, "reason") else ""},
                level="DEBUG", also_overall=False,
            )
            return decision
        if strategy == "STAND_ASIDE":
            decision = ManagerDecision(
                product_id, "WAIT", strategy, risk_mode, 0.75, 0.0,
                f"strategy_stand_aside;{strategy_info.get('reason')}",
            )
            debug_every(
                MODULE_NAME, f"level5_decide:{product_id}", 20.0, "level5_decision",
                data={"product_id": product_id, "action": decision.action if "decision" in locals() and hasattr(decision, "action") else "", "strategy": decision.strategy if "decision" in locals() and hasattr(decision, "strategy") else "", "risk_mode": decision.risk_mode if "decision" in locals() and hasattr(decision, "risk_mode") else "", "confidence": decision.confidence if "decision" in locals() and hasattr(decision, "confidence") else 0.0, "reason": decision.reason if "decision" in locals() and hasattr(decision, "reason") else ""},
                level="DEBUG", also_overall=False,
            )
            return decision

        ai_action = str(ai.get("action", "")).upper()
        ai_confidence = float(ai.get("confidence", 0.0) or 0.0)
        ai_expected = float(ai.get("expected_move_30m_bps", 0.0) or 0.0)
        ai_adverse = float(ai.get("expected_adverse_bps", 0.0) or 0.0)
        if ai_action == "BLOCK_BUY" and ai_confidence >= 0.20:
            decision = ManagerDecision(
                product_id, "BLOCK", strategy, risk_mode, ai_confidence, 0.0,
                f"ai_block;move={ai_expected:.2f};adverse={ai_adverse:.2f}",
            )
            debug_every(
                MODULE_NAME, f"level5_decide:{product_id}", 20.0, "level5_decision",
                data={"product_id": product_id, "action": decision.action if "decision" in locals() and hasattr(decision, "action") else "", "strategy": decision.strategy if "decision" in locals() and hasattr(decision, "strategy") else "", "risk_mode": decision.risk_mode if "decision" in locals() and hasattr(decision, "risk_mode") else "", "confidence": decision.confidence if "decision" in locals() and hasattr(decision, "confidence") else 0.0, "reason": decision.reason if "decision" in locals() and hasattr(decision, "reason") else ""},
                level="DEBUG", also_overall=False,
            )
            return decision
        if ai_expected < ai_adverse and ai_confidence >= 0.20:
            decision = ManagerDecision(
                product_id, "WAIT", strategy, risk_mode, ai_confidence, 0.0,
                f"ai_wait_adverse_gt_move;move={ai_expected:.2f};"
                f"adverse={ai_adverse:.2f}",
            )
            debug_every(
                MODULE_NAME, f"level5_decide:{product_id}", 20.0, "level5_decision",
                data={"product_id": product_id, "action": decision.action if "decision" in locals() and hasattr(decision, "action") else "", "strategy": decision.strategy if "decision" in locals() and hasattr(decision, "strategy") else "", "risk_mode": decision.risk_mode if "decision" in locals() and hasattr(decision, "risk_mode") else "", "confidence": decision.confidence if "decision" in locals() and hasattr(decision, "confidence") else 0.0, "reason": decision.reason if "decision" in locals() and hasattr(decision, "reason") else ""},
                level="DEBUG", also_overall=False,
            )
            return decision

        base_pct = {
            "DEFENSIVE": 0.03,
            "OPPORTUNISTIC": 0.08,
        }.get(risk_mode, 0.05)
        if strategy == "BREAKOUT_CONTINUATION":
            base_pct *= 0.8
        elif strategy == "MEAN_REVERSION_BOUNCE":
            base_pct *= 0.7

        decision = ManagerDecision(
            product_id=product_id,
            action="HOLD" if has_open_position else "ALLOW_BUY",
            strategy=strategy,
            risk_mode=risk_mode,
            confidence=max(0.25, ai_confidence),
            max_position_pct=float(base_pct),
            reason=(
                f"manager_allow;strategy={strategy};risk={risk_mode};"
                f"session={session.get('reason')};product={product.get('reason')};"
                f"ai_action={ai_action};ai_conf={ai_confidence:.3f}"
            ),
        )
        debug_every(
            MODULE_NAME,
            f"level5_decide:{product_id}",
            20.0,
            "level5_decision",
            data={
                "product_id": product_id,
                "action": decision.action if "decision" in locals() and hasattr(decision, "action") else "",
                "strategy": decision.strategy if "decision" in locals() and hasattr(decision, "strategy") else "",
                "risk_mode": decision.risk_mode if "decision" in locals() and hasattr(decision, "risk_mode") else "",
                "confidence": decision.confidence if "decision" in locals() and hasattr(decision, "confidence") else 0.0,
                "reason": decision.reason if "decision" in locals() and hasattr(decision, "reason") else "",
            },
            level="DEBUG",
            also_overall=False,
        )
        return decision
