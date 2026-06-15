"""Level 8 trading council capital allocation and risk guidance."""

import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` to the inclusive range bounded by minimum and maximum."""
    return max(minimum, min(maximum, value))


class Level8Council:
    """Outcome-adaptive council with an 80% maximum portfolio deployment."""

    def __init__(self) -> None:
        self.base_buy_threshold = 0.58
        self.base_sell_threshold = 0.55

        self.min_buy_threshold = 0.46
        self.max_buy_threshold = 0.78
        self.min_sell_threshold = 0.46
        self.max_sell_threshold = 0.76

        self.max_agent_adjustment = 0.25
        self.min_agent_reliability = 0.25
        self.max_agent_reliability = 1.50

        self.min_truth_to_trade = 0.35
        self.min_truth_to_core_trade = 0.50

        # Portfolio allocation model.
        # The only hard spending ceiling is 80% deployed / 20% reserve.
        self.reserve_bucket_pct = 0.20
        self.max_single_asset_pct = 0.80
        self.max_total_exposure_pct = 0.80

        # Council-controlled sizing.
        self.test_bucket_trade_pct = 0.05
        self.min_core_trade_pct = 0.10
        self.max_core_trade_pct = 0.80

        # These are descriptive only now; they do not hard-block spending.
        self.test_bucket_pct = 0.10
        self.core_bucket_pct = 0.70

        self.last_summary: Dict[str, Any] = {}

    def _recent_trades(self, lookback_rows: int = 80) -> pd.DataFrame:
        """Return recent trades, tolerating absent or malformed history."""
        try:
            if not os.path.exists(TRADES_CSV):
                return pd.DataFrame()
            trades = pd.read_csv(TRADES_CSV)
        except Exception:
            return pd.DataFrame()

        if trades.empty:
            return pd.DataFrame()
        if "ts" in trades.columns:
            trades["ts"] = pd.to_numeric(trades["ts"], errors="coerce")
            trades = trades.sort_values("ts")
        return trades.tail(lookback_rows).copy()

    def session_health(self) -> Dict[str, Any]:
        """Summarize session outcomes without imposing a hard pause mode."""
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
            return summary

        trades["net_pnl_usd"] = pd.to_numeric(
            trades["net_pnl_usd"], errors="coerce"
        ).fillna(0.0)
        if "event" in trades.columns:
            sells = trades[
                trades["event"].astype(str).str.upper() == "SELL"
            ].copy()
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
            risk_mode = "DEFENSIVE"
        elif loss_streak >= 2 or session_net <= -1.00:
            risk_mode = "CAUTIOUS"
        elif session_net >= 0.75 and loss_streak == 0:
            risk_mode = "AGGRESSIVE"
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
        return summary

    def risk_agent(
        self,
        risk_mode: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Return risk-agent votes that influence rather than veto the council."""
        mode = str(
            risk_mode or self.session_health().get("risk_mode", "NORMAL")
        ).upper()

        if mode == "DEFENSIVE":
            buy, sell, hold, wait = 0.38, 0.58, 0.48, 0.62
            conf = 0.80
        elif mode == "CAUTIOUS":
            buy, sell, hold, wait = 0.44, 0.52, 0.50, 0.56
            conf = 0.65
        elif mode == "AGGRESSIVE":
            buy, sell, hold, wait = 0.72, 0.35, 0.62, 0.25
            conf = 0.65
        else:
            buy, sell, hold, wait = 0.55, 0.42, 0.55, 0.40
            conf = 0.50

        return {
            "agent": "risk",
            "risk_mode": mode,
            "buy": buy,
            "sell": sell,
            "hold": hold,
            "wait": wait,
            "confidence": conf,
        }

    def _position_pct_from_decision(
        self,
        *,
        final_buy_score: float,
        threshold: float,
        truth_score: float,
        risk_mode: str,
    ) -> Tuple[str, float, str]:
        """
        Aggressive Level 8 sizing model.

        The council may scale up to 80% of portfolio value on very strong decisions.
        The only hard portfolio spending ceiling remains 20% reserve / 80% max deployment.
        """
        margin = float(final_buy_score) - float(threshold)

        if margin < 0:
            return "SHADOW", 0.0, "below_threshold_shadow_only"

        if truth_score < self.min_truth_to_trade:
            return "SHADOW", 0.0, "truth_below_live_trade_min"

        # Base position from score strength.
        # Small pass = small live test.
        # Large pass + strong truth = large core position.
        if margin < 0.05 or truth_score < self.min_truth_to_core_trade:
            pct = self.test_bucket_trade_pct + max(0.0, margin) * 0.50
            bucket = "TEST"
        else:
            pct = (
                self.min_core_trade_pct
                + margin * 1.25
                + max(0.0, truth_score - self.min_truth_to_core_trade) * 0.75
            )
            bucket = "CORE"

        risk_mode_u = str(risk_mode).upper()

        if risk_mode_u == "DEFENSIVE":
            pct *= 0.70
        elif risk_mode_u == "CAUTIOUS":
            pct *= 0.85
        elif risk_mode_u == "AGGRESSIVE":
            pct *= 1.25

        pct = clamp(pct, 0.0, self.max_single_asset_pct)

        return bucket, pct, (
            f"{bucket.lower()}_bucket margin={margin:.3f};"
            f"truth={truth_score:.3f};risk={risk_mode_u};pct={pct:.3f}"
        )
