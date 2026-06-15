"""Level 8 trading council capital allocation and risk guidance."""

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` to the inclusive range bounded by minimum and maximum."""
    return max(minimum, min(maximum, value))


@dataclass
class AgentVote:
    """Normalized council vote, including its outcome-based adjustments."""

    agent: str
    buy: float
    sell: float
    hold: float
    wait: float
    confidence: float
    reliability: float = 0.80
    adjusted_buy_score: float = 0.0
    adjusted_sell_score: float = 0.0
    adjusted_hold_score: float = 0.0
    adjusted_wait_score: float = 0.0


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

    def _outcome_stats(
        self,
        *,
        agent: Optional[str] = None,
        product_id: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, float]:
        """Summarize closed outcomes for adaptive votes and thresholds."""
        trades = self._recent_trades(80)
        if trades.empty:
            return {"n": 0.0, "win_rate": 0.5, "avg_move": 0.0, "avg_adverse": 0.0}

        filters = (
            ("agent", agent),
            ("product_id", product_id),
            ("strategy", strategy),
        )
        for column, value in filters:
            if value is not None and column in trades.columns:
                trades = trades[trades[column].astype(str) == str(value)]

        if "event" in trades.columns:
            trades = trades[trades["event"].astype(str).str.upper() == "SELL"]
        if trades.empty:
            return {"n": 0.0, "win_rate": 0.5, "avg_move": 0.0, "avg_adverse": 0.0}

        pnl_column = next(
            (column for column in ("net_pnl_usd", "pnl", "move_bps") if column in trades),
            None,
        )
        pnl = (
            pd.to_numeric(trades[pnl_column], errors="coerce").fillna(0.0)
            if pnl_column
            else pd.Series(0.0, index=trades.index)
        )
        adverse_column = next(
            (column for column in ("adverse_move_bps", "max_adverse_bps", "adverse") if column in trades),
            None,
        )
        adverse = (
            pd.to_numeric(trades[adverse_column], errors="coerce").fillna(0.0).abs()
            if adverse_column
            else pd.Series(0.0, index=trades.index)
        )
        return {
            "n": float(len(trades)),
            "win_rate": float((pnl > 0.0).mean()),
            "avg_move": float(pnl.mean()),
            "avg_adverse": float(adverse.mean()),
        }

    def _agent_adjustments(
        self,
        agent: str,
        product_id: str,
        strategy: str,
    ) -> Dict[str, float]:
        """Calculate bounded adjustments from an agent's historical outcomes."""
        agent_stats = self._outcome_stats(agent=agent)
        product_stats = self._outcome_stats(agent=agent, product_id=product_id)
        strategy_stats = self._outcome_stats(agent=agent, strategy=strategy)

        recent = self._recent_trades(20)
        if "agent" in recent.columns:
            recent = recent[recent["agent"].astype(str) == str(agent)]
        if "event" in recent.columns:
            recent = recent[recent["event"].astype(str).str.upper() == "SELL"]
        pnl_column = next(
            (column for column in ("net_pnl_usd", "pnl", "move_bps") if column in recent),
            None,
        )
        recent_win_rate = 0.5
        if not recent.empty and pnl_column:
            pnl = pd.to_numeric(recent[pnl_column], errors="coerce").fillna(0.0)
            recent_win_rate = float((pnl > 0.0).mean())

        n = float(agent_stats["n"])
        stats = {
            "agent_win_rate": agent_stats["win_rate"],
            "product_win_rate": product_stats["win_rate"],
            "strategy_win_rate": strategy_stats["win_rate"],
            "recent_win_rate": recent_win_rate,
        }
        sample_factor = clamp(n / 20.0, 0.0, 1.0)

        product_adj = (float(stats.get("product_win_rate", 0.5)) - 0.5) * 0.45 * sample_factor
        strategy_adj = (float(stats.get("strategy_win_rate", 0.5)) - 0.5) * 0.40 * sample_factor
        recent_adj = (float(stats.get("recent_win_rate", 0.5)) - 0.5) * 0.60 * sample_factor

        product_adj = clamp(product_adj, -self.max_agent_adjustment, self.max_agent_adjustment)
        strategy_adj = clamp(strategy_adj, -self.max_agent_adjustment, self.max_agent_adjustment)
        recent_adj = clamp(recent_adj, -self.max_agent_adjustment, self.max_agent_adjustment)

        base_reliability = 0.80 + (float(stats.get("agent_win_rate", 0.5)) - 0.5) * 0.85 * sample_factor
        return {
            "product": product_adj,
            "strategy": strategy_adj,
            "recent": recent_adj,
            "directional": clamp(
                product_adj + strategy_adj + recent_adj,
                -self.max_agent_adjustment,
                self.max_agent_adjustment,
            ),
            "reliability": clamp(
                base_reliability,
                self.min_agent_reliability,
                self.max_agent_reliability,
            ),
        }

    def _adjust_vote(
        self,
        vote: Dict[str, Any],
        product_id: str,
        strategy: str,
    ) -> AgentVote:
        """Apply outcome-derived direction and reliability to a raw vote."""
        adjustments = self._agent_adjustments(
            str(vote.get("agent", "unknown")), product_id, strategy
        )
        directional_adj = adjustments["directional"]
        raw_buy = float(vote.get("buy", 0.0))
        raw_sell = float(vote.get("sell", 0.0))
        raw_hold = float(vote.get("hold", 0.0))
        raw_wait = float(vote.get("wait", 0.0))

        # Directional adjustment rewards agents when their category has been right.
        # It does not make one agent infinite; reliability and truth still control final weight.
        buy = clamp(raw_buy + directional_adj, 0.0, 1.0)
        sell = clamp(raw_sell + directional_adj, 0.0, 1.0)
        hold = clamp(raw_hold + directional_adj * 0.25, 0.0, 1.0)
        wait = clamp(raw_wait - directional_adj * 0.75, 0.0, 1.0)
        confidence = clamp(float(vote.get("confidence", 0.5)), 0.0, 1.0)
        reliability = adjustments["reliability"]

        return AgentVote(
            agent=str(vote.get("agent", "unknown")),
            buy=raw_buy,
            sell=raw_sell,
            hold=raw_hold,
            wait=raw_wait,
            confidence=confidence,
            reliability=reliability,
            adjusted_buy_score=buy,
            adjusted_sell_score=sell,
            adjusted_hold_score=hold,
            adjusted_wait_score=wait,
        )

    def adaptive_thresholds(self, product_id: str, strategy: str) -> Dict[str, Any]:
        """Return stable thresholds that adapt only after meaningful samples."""
        health = self.session_health()
        risk_mode = health.get("risk_mode", "NORMAL")
        product_stats = self._outcome_stats(product_id=product_id)
        buy = self.base_buy_threshold
        sell = self.base_sell_threshold

        risk_mode_u = str(risk_mode).upper()

        if risk_mode_u == "DEFENSIVE":
            buy += 0.08
            sell += 0.03
        elif risk_mode_u == "CAUTIOUS":
            buy += 0.04
            sell += 0.02
        elif risk_mode_u == "AGGRESSIVE":
            buy -= 0.05
            sell -= 0.02

        n = float(product_stats.get("n", 0.0))
        wr = float(product_stats.get("win_rate", 0.5))
        avg = float(product_stats.get("avg_move", 0.0))
        adverse = float(product_stats.get("avg_adverse", 0.0))

        if n >= 10:
            if wr < 0.38 or avg < -60:
                buy += 0.08
                sell += 0.03
            elif wr > 0.62 and avg > 35:
                buy -= 0.06
                sell -= 0.02

        if n >= 20:
            if adverse > 120:
                buy += 0.04
            elif adverse < 45 and avg > 20:
                buy -= 0.03

        if strategy == "BREAKOUT_CONTINUATION":
            buy += 0.02
        elif strategy == "MEAN_REVERSION_BOUNCE":
            buy += 0.01
        elif strategy == "PULLBACK_CONTINUATION":
            buy -= 0.01
        elif strategy == "STAND_ASIDE":
            buy += 0.18

        buy = clamp(buy, self.min_buy_threshold, self.max_buy_threshold)
        sell = clamp(sell, self.min_sell_threshold, self.max_sell_threshold)
        return {
            "buy_threshold": buy,
            "sell_threshold": sell,
            "risk_mode": risk_mode_u,
            "product_stats": product_stats,
        }

    def decide_buy(
        self,
        product_id: str,
        strategy: str,
        votes: list[Dict[str, Any]],
        truth_vote: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine adjusted votes into a buy, shadow, or wait decision."""
        adjusted = [self._adjust_vote(vote, product_id, strategy) for vote in votes]
        adjusted_truth = self._adjust_vote(truth_vote, product_id, strategy)
        weighted = [
            (vote, max(0.0, vote.confidence * vote.reliability))
            for vote in adjusted
        ]
        weight_total = sum(weight for _, weight in weighted) or 1.0
        combined = {
            "adj_buy": sum(v.adjusted_buy_score * w for v, w in weighted) / weight_total,
            "adj_sell": sum(v.adjusted_sell_score * w for v, w in weighted) / weight_total,
        }
        truth_score = clamp(
            (
                adjusted_truth.adjusted_buy_score * 0.55
                + adjusted_truth.confidence * 0.25
                + adjusted_truth.reliability * 0.20
            ),
            0.0,
            1.0,
        )
        final_buy = clamp(
            combined["adj_buy"] * (0.65 + truth_score * 0.35),
            0.0,
            1.0,
        )
        final_sell = clamp(
            combined["adj_sell"] * (0.65 + truth_score * 0.35),
            0.0,
            1.0,
        )
        thresholds = self.adaptive_thresholds(product_id, strategy)
        buy_threshold = thresholds["buy_threshold"]
        bucket, position_pct, sizing_reason = self._position_pct_from_decision(
            final_buy_score=final_buy,
            threshold=buy_threshold,
            truth_score=truth_score,
            risk_mode=thresholds["risk_mode"],
        )

        if final_buy >= buy_threshold and bucket in ("TEST", "CORE"):
            action = "ALLOW_BUY"
        elif final_buy >= buy_threshold and bucket == "SHADOW":
            action = "SHADOW"
        else:
            action = "WAIT"

        return {
            "action": action,
            "final_buy": final_buy,
            "final_sell": final_sell,
            "truth_score": truth_score,
            "bucket": bucket,
            "position_pct": position_pct,
            "sizing_reason": sizing_reason,
            **thresholds,
            "votes": [asdict(vote) for vote in adjusted],
            "truth_vote": asdict(adjusted_truth),
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
