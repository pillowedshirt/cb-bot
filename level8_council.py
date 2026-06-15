"""Level 8 trading council capital allocation and risk guidance."""

import csv
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")
MISSED_OPPORTUNITIES_CSV = os.path.join(BASE_DIR, "missed_opportunities.csv")
AGENT_PERFORMANCE_CSV = os.path.join(BASE_DIR, "agent_performance.csv")
COUNCIL_OBSERVATION_OUTCOMES_CSV = os.path.join(BASE_DIR, "council_observation_outcomes.csv")
AGENT_ADJUSTMENTS_CSV = os.path.join(BASE_DIR, "agent_adjustments.csv")
ADAPTIVE_THRESHOLDS_CSV = os.path.join(BASE_DIR, "adaptive_thresholds.csv")
SHADOW_TRADES_CSV = os.path.join(BASE_DIR, "shadow_trades.csv")
AGENT_LEADERBOARD_CSV = os.path.join(BASE_DIR, "agent_leaderboard.csv")
LEVEL8_EVENTS_DB = os.path.join(BASE_DIR, "level8_events.sqlite3")


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` to the inclusive range bounded by minimum and maximum."""
    return max(minimum, min(maximum, value))


def utc_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def utc_dt(ts: Optional[float] = None) -> str:
    value = float(ts if ts is not None else utc_ts())
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def append_sqlite_event(
    *,
    event_type: str,
    source_path: str,
    row: Dict[str, Any],
) -> None:
    """
    Durable Level 8 event mirror.

    CSV remains the viewer-friendly format.
    SQLite becomes the safer long-term learning/event ledger.
    """
    try:
        payload = json.dumps(row, default=str)

        conn = sqlite3.connect(LEVEL8_EVENTS_DB)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS level8_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    dt_utc TEXT,
                    event_type TEXT,
                    source_path TEXT,
                    decision_id TEXT,
                    product_id TEXT,
                    agent TEXT,
                    strategy TEXT,
                    payload_json TEXT
                )
                """
            )

            conn.execute(
                """
                INSERT INTO level8_events (
                    ts, dt_utc, event_type, source_path, decision_id,
                    product_id, agent, strategy, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    float(row.get("ts", utc_ts()) or utc_ts()),
                    str(row.get("dt_utc", utc_dt())),
                    event_type,
                    os.path.basename(source_path),
                    str(row.get("decision_id", "")),
                    str(row.get("product_id", "")),
                    str(row.get("agent", "")),
                    str(row.get("strategy", "")),
                    payload,
                ),
            )

            conn.commit()
        finally:
            conn.close()

    except Exception:
        pass


def append_csv_row(path: str, columns: list[str], row: Dict[str, Any]) -> None:
    exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(columns)

        writer.writerow([row.get(column, "") for column in columns])

    append_sqlite_event(
        event_type=os.path.splitext(os.path.basename(path))[0],
        source_path=path,
        row=row,
    )


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

    product_adjustment: float = 0.0
    strategy_adjustment: float = 0.0
    recent_performance_adjustment: float = 0.0
    weight: float = 0.0
    leaderboard_rank: float = 999.0
    leaderboard_score: float = 0.5
    leader_bonus: float = 0.0
    leader_penalty: float = 0.0
    reason: str = ""


class Level8Council:
    """Outcome-adaptive council with an 80% maximum portfolio deployment."""

    def __init__(self) -> None:
        # Start risky so the bot learns from real outcomes.
        self.base_buy_threshold = 0.30
        self.base_sell_threshold = 0.44

        self.min_buy_threshold = 0.18
        self.max_buy_threshold = 0.76
        self.min_sell_threshold = 0.30
        self.max_sell_threshold = 0.76

        self.max_agent_adjustment = 0.32
        self.min_agent_reliability = 0.20
        self.max_agent_reliability = 1.65

        self.min_truth_to_trade = 0.05
        self.min_truth_to_core_trade = 0.16

        # Portfolio allocation model.
        # The only hard spending ceiling is 80% deployed / 20% reserve.
        self.reserve_bucket_pct = 0.20
        self.max_single_asset_pct = 0.80
        self.max_total_exposure_pct = 0.80

        # Council-controlled sizing.
        self.test_bucket_trade_pct = 0.08
        self.min_core_trade_pct = 0.14
        self.max_core_trade_pct = 0.80

        # These are descriptive only now; they do not hard-block spending.
        self.test_bucket_pct = 0.10
        self.core_bucket_pct = 0.70

        self.last_summary: Dict[str, Any] = {}

    def _neutral_stats(self, reason: str = "no_matching_data") -> Dict[str, float]:
        return {
            "n": 0.0,
            "win_rate": 0.5,
            "avg_move": 0.0,
            "avg_adverse": 0.0,
            "avg_credit": 0.5,
            "weighted_credit": 0.5,
            "real_trade_n": 0.0,
            "observation_n": 0.0,
            "reason": reason,
        }

    def _missed_opportunity_relief(self, product_id: str) -> float:
        """Reduce strictness after repeated WAIT/SHADOW decisions missed jumps."""
        try:
            if not os.path.exists(MISSED_OPPORTUNITIES_CSV):
                return 0.0
            frame = pd.read_csv(MISSED_OPPORTUNITIES_CSV)
            if frame.empty or "product_id" not in frame.columns:
                return 0.0
            frame = frame[
                frame["product_id"].astype(str) == str(product_id)
            ].copy()
            if frame.empty:
                return 0.0
            frame["move_bps"] = pd.to_numeric(
                frame["move_bps"], errors="coerce"
            ).fillna(0.0)
            recent = frame.tail(20)
            big_misses = int((recent["move_bps"] >= 120.0).sum())
            huge_misses = int((recent["move_bps"] >= 250.0).sum())
            # Missed jumps should strongly teach the council that it was too strict.
            relief = big_misses * 0.018 + huge_misses * 0.030
            return clamp(relief, 0.0, 0.20)
        except Exception:
            return 0.0

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
        """
        Summarize real trades, chart-only outcomes, and agent-specific credit.

        Important behavior:
        - Real filled trade outcomes are weighted more heavily than observations.
        - Heartbeat/observation outcomes still teach, but they do not overwhelm fills.
        - Missing agent/product/strategy data returns neutral stats instead of silently
          falling back to broad unrelated data.
        """
        frames = []

        try:
            if os.path.exists(AGENT_PERFORMANCE_CSV):
                perf = pd.read_csv(AGENT_PERFORMANCE_CSV)

                if not perf.empty:
                    perf = perf.rename(columns={
                        "outcome_move_bps": "move_bps",
                        "outcome_success": "success",
                    })

                    if "outcome_source" in perf.columns:
                        perf["source"] = perf["outcome_source"].astype(str)
                    elif "source" not in perf.columns:
                        perf["source"] = "agent_performance"

                    frames.append(perf)
        except Exception:
            pass

        try:
            if os.path.exists(COUNCIL_OBSERVATION_OUTCOMES_CSV):
                obs = pd.read_csv(COUNCIL_OBSERVATION_OUTCOMES_CSV)

                if not obs.empty:
                    obs = obs.rename(columns={
                        "decision_strategy": "strategy",
                        "would_have_won": "success",
                    })
                    obs["agent"] = obs.get("agent", "council_observation")
                    obs["source"] = "observation_outcome"
                    frames.append(obs)
        except Exception:
            pass

        try:
            trades = self._recent_trades(240)

            if not trades.empty:
                if "net_pnl_usd" in trades.columns:
                    trades["move_bps"] = pd.to_numeric(
                        trades["net_pnl_usd"], errors="coerce"
                    ).fillna(0.0)
                elif "move_bps" not in trades.columns:
                    trades["move_bps"] = 0.0

                trades["success"] = (
                    pd.to_numeric(trades["move_bps"], errors="coerce").fillna(0.0) > 0
                ).astype(int)

                trades["source"] = "real_trade"
                frames.append(trades)
        except Exception:
            pass

        if not frames:
            return self._neutral_stats("no_frames")

        data = pd.concat(frames, ignore_index=True, sort=False)

        if data.empty:
            return self._neutral_stats("empty_data")

        for column, value in (
            ("agent", agent),
            ("product_id", product_id),
            ("strategy", strategy),
        ):
            if value is not None:
                if column not in data.columns:
                    return self._neutral_stats(f"missing_column:{column}")

                rows = data[data[column].astype(str) == str(value)].copy()

                if rows.empty:
                    return self._neutral_stats(f"no_match:{column}={value}")

                data = rows

        if data.empty:
            return self._neutral_stats("empty_after_filters")

        move_source = data["move_bps"] if "move_bps" in data.columns else pd.Series(0.0, index=data.index)
        move = pd.to_numeric(move_source, errors="coerce").fillna(0.0)

        if "weighted_agent_credit_score" in data.columns:
            credit = pd.to_numeric(data["weighted_agent_credit_score"], errors="coerce").fillna(0.5)
            success = (credit >= 0.5).astype(int)
        elif "agent_credit_score" in data.columns:
            credit = pd.to_numeric(data["agent_credit_score"], errors="coerce").fillna(0.5)
            success = (credit >= 0.5).astype(int)
        elif "success" in data.columns:
            success = pd.to_numeric(data["success"], errors="coerce").fillna((move > 0).astype(int))
            credit = success.astype(float)
        else:
            success = (move > 0).astype(int)
            credit = success.astype(float)

        adverse_col = next(
            (
                c for c in (
                    "adverse_move_bps",
                    "max_adverse_bps",
                    "avg_adverse",
                    "adverse",
                )
                if c in data.columns
            ),
            None,
        )

        if adverse_col:
            adverse = pd.to_numeric(data[adverse_col], errors="coerce").fillna(0.0).abs()
        else:
            adverse = pd.Series(0.0, index=data.index)

        source = data["source"].astype(str) if "source" in data.columns else pd.Series("unknown", index=data.index)
        source_weight = source.map({
            "real_trade": 1.00,
            "trade_outcome": 1.00,
            "agent_performance": 0.80,
            "level8_observation": 0.45,
            "observation_outcome": 0.35,
            "unknown": 0.35,
        }).fillna(0.35).astype(float)

        if source_weight.sum() > 0:
            weighted_credit = float((credit * source_weight).sum() / source_weight.sum())
            weighted_success = float((success * source_weight).sum() / source_weight.sum())
            weighted_move = float((move * source_weight).sum() / source_weight.sum())
            weighted_adverse = float((adverse * source_weight).sum() / source_weight.sum())
        else:
            weighted_credit = 0.5
            weighted_success = 0.5
            weighted_move = 0.0
            weighted_adverse = 0.0

        real_trade_n = float(source.isin(["real_trade", "trade_outcome"]).sum())
        observation_n = float(source.isin(["level8_observation", "observation_outcome"]).sum())

        return {
            "n": float(len(data)),
            "win_rate": weighted_success,
            "avg_move": weighted_move,
            "avg_adverse": weighted_adverse,
            "avg_credit": float(credit.mean()),
            "weighted_credit": weighted_credit,
            "real_trade_n": real_trade_n,
            "observation_n": observation_n,
            "reason": "weighted_stats",
        }

    def _agent_competition_score(self, agent: str) -> Dict[str, float]:
        """Give each council member a bounded competitive score."""
        neutral = {
            "leaderboard_rank": 999.0,
            "leaderboard_score": 0.5,
            "leader_bonus": 0.0,
            "leader_penalty": 0.0,
            "sample_size": 0.0,
        }
        try:
            if not os.path.exists(AGENT_PERFORMANCE_CSV):
                return neutral

            frame = pd.read_csv(AGENT_PERFORMANCE_CSV)
            if frame.empty or "agent" not in frame.columns or "agent_credit_score" not in frame.columns:
                return neutral

            credit_col = "weighted_agent_credit_score" if "weighted_agent_credit_score" in frame.columns else "agent_credit_score"
            frame[credit_col] = pd.to_numeric(frame[credit_col], errors="coerce").fillna(0.5)
            source = frame["outcome_source"].astype(str) if "outcome_source" in frame.columns else pd.Series("unknown", index=frame.index)
            frame["_source_weight"] = source.map({
                "trade_outcome": 1.00,
                "real_trade": 1.00,
                "observation_outcome": 0.40,
                "level8_observation": 0.40,
                "unknown": 0.35,
            }).fillna(0.35).astype(float)

            rows = []
            for name, group in frame.groupby(frame["agent"].astype(str)):
                n = float(len(group))
                if n <= 0:
                    continue
                weighted_credit = float(
                    (group[credit_col] * group["_source_weight"]).sum()
                    / max(group["_source_weight"].sum(), 1e-9)
                )
                recent = group.tail(50)
                recent_credit = float(recent[credit_col].mean()) if not recent.empty else weighted_credit
                sample_factor = clamp(n / 30.0, 0.0, 1.0)
                leaderboard_score = clamp(
                    weighted_credit * 0.70 + recent_credit * 0.20 + sample_factor * 0.10,
                    0.0,
                    1.0,
                )
                rows.append({
                    "agent": str(name),
                    "sample_size": n,
                    "weighted_credit": weighted_credit,
                    "recent_credit": recent_credit,
                    "leaderboard_score": leaderboard_score,
                })

            if not rows:
                return neutral

            board = pd.DataFrame(rows).sort_values("leaderboard_score", ascending=False).reset_index(drop=True)
            board["leaderboard_rank"] = board.index + 1
            ts = utc_ts()
            for _, row in board.iterrows():
                try:
                    rank = float(row["leaderboard_rank"])
                    score = float(row["leaderboard_score"])
                    n = float(row["sample_size"])
                    leader_bonus = 0.0
                    leader_penalty = 0.0
                    if n >= 10:
                        if rank == 1 and score > 0.56:
                            leader_bonus = 0.060
                        elif rank <= 3 and score > 0.54:
                            leader_bonus = 0.035
                        elif score < 0.46:
                            leader_penalty = 0.050
                        elif score < 0.49:
                            leader_penalty = 0.025
                    append_csv_row(
                        AGENT_LEADERBOARD_CSV,
                        [
                            "ts", "dt_utc", "agent", "leaderboard_rank",
                            "leaderboard_score", "weighted_credit", "recent_credit",
                            "sample_size", "leader_bonus", "leader_penalty", "reason",
                        ],
                        {
                            "ts": f"{ts:.6f}",
                            "dt_utc": utc_dt(ts),
                            "agent": str(row["agent"]),
                            "leaderboard_rank": f"{rank:.0f}",
                            "leaderboard_score": f"{score:.6f}",
                            "weighted_credit": f"{float(row['weighted_credit']):.6f}",
                            "recent_credit": f"{float(row['recent_credit']):.6f}",
                            "sample_size": f"{n:.0f}",
                            "leader_bonus": f"{leader_bonus:.6f}",
                            "leader_penalty": f"{leader_penalty:.6f}",
                            "reason": (
                                f"competitive_agent_goal;rank={rank:.0f};score={score:.3f};"
                                f"bonus={leader_bonus:.3f};penalty={leader_penalty:.3f}"
                            ),
                        },
                    )
                except Exception:
                    pass

            own = board[board["agent"].astype(str) == str(agent)]
            if own.empty:
                return neutral
            row = own.iloc[0]
            rank = float(row["leaderboard_rank"])
            score = float(row["leaderboard_score"])
            n = float(row["sample_size"])
            leader_bonus = 0.0
            leader_penalty = 0.0
            if n >= 10:
                if rank == 1 and score > 0.56:
                    leader_bonus = 0.060
                elif rank <= 3 and score > 0.54:
                    leader_bonus = 0.035
                elif score < 0.46:
                    leader_penalty = 0.050
                elif score < 0.49:
                    leader_penalty = 0.025
            return {
                "leaderboard_rank": rank,
                "leaderboard_score": score,
                "leader_bonus": leader_bonus,
                "leader_penalty": leader_penalty,
                "sample_size": n,
            }
        except Exception:
            return neutral

    def _agent_adjustments(
        self,
        agent: str,
        product_id: str,
        strategy: str,
    ) -> Dict[str, float]:
        """Calculate bounded adjustments from outcomes and agent competition."""
        agent_stats = self._outcome_stats(agent=agent)
        product_stats = self._outcome_stats(agent=agent, product_id=product_id)
        strategy_stats = self._outcome_stats(agent=agent, strategy=strategy)

        recent = self._recent_trades(20)
        if "agent" in recent.columns:
            recent = recent[recent["agent"].astype(str) == str(agent)]
        if "event" in recent.columns:
            recent = recent[recent["event"].astype(str).str.upper() == "SELL"]
        pnl_column = next((column for column in ("net_pnl_usd", "pnl", "move_bps") if column in recent), None)
        recent_win_rate = 0.5
        if not recent.empty and pnl_column:
            pnl = pd.to_numeric(recent[pnl_column], errors="coerce").fillna(0.0)
            recent_win_rate = float((pnl > 0.0).mean())

        n = float(agent_stats.get("n", 0.0))
        sample_factor = clamp(n / 12.0, 0.0, 1.0)
        agent_credit = float(agent_stats.get("weighted_credit", agent_stats.get("avg_credit", 0.5)))
        product_win_rate = float(product_stats.get("win_rate", 0.5))
        strategy_win_rate = float(strategy_stats.get("win_rate", 0.5))
        product_adj = (product_win_rate - 0.5) * 0.65 * sample_factor
        strategy_adj = (strategy_win_rate - 0.5) * 0.60 * sample_factor
        recent_adj = (recent_win_rate - 0.5) * 0.85 * sample_factor
        competition = self._agent_competition_score(agent)
        leader_bonus = float(competition.get("leader_bonus", 0.0))
        leader_penalty = float(competition.get("leader_penalty", 0.0))
        product_adj = clamp(product_adj, -self.max_agent_adjustment, self.max_agent_adjustment)
        strategy_adj = clamp(strategy_adj, -self.max_agent_adjustment, self.max_agent_adjustment)
        recent_adj = clamp(recent_adj, -self.max_agent_adjustment, self.max_agent_adjustment)
        directional = clamp(product_adj + strategy_adj + recent_adj + leader_bonus - leader_penalty, -self.max_agent_adjustment, self.max_agent_adjustment)
        base_reliability = 0.80 + (agent_credit - 0.5) * 1.35 * sample_factor
        reliability = clamp(base_reliability + leader_bonus - leader_penalty, self.min_agent_reliability, self.max_agent_reliability)

        try:
            ts = utc_ts()
            append_csv_row(
                AGENT_ADJUSTMENTS_CSV,
                [
                    "ts", "dt_utc", "agent", "product_id", "strategy",
                    "base_reliability", "product_adjustment", "strategy_adjustment",
                    "recent_performance_adjustment", "directional_adjustment",
                    "final_reliability", "sample_size", "agent_credit",
                    "product_win_rate", "strategy_win_rate", "recent_win_rate",
                    "leaderboard_rank", "leaderboard_score", "leader_bonus",
                    "leader_penalty", "reason",
                ],
                {
                    "ts": f"{ts:.6f}", "dt_utc": utc_dt(ts), "agent": agent,
                    "product_id": product_id, "strategy": strategy,
                    "base_reliability": f"{base_reliability:.6f}",
                    "product_adjustment": f"{product_adj:.6f}",
                    "strategy_adjustment": f"{strategy_adj:.6f}",
                    "recent_performance_adjustment": f"{recent_adj:.6f}",
                    "directional_adjustment": f"{directional:.6f}",
                    "final_reliability": f"{reliability:.6f}",
                    "sample_size": f"{n:.0f}",
                    "agent_credit": f"{agent_credit:.6f}",
                    "product_win_rate": f"{product_win_rate:.6f}",
                    "strategy_win_rate": f"{strategy_win_rate:.6f}",
                    "recent_win_rate": f"{recent_win_rate:.6f}",
                    "leaderboard_rank": f"{float(competition.get('leaderboard_rank', 999.0)):.0f}",
                    "leaderboard_score": f"{float(competition.get('leaderboard_score', 0.5)):.6f}",
                    "leader_bonus": f"{leader_bonus:.6f}",
                    "leader_penalty": f"{leader_penalty:.6f}",
                    "reason": (
                        f"agent={agent};competitive_goal=highest_weight;credit={agent_credit:.3f};"
                        f"leader_rank={float(competition.get('leaderboard_rank', 999.0)):.0f};"
                        f"leader_score={float(competition.get('leaderboard_score', 0.5)):.3f};"
                        f"bonus={leader_bonus:.3f};penalty={leader_penalty:.3f}"
                    ),
                },
            )
        except Exception:
            pass
        return {
            "product": product_adj, "strategy": strategy_adj, "recent": recent_adj,
            "directional": directional, "reliability": reliability,
            "leaderboard_rank": float(competition.get("leaderboard_rank", 999.0)),
            "leaderboard_score": float(competition.get("leaderboard_score", 0.5)),
            "leader_bonus": leader_bonus, "leader_penalty": leader_penalty,
        }

    def _adjust_vote(
        self,
        vote: Dict[str, Any],
        product_id: str,
        strategy: str,
    ) -> AgentVote:
        """Apply outcome-derived direction and reliability to a raw vote."""
        agent_name = str(vote.get("agent", "unknown"))

        adjustments = self._agent_adjustments(
            agent_name,
            product_id,
            strategy,
        )

        directional_adj = float(adjustments["directional"])

        raw_buy = float(vote.get("buy", 0.0) or 0.0)
        raw_sell = float(vote.get("sell", 0.0) or 0.0)
        raw_hold = float(vote.get("hold", 0.0) or 0.0)
        raw_wait = float(vote.get("wait", 0.0) or 0.0)

        buy = clamp(raw_buy + directional_adj, 0.0, 1.0)
        sell = clamp(raw_sell + directional_adj, 0.0, 1.0)
        hold = clamp(raw_hold + directional_adj * 0.25, 0.0, 1.0)
        wait = clamp(raw_wait - directional_adj * 0.75, 0.0, 1.0)

        confidence = clamp(float(vote.get("confidence", 0.5) or 0.5), 0.0, 1.0)
        reliability = float(adjustments["reliability"])
        weight = max(0.0, confidence * reliability)

        return AgentVote(
            agent=agent_name,
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
            product_adjustment=float(adjustments["product"]),
            strategy_adjustment=float(adjustments["strategy"]),
            recent_performance_adjustment=float(adjustments["recent"]),
            weight=weight,
            leaderboard_rank=float(adjustments.get("leaderboard_rank", 999.0)),
            leaderboard_score=float(adjustments.get("leaderboard_score", 0.5)),
            leader_bonus=float(adjustments.get("leader_bonus", 0.0)),
            leader_penalty=float(adjustments.get("leader_penalty", 0.0)),
            reason=str(vote.get("reason", "")),
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
            buy += 0.020
            sell += 0.015
        elif risk_mode_u == "CAUTIOUS":
            buy += 0.010
            sell += 0.008
        elif risk_mode_u == "AGGRESSIVE":
            buy -= 0.065
            sell -= 0.020

        n = float(product_stats.get("n", 0.0))
        wr = float(product_stats.get("win_rate", 0.5))
        avg = float(product_stats.get("avg_move", 0.0))
        adverse = float(product_stats.get("avg_adverse", 0.0))

        if n >= 8:
            if wr < 0.35 or avg < -60:
                buy += 0.025
                sell += 0.015
            elif wr > 0.55 and avg > 15:
                buy -= 0.075
                sell -= 0.020

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

        missed_relief = self._missed_opportunity_relief(product_id)
        buy -= missed_relief
        buy = clamp(buy, self.min_buy_threshold, self.max_buy_threshold)
        sell = clamp(sell, self.min_sell_threshold, self.max_sell_threshold)

        try:
            ts = utc_ts()

            append_csv_row(
                ADAPTIVE_THRESHOLDS_CSV,
                [
                    "ts", "dt_utc", "scope", "product_id", "strategy",
                    "buy_threshold", "sell_threshold", "risk_mode",
                    "sample_size", "win_rate", "avg_move", "avg_adverse",
                    "missed_opportunity_relief", "reason",
                ],
                {
                    "ts": f"{ts:.6f}",
                    "dt_utc": utc_dt(ts),
                    "scope": "product_strategy",
                    "product_id": product_id,
                    "strategy": strategy,
                    "buy_threshold": f"{buy:.6f}",
                    "sell_threshold": f"{sell:.6f}",
                    "risk_mode": risk_mode_u,
                    "sample_size": f"{n:.0f}",
                    "win_rate": f"{wr:.6f}",
                    "avg_move": f"{avg:.6f}",
                    "avg_adverse": f"{adverse:.6f}",
                    "missed_opportunity_relief": f"{missed_relief:.6f}",
                    "reason": (
                        f"risk={risk_mode_u};n={n:.0f};wr={wr:.3f};"
                        f"avg={avg:.2f};adverse={adverse:.2f};"
                        f"missed_relief={missed_relief:.3f}"
                    ),
                },
            )
        except Exception:
            pass

        return {
            "buy_threshold": buy,
            "sell_threshold": sell,
            "risk_mode": risk_mode_u,
            "product_stats": product_stats,
            "missed_opportunity_relief": missed_relief,
        }

    def decide_buy(
        self,
        product_id: str,
        strategy: str,
        votes: list[Dict[str, Any]],
        truth_vote: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine adjusted votes into a buy, shadow, or wait decision."""
        decision_id = f"l8buy-{product_id}-{int(utc_ts())}-{uuid.uuid4().hex[:8]}"
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
            "confidence": sum(v.confidence * w for v, w in weighted) / weight_total,
        }

        raw_learning_scores = []

        for vote in votes:
            try:
                raw_learning_scores.append(float(vote.get("learning_score", 0.0) or 0.0))
            except Exception:
                pass

        learning_score = clamp(
            sum(raw_learning_scores) / len(raw_learning_scores) if raw_learning_scores else 0.0,
            0.0,
            1.0,
        )

        experience_n = float(self._outcome_stats(product_id=product_id).get("n", 0.0))
        exploration_decay = clamp(experience_n / 120.0, 0.0, 1.0)
        exploration_weight = 0.38 * (1.0 - exploration_decay) + 0.08 * exploration_decay

        truth_score = clamp(
            (
                adjusted_truth.adjusted_buy_score * 0.55
                + adjusted_truth.confidence * 0.25
                + adjusted_truth.reliability * 0.20
            ),
            0.0,
            1.0,
        )
        # Truth modulates the score, but learning mode should not let low early sample
        # quality completely suppress all buys.
        base_final_buy = clamp(
            combined["adj_buy"] * (0.85 + truth_score * 0.15),
            0.0,
            1.0,
        )

        final_buy = clamp(
            base_final_buy * (1.0 - exploration_weight)
            + learning_score * exploration_weight,
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

        if bucket in ("TEST", "CORE"):
            action = "ALLOW_BUY"
        elif final_buy >= buy_threshold and bucket == "SHADOW":
            action = "SHADOW"
        else:
            action = "WAIT"

        if action == "SHADOW":
            try:
                ts = utc_ts()

                append_csv_row(
                    SHADOW_TRADES_CSV,
                    [
                        "ts", "dt_utc", "decision_id", "product_id", "strategy",
                        "shadow_action", "council_buy_score", "buy_threshold",
                        "truth_score", "recommended_position_pct", "reason",
                    ],
                    {
                        "ts": f"{ts:.6f}",
                        "dt_utc": utc_dt(ts),
                        "decision_id": decision_id,
                        "product_id": product_id,
                        "strategy": strategy,
                        "shadow_action": "BUY",
                        "council_buy_score": f"{final_buy:.6f}",
                        "buy_threshold": f"{buy_threshold:.6f}",
                        "truth_score": f"{truth_score:.6f}",
                        "recommended_position_pct": f"{position_pct:.6f}",
                        "reason": sizing_reason,
                    },
                )
            except Exception:
                pass

        return {
            "decision_id": decision_id,
            "action": action,
            "final_buy": final_buy,
            "final_sell": final_sell,
            "truth_score": truth_score,
            "bucket": bucket,
            "position_pct": position_pct,
            "sizing_reason": sizing_reason,
            "learning_score": learning_score,
            "exploration_weight": exploration_weight,
            "confidence": combined["confidence"],
            **thresholds,
            "votes": [asdict(vote) for vote in adjusted],
            "truth_vote": asdict(adjusted_truth),
        }

    def decide_exit(
        self,
        product_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Level 8 sell council.

        This is separate from buy logic because selling answers a different question:
        should we capture profit, hold for continuation, or exit a failing position?
        """
        decision_id = f"l8exit-{product_id}-{int(utc_ts())}-{uuid.uuid4().hex[:8]}"
        unrealized_bps = float(context.get("unrealized_bps", 0.0) or 0.0)
        spread_bps = float(context.get("spread_bps", 0.0) or 0.0)
        cost_bps = float(context.get("cost_bps", 0.0) or 0.0)

        profit_capture = clamp(
            0.35 + max(0.0, unrealized_bps) / 220.0,
            0.0,
            1.0,
        )

        loss_exit = clamp(
            0.30 + max(0.0, -unrealized_bps) / 260.0,
            0.0,
            1.0,
        )

        continuation_hold = clamp(
            0.55 + max(0.0, unrealized_bps) / 400.0 - max(0.0, -unrealized_bps) / 300.0,
            0.0,
            1.0,
        )

        execution_sell_quality = clamp(
            1.0 - max(0.0, spread_bps) / 120.0,
            0.0,
            1.0,
        )

        fee_recovery = clamp(
            0.40 + (unrealized_bps - cost_bps) / 280.0,
            0.0,
            1.0,
        )

        votes = [
            {
                "agent": "profit_capture",
                "buy": 0.0,
                "sell": profit_capture,
                "hold": 1.0 - profit_capture * 0.55,
                "wait": 0.20,
                "confidence": 0.70,
            },
            {
                "agent": "drawdown_exit",
                "buy": 0.0,
                "sell": loss_exit,
                "hold": 1.0 - loss_exit * 0.70,
                "wait": 0.25,
                "confidence": 0.70,
            },
            {
                "agent": "continuation_hold",
                "buy": 0.0,
                "sell": 1.0 - continuation_hold,
                "hold": continuation_hold,
                "wait": 0.20,
                "confidence": 0.60,
            },
            {
                "agent": "execution",
                "buy": 0.0,
                "sell": execution_sell_quality,
                "hold": 0.40,
                "wait": 1.0 - execution_sell_quality,
                "confidence": 0.65,
            },
            {
                "agent": "fee_recovery",
                "buy": 0.0,
                "sell": fee_recovery,
                "hold": 1.0 - fee_recovery * 0.50,
                "wait": 0.30,
                "confidence": 0.65,
            },
        ]

        truth_vote = {
            "agent": "exit_truth",
            "buy": 0.0,
            "sell": clamp(
                profit_capture * 0.28
                + loss_exit * 0.24
                + execution_sell_quality * 0.18
                + fee_recovery * 0.20
                + (1.0 - continuation_hold) * 0.10,
                0.0,
                1.0,
            ),
            "hold": continuation_hold,
            "wait": 1.0 - execution_sell_quality,
            "confidence": 0.70,
        }

        adjusted = [self._adjust_vote(vote, product_id, "EXIT_REVIEW") for vote in votes]
        adjusted_truth = self._adjust_vote(truth_vote, product_id, "EXIT_REVIEW")

        weighted = [
            (vote, max(0.0, vote.confidence * vote.reliability))
            for vote in adjusted
        ]

        weight_total = sum(weight for _, weight in weighted) or 1.0

        final_sell = clamp(
            sum(v.adjusted_sell_score * w for v, w in weighted) / weight_total,
            0.0,
            1.0,
        )

        final_hold = clamp(
            sum(v.adjusted_hold_score * w for v, w in weighted) / weight_total,
            0.0,
            1.0,
        )

        truth_score = clamp(
            adjusted_truth.adjusted_sell_score * 0.55
            + adjusted_truth.confidence * 0.25
            + adjusted_truth.reliability * 0.20,
            0.0,
            1.0,
        )

        thresholds = self.adaptive_thresholds(product_id, "EXIT_REVIEW")
        sell_threshold = float(thresholds["sell_threshold"])

        if final_sell >= sell_threshold:
            action = "ALLOW_SELL"
        elif abs(unrealized_bps) >= 90.0 and final_sell >= sell_threshold - 0.08:
            action = "ALLOW_SELL"
        else:
            action = "HOLD"

        return {
            "decision_id": decision_id,
            "action": action,
            "final_sell": final_sell,
            "final_hold": final_hold,
            "truth_score": truth_score,
            "sell_threshold": sell_threshold,
            "buy_threshold": thresholds["buy_threshold"],
            "risk_mode": thresholds["risk_mode"],
            "votes": [asdict(vote) for vote in adjusted],
            "truth_vote": asdict(adjusted_truth),
            "reason": (
                f"exit_council;unrealized_bps={unrealized_bps:.2f};"
                f"spread_bps={spread_bps:.2f};cost_bps={cost_bps:.2f};"
                f"final_sell={final_sell:.3f};threshold={sell_threshold:.3f};"
                f"truth={truth_score:.3f}"
            ),
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

        if margin < -0.08:
            return "SHADOW", 0.0, "far_below_threshold_shadow_only"

        # Allow tiny below-threshold learning trades if truth is not completely absent.
        if margin < 0 and truth_score >= self.min_truth_to_trade:
            pct = self.test_bucket_trade_pct
            return "TEST", pct, (
                f"slightly_below_threshold_learning_test margin={margin:.3f};"
                f"truth={truth_score:.3f}"
            )

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
