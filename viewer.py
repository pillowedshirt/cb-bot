import html
import json
import os
import time
import traceback
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

try:
    from debug_tools import (
        module_debug,
        module_exception,
        debug_every,
        initialize_all_module_debug_logs,
        dataframe_debug_summary,
        viewer_snapshot_summary,
        csv_runtime_status,
    )
except Exception:
    def module_debug(*args, **kwargs): pass
    def module_exception(*args, **kwargs): pass
    def debug_every(*args, **kwargs): pass
    def initialize_all_module_debug_logs(*args, **kwargs): pass
    def dataframe_debug_summary(*args, **kwargs): return {}
    def viewer_snapshot_summary(*args, **kwargs): return {}
    def csv_runtime_status(*args, **kwargs): return {}

MODULE_NAME = "viewer"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)
VIEWER_SNAPSHOT_PATH = os.path.join(BASE_DIR, "viewer_snapshot.json")
VIEWER_SNAPSHOT_CSV_SAFE_PATH = VIEWER_SNAPSHOT_PATH
CALCULATION_STATUS_JSON_PATH = os.path.join(BASE_DIR, "calculation_status.json")
CALCULATION_COMPLETE_LATCH_JSON_PATH = os.path.join(BASE_DIR, "calculation_complete_latch.json")
MARKET_CSV_PATH = os.path.join(BASE_DIR, "market.csv")
TRADES_CSV_PATH = os.path.join(BASE_DIR, "trades.csv")
POSITION_TARGETS_PATH = os.path.join(BASE_DIR, "position_targets.csv")
COUNCIL_DECISIONS_PATH = os.path.join(BASE_DIR, "council_decisions.csv")
COUNCIL_VOTES_CSV_PATH = os.path.join(BASE_DIR, "council_votes.csv")
ORDERS_CSV_PATH = os.path.join(BASE_DIR, "orders.csv")
WALK_FORWARD_VALIDATION_PATH = os.path.join(BASE_DIR, "walk_forward_validation.csv")
AGENT_ABLATION_PATH = os.path.join(BASE_DIR, "agent_ablation.csv")
AI_FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "ai_feature_importance.csv")
ORDER_BOOK_SNAPSHOTS_PATH = os.path.join(BASE_DIR, "order_book_snapshots.csv")
MICRO_HISTORY_CSV_PATH = os.path.join(BASE_DIR, "micro_history.csv")
MACRO_DAY_CSV_PATH = os.path.join(BASE_DIR, "macro_day.csv")
MACRO_WEEK_CSV_PATH = os.path.join(BASE_DIR, "macro_week.csv")
SHADOW_TRADES_CSV_PATH = os.path.join(BASE_DIR, "shadow_trades.csv")
SHADOW_SELL_REPLAY_CSV_PATH = os.path.join(BASE_DIR, "shadow_sell_replay.csv")
HISTORICAL_SHADOW_REPLAY_CSV_PATH = os.path.join(BASE_DIR, "historical_shadow_replay.csv")
HISTORICAL_REPLAY_SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "historical_replay_summary.csv")
HISTORICAL_REPLAY_MANIFEST_JSON_PATH = os.path.join(BASE_DIR, "historical_replay_manifest.json")
HIST_REPLAY_15M_90D_CSV_PATH = os.path.join(BASE_DIR, "historical_replay_15m_90d.csv")
HIST_REPLAY_1H_365D_CSV_PATH = os.path.join(BASE_DIR, "historical_replay_1h_365d.csv")

STARTUP_CALC_REQUIRED_MICRO_ROWS_PER_PRODUCT = 120
STARTUP_CALC_REQUIRED_15M_CANDLE_ROWS_PER_PRODUCT = int(90 * 24 * 4 * 0.92)
STARTUP_CALC_REQUIRED_1H_CANDLE_ROWS_PER_PRODUCT = int(365 * 24 * 0.92)
STARTUP_CALC_REQUIRED_15M_REPLAY_ROWS_PER_PRODUCT = 300
STARTUP_CALC_REQUIRED_1H_REPLAY_ROWS_PER_PRODUCT = 100

STRATEGY_VARIANT_REPLAY_SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "strategy_variant_replay_summary.csv")
REPLAY_FEE_COMPARISON_SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "replay_fee_comparison_summary.csv")
EXCHANGE_PRODUCT_MAP_CSV_PATH = os.path.join(BASE_DIR, "exchange_product_map.csv")
MISSED_OPPORTUNITIES_CSV_PATH = os.path.join(BASE_DIR, "missed_opportunities.csv")
CHART_1M_7D_CSV_PATH = os.path.join(BASE_DIR, "chart_1m_7d.csv")
CHART_15M_30D_CSV_PATH = os.path.join(BASE_DIR, "chart_15m_30d.csv")
CHART_1H_90D_CSV_PATH = os.path.join(BASE_DIR, "chart_1h_90d.csv")
CHART_1D_2Y_CSV_PATH = os.path.join(BASE_DIR, "chart_1d_2y.csv")
CANDIDATE_REPLAY_PATH = os.path.join(BASE_DIR, "candidate_replay.csv")
AGENT_ADJUSTMENTS_PATH = os.path.join(BASE_DIR, "agent_adjustments.csv")
AGENT_PERFORMANCE_PATH = os.path.join(BASE_DIR, "agent_performance.csv")
AGENT_COMPONENT_REPLAY_ATTRIBUTION_CSV_PATH = os.path.join(BASE_DIR, "agent_component_replay_attribution.csv")
AGENT_TRADE_POLICY_CSV_PATH = os.path.join(BASE_DIR, "agent_trade_policy.csv")
AGENT_SIDE_RATINGS_PATH = os.path.join(BASE_DIR, "agent_side_ratings.csv")
FOUR_PASS_AGENT_BUY_PATH = os.path.join(BASE_DIR, "four_pass_agent_buy_timing.csv")
FOUR_PASS_COUNCIL_BUY_PATH = os.path.join(BASE_DIR, "four_pass_council_buy_timing.csv")
FOUR_PASS_AGENT_SELL_PATH = os.path.join(BASE_DIR, "four_pass_agent_sell_timing.csv")
FOUR_PASS_COUNCIL_SELL_PATH = os.path.join(BASE_DIR, "four_pass_council_sell_timing.csv")
FOUR_PASS_FINAL_AGENT_RATINGS_PATH = os.path.join(BASE_DIR, "four_pass_final_agent_ratings.csv")
DECISION_AUDIT_PATH = os.path.join(BASE_DIR, "decision_audit.csv")

SNAPSHOT_STALE_WARN_SEC = 20.0
CHART_STALE_WARN_SEC_DAY = 180.0
CHART_STALE_WARN_SEC_WEEK = 3600.0
COUNCIL_STALE_WARN_SEC = 60.0
HISTORY_ROWS_PER_COIN = 600

st.set_page_config(
    page_title="Crypto Strategy HUD",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
initialize_all_module_debug_logs(BASE_DIR)
module_debug(MODULE_NAME, "viewer_module_loaded", data={"base_dir": BASE_DIR, "file": __file__}, level="INFO", also_overall=True)

LEAD_DECISION_MAKER_TITLE = "🧠 The Signal Core"
LEAD_DECISION_MAKER_SUBTITLE = "Final strategy verdict"
VOLUME_LEADER_TITLE = "🧭 The Volume Oracle"
VOLUME_LEADER_SUBTITLE = "Value, volume, and market-location verdict"

AGENT_TITLES = {
    "volume_profile_leader": VOLUME_LEADER_TITLE, "volume_profile_agent": "📊 Value Area Analyst",
    "trend": "📈 Trend Analyst", "mean_reversion": "🔁 Reversion Analyst", "breakout": "🚀 Breakout Analyst",
    "ai_outcome": "🧠 AI Outcome Coach", "execution": "⚙️ Execution Analyst", "order_book_liquidity_agent": "🧱 Order Book Analyst",
    "previous_session_volume_profile_agent": "🗂 Prior Session Analyst", "quant_boundary_agent": "🔬 Quant Boundary Analyst",
    "candle_context_agent": "🕯 Candle Context Analyst", "candle_sequence_agent": "🎞 Candle Sequence Analyst",
    "candle_exhaustion_agent": "🔋 Exhaustion Analyst", "market_structure_agent": "🏗 Structure Analyst",
    "validated_liquidity_agent": "💧 Liquidity Analyst", "fresh_zone_retest_agent": "🎯 Retest Analyst",
    "fair_value_gap_agent": "🕳 Gap Analyst", "smt_divergence_agent": "🪞 Divergence Analyst",
    "setup_performance_agent": "📚 Setup History Coach", "utility_leader": "💰 Utility Analyst",
    "risk": "🛡 Risk Officer", "exploration": "🧪 Exploration Coach", "truth": "⚖️ Truth Arbiter",
    "fallback": "🛰 Strategy Agent",
}

AGENT_MONITOR_DESCRIPTIONS = {
    "volume_profile_leader": "Monitors value area, POC, VAH/VAL, volume nodes, and whether price is accepting or rejecting important volume zones.",
    "volume_profile_agent": "Watches whether price is inside value, above value, below value, or moving through low/high-volume areas.",
    "trend": "Measures short-term and medium-term momentum to decide whether price is trending or stalling.",
    "mean_reversion": "Looks for stretched moves that may snap back toward fair value or the recent mean.",
    "breakout": "Looks for acceptance above resistance, continuation through clean levels, and room for price to expand.",
    "ai_outcome": "Uses learned outcome patterns to estimate whether similar setups recently led to favorable movement.",
    "execution": "Checks whether the trade can be executed cleanly without stale quotes, excessive spread, or fill risk.",
    "order_book_liquidity_agent": "Reads order-book pressure, bid/ask liquidity, spread quality, and whether enough depth supports the move.",
    "previous_session_volume_profile_agent": "Compares current price to prior-session POC, VAH, VAL, and previous value reactions.",
    "quant_boundary_agent": "Checks statistical boundaries, expected movement, and whether price is near a probable edge or danger zone.",
    "candle_context_agent": "Reads the latest candles for rejection, continuation, wick behavior, and candle quality.",
    "candle_sequence_agent": "Looks at candle order and rhythm to judge whether the move is building or fading.",
    "candle_exhaustion_agent": "Looks for signs that a move is exhausted and may need to pause, reverse, or harvest profit.",
    "market_structure_agent": "Tracks swing highs, swing lows, higher highs, lower lows, breaks of structure, and trend structure.",
    "validated_liquidity_agent": "Watches liquidity sweeps, reclaims, stop zones, and whether liquidity was taken cleanly.",
    "fresh_zone_retest_agent": "Looks for clean retests of newly created support or resistance zones.",
    "fair_value_gap_agent": "Tracks fair value gaps, gap fills, reclaim behavior, and rejection from imbalance zones.",
    "smt_divergence_agent": "Looks for divergence between related markets that may warn of weak continuation or hidden strength.",
    "setup_performance_agent": "Compares the current setup type against backlog results and historical replay performance.",
    "utility_leader": "Judges whether expected reward is large enough after fees, spread, slippage, and wait value.",
    "risk": "Controls downside, exposure, stop risk, portfolio concentration, and whether the trade is worth live funds.",
    "exploration": "Allows controlled learning when the bot needs more outcome data but should not over-risk live money.",
    "truth": "Combines the strongest economic and technical evidence into a final reality check.",
    "exit_truth": "Combines sell-side evidence to decide whether an open position should be harvested or held.",
    "sell_utility_leader": "Checks whether selling now captures enough profit compared with holding longer.",
    "drawdown_exit": "Watches drawdown and invalidation to prevent a winning or neutral position from turning into a poor hold.",
    "fee_recovery": "Checks whether the position has cleared fees, spread, and minimum profitable exit requirements.",
    "fallback": "General strategy analyst monitoring the current market state.",
}


def agent_monitor_description(agent: Any) -> str:
    return AGENT_MONITOR_DESCRIPTIONS.get(str(agent), AGENT_MONITOR_DESCRIPTIONS["fallback"])


def latest_agent_side_ratings_map(agent_side_ratings_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    try:
        if agent_side_ratings_df is None or agent_side_ratings_df.empty or "agent" not in agent_side_ratings_df.columns:
            return {}
        frame = agent_side_ratings_df.copy()
        if "ts" in frame.columns:
            frame["ts_num"] = pd.to_numeric(frame["ts"], errors="coerce")
            frame = frame.sort_values("ts_num")
        latest = frame.groupby(frame["agent"].astype(str), as_index=False).tail(1)
        out = {}
        for _, row in latest.iterrows():
            agent = str(row.get("agent") or "")
            if agent:
                out[agent] = row.to_dict()
        return out
    except Exception:
        return {}


def _pct_text(value: Any, default: str = "—") -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except Exception:
        return default


def _weight_pct_text(value: Any, default: str = "—") -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "": return float(default)
        return float(value)
    except Exception:
        return float(default)


def _html(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")



def local_tzinfo():
    try:
        return datetime.now().astimezone().tzinfo
    except Exception:
        return timezone.utc


def local_timezone_label() -> str:
    try:
        dt = datetime.now().astimezone()
        return dt.tzname() or str(dt.tzinfo) or "local time"
    except Exception:
        return "local time"


def format_local_datetime(ts_value: Any) -> str:
    try:
        ts_float = _safe_float(ts_value, 0.0)
        if ts_float <= 0:
            return "unknown"
        dt = datetime.fromtimestamp(ts_float, tz=local_tzinfo())
        return dt.strftime("%Y-%m-%d %I:%M:%S %p %Z")
    except Exception:
        return "unknown"


def format_hold_duration(seconds_value: Any) -> str:
    try:
        seconds = max(0, int(float(seconds_value or 0)))
    except Exception:
        seconds = 0
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def signed_usd(value: Any) -> str:
    amount = _safe_float(value, 0.0)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):.2f}"


def signed_pct(value: Any) -> str:
    amount = _safe_float(value, 0.0)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}{abs(amount):.2f}%"


def parse_note_float(note: Any, key: str, default: float = 0.0) -> float:
    try:
        text = str(note or "")
        marker = f"{key}="
        if marker not in text:
            return float(default)
        raw = text.split(marker, 1)[1].split()[0].split(";")[0].strip()
        return float(raw)
    except Exception:
        return float(default)


def inject_crypto_game_css() -> None:
    st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, rgba(27, 118, 255, 0.22), transparent 34%), radial-gradient(circle at top right, rgba(0, 255, 194, 0.13), transparent 30%), linear-gradient(180deg, #07111f 0%, #05070d 100%); color: #d9f5ff; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px; }
.hud-header { border: 1px solid rgba(80, 220, 255, 0.30); border-radius: 22px; padding: 1rem 1.2rem; background: linear-gradient(90deg, rgba(12, 35, 62, 0.95), rgba(8, 18, 32, 0.95)); margin-bottom: 1rem; }
.hud-title { font-size: 2.1rem; font-weight: 900; letter-spacing: 0.04em; color: #e8fbff; }
.hud-subtitle { color: #8db7c8; font-size: 0.95rem; }
.status-strip { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.6rem; margin: 0.75rem 0 1rem 0; }
.status-pill { border: 1px solid rgba(80, 220, 255, 0.20); border-radius: 14px; padding: 0.65rem 0.75rem; background: rgba(6, 20, 34, 0.86); }
.pill-label { color: #83aabc; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.pill-value { color: #e8fbff; font-size: 1.05rem; font-weight: 800; }
.arena-grid { display: grid; grid-template-columns: 1.05fr 1.95fr; gap: 0.85rem; margin-bottom: 1rem; }
.chief-card { border: 1px solid rgba(0, 255, 194, 0.35); border-radius: 20px; padding: 1rem; background: linear-gradient(180deg, rgba(0, 73, 92, 0.44), rgba(6, 13, 24, 0.94)); }
.leadership-grid { display: grid; grid-template-columns: 1fr; gap: 0.9rem; margin: 1rem 0 1.1rem 0; }
.leadership-card { border: 1px solid rgba(80, 220, 255, 0.22); border-radius: 20px; padding: 1rem; background: linear-gradient(180deg, rgba(6, 18, 32, 0.96), rgba(4, 12, 24, 0.94)); box-shadow: 0 0 22px rgba(80, 220, 255, 0.06); }
.leadership-card.oracle { border-color: rgba(255, 214, 102, 0.35); box-shadow: 0 0 22px rgba(255, 214, 102, 0.07); }
.leadership-title { font-size: 1.15rem; font-weight: 900; margin-bottom: 0.2rem; }
.leadership-subtitle { color: #8db7c8; font-size: 0.86rem; margin-bottom: 0.65rem; }
.leadership-verdict { font-size: 1.05rem; font-weight: 900; margin-bottom: 0.5rem; }
.leadership-paragraph { color: #d9f5ff; line-height: 1.35rem; font-size: 0.93rem; }
.leadership-learning { margin-top: 0.7rem; padding: 0.65rem; border-radius: 14px; background: rgba(57, 245, 163, 0.06); border: 1px solid rgba(57, 245, 163, 0.18); color: #c9f8e2; font-size: 0.88rem; line-height: 1.25rem; }
@media (max-width: 900px) { .leadership-grid { grid-template-columns: 1fr; } }

.agent-card {
    border: 1px solid rgba(80, 220, 255, 0.20);
    border-radius: 16px;
    padding: 0.85rem;
    background: rgba(7, 18, 32, 0.92);
    min-height: 420px !important;
    height: 420px !important;
    max-height: 420px !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start;
}
.agent-card .agent-title { font-weight: 900; margin-bottom: 0.35rem; }
.agent-card .agent-summary { flex: 1; overflow: visible !important; color: #8db7c8; font-size: 0.88rem; line-height: 1.22rem; }
.agent-card .agent-metrics { margin-top: 0.45rem; font-size: 0.85rem; }
.agent-card-buy { border-color: rgba(0, 255, 160, 0.45); } .agent-card-sell { border-color: rgba(255, 87, 116, 0.45); } .agent-card-hold { border-color: rgba(255, 214, 102, 0.42); } .agent-card-wait { border-color: rgba(135, 159, 180, 0.38); }
.inquiry-panel { border: 1px solid rgba(0, 255, 194, 0.25); border-radius: 18px; padding: 1rem; background: rgba(3, 22, 30, 0.88); margin-top: 0.8rem; }
.codex-panel { border: 1px solid rgba(80, 220, 255, 0.18); border-radius: 18px; padding: 1rem; background: rgba(5, 13, 24, 0.92); }
.good { color: #39f5a3; font-weight: 800; } .warn { color: #ffd166; font-weight: 800; } .danger { color: #ff5c7a; font-weight: 800; } .muted { color: #8db7c8; }
div[data-testid="stMetric"] { background: rgba(6,20,34,.75); border: 1px solid rgba(80,220,255,.15); padding: 8px 10px; border-radius: 12px; }
.screen-section { width: 100%; display: block; padding: 0.35rem 0 0.75rem 0; margin: 0; border-bottom: 1px solid rgba(80, 220, 255, 0.08); }
.calibration-gate { max-width: 1080px; margin: 2rem auto; padding: 1.25rem; border: 1px solid rgba(80, 220, 255, 0.22); border-radius: 24px; background: linear-gradient(180deg, rgba(6, 18, 32, 0.96), rgba(3, 9, 18, 0.98)); box-shadow: 0 0 32px rgba(80, 220, 255, 0.08); }
.calibration-title { font-size: 1.7rem; font-weight: 900; margin-bottom: 0.35rem; }
.calibration-subtitle { color: #8db7c8; margin-bottom: 1rem; }
.calibration-phase-card { border: 1px solid rgba(80, 220, 255, 0.18); border-radius: 16px; padding: 0.85rem; background: rgba(7, 18, 32, 0.88); margin-bottom: 0.75rem; }
.calibration-elapsed { max-width: 760px; margin: 0.75rem auto 0.55rem auto; text-align: center; font-size: 1.15rem; font-weight: 900; letter-spacing: 0.03em; color: #d9f5ff; border: 1px solid rgba(80, 220, 255, 0.20); border-radius: 999px; padding: 0.65rem 0.9rem; background: rgba(6, 18, 32, 0.82); box-shadow: 0 0 20px rgba(80, 220, 255, 0.07); }
.calibration-elapsed span { color: #39f5a3; }
.calibration-product-table { font-size: 0.9rem; }
.held-banner {
    max-width: 760px;
    margin: 0.9rem auto 1rem auto;
    border: 1px solid rgba(255, 214, 102, 0.55);
    border-radius: 18px;
    padding: 0.85rem 1rem;
    background: rgba(255, 214, 102, 0.12);
    color: #ffd166;
    text-align: center;
    font-weight: 900;
    letter-spacing: 0.02em;
}
.held-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
.held-position-card { position: relative; border: 1px solid rgba(255, 214, 102, 0.32); border-radius: 18px; padding: 0.9rem 0.9rem 3.2rem 0.9rem; background: rgba(6, 18, 32, 0.90); min-height: 270px; overflow: hidden; }
.held-position-card.sold-profit { border-color: rgba(57, 245, 163, 0.55); background: linear-gradient(180deg, rgba(7, 42, 30, 0.92), rgba(6, 18, 32, 0.94)); }
.held-position-card.sold-loss { border-color: rgba(255, 92, 122, 0.60); background: linear-gradient(180deg, rgba(56, 12, 22, 0.92), rgba(6, 18, 32, 0.94)); }
.position-pnl-banner { position: absolute; left: 0; right: 0; bottom: 0; padding: 0.65rem; font-weight: 900; text-align: center; }
.position-pnl-banner.positive { background: rgba(57, 245, 163, 0.18); color: #39f5a3; border-top: 1px solid rgba(57, 245, 163, 0.45); }
.position-pnl-banner.negative { background: rgba(255, 92, 122, 0.18); color: #ff5c7a; border-top: 1px solid rgba(255, 92, 122, 0.45); }
.sale-result-overlay { position: absolute; top: 0.75rem; right: 0.75rem; border-radius: 999px; padding: 0.25rem 0.6rem; font-size: 0.78rem; font-weight: 900; }
.sale-result-overlay.profit { background: rgba(57, 245, 163, 0.20); color: #39f5a3; border: 1px solid rgba(57, 245, 163, 0.50); }
.sale-result-overlay.loss { background: rgba(255, 92, 122, 0.20); color: #ff5c7a; border: 1px solid rgba(255, 92, 122, 0.50); }
@media (max-width: 900px) { .held-grid { grid-template-columns: 1fr; } }

.screen-section.command-deck { min-height: auto; }
.screen-section.strategy-arena { min-height: auto; }
.screen-section.deep-learning { min-height: auto; }
.screen-section.debug-health { min-height: auto; }
.screen-card { border: 1px solid rgba(80, 220, 255, 0.22); border-radius: 22px; padding: 1rem; background: linear-gradient(180deg, rgba(7, 22, 39, 0.94), rgba(4, 9, 18, 0.97)); box-shadow: 0 0 28px rgba(0, 180, 255, 0.08); margin-bottom: 1rem; }
.overview-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }
.coin-overview-card {
    position: relative;
    border: 1px solid rgba(80, 220, 255, 0.20);
    border-radius: 18px;
    padding: 0.8rem;
    background: rgba(6, 18, 32, 0.90);
    min-height: 285px;
    transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
}
.coin-overview-card:hover { transform: translateY(-2px); border-color: rgba(57, 245, 163, 0.65); box-shadow: 0 0 24px rgba(57, 245, 163, 0.10); }
/* Themed Streamlit click buttons under coin and agent cards.
   These intentionally remain visible because transparent overlays are unreliable
   across Streamlit/browser versions. */
div[data-testid="stButton"] {
    margin-top: -0.35rem !important;
    margin-bottom: 0.75rem !important;
}

div[data-testid="stButton"] button {
    background: rgba(6, 18, 32, 0.90) !important;
    color: #d9f5ff !important;
    border: 1px solid rgba(80, 220, 255, 0.22) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    min-height: 2.5rem !important;
}

div[data-testid="stButton"] button:hover {
    background: rgba(7, 24, 42, 0.96) !important;
    color: #e8fbff !important;
    border-color: rgba(57, 245, 163, 0.55) !important;
    box-shadow: 0 0 18px rgba(57, 245, 163, 0.10) !important;
}

div[data-testid="stButton"] button:focus {
    outline: 2px solid rgba(57, 245, 163, 0.45) !important;
    outline-offset: 2px !important;
}

.coin-overview-card,
.agent-card {
    position: relative;
}

.coin-overview-card:hover,
.agent-card:hover {
    transform: translateY(-2px);
    border-color: rgba(57, 245, 163, 0.65);
    box-shadow: 0 0 24px rgba(57, 245, 163, 0.10);
}
.coin-overview-card.buy { border-color: rgba(0, 255, 160, 0.48); }
.coin-overview-card.shadow { border-color: rgba(255, 214, 102, 0.45); }
.coin-overview-card.wait { border-color: rgba(135, 159, 180, 0.38); }
.coin-overview-card.blocked { border-color: rgba(255, 92, 122, 0.45); }
.rank-badge { display: inline-block; border: 1px solid rgba(57, 245, 163, 0.45); border-radius: 999px; padding: 0.18rem 0.5rem; font-size: 0.78rem; color: #39f5a3; background: rgba(57, 245, 163, 0.08); margin-right: 0.35rem; }
.viability-score { font-size: 1.35rem; font-weight: 900; color: #e8fbff; }
.viability-reason { color: #8db7c8; font-size: 0.86rem; line-height: 1.25rem; }
.leadership-grid { display: grid; grid-template-columns: 1fr; gap: 0.9rem; margin: 1rem 0 1.1rem 0; }
.leadership-card { border: 1px solid rgba(80, 220, 255, 0.22); border-radius: 20px; padding: 1rem; background: linear-gradient(180deg, rgba(6, 18, 32, 0.96), rgba(4, 12, 24, 0.94)); box-shadow: 0 0 22px rgba(80, 220, 255, 0.06); }
.leadership-card.oracle { border-color: rgba(255, 214, 102, 0.35); box-shadow: 0 0 22px rgba(255, 214, 102, 0.07); }
.leadership-title { font-size: 1.15rem; font-weight: 900; margin-bottom: 0.2rem; }
.leadership-subtitle { color: #8db7c8; font-size: 0.86rem; margin-bottom: 0.65rem; }
.leadership-verdict { font-size: 1.05rem; font-weight: 900; margin-bottom: 0.5rem; }
.leadership-paragraph { color: #d9f5ff; line-height: 1.35rem; font-size: 0.93rem; }
.leadership-learning { margin-top: 0.7rem; padding: 0.65rem; border-radius: 14px; background: rgba(57, 245, 163, 0.06); border: 1px solid rgba(57, 245, 163, 0.18); color: #c9f8e2; font-size: 0.88rem; line-height: 1.25rem; }
@media (max-width: 900px) { .leadership-grid { grid-template-columns: 1fr; } }
.context-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.8rem; }
.context-card { border: 1px solid rgba(80, 220, 255, 0.18); border-radius: 18px; padding: 0.9rem; background: rgba(6, 18, 32, 0.86); }
.context-card h3 { margin-top: 0; margin-bottom: 0.45rem; }
.watch-list { border-left: 3px solid rgba(57, 245, 163, 0.75); padding: 0.65rem 0.8rem; background: rgba(57, 245, 163, 0.06); border-radius: 12px; margin-top: 0.6rem; }
.live-pulse { display: inline-block; width: 0.65rem; height: 0.65rem; border-radius: 50%; background: #39f5a3; box-shadow: 0 0 14px rgba(57, 245, 163, 0.9); margin-right: 0.4rem; }
.agent-ticker { border: 1px solid rgba(0, 255, 194, 0.24); border-radius: 18px; padding: 0.9rem; background: rgba(3, 22, 30, 0.88); margin: 0.75rem 0; }
.agent-row { border-left: 3px solid rgba(80, 220, 255, 0.35); padding: 0.55rem 0.75rem; margin: 0.45rem 0; background: rgba(6, 20, 34, 0.55); border-radius: 12px; }
.agent-row.active { border-left-color: #39f5a3; box-shadow: 0 0 18px rgba(57, 245, 163, 0.12); }
@media (max-width: 900px) { .overview-grid { grid-template-columns: 1fr; } .context-grid { grid-template-columns: 1fr; } }

.leadership-grid {
    grid-template-columns: 1fr !important;
    width: 100% !important;
}
.leadership-card {
    width: 100% !important;
}
.agent-card .agent-summary,
.agent-card .agent-metrics,
.agent-card .agent-description {
    overflow: visible !important;
}
.agent-description {
    margin-top: 8px;
    margin-bottom: 10px;
    font-size: 0.86rem;
    line-height: 1.35;
    opacity: 0.88;
}
</style>
    """, unsafe_allow_html=True)


def file_signature(path: str) -> tuple:
    try:
        if not os.path.exists(path): return (path, False, 0, 0)
        stat = os.stat(path)
        return (path, True, int(stat.st_size), int(stat.st_mtime_ns))
    except Exception:
        return (path, False, 0, 0)


@st.cache_data(show_spinner=False)
def _load_csv_cached(path: str, exists: bool, size_bytes: int, mtime_ns: int, usecols_key: tuple | None = None) -> pd.DataFrame:
    if not exists: return pd.DataFrame()
    usecols = list(usecols_key) if usecols_key else None
    return pd.read_csv(path, usecols=usecols)


def load_csv(path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    sig = file_signature(path); usecols_key = tuple(usecols) if usecols else None
    try:
        frame = _load_csv_cached(sig[0], sig[1], sig[2], sig[3], usecols_key)
        module_debug(MODULE_NAME, "viewer_csv_loaded", data={"path": path, "exists": sig[1], "size_bytes": sig[2], "mtime_ns": sig[3], "rows": int(len(frame)) if hasattr(frame, "__len__") else 0, "columns": list(frame.columns)[:80] if hasattr(frame, "columns") else []}, level="DEBUG", also_overall=False)
        return frame
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer_csv_load_failed", exc, data={"path": path, "signature": sig, "traceback": traceback.format_exc()}, also_overall=True)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_csv_tail_cached(
    path: str,
    exists: bool,
    size_bytes: int,
    mtime_ns: int,
    max_lines: int,
    usecols_key: tuple | None = None,
) -> pd.DataFrame:
    if not exists:
        return pd.DataFrame()

    usecols = list(usecols_key) if usecols_key else None

    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            header = f.readline()
            tail_lines = deque(f, maxlen=max(1, int(max_lines)))

        if not header:
            return pd.DataFrame()

        text = header + "".join(tail_lines)
        return pd.read_csv(StringIO(text), usecols=usecols)

    except Exception:
        if size_bytes <= 5_000_000:
            return pd.read_csv(path, usecols=usecols)
        return pd.DataFrame()


def load_csv_tail(path: str, max_lines: int = 25000, usecols: list[str] | None = None) -> pd.DataFrame:
    sig = file_signature(path)
    usecols_key = tuple(usecols) if usecols else None

    try:
        frame = _load_csv_tail_cached(
            sig[0],
            sig[1],
            sig[2],
            sig[3],
            int(max_lines),
            usecols_key,
        )
        module_debug(MODULE_NAME, "viewer_csv_tail_loaded", data={"path": path, "exists": sig[1], "size_bytes": sig[2], "mtime_ns": sig[3], "rows": int(len(frame)) if hasattr(frame, "__len__") else 0, "max_lines": int(max_lines), "columns": list(frame.columns)[:80] if hasattr(frame, "columns") else []}, level="DEBUG", also_overall=False)
        return frame
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer_csv_tail_load_failed", exc, data={"path": path, "signature": sig, "max_lines": int(max_lines), "traceback": traceback.format_exc()}, also_overall=True)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_viewer_snapshot_cached(path: str, exists: bool, size_bytes: int, mtime_ns: int) -> Dict[str, Any]:
    if not exists:
        return {"updated_ts": 0.0, "coins": {}, "top_products": [], "live_positions": [], "readiness": {"startup_state": "waiting_for_first_bot_snapshot"}, "_startup_waiting": True}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_viewer_snapshot() -> Dict[str, Any]:
    sig = file_signature(VIEWER_SNAPSHOT_PATH)
    try:
        snapshot = _load_viewer_snapshot_cached(sig[0], sig[1], sig[2], sig[3])
        if not sig[1]:
            module_debug(MODULE_NAME, "viewer_snapshot_missing_startup_wait", data={"path": VIEWER_SNAPSHOT_PATH}, level="INFO", also_overall=False)
        else:
            module_debug(MODULE_NAME, "viewer_snapshot_loaded", data={"path": VIEWER_SNAPSHOT_PATH, "size_bytes": sig[2], "mtime_ns": sig[3], "coin_count": len((snapshot.get("coins") or {})), "updated_ts": snapshot.get("updated_ts")}, level="DEBUG", also_overall=False)
        return snapshot
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer_snapshot_load_failed", exc, data={"path": VIEWER_SNAPSHOT_PATH, "signature": sig, "traceback": traceback.format_exc()}, also_overall=True)
        return {"updated_ts": 0.0, "coins": {}, "top_products": [], "live_positions": [], "readiness": {}, "_viewer_snapshot_error": f"{type(exc).__name__}: {exc}"}


@st.cache_data(show_spinner=False)
def _load_calculation_status_cached(path: str, exists: bool, size_bytes: int, mtime_ns: int) -> dict:
    if not exists:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        module_exception(MODULE_NAME, "calculation_status_load_failed", exc, data={"path": path, "traceback": traceback.format_exc()}, also_overall=False)
        return {}


def _viewer_count_rows_by_product(path: str) -> Dict[str, int]:
    try:
        df = load_csv_tail(path, max_lines=300000)
        if df is None or df.empty or "product_id" not in df.columns:
            return {}
        return df["product_id"].astype(str).value_counts().to_dict()
    except Exception:
        return {}


def _viewer_product_ids(snapshot: dict, manifest: dict, micro_counts: Dict[str, int]) -> list:
    ids = []

    try:
        ids.extend([str(x) for x in ((snapshot or {}).get("coins") or {}).keys() if str(x)])
    except Exception:
        pass

    try:
        jobs = ((manifest or {}).get("jobs") or {}).values()
        ids.extend([str(j.get("product_id") or "") for j in jobs if str(j.get("product_id") or "")])
    except Exception:
        pass

    try:
        ids.extend([str(x) for x in micro_counts.keys() if str(x)])
    except Exception:
        pass

    try:
        exchange_map = load_csv(EXCHANGE_PRODUCT_MAP_CSV_PATH)
        if exchange_map is not None and not exchange_map.empty:
            if "canonical_product_id" in exchange_map.columns:
                ids.extend([str(x) for x in exchange_map["canonical_product_id"].dropna().astype(str).tolist() if str(x)])
            elif "coinbase_product_id" in exchange_map.columns:
                ids.extend([str(x) for x in exchange_map["coinbase_product_id"].dropna().astype(str).tolist() if str(x)])
    except Exception:
        pass

    out = []
    seen = set()
    for product_id in ids:
        product_id = str(product_id).strip()
        if not product_id or product_id in seen:
            continue
        seen.add(product_id)
        out.append(product_id)

    return out


def _viewer_manifest_progress(manifest: dict) -> dict:
    jobs = list(((manifest or {}).get("jobs") or {}).values())
    total = len(jobs)

    counts = {}
    for job in jobs:
        status = str(job.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1

    complete = counts.get("done", 0) + counts.get("merged", 0)

    return {
        "total_jobs": int(total),
        "done_jobs": int(counts.get("done", 0)),
        "merged_jobs": int(counts.get("merged", 0)),
        "failed_jobs": int(counts.get("failed", 0)),
        "running_jobs": int(counts.get("running", 0)),
        "pending_jobs": int(counts.get("pending", 0)),
        "progress": float(complete / max(1, total)),
        "progress_pct": float(complete / max(1, total) * 100.0),
        "running_jobs_detail": [
            {
                "job_id": str(j.get("job_id") or ""),
                "product_id": str(j.get("product_id") or ""),
                "timeframe": str(j.get("timeframe") or ""),
                "started_ts": float(j.get("started_ts", 0.0) or 0.0),
            }
            for j in jobs
            if str(j.get("status") or "") == "running"
        ][:10],
        "failed_job_errors": [
            {
                "job_id": str(j.get("job_id") or ""),
                "product_id": str(j.get("product_id") or ""),
                "timeframe": str(j.get("timeframe") or ""),
                "attempts": int(j.get("attempts", 0) or 0),
                "error": str(j.get("error") or ""),
            }
            for j in jobs
            if str(j.get("status") or "") == "failed"
        ][-10:],
        "next_pending_jobs": [
            {
                "job_id": str(j.get("job_id") or ""),
                "product_id": str(j.get("product_id") or ""),
                "timeframe": str(j.get("timeframe") or ""),
            }
            for j in jobs
            if str(j.get("status") or "pending") == "pending"
        ][:10],
    }


def _load_calculation_complete_latch_for_viewer() -> dict:
    try:
        if (
            not os.path.exists(CALCULATION_COMPLETE_LATCH_JSON_PATH)
            or os.path.getsize(CALCULATION_COMPLETE_LATCH_JSON_PATH) <= 0
        ):
            return {}
        with open(CALCULATION_COMPLETE_LATCH_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _status_is_startup_complete(status: dict) -> bool:
    try:
        if not isinstance(status, dict):
            return False
        return bool(
            status.get("calculation_complete_latched")
            or status.get("calculation_work_complete")
            or status.get("full_viewer_unlocked")
        )
    except Exception:
        return False


def _normalize_completed_calculation_status(status: dict, *, source: str) -> dict:
    """Force completed startup status to stay completed.

    Once startup is latched, the viewer must not go back to a fake loading bar
    because live data freshness temporarily looks stale.
    """
    out = dict(status or {})
    now_value = time.time()

    out["ts"] = float(out.get("ts", 0.0) or now_value)
    out["full_viewer_unlocked"] = True
    out["calculation_work_complete"] = True
    out["calculation_complete_latched"] = True
    out["overall_progress"] = 1.0
    out["overall_progress_pct"] = 100.0
    out["phase_label"] = "Complete"

    phase_progress = dict(out.get("phase_progress") or {})
    phase_progress["micro_backlog"] = 1.0
    phase_progress["historical_candle_backlog"] = 1.0
    phase_progress["historical_replay"] = 1.0
    phase_progress["replay_calibration_verdicts"] = 1.0
    phase_progress["live_data"] = max(0.0, min(1.0, float(phase_progress.get("live_data", 1.0) or 1.0)))
    out["phase_progress"] = phase_progress

    product_status = out.get("product_status") or {}
    if isinstance(product_status, dict):
        out["product_count"] = int(out.get("product_count", len(product_status)) or len(product_status))
        out["complete_products"] = int(out.get("complete_products", len(product_status)) or len(product_status))
        out["incomplete_products"] = 0

    out["viewer_status_source"] = source
    out["viewer_status_reason"] = "startup completion latch exists, so viewer remains unlocked even if calculation_status.json is stale"

    return out


def _synthesize_calculation_status_for_viewer(snapshot: dict) -> dict:
    """Viewer-side fallback when bot has not written calculation_status.json yet.

    This prevents the loading screen from lying with 0.0% and 0 seconds while
    startup files already prove calculation/backlog work is underway.
    """
    now_value = time.time()

    manifest = {}
    if os.path.exists(HISTORICAL_REPLAY_MANIFEST_JSON_PATH):
        try:
            with open(HISTORICAL_REPLAY_MANIFEST_JSON_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    micro_counts = _viewer_count_rows_by_product(MICRO_HISTORY_CSV_PATH)
    candle_15m_counts = _viewer_count_rows_by_product(HIST_REPLAY_15M_90D_CSV_PATH)
    candle_1h_counts = _viewer_count_rows_by_product(HIST_REPLAY_1H_365D_CSV_PATH)

    replay_counts_15m = {}
    replay_counts_1h = {}

    replay_df = load_csv_tail(HISTORICAL_SHADOW_REPLAY_CSV_PATH, max_lines=300000)
    if replay_df is not None and not replay_df.empty and "product_id" in replay_df.columns:
        replay_df = replay_df.copy()
        replay_df["product_id"] = replay_df["product_id"].astype(str)
        timeframe_series = replay_df["timeframe"].astype(str) if "timeframe" in replay_df.columns else pd.Series([""] * len(replay_df))

        replay_counts_15m = replay_df[
            timeframe_series.str.contains("15m|primary", case=False, regex=True, na=False)
        ]["product_id"].value_counts().to_dict()

        replay_counts_1h = replay_df[
            timeframe_series.str.contains("1h|regime", case=False, regex=True, na=False)
        ]["product_id"].value_counts().to_dict()

    product_ids = _viewer_product_ids(snapshot, manifest, micro_counts)
    product_status = {}

    for product_id in product_ids:
        micro_rows = int(micro_counts.get(product_id, 0) or 0)
        candle_15m_rows = int(candle_15m_counts.get(product_id, 0) or 0)
        candle_1h_rows = int(candle_1h_counts.get(product_id, 0) or 0)
        replay_15m_rows = int(replay_counts_15m.get(product_id, 0) or 0)
        replay_1h_rows = int(replay_counts_1h.get(product_id, 0) or 0)

        micro_progress = min(1.0, micro_rows / max(1.0, float(STARTUP_CALC_REQUIRED_MICRO_ROWS_PER_PRODUCT)))
        candle_15m_progress = min(1.0, candle_15m_rows / max(1.0, float(STARTUP_CALC_REQUIRED_15M_CANDLE_ROWS_PER_PRODUCT)))
        candle_1h_progress = min(1.0, candle_1h_rows / max(1.0, float(STARTUP_CALC_REQUIRED_1H_CANDLE_ROWS_PER_PRODUCT)))
        replay_15m_progress = min(1.0, replay_15m_rows / max(1.0, float(STARTUP_CALC_REQUIRED_15M_REPLAY_ROWS_PER_PRODUCT)))
        replay_1h_progress = min(1.0, replay_1h_rows / max(1.0, float(STARTUP_CALC_REQUIRED_1H_REPLAY_ROWS_PER_PRODUCT)))

        historical_candle_progress = (candle_15m_progress + candle_1h_progress) / 2.0
        historical_replay_progress = (replay_15m_progress + replay_1h_progress) / 2.0

        complete = bool(
            micro_progress >= 1.0
            and historical_candle_progress >= 1.0
            and historical_replay_progress >= 1.0
        )

        product_status[product_id] = {
            "product_id": product_id,
            "micro_rows": micro_rows,
            "historical_15m_candle_rows": candle_15m_rows,
            "required_15m_candle_rows": STARTUP_CALC_REQUIRED_15M_CANDLE_ROWS_PER_PRODUCT,
            "historical_1h_candle_rows": candle_1h_rows,
            "required_1h_candle_rows": STARTUP_CALC_REQUIRED_1H_CANDLE_ROWS_PER_PRODUCT,
            "primary_15m_90d_rows": replay_15m_rows,
            "regime_1h_365d_rows": replay_1h_rows,
            "qualified_rows": replay_15m_rows + replay_1h_rows,
            "avg_net_pnl_bps": 0.0,
            "verdict": "viewer_synthesized_status",
            "complete": complete,
            "profit_ready": False,
            "live_trade_allowed": False,
            "reason": "viewer_synthesized_from_csvs_because_calculation_status_json_missing_or_stale",
            "micro_progress": float(micro_progress),
            "historical_candle_progress": float(historical_candle_progress),
            "historical_replay_progress": float(historical_replay_progress),
            "calibration_verdict_progress": 0.0,
            "overall_product_progress": float(
                (micro_progress * 0.15)
                + (historical_candle_progress * 0.20)
                + (historical_replay_progress * 0.35)
            ),
        }

    product_count = max(1, len(product_ids))

    phase_progress = {
        "live_data": 1.0 if ((snapshot or {}).get("updated_ts") or os.path.exists(MARKET_CSV_PATH)) else 0.0,
        "micro_backlog": sum(v["micro_progress"] for v in product_status.values()) / product_count,
        "historical_candle_backlog": sum(v["historical_candle_progress"] for v in product_status.values()) / product_count,
        "historical_replay": sum(v["historical_replay_progress"] for v in product_status.values()) / product_count,
        "replay_calibration_verdicts": 0.0,
    }

    overall_progress = (
        phase_progress["live_data"] * 0.10
        + phase_progress["micro_backlog"] * 0.15
        + phase_progress["historical_candle_backlog"] * 0.20
        + phase_progress["historical_replay"] * 0.35
        + phase_progress["replay_calibration_verdicts"] * 0.20
    )

    start_ts_candidates = [
        float((manifest or {}).get("created_ts") or 0.0),
        float((snapshot or {}).get("calculation_started_ts") or 0.0),
        float((snapshot or {}).get("updated_ts") or 0.0),
    ]
    start_ts_candidates = [x for x in start_ts_candidates if x > 0.0]

    calculation_started_ts = min(start_ts_candidates) if start_ts_candidates else now_value
    elapsed_sec = max(0.0, now_value - calculation_started_ts)

    if phase_progress["historical_candle_backlog"] < 1.0:
        phase_label = "Building historical candle backlogs"
    elif phase_progress["historical_replay"] < 1.0:
        phase_label = "Running historical replay across all products"
    elif phase_progress["replay_calibration_verdicts"] < 1.0:
        phase_label = "Calculating replay-based product verdicts"
    else:
        phase_label = "Final readiness checks"

    worker_manifest = _viewer_manifest_progress(manifest)

    return {
        "ts": now_value,
        "calculation_started_ts": float(calculation_started_ts),
        "calculation_elapsed_sec": float(elapsed_sec),
        "full_viewer_unlocked": False,
        "calculation_work_complete": False,
        "calculation_complete_latched": False,
        "overall_progress": float(max(0.0, min(1.0, overall_progress))),
        "overall_progress_pct": float(max(0.0, min(100.0, overall_progress * 100.0))),
        "phase_label": phase_label,
        "phase_progress": phase_progress,
        "product_count": int(len(product_ids)),
        "complete_products": int(sum(1 for v in product_status.values() if v.get("complete"))),
        "profit_ready_products": 0,
        "blocked_products": 0,
        "incomplete_products": int(len(product_ids) - sum(1 for v in product_status.values() if v.get("complete"))),
        "product_status": product_status,
        "historical_replay_worker_manifest": worker_manifest,
        "readiness": (snapshot or {}).get("readiness", {}),
        "policy": (snapshot or {}).get("readiness", {}),
        "viewer_status_source": "synthesized_from_csvs_and_manifest",
        "viewer_status_reason": "calculation_status.json was missing or stale, so viewer calculated progress from existing runtime files",
    }


def load_calculation_status(snapshot: dict | None = None) -> dict:
    sig = file_signature(CALCULATION_STATUS_JSON_PATH)
    status = _load_calculation_status_cached(sig[0], sig[1], sig[2], sig[3])
    latch = _load_calculation_complete_latch_for_viewer()

    # Permanent completion latch wins over everything.
    # This prevents the viewer from falling back into fake loading mode.
    if _status_is_startup_complete(latch):
        merged = dict(status or {})
        merged.update(latch)
        return _normalize_completed_calculation_status(
            merged,
            source="calculation_complete_latch.json",
        )

    # Fresh completed calculation status also wins.
    if _status_is_startup_complete(status or {}):
        return _normalize_completed_calculation_status(
            status,
            source="calculation_status.json",
        )

    # If the status file is fresh and not complete yet, use it normally.
    if status:
        try:
            status_ts = float(status.get("ts", 0.0) or 0.0)
            if status_ts > 0 and time.time() - status_ts <= 15.0:
                return status
        except Exception:
            return status

    # Only synthesize fallback progress when there is no completed latch/status.
    synthesized = _synthesize_calculation_status_for_viewer(snapshot or load_viewer_snapshot())

    if status:
        synthesized["bot_calculation_status_stale"] = True
        synthesized["bot_calculation_status_age_sec"] = max(
            0.0,
            time.time() - float(status.get("ts", 0.0) or 0.0),
        )
    else:
        synthesized["bot_calculation_status_missing"] = True

    return synthesized


def dataframe_latest_age_sec(frame: pd.DataFrame) -> float:
    try:
        if frame.empty or "ts" not in frame.columns: return 999999.0
        ts = pd.to_numeric(frame["ts"], errors="coerce").dropna()
        return max(0.0, time.time() - float(ts.max())) if not ts.empty else 999999.0
    except Exception:
        return 999999.0


def format_age(age_sec: float) -> str:
    age_sec = max(0.0, float(age_sec or 0.0))
    return f"{age_sec:.0f}s" if age_sec < 60 else f"{age_sec / 60.0:.1f}m"


def format_elapsed_duration(seconds: float) -> str:
    try:
        seconds = max(0, int(float(seconds)))
    except Exception:
        seconds = 0
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def freshness_class(age_sec: float, warn: float, danger: float) -> str:
    return "good" if age_sec <= warn else "warn" if age_sec <= danger else "danger"


def get_refresh_config() -> dict:
    """
    Viewer refresh is intentionally automatic and hidden.
    The loading/calibration screen and the unlocked live dashboard must both
    refresh from the same auto-refresh path.
    """
    return {
        "live_enabled": True,
        "interval_label": "2s",
        "fallback_interval_sec": 2.0,
        "manual_tick": int(st.session_state.get("_manual_refresh_tick", 0)),
        "fragment_supported": callable(getattr(st, "fragment", None)),
    }


def run_every_value(refresh_config: dict):
    if not refresh_config.get("live_enabled"): return None
    if not refresh_config.get("fragment_supported"): return None
    return refresh_config.get("interval_label", "5s")


@contextmanager
def render_section(name: str):
    module_debug(MODULE_NAME, "render_section_start", data={"section": name}, level="DEBUG", also_overall=False)
    try:
        yield
        module_debug(MODULE_NAME, "render_section_end", data={"section": name}, level="DEBUG", also_overall=False)
    except Exception as exc:
        module_exception(MODULE_NAME, f"render_section_failed:{name}", exc, also_overall=True); raise


def render_crypto_header() -> None:
    st.markdown('<div class="hud-header"><div class="hud-title">🛰️ Crypto Strategy HUD</div><div class="hud-subtitle">Strategy Arena for live crypto learning, agent consensus, and Binance.US chart context.</div></div>', unsafe_allow_html=True)


def get_available_products(snapshot: Dict[str, Any]) -> list[str]:
    """
    Product list priority:
    1. viewer_snapshot top_products / coins
    2. products_active.csv
    3. market.csv
    4. council_decisions.csv
    5. position_targets.csv

    This keeps the viewer usable even when viewer_snapshot.json is briefly
    overwritten by a startup snapshot with zero coins.
    """
    products: list[str] = []

    def add_product(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in products:
            products.append(text)

    try:
        coins = snapshot.get("coins", {}) or {}
        top = snapshot.get("top_products", []) or []
        for p in top:
            add_product(p)
        for p in coins.keys():
            add_product(p)
    except Exception:
        pass

    if products:
        return products

    fallback_paths = [os.path.join(BASE_DIR, "products_active.csv"), MARKET_CSV_PATH, COUNCIL_DECISIONS_PATH, POSITION_TARGETS_PATH]
    for path in fallback_paths:
        try:
            frame = load_csv(path, usecols=["product_id"])
            if frame.empty or "product_id" not in frame.columns:
                continue
            for p in frame["product_id"].dropna().astype(str).unique().tolist():
                add_product(p)
            if products:
                module_debug(MODULE_NAME, "available_products_loaded_from_fallback", data={"path": path, "count": len(products)}, level="INFO", also_overall=False)
                return products
        except Exception as exc:
            module_exception(MODULE_NAME, "available_products_fallback_failed", exc, data={"path": path}, also_overall=False)
    return products

def pick_selected_coin(snapshot: Dict[str, Any]) -> str | None:
    available = get_available_products(snapshot)
    if not available:
        return None
    if st.session_state.get("selected_coin") not in available:
        st.session_state["selected_coin"] = available[0]
    return st.session_state["selected_coin"]


def normalize_timeframe_label(label: str) -> str:
    mapping = {"1D · 1m": "1d_1m", "7D · 1m": "7d_1m", "30D · 15m": "30d_15m", "90D · 1h": "90d_1h", "2Y · 1d": "2y_1d"}
    return mapping.get(str(label), "1d_1m")


def render_overlay_controls():
    defaults = {
        "volume": True,
        "confirmed_trades": True,
        "shadow_trades": True,
        "profile": True,
        "prior_profile": True,
        "average_entry": True,
        "targets": True,
        "vwap": True,
        "structure": False,
        "level8_markers": True,
    }
    labels = {
        "volume": "Volume bars",
        "confirmed_trades": "Confirmed live buys/sells",
        "shadow_trades": "Shadow trades",
        "profile": "POC / VAH / VAL",
        "prior_profile": "Prior-session POC / VAH / VAL",
        "average_entry": "Average entry",
        "targets": "Target buy / target sell / stop / min-profitable exit",
        "vwap": "VWAP / anchored VWAP",
        "structure": "Trend / structure / liquidity lines",
        "level8_markers": "Level 8 action markers",
    }

    with st.expander("Chart overlays", expanded=False):
        st.caption("Toggle every line or marker the bot can display on the chart.")
        cols = st.columns(2)
        toggles = {}
        for idx, (key, default) in enumerate(defaults.items()):
            with cols[idx % 2]:
                toggles[key] = st.checkbox(labels[key], value=default, key=f"overlay_{key}")
        return toggles



def build_open_position_lots_from_trades(trades_df: pd.DataFrame, market_df: pd.DataFrame, targets_df: pd.DataFrame) -> list[dict]:
    '''Reconstruct currently open lots from trades.csv using FIFO.'''
    open_lots_by_product: dict[str, list[dict]] = {}
    if trades_df is not None and not trades_df.empty and "product_id" in trades_df.columns:
        trades = trades_df.copy()
        trades["ts_num"] = pd.to_numeric(trades.get("ts"), errors="coerce")
        trades = trades.dropna(subset=["ts_num"]).sort_values("ts_num")
        for _, row in trades.iterrows():
            product_id = str(row.get("product_id") or "").strip()
            if not product_id:
                continue
            event = str(row.get("event") or "").upper().strip()
            side = str(row.get("side") or "").upper().strip()
            qty = _safe_float(row.get("qty"), 0.0)
            if qty <= 0:
                continue
            if event == "BUY" or side == "BUY":
                raw_price = _safe_float(row.get("price"), 0.0)
                entry_price = _safe_float(row.get("entry_price"), 0.0)
                note_text = str(row.get("note", ""))
                requested_quote = parse_note_float(note_text, "requested_quote_usd", 0.0)
                filled_notional_note = parse_note_float(note_text, "filled_notional_usd", 0.0)
                partial_fill_ratio = parse_note_float(note_text, "partial_fill_ratio", 0.0)
                all_in_price = parse_note_float(note_text, "all_in_entry_price", 0.0)
                if all_in_price > 0:
                    entry_price = all_in_price
                elif entry_price <= 0:
                    entry_price = raw_price
                open_lots_by_product.setdefault(product_id, []).append({
                    "source": "trades_csv", "product_id": product_id, "entry_ts": _safe_float(row.get("ts"), 0.0),
                    "qty": qty, "entry_price": entry_price,
                    "requested_quote_usd": requested_quote,
                    "filled_notional_usd": filled_notional_note or _safe_float(row.get("notional_usd"), qty * raw_price),
                    "notional_usd": _safe_float(row.get("notional_usd"), qty * entry_price),
                    "fee_usd": _safe_float(row.get("fee_usd"), 0.0),
                    "partial_fill_ratio": partial_fill_ratio,
                    "note": note_text,
                })
            elif event == "SELL" or side == "SELL" or event == "STARTUP_LIQUIDATION":
                remaining_sell_qty = qty
                lots = open_lots_by_product.get(product_id, [])
                while remaining_sell_qty > 1e-12 and lots:
                    first = lots[0]
                    first_qty = _safe_float(first.get("qty"), 0.0)
                    if first_qty <= remaining_sell_qty + 1e-12:
                        remaining_sell_qty -= first_qty
                        lots.pop(0)
                    else:
                        first["qty"] = first_qty - remaining_sell_qty
                        remaining_sell_qty = 0.0
                open_lots_by_product[product_id] = lots
    open_lots = []
    for product_id, lots in open_lots_by_product.items():
        market = latest_row_for_product(market_df, product_id)
        target = latest_row_for_product(targets_df, product_id)
        current_price = _safe_float(market.get("mid") or target.get("current_bid") or target.get("current_ask"), 0.0)
        for lot in lots:
            qty = _safe_float(lot.get("qty"), 0.0)
            entry_price = _safe_float(lot.get("entry_price"), 0.0)
            entry_ts = _safe_float(lot.get("entry_ts"), 0.0)
            current_value = qty * current_price if current_price > 0 else 0.0
            cost_basis = qty * entry_price if entry_price > 0 else _safe_float(lot.get("notional_usd"), 0.0)
            open_lots.append({**lot, "current_price": current_price, "current_value": current_value, "cost_basis": cost_basis, "unrealized_pnl_usd": current_value - cost_basis, "unrealized_pnl_pct": ((current_price / entry_price) - 1.0) * 100.0 if current_price > 0 and entry_price > 0 else 0.0, "held_seconds": max(0.0, time.time() - entry_ts) if entry_ts > 0 else 0.0})
    if open_lots:
        return open_lots
    if targets_df is not None and not targets_df.empty and "has_position" in targets_df.columns:
        for _, row in targets_df.iterrows():
            if not boolish(row.get("has_position")):
                continue
            product_id = str(row.get("product_id") or "").strip()
            qty = _safe_float(row.get("position_qty"), 0.0)
            entry_price = _safe_float(row.get("avg_entry_price"), 0.0)
            current_price = _safe_float(row.get("current_bid") or row.get("current_ask"), 0.0)
            entry_ts = _safe_float(row.get("entry_ts"), 0.0)
            if not product_id or qty <= 0:
                continue
            cost_basis = qty * entry_price if entry_price > 0 else 0.0
            current_value = qty * current_price if current_price > 0 else 0.0
            open_lots.append({"source": "position_targets_fallback", "product_id": product_id, "entry_ts": entry_ts, "qty": qty, "entry_price": entry_price, "notional_usd": cost_basis, "fee_usd": 0.0, "current_price": current_price, "current_value": current_value, "cost_basis": cost_basis, "unrealized_pnl_usd": current_value - cost_basis, "unrealized_pnl_pct": ((current_price / entry_price) - 1.0) * 100.0 if current_price > 0 and entry_price > 0 else 0.0, "held_seconds": max(0.0, time.time() - entry_ts) if entry_ts > 0 else 0.0, "note": str(row.get("exit_plan_note", ""))})
    return open_lots


def build_recent_closed_sales(trades_df: pd.DataFrame, limit: int = 10) -> list[dict]:
    if trades_df is None or trades_df.empty or "product_id" not in trades_df.columns:
        return []
    df = trades_df.copy()
    df["ts_num"] = pd.to_numeric(df.get("ts"), errors="coerce")
    df = df.dropna(subset=["ts_num"])
    event_text = df.get("event", pd.Series("", index=df.index)).astype(str).str.upper()
    side_text = df.get("side", pd.Series("", index=df.index)).astype(str).str.upper()
    sales = df[(event_text.eq("SELL")) | (side_text.eq("SELL")) | (event_text.eq("STARTUP_LIQUIDATION"))].copy()
    if sales.empty:
        return []
    out = []
    for _, row in sales.sort_values("ts_num", ascending=False).head(int(limit)).iterrows():
        net_pnl = _safe_float(row.get("net_pnl_usd"), 0.0)
        gross_pnl = _safe_float(row.get("gross_pnl_usd"), 0.0)
        note = str(row.get("note", ""))
        exit_role = str(row.get("exit_role", ""))
        is_stop = net_pnl < 0 or "stop" in note.lower() or "hard_stop" in note.lower() or "stop" in exit_role.lower()
        out.append({"product_id": str(row.get("product_id") or ""), "sell_ts": _safe_float(row.get("ts"), 0.0), "qty": _safe_float(row.get("qty"), 0.0), "exit_price": _safe_float(row.get("exit_price") or row.get("price"), 0.0), "entry_price": _safe_float(row.get("entry_price"), 0.0), "notional_usd": _safe_float(row.get("notional_usd"), 0.0), "fee_usd": _safe_float(row.get("fee_usd"), 0.0), "gross_pnl_usd": gross_pnl, "net_pnl_usd": net_pnl, "exit_role": exit_role, "note": note, "result_label": "STOP LOSS" if is_stop else "PROFIT", "result_class": "sold-loss" if is_stop else "sold-profit"})
    return out


def render_held_positions(snapshot: Dict[str, Any], market_df: pd.DataFrame, trades_df: pd.DataFrame, targets_df: pd.DataFrame) -> None:
    open_lots = build_open_position_lots_from_trades(trades_df, market_df, targets_df)
    recent_sales = build_recent_closed_sales(trades_df, limit=10)
    st.markdown(f'''<div class="held-banner">Currently Held Positions · Local timezone: {_html(local_timezone_label())}</div>''', unsafe_allow_html=True)
    if not open_lots:
        st.info("No currently held positions.")
    else:
        html = ['<div class="held-grid">']
        for lot in open_lots:
            product_id = str(lot.get("product_id") or "")
            qty = _safe_float(lot.get("qty"), 0.0)
            entry_price = _safe_float(lot.get("entry_price"), 0.0)
            current_price = _safe_float(lot.get("current_price"), 0.0)
            notional = _safe_float(lot.get("notional_usd") or lot.get("cost_basis"), 0.0)
            requested_quote = _safe_float(lot.get("requested_quote_usd"), 0.0)
            filled_notional = _safe_float(lot.get("filled_notional_usd"), 0.0)
            partial_fill_ratio = _safe_float(lot.get("partial_fill_ratio"), 0.0)
            partial_fill_html = f'<div class="muted">Partial fill: <b>{partial_fill_ratio * 100.0:.1f}%</b></div>' if partial_fill_ratio > 0 and partial_fill_ratio < 0.98 else ''
            current_value = _safe_float(lot.get("current_value"), 0.0)
            unrealized = _safe_float(lot.get("unrealized_pnl_usd"), 0.0)
            unrealized_pct = _safe_float(lot.get("unrealized_pnl_pct"), 0.0)
            entry_ts = _safe_float(lot.get("entry_ts"), 0.0)
            held_seconds = _safe_float(lot.get("held_seconds"), 0.0)
            pnl_class = "positive" if unrealized >= 0 else "negative"
            html.append(f'''<div class="held-position-card"><div style="font-size:1.25rem;font-weight:900;">{_html(product_id)}</div><div class="muted">Open position</div><div>Quantity: <b>{qty:.12f}</b></div><div>Requested order: <b>{signed_usd(requested_quote if requested_quote > 0 else notional)}</b></div><div>Filled amount: <b>{signed_usd(filled_notional if filled_notional > 0 else notional)}</b></div>{partial_fill_html}<div>Entry price: <b>{entry_price:.8f}</b></div><div>Current price: <b>{current_price:.8f}</b></div><div>Current value: <b>{signed_usd(current_value)}</b></div><div>Purchased: <b>{_html(format_local_datetime(entry_ts))}</b></div><div>Held: <b>{_html(format_hold_duration(held_seconds))}</b></div><div class="position-pnl-banner {pnl_class}">Currently {signed_usd(unrealized)} · {signed_pct(unrealized_pct)}</div></div>''')
        html.append('</div>')
        st.markdown("".join(html), unsafe_allow_html=True)
    if recent_sales:
        st.markdown('<div class="held-banner">Most Recent Closed Sales</div>', unsafe_allow_html=True)
        html = ['<div class="held-grid">']
        for sale in recent_sales:
            product_id = str(sale.get("product_id") or "")
            result_class = str(sale.get("result_class") or "sold-profit")
            is_profit = result_class == "sold-profit"
            overlay_class = "profit" if is_profit else "loss"
            label = "PROFIT" if is_profit else "STOP LOSS"
            html.append(f'''<div class="held-position-card {result_class}"><div class="sale-result-overlay {overlay_class}">{label}</div><div style="font-size:1.25rem;font-weight:900;">{_html(product_id)}</div><div class="muted">Closed sale</div><div>Quantity sold: <b>{_safe_float(sale.get("qty")):.12f}</b></div><div>Entry price: <b>{_safe_float(sale.get("entry_price")):.8f}</b></div><div>Exit price: <b>{_safe_float(sale.get("exit_price")):.8f}</b></div><div>Sold: <b>{_html(format_local_datetime(sale.get("sell_ts")))}</b></div><div>Fee: <b>{signed_usd(sale.get("fee_usd"))}</b></div><div>Gross P/L: <b>{signed_usd(sale.get("gross_pnl_usd"))}</b></div><div class="position-pnl-banner {'positive' if is_profit else 'negative'}">Net {signed_usd(sale.get("net_pnl_usd"))}</div></div>''')
        html.append('</div>')
        st.markdown("".join(html), unsafe_allow_html=True)

def latest_targets_for_coin(df: pd.DataFrame, product_id: str) -> Dict[str, Any]:
    if df.empty or "product_id" not in df.columns: return {}
    sub = df[df["product_id"].astype(str) == str(product_id)].copy()
    if sub.empty: return {}
    if "ts" in sub.columns:
        sub["ts_num"] = pd.to_numeric(sub["ts"], errors="coerce"); sub = sub.sort_values("ts_num")
    return sub.iloc[-1].to_dict()


def confirmed_trades_only(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    if df.empty or "product_id" not in df.columns: return pd.DataFrame()
    sub = df[df["product_id"].astype(str) == str(product_id)].copy()
    for status_col in [c for c in ["status", "trade_status", "result", "fill_status"] if c in sub.columns]:
        mask = sub[status_col].astype(str).str.lower().isin(["filled", "confirmed", "executed", "success", "complete", "completed"])
        if mask.any(): sub = sub[mask].copy(); break
    if "ts" in sub.columns:
        sub["ts_num"] = pd.to_numeric(sub["ts"], errors="coerce"); sub = sub.sort_values("ts_num", ascending=False)
    return sub.head(20)


def load_chart_history(product_id: str, timeframe: str) -> tuple[pd.DataFrame, dict]:
    tf = str(timeframe or "1d_1m").lower()
    if tf == "7d_1m":
        source_path, source_name, fallback_path, fallback_name, max_rows = CHART_1M_7D_CSV_PATH, "chart_1m_7d.csv", MICRO_HISTORY_CSV_PATH, "micro_history.csv", 7 * 24 * 60 + 100
    elif tf == "30d_15m":
        source_path, source_name, fallback_path, fallback_name, max_rows = CHART_15M_30D_CSV_PATH, "chart_15m_30d.csv", MACRO_WEEK_CSV_PATH, "macro_week.csv", 30 * 24 * 4 + 100
    elif tf == "90d_1h":
        source_path, source_name, fallback_path, fallback_name, max_rows = CHART_1H_90D_CSV_PATH, "chart_1h_90d.csv", MACRO_WEEK_CSV_PATH, "macro_week.csv", 90 * 24 + 100
    elif tf == "2y_1d":
        source_path, source_name, fallback_path, fallback_name, max_rows = CHART_1D_2Y_CSV_PATH, "chart_1d_2y.csv", MACRO_WEEK_CSV_PATH, "macro_week.csv", 2 * 365 + 30
    else:
        source_path, source_name, fallback_path, fallback_name, max_rows = MICRO_HISTORY_CSV_PATH, "micro_history.csv", MACRO_DAY_CSV_PATH, "macro_day.csv", 24 * 60 + 100
    frame = load_csv(source_path)
    if frame.empty and fallback_path:
        frame = load_csv(fallback_path); source_path = fallback_path; source_name = fallback_name
    required = ["ts", "product_id", "open", "high", "low", "close", "volume"]
    meta = {"product_id": product_id, "timeframe": tf, "source": source_name, "path": source_path, "rows_before_filter": int(len(frame)) if hasattr(frame, "__len__") else 0, "rows": 0, "has_volume": False, "age_sec": 999999.0, "missing_columns": []}
    if frame.empty: return pd.DataFrame(), meta
    missing = [c for c in required if c not in frame.columns]; meta["missing_columns"] = missing
    if missing:
        module_debug(MODULE_NAME, "chart_history_missing_columns", data=meta, level="WARN", also_overall=True); return pd.DataFrame(), meta
    out = frame[frame["product_id"].astype(str) == str(product_id)].copy()
    for col in ["ts", "open", "high", "low", "close", "volume"]: out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
    out["dt"] = pd.to_datetime(out["ts"], unit="s", errors="coerce", utc=True)
    out = out.tail(max_rows)
    meta.update({"rows": int(len(out)), "has_volume": bool(pd.to_numeric(out.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() > 0), "age_sec": dataframe_latest_age_sec(out)})
    module_debug(MODULE_NAME, "chart_history_selected", data=meta, level="INFO", also_overall=False)
    return out, meta


def _first_price(*vals):
    for v in vals:
        f = _safe_float(v, 0.0)
        if f > 0: return f
    return 0.0


def _nearest_close(chart_df: pd.DataFrame, ts_value: Any) -> float:
    try:
        t = pd.to_numeric(pd.Series([ts_value]), errors="coerce").iloc[0]
        if pd.isna(t):
            dt = pd.to_datetime(ts_value, errors="coerce", utc=True); t = dt.timestamp() if pd.notna(dt) else 0
        idx = (pd.to_numeric(chart_df["ts"], errors="coerce") - float(t)).abs().idxmin()
        return float(chart_df.loc[idx, "close"])
    except Exception: return 0.0


def _marker_df(df: pd.DataFrame, product_id: str, chart_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "product_id" not in df.columns or "ts" not in df.columns: return pd.DataFrame()
    out = df[df["product_id"].astype(str) == str(product_id)].copy()
    if out.empty: return out
    out["dt"] = pd.to_datetime(pd.to_numeric(out["ts"], errors="coerce"), unit="s", errors="coerce", utc=True)
    if "price" not in out.columns: out["price"] = 0.0
    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
    if not chart_df.empty: out.loc[out["price"] <= 0, "price"] = out.loc[out["price"] <= 0, "ts"].apply(lambda x: _nearest_close(chart_df, x))
    return out[out["price"] > 0]


def build_coin_chart(chart_df, chart_meta, coin_state, market_df, confirmed_trades_df, shadow_trades_df, decisions_df, target_state, overlay_toggles, full_chart: bool = False) -> go.Figure:
    product_id = str(chart_meta.get("product_id") or coin_state.get("product_id") or "")
    has_volume = bool(chart_meta.get("has_volume")) and bool(overlay_toggles.get("volume", True))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.76, 0.24])
    overlay_count = 0
    if not chart_df.empty:
        fig.add_trace(go.Candlestick(x=chart_df["dt"], open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=chart_df["close"], name="OHLC"), row=1, col=1)
        if has_volume:
            fig.add_trace(go.Bar(x=chart_df["dt"], y=pd.to_numeric(chart_df["volume"], errors="coerce").fillna(0), name="Volume", marker_color="rgba(86,139,211,0.35)"), row=2, col=1); overlay_count += 1
    def hline(label, y, color, dash="dash"):
        nonlocal overlay_count
        y=_safe_float(y)
        if y>0: fig.add_hline(y=y, line_width=1.2, line_color=color, line_dash=dash, annotation_text=label, annotation_position="right", row=1, col=1); overlay_count += 1
    if overlay_toggles.get("profile", True):
        hline("POC", _first_price(target_state.get("point_of_control"), coin_state.get("point_of_control")), "#ff9f5a"); hline("VAH", _first_price(target_state.get("value_area_high"), coin_state.get("value_area_high")), "#e8c16f", "dot"); hline("VAL", _first_price(target_state.get("value_area_low"), coin_state.get("value_area_low")), "#e8c16f", "dot")
    if overlay_toggles.get("prior_profile", True):
        hline("Prior POC", coin_state.get("previous_session_profile_poc"), "#facc15"); hline("Prior VAH", coin_state.get("previous_session_profile_vah"), "#fde68a", "dot"); hline("Prior VAL", coin_state.get("previous_session_profile_val"), "#fde68a", "dot")
    if overlay_toggles.get("average_entry", True): hline("Avg Entry", _first_price(coin_state.get("avg_entry"), target_state.get("avg_entry_price")), "#a5dcff", "solid")
    if overlay_toggles.get("targets", True):
        hline("Target Buy", _first_price(target_state.get("target_buy_price"), coin_state.get("selected_target_buy_price")), "#78d6a8"); hline("Target Sell", _first_price(target_state.get("target_sell_price"), target_state.get("scalp_target_price"), target_state.get("core_target_price"), coin_state.get("selected_target_sell_price")), "#ff8e8e"); hline("Stop", _first_price(target_state.get("target_stop_price"), coin_state.get("selected_target_stop_price")), "#d46cff", "dot"); hline("Min Profitable Exit", _first_price(target_state.get("min_profitable_exit_price"), coin_state.get("min_profitable_exit_price")), "#facc15")
    if overlay_toggles.get("structure", False):
        for key in ["validated_high","validated_low","zone_low","zone_high","bullish_fvg_low","bullish_fvg_high","bearish_fvg_low","bearish_fvg_high"]: hline(key, coin_state.get(key), "rgba(255,255,255,0.35)", "dot")
    if overlay_toggles.get("vwap", True) and not market_df.empty and "anchored_vwap" in market_df.columns and "product_id" in market_df.columns:
        m=market_df[market_df["product_id"].astype(str)==product_id].copy()
        if not m.empty and "ts" in m.columns:
            m["dt"]=pd.to_datetime(pd.to_numeric(m["ts"],errors="coerce"),unit="s",errors="coerce",utc=True); m["anchored_vwap"]=pd.to_numeric(m["anchored_vwap"],errors="coerce"); m=m.dropna(subset=["dt","anchored_vwap"])
            if not m.empty: fig.add_trace(go.Scatter(x=m["dt"], y=m["anchored_vwap"], mode="lines", name="Anchored VWAP", line=dict(color="#9b87f5", width=1.5)), row=1, col=1); overlay_count += 1
    for df, name, marker in [(confirmed_trades_df, "Confirmed", "triangle-up"), (shadow_trades_df, "Shadow", "circle")]:
        if overlay_toggles.get("confirmed_trades" if name=="Confirmed" else "shadow_trades", True):
            t=_marker_df(df, product_id, chart_df)
            if not t.empty:
                color = "#78d6a8" if name == "Confirmed" else "#ffd37c"
                fig.add_trace(go.Scatter(x=t["dt"], y=t["price"], mode="markers", name=f"{name} Trades", marker=dict(symbol=marker, size=9, color=color)), row=1, col=1); overlay_count += 1
    latest_decision_id=""
    if overlay_toggles.get("level8_markers", True):
        d=_marker_df(decisions_df, product_id, chart_df)
        if not d.empty:
            latest_decision_id=str(d.iloc[-1].get("decision_id", "")); actions=d["action"] if "action" in d.columns else ["L8"]*len(d)
            fig.add_trace(go.Scatter(x=d["dt"], y=d["price"], mode="markers+text", text=actions, textposition="top center", name="Level 8", marker=dict(symbol="diamond", size=9, color="#38bdf8")), row=1, col=1); overlay_count += 1
    chart_height = 900 if full_chart else 620
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font=dict(color="#d7dde8"), legend=dict(orientation="h", y=1.02), margin=dict(l=15,r=15,t=45,b=15), height=chart_height, xaxis_rangeslider_visible=False, uirevision=f"{chart_meta.get('product_id', '')}-{chart_meta.get('timeframe', '')}")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)"); fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    module_debug(MODULE_NAME, "chart_debug", data={"selected_coin": product_id, "active_timeframe": chart_meta.get("timeframe"), "chart_source_file": chart_meta.get("source"), "selected_candle_rows": int(len(chart_df)), "selected_volume_rows": int(len(chart_df)) if has_volume else 0, "volume_available": bool(chart_meta.get("has_volume")), "overlay_toggles": overlay_toggles, "overlay_count": overlay_count, "trace_count": len(fig.data), "chart_data_age": chart_meta.get("age_sec"), "latest_decision_id": latest_decision_id}, level="INFO", also_overall=False)
    return fig


def latest_council_votes_for_coin(council_votes_df, decisions_df, product_id):
    """
    Return the latest usable Level 8 decision and the matching council vote rows.

    Heartbeat / COMMENTARY rows can have blank or NaN decision_id values.
    If "nan" is treated as a real decision_id, matching fails and callers can
    accidentally render every historical vote for the product.
    """
    latest_decision_id = ""
    latest_row = {}

    def valid_decision_id(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none", "null", "nat"}:
            return ""
        return text

    try:
        decisions = (
            decisions_df[decisions_df["product_id"].astype(str) == str(product_id)].copy()
            if decisions_df is not None
            and not decisions_df.empty
            and "product_id" in decisions_df.columns
            else pd.DataFrame()
        )

        votes = (
            council_votes_df[council_votes_df["product_id"].astype(str) == str(product_id)].copy()
            if council_votes_df is not None
            and not council_votes_df.empty
            and "product_id" in council_votes_df.columns
            else pd.DataFrame()
        )

        if not decisions.empty:
            decisions["ts_num"] = pd.to_numeric(decisions.get("ts"), errors="coerce")
            decisions = decisions.sort_values("ts_num")
            usable_decisions = decisions[
                decisions.get("decision_id", pd.Series("", index=decisions.index))
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": "", "none": "", "null": ""})
                .ne("")
            ].copy()

            latest_row = (usable_decisions.iloc[-1] if not usable_decisions.empty else decisions.iloc[-1]).to_dict()
            latest_decision_id = valid_decision_id(latest_row.get("decision_id", ""))

        if votes.empty:
            return latest_decision_id, latest_row, pd.DataFrame()

        if "ts" in votes.columns:
            votes["ts_num"] = pd.to_numeric(votes.get("ts"), errors="coerce")
            votes = votes.sort_values("ts_num")

        matched = pd.DataFrame()

        if latest_decision_id and "decision_id" in votes.columns:
            matched = votes[votes["decision_id"].astype(str) == latest_decision_id].copy()

        if matched.empty and "decision_id" in votes.columns:
            clean_vote_ids = votes["decision_id"].astype(str).str.strip()
            usable_votes = votes[
                clean_vote_ids.ne("")
                & clean_vote_ids.str.lower().ne("nan")
                & clean_vote_ids.str.lower().ne("none")
                & clean_vote_ids.str.lower().ne("null")
            ].copy()

            if not usable_votes.empty:
                latest_decision_id = str(usable_votes.iloc[-1].get("decision_id", "")).strip()
                matched = usable_votes[usable_votes["decision_id"].astype(str) == latest_decision_id].copy()

        if matched.empty:
            matched = votes.tail(25).copy()

        sort_cols = [c for c in ["leaderboard_rank", "agent"] if c in matched.columns]
        if sort_cols:
            matched = matched.sort_values(sort_cols)

        module_debug(
            MODULE_NAME,
            "latest_council_votes_selected",
            data={
                "product_id": product_id,
                "decision_id": latest_decision_id,
                "vote_rows": int(len(matched)),
                "unique_agents": int(matched["agent"].nunique()) if "agent" in matched.columns else 0,
            },
            level="DEBUG",
            also_overall=False,
        )

        return latest_decision_id, latest_row, matched

    except Exception as exc:
        module_exception(
            MODULE_NAME,
            "latest_council_votes_for_coin_failed",
            exc,
            data={"product_id": product_id, "traceback": traceback.format_exc()},
            also_overall=True,
        )
        return latest_decision_id, latest_row, pd.DataFrame()

def vote_leaning(row):
    scores={"BUY":_safe_float(row.get("adjusted_buy_score", row.get("buy_score"))),"SELL":_safe_float(row.get("adjusted_sell_score", row.get("sell_score"))),"HOLD":_safe_float(row.get("adjusted_hold_score", row.get("hold_score"))),"WAIT":_safe_float(row.get("adjusted_wait_score", row.get("wait_score")))}
    return max(scores, key=scores.get)


def strongest_vote_score(row): return max(_safe_float(row.get(k)) for k in ["adjusted_buy_score","adjusted_sell_score","adjusted_hold_score","adjusted_wait_score","buy_score","sell_score","hold_score","wait_score"])

def agent_title_icon(agent): return AGENT_TITLES.get(str(agent), AGENT_TITLES["fallback"])


def plain_reason(reason: Any) -> str:
    r=str(reason or ""); low=r.lower(); parts=[]
    mapping=[("entry_mode=no_position","The bot is evaluating a new entry because it does not currently hold this coin."),("sell_score","Sell pressure is being used as a warning against buying, not necessarily as an instruction to sell."),("map_to_avoid_buy","That signal maps to avoiding a new buy right now."),("ev_below_calibrated_target","Expected value is below the coin’s calibrated target."),("score_below_calibrated_target","The live score is below the coin’s calibrated buy target."),("inside_value_area","Price is inside the value area, so this analyst is cautious about chasing."),("near_poc","Price is near the point of control, where chop is more likely."),("low_volume_node","Price may move faster through a low-volume area."),("stale_market_data","The bot does not trust the quote age enough for live trading."),("expected_utility_too_low","The projected reward is not strong enough after costs."),("probability_below","The probability model wants stronger odds before entry."),("spread","Spread and fees are reducing the net edge."),("fee","Spread and fees are reducing the net edge.")]
    for key, text in mapping:
        if key in low and text not in parts: parts.append(text)
    return " ".join(parts) or (r[:220] if r else "No reason text was published for this row.")




def agent_plain_summary(row: dict) -> str:
    reason = str(row.get("reason", "") or "")
    low = reason.lower()
    leaning = vote_leaning(row)
    summaries = []
    if "entry_mode=no_position" in low or "no position" in low:
        summaries.append("This analyst is evaluating a fresh entry because the bot does not currently hold this coin.")
    if "sell_score" in low and "map_to_avoid_buy" in low:
        summaries.append("The sell pressure is being interpreted as a reason to avoid buying right now, not as an instruction to sell.")
    if "expected_utility_too_low" in low:
        summaries.append("The setup may look active, but the projected reward is not strong enough after fees, spread, uncertainty, and context penalties.")
    if "inside_value" in low or "inside_value_area" in low:
        summaries.append("Price is inside the main value area, where chop is more likely and breakout trades need stronger confirmation.")
    if "near_poc" in low or "poc_distance" in low:
        summaries.append("Price is near the point of control, which often means the market is balanced instead of clearly trending.")
    if "high_volume_node" in low:
        summaries.append("Price is sitting in a dense volume zone, so movement may be slower and harder to scalp cleanly.")
    if "low_volume_node" in low or "lvn" in low:
        summaries.append("A nearby low-volume area may create a faster move if price breaks into it cleanly.")
    if "score_below_calibrated_target" in low:
        summaries.append("The live score is below this coin’s calibrated buy target.")
    if "probability_below_calibrated_target" in low:
        summaries.append("The probability model wants stronger odds before risking live money.")
    if "ev_below_calibrated_target" in low:
        summaries.append("The expected value is below the calibrated target for this coin.")
    if "spread" in low:
        summaries.append("Spread and execution cost are reducing the quality of the entry.")
    if "maker_adjusted_ev_too_low" in low:
        summaries.append("The maker-adjusted edge is too low, so even a limit-order-first entry does not look attractive enough.")
    if "buy_vs_wait" in low:
        summaries.append("The bot thinks waiting may be better than buying immediately.")
    if not summaries:
        if leaning == "BUY": summaries.append("This analyst sees enough evidence to support a buy, but Level 8 still weighs it against fees, utility, and other agents.")
        elif leaning == "SELL": summaries.append("This analyst sees downside or exit pressure and is not supporting a new buy.")
        elif leaning == "HOLD": summaries.append("This analyst sees a reason to stay patient rather than force an entry.")
        else: summaries.append("This analyst is waiting for cleaner confirmation before supporting live execution.")
    return " ".join(summaries[:2])



def agent_full_plain_summary(row: dict) -> str:
    """Full paragraph version for the selected agent dialogue panel."""
    agent = str(row.get("agent", "fallback"))
    reason = str(row.get("reason", "") or "")
    low = reason.lower()
    leaning = vote_leaning(row)
    opening = (f"{agent_title_icon(agent)} is currently leaning {leaning}. "
        f"Its adjusted scores are buy {_safe_float(row.get('adjusted_buy_score')):.3f}, sell {_safe_float(row.get('adjusted_sell_score')):.3f}, "
        f"hold {_safe_float(row.get('adjusted_hold_score')):.3f}, and wait {_safe_float(row.get('adjusted_wait_score')):.3f}, with confidence {_safe_float(row.get('confidence')):.3f}. ")
    observations = []
    patterns = [
        (("entry_mode=no_position", "no position", "owns_position=false"), "It is judging this as a fresh-entry situation because the bot does not currently hold the coin, so sell pressure is being interpreted as a reason to avoid a new buy rather than as an instruction to exit an existing position."),
        (("expected_utility_too_low", "expected_utility="), "The most important issue is expected utility: after Binance.US fees, spread, uncertainty, wait utility, and context penalties, the setup does not appear to offer enough net reward for live execution."),
        (("maker_adjusted_ev_too_low", "maker_ev"), "Even if the bot tries to enter with maker-style execution, the maker-adjusted edge is still weak, so the analyst is not convinced that the trade can overcome costs cleanly."),
        (("inside_value_area", "inside_value"), "Price appears to be inside the main value area, which often means the market is balanced or choppy rather than clearly trending. That makes a new buy less attractive unless price breaks and accepts outside value."),
        (("near_poc", "poc_distance"), "Price is near the point of control, where a lot of trading has already happened. That can make the market more likely to chop around instead of moving cleanly toward a target."),
        (("above_value_area", "accepted_above"), "There is some bullish context because price is above or trying to accept above value. The analyst still wants confirmation that this is acceptance, not a quick rejection back into the prior range."),
        (("below_value_area", "accepted_lower"), "The value-area context is weaker because price is below or accepting lower prices, which can make a buy premature unless price reclaims value with strength."),
        (("sweep_reclaim", "swept", "reclaimed"), "The analyst is watching for a sweep-and-reclaim style setup, where price takes liquidity and then reclaims a level. That can be useful, but it needs cleaner follow-through before it becomes a strong live entry."),
        (("upper_rejection=true",), "The candle structure shows upper-wick rejection, which means buyers pushed price up but sellers pushed it back down. That weakens the immediate buy case."),
        (("lower_rejection=true",), "The candle structure shows lower-wick rejection, which can mean buyers defended lower prices. That can support a buy only if the rest of the setup also confirms."),
        (("bullish_fvg",), "There is a bullish fair-value-gap context on the chart. The bot is watching whether price respects that imbalance or fails back through it."),
        (("bearish_fvg",), "There is a bearish fair-value-gap context on the chart. That can create overhead pressure unless price breaks through it convincingly."),
        (("volume_profile_unavailable",), "The volume-profile data for this specific decision is weak or unavailable, so this analyst has less confidence in value-area levels for the current coin."),
        (("low_volume_node", "lvn"), "A nearby low-volume area may create a faster move if price breaks into it cleanly, but the bot still needs the setup to clear fees and execution cost."),
        (("high_volume_node",), "A high-volume node can slow price down because many trades have already occurred there, so the bot is cautious about expecting a clean scalp through that zone."),
        (("walk_forward",), "The walk-forward validation layer is not strongly supportive yet. That does not automatically mean the setup is bad, but it reduces confidence because similar historical examples have not proven themselves cleanly."),
        (("buy_vs_wait",), "The buy-versus-wait calculation favors patience. In plain English, the bot believes waiting may offer a better risk/reward than buying immediately."),
        (("probability_below", "calibrated_p_win"), "The calibrated probability model is not giving enough edge yet. The analyst wants stronger odds before supporting a live trade."),
        (("payoff",), "The payoff ratio is part of the hesitation: the expected win is not large enough compared with the expected loss and cost burden."),
    ]
    for keys, text in patterns:
        if any(key in low for key in keys): observations.append(text)
    if not observations:
        observations.append("The raw reason does not map to a known explanation pattern yet. The safest interpretation is that this analyst is weighing the chart, score, confidence, and Level 8 context before supporting live execution.")
    closing = f"Overall, this analyst is saying {leaning} because it wants a cleaner alignment between chart structure, volume context, probability, and fee-adjusted profitability before Level 8 risks live money."
    return opening + " ".join(observations[:5]) + " " + closing

def latest_decision_for_product(decisions_df: pd.DataFrame, product_id: str) -> dict:
    try:
        if decisions_df is None or decisions_df.empty or "product_id" not in decisions_df.columns:
            return {}
        sub = decisions_df[decisions_df["product_id"].astype(str).eq(str(product_id))].copy()
        if sub.empty:
            return {}
        if "ts" in sub.columns:
            sub["ts_num"] = pd.to_numeric(sub["ts"], errors="coerce")
            sub = sub.sort_values("ts_num")
        return sub.iloc[-1].to_dict()
    except Exception:
        return {}


def latest_volume_oracle_vote(council_votes_df: pd.DataFrame, decisions_df: pd.DataFrame, product_id: str) -> dict:
    try:
        _, _, votes = latest_council_votes_for_coin(council_votes_df, decisions_df, product_id)
        if votes.empty or "agent" not in votes.columns:
            return {}
        sub = votes[votes["agent"].astype(str).eq("volume_profile_leader")].copy()
        if sub.empty:
            return {}
        if "ts" in sub.columns:
            sub["ts_num"] = pd.to_numeric(sub["ts"], errors="coerce")
            sub = sub.sort_values("ts_num")
        return sub.iloc[-1].to_dict()
    except Exception:
        return {}


def signal_core_paragraph(decision: dict, top_row: dict) -> str:
    action = str(decision.get("action") or top_row.get("action") or "WAIT").upper()
    product_id = str(decision.get("product_id") or top_row.get("product_id") or "the top candidate")
    expected_utility = _safe_float(decision.get("expected_utility_bps") or top_row.get("expected_utility_bps"))
    score = _safe_float(decision.get("final_buy_score") or top_row.get("final_buy_score"))
    threshold = _safe_float(decision.get("buy_threshold") or top_row.get("buy_threshold"))
    reason = plain_reason(decision.get("reason") or top_row.get("blocker"))
    if action == "BUY":
        verdict = f"The Signal Core is allowing {product_id} to move toward live execution because the candidate has cleared the highest-level strategy checks."
    elif action == "SHADOW":
        verdict = f"The Signal Core is shadowing {product_id}. It sees enough structure to study the setup, but not enough net edge to risk live money yet."
    elif action == "COMMENTARY":
        verdict = f"The Signal Core is treating {product_id} as commentary. The setup is useful for learning, but it is not close enough to a live entry."
    else:
        verdict = f"The Signal Core is waiting on {product_id}. It is not seeing a strong enough combination of score, utility, and execution quality."
    return f"{verdict} The current score is {score:.3f} against a threshold of {threshold:.3f}, and expected utility is {expected_utility:.2f} bps. In plain English, its decision is being driven by whether the trade can become net-profitable after fees, spread, wait utility, and context penalties. {reason}"


def signal_core_learning_paragraph(decision: dict, top_row: dict) -> str:
    product_id = str(decision.get("product_id") or top_row.get("product_id") or "this product")
    reason_blob = str(decision.get("reason") or top_row.get("blocker") or "").lower()
    priorities = []
    if "expected_utility_too_low" in reason_blob:
        priorities.append("it is learning whether the utility penalty is too harsh or correctly avoiding fee-negative trades")
    if "inside_value" in reason_blob or "near_poc" in reason_blob:
        priorities.append("it is studying whether value-area chop should keep blocking entries")
    if "buy_vs_wait" in reason_blob:
        priorities.append("it is comparing immediate buys against the value of waiting")
    if "adaptive_waiting_for_reviews" in reason_blob:
        priorities.append("it is waiting for more reviewed outcomes before trusting adaptive thresholds")
    if "historical" in reason_blob or "replay" in reason_blob:
        priorities.append("it is using replay outcomes to decide whether this setup class is actually profitable")
    if not priorities:
        priorities.append("it is learning which analysts consistently predict net-positive outcomes after the sell model is applied")
    return f"Right now, The Signal Core is trying to learn whether {product_id} belongs in a profitable setup family. It is prioritizing the analysts whose signals affect net P/L the most: utility, volume/profile location, order-book execution quality, price action, and historical replay. At this moment, " + "; ".join(priorities[:3]) + "."


def volume_oracle_paragraph(vote: dict, product_id: str) -> str:
    if not vote:
        return f"The Volume Oracle does not have a fresh vote for {product_id} yet. Once the council vote rows update, this panel will explain value area, POC, VAH, VAL, and volume-node context."
    leaning = vote_leaning(vote)
    confidence = _safe_float(vote.get("confidence"))
    buy_score = _safe_float(vote.get("adjusted_buy_score"))
    sell_score = _safe_float(vote.get("adjusted_sell_score"))
    wait_score = _safe_float(vote.get("adjusted_wait_score"))
    return f"The Volume Oracle is leaning {leaning} on {product_id} with confidence {confidence:.3f}. Its buy score is {buy_score:.3f}, sell score is {sell_score:.3f}, and wait score is {wait_score:.3f}. This verdict is mainly about whether price is in a clean location or trapped in value-area chop near POC. {agent_full_plain_summary(vote)}"


def volume_oracle_learning_paragraph(vote: dict, product_id: str) -> str:
    reason_blob = str(vote.get("reason") or "").lower() if vote else ""
    focus = []
    if "inside_value" in reason_blob:
        focus.append("inside-value trades are being treated cautiously because they often chop")
    if "near_poc" in reason_blob:
        focus.append("POC proximity is being monitored as a balance/chop warning")
    if "high_volume_node" in reason_blob:
        focus.append("high-volume nodes are being studied as places where moves can stall")
    if "low_volume_node" in reason_blob:
        focus.append("low-volume paths are being studied as possible fast-move zones")
    if "accepted_above" in reason_blob or "reclaimed_value" in reason_blob:
        focus.append("acceptance and reclaim behavior are being watched for cleaner directional setups")
    if not focus:
        focus.append("it is learning whether value-area position improves or weakens final net trade outcomes")
    return f"The Volume Oracle is currently learning how {product_id}'s value-area location affects the bot's final profit. It is prioritizing POC distance, VAH/VAL acceptance, high-volume-node resistance, low-volume-node opportunity, and whether price is accepting or rejecting value. Current focus: " + "; ".join(focus[:3]) + "."


def render_leadership_verdicts(rows: list[dict], decisions_df: pd.DataFrame, council_votes_df: pd.DataFrame):
    if not rows:
        return
    top_row = rows[0]
    product_id = str(top_row.get("product_id") or "")
    decision = latest_decision_for_product(decisions_df, product_id)
    action = str(decision.get("action") or top_row.get("action") or "WAIT").upper()
    expected_utility = _safe_float(decision.get("expected_utility_bps") or top_row.get("expected_utility_bps"))
    st.markdown(
        f'''<div class="leadership-grid"><div class="leadership-card"><div class="leadership-title">{_html(LEAD_DECISION_MAKER_TITLE)}</div><div class="leadership-subtitle">{_html(LEAD_DECISION_MAKER_SUBTITLE)} · {_html(product_id)}</div><div class="leadership-verdict">Current verdict: {_html(action)} · Utility {expected_utility:.2f} bps</div><div class="leadership-paragraph">{_html(signal_core_paragraph(decision, top_row))}</div><div class="leadership-learning"><b>Current learning:</b> {_html(signal_core_learning_paragraph(decision, top_row))}</div></div></div>''',
        unsafe_allow_html=True,
    )

def set_selected_coin(product_id: str) -> None:
    product_id = str(product_id or "").strip()
    if not product_id:
        return

    st.session_state["selected_coin"] = product_id
    st.session_state["strategy_arena_coin"] = product_id
    st.session_state["_scroll_to_strategy_arena"] = True

    module_debug(
        MODULE_NAME,
        "coin_card_selected",
        data={"product_id": product_id},
        level="INFO",
        also_overall=False,
    )


def set_selected_agent(agent_name: str) -> None:
    agent_name = str(agent_name or "").strip()
    if not agent_name:
        return

    st.session_state["selected_agent_dialogue"] = agent_name
    st.session_state["_scroll_to_agent_dialogue"] = True

    module_debug(
        MODULE_NAME,
        "agent_card_selected",
        data={"agent": agent_name},
        level="INFO",
        also_overall=False,
    )


def scroll_to_anchor_if_requested(flag_key: str, anchor_id: str) -> None:
    """
    Scrolls inside the current Streamlit page after a native Streamlit click.
    This does not use href links, javascript:void links, or query params.
    """
    if not st.session_state.pop(flag_key, False):
        return

    safe_anchor = str(anchor_id).replace('"', '').replace("'", "")

    components.html(
        f"""
        <script>
        const parentWindow = window.parent || window;
        const doc = parentWindow.document;
        setTimeout(function() {{
            const el = doc.getElementById("{safe_anchor}");
            if (el) {{
                el.scrollIntoView({{behavior: "smooth", block: "start"}});
            }}
        }}, 150);
        </script>
        """,
        height=0,
    )


def scroll_to_strategy_arena_if_requested() -> None:
    scroll_to_anchor_if_requested("_scroll_to_strategy_arena", "strategy-arena-anchor")


def scroll_to_agent_dialogue_if_requested() -> None:
    scroll_to_anchor_if_requested("_scroll_to_agent_dialogue", "agent-dialogue-anchor")


def render_agent_disagreement_summary(votes: pd.DataFrame) -> Dict[str, Any]:
    if votes.empty: return {"BUY":0,"SELL":0,"HOLD":0,"WAIT":0,"consensus":"WAIT","main_blocker":"No agent votes yet"}
    rows = votes.to_dict("records"); leanings=[vote_leaning(r) for r in rows]
    counts={k: leanings.count(k) for k in ["BUY","SELL","HOLD","WAIT"]}
    avg_conf=sum(_safe_float(r.get("confidence")) for r in rows)/max(1,len(rows))
    buy_rows=[r for r in rows if vote_leaning(r)=="BUY"]; blockers=[r for r in rows if vote_leaning(r)!="BUY"]
    best_buy=max(buy_rows, key=lambda r:_safe_float(r.get("confidence")), default={})
    best_block=max(blockers, key=lambda r:_safe_float(r.get("confidence")), default={})
    consensus=max(counts, key=counts.get)
    summary={"BUY":counts["BUY"],"SELL":counts["SELL"],"HOLD":counts["HOLD"],"WAIT":counts["WAIT"],"highest_confidence_buy_agent":agent_title_icon(best_buy.get("agent", "")) if best_buy else "—","highest_confidence_blocker":agent_title_icon(best_block.get("agent", "")) if best_block else "—","average_confidence":avg_conf,"consensus":consensus,"main_blocker":plain_reason(best_block.get("reason", "")) if best_block else "No blocker published"}
    st.markdown(f'<div class="codex-panel"><b>Agent Consensus</b><br><span class="muted">BUY {counts["BUY"]} · SELL {counts["SELL"]} · HOLD {counts["HOLD"]} · WAIT {counts["WAIT"]} · Avg confidence {avg_conf:.3f}</span><br><b>Consensus:</b> {consensus}<br><b>Main blocker:</b> {_html(summary["main_blocker"])}</div>', unsafe_allow_html=True)
    return summary


def render_status_strip(snapshot, selected, coin, chart_meta, votes, drow, market_df, refresh_config):
    now=time.time(); snapshot_age=max(0.0, now-_safe_float(snapshot.get("updated_ts"))) if _safe_float(snapshot.get("updated_ts"))>0 else 999999.0
    chart_age=float(chart_meta.get("age_sec",999999.0)); council_age=dataframe_latest_age_sec(votes); readiness=snapshot.get("readiness",{}) or {}
    live_mode = "Live" if readiness.get("live_trading_enabled", coin.get("live_trading_enabled", False)) else "Shadow"
    action = drow.get("action", coin.get("decision_action", "WAIT")) if isinstance(drow, dict) else coin.get("decision_action", "WAIT")
    maker=_safe_float(coin.get("maker_fee_rate", coin.get("maker_fee_bps", 0.0))); taker=_safe_float(coin.get("taker_fee_rate", coin.get("taker_fee_bps", 0.0)))
    blocker=str(coin.get("main_blocker", coin.get("buy_blocker", drow.get("reason", "—") if isinstance(drow,dict) else "—")))[:80]
    refresh_mode = "fragment data refresh" if refresh_config.get("fragment_supported") and refresh_config.get("live_enabled") else "manual only until Streamlit is updated" if not refresh_config.get("fragment_supported") else "manual"
    pills=[("Selected",selected,"good"),("Action",action,"good" if str(action).upper()=="BUY" else "warn"),("Mode",live_mode,"good" if live_mode=="Live" else "warn"),("Snapshot",format_age(snapshot_age),freshness_class(snapshot_age,8, SNAPSHOT_STALE_WARN_SEC)),("Chart",format_age(chart_age),freshness_class(chart_age,60, CHART_STALE_WARN_SEC_WEEK)),("Council",format_age(council_age),freshness_class(council_age,20, COUNCIL_STALE_WARN_SEC)),("Fees",f"M {maker:.4g} / T {taker:.4g}","warn" if coin.get("high_fee_tier_active") else "good"),("Readiness",str(readiness.get("state", coin.get("trade_readiness", "—"))),"good" if str(readiness.get("state","")).lower() in {"ready","live"} else "warn"),("Blocker",blocker,"danger" if blocker and blocker != "—" else "good"),("Refresh",refresh_mode,"good" if "fragment" in refresh_mode else "warn")]
    html='<div class="status-strip">' + ''.join(f'<div class="status-pill"><div class="pill-label">{_html(l)}</div><div class="pill-value {_html(cls)}">{_html(v)}</div></div>' for l,v,cls in pills) + '</div>'
    st.markdown(html, unsafe_allow_html=True)
    if coin.get("high_fee_tier_active"): st.warning("Profit-First Fee-Aware Mode is active. The bot can trade when projected net profit clears fees, spread, and execution cost; it is not observe-only mode.")
    return snapshot_age, chart_age, council_age, refresh_mode


def render_targets_panel(coin: Dict[str, Any], target: Dict[str, Any]) -> None:
    st.markdown('### Targets / Sell Plan')
    cols=st.columns(4); cols[0].metric("Current Price", f"{_safe_float(coin.get('price')):.8f}"); cols[1].metric("Buy Target", f"{_safe_float(target.get('target_buy_price', coin.get('selected_target_buy_price'))):.8f}"); cols[2].metric("Sell Target", f"{_safe_float(target.get('target_sell_price', coin.get('selected_target_sell_price'))):.8f}"); cols[3].metric("Stop", f"{_safe_float(target.get('target_stop_price', coin.get('selected_target_stop_price'))):.8f}")


def render_confirmed_trades(trades: pd.DataFrame) -> None:
    st.markdown('### Confirmed Trades Only')
    if trades.empty: st.info("No confirmed trades yet."); return
    cols=[c for c in ["ts","product_id","side","price","qty","size","fee","fee_usd","order_id"] if c in trades.columns]
    st.dataframe(trades[cols] if cols else trades, width="stretch", hide_index=True)


def render_coin_analytics(coin: Dict[str, Any]) -> None:
    st.markdown('### Selected-Coin Analytics')
    cols=st.columns(4); metrics=[("Truth","truth_score"),("Final Buy","final_buy_score"),("Expected Utility bps","expected_utility_bps"),("Buy vs Wait bps","buy_vs_wait_edge_bps")]
    for c,(label,key) in zip(cols,metrics): c.metric(label, f"{_safe_float(coin.get(key)):.3f}")


def viewer_runtime_audit(*, snapshot, selected, market_df, trades_df, targets_df, decisions_df, council_votes_df, orders_df, walk_forward_df, agent_ablation_df, ai_importance_df, chart_meta=None, latest_votes=None, refresh_config=None) -> Dict[str, Any]:
    chart_meta=chart_meta or {}; latest_votes=latest_votes if latest_votes is not None else pd.DataFrame(); refresh_config=refresh_config or {}
    coins=dict(snapshot.get("coins",{}) or {}); coin=dict(coins.get(selected,{}) or {})
    selected_market_rows=0 if market_df.empty or "product_id" not in market_df.columns else int((market_df["product_id"].astype(str)==str(selected)).sum())
    missing_files=[]; empty_files=[]
    normal_messages={POSITION_TARGETS_PATH:"No position targets file yet. This is normal until the bot publishes target rows.", AI_FEATURE_IMPORTANCE_PATH:"AI feature importance pending until enough labeled training rows exist.", TRADES_CSV_PATH:"No confirmed trades yet.", ORDERS_CSV_PATH:"No backend order attempts yet.", AGENT_ABLATION_PATH:"Agent ablation is waiting for enough reviewed outcomes.", WALK_FORWARD_VALIDATION_PATH:"Walk-forward validation is waiting for enough reviewed outcomes."}
    for p in [POSITION_TARGETS_PATH, AI_FEATURE_IMPORTANCE_PATH, TRADES_CSV_PATH, ORDERS_CSV_PATH, AGENT_ABLATION_PATH, WALK_FORWARD_VALIDATION_PATH]:
        sig=file_signature(p)
        if not sig[1]: missing_files.append({"file":os.path.basename(p),"message":normal_messages[p]})
        elif sig[2] == 0: empty_files.append({"file":os.path.basename(p),"message":normal_messages[p]})
    latest_decision_id, _, _ = latest_council_votes_for_coin(council_votes_df, decisions_df, selected)
    health={"selected_coin":selected,"snapshot_age_sec":max(0.0,time.time()-_safe_float(snapshot.get("updated_ts"))) if _safe_float(snapshot.get("updated_ts"))>0 else 999999.0,"chart_source":chart_meta.get("source"),"chart_row_count":chart_meta.get("rows",0),"chart_volume_availability":chart_meta.get("has_volume",False),"latest_decision_id":latest_decision_id,"latest_agent_vote_count":int(len(latest_votes)),"selected_market_rows":selected_market_rows,"stale_market_data_blockers":"stale_market_data" in str(coin).lower(),"missing_files":missing_files,"empty_files":empty_files,"refresh_mode":"fragment data refresh" if refresh_config.get("fragment_supported") else "manual only until Streamlit is updated","fragment_support":bool(refresh_config.get("fragment_supported")),"full_page_refresh_disabled":True,"csv_audit_healthy":True,"message":"Viewer refresh mode: fragment data refresh" if refresh_config.get("fragment_supported") else "Viewer refresh mode: manual only until Streamlit is updated"}
    debug_every(MODULE_NAME, f"viewer_runtime_audit:{selected}", 10.0, "viewer_runtime_audit", data=health, level="INFO", also_overall=False)
    return health



def latest_row_for_product(df: pd.DataFrame, product_id: str) -> dict:
    if df.empty or "product_id" not in df.columns:
        return {}
    sub = df[df["product_id"].astype(str) == str(product_id)].copy()
    if sub.empty:
        return {}
    if "ts" in sub.columns:
        sub["ts_num"] = pd.to_numeric(sub["ts"], errors="coerce")
        sub = sub.sort_values("ts_num")
    return sub.iloc[-1].to_dict()


def boolish(value: Any) -> bool:
    if isinstance(value, bool): return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ready", "buy_ready"}


def calculate_coin_viability(row: dict) -> tuple[float, str]:
    action = str(row.get("action") or "").upper(); consensus = str(row.get("consensus") or "").upper()
    final_buy_score = _safe_float(row.get("final_buy_score")); buy_threshold = _safe_float(row.get("buy_threshold")); expected_utility_bps = _safe_float(row.get("expected_utility_bps")); buy_votes = _safe_float(row.get("buy_votes")); wait_votes = _safe_float(row.get("wait_votes")); spread_bps = _safe_float(row.get("spread_bps")); recommended_position_pct = _safe_float(row.get("recommended_position_pct"))
    buy_gate_tradeable = boolish(row.get("buy_gate_tradeable")); buy_gate_strict_ok = boolish(row.get("buy_gate_strict_ok")); buy_gate_spread_ok = boolish(row.get("buy_gate_spread_ok")); buy_gate_ev_ok = boolish(row.get("buy_gate_ev_ok")); buy_gate_score_ok = boolish(row.get("buy_gate_score_ok")); buy_gate_prob_ok = boolish(row.get("buy_gate_prob_ok"))
    viability = 0.0; reasons = []; score_margin = final_buy_score - buy_threshold
    if action == "BUY": viability += 25.0; reasons.append("final action is BUY")
    elif action == "SHADOW": viability += 12.0; reasons.append("shadow-buy candidate")
    elif action == "COMMENTARY": viability -= 5.0; reasons.append("commentary only")
    if consensus == "BUY": viability += 10.0; reasons.append("agent consensus leans BUY")
    elif consensus == "WAIT": viability -= 4.0; reasons.append("agent consensus leans WAIT")
    elif consensus == "SELL": viability -= 8.0; reasons.append("agent consensus leans SELL")
    viability += max(-15.0, min(25.0, score_margin * 45.0)) + max(-20.0, min(30.0, expected_utility_bps / 10.0)) + max(0.0, min(10.0, buy_votes * 0.8)) - max(0.0, min(10.0, wait_votes * 0.25)) + max(0.0, min(8.0, recommended_position_pct * 40.0))
    if buy_gate_tradeable: viability += 14.0; reasons.append("market telemetry says tradeable")
    if buy_gate_strict_ok: viability += 6.0
    if buy_gate_score_ok: viability += 4.0
    if buy_gate_prob_ok: viability += 4.0
    if buy_gate_ev_ok: viability += 6.0
    if buy_gate_spread_ok: viability += 4.0
    else: viability -= 8.0; reasons.append("spread gate is not clean")
    if spread_bps > 20: viability -= 10.0; reasons.append("wide spread")
    elif spread_bps > 10: viability -= 5.0; reasons.append("moderate spread")
    blocker = str(row.get("blocker") or "").lower()
    if "stale_market_data" in blocker: viability -= 18.0; reasons.append("stale top-of-book")
    if "expected_utility_too_low" in blocker: viability -= 16.0; reasons.append("utility too low")
    if "probability_too_low" in blocker or "probability_below" in blocker: viability -= 10.0; reasons.append("probability not high enough")
    if "score_below" in blocker: viability -= 8.0; reasons.append("score below calibrated target")
    if boolish(row.get("owns_position")): viability += 6.0; reasons.append("already held")
    return round(float(viability), 3), "; ".join(reasons[:4]) or "ranked by score, utility, gates, spread, and consensus"


def build_all_coin_rows(snapshot, market_df, decisions_df, council_votes_df, targets_df) -> list[dict]:
    coins = snapshot.get("coins", {}) or {}
    products = get_available_products(snapshot)

    for frame in [market_df, decisions_df, targets_df, council_votes_df]:
        if frame is not None and not frame.empty and "product_id" in frame.columns:
            for p in frame["product_id"].dropna().astype(str).unique().tolist():
                if p not in products:
                    products.append(p)

    rows = []
    for product in products:
        coin = dict(coins.get(product, {}) or {}); market = latest_row_for_product(market_df, product); decision = latest_row_for_product(decisions_df, product); target = latest_row_for_product(targets_df, product); latest_decision_id, _, votes = latest_council_votes_for_coin(council_votes_df, decisions_df, product)
        if not votes.empty:
            leanings = [vote_leaning(r) for r in votes.to_dict("records")]; consensus = max(["BUY", "SELL", "HOLD", "WAIT"], key=lambda x: leanings.count(x)); buy_votes = leanings.count("BUY"); wait_votes = leanings.count("WAIT"); sell_votes = leanings.count("SELL"); hold_votes = leanings.count("HOLD")
        else: consensus = "WAIT"; buy_votes = wait_votes = sell_votes = hold_votes = 0
        action = str(decision.get("action") or coin.get("decision_action") or market.get("buy_gate_tradeable") or "WAIT").upper()
        blocker = str(market.get("buy_gate_blocker") or coin.get("main_blocker") or coin.get("buy_blocker") or decision.get("reason") or decision.get("main_reason") or "")
        row = {"product_id": product, "price": _safe_float(market.get("mid") or coin.get("price")), "spread_bps": _safe_float(market.get("spread_bps") or coin.get("spread_bps")), "action": action, "consensus": consensus, "buy_votes": buy_votes, "sell_votes": sell_votes, "hold_votes": hold_votes, "wait_votes": wait_votes, "owns_position": boolish(coin.get("owns_position") or target.get("has_position")), "final_buy_score": _safe_float(decision.get("final_buy_score") or coin.get("final_buy_score")), "buy_threshold": _safe_float(decision.get("buy_threshold") or coin.get("buy_threshold")), "expected_utility_bps": _safe_float(decision.get("expected_utility_bps") or coin.get("expected_utility_bps")), "recommended_position_pct": _safe_float(decision.get("recommended_position_pct") or coin.get("recommended_position_pct")), "buy_gate_tradeable": market.get("buy_gate_tradeable"), "buy_gate_strict_ok": market.get("buy_gate_strict_ok"), "buy_gate_spread_ok": market.get("buy_gate_spread_ok"), "buy_gate_ev_ok": market.get("buy_gate_ev_ok"), "buy_gate_score_ok": market.get("buy_gate_score_ok"), "buy_gate_prob_ok": market.get("buy_gate_prob_ok"), "blocker": plain_reason(blocker), "decision_id": latest_decision_id, "historical_replay_ready": boolish(coin.get("historical_replay_ready")), "historical_replay_rows": _safe_float(coin.get("historical_replay_rows")), "calibration_source": str(coin.get("calibration_source") or "unknown")}
        row["viability_score"], row["viability_reason"] = calculate_coin_viability(row); rows.append(row)
    rows.sort(key=lambda r: (_safe_float(r.get("viability_score")), boolish(r.get("buy_gate_tradeable")), _safe_float(r.get("expected_utility_bps")), _safe_float(r.get("final_buy_score")) - _safe_float(r.get("buy_threshold")), _safe_float(r.get("buy_votes"))), reverse=True)
    for idx, row in enumerate(rows, start=1): row["rank"] = idx
    return rows

def render_all_coin_landing_page(snapshot, market_df, decisions_df, council_votes_df, targets_df, trades_df, refresh_config):
    rows = build_all_coin_rows(snapshot, market_df, decisions_df, council_votes_df, targets_df)
    readiness = snapshot.get("readiness", {}) or {}
    updated_ts = _safe_float(snapshot.get("updated_ts"))
    age = max(0.0, time.time() - updated_ts) if updated_ts > 0 else 999999.0
    st.markdown('<div class="hud-header"><div class="hud-title"><span class="live-pulse"></span>All-Coin Command Deck</div><div class="hud-subtitle">One-glance live stance across every tracked Binance.US product.</div></div>', unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("Tracked Coins", len(rows))
    cols[1].metric("Top Candidate", rows[0]["product_id"] if rows else "None")
    cols[2].metric("Top Viability", f'{rows[0]["viability_score"]:.1f}' if rows else "0.0")
    cols[3].metric("BUY Actions", sum(1 for r in rows if r["action"] == "BUY"))
    cols[4].metric("Snapshot Age", format_age(age))
    if rows:
        st.caption(f'Continuously sorted by viability score. Current leader: {rows[0]["product_id"]} — {rows[0]["viability_reason"]}')
    render_leadership_verdicts(rows, decisions_df, council_votes_df)
    if readiness.get("high_fee_tier_active"):
        st.warning("Profit-First Fee-Aware Mode is active because Binance.US fees are high. The bot can still trade, but projected net profit must clear maker/taker fees, spread, and execution cost.")
    render_held_positions(snapshot, market_df, trades_df, targets_df)
    st.markdown('<div class="muted">Tap a coin card to open it in Strategy Arena.</div>', unsafe_allow_html=True)
    for i in range(0, len(rows), 3):
        cols = st.columns(3)
        for col, row in zip(cols, rows[i:i + 3]):
            product_id = str(row.get("product_id") or "")
            card_state = ("buy" if row.get("action") == "BUY" else "shadow" if "SHADOW" in str(row.get("action") or "") else "blocked" if row.get("blocker") else "wait")
            with col:
                st.markdown(f'''
                    <div class="coin-overview-card {card_state}">
                        <div style="font-size:1.25rem;font-weight:900;"><span class="rank-badge">#{row["rank"]}</span>{_html(product_id)}</div>
                        <div class="viability-score">Viability {row["viability_score"]:.1f}</div>
                        <div class="viability-reason">{_html(row["viability_reason"])}</div>
                        <div class="muted">Action: <b>{_html(row["action"])}</b> · Consensus: <b>{_html(row["consensus"])}</b></div>
                        <div>Votes: BUY <b>{row["buy_votes"]}</b> · WAIT <b>{row["wait_votes"]}</b> · SELL <b>{row["sell_votes"]}</b> · HOLD <b>{row["hold_votes"]}</b></div>
                        <div>Price: <b>{row["price"]:.8f}</b></div>
                        <div>Spread: <b>{row["spread_bps"]:.2f} bps</b></div>
                        <div>Buy score: <b>{row["final_buy_score"]:.3f}</b> / threshold <b>{row["buy_threshold"]:.3f}</b></div>
                        <div>Utility: <b>{row["expected_utility_bps"]:.2f} bps</b></div>
                        <div>Calibration: <b>{_html(row["calibration_source"])}</b></div>
                        <div>Replay rows: <b>{int(row["historical_replay_rows"])}</b></div>
                        <div class="muted">Blocker: {_html(row["blocker"][:160] or "No blocker published.")}</div>
                    </div>
                    ''', unsafe_allow_html=True)

                if st.button(
                    "Click here",
                    key=f"coin_card_click_{product_id}",
                    width="stretch",
                    type="secondary",
                ):
                    set_selected_coin(product_id)
    with st.expander("All-coin sortable table", expanded=False):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

def render_agent_debate_stream(selected_coin: str, latest_decision_id: str, decision_row: dict, votes: pd.DataFrame):
    st.markdown('<div class="agent-ticker"><b>Live Agent Debate</b><div class="muted">The highlighted analyst rotates automatically as the viewer refreshes.</div></div>', unsafe_allow_html=True)
    if votes.empty: st.info("No agent debate rows yet."); return
    rows = votes.to_dict("records"); tick = int(time.time() // 2); active_index = tick % len(rows); highlighted = []
    for offset in range(min(4, len(rows))):
        row = rows[(active_index + offset) % len(rows)]; active = "active" if offset == 0 else ""; agent = str(row.get("agent", "fallback")); leaning = vote_leaning(row); confidence = _safe_float(row.get("confidence")); reason = plain_reason(row.get("reason", ""))
        highlighted.append(f'''<div class="agent-row {active}"><b>{_html(agent_title_icon(agent))}</b><span class="muted"> · leaning <b>{_html(leaning)}</b> · confidence <b>{confidence:.3f}</b></span><br>{_html(reason)}</div>''')
    st.markdown("".join(highlighted), unsafe_allow_html=True)
    with st.expander("Full debate transcript for this decision", expanded=False):
        display_cols = [c for c in ["agent", "adjusted_buy_score", "adjusted_sell_score", "adjusted_hold_score", "adjusted_wait_score", "confidence", "reason"] if c in votes.columns]
        st.dataframe(votes[display_cols] if display_cols else votes, width="stretch", hide_index=True)


def render_agent_dialogue_panel(agent_name: str, votes: pd.DataFrame):
    sub = votes[votes["agent"].astype(str) == str(agent_name)] if not votes.empty and "agent" in votes.columns else pd.DataFrame()
    if not sub.empty and "ts" in sub.columns:
        sub = sub.copy()
        sub["ts_num"] = pd.to_numeric(sub.get("ts"), errors="coerce")
        sub = sub.sort_values("ts_num")

    if sub.empty:
        st.info("Select an analyst card to view its full dialogue.")
        return

    row = sub.iloc[-1].to_dict()
    reason = str(row.get("reason", ""))

    st.markdown(
        f'''
        <div class="screen-card">
            <h2>{_html(agent_title_icon(agent_name))}</h2>
            <div class="muted">{_html(agent_monitor_description(agent_name))}</div>
            <p>{_html(agent_full_plain_summary(row))}</p>
            <div class="context-grid">
                <div class="context-card"><h3>Current leaning</h3><p><b>{_html(vote_leaning(row))}</b></p></div>
                <div class="context-card"><h3>Confidence</h3><p><b>{_safe_float(row.get("confidence")):.3f}</b></p></div>
                <div class="context-card"><h3>Strongest score</h3><p><b>{strongest_vote_score(row):.3f}</b></p></div>
                <div class="context-card"><h3>Leaderboard rank</h3><p><b>{_html(row.get("leaderboard_rank", "not published"))}</b></p></div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.expander("Raw coded analyst reason", expanded=False):
        st.text(reason)

def render_agent_roster_no_buttons(selected_coin: str, votes: pd.DataFrame, agent_side_ratings_df: pd.DataFrame = None):
    st.markdown("### Analyst Roster")

    if votes.empty:
        st.info("No analyst rows yet.")
        return

    if "agent" in votes.columns:
        display_votes = votes.copy()
        if "ts" in display_votes.columns:
            display_votes["ts_num"] = pd.to_numeric(display_votes.get("ts"), errors="coerce")
            display_votes = display_votes.sort_values("ts_num")
        display_votes = display_votes.drop_duplicates(subset=["agent"], keep="last")
    else:
        display_votes = votes.copy()

    rows = display_votes.to_dict("records")
    side_rating_map = latest_agent_side_ratings_map(agent_side_ratings_df)

    if not st.session_state.get("selected_agent_dialogue"):
        if "agent" in votes.columns and "volume_profile_leader" in votes["agent"].astype(str).tolist():
            st.session_state["selected_agent_dialogue"] = "volume_profile_leader"
        elif rows:
            st.session_state["selected_agent_dialogue"] = str(rows[0].get("agent", "fallback"))

    for i in range(0, len(rows), 3):
        cols = st.columns(3)

        for col, row in zip(cols, rows[i:i + 3]):
            agent = str(row.get("agent", "fallback"))
            leaning = vote_leaning(row).lower()
            selected_class = "active" if agent == st.session_state.get("selected_agent_dialogue") else ""
            side_rating = side_rating_map.get(agent, {})
            buy_weight_pct = _weight_pct_text(side_rating.get("buy_weight_pct"))
            sell_weight_pct = _weight_pct_text(side_rating.get("sell_weight_pct"))
            buy_accuracy = _pct_text(side_rating.get("buy_accuracy"))
            sell_accuracy = _pct_text(side_rating.get("sell_accuracy"))
            buy_rows = int(float(side_rating.get("buy_rows", 0) or 0))
            sell_rows = int(float(side_rating.get("sell_rows", 0) or 0))

            with col:
                st.markdown(
                    f'''
                    <div class="agent-card agent-card-{leaning} {selected_class}">
                        <div class="agent-title">{_html(agent_title_icon(agent))}</div>
                        <div class="agent-description">
                            {_html(agent_monitor_description(agent))}
                        </div>
                        <div class="agent-metrics">
                            Leaning: <b>{leaning.upper()}</b><br>
                            Confidence: <b>{_safe_float(row.get("confidence")):.3f}</b><br>
                            Strongest score: <b>{strongest_vote_score(row):.3f}</b><br>
                            Buy weight: <b>{buy_weight_pct}</b> · Buy accuracy: <b>{buy_accuracy}</b> · n=<b>{buy_rows}</b><br>
                            Sell weight: <b>{sell_weight_pct}</b> · Sell accuracy: <b>{sell_accuracy}</b> · n=<b>{sell_rows}</b>
                        </div>
                        <div class="agent-summary">
                            {_html(agent_plain_summary(row))}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

                decision_key = str(row.get("decision_id") or "no_decision")
                rank_key = str(row.get("leaderboard_rank") or i)
                row_key = f"{i}_{agent}_{decision_key}_{rank_key}".replace(" ", "_").replace("/", "_")

                if st.button(
                    "Click here",
                    key=f"agent_card_click_{selected_coin}_{row_key}",
                    width="stretch",
                    type="secondary",
                ):
                    set_selected_agent(agent)

    st.markdown('<div id="agent-dialogue-anchor"></div>', unsafe_allow_html=True)
    scroll_to_agent_dialogue_if_requested()

    selected_agent = st.session_state.get("selected_agent_dialogue")
    if selected_agent:
        render_agent_dialogue_panel(selected_agent, votes)

def render_topic_explanation(topic, selected_coin, votes, decisions_df, market_df, snapshot):
    coin = dict((snapshot.get("coins", {}) or {}).get(selected_coin, {}) or {}); _, drow, _ = latest_council_votes_for_coin(votes, decisions_df, selected_coin)
    if topic == "Why the bot is not buying live": st.info(plain_reason(coin.get("main_blocker") or coin.get("buy_blocker") or drow.get("reason", "No live-buy blocker is currently published.")))
    elif topic == "What would need to change": st.info("The bot needs fresher data, stronger expected utility, lower spread/fees, stronger agent confidence, or removal of the currently published blocker.")
    elif topic == "Chart levels": st.info(f"Published levels: POC {_safe_float(coin.get('point_of_control')):.8f}, VAH {_safe_float(coin.get('value_area_high')):.8f}, VAL {_safe_float(coin.get('value_area_low')):.8f}.")
    elif topic == "Fee impact": st.info("Profit-First Fee-Aware Mode is active: entries need projected net profit after Binance.US fees and spread." if coin.get("high_fee_tier_active") else "No explicit fee blocker is present in the selected snapshot row.")
    elif topic == "Agent disagreement": render_agent_disagreement_summary(votes)


def render_learning_console(selected_coin, votes, decisions_df, market_df, snapshot):
    st.markdown('<div class="screen-card"><h2>Learning Console</h2><div class="muted">Read what the bot is watching without clicking Ask buttons.</div></div>', unsafe_allow_html=True)
    topic = st.selectbox("Learning topic", ["Why the bot is not buying live", "What would need to change", "Chart levels", "Fee impact", "Agent disagreement", "Selected analyst details"], key=f"learning_topic_{selected_coin}")
    agent_options = [str(a) for a in votes["agent"].dropna().astype(str).unique().tolist()] if not votes.empty and "agent" in votes.columns else []
    if topic == "Selected analyst details" and agent_options:
        selected_agent = st.selectbox("Focus analyst", agent_options, format_func=agent_title_icon, key=f"focus_agent_{selected_coin}"); render_agent_dialogue_panel(selected_agent, votes); return
    render_topic_explanation(topic, selected_coin, votes, decisions_df, market_df, snapshot)


def render_strategy_screen(selected, snapshot, market_df, decisions_df, council_votes_df, targets_df, trades_df, shadow_df, agent_side_ratings_df=None):
    available = get_available_products(snapshot)
    if not available:
        st.info("No products are available yet. Waiting for bot files to populate.")
        return

    current = st.session_state.get("selected_coin") or st.session_state.get("strategy_arena_coin") or available[0]
    if current not in available:
        current = available[0]
    st.session_state["selected_coin"] = current
    st.session_state["strategy_arena_coin"] = current
    selected = st.selectbox(
        "Strategy Arena Coin",
        available,
        index=available.index(current),
        key="strategy_arena_coin",
    )
    st.session_state["selected_coin"] = selected

    timeframe_label = st.radio(
        "Chart Mode",
        ["1D · 1m", "7D · 1m", "30D · 15m", "90D · 1h", "2Y · 1d"],
        horizontal=True,
        key="chart_timeframe_label",
    )
    timeframe = normalize_timeframe_label(timeframe_label)
    overlays = render_overlay_controls()

    st.markdown(
        f'<div class="hud-header"><div class="hud-title">Strategy Arena</div>'
        f'<div class="hud-subtitle">{_html(selected)} · chart first, analyst debate below.</div></div>',
        unsafe_allow_html=True,
    )
    chart_df, chart_meta = load_chart_history(selected, timeframe)
    confirmed = confirmed_trades_only(trades_df, selected)
    target = latest_targets_for_coin(targets_df, selected)
    fig = build_coin_chart(chart_df, chart_meta, dict((snapshot.get("coins") or {}).get(selected, {}) or {}), market_df, confirmed, shadow_df, decisions_df, target, overlays, full_chart=True)
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"main_chart_{selected}_{timeframe}",
        config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
    )
    latest_decision_id, drow, votes = latest_council_votes_for_coin(council_votes_df, decisions_df, selected)
    render_agent_debate_stream(selected, latest_decision_id, drow, votes)
    render_agent_roster_no_buttons(selected, votes, agent_side_ratings_df)


def explain_current_trade_state(blocker: str, decision: dict, market: dict) -> str:
    text = str(blocker or "").lower()
    if "stale_market_data" in text:
        return "The setup may look interesting, but the bot does not trust the latest top-of-book quote age enough to place live money."
    if "expected_utility_too_low" in text:
        return "The bot sees the trade idea, but the reward after spread and fees is not strong enough."
    if "probability" in text and "too_low" in text:
        return "The bot wants a better probability edge before risking live money."
    if "score_below" in text:
        return "The setup score is below the calibrated threshold for this product."
    if boolish(market.get("buy_gate_tradeable")):
        return "The market telemetry says the coin is currently tradeable, but Level 8 still controls whether this becomes a live entry."
    if str(decision.get("action") or "").upper() == "SHADOW":
        return "The bot is shadow-trading this idea for learning, but at least one live-money safety rule is still blocking execution."
    return "No specific blocker was published. Use the score, utility, spread, and agent disagreement panels to inspect the setup."


def build_watch_items(selected_coin: str, coin: dict, market: dict, decision: dict, order_book: dict, target: dict, votes: pd.DataFrame) -> dict:
    blocker = str(market.get("buy_gate_blocker") or decision.get("reason") or "").lower()
    spread = _safe_float(market.get("spread_bps"))
    utility = _safe_float(decision.get("expected_utility_bps") or coin.get("expected_utility_bps"))
    score = _safe_float(decision.get("final_buy_score") or coin.get("final_buy_score"))
    threshold = _safe_float(decision.get("buy_threshold") or coin.get("buy_threshold"))
    main_watch = "Watch for the coin to improve score, utility, spread, and quote freshness at the same time."
    if "stale_market_data" in blocker:
        main_watch = "Top-of-book quote freshness is the immediate issue. Patch the keeper loop so the bot can trust live bid/ask."
    elif utility < 35:
        main_watch = "Utility is the main weakness. The bot needs more upside after Binance.US fees and spread."
    elif score < threshold:
        main_watch = "Score is the main weakness. Wait for stronger alignment across agents."
    elif boolish(market.get("buy_gate_tradeable")):
        main_watch = "Telemetry says BUY_READY. Watch whether Level 8 also confirms action."
    chart_watch = "Watch whether price is accepted above VAH, rejected at POC, or loses VAL."
    value_state = str(coin.get("value_acceptance_state") or "").lower()
    if "inside" in value_state:
        chart_watch = "Price appears inside value. Expect chop unless it accepts above VAH or rejects below VAL."
    elif "above" in value_state:
        chart_watch = "Price is above value. Watch whether it holds acceptance or rejects back inside."
    elif "below" in value_state:
        chart_watch = "Price is below value. Watch whether it reclaims VAL or continues weakness."
    order_book_watch = "Watch spread, imbalance, and liquidity risk before trusting live execution."
    if spread > 20:
        order_book_watch = "Spread is wide. Live entries need either tighter spread or much stronger projected edge."
    elif _safe_float(order_book.get("imbalance")) > 0.2:
        order_book_watch = "Bid-side imbalance is supportive. Watch whether it persists while score and utility improve."
    elif _safe_float(order_book.get("imbalance")) < -0.2:
        order_book_watch = "Ask-side imbalance is a headwind. The bot may wait for selling pressure to fade."
    target_watch = "No live position yet, so target plan is waiting for an entry."
    if boolish(target.get("has_position")):
        target_watch = "Position is open. Watch min profitable exit, scalp/core arming, and pullback trigger prices."
    return {"main_watch": main_watch, "chart_watch": chart_watch, "order_book_watch": order_book_watch, "target_watch": target_watch}


def render_chart_overlay_guide() -> None:
    st.markdown('''
        <div class="screen-card"><h2>Chart Overlay Guide</h2><div class="context-grid">
            <div class="context-card"><h3>POC / VAH / VAL</h3><p>POC is the price where the most volume traded. VAH and VAL mark the upper and lower edges of the value area. The bot watches these to decide whether price is balanced, breaking out, or rejecting value.</p></div>
            <div class="context-card"><h3>Prior Session POC / VAH / VAL</h3><p>These are yesterday or prior-session value levels. They help the bot judge whether today’s price is accepting above, rejecting from, or reclaiming older value.</p></div>
            <div class="context-card"><h3>Anchored VWAP</h3><p>Anchored VWAP estimates the average traded price from a chosen anchor window. Price above it can show buyers in control; price below it can show weaker demand.</p></div>
            <div class="context-card"><h3>Average Entry / Targets / Stop</h3><p>Average entry, target sell, stop, and min-profitable-exit lines are position-management levels. They matter after a live entry and keep exits fee-aware.</p></div>
            <div class="context-card"><h3>Shadow Trades</h3><p>Shadow trades are simulated learning decisions. They show where Level 8 wanted to learn from a setup without risking live money.</p></div>
            <div class="context-card"><h3>Level 8 Markers</h3><p>Level 8 markers show major strategy decisions. They help you connect the chart to what the council decided at that moment.</p></div>
        </div></div>
    ''', unsafe_allow_html=True)


def render_setup_type_guide(selected_coin: str, market: dict, decision: dict, votes: pd.DataFrame) -> None:
    reason_blob = " ".join([
        str(market.get("buy_gate_blocker", "")),
        str(decision.get("reason", "")),
        " ".join(votes["reason"].astype(str).tail(25).tolist()) if not votes.empty and "reason" in votes.columns else "",
    ]).lower()
    setup_notes = []
    if "sweep" in reason_blob or "reclaim" in reason_blob:
        setup_notes.append(("Sweep + Reclaim", "The bot is watching for price to sweep liquidity, reclaim a level, and then hold above it. This can indicate a reversal attempt."))
    if "breakout" in reason_blob or "accepted_above" in reason_blob:
        setup_notes.append(("Acceptance Breakout", "The bot is watching for price to accept above value instead of rejecting back inside. This is cleaner when spread is tight and utility is positive."))
    if "range_chop" in reason_blob or "inside_value" in reason_blob:
        setup_notes.append(("Range / Chop Avoidance", "The bot sees a balanced market. It may avoid buying because the move can stall near POC or high-volume nodes."))
    if "low_volume_node" in reason_blob or "lvn" in reason_blob:
        setup_notes.append(("Low-Volume Path", "The bot is watching whether price can move through a low-volume zone quickly. This can create better upside if confirmation is strong."))
    if "mean_reversion" in reason_blob or "reversion" in reason_blob:
        setup_notes.append(("Mean Reversion", "The bot is watching whether price has stretched too far and may snap back toward fair value."))
    if "expected_utility_too_low" in reason_blob:
        setup_notes.append(("Utility Filter", "The main issue is not that nothing is happening. The issue is that the projected reward is negative after fees, spread, uncertainty, and context penalties."))
    if not setup_notes:
        setup_notes.append(("Current Focus", "The bot is waiting for a cleaner setup with positive expected utility, stronger score/probability alignment, and acceptable execution cost."))
    html_parts = ['<div class="screen-card"><h2>Common Setup Types Being Monitored</h2><div class="context-grid">']
    for title, body in setup_notes[:6]:
        html_parts.append(f'<div class="context-card"><h3>{_html(title)}</h3><p>{_html(body)}</p></div>')
    html_parts.append('</div></div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_deep_learning_screen(selected, snapshot, market_df, decisions_df, council_votes_df, order_book_df, targets_df):
    selected_coin = st.session_state.get("selected_coin", selected)
    coin = dict((snapshot.get("coins", {}) or {}).get(selected_coin, {}) or {})
    market = latest_row_for_product(market_df, selected_coin)
    decision = latest_row_for_product(decisions_df, selected_coin)
    order_book = latest_row_for_product(order_book_df, selected_coin)
    target = latest_row_for_product(targets_df, selected_coin)
    _, _, votes = latest_council_votes_for_coin(council_votes_df, decisions_df, selected_coin)
    st.markdown(f'<div class="hud-header"><div class="hud-title">Deep Learning Context</div><div class="hud-subtitle">{_html(selected_coin)} · what the bot is watching, why it is waiting, and what would need to change.</div></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Decision", str(decision.get("action") or coin.get("decision_action") or "WAIT"))
    col2.metric("Buy Score", f'{_safe_float(decision.get("final_buy_score") or coin.get("final_buy_score")):.3f}')
    col3.metric("Threshold", f'{_safe_float(decision.get("buy_threshold") or coin.get("buy_threshold")):.3f}')
    col4.metric("Expected Utility", f'{_safe_float(decision.get("expected_utility_bps") or coin.get("expected_utility_bps")):.2f} bps')
    blocker = str(market.get("buy_gate_blocker") or coin.get("main_blocker") or coin.get("buy_blocker") or decision.get("reason") or "")
    watch_items = build_watch_items(selected_coin, coin, market, decision, order_book, target, votes)
    st.markdown('<div class="context-grid">', unsafe_allow_html=True)
    st.markdown(f'''<div class="context-card"><h3>1. Current Trade Lesson</h3><b>Plain-English state:</b><br>{_html(explain_current_trade_state(blocker, decision, market))}<div class="watch-list"><b>Watch next:</b><br>{_html(watch_items.get("main_watch", "Wait for a cleaner published setup."))}</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="context-card"><h3>2. Chart + Volume Profile</h3><b>POC:</b> {_safe_float(coin.get("point_of_control")):.8f}<br><b>VAH:</b> {_safe_float(coin.get("value_area_high")):.8f}<br><b>VAL:</b> {_safe_float(coin.get("value_area_low")):.8f}<br><b>Value state:</b> {_html(coin.get("value_acceptance_state") or "not published")}<br><b>Volume node:</b> {_html(coin.get("volume_node_state") or "not published")}<div class="watch-list"><b>Watch next:</b><br>{_html(watch_items.get("chart_watch", "Look for price accepting above value or rejecting below value."))}</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="context-card"><h3>3. Order Book + Fees</h3><b>Spread:</b> {_safe_float(market.get("spread_bps")):.2f} bps<br><b>Bid depth:</b> {_safe_float(order_book.get("bid_depth_usd")):.2f}<br><b>Ask depth:</b> {_safe_float(order_book.get("ask_depth_usd")):.2f}<br><b>Imbalance:</b> {_safe_float(order_book.get("imbalance")):.3f}<br><b>Liquidity risk:</b> {_safe_float(order_book.get("liquidity_risk_score")):.3f}<div class="watch-list"><b>Watch next:</b><br>{_html(watch_items.get("order_book_watch", "Wait for tighter spread and healthier bid/ask depth."))}</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="context-card"><h3>4. Targets + Sell Plan</h3><b>Has position:</b> {_html(target.get("has_position", False))}<br><b>Average entry:</b> {_safe_float(target.get("avg_entry_price")):.8f}<br><b>Min profitable exit:</b> {_safe_float(target.get("min_profitable_exit_price")):.8f}<br><b>Scalp target:</b> {_safe_float(target.get("scalp_target_price")):.8f}<br><b>Core target:</b> {_safe_float(target.get("core_target_price")):.8f}<div class="watch-list"><b>Watch next:</b><br>{_html(watch_items.get("target_watch", "No live position yet, so target plan is waiting for entry."))}</div></div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    render_chart_overlay_guide()
    render_setup_type_guide(selected_coin, market, decision, votes)
    with st.expander("Agent disagreement behind this context", expanded=False):
        render_agent_disagreement_summary(votes)
        display_cols = [c for c in ["agent", "adjusted_buy_score", "adjusted_sell_score", "adjusted_hold_score", "adjusted_wait_score", "confidence", "reason"] if c in votes.columns]
        if display_cols and not votes.empty:
            st.dataframe(votes[display_cols], width="stretch", hide_index=True)
        else:
            st.info("No council vote rows yet for this coin.")
    with st.expander("Raw context rows", expanded=False):
        st.write("Latest market row"); st.json(market); st.write("Latest decision row"); st.json(decision); st.write("Latest order-book row"); st.json(order_book); st.write("Latest target row"); st.json(target)


def replay_calibration_eligible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "accepted_for_calibration" in out.columns:
        accepted = pd.to_numeric(out.get("accepted_for_calibration"), errors="coerce").fillna(0)
        out = out[accepted.astype(int).eq(1)].copy()
    else:
        out = out[~out.get("timeframe", pd.Series("", index=out.index)).astype(str).str.contains("daily", case=False, na=False)].copy()
    if "replay_candidate_qualified" in out.columns:
        qualified = pd.to_numeric(out.get("replay_candidate_qualified"), errors="coerce").fillna(0)
        out = out[qualified.astype(int).eq(1)].copy()
    return out


def render_historical_replay_profitability_box(historical_replay_df: pd.DataFrame, historical_replay_summary_df: pd.DataFrame = None) -> None:
    eligible = replay_calibration_eligible_frame(historical_replay_df)
    if eligible is None or eligible.empty:
        st.markdown('<div class="screen-card"><h2>Replay Profitability Estimate</h2><p>No qualified replay rows are available yet. Once historical replay finishes, this box will show projected profitability from accepted replay candidates.</p></div>', unsafe_allow_html=True)
        return
    frame = eligible.copy()
    if "net_pnl_bps" not in frame.columns:
        st.info("Qualified replay rows exist, but net_pnl_bps is missing.")
        return
    net_bps = pd.to_numeric(frame["net_pnl_bps"], errors="coerce").dropna()
    if net_bps.empty:
        st.info("Qualified replay rows exist, but net P/L values are not numeric yet.")
        return
    if "net_pnl_usd" in frame.columns:
        net_usd = pd.to_numeric(frame["net_pnl_usd"], errors="coerce").fillna(0.0)
    else:
        notional = pd.to_numeric(frame.get("synthetic_notional_usd", 5.0), errors="coerce").fillna(5.0)
        net_usd = notional * (pd.to_numeric(frame["net_pnl_bps"], errors="coerce").fillna(0.0) / 10000.0)
    wins = int((net_bps > 0).sum())
    losses = int((net_bps <= 0).sum())
    win_rate = wins / max(1, wins + losses)
    total_net_usd = float(net_usd.sum())
    reference_portfolio_usd = 100.0
    total_portfolio_return_pct = total_net_usd / reference_portfolio_usd * 100.0
    daily_avg_pct = 0.0
    thirty_day_simple_pct = 0.0
    if "replay_ts" in frame.columns:
        dated = frame.copy()
        dated["replay_dt"] = pd.to_datetime(pd.to_numeric(dated["replay_ts"], errors="coerce"), unit="s", utc=True, errors="coerce")
        dated["_net_usd"] = net_usd
        dated = dated.dropna(subset=["replay_dt"])
        if not dated.empty:
            dated["replay_day"] = dated["replay_dt"].dt.date
            daily = dated.groupby("replay_day")["_net_usd"].sum()
            if not daily.empty:
                daily_avg_pct = float((daily / reference_portfolio_usd * 100.0).mean())
                thirty_day_simple_pct = daily_avg_pct * 30.0
    verdict_class = "good" if total_portfolio_return_pct > 0 and win_rate >= 0.50 else "warn"
    st.markdown(f'''<div class="screen-card"><h2>Replay Profitability Estimate</h2><p>Based only on qualified historical replay rows, the accepted replay set is currently <b>{total_portfolio_return_pct:.2f}%</b> on a $100 reference portfolio. Average active-day replay projection is <b>{daily_avg_pct:.2f}%</b>, which equals about <b>{thirty_day_simple_pct:.2f}%</b> over 30 active replay days if conditions repeated.</p></div>''', unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("Replay Portfolio Return", f"{total_portfolio_return_pct:.2f}%")
    cols[1].metric("30-Day Replay Projection", f"{thirty_day_simple_pct:.2f}%")
    cols[2].metric("Qualified Win Rate", f"{win_rate * 100.0:.1f}%")
    cols[3].metric("Avg Net / Trade", f"{float(net_bps.mean()):.2f} bps")
    cols[4].metric("Median Net / Trade", f"{float(net_bps.median()):.2f} bps")
    if verdict_class == "good":
        st.success("Qualified replay is showing positive portfolio return and at least 50% win rate.")
    else:
        st.warning("Replay profitability is not strong enough yet to assume live profitability. Treat this as a calibration signal, not proof.")

def render_calibration_loading_screen(calc_status: dict, snapshot: dict) -> None:
    progress = float(calc_status.get("overall_progress", 0.0) or 0.0)
    progress = max(0.0, min(1.0, progress))
    progress_pct = progress * 100.0
    phase_label = str(calc_status.get("phase_label") or "Calculating and calibrating")
    complete_products = int(calc_status.get("complete_products", 0) or 0)
    product_count = int(calc_status.get("product_count", 0) or 0)
    profit_ready_products = int(calc_status.get("profit_ready_products", 0) or 0)
    blocked_products = int(calc_status.get("blocked_products", 0) or 0)
    incomplete_products = int(calc_status.get("incomplete_products", 0) or 0)
    st.markdown(
        f"""<div class=\"calibration-gate\"><div class=\"calibration-title\">Calculating and calibrating</div><div class=\"calibration-subtitle\">The full viewer is locked only until startup calculation is complete. After that, the bot may still block live buys if replay profitability is not strong enough.</div><div class=\"calibration-phase-card\"><b>Current phase:</b> {_html(phase_label)}<br><b>Overall completion:</b> {progress_pct:.1f}%</div></div>""",
        unsafe_allow_html=True,
    )
    calculation_started_ts = float(calc_status.get("calculation_started_ts", 0.0) or 0.0)
    if calculation_started_ts > 0.0:
        elapsed_sec = max(0.0, time.time() - calculation_started_ts)
    else:
        elapsed_sec = float(calc_status.get("calculation_elapsed_sec", 0.0) or 0.0)
    elapsed_label = format_elapsed_duration(elapsed_sec)
    st.markdown(
        f'''
        <div class="calibration-elapsed">
            Time elapsed calculating and calibrating: <span>{_html(elapsed_label)}</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.progress(progress)
    status_ts = float(calc_status.get("ts", 0.0) or 0.0)
    age_sec = max(0.0, time.time() - status_ts) if status_ts > 0 else 0.0
    status_source = str(calc_status.get("viewer_status_source") or "calculation_status.json")
    st.caption(f"Status age: {age_sec:.1f}s · Source: {status_source} · Auto-refresh should update this every 2 seconds.")
    cols = st.columns(4)
    cols[0].metric("Complete products", f"{complete_products}/{product_count}")
    cols[1].metric("Profit-ready products", profit_ready_products)
    cols[2].metric("Blocked by replay", blocked_products)
    cols[3].metric("Still calculating", incomplete_products)
    readiness = calc_status.get("readiness", {}) or {}
    if calc_status.get("full_viewer_unlocked") and not all([readiness.get("viewer_snapshot_recent"), readiness.get("websocket_recent"), readiness.get("market_csv_recent")]):
        st.warning("Calculation is complete, but live market data freshness is not perfect. The viewer remains unlocked; check websocket, market.csv, and viewer_snapshot freshness before unattended trading.")
    policy = calc_status.get("policy", {}) or {}
    st.markdown("### Historical data and exchange mode")
    exchange_cols = st.columns(4)
    exchange_cols[0].metric("Live execution", str(policy.get("live_execution_exchange") or readiness.get("live_execution_exchange") or "binance_us"))
    exchange_cols[1].metric("Binance bulk history", str(policy.get("binance_bulk_historical_backfill_enabled") or readiness.get("binance_bulk_historical_backfill_enabled")))
    exchange_cols[2].metric("Binance live trading", str(policy.get("binance_live_execution_enabled") or readiness.get("binance_live_execution_enabled")))
    exchange_cols[3].metric("Historical priority", " → ".join(policy.get("historical_source_priority") or readiness.get("historical_source_priority") or []))
    st.markdown("### Startup engine")
    startup_cols = st.columns(5)
    startup_cols[0].metric("Parallel replay", str(policy.get("historical_replay_parallel_startup_enabled", False)))
    startup_cols[1].metric("Parallel jobs", int(policy.get("historical_replay_startup_parallel_jobs", 0) or 0))
    startup_cols[2].metric("Parallel fetches", int(policy.get("historical_replay_max_parallel_fetches", 0) or 0))
    startup_cols[3].metric("CPU worker replay", str(policy.get("full_replay_math_in_process_workers", False)))
    startup_cols[4].metric("Worker import", str(policy.get("historical_replay_worker_import_ok", False)))
    st.markdown("### Replay fee comparison")
    fee_cols = st.columns(4)
    fee_cols[0].metric("Fee comparison", str(policy.get("replay_exchange_fee_comparison_enabled", False)))
    fee_cols[1].metric("Primary model", str(policy.get("replay_primary_fee_model", "binance_us")))
    fee_cols[2].metric("Comparison model", str(policy.get("replay_comparison_fee_model", "binance_us")))
    fee_cols[3].metric("Binance taker", f"{float(policy.get('binance_us_comparison_taker_fee_bps', 0.0) or 0.0):.2f} bps")
    worker_import_error = str(policy.get("historical_replay_worker_import_error", "") or "")
    if worker_import_error:
        st.error(f"CPU worker import error: {worker_import_error}")
    worker_manifest = calc_status.get("historical_replay_worker_manifest", {}) or {}
    if worker_manifest:
        st.markdown("### Replay worker manifest")
        cols = st.columns(5)
        cols[0].metric("Worker jobs", int(worker_manifest.get("total_jobs", 0) or 0))
        cols[1].metric("Merged", int(worker_manifest.get("merged_jobs", 0) or 0))
        cols[2].metric("Done", int(worker_manifest.get("done_jobs", 0) or 0))
        cols[3].metric("Running", int(worker_manifest.get("running_jobs", 0) or 0))
        cols[4].metric("Failed", int(worker_manifest.get("failed_jobs", 0) or 0))
        st.progress(float(worker_manifest.get("progress", 0.0) or 0.0))

        failed_errors = worker_manifest.get("failed_job_errors", []) or []
        running_detail = worker_manifest.get("running_jobs_detail", []) or []
        pending_detail = worker_manifest.get("next_pending_jobs", []) or []

        if failed_errors:
            st.error("Historical replay workers are failing. Open the table below before waiting longer; the loading bar will not advance while these jobs fail.")
            st.dataframe(pd.DataFrame(failed_errors), width="stretch", hide_index=True)

        if running_detail:
            with st.expander("Currently running replay worker jobs", expanded=False):
                st.dataframe(pd.DataFrame(running_detail), width="stretch", hide_index=True)

        if pending_detail:
            with st.expander("Next pending replay worker jobs", expanded=False):
                st.dataframe(pd.DataFrame(pending_detail), width="stretch", hide_index=True)
    exchange_map_df = load_csv(EXCHANGE_PRODUCT_MAP_CSV_PATH)
    if exchange_map_df is not None and not exchange_map_df.empty:
        with st.expander("Canonical product ↔ Binance.US symbol mapping", expanded=False):
            st.dataframe(exchange_map_df, width="stretch", hide_index=True)
    phase = calc_status.get("phase_progress", {}) or {}
    st.markdown("### Phase progress")
    pcols = st.columns(5)
    pcols[0].metric("Live data", f"{float(phase.get('live_data', 0.0)) * 100.0:.1f}%")
    pcols[1].metric("Micro backlog", f"{float(phase.get('micro_backlog', 0.0)) * 100.0:.1f}%")
    pcols[2].metric("Candle backlog", f"{float(phase.get('historical_candle_backlog', 0.0)) * 100.0:.1f}%")
    pcols[3].metric("Historical replay", f"{float(phase.get('historical_replay', 0.0)) * 100.0:.1f}%")
    pcols[4].metric("Replay verdicts", f"{float(phase.get('replay_calibration_verdicts', 0.0)) * 100.0:.1f}%")
    product_status = calc_status.get("product_status", {}) or {}
    if product_status:
        rows = []
        for product_id, status in product_status.items():
            rows.append({"product_id": product_id, "overall_progress_pct": round(float(status.get("overall_product_progress", 0.0)) * 100.0, 1), "verdict": status.get("verdict", "unknown"), "micro_rows": status.get("micro_rows", 0), "15m_candles": status.get("historical_15m_candle_rows", 0), "15m_required": status.get("required_15m_candle_rows", 0), "1h_candles": status.get("historical_1h_candle_rows", 0), "1h_required": status.get("required_1h_candle_rows", 0), "15m_replay": status.get("primary_15m_90d_rows", 0), "1h_replay": status.get("regime_1h_365d_rows", 0), "qualified_rows": status.get("qualified_rows", 0), "avg_net_bps": round(float(status.get("avg_net_pnl_bps", 0.0)), 2), "complete": bool(status.get("complete")), "live_trade_allowed": bool(status.get("live_trade_allowed")), "reason": status.get("reason", "")})
        df = pd.DataFrame(rows).sort_values(["complete", "overall_progress_pct"], ascending=[True, True])
        st.markdown("### Product calculation status")
        st.dataframe(df, width="stretch", hide_index=True)
    with st.expander("Raw calculation status", expanded=False):
        st.json(calc_status)
    st.info("When this reaches 100%, refreshing localhost will show the normal All-Coin Command Deck. If a product is replay-complete but unprofitable, it still counts as calculated, but live buys remain blocked for that product.")


def render_replay_fee_comparison_panel():
    df = load_csv_tail(REPLAY_FEE_COMPARISON_SUMMARY_CSV_PATH, max_lines=5000)
    st.markdown("### Binance.US Replay Fee Scenarios")
    if df is None or df.empty:
        st.info("No replay fee comparison summary yet. It will populate after historical worker outputs merge.")
        return
    numeric_cols = ["rows", "primary_avg_net_bps", "primary_median_net_bps", "primary_win_rate", "comparison_avg_net_bps", "comparison_median_net_bps", "comparison_win_rate", "avg_improvement_bps", "median_improvement_bps", "rows_flipped_to_profit_by_comparison"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    latest = df.sort_values("ts").groupby(["product_id", "timeframe"], as_index=False).tail(1) if all(c in df.columns for c in ["ts", "product_id", "timeframe"]) else df.tail(200)
    total_rows = int(latest["rows"].sum()) if "rows" in latest.columns else 0
    avg_improvement = float(latest["avg_improvement_bps"].mean()) if "avg_improvement_bps" in latest.columns and not latest.empty else 0.0
    flipped = int(latest["rows_flipped_to_profit_by_comparison"].sum()) if "rows_flipped_to_profit_by_comparison" in latest.columns else 0
    cols = st.columns(4)
    cols[0].metric("Compared rows", total_rows)
    cols[1].metric("Avg Binance improvement", f"{avg_improvement:.2f} bps")
    cols[2].metric("Rows flipped profitable", flipped)
    cols[3].metric("Products/timeframes", len(latest))
    show_cols = [c for c in ["product_id", "timeframe", "rows", "primary_avg_net_bps", "comparison_avg_net_bps", "avg_improvement_bps", "primary_win_rate", "comparison_win_rate", "rows_flipped_to_profit_by_comparison", "primary_fee_model", "comparison_fee_model"] if c in latest.columns]
    st.dataframe(latest[show_cols], width="stretch", hide_index=True)

def render_four_pass_backtest_box(
    four_pass_agent_buy_df: pd.DataFrame,
    four_pass_council_buy_df: pd.DataFrame,
    four_pass_agent_sell_df: pd.DataFrame,
    four_pass_council_sell_df: pd.DataFrame,
    four_pass_final_agent_ratings_df: pd.DataFrame,
) -> None:
    st.markdown("### Four-Pass Council Backtest")

    buy_agent_rows = 0 if four_pass_agent_buy_df is None or four_pass_agent_buy_df.empty else len(four_pass_agent_buy_df)
    buy_council_rows = 0 if four_pass_council_buy_df is None or four_pass_council_buy_df.empty else len(four_pass_council_buy_df)
    sell_agent_rows = 0 if four_pass_agent_sell_df is None or four_pass_agent_sell_df.empty else len(four_pass_agent_sell_df)
    sell_council_rows = 0 if four_pass_council_sell_df is None or four_pass_council_sell_df.empty else len(four_pass_council_sell_df)

    cols = st.columns(4)
    cols[0].metric("BUY Pass 1 Agents", buy_agent_rows)
    cols[1].metric("BUY Pass 2 Council", buy_council_rows)
    cols[2].metric("SELL Pass 1 Agents", sell_agent_rows)
    cols[3].metric("SELL Pass 2 Council", sell_council_rows)

    if buy_agent_rows > 0:
        st.markdown("#### BUY Agent Weights")
        display_cols = [c for c in ["agent", "selected_count", "win_rate", "avg_net_bps", "median_net_bps", "score", "buy_weight_pct"] if c in four_pass_agent_buy_df.columns]
        st.dataframe(four_pass_agent_buy_df.sort_values("buy_weight_pct", ascending=False)[display_cols], use_container_width=True, hide_index=True)

    if buy_council_rows > 0:
        st.markdown("#### Weighted BUY Council")
        display_cols = [c for c in ["product_id", "selected_count", "threshold", "win_rate", "avg_net_bps", "median_net_bps", "portfolio_return_pct_100_ref", "score"] if c in four_pass_council_buy_df.columns]
        st.dataframe(four_pass_council_buy_df.sort_values("score", ascending=False)[display_cols], use_container_width=True, hide_index=True)

    if sell_agent_rows > 0:
        st.markdown("#### SELL Analyst Weights")
        display_cols = [c for c in ["agent", "selected_count", "good_exit_rate", "too_early_rate", "avg_move_after_sell_bps", "score", "sell_weight_pct"] if c in four_pass_agent_sell_df.columns]
        st.dataframe(four_pass_agent_sell_df.sort_values("sell_weight_pct", ascending=False)[display_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("Sell-side four-pass data is not ready yet. This is expected until sell_outcomes.csv or synthetic sell replay rows exist.")

    if four_pass_final_agent_ratings_df is not None and not four_pass_final_agent_ratings_df.empty:
        with st.expander("Final four-pass side-specific analyst ratings", expanded=False):
            st.dataframe(four_pass_final_agent_ratings_df.tail(200), width="stretch", hide_index=True)


def render_debug_launch_screen(snapshot, market_df, decisions_df, council_votes_df, trades_df, orders_df, missed_df=None, shadow_sell_replay_df=None, historical_replay_df=None, historical_replay_summary_df=None, four_pass_agent_buy_df=None, four_pass_council_buy_df=None, four_pass_agent_sell_df=None, four_pass_council_sell_df=None, four_pass_final_agent_ratings_df=None):
    st.markdown('<div class="hud-header"><div class="hud-title">Launch / Debug Health</div><div class="hud-subtitle">Startup readiness, early-learning files, orders, and raw health.</div></div>', unsafe_allow_html=True)
    readiness = snapshot.get("readiness", {}) or {}
    st.metric("Trading Mode", readiness.get("live_trading_mode_label", readiness.get("trading_aggression_mode", "unknown")))
    cols = st.columns(6)
    cols[0].metric("WebSocket Recent", str(readiness.get("websocket_recent")))
    cols[1].metric("Safe Overnight", str(readiness.get("safe_to_run_overnight")))
    cols[2].metric("Micro Ready", str(readiness.get("micro_history_ready")))
    cols[3].metric("Macro Ready", str(readiness.get("macro_ready")))
    cols[4].metric("Calibrated", f'{readiness.get("product_calibration_ready_count", 0)}/{readiness.get("product_count", 0)}')
    cols[5].metric("TOB Keeper", str(readiness.get("top_of_book_keeper_running")))
    last_tob_keeper = _safe_float(readiness.get("last_tob_keeper_cycle_ts"))
    if last_tob_keeper > 0:
        st.caption(f"Last top-of-book keeper cycle age: {format_age(max(0.0, time.time() - last_tob_keeper))}")
    if readiness.get("websocket_recent") is False:
        st.warning("WebSocket/top-of-book freshness is not healthy. The bot may shadow valid setups as stale_market_data until top-of-book refresh is repaired.")
    if readiness.get("safe_to_run_overnight") is False:
        st.warning("safe_to_run_overnight is false. Check websocket freshness, duplicate process status, writable logs, and risk pause status before unattended running.")


    st.markdown("### Historical Shadow Replay Calibration")
    render_historical_replay_profitability_box(historical_replay_df, historical_replay_summary_df)
    render_four_pass_backtest_box(
        four_pass_agent_buy_df,
        four_pass_council_buy_df,
        four_pass_agent_sell_df,
        four_pass_council_sell_df,
        four_pass_final_agent_ratings_df,
    )
    hist_enabled = readiness.get("historical_replay_enabled")
    hist_running = readiness.get("historical_replay_running")
    hist_ready = readiness.get("historical_replay_ready_count", 0)
    hist_total = readiness.get("historical_replay_product_count", 0)
    eligible_replay = replay_calibration_eligible_frame(historical_replay_df)
    replay_cols = st.columns(4)
    replay_cols[0].metric("All Replay Rows", len(historical_replay_df) if historical_replay_df is not None else 0)
    replay_cols[1].metric("Qualified Calibration Rows", len(eligible_replay))
    replay_cols[2].metric("Ready Products", f"{hist_ready}/{hist_total}")
    replay_cols[3].metric("Replay Running", str(hist_running))
    st.caption(f"Replay enabled={hist_enabled}; calibration mode={'Profit Replay' if readiness.get('profit_replay_based_calibration_enabled') else 'Fallback'}")
    if eligible_replay is not None and not eligible_replay.empty and "net_pnl_bps" in eligible_replay.columns:
        net = pd.to_numeric(eligible_replay["net_pnl_bps"], errors="coerce").dropna()
        wins = int((net > 0).sum())
        losses = int((net <= 0).sum())
        profit_cols = st.columns(4)
        profit_cols[0].metric("Qualified Win Rate", f"{wins / max(1, wins + losses) * 100.0:.1f}%")
        profit_cols[1].metric("Median Net", f"{net.median():.2f} bps")
        profit_cols[2].metric("Average Net", f"{net.mean():.2f} bps")
        profit_cols[3].metric("Total Net", f"{net.sum():.2f} bps")
        if net.mean() > 0 and net.median() > 0:
            st.success("Qualified replay is net-positive on average and at the median.")
        elif net.mean() > 0:
            st.warning("Qualified replay is positive on average but negative at the median. A few large winners may be carrying many losers.")
        else:
            st.warning("Qualified replay is not net-positive yet. The bot should not trust this product family for live scaling.")
    else:
        st.info("No qualified calibration replay rows yet. Wait for 15m/90d and 1h/365d rows to build across products.")
    if historical_replay_summary_df is None or historical_replay_summary_df.empty:
        st.info("No historical replay summary rows yet. The bot will fill historical_shadow_replay.csv in the background, then use net replay profit to calibrate each product.")
    else:
        summary = historical_replay_summary_df.copy()
        if "ts" in summary.columns:
            summary["ts_num"] = pd.to_numeric(summary["ts"], errors="coerce")
            summary = summary.sort_values("ts_num").groupby("product_id").tail(1)
        show_cols = [c for c in ["product_id", "rows", "wins", "losses", "win_rate", "median_net_pnl_bps", "avg_net_pnl_bps", "days_covered", "calibration_ready", "recommended_min_score", "recommended_min_probability", "recommended_min_expected_value_bps"] if c in summary.columns]
        st.dataframe(summary[show_cols], width="stretch", hide_index=True)
        net = pd.to_numeric(summary.get("median_net_pnl_bps", pd.Series(dtype=float)), errors="coerce").dropna()
        if not net.empty:
            st.write(f"Across ready products, median replay net P/L is {net.median():.2f} bps. Positive values mean the entry+sell model was net-profitable in historical replay.")
    if historical_replay_df is not None and not historical_replay_df.empty:
        with st.expander("Latest historical replay rows", expanded=False):
            show_cols = [
                c
                for c in [
                    "product_id",
                    "timeframe",
                    "replay_source",
                    "historical_source_exchange",
                    "historical_source_symbol",
                    "replay_candidate_qualified",
                    "accepted_for_calibration",
                    "net_pnl_bps",
                    "replay_filter_reason",
                    "historical_source_note",
                ]
                if c in historical_replay_df.columns
            ]
            st.dataframe(historical_replay_df.tail(200)[show_cols], width="stretch", hide_index=True)

    with st.expander("Binance.US replay fee scenarios", expanded=False):
        render_replay_fee_comparison_panel()

    fee_comparison_df = load_csv_tail(REPLAY_FEE_COMPARISON_SUMMARY_CSV_PATH, max_lines=5000)
    if fee_comparison_df is not None and not fee_comparison_df.empty:
        with st.expander("replay_fee_comparison_summary.csv", expanded=False):
            st.dataframe(fee_comparison_df, width="stretch", hide_index=True)

    explanations = readiness.get("readiness_explanation") or []
    if explanations:
        st.warning("System readiness notes:")
        for item in explanations:
            st.write(f"- {item}")

    coin_count = len((snapshot.get("coins") or {}))
    if coin_count == 0:
        st.info("viewer_snapshot.json currently has zero coins. The viewer is using CSV fallback data from market.csv, council_decisions.csv, council_votes.csv, and position_targets.csv until the bot publishes a full snapshot.")

    st.markdown("### Why no live trades yet?")
    latest_decisions = decisions_df.copy()
    if not latest_decisions.empty and "ts" in latest_decisions.columns:
        latest_decisions["ts_num"] = pd.to_numeric(latest_decisions["ts"], errors="coerce")
        latest_decisions = latest_decisions.sort_values("ts_num").groupby("product_id").tail(1) if "product_id" in latest_decisions.columns else latest_decisions.sort_values("ts_num").tail(20)
    if latest_decisions.empty:
        st.info("No Level 8 decisions are published yet.")
    else:
        actions = latest_decisions["action"].astype(str).value_counts().to_dict() if "action" in latest_decisions.columns else {}
        reasons_blob = " ".join(latest_decisions.get("reason", pd.Series(dtype=str)).fillna("").astype(str).tolist()).lower()
        counts = {
            "expected_utility_too_low": reasons_blob.count("expected_utility_too_low"),
            "maker_adjusted_ev_too_low": reasons_blob.count("maker_adjusted_ev_too_low"),
            "buy_does_not_beat_wait": reasons_blob.count("buy_does_not_beat_wait"),
            "score_below_target": reasons_blob.count("score_below"),
            "probability_below_target": reasons_blob.count("probability_below"),
        }
        count_cols = st.columns(2)
        count_cols[0].metric("Latest SHADOW decisions", actions.get("SHADOW", 0))
        count_cols[1].metric("Latest COMMENTARY decisions", actions.get("COMMENTARY", 0))

        with st.expander("Raw latest no-trade counts", expanded=False):
            st.write({"actions": actions, "blockers": counts})

        if counts["expected_utility_too_low"] > 0:
            st.warning("The main reason no trades are firing is expected_utility_too_low. That means Level 8 thinks the setup does not make enough net profit after Binance.US fees, spread, uncertainty, wait utility, and context penalties.")
            st.info("This is not a viewer failure. It means the bot is finding activity, but Level 8 does not believe the setups are net-profitable enough yet after Binance.US costs and context penalties.")


    st.markdown("### Shadow Sell Replay")
    if shadow_sell_replay_df is None or shadow_sell_replay_df.empty:
        st.info("shadow_sell_replay.csv is empty. This means runtime shadow entries have not yet been replayed through the sell model. After this patch, check bot.debug.log for shadow_sell_replay_no_rows to see whether the cause is no future candles, duplicate keys, or not enough aged shadow decisions.")
    else:
        replay = shadow_sell_replay_df.copy()
        wins = pd.to_numeric(replay.get("would_have_won", pd.Series(dtype=float)), errors="coerce").fillna(0)
        net = pd.to_numeric(replay.get("net_pnl_bps", pd.Series(dtype=float)), errors="coerce").dropna()
        stops = pd.to_numeric(replay.get("would_have_hit_stop", pd.Series(dtype=float)), errors="coerce").fillna(0)
        min_profit = pd.to_numeric(replay.get("would_have_hit_min_profit", pd.Series(dtype=float)), errors="coerce").fillna(0)
        cols = st.columns(5)
        cols[0].metric("Replay Rows", len(replay))
        cols[1].metric("Win Rate", f"{wins.mean() * 100.0:.1f}%" if len(wins) else "0.0%")
        cols[2].metric("Median Net", f"{net.median():.2f} bps" if not net.empty else "0.00 bps")
        cols[3].metric("Stop Hit Rate", f"{stops.mean() * 100.0:.1f}%" if len(stops) else "0.0%")
        cols[4].metric("Min Profit Hit", f"{min_profit.mean() * 100.0:.1f}%" if len(min_profit) else "0.0%")
        if not net.empty:
            if net.median() > 0:
                st.success("The shadow sell replay is currently net-positive at the median. This suggests the sell model may be able to turn some shadow entries into profitable exits.")
            else:
                st.warning("The shadow sell replay is currently net-negative at the median. This means buying all shadow entries would still likely lose money with the current sell model.")
        with st.expander("Latest shadow sell replay rows", expanded=False):
            show_cols = [c for c in ["product_id", "decision_id", "entry_price", "exit_price", "exit_reason", "net_pnl_bps", "max_favorable_bps", "max_adverse_bps", "would_have_won", "would_have_hit_stop", "would_have_hit_min_profit"] if c in replay.columns]
            st.dataframe(replay.tail(200)[show_cols], width="stretch", hide_index=True)

    st.markdown("### Overnight Run Summary")
    try:
        total_decisions = int(len(decisions_df)) if decisions_df is not None else 0
        total_votes = int(len(council_votes_df)) if council_votes_df is not None else 0
        total_trades = int(len(trades_df)) if trades_df is not None else 0
        total_orders = int(len(orders_df)) if orders_df is not None else 0
        action_counts = (
            decisions_df["action"].astype(str).value_counts().to_dict()
            if decisions_df is not None and not decisions_df.empty and "action" in decisions_df.columns
            else {}
        )
        st.write(
            f"During the loaded runtime window, the bot published {total_decisions} Level 8 decisions, "
            f"{total_votes} agent votes, {total_orders} backend order rows, and {total_trades} confirmed trade rows."
        )
        st.write(f"Latest loaded action mix: {action_counts}")
        if total_trades == 0 and total_orders == 0:
            st.info(
                "No live orders or confirmed trades were recorded in the loaded runtime files. "
                "The bot was evaluating and shadowing setups, but it did not approve live execution."
            )
        if decisions_df is not None and not decisions_df.empty and "expected_utility_bps" in decisions_df.columns:
            util = pd.to_numeric(decisions_df["expected_utility_bps"], errors="coerce").dropna()
            if not util.empty:
                st.write(
                    f"Expected utility summary: median {util.median():.2f} bps, "
                    f"best {util.max():.2f} bps, worst {util.min():.2f} bps."
                )
    except Exception as exc:
        module_exception(MODULE_NAME, "overnight_summary_render_failed", exc, data={"traceback": traceback.format_exc()}, also_overall=False)

    if missed_df is not None and not missed_df.empty:
        st.markdown("### Missed Opportunity Learning")
        count = int(len(missed_df))
        st.write(f"The bot logged {count} missed opportunity review rows in the loaded runtime window.")
        if "product_id" in missed_df.columns:
            top_missed = missed_df["product_id"].astype(str).value_counts().head(5).to_dict()
            st.write(f"Most common missed-opportunity products: {top_missed}")
        if "move_bps" in missed_df.columns:
            moves = pd.to_numeric(missed_df["move_bps"], errors="coerce").dropna()
            if not moves.empty:
                st.write(f"Missed move size: median {moves.median():.2f} bps, max {moves.max():.2f} bps.")
        st.info(
            "Missed opportunities do not automatically mean the bot should have live traded. "
            "They are learning rows that show where the bot avoided or shadowed a setup that later moved."
        )

    with st.expander("Raw readiness JSON", expanded=False):
        st.json(readiness)
    for name, df in [("trades", trades_df), ("orders", orders_df), ("market", market_df), ("council_decisions", decisions_df), ("council_votes", council_votes_df)]:
        with st.expander(name, expanded=False):
            st.dataframe(df.tail(100), width="stretch", hide_index=True) if not df.empty else st.info(f"{name}.csv has no rows yet.")


def render_strategy_variant_replay_panel():
    df = load_csv_tail(STRATEGY_VARIANT_REPLAY_SUMMARY_CSV_PATH, max_lines=5000)
    st.markdown("### Strategy Variant Replay Comparison")
    if df is None or df.empty:
        st.info("No strategy variant replay summary yet. It will populate after historical worker outputs merge.")
        return
    numeric_cols = [c for c in df.columns if c.endswith("_avg_bps") or c.endswith("_win_rate") or c in ["rows", "hard_stop_rate", "profit_pullback_rate", "avg_mfe_bps", "avg_mae_bps"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    latest = df.sort_values("ts").groupby(["product_id", "timeframe", "strategy_variant"], as_index=False).tail(1)
    show_cols = [c for c in [
        "product_id", "timeframe", "strategy_variant", "rows",
        "binance_maker_maker_avg_bps", "binance_maker_maker_win_rate",
        "binance_maker_taker_avg_bps", "binance_maker_taker_win_rate",
        "binance_taker_taker_avg_bps", "binance_taker_taker_win_rate",
        "hard_stop_rate", "profit_pullback_rate", "early_adverse_exit_rate",
        "avg_mfe_bps", "avg_mae_bps",
    ] if c in latest.columns]
    st.dataframe(latest[show_cols], width="stretch", hide_index=True)
    if "strategy_variant" in latest.columns:
        by_variant = latest.groupby("strategy_variant").agg({
            "rows": "sum",
            "binance_maker_maker_avg_bps": "mean",
            "binance_maker_maker_win_rate": "mean",
            "binance_maker_taker_avg_bps": "mean",
            "binance_maker_taker_win_rate": "mean",
            "binance_taker_taker_avg_bps": "mean",
            "binance_taker_taker_win_rate": "mean",
            "hard_stop_rate": "mean",
            "profit_pullback_rate": "mean",
        }).reset_index()
        st.markdown("#### Overall by variant")
        st.dataframe(by_variant, width="stretch", hide_index=True)

def render_profitability_diagnostics_panel():
    st.markdown("### Profitability Diagnostics")
    variant_df = load_csv_tail(STRATEGY_VARIANT_REPLAY_SUMMARY_CSV_PATH, max_lines=10000)
    agent_policy_df = load_csv_tail(AGENT_TRADE_POLICY_CSV_PATH, max_lines=5000)
    component_df = load_csv_tail(AGENT_COMPONENT_REPLAY_ATTRIBUTION_CSV_PATH, max_lines=5000)
    if variant_df is not None and not variant_df.empty:
        st.markdown("#### Strategy variant performance")
        st.dataframe(variant_df.sort_values("ts").groupby(["product_id", "timeframe", "strategy_variant"], as_index=False).tail(1), width="stretch", hide_index=True)
    if agent_policy_df is not None and not agent_policy_df.empty:
        st.markdown("#### Agent trade policy")
        latest = agent_policy_df.sort_values("ts").groupby("agent", as_index=False).tail(1)
        st.dataframe(latest, width="stretch", hide_index=True)
    if component_df is not None and not component_df.empty:
        st.markdown("#### Replay component attribution")
        latest = component_df.sort_values("ts").groupby("component", as_index=False).tail(1)
        st.dataframe(latest, width="stretch", hide_index=True)


def render_live_dashboard(selected, refresh_config):
    now_tick = int(time.time()); st.session_state["_viewer_live_tick"] = now_tick
    module_debug(MODULE_NAME, "viewer_live_tick", data={"tick": now_tick, "selected_coin": selected, "timeframe": st.session_state.get("chart_timeframe_label", "1D · 1m"), "interval_label": refresh_config.get("interval_label")}, level="DEBUG", also_overall=False)
    snapshot = load_viewer_snapshot()
    calc_status = load_calculation_status(snapshot)
    if not bool(calc_status.get("full_viewer_unlocked", False)):
        render_calibration_loading_screen(calc_status, snapshot)
        return
    market_df = load_csv_tail(MARKET_CSV_PATH, max_lines=6000)
    decisions_df = load_csv_tail(COUNCIL_DECISIONS_PATH, max_lines=6000)
    council_votes_df = load_csv_tail(COUNCIL_VOTES_CSV_PATH, max_lines=40000)
    agent_side_ratings_df = load_csv_tail(AGENT_SIDE_RATINGS_PATH, max_lines=5000)
    targets_df = load_csv(POSITION_TARGETS_PATH)
    trades_df = load_csv(TRADES_CSV_PATH)
    orders_df = load_csv(ORDERS_CSV_PATH)
    shadow_df = load_csv_tail(SHADOW_TRADES_CSV_PATH, max_lines=6000)
    order_book_df = load_csv_tail(ORDER_BOOK_SNAPSHOTS_PATH, max_lines=6000)
    missed_df = load_csv_tail(MISSED_OPPORTUNITIES_CSV_PATH, max_lines=5000)
    shadow_sell_replay_df = load_csv_tail(SHADOW_SELL_REPLAY_CSV_PATH, max_lines=20000)
    historical_replay_df = load_csv_tail(HISTORICAL_SHADOW_REPLAY_CSV_PATH, max_lines=50000)
    historical_replay_summary_df = load_csv_tail(HISTORICAL_REPLAY_SUMMARY_CSV_PATH, max_lines=5000)
    four_pass_agent_buy_df = load_csv_tail(FOUR_PASS_AGENT_BUY_PATH, max_lines=5000)
    four_pass_council_buy_df = load_csv_tail(FOUR_PASS_COUNCIL_BUY_PATH, max_lines=5000)
    four_pass_agent_sell_df = load_csv_tail(FOUR_PASS_AGENT_SELL_PATH, max_lines=5000)
    four_pass_council_sell_df = load_csv_tail(FOUR_PASS_COUNCIL_SELL_PATH, max_lines=5000)
    four_pass_final_agent_ratings_df = load_csv_tail(FOUR_PASS_FINAL_AGENT_RATINGS_PATH, max_lines=5000)
    with st.container(): st.markdown('<section class="screen-section command-deck">', unsafe_allow_html=True); render_all_coin_landing_page(snapshot, market_df, decisions_df, council_votes_df, targets_df, trades_df, refresh_config); st.markdown('</section>', unsafe_allow_html=True)
    with st.container(): st.markdown('<div id="strategy-arena-anchor"></div>', unsafe_allow_html=True); scroll_to_strategy_arena_if_requested(); st.markdown('<section class="screen-section strategy-arena">', unsafe_allow_html=True); render_strategy_screen(selected, snapshot, market_df, decisions_df, council_votes_df, targets_df, trades_df, shadow_df, agent_side_ratings_df); st.markdown('</section>', unsafe_allow_html=True)
    with st.container(): st.markdown('<section class="screen-section deep-learning">', unsafe_allow_html=True); render_deep_learning_screen(selected, snapshot, market_df, decisions_df, council_votes_df, order_book_df, targets_df); st.markdown('</section>', unsafe_allow_html=True)
    with st.expander("Profitability diagnostics", expanded=True):
        render_profitability_diagnostics_panel()
    with st.expander("Strategy variant replay comparison", expanded=False):
        render_strategy_variant_replay_panel()
    with st.container(): st.markdown('<section class="screen-section debug-health">', unsafe_allow_html=True); render_debug_launch_screen(snapshot, market_df, decisions_df, council_votes_df, trades_df, orders_df, missed_df, shadow_sell_replay_df, historical_replay_df, historical_replay_summary_df, four_pass_agent_buy_df, four_pass_council_buy_df, four_pass_agent_sell_df, four_pass_council_sell_df, four_pass_final_agent_ratings_df); st.markdown('</section>', unsafe_allow_html=True)


def render_viewer_tick(refresh_config: dict) -> None:
    """Render one live viewer tick.

    The calculation gate must be checked before selecting/rendering coins.
    Once the completion latch exists, the viewer should enter the live dashboard
    even if all products are blocked by profitability.
    """
    snapshot = load_viewer_snapshot()
    calc_status = load_calculation_status(snapshot)

    if not bool(calc_status.get("full_viewer_unlocked", False)):
        render_calibration_loading_screen(calc_status, snapshot)
        return

    selected = pick_selected_coin(snapshot)

    if selected:
        render_live_dashboard(selected, refresh_config)
        return

    st.success("Startup calculation is complete and the viewer is unlocked.")
    st.info("Waiting for selectable coin rows in viewer_snapshot.json, market.csv, or products_active.csv.")

    product_status = calc_status.get("product_status", {}) or {}
    if isinstance(product_status, dict) and product_status:
        rows = []
        for product_id, row in product_status.items():
            rows.append({
                "product_id": product_id,
                "complete": bool(row.get("complete")),
                "profit_ready": bool(row.get("profit_ready")),
                "live_trade_allowed": bool(row.get("live_trade_allowed")),
                "verdict": str(row.get("verdict") or ""),
                "reason": str(row.get("reason") or ""),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def main() -> None:
    inject_crypto_game_css()
    refresh_config = get_refresh_config()
    render_crypto_header()

    run_every = run_every_value(refresh_config)

    if callable(getattr(st, "fragment", None)):
        @st.fragment(run_every=run_every)
        def viewer_auto_refresh_fragment():
            render_viewer_tick(refresh_config)

        viewer_auto_refresh_fragment()
        return

    st.warning("Streamlit fragment auto-refresh is unavailable. Using full-page fallback refresh instead.")
    render_viewer_tick(refresh_config)

    if refresh_config.get("live_enabled"):
        time.sleep(float(refresh_config.get("fallback_interval_sec", 2.0) or 2.0))
        st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer main crashed", exc, also_overall=True)
        try:
            st.error("Viewer crashed. Check debug/viewer.debug.log for the full traceback."); st.exception(exc)
        except Exception: pass
        raise
