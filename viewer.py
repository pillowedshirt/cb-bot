import html
import json
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import quote, unquote

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

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
VIEWER_SNAPSHOT_PATH = os.path.join(BASE_DIR, "viewer_snapshot.json")
VIEWER_SNAPSHOT_CSV_SAFE_PATH = VIEWER_SNAPSHOT_PATH
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
CHART_1M_7D_CSV_PATH = os.path.join(BASE_DIR, "chart_1m_7d.csv")
CHART_15M_30D_CSV_PATH = os.path.join(BASE_DIR, "chart_15m_30d.csv")
CHART_1H_90D_CSV_PATH = os.path.join(BASE_DIR, "chart_1h_90d.csv")
CHART_1D_2Y_CSV_PATH = os.path.join(BASE_DIR, "chart_1d_2y.csv")
CANDIDATE_REPLAY_PATH = os.path.join(BASE_DIR, "candidate_replay.csv")
AGENT_ADJUSTMENTS_PATH = os.path.join(BASE_DIR, "agent_adjustments.csv")
AGENT_PERFORMANCE_PATH = os.path.join(BASE_DIR, "agent_performance.csv")
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

AGENT_TITLES = {
    "volume_profile_leader": "🧭 Chief Market Strategist", "volume_profile_agent": "📊 Value Area Analyst",
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "": return float(default)
        return float(value)
    except Exception:
        return float(default)


def _html(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
.agent-card {
    border: 1px solid rgba(80, 220, 255, 0.20);
    border-radius: 16px;
    padding: 0.85rem;
    background: rgba(7, 18, 32, 0.92);
    height: 260px;
    min-height: 260px;
    max-height: 260px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}
.agent-card .agent-title { font-weight: 900; margin-bottom: 0.35rem; }
.agent-card .agent-summary { flex: 1; overflow: hidden; color: #8db7c8; font-size: 0.88rem; line-height: 1.22rem; }
.agent-card .agent-metrics { margin-top: 0.45rem; font-size: 0.85rem; }
.agent-card-buy { border-color: rgba(0, 255, 160, 0.45); } .agent-card-sell { border-color: rgba(255, 87, 116, 0.45); } .agent-card-hold { border-color: rgba(255, 214, 102, 0.42); } .agent-card-wait { border-color: rgba(135, 159, 180, 0.38); }
.inquiry-panel { border: 1px solid rgba(0, 255, 194, 0.25); border-radius: 18px; padding: 1rem; background: rgba(3, 22, 30, 0.88); margin-top: 0.8rem; }
.codex-panel { border: 1px solid rgba(80, 220, 255, 0.18); border-radius: 18px; padding: 1rem; background: rgba(5, 13, 24, 0.92); }
.good { color: #39f5a3; font-weight: 800; } .warn { color: #ffd166; font-weight: 800; } .danger { color: #ff5c7a; font-weight: 800; } .muted { color: #8db7c8; }
div[data-testid="stMetric"] { background: rgba(6,20,34,.75); border: 1px solid rgba(80,220,255,.15); padding: 8px 10px; border-radius: 12px; }
.screen-section { width: 100%; display: block; padding: 0.35rem 0 0.75rem 0; margin: 0; border-bottom: 1px solid rgba(80, 220, 255, 0.08); }
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
.clickable-coin-card { cursor: pointer; }
.coin-card-hitbox { position: absolute; inset: 0; z-index: 8; border-radius: 18px; background: rgba(255, 255, 255, 0); text-decoration: none; }
.coin-card-hitbox:hover { background: rgba(57, 245, 163, 0.035); }
.tv-chart-shell { width: 100%; height: 900px; min-height: 900px; border: 1px solid rgba(80, 220, 255, 0.18); border-radius: 18px; overflow: hidden; background: #0b0f14; }
.coin-overview-card.buy { border-color: rgba(0, 255, 160, 0.48); }
.coin-overview-card.shadow { border-color: rgba(255, 214, 102, 0.45); }
.coin-overview-card.wait { border-color: rgba(135, 159, 180, 0.38); }
.coin-overview-card.blocked { border-color: rgba(255, 92, 122, 0.45); }
.rank-badge { display: inline-block; border: 1px solid rgba(57, 245, 163, 0.45); border-radius: 999px; padding: 0.18rem 0.5rem; font-size: 0.78rem; color: #39f5a3; background: rgba(57, 245, 163, 0.08); margin-right: 0.35rem; }
.viability-score { font-size: 1.35rem; font-weight: 900; color: #e8fbff; }
.viability-reason { color: #8db7c8; font-size: 0.86rem; line-height: 1.25rem; }
.context-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.8rem; }
.context-card { border: 1px solid rgba(80, 220, 255, 0.18); border-radius: 18px; padding: 0.9rem; background: rgba(6, 18, 32, 0.86); }
.context-card h3 { margin-top: 0; margin-bottom: 0.45rem; }
.watch-list { border-left: 3px solid rgba(57, 245, 163, 0.75); padding: 0.65rem 0.8rem; background: rgba(57, 245, 163, 0.06); border-radius: 12px; margin-top: 0.6rem; }
.live-pulse { display: inline-block; width: 0.65rem; height: 0.65rem; border-radius: 50%; background: #39f5a3; box-shadow: 0 0 14px rgba(57, 245, 163, 0.9); margin-right: 0.4rem; }
.agent-ticker { border: 1px solid rgba(0, 255, 194, 0.24); border-radius: 18px; padding: 0.9rem; background: rgba(3, 22, 30, 0.88); margin: 0.75rem 0; }
.agent-row { border-left: 3px solid rgba(80, 220, 255, 0.35); padding: 0.55rem 0.75rem; margin: 0.45rem 0; background: rgba(6, 20, 34, 0.55); border-radius: 12px; }
.agent-row.active { border-left-color: #39f5a3; box-shadow: 0 0 18px rgba(57, 245, 163, 0.12); }
@media (max-width: 900px) { .overview-grid { grid-template-columns: 1fr; } .context-grid { grid-template-columns: 1fr; } }
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


def freshness_class(age_sec: float, warn: float, danger: float) -> str:
    return "good" if age_sec <= warn else "warn" if age_sec <= danger else "danger"


def get_refresh_config() -> dict:
    """
    Viewer refresh is intentionally automatic and hidden.
    Keep the UI clean by removing the manual live-data settings.
    """
    return {
        "live_enabled": True,
        "interval_label": "2s",
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
    st.markdown('<div class="hud-header"><div class="hud-title">🛰️ Crypto Strategy HUD</div><div class="hud-subtitle">Strategy Arena for live crypto learning, agent consensus, and Coinbase-style chart context.</div></div>', unsafe_allow_html=True)


def get_available_products(snapshot: Dict[str, Any]) -> list[str]:
    coins = snapshot.get("coins", {}) or {}
    top = snapshot.get("top_products", []) or []
    available = [c for c in list(top) + [c for c in coins.keys() if c not in top] if str(c).strip()]
    return available


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

    with st.expander("Chart overlays", expanded=True):
        st.caption("Toggle every line or marker the bot can display on the chart.")
        cols = st.columns(2)
        toggles = {}
        for idx, (key, default) in enumerate(defaults.items()):
            with cols[idx % 2]:
                toggles[key] = st.checkbox(labels[key], value=default, key=f"overlay_{key}")
        return toggles


def render_held_positions(snapshot: Dict[str, Any]) -> None:
    coins = snapshot.get("coins", {}) or {}; held = [(p, dict(c or {})) for p,c in coins.items() if bool((c or {}).get("owns_position", False))]
    st.markdown('<div class="codex-panel"><b>Held Positions</b><div class="muted">Owned coins appear here before the selected chart.</div></div>', unsafe_allow_html=True)
    if not held: st.info("No currently held positions."); return
    cols = st.columns(min(4, max(1, len(held))))
    for idx, (product_id, coin) in enumerate(held):
        with cols[idx % len(cols)]:
            st.metric(product_id, f"{_safe_float(coin.get('price')):.8f}", f"exit {_safe_float(coin.get('net_after_exit_bps')):.2f} bps")


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
    latest_decision_id, latest_row = "", {}
    try:
        d=decisions_df[decisions_df["product_id"].astype(str)==str(product_id)].copy() if not decisions_df.empty and "product_id" in decisions_df.columns else pd.DataFrame()
        if not d.empty:
            d["ts_num"]=pd.to_numeric(d.get("ts"), errors="coerce"); d=d.sort_values("ts_num"); latest_row=d.iloc[-1].to_dict(); latest_decision_id=str(latest_row.get("decision_id", ""))
        v=council_votes_df[council_votes_df["product_id"].astype(str)==str(product_id)].copy() if not council_votes_df.empty and "product_id" in council_votes_df.columns else pd.DataFrame()
        if not v.empty and latest_decision_id and "decision_id" in v.columns:
            matched=v[v["decision_id"].astype(str)==latest_decision_id].copy(); v=matched if not matched.empty else v
        if not v.empty and (not latest_decision_id) and "decision_id" in v.columns:
            v["ts_num"]=pd.to_numeric(v.get("ts"), errors="coerce"); v=v.sort_values("ts_num"); latest_decision_id=str(v.iloc[-1].get("decision_id", "")); v=v[v["decision_id"].astype(str)==latest_decision_id].copy()
        sort_cols=[c for c in ["leaderboard_rank","agent"] if c in v.columns]
        if sort_cols: v=v.sort_values(sort_cols)
        return latest_decision_id, latest_row, v
    except Exception as exc:
        module_exception(MODULE_NAME, "latest_council_votes_for_coin_failed", exc, also_overall=True); return latest_decision_id, latest_row, pd.DataFrame()


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


def set_selected_coin(product_id: str) -> None:
    product_id = str(product_id or "").strip()
    if not product_id:
        return
    st.session_state["selected_coin"] = product_id
    st.session_state["strategy_arena_coin"] = product_id
    st.session_state["_scroll_to_strategy_arena"] = True


def apply_query_selected_coin(snapshot: Dict[str, Any]) -> None:
    """Reads ?coin=PRODUCT and applies it to the Strategy Arena selection."""
    try:
        available = get_available_products(snapshot)
        raw_coin = st.query_params.get("coin", None)
        if isinstance(raw_coin, list):
            raw_coin = raw_coin[0] if raw_coin else None
        selected = unquote(str(raw_coin or "")).strip()
        if selected and selected in available:
            st.session_state["selected_coin"] = selected
            st.session_state["strategy_arena_coin"] = selected
            st.session_state["_scroll_to_strategy_arena"] = True
    except Exception as exc:
        module_exception(MODULE_NAME, "apply_query_selected_coin_failed", exc, also_overall=False)


def scroll_to_strategy_arena_if_requested() -> None:
    if not st.session_state.pop("_scroll_to_strategy_arena", False):
        return
    components.html("""<script>const doc = window.parent.document; setTimeout(function() { const el = doc.getElementById("strategy-arena-anchor"); if (el) { el.scrollIntoView({behavior: "smooth", block: "start"}); } }, 150);</script>""", height=0)

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
    st.dataframe(trades[cols] if cols else trades, use_container_width=True, hide_index=True)


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
    coins = snapshot.get("coins", {}) or {}; products = list(snapshot.get("top_products") or []); products += [p for p in coins.keys() if p not in products]; rows = []
    for product in products:
        coin = dict(coins.get(product, {}) or {}); market = latest_row_for_product(market_df, product); decision = latest_row_for_product(decisions_df, product); target = latest_row_for_product(targets_df, product); latest_decision_id, _, votes = latest_council_votes_for_coin(council_votes_df, decisions_df, product)
        if not votes.empty:
            leanings = [vote_leaning(r) for r in votes.to_dict("records")]; consensus = max(["BUY", "SELL", "HOLD", "WAIT"], key=lambda x: leanings.count(x)); buy_votes = leanings.count("BUY"); wait_votes = leanings.count("WAIT"); sell_votes = leanings.count("SELL"); hold_votes = leanings.count("HOLD")
        else: consensus = "WAIT"; buy_votes = wait_votes = sell_votes = hold_votes = 0
        action = str(decision.get("action") or coin.get("decision_action") or market.get("buy_gate_tradeable") or "WAIT").upper()
        blocker = str(market.get("buy_gate_blocker") or coin.get("main_blocker") or coin.get("buy_blocker") or decision.get("reason") or decision.get("main_reason") or "")
        row = {"product_id": product, "price": _safe_float(market.get("mid") or coin.get("price")), "spread_bps": _safe_float(market.get("spread_bps") or coin.get("spread_bps")), "action": action, "consensus": consensus, "buy_votes": buy_votes, "sell_votes": sell_votes, "hold_votes": hold_votes, "wait_votes": wait_votes, "owns_position": boolish(coin.get("owns_position") or target.get("has_position")), "final_buy_score": _safe_float(decision.get("final_buy_score") or coin.get("final_buy_score")), "buy_threshold": _safe_float(decision.get("buy_threshold") or coin.get("buy_threshold")), "expected_utility_bps": _safe_float(decision.get("expected_utility_bps") or coin.get("expected_utility_bps")), "recommended_position_pct": _safe_float(decision.get("recommended_position_pct") or coin.get("recommended_position_pct")), "buy_gate_tradeable": market.get("buy_gate_tradeable"), "buy_gate_strict_ok": market.get("buy_gate_strict_ok"), "buy_gate_spread_ok": market.get("buy_gate_spread_ok"), "buy_gate_ev_ok": market.get("buy_gate_ev_ok"), "buy_gate_score_ok": market.get("buy_gate_score_ok"), "buy_gate_prob_ok": market.get("buy_gate_prob_ok"), "blocker": plain_reason(blocker), "decision_id": latest_decision_id}
        row["viability_score"], row["viability_reason"] = calculate_coin_viability(row); rows.append(row)
    rows.sort(key=lambda r: (_safe_float(r.get("viability_score")), boolish(r.get("buy_gate_tradeable")), _safe_float(r.get("expected_utility_bps")), _safe_float(r.get("final_buy_score")) - _safe_float(r.get("buy_threshold")), _safe_float(r.get("buy_votes"))), reverse=True)
    for idx, row in enumerate(rows, start=1): row["rank"] = idx
    return rows

def render_all_coin_landing_page(snapshot, market_df, decisions_df, council_votes_df, targets_df, refresh_config):
    rows = build_all_coin_rows(snapshot, market_df, decisions_df, council_votes_df, targets_df)
    readiness = snapshot.get("readiness", {}) or {}
    updated_ts = _safe_float(snapshot.get("updated_ts"))
    age = max(0.0, time.time() - updated_ts) if updated_ts > 0 else 999999.0
    st.markdown('<div class="hud-header"><div class="hud-title"><span class="live-pulse"></span>All-Coin Command Deck</div><div class="hud-subtitle">One-glance live stance across every tracked Coinbase product.</div></div>', unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("Tracked Coins", len(rows))
    cols[1].metric("Top Candidate", rows[0]["product_id"] if rows else "None")
    cols[2].metric("Top Viability", f'{rows[0]["viability_score"]:.1f}' if rows else "0.0")
    cols[3].metric("BUY Actions", sum(1 for r in rows if r["action"] == "BUY"))
    cols[4].metric("Snapshot Age", format_age(age))
    if rows:
        st.caption(f'Continuously sorted by viability score. Current leader: {rows[0]["product_id"]} — {rows[0]["viability_reason"]}')
    if readiness.get("high_fee_tier_active"):
        st.warning("Profit-First Fee-Aware Mode is active because Coinbase fees are high. The bot can still trade, but projected net profit must clear maker/taker fees, spread, and execution cost.")
    st.markdown('<div class="muted">Tap a coin card to open it in Strategy Arena.</div>', unsafe_allow_html=True)
    for i in range(0, len(rows), 3):
        cols = st.columns(3)
        for col, row in zip(cols, rows[i:i + 3]):
            product_id = str(row.get("product_id") or "")
            card_state = ("buy" if row.get("action") == "BUY" else "shadow" if "SHADOW" in str(row.get("action") or "") else "blocked" if row.get("blocker") else "wait")
            with col:
                encoded_coin = quote(product_id, safe="")
                card_href = f"?coin={encoded_coin}#strategy-arena-anchor"
                st.markdown(f'''
                    <div class="coin-overview-card {card_state} clickable-coin-card">
                        <a class="coin-card-hitbox" href="{card_href}" aria-label="Open {html.escape(product_id)} in Strategy Arena"></a>
                        <div style="font-size:1.25rem;font-weight:900;"><span class="rank-badge">#{row["rank"]}</span>{_html(product_id)}</div>
                        <div class="viability-score">Viability {row["viability_score"]:.1f}</div>
                        <div class="viability-reason">{_html(row["viability_reason"])}</div>
                        <div class="muted">Action: <b>{_html(row["action"])}</b> · Consensus: <b>{_html(row["consensus"])}</b></div>
                        <div>Votes: BUY <b>{row["buy_votes"]}</b> · WAIT <b>{row["wait_votes"]}</b> · SELL <b>{row["sell_votes"]}</b> · HOLD <b>{row["hold_votes"]}</b></div>
                        <div>Price: <b>{row["price"]:.8f}</b></div>
                        <div>Spread: <b>{row["spread_bps"]:.2f} bps</b></div>
                        <div>Buy score: <b>{row["final_buy_score"]:.3f}</b> / threshold <b>{row["buy_threshold"]:.3f}</b></div>
                        <div>Utility: <b>{row["expected_utility_bps"]:.2f} bps</b></div>
                        <div class="muted">Blocker: {_html(row["blocker"][:160] or "No blocker published.")}</div>
                    </div>
                    ''', unsafe_allow_html=True)
    with st.expander("All-coin sortable table", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def tradingview_symbol(product_id: str) -> str:
    return f"COINBASE:{str(product_id or '').replace('-', '')}"


def render_tradingview_chart(product_id: str) -> None:
    symbol = tradingview_symbol(product_id)

    html_block = f'''
    <div class="tv-chart-shell">
        <div class="tradingview-widget-container" style="height:100%;width:100%;">
            <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
            {{
                "autosize": true,
                "symbol": "{symbol}",
                "interval": "15",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "allow_symbol_change": false,
                "hide_side_toolbar": false,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "hide_volume": false,
                "calendar": false,
                "support_host": "https://www.tradingview.com"
            }}
            </script>
        </div>
    </div>
    '''

    components.html(html_block, height=930, scrolling=False)


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
        st.dataframe(votes[display_cols] if display_cols else votes, use_container_width=True, hide_index=True)


def render_agent_detail_panel(agent_name: str, votes: pd.DataFrame):
    sub = votes[votes["agent"].astype(str) == str(agent_name)] if not votes.empty and "agent" in votes.columns else pd.DataFrame()
    if sub.empty: return
    row = sub.iloc[-1].to_dict(); reason = str(row.get("reason", ""))
    st.markdown(f'''<div class="screen-card"><h3>{_html(agent_title_icon(agent_name))}</h3><b>Leaning:</b> {vote_leaning(row)}<br><b>Confidence:</b> {_safe_float(row.get("confidence")):.3f}<br><b>Strongest score:</b> {strongest_vote_score(row):.3f}<br><b>Plain-English reason:</b> {_html(plain_reason(reason))}<br></div>''', unsafe_allow_html=True)
    with st.expander("Raw analyst reason", expanded=False): st.text(reason)


def render_agent_roster_no_buttons(selected_coin: str, votes: pd.DataFrame):
    st.markdown("### Analyst Roster")
    if votes.empty: st.info("No analyst rows yet."); return
    focus_options = votes["agent"].dropna().astype(str).unique().tolist() if "agent" in votes.columns else []
    focused = st.selectbox("Focus analyst", focus_options, format_func=agent_title_icon, key=f"focus_analyst_{selected_coin}") if focus_options else ""
    rows = votes.to_dict("records")
    for i in range(0, len(rows), 3):
        cols = st.columns(3)
        for col, row in zip(cols, rows[i:i + 3]):
            agent = str(row.get("agent", "fallback")); leaning = vote_leaning(row).lower()
            with col:
                st.markdown(
                    f'''
                    <div class="agent-card agent-card-{leaning}">
                        <div class="agent-title">{_html(agent_title_icon(agent))}</div>
                        <div class="agent-metrics">
                            Leaning: <b>{leaning.upper()}</b><br>
                            Confidence: <b>{_safe_float(row.get("confidence")):.3f}</b><br>
                            Strongest score: <b>{strongest_vote_score(row):.3f}</b>
                        </div>
                        <div class="agent-summary">
                            {_html(agent_plain_summary(row))}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
    if focused: render_agent_detail_panel(focused, votes)


def render_topic_explanation(topic, selected_coin, votes, decisions_df, market_df, snapshot):
    coin = dict((snapshot.get("coins", {}) or {}).get(selected_coin, {}) or {}); _, drow, _ = latest_council_votes_for_coin(votes, decisions_df, selected_coin)
    if topic == "Why the bot is not buying live": st.info(plain_reason(coin.get("main_blocker") or coin.get("buy_blocker") or drow.get("reason", "No live-buy blocker is currently published.")))
    elif topic == "What would need to change": st.info("The bot needs fresher data, stronger expected utility, lower spread/fees, stronger agent confidence, or removal of the currently published blocker.")
    elif topic == "Chart levels": st.info(f"Published levels: POC {_safe_float(coin.get('point_of_control')):.8f}, VAH {_safe_float(coin.get('value_area_high')):.8f}, VAL {_safe_float(coin.get('value_area_low')):.8f}.")
    elif topic == "Fee impact": st.info("Profit-First Fee-Aware Mode is active: entries need projected net profit after Coinbase fees and spread." if coin.get("high_fee_tier_active") else "No explicit fee blocker is present in the selected snapshot row.")
    elif topic == "Agent disagreement": render_agent_disagreement_summary(votes)


def render_learning_console(selected_coin, votes, decisions_df, market_df, snapshot):
    st.markdown('<div class="screen-card"><h2>Learning Console</h2><div class="muted">Read what the bot is watching without clicking Ask buttons.</div></div>', unsafe_allow_html=True)
    topic = st.selectbox("Learning topic", ["Why the bot is not buying live", "What would need to change", "Chart levels", "Fee impact", "Agent disagreement", "Selected analyst details"], key=f"learning_topic_{selected_coin}")
    agent_options = [str(a) for a in votes["agent"].dropna().astype(str).unique().tolist()] if not votes.empty and "agent" in votes.columns else []
    if topic == "Selected analyst details" and agent_options:
        selected_agent = st.selectbox("Focus analyst", agent_options, format_func=agent_title_icon, key=f"focus_agent_{selected_coin}"); render_agent_detail_panel(selected_agent, votes); return
    render_topic_explanation(topic, selected_coin, votes, decisions_df, market_df, snapshot)


def render_strategy_screen(selected, timeframe, overlays, snapshot, market_df, decisions_df, council_votes_df, targets_df, trades_df, shadow_df):
    available = get_available_products(snapshot)
    if available:
        current = st.session_state.get("selected_coin", available[0])
        if current not in available:
            current = available[0]
            st.session_state["selected_coin"] = current
        if st.session_state.get("strategy_arena_coin") not in available:
            st.session_state["strategy_arena_coin"] = current
        selected = st.selectbox("Strategy Arena Coin", available, index=available.index(st.session_state.get("strategy_arena_coin", current)), key="strategy_arena_coin")
        st.session_state["selected_coin"] = selected
    st.markdown(f'<div class="hud-header"><div class="hud-title">Strategy Arena</div><div class="hud-subtitle">{_html(selected)} · chart first, analyst debate below.</div></div>', unsafe_allow_html=True)
    chart_choice = st.radio("Chart source", ["Bot overlay chart", "TradingView visual chart"], horizontal=True, key=f"chart_source_{selected}")
    if chart_choice == "TradingView visual chart":
        render_tradingview_chart(selected); st.info("TradingView is visual-only. The bot still learns from Coinbase API data and internal CSV history.")
    else:
        chart_df, chart_meta = load_chart_history(selected, timeframe); confirmed = confirmed_trades_only(trades_df, selected); target = latest_targets_for_coin(targets_df, selected)
        fig = build_coin_chart(chart_df, chart_meta, dict((snapshot.get("coins") or {}).get(selected, {}) or {}), market_df, confirmed, shadow_df, decisions_df, target, overlays, full_chart=True)
        st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{selected}_{timeframe}", config={"displayModeBar": True, "scrollZoom": True, "responsive": True})
    latest_decision_id, drow, votes = latest_council_votes_for_coin(council_votes_df, decisions_df, selected)
    render_agent_debate_stream(selected, latest_decision_id, drow, votes); render_agent_roster_no_buttons(selected, votes)


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
        main_watch = "Utility is the main weakness. The bot needs more upside after Coinbase fees and spread."
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
            st.dataframe(votes[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No council vote rows yet for this coin.")
    with st.expander("Raw context rows", expanded=False):
        st.write("Latest market row"); st.json(market); st.write("Latest decision row"); st.json(decision); st.write("Latest order-book row"); st.json(order_book); st.write("Latest target row"); st.json(target)


def render_debug_launch_screen(snapshot, market_df, decisions_df, council_votes_df, trades_df, orders_df):
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
        st.write(f"Latest action counts: {actions}")
        st.write(f"Latest blocker counts: {counts}")
        if counts["expected_utility_too_low"] > 0:
            st.warning("The main reason no trades are firing is expected_utility_too_low. That means Level 8 thinks the setup does not make enough net profit after Coinbase fees, spread, uncertainty, wait utility, and context penalties.")

    st.json(readiness)
    for name, df in [("trades", trades_df), ("orders", orders_df), ("market", market_df), ("council_decisions", decisions_df), ("council_votes", council_votes_df)]:
        with st.expander(name, expanded=False):
            st.dataframe(df.tail(100), use_container_width=True, hide_index=True) if not df.empty else st.info(f"{name}.csv has no rows yet.")

def render_live_dashboard(selected, timeframe, overlays, refresh_config):
    now_tick = int(time.time()); st.session_state["_viewer_live_tick"] = now_tick
    module_debug(MODULE_NAME, "viewer_live_tick", data={"tick": now_tick, "selected_coin": selected, "timeframe": timeframe, "interval_label": refresh_config.get("interval_label")}, level="DEBUG", also_overall=False)
    snapshot = load_viewer_snapshot(); market_df = load_csv(MARKET_CSV_PATH); decisions_df = load_csv(COUNCIL_DECISIONS_PATH); council_votes_df = load_csv(COUNCIL_VOTES_CSV_PATH); targets_df = load_csv(POSITION_TARGETS_PATH); trades_df = load_csv(TRADES_CSV_PATH); orders_df = load_csv(ORDERS_CSV_PATH); shadow_df = load_csv(SHADOW_TRADES_CSV_PATH); order_book_df = load_csv(ORDER_BOOK_SNAPSHOTS_PATH)
    with st.container(): st.markdown('<section class="screen-section command-deck">', unsafe_allow_html=True); render_all_coin_landing_page(snapshot, market_df, decisions_df, council_votes_df, targets_df, refresh_config); st.markdown('</section>', unsafe_allow_html=True)
    with st.container(): st.markdown('<div id="strategy-arena-anchor"></div>', unsafe_allow_html=True); scroll_to_strategy_arena_if_requested(); st.markdown('<section class="screen-section strategy-arena">', unsafe_allow_html=True); render_strategy_screen(selected, timeframe, overlays, snapshot, market_df, decisions_df, council_votes_df, targets_df, trades_df, shadow_df); st.markdown('</section>', unsafe_allow_html=True)
    with st.container(): st.markdown('<section class="screen-section deep-learning">', unsafe_allow_html=True); render_deep_learning_screen(selected, snapshot, market_df, decisions_df, council_votes_df, order_book_df, targets_df); st.markdown('</section>', unsafe_allow_html=True)
    with st.container(): st.markdown('<section class="screen-section debug-health">', unsafe_allow_html=True); render_debug_launch_screen(snapshot, market_df, decisions_df, council_votes_df, trades_df, orders_df); st.markdown('</section>', unsafe_allow_html=True)


def main() -> None:
    inject_crypto_game_css()
    refresh_config = get_refresh_config()
    render_crypto_header()
    snapshot_static = load_viewer_snapshot()
    apply_query_selected_coin(snapshot_static)
    selected = pick_selected_coin(snapshot_static)
    if not selected:
        st.info("Waiting for bot data. Start the bot and wait for viewer_snapshot.json to update.")
        return
    timeframe_label = st.radio("Chart Mode", ["1D · 1m", "7D · 1m", "30D · 15m", "90D · 1h", "2Y · 1d"], horizontal=True, key="chart_timeframe")
    timeframe = normalize_timeframe_label(timeframe_label)
    overlays = render_overlay_controls()
    run_every = run_every_value(refresh_config)
    if callable(getattr(st, "fragment", None)):
        @st.fragment(run_every=run_every)
        def live_dashboard_fragment():
            render_live_dashboard(selected, timeframe, overlays, refresh_config)
        live_dashboard_fragment()
    else:
        st.warning("Subtle auto-refresh needs Streamlit 1.37+. Manual refresh still works.")
        render_live_dashboard(selected, timeframe, overlays, refresh_config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer main crashed", exc, also_overall=True)
        try:
            st.error("Viewer crashed. Check debug/viewer.debug.log for the full traceback."); st.exception(exc)
        except Exception: pass
        raise
