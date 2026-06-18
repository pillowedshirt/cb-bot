import json
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
.agent-card { border: 1px solid rgba(80, 220, 255, 0.20); border-radius: 16px; padding: 0.78rem; background: rgba(7, 18, 32, 0.92); min-height: 145px; }
.agent-card-buy { border-color: rgba(0, 255, 160, 0.45); } .agent-card-sell { border-color: rgba(255, 87, 116, 0.45); } .agent-card-hold { border-color: rgba(255, 214, 102, 0.42); } .agent-card-wait { border-color: rgba(135, 159, 180, 0.38); }
.inquiry-panel { border: 1px solid rgba(0, 255, 194, 0.25); border-radius: 18px; padding: 1rem; background: rgba(3, 22, 30, 0.88); margin-top: 0.8rem; }
.codex-panel { border: 1px solid rgba(80, 220, 255, 0.18); border-radius: 18px; padding: 1rem; background: rgba(5, 13, 24, 0.92); }
.good { color: #39f5a3; font-weight: 800; } .warn { color: #ffd166; font-weight: 800; } .danger { color: #ff5c7a; font-weight: 800; } .muted { color: #8db7c8; }
div[data-testid="stMetric"] { background: rgba(6,20,34,.75); border: 1px solid rgba(80,220,255,.15); padding: 8px 10px; border-radius: 12px; }
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
def load_viewer_snapshot() -> Dict[str, Any]:
    try:
        sig = file_signature(VIEWER_SNAPSHOT_CSV_SAFE_PATH)
        if not sig[1]:
            module_debug(MODULE_NAME, "viewer_snapshot_missing", data={"path": VIEWER_SNAPSHOT_CSV_SAFE_PATH}, level="INFO", also_overall=False)
            return {"updated_ts": 0.0, "coins": {}, "top_products": [], "live_positions": [], "readiness": {"startup_state": "waiting_for_first_bot_snapshot"}, "_startup_waiting": True}
        with open(VIEWER_SNAPSHOT_CSV_SAFE_PATH, "r", encoding="utf-8") as f: snapshot = json.load(f)
        module_debug(MODULE_NAME, "viewer_snapshot_loaded", data=viewer_snapshot_summary(snapshot), level="INFO", also_overall=False)
        return snapshot
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer_snapshot_corrupt", exc, data={"traceback": traceback.format_exc()}, also_overall=True)
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
    with st.sidebar:
        st.markdown("### Live Data")
        live_enabled = st.toggle("Live update data", value=True)
        interval_label = st.selectbox("Update interval", ["2s", "3s", "5s", "10s", "15s", "30s"], index=2)
        if st.button("Refresh data now"):
            st.cache_data.clear()
            st.session_state["_manual_refresh_tick"] = int(st.session_state.get("_manual_refresh_tick", 0)) + 1
    return {"live_enabled": live_enabled, "interval_label": interval_label, "manual_tick": int(st.session_state.get("_manual_refresh_tick", 0)), "fragment_supported": callable(getattr(st, "fragment", None))}


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


def pick_selected_coin(snapshot: Dict[str, Any]) -> str | None:
    coins = snapshot.get("coins", {}) or {}; top = snapshot.get("top_products", []) or []
    available = [c for c in list(top) + [c for c in coins.keys() if c not in top] if str(c).strip()]
    if not available: return None
    if st.session_state.get("selected_coin") not in available: st.session_state.selected_coin = available[0]
    selected = st.selectbox("Select Coin", options=available, index=available.index(st.session_state.selected_coin), key="selected_coin_selectbox")
    st.session_state.selected_coin = selected
    return selected


def render_overlay_controls():
    defaults={"volume":True,"confirmed_trades":True,"shadow_trades":True,"profile":True,"prior_profile":True,"average_entry":True,"targets":True,"vwap":True,"structure":False,"level8_markers":True}
    labels={"volume":"Volume","confirmed_trades":"Confirmed buys/sells","shadow_trades":"Shadow trades","profile":"POC / VAH / VAL","prior_profile":"Prior-session POC/VAH/VAL","average_entry":"Average entry","targets":"Targets/sell plan","vwap":"VWAP / anchored VWAP","structure":"Trend / structure lines","level8_markers":"Level 8 action markers"}
    with st.expander("Chart overlays", expanded=False):
        return {k: st.checkbox(labels[k], value=v, key=f"overlay_{k}") for k,v in defaults.items()}


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
    timeframe = str(timeframe or "day").lower()
    source_path, source_name = (MACRO_WEEK_CSV_PATH, "macro_week.csv") if timeframe == "week" else (MICRO_HISTORY_CSV_PATH, "micro_history.csv")
    required = ["ts", "product_id", "open", "high", "low", "close", "volume"]
    frame = load_csv(source_path)
    if timeframe == "day" and frame.empty:
        source_path, source_name = MACRO_DAY_CSV_PATH, "macro_day.csv"; frame = load_csv(source_path)
    meta = {"product_id": product_id, "timeframe": timeframe, "source": source_name, "path": source_path, "rows_before_filter": int(len(frame)) if hasattr(frame, "__len__") else 0, "rows": 0, "has_volume": False, "age_sec": 999999.0, "missing_columns": []}
    if frame.empty: return pd.DataFrame(), meta
    missing = [c for c in required if c not in frame.columns]; meta["missing_columns"] = missing
    if missing: module_debug(MODULE_NAME, "chart_history_missing_columns", data=meta, level="WARN", also_overall=True); return pd.DataFrame(), meta
    out = frame[frame["product_id"].astype(str) == str(product_id)].copy()
    for col in ["ts", "open", "high", "low", "close", "volume"]: out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
    out["dt"] = pd.to_datetime(out["ts"], unit="s", errors="coerce", utc=True)
    out = out.tail(7 * 24 * 4 + 50 if timeframe == "week" else 24 * 60 + 200)
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
    mapping=[("inside_value_area","Price is inside the value area, so this analyst is cautious about chasing."),("near_poc","Price is near the point of control, where chop is more likely."),("low_volume_node","Price may move faster through a low-volume area."),("stale_market_data","The bot does not trust the quote age enough for live trading."),("expected_utility_too_low","The projected reward is not strong enough after costs."),("probability_below","The probability model wants stronger odds before entry."),("spread","Spread and fees are reducing the net edge."),("fee","Spread and fees are reducing the net edge.")]
    for key, text in mapping:
        if key in low and text not in parts: parts.append(text)
    return " ".join(parts) or (r[:220] if r else "No reason text was published for this row.")


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


def render_strategy_arena(council_votes_df, decisions_df, selected_coin: str):
    latest_decision_id, drow, votes = latest_council_votes_for_coin(council_votes_df, decisions_df, selected_coin)
    action = drow.get("action", drow.get("final_action", "—")) if isinstance(drow, dict) else "—"
    st.markdown(f'<div class="hud-header"><div class="hud-title">Strategy Arena</div><div class="hud-subtitle">{_html(selected_coin)} · Latest Level 8 action: <b>{_html(action)}</b> · decision_id: <b>{_html(latest_decision_id or "—")}</b></div></div>', unsafe_allow_html=True)
    if votes.empty:
        st.info("No Strategy Arena vote statements found for this selected coin yet."); return latest_decision_id, votes
    chief = votes[votes.get("agent", pd.Series(dtype=str)).astype(str)=="volume_profile_leader"] if "agent" in votes.columns else pd.DataFrame()
    chief_row = chief.iloc[-1].to_dict() if not chief.empty else votes.iloc[0].to_dict()
    col1, col2 = st.columns([1.05,1.95])
    with col1:
        st.markdown(f'<div class="chief-card"><b>{agent_title_icon("volume_profile_leader")}</b><br>Leaning: <b>{vote_leaning(chief_row)}</b><br>Confidence: <b>{_safe_float(chief_row.get("confidence")):.3f}</b><br>Strongest score: <b>{strongest_vote_score(chief_row):.3f}</b><br><span class="muted">{_html(plain_reason(chief_row.get("reason", "")))}</span></div>', unsafe_allow_html=True)
        if st.button("Ask", key=f"ask_agent_{selected_coin}_{latest_decision_id}_volume_profile_leader"):
            st.session_state["inquiry_agent"] = "volume_profile_leader"; st.session_state["inquiry_decision_id"] = latest_decision_id
    with col2:
        st.markdown(f'<div class="codex-panel"><b>⚖️ Level 8 Arbiter</b><br>decision_id: <b>{_html(latest_decision_id or "—")}</b> · action: <b>{_html(action)}</b><br>Final buy score: <b>{_safe_float(drow.get("final_buy_score")):.3f}</b> · Expected utility: <b>{_safe_float(drow.get("expected_utility_bps")):.2f} bps</b><br><span class="muted">{_html(plain_reason(drow.get("reason", drow.get("main_reason", ""))))}</span></div>', unsafe_allow_html=True)
        if st.button("Ask Level 8 Arbiter", key=f"ask_agent_{selected_coin}_{latest_decision_id}_level8"):
            st.session_state["inquiry_agent"] = "level8_arbiter"; st.session_state["inquiry_decision_id"] = latest_decision_id
        render_agent_disagreement_summary(votes)
    cards=[r for r in votes.to_dict("records") if str(r.get("agent")) != "volume_profile_leader"]
    for i in range(0, len(cards), 4):
        cols=st.columns(4)
        for col,row in zip(cols,cards[i:i+4]):
            agent=str(row.get("agent","fallback")); lean=vote_leaning(row).lower()
            with col:
                st.markdown(f'<div class="agent-card agent-card-{lean}"><b>{_html(agent_title_icon(agent))}</b><br>Leaning: <b>{lean.upper()}</b><br>Confidence: <b>{_safe_float(row.get("confidence")):.3f}</b><br>Strongest: <b>{strongest_vote_score(row):.3f}</b><br><span class="muted">{_html(plain_reason(row.get("reason", "")))}</span></div>', unsafe_allow_html=True)
                if st.button("Ask", key=f"ask_agent_{selected_coin}_{latest_decision_id}_{agent}"):
                    st.session_state["inquiry_agent"] = agent; st.session_state["inquiry_decision_id"] = latest_decision_id
    return latest_decision_id, votes


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
    if coin.get("high_fee_tier_active"): st.warning("Strict mode: Coinbase fees are high, so the bot needs stronger edge before live entry.")
    return snapshot_age, chart_age, council_age, refresh_mode


def render_inquiry_panel(selected_coin, votes, decisions_df, market_df, snapshot):
    st.markdown('<div class="inquiry-panel"><b>Ask the Bot</b><div class="muted">Deterministic explanations from current Strategy Arena rows, market telemetry, and the viewer snapshot.</div></div>', unsafe_allow_html=True)
    buttons=[("Ask Chief Market Strategist","volume_profile_leader"),("Ask Level 8 Arbiter","level8_arbiter"),("Explain why no live buy","why_no_live_buy"),("Explain what would need to change","what_change"),("Explain chart levels","chart_levels"),("Explain fee impact","fee_impact"),("Explain agent disagreement","agent_disagreement")]
    cols=st.columns(4)
    for i,(label,key) in enumerate(buttons):
        if cols[i%4].button(label, key=f"inquiry_{selected_coin}_{key}"): st.session_state["inquiry_agent"] = key
    choice=st.session_state.get("inquiry_agent", "volume_profile_leader")
    coin=dict((snapshot.get("coins",{}) or {}).get(selected_coin,{}) or {})
    if choice == "level8_arbiter":
        _, drow, _ = latest_council_votes_for_coin(votes, decisions_df, selected_coin)
        st.info(f"Level 8 action: {drow.get('action', '—')}. Reason: {plain_reason(drow.get('reason', drow.get('main_reason', '')))}")
        return
    if choice in {"why_no_live_buy","what_change","chart_levels","fee_impact","agent_disagreement"}:
        text={"why_no_live_buy": plain_reason(coin.get("main_blocker", coin.get("buy_blocker", "expected_utility_too_low"))), "what_change":"The current rows suggest the bot needs fresher data, stronger expected utility, lower spread/fees, or stronger agent confidence when those blockers appear in the published reasons.", "chart_levels":f"Published levels: POC {_safe_float(coin.get('point_of_control')):.8f}, VAH {_safe_float(coin.get('value_area_high')):.8f}, VAL {_safe_float(coin.get('value_area_low')):.8f}.", "fee_impact":plain_reason("fee spread") if coin.get("high_fee_tier_active") or _safe_float(coin.get("taker_fee_rate", coin.get("taker_fee_bps"))) else "No explicit fee blocker is present in the selected snapshot row.", "agent_disagreement":"Review the Agent Consensus box for BUY/SELL/HOLD/WAIT counts and highest-confidence blocker."}[choice]
        st.info(text); return
    agent=choice
    row={}
    if not votes.empty and "agent" in votes.columns:
        m=votes[votes["agent"].astype(str)==str(agent)]
        if not m.empty: row=m.iloc[-1].to_dict()
    if not row and not votes.empty: row=votes.iloc[0].to_dict()
    reason=row.get("reason", "")
    st.markdown(f"""
<div class="codex-panel"><b>{_html(agent_title_icon(row.get('agent', agent)))}</b><br>
<b>What this analyst is watching:</b> {_html(row.get('agent', agent))}<br>
<b>Current leaning:</b> {vote_leaning(row) if row else '—'}<br>
<b>Confidence:</b> {_safe_float(row.get('confidence')):.3f}<br>
<b>Strongest score:</b> {strongest_vote_score(row) if row else 0:.3f}<br>
<b>Reason in plain English:</b> {_html(plain_reason(reason))}<br>
<b>Raw reason:</b> <span class="muted">{_html(reason)}</span><br>
<b>What would change its mind:</b> <span class="muted">A new row with stronger scores or a reason no longer mentioning the current blocker.</span></div>
""", unsafe_allow_html=True)


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


def render_live_dashboard(selected, timeframe, overlays, full_chart, refresh_config):
    snapshot=load_viewer_snapshot(); coin=dict((snapshot.get("coins",{}) or {}).get(selected,{}) or {})
    market_df=load_csv(MARKET_CSV_PATH); trades_df=load_csv(TRADES_CSV_PATH); shadow_df=load_csv(SHADOW_TRADES_CSV_PATH); targets_df=load_csv(POSITION_TARGETS_PATH); decisions_df=load_csv(COUNCIL_DECISIONS_PATH); council_votes_df=load_csv(COUNCIL_VOTES_CSV_PATH); order_book_df=load_csv(ORDER_BOOK_SNAPSHOTS_PATH)
    chart_df, chart_meta=load_chart_history(selected, timeframe); confirmed=confirmed_trades_only(trades_df, selected); target=latest_targets_for_coin(targets_df, selected)
    latest_decision_id, drow, latest_votes=latest_council_votes_for_coin(council_votes_df, decisions_df, selected)
    snapshot_age, chart_age, council_age, _ = render_status_strip(snapshot, selected, coin, chart_meta, latest_votes, drow, market_df, refresh_config)
    render_held_positions(snapshot)
    latest_decision_id, latest_votes=render_strategy_arena(council_votes_df, decisions_df, selected)
    st.markdown('<div class="codex-panel"><b>Coinbase-style Chart</b><div class="muted">Day mode uses micro_history.csv with macro_day.csv fallback. Week mode uses macro_week.csv. market.csv is telemetry only.</div></div>', unsafe_allow_html=True)
    fig=build_coin_chart(chart_df, chart_meta, coin, market_df, confirmed, shadow_df, decisions_df, target, overlays, full_chart=full_chart)
    trace_count=len(fig.data)
    st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{selected}_{timeframe}", config={"displayModeBar": True, "scrollZoom": True, "responsive": True})
    render_inquiry_panel(selected, latest_votes, decisions_df, market_df, snapshot)
    with st.expander("Learning Panels", expanded=False):
        st.markdown("### Agent Statements"); st.dataframe(latest_votes, use_container_width=True, hide_index=True) if not latest_votes.empty else st.info("Agent statements are pending.")
    with st.expander("Context Panels", expanded=False):
        st.json({k: coin.get(k) for k in ["value_acceptance_state","volume_node_state","previous_session_profile_reaction_state","quant_boundary_state","order_book_reason"]})
    render_confirmed_trades(confirmed); render_targets_panel(coin, target); render_coin_analytics(coin)
    orders_df=load_csv(ORDERS_CSV_PATH); walk_forward_df=load_csv(WALK_FORWARD_VALIDATION_PATH); agent_ablation_df=load_csv(AGENT_ABLATION_PATH); ai_importance_df=load_csv(AI_FEATURE_IMPORTANCE_PATH)
    viewer_health=viewer_runtime_audit(snapshot=snapshot, selected=selected, market_df=market_df, trades_df=trades_df, targets_df=targets_df, decisions_df=decisions_df, council_votes_df=council_votes_df, orders_df=orders_df, walk_forward_df=walk_forward_df, agent_ablation_df=agent_ablation_df, ai_importance_df=ai_importance_df, chart_meta=chart_meta, latest_votes=latest_votes, refresh_config=refresh_config)
    with st.expander("Debug Health", expanded=False): st.json(viewer_health)
    with st.expander("Validation / Overfitting", expanded=False):
        st.info("Walk-forward validation is waiting for enough reviewed outcomes.") if walk_forward_df.empty else st.dataframe(walk_forward_df.tail(50), use_container_width=True, hide_index=True)
        st.info("Agent ablation is waiting for enough reviewed outcomes.") if agent_ablation_df.empty else st.dataframe(agent_ablation_df.tail(50), use_container_width=True, hide_index=True)
        st.info("AI feature importance pending until enough labeled training rows exist.") if ai_importance_df.empty else st.dataframe(ai_importance_df.head(40), use_container_width=True, hide_index=True)
    with st.expander("Raw Tables", expanded=False):
        for name,df in [("council_decisions",decisions_df),("council_votes",council_votes_df),("market",market_df),("order_book_snapshots",order_book_df),("shadow_trades",shadow_df)]:
            st.markdown(f"### {name}"); st.dataframe(df.tail(100), use_container_width=True, hide_index=True) if not df.empty else st.info(f"{name}.csv has no rows yet.")
        if st.checkbox("Load heavy raw tables", value=False):
            candidate_df=load_csv(CANDIDATE_REPLAY_PATH); agent_adjustments_df=load_csv(AGENT_ADJUSTMENTS_PATH); agent_performance_df=load_csv(AGENT_PERFORMANCE_PATH); decision_audit_df=load_csv(DECISION_AUDIT_PATH)
            for name,df in [("candidate_replay",candidate_df),("agent_adjustments",agent_adjustments_df),("agent_performance",agent_performance_df),("decision_audit",decision_audit_df)]:
                st.markdown(f"### {name}"); st.dataframe(df.tail(100), use_container_width=True, hide_index=True) if not df.empty else st.info(f"{name}.csv has no rows yet.")
    with st.expander("Backend Order Attempts", expanded=False): st.info("No backend order attempts yet.") if orders_df.empty else st.dataframe(orders_df.tail(100), use_container_width=True, hide_index=True)
    module_debug(MODULE_NAME, "viewer_fragment_refresh", data={"selected_coin": selected, "refresh_mode": "fragment", "interval_label": refresh_config.get("interval_label"), "snapshot_age_sec": snapshot_age, "chart_age_sec": chart_age, "council_age_sec": council_age, "chart_rows": chart_meta.get("rows"), "council_rows": int(len(latest_votes)), "decision_id": latest_decision_id, "trace_count": trace_count}, level="DEBUG", also_overall=False)


def main() -> None:
    inject_crypto_game_css()
    refresh_config = get_refresh_config()
    render_crypto_header()
    snapshot_static = load_viewer_snapshot()
    selected = pick_selected_coin(snapshot_static)
    if not selected:
        st.info("Waiting for bot data. Start the bot and wait for viewer_snapshot.json to update.")
        return
    timeframe = st.radio("Chart Mode", ["day", "week"], horizontal=True, key="chart_timeframe", format_func=lambda x: x.title())
    full_chart = st.toggle("Full chart mode", value=False)
    overlays = render_overlay_controls()
    run_every = run_every_value(refresh_config)
    if callable(getattr(st, "fragment", None)):
        @st.fragment(run_every=run_every)
        def live_dashboard_fragment():
            render_live_dashboard(selected, timeframe, overlays, full_chart, refresh_config)
        live_dashboard_fragment()
    else:
        st.warning("Subtle auto-refresh needs Streamlit 1.37+. Manual refresh still works.")
        render_live_dashboard(selected, timeframe, overlays, full_chart, refresh_config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer main crashed", exc, also_overall=True)
        try:
            st.error("Viewer crashed. Check debug/viewer.debug.log for the full traceback."); st.exception(exc)
        except Exception: pass
        raise
