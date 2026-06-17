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
    import streamlit.components.v1 as components
except Exception:
    components = None

try:
    from debug_tools import (
        module_debug,
        module_exception,
        debug_every,
        debug_timer,
        initialize_all_module_debug_logs,
        dataframe_debug_summary,
        viewer_snapshot_summary,
        csv_debug_summary,
        csv_runtime_status,
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
    def initialize_all_module_debug_logs(*args, **kwargs):
        pass
    def dataframe_debug_summary(*args, **kwargs):
        return {}
    def viewer_snapshot_summary(*args, **kwargs):
        return {}
    def csv_debug_summary(*args, **kwargs):
        return {}
    def csv_runtime_status(*args, **kwargs):
        return {}

MODULE_NAME = "viewer"

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
initialize_all_module_debug_logs(BASE_DIR)
module_debug(
    MODULE_NAME,
    "viewer_per_file_debug_logs_initialized",
    data={
        "debug_dir": os.path.join(BASE_DIR, "debug"),
        "viewer_snapshot_path": VIEWER_SNAPSHOT_PATH if "VIEWER_SNAPSHOT_PATH" in globals() else "",
    },
    level="INFO",
    also_overall=True,
)
module_debug(
    MODULE_NAME,
    "viewer_module_loaded",
    data={
        "base_dir": BASE_DIR,
        "file": __file__,
    },
    level="INFO",
    also_overall=True,
)
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

SNAPSHOT_STALE_WARN_SEC = 20.0
CHART_STALE_WARN_SEC_DAY = 180.0
CHART_STALE_WARN_SEC_WEEK = 3600.0
COUNCIL_STALE_WARN_SEC = 60.0

FAST_REFRESH_MS = 3000
FAST_TTL_SEC = 2
SLOW_TTL_SEC = 15
HISTORY_ROWS_PER_COIN = 600

st.set_page_config(page_title="Council Chamber", page_icon="♛", layout="wide", initial_sidebar_state="collapsed")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def inject_medieval_css() -> None:
    st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top, #2b241c 0%, #19140f 45%, #100c09 100%); color: #f2e6c9; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px; }
.council-title { font-size: 2.2rem; font-weight: 800; color: #f7e7b2; text-align: center; letter-spacing: 0.04em; margin-bottom: 0.2rem; }
.council-subtitle { text-align: center; color: #d5c49a; font-size: 1rem; margin-bottom: 1.2rem; }
.panel-card { background: linear-gradient(180deg, rgba(58,42,29,0.95) 0%, rgba(33,24,18,0.96) 100%); border: 1px solid rgba(196,168,112,0.35); border-radius: 18px; padding: 0.9rem 1rem; box-shadow: 0 6px 18px rgba(0,0,0,0.25); color: #f3ead6; }
.leader-card { background: linear-gradient(180deg, rgba(87,62,32,0.98) 0%, rgba(44,31,20,0.98) 100%); border: 1px solid rgba(230,198,113,0.50); border-radius: 22px; padding: 1rem 1.1rem; box-shadow: 0 10px 24px rgba(0,0,0,0.30); color: #fff3d1; }
.section-title { font-size: 1.1rem; font-weight: 800; color: #f7e7b2; margin-bottom: 0.5rem; }
.muted { color: #c7b897; font-size: 0.92rem; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.7rem; }
.metric-card { background: rgba(37,29,23,0.95); border: 1px solid rgba(196,168,112,0.30); border-radius: 14px; padding: 0.65rem 0.8rem; }
.metric-label { color: #cbb88c; font-size: 0.82rem; margin-bottom: 0.18rem; }
.metric-value { color: #fff3d1; font-size: 1.12rem; font-weight: 800; }
.fresh-good { color: #8be6b4; font-weight: 800; } .fresh-warn { color: #ffd37c; font-weight: 800; } .fresh-bad { color: #ff8c8c; font-weight: 800; }
.trade-table-note { color: #c7b897; font-size: 0.88rem; margin-top: 0.25rem; }
div[data-testid="stMetric"] { background: rgba(37,29,23,0.95); border: 1px solid rgba(196,168,112,0.24); padding: 8px 10px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=FAST_TTL_SEC, show_spinner=False)
def load_viewer_snapshot() -> Dict[str, Any]:
    try:
        if not os.path.exists(VIEWER_SNAPSHOT_CSV_SAFE_PATH):
            module_debug(
                MODULE_NAME,
                "viewer_snapshot_missing",
                data={"path": VIEWER_SNAPSHOT_CSV_SAFE_PATH, "startup_state": "waiting_for_first_bot_snapshot"},
                level="INFO",
                also_overall=False,
            )
            return {
                "updated_ts": 0.0,
                "coins": {},
                "top_products": [],
                "live_positions": [],
                "readiness": {"startup_state": "waiting_for_first_bot_snapshot"},
                "_viewer_snapshot_error": "viewer_snapshot.json not written yet",
                "_startup_waiting": True,
            }
        with open(VIEWER_SNAPSHOT_CSV_SAFE_PATH, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
        summary = viewer_snapshot_summary(snapshot)
        module_debug(MODULE_NAME, "viewer_snapshot_loaded", data=summary, level="INFO", also_overall=False)
        module_debug(MODULE_NAME, "viewer_snapshot_coin_count", data={"coin_count": summary.get("coin_count", 0)}, level="INFO", also_overall=False)
        return snapshot
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer_snapshot_corrupt", exc, data={"path": VIEWER_SNAPSHOT_CSV_SAFE_PATH, "traceback": traceback.format_exc()}, also_overall=True)
        return {"updated_ts": 0.0, "coins": {}, "top_products": [], "live_positions": [], "readiness": {}, "_viewer_snapshot_error": f"{type(exc).__name__}: {exc}"}


def csv_file_age_sec(path: str) -> float:
    try:
        if not os.path.exists(path):
            return 999999.0
        return max(0.0, time.time() - os.path.getmtime(path))
    except Exception:
        return 999999.0


def dataframe_latest_age_sec(frame: pd.DataFrame) -> float:
    try:
        if frame.empty or "ts" not in frame.columns:
            return 999999.0
        ts = pd.to_numeric(frame["ts"], errors="coerce").dropna()
        if ts.empty:
            return 999999.0
        latest = float(ts.max())
        return max(0.0, time.time() - latest)
    except Exception:
        return 999999.0


@st.cache_data(ttl=SLOW_TTL_SEC, show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    required_by_name = {
        "micro_history.csv": ["ts", "product_id", "open", "high", "low", "close", "volume"],
        "macro_day.csv": ["ts", "product_id", "open", "high", "low", "close", "volume"],
        "macro_week.csv": ["ts", "product_id", "open", "high", "low", "close", "volume"],
    }
    name = os.path.basename(path)
    required = required_by_name.get(name, [])
    try:
        if not os.path.exists(path):
            module_debug(MODULE_NAME, "viewer_csv_missing", data=csv_runtime_status(path, required_columns=required, name=name), level="INFO", also_overall=False)
            return pd.DataFrame()
        frame = pd.read_csv(path)
        status = {**csv_runtime_status(path, required_columns=required, name=name), **dataframe_debug_summary(frame, required_columns=required, name=name)}
        module_debug(MODULE_NAME, "viewer_csv_loaded", data=status, level="DEBUG", also_overall=False)
        return frame
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer_csv_load_failed", exc, data={"path": path, "traceback": traceback.format_exc(), **csv_runtime_status(path, required_columns=required, name=name)}, also_overall=True)
        return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def load_walk_forward_validation_df() -> pd.DataFrame:
    return load_csv(WALK_FORWARD_VALIDATION_PATH)


@st.cache_data(ttl=20, show_spinner=False)
def load_agent_ablation_df() -> pd.DataFrame:
    return load_csv(AGENT_ABLATION_PATH)


@st.cache_data(ttl=20, show_spinner=False)
def load_ai_feature_importance_df() -> pd.DataFrame:
    return load_csv(AI_FEATURE_IMPORTANCE_PATH)


@st.cache_data(ttl=FAST_TTL_SEC, show_spinner=False)
def load_council_votes_df() -> pd.DataFrame:
    return load_csv(COUNCIL_VOTES_CSV_PATH)


def freshness_class(age_sec: float) -> str:
    return "fresh-good" if age_sec <= 8 else "fresh-warn" if age_sec <= 20 else "fresh-bad"


def format_age(age_sec: float) -> str:
    age_sec = max(0.0, age_sec)
    return f"{age_sec:.0f}s old" if age_sec < 60 else f"{age_sec / 60.0:.1f}m old"


def coin_market_history(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    if df.empty or "product_id" not in df.columns:
        return pd.DataFrame()
    sub = df[df["product_id"].astype(str) == str(product_id)].copy()
    if "ts" in sub.columns:
        sub["ts"] = pd.to_datetime(sub["ts"], errors="coerce", utc=True)
        sub = sub.sort_values("ts")
    return sub.tail(HISTORY_ROWS_PER_COIN)


def confirmed_trades_only(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    if df.empty or "product_id" not in df.columns:
        return pd.DataFrame()
    sub = df[df["product_id"].astype(str) == str(product_id)].copy()
    for status_col in [c for c in ["status", "trade_status", "result", "fill_status"] if c in sub.columns]:
        mask = sub[status_col].astype(str).str.lower().isin(["filled", "confirmed", "executed", "success", "complete", "completed"])
        if mask.any():
            sub = sub[mask].copy()
            break
    if "ts" in sub.columns:
        sub["ts"] = pd.to_datetime(sub["ts"], errors="coerce", utc=True)
        sub = sub.sort_values("ts", ascending=False)
    return sub.head(20)


def confirmed_trades_for_coin(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    return confirmed_trades_only(df, product_id)

def latest_targets_for_coin(df: pd.DataFrame, product_id: str) -> Dict[str, Any]:
    if df.empty or "product_id" not in df.columns:
        return {}
    sub = df[df["product_id"].astype(str) == str(product_id)].copy()
    if sub.empty:
        return {}
    if "ts" in sub.columns:
        sub["ts"] = pd.to_datetime(sub["ts"], errors="coerce", utc=True)
        sub = sub.sort_values("ts")
    return sub.iloc[-1].to_dict()


def build_coin_chart(history_df: pd.DataFrame, coin_state: Dict[str, Any], confirmed_trades_df: pd.DataFrame, target_state: Dict[str, Any]) -> go.Figure:
    selected_product_for_debug = str(coin_state.get("product_id", "unknown"))
    debug_every(
        MODULE_NAME,
        f"chart_input:{selected_product_for_debug}",
        10.0,
        "chart_build_input",
        data={
            "product_id": selected_product_for_debug,
            "history": dataframe_debug_summary(history_df, required_columns=["ts"], name="history_df"),
            "confirmed_trades": dataframe_debug_summary(
                confirmed_trades_df,
                required_columns=["ts", "side", "price"],
                name="confirmed_trades_df",
            ),
            "target_keys": list(target_state.keys()) if isinstance(target_state, dict) else [],
            "coin_state_keys": list(coin_state.keys())[:120] if isinstance(coin_state, dict) else [],
        },
        level="DEBUG",
        also_overall=False,
    )
    has_volume = "volume" in history_df.columns if not history_df.empty else False
    if has_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.76, 0.24])
    else:
        fig = go.Figure()
    if history_df.empty:
        fig.update_layout(template="plotly_dark", title="No market history available", paper_bgcolor="#221a14", plot_bgcolor="#221a14", font=dict(color="#f3ead6"), height=520)
        try:
            debug_every(
                MODULE_NAME,
                f"chart_output:{selected_product_for_debug}",
                10.0,
                "chart_build_output",
                data={
                    "product_id": selected_product_for_debug,
                    "has_volume": False,
                    "trace_count": len(getattr(fig, "data", []) or []),
                    "height": 520,
                    "line_count": 0,
                    "empty_history": True,
                },
                level="DEBUG",
                also_overall=False,
            )
        except Exception:
            pass
        return fig
    ts_col = "ts" if "ts" in history_df.columns else history_df.columns[0]
    close_col = "close" if "close" in history_df.columns else "mid" if "mid" in history_df.columns else "price"
    if all(c in history_df.columns for c in ["open", "high", "low", close_col]):
        price_trace = go.Candlestick(x=history_df[ts_col], open=history_df["open"], high=history_df["high"], low=history_df["low"], close=history_df[close_col], name="Price")
        fig.add_trace(price_trace, row=1, col=1) if has_volume else fig.add_trace(price_trace)
    elif close_col in history_df.columns:
        price_trace = go.Scatter(x=history_df[ts_col], y=history_df[close_col], mode="lines", name="Price", line=dict(color="#8ee0ff", width=2))
        fig.add_trace(price_trace, row=1, col=1) if has_volume else fig.add_trace(price_trace)
    if has_volume:
        fig.add_trace(go.Bar(x=history_df[ts_col], y=pd.to_numeric(history_df["volume"], errors="coerce").fillna(0.0), name="Volume", marker=dict(color="rgba(232,193,111,0.35)")), row=2, col=1)
    lines = [
        ("VAH", target_state.get("value_area_high", coin_state.get("value_area_high", 0.0)), "#e8c16f", "dot"),
        ("VAL", target_state.get("value_area_low", coin_state.get("value_area_low", 0.0)), "#e8c16f", "dot"),
        ("POC", target_state.get("point_of_control", coin_state.get("point_of_control", 0.0)), "#ff9f5a", "dash"),
        ("Buy Target", target_state.get("target_buy_price", coin_state.get("selected_target_buy_price", 0.0)), "#78d6a8", "dash"),
        ("Sell Target", target_state.get("target_sell_price", coin_state.get("selected_target_sell_price", 0.0)), "#ff8e8e", "dash"),
        ("Min Profitable Exit", target_state.get("min_profitable_exit_price", coin_state.get("min_profitable_exit_price", 0.0)), "#facc15", "dash"),
        ("Stop", target_state.get("target_stop_price", coin_state.get("selected_target_stop_price", 0.0)), "#d46cff", "dot"),
        ("Avg Entry", coin_state.get("avg_entry", 0.0), "#a5dcff", "solid"),
    ]
    prior_poc = coin_state.get("previous_session_profile_poc", 0.0)
    prior_vah = coin_state.get("previous_session_profile_vah", 0.0)
    prior_val = coin_state.get("previous_session_profile_val", 0.0)
    prior_poc = _safe_float(prior_poc)
    prior_vah = _safe_float(prior_vah)
    prior_val = _safe_float(prior_val)
    if prior_poc > 0:
        lines.append(("Prior POC", prior_poc, "#facc15", "dash"))
    if prior_vah > 0:
        lines.append(("Prior VAH", prior_vah, "#fde68a", "dot"))
    if prior_val > 0:
        lines.append(("Prior VAL", prior_val, "#fde68a", "dot"))

    for label, y, color, dash in lines:
        y = _safe_float(y)
        if y > 0:
            if has_volume:
                fig.add_hline(y=y, line_width=1.5, line_color=color, line_dash=dash, annotation_text=label, annotation_position="right", row=1, col=1)
            else:
                fig.add_hline(y=y, line_width=1.5, line_color=color, line_dash=dash, annotation_text=label, annotation_position="right")
    if not confirmed_trades_df.empty and {"side", "price", "ts"}.issubset(confirmed_trades_df.columns):
        trades_for_chart = confirmed_trades_df.copy()
        trades_for_chart["price"] = pd.to_numeric(trades_for_chart["price"], errors="coerce")
        trades_for_chart = trades_for_chart.dropna(subset=["price"])
        buys = trades_for_chart[trades_for_chart["side"].astype(str).str.upper() == "BUY"]
        sells = trades_for_chart[trades_for_chart["side"].astype(str).str.upper() == "SELL"]
        if not buys.empty:
            buy_trace = go.Scatter(x=buys["ts"], y=buys["price"], mode="markers", name="Confirmed Buys", marker=dict(symbol="triangle-up", size=10, color="#78d6a8"))
            fig.add_trace(buy_trace, row=1, col=1) if has_volume else fig.add_trace(buy_trace)
        if not sells.empty:
            sell_trace = go.Scatter(x=sells["ts"], y=sells["price"], mode="markers", name="Confirmed Sells", marker=dict(symbol="triangle-down", size=10, color="#ff8e8e"))
            fig.add_trace(sell_trace, row=1, col=1) if has_volume else fig.add_trace(sell_trace)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#221a14", plot_bgcolor="#221a14", font=dict(color="#f3ead6"), legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0), margin=dict(l=15, r=15, t=55, b=15), height=650 if has_volume else 560, xaxis_title=None, yaxis_title="Price")
    if has_volume:
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    try:
        debug_every(
            MODULE_NAME,
            f"chart_output:{selected_product_for_debug}",
            10.0,
            "chart_build_output",
            data={
                "product_id": selected_product_for_debug,
                "has_volume": bool(has_volume),
                "trace_count": len(getattr(fig, "data", []) or []),
                "height": 650 if has_volume else 560,
                "line_count": len(lines),
            },
            level="DEBUG",
            also_overall=False,
        )
    except Exception:
        pass
    return fig


def render_header() -> None:
    st.markdown('<div class="council-title">Council Chamber</div>', unsafe_allow_html=True)
    st.markdown('<div class="council-subtitle">Volume-first market judgment, per coin, with live chamber updates.</div>', unsafe_allow_html=True)


def pick_selected_coin(snapshot: Dict[str, Any]) -> str | None:
    coins = snapshot.get("coins", {}) or {}
    top = snapshot.get("top_products", []) or []
    available = [c for c in list(top) + [c for c in coins.keys() if c not in top] if str(c).strip()]
    if not available:
        return None
    if st.session_state.get("selected_coin") not in available:
        st.session_state.selected_coin = available[0]
    selected = st.selectbox("Select Coin", options=available, index=available.index(st.session_state.selected_coin), key="selected_coin_selectbox")
    st.session_state.selected_coin = selected
    return selected


def render_held_positions(snapshot: Dict[str, Any]) -> None:
    coins = snapshot.get("coins", {}) or {}
    held = []
    for product_id, coin in coins.items():
        if bool(coin.get("owns_position", False)):
            held.append((product_id, dict(coin or {})))

    st.markdown('<div class="panel-card"><div class="section-title">Currently Held Positions</div><div class="muted">Owned coins are shown here before choosing a chart.</div></div>', unsafe_allow_html=True)
    if not held:
        st.info("No currently held positions.")
        return

    cols = st.columns(min(4, max(1, len(held))))
    for idx, (product_id, coin) in enumerate(held):
        with cols[idx % len(cols)]:
            bought_at = _safe_float(coin.get("avg_entry", 0.0))
            current_price = _safe_float(coin.get("price", 0.0))
            net_after_exit = _safe_float(coin.get("net_after_exit_bps", 0.0))
            peak = _safe_float(coin.get("peak_unrealized_bps", 0.0))
            sell_target = _safe_float(coin.get("selected_target_sell_price", coin.get("target_sell_price", coin.get("min_profitable_exit_price", 0.0))))
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{product_id}</div>
                    <div class="metric-value">{current_price:.8f}</div>
                    <div class="muted">
                        Bought at: <b>{bought_at:.8f}</b><br>
                        Looking to sell near: <b>{sell_target:.8f}</b><br>
                        Net after exit: <b>{net_after_exit:.2f} bps</b><br>
                        Peak unrealized: <b>{peak:.2f} bps</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_leader_and_council(snapshot: Dict[str, Any], selected_coin: str) -> None:
    coin = dict((snapshot.get("coins", {}) or {}).get(selected_coin, {}) or {})
    age = time.time() - _safe_float(coin.get("updated_ts", snapshot.get("updated_ts", 0.0)), 0.0)
    st.markdown(f"""
<div class="leader-card"><div class="section-title">♛ The King of the Council — Volume Profile Leader</div>
<div class="muted">Selected Coin: <b>{selected_coin}</b> &nbsp; | &nbsp; Freshness: <span class="{freshness_class(age)}">{format_age(age)}</span></div>
<div style="margin-top:0.6rem;" class="metric-grid">
<div class="metric-card"><div class="metric-label">Leader Buy Score</div><div class="metric-value">{_safe_float(coin.get('volume_profile_leader_buy_score', coin.get('leader_buy_score'))):.3f}</div></div>
<div class="metric-card"><div class="metric-label">Leader Sell Score</div><div class="metric-value">{_safe_float(coin.get('volume_profile_leader_sell_score', coin.get('leader_sell_score'))):.3f}</div></div>
<div class="metric-card"><div class="metric-label">Leader Hold Score</div><div class="metric-value">{_safe_float(coin.get('volume_profile_leader_hold_score', coin.get('leader_hold_score'))):.3f}</div></div>
<div class="metric-card"><div class="metric-label">Council Mode</div><div class="metric-value">{coin.get('council_mode', '')}</div></div>
</div>
<div style="margin-top:0.6rem;" class="metric-grid">
<div class="metric-card"><div class="metric-label">Prev Session Reaction</div><div class="metric-value" style="font-size:1rem;">{coin.get('previous_session_profile_reaction_state', '')}</div></div>
<div class="metric-card"><div class="metric-label">Prev Session Bias</div><div class="metric-value">{coin.get('previous_session_profile_bias', '')}</div></div>
<div class="metric-card"><div class="metric-label">Quant Boundary</div><div class="metric-value" style="font-size:1rem;">{coin.get('quant_boundary_state', '')}</div></div>
<div class="metric-card"><div class="metric-label">Stationarity</div><div class="metric-value">{_safe_float(coin.get('quant_stationarity_score')):.3f}</div></div>
</div>
<div style="margin-top:0.75rem;" class="muted"><b>Decision:</b> {coin.get('decision_action', '')}<br/><b>Volume State:</b> {coin.get('value_acceptance_state', '')} / {coin.get('volume_node_state', '')}<br/><b>Prev Session Levels:</b> POC {_safe_float(coin.get('previous_session_profile_poc')):.8f} / VAH {_safe_float(coin.get('previous_session_profile_vah')):.8f} / VAL {_safe_float(coin.get('previous_session_profile_val')):.8f}<br/><b>Quant:</b> forecast {_safe_float(coin.get('quant_forecast_return_bps')):.2f} bps; peer {coin.get('quant_peer_product', '')} {coin.get('quant_peer_state', '')}<br/><b>Leader Reason:</b> {str(coin.get('volume_profile_utility_reason', coin.get('leader_reason', '')))[:500]}</div></div>
""", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (label, key, desc) in zip(cols, [("Truth", "truth_score", "How strongly the chamber believes the setup."), ("Final Buy", "final_buy_score", "Final buy score for this specific coin."), ("Expected Utility", "expected_utility_bps", "Net expected value after costs."), ("Buy vs Wait", "buy_vs_wait_edge_bps", "Whether action beats waiting.")]):
        with col:
            st.markdown(f'<div class="panel-card"><div class="section-title">{label}</div><div style="font-size:1.35rem;font-weight:800;color:#fff3d1;">{_safe_float(coin.get(key)):.3f}</div><div class="muted">{desc}</div></div>', unsafe_allow_html=True)


def render_freshness_banner(snapshot: Dict[str, Any], selected_coin: str) -> None:
    coin = dict((snapshot.get("coins", {}) or {}).get(selected_coin, {}) or {})
    age = time.time() - _safe_float(coin.get("updated_ts", snapshot.get("updated_ts", 0.0)), 0.0)
    st.markdown(f'<div class="panel-card"><div class="section-title">Live Freshness</div><div class="{freshness_class(age)}">Selected Coin Data: {format_age(age)}</div><div class="muted">Fast snapshot refreshes are lightweight. Historical chart files refresh on a slower cache TTL for performance.</div></div>', unsafe_allow_html=True)


def render_targets_panel(coin: Dict[str, Any], target: Dict[str, Any]) -> None:
    st.markdown('<div class="panel-card"><div class="section-title">Price Targets / Sell Plan</div></div>', unsafe_allow_html=True)
    buy = _safe_float(target.get("target_buy_price", coin.get("selected_target_buy_price", 0.0)))
    sell = _safe_float(target.get("target_sell_price", coin.get("selected_target_sell_price", 0.0)))
    stop = _safe_float(target.get("target_stop_price", coin.get("selected_target_stop_price", 0.0)))
    cols = st.columns(4)
    cols[0].metric("Current Price", f"{_safe_float(coin.get('price')):.8f}")
    cols[1].metric("Buy Target", f"{buy:.8f}" if buy > 0 else "—")
    cols[2].metric("Sell Target", f"{sell:.8f}" if sell > 0 else "—")
    cols[3].metric("Stop / Risk Level", f"{stop:.8f}" if stop > 0 else "—")
    cols = st.columns(4)
    cols[0].metric("Recommended Position %", f"{_safe_float(coin.get('recommended_position_pct')):.2f}%")
    cols[1].metric("Position Owned", "Yes" if coin.get("owns_position", False) else "No")
    cols[2].metric("Position Qty", f"{_safe_float(coin.get('position_qty')):.8f}")
    cols[3].metric("Net After Exit (bps)", f"{_safe_float(coin.get('net_after_exit_bps')):.2f}")


def render_confirmed_trades(trades: pd.DataFrame) -> None:
    st.markdown('<div class="panel-card"><div class="section-title">Confirmed Trades</div></div>', unsafe_allow_html=True)
    if trades.empty:
        st.info("No confirmed trades found for this coin.")
        return
    cols = [c for c in ["ts", "product_id", "side", "price", "qty", "size", "fee", "fee_usd", "order_id"] if c in trades.columns]
    st.dataframe(trades[cols] if cols else trades, use_container_width=True, hide_index=True)
    st.markdown('<div class="trade-table-note">Order attempts are intentionally hidden here. This section is for confirmed trades only.</div>', unsafe_allow_html=True)


def render_coin_analytics(coin: Dict[str, Any]) -> None:
    st.markdown('<div class="panel-card"><div class="section-title">Selected Coin Analytics</div></div>', unsafe_allow_html=True)
    rows = [[("Truth Score", "truth_score", ".3f"), ("Final Buy Score", "final_buy_score", ".3f"), ("Buy Threshold", "buy_threshold", ".3f"), ("Sell Threshold", "sell_threshold", ".3f")], [("Expected Utility (bps)", "expected_utility_bps", ".2f"), ("Buy vs Wait (bps)", "buy_vs_wait_edge_bps", ".2f"), ("POC Distance (bps)", "poc_distance_bps", ".2f"), ("Peak Unrealized (bps)", "peak_unrealized_bps", ".2f")], [("OB Imbalance", "order_book_imbalance", ".3f"), ("OB Top Depth ($)", "order_book_top_depth_usd", ".0f"), ("Spread Instability", "spread_instability_bps", ".2f"), ("Liquidity Risk", "liquidity_risk_score", ".3f")]]
    for row in rows:
        cols = st.columns(4)
        for col, (label, key, fmt) in zip(cols, row):
            col.metric(label, format(_safe_float(coin.get(key)), fmt))
    cols = st.columns(4)
    cols[0].metric("Value Acceptance", str(coin.get("value_acceptance_state", "")) or "—")
    cols[1].metric("Volume Node", str(coin.get("volume_node_state", "")) or "—")
    cols[2].metric("Low Volume Path Up (bps)", f"{_safe_float(coin.get('low_volume_path_up_bps')):.2f}")
    cols[3].metric("Low Volume Path Down (bps)", f"{_safe_float(coin.get('low_volume_path_down_bps')):.2f}")
    st.caption(str(coin.get("order_book_reason", ""))[:260] or "Order-book context pending.")


def render_agent_statements(votes_df: pd.DataFrame, selected_coin: str) -> None:
    st.markdown('<div class="panel-card"><div class="section-title">Council Agent Statements</div><div class="muted">The Volume Profile King leads, but the full council is still shown here.</div></div>', unsafe_allow_html=True)
    if votes_df.empty or "product_id" not in votes_df.columns:
        st.info("No council vote statements found yet.")
        return
    df = votes_df[votes_df["product_id"].astype(str) == str(selected_coin)].copy()
    if df.empty:
        st.info("No council vote statements found for this selected coin yet.")
        return
    if "ts" in df.columns:
        df["ts_num"] = pd.to_numeric(df["ts"], errors="coerce")
        latest_decision_ts = df["ts_num"].max()
        df = df[df["ts_num"] == latest_decision_ts].copy()
    important_order = {"volume_profile_leader": 0, "previous_session_volume_profile_agent": 1, "quant_boundary_agent": 2, "utility_leader": 3, "setup_performance_agent": 4, "session_liquidity": 5, "candle_context_agent": 6, "candle_sequence_agent": 7, "candle_exhaustion_agent": 8, "market_structure_agent": 9, "validated_liquidity_agent": 10, "fresh_zone_retest_agent": 11, "fair_value_gap_agent": 12, "volume_profile_agent": 13, "smt_divergence_agent": 14, "risk": 15, "truth": 16, "volume_profile_leader_exit": 17, "previous_session_profile_exit": 18, "quant_boundary_exit": 19, "spike_profit_protection": 20}

    def sort_key(agent: str) -> int:
        a = str(agent)
        if a in important_order:
            return important_order[a]
        if "session" in a or "liquidity" in a:
            return 3
        return 99

    if "agent" in df.columns:
        df["agent_sort"] = df["agent"].map(sort_key)
        df = df.sort_values(["agent_sort"])
    cards = df.head(16).to_dict("records")
    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, row in zip(cols, cards[i:i + 4]):
            agent = str(row.get("agent", "unknown"))
            buy = _safe_float(row.get("adjusted_buy_score", row.get("raw_buy_score", 0.0)))
            sell = _safe_float(row.get("adjusted_sell_score", row.get("raw_sell_score", 0.0)))
            hold = _safe_float(row.get("adjusted_hold_score", row.get("raw_hold_score", 0.0)))
            wait = _safe_float(row.get("adjusted_wait_score", row.get("raw_wait_score", 0.0)))
            confidence = _safe_float(row.get("confidence", 0.0))
            reason = str(row.get("reason", ""))[:280]
            with col:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">{agent}</div><div class="muted">Buy: <b>{buy:.3f}</b> · Sell: <b>{sell:.3f}</b><br>Hold: <b>{hold:.3f}</b> · Wait: <b>{wait:.3f}</b><br>Confidence: <b>{confidence:.3f}</b><br><span>{reason}</span></div></div>""", unsafe_allow_html=True)


def render_volume_context_note(coin: Dict[str, Any]) -> None:
    st.markdown(f"""
        <div class="panel-card">
            <div class="section-title">Volume Profile Context</div>
            <div class="muted">
                Value acceptance: <b>{coin.get("value_acceptance_state", "—")}</b><br>
                Volume node: <b>{coin.get("volume_node_state", "—")}</b><br>
                POC distance: <b>{_safe_float(coin.get("poc_distance_bps", 0.0)):.2f} bps</b><br>
                Low-volume path up: <b>{_safe_float(coin.get("low_volume_path_up_bps", 0.0)):.2f} bps</b><br>
                Low-volume path down: <b>{_safe_float(coin.get("low_volume_path_down_bps", 0.0)):.2f} bps</b><br>
                Unfair trade score: <b>{_safe_float(coin.get("unfair_trade_score", 0.0)):.3f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_order_book_context(coin: Dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="section-title">Order Book / Liquidity Context</div>
            <div class="muted">
                Available: <b>{bool(coin.get("order_book_available", False))}</b><br>
                Imbalance: <b>{_safe_float(coin.get("order_book_imbalance", 0.0)):.3f}</b><br>
                Bid depth: <b>${_safe_float(coin.get("order_book_bid_depth_usd", 0.0)):.2f}</b><br>
                Ask depth: <b>${_safe_float(coin.get("order_book_ask_depth_usd", 0.0)):.2f}</b><br>
                Top depth: <b>${_safe_float(coin.get("order_book_top_depth_usd", 0.0)):.2f}</b><br>
                Spread instability: <b>{_safe_float(coin.get("spread_instability_bps", 0.0)):.2f} bps</b><br>
                Liquidity risk: <b>{_safe_float(coin.get("liquidity_risk_score", 0.0)):.3f}</b><br>
                Market data age: <b>{_safe_float(coin.get("market_data_age_sec", 0.0)):.1f}s</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_transcript_strategy_context(coin: Dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="section-title">Previous-Session Profile + Quant Boundary</div>
            <div class="muted">
                Prior session: <b>{coin.get("previous_session_profile_session_key", "—")}</b><br>
                Reaction: <b>{coin.get("previous_session_profile_reaction_state", "—")}</b><br>
                Higher-timeframe bias: <b>{coin.get("previous_session_profile_bias", "—")}</b><br>
                Prior POC / VAH / VAL: <b>{_safe_float(coin.get("previous_session_profile_poc", 0.0)):.8f}</b> / <b>{_safe_float(coin.get("previous_session_profile_vah", 0.0)):.8f}</b> / <b>{_safe_float(coin.get("previous_session_profile_val", 0.0)):.8f}</b><br><br>
                Quant boundary: <b>{coin.get("quant_boundary_state", "—")}</b><br>
                Volatility cluster: <b>{coin.get("quant_volatility_cluster_state", "—")}</b><br>
                Stationarity score: <b>{_safe_float(coin.get("quant_stationarity_score", 0.0)):.3f}</b><br>
                Forecast return: <b>{_safe_float(coin.get("quant_forecast_return_bps", 0.0)):.2f} bps</b><br>
                Conditional volatility: <b>{_safe_float(coin.get("quant_conditional_volatility_bps", 0.0)):.2f} bps</b><br>
                Peer: <b>{coin.get("quant_peer_product", "—")}</b> / <b>{coin.get("quant_peer_state", "—")}</b> / z=<b>{_safe_float(coin.get("quant_peer_spread_z", 0.0)):.2f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def viewer_runtime_audit(
    *,
    snapshot: Dict[str, Any],
    selected: str,
    market_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    council_votes_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    walk_forward_df: pd.DataFrame,
    agent_ablation_df: pd.DataFrame,
    ai_importance_df: pd.DataFrame,
) -> Dict[str, Any]:
    coins = dict(snapshot.get("coins", {}) or {})
    coin = dict(coins.get(selected, {}) or {})
    required_coin_fields = [
        "product_id",
        "price",
        "truth_score",
        "final_buy_score",
        "expected_utility_bps",
        "buy_vs_wait_edge_bps",
        "volume_profile_leader_buy_score",
        "volume_profile_leader_sell_score",
        "value_acceptance_state",
        "volume_node_state",
        "previous_session_profile_reaction_state",
        "quant_boundary_state",
        "order_book_imbalance",
        "liquidity_risk_score",
    ]
    missing_coin_fields = [field for field in required_coin_fields if field not in coin]
    if market_df.empty or "product_id" not in market_df.columns:
        selected_market_rows = 0
    else:
        selected_market_rows = int((market_df["product_id"].astype(str) == str(selected)).sum())
    if trades_df.empty or "product_id" not in trades_df.columns:
        selected_trade_rows = 0
    else:
        selected_trade_rows = int((trades_df["product_id"].astype(str) == str(selected)).sum())
    if council_votes_df.empty or "product_id" not in council_votes_df.columns:
        selected_vote_rows = 0
    else:
        selected_vote_rows = int((council_votes_df["product_id"].astype(str) == str(selected)).sum())
    updated_ts = _safe_float(snapshot.get("updated_ts", 0.0))
    snapshot_age = max(0.0, time.time() - updated_ts) if updated_ts > 0 else 999999.0
    health = {
        "selected": selected,
        "snapshot_age_sec": snapshot_age,
        "coin_count": len(coins),
        "selected_coin_present": bool(coin),
        "missing_coin_fields": missing_coin_fields,
        "selected_market_rows": selected_market_rows,
        "selected_trade_rows": selected_trade_rows,
        "selected_vote_rows": selected_vote_rows,
        "market_df": dataframe_debug_summary(market_df, required_columns=["ts", "product_id"], name="market.csv"),
        "trades_df": dataframe_debug_summary(
            trades_df, required_columns=["ts", "product_id", "side", "price"], name="trades.csv"
        ),
        "targets_df": dataframe_debug_summary(targets_df, required_columns=["product_id"], name="position_targets.csv"),
        "decisions_df": dataframe_debug_summary(decisions_df, required_columns=["product_id"], name="council_decisions.csv"),
        "council_votes_df": dataframe_debug_summary(
            council_votes_df, required_columns=["product_id", "agent"], name="council_votes.csv"
        ),
        "orders_df": dataframe_debug_summary(orders_df, name="orders.csv"),
        "walk_forward_df": dataframe_debug_summary(walk_forward_df, name="walk_forward_validation.csv"),
        "agent_ablation_df": dataframe_debug_summary(agent_ablation_df, name="agent_ablation.csv"),
        "ai_importance_df": dataframe_debug_summary(ai_importance_df, name="ai_feature_importance.csv"),
    }
    level = "INFO"
    if missing_coin_fields or snapshot_age > 20 or selected_market_rows <= 0:
        level = "WARN"
    debug_every(
        MODULE_NAME,
        f"viewer_runtime_audit:{selected}",
        10.0,
        "viewer_runtime_audit",
        data=health,
        level=level,
        also_overall=(level == "WARN"),
    )
    return health

def main() -> None:
    inject_medieval_css()
    if st_autorefresh is not None:
        st_autorefresh(interval=FAST_REFRESH_MS, key="council_auto_refresh")
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    render_header()
    snapshot = load_viewer_snapshot()
    render_held_positions(snapshot)
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    selected = pick_selected_coin(snapshot)
    if not selected:
        st.warning("No coin data is available yet. Start the bot and wait for viewer_snapshot.json to update.")
        return
    market_df = load_csv(MARKET_CSV_PATH)
    trades_df = load_csv(TRADES_CSV_PATH)
    targets_df = load_csv(POSITION_TARGETS_PATH)
    decisions_df = load_csv(COUNCIL_DECISIONS_PATH)
    council_votes_df = load_council_votes_df()
    orders_df = load_csv(ORDERS_CSV_PATH)
    walk_forward_df = load_walk_forward_validation_df()
    agent_ablation_df = load_agent_ablation_df()
    ai_importance_df = load_ai_feature_importance_df()
    viewer_health = viewer_runtime_audit(
        snapshot=snapshot,
        selected=selected,
        market_df=market_df,
        trades_df=trades_df,
        targets_df=targets_df,
        decisions_df=decisions_df,
        council_votes_df=council_votes_df,
        orders_df=orders_df,
        walk_forward_df=walk_forward_df,
        agent_ablation_df=agent_ablation_df,
        ai_importance_df=ai_importance_df,
    )
    render_leader_and_council(snapshot, selected)
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    render_agent_statements(council_votes_df, selected)
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    render_freshness_banner(snapshot, selected)
    coin = dict((snapshot.get("coins", {}) or {}).get(selected, {}) or {})
    history = coin_market_history(market_df, selected)
    confirmed = confirmed_trades_only(trades_df, selected)
    target = latest_targets_for_coin(targets_df, selected)
    st.markdown('<div class="panel-card"><div class="section-title">Primary Market Chart</div><div class="muted">This is the main chart the chamber should prioritize for buy and sell judgment.</div></div>', unsafe_allow_html=True)
    st.plotly_chart(build_coin_chart(history, coin, confirmed, target), use_container_width=True)
    render_volume_context_note(coin)
    render_transcript_strategy_context(coin)
    render_order_book_context(coin)
    render_confirmed_trades(confirmed)
    render_targets_panel(coin, target)
    render_coin_analytics(coin)
    with st.expander("Account / Exposure", expanded=False):
        st.json({"live_positions": snapshot.get("live_positions", []), "top_products": snapshot.get("top_products", [])})
    with st.expander("Live Readiness", expanded=False):
        st.json(snapshot.get("readiness", {}))
    with st.expander("Viewer Debug Health", expanded=False):
        st.json(viewer_health)
    with st.expander("Raw Council Tables", expanded=False):
        if decisions_df.empty:
            st.info("No council decision log found.")
        else:
            sub = decisions_df[decisions_df["product_id"].astype(str) == str(selected)] if "product_id" in decisions_df.columns else decisions_df
            st.dataframe(sub.tail(50), use_container_width=True, hide_index=True)
    with st.expander("Backend Order Attempts", expanded=False):
        if orders_df.empty:
            st.info("No order attempts found.")
        else:
            st.dataframe(orders_df.tail(100), use_container_width=True, hide_index=True)
    with st.expander("Validation / Overfitting Controls", expanded=False):
        st.markdown("### Walk-Forward Validation")
        if not walk_forward_df.empty:
            st.dataframe(walk_forward_df.tail(50), use_container_width=True, hide_index=True)
        else:
            st.info("No walk-forward validation rows yet.")
        st.markdown("### Agent Ablation")
        if not agent_ablation_df.empty:
            agent_ablation_view = agent_ablation_df.copy()
            if "ablation_score" in agent_ablation_view.columns:
                agent_ablation_view["ablation_score_num"] = pd.to_numeric(
                    agent_ablation_view["ablation_score"],
                    errors="coerce",
                ).fillna(0.0)
                agent_ablation_view = agent_ablation_view.sort_values(
                    "ablation_score_num",
                    ascending=False,
                ).head(50)
            else:
                agent_ablation_view = agent_ablation_view.tail(50)
            st.dataframe(agent_ablation_view, use_container_width=True, hide_index=True)
        else:
            st.info("No agent ablation rows yet.")
    with st.expander("AI Feature Importance", expanded=False):
        if not ai_importance_df.empty:
            st.dataframe(ai_importance_df.head(40), use_container_width=True, hide_index=True)
        else:
            st.info("No AI feature importance report yet.")



def load_chart_history(product_id: str, timeframe: str) -> tuple[pd.DataFrame, dict]:
    timeframe = str(timeframe or "day").lower()
    if timeframe == "week":
        source_path, source_name = MACRO_WEEK_CSV_PATH, "macro_week.csv"
    else:
        source_path, source_name = MICRO_HISTORY_CSV_PATH, "micro_history.csv"
    required = ["ts", "product_id", "open", "high", "low", "close", "volume"]
    frame = load_csv(source_path)
    if timeframe == "day" and frame.empty:
        source_path, source_name = MACRO_DAY_CSV_PATH, "macro_day.csv"
        frame = load_csv(source_path)
    meta = {"product_id": product_id, "timeframe": timeframe, "source": source_name, "path": source_path, "rows_before_filter": int(len(frame)) if hasattr(frame, "__len__") else 0, "rows": 0, "has_volume": False, "age_sec": 999999.0, "missing_columns": []}
    if frame.empty:
        module_debug(MODULE_NAME, "chart_history_empty", data=meta, level="WARN", also_overall=False)
        return pd.DataFrame(), meta
    missing = [c for c in required if c not in frame.columns]
    meta["missing_columns"] = missing
    if missing:
        module_debug(MODULE_NAME, "chart_history_missing_columns", data=meta, level="WARN", also_overall=True)
        return pd.DataFrame(), meta
    out = frame[frame["product_id"].astype(str) == str(product_id)].copy()
    for col in ["ts", "open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
    out["dt"] = pd.to_datetime(out["ts"], unit="s", errors="coerce", utc=True)
    out = out.tail(7 * 24 * 4 + 50 if timeframe == "week" else 24 * 60 + 200)
    meta["rows"] = int(len(out)); meta["has_volume"] = bool("volume" in out.columns and pd.to_numeric(out["volume"], errors="coerce").fillna(0).sum() > 0); meta["age_sec"] = dataframe_latest_age_sec(out)
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
    except Exception:
        return 0.0


def _marker_df(df: pd.DataFrame, product_id: str, chart_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "product_id" not in df.columns or "ts" not in df.columns: return pd.DataFrame()
    out = df[df["product_id"].astype(str) == str(product_id)].copy()
    if out.empty: return out
    out["dt"] = pd.to_datetime(pd.to_numeric(out["ts"], errors="coerce"), unit="s", errors="coerce", utc=True)
    if "price" not in out.columns: out["price"] = 0.0
    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
    out.loc[out["price"] <= 0, "price"] = out.loc[out["price"] <= 0, "ts"].apply(lambda x: _nearest_close(chart_df, x))
    return out[out["price"] > 0]


def build_coin_chart(chart_df, chart_meta, coin_state, market_df, confirmed_trades_df, shadow_trades_df, decisions_df, target_state, overlay_toggles) -> go.Figure:
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
        if y>0:
            fig.add_hline(y=y, line_width=1.2, line_color=color, line_dash=dash, annotation_text=label, annotation_position="right", row=1, col=1); overlay_count += 1
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
    if overlay_toggles.get("confirmed_trades", True):
        t=_marker_df(confirmed_trades_df, product_id, chart_df)
        if not t.empty and "side" in t.columns:
            buys=t[t["side"].astype(str).str.upper()=="BUY"]; sells=t[t["side"].astype(str).str.upper()=="SELL"]
            if not buys.empty: fig.add_trace(go.Scatter(x=buys["dt"], y=buys["price"], mode="markers", name="Confirmed Buys", marker=dict(symbol="triangle-up", size=11, color="#78d6a8")), row=1, col=1); overlay_count += 1
            if not sells.empty: fig.add_trace(go.Scatter(x=sells["dt"], y=sells["price"], mode="markers", name="Confirmed Sells", marker=dict(symbol="triangle-down", size=11, color="#ff8e8e")), row=1, col=1); overlay_count += 1
    if overlay_toggles.get("shadow_trades", True):
        sh=_marker_df(shadow_trades_df, product_id, chart_df)
        if not sh.empty: fig.add_trace(go.Scatter(x=sh["dt"], y=sh["price"], mode="markers", name="Shadow Trades", marker=dict(symbol="circle", size=7, color="#ffd37c")), row=1, col=1); overlay_count += 1
    latest_decision_id=""
    if overlay_toggles.get("level8_markers", True):
        d=_marker_df(decisions_df, product_id, chart_df)
        if not d.empty:
            latest_decision_id=str(d.iloc[-1].get("decision_id", "")); actions=d["action"] if "action" in d.columns else ["L8"]*len(d)
            fig.add_trace(go.Scatter(x=d["dt"], y=d["price"], mode="markers+text", text=actions, textposition="top center", name="Level 8", marker=dict(symbol="diamond", size=9, color="#38bdf8")), row=1, col=1); overlay_count += 1
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font=dict(color="#d7dde8"), legend=dict(orientation="h", y=1.02), margin=dict(l=15,r=15,t=45,b=15), height=760, xaxis_rangeslider_visible=False)
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
            latest_decision_id=str(v.sort_values("ts").iloc[-1].get("decision_id", "")) if "ts" in v.columns else str(v.iloc[-1].get("decision_id", "")); v=v[v["decision_id"].astype(str)==latest_decision_id].copy()
        sort_cols=[c for c in ["leaderboard_rank","agent"] if c in v.columns]
        if sort_cols: v=v.sort_values(sort_cols)
        return latest_decision_id, latest_row, v
    except Exception as exc:
        module_exception(MODULE_NAME, "latest_council_votes_for_coin_failed", exc, also_overall=True); return latest_decision_id, latest_row, pd.DataFrame()


def vote_leaning(row):
    scores={"BUY":_safe_float(row.get("adjusted_buy_score")),"SELL":_safe_float(row.get("adjusted_sell_score")),"HOLD":_safe_float(row.get("adjusted_hold_score")),"WAIT":_safe_float(row.get("adjusted_wait_score"))}
    return max(scores, key=scores.get)

def strongest_vote_score(row): return max(_safe_float(row.get(k)) for k in ["adjusted_buy_score","adjusted_sell_score","adjusted_hold_score","adjusted_wait_score"])

def agent_title_icon(agent):
    m={"volume_profile_leader":"♛ King / Volume Profile Leader","volume_profile_agent":"🏰 Value Cartographer","trend":"⚔ Trend Knight","mean_reversion":"🜁 Mean Reversion Oracle","breakout":"🐎 Breakout Cavalier","ai_outcome":"🧠 AI Seer","execution":"⚖ Execution Chancellor","order_book_liquidity_agent":"🛡 Order Book Guard","previous_session_volume_profile_agent":"📜 Prior Session Archivist","quant_boundary_agent":"🔭 Quant Astrologer","candle_context_agent":"🕯 Candle Scribe","candle_sequence_agent":"🔥 Sequence Herald","candle_exhaustion_agent":"💤 Exhaustion Watchman","market_structure_agent":"🏗 Structure Mason","validated_liquidity_agent":"💧 Liquidity Scout","fresh_zone_retest_agent":"🌿 Fresh Zone Ranger","fair_value_gap_agent":"🕳 Fair Value Gap Keeper","smt_divergence_agent":"🪞 SMT Mirror Mage","setup_performance_agent":"📊 Setup Historian","utility_leader":"💰 Utility Treasurer","risk":"🧯 Risk Warden","exploration":"🧪 Exploration Alchemist","truth":"⚜ Truth Arbiter"}
    return m.get(str(agent), "🪑 Council Agent")

def whimsical_blurb(reason):
    r=str(reason or ""); low=r.lower(); blurb="The council keeps its counsel."
    if "spread" in low or "cost" in low or "fee" in low: blurb="The chamber eyes the toll bridge before marching."
    elif "poc" in low or "value" in low: blurb="The king sees price wandering near the value throne."
    elif "prob" in low or "odds" in low: blurb="The oracle asks for stronger odds before blessing a trade."
    elif "stale_market_data" in low or "stale" in low: blurb="The scouts demand fresher market scrolls."
    return f"{blurb} — {r[:160]}"


def render_leader_and_council(council_votes_df, decisions_df, selected_coin: str):
    did, drow, votes = latest_council_votes_for_coin(council_votes_df, decisions_df, selected_coin)
    action = drow.get("action", drow.get("final_action", "—")) if isinstance(drow, dict) else "—"
    module_debug(MODULE_NAME, "latest_council_votes", data={"product_id": selected_coin, "latest_decision_id": did, "vote_rows": int(len(votes))}, level="INFO", also_overall=False)
    st.markdown(f'<div class="leader-card"><div class="section-title">🏰 Medieval Council Chamber — {selected_coin}</div><div class="muted">Latest Level 8 action: <b>{action}</b> · decision_id: <b>{did or "—"}</b></div></div>', unsafe_allow_html=True)
    if votes.empty:
        st.info("No council vote statements found for this selected coin yet."); return did, votes
    king = votes[votes.get("agent", pd.Series(dtype=str)).astype(str)=="volume_profile_leader"] if "agent" in votes.columns else pd.DataFrame()
    rest = votes.drop(king.index) if not king.empty else votes
    if not king.empty:
        row=king.iloc[-1].to_dict(); st.markdown(f'<div class="leader-card"><b>{agent_title_icon("volume_profile_leader")}</b><br>Leaning: <b>{vote_leaning(row)}</b> · Confidence: <b>{_safe_float(row.get("confidence")):.3f}</b> · Strongest score: <b>{strongest_vote_score(row):.3f}</b><br><span class="muted">{whimsical_blurb(row.get("reason", ""))}</span></div>', unsafe_allow_html=True)
    cards=rest.to_dict("records")
    for i in range(0,len(cards),4):
        cols=st.columns(4)
        for col,row in zip(cols,cards[i:i+4]):
            agent=str(row.get("agent","unknown"))
            col.markdown(f'<div class="metric-card"><div class="metric-label">{agent}</div><b>{agent_title_icon(agent)}</b><br><span class="muted">Leaning: <b>{vote_leaning(row)}</b><br>Confidence: <b>{_safe_float(row.get("confidence")):.3f}</b><br>Strongest: <b>{strongest_vote_score(row):.3f}</b><br>{whimsical_blurb(row.get("reason", ""))}</span></div>', unsafe_allow_html=True)
    with st.expander("Raw council vote table", expanded=False): st.dataframe(votes, use_container_width=True, hide_index=True)
    return did, votes


def trigger_viewer_refresh(interval_ms: int) -> dict:
    tick=0; method="none"
    try:
        if st_autorefresh is not None:
            tick=st_autorefresh(interval=interval_ms, key="council_auto_refresh"); method="streamlit_autorefresh"
        elif components is not None:
            tick=int(st.session_state.get("_viewer_refresh_tick",0))+1; st.session_state["_viewer_refresh_tick"]=tick; components.html(f"<script>setTimeout(function(){{window.parent.location.reload();}}, {int(interval_ms)});</script>", height=0); method="components_js_reload"
        else:
            tick=int(st.session_state.get("_viewer_refresh_tick",0)); method="manual_refresh_only"
        module_debug(MODULE_NAME, "viewer_refresh_tick", data={"tick":tick,"method":method,"interval_ms":interval_ms}, level="DEBUG", also_overall=False); return {"tick":tick,"method":method,"interval_ms":interval_ms}
    except Exception as exc:
        module_exception(MODULE_NAME, "viewer_refresh_failed", exc, also_overall=True); return {"tick":tick,"method":"refresh_failed","interval_ms":interval_ms}

@contextmanager
def render_section(name: str):
    module_debug(MODULE_NAME, "render_section_start", data={"section": name}, level="DEBUG", also_overall=False)
    try:
        yield
        module_debug(MODULE_NAME, "render_section_end", data={"section": name}, level="DEBUG", also_overall=False)
    except Exception as exc:
        module_exception(MODULE_NAME, f"render_section_failed:{name}", exc, also_overall=True); raise


def _overlay_controls():
    defaults={"volume":True,"confirmed_trades":True,"shadow_trades":True,"profile":True,"prior_profile":True,"average_entry":True,"targets":True,"vwap":True,"structure":False,"level8_markers":True}
    labels={"volume":"Volume","confirmed_trades":"Confirmed buys/sells","shadow_trades":"Shadow trades","profile":"POC / VAH / VAL","prior_profile":"Prior-session POC/VAH/VAL","average_entry":"Average entry","targets":"Targets/sell plan","vwap":"VWAP / anchored VWAP","structure":"Trend / structure lines","level8_markers":"Level 8 action markers"}
    with st.expander("Chart overlays", expanded=False):
        return {k: st.checkbox(labels[k], value=v, key=f"overlay_{k}") for k,v in defaults.items()}


def _render_freshness(snapshot, chart_meta, votes, refresh_info, timeframe):
    now=time.time(); snap_ts=_safe_float(snapshot.get("updated_ts")); snap_age=max(0, now-snap_ts) if snap_ts>0 else 999999.0; council_age=dataframe_latest_age_sec(votes); chart_age=float(chart_meta.get("age_sec",999999.0));
    cols=st.columns(7); cols[0].metric("Current time", datetime.now(timezone.utc).strftime("%H:%M:%S UTC")); cols[1].metric("Snapshot updated", format_age(snap_age)); cols[2].metric("Snapshot age", format_age(snap_age)); cols[3].metric("Chart age", format_age(chart_age)); cols[4].metric("Council age", format_age(council_age)); cols[5].metric("Refresh", refresh_info.get("method")); cols[6].metric("Tick", refresh_info.get("tick"))
    if snap_age>SNAPSHOT_STALE_WARN_SEC: st.warning("Snapshot data is stale."); module_debug(MODULE_NAME,"stale_snapshot",data={"age_sec":snap_age},level="WARN",also_overall=False)
    limit=CHART_STALE_WARN_SEC_WEEK if timeframe=="week" else CHART_STALE_WARN_SEC_DAY
    if chart_age>limit: st.warning("Chart data is stale."); module_debug(MODULE_NAME,"stale_chart_data",data={"age_sec":chart_age,"timeframe":timeframe},level="WARN",also_overall=False)
    if council_age>COUNCIL_STALE_WARN_SEC: st.warning("Council votes are stale."); module_debug(MODULE_NAME,"stale_council_votes",data={"age_sec":council_age},level="WARN",also_overall=False)


def main() -> None:
    inject_medieval_css()
    with st.sidebar:
        interval_label=st.selectbox("Refresh interval", ["2s","3s","5s","10s","15s","30s"], index=1); interval_ms=int(interval_label.rstrip("s"))*1000
    refresh_info=trigger_viewer_refresh(interval_ms)
    with render_section("header"): render_header()
    with render_section("snapshot_load"):
        snapshot=load_viewer_snapshot()
        if snapshot.get("_startup_waiting"): st.info("Waiting for the first bot snapshot. This is normal during startup before the bot completes its first evaluation cycle.")
    with render_section("held_positions"): render_held_positions(snapshot)
    with render_section("selected_coin"):
        selected=pick_selected_coin(snapshot)
        if not selected: module_debug(MODULE_NAME,"missing_selected_coin",data={"snapshot_coin_count":len(snapshot.get("coins",{}) or {})},level="WARN",also_overall=False); st.warning("No coin data is available yet. Start the bot and wait for viewer_snapshot.json to update."); return
        module_debug(MODULE_NAME,"selected_coin_present",data={"selected":selected},level="INFO",also_overall=False)
        timeframe=st.radio("Chart mode", ["day","week"], horizontal=True, format_func=lambda x: x.title())
        overlays=_overlay_controls()
    market_df=load_csv(MARKET_CSV_PATH); trades_df=load_csv(TRADES_CSV_PATH); shadow_df=load_csv(SHADOW_TRADES_CSV_PATH); targets_df=load_csv(POSITION_TARGETS_PATH); decisions_df=load_csv(COUNCIL_DECISIONS_PATH); council_votes_df=load_council_votes_df(); orders_df=load_csv(ORDERS_CSV_PATH); walk_forward_df=load_walk_forward_validation_df(); agent_ablation_df=load_agent_ablation_df(); ai_importance_df=load_ai_feature_importance_df(); order_book_df=load_csv(ORDER_BOOK_SNAPSHOTS_PATH)
    coin=dict((snapshot.get("coins",{}) or {}).get(selected,{}) or {}); target=latest_targets_for_coin(targets_df, selected); chart_df, chart_meta=load_chart_history(selected, timeframe); confirmed=confirmed_trades_only(trades_df, selected)
    with render_section("council_chamber"): latest_decision_id, latest_votes=render_leader_and_council(council_votes_df, decisions_df, selected)
    _render_freshness(snapshot, chart_meta, latest_votes, refresh_info, timeframe)
    with render_section("chart"):
        st.markdown('<div class="panel-card"><div class="section-title">Coinbase-Style OHLCV Chart</div><div class="muted">Candles come from micro/macro OHLCV files; market.csv is telemetry only.</div></div>', unsafe_allow_html=True)
        fig=build_coin_chart(chart_df, chart_meta, coin, market_df, confirmed, shadow_df, decisions_df, target, overlays); st.plotly_chart(fig, use_container_width=True)
        if confirmed.empty: st.info("No confirmed trades yet.")
    with render_section("context_panels"):
        render_volume_context_note(coin); render_transcript_strategy_context(coin); render_order_book_context(coin)
    with render_section("confirmed_trades"): render_confirmed_trades(confirmed)
    with render_section("targets"): render_targets_panel(coin, target)
    with render_section("analytics"): render_coin_analytics(coin)
    viewer_health=viewer_runtime_audit(snapshot=snapshot, selected=selected, market_df=market_df, trades_df=trades_df, targets_df=targets_df, decisions_df=decisions_df, council_votes_df=council_votes_df, orders_df=orders_df, walk_forward_df=walk_forward_df, agent_ablation_df=agent_ablation_df, ai_importance_df=ai_importance_df)
    viewer_health.update({"selected_candle_rows": int(len(chart_df)), "selected_volume_rows": int(len(chart_df)) if chart_meta.get("has_volume") else 0, "chart_trace_count": len(fig.data), "active_timeframe": timeframe, "chart_overlay_toggles": overlays, "latest_decision_id": latest_decision_id, "latest_agent_vote_row_count": int(len(latest_votes))})
    with render_section("debug_health"):
        with st.expander("Viewer Debug Health", expanded=False): st.json(viewer_health)
    with render_section("validation"):
        with st.expander("Validation / Overfitting Controls", expanded=False):
            st.markdown("### Walk-Forward Validation"); st.info("Walk-forward validation is waiting for enough reviewed outcomes.") if walk_forward_df.empty else st.dataframe(walk_forward_df.tail(50), use_container_width=True, hide_index=True)
            st.markdown("### Agent Ablation"); st.info("Agent ablation is waiting for enough reviewed outcomes.") if agent_ablation_df.empty else st.dataframe(agent_ablation_df.tail(50), use_container_width=True, hide_index=True)
            st.markdown("### AI Feature Importance"); st.info("AI feature importance will appear after the AI brain has enough training rows.") if ai_importance_df.empty else st.dataframe(ai_importance_df.head(40), use_container_width=True, hide_index=True)
    with render_section("raw_tables"):
        with st.expander("Raw Tables", expanded=False):
            for name,df in [("council_decisions",decisions_df),("council_votes",council_votes_df),("market",market_df),("order_book_snapshots",order_book_df),("shadow_trades",shadow_df)]:
                st.markdown(f"### {name}"); st.dataframe(df.tail(100), use_container_width=True, hide_index=True) if not df.empty else st.info(f"{name}.csv has no rows yet.")
    with render_section("backend_orders"):
        with st.expander("Backend Order Attempts", expanded=False): st.info("No backend order attempts yet.") if orders_df.empty else st.dataframe(orders_df.tail(100), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        module_exception(
            MODULE_NAME,
            "viewer main crashed",
            exc,
            also_overall=True,
        )
        try:
            st.error("Viewer crashed. Check debug/viewer.debug.log for the full traceback.")
            st.exception(exc)
        except Exception:
            pass
        raise
