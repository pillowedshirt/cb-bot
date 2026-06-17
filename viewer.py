import json
import os
import time
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIEWER_SNAPSHOT_PATH = os.path.join(BASE_DIR, "viewer_snapshot.json")
VIEWER_SNAPSHOT_CSV_SAFE_PATH = VIEWER_SNAPSHOT_PATH
MARKET_CSV_PATH = os.path.join(BASE_DIR, "market.csv")
TRADES_CSV_PATH = os.path.join(BASE_DIR, "trades.csv")
POSITION_TARGETS_PATH = os.path.join(BASE_DIR, "position_targets.csv")
COUNCIL_DECISIONS_PATH = os.path.join(BASE_DIR, "council_decisions.csv")
COUNCIL_VOTES_CSV_PATH = os.path.join(BASE_DIR, "council_votes.csv")
ORDERS_CSV_PATH = os.path.join(BASE_DIR, "orders.csv")

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
    if not os.path.exists(VIEWER_SNAPSHOT_CSV_SAFE_PATH):
        return {"updated_ts": 0.0, "coins": {}, "top_products": [], "live_positions": []}
    with open(VIEWER_SNAPSHOT_CSV_SAFE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=SLOW_TTL_SEC, show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=FAST_TTL_SEC, show_spinner=False)
def load_council_votes_df() -> pd.DataFrame:
    if not os.path.exists(COUNCIL_VOTES_CSV_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(COUNCIL_VOTES_CSV_PATH)
    except Exception:
        return pd.DataFrame()


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
    has_volume = "volume" in history_df.columns if not history_df.empty else False
    if has_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.76, 0.24])
    else:
        fig = go.Figure()
    if history_df.empty:
        fig.update_layout(template="plotly_dark", title="No market history available", paper_bgcolor="#221a14", plot_bgcolor="#221a14", font=dict(color="#f3ead6"), height=520)
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
        buys = confirmed_trades_df[confirmed_trades_df["side"].astype(str).str.upper() == "BUY"]
        sells = confirmed_trades_df[confirmed_trades_df["side"].astype(str).str.upper() == "SELL"]
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


if __name__ == "__main__":
    main()
