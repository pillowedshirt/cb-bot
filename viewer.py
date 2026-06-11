import os
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


TZ = "America/Phoenix"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MARKET_CSV = os.path.join(BASE_DIR, "market.csv")
TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")
ORDERS_CSV = os.path.join(BASE_DIR, "orders.csv")
MACRO_LEVELS_CSV = os.path.join(BASE_DIR, "macro_levels.csv")
MACRO_FILES = {
    "Past day": os.path.join(BASE_DIR, "macro_day.csv"),
    "Past week": os.path.join(BASE_DIR, "macro_week.csv"),
}

st.set_page_config(
    page_title="Coinbase Bot Viewer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# Styling optimized for a 1080px-wide vertical monitor
# =============================================================================

st.markdown(
    """
<style>
:root {
  --bg: #060912;
  --panel: rgba(15, 23, 42, 0.82);
  --panel2: rgba(9, 14, 27, 0.92);
  --border: rgba(148, 163, 184, 0.18);
  --border2: rgba(148, 163, 184, 0.30);
  --text: #E5E7EB;
  --muted: #94A3B8;
  --soft: #CBD5E1;
  --blue: #60A5FA;
  --green: #34D399;
  --red: #FB7185;
  --yellow: #FBBF24;
  --purple: #A78BFA;
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 20% 0%, rgba(96,165,250,0.14), transparent 26%),
    radial-gradient(circle at 90% 10%, rgba(167,139,250,0.12), transparent 28%),
    linear-gradient(180deg, #060912 0%, #09111F 42%, #060912 100%) !important;
  color: var(--text) !important;
}

[data-testid="stHeader"] {
  background: rgba(6, 9, 18, 0.82) !important;
  backdrop-filter: blur(16px);
}

.block-container {
  padding-top: 0.7rem;
  padding-left: 0.7rem;
  padding-right: 0.7rem;
  padding-bottom: 1.2rem;
  max-width: 1060px;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(11,18,32,0.97), rgba(6,9,18,0.98)) !important;
  border-right: 1px solid var(--border);
}

.cb-hero {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 0.72rem 0.92rem;
  background:
    linear-gradient(135deg, rgba(96,165,250,0.14), rgba(167,139,250,0.08)),
    rgba(15,23,42,0.72);
  box-shadow: 0 14px 42px rgba(0,0,0,0.30);
  margin-bottom: 0.55rem;
}

.cb-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}

.cb-title {
  font-size: 1.34rem;
  font-weight: 820;
  line-height: 1.06;
  color: var(--text);
}

.cb-sub {
  color: var(--muted);
  font-size: 0.78rem;
  margin-top: 0.15rem;
}

.cb-pill {
  border: 1px solid var(--border2);
  border-radius: 999px;
  padding: 0.25rem 0.55rem;
  color: var(--soft);
  background: rgba(15,23,42,0.64);
  font-size: 0.74rem;
  white-space: nowrap;
}

.cb-section {
  font-size: 0.92rem;
  font-weight: 780;
  color: var(--text);
  margin: 0.55rem 0 0.33rem 0;
}

.cb-card {
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 0.63rem 0.70rem;
  background: var(--panel);
  box-shadow: 0 10px 32px rgba(0,0,0,0.24);
  margin-bottom: 0.50rem;
}

.cb-mini-card {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.50rem 0.58rem;
  background: rgba(15,23,42,0.72);
}

.cb-label {
  color: var(--muted);
  font-size: 0.70rem;
  line-height: 1.0;
  margin-bottom: 0.12rem;
}

.cb-value {
  color: var(--text);
  font-size: 1.02rem;
  line-height: 1.12;
  font-weight: 780;
  overflow-wrap: anywhere;
}

.cb-small {
  color: var(--muted);
  font-size: 0.72rem;
}

.cb-kv {
  display: grid;
  grid-template-columns: 0.92fr 1.08fr;
  gap: 0.25rem 0.60rem;
  font-size: 0.76rem;
}

.cb-kv .k { color: var(--muted); }
.cb-kv .v { color: var(--soft); font-weight: 650; overflow-wrap: anywhere; }

div[data-testid="stMetric"] {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.45rem 0.56rem;
  background: rgba(15,23,42,0.72);
  box-shadow: 0 8px 28px rgba(0,0,0,0.20);
}

[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  font-size: 0.68rem !important;
}

[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-size: 1.05rem !important;
}

.stPlotlyChart, [data-testid="stImage"], [data-testid="stPyplot"] {
  border-radius: 15px;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  background-color: rgba(15,23,42,0.82) !important;
  border-color: var(--border2) !important;
}

.stButton > button {
  border-radius: 999px;
  border: 1px solid var(--border2);
  background: linear-gradient(135deg, rgba(96,165,250,0.20), rgba(167,139,250,0.15));
  color: var(--text);
  padding: 0.35rem 0.80rem;
}

hr {
  border-color: var(--border);
  margin: 0.5rem 0;
}

[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  background: rgba(15,23,42,0.48) !important;
}

@media (max-width: 1100px) {
  .block-container {
    max-width: 100vw;
    padding-left: 0.45rem;
    padding-right: 0.45rem;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Data helpers
# =============================================================================

def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_dt_mst(ts_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(pd.to_numeric(ts_series, errors="coerce"), unit="s", utc=True)
    return dt.dt.tz_convert(TZ)


def fmt_money(x: Any, digits: int = 2) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"${float(x):,.{digits}f}"
    except Exception:
        return "—"


def fmt_num(x: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):,.{digits}f}{suffix}"
    except Exception:
        return "—"


def fmt_pct(x: Any, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x) * 100.0:.{digits}f}%"
    except Exception:
        return "—"


def compact_order_line(row: pd.Series) -> str:
    side = str(row.get("side", "—"))
    product = str(row.get("product_id", "—"))
    status = str(row.get("status", "—"))
    mode = str(row.get("mode", "—"))
    quote = fmt_money(row.get("requested_quote_usd", np.nan))
    qty = fmt_num(row.get("filled_qty", np.nan), 8)
    return f"{side} · {product} · {status} · {mode} · request {quote} · fill qty {qty}"


def compact_trade_line(row: pd.Series) -> str:
    side = str(row.get("side", "—"))
    product = str(row.get("product_id", "—"))
    price = fmt_money(row.get("price", np.nan), 4)
    qty = fmt_num(row.get("qty", np.nan), 8)
    pnl = fmt_money(row.get("net_pnl_usd", np.nan))
    return f"{side} · {product} · price {price} · qty {qty} · net P/L {pnl}"


def latest_macro_levels(macro_levels: pd.DataFrame, product: str, timeframe: str) -> dict:
    if macro_levels.empty or "product_id" not in macro_levels.columns or "timeframe" not in macro_levels.columns:
        return {}
    d = macro_levels[(macro_levels["product_id"] == product) & (macro_levels["timeframe"] == timeframe)].copy()
    if d.empty:
        return {}
    return d.sort_values("ts").iloc[-1].to_dict()


def plot_price(ax, d: pd.DataFrame, *, title: str, show_bid_ask: bool = False, trades: pd.DataFrame = pd.DataFrame()):
    ax.set_facecolor("#09111F")
    ax.plot(d["dt"], d["mid"], linewidth=1.6, label="mid", color="#93C5FD")

    if show_bid_ask and "bid" in d.columns and "ask" in d.columns:
        ax.plot(d["dt"], d["bid"], linewidth=0.75, label="bid", color="#34D399")
        ax.plot(d["dt"], d["ask"], linewidth=0.75, label="ask", color="#FB7185")

    if "anchored_vwap" in d.columns and not d["anchored_vwap"].isna().all():
        ax.plot(d["dt"], d["anchored_vwap"], linewidth=1.0, linestyle="--", label="VWAP", color="#C4B5FD")

    if "fair_value" in d.columns and not d["fair_value"].isna().all():
        ax.plot(d["dt"], d["fair_value"], linewidth=1.0, linestyle="--", label="fair", color="#FBBF24")

    if not trades.empty:
        if "event" in trades.columns:
            buys = trades[(trades["event"] == "BUY") & (trades["side"] == "BUY")]
            sells = trades[(trades["event"].isin(["SELL", "STARTUP_LIQUIDATION"])) & (trades["side"] == "SELL")]
        else:
            buys = trades[trades["side"] == "BUY"]
            sells = trades[trades["side"] == "SELL"]

        if not buys.empty:
            ax.scatter(buys["dt"], buys["price"], marker="^", s=58, color="#60A5FA", edgecolors="white", linewidths=0.5, zorder=6, label="BUY")
        if not sells.empty:
            ax.scatter(sells["dt"], sells["price"], marker="v", s=58, color="#F43F5E", edgecolors="white", linewidths=0.5, zorder=7, label="SELL")

    ax.set_title(title, color="#E5E7EB", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors="#94A3B8", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=d["dt"].dt.tz))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, alpha=0.13, color="#94A3B8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, fontsize=7, frameon=False, labelcolor="#CBD5E1")


def plot_macro(ax, df: pd.DataFrame, levels: dict, title: str):
    d = df.dropna(subset=["ts", "close"]).copy()
    d["dt"] = to_dt_mst(d["ts"])

    ax.set_facecolor("#09111F")
    ax.plot(d["dt"], d["close"], linewidth=1.35, label="close", color="#93C5FD")

    if levels:
        sup_lo = levels.get("support_zone_low")
        sup_hi = levels.get("support_zone_high")
        res_lo = levels.get("resistance_zone_low")
        res_hi = levels.get("resistance_zone_high")

        if pd.notna(sup_lo) and pd.notna(sup_hi):
            ax.axhspan(float(sup_lo), float(sup_hi), alpha=0.13, color="#34D399", label="support")
        if pd.notna(res_lo) and pd.notna(res_hi):
            ax.axhspan(float(res_lo), float(res_hi), alpha=0.13, color="#FB7185", label="resistance")

        for key, lbl, style, color in [
            ("vwap", "VWAP", "-", "#C4B5FD"),
            ("val", "VAL", "-", "#67E8F9"),
            ("vah", "VAH", "-", "#F0ABFC"),
            ("breakout", "breakout", ":", "#FBBF24"),
        ]:
            v = levels.get(key)
            if pd.notna(v):
                ax.axhline(float(v), linestyle=style, linewidth=0.9, color=color, label=lbl)

    ax.set_title(title, color="#E5E7EB", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors="#94A3B8", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=d["dt"].dt.tz))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, alpha=0.13, color="#94A3B8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=7, frameon=False, labelcolor="#CBD5E1")


def mini_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
<div class="cb-mini-card">
  <div class="cb-label">{label}</div>
  <div class="cb-value">{value}</div>
  <div class="cb-small">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# Live update controls
# =============================================================================

with st.sidebar:
    st.markdown("### Viewer controls")
    live_update = st.checkbox("Live data update", value=True)
    refresh_sec = st.slider("Update interval", 1, 15, 2, help="Uses Streamlit-native reruns, not browser page reload.")
    if st.button("Update now"):
        st.rerun()

    st.divider()
    window_minutes = st.slider("Micro window", 5, 240, 45, help="Shorter windows fit better on a vertical monitor.")
    show_bid_ask = st.checkbox("Show bid/ask lines", value=True)
    show_macro = st.checkbox("Show compact macro panel", value=True)
    show_debug_tables = st.checkbox("Show detailed tables", value=False)

if live_update:
    if st_autorefresh is None:
        st.warning("Install streamlit-autorefresh for live updates without browser page reload: pip install streamlit-autorefresh")
    else:
        refresh_count = st_autorefresh(interval=int(refresh_sec * 1000), key="live_data_update")
else:
    refresh_count = 0


# =============================================================================
# Load data
# =============================================================================

m = load_csv(MARKET_CSV)
t = load_csv(TRADES_CSV)
o = load_csv(ORDERS_CSV)
ml = load_csv(MACRO_LEVELS_CSV)

st.markdown(
    """
<div class="cb-hero">
  <div class="cb-row">
    <div>
      <div class="cb-title">Coinbase Bot Viewer</div>
      <div class="cb-sub">Vertical live console · probability sizing · orders · confirmed fills</div>
    </div>
    <div class="cb-pill">Live data update</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if m.empty:
    st.info("Waiting for market.csv. Start bot.py and let it write telemetry.")
    st.stop()

m = numeric(m, [
    "ts", "mid", "bid", "ask", "spread_bps",
    "exposures_usd", "position_qty", "avg_entry_price",
    "anchored_vwap", "fair_value", "sigma_bps", "weekly_bias",
    "cash_usd", "equity_usd",
    "entry_score", "entry_tier", "expected_net_edge_bps",
    "estimated_prob_up", "position_pct", "target_bps", "cost_bps",
    "current_maker_fee_bps", "current_taker_fee_bps",
    "dip_depth_score", "dip_speed_score", "reversal_score", "support_score",
    "room_score", "regime_score", "spread_penalty", "cost_penalty"
])

ml = numeric(ml, [
    "ts", "support_zone_low", "support_zone_high", "resistance_zone_low", "resistance_zone_high",
    "breakout", "range_low", "range_high", "prev_low", "prev_high", "vwap", "val", "vah", "price_now"
])

if not t.empty:
    t = numeric(t, ["ts", "price", "qty", "notional_usd", "cum_pnl_usd", "net_pnl_usd"])
    if "event" in t.columns:
        t = t[t["event"].isin(["BUY", "SELL", "STARTUP_LIQUIDATION"])].copy()
    if "qty" in t.columns:
        t = t[pd.to_numeric(t["qty"], errors="coerce").fillna(0.0) > 0].copy()
    if "price" in t.columns:
        t = t[pd.to_numeric(t["price"], errors="coerce").fillna(0.0) > 0].copy()
    if "ts" in t.columns and not t.empty:
        t["dt"] = to_dt_mst(t["ts"])

if not o.empty:
    o = numeric(o, [
        "ts", "requested_quote_usd", "requested_base_qty",
        "filled_qty", "avg_price", "filled_notional_usd", "fee_usd"
    ])
    if "ts" in o.columns and not o.empty:
        o["dt"] = to_dt_mst(o["ts"])


# =============================================================================
# Product selection and latest rows
# =============================================================================

cutoff = pd.Timestamp.utcnow().timestamp() - float(window_minutes) * 60.0
m_view = m[m["ts"] >= cutoff].copy()
products = sorted([p for p in m["product_id"].dropna().unique().tolist() if isinstance(p, str)])

if not products:
    st.warning("No products found in market.csv yet.")
    st.stop()

top_a, top_b = st.columns([0.48, 0.52], vertical_alignment="center")
with top_a:
    default_idx = products.index("BTC-USD") if "BTC-USD" in products else 0
    product = st.selectbox("Product", products, index=default_idx, label_visibility="collapsed")

m_prod = m_view[m_view["product_id"] == product].dropna(subset=["ts", "mid"]).copy()
if m_prod.empty:
    st.warning(f"No recent rows for {product} in the selected window.")
    st.stop()

m_prod["dt"] = to_dt_mst(m_prod["ts"])
scored_rows = m_prod.dropna(subset=["entry_score"]) if "entry_score" in m_prod.columns else pd.DataFrame()
latest_row = scored_rows.iloc[-1] if not scored_rows.empty else m_prod.iloc[-1]

last_seen = to_dt_mst(pd.Series([m["ts"].dropna().iloc[-1]])).iloc[0] if not m["ts"].dropna().empty else None
with top_b:
    st.caption(
        f"Last telemetry: {last_seen.strftime('%H:%M:%S %Z') if last_seen is not None else '—'} · "
        f"interval: {refresh_sec}s · cycle: {refresh_count}"
    )

t_prod = pd.DataFrame()
if not t.empty and "product_id" in t.columns:
    t_prod = t[(t["product_id"] == product) & (t["ts"] >= cutoff)].copy()

o_sorted = o.sort_values("ts", ascending=False).copy() if not o.empty and "ts" in o.columns else pd.DataFrame()
t_sorted = t.sort_values("ts", ascending=False).copy() if not t.empty and "ts" in t.columns else pd.DataFrame()


# =============================================================================
# Top recent order/trade strip
# =============================================================================

st.markdown('<div class="cb-section">Most recent activity</div>', unsafe_allow_html=True)

act1, act2 = st.columns(2)
with act1:
    if not o_sorted.empty:
        row = o_sorted.iloc[0]
        mini_card("Latest order attempt", compact_order_line(row), str(row.get("dt_mst", "")) or str(row.get("dt", "")))
    else:
        mini_card("Latest order attempt", "No order attempts yet", "")

with act2:
    if not t_sorted.empty:
        row = t_sorted.iloc[0]
        mini_card("Latest confirmed trade", compact_trade_line(row), str(row.get("dt_mst", "")) or str(row.get("dt", "")))
    else:
        mini_card("Latest confirmed trade", "No confirmed trades yet", "")


# =============================================================================
# Compact account and probability cards
# =============================================================================

cash = latest_row.get("cash_usd", np.nan)
equity = latest_row.get("equity_usd", np.nan)
exposure = latest_row.get("exposures_usd", np.nan)
position_qty = latest_row.get("position_qty", np.nan)
spread = latest_row.get("spread_bps", np.nan)

prob = latest_row.get("estimated_prob_up", np.nan)
pos_pct = latest_row.get("position_pct", np.nan)
target_bps = latest_row.get("target_bps", np.nan)
cost_bps = latest_row.get("cost_bps", np.nan)
maker_bps = latest_row.get("current_maker_fee_bps", np.nan)
taker_bps = latest_row.get("current_taker_fee_bps", np.nan)
fee_reason = latest_row.get("fee_tier_reason", "")
projected_size = float(equity) * float(pos_pct) if pd.notna(equity) and pd.notna(pos_pct) else np.nan

st.markdown('<div class="cb-section">Live account and sizing</div>', unsafe_allow_html=True)

a1, a2, a3, a4 = st.columns(4)
with a1:
    mini_card("Cash", fmt_money(cash), "Coinbase available USD")
with a2:
    mini_card("Equity", fmt_money(equity), "Cash + live positions")
with a3:
    mini_card("Exposure", fmt_money(exposure), product)
with a4:
    mini_card("Spread", fmt_num(spread, 2, " bps"), "top of book")

p1, p2, p3, p4 = st.columns(4)
with p1:
    mini_card("Probability up", fmt_pct(prob), "estimated")
with p2:
    mini_card("Position size", fmt_pct(pos_pct), "of total equity")
with p3:
    mini_card("Projected buy", fmt_money(projected_size), "if triggered")
with p4:
    mini_card("Position qty", fmt_num(position_qty, 8), product)

s1, s2 = st.columns([0.47, 0.53])
with s1:
    signal_html = f"""
<div class="cb-kv">
  <div class="k">Entry score</div><div class="v">{fmt_num(latest_row.get('entry_score', np.nan), 1)}</div>
  <div class="k">Tier</div><div class="v">{latest_row.get('entry_tier', '—')}</div>
  <div class="k">Expected edge</div><div class="v">{fmt_num(latest_row.get('expected_net_edge_bps', np.nan), 1, ' bps')}</div>
  <div class="k">Target</div><div class="v">{fmt_num(target_bps, 1, ' bps')}</div>
  <div class="k">Cost model</div><div class="v">{fmt_num(cost_bps, 1, ' bps')}</div>
</div>
"""
    st.markdown(f'<div class="cb-card">{signal_html}</div>', unsafe_allow_html=True)

with s2:
    fee_html = f"""
<div class="cb-kv">
  <div class="k">Maker fee</div><div class="v">{fmt_num(maker_bps, 2, ' bps')}</div>
  <div class="k">Taker fee</div><div class="v">{fmt_num(taker_bps, 2, ' bps')}</div>
  <div class="k">Fee source</div><div class="v">{fee_reason if fee_reason else '—'}</div>
  <div class="k">Reason</div><div class="v">{latest_row.get('entry_reason', '—')}</div>
</div>
"""
    st.markdown(f'<div class="cb-card">{fee_html}</div>', unsafe_allow_html=True)


# =============================================================================
# Main chart for vertical display
# =============================================================================

st.markdown(f'<div class="cb-section">{product} live micro chart</div>', unsafe_allow_html=True)

fig = plt.figure(figsize=(7.5, 2.65), facecolor="#060912")
ax = plt.gca()
plot_price(ax, m_prod, title=f"{product} · last {window_minutes} min", show_bid_ask=show_bid_ask, trades=t_prod)
st.pyplot(fig, clear_figure=True, use_container_width=True)


# =============================================================================
# Compact macro panel
# =============================================================================

if show_macro:
    st.markdown('<div class="cb-section">Macro structure</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Past day", "Past week"])
    for tab, label in zip(tabs, ["Past day", "Past week"]):
        with tab:
            df_macro = load_csv(MACRO_FILES[label])
            if df_macro.empty or "product_id" not in df_macro.columns:
                st.caption(f"Waiting for {label} macro file.")
            else:
                df_macro = numeric(df_macro, ["ts", "open", "high", "low", "close", "volume"])
                df_macro = df_macro[df_macro["product_id"] == product].dropna(subset=["ts", "close"]).copy()
                if df_macro.empty:
                    st.caption(f"No {label} macro data for {product}.")
                else:
                    timeframe = "day" if label == "Past day" else "week"
                    levels = latest_macro_levels(ml, product, timeframe)
                    figm = plt.figure(figsize=(7.5, 2.10), facecolor="#060912")
                    axm = plt.gca()
                    plot_macro(axm, df_macro, levels, f"{product} · {label}")
                    st.pyplot(figm, clear_figure=True, use_container_width=True)


# =============================================================================
# Compact recent rows, then expandable detailed tables
# =============================================================================

st.markdown('<div class="cb-section">Recent rows</div>', unsafe_allow_html=True)

r1, r2 = st.columns(2)
with r1:
    st.caption("Recent order attempts")
    if o_sorted.empty:
        st.write("No order attempts.")
    else:
        compact_cols = [
            c for c in [
                "dt_mst", "event", "product_id", "side", "mode",
                "requested_quote_usd", "ok", "status", "filled_qty", "avg_price", "raw_error"
            ] if c in o_sorted.columns
        ]
        st.dataframe(o_sorted[compact_cols].head(5), use_container_width=True, height=180, hide_index=True)

with r2:
    st.caption("Recent confirmed trades")
    if t_sorted.empty:
        st.write("No confirmed trades.")
    else:
        compact_cols = [
            c for c in [
                "dt_mst", "event", "product_id", "side", "qty", "price",
                "fee_usd", "net_pnl_usd", "note"
            ] if c in t_sorted.columns
        ]
        st.dataframe(t_sorted[compact_cols].head(5), use_container_width=True, height=180, hide_index=True)


with st.expander("Older order attempts"):
    if o_sorted.empty:
        st.write("No order attempts logged yet.")
    else:
        show_cols = [
            c for c in [
                "dt_mst", "event", "product_id", "side", "mode",
                "requested_quote_usd", "requested_base_qty",
                "ok", "status", "filled_qty", "avg_price",
                "filled_notional_usd", "fee_usd", "reason", "raw_error"
            ] if c in o_sorted.columns
        ]
        st.dataframe(o_sorted[show_cols].head(100), use_container_width=True, height=420, hide_index=True)

with st.expander("Older confirmed trades"):
    if t_sorted.empty:
        st.write("No confirmed trades logged yet.")
    else:
        show_cols = [
            c for c in [
                "dt_mst", "event", "product_id", "side", "qty", "price",
                "fee_usd", "gross_pnl_usd", "net_pnl_usd", "cum_pnl_usd",
                "entry_price", "exit_price", "exit_role", "note"
            ] if c in t_sorted.columns
        ]
        st.dataframe(t_sorted[show_cols].head(100), use_container_width=True, height=420, hide_index=True)

if show_debug_tables:
    with st.expander("Latest market telemetry debug"):
        debug_cols = [
            c for c in [
                "ts", "product_id", "bid", "ask", "mid", "spread_bps",
                "cash_usd", "equity_usd", "entry_score", "entry_tier",
                "estimated_prob_up", "position_pct", "target_bps", "cost_bps",
                "entry_reason"
            ] if c in m.columns
        ]
        st.dataframe(m.sort_values("ts", ascending=False)[debug_cols].head(200), use_container_width=True, height=420, hide_index=True)
