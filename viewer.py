import os
import math
import time
from typing import Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


TZ = "America/Phoenix"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MARKET_CSV = os.path.join(BASE_DIR, "market.csv")
TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")
ORDERS_CSV = os.path.join(BASE_DIR, "orders.csv")
MACRO_FILES = {
    "Past week (15m)": os.path.join(BASE_DIR, "macro_week.csv"),
    "Past day (1m)": os.path.join(BASE_DIR, "macro_day.csv"),
}
MACRO_LEVELS_CSV = os.path.join(BASE_DIR, "macro_levels.csv")

st.set_page_config(
    page_title="Coinbase Bot Viewer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# High-end dashboard styling
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
  --cb-bg: #070A12;
  --cb-panel: rgba(17, 24, 39, 0.78);
  --cb-panel-2: rgba(11, 18, 32, 0.92);
  --cb-border: rgba(148, 163, 184, 0.18);
  --cb-border-strong: rgba(148, 163, 184, 0.30);
  --cb-text: #E5E7EB;
  --cb-muted: #94A3B8;
  --cb-soft: #CBD5E1;
  --cb-accent: #60A5FA;
  --cb-accent-2: #A78BFA;
  --cb-good: #34D399;
  --cb-warn: #FBBF24;
  --cb-bad: #FB7185;
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 15% 0%, rgba(96, 165, 250, 0.16), transparent 30%),
    radial-gradient(circle at 85% 10%, rgba(167, 139, 250, 0.13), transparent 32%),
    linear-gradient(180deg, #070A12 0%, #0B1020 45%, #070A12 100%) !important;
  color: var(--cb-text) !important;
}

[data-testid="stHeader"] {
  background: rgba(7, 10, 18, 0.70) !important;
  backdrop-filter: blur(18px);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(11, 18, 32, 0.97), rgba(7, 10, 18, 0.98)) !important;
  border-right: 1px solid var(--cb-border);
}

[data-testid="stSidebar"] * {
  color: var(--cb-soft);
}

.block-container {
  padding-top: 1.4rem;
  padding-bottom: 3rem;
  max-width: 1500px;
}

.cb-hero {
  padding: 1.2rem 1.35rem;
  border: 1px solid var(--cb-border);
  border-radius: 24px;
  background:
    linear-gradient(135deg, rgba(96, 165, 250, 0.13), rgba(167, 139, 250, 0.08)),
    rgba(11, 18, 32, 0.74);
  box-shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
  margin-bottom: 1.0rem;
}

.cb-eyebrow {
  color: var(--cb-muted);
  font-size: 0.78rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}

.cb-title {
  color: var(--cb-text);
  font-size: 2.0rem;
  font-weight: 780;
  line-height: 1.12;
  margin-bottom: 0.35rem;
}

.cb-subtitle {
  color: var(--cb-muted);
  font-size: 0.98rem;
}

.cb-card {
  border: 1px solid var(--cb-border);
  border-radius: 20px;
  padding: 1rem 1.05rem;
  background: var(--cb-panel);
  box-shadow: 0 14px 46px rgba(0, 0, 0, 0.28);
  margin-bottom: 1rem;
}

.cb-card h3 {
  margin: 0 0 0.3rem 0;
  font-size: 0.95rem;
  color: var(--cb-text);
}

.cb-muted {
  color: var(--cb-muted);
}

.cb-section-title {
  font-size: 1.08rem;
  font-weight: 720;
  color: var(--cb-text);
  margin: 1.05rem 0 0.55rem 0;
}

.cb-kv {
  display: grid;
  grid-template-columns: minmax(120px, 0.8fr) minmax(180px, 1.2fr);
  gap: 0.45rem 0.9rem;
  font-size: 0.88rem;
}

.cb-kv .k {
  color: var(--cb-muted);
}

.cb-kv .v {
  color: var(--cb-text);
  font-weight: 620;
  overflow-wrap: anywhere;
}

[data-testid="stMetric"] {
  border: 1px solid var(--cb-border);
  border-radius: 18px;
  padding: 0.85rem 0.95rem;
  background: rgba(15, 23, 42, 0.70);
  box-shadow: 0 10px 34px rgba(0, 0, 0, 0.22);
}

[data-testid="stMetricLabel"] {
  color: var(--cb-muted) !important;
}

[data-testid="stMetricValue"] {
  color: var(--cb-text) !important;
}

.stPlotlyChart, [data-testid="stImage"], [data-testid="stPyplot"] {
  border-radius: 20px;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--cb-border);
  border-radius: 18px;
  overflow: hidden;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  background-color: rgba(15, 23, 42, 0.78) !important;
  border-color: var(--cb-border-strong) !important;
}

.stButton > button {
  border-radius: 999px;
  border: 1px solid var(--cb-border-strong);
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.20), rgba(167, 139, 250, 0.15));
  color: var(--cb-text);
}

hr {
  border-color: var(--cb-border);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

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


def latest_macro_levels(macro_levels: pd.DataFrame, product: str, timeframe: str) -> dict:
    if macro_levels.empty:
        return {}
    d = macro_levels[(macro_levels["product_id"] == product) & (macro_levels["timeframe"] == timeframe)].copy()
    if d.empty:
        return {}
    d = d.sort_values("ts")
    return d.iloc[-1].to_dict()


def card_html(title: str, body: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="cb-muted">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
<div class="cb-card">
  <h3>{title}</h3>
  {subtitle_html}
  {body}
</div>
""",
        unsafe_allow_html=True,
    )


def plot_macro(ax, df: pd.DataFrame, levels: dict, title: str, line_width: int, show_grid: bool):
    d = df.dropna(subset=["ts", "close"]).copy()
    d["dt"] = to_dt_mst(d["ts"])

    ax.set_facecolor("#0B1020")
    ax.plot(d["dt"], d["close"], linewidth=line_width, label="price (close)", color="#93C5FD")

    if levels:
        sup_lo = levels.get("support_zone_low")
        sup_hi = levels.get("support_zone_high")
        res_lo = levels.get("resistance_zone_low")
        res_hi = levels.get("resistance_zone_high")

        if pd.notna(sup_lo) and pd.notna(sup_hi):
            ax.axhspan(float(sup_lo), float(sup_hi), alpha=0.16, color="#34D399", label="support zone")
        if pd.notna(res_lo) and pd.notna(res_hi):
            ax.axhspan(float(res_lo), float(res_hi), alpha=0.16, color="#FB7185", label="resistance zone")

        for key, lbl, style, color in [
            ("breakout", "breakout", ":", "#FBBF24"),
            ("prev_low", "prev low", ":", "#A7F3D0"),
            ("prev_high", "prev high", ":", "#FECDD3"),
            ("vwap", "vwap", "-", "#C4B5FD"),
            ("val", "activity value area low", "-", "#67E8F9"),
            ("vah", "activity value area high", "-", "#F0ABFC"),
        ]:
            v = levels.get(key)
            if pd.notna(v):
                ax.axhline(float(v), linestyle=style, linewidth=1.0, color=color, label=lbl)

    ax.set_title(title, color="#E5E7EB", fontsize=12, pad=10)
    ax.set_xlabel("time (MST)", color="#94A3B8")
    ax.set_ylabel("price", color="#94A3B8")
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=d["dt"].dt.tz))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for spine in ax.spines.values():
        spine.set_color("#334155")
    if show_grid:
        ax.grid(True, alpha=0.18, color="#94A3B8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8, frameon=False, labelcolor="#CBD5E1")


# ---------------------------------------------------------------------------
# Sidebar controls + working auto-refresh
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh viewer", value=True)
refresh_sec = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 2)
manual_refresh = st.sidebar.button("Refresh now")

if manual_refresh:
    st.rerun()

if auto_refresh:
    if st_autorefresh is not None:
        refresh_count = st_autorefresh(interval=int(refresh_sec * 1000), key="viewer_autorefresh")
        st.sidebar.caption(f"Auto-refresh active · {refresh_sec}s · cycle {refresh_count}")
    else:
        components.html(
            f"""
<script>
const delay = {int(refresh_sec * 1000)};
setTimeout(function() {{
  try {{
    window.parent.location.reload();
  }} catch (e) {{
    window.location.reload();
  }}
}}, delay);
</script>
""",
            height=0,
        )
        st.sidebar.caption(f"Auto-refresh active · {refresh_sec}s · browser fallback")
else:
    st.sidebar.caption("Auto-refresh paused")

st.sidebar.divider()
st.sidebar.markdown("### Micro window")
window_minutes = st.sidebar.slider("Micro market window (minutes)", 1, 1440, 60)

st.sidebar.divider()
st.sidebar.markdown("### Chart style")
macro_height = st.sidebar.slider("Macro chart height (px)", 220, 520, 280, step=10)
micro_height = st.sidebar.slider("Micro chart height (px)", 220, 700, 340, step=10)
line_width = st.sidebar.slider("Line width", 1, 5, 2)
show_grid = st.sidebar.checkbox("Grid", True)

st.sidebar.divider()
st.sidebar.markdown("### Key")
st.sidebar.caption("Support/Resistance: bot-provided zones.")
st.sidebar.caption("VAL/VAH: approximate activity-weighted value area from bot output.")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

m = load_csv(MARKET_CSV)
t = load_csv(TRADES_CSV)
o = load_csv(ORDERS_CSV)
ml = load_csv(MACRO_LEVELS_CSV)

st.markdown(
    """
<div class="cb-hero">
  <div class="cb-eyebrow">Live Coinbase Trading Console</div>
  <div class="cb-title">Coinbase Bot Viewer</div>
  <div class="cb-subtitle">Macro structure, micro execution, probability sizing, confirmed trades, and order diagnostics.</div>
</div>
""",
    unsafe_allow_html=True,
)

if m.empty:
    st.info("Waiting for market.csv... start bot.py and let it write telemetry.")
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

cutoff = pd.Timestamp.utcnow().timestamp() - window_minutes * 60
m_view = m[m["ts"] >= cutoff].copy()
products = sorted([p for p in m_view["product_id"].dropna().unique().tolist() if isinstance(p, str)])
if not products:
    st.warning("No products found in the selected micro window yet.")
    st.stop()

default_idx = products.index("BTC-USD") if "BTC-USD" in products else 0
top_left, top_right = st.columns([0.55, 0.45], vertical_alignment="center")
with top_left:
    product = st.selectbox("Product", products, index=default_idx, label_visibility="collapsed")
with top_right:
    latest_ts = pd.to_numeric(m["ts"], errors="coerce").dropna()
    if not latest_ts.empty:
        last_seen = to_dt_mst(pd.Series([latest_ts.iloc[-1]])).iloc[0]
        st.caption(f"Last telemetry update: {last_seen.strftime('%Y-%m-%d %H:%M:%S %Z')}")


macro_dfs = {}
for label, path in MACRO_FILES.items():
    dfc = load_csv(path)
    if not dfc.empty and "product_id" in dfc.columns:
        dfc = numeric(dfc, ["ts", "open", "high", "low", "close", "volume"])
        dfc = dfc[dfc["product_id"] == product].dropna(subset=["ts", "close"])
    macro_dfs[label] = dfc

m_prod = m_view[m_view["product_id"] == product].dropna(subset=["ts", "mid"]).copy()
m_prod["dt"] = to_dt_mst(m_prod["ts"])

scored_rows = m_prod.dropna(subset=["entry_score"]) if "entry_score" in m_prod.columns else pd.DataFrame()
latest_row = scored_rows.iloc[-1] if not scored_rows.empty else (m_prod.iloc[-1] if not m_prod.empty else None)


# ---------------------------------------------------------------------------
# Executive metrics
# ---------------------------------------------------------------------------

last_row = m_prod.dropna(subset=["cash_usd", "equity_usd"]).tail(1)
cash = last_row["cash_usd"].iloc[0] if not last_row.empty else np.nan
equity = last_row["equity_usd"].iloc[0] if not last_row.empty else np.nan
position_qty = latest_row.get("position_qty", np.nan) if latest_row is not None else np.nan
exposure = latest_row.get("exposures_usd", np.nan) if latest_row is not None else np.nan
spread = latest_row.get("spread_bps", np.nan) if latest_row is not None else np.nan

st.markdown('<div class="cb-section-title">Account snapshot</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cash", fmt_money(cash))
c2.metric("Equity", fmt_money(equity))
c3.metric(f"{product} exposure", fmt_money(exposure))
c4.metric("Position qty", fmt_num(position_qty, 8))
c5.metric("Spread", fmt_num(spread, 2, " bps"))


# ---------------------------------------------------------------------------
# Probability sizing / signal summary
# ---------------------------------------------------------------------------

if latest_row is not None:
    prob = latest_row.get("estimated_prob_up", np.nan)
    pos_pct = latest_row.get("position_pct", np.nan)
    target_bps = latest_row.get("target_bps", np.nan)
    cost_bps = latest_row.get("cost_bps", np.nan)
    maker_bps = latest_row.get("current_maker_fee_bps", np.nan)
    taker_bps = latest_row.get("current_taker_fee_bps", np.nan)
    fee_reason = latest_row.get("fee_tier_reason", "")
    projected_size = float(equity) * float(pos_pct) if pd.notna(equity) and pd.notna(pos_pct) else np.nan

    st.markdown('<div class="cb-section-title">Probability sizing</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Estimated probability up", fmt_pct(prob))
    p2.metric("Position % of equity", fmt_pct(pos_pct))
    p3.metric("Projected buy size", fmt_money(projected_size))
    p4.metric("Available cash", fmt_money(cash))

    signal_body = f"""
<div class="cb-kv">
  <div class="k">Entry score</div><div class="v">{fmt_num(latest_row.get('entry_score', np.nan), 1)}</div>
  <div class="k">Tier</div><div class="v">{latest_row.get('entry_tier', '—')}</div>
  <div class="k">Expected net edge</div><div class="v">{fmt_num(latest_row.get('expected_net_edge_bps', np.nan), 1, ' bps')}</div>
  <div class="k">Target move</div><div class="v">{fmt_num(target_bps, 1, ' bps')}</div>
  <div class="k">Cost model</div><div class="v">{fmt_num(cost_bps, 1, ' bps')}</div>
  <div class="k">Coinbase maker fee</div><div class="v">{fmt_num(maker_bps, 2, ' bps')}</div>
  <div class="k">Coinbase taker fee</div><div class="v">{fmt_num(taker_bps, 2, ' bps')}</div>
  <div class="k">Fee source</div><div class="v">{fee_reason if fee_reason else '—'}</div>
  <div class="k">Reason</div><div class="v">{latest_row.get('entry_reason', '—')}</div>
</div>
"""
    card_html("Signal detail", signal_body)


# ---------------------------------------------------------------------------
# Macro charts
# ---------------------------------------------------------------------------

st.markdown(f'<div class="cb-section-title">{product} macro structure</div>', unsafe_allow_html=True)
colA, colB = st.columns(2)

def macro_panel(container, label, timeframe):
    dfc = macro_dfs.get(label, pd.DataFrame())
    levels = latest_macro_levels(ml, product, timeframe)
    if dfc is None or dfc.empty:
        container.info(f"{label}: waiting for {MACRO_FILES[label]} (bot downloads periodically).")
        return
    fig = plt.figure(figsize=(8, 3.2), facecolor="#070A12")
    ax = plt.gca()
    plot_macro(ax, dfc, levels, title=label, line_width=line_width, show_grid=show_grid)
    fig.set_size_inches(8, macro_height / 100.0)
    container.pyplot(fig, clear_figure=True)

with colA:
    macro_panel(st, "Past week (15m)", "week")
with colB:
    macro_panel(st, "Past day (1m)", "day")


# ---------------------------------------------------------------------------
# Trades for current product
# ---------------------------------------------------------------------------

if not t.empty and all(c in t.columns for c in ["ts", "product_id", "side", "price"]):
    t = numeric(t, ["ts", "price", "qty", "notional_usd", "cum_pnl_usd", "net_pnl_usd"])

    if "event" in t.columns:
        t = t[t["event"].isin(["BUY", "SELL", "STARTUP_LIQUIDATION"])].copy()

    if "qty" in t.columns:
        t = t[pd.to_numeric(t["qty"], errors="coerce").fillna(0.0) > 0].copy()

    if "price" in t.columns:
        t = t[pd.to_numeric(t["price"], errors="coerce").fillna(0.0) > 0].copy()

    t_prod = t[t["product_id"] == product].copy()
    if not t_prod.empty:
        t_prod["dt"] = to_dt_mst(t_prod["ts"])
        t_prod = t_prod[t_prod["ts"] >= cutoff].copy()
else:
    t_prod = pd.DataFrame()


# ---------------------------------------------------------------------------
# Micro chart
# ---------------------------------------------------------------------------

st.markdown(f'<div class="cb-section-title">{product} micro price · last {window_minutes} minutes</div>', unsafe_allow_html=True)
figp = plt.figure(figsize=(12, 3.6), facecolor="#070A12")
axp = plt.gca()
axp.set_facecolor("#0B1020")
axp.plot(m_prod["dt"], m_prod["mid"], linewidth=line_width, label="price: mid", color="#93C5FD")
if "bid" in m_prod.columns:
    axp.plot(m_prod["dt"], m_prod["bid"], linewidth=1, label="bid", color="#34D399")
if "ask" in m_prod.columns:
    axp.plot(m_prod["dt"], m_prod["ask"], linewidth=1, label="ask", color="#FB7185")
if "anchored_vwap" in m_prod.columns:
    av = m_prod["anchored_vwap"].astype(float).where(~m_prod["anchored_vwap"].isna())
    if not av.isna().all():
        axp.plot(m_prod["dt"], av, linewidth=1, linestyle="--", label="anchored VWAP", color="#C4B5FD")
if "fair_value" in m_prod.columns:
    fv = m_prod["fair_value"].astype(float).where(~m_prod["fair_value"].isna())
    if not fv.isna().all():
        axp.plot(m_prod["dt"], fv, linewidth=1, linestyle="--", label="fair value", color="#FBBF24")

if not t_prod.empty:
    if "event" in t_prod.columns:
        buys = t_prod[(t_prod["event"] == "BUY") & (t_prod["side"] == "BUY")]
        sells = t_prod[(t_prod["event"].isin(["SELL", "STARTUP_LIQUIDATION"])) & (t_prod["side"] == "SELL")]
    else:
        buys = t_prod[t_prod["side"] == "BUY"]
        sells = t_prod[t_prod["side"] == "SELL"]
    if not buys.empty:
        axp.scatter(buys["dt"], buys["price"], marker="^", s=90, color="#60A5FA", edgecolors="white", linewidths=0.7, zorder=6, label="BUY")
    if not sells.empty:
        axp.scatter(sells["dt"], sells["price"], marker="v", s=90, color="#F43F5E", edgecolors="white", linewidths=0.7, zorder=7, label="SELL")

axp.set_xlabel("time (MST)", color="#94A3B8")
axp.set_ylabel("price", color="#94A3B8")
axp.tick_params(colors="#94A3B8", labelsize=8)
axp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=m_prod["dt"].dt.tz))
axp.xaxis.set_major_locator(mdates.AutoDateLocator())
for spine in axp.spines.values():
    spine.set_color("#334155")
if show_grid:
    axp.grid(True, alpha=0.18, color="#94A3B8")
axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, frameon=False, labelcolor="#CBD5E1")
figp.set_size_inches(12, micro_height / 100.0)
st.pyplot(figp, clear_figure=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

st.markdown('<div class="cb-section-title">Confirmed trades</div>', unsafe_allow_html=True)
if t.empty:
    st.write("No confirmed trades yet.")
else:
    t_sorted = t.sort_values("ts", ascending=False).copy()
    if "dt_mst" not in t_sorted.columns and "ts" in t_sorted.columns:
        try:
            t_sorted["dt_mst"] = to_dt_mst(t_sorted["ts"]).astype(str)
        except Exception:
            pass
    if "dt_mst" in t_sorted.columns:
        cols = ["dt_mst"] + [c for c in t_sorted.columns if c != "dt_mst"]
        t_sorted = t_sorted[cols]
    st.dataframe(t_sorted, use_container_width=True, height=620)

st.markdown('<div class="cb-section-title">Recent order attempts</div>', unsafe_allow_html=True)
if not o.empty:
    o = numeric(o, [
        "ts", "requested_quote_usd", "requested_base_qty",
        "filled_qty", "avg_price", "filled_notional_usd", "fee_usd"
    ])
    if "ts" in o.columns:
        o = o.sort_values("ts", ascending=False)
    show_cols = [
        c for c in [
            "dt_mst", "event", "product_id", "side", "mode",
            "requested_quote_usd", "requested_base_qty",
            "ok", "status", "filled_qty", "avg_price",
            "filled_notional_usd", "fee_usd", "reason", "raw_error"
        ] if c in o.columns
    ]
    st.dataframe(o[show_cols].head(40), use_container_width=True, height=420)
else:
    st.write("No order attempts logged yet.")
