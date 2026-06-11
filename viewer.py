import importlib.util
import os
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

if importlib.util.find_spec("streamlit_autorefresh") is not None:
    from streamlit_autorefresh import st_autorefresh
else:
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

.cb-status-ok { color: var(--green); font-weight: 800; }
.cb-status-warn { color: var(--yellow); font-weight: 800; }
.cb-status-bad { color: var(--red); font-weight: 800; }

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


def numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def latest_by_product(m: pd.DataFrame) -> pd.DataFrame:
    if m.empty or "product_id" not in m.columns or "ts" not in m.columns:
        return pd.DataFrame()
    d = m.dropna(subset=["product_id", "ts"]).sort_values("ts").copy()
    return d.groupby("product_id", as_index=False).tail(1).sort_values("product_id")


def previous_by_product(m: pd.DataFrame, lookback_rows: int = 20) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if m.empty or "product_id" not in m.columns or "ts" not in m.columns:
        return out
    for product, d in m.dropna(subset=["product_id", "ts"]).sort_values("ts").groupby("product_id"):
        if len(d) >= 2:
            idx = max(0, len(d) - lookback_rows)
            out[str(product)] = d.iloc[idx]
    return out


def age_seconds(ts_value: Any) -> float:
    try:
        return max(0.0, pd.Timestamp.utcnow().timestamp() - float(ts_value))
    except Exception:
        return np.nan


def status_for_age(age: float) -> str:
    if pd.isna(age):
        return "unknown"
    if age <= 8:
        return "live"
    if age <= 30:
        return "delayed"
    return "stale"


def status_class(status: str) -> str:
    if status == "live":
        return "cb-status-ok"
    if status == "delayed":
        return "cb-status-warn"
    return "cb-status-bad"


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


def compact_order_line(row: pd.Series) -> str:
    side = str(row.get("side", "—"))
    product = str(row.get("product_id", "—"))
    status = str(row.get("status", "—"))
    mode = str(row.get("mode", "—"))
    quote = fmt_money(row.get("requested_quote_usd", np.nan))
    qty = fmt_num(row.get("filled_qty", np.nan), 8)
    return f"{side} · {product} · {status} · {mode} · request {quote} · fill {qty}"


def compact_trade_line(row: pd.Series) -> str:
    side = str(row.get("side", "—"))
    product = str(row.get("product_id", "—"))
    price = fmt_money(row.get("price", np.nan), 4)
    qty = fmt_num(row.get("qty", np.nan), 8)
    pnl = fmt_money(row.get("net_pnl_usd", np.nan))
    return f"{side} · {product} · price {price} · qty {qty} · net {pnl}"


def latest_macro_levels(macro_levels: pd.DataFrame, product: str, timeframe: str) -> dict:
    if macro_levels.empty or "product_id" not in macro_levels.columns or "timeframe" not in macro_levels.columns:
        return {}
    d = macro_levels[(macro_levels["product_id"] == product) & (macro_levels["timeframe"] == timeframe)].copy()
    if d.empty:
        return {}
    return d.sort_values("ts").iloc[-1].to_dict()


def clean_trades(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty:
        return t
    t = numeric(t, ["ts", "price", "qty", "notional_usd", "cum_pnl_usd", "net_pnl_usd", "fee_usd"])
    if "event" in t.columns:
        t = t[t["event"].isin(["BUY", "SELL", "STARTUP_LIQUIDATION"])].copy()
    if "qty" in t.columns:
        t = t[pd.to_numeric(t["qty"], errors="coerce").fillna(0.0) > 0].copy()
    if "price" in t.columns:
        t = t[pd.to_numeric(t["price"], errors="coerce").fillna(0.0) > 0].copy()
    if "ts" in t.columns and not t.empty:
        t["dt"] = to_dt_mst(t["ts"])
    return t


def plot_price(ax, d: pd.DataFrame, *, title: str, show_bid_ask: bool, trades: pd.DataFrame):
    ax.set_facecolor("#09111F")
    ax.plot(d["dt"], d["mid"], linewidth=1.7, label="mid", color="#93C5FD", zorder=2)

    if show_bid_ask and "bid" in d.columns and "ask" in d.columns:
        ax.plot(d["dt"], d["bid"], linewidth=0.75, label="bid", color="#34D399", alpha=0.8, zorder=1)
        ax.plot(d["dt"], d["ask"], linewidth=0.75, label="ask", color="#FB7185", alpha=0.8, zorder=1)

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
            ax.scatter(buys["dt"], buys["price"], marker="^", s=80, color="#60A5FA", edgecolors="white", linewidths=0.7, zorder=10, label="BUY")
            for _, r in buys.tail(5).iterrows():
                ax.annotate("BUY", (r["dt"], r["price"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="#BFDBFE", zorder=11)
        if not sells.empty:
            ax.scatter(sells["dt"], sells["price"], marker="v", s=80, color="#F43F5E", edgecolors="white", linewidths=0.7, zorder=10, label="SELL")
            for _, r in sells.tail(5).iterrows():
                ax.annotate("SELL", (r["dt"], r["price"]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color="#FECDD3", zorder=11)

    ax.set_title(title, color="#E5E7EB", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors="#94A3B8", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=d["dt"].dt.tz))
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



# =============================================================================
# Live update controls
# =============================================================================

with st.sidebar:
    st.markdown("### Viewer controls")
    live_update = st.checkbox("Live data update", value=True)
    refresh_sec = st.slider("Update interval", 1, 15, 2)
    if st.button("Update now"):
        st.rerun()

    st.divider()
    window_minutes = st.slider("Micro chart window", 5, 240, 45)
    overview_lookback_rows = st.slider("Overview change lookback rows", 2, 60, 20)
    show_bid_ask = st.checkbox("Show bid/ask lines", value=True)
    show_macro = st.checkbox("Show macro tabs", value=True)
    show_debug_tables = st.checkbox("Show debug telemetry table", value=False)

refresh_count = 0
refresh_status = "paused"

if live_update:
    if st_autorefresh is None:
        refresh_status = "missing package"
        st.error("Live update requires streamlit-autorefresh. Run: pip install streamlit-autorefresh")
    else:
        refresh_count = st_autorefresh(interval=int(refresh_sec * 1000), key="live_data_update")
        refresh_status = "active"


# =============================================================================
# Load data every rerun
# =============================================================================

m = load_csv(MARKET_CSV)
t = clean_trades(load_csv(TRADES_CSV))
o = load_csv(ORDERS_CSV)
ml = load_csv(MACRO_LEVELS_CSV)

st.markdown(
    """
<div class="cb-hero">
  <div class="cb-row">
    <div>
      <div class="cb-title">Coinbase Bot Viewer</div>
      <div class="cb-sub">Live vertical console · all-coins overview · probability sizing · confirmed fills</div>
    </div>
    <div class="cb-pill">CSV live monitor</div>
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

if not o.empty:
    o = numeric(o, [
        "ts", "requested_quote_usd", "requested_base_qty",
        "filled_qty", "avg_price", "filled_notional_usd", "fee_usd"
    ])
    if "ts" in o.columns and not o.empty:
        o["dt"] = to_dt_mst(o["ts"])


# =============================================================================
# Overview data
# =============================================================================

latest_all = latest_by_product(m)
previous_map = previous_by_product(m, overview_lookback_rows)

overview_rows = []
for _, r in latest_all.iterrows():
    product_id = str(r.get("product_id", ""))
    prev = previous_map.get(product_id)

    mid = safe_float(r.get("mid"))
    prev_mid = safe_float(prev.get("mid")) if prev is not None else np.nan
    mid_change_bps = ((mid / prev_mid) - 1.0) * 10000.0 if pd.notna(mid) and pd.notna(prev_mid) and prev_mid > 0 else np.nan

    prob = safe_float(r.get("estimated_prob_up"))
    pos_pct = safe_float(r.get("position_pct"))
    equity = safe_float(r.get("equity_usd"))
    projected_buy = equity * pos_pct if pd.notna(equity) and pd.notna(pos_pct) else np.nan
    age = age_seconds(r.get("ts"))
    status = status_for_age(age)

    product_rows = m[m["product_id"] == product_id] if "product_id" in m.columns else pd.DataFrame()

    overview_rows.append({
        "Product": product_id,
        "Status": status,
        "Age": f"{age:.0f}s" if pd.notna(age) else "—",
        "Prob": prob,
        "Score": safe_float(r.get("entry_score")),
        "Mid": mid,
        "Δ bps": mid_change_bps,
        "Spread": safe_float(r.get("spread_bps")),
        "Pos %": pos_pct,
        "Projected": projected_buy,
        "Exposure": safe_float(r.get("exposures_usd")),
        "Rows": len(product_rows),
    })
overview = pd.DataFrame(overview_rows)

latest_ts = pd.to_numeric(m["ts"], errors="coerce").dropna()
last_seen = to_dt_mst(pd.Series([latest_ts.iloc[-1]])).iloc[0] if not latest_ts.empty else None
global_age = age_seconds(latest_ts.iloc[-1]) if not latest_ts.empty else np.nan
global_status = status_for_age(global_age)
status_cls = status_class(global_status)

st.markdown(
    f"""
<div class="cb-row">
  <div class="cb-small">
    Last telemetry:
    <span class="{status_cls}">{last_seen.strftime('%H:%M:%S %Z') if last_seen is not None else '—'}</span>
    · age {fmt_num(global_age, 0, 's')}
    · update {refresh_status}
    · interval {refresh_sec}s
    · cycle {refresh_count}
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Top most recent activity
# =============================================================================

o_sorted = o.sort_values("ts", ascending=False).copy() if not o.empty and "ts" in o.columns else pd.DataFrame()
t_sorted = t.sort_values("ts", ascending=False).copy() if not t.empty and "ts" in t.columns else pd.DataFrame()

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
# All-coins overview
# =============================================================================

st.markdown('<div class="cb-section">All monitored coins overview</div>', unsafe_allow_html=True)
if overview.empty:
    st.warning("No product overview rows available yet.")
else:
    display_overview = overview.copy()
    display_overview["Prob"] = display_overview["Prob"].map(lambda x: fmt_pct(x))
    display_overview["Score"] = display_overview["Score"].map(lambda x: fmt_num(x, 1))
    display_overview["Mid"] = display_overview["Mid"].map(lambda x: fmt_num(x, 6))
    display_overview["Δ bps"] = display_overview["Δ bps"].map(lambda x: fmt_num(x, 1))
    display_overview["Spread"] = display_overview["Spread"].map(lambda x: fmt_num(x, 1))
    display_overview["Pos %"] = display_overview["Pos %"].map(lambda x: fmt_pct(x))
    display_overview["Projected"] = display_overview["Projected"].map(lambda x: fmt_money(x))
    display_overview["Exposure"] = display_overview["Exposure"].map(lambda x: fmt_money(x))
    st.dataframe(
        display_overview[["Product", "Status", "Age", "Prob", "Score", "Mid", "Δ bps", "Spread", "Pos %", "Projected", "Exposure", "Rows"]],
        use_container_width=True,
        hide_index=True,
        height=210,
    )


# =============================================================================
# Product selection
# =============================================================================

products = overview["Product"].tolist() if not overview.empty else sorted(m["product_id"].dropna().unique().tolist())
if not products:
    st.warning("No products found.")
    st.stop()

top_a, top_b = st.columns([0.52, 0.48], vertical_alignment="center")
with top_a:
    default_idx = products.index("BTC-USD") if "BTC-USD" in products else 0
    product = st.selectbox("Selected coin", products, index=default_idx, label_visibility="collapsed")

cutoff = pd.Timestamp.utcnow().timestamp() - float(window_minutes) * 60.0
m_prod = m[(m["product_id"] == product) & (m["ts"] >= cutoff)].dropna(subset=["ts", "mid"]).copy()
if m_prod.empty:
    st.warning(f"No recent telemetry rows for {product} in the selected window.")
    st.stop()

m_prod["dt"] = to_dt_mst(m_prod["ts"])
scored_rows = m_prod.dropna(subset=["entry_score"]) if "entry_score" in m_prod.columns else pd.DataFrame()
latest_row = scored_rows.iloc[-1] if not scored_rows.empty else m_prod.iloc[-1]
with top_b:
    selected_age = age_seconds(latest_row.get("ts"))
    selected_status = status_for_age(selected_age)
    st.caption(f"{product} status: {selected_status} · age {fmt_num(selected_age, 0, 's')} · rows in chart {len(m_prod)}")

t_prod = pd.DataFrame()
if not t.empty and "product_id" in t.columns:
    t_prod = t[(t["product_id"] == product) & (t["ts"] >= cutoff)].copy()


# =============================================================================
# Selected coin cards
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

st.markdown(f'<div class="cb-section">{product} live account and sizing</div>', unsafe_allow_html=True)
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
    mini_card("Probability up", fmt_pct(prob), "estimated by bot")
with p2:
    mini_card("Position size", fmt_pct(pos_pct), "of total equity")
with p3:
    mini_card("Projected buy", fmt_money(projected_size), "if triggered")
with p4:
    mini_card("Position qty", fmt_num(position_qty, 8), product)


# =============================================================================
# Selected coin signal detail
# =============================================================================

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
# Main live chart with buy/sell overlays
# =============================================================================

st.markdown(f'<div class="cb-section">{product} live chart with buy/sell overlays</div>', unsafe_allow_html=True)
fig = plt.figure(figsize=(7.5, 2.75), facecolor="#050814")
ax = plt.gca()
plot_price(ax, m_prod, title=f"{product} · last {window_minutes} min", show_bid_ask=show_bid_ask, trades=t_prod)
st.pyplot(fig, clear_figure=True, use_container_width=True)


# =============================================================================
# Compact recent rows
# =============================================================================

st.markdown('<div class="cb-section">Recent orders and trades</div>', unsafe_allow_html=True)
r1, r2 = st.columns(2)
with r1:
    st.caption("Recent order attempts")
    if o_sorted.empty:
        st.write("No order attempts.")
    else:
        compact_cols = [c for c in ["dt_mst", "event", "product_id", "side", "mode", "requested_quote_usd", "ok", "status", "filled_qty", "avg_price", "raw_error"] if c in o_sorted.columns]
        st.dataframe(o_sorted[compact_cols].head(5), use_container_width=True, height=175, hide_index=True)
with r2:
    st.caption("Recent confirmed trades")
    if t_sorted.empty:
        st.write("No confirmed trades.")
    else:
        compact_cols = [c for c in ["dt_mst", "event", "product_id", "side", "qty", "price", "fee_usd", "net_pnl_usd", "note"] if c in t_sorted.columns]
        st.dataframe(t_sorted[compact_cols].head(5), use_container_width=True, height=175, hide_index=True)


# =============================================================================
# Macro data in compact tabs
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
                    figm = plt.figure(figsize=(7.5, 2.05), facecolor="#050814")
                    axm = plt.gca()
                    plot_macro(axm, df_macro, levels, f"{product} · {label}")
                    st.pyplot(figm, clear_figure=True, use_container_width=True)


# =============================================================================
# Expandable older data
# =============================================================================

with st.expander("Older order attempts"):
    if o_sorted.empty:
        st.write("No order attempts logged yet.")
    else:
        show_cols = [c for c in ["dt_mst", "event", "product_id", "side", "mode", "requested_quote_usd", "requested_base_qty", "ok", "status", "filled_qty", "avg_price", "filled_notional_usd", "fee_usd", "reason", "raw_error"] if c in o_sorted.columns]
        st.dataframe(o_sorted[show_cols].head(150), use_container_width=True, height=420, hide_index=True)
with st.expander("Older confirmed trades"):
    if t_sorted.empty:
        st.write("No confirmed trades logged yet.")
    else:
        show_cols = [c for c in ["dt_mst", "event", "product_id", "side", "qty", "price", "fee_usd", "gross_pnl_usd", "net_pnl_usd", "cum_pnl_usd", "entry_price", "exit_price", "exit_role", "note"] if c in t_sorted.columns]
        st.dataframe(t_sorted[show_cols].head(150), use_container_width=True, height=420, hide_index=True)
if show_debug_tables:
    with st.expander("Market telemetry debug"):
        debug_cols = [c for c in ["ts", "product_id", "bid", "ask", "mid", "spread_bps", "cash_usd", "equity_usd", "entry_score", "entry_tier", "estimated_prob_up", "position_pct", "target_bps", "cost_bps", "entry_reason"] if c in m.columns]
        st.dataframe(m.sort_values("ts", ascending=False)[debug_cols].head(300), use_container_width=True, height=420, hide_index=True)
