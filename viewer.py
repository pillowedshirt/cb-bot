import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

TZ = "America/Phoenix"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MARKET_CSV = os.path.join(BASE_DIR, "market.csv")
TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")
MACRO_FILES = {
    "Past week (15m)": os.path.join(BASE_DIR, "macro_week.csv"),
    "Past day (1m)": os.path.join(BASE_DIR, "macro_day.csv"),
}
MACRO_LEVELS_CSV = os.path.join(BASE_DIR, "macro_levels.csv")

st.set_page_config(page_title="Coinbase Bot Viewer", layout="wide")
st.title("Coinbase Bot — Macro + Micro View (MST)")


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_dt_mst(ts_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(pd.to_numeric(ts_series, errors="coerce"), unit="s", utc=True)
    return dt.dt.tz_convert(TZ)


def psych_step(x: float) -> float:
    if x <= 0:
        return 1.0
    if x < 10:
        return 0.5
    if x < 100:
        return 5.0
    if x < 1000:
        return 25.0
    return 100.0


def latest_macro_levels(macro_levels: pd.DataFrame, product: str, timeframe: str) -> dict:
    if macro_levels.empty:
        return {}
    d = macro_levels[(macro_levels["product_id"] == product) & (macro_levels["timeframe"] == timeframe)].copy()
    if d.empty:
        return {}
    d = d.sort_values("ts")
    return d.iloc[-1].to_dict()


def plot_macro(ax, df: pd.DataFrame, levels: dict, title: str, line_width: int, show_grid: bool):
    d = df.dropna(subset=["ts", "close"]).copy()
    d["dt"] = to_dt_mst(d["ts"])
    ax.plot(d["dt"], d["close"], linewidth=line_width, label="price (close)")

    if levels:
        sup_lo = levels.get("support_zone_low")
        sup_hi = levels.get("support_zone_high")
        res_lo = levels.get("resistance_zone_low")
        res_hi = levels.get("resistance_zone_high")

        if pd.notna(sup_lo) and pd.notna(sup_hi):
            ax.axhspan(float(sup_lo), float(sup_hi), alpha=0.15, label="support zone")
        if pd.notna(res_lo) and pd.notna(res_hi):
            ax.axhspan(float(res_lo), float(res_hi), alpha=0.15, label="resistance zone")

        for key, lbl, style in [
            ("breakout", "breakout", ":"),
            ("prev_low", "prev low", ":"),
            ("prev_high", "prev high", ":"),
            ("vwap", "vwap", "-"),
            ("val", "activity value area low", "-"),
            ("vah", "activity value area high", "-"),
        ]:
            v = levels.get(key)
            if pd.notna(v):
                ax.axhline(float(v), linestyle=style, linewidth=1.0, label=lbl)

    ax.set_title(title)
    ax.set_xlabel("time (MST)")
    ax.set_ylabel("price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=d["dt"].dt.tz))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    if show_grid:
        ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8, frameon=False)


st.sidebar.header("Controls")
refresh_sec = st.sidebar.slider("Refresh (seconds)", 0.5, 5.0, 1.0)
manual_refresh = st.sidebar.button("Refresh now")
if manual_refresh:
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Micro window")
window_minutes = st.sidebar.slider("Micro market window (minutes)", 1, 1440, 60)

st.sidebar.divider()
st.sidebar.subheader("Chart style")
macro_height = st.sidebar.slider("Macro chart height (px)", 220, 520, 280, step=10)
micro_height = st.sidebar.slider("Micro chart height (px)", 220, 700, 340, step=10)
line_width = st.sidebar.slider("Line width", 1, 5, 2)
show_grid = st.sidebar.checkbox("Grid", True)

st.sidebar.divider()
st.sidebar.subheader("Key (macro)")
st.sidebar.caption("Support/Resistance: bot-provided zones.")
st.sidebar.caption("VAL/VAH: approximate activity-weighted value area from bot output.")

m = load_csv(MARKET_CSV)
t = load_csv(TRADES_CSV)
ml = load_csv(MACRO_LEVELS_CSV)

if m.empty:
    st.info("Waiting for market.csv... (start bot.py and let it run)")
    st.stop()

m = numeric(m, [
    "ts", "mid", "bid", "ask", "spread_bps",
    "exposures_usd", "position_qty", "avg_entry_price",
    "anchored_vwap", "fair_value", "sigma_bps", "weekly_bias",
    "cash_usd", "equity_usd"
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
product = st.selectbox("Product", products, index=default_idx)

macro_dfs = {}
for label, path in MACRO_FILES.items():
    dfc = load_csv(path)
    if not dfc.empty and "product_id" in dfc.columns:
        dfc = numeric(dfc, ["ts", "open", "high", "low", "close", "volume"])
        dfc = dfc[dfc["product_id"] == product].dropna(subset=["ts", "close"])
    macro_dfs[label] = dfc

st.subheader(f"{product} — Macro structure (MST)")
colA, colB = st.columns(2)

def macro_panel(container, label, timeframe):
    dfc = macro_dfs.get(label, pd.DataFrame())
    levels = latest_macro_levels(ml, product, timeframe)
    if dfc is None or dfc.empty:
        container.info(f"{label}: waiting for {MACRO_FILES[label]} (bot downloads periodically).")
        return
    fig = plt.figure(figsize=(8, 3.2))
    ax = plt.gca()
    plot_macro(ax, dfc, levels, title=label, line_width=line_width, show_grid=show_grid)
    fig.set_size_inches(8, macro_height / 100.0)
    container.pyplot(fig, clear_figure=True)

with colA:
    macro_panel(st, "Past week (15m)", "week")
with colB:
    macro_panel(st, "Past day (1m)", "day")

m_prod = m_view[m_view["product_id"] == product].dropna(subset=["ts", "mid"]).copy()
m_prod["dt"] = to_dt_mst(m_prod["ts"])

if not t.empty and all(c in t.columns for c in ["ts", "product_id", "side", "price"]):
    t = numeric(t, ["ts", "price", "cum_pnl_usd", "net_pnl_usd"])
    t_prod = t[t["product_id"] == product].copy()
    if not t_prod.empty:
        t_prod["dt"] = to_dt_mst(t_prod["ts"])
        t_prod = t_prod[t_prod["ts"] >= cutoff].copy()
else:
    t_prod = pd.DataFrame()

st.subheader(f"{product} — Micro price (last {window_minutes} min, MST)")
figp = plt.figure(figsize=(12, 3.6))
axp = plt.gca()
axp.plot(m_prod["dt"], m_prod["mid"], linewidth=line_width, label="price: mid")
if "bid" in m_prod.columns:
    axp.plot(m_prod["dt"], m_prod["bid"], linewidth=1, label="bid")
if "ask" in m_prod.columns:
    axp.plot(m_prod["dt"], m_prod["ask"], linewidth=1, label="ask")
if "anchored_vwap" in m_prod.columns:
    av = m_prod["anchored_vwap"].astype(float).where(~m_prod["anchored_vwap"].isna())
    if not av.isna().all():
        axp.plot(m_prod["dt"], av, linewidth=1, linestyle="--", label="anchored VWAP")
if "fair_value" in m_prod.columns:
    fv = m_prod["fair_value"].astype(float).where(~m_prod["fair_value"].isna())
    if not fv.isna().all():
        axp.plot(m_prod["dt"], fv, linewidth=1, linestyle="--", label="fair value")

if not t_prod.empty:
    buys = t_prod[t_prod["side"] == "BUY"]
    sells = t_prod[t_prod["side"] == "SELL"]
    if not buys.empty:
        axp.scatter(buys["dt"], buys["price"], marker="^", s=80, color="blue", edgecolors="white", linewidths=0.5, zorder=6, label="BUY")
    if not sells.empty:
        axp.scatter(sells["dt"], sells["price"], marker="v", s=80, color="red", edgecolors="white", linewidths=0.5, zorder=7, label="SELL")

axp.set_xlabel("time (MST)")
axp.set_ylabel("price")
axp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=m_prod["dt"].dt.tz))
axp.xaxis.set_major_locator(mdates.AutoDateLocator())
if show_grid:
    axp.grid(True, alpha=0.25)
axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, frameon=False)
figp.set_size_inches(12, micro_height / 100.0)
st.pyplot(figp, clear_figure=True)

st.subheader("Account (paper mode)")
c1, c2, c3 = st.columns(3)
last_row = m_prod.dropna(subset=["cash_usd", "equity_usd"]).tail(1)
if not last_row.empty:
    c1.metric("Cash (USD)", f"{float(last_row['cash_usd'].iloc[0]):.2f}")
    c2.metric("Equity (USD)", f"{float(last_row['equity_usd'].iloc[0]):.2f}")
if not t.empty and "cum_pnl_usd" in t.columns:
    lastp = pd.to_numeric(t["cum_pnl_usd"], errors="coerce").dropna()
    if not lastp.empty:
        c3.metric("Cumulative P&L (USD)", f"{float(lastp.iloc[-1]):.6f}")

st.subheader("Trades (most recent first)")
if t.empty:
    st.write("No trades yet.")
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
    st.dataframe(t_sorted, width='stretch', height=780)
