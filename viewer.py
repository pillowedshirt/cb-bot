import os
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

TZ = "America/Phoenix"  # MST (no DST)

# Resolve data file paths relative to this script so that the viewer finds
# the CSVs regardless of the current working directory.  Without this,
# launching streamlit from a different folder would fail to locate the files.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MARKET_CSV = os.path.join(BASE_DIR, "market.csv")
TRADES_CSV = os.path.join(BASE_DIR, "trades.csv")

# Only include the necessary macro windows for this model: week and day.
# The bot writes macro_week.csv and macro_day.csv into the current working
# directory, so we reference them directly here.  Avoid anchoring these
# filenames to BASE_DIR; otherwise Streamlit will look for the files in the
# script directory instead of the project root, causing the "waiting" message.
MACRO_FILES = {
    "Past week (15m)": "macro_week.csv",
    "Past day (1m)": "macro_day.csv",
}

st.set_page_config(page_title="Coinbase Bot Viewer", layout="wide")
st.title("Coinbase Bot — Macro + Micro View (MST)")

# ------------------------------------------------------------
# CSV utilities
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Macro level computation (8 horizontal line families)
# ------------------------------------------------------------

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

def compute_macro_lines(df: pd.DataFrame) -> dict:
    """
    Expects columns: ts, open, high, low, close, volume
    Returns dict of levels for plotting.
    """
    if df.empty or not all(c in df.columns for c in ["open", "high", "low", "close"]):
        return {}

    d = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if len(d) < 50:
        return {}

    o = d["open"].values.astype(float)
    h = d["high"].values.astype(float)
    l = d["low"].values.astype(float)
    c = d["close"].values.astype(float)
    v = d["volume"].values.astype(float) if "volume" in d.columns else np.ones_like(c)

    price_now = float(c[-1])

    # Range high/low
    range_low = float(np.min(l))
    range_high = float(np.max(h))

    # Prev high/low (previous half of the window as robust fallback)
    half = max(10, len(d) // 2)
    prev = d.iloc[:-half]
    prev_high = float(prev["high"].max()) if not prev.empty else float(h[0])
    prev_low = float(prev["low"].min()) if not prev.empty else float(l[0])

    # VWAP
    tp = (h + l + c) / 3.0
    vsum = float(np.sum(v))
    if vsum <= 1e-9:
        v = np.ones_like(c)
        vsum = float(np.sum(v))
    vwap = float(np.sum(tp * v) / vsum)

    # Psychological levels
    step = psych_step(price_now)
    psych_low = float(math.floor(price_now / step) * step)
    psych_high = float(math.ceil(price_now / step) * step)

    # Value area (volume-by-price approx): bin volume by close price
    bins = 60
    pmin, pmax = float(np.min(l)), float(np.max(h))
    if pmax <= pmin:
        return {}
    edges = np.linspace(pmin, pmax, bins + 1)
    hist = np.zeros(bins, dtype=float)
    idx = np.clip(np.digitize(c, edges) - 1, 0, bins - 1)
    for i, vv in zip(idx, v):
        hist[i] += float(vv)

    total = float(np.sum(hist))
    if total <= 1e-9:
        hist += 1.0
        total = float(np.sum(hist))

    poc_i = int(np.argmax(hist))
    poc = float((edges[poc_i] + edges[poc_i + 1]) / 2.0)

    target = 0.70 * total
    left = right = poc_i
    covered = float(hist[poc_i])
    while covered < target and (left > 0 or right < bins - 1):
        left_vol = hist[left - 1] if left > 0 else -1
        right_vol = hist[right + 1] if right < bins - 1 else -1
        if right_vol >= left_vol:
            right += 1
            covered += float(hist[right])
        else:
            left -= 1
            covered += float(hist[left])

    val = float(edges[left])
    vah = float(edges[right + 1])

    # Support/Resistance from extrema clustering
    w = 3
    lows_cand = []
    highs_cand = []
    lows_series = d["low"].values
    highs_series = d["high"].values
    for i in range(w, len(d) - w):
        lo = float(lows_series[i])
        hi = float(highs_series[i])
        if all(lo <= float(lows_series[j]) for j in range(i - w, i + w + 1)):
            lows_cand.append(lo)
        if all(hi >= float(highs_series[j]) for j in range(i - w, i + w + 1)):
            highs_cand.append(hi)

    def cluster_levels(levels, tol_pct=0.35):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        cur = [levels[0]]
        for x in levels[1:]:
            ref = float(np.mean(cur))
            tol = ref * (tol_pct / 100.0)
            if abs(x - ref) <= tol:
                cur.append(x)
            else:
                clusters.append((float(np.mean(cur)), len(cur)))
                cur = [x]
        clusters.append((float(np.mean(cur)), len(cur)))
        clusters.sort(key=lambda t: t[1], reverse=True)
        return clusters

    low_clusters = cluster_levels(lows_cand)
    high_clusters = cluster_levels(highs_cand)

    support = float(low_clusters[0][0]) if low_clusters else float(np.percentile(l, 15))
    resistance = float(high_clusters[0][0]) if high_clusters else float(np.percentile(h, 85))

    # Breakout level: strongest resistance above current (fallback to resistance)
    breakout = float(resistance)

    return {
        "support": support,
        "resistance": resistance,
        "breakout": breakout,
        "range_low": range_low,
        "range_high": range_high,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "vwap": vwap,
        "psych_low": psych_low,
        "psych_high": psych_high,
        "val": val,
        "vah": vah,
        "poc": poc,
        "price_now": price_now,
    }

def plot_macro(ax, df: pd.DataFrame, title: str, line_width: int, show_grid: bool):
    d = df.dropna(subset=["ts", "close"]).copy()
    d["dt"] = to_dt_mst(d["ts"])
    lines = compute_macro_lines(d)

    ax.plot(d["dt"], d["close"], linewidth=line_width, label="price (close)")

    if lines:
        # 8 families -> we draw multiple lines, but label clearly
        ax.axhline(lines["support"], linestyle="--", linewidth=1.2, label="support")
        ax.axhline(lines["resistance"], linestyle="--", linewidth=1.2, label="resistance")
        ax.axhline(lines["breakout"], linestyle=":", linewidth=1.2, label="breakout level")

        ax.axhline(lines["range_low"], linestyle="-.", linewidth=1.0, label="range low")
        ax.axhline(lines["range_high"], linestyle="-.", linewidth=1.0, label="range high")

        ax.axhline(lines["prev_low"], linestyle=":", linewidth=1.0, label="prev low")
        ax.axhline(lines["prev_high"], linestyle=":", linewidth=1.0, label="prev high")

        ax.axhline(lines["vwap"], linestyle="-", linewidth=1.0, label="vwap")

        ax.axhline(lines["psych_low"], linestyle="--", linewidth=1.0, label="psych low")
        ax.axhline(lines["psych_high"], linestyle="--", linewidth=1.0, label="psych high")

        ax.axhline(lines["val"], linestyle="-", linewidth=1.0, label="value area low (VAL)")
        ax.axhline(lines["vah"], linestyle="-", linewidth=1.0, label="value area high (VAH)")

    ax.set_title(title)
    ax.set_xlabel("time (MST)")
    ax.set_ylabel("price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=d["dt"].dt.tz))
    # Use default AutoDateLocator which avoids warnings about missing interval choices
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    if show_grid:
        ax.grid(True, alpha=0.25)

    # legend UNDER chart
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8, frameon=False)

    return lines

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------

st.sidebar.header("Controls")
refresh_sec = st.sidebar.slider("Refresh (seconds)", 0.5, 5.0, 1.0)

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
st.sidebar.subheader("Key (macro lines)")
st.sidebar.caption("Support/Resistance: clustered swing lows/highs (zones approximated as lines).")
st.sidebar.caption("Breakout: strongest resistance level (watch for close above + retest).")
st.sidebar.caption("Range hi/lo: max/min of the timeframe.")
st.sidebar.caption("Prev hi/lo: previous half-window high/low (robust fallback).")
st.sidebar.caption("VWAP: volume-weighted average price (volume is approximate if missing).")
st.sidebar.caption("Psych levels: nearest round-number steps.")
st.sidebar.caption("Value area (VAL/VAH): 70% volume-by-price region (approx).")

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

while True:
    m = load_csv(MARKET_CSV)
    t = load_csv(TRADES_CSV)

    if m.empty:
        st.info("Waiting for market.csv... (start bot.py and let it run)")
        time.sleep(refresh_sec)
        st.rerun()

    # Convert relevant numeric columns.  The market.csv produced by the bot contains
    # exposures, anchored_vwap, fair_value, sigma_bps and weekly_bias instead of
    # the old dip/rsi/ev metrics.  We omit conversion for the deprecated fields.
    m = numeric(m, [
        "ts", "mid", "bid", "ask", "spread_bps",
        "exposures_usd", "position_qty", "avg_entry_price",
        "anchored_vwap", "fair_value", "sigma_bps", "weekly_bias",
        "cash_usd", "equity_usd"
    ])
    cutoff = time.time() - window_minutes * 60
    m_view = m[m["ts"] >= cutoff].copy()

    products = sorted([p for p in m_view["product_id"].dropna().unique().tolist() if isinstance(p, str)])
    if not products:
        st.warning("No products found in the selected micro window yet.")
        time.sleep(refresh_sec)
        st.rerun()

    default_idx = products.index("BTC-USD") if "BTC-USD" in products else 0
    product = st.selectbox("Product", products, index=default_idx)

    # Load macro candle files
    macro_dfs = {}
    for label, path in MACRO_FILES.items():
        dfc = load_csv(path)
        if not dfc.empty and "product_id" in dfc.columns:
            dfc = numeric(dfc, ["ts", "open", "high", "low", "close", "volume"])
            dfc = dfc[dfc["product_id"] == product].dropna(subset=["ts", "close"])
        macro_dfs[label] = dfc

    # -----------------------
    # Macro quadrant charts
    # -----------------------
    st.subheader(f"{product} — Macro structure (MST)")

    # Use two columns for the macro views (week/day)
    colA, colB = st.columns(2)

    def macro_panel(container, label):
        dfc = macro_dfs.get(label, pd.DataFrame())
        if dfc is None or dfc.empty:
            container.info(f"{label}: waiting for {MACRO_FILES[label]} (bot downloads periodically).")
            return

        fig = plt.figure(figsize=(8, 3.2))
        ax = plt.gca()
        plot_macro(ax, dfc, title=label, line_width=line_width, show_grid=show_grid)
        fig.set_size_inches(8, macro_height / 100.0)
        container.pyplot(fig, clear_figure=True)

    with colA:
        macro_panel(st, "Past week (15m)")
    with colB:
        macro_panel(st, "Past day (1m)")

    # -----------------------
    # Micro charts
    # -----------------------
    m_prod = m_view[m_view["product_id"] == product].dropna(subset=["ts", "mid"]).copy()
    m_prod["dt"] = to_dt_mst(m_prod["ts"])

    # Trades for markers
    t_prod = pd.DataFrame()
    if not t.empty and all(c in t.columns for c in ["ts", "product_id", "side", "price"]):
        t = numeric(t, ["ts", "price", "cum_pnl_usd", "net_pnl_usd"])
        t_prod = t[t["product_id"] == product].copy()
        if not t_prod.empty:
            t_prod["dt"] = to_dt_mst(t_prod["ts"])
            # Ensure trade markers fall off cleanly outside the selected micro window
            t_prod = t_prod[t_prod["ts"] >= cutoff].copy()

    st.subheader(f"{product} — Micro price (last {window_minutes} min, MST)")
    figp = plt.figure(figsize=(12, 3.6))
    axp = plt.gca()

    axp.plot(m_prod["dt"], m_prod["mid"], linewidth=line_width, label="price: mid")
    if "bid" in m_prod.columns:
        axp.plot(m_prod["dt"], m_prod["bid"], linewidth=1, label="bid")
    if "ask" in m_prod.columns:
        axp.plot(m_prod["dt"], m_prod["ask"], linewidth=1, label="ask")

    # Overlay anchored VWAP and fair value if available
    if "anchored_vwap" in m_prod.columns:
        # Only plot where anchored_vwap is finite
        av_series = m_prod["anchored_vwap"].astype(float).where(~m_prod["anchored_vwap"].isna())
        if not av_series.isna().all():
            axp.plot(m_prod["dt"], av_series, linewidth=1, linestyle="--", label="anchored VWAP")
    if "fair_value" in m_prod.columns:
        fv_series = m_prod["fair_value"].astype(float).where(~m_prod["fair_value"].isna())
        if not fv_series.isna().all():
            axp.plot(m_prod["dt"], fv_series, linewidth=1, linestyle="--", label="fair value")

    if not t_prod.empty:
        buys = t_prod[t_prod["side"] == "BUY"]
        sells = t_prod[t_prod["side"] == "SELL"]
        if not buys.empty:
            axp.scatter(buys["dt"], buys["price"], marker="^", s=80, color="blue", edgecolors="white", linewidths=0.5, zorder=6, label="BUY")
        if not sells.empty:
            axp.scatter(
                sells["dt"], sells["price"],
                marker="v", s=80,
                color="red",
                edgecolors="white", linewidths=0.5,
                zorder=7,
                label="SELL"
            )

    axp.set_xlabel("time (MST)")
    axp.set_ylabel("price")
    axp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=m_prod["dt"].dt.tz))
    # Use default AutoDateLocator to avoid warnings about missing interval choices
    axp.xaxis.set_major_locator(mdates.AutoDateLocator())
    if show_grid:
        axp.grid(True, alpha=0.25)

    # legend under
    axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, frameon=False)
    figp.set_size_inches(12, micro_height / 100.0)
    st.pyplot(figp, clear_figure=True)

    # --------------------------------------
    # Signals chart (structural mean‑reversion context)
    #
    # We plot the deviation of anchored VWAP and fair value from the current mid price in
    # basis points, the volatility (sigma) in bps, and the weekly bias scaled to a
    # percentage.  These metrics explain why ladder entries are taken and when exits
    # occur.
    st.subheader("Signals (micro + macro context)")
    figs = plt.figure(figsize=(12, 3.6))
    axs = plt.gca()

    # Compute deltas in basis points where possible
    if "anchored_vwap" in m_prod.columns and "mid" in m_prod.columns:
        anchored_delta = np.where(
            (m_prod["mid"] > 0) & (~m_prod["anchored_vwap"].isna()),
            ((m_prod["anchored_vwap"] - m_prod["mid"]) / m_prod["mid"]) * 10000.0,
            np.nan,
        )
        axs.plot(m_prod["dt"], anchored_delta, linewidth=line_width, label="anchored_vwap_delta_bps")
        axs.axhline(0.0, linestyle=":", linewidth=1.0, label="anchored VWAP parity")

    if "fair_value" in m_prod.columns and "mid" in m_prod.columns:
        fair_delta = np.where(
            (m_prod["mid"] > 0) & (~m_prod["fair_value"].isna()),
            ((m_prod["fair_value"] - m_prod["mid"]) / m_prod["mid"]) * 10000.0,
            np.nan,
        )
        axs.plot(m_prod["dt"], fair_delta, linewidth=line_width, label="fair_value_delta_bps")

    if "sigma_bps" in m_prod.columns:
        axs.plot(m_prod["dt"], m_prod["sigma_bps"], linewidth=line_width, label="sigma_bps (volatility)")

    if "weekly_bias" in m_prod.columns:
        # Scale bias to ±100 for visibility
        bias_scaled = m_prod["weekly_bias"] * 100.0
        axs.plot(m_prod["dt"], bias_scaled, linewidth=line_width, label="weekly_bias (% of range)")
        axs.axhline(0.0, linestyle=":", linewidth=1.0, label="bias neutral")

    axs.set_xlabel("time (MST)")
    axs.set_ylabel("signal (bps / %)")
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=m_prod["dt"].dt.tz))
    # Use default AutoDateLocator to avoid warnings about missing interval choices
    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    if show_grid:
        axs.grid(True, alpha=0.25)

    # Place legend underneath
    axs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=9, frameon=False)
    figs.set_size_inches(12, micro_height / 100.0)
    st.pyplot(figs, clear_figure=True)

    # -----------------------
    # Account + Trades (PnL above)
    # -----------------------
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
        # Put dt_mst near the front for readability
        if "dt_mst" in t_sorted.columns:
            cols = ["dt_mst"] + [c for c in t_sorted.columns if c != "dt_mst"]
            t_sorted = t_sorted[cols]
        st.dataframe(t_sorted, width='stretch', height=780)

    time.sleep(refresh_sec)
    st.rerun()
