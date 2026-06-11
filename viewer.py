import importlib.util
import os
from typing import Any, Callable, Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
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
CALIBRATION_CSV = os.path.join(BASE_DIR, "calibration.csv")
MICRO_HISTORY_CSV = os.path.join(BASE_DIR, "micro_history.csv")
POSITION_TARGETS_CSV = os.path.join(BASE_DIR, "position_targets.csv")
CANDIDATE_REPLAY_CSV = os.path.join(BASE_DIR, "candidate_replay.csv")
PRODUCTS_ACTIVE_CSV = os.path.join(BASE_DIR, "products_active.csv")
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

def load_csv(path: str, required_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Load CSV safely.

    If the file is missing, empty, partially written, or missing required columns,
    return an empty dataframe instead of crashing the viewer.
    """
    try:
        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path)

        if df.empty:
            return pd.DataFrame()

        if required_cols:
            missing = [column for column in required_cols if column not in df.columns]
            if missing:
                st.warning(
                    f"{os.path.basename(path)} is missing required columns {missing}. "
                    f"Detected columns: {list(df.columns)}. "
                    "Restart bot.py or clear the old CSV so it can regenerate."
                )
                return pd.DataFrame()

        return df
    except Exception as exc:
        st.warning(f"Could not read {os.path.basename(path)}: {exc}")
        return pd.DataFrame()


def numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def truthy_cell(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def pass_wait_label(value: Any) -> str:
    return "PASS" if truthy_cell(value) else "waiting"


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


def valid_target_number(value: Any) -> bool:
    """Return True only for finite, strictly positive calibration targets."""
    try:
        target = float(value)
        return bool(np.isfinite(target) and target > 0.0)
    except Exception:
        return False


def display_target_value(row: pd.Series, column: str, formatter: Any) -> str:
    if not bool(row.get("Calibrated", False)):
        return "Awaiting calibration"
    value = row.get(column, np.nan)
    if not valid_target_number(value):
        return "Awaiting calibration"
    return formatter(value)


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


def latest_calibration_for_product(cal: pd.DataFrame, product: str) -> pd.Series | None:
    """Return the latest active learned profile, then fall back to the latest row."""
    if cal.empty or "product_id" not in cal.columns or "ts" not in cal.columns:
        return None
    rows = cal[cal["product_id"].astype(str) == str(product)].copy()
    if rows.empty:
        return None
    rows = rows.sort_values("ts")
    learned_rows = rows[
        rows.apply(
            lambda row: (
                truthy_cell(row.get("is_calibrated", False))
                and valid_target_number(row.get("min_score", np.nan))
                and valid_target_number(row.get("min_probability", np.nan))
                and valid_target_number(
                    row.get("min_expected_value_bps", np.nan)
                )
            ),
            axis=1,
        )
    ]
    if not learned_rows.empty:
        return learned_rows.iloc[-1]
    return rows.iloc[-1]


def latest_position_target_for_product(pt: pd.DataFrame, product: str) -> pd.Series | None:
    if pt.empty or "product_id" not in pt.columns or "ts" not in pt.columns:
        return None
    rows = pt[pt["product_id"] == product].copy()
    if rows.empty:
        return None
    return rows.sort_values("ts").iloc[-1]


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


def plot_price_plotly(
    d: pd.DataFrame,
    *,
    title: str,
    show_bid_ask: bool,
    trades: pd.DataFrame,
    sell_target_row: pd.Series | None = None,
):
    """
    Live chart rendered with Plotly instead of st.pyplot.

    This avoids Streamlit temporary PNG media-file errors during rapid auto-refresh.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=d["dt"],
        y=d["mid"],
        mode="lines",
        name="mid",
        line=dict(width=2),
        hovertemplate="mid=%{y}<br>%{x}<extra></extra>",
    ))

    if show_bid_ask and "bid" in d.columns and "ask" in d.columns:
        fig.add_trace(go.Scatter(
            x=d["dt"],
            y=d["bid"],
            mode="lines",
            name="bid",
            line=dict(width=1),
            hovertemplate="bid=%{y}<br>%{x}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=d["dt"],
            y=d["ask"],
            mode="lines",
            name="ask",
            line=dict(width=1),
            hovertemplate="ask=%{y}<br>%{x}<extra></extra>",
        ))

    if "anchored_vwap" in d.columns and not d["anchored_vwap"].isna().all():
        fig.add_trace(go.Scatter(
            x=d["dt"],
            y=d["anchored_vwap"],
            mode="lines",
            name="VWAP",
            line=dict(width=1, dash="dash"),
            hovertemplate="VWAP=%{y}<br>%{x}<extra></extra>",
        ))

    if "fair_value" in d.columns and not d["fair_value"].isna().all():
        fig.add_trace(go.Scatter(
            x=d["dt"],
            y=d["fair_value"],
            mode="lines",
            name="fair",
            line=dict(width=1, dash="dash"),
            hovertemplate="fair=%{y}<br>%{x}<extra></extra>",
        ))

    if trades is not None and not trades.empty:
        buys = pd.DataFrame()
        sells = pd.DataFrame()

        if "event" in trades.columns:
            buys = trades[(trades["event"] == "BUY") & (trades["side"] == "BUY")]
            sells = trades[(trades["event"].isin(["SELL", "STARTUP_LIQUIDATION"])) & (trades["side"] == "SELL")]
        elif "side" in trades.columns:
            buys = trades[trades["side"] == "BUY"]
            sells = trades[trades["side"] == "SELL"]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["dt"],
                y=buys["price"],
                mode="markers+text",
                name="BUY",
                text=["BUY"] * len(buys),
                textposition="top center",
                marker=dict(symbol="triangle-up", size=12, line=dict(width=1)),
                hovertemplate="BUY<br>price=%{y}<br>%{x}<extra></extra>",
            ))

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["dt"],
                y=sells["price"],
                mode="markers+text",
                name="SELL",
                text=["SELL"] * len(sells),
                textposition="bottom center",
                marker=dict(symbol="triangle-down", size=12, line=dict(width=1)),
                hovertemplate="SELL<br>price=%{y}<br>%{x}<extra></extra>",
            ))

    if sell_target_row is not None and str(sell_target_row.get("has_position", "False")).lower() in ("true", "1", "yes"):
        line_specs = [
            ("Min profitable exit", sell_target_row.get("min_profitable_exit_price"), "dot"),
            ("Scalp target", sell_target_row.get("scalp_target_price"), "dash"),
            ("Core target", sell_target_row.get("core_target_price"), "dash"),
            ("Scalp trigger", sell_target_row.get("scalp_pullback_trigger_price"), "dot"),
            ("Core trigger", sell_target_row.get("core_pullback_trigger_price"), "dot"),
        ]
        for label, value, dash in line_specs:
            try:
                y = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(y) or y <= 0:
                continue
            fig.add_hline(
                y=y,
                line_dash=dash,
                annotation_text=label,
                annotation_position="top left",
                opacity=0.72,
            )

    fig.update_layout(
        title=title,
        height=305,
        margin=dict(l=8, r=8, t=36, b=8),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#09111F",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.13)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.13)",
            zeroline=False,
        ),
    )

    return fig


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
    # Default to max so the first opened view shows the broadest available run context.
    window_minutes = st.slider("Micro chart window", 5, 1440, 1440)
    overview_lookback_rows = st.slider("Overview change lookback rows", 2, 120, 120)
    show_bid_ask = st.checkbox("Show bid/ask lines", value=True)
    show_macro = st.checkbox("Show macro tabs", value=False)
    show_debug_tables = st.checkbox("Show debug telemetry table", value=False)

# Keep a viewer session start time so the display does not feel like it resets.
if "viewer_started_at_ts" not in st.session_state:
    st.session_state["viewer_started_at_ts"] = pd.Timestamp.utcnow().timestamp()

# Count visible data refreshes across automatic updates.
if "viewer_refresh_count" not in st.session_state:
    st.session_state["viewer_refresh_count"] = 0

refresh_count = int(st.session_state.get("viewer_refresh_count", 0))
refresh_status = "paused"
refresh_mode = "manual"

# Prefer Streamlit-native fragment refresh. It updates the live dashboard section
# without using a browser page reload.
HAS_NATIVE_FRAGMENT_REFRESH = hasattr(st, "fragment")


def render_live_dashboard() -> None:
    global refresh_count, refresh_status, refresh_mode

    st.session_state["viewer_refresh_count"] = int(st.session_state.get("viewer_refresh_count", 0)) + 1
    refresh_count = int(st.session_state["viewer_refresh_count"])

    if live_update and HAS_NATIVE_FRAGMENT_REFRESH:
        refresh_status = "active"
        refresh_mode = "native fragment"
    elif live_update and st_autorefresh is not None:
        refresh_status = "active"
        refresh_mode = "streamlit-autorefresh"
    elif live_update:
        refresh_status = "missing refresh engine"
        refresh_mode = "not running"
    else:
        refresh_status = "paused"
        refresh_mode = "manual"

    # =============================================================================
    # Load data every rerun
    # =============================================================================

    m = load_csv(MARKET_CSV, required_cols=["ts", "product_id", "mid"])
    t = clean_trades(load_csv(TRADES_CSV))
    o = load_csv(ORDERS_CSV)
    ml = load_csv(MACRO_LEVELS_CSV)
    cal = load_csv(CALIBRATION_CSV)
    hist = load_csv(MICRO_HISTORY_CSV)
    pt = load_csv(POSITION_TARGETS_CSV)
    cr = load_csv(CANDIDATE_REPLAY_CSV)
    active_products_df = load_csv(PRODUCTS_ACTIVE_CSV)

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
        st.info(
            "Waiting for a valid market.csv. Start bot.py and let it write telemetry. "
            "If bot.py is already running, delete/rename the old market.csv and restart bot.py."
        )
        st.stop()

    m = numeric(m, [
        "ts", "mid", "bid", "ask", "spread_bps",
        "exposures_usd", "position_qty", "avg_entry_price",
        "anchored_vwap", "fair_value", "sigma_bps", "weekly_bias",
        "cash_usd", "equity_usd",
        "entry_score", "entry_tier", "expected_net_edge_bps",
        "estimated_prob_up", "position_pct",
        "target_bps", "projected_forward_gain_bps", "cost_bps",
        "calibrated_time_to_min_profit_minutes", "calibrated_forward_window_minutes",
        "current_maker_fee_bps", "current_taker_fee_bps",
        "dip_depth_score", "dip_speed_score", "reversal_score", "support_score",
        "room_score", "regime_score", "spread_penalty", "cost_penalty"
    ])

    m_eval = m[m["source"].astype(str).str.lower() == "eval"].copy() if "source" in m.columns else m.copy()
    m_telemetry = m[m["source"].astype(str).str.lower() == "telemetry"].copy() if "source" in m.columns else m.copy()
    m_chart = m.copy()

    cal = numeric(cal, [
        "ts", "min_score", "min_probability", "min_expected_value_bps",
        "scalp_pullback_pct", "core_pullback_pct",
        "day_sample_count", "week_sample_count",
        "day_win_rate", "week_win_rate", "blended_win_rate",
        "avg_win_bps", "avg_loss_bps", "expected_value_bps",
        "calibrated_projected_gross_bps",
        "calibrated_projected_net_bps",
        "calibrated_time_to_min_profit_minutes",
        "calibrated_forward_window_minutes",
        "calibrated_selected_window_minutes",
        "calibrated_post_profit_breathing_minutes",
        "calibrated_post_profit_extra_gain_bps",
        "calibrated_max_adverse_before_profit_bps",
        "calibrated_expected_bps_per_minute",
        "calibrated_raw_probability_median",
        "calibrated_empirical_win_rate",
    ])
    hist = numeric(hist, ["ts", "open", "high", "low", "close", "volume"])
    pt = numeric(pt, [
        "ts", "position_qty", "avg_entry_price", "current_bid", "current_ask",
        "min_profitable_exit_price", "scalp_target_price", "core_target_price",
        "scalp_arm_peak", "core_arm_peak",
        "scalp_pullback_pct", "core_pullback_pct",
        "scalp_pullback_trigger_price", "core_pullback_trigger_price",
        "distance_to_min_profit_bps", "distance_to_scalp_bps", "distance_to_core_bps",
        "profit_lock_price", "min_profitable_exit_price_from_lot",
        "calibrated_forward_window_minutes", "calibrated_post_profit_breathing_minutes",
    ])
    cr = numeric(cr, [
        "ts", "score", "probability", "expected_net_edge_bps",
        "target_bps", "cost_bps", "spread_bps",
        "selected_forward_window_minutes", "max_favorable_bps", "max_adverse_bps",
        "adverse_before_profit_bps", "time_to_min_profit_minutes",
        "forward_window_minutes", "post_profit_max_favorable_bps",
        "post_profit_extra_gain_bps",
    ])

    ml = numeric(ml, [
        "ts", "support_zone_low", "support_zone_high", "resistance_zone_low", "resistance_zone_high",
        "breakout", "range_low", "range_high", "prev_low", "prev_high", "vwap", "val", "vah", "price_now"
    ])

    configured_products = []
    if not active_products_df.empty and "product_id" in active_products_df.columns:
        configured_products = [
            str(x) for x in active_products_df["product_id"].dropna().tolist()
        ]

    if not configured_products:
        # Display fallback until bot.py publishes products_active.csv.
        configured_products = [
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
            "DOGE-USD", "ADA-USD", "LINK-USD", "AVAX-USD", "XLM-USD",
            "LTC-USD", "BCH-USD", "SHIB-USD", "DOT-USD", "SUI-USD",
        ]

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

    latest_all = latest_by_product(m_eval)

    # Build the overview from configured products, not only products with eval rows.
    eval_products = (
        sorted(set(str(x) for x in latest_all["product_id"].dropna().tolist()))
        if not latest_all.empty and "product_id" in latest_all.columns
        else []
    )
    cal_products = (
        sorted(set(str(x) for x in cal["product_id"].dropna().tolist()))
        if not cal.empty and "product_id" in cal.columns
        else []
    )
    market_products = (
        sorted(set(str(x) for x in m["product_id"].dropna().tolist()))
        if not m.empty and "product_id" in m.columns
        else []
    )
    all_products_for_overview = sorted(
        set(configured_products + eval_products + cal_products + market_products)
    )

    previous_map = previous_by_product(m_chart, overview_lookback_rows)
    overview_rows = []
    latest_all_product_ids = (
        set(latest_all["product_id"].astype(str).tolist())
        if not latest_all.empty and "product_id" in latest_all.columns
        else set()
    )

    for product_id in all_products_for_overview:
        if product_id in latest_all_product_ids:
            r = latest_all[
                latest_all["product_id"].astype(str) == product_id
            ].iloc[-1]
        else:
            r = pd.Series({"product_id": product_id})
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
        calibration = latest_calibration_for_product(cal, product_id)
        is_product_calibrated = False
        calibration_status = "awaiting_calibration"
        calibration_reason = "No calibration row yet"

        if calibration is not None:
            is_product_calibrated = truthy_cell(
                calibration.get("is_calibrated", False)
            )
            calibration_status = str(
                calibration.get("calibration_status", "unknown")
            )
            calibration_reason = str(calibration.get("reason", ""))

        raw_score_target = (
            calibration.get("min_score", np.nan)
            if calibration is not None else np.nan
        )
        raw_prob_target = (
            calibration.get("min_probability", np.nan)
            if calibration is not None else np.nan
        )
        raw_ev_target = (
            calibration.get("min_expected_value_bps", np.nan)
            if calibration is not None else np.nan
        )
        targets_are_valid = bool(
            is_product_calibrated
            and valid_target_number(raw_score_target)
            and valid_target_number(raw_prob_target)
            and valid_target_number(raw_ev_target)
        )

        if targets_are_valid:
            buy_score_target = raw_score_target
            buy_prob_target = raw_prob_target
            buy_ev_target = raw_ev_target
        else:
            buy_score_target = np.nan
            buy_prob_target = np.nan
            buy_ev_target = np.nan

        overview_rows.append({
            "Product": product_id,
            "Status": status,
            "Age": f"{age:.0f}s" if pd.notna(age) else "—",
            "Prob": prob,
            "Score": safe_float(r.get("entry_score")),
            "Buy Score Target": buy_score_target,
            "Buy Prob Target": buy_prob_target,
            "Buy EV Target": buy_ev_target,
            "Calibrated": targets_are_valid,
            "Calibration Status": calibration_status,
            "Calibration Reason": calibration_reason,
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
    latest_market_ts = latest_ts.max() if not latest_ts.empty else np.nan
    earliest_market_ts = latest_ts.min() if not latest_ts.empty else np.nan
    last_seen = to_dt_mst(pd.Series([latest_market_ts])).iloc[0] if pd.notna(latest_market_ts) else None
    global_age = age_seconds(latest_market_ts) if pd.notna(latest_market_ts) else np.nan
    global_status = status_for_age(global_age)

    viewer_runtime_sec = pd.Timestamp.utcnow().timestamp() - float(
        st.session_state.get("viewer_started_at_ts", pd.Timestamp.utcnow().timestamp())
    )

    bot_runtime_sec = np.nan
    try:
        if pd.notna(earliest_market_ts) and pd.notna(latest_market_ts):
            bot_runtime_sec = float(latest_market_ts) - float(earliest_market_ts)
    except Exception:
        bot_runtime_sec = np.nan

    status_cls = status_class(global_status)

    st.markdown(
        f"""
    <div class="cb-row">
      <div class="cb-small">
        Last telemetry:
        <span class="{status_cls}">{last_seen.strftime('%H:%M:%S %Z') if last_seen is not None else '—'}</span>
        · data age {fmt_num(global_age, 0, 's')}
        · bot runtime {fmt_num(bot_runtime_sec / 60.0 if pd.notna(bot_runtime_sec) else np.nan, 1, ' min')}
        · viewer runtime {fmt_num(viewer_runtime_sec / 60.0, 1, ' min')}
        · update {refresh_status}
        · mode {refresh_mode}
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
        # Sort by estimated probability, then score, so the strongest setup is visually first.
        overview_sorted = overview.copy()
        overview_sorted["ProbRaw"] = pd.to_numeric(overview_sorted["Prob"], errors="coerce")
        overview_sorted["ScoreRaw"] = pd.to_numeric(overview_sorted["Score"], errors="coerce")
        overview_sorted = overview_sorted.sort_values(["ProbRaw", "ScoreRaw"], ascending=False)

        cols_per_row = 3
        rows = [overview_sorted.iloc[i:i + cols_per_row] for i in range(0, len(overview_sorted), cols_per_row)]

        for chunk in rows:
            cols = st.columns(cols_per_row)
            for col, (_, row) in zip(cols, chunk.iterrows()):
                status = str(row.get("Status", "unknown"))
                cls = status_class(status)
                product_label = str(row.get("Product", "—"))

                prob_val = row.get("Prob", np.nan)
                score_val = row.get("Score", np.nan)
                mid_val = row.get("Mid", np.nan)
                change_val = row.get("Δ bps", np.nan)
                spread_val = row.get("Spread", np.nan)
                projected_val = row.get("Projected", np.nan)
                exposure_val = row.get("Exposure", np.nan)
                rows_val = row.get("Rows", "—")
                is_row_calibrated = bool(row.get("Calibrated", False))

                buy_score_display = (
                    fmt_num(row.get("Buy Score Target", np.nan), 3)
                    if is_row_calibrated
                    and valid_target_number(row.get("Buy Score Target", np.nan))
                    else "Awaiting calibration"
                )
                buy_prob_display = (
                    fmt_pct(row.get("Buy Prob Target", np.nan), 3)
                    if is_row_calibrated
                    and valid_target_number(row.get("Buy Prob Target", np.nan))
                    else "Awaiting calibration"
                )
                buy_ev_display = (
                    fmt_num(row.get("Buy EV Target", np.nan), 1, " bps")
                    if is_row_calibrated
                    and valid_target_number(row.get("Buy EV Target", np.nan))
                    else "Awaiting calibration"
                )
                calibration_display = (
                    "READY" if is_row_calibrated else "AWAITING"
                )

                body = f"""
    <div class="cb-card">
      <div class="cb-row">
        <div class="cb-value">{product_label}</div>
        <div class="{cls}">{status.upper()}</div>
      </div>
      <div class="cb-kv" style="margin-top:0.35rem;">
        <div class="k">Probability</div><div class="v">{fmt_pct(prob_val)}</div>
        <div class="k">Score</div><div class="v">{fmt_num(score_val, 1)}</div>
        <div class="k">Calibration</div><div class="v">{calibration_display}</div>
        <div class="k">Buy score target</div><div class="v">{buy_score_display}</div>
        <div class="k">Buy prob target</div><div class="v">{buy_prob_display}</div>
        <div class="k">Buy EV target</div><div class="v">{buy_ev_display}</div>
        <div class="k">Mid</div><div class="v">{fmt_num(mid_val, 6)}</div>
        <div class="k">Change</div><div class="v">{fmt_num(change_val, 1, ' bps')}</div>
        <div class="k">Spread</div><div class="v">{fmt_num(spread_val, 1, ' bps')}</div>
        <div class="k">Projected buy</div><div class="v">{fmt_money(projected_val)}</div>
        <div class="k">Exposure</div><div class="v">{fmt_money(exposure_val)}</div>
        <div class="k">Rows</div><div class="v">{rows_val}</div>
      </div>
    </div>
    """
                col.markdown(body, unsafe_allow_html=True)

        with st.expander("Overview table"):
            display_overview = overview.copy()
            display_overview["Prob"] = display_overview["Prob"].map(lambda x: fmt_pct(x))
            display_overview["Score"] = display_overview["Score"].map(lambda x: fmt_num(x, 1))
            display_overview["Buy Score Target"] = display_overview.apply(
                lambda row: display_target_value(
                    row, "Buy Score Target", lambda x: fmt_num(x, 3)
                ),
                axis=1,
            )
            display_overview["Buy Prob Target"] = display_overview.apply(
                lambda row: display_target_value(
                    row, "Buy Prob Target", lambda x: fmt_pct(x, 3)
                ),
                axis=1,
            )
            display_overview["Buy EV Target"] = display_overview.apply(
                lambda row: display_target_value(
                    row, "Buy EV Target", lambda x: fmt_num(x, 1, " bps")
                ),
                axis=1,
            )
            display_overview["Mid"] = display_overview["Mid"].map(lambda x: fmt_num(x, 6))
            display_overview["Δ bps"] = display_overview["Δ bps"].map(lambda x: fmt_num(x, 1))
            display_overview["Spread"] = display_overview["Spread"].map(lambda x: fmt_num(x, 1))
            display_overview["Pos %"] = display_overview["Pos %"].map(lambda x: fmt_pct(x))
            display_overview["Projected"] = display_overview["Projected"].map(lambda x: fmt_money(x))
            display_overview["Exposure"] = display_overview["Exposure"].map(lambda x: fmt_money(x))

            st.dataframe(
                display_overview[
                    ["Product", "Status", "Age", "Prob", "Score", "Buy Score Target", "Buy Prob Target", "Buy EV Target", "Mid", "Δ bps", "Spread", "Pos %", "Projected", "Exposure", "Rows"]
                ],
                width="stretch",
                hide_index=True,
                height=210,
            )


    st.markdown(
        '<div class="cb-section">Live calibration targets by coin</div>',
        unsafe_allow_html=True,
    )
    target_rows = []

    for product_id in all_products_for_overview:
        calibration = latest_calibration_for_product(cal, product_id)
        latest_eval = (
            latest_all[
                latest_all["product_id"].astype(str) == product_id
            ].iloc[-1]
            if product_id in latest_all_product_ids
            else pd.Series({"product_id": product_id})
        )
        is_calibrated = bool(
            calibration is not None
            and truthy_cell(calibration.get("is_calibrated", False))
        )
        min_score = (
            calibration.get("min_score", np.nan)
            if calibration is not None else np.nan
        )
        min_prob = (
            calibration.get("min_probability", np.nan)
            if calibration is not None else np.nan
        )
        min_ev = (
            calibration.get("min_expected_value_bps", np.nan)
            if calibration is not None else np.nan
        )
        valid_targets = bool(
            is_calibrated
            and valid_target_number(min_score)
            and valid_target_number(min_prob)
            and valid_target_number(min_ev)
        )

        target_rows.append({
            "Product": product_id,
            "Calibration": "LEARNED" if valid_targets else "AWAITING",
            "Current Score": fmt_num(latest_eval.get("entry_score", np.nan), 3),
            "Score Target": (
                fmt_num(min_score, 3)
                if valid_targets else "Awaiting calibration"
            ),
            "Current Prob": fmt_pct(
                latest_eval.get("estimated_prob_up", np.nan), 3
            ),
            "Prob Target": (
                fmt_pct(min_prob, 3)
                if valid_targets else "Awaiting calibration"
            ),
            "Current EV": fmt_num(
                latest_eval.get("expected_net_edge_bps", np.nan), 1, " bps"
            ),
            "EV Target": (
                fmt_num(min_ev, 1, " bps")
                if valid_targets else "Awaiting calibration"
            ),
            "Raw Prob Median": (
                fmt_pct(
                    calibration.get(
                        "calibrated_raw_probability_median", np.nan
                    ),
                    3,
                )
                if calibration is not None else "—"
            ),
            "Empirical Win Rate": (
                fmt_pct(
                    calibration.get(
                        "calibrated_empirical_win_rate", np.nan
                    ),
                    3,
                )
                if calibration is not None else "—"
            ),
            "Projected Forward": fmt_num(
                latest_eval.get("projected_forward_gain_bps", np.nan),
                1,
                " bps",
            ),
            "Modeled Cost": fmt_num(
                latest_eval.get("cost_bps", np.nan), 1, " bps"
            ),
            "Status": (
                str(calibration.get("calibration_status", "no row"))
                if calibration is not None else "no row"
            ),
        })

    st.dataframe(pd.DataFrame(target_rows), width="stretch", hide_index=True)


    # =============================================================================
    # Product selection
    # =============================================================================

    products = all_products_for_overview
    if not products:
        st.warning("No products found.")
        st.stop()

    top_a, top_b = st.columns([0.52, 0.48], vertical_alignment="center")
    with top_a:
        default_idx = products.index("BTC-USD") if "BTC-USD" in products else 0
        product = st.selectbox("Selected coin", products, index=default_idx, label_visibility="collapsed")

    if pd.notna(latest_market_ts):
        cutoff = float(latest_market_ts) - float(window_minutes) * 60.0
    else:
        cutoff = pd.Timestamp.utcnow().timestamp() - float(window_minutes) * 60.0

    m_prod_live = m_chart[(m_chart["product_id"] == product) & (m["ts"] >= cutoff)].dropna(subset=["ts", "mid"]).copy()
    hist_prod = pd.DataFrame()
    if not hist.empty and "product_id" in hist.columns:
        hist_prod = hist[(hist["product_id"] == product) & (hist["ts"] >= cutoff)].copy()
        if not hist_prod.empty:
            hist_prod["mid"] = hist_prod["close"]
            hist_prod["bid"] = np.nan
            hist_prod["ask"] = np.nan
            hist_prod["spread_bps"] = np.nan

    m_prod = pd.concat([
        hist_prod[[c for c in ["ts", "mid", "bid", "ask", "spread_bps"] if c in hist_prod.columns]],
        m_prod_live[[c for c in ["ts", "mid", "bid", "ask", "spread_bps", "anchored_vwap", "fair_value", "entry_score", "estimated_prob_up", "expected_net_edge_bps", "cash_usd", "equity_usd", "exposures_usd", "position_qty", "target_bps", "projected_forward_gain_bps", "cost_bps", "calibrated_time_to_min_profit_minutes", "calibrated_forward_window_minutes", "position_pct", "current_maker_fee_bps", "current_taker_fee_bps", "fee_tier_reason", "entry_tier", "entry_reason"] if c in m_prod_live.columns]],
    ], ignore_index=True)
    m_prod = m_prod.dropna(subset=["ts", "mid"]).drop_duplicates(subset=["ts"], keep="last").sort_values("ts").copy()
    if m_prod.empty:
        st.warning(f"No recent telemetry rows for {product} in the selected window.")
        st.stop()

    m_prod["dt"] = to_dt_mst(m_prod["ts"])
    latest_eval_rows = latest_by_product(m_eval)
    selected_eval_rows = latest_eval_rows[latest_eval_rows["product_id"] == product] if not latest_eval_rows.empty else pd.DataFrame()
    if not selected_eval_rows.empty:
        latest_row = selected_eval_rows.iloc[-1]
    else:
        latest_row = m_prod.iloc[-1]
        st.warning(f"No eval row is available for {product} yet; buy requirements are waiting for the evaluation loop.")
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

    selected_cal = latest_calibration_for_product(cal, product)
    is_calibrated = False
    calibration_status = "unknown"
    if selected_cal is not None:
        is_calibrated = truthy_cell(
            selected_cal.get("is_calibrated", False)
        )
        calibration_status = str(
            selected_cal.get("calibration_status", "unknown")
        )

    st.markdown(f'<div class="cb-section">{product} buy requirements</div>', unsafe_allow_html=True)
    c0, c1 = st.columns(2)
    with c0:
        mini_card(
            "Calibration",
            "READY" if is_calibrated else "NOT READY",
            calibration_status,
        )
    with c1:
        mini_card(
            "Calibration reason",
            (
                str(selected_cal.get("reason", "—"))
                if selected_cal is not None
                else "No calibration row yet"
            ),
            "",
        )

    if selected_cal is None:
        st.warning("No calibration profile available yet. Waiting for calibration.csv.")
    else:
        current_score = latest_row.get("entry_score", np.nan)
        current_prob = latest_row.get("estimated_prob_up", np.nan)
        current_ev = latest_row.get("expected_net_edge_bps", np.nan)
        min_score = selected_cal.get("min_score", np.nan)
        min_prob = selected_cal.get("min_probability", np.nan)
        min_ev = selected_cal.get("min_expected_value_bps", np.nan)
        # Use the exact persisted booleans from the bot's live buy decision.
        # Recomputing these in the viewer can drift from operational thresholds.
        score_target_ok = truthy_cell(
            latest_row.get("buy_gate_score_ok", False)
        )
        prob_target_ok = truthy_cell(
            latest_row.get("buy_gate_prob_ok", False)
        )
        ev_target_ok = truthy_cell(
            latest_row.get("buy_gate_ev_ok", False)
        )

        targets_are_valid = bool(
            is_calibrated
            and valid_target_number(min_score)
            and valid_target_number(min_prob)
            and valid_target_number(min_ev)
        )

        if targets_are_valid:
            score_display = (
                f"{fmt_num(current_score, 3)} / {fmt_num(min_score, 3)}"
            )
            prob_display = (
                f"{fmt_pct(current_prob, 3)} / {fmt_pct(min_prob, 3)}"
            )
            ev_display = (
                f"{fmt_num(current_ev, 1, ' bps')} / "
                f"{fmt_num(min_ev, 1, ' bps')}"
            )
        else:
            score_display = "Awaiting calibration"
            prob_display = "Awaiting calibration"
            ev_display = "Awaiting calibration"

        b1, b2, b3 = st.columns(3)
        with b1:
            mini_card(
                "Calibrated buy score target",
                score_display,
                "PASS" if score_target_ok and targets_are_valid else "waiting",
            )
        with b2:
            mini_card(
                "Calibrated buy probability target",
                prob_display,
                "PASS" if prob_target_ok and targets_are_valid else "waiting",
            )
        with b3:
            mini_card(
                "Projected net edge",
                ev_display,
                "PASS" if ev_target_ok and targets_are_valid else "waiting",
            )

        st.markdown(
            f'<div class="cb-section">{product} buy requirements</div>',
            unsafe_allow_html=True,
        )

        gate_items = [
            ("Score target", score_target_ok),
            ("Probability target", prob_target_ok),
            ("EV target", ev_target_ok),
        ]

        gcols = st.columns(3)
        for idx, (label, val) in enumerate(gate_items):
            with gcols[idx % 3]:
                mini_card(label, "PASS" if truthy_cell(val) else "BLOCKED", "")

        st.markdown(
            f'<div class="cb-section">{product} execution readiness</div>',
            unsafe_allow_html=True,
        )

        execution_items = [
            ("Fee data ready", latest_row.get("buy_gate_fee_ok", False)),
        ]

        ecols = st.columns(3)
        for idx, (label, val) in enumerate(execution_items):
            with ecols[idx % 3]:
                mini_card(label, "READY" if truthy_cell(val) else "WAITING", "")

        blocker = latest_row.get("buy_gate_blocker", "")
        if pd.notna(blocker) and str(blocker).strip():
            st.caption(f"Buy status: {blocker}")

        with st.expander("Old diagnostic gates"):
            st.write({
                "spread_gate": latest_row.get("buy_gate_spread_ok", False),
                "setup_reversal_gate": latest_row.get("buy_gate_strict_ok", False),
                "target_cost_gate": latest_row.get(
                    "buy_gate_target_cost_ok", False
                ),
                "calibrated_gate": latest_row.get(
                    "buy_gate_calibrated_ok", False
                ),
                "tradeable_signal": latest_row.get(
                    "buy_gate_tradeable", False
                ),
            })

        current_projected_forward = latest_row.get("projected_forward_gain_bps", np.nan)
        current_cost = latest_row.get("cost_bps", np.nan)
        time_to_profit = latest_row.get("calibrated_time_to_min_profit_minutes", np.nan)
        forward_window = latest_row.get("calibrated_forward_window_minutes", np.nan)

        e1, e2, e3 = st.columns(3)
        with e1:
            mini_card(
                "Projected forward gain",
                fmt_num(current_projected_forward, 1, " bps"),
                "historical similar-setup projection",
            )
        with e2:
            mini_card(
                "Modeled cost",
                fmt_num(current_cost, 1, " bps"),
                "fees + spread + buffers",
            )
        with e3:
            mini_card(
                "Calibrated time window",
                fmt_num(time_to_profit, 1, " min"),
                f"window {fmt_num(forward_window, 1, ' min')}",
            )

        st.caption(f"Calibration reason: {selected_cal.get('reason', '—')}")

    st.markdown(f'<div class="cb-section">{product} calibration quality</div>', unsafe_allow_html=True)
    if selected_cal is not None:
        q1, q2, q3 = st.columns(3)
        with q1:
            mini_card(
                "Selected window",
                fmt_num(selected_cal.get("calibrated_selected_window_minutes", np.nan), 1, " min"),
                "historical forward window",
            )
        with q2:
            mini_card(
                "Time to min profit",
                fmt_num(selected_cal.get("calibrated_time_to_min_profit_minutes", np.nan), 1, " min"),
                "median survivable winner",
            )
        with q3:
            mini_card(
                "Extra after min profit",
                fmt_num(selected_cal.get("calibrated_post_profit_extra_gain_bps", np.nan), 1, " bps"),
                "breathing-room upside",
            )

        q4, q5 = st.columns(2)
        with q4:
            mini_card(
                "Raw prob median",
                fmt_pct(
                    selected_cal.get(
                        "calibrated_raw_probability_median", np.nan
                    ),
                    3,
                ),
                "calibration source",
            )
        with q5:
            mini_card(
                "Empirical win rate",
                fmt_pct(
                    selected_cal.get(
                        "calibrated_empirical_win_rate", np.nan
                    ),
                    3,
                ),
                "source outcome rate",
            )

    selected_pt = latest_position_target_for_product(pt, product)
    st.markdown(f'<div class="cb-section">{product} sell plan</div>', unsafe_allow_html=True)
    if selected_pt is None:
        st.info("No sell target snapshot available yet.")
    elif str(selected_pt.get("has_position", "False")).lower() not in ("true", "1", "yes"):
        st.info("No open position. Sell targets will appear after a confirmed buy.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            mini_card("Current bid", fmt_money(selected_pt.get("current_bid", np.nan), 6), f"ask {fmt_money(selected_pt.get('current_ask', np.nan), 6)}")
        with s2:
            mini_card("Minimum profitable exit", fmt_money(selected_pt.get("min_profitable_exit_price", np.nan), 6), f"distance {fmt_num(selected_pt.get('distance_to_min_profit_bps', np.nan), 1, ' bps')}")
        with s3:
            mini_card("Scalp target", fmt_money(selected_pt.get("scalp_target_price", np.nan), 6), f"distance {fmt_num(selected_pt.get('distance_to_scalp_bps', np.nan), 1, ' bps')} · armed: {selected_pt.get('scalp_armed', False)} · pullback {fmt_pct(selected_pt.get('scalp_pullback_pct', np.nan), 2)}")
        with s4:
            mini_card("Core target", fmt_money(selected_pt.get("core_target_price", np.nan), 6), f"distance {fmt_num(selected_pt.get('distance_to_core_bps', np.nan), 1, ' bps')} · armed: {selected_pt.get('core_armed', False)} · pullback {fmt_pct(selected_pt.get('core_pullback_pct', np.nan), 2)}")

        s5, s6 = st.columns(2)
        with s5:
            mini_card("Scalp pullback trigger", fmt_money(selected_pt.get("scalp_pullback_trigger_price", np.nan), 6), f"peak {fmt_money(selected_pt.get('scalp_arm_peak', np.nan), 6)}")
        with s6:
            mini_card("Core pullback trigger", fmt_money(selected_pt.get("core_pullback_trigger_price", np.nan), 6), f"peak {fmt_money(selected_pt.get('core_arm_peak', np.nan), 6)}")
        p1, p2, p3 = st.columns(3)
        with p1:
            mini_card(
                "Profit lock",
                "ARMED" if truthy_cell(selected_pt.get("profit_lock_armed", False)) else "waiting",
                fmt_money(selected_pt.get("profit_lock_price", np.nan), 6),
            )
        with p2:
            mini_card(
                "Forward window",
                fmt_num(selected_pt.get("calibrated_forward_window_minutes", np.nan), 1, " min"),
                "expected time horizon",
            )
        with p3:
            mini_card(
                "Breathing room",
                fmt_num(selected_pt.get("calibrated_post_profit_breathing_minutes", np.nan), 1, " min"),
                "after min profit",
            )
        st.caption(str(selected_pt.get("exit_plan_note", "")))


    with st.expander(f"{product} calibration replay candidates"):
        if cr.empty or "product_id" not in cr.columns:
            st.info("No candidate_replay.csv rows yet.")
        else:
            crp = cr[cr["product_id"] == product].copy()
            if crp.empty:
                st.info("No replay candidates for this product yet.")
            else:
                show_cols = [
                    "dt_mst", "timeframe", "score", "probability",
                    "expected_net_edge_bps", "cost_bps",
                    "selected_forward_window_minutes", "max_favorable_bps",
                    "max_adverse_bps", "adverse_before_profit_bps",
                    "time_to_min_profit_minutes", "post_profit_extra_gain_bps",
                    "reached_min_profit", "survived_to_profit",
                    "accepted_by_calibration",
                ]
                show_cols = [column for column in show_cols if column in crp.columns]
                st.dataframe(
                    crp.sort_values("ts", ascending=False)[show_cols].head(250),
                    width="stretch",
                    hide_index=True,
                )

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
    fig = plot_price_plotly(
        m_prod,
        title=f"{product} · last {window_minutes} min",
        show_bid_ask=show_bid_ask,
        trades=t_prod,
        sell_target_row=selected_pt,
    )
    st.plotly_chart(fig, width="stretch", key=f"live_price_chart_{product}")


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
            st.dataframe(o_sorted[compact_cols].head(5), width="stretch", height=175, hide_index=True)
    with r2:
        st.caption("Recent confirmed trades")
        if t_sorted.empty:
            st.write("No confirmed trades.")
        else:
            compact_cols = [c for c in ["dt_mst", "event", "product_id", "side", "qty", "price", "fee_usd", "net_pnl_usd", "note"] if c in t_sorted.columns]
            st.dataframe(t_sorted[compact_cols].head(5), width="stretch", height=175, hide_index=True)


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
                        st.pyplot(figm, clear_figure=True, width="stretch")


    # =============================================================================
    # Expandable older data
    # =============================================================================

    with st.expander("Older order attempts"):
        if o_sorted.empty:
            st.write("No order attempts logged yet.")
        else:
            show_cols = [c for c in ["dt_mst", "event", "product_id", "side", "mode", "requested_quote_usd", "requested_base_qty", "ok", "status", "filled_qty", "avg_price", "filled_notional_usd", "fee_usd", "reason", "raw_error"] if c in o_sorted.columns]
            st.dataframe(o_sorted[show_cols].head(150), width="stretch", height=420, hide_index=True)
    with st.expander("Older confirmed trades"):
        if t_sorted.empty:
            st.write("No confirmed trades logged yet.")
        else:
            show_cols = [c for c in ["dt_mst", "event", "product_id", "side", "qty", "price", "fee_usd", "gross_pnl_usd", "net_pnl_usd", "cum_pnl_usd", "entry_price", "exit_price", "exit_role", "note"] if c in t_sorted.columns]
            st.dataframe(t_sorted[show_cols].head(150), width="stretch", height=420, hide_index=True)
    if show_debug_tables:
        with st.expander("Market telemetry debug"):
            debug_cols = [c for c in ["ts", "product_id", "bid", "ask", "mid", "spread_bps", "cash_usd", "equity_usd", "entry_score", "entry_tier", "estimated_prob_up", "position_pct", "target_bps", "projected_forward_gain_bps", "cost_bps", "calibrated_time_to_min_profit_minutes", "calibrated_forward_window_minutes", "entry_reason"] if c in m.columns]
            st.dataframe(m.sort_values("ts", ascending=False)[debug_cols].head(300), width="stretch", height=420, hide_index=True)

if live_update and HAS_NATIVE_FRAGMENT_REFRESH:
    # Streamlit-native auto-refresh. This is the preferred mode.
    @st.fragment(run_every=f"{int(refresh_sec)}s")
    def _live_fragment() -> None:
        render_live_dashboard()

    _live_fragment()

elif live_update and st_autorefresh is not None:
    # Fallback if native fragments are unavailable.
    st_autorefresh(interval=int(refresh_sec * 1000), key="live_data_update_fallback")
    render_live_dashboard()

else:
    # Manual mode or missing refresh engine.
    if live_update and st_autorefresh is None and not HAS_NATIVE_FRAGMENT_REFRESH:
        st.error(
            "Live update is enabled, but neither st.fragment nor streamlit-autorefresh is available. "
            "Upgrade Streamlit or install streamlit-autorefresh."
        )
    render_live_dashboard()
