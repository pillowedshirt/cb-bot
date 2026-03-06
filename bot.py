# ============================================================
# UPDATED: Option-1 BUY entry gate (Support -> Reversal -> Room)
# - Replaced the previous time-windowed bottoming/stabilization entry gate with:
#     A) Support zone proximity (MacroLevels day/week)
#     B) Event-based reversal confirmation (2-of-3: RSI flip, EMA reclaim, HL->break)
#     C) Room to +2% before nearby resistance/mean magnets
# - Existing spread filter, buy cooldown, sizing, sell logic, fill-truth, logs/CSVs unchanged.
# ============================================================

# ============================================================
# CHANGELOG (SELL SIMPLIFICATION)
# - TradingBot.run: replaced all staggered/target/volatility-based sell logic with:
#     * HARD_PEAK_STOP_PCT (0.5% drawdown from peak bid since entry) => full exit
#     * TRAIL_ARM_PCT (1.5% peak profit) + TRAIL_DRAWDOWN_PCT (0.25% drawdown) => full exit
# - TradingBot.run._sell_all: full-exit reset now clears ladder_plan + lot_tags in addition to prior state resets
# - Configuration constants: disabled legacy sell constants and added TRAIL_ARM_PCT / TRAIL_DRAWDOWN_PCT / HARD_PEAK_STOP_PCT
# ============================================================

import os
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import csv
import math
import asyncio
import uuid
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Deque, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
import websockets
from dotenv import load_dotenv

from coinbase.rest import RESTClient
from coinbase import jwt_generator


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
TZ_NAME: str = "America/Phoenix"
TZ = ZoneInfo(TZ_NAME)

# ============================================================
# CONFIGURATION
# ============================================================

# Paper trading mode. If True, no real orders are sent on Coinbase.
PAPER_TRADING: bool = False

# Starting account balance in USD (paper mode)
STARTING_CASH_USD: float = 100.0


# If True, treat USD 'hold' as tradable buying power (useful for instant-deposit trading).
# Coinbase may allow trading immediately while keeping deposits on a withdrawal hold; in that case
# available_balance can be 0 while hold is positive. Enabling this will use available_balance + hold.
USE_USD_HOLD_AS_TRADABLE: bool = True

# Optional session filter (UTC). Disabled by default to preserve existing behaviour.
# If enabled, the entry gate will only allow buys during the configured UTC hours.
ENABLE_SESSION_FILTER: bool = False
SESSION_ALLOWED_UTC_HOURS: Optional[List[int]] = list(range(13, 23))  # 13:00–22:59 UTC (US/EU overlap)

# Fixed trade sizing parameters
MAX_EXPOSURE_PER_PRODUCT_USD: float = 40.0  # maximum inventory cost per product
ENTRY_SIZE_USD: float = 4.0                # amount to spend on each laddered entry
SELL_PROPORTIONS: List[float] = []  # DISABLED (no staggered/partial sells)

# Fees (basis points). 1 bp = 0.01%.
MAKER_FEE_BPS: float = 6.0
TAKER_FEE_BPS: float = 10.0

# Disable all legacy take-profit / stagger / stop logic (replaced by peak-based full-exit)
MIN_TAKE_PROFIT_BPS: float = 0.0  # DISABLED
MIN_TAKE_PROFIT_PCT: float = 0.0  # DISABLED
STOP_LOSS_PCT: float = 0.0        # DISABLED (replaced by peak-based HARD_PEAK_STOP_PCT)
PROFIT_LOCK_IN_PCT: float = 0.0   # DISABLED
PROFIT_LOCK_IN_DRAWDOWN_PCT: float = 0.0  # DISABLED

# Simplified SELL logic (peak-based trailing + peak stop)
TRAIL_ARM_PCT: float = 0.015          # +1.5% arms trailing
TRAIL_DRAWDOWN_PCT: float = 0.0025    # 0.25% drop from peak triggers sell AFTER armed
HARD_PEAK_STOP_PCT: float = 0.005     # 0.5% drop from peak triggers sell ANYTIME

# Disable volatility-weighted trailing band system (legacy)
TRAIL_VOL_WINDOW_MIN: int = 0  # DISABLED
TRAIL_K_BASE: float = 0.0  # DISABLED
TRAIL_K_MIN: float = 0.0  # DISABLED
TRAIL_K_MAX: float = 0.0  # DISABLED
TRAIL_MIN_DRAWDOWN_PCT: float = 0.0  # DISABLED
TRAIL_MAX_DRAWDOWN_PCT: float = 0.0  # DISABLED

# Laddered entry sizing (fractions of all-in notional).
LADDER_FRACS: Tuple[float, float, float] = (0.50, 0.30, 0.20)

# Tranche-2 add trigger: add when price dips below entry1 by ADD2_K * σ (σ from 60m rolling 1m returns).
ADD2_K_SIGMA: float = 0.75

# Tranche-3 add trigger: add when price reclaims 24h anchored VWAP.
ADD3_RECLAIM_AVWAP: bool = True

# Exit targets for tranches:
TRANCHE1_TP_PCT: float = 0.010    # +1% baseline gain
TRANCHE2_TARGET_SIGMA: float = 2.0 # anchored VWAP + 2σ (price units)
# Product universe filter: require ~30% single-day range volatility (high-low)/close.
# This is computed from the most recent daily candle via Coinbase Exchange public market-data.
MIN_DAILY_RANGE_PCT: float = 0.30

# Products to trade. High liquidity pairs only.
# If AUTO_SELECT_PRODUCTS is True, this list is treated as a fallback default.
PRODUCTS_DEFAULT: List[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD"
]

# Auto-selection (diversify volatility + liquidity):
# We pick a set of USD pairs that (a) are liquid on Coinbase Exchange and
# (b) tend to have higher realized volatility when BTC is quiet, while keeping
# correlations in the basket lower.
AUTO_SELECT_PRODUCTS: bool = False
TARGET_PRODUCT_COUNT: int = 5          # total products to trade (includes BTC if available)
CANDIDATE_TOP_BY_USD_VOL: int = 40      # only consider the top-N USD-volume products (liquidity filter)
SELECTION_LOOKBACK_DAYS: int = 140      # daily bars to pull for correlation/volatility scoring
SELECTION_BTC_QUIET_ROLL_DAYS: int = 14 # define "BTC quiet" by rolling vol over this many days
SELECTION_REFRESH_SEC: int = 6 * 60 * 60  # recompute selection every 6 hours
# Correlation / stress controls (reduce synchronized dumps)
SELECTION_BTC_STRESS_QUANTILE: float = 0.75   # top quartile of BTC rolling vol = "stress"
MAX_ABS_CORR_TO_BTC: float = 0.88             # hard cap on |corr| to BTC for non-BTC assets
MAX_AVG_ABS_CORR: float = 0.82                # soft cap on avg |corr| to current basket
PRODUCTS_CACHE_PATH: str = os.path.join(BASE_DIR, "products_selected.json")

# Runtime products list (populated at startup in main()).
PRODUCTS: List[str] = list(PRODUCTS_DEFAULT)


# File paths for logging
# Resolve paths relative to this script so that bot and viewer always refer to
# the same files regardless of the current working directory.
TRADES_CSV_PATH: str = os.path.join(BASE_DIR, "trades.csv")
MARKET_CSV_PATH: str = os.path.join(BASE_DIR, "market.csv")
#
# Macro candle CSVs should live in the current working directory rather than being
# anchored to the script directory.  In practice, the viewer and bot run from
# the project root, so keeping these as plain filenames ensures both components
# read/write the same files.  Use relative filenames here instead of
# os.path.join(BASE_DIR, ...).
MACRO_WEEK_CSV: str = "macro_week.csv"  # 15‑minute candles (past week)
MACRO_DAY_CSV: str = "macro_day.csv"    # 1‑minute candles (past day)

# Cadence for macro refresh
MACRO_REFRESH_EVERY_SEC: int = 3 * 60  # every 3 minutes

# WebSocket configuration
WS_MARKET_URL: str = "wss://advanced-trade-ws.coinbase.com"
WS_PING_INTERVAL: int = 20
WS_PING_TIMEOUT: int = 20
WS_RECONNECT_DELAY_SEC: int = 3

# Preload micro context (minutes of 1m candles) on startup
MICRO_PRELOAD_MINUTES: int = 1440


# Sigma window (minutes of 1m candles for volatility calc)
SIGMA_WINDOW_MINUTES: int = 60

# Require a full day of 1m candles before using in-memory live_1m for daily levels.
DAY_CANDLES_MIN_FOR_LIVE: int = 60 * 24  # 1440

# Evaluation loop cadence (seconds). Slows decision making to 1m-scale behavior.
EVAL_TICK_SEC: float = 2.0

# After any full exit, wait before allowing a new entry (prevents rapid flip-flopping).
POST_EXIT_COOLDOWN_SEC: float = 5 * 60.0

# Price-based re-entry re-arm: after any full exit, the bot will NOT re-enter
# until price has first moved ABOVE the day support zone by this buffer (in bps),
# and then later returns back into the support zone.
REENTRY_REARM_BPS: float = 15.0

# Bias gating: do not open new longs if weekly bias is below this threshold
WEEKLY_BIAS_THRESHOLD: float = -0.5

# ============================================================
# BUY/SELL LOGIC TUNING (paper mode)
#
# These parameters ONLY affect trade decision logic and sizing.
# All other bot behaviors (feeds, logging, viewer formats) remain unchanged.
# ============================================================

# Warm-up: do not permit the very first buy until this many seconds after startup.
FIRST_BUY_DELAY_SEC: float = 0.0 * 60.0  # 30 minutes


# Compounding sizing: position sizes scale with equity so profits are reinvested.
ENTRY_RISK_FRAC: float = 0.008        # 0.8% of equity per entry
MAX_ENTRY_FRAC: float = 0.03          # cap any single entry at 3% equity
MAX_EXPOSURE_FRAC: float = 0.18       # cap per-product exposure at 18% equity

# Dynamic cash utilization scaling:
# - Base utilization target starts around 50% (your current behavior)
# - As more ladder entries are taken (more "active buy orders"), we allow higher utilization
# - This increases per-entry notional when we're under-utilized and tempers it if we're over
CASH_UTIL_BASE: float = 0.50          # baseline target utilization (50%)
CASH_UTIL_MAX: float = 0.90           # never target more than 90% utilization
CASH_UTIL_PER_BUY: float = 0.08       # +8% target utilization per filled ladder level (cap at CASH_UTIL_MAX)
CASH_UTIL_GAIN: float = 1.50          # how aggressively to scale up notional when under target
CASH_UTIL_DECAY: float = 1.00         # how aggressively to scale down notional when above target
CASH_UTIL_MIN_SCALE: float = 0.50     # never scale a base entry below 50%
MIN_ENTRY_USD: float = 1.0             # safety floor; primary sizing is % of available cash


# Cash-based entry sizing (requested):
# We treat each PositionLot as one "position slot" across the whole account.
# The next buy uses a percentage of AVAILABLE cash that increases as more slots are filled:
#   0 other open slots -> 10.0% of available cash
#   1 other open slots -> 12.5%
#   2 other open slots -> 15.0%
#   ...
# Up to MAX_OPEN_POSITIONS slots total; the final slot invests the remaining cash (minus fees).
MAX_OPEN_POSITIONS: int = 10
ENTRY_PCT_BASE: float = 0.10
ENTRY_PCT_STEP: float = 0.025

# Execution quality filters
MAX_SPREAD_BPS: float = 20.0          # skip trading if spread too wide

# ============================================================
# TIERED BUY/SELL EXTENSION
# ============================================================

# Tier names (higher number = higher priority)
TIER_LOW = 1
TIER_MID = 2
TIER_HIGH = 3

# Profit-taking targets for tiers (these are TP sells; risk exits still use peak stop logic)
TIER_TP_PCT = {
    TIER_LOW: 0.0025,   # +0.25%
    TIER_MID: 0.0075,   # +0.75% (in your requested 0.5–1% band)
    TIER_HIGH: 0.015,   # +1.5% (leave room for +2%+ runners)
}

# How much of the position to sell at tier TP when hit
# (Low tier: mostly scalp out; Mid: partial; High: smallest partial to keep runners)
TIER_TP_SELL_FRAC = {
    TIER_LOW: 0.85,
    TIER_MID: 0.55,
    TIER_HIGH: 0.25,
}

# Prevent TP spam (seconds between tier TP sells per product)
TIER_TP_COOLDOWN_SEC = 90.0

# Rotation rule: if holding tier < candidate_tier by at least this delta, rotate
ROTATE_MIN_TIER_DELTA = 1

# Require minimum spread quality for scalps (you can tighten later)
SCALP_MAX_SPREAD_BPS: float = 12.0

# Option-1 / tiered-entry support-zone buffer (basis points)
SUPPORT_BUFFER_BPS: float = 20.0
RSI_BUY_THRESHOLD: float = 35.0
EMA_ENTRY_FAST: int = 9
EMA_ENTRY_SLOW: int = 20
EMA_SLOPE_MAX_DOWN_BPS: float = 12.0
PIVOT_W: int = 2
RESIST_BUFFER_BPS: float = 15.0
REQUIRE_CONFIRMATIONS: int = 2

# Ladder discipline
BUY_COOLDOWN_SEC: float = 45.0        # minimum time between entries
BUY_MIN_SPACING_SIGMA: float = 0.25   # require mid to be <= last_buy_price - (0.25*sigma)

# Regime filter (avoid mean reversion entries during strong downtrends)
EMA_FAST_MINUTES: int = 20
EMA_SLOW_MINUTES: int = 60
MAX_TREND_STRENGTH_BPS: float = 35.0  # if |EMA_fast-EMA_slow|/price > 35 bps => trending
# Entry gating (bottoming + stabilization model on 1m candles)
LOCAL_LOW_LOOKBACK_MIN: int = 75      # local low must occur within this many minutes
LOCAL_LOW_MIN_AGE_MIN: int = 8        # require at least this many minutes since the low (avoid buying the exact knife tip)
BASE_BUILD_MIN: int = 12              # "no new lows" base-building time requirement
DECEL_WINDOW_MIN: int = 15            # window for momentum slope checks
VOL_CONTRACTION_SHORT_MIN: int = 10   # short vol window
VOL_CONTRACTION_LONG_MIN: int = 30    # long vol window
RECLAIM_VWAP_BPS: float = 0.0         # require mid >= anchored_vwap_24h * (1+RECLAIM_VWAP_BPS)
ENTRY_NEAR_LOW_MAX_PCT: float = 0.012 # must enter within 1.2% of local low (upper bound)
ENTRY_NEAR_LOW_SIGMA_MULT: float = 0.90  # or within this many sigma above the low (sigma from 60m 1m-returns)
MIN_RELIEF_RALLY_BPS: float = -8.0    # last few minutes should not be strongly negative (avoid catching knife)


# Exit tuning

# Target staggering:
# We enforce a minimum time gap between profit-target sells so they don't all fire within minutes.
# Base desired spacing is ~10 minutes between targets; we adapt slightly to fast/slow micro conditions.
SELL_TARGET_GAP_SEC_BASE: float = 0.0   # DISABLED
SELL_TARGET_GAP_SEC_MIN: float = 0.0    # DISABLED
SELL_TARGET_GAP_SEC_MAX: float = 0.0    # DISABLED
EXTRA_PROFIT_BPS: float = 0.0  # DISABLED

# Fair value smoothing (reduce sudden "dips" in fair value):
# - We compute a raw fair value (anchored VWAP / value-area midpoint / vwap fallback)
# - Then apply a small median filter + EWMA + per-tick max-step clamp in bps.
FAIR_VALUE_MEDIAN_WINDOW: int = 9
FAIR_VALUE_SMOOTH_ALPHA: float = 0.12
FAIR_VALUE_MAX_STEP_BPS: float = 30.0
FAIR_VALUE_MAX_STEP_DOWN_BPS: float = 18.0

# Runner + safety exits
TRAIL_AFTER_TARGET: int = 0           # DISABLED
TRAIL_SIGMA_MULT: float = 0.0        # DISABLED
TIME_STOP_SEC: int = 6 * 60 * 60      # 6 hours
RISK_OFF_REDUCTION_FRAC: float = 0.05  # reduce position by 5% on risk-off events
RISK_OFF_COOLDOWN_SEC: float = 60.0     # throttle risk-off/time-stop reductions to once per minute
RISK_OFF_MIN_NOTIONAL_USD: float = 1.0  # if reduction notional < $1, liquidate remainder to avoid dust spam

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def now_ts() -> float:
    """Return current UNIX timestamp as float."""
    return time.time()


def now_ts_i() -> int:
    """Return current UNIX timestamp as int seconds."""
    return int(time.time())


def bps_to_mult(bps: float) -> float:
    """Convert basis points to multiplicative factor (e.g. 50 bps → 1.005)."""
    return 1.0 + (bps / 10_000.0)


def fee_usd(notional_usd: float, fee_bps: float) -> float:
    """Return fee in USD for a given notional and fee rate."""
    return notional_usd * (fee_bps / 10_000.0)


def safe_float(x: Any) -> Optional[float]:
    """Convert to float if possible, else None."""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into the inclusive range [lo, hi]."""
    return float(max(lo, min(hi, x)))

# ------------------------------------------------------------
# Product auto-selection (liquidity + diversification)
# ------------------------------------------------------------

def _http_get_json(url: str, timeout: float = 12.0) -> Any:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "cb-bot/1.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _iso(ts: int) -> str:
    # Coinbase Exchange REST expects ISO8601 timestamps.
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def _fetch_exchange_products() -> List[Dict[str, Any]]:
    data = _http_get_json("https://api.exchange.coinbase.com/products")
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def _fetch_volume_summary() -> Dict[str, Dict[str, float]]:
    # Docs: GET /products/volume-summary (Coinbase Exchange market-data)
    # Returns 24h + 30d volumes for all products.
    # Shape varies; we normalize into {product_id: {usd_vol_24h, base_vol_24h}}.
    out: Dict[str, Dict[str, float]] = {}
    data = _http_get_json("https://api.exchange.coinbase.com/products/volume-summary")
    if not data:
        return out

    items = None
    if isinstance(data, dict):
        for k in ("data", "products", "volume_summary", "volumeSummary", "volume-summary"):
            v = data.get(k)
            if isinstance(v, list):
                items = v
                break
    if items is None and isinstance(data, list):
        items = data

    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        pid = it.get("product_id") or it.get("productId") or it.get("id")
        if not isinstance(pid, str):
            continue
        # try a few likely keys
        quote_vol = safe_float(it.get("quote_volume_24h") or it.get("quote_volume") or it.get("quoteVolume"))
        base_vol = safe_float(it.get("base_volume_24h") or it.get("volume_24h") or it.get("volume") or it.get("baseVolume"))
        if quote_vol is None:
            quote_vol = 0.0
        if base_vol is None:
            base_vol = 0.0
        out[pid] = {"usd_vol_24h": float(quote_vol), "base_vol_24h": float(base_vol)}
    return out


def _fetch_daily_closes(product_id: str, days: int) -> Optional[List[Tuple[int, float]]]:
    # Docs: GET /products/{product_id}/candles with granularity=86400 for daily candles.
    # Candle format is [time, low, high, open, close, volume].
    end_ts = int(now_ts())
    start_ts = end_ts - int(days) * 86400
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles?granularity=86400&start={_iso(start_ts)}&end={_iso(end_ts)}"
    data = _http_get_json(url)
    if not isinstance(data, list):
        return None
    out: List[Tuple[int, float]] = []
    for row in data:
        if isinstance(row, (list, tuple)) and len(row) >= 5:
            t = int(float(row[0]))
            close = float(row[4])
            if t > 0 and close > 0:
                out.append((t, close))
        elif isinstance(row, dict):
            t = int(float(row.get("time") or row.get("start") or row.get("ts") or 0))
            close = safe_float(row.get("close"))
            if t > 0 and close is not None and close > 0:
                out.append((t, float(close)))
    if not out:
        return None
    out.sort(key=lambda x: x[0])
    # Deduplicate timestamps
    uniq: Dict[int, float] = {}
    for t, c in out:
        uniq[t] = c
    merged = sorted(uniq.items(), key=lambda x: x[0])
    return merged


def _fetch_candles_public(
    *,
    product_id: str,
    granularity: int,
    limit: int = 300,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> List[List[float]]:
    """Fetch public candles from Coinbase Exchange market-data endpoint.

    Returns rows in the canonical Exchange format:
        [time, low, high, open, close, volume]

    Notes:
      - This is a *public* endpoint used only for product universe selection / filters.
      - It is intentionally synchronous because it's only used at startup / periodic selection.
      - Callers must not assume ordering; we sort by time ascending before returning.
    """
    try:
        pid = str(product_id)
        gran = int(granularity)
        lim = int(limit)
        lim = max(1, min(lim, 300))  # Exchange endpoint returns up to ~300 rows per call

        if end is None:
            end_ts = int(now_ts())
        else:
            end_ts = int(end)
        if start is None:
            start_ts = end_ts - (lim * gran)
        else:
            start_ts = int(start)

        url = (
            f"https://api.exchange.coinbase.com/products/{pid}/candles"
            f"?granularity={gran}&start={_iso(start_ts)}&end={_iso(end_ts)}"
        )
        data = _http_get_json(url)
        if not isinstance(data, list):
            return []
        rows: List[List[float]] = []
        for row in data:
            if isinstance(row, (list, tuple)) and len(row) >= 6:
                t = int(float(row[0]))
                lo = float(row[1]); hi = float(row[2]); op = float(row[3]); cl = float(row[4]); vol = float(row[5])
                rows.append([t, lo, hi, op, cl, vol])
            elif isinstance(row, dict):
                t = int(float(row.get("time") or row.get("start") or row.get("ts") or 0))
                lo = safe_float(row.get("low")); hi = safe_float(row.get("high"))
                op = safe_float(row.get("open")); cl = safe_float(row.get("close"))
                vol = safe_float(row.get("volume")) or 0.0
                if t > 0 and lo is not None and hi is not None and op is not None and cl is not None:
                    rows.append([t, float(lo), float(hi), float(op), float(cl), float(vol)])
        if not rows:
            return []
        rows.sort(key=lambda r: r[0])
        # Keep only the most recent `limit` rows (ascending)
        if len(rows) > lim:
            rows = rows[-lim:]
        return rows
    except Exception:
        return []


def _fetch_recent_daily_range_pct(product_id: str) -> Optional[float]:
    """
    Returns the most recent daily (high-low)/close range as a fraction, e.g. 0.30 = 30%.
    Uses Coinbase Exchange public candles endpoint (86400 granularity).
    """
    try:
        # Pull the last ~3 days to be robust to partial current-day candles.
        rows = _fetch_candles_public(product_id=product_id, granularity=86400, limit=3)
        if not rows:
            return None
        # Rows are [time, low, high, open, close, volume]
        rows = sorted(rows, key=lambda r: r[0])
        # Choose the last FULL candle (often the last item is current partial day). Heuristic:
        # - If last candle timestamp is within the past 18 hours, use the prior candle.
        now = int(time.time())
        last = rows[-1]
        t_last = int(last[0])
        use = last
        if (now - t_last) < int(18 * 60 * 60) and len(rows) >= 2:
            use = rows[-2]
        low = float(use[1]); high = float(use[2]); close = float(use[4])
        if close <= 0:
            return None
        return max(0.0, (high - low) / close)
    except Exception:
        return None


def _series_to_returns(series: List[Tuple[int, float]]) -> pd.Series:
    # index by date (ts)
    s = pd.Series({t: c for t, c in series}).sort_index()
    return s.pct_change().dropna()


def select_diversified_products() -> List[str]:
    """Select USD products with (1) liquidity and (2) volatility when BTC is quiet, while reducing correlation."""
    # Cache first
    try:
        if os.path.exists(PRODUCTS_CACHE_PATH):
            with open(PRODUCTS_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict):
                ts = safe_float(cached.get("ts"))
                prods = cached.get("products")
                if ts is not None and isinstance(prods, list) and (now_ts() - ts) < SELECTION_REFRESH_SEC:
                    prods2 = [p for p in prods if isinstance(p, str)]
                    if len(prods2) >= 2:
                        return prods2
    except Exception:
        pass

    # Pull Coinbase Exchange product list + volume summary (public market-data APIs).
    products = _fetch_exchange_products()
    vol_map = _fetch_volume_summary()

    usd_pairs: List[str] = []
    for p in products:
        if p.get("quote_currency") != "USD":
            continue
        if p.get("status") not in (None, "online"):
            continue
        if p.get("trading_disabled") is True:
            continue
        pid = p.get("id")
        if isinstance(pid, str) and "-" in pid:
            usd_pairs.append(pid)

    # Liquidity filter: keep top-N by quote (USD) volume when available.
    scored: List[Tuple[str, float]] = []
    for pid in usd_pairs:
        v = vol_map.get(pid, {}).get("usd_vol_24h", 0.0)
        scored.append((pid, float(v)))
    scored.sort(key=lambda x: x[1], reverse=True)
    candidates = [pid for pid, _ in scored[:max(10, CANDIDATE_TOP_BY_USD_VOL)]]

    # Hard volatility filter: require ~30% single-day range volatility (high-low)/close.
    # This matches your request to trade ONLY coins with large 24h swings.
    vol_ok: List[str] = []
    for pid in candidates:
        rng = _fetch_recent_daily_range_pct(pid)
        if rng is None:
            continue
        if rng >= MIN_DAILY_RANGE_PCT:
            vol_ok.append(pid)

    # If the strict filter removes everything, fall back to the original candidates so the bot can still run.
    # (You can tighten/loosen MIN_DAILY_RANGE_PCT at the top.)
    if vol_ok:
        candidates = vol_ok

    # Ensure BTC is considered (anchor).
    if "BTC-USD" not in candidates and "BTC-USD" in usd_pairs:
        candidates = ["BTC-USD"] + candidates[:-1]

    # Fetch daily returns for candidates.
    rets: Dict[str, pd.Series] = {}
    for pid in candidates:
        series = _fetch_daily_closes(pid, SELECTION_LOOKBACK_DAYS)
        if not series:
            continue
        r = _series_to_returns(series)
        if len(r) >= 60:
            rets[pid] = r

    if "BTC-USD" not in rets:
        # if BTC data missing, fall back
        return list(PRODUCTS_DEFAULT)

    # Align on common dates
    df = pd.DataFrame({k: v for k, v in rets.items()}).dropna(how="any")
    if df.empty or df.shape[0] < 60:
        return list(PRODUCTS_DEFAULT)

    # Liquidity proxy for scoring
    usd_vol = pd.Series({pid: vol_map.get(pid, {}).get("usd_vol_24h", 0.0) for pid in df.columns}).replace(0.0, np.nan)
    usd_vol = usd_vol.fillna(usd_vol.median() if not usd_vol.dropna().empty else 1.0)

    # BTC "quiet" days = bottom quartile of rolling vol
    btc = df["BTC-USD"]
    btc_roll = btc.rolling(SELECTION_BTC_QUIET_ROLL_DAYS).std()
    thresh = float(np.nanquantile(btc_roll.values, 0.25))
    quiet_mask = (btc_roll <= thresh)
    quiet_df = df[quiet_mask].dropna(how="any")
    if quiet_df.shape[0] < 20:
        quiet_df = df.copy()

    # Volatility on quiet days (std of returns)
    vol_quiet = quiet_df.std().replace(0.0, np.nan)

    # Correlation matrix (full period)
    corr = df.corr().fillna(0.0)

    # BTC "stress" days = top quantile of rolling vol (risk-off cascades tend to synchronize here)
    stress_thresh = float(np.nanquantile(btc_roll.values, SELECTION_BTC_STRESS_QUANTILE))
    stress_mask = (btc_roll >= stress_thresh)
    stress_df = df[stress_mask].dropna(how="any")
    corr_stress = stress_df.corr().fillna(0.0) if stress_df.shape[0] >= 20 else corr

    # Standardize scoring components
    def zscore(s: pd.Series) -> pd.Series:
        mu = float(s.mean())
        sd = float(s.std()) if float(s.std()) > 1e-12 else 1.0
        return (s - mu) / sd

    vol_z = zscore(vol_quiet)
    liq_z = zscore(np.log1p(usd_vol))

    selected: List[str] = []
    if "BTC-USD" in df.columns:
        selected.append("BTC-USD")

    # Greedy add: maximize volatility when BTC is quiet, penalize correlation to selected
    while len(selected) < max(2, TARGET_PRODUCT_COUNT):
        best_pid = None
        best_score = -1e9
        for pid in df.columns:
            if pid in selected:
                continue
            # average absolute correlation to current basket
            avg_corr = float(np.mean([abs(float(corr.loc[pid, s])) for s in selected])) if selected else 0.0
            corr_to_btc = abs(float(corr.loc[pid, "BTC-USD"])) if "BTC-USD" in corr.columns else 0.0
            avg_corr_stress = float(np.mean([abs(float(corr_stress.loc[pid, s])) for s in selected])) if selected else 0.0

            # Hard correlation gate to avoid "everything dumps together" baskets.
            if pid != "BTC-USD" and corr_to_btc > MAX_ABS_CORR_TO_BTC:
                continue
            if selected and avg_corr > MAX_AVG_ABS_CORR and len(selected) >= 2:
                continue

            # Score: prefer (a) volatility on BTC-quiet days, (b) liquidity, and
            # penalize correlation both in normal conditions and in BTC stress regimes.
            score = float(
                0.55 * vol_z.get(pid, 0.0)
                + 0.20 * liq_z.get(pid, 0.0)
                - 0.22 * avg_corr
                - 0.22 * avg_corr_stress
                - 0.12 * corr_to_btc
            )
            if score > best_score:
                best_score = score
                best_pid = pid
        if best_pid is None:
            break
        selected.append(best_pid)

    # Cache
    try:
        with open(PRODUCTS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"ts": now_ts(), "products": selected}, f, indent=2)
    except Exception:
        pass

    return selected


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MacroLevels:
    support_zone_low: float
    support_zone_high: float
    resistance_zone_low: float
    resistance_zone_high: float
    breakout: float
    range_low: float
    range_high: float
    prev_low: float
    prev_high: float
    vwap: float
    psych_low: float
    psych_high: float
    val: float
    vah: float
    price_now: float


@dataclass
class PositionLot:
    qty: float
    price: float


class MacroManager:
    """
    Stores macro levels per product and timeframe (week or day).
    Provides a function to compute weekly bias for gating.
    """

    def __init__(self) -> None:
        self.levels: Dict[str, Dict[str, MacroLevels]] = {p: {} for p in PRODUCTS}

    def set_levels(self, product_id: str, timeframe: str, levels: MacroLevels) -> None:
        if levels is not None:
            self.levels.setdefault(product_id, {})[timeframe] = levels

    def get_levels(self, product_id: str, timeframe: str) -> Optional[MacroLevels]:
        return self.levels.get(product_id, {}).get(timeframe)

    def compute_weekly_bias(self, product_id: str, price: float) -> Optional[float]:
        """
        Compute weekly bias in [-1,+1] using weekly macro levels and current price.
        A positive score indicates price near support or below VWAP; a negative score indicates
        price near resistance or above VWAP. We use a similar formula as before, but
        emphasise support proximity and value area position.
        """
        levels = self.get_levels(product_id, "week")
        if not levels or price <= 0:
            return None
        # Distance to support/resistance centres
        sup_mid = (levels.support_zone_low + levels.support_zone_high) / 2.0
        sup_width = levels.support_zone_high - levels.support_zone_low
        res_mid = (levels.resistance_zone_low + levels.resistance_zone_high) / 2.0
        res_width = levels.resistance_zone_high - levels.resistance_zone_low
        parts: List[float] = []
        if sup_width > 0:
            d_sup = (price - sup_mid) / sup_width
            parts.append(clamp(1.0 - d_sup, -1.0, 1.0) * 0.40)
        if res_width > 0:
            d_res = (res_mid - price) / res_width
            parts.append(clamp(d_res, -1.0, 1.0) * 0.25)
        # VWAP bias: below vwap yields positive contribution
        dv = (price - levels.vwap) / levels.vwap if levels.vwap > 0 else 0.0
        parts.append(clamp(-dv, -1.0, 1.0) * 0.20)
        # Value area position: below VAL preferred, above VAH negative
        if levels.vah > levels.val > 0:
            if price < levels.val:
                parts.append(0.15)
            elif price > levels.vah:
                parts.append(-0.15)
            else:
                parts.append(0.0)
        if not parts:
            return None
        score = sum(parts)
        return clamp(score, -1.0, 1.0)


class RollingMidSeries:
    """Maintains a rolling buffer of (timestamp, mid) for volatility estimation."""

    def __init__(self, maxlen: int = 200_000) -> None:
        self.buf: Deque[Tuple[float, float]] = deque(maxlen=maxlen)

    def push(self, ts: float, mid: float) -> None:
        self.buf.append((ts, mid))

    def returns(self, start_ts: float) -> List[float]:
        """
        Compute a list of mid returns in bps since start_ts. Returns empty list if insufficient data.
        """
        rets: List[float] = []
        prev_mid: Optional[float] = None
        for ts, mid in self.buf:
            if ts < start_ts:
                continue
            if prev_mid is not None and prev_mid > 0 and mid > 0:
                rets.append((mid / prev_mid - 1.0) * 10_000.0)
            prev_mid = mid
        return rets


class LiveMinuteCandleSeries:
    """
    Build 1‑minute candles from the mid price.  Each tick is treated as volume=1.
    These candles are used for computing daily macro structure and anchored VWAP.
    """

    def __init__(self, maxlen: int = 3_000) -> None:
        self.candles: Deque['MinuteCandle'] = deque(maxlen=maxlen)
        self._cur_minute: Optional[int] = None
        self._o = self._h = self._l = self._c = None
        self._v = 0.0

    def _bucket(self, ts: float) -> int:
        return int(ts // 60) * 60

    def push_mid(self, ts: float, mid: float) -> None:
        m = self._bucket(ts)
        if self._cur_minute is None:
            self._cur_minute = m
            self._o = self._h = self._l = self._c = mid
            self._v = 1.0
            return
        if m == self._cur_minute:
            self._h = max(self._h, mid)
            self._l = min(self._l, mid)
            self._c = mid
            self._v += 1.0
            return
        # Finalise previous candle
        self.candles.append(MinuteCandle(
            minute_start_ts=self._cur_minute,
            open=float(self._o),
            high=float(self._h),
            low=float(self._l),
            close=float(self._c),
            volume=float(self._v),
        ))
        # Start new candle
        self._cur_minute = m
        self._o = self._h = self._l = self._c = mid
        self._v = 1.0

    def export_rows(self, product_id: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for c in self.candles:
            rows.append({
                "ts": c.minute_start_ts,
                "product_id": product_id,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })
        return rows


@dataclass
class MinuteCandle:
    minute_start_ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class TradeLogger:
    """
    Writes executed trades to a CSV file and maintains cumulative P&L.
    Trades are logged with event (BUY/SELL), qty, price and PnL metrics.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.cum_pnl_usd: float = 0.0
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "dt_mst", "event", "product_id", "side", "qty", "price", "notional_usd",
                "fee_usd", "gross_pnl_usd", "net_pnl_usd", "cum_pnl_usd",
                "entry_price", "exit_price", "weekly_bias", "note"
            ])

    def log_trade(
        self,
        *,
        event: str,
        product_id: str,
        side: str,
        qty: float,
        price: float,
        fee_usd_val: float,
        gross_pnl_usd: float,
        net_pnl_usd: float,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        weekly_bias: Optional[float] = None,
        note: str = "",
    ) -> None:
        notional = qty * price
        self.cum_pnl_usd += net_pnl_usd
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            tsv = now_ts()
            dt_mst = datetime.fromtimestamp(tsv, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([
                f"{tsv:.6f}", dt_mst, event, product_id, side,
                f"{qty:.10f}", f"{price:.10f}", f"{notional:.10f}",
                f"{fee_usd_val:.10f}", f"{gross_pnl_usd:.10f}", f"{net_pnl_usd:.10f}",
                f"{self.cum_pnl_usd:.10f}",
                "" if entry_price is None else f"{entry_price:.10f}",
                "" if exit_price is None else f"{exit_price:.10f}",
                "" if weekly_bias is None else f"{weekly_bias:.6f}",
                note
            ])


class MarketLogger:
    """
    Writes periodic market snapshots for viewer consumption.  Each row includes spreads,
    exposures, anchor VWAP, fair value and other metrics.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "dt_mst", "product_id", "bid", "ask", "mid", "spread_bps",
                "exposures_usd", "position_qty", "avg_entry_price",
                "anchored_vwap", "fair_value", "sigma_bps", "weekly_bias",
                "state", "cash_usd", "equity_usd"
            ])

    def log_snapshot(
        self,
        *,
        ts: float,
        product_id: str,
        bid: float,
        ask: float,
        mid: float,
        spread_bps: float,
        exposures_usd: float,
        position_qty: float,
        avg_entry_price: Optional[float],
        anchored_vwap: Optional[float],
        fair_value: Optional[float],
        sigma_bps: Optional[float],
        weekly_bias: Optional[float],
        state: str,
        cash_usd: float,
        equity_usd: float,
    ) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            dt_mst = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([
                f"{ts:.6f}", dt_mst, product_id, f"{bid:.10f}", f"{ask:.10f}", f"{mid:.10f}", f"{spread_bps:.6f}",
                f"{exposures_usd:.10f}", f"{position_qty:.10f}",
                "" if avg_entry_price is None else f"{avg_entry_price:.10f}",
                "" if anchored_vwap is None else f"{anchored_vwap:.10f}",
                "" if fair_value is None else f"{fair_value:.10f}",
                "" if sigma_bps is None else f"{sigma_bps:.6f}",
                "" if weekly_bias is None else f"{weekly_bias:.6f}",
                state,
                f"{cash_usd:.6f}", f"{equity_usd:.6f}"
            ])


class CandleCSVWriter:
    """
    Writes a list of candle dictionaries to a CSV file.  Always overwrites (atomic).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        # Ensure a header file exists on initialization so that the viewer
        # can read the file even before the first macro update.  Without this,
        # the viewer may show a "waiting" message if the macro loop has not
        # yet written any rows.
        if not os.path.exists(self.path):
            try:
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["ts", "product_id", "open", "high", "low", "close", "volume"])
            except Exception:
                # swallow any errors; file will be created on first write
                pass

    async def write(self, rows: List[Dict[str, Any]]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts", "product_id", "open", "high", "low", "close", "volume"])
            for r in rows:
                w.writerow([
                    int(r["ts"]), r["product_id"], f"{float(r['open']):.10f}",
                    f"{float(r['high']):.10f}", f"{float(r['low']):.10f}", f"{float(r['close']):.10f}",
                    f"{float(r.get('volume', 0.0)):.10f}"
                ])
        os.replace(tmp, self.path)


class MacroFetcher:
    """
    Fetches historical candles via the Coinbase REST API.  Provides chunked fetch to
    respect the <350 candle limit per request.
    """

    def __init__(self, rest: RESTClient) -> None:
        self.rest = rest

    async def fetch(self, product_id: str, start: int, end: int, granularity: str) -> List[Candle]:
        try:
            print(f"[macro] get_candles {product_id} {granularity} start={int(start)} end={int(end)} span={(int(end)-int(start))}")
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.rest.get_candles(
                    product_id=product_id,
                    start=str(int(start)),
                    end=str(int(end)),
                    granularity=granularity,
                )
            )
            candles = _parse_candles_response(resp)
            if candles:
                return candles

            # Fallback: Coinbase Exchange public candles endpoint (more reliable formatting)
            gran_map = {
                "ONE_MINUTE": 60,
                "FIVE_MINUTE": 300,
                "FIFTEEN_MINUTE": 900,
                "ONE_HOUR": 3600,
                "ONE_DAY": 86400,
            }
            g = gran_map.get(granularity)
            if g:
                rows = _fetch_candles_public(product_id=product_id, granularity=g, start=start, end=end, limit=300)
                out = []
                for r in rows:
                    # r = [time, low, high, open, close, volume]
                    out.append(Candle(ts=int(r[0]), open=float(r[3]), high=float(r[2]), low=float(r[1]), close=float(r[4]), volume=float(r[5])))
                return out

            return []
        except Exception as e:
            print(f"[macro] fetch failed for {product_id} {granularity}: {e}")
            return []

    async def fetch_chunked(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str,
        max_candles_per_req: int = 300,
    ) -> List[Candle]:
        gran_sec_map = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600,
            "ONE_DAY": 86400,
        }
        span = gran_sec_map.get(granularity)
        if not span:
            return await self.fetch(product_id, start, end, granularity)

        # --- ALIGN timestamps to candle boundaries (CRITICAL) ---
        start_i = int(start)
        end_i = int(end)

        # floor to boundary
        start_i = start_i - (start_i % span)
        end_i = end_i - (end_i % span)

        # ensure end is strictly after start by at least one bucket
        if end_i <= start_i:
            end_i = start_i + span

        chunk_span = span * max_candles_per_req
        out: List[Candle] = []
        cursor = start_i

        while cursor < end_i:
            chunk_end = min(end_i, cursor + chunk_span)
            if chunk_end <= cursor:
                break
            chunk = await self.fetch(product_id, cursor, chunk_end, granularity)
            if chunk:
                out.extend(chunk)
            cursor = chunk_end
            await asyncio.sleep(0.05)
        # deduplicate by ts
        if not out:
            return []
        uniq: Dict[int, Candle] = {}
        for c in out:
            uniq[int(c.ts)] = c
        merged = list(uniq.values())
        merged.sort(key=lambda x: x.ts)
        return merged


def _parse_candles_response(resp: Any) -> List[Candle]:
    """
    Normalise Coinbase SDK responses into a list of Candle objects.  Handles several
    possible shapes of the response.
    """
    out: List[Candle] = []

    # The Coinbase Advanced SDK may return:
    # - a dict with a "candles" key
    # - a raw list
    # - a typed response object with `.candles` and/or `.to_dict()`
    items: Optional[List[Any]] = None

    if resp is None:
        return out

    # Typed response object (coinbase-advanced-py)
    if items is None and hasattr(resp, "candles"):
        try:
            v = getattr(resp, "candles")
            if isinstance(v, list):
                items = v
        except Exception:
            pass

    # Typed response object that supports to_dict()
    if items is None and hasattr(resp, "to_dict"):
        try:
            d = resp.to_dict()  # type: ignore[attr-defined]
            if isinstance(d, dict):
                for k in ("candles", "data", "results"):
                    v = d.get(k)
                    if isinstance(v, list):
                        items = v
                        break
        except Exception:
            pass

    # Dict response
    if items is None and isinstance(resp, dict):
        for k in ("candles", "data", "results"):
            v = resp.get(k)
            if isinstance(v, list):
                items = v
                break

    # Raw list response
    if items is None and isinstance(resp, list):
        items = resp

    if not items:
        return out

    def _get_attr(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    for it in items:
        # Candle dicts
        if isinstance(it, dict):
            ts = int(float(it.get("start") or it.get("time") or it.get("ts") or 0))
            o = float(it.get("open") or 0)
            h = float(it.get("high") or 0)
            l = float(it.get("low") or 0)
            c = float(it.get("close") or 0)
            v = float(it.get("volume") or 0)
            if ts > 0 and c > 0:
                out.append(Candle(ts=ts, open=o, high=h, low=l, close=c, volume=v))
            continue

        # Tuple/list shape: [start, low, high, open, close, volume]
        if isinstance(it, (list, tuple)) and len(it) >= 6:
            ts = int(float(it[0]))
            l = float(it[1]); h = float(it[2]); o = float(it[3]); c = float(it[4]); v = float(it[5])
            if ts > 0 and c > 0:
                out.append(Candle(ts=ts, open=o, high=h, low=l, close=c, volume=v))
            continue

        # Candle objects from the SDK
        try:
            ts_raw = _get_attr(it, "start") or _get_attr(it, "time") or _get_attr(it, "ts")
            ts = int(float(ts_raw or 0))
            o = float(_get_attr(it, "open") or 0)
            h = float(_get_attr(it, "high") or 0)
            l = float(_get_attr(it, "low") or 0)
            c = float(_get_attr(it, "close") or 0)
            v = float(_get_attr(it, "volume") or 0)
            if ts > 0 and c > 0:
                out.append(Candle(ts=ts, open=o, high=h, low=l, close=c, volume=v))
        except Exception:
            continue

    out.sort(key=lambda x: x.ts)
    return out


def compute_macro_levels(candles: List[Candle]) -> Optional[MacroLevels]:
    """
    Compute structural levels and zones from a list of candles.  Requires at least 50
    candles to provide a robust estimate.  Expands support/resistance into zones and
    computes additional metrics (range, VWAP, value area, psychological levels).
    """
    if not candles or len(candles) < 50:
        return None
    o = np.array([c.open for c in candles], dtype=float)
    h = np.array([c.high for c in candles], dtype=float)
    l = np.array([c.low for c in candles], dtype=float)
    cprices = np.array([c.close for c in candles], dtype=float)
    v = np.array([c.volume for c in candles], dtype=float)

    price_now = float(cprices[-1])
    if price_now <= 0:
        return None
    # Range high/low
    range_low = float(np.min(l))
    range_high = float(np.max(h))
    # Previous high/low from earlier half of window
    half = max(10, len(candles) // 2)
    prev = candles[:-half]
    prev_high = float(max((c.high for c in prev), default=h[0]))
    prev_low = float(min((c.low for c in prev), default=l[0]))
    # VWAP
    typical_price = (h + l + cprices) / 3.0
    vol = v.copy()
    vsum = float(np.sum(vol))
    if vsum <= 1e-9:
        vol = np.ones_like(cprices)
        vsum = float(np.sum(vol))
    vwap = float(np.sum(typical_price * vol) / vsum)
    # Psychological levels
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
    step = psych_step(price_now)
    psych_low = float(math.floor(price_now / step) * step)
    psych_high = float(math.ceil(price_now / step) * step)
    # Value area (volume‑by‑price) approximation on closes
    bins = 60
    pmin, pmax = float(np.min(l)), float(np.max(h))
    if pmax <= pmin:
        return None
    edges = np.linspace(pmin, pmax, bins + 1)
    hist = np.zeros(bins, dtype=float)
    idx = np.clip(np.digitize(cprices, edges) - 1, 0, bins - 1)
    for i, vv in zip(idx, vol):
        hist[i] += float(vv)
    total = float(np.sum(hist))
    if total <= 1e-9:
        hist += 1.0
        total = float(np.sum(hist))
    poc_i = int(np.argmax(hist))
    poc_price = float((edges[poc_i] + edges[poc_i + 1]) / 2.0)
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
    # Support/resistance via clustering of extrema
    w = 3
    lows_cand: List[float] = []
    highs_cand: List[float] = []
    lows_series = l
    highs_series = h
    for i in range(w, len(candles) - w):
        lo = float(lows_series[i])
        hi = float(highs_series[i])
        if all(lo <= float(lows_series[j]) for j in range(i - w, i + w + 1)):
            lows_cand.append(lo)
        if all(hi >= float(highs_series[j]) for j in range(i - w, i + w + 1)):
            highs_cand.append(hi)
    def cluster_levels(levels: List[float], tol_pct: float = 0.35) -> List[Tuple[float, int]]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: List[List[float]] = []
        cur: List[float] = [levels[0]]
        for x in levels[1:]:
            ref = float(np.mean(cur))
            tol = ref * (tol_pct / 100.0)
            if abs(x - ref) <= tol:
                cur.append(x)
            else:
                clusters.append(list(cur))
                cur = [x]
        clusters.append(list(cur))
        clusters.sort(key=lambda c: len(c), reverse=True)
        out: List[Tuple[float, int]] = []
        for cset in clusters:
            out.append((float(np.mean(cset)), len(cset)))
        return out
    low_clusters = cluster_levels(lows_cand)
    high_clusters = cluster_levels(highs_cand)
    support = float(low_clusters[0][0]) if low_clusters else float(np.percentile(l, 15))
    resistance = float(high_clusters[0][0]) if high_clusters else float(np.percentile(h, 85))
    # Breakout level: highest resistance above price
    breakout = resistance
    # Convert support/resistance into zones (bands).  Use a small fraction of price to set zone width.
    sup_width = max(price_now * 0.003, price_now * 0.0015)  # ~0.3% band
    res_width = max(price_now * 0.003, price_now * 0.0015)
    support_zone_low = support - sup_width
    support_zone_high = support + sup_width
    resistance_zone_low = resistance - res_width
    resistance_zone_high = resistance + res_width
    return MacroLevels(
        support_zone_low=support_zone_low,
        support_zone_high=support_zone_high,
        resistance_zone_low=resistance_zone_low,
        resistance_zone_high=resistance_zone_high,
        breakout=float(breakout),
        range_low=float(range_low),
        range_high=float(range_high),
        prev_low=float(prev_low),
        prev_high=float(prev_high),
        vwap=float(vwap),
        psych_low=float(psych_low),
        psych_high=float(psych_high),
        val=float(val),
        vah=float(vah),
        price_now=float(price_now),
    )


def compute_sigma_bps(series: RollingMidSeries, window_sec: int = 60 * 60) -> Optional[float]:
    """
    Compute volatility (sigma) in basis points from the mid series over the given window.
    Returns None if insufficient data.  Uses standard deviation of returns.
    """
    if not series.buf:
        return None
    now_ts_val = series.buf[-1][0]
    start_ts = now_ts_val - window_sec
    rets = series.returns(start_ts)
    if len(rets) < 10:
        return None
    sigma = float(np.std(rets))  # already in bps because series.returns uses bps
    return sigma


# ------------------------------------------------------------
# Option-1 entry helpers (support -> reversal -> room to +2%)
# ------------------------------------------------------------

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    s = pd.Series(arr)
    return s.ewm(span=int(span), adjust=False).mean().to_numpy(dtype=float)

def _rsi(closes: np.ndarray, period: int = 14) -> Optional[float]:
    if closes.size < period + 2:
        return None
    diff = np.diff(closes.astype(float))
    up = np.maximum(diff, 0.0)
    down = np.maximum(-diff, 0.0)
    roll_up = pd.Series(up).ewm(alpha=1.0/period, adjust=False).mean()
    roll_down = pd.Series(down).ewm(alpha=1.0/period, adjust=False).mean()
    rs = roll_up.iloc[-1] / (roll_down.iloc[-1] + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)

def _pivot_high(highs: np.ndarray, w: int) -> Optional[int]:
    n = highs.size
    if n < (2*w + 3):
        return None
    i = n - w - 2
    lo = max(0, i - w)
    hi = min(n, i + w + 1)
    v = highs[i]
    if np.all(v >= highs[lo:hi]):
        return int(i)
    return None

def _pivot_low(lows: np.ndarray, w: int) -> Optional[int]:
    n = lows.size
    if n < (2*w + 3):
        return None
    i = n - w - 2
    lo = max(0, i - w)
    hi = min(n, i + w + 1)
    v = lows[i]
    if np.all(v <= lows[lo:hi]):
        return int(i)
    return None

def option1_in_support_zone(mid: float, day: Optional['MacroLevels'], week: Optional['MacroLevels'], support_buffer_bps: float) -> Tuple[bool, str]:
    if mid <= 0:
        return False, "bad_mid"
    buf = mid * (support_buffer_bps / 10_000.0)
    checks = []
    if day:
        checks += [
            ("day_support_zone", day.support_zone_low - buf, day.support_zone_high + buf),
            ("day_VAL", day.val - buf, day.val + buf),
            ("day_prev_low", day.prev_low - buf, day.prev_low + buf),
        ]
    if week:
        checks += [
            ("week_support_zone", week.support_zone_low - buf, week.support_zone_high + buf),
            ("week_VAL", week.val - buf, week.val + buf),
            ("week_prev_low", week.prev_low - buf, week.prev_low + buf),
        ]
    for name, lo, hi in checks:
        if lo <= mid <= hi:
            return True, name
    return False, "not_in_support_zone"

def option1_reversal_confirmation(minute_candles: List['MinuteCandle']) -> Tuple[bool, str]:
    # Uses last ~120 minutes if available.
    if not minute_candles or len(minute_candles) < 30:
        return False, "insufficient_1m"
    recent = minute_candles[-120:] if len(minute_candles) > 120 else minute_candles[:]
    closes = np.array([c.close for c in recent], dtype=float)
    highs = np.array([c.high for c in recent], dtype=float)
    lows = np.array([c.low for c in recent], dtype=float)

    passed: List[str] = []

    # (1) Momentum flip
    rsi = _rsi(closes, 14)
    if rsi is not None and rsi >= RSI_BUY_THRESHOLD:
        passed.append("rsi")

    # (2) Structure reclaim
    ema9 = _ema(closes, EMA_ENTRY_FAST)
    ema20 = _ema(closes, EMA_ENTRY_SLOW)
    if ema20.size >= 12:
        # EMA20 slope over ~10 mins in bps
        slope = (ema20[-1] / ema20[-11] - 1.0) * 10_000.0
    else:
        slope = 0.0
    if closes[-1] > ema20[-1] and (ema9[-1] >= ema20[-1]) and slope >= (-EMA_SLOPE_MAX_DOWN_BPS):
        passed.append("ema_reclaim")

    # (3) Micro structure: higher-low then break above pivot high
    ph_i = _pivot_high(highs, PIVOT_W)
    pl_i = _pivot_low(lows, PIVOT_W)
    if ph_i is not None and pl_i is not None and pl_i < ph_i:
        # higher-low: last low pivot above previous low in window
        # simple: compare last 2 pivot lows if we can find them
        pivot_lows: List[Tuple[int, float]] = []
        for i in range(len(lows) - (2*PIVOT_W + 2)):
            if i < PIVOT_W or i > len(lows) - PIVOT_W - 2:
                continue
            v = lows[i]
            if np.all(v <= lows[i-PIVOT_W:i+PIVOT_W+1]):
                pivot_lows.append((i, float(v)))
        if len(pivot_lows) >= 2:
            (i1, v1), (i2, v2) = pivot_lows[-2], pivot_lows[-1]
            if v2 > v1:
                # break above the most recent pivot high price
                pivot_highs: List[Tuple[int, float]] = []
                for i in range(len(highs) - (2*PIVOT_W + 2)):
                    if i < PIVOT_W or i > len(highs) - PIVOT_W - 2:
                        continue
                    v = highs[i]
                    if np.all(v >= highs[i-PIVOT_W:i+PIVOT_W+1]):
                        pivot_highs.append((i, float(v)))
                if pivot_highs:
                    _, last_ph = pivot_highs[-1]
                    if closes[-1] >= last_ph:
                        passed.append("hl_break")

    ok = (len(passed) >= REQUIRE_CONFIRMATIONS)
    if ok:
        return True, "pass=" + "+".join(passed)
    return False, "fail=" + "+".join(passed) if passed else "fail=none"

def option1_room_to_target(mid: float, day: Optional['MacroLevels'], week: Optional['MacroLevels'], resist_buffer_bps: float) -> Tuple[bool, str]:
    if mid <= 0:
        return False, "bad_mid"
    target = mid * 1.02
    buf = mid * (resist_buffer_bps / 10_000.0)

    resist_levels: List[Tuple[str, float]] = []
    if day:
        resist_levels += [
            ("day_res_zone_low", day.resistance_zone_low),
            ("day_vwap", day.vwap),
            ("day_VAH", day.vah),
        ]
    if week:
        resist_levels += [
            ("week_res_zone_low", week.resistance_zone_low),
            ("week_vwap", week.vwap),
            ("week_VAH", week.vah),
        ]

    # Find nearest resistance above mid
    above = [(name, lvl) for name, lvl in resist_levels if lvl and lvl > mid]
    if not above:
        return True, "no_resistance_above"
    name, lvl = min(above, key=lambda x: x[1])

    # If resistance is below target (with buffer), reject: not enough room to +2%
    if (lvl + buf) < target:
        return False, f"capped_by={name}"
    return True, f"room_ok_next={name}"


def _room_to_target_pct(mid: float, day: Optional['MacroLevels'], week: Optional['MacroLevels'], target_pct: float, resist_buffer_bps: float) -> Tuple[bool, str]:
    if mid <= 0:
        return False, "bad_mid"
    target = mid * (1.0 + float(target_pct))
    buf = mid * (resist_buffer_bps / 10_000.0)

    resist_levels: List[Tuple[str, float]] = []
    if day:
        resist_levels += [
            ("day_res_zone_low", day.resistance_zone_low),
            ("day_vwap", day.vwap),
            ("day_VAH", day.vah),
        ]
    if week:
        resist_levels += [
            ("week_res_zone_low", week.resistance_zone_low),
            ("week_vwap", week.vwap),
            ("week_VAH", week.vah),
        ]

    above = [(name, lvl) for name, lvl in resist_levels if lvl and lvl > mid]
    if not above:
        return True, "no_resistance_above"
    name, lvl = min(above, key=lambda x: x[1])

    if (lvl + buf) < target:
        return False, f"capped_by={name}"
    return True, f"room_ok_next={name}"


def tiered_entry_gate(
    *,
    mid: float,
    spread_bps: float,
    levels_day: Optional['MacroLevels'],
    levels_week: Optional['MacroLevels'],
    minute_candles: List['MinuteCandle'],
    weekly_bias: Optional[float],
    trending_down: bool,
    support_buffer_bps: float,
    resist_buffer_bps: float,
) -> Tuple[bool, int, str]:
    """
    Returns: (ok, tier, reason)
      - tier: 1=low, 2=mid, 3=high
    """

    # Hard risk-off block stays: don’t scalp aggressively in extreme risk-off
    risk_off = trending_down or (weekly_bias is not None and weekly_bias < WEEKLY_BIAS_THRESHOLD)

    # A) Support proximity (re-use option1 support)
    sup_ok, sup_reason = option1_in_support_zone(mid, levels_day, levels_week, support_buffer_bps)

    # B) Reversal confirmation (option1 is 2-of-3)
    rev_ok, rev_reason = option1_reversal_confirmation(minute_candles)

    # HIGH: strict Option-1 (your original intent)
    if (not risk_off) and sup_ok and rev_ok:
        room_ok, room_reason = option1_room_to_target(mid, levels_day, levels_week, resist_buffer_bps)
        if room_ok:
            return True, TIER_HIGH, f"HIGH:A={sup_reason};B={rev_reason};C={room_reason}"

    # MID: allow if support + reversal, but only require room to ~0.75%
    if sup_ok and rev_ok and (not risk_off):
        room_ok, room_reason = _room_to_target_pct(mid, levels_day, levels_week, target_pct=TIER_TP_PCT[TIER_MID], resist_buffer_bps=resist_buffer_bps)
        if room_ok:
            return True, TIER_MID, f"MID:A={sup_reason};B={rev_reason};C={room_reason}"

    # LOW (scalp): require tight spread + either support proximity OR mild reversal,
    # and do NOT require room to 2% (only room to 0.25%)
    if spread_bps <= SCALP_MAX_SPREAD_BPS and (not risk_off):
        # Looser reversal: accept if RSI or EMA reclaim passed (not necessarily 2-of-3)
        # We parse the option1 string: "pass=rsi+ema_reclaim" etc.
        passed_any = ("pass=" in rev_reason) and any(k in rev_reason for k in ("rsi", "ema_reclaim"))
        if sup_ok or passed_any:
            room_ok, room_reason = _room_to_target_pct(mid, levels_day, levels_week, target_pct=TIER_TP_PCT[TIER_LOW], resist_buffer_bps=resist_buffer_bps)
            if room_ok:
                return True, TIER_LOW, f"LOW:A={sup_reason};B={rev_reason};C={room_reason}"

    return False, 0, f"NO: A={sup_reason}; B={rev_reason}"

# ------------------------------------------------------------
# Portfolio management
# ------------------------------------------------------------

class PaperPortfolio:
    """Simple portfolio holding only USD. Tracks available cash."""
    def __init__(self, starting_cash: float) -> None:
        self.cash_usd: float = float(starting_cash)

    def can_afford(self, notional_usd: float, fee_bps: float) -> bool:
        total_cost = notional_usd + fee_usd(notional_usd, fee_bps)
        return self.cash_usd >= total_cost

    def debit(self, notional_usd: float, fee_bps: float) -> float:
        fee_val = fee_usd(notional_usd, fee_bps)
        self.cash_usd -= (notional_usd + fee_val)
        return fee_val

    def credit(self, notional_usd: float, fee_bps: float) -> float:
        fee_val = fee_usd(notional_usd, fee_bps)
        self.cash_usd += (notional_usd - fee_val)
        return fee_val



@dataclass
class ExecutionResult:
    ok: bool
    order_id: Optional[str]
    client_order_id: str
    product_id: str
    side: str  # "BUY" | "SELL"
    filled_qty: float
    avg_price: Optional[float]
    fee_usd: float
    filled_notional_usd: Optional[float]
    status: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "product_id": self.product_id,
            "side": self.side,
            "filled_qty": float(self.filled_qty),
            "avg_price": (None if self.avg_price is None else float(self.avg_price)),
            "fee_usd": float(self.fee_usd),
            "filled_notional_usd": (None if self.filled_notional_usd is None else float(self.filled_notional_usd)),
            "status": str(self.status),
            "error": self.error,
        }


class LivePortfolio:
    """Live-portfolio wrapper for Coinbase Advanced Trade.

    Coinbase must be the source of truth for balances, fills, and fees.

    Key design points:
      - Account balances are refreshed from Coinbase and used for cash/equity.
      - Orders are NOT treated as successful unless we have an order_id and
        we can confirm a non-zero filled quantity from a terminal order state
        and/or fills.
      - Fills are preferred for accounting. If the installed Coinbase SDK does
        not expose a fills endpoint, we fall back to order-level fields.
      - All order placement returns a canonical ExecutionResult payload.
    """

    def __init__(self, rest: RESTClient) -> None:
        self.rest = rest

        # Cached snapshot from get_accounts() to avoid hammering the API.
        self._snapshot_ts: float = 0.0
        self._snapshot: Dict[str, Dict[str, float]] = {}

        # Public value used by the bot and written into snapshots.
        self.cash_usd: float = 0.0

        # Prime the cache.
        self.refresh_snapshot(force=True)

    # ---------------------------
    # Snapshot + parsing helpers
    # ---------------------------

    def _to_dict(self, resp: Any) -> dict:
        """Best-effort conversion of Coinbase SDK response to a plain dict."""
        if resp is None:
            return {}
        if isinstance(resp, dict):
            return resp
        if hasattr(resp, "to_dict"):
            try:
                d = resp.to_dict()  # type: ignore[attr-defined]
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        # Sometimes the SDK returns dataclasses with __dict__
        try:
            d = dict(resp.__dict__)  # type: ignore[attr-defined]
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _as_list(self, x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            for k in ("accounts", "data", "items", "results"):
                v = x.get(k)
                if isinstance(v, list):
                    return v
        for attr in ("accounts", "data", "items", "results"):
            v = getattr(x, attr, None)
            if isinstance(v, list):
                return v
        return []

    def _get(self, obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _get_value(self, x: Any) -> Any:
        """Extract numeric-like 'value' from dict/object/string."""
        if x is None:
            return None
        if isinstance(x, dict):
            return x.get("value")
        v = getattr(x, "value", None)
        if v is not None:
            return v
        return x

    def refresh_snapshot(self, force: bool = False, ttl_sec: float = 1.25) -> Dict[str, Dict[str, float]]:
        """Refresh and cache balances from Coinbase accounts.

        Returns a mapping:
            { "USD": {"available": ..., "hold": ..., "total": ...}, "BTC": {...}, ... }
        """
        nowv = time.time()
        if (not force) and self._snapshot and (nowv - self._snapshot_ts) <= float(ttl_sec):
            return self._snapshot

        resp = self.rest.get_accounts()
        data = getattr(resp, "accounts", None) or getattr(resp, "data", None) or resp
        if hasattr(data, "to_dict"):
            try:
                data = data.to_dict()  # type: ignore[attr-defined]
            except Exception:
                pass

        snap: Dict[str, Dict[str, float]] = {}

        for acct in self._as_list(data):
            cur = (
                self._get(acct, "currency")
                or self._get(acct, "asset")
                or self._get(acct, "symbol")
                or self._get(acct, "currency_code")
                or self._get(acct, "currencyCode")
            )
            if not isinstance(cur, str) or not cur:
                continue

            ab = self._get_value(self._get(acct, "available_balance"))
            hb = self._get_value(self._get(acct, "hold"))
            bal = self._get_value(self._get(acct, "balance"))

            # Some SDKs use alternate keys
            if ab is None:
                ab = self._get_value(self._get(acct, "available"))
            if hb is None:
                hb = self._get_value(self._get(acct, "holds"))
            if bal is None:
                bal = self._get_value(self._get(acct, "balance_amount")) or self._get_value(self._get(acct, "total_balance"))

            try:
                available = float(ab or 0.0)
            except Exception:
                available = 0.0
            try:
                hold = float(hb or 0.0)
            except Exception:
                hold = 0.0
            try:
                total = float(bal) if bal is not None else float(available + hold)
            except Exception:
                total = float(available + hold)

            snap[cur] = {"available": float(available), "hold": float(hold), "total": float(total)}

        self._snapshot = snap
        self._snapshot_ts = nowv

        # Update public cash view
        self.cash_usd = self.get_tradable_usd(snapshot=snap)
        return snap

    def get_tradable_usd(self, snapshot: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """Return tradable USD (available + optional hold)."""
        snap = snapshot if snapshot is not None else self.refresh_snapshot()
        usd = snap.get("USD", {})
        avail = float(usd.get("available", 0.0))
        hold = float(usd.get("hold", 0.0))
        tradable = avail + (hold if USE_USD_HOLD_AS_TRADABLE else 0.0)
        return float(max(0.0, tradable))

    def get_total_asset(self, asset: str, snapshot: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """Return total balance for an asset (best-effort)."""
        snap = snapshot if snapshot is not None else self.refresh_snapshot()
        d = snap.get(asset, {})
        return float(max(0.0, float(d.get("total", 0.0))))

    def compute_equity_usd(
        self,
        *,
        mid_by_product: Dict[str, float],
        snapshot: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        """Compute account equity in USD using snapshot balances + current mids."""
        snap = snapshot if snapshot is not None else self.refresh_snapshot()
        equity = 0.0

        usd = snap.get("USD", {})
        equity += float(usd.get("total", 0.0))

        stable_1 = {"USDC", "USDT", "DAI", "TUSD", "FDUSD"}

        for asset, vals in snap.items():
            if asset == "USD":
                continue
            qty = float(vals.get("total", 0.0))
            if qty <= 0:
                continue
            if asset in stable_1:
                equity += qty
                continue

            pid = f"{asset}-USD"
            mid = mid_by_product.get(pid)
            if mid is None or mid <= 0:
                continue
            equity += qty * float(mid)

        return float(max(0.0, equity))

    def sync_after_trade(self, attempts: int = 8, sleep_sec: float = 0.5) -> None:
        """Force-refresh balances a few times to let Coinbase settle post-trade."""
        for _ in range(max(1, int(attempts))):
            self.refresh_snapshot(force=True, ttl_sec=0.0)
            time.sleep(float(sleep_sec))

    # ---------------------------
    # PaperPortfolio-compatible API
    # ---------------------------

    def refresh_cash(self) -> float:
        self.refresh_snapshot(force=True, ttl_sec=0.0)
        return self.cash_usd

    def can_afford(self, notional_usd: float, fee_bps: float) -> bool:
        cash = self.refresh_cash()
        est_total = float(notional_usd) * (1.0 + (float(fee_bps) / 10_000.0))
        return cash >= est_total

    def debit(self, notional_usd: float, fee_bps: float) -> float:
        n = float(max(0.0, notional_usd))
        fee = float(fee_usd(n, float(fee_bps)))
        self.refresh_snapshot(force=True, ttl_sec=0.0)
        return fee

    def credit(self, notional_usd: float, fee_bps: float) -> float:
        n = float(max(0.0, notional_usd))
        fee = float(fee_usd(n, float(fee_bps)))
        self.refresh_snapshot(force=True, ttl_sec=0.0)
        return fee

    # ---------------------------
    # Order placement + reconciliation
    # ---------------------------

    def _extract_order_id(self, d: dict) -> Optional[str]:
        for key in ("order_id", "orderId", "id"):
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
        for nest in ("success_response", "successResponse", "order", "result", "data"):
            v = d.get(nest)
            if isinstance(v, dict):
                oid = v.get("order_id") or v.get("orderId") or v.get("id")
                if isinstance(oid, str) and oid:
                    return oid
        return None

    def _extract_status(self, d: dict) -> str:
        # Common keys in Advanced Trade order payloads
        for key in ("status", "order_status", "orderStatus"):
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
        if isinstance(d.get("order"), dict):
            v = d["order"].get("status") or d["order"].get("order_status") or d["order"].get("orderStatus")
            if isinstance(v, str) and v:
                return v
        if isinstance(d.get("data"), dict):
            v = d["data"].get("status") or d["data"].get("order_status") or d["data"].get("orderStatus")
            if isinstance(v, str) and v:
                return v
        return ""

    def _extract_error(self, d: dict) -> Optional[str]:
        for key in ("message", "error", "failure_reason", "failureReason", "error_details", "errorDetails"):
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, dict):
                msg = v.get("message") or v.get("error") or v.get("reason")
                if isinstance(msg, str) and msg:
                    return msg
        # sometimes nested
        for nest in ("error_response", "errorResponse", "failure_response", "failureResponse"):
            v = d.get(nest)
            if isinstance(v, dict):
                msg = v.get("message") or v.get("error") or v.get("reason")
                if isinstance(msg, str) and msg:
                    return msg
        return None

    def _extract_success(self, d: dict) -> Optional[bool]:
        v = d.get("success")
        if isinstance(v, bool):
            return v
        # Many SDKs wrap success responses
        if isinstance(d.get("success_response"), dict) or isinstance(d.get("successResponse"), dict):
            return True
        return None

    def _wait_for_order(self, order_id: str, timeout_sec: float = 20.0, poll_sec: float = 0.6) -> dict:
        """Poll order status until terminal or timeout. Returns an order dict (may be partial)."""
        t0 = time.time()
        last: dict = {}
        while (time.time() - t0) < float(timeout_sec):
            try:
                if hasattr(self.rest, "get_order"):
                    try:
                        resp = self.rest.get_order(order_id=order_id)  # type: ignore[arg-type]
                    except TypeError:
                        resp = self.rest.get_order(order_id)  # type: ignore[misc]
                    last = self._to_dict(resp)
            except Exception:
                pass

            st = self._extract_status(last).upper()
            if st in ("FILLED", "DONE", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "FAILED"):
                break

            # Some responses include completion percentage
            pct = last.get("completion_percentage") or last.get("completionPercentage")
            try:
                if pct is not None and float(pct) >= 100.0:
                    break
            except Exception:
                pass

            time.sleep(float(poll_sec))

        return last

    def _parse_order_fill_fields(self, order_d: dict) -> Tuple[float, Optional[float], float, Optional[float]]:
        """Fallback: extract (filled_qty, avg_price, fee_usd, filled_notional_usd) from an order dict."""
        d = order_d
        if isinstance(order_d.get("order"), dict):
            d = order_d.get("order")  # type: ignore[assignment]
        if isinstance(order_d.get("data"), dict):
            d = order_d.get("data")  # type: ignore[assignment]

        def fget(*keys: str) -> Any:
            for k in keys:
                if k in d:
                    return d.get(k)
            return None

        filled_qty = safe_float(fget("filled_size", "filledSize", "filled_qty", "filledQty", "filled_quantity", "filledQuantity")) or 0.0
        avg_price = safe_float(fget("average_filled_price", "averageFilledPrice", "avg_price", "avgPrice", "average_price", "averagePrice"))

        # Filled value / notional in quote (USD for -USD pairs)
        filled_value = safe_float(fget("filled_value", "filledValue", "executed_value", "executedValue", "filled_notional", "filledNotional"))

        fee = fget("total_fees", "totalFees", "fee", "fees", "total_fee", "totalFee")
        fee_usd_val = 0.0
        if isinstance(fee, dict):
            fee_usd_val = safe_float(fee.get("value")) or 0.0
        else:
            fee_usd_val = safe_float(fee) or 0.0

        return float(filled_qty), (float(avg_price) if avg_price is not None else None), float(fee_usd_val), (float(filled_value) if filled_value is not None else None)

    def _fetch_fills_for_order(self, order_id: str, product_id: str) -> List[dict]:
        """Best-effort fills fetch. Uses SDK if available, else returns [].

        Different versions of Coinbase Advanced Trade SDK expose fills with different names.
        We probe a small set of likely methods at runtime.
        """
        candidates = [
            "get_fills",
            "list_fills",
            "get_fills_for_order",
            "list_fills_for_order",
            "get_order_fills",
            "list_order_fills",
            "get_fills_by_order_id",
        ]
        for name in candidates:
            fn = getattr(self.rest, name, None)
            if not callable(fn):
                continue
            # Try common call signatures
            for kwargs in (
                {"order_id": order_id},
                {"orderId": order_id},
                {"order_id": order_id, "product_id": product_id},
                {"orderId": order_id, "product_id": product_id},
                {"product_id": product_id, "order_id": order_id},
            ):
                try:
                    resp = fn(**kwargs)  # type: ignore[misc]
                    d = self._to_dict(resp)
                    items = None
                    if isinstance(d.get("fills"), list):
                        items = d.get("fills")
                    elif isinstance(d.get("data"), list):
                        items = d.get("data")
                    elif isinstance(d.get("results"), list):
                        items = d.get("results")
                    elif isinstance(d.get("items"), list):
                        items = d.get("items")
                    elif isinstance(resp, list):
                        items = resp
                    if isinstance(items, list):
                        return [x for x in items if isinstance(x, dict)]
                except Exception:
                    continue
            # Try positional
            try:
                resp = fn(order_id)  # type: ignore[misc]
                d = self._to_dict(resp)
                items = None
                if isinstance(d.get("fills"), list):
                    items = d.get("fills")
                elif isinstance(d.get("data"), list):
                    items = d.get("data")
                elif isinstance(d.get("results"), list):
                    items = d.get("results")
                elif isinstance(d.get("items"), list):
                    items = d.get("items")
                elif isinstance(resp, list):
                    items = resp
                if isinstance(items, list):
                    return [x for x in items if isinstance(x, dict)]
            except Exception:
                pass
        return []

    def _aggregate_fills(self, fills: List[dict]) -> Tuple[float, Optional[float], float, Optional[float]]:
        """Aggregate fills into (qty, avg_price, fee_usd, notional_usd)."""
        total_qty = 0.0
        notional = 0.0
        fee_total = 0.0

        for f in fills:
            # Common keys: size, price, commission, fee
            sz = safe_float(f.get("size") or f.get("filled_size") or f.get("filledSize") or f.get("qty") or f.get("quantity"))
            px = safe_float(f.get("price") or f.get("fill_price") or f.get("fillPrice") or f.get("trade_price") or f.get("tradePrice"))
            fee = f.get("commission") or f.get("fee") or f.get("fees") or f.get("total_fee") or f.get("totalFee")
            if isinstance(fee, dict):
                fee_val = safe_float(fee.get("value"))
            else:
                fee_val = safe_float(fee)
            if sz is None or px is None:
                continue
            if sz <= 0 or px <= 0:
                continue
            total_qty += float(sz)
            notional += float(sz) * float(px)
            if fee_val is not None and fee_val > 0:
                fee_total += float(fee_val)

        if total_qty <= 0:
            return 0.0, None, 0.0, None
        avg_price = notional / total_qty
        return float(total_qty), float(avg_price), float(fee_total), float(notional)

    def _market_order(self, *, side: str, product_id: str, quote_usd: Optional[float] = None, base_qty: Optional[float] = None) -> dict:
        """Place a market order, confirm terminal state, fetch fills, and return canonical result."""
        side_u = str(side).upper().strip()
        if side_u not in ("BUY", "SELL"):
            return ExecutionResult(
                ok=False, order_id=None, client_order_id="", product_id=product_id, side=side_u,
                filled_qty=0.0, avg_price=None, fee_usd=0.0, filled_notional_usd=None,
                status="INVALID", error="invalid_side"
            ).to_dict()

        client_order_id = str(uuid.uuid4())

        # Place order
        try:
            if side_u == "BUY":
                if quote_usd is None or float(quote_usd) <= 0:
                    return ExecutionResult(
                        ok=False, order_id=None, client_order_id=client_order_id, product_id=product_id, side=side_u,
                        filled_qty=0.0, avg_price=None, fee_usd=0.0, filled_notional_usd=None,
                        status="INVALID", error="quote_usd<=0"
                    ).to_dict()
                resp = self.rest.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size=str(round(float(quote_usd), 2)),
                )
            else:
                if base_qty is None or float(base_qty) <= 0:
                    return ExecutionResult(
                        ok=False, order_id=None, client_order_id=client_order_id, product_id=product_id, side=side_u,
                        filled_qty=0.0, avg_price=None, fee_usd=0.0, filled_notional_usd=None,
                        status="INVALID", error="base_qty<=0"
                    ).to_dict()
                resp = self.rest.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=format(float(base_qty), ".10f").rstrip("0").rstrip("."),
                )
        except Exception as e:
            return ExecutionResult(
                ok=False, order_id=None, client_order_id=client_order_id, product_id=product_id, side=side_u,
                filled_qty=0.0, avg_price=None, fee_usd=0.0, filled_notional_usd=None,
                status="ERROR", error=str(e)
            ).to_dict()

        d0 = self._to_dict(resp)
        ok0 = self._extract_success(d0)
        order_id = self._extract_order_id(d0)
        err0 = self._extract_error(d0)
        if ok0 is False:
            return ExecutionResult(
                ok=False, order_id=order_id, client_order_id=client_order_id, product_id=product_id, side=side_u,
                filled_qty=0.0, avg_price=None, fee_usd=0.0, filled_notional_usd=None,
                status=self._extract_status(d0) or "REJECTED",
                error=err0 or "order_rejected"
            ).to_dict()

        # Guardrail: no order_id -> cannot reconcile -> treat as failure
        if not order_id:
            return ExecutionResult(
                ok=False, order_id=None, client_order_id=client_order_id, product_id=product_id, side=side_u,
                filled_qty=0.0, avg_price=None, fee_usd=0.0, filled_notional_usd=None,
                status=self._extract_status(d0) or "UNKNOWN",
                error=err0 or "missing_order_id"
            ).to_dict()

        # Wait for terminal or timeout
        order_d = self._wait_for_order(order_id=order_id, timeout_sec=20.0, poll_sec=0.6)
        status = (self._extract_status(order_d) or self._extract_status(d0) or "").upper()

        # Fills-first accounting (best-effort)
        fills = self._fetch_fills_for_order(order_id=order_id, product_id=product_id)
        qty_f, avg_px_f, fee_f, notional_f = self._aggregate_fills(fills)

        if qty_f <= 0:
            qty_o, avg_px_o, fee_o, notional_o = self._parse_order_fill_fields(order_d)
            qty_f, avg_px_f, fee_f, notional_f = qty_o, avg_px_o, fee_o, notional_o

        # Determine outcome
        if qty_f <= 1e-12:
            # Terminal-but-unfilled should not be logged as success.
            terminal = status in ("FILLED", "DONE", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "FAILED")
            err = self._extract_error(order_d) or err0 or ("terminal_zero_fill" if terminal else "ambiguous_fill")
            ok_final = False
        else:
            ok_final = True
            err = None
            # if status missing but we have fills, treat as filled
            if not status:
                status = "FILLED"

        # Sync balances
        try:
            self.sync_after_trade(attempts=6, sleep_sec=0.5)
        except Exception:
            pass

        return ExecutionResult(
            ok=ok_final,
            order_id=order_id,
            client_order_id=client_order_id,
            product_id=product_id,
            side=side_u,
            filled_qty=float(qty_f),
            avg_price=(None if avg_px_f is None else float(avg_px_f)),
            fee_usd=float(fee_f),
            filled_notional_usd=(None if notional_f is None else float(notional_f)),
            status=status or "UNKNOWN",
            error=err,
        ).to_dict()

    def buy_market(self, product_id: str, quote_usd: float) -> dict:
        return self._market_order(side="BUY", product_id=product_id, quote_usd=float(quote_usd))

    def sell_market(self, product_id: str, base_qty: float) -> dict:
        return self._market_order(side="SELL", product_id=product_id, base_qty=float(base_qty))


class TradingBot:
    """
    A structurally mean‑reverting trading bot.  Uses a three‑layer model:
    - Weekly macro to compute bias and support/resistance zones.
    - Daily macro to refine support and compute value area for fair value.
    - Micro (per‑second) to place laddered entries and strength‑based exits.
    """
    def __init__(self, rest: RESTClient, api_key: str, pem_secret: str) -> None:
        self.rest = rest
        self.api_key = api_key
        self.pem_secret = pem_secret
        self.fetcher = MacroFetcher(rest)
        self.macro = MacroManager()
        self.tlog = TradeLogger(TRADES_CSV_PATH)
        self.mlog = MarketLogger(MARKET_CSV_PATH)
        self.week_writer = CandleCSVWriter(MACRO_WEEK_CSV)
        self.day_writer = CandleCSVWriter(MACRO_DAY_CSV)
        # top of book per product
        self.tob: Dict[str, Optional[TopOfBook]] = {p: None for p in PRODUCTS}
        # Rolling mid price series per product
        self.mid_series: Dict[str, RollingMidSeries] = {p: RollingMidSeries() for p in PRODUCTS}
        # 1m candle series per product
        self.live_1m: Dict[str, LiveMinuteCandleSeries] = {p: LiveMinuteCandleSeries() for p in PRODUCTS}
        # positions per product: list of PositionLot
        self.positions: Dict[str, List[PositionLot]] = {p: [] for p in PRODUCTS}
        # parallel metadata for lots (so we can do tranche-specific exits without changing CSV schema)
        self.lot_tags: Dict[str, List[str]] = {p: [] for p in PRODUCTS}  # e.g., ["L1","L2","L3"]

        # Ladder plan per product (active when position exists). Keeps deterministic staged entry rules.
        # Structure:
        #   {"total_notional": float, "fracs": (f1,f2,f3), "notional_done": [n1,n2,n3],
        #    "entry1_price": float, "armed": bool}
        self.ladder_plan: Dict[str, Optional[Dict[str, Any]]] = {p: None for p in PRODUCTS}
        # executed buy levels per product (indices)
        self.executed_buy_idx: Dict[str, set] = {p: set() for p in PRODUCTS}
        # executed sell levels per product (indices)
        self.executed_sell_idx: Dict[str, set] = {p: set() for p in PRODUCTS}
        # anchor timestamp per product for anchored VWAP
        self.anchor_ts: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        # fair value smoothing state (per product)
        self.fair_value_smooth: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        self.fair_value_raw_hist: Dict[str, Deque[float]] = {p: deque(maxlen=FAIR_VALUE_MEDIAN_WINDOW) for p in PRODUCTS}

        # --- buy/sell logic state (decisioning only) ---
        # Previous mid for cross-detection
        self.prev_mid: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        # Entry cooldown + spacing
        self.last_buy_ts: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        self.last_buy_price: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        # Profit-target staggering timestamp
        self.last_target_sell_ts: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        # When the current position lifecycle began (for time-stop)
        self.position_start_ts: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        # Runner trailing state
        self.peak_bid: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        self.trailing_active: Dict[str, bool] = {p: False for p in PRODUCTS}
        # Tier tracking for the active position
        self.position_tier: Dict[str, int] = {p: 0 for p in PRODUCTS}
        self.last_tier_tp_ts: Dict[str, float] = {p: 0.0 for p in PRODUCTS}
        self.position_entry_price: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        # Risk-off throttling state (per product)
        self.last_risk_off_ts: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
        # Global exit timestamp (used to throttle re-entry after any liquidation)
        self.last_exit_ts: Optional[float] = None
        # Price-based re-entry gating (prevents rapid churn without using time)
        self.rearm_required: Dict[str, bool] = {p: False for p in PRODUCTS}
        # portfolio
        self.portfolio = PaperPortfolio(STARTING_CASH_USD) if PAPER_TRADING else LivePortfolio(rest)
        # last macro update time
        self.last_macro_update: float = 0.0
        # stop event
        self._stop_event = asyncio.Event()
        # startup timestamp (used for FIRST_BUY_DELAY_SEC warm-up)
        self.bot_start_ts: float = now_ts()
    async def preload_micro_history(self) -> None:
        """Preload the last MICRO_PRELOAD_MINUTES of 1m candles into micro buffers at startup.

        This ensures the micro price charts (RollingMidSeries) and signal charts (LiveMinuteCandleSeries)
        have immediate context when the bot starts.
        """
        try:
            end_ts = int(now_ts())
            start_ts = end_ts - int(MICRO_PRELOAD_MINUTES) * 60
            # Pull 1m candles for each product and seed both mid_series and live_1m.
            for product in PRODUCTS:
                candles = await self.fetcher.fetch_chunked(product, start_ts, end_ts, "ONE_MINUTE")
                if not candles:
                    continue
                for c in candles:
                    # Candle timestamps are minute-starts; place a synthetic mid tick mid-minute.
                    ts = float(int(c.ts) + 30)
                    mid = float(c.close)
                    if mid <= 0:
                        continue
                    self.mid_series[product].push(ts, mid)
                    self.live_1m[product].push_mid(ts, mid)
            print(f"[startup] preloaded {MICRO_PRELOAD_MINUTES}m micro context for {len(PRODUCTS)} products")
        except Exception as e:
            print(f"[startup] micro preload failed: {e}")


    # --------------------------------------------------------
    # Micro metrics (24h aware)
    # --------------------------------------------------------
    def _compute_anchored_vwap_24h(self, product_id: str, now_ts: float) -> Optional[float]:
        """Compute a 24h anchored VWAP using the in-memory 1m candle series.

        We use typical price (H+L+C)/3 weighted by per-minute volume proxy.
        If volume is missing/zero, we fall back to equal weights.
        """
        series = self.live_1m.get(product_id)
        if not series or not series.candles:
            return None
        start = int(now_ts) - 24 * 60 * 60
        candles = [c for c in list(series.candles) if int(c.minute_start_ts) >= start]
        if len(candles) < 30:
            return None
        tp = np.array([(float(c.high) + float(c.low) + float(c.close)) / 3.0 for c in candles], dtype=float)
        vol = np.array([float(getattr(c, "volume", 0.0) or 0.0) for c in candles], dtype=float)
        vsum = float(np.sum(vol))
        if vsum <= 1e-9:
            vol = np.ones_like(tp)
            vsum = float(np.sum(vol))
        return float(np.sum(tp * vol) / vsum) if vsum > 0 else None

    def _compute_value_area_mid(self, product_id: str) -> Optional[float]:
        """Return (VAL+VAH)/2 from day macro if available."""
        lv = self.macro.get_levels(product_id, "day")
        if not lv:
            return None
        if lv.val and lv.vah and lv.vah > 0 and lv.val > 0:
            return float((lv.val + lv.vah) / 2.0)
        return None


    def _require_live_fill(
        self,
        r: Any,
        *,
        product_id: str,
        side: str,
    ) -> Optional[Tuple[float, float, float, float, str]]:
        """Validate a live execution result and extract fill truth.

        Returns (filled_qty, avg_price, fee_usd, filled_notional_usd, order_id) or None.

        Safety rules (LIVE):
          - r must be a dict
          - r['ok'] is True
          - order_id is present (so we can reconcile)
          - filled_qty > 0
          - avg_price determinable and > 0 (from avg_price or filled_notional/qty)
        """
        side_u = str(side).upper().strip()
        if not isinstance(r, dict):
            print(f"[{side_u.lower()}] non-dict execution result for {product_id}: {type(r)}")
            return None
        if r.get("ok") is not True:
            err = r.get("error") or "exec_not_ok"
            print(f"[{side_u.lower()}] execution failed for {product_id}: {err}")
            return None

        order_id = r.get("order_id")
        if not isinstance(order_id, str) or not order_id.strip():
            # LivePortfolio should already fail without an order_id, but we double-enforce here.
            print(f"[{side_u.lower()}] missing order_id; refusing to mutate local state for {product_id}")
            return None

        filled_qty = safe_float(r.get("filled_qty")) or 0.0
        if filled_qty <= 1e-12:
            print(f"[{side_u.lower()}] zero fill for {product_id} (order_id={order_id})")
            return None

        fee_val = safe_float(r.get("fee_usd")) or 0.0
        filled_notional = safe_float(r.get("filled_notional_usd"))
        avg_px = safe_float(r.get("avg_price"))

        # If avg_price missing, derive it from notional/qty (still fill-truth, not a quote fallback).
        if (avg_px is None or avg_px <= 0) and filled_notional is not None and filled_notional > 0:
            avg_px = float(filled_notional) / float(filled_qty)

        if avg_px is None or avg_px <= 0:
            # Do NOT fall back to bid/ask. Without a fill price we cannot log truthfully.
            print(
                f"[{side_u.lower()}] missing avg_price and filled_notional_usd; refusing local state mutation for {product_id} (order_id={order_id})"
            )
            return None

        if filled_notional is None or filled_notional <= 0:
            filled_notional = float(filled_qty) * float(avg_px)

        return float(filled_qty), float(avg_px), float(fee_val), float(filled_notional), str(order_id)

    def _fifo_cost_basis(self, lots: List[PositionLot], qty: float) -> Tuple[float, Optional[float]]:
        """Compute FIFO cost basis for selling `qty` from `lots` without mutating them.

        Returns (cost_usd, fifo_avg_price). If qty <= 0 or insufficient lots, returns (0.0, None).
        """
        q = float(qty)
        if q <= 0:
            return 0.0, None
        remaining = q
        cost = 0.0
        for lot in lots:
            if remaining <= 0:
                break
            take = min(float(lot.qty), remaining)
            if take > 0:
                cost += take * float(lot.price)
                remaining -= take
        sold = q - remaining
        if sold <= 1e-12:
            return 0.0, None
        return float(cost), float(cost / sold)

    def _fifo_reduce_lots(self, product: str, qty_to_remove: float) -> Tuple[float, Optional[float]]:
        """Reduce position lots FIFO by qty_to_remove. Returns (removed_qty, fifo_avg_entry_price)."""
        lots = self.positions.get(product, [])
        if not lots or qty_to_remove <= 0:
            return 0.0, None
        remaining = float(qty_to_remove)
        removed_qty = 0.0
        removed_cost = 0.0
        new_lots: List[PositionLot] = []
        new_tags: List[str] = []
        tags = self.lot_tags.get(product, [])

        for i, lot in enumerate(lots):
            tag = tags[i] if i < len(tags) else ""
            if remaining <= 1e-12:
                new_lots.append(lot)
                new_tags.append(tag)
                continue
            take = min(float(lot.qty), remaining)
            removed_qty += take
            removed_cost += take * float(lot.price)
            left = float(lot.qty) - take
            remaining -= take
            if left > 1e-12:
                new_lots.append(PositionLot(qty=left, price=float(lot.price)))
                new_tags.append(tag)

        self.positions[product] = new_lots
        self.lot_tags[product] = new_tags
        avg_entry = (removed_cost / removed_qty) if removed_qty > 1e-12 else None
        return removed_qty, avg_entry

    async def _sell_partial(self, product: str, qty_to_sell: float, note: str) -> Optional[Tuple[float, float, float]]:
        """Sell qty_to_sell and return (sold_qty, exec_price, fee_usd) if filled."""
        tob = self.tob.get(product)
        if not tob:
            return None
        bid = tob.bid

        qty_to_sell = float(qty_to_sell)
        if qty_to_sell <= 1e-12:
            return None

        exec_price = float(bid)
        fee = 0.0
        sold_qty = qty_to_sell

        if isinstance(self.portfolio, LivePortfolio):
            r = await self._live_sell_market(product_id=product, base_qty=qty_to_sell)
            fill = self._require_live_fill(r, product_id=product, side="SELL")
            if fill is None:
                return None
            filled_qty, avg_px, fee_val, filled_notional, order_id = fill
            exec_price = float(avg_px)
            fee = float(fee_val)
            sold_qty = float(min(qty_to_sell, filled_qty))
            try:
                self.portfolio.cash_usd = float(await self._live_refresh_cash())
            except Exception:
                pass
        else:
            if self.portfolio:
                self.portfolio.credit(sold_qty * exec_price, TAKER_FEE_BPS)

        return sold_qty, exec_price, fee

    async def _force_sell_product(self, product: str, note: str = "") -> None:
        tob = self.tob.get(product)
        if not tob:
            return
        bid = tob.bid

        lots = self.positions.get(product, [])
        qty = sum(l.qty for l in lots) if lots else 0.0
        if qty <= 0:
            return

        if isinstance(self.portfolio, LivePortfolio):
            r = await self._live_sell_market(product_id=product, base_qty=float(qty))
            fill = self._require_live_fill(r, product_id=product, side="SELL")
            if fill is None:
                return
            filled_qty, avg_px, fee_val, filled_notional, order_id = fill
            exec_price = float(avg_px)
            fee = float(fee_val)
            qty_sold = float(min(qty, filled_qty))
            try:
                self.portfolio.cash_usd = float(await self._live_refresh_cash())
            except Exception:
                pass
        else:
            exec_price = float(bid)
            fee = 0.0
            qty_sold = float(qty)
            if self.portfolio:
                self.portfolio.credit(qty_sold * exec_price, TAKER_FEE_BPS)

        self.tlog.log_trade(
            event="SELL",
            product_id=product,
            side="SELL",
            qty=qty_sold,
            price=exec_price,
            fee_usd_val=fee,
            gross_pnl_usd=0.0,
            net_pnl_usd=-fee,
            entry_price=self.position_entry_price.get(product),
            exit_price=exec_price,
            weekly_bias=self.macro.compute_weekly_bias(product, tob.mid),
            note=note,
        )

        ts_now = now_ts()
        self.positions[product] = []
        self.lot_tags[product] = []
        self.ladder_plan[product] = None
        self.peak_bid[product] = None
        self.trailing_active[product] = False
        self.position_tier[product] = 0
        self.position_entry_price[product] = None
        self.last_tier_tp_ts[product] = ts_now
        self.last_exit_ts = ts_now
        self.rearm_required[product] = True


    def _compute_fair_value(self, product_id: str, mid: float, avwap_24h: Optional[float]) -> Optional[float]:
        """Compute and smooth a 24h-aware fair value.

        fair_value_raw = w1*anchored_vwap_24h + w2*value_area_mid + w3*mid

        Then smooth with (median -> EWMA -> per-tick step clamp) using per-product state.
        """
        if mid <= 0:
            return None

        value_mid = self._compute_value_area_mid(product_id)
        # Weights: anchored VWAP dominates; value-area midpoint adds macro value anchor; mid adds responsiveness.
        if avwap_24h is not None and value_mid is not None:
            w1, w2, w3 = 0.62, 0.23, 0.15
            raw = w1 * float(avwap_24h) + w2 * float(value_mid) + w3 * float(mid)
        elif avwap_24h is not None:
            w1, w3 = 0.80, 0.20
            raw = w1 * float(avwap_24h) + w3 * float(mid)
        elif value_mid is not None:
            w2, w3 = 0.70, 0.30
            raw = w2 * float(value_mid) + w3 * float(mid)
        else:
            raw = float(mid)

        # Update raw history (for median filter)
        hist = self.fair_value_raw_hist.get(product_id)
        if hist is None:
            self.fair_value_raw_hist[product_id] = deque(maxlen=FAIR_VALUE_MEDIAN_WINDOW)
            hist = self.fair_value_raw_hist[product_id]
        hist.append(float(raw))

        # Median filter (robust to occasional spikes)
        med = float(np.median(np.array(hist, dtype=float))) if len(hist) >= 3 else float(raw)

        # EWMA
        prev = self.fair_value_smooth.get(product_id)
        if prev is None:
            sm = med
        else:
            sm = (FAIR_VALUE_SMOOTH_ALPHA * med) + ((1.0 - FAIR_VALUE_SMOOTH_ALPHA) * float(prev))

        # Step clamp in bps (allows turns, but prevents whips)
        prev2 = self.fair_value_smooth.get(product_id)
        if prev2 is not None and prev2 > 0:
            step_bps = ((sm / float(prev2)) - 1.0) * 10_000.0
            max_up = FAIR_VALUE_MAX_STEP_BPS
            max_dn = FAIR_VALUE_MAX_STEP_DOWN_BPS
            step_bps = clamp(step_bps, -max_dn, max_up)
            sm = float(prev2) * bps_to_mult(step_bps)

        self.fair_value_smooth[product_id] = float(sm)
        return float(sm)

    def _compute_sigma_bps_from_1m(self, product_id: str) -> Optional[float]:
        """Sigma (bps) from 1m closes over SIGMA_WINDOW_MINUTES."""
        series = self.live_1m.get(product_id)
        if not series or not series.candles:
            return None
        closes = [float(c.close) for c in list(series.candles)[-SIGMA_WINDOW_MINUTES:]]
        if len(closes) < 12:
            return None
        rets = []
        for i in range(1, len(closes)):
            a = closes[i - 1]
            b = closes[i]
            if a > 0:
                rets.append((b / a) - 1.0)
        if len(rets) < 8:
            return None
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
        sig = math.sqrt(max(var, 0.0))
        return float(sig * 10_000.0)

    def _rolling_sigma_pct_from_1m(self, product_id: str, window_min: int) -> Optional[float]:
        """Rolling stddev of 1m simple returns over `window_min` minutes, returned as a decimal (e.g. 0.001 = 0.1%)."""
        series = self.live_1m.get(product_id)
        if not series or not series.candles:
            return None
        n = max(int(window_min), 2)
        closes = [float(c.close) for c in list(series.candles)[-n:]]
        if len(closes) < 3:
            return None
        rets: List[float] = []
        for i in range(1, len(closes)):
            a = closes[i - 1]
            b = closes[i]
            if a > 0:
                rets.append((b / a) - 1.0)
        if len(rets) < 2:
            return None
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
        return float(math.sqrt(max(var, 0.0)))

    def _mtf_ema_slope_ok(self, product_id: str, mid: float, avwap_24h: Optional[float]) -> Tuple[bool, Dict[str, float]]:
        """Multi-timeframe momentum filter derived from the 1m candle stream.
        We resample to 1h and 4h closes, compute EMA20/EMA60, and require positive slopes.
        """
        dbg: Dict[str, float] = {}
        series = self.live_1m.get(product_id)
        if not series or len(series.candles) < 240:  # need at least ~4h
            return True, dbg  # not enough data => do not block
        closes = [float(c.close) for c in series.candles]
        # Build 1h and 4h closes by simple bucket sampling from 1m series (last close in bucket).
        # Keep it deterministic and light.
        def bucket_closes(step_min: int) -> List[float]:
            out: List[float] = []
            for i in range(step_min - 1, len(closes), step_min):
                out.append(closes[i])
            return out

        def ema(vals: List[float], span: int) -> List[float]:
            if not vals:
                return []
            alpha = 2.0 / (span + 1.0)
            e = vals[0]
            out = [e]
            for v in vals[1:]:
                e = (alpha * v) + ((1.0 - alpha) * e)
                out.append(e)
            return out

        ok = True
        for label, step in (("h1", 60), ("h4", 240)):
            bc = bucket_closes(step)
            if len(bc) < 70:  # need enough samples for EMA60 to settle
                continue
            e20 = ema(bc, 20)
            e60 = ema(bc, 60)
            if len(e20) < 5 or len(e60) < 5:
                continue
            slope20 = e20[-1] - e20[-4]
            slope60 = e60[-1] - e60[-4]
            dbg[f"ema20_slope_{label}"] = float(slope20)
            dbg[f"ema60_slope_{label}"] = float(slope60)
            if slope20 <= 0 or slope60 <= 0:
                ok = False

        # Additional constraint: prefer price at/above anchored VWAP to avoid relief-rally longs.
        if avwap_24h is not None and mid < avwap_24h:
            dbg["below_avwap_24h"] = 1.0
            ok = False

        return ok, dbg

    def _allowed_session_hour(self, ts_now: float) -> bool:
        """Optional session filter. Disabled by default (returns True)."""
        if not globals().get("ENABLE_SESSION_FILTER", False):
            return True
        allowed = globals().get("SESSION_ALLOWED_UTC_HOURS", None)
        if not allowed:
            return True
        h = datetime.fromtimestamp(ts_now, tz=timezone.utc).hour
        return h in allowed

    def _adaptive_trail_k(self, product_id: str) -> float:
        """Modestly adapt trailing band multiplier using 5-day realized vol (if available)."""
        base = float(TRAIL_K_BASE)
        # If daily macro history exists in memory (macro manager), we can infer 5d realized vol from day candles we store.
        # We don't fetch extra data here; if insufficient, return base.
        series = self.live_1m.get(product_id)
        if not series or len(series.candles) < 60:
            return float(min(max(base, TRAIL_K_MIN), TRAIL_K_MAX))
        # Use 24h 1m series as a proxy for "recent realized vol" and map into a gentle scaling.
        sig = self._rolling_sigma_pct_from_1m(product_id, TRAIL_VOL_WINDOW_MIN)
        if sig is None:
            return float(min(max(base, TRAIL_K_MIN), TRAIL_K_MAX))
        # Map sigma to scaling in [0.9, 1.15] roughly.
        # Typical 1m sigma might be 0.0002..0.001 for majors; keep this bounded.
        scale = 0.90 + 0.25 * min(max(sig / 0.0007, 0.0), 1.0)
        k = base * scale
        return float(min(max(k, TRAIL_K_MIN), TRAIL_K_MAX))


    def _entry_gate_bottoming(
        self,
        *,
        product_id: str,
        mid: float,
        avwap_24h: Optional[float],
        trending_down: bool,
        weekly_bias: Optional[float],
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Option-1 entry gate (support zone -> reversal confirmation -> room to +2%).

        NOTE:
          - This replaces the previous time-windowed "bottoming/stabilization/local-low/base-build" gate.
          - Spread filters, buy cooldown, sizing, sell logic, live fill-truth, and CSV schemas are unchanged.
        """
        if mid <= 0:
            return False, "BUY_REJECT:A:bad_mid", {}

        # Macro context already computed by MacroManager
        day = self.macro.get_levels(product_id, "day")
        week = self.macro.get_levels(product_id, "week")

        ok_a, why_a = option1_in_support_zone(float(mid), day, week, float(SUPPORT_BUFFER_BPS))
        if not ok_a:
            return False, f"BUY_REJECT:A:{why_a}", {"A": 0.0}

        # Use in-memory 1m candles for this product (built from mid ticks)
        candles_deque = self.live_1m.get(product_id).candles if self.live_1m.get(product_id) is not None else None
        minute_candles = list(candles_deque) if candles_deque is not None else []
        ok_b, why_b = option1_reversal_confirmation(minute_candles)
        if not ok_b:
            return False, f"BUY_REJECT:B:{why_b}", {"B": 0.0}

        ok_c, why_c = option1_room_to_target(float(mid), day, week, float(RESIST_BUFFER_BPS))
        if not ok_c:
            return False, f"BUY_REJECT:C:{why_c}", {"C": 0.0}

        # For downstream candidate scoring/debug: approximate "dist_low_pct" vs the nearest major support anchor.
        support_refs: List[float] = []
        if day:
            support_refs += [float(day.support_zone_low), float(day.prev_low), float(day.val)]
        if week:
            support_refs += [float(week.support_zone_low), float(week.prev_low), float(week.val)]
        support_refs = [x for x in support_refs if x and x > 0]
        support_ref = min(support_refs) if support_refs else float(mid)
        dist_low_pct = (float(mid) / max(float(support_ref), 1e-12)) - 1.0

        reason = f"BUY_OK:A={why_a};B={why_b};C={why_c}"
        dbg = {
            "dist_low_pct": float(dist_low_pct),
        }
        return True, reason, dbg



    # --------------------------------------------------------
    # Live execution helpers (avoid blocking asyncio loop)
    # --------------------------------------------------------
    async def _live_buy_market(self, *, product_id: str, quote_usd: float) -> Any:
        """Run LivePortfolio.buy_market in a worker thread to avoid blocking the event loop."""
        if not isinstance(self.portfolio, LivePortfolio):
            raise RuntimeError("live buy called without LivePortfolio")
        return await asyncio.to_thread(self.portfolio.buy_market, product_id=product_id, quote_usd=float(quote_usd))


    async def _live_sell_market(self, *, product_id: str, base_qty: float) -> Any:
        """Run LivePortfolio.sell_market in a worker thread to avoid blocking the event loop."""
        if not isinstance(self.portfolio, LivePortfolio):
            raise RuntimeError("live sell called without LivePortfolio")
        return await asyncio.to_thread(self.portfolio.sell_market, product_id=product_id, base_qty=float(base_qty))


    async def _live_refresh_cash(self) -> float:
        """Refresh live cash snapshot in a thread (API calls can block)."""
        if not isinstance(self.portfolio, LivePortfolio):
            return float(getattr(self.portfolio, "cash_usd", 0.0)) if self.portfolio else 0.0
        return await asyncio.to_thread(self.portfolio.refresh_cash)



    async def _live_can_afford(self, notional_usd: float, fee_bps: float) -> bool:
        """Non-blocking afford-check for LivePortfolio (runs REST snapshot in a worker thread)."""
        if not isinstance(self.portfolio, LivePortfolio):
            return bool(self.portfolio.can_afford(notional_usd, fee_bps)) if self.portfolio else False
        return bool(await asyncio.to_thread(self.portfolio.can_afford, float(notional_usd), float(fee_bps)))

    async def _live_refresh_snapshot(self, *, force: bool = True, ttl_sec: float = 0.0) -> Optional[Dict[str, Dict[str, float]]]:
        """Refresh live balances snapshot from Coinbase in a worker thread (non-blocking for event loop)."""
        if not isinstance(self.portfolio, LivePortfolio):
            return None
        return await asyncio.to_thread(self.portfolio.refresh_snapshot, force=bool(force), ttl_sec=float(ttl_sec))

    async def run(self) -> None:
        """Launch all asynchronous loops and wait for them to finish."""
        # Preload micro + signal context (last 24 hours of 1m candles)
        await self.preload_micro_history()
        tasks = [
            asyncio.create_task(self.ws_loop()),
            asyncio.create_task(self.macro_loop()),
            asyncio.create_task(self.eval_loop()),
            asyncio.create_task(self.telemetry_loop()),
        ]
        await asyncio.gather(*tasks)

    # --------------------------------------------------------
    # WebSocket loop
    # --------------------------------------------------------
    async def ws_loop(self) -> None:
        """Connect to Coinbase WebSocket and update top‑of‑book and mid data."""
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    WS_MARKET_URL,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                ) as ws:
                    # authenticate and subscribe to ticker and heartbeats
                    jwt_token = jwt_generator.build_ws_jwt(self.api_key, self.pem_secret)
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channel": "ticker",
                        "product_ids": PRODUCTS,
                        "jwt": jwt_token
                    }))
                    jwt_token = jwt_generator.build_ws_jwt(self.api_key, self.pem_secret)
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channel": "heartbeats",
                        "jwt": jwt_token
                    }))
                    print("Subscribed to Coinbase WS ticker for", PRODUCTS)
                    async for message in ws:
                        if self._stop_event.is_set():
                            break
                        try:
                            data = json.loads(message)
                        except Exception:
                            continue
                        if data.get("type") in ("error", "subscriptions"):
                            continue
                        if data.get("channel") != "ticker":
                            continue
                        events = data.get("events") or []
                        for ev in events:
                            tickers = ev.get("tickers") or []
                            for t in tickers:
                                if not isinstance(t, dict):
                                    continue
                                product_id = t.get("product_id")
                                if product_id not in PRODUCTS:
                                    continue
                                bid = safe_float(t.get("best_bid"))
                                ask = safe_float(t.get("best_ask"))
                                if bid is None or ask is None:
                                    continue
                                ts = now_ts_i()
                                self.tob[product_id] = TopOfBook(bid=bid, ask=ask, ts=ts)
                                mid = (bid + ask) / 2.0
                                # update mid series and 1m candles
                                self.mid_series[product_id].push(ts, mid)
                                self.live_1m[product_id].push_mid(ts, mid)
            except Exception as e:
                print(f"WS error: {e}, reconnecting in {WS_RECONNECT_DELAY_SEC}s")
                await asyncio.sleep(WS_RECONNECT_DELAY_SEC)

    # --------------------------------------------------------
    # Macro loop
    # --------------------------------------------------------
    async def macro_loop(self) -> None:
        """
        Periodically fetch macro candles, compute macro levels and write CSVs for viewer.
        Uses chunked REST requests to avoid exceeding API limits.
        """
        while not self._stop_event.is_set():
            start_week = int(now_ts()) - 7 * 24 * 60 * 60
            start_day = int(now_ts_i()) - 24 * 60 * 60
            end_ts = int(now_ts_i())
            week_rows: List[Dict[str, Any]] = []
            day_rows: List[Dict[str, Any]] = []
            for product in PRODUCTS:
                # Weekly 15‑minute candles
                candles_week = await self.fetcher.fetch_chunked(product, start_week, end_ts, "FIFTEEN_MINUTE")
                if candles_week:
                    levels_week = compute_macro_levels(candles_week)
                    if levels_week:
                        self.macro.set_levels(product, "week", levels_week)
                    for c in candles_week:
                        week_rows.append({
                            "ts": c.ts,
                            "product_id": product,
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "volume": c.volume,
                        })
                # Daily 1‑minute candles: use live_1m if enough data else REST
                live_rows = self.live_1m[product].export_rows(product)
                if len(live_rows) >= 120:
                    candles_day: List[Candle] = [
                        Candle(
                            ts=int(r["ts"]),
                            open=float(r["open"]),
                            high=float(r["high"]),
                            low=float(r["low"]),
                            close=float(r["close"]),
                            volume=float(r.get("volume", 0.0))
                        ) for r in live_rows
                    ]
                else:
                    candles_day = await self.fetcher.fetch_chunked(product, start_day, end_ts, "ONE_MINUTE")
                if not candles_week:
                    print(f"[macro] week empty for {product}")
                if not candles_day:
                    print(f"[macro] day empty for {product} (live_rows={len(live_rows)})")
                if candles_day:
                    levels_day = compute_macro_levels(candles_day)
                    if levels_day:
                        self.macro.set_levels(product, "day", levels_day)
                    for c in candles_day:
                        day_rows.append({
                            "ts": c.ts,
                            "product_id": product,
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "volume": c.volume,
                        })
            # Write weekly and daily candles for viewer
            try:
                await self.week_writer.write(week_rows)
                await self.day_writer.write(day_rows)
            except Exception as e:
                print("[macro] write failed:", e)
            # update last macro time
            self.last_macro_update = now_ts_i()
            await asyncio.sleep(MACRO_REFRESH_EVERY_SEC)

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------
    async def eval_loop(self) -> None:
        """
        Support/Resistance (S/R) all-in model.

        - Only ONE position is allowed across the whole account at any time.
        - Entry: buy with (almost) all available USD when price is inside the macro SUPPORT zone
          (daily preferred, else weekly) and the 1m regime filter is not strongly trending down.
        - Exit: sell the entire position when price reaches the macro RESISTANCE zone.
        - Stop loss: sell the entire position if price breaks materially below the support zone.
        - Risk-off: if risk_off triggers (downtrend or weekly bias below threshold), LIQUIDATE
          the entire position immediately.

        NOTE: This replaces the previous fair-value dip + sigma exit behavior, but keeps all
        other system behaviors (feeds, macro writer, logs, telemetry format) the same.
        """
        while not self._stop_event.is_set():
            ts_now = now_ts_i()

            warmup_done = (ts_now - self.bot_start_ts) >= FIRST_BUY_DELAY_SEC

            # Compute current cash/equity.
            mid_by_product: Dict[str, float] = {}
            for _pid, _tob in self.tob.items():
                try:
                    if _tob is not None and float(getattr(_tob, "mid", 0.0)) > 0:
                        mid_by_product[_pid] = float(getattr(_tob, "mid"))
                except Exception:
                    continue

            snap_live: Optional[Dict[str, Dict[str, float]]] = None
            if isinstance(self.portfolio, LivePortfolio):
                # Coinbase account is the source of truth for cash.
                try:
                    snap_live = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                    cash_usd = float(self.portfolio.get_tradable_usd(snapshot=snap_live))
                except Exception:
                    cash_usd = float(self.portfolio.cash_usd)
            else:
                cash_usd = float(self.portfolio.cash_usd) if self.portfolio else 0.0

            pos_value = 0.0
            active_product: Optional[str] = None
            active_qty = 0.0
            for p, lots in self.positions.items():
                if not lots:
                    continue
                tob_p = self.tob.get(p)
                if tob_p:
                    q = sum(lot.qty for lot in lots)
                    pos_value += q * tob_p.mid
                    if q > 0:
                        active_product = p
                        active_qty = q
            # Equity: in live mode, compute directly from Coinbase account balances.
            if isinstance(self.portfolio, LivePortfolio):
                try:
                    if snap_live is None:
                        snap_live = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                    # Override active qty using account balance for accurate sell sizing.
                    if active_product:
                        _acct_qty = self.portfolio.get_total_asset(active_product.split("-")[0], snapshot=snap_live)
                        if _acct_qty > 0:
                            active_qty = float(_acct_qty)
                    cash_usd = float(self.portfolio.get_tradable_usd(snapshot=snap_live))
                    equity_usd = float(self.portfolio.compute_equity_usd(mid_by_product=mid_by_product, snapshot=snap_live))
                except Exception:
                    equity_usd = cash_usd + pos_value
            else:
                equity_usd = cash_usd + pos_value

            # Local helpers (kept local to avoid touching global structure).
            def _ema(values: List[float], period: int) -> Optional[float]:
                if not values or period <= 1 or len(values) < period:
                    return None
                alpha = 2.0 / (period + 1.0)
                ema_v = float(values[0])
                for x in values[1:]:
                    ema_v = alpha * float(x) + (1.0 - alpha) * float(ema_v)
                return float(ema_v)

            def _regime_is_trending_down(product_id: str) -> bool:
                """Return True if 1m EMA separation indicates a strong downtrend."""
                series = self.live_1m.get(product_id)
                if not series or not series.candles:
                    return False
                closes = [float(c.close) for c in list(series.candles)[-max(EMA_SLOW_MINUTES, EMA_FAST_MINUTES, 10):]]
                ef = _ema(closes[-EMA_FAST_MINUTES:], EMA_FAST_MINUTES)
                es = _ema(closes[-EMA_SLOW_MINUTES:], EMA_SLOW_MINUTES)
                if ef is None or es is None:
                    return False
                price = float(closes[-1])
                trend_strength = abs(ef - es) / max(price, 1e-12)
                return (ef < es) and (trend_strength > MAX_TREND_STRENGTH_BPS)

            def _sigma_bps_from_1m(product_id: str, mid: float) -> Optional[float]:
                series = self.live_1m.get(product_id)
                if not series or not series.candles or len(series.candles) < 10:
                    return None
                closes = [float(c.close) for c in list(series.candles)[-SIGMA_WINDOW_MINUTES:]]
                if len(closes) < 10:
                    return None
                rets = []
                for i in range(1, len(closes)):
                    a = closes[i - 1]
                    b = closes[i]
                    if a > 0:
                        rets.append((b / a) - 1.0)
                if not rets:
                    return None
                mu = sum(rets) / len(rets)
                var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
                sig = math.sqrt(max(var, 0.0))
                return float(sig * 10000.0)

            def _all_in_notional(cash_usd_val: float) -> float:
                # Keep a small buffer so fee debit doesn't fail on rounding.
                buf = max(1.0, cash_usd_val * 0.0025)  # 0.25% or $1
                return max(0.0, cash_usd_val - buf)

            best_candidate = None  # dict with keys used below

            for product in PRODUCTS:
                tob = self.tob.get(product)
                if not tob:
                    continue
                bid = tob.bid
                ask = tob.ask
                mid = tob.mid
                spread_bps = tob.spread_bps

                levels_day = self.macro.get_levels(product, "day")
                levels_week = self.macro.get_levels(product, "week")
                # Use DAY levels for support/resistance decisions (not micro/second-level structure)
                levels = levels_day

                weekly_bias = self.macro.compute_weekly_bias(product, mid) if levels_week else None
                trending_down = _regime_is_trending_down(product)
                sigma_bps = _sigma_bps_from_1m(product, mid)

                risk_off = trending_down or (weekly_bias is not None and weekly_bias < WEEKLY_BIAS_THRESHOLD)

                # -------------------------
                # Manage OPEN position
                # -------------------------
                if active_product == product and active_qty > 0:
                    lots = self.positions.get(product, [])
                    avg_entry_price = None
                    if lots:
                        qsum = sum(l.qty for l in lots)
                        if qsum > 0:
                            avg_entry_price = sum(l.qty * l.price for l in lots) / qsum

                    state = "HOLD"

                    async def _sell_all(note: str) -> None:
                        nonlocal active_qty, cash_usd, equity_usd
                        qty_to_sell = active_qty
                        if isinstance(self.portfolio, LivePortfolio):
                            # Use Coinbase account balance as source-of-truth for sell sizing.
                            try:
                                _snap = await self._live_refresh_snapshot(force=False, ttl_sec=0.0) or {}
                                _acct_qty = self.portfolio.get_total_asset(product.split("-")[0], snapshot=_snap)
                                if _acct_qty > 0:
                                    qty_to_sell = min(float(active_qty), float(_acct_qty))
                            except Exception:
                                pass

                        if qty_to_sell <= 0:
                            return
                        lots_before = list(self.positions.get(product, []))
                        notional_usd = qty_to_sell * bid
                        exec_price = bid
                        fee = 0.0
                        if isinstance(self.portfolio, LivePortfolio):
                            r = await self._live_sell_market(product_id=product, base_qty=qty_to_sell)
                            fill = self._require_live_fill(r, product_id=product, side="SELL")
                            if fill is None:
                                return
                            filled_qty, avg_px, fee_val, filled_notional, order_id = fill
                            exec_price = float(avg_px)
                            fee = float(fee_val)
                            qty_sold = min(float(qty_to_sell), float(filled_qty))
                            qty_to_sell = float(qty_sold)
                            notional_usd = float(filled_notional) if abs(qty_sold - filled_qty) <= 1e-9 else float(qty_sold) * float(avg_px)
                            # Refresh live cash from Coinbase (source of truth)
                            try:
                                cash_usd = float(await self._live_refresh_cash())
                            except Exception:
                                pass
                        else:
                            notional_usd = qty_to_sell * bid
                            if self.portfolio:
                                fee = self.portfolio.credit(notional_usd, TAKER_FEE_BPS)

                        fifo_cost, fifo_avg_entry = self._fifo_cost_basis(lots_before, qty_to_sell)
                        # Realized gross P&L uses FIFO lot cost basis for the exact sold quantity.
                        pnl_gross = float(notional_usd) - float(fifo_cost)

                        self.tlog.log_trade(
                            event="SELL",
                            product_id=product,
                            side="SELL",
                            qty=qty_to_sell,
                            price=exec_price,
                            fee_usd_val=fee,
                            gross_pnl_usd=pnl_gross,
                            net_pnl_usd=pnl_gross - fee,
                            entry_price=(fifo_avg_entry if fifo_avg_entry is not None else avg_entry_price),
                            exit_price=exec_price,
                            weekly_bias=weekly_bias,
                            note=note,
                        )

                        self.positions[product] = []
                        self.lot_tags[product] = []
                        self.ladder_plan[product] = None
                        active_qty = 0.0

                        self.executed_buy_idx[product].clear()
                        self.executed_sell_idx[product].clear()
                        self.anchor_ts[product] = None
                        self.last_buy_ts[product] = None
                        self.last_buy_price[product] = None
                        self.position_start_ts[product] = None
                        self.peak_bid[product] = None
                        self.trailing_active[product] = False
                        self.position_tier[product] = 0
                        self.position_entry_price[product] = None
                        self.last_tier_tp_ts[product] = ts_now
                        self.last_risk_off_ts[product] = ts_now
                        self.last_exit_ts = ts_now
                        # Require price to move ABOVE day support before re-entry (price-based, not time-based)
                        self.rearm_required[product] = True

                    if risk_off and (self.last_risk_off_ts.get(product) is None or (ts_now - self.last_risk_off_ts.get(product, 0.0) >= RISK_OFF_COOLDOWN_SEC)):
                        state = "RISK_OFF_LIQUIDATE"
                        await _sell_all("risk_off_liq")
                    else:
                        # Update peak bid for peak-based exits
                        peak_bid_now = float(self.peak_bid.get(product) or bid)
                        if bid > peak_bid_now:
                            peak_bid_now = float(bid)
                            self.peak_bid[product] = peak_bid_now

                        # Peak profit since entry (based on avg entry)
                        if avg_entry_price and avg_entry_price > 0:
                            peak_profit_pct = (peak_bid_now / float(avg_entry_price)) - 1.0
                        else:
                            peak_profit_pct = 0.0

                        held_tier = int(self.position_tier.get(product, 0) or 0)
                        entry_px = self.position_entry_price.get(product) or avg_entry_price
                        if entry_px and entry_px > 0 and held_tier in TIER_TP_PCT:
                            tp_pct = float(TIER_TP_PCT[held_tier])
                            tp_price = float(entry_px) * (1.0 + tp_pct)

                            # cooldown
                            if (ts_now - float(self.last_tier_tp_ts.get(product, 0.0))) >= TIER_TP_COOLDOWN_SEC:
                                if bid >= tp_price:
                                    # sell fraction of current active qty
                                    frac = float(TIER_TP_SELL_FRAC.get(held_tier, 0.0))
                                    qty_to_sell = float(active_qty) * frac
                                    res = await self._sell_partial(product, qty_to_sell, note=f"TP_TIER{held_tier}")
                                    if res is not None:
                                        sold_qty, exec_price, fee = res
                                        removed_qty, fifo_entry = self._fifo_reduce_lots(product, sold_qty)
                                        active_qty = max(0.0, active_qty - removed_qty)
                                        self.last_tier_tp_ts[product] = ts_now

                                        self.tlog.log_trade(
                                            event="SELL",
                                            product_id=product,
                                            side="SELL",
                                            qty=sold_qty,
                                            price=exec_price,
                                            fee_usd_val=fee,
                                            gross_pnl_usd=0.0,
                                            net_pnl_usd=-fee,
                                            entry_price=(fifo_entry if fifo_entry is not None else entry_px),
                                            exit_price=exec_price,
                                            weekly_bias=weekly_bias,
                                            note=f"tier_tp|tier={held_tier}",
                                        )

                        # HARD peak stop: sell on 0.5% drop from peak at ANY time
                        hard_stop_price = peak_bid_now * (1.0 - HARD_PEAK_STOP_PCT)
                        if bid <= hard_stop_price:
                            state = "HARD_PEAK_STOP"
                            await _sell_all("hard_peak_stop")
                            await asyncio.sleep(EVAL_TICK_SEC)
                            continue

                        # Arm trailing only AFTER +1.5% peak profit
                        if (not self.trailing_active.get(product, False)) and (peak_profit_pct >= TRAIL_ARM_PCT):
                            self.trailing_active[product] = True

                        # If armed, sell if price drops 0.25% from peak
                        if self.trailing_active.get(product, False):
                            trail_price = peak_bid_now * (1.0 - TRAIL_DRAWDOWN_PCT)
                            if bid <= trail_price:
                                state = "TRAIL_EXIT"
                                await _sell_all("trail_exit")
                                await asyncio.sleep(EVAL_TICK_SEC)
                                continue

                    # If we just exited, skip add/exit logic and move on.
                    if active_qty <= 0:
                        cash_usd = self.portfolio.cash_usd if self.portfolio else cash_usd
                        lots2 = self.positions.get(product, [])
                        pos_value2 = sum(l.qty for l in lots2) * mid if lots2 else 0.0
                        equity_usd = cash_usd + pos_value2
                        self.mlog.log_snapshot(
                            ts=ts_now,
                            product_id=product,
                            bid=bid,
                            ask=ask,
                            mid=mid,
                            spread_bps=spread_bps,
                            exposures_usd=pos_value2,
                            position_qty=sum(l.qty for l in lots2),
                            avg_entry_price=avg_entry_price,
                            anchored_vwap=self._compute_anchored_vwap_24h(product, ts_now),
                            fair_value=self._compute_fair_value(product, mid, self._compute_anchored_vwap_24h(product, ts_now)),
                            sigma_bps=sigma_bps,
                            weekly_bias=weekly_bias,
                            state=state,
                            cash_usd=cash_usd,
                            equity_usd=equity_usd,
                        )
                        continue

                    # Volatility estimate (decimal) from 1m returns over SIGMA_WINDOW_MINUTES (used for ladder add triggers).
                    sigma_pct = self._rolling_sigma_pct_from_1m(product, SIGMA_WINDOW_MINUTES) or 0.0

                    # Anchored VWAP for add3.
                    avwap_24h_now = self._compute_anchored_vwap_24h(product, ts_now)

                    # Laddered adds (only if plan exists and we still have cash).
                    plan = self.ladder_plan.get(product)
                    can_add = (spread_bps <= MAX_SPREAD_BPS) and (not trending_down) and (weekly_bias is None or weekly_bias >= WEEKLY_BIAS_THRESHOLD)
                    if plan and self.portfolio and can_add:
                        total_notional = float(plan.get("total_notional", 0.0))
                        f1, f2, f3 = plan.get("fracs", LADDER_FRACS)
                        n2 = total_notional * float(f2)
                        n3 = total_notional * float(f3)
                        not_done = plan.get("notional_done", [0.0, 0.0, 0.0])
                        # Tranche 2: add if price dips below entry1 by ~0.75σ (using 60m rolling σ) and we're still in support.
                        if not_done[1] <= 0.0 and n2 > 0 and sigma_pct > 0 and levels_day is not None:
                            entry1_price = float(plan.get("entry1_price", avg_entry_price or mid))
                            dip_trigger = entry1_price * (1.0 - (ADD2_K_SIGMA * sigma_pct))
                            if (levels_day.support_zone_low <= mid <= levels_day.support_zone_high) and (mid <= dip_trigger):
                                if await self._live_can_afford(n2, TAKER_FEE_BPS):
                                    if isinstance(self.portfolio, LivePortfolio):
                                        r = await self._live_buy_market(product_id=product, quote_usd=n2)
                                        fill = self._require_live_fill(r, product_id=product, side="BUY")
                                        if fill is None:
                                            continue
                                        filled_qty, avg_px, fee_val, filled_notional, order_id = fill
                                        fee2 = float(fee_val)
                                        eff_price2 = float((filled_notional + fee2) / filled_qty) if filled_qty > 0 else float(avg_px)
                                        # Refresh live cash from Coinbase (source of truth)
                                        try:
                                            await self._live_refresh_cash()
                                        except Exception:
                                            pass
                                        self.positions[product].append(PositionLot(qty=filled_qty, price=eff_price2))
                                        qty2 = filled_qty
                                        buy_px2 = avg_px
                                    else:
                                        qty2 = n2 / ask
                                        fee2 = self.portfolio.debit(n2, TAKER_FEE_BPS)
                                        eff_price2 = float((n2 + fee2) / qty2) if qty2 > 0 else float(ask)
                                        self.positions[product].append(PositionLot(qty=qty2, price=eff_price2))
                                        buy_px2 = ask
                                    self.lot_tags[product].append(f"T{self.position_tier.get(product, TIER_LOW)}")
                                    not_done[1] = float(n2)
                                    plan["notional_done"] = not_done
                                    self.last_buy_ts[product] = ts_now
                                    self.last_buy_price[product] = ask
                                    self.tlog.log_trade(
                                        event="BUY",
                                        product_id=product,
                                        side="BUY",
                                        qty=qty2,
                                        price=buy_px2,
                                        fee_usd_val=fee2,
                                        gross_pnl_usd=0.0,
                                        net_pnl_usd=-fee2,
                                        entry_price=buy_px2,
                                        exit_price=None,
                                        weekly_bias=weekly_bias,
                                        note="ladder_L2|dip_add",
                                    )

                        # Tranche 3: add when price reclaims anchored VWAP (often marks transition out of base).
                        if ADD3_RECLAIM_AVWAP and not_done[2] <= 0.0 and n3 > 0 and avwap_24h_now is not None:
                            if mid >= float(avwap_24h_now):
                                if await self._live_can_afford(n3, TAKER_FEE_BPS):
                                    if isinstance(self.portfolio, LivePortfolio):
                                        r = await self._live_buy_market(product_id=product, quote_usd=n3)
                                        fill = self._require_live_fill(r, product_id=product, side="BUY")
                                        if fill is None:
                                            continue
                                        filled_qty, avg_px, fee_val, filled_notional, order_id = fill
                                        fee3 = float(fee_val)
                                        eff_price3 = float((filled_notional + fee3) / filled_qty) if filled_qty > 0 else float(avg_px)
                                        # Refresh live cash from Coinbase (source of truth)
                                        try:
                                            await self._live_refresh_cash()
                                        except Exception:
                                            pass
                                        px3 = eff_price3
                                        qty3 = filled_qty
                                        buy_px3 = avg_px
                                    else:
                                        qty3 = n3 / ask
                                        buy_px3 = ask
                                        fee3 = self.portfolio.debit(n3, TAKER_FEE_BPS)
                                        px3 = float((n3 + fee3) / qty3) if qty3 > 0 else float(ask)
                                    self.positions[product].append(PositionLot(qty=qty3, price=px3))
                                    self.lot_tags[product].append(f"T{self.position_tier.get(product, TIER_LOW)}")
                                    not_done[2] = float(n3)
                                    plan["notional_done"] = not_done
                                    self.last_buy_ts[product] = ts_now
                                    self.last_buy_price[product] = ask
                                    self.tlog.log_trade(
                                        event="BUY",
                                        product_id=product,
                                        side="BUY",
                                        qty=qty3,
                                        price=buy_px3,
                                        fee_usd_val=fee3,
                                        gross_pnl_usd=0.0,
                                        net_pnl_usd=-fee3,
                                        entry_price=buy_px3,
                                        exit_price=None,
                                        weekly_bias=weekly_bias,
                                        note="ladder_L3|reclaim_avwap",
                                    )

                    # Helper: sell a specific qty (FIFO) while keeping CSV schema identical.
                    async def _sell_qty(note: str, qty_to_sell: float) -> None:
                        nonlocal active_qty, cash_usd, equity_usd
                        if qty_to_sell <= 0:
                            return
                        qty_to_sell = min(float(qty_to_sell), float(active_qty))
                        if isinstance(self.portfolio, LivePortfolio):
                            # Cap by Coinbase account balance (source-of-truth).
                            try:
                                _snap = await self._live_refresh_snapshot(force=False, ttl_sec=0.0) or {}
                                _acct_qty = self.portfolio.get_total_asset(product.split("-")[0], snapshot=_snap)
                                if _acct_qty > 0:
                                    qty_to_sell = min(float(qty_to_sell), float(_acct_qty))
                            except Exception:
                                pass

                        if qty_to_sell <= 0:
                            return
                        lots_before = list(self.positions.get(product, []))
                        notional_usd = qty_to_sell * bid
                        exec_price = bid
                        fee = 0.0
                        if isinstance(self.portfolio, LivePortfolio):
                            r = await self._live_sell_market(product_id=product, base_qty=qty_to_sell)
                            fill = self._require_live_fill(r, product_id=product, side="SELL")
                            if fill is None:
                                return
                            filled_qty, avg_px, fee_val, filled_notional, order_id = fill
                            exec_price = float(avg_px)
                            fee = float(fee_val)
                            qty_sold = min(float(qty_to_sell), float(filled_qty))
                            qty_to_sell = float(qty_sold)
                            notional_usd = float(filled_notional) if abs(qty_sold - filled_qty) <= 1e-9 else float(qty_sold) * float(avg_px)
                            # Refresh live cash from Coinbase (source of truth)
                            try:
                                cash_usd = float(await self._live_refresh_cash())
                            except Exception:
                                pass
                        else:
                            notional_usd = qty_to_sell * bid
                            if self.portfolio:
                                fee = self.portfolio.credit(notional_usd, TAKER_FEE_BPS)

                        fifo_cost, fifo_avg_entry = self._fifo_cost_basis(lots_before, qty_to_sell)
                        pnl_gross = float(notional_usd) - float(fifo_cost)

                        self.tlog.log_trade(
                            event="SELL",
                            product_id=product,
                            side="SELL",
                            qty=qty_to_sell,
                            price=exec_price,
                            fee_usd_val=fee,
                            gross_pnl_usd=pnl_gross,
                            net_pnl_usd=pnl_gross - fee,
                            entry_price=(fifo_avg_entry if fifo_avg_entry is not None else avg_entry_price),
                            exit_price=exec_price,
                            weekly_bias=weekly_bias,
                            note=note,
                        )

                        # Reduce lots FIFO.
                        lots_local = self.positions.get(product, [])
                        tags_local = self.lot_tags.get(product, [])
                        remaining = qty_to_sell
                        new_lots: List[PositionLot] = []
                        new_tags: List[str] = []
                        for lot, tag in zip(lots_local, tags_local):
                            if remaining <= 0:
                                new_lots.append(lot)
                                new_tags.append(tag)
                                continue
                            if lot.qty <= remaining + 1e-12:
                                remaining -= float(lot.qty)
                                continue
                            # partial reduce
                            new_lots.append(PositionLot(qty=float(lot.qty) - remaining, price=float(lot.price)))
                            new_tags.append(tag)
                            remaining = 0.0
                        self.positions[product] = new_lots
                        self.lot_tags[product] = new_tags
                        active_qty = sum(l.qty for l in new_lots)

                        if active_qty <= 1e-12:
                            # Fully flat: reset per-product state
                            self.executed_buy_idx[product].clear()
                            self.executed_sell_idx[product].clear()
                            self.anchor_ts[product] = None
                            self.last_buy_ts[product] = None
                            self.last_buy_price[product] = None
                            self.position_start_ts[product] = None
                            self.peak_bid[product] = None
                            self.trailing_active[product] = False
                            self.ladder_plan[product] = None
                            self.last_risk_off_ts[product] = ts_now
                            self.last_exit_ts = ts_now
                            self.rearm_required[product] = True

                    # Profit targets / staggered exits / volatility trailing DISABLED.
                    # (Exits are handled exclusively by HARD_PEAK_STOP_PCT and the armed trailing drawdown.)

                    cash_usd = self.portfolio.cash_usd if self.portfolio else cash_usd
                    lots2 = self.positions.get(product, [])
                    pos_value2 = sum(l.qty for l in lots2) * mid if lots2 else 0.0
                    equity_usd = cash_usd + pos_value2

                    self.mlog.log_snapshot(
                        ts=ts_now,
                        product_id=product,
                        bid=bid,
                        ask=ask,
                        mid=mid,
                        spread_bps=spread_bps,
                        exposures_usd=pos_value2,
                        position_qty=sum(l.qty for l in lots2),
                        avg_entry_price=avg_entry_price,
                        anchored_vwap=self._compute_anchored_vwap_24h(product, ts_now),
                        fair_value=self._compute_fair_value(product, mid, self._compute_anchored_vwap_24h(product, ts_now)),
                        sigma_bps=sigma_bps,
                        weekly_bias=weekly_bias,
                        state=state,
                        cash_usd=cash_usd,
                        equity_usd=equity_usd,
                    )
                    continue

                # -------------------------
                # Look for entries (tiered gate)
                # -------------------------
                if warmup_done:
                    if spread_bps > MAX_SPREAD_BPS or levels is None:
                        pass
                    else:
                        # Price-based re-arm: after any exit on this product, require price to first
                        # trade ABOVE the day support zone by a buffer before allowing a new entry.
                        if self.rearm_required.get(product, False):
                            rearm_level = levels.support_zone_high * (1.0 + (REENTRY_REARM_BPS / 10_000.0))
                            if mid >= rearm_level:
                                self.rearm_required[product] = False
                        if not self.rearm_required.get(product, False):
                            minute_candles = list(self.live_1m.get(product).candles) if self.live_1m.get(product) else []

                            ok, tier, reason = tiered_entry_gate(
                                mid=mid,
                                spread_bps=spread_bps,
                                levels_day=levels_day,
                                levels_week=levels_week,
                                minute_candles=minute_candles,
                                weekly_bias=weekly_bias,
                                trending_down=trending_down,
                                support_buffer_bps=SUPPORT_BUFFER_BPS,
                                resist_buffer_bps=RESIST_BUFFER_BPS,
                            )

                            if ok:
                                cand = {
                                    "product": product,
                                    "tier": tier,
                                    "mid": mid,
                                    "bid": bid,
                                    "ask": ask,
                                    "weekly_bias": weekly_bias,
                                    "sigma_bps": sigma_bps,
                                    "entry_reason": reason,
                                }

                                # Choose highest tier; tie-break by lowest spread, then highest weekly bias
                                if (best_candidate is None
                                    or cand["tier"] > best_candidate["tier"]
                                    or (cand["tier"] == best_candidate["tier"] and spread_bps < best_candidate.get("spread_bps", 1e9))
                                    or (cand["tier"] == best_candidate["tier"] and (weekly_bias or 0.0) > (best_candidate.get("weekly_bias") or 0.0))
                                ):
                                    cand["spread_bps"] = spread_bps
                                    best_candidate = cand

                # Always log snapshots for non-active products too.
                pos_qty = sum(l.qty for l in self.positions.get(product, []))
                avg_entry_price = None
                lots = self.positions.get(product, [])
                if lots:
                    qsum = sum(l.qty for l in lots)
                    if qsum > 0:
                        avg_entry_price = sum(l.qty * l.price for l in lots) / qsum

                self.mlog.log_snapshot(
                    ts=ts_now,
                    product_id=product,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    spread_bps=spread_bps,
                    exposures_usd=pos_qty * mid,
                    position_qty=pos_qty,
                    avg_entry_price=avg_entry_price,
                    anchored_vwap=self._compute_anchored_vwap_24h(product, ts_now),
                    fair_value=self._compute_fair_value(product, mid, self._compute_anchored_vwap_24h(product, ts_now)),
                    sigma_bps=sigma_bps,
                    weekly_bias=weekly_bias,
                    state="WATCH" if active_product is None else "IGNORED",
                    cash_usd=cash_usd,
                    equity_usd=equity_usd,
                )

            # Rotation: if holding a lower tier and a higher-tier candidate appears, exit then rotate
            if active_product is not None and active_qty > 0 and best_candidate is not None:
                held_tier = int(self.position_tier.get(active_product, 0) or 0)
                cand_tier = int(best_candidate.get("tier", 0) or 0)

                if best_candidate["product"] != active_product and cand_tier >= (held_tier + ROTATE_MIN_TIER_DELTA):
                    await self._force_sell_product(active_product, note=f"ROTATE:{held_tier}->{cand_tier}")
                    cash_usd = self.portfolio.cash_usd if self.portfolio else cash_usd
                    active_product = None
                    active_qty = 0.0

            # Execute entry after scanning all products.
            if best_candidate is not None and self.portfolio is not None and (active_product is None or active_qty <= 0):
                product = best_candidate["product"]
                tob = self.tob.get(product)
                if tob:
                    bid = tob.bid
                    ask = tob.ask
                    mid = tob.mid
                    spread_bps = tob.spread_bps

                    weekly_bias = best_candidate["weekly_bias"]
                    sigma_bps = best_candidate["sigma_bps"]

                    total_notional = _all_in_notional(self.portfolio.cash_usd)
                    f1, f2, f3 = LADDER_FRACS
                    n1 = total_notional * float(f1)
                    n2 = total_notional * float(f2)
                    n3 = total_notional * float(f3)

                    if await self._live_can_afford(n1, TAKER_FEE_BPS):
                            if isinstance(self.portfolio, LivePortfolio):
                                r = await self._live_buy_market(product_id=product, quote_usd=n1)
                                fill = self._require_live_fill(r, product_id=product, side="BUY")
                                if fill is None:
                                    await asyncio.sleep(EVAL_TICK_SEC)
                                    continue
                                filled_qty, avg_px, fee_val, filled_notional, order_id = fill
                                fee1 = float(fee_val)
                                eff_price1 = float((filled_notional + fee1) / filled_qty) if filled_qty > 0 else float(avg_px)
                                # Refresh live cash from Coinbase (source of truth)
                                try:
                                    await self._live_refresh_cash()
                                except Exception:
                                    pass
                                self.positions[product] = [PositionLot(qty=filled_qty, price=eff_price1)]
                                qty1 = filled_qty
                                buy_px1 = avg_px
                            else:
                                qty1 = n1 / ask
                                buy_px1 = ask
                                fee1 = self.portfolio.debit(n1, TAKER_FEE_BPS)

                                eff_price1 = float((n1 + fee1) / qty1) if qty1 > 0 else float(ask)
                                self.positions[product] = [PositionLot(qty=qty1, price=eff_price1)]
                            buy_tier = int(best_candidate.get("tier", TIER_LOW))
                            self.position_tier[product] = buy_tier
                            self.position_entry_price[product] = float(eff_price1)
                            self.last_tier_tp_ts[product] = 0.0
                            self.lot_tags[product] = [f"T{self.position_tier[product]}"]
                            self.ladder_plan[product] = {
                                "total_notional": float(total_notional),
                                "fracs": (float(f1), float(f2), float(f3)),
                                "notional_done": [float(n1), 0.0, 0.0],
                                "entry1_price": float(ask),
                                "armed": False,
                            }

                            self.position_start_ts[product] = ts_now
                            self.last_buy_ts[product] = ts_now
                            self.last_buy_price[product] = ask
                            self.anchor_ts[product] = ts_now
                            self.peak_bid[product] = bid
                            self.trailing_active[product] = False

                            self.tlog.log_trade(
                                event="BUY",
                                product_id=product,
                                side="BUY",
                                qty=qty1,
                                price=buy_px1,
                                fee_usd_val=fee1,
                                gross_pnl_usd=0.0,
                                net_pnl_usd=-fee1,
                                entry_price=buy_px1,
                                exit_price=None,
                                weekly_bias=weekly_bias,
                                note=f"ladder_L1|{best_candidate.get('entry_reason', 'OK')}",
                            )

                            cash_usd2 = self.portfolio.cash_usd
                            pos_value2 = qty1 * mid
                            equity_usd2 = cash_usd2 + pos_value2
                            self.mlog.log_snapshot(
                                ts=ts_now,
                                product_id=product,
                                bid=bid,
                                ask=ask,
                                mid=mid,
                                spread_bps=spread_bps,
                                exposures_usd=pos_value2,
                                position_qty=qty1,
                                avg_entry_price=ask,
                                anchored_vwap=self._compute_anchored_vwap_24h(product, ts_now),
                                fair_value=self._compute_fair_value(product, mid, self._compute_anchored_vwap_24h(product, ts_now)),
                                sigma_bps=sigma_bps,
                                weekly_bias=weekly_bias,
                                state="BUY_L1",
                                cash_usd=cash_usd2,
                                equity_usd=equity_usd2,
                            )
            await asyncio.sleep(EVAL_TICK_SEC)


    async def telemetry_loop(self) -> None:
        """
        Periodically log market snapshots for viewer.  Includes exposures,
        volatility, anchored VWAP and fair value.
        """
        while not self._stop_event.is_set():
            ts_now = now_ts_i()
            total_equity = 0.0
            cash_usd = self.portfolio.cash_usd if self.portfolio else 0.0
            # Compute total equity across all products
            for product, lots in self.positions.items():
                tob = self.tob.get(product)
                if tob and lots:
                    mid = (tob.bid + tob.ask) / 2.0
                    total_equity += sum(lot.qty for lot in lots) * mid
            equity_usd = cash_usd + total_equity
            # Log per product snapshot
            for product in PRODUCTS:
                tob = self.tob.get(product)
                if not tob:
                    continue
                mid = (tob.bid + tob.ask) / 2.0
                spread_bps = ((tob.ask - tob.bid) / mid) * 10_000.0 if mid > 0 else 0.0
                positions = self.positions[product]
                exposures_usd = sum(lot.qty * lot.price for lot in positions)
                position_qty = sum(lot.qty for lot in positions)
                avg_entry_price = (exposures_usd / position_qty) if position_qty > 0 else None
                # anchored vwap (24h anchored, always-on)
                avwap = self._compute_anchored_vwap_24h(product, ts_now)

                levels_day = self.macro.get_levels(product, "day")
                levels_week = self.macro.get_levels(product, "week")
                # 24h-aware fair value blend + smoothing
                fair_value = self._compute_fair_value(product, mid, avwap)

                sigma_bps = self._compute_sigma_bps_from_1m(product)
                weekly_bias = self.macro.compute_weekly_bias(product, mid)
                state = "long" if positions else "flat"
                self.mlog.log_snapshot(
                    ts=ts_now,
                    product_id=product,
                    bid=tob.bid,
                    ask=tob.ask,
                    mid=mid,
                    spread_bps=spread_bps,
                    exposures_usd=exposures_usd,
                    position_qty=position_qty,
                    avg_entry_price=avg_entry_price,
                    anchored_vwap=avwap,
                    fair_value=fair_value,
                    sigma_bps=sigma_bps,
                    weekly_bias=weekly_bias,
                    state=state,
                    cash_usd=cash_usd,
                    equity_usd=equity_usd,
                )
            await asyncio.sleep(EVAL_TICK_SEC)


# TopOfBook definition used in the bot
@dataclass
class TopOfBook:
    bid: float
    ask: float
    ts: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return ((self.ask - self.bid) / m) * 10_000.0 if m > 0 else 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


# ------------------------------------------------------------
# Authentication helpers
# ------------------------------------------------------------

def load_pem_secret_from_env() -> str:
    """Load the PEM private key from environment or file."""
    load_dotenv()
    secret_file = (os.environ.get("COINBASE_API_SECRET_FILE") or "").strip()
    inline_secret = (os.environ.get("COINBASE_API_SECRET") or "").strip()
    pem = ""
    if secret_file:
        if not os.path.isabs(secret_file):
            secret_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), secret_file)
        if not os.path.exists(secret_file):
            raise RuntimeError(f"COINBASE_API_SECRET_FILE='{secret_file}' not found.")
        with open(secret_file, "r", encoding="utf-8-sig") as f:
            pem = f.read()
    elif inline_secret:
        pem = inline_secret.replace("\n", "\n")
    else:
        raise RuntimeError("Provide COINBASE_API_SECRET_FILE or COINBASE_API_SECRET in .env")
    pem = pem.strip()
    if not pem.startswith("-----BEGIN") or "PRIVATE KEY" not in pem:
        raise RuntimeError("API secret does not look like PEM text.")
    return pem


def load_coinbase_client() -> RESTClient:
    """Instantiate the Coinbase REST client using env credentials."""
    load_dotenv()
    api_key = (os.environ.get("COINBASE_API_KEY") or "").strip()
    pem = load_pem_secret_from_env()
    if not api_key:
        raise RuntimeError("Missing COINBASE_API_KEY in .env")
    if not api_key.startswith("organizations/"):
        raise RuntimeError("COINBASE_API_KEY must start with 'organizations/.../apiKeys/...'")
    return RESTClient(api_key=api_key, api_secret=pem)


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

async def main() -> None:
    global PRODUCTS
    rest = load_coinbase_client()
    load_dotenv()
    api_key = (os.environ.get("COINBASE_API_KEY") or "").strip()
    pem = load_pem_secret_from_env()

    if AUTO_SELECT_PRODUCTS:
        try:
            PRODUCTS = await asyncio.to_thread(select_diversified_products)
            if not PRODUCTS:
                PRODUCTS = list(PRODUCTS_DEFAULT)
        except Exception as e:
            print("[select] failed, using default products:", e)
            PRODUCTS = list(PRODUCTS_DEFAULT)
    else:
        PRODUCTS = list(PRODUCTS_DEFAULT)

    # Currency safety: enforce USD quote pairs.
    PRODUCTS = [p for p in PRODUCTS if p.endswith("-USD")]

    print("[config] Trading products:", PRODUCTS)

    bot = TradingBot(rest=rest, api_key=api_key, pem_secret=pem)
    if not hasattr(bot, "run"):
        raise RuntimeError("TradingBot instance has no run(); ensure you are running the updated bot.py file.")
    await bot.run()
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot interrupted by user.")