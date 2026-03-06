# STRATEGY SUMMARY
# - Entry model: support-zone proximity + reversal confirmation + room-to-target
# - Exit model: full exit only via hard peak stop or armed trailing drawdown
# - Sizing model: slot-based percentage of available cash
# - Logging model: Coinbase fills are source of truth in live mode
# - Viewer model: reads bot-generated CSV outputs only; does not recompute alternate trade logic

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
USE_USD_HOLD_AS_TRADABLE: bool = False

# Optional session filter (UTC). Disabled by default to preserve existing behaviour.
# If enabled, the entry gate will only allow buys during the configured UTC hours.
ENABLE_SESSION_FILTER: bool = False
SESSION_ALLOWED_UTC_HOURS: Optional[List[int]] = list(range(13, 23))  # 13:00–22:59 UTC (US/EU overlap)


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
MACRO_WEEK_CSV: str = os.path.join(BASE_DIR, "macro_week.csv")  # 15-minute candles (past week)
MACRO_DAY_CSV: str = os.path.join(BASE_DIR, "macro_day.csv")    # 1-minute candles (past day)
MACRO_LEVELS_CSV: str = os.path.join(BASE_DIR, "macro_levels.csv")

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


# Price-based re-entry re-arm: after any full exit, the bot will NOT re-enter
# until price has first moved ABOVE the day support zone by this buffer (in bps),
# and then later returns back into the support zone.
REENTRY_REARM_BPS: float = 15.0


# ============================================================
# CANONICAL STRATEGY CONFIG
# ============================================================

# Warm-up / cadence
FIRST_BUY_DELAY_SEC: float = 30.0 * 60.0
BUY_COOLDOWN_SEC: float = 45.0
POST_EXIT_COOLDOWN_SEC: float = 5 * 60.0
EVAL_TICK_SEC: float = 2.0

# Exposure / allocation
MAX_OPEN_POSITIONS: int = 20
MIN_ENTRY_USD: float = 1.0
MAX_EXPOSURE_PER_PRODUCT_USD: float = 40.0
TARGET_UTIL_MIN: float = 0.25
TARGET_UTIL_MID: float = 0.60
TARGET_UTIL_MAX: float = 0.90
HIGH_SCORE_UTIL_THRESHOLD: float = 80.0
MID_SCORE_UTIL_THRESHOLD: float = 65.0
MAX_NEW_ENTRIES_PER_EVAL: int = 3

# Execution friction
MAX_SPREAD_BPS: float = 20.0
SCALP_MAX_SPREAD_BPS: float = 12.0
MAKER_FEE_BPS: float = 6.0
TAKER_FEE_BPS: float = 10.0
EST_SLIPPAGE_BPS: float = 6.0
EST_ADVERSE_FILL_BPS: float = 6.0

# Dip / reversal detection
DIP_LOOKBACK_MIN = 90
DIP_MAX_AGE_MIN = 15
DIP_MIN_PCT = 0.0020
DIP_RATE_MIN_BPS_PER_MIN = 6.0
REV_MIN_UP_CANDLES = 2
REV_RECLAIM_BPS = 8.0

# Tier score bands
TIER_LOW = 1
TIER_MID = 2
TIER_HIGH = 3

TIER_SCORE_BANDS = {
    TIER_LOW: (50.0, 64.9999),
    TIER_MID: (65.0, 79.9999),
    TIER_HIGH: (80.0, 100.0),
}

# Entry context
SUPPORT_BUFFER_BPS: float = 20.0
RESIST_BUFFER_BPS: float = 15.0
RSI_BUY_THRESHOLD: float = 35.0
EMA_ENTRY_FAST: int = 9
EMA_ENTRY_SLOW: int = 20
EMA_SLOPE_MAX_DOWN_BPS: float = 12.0
PIVOT_W: int = 2
REQUIRE_CONFIRMATIONS: int = 2

# Regime filter
EMA_FAST_MINUTES: int = 20
EMA_SLOW_MINUTES: int = 60
MAX_TREND_STRENGTH_BPS: float = 35.0
WEEKLY_BIAS_THRESHOLD: float = -0.5

# Score weights
SCORE_DIP_DEPTH_W: float = 24.0
SCORE_DIP_SPEED_W: float = 16.0
SCORE_REVERSAL_W: float = 20.0
SCORE_SUPPORT_W: float = 14.0
SCORE_ROOM_W: float = 18.0
SCORE_REGIME_W: float = 8.0
SCORE_SPREAD_PENALTY_W: float = 12.0
SCORE_COST_PENALTY_W: float = 10.0

# Exit inventory fractions by tier
EXIT_PLAN = {
    TIER_LOW:  {"scalp_frac": 0.80, "core_frac": 0.20, "runner_frac": 0.00},
    TIER_MID:  {"scalp_frac": 0.45, "core_frac": 0.40, "runner_frac": 0.15},
    TIER_HIGH: {"scalp_frac": 0.20, "core_frac": 0.45, "runner_frac": 0.35},
}

# Volatility-scaled objective multipliers
SCALP_SIGMA_MULT = {
    TIER_LOW: 0.75,
    TIER_MID: 0.90,
    TIER_HIGH: 1.10,
}
CORE_SIGMA_MULT = {
    TIER_LOW: 1.20,
    TIER_MID: 1.60,
    TIER_HIGH: 2.20,
}

# Peak-based protective exits
TRAIL_ARM_PCT: float = 0.015
TRAIL_DRAWDOWN_PCT: float = 0.0025
HARD_PEAK_STOP_PCT: float = 0.005

# Safety exits
TIME_STOP_SEC: int = 6 * 60 * 60
RISK_OFF_REDUCTION_FRAC: float = 0.05
RISK_OFF_COOLDOWN_SEC: float = 60.0
RISK_OFF_MIN_NOTIONAL_USD: float = 1.0

# Product universe filter
MIN_DAILY_RANGE_PCT: float = 0.05

# Legacy trailing-band constants kept disabled for compatibility with helper methods
TRAIL_VOL_WINDOW_MIN: int = 0
TRAIL_K_BASE: float = 0.0
TRAIL_K_MIN: float = 0.0
TRAIL_K_MAX: float = 0.0
TRAIL_MIN_DRAWDOWN_PCT: float = 0.0
TRAIL_MAX_DRAWDOWN_PCT: float = 0.0

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
    else:
        print("[selection] strict daily-range filter returned no products; using fallback candidate list")

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
    tier: int = TIER_LOW
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


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
    Build 1-minute synthetic activity candles from mid-price updates.
    These candles are suitable for price structure and approximate recency weighting,
    but they are not true exchange-trade volume candles.
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
        # Finalise previous synthetic activity candle
        self.candles.append(MinuteCandle(
            minute_start_ts=self._cur_minute,
            open=float(self._o),
            high=float(self._h),
            low=float(self._l),
            close=float(self._c),
            volume=float(self._v),
        ))
        # Start new synthetic activity candle
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
        try:
            if os.path.exists(self.path):
                df = pd.read_csv(self.path)
                if not df.empty and "cum_pnl_usd" in df.columns:
                    last_val = pd.to_numeric(df["cum_pnl_usd"], errors="coerce").dropna()
                    if not last_val.empty:
                        self.cum_pnl_usd = float(last_val.iloc[-1])
        except Exception:
            self.cum_pnl_usd = 0.0

    def _ensure_header(self) -> None:
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "dt_mst", "event", "product_id", "side", "qty", "price", "notional_usd",
                "fee_usd", "gross_pnl_usd", "net_pnl_usd", "cum_pnl_usd",
                "entry_price", "exit_price", "weekly_bias", "note",
                "entry_score", "entry_tier", "entry_reason", "expected_net_edge_bps", "lot_role",
                "exit_role", "exit_reason"
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
        filled_notional_usd: Optional[float] = None,
        entry_score: Optional[float] = None,
        entry_tier: Optional[int] = None,
        entry_reason: str = "",
        expected_net_edge_bps: Optional[float] = None,
        lot_role: str = "",
        exit_role: str = "",
        exit_reason: str = "",
    ) -> None:
        notional = float(filled_notional_usd) if filled_notional_usd is not None else (float(qty) * float(price))
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
                note,
                "" if entry_score is None else f"{entry_score:.6f}",
                "" if entry_tier is None else str(entry_tier),
                entry_reason,
                "" if expected_net_edge_bps is None else f"{expected_net_edge_bps:.6f}",
                lot_role,
                exit_role,
                exit_reason,
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
                "state", "cash_usd", "equity_usd",
                "entry_score", "entry_tier", "entry_reason", "expected_net_edge_bps",
                "dip_depth_score", "dip_speed_score", "reversal_score", "support_score",
                "room_score", "regime_score", "spread_penalty", "cost_penalty"
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
        entry_score: Optional[float] = None,
        entry_tier: Optional[int] = None,
        entry_reason: str = "",
        expected_net_edge_bps: Optional[float] = None,
        dip_depth_score: Optional[float] = None,
        dip_speed_score: Optional[float] = None,
        reversal_score: Optional[float] = None,
        support_score: Optional[float] = None,
        room_score: Optional[float] = None,
        regime_score: Optional[float] = None,
        spread_penalty: Optional[float] = None,
        cost_penalty: Optional[float] = None,
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
                f"{cash_usd:.6f}", f"{equity_usd:.6f}",
                "" if entry_score is None else f"{entry_score:.6f}",
                "" if entry_tier is None else str(entry_tier),
                entry_reason,
                "" if expected_net_edge_bps is None else f"{expected_net_edge_bps:.6f}",
                "" if dip_depth_score is None else f"{dip_depth_score:.6f}",
                "" if dip_speed_score is None else f"{dip_speed_score:.6f}",
                "" if reversal_score is None else f"{reversal_score:.6f}",
                "" if support_score is None else f"{support_score:.6f}",
                "" if room_score is None else f"{room_score:.6f}",
                "" if regime_score is None else f"{regime_score:.6f}",
                "" if spread_penalty is None else f"{spread_penalty:.6f}",
                "" if cost_penalty is None else f"{cost_penalty:.6f}"
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


class MacroLevelsCSVWriter:
    """Writes macro levels produced by the bot for viewer consumption."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.columns = [
            "ts", "product_id", "timeframe",
            "support_zone_low", "support_zone_high",
            "resistance_zone_low", "resistance_zone_high",
            "breakout", "range_low", "range_high",
            "prev_low", "prev_high", "vwap",
            "psych_low", "psych_high", "val", "vah", "price_now",
        ]
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.columns)

    async def write(self, rows: List[Dict[str, Any]]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.columns)
            for r in rows:
                w.writerow([r.get(c, "") for c in self.columns])
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
    computes additional metrics (range, VWAP, approximate activity-weighted value area, psychological levels).
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
    # Approximate activity-weighted value area on closes (uses synthetic tick activity if true trade volume is unavailable)
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
# Scored entry helpers
# ------------------------------------------------------------

def _dip_metrics(minute_candles: List['MinuteCandle']) -> Optional[Dict[str, float]]:
    if not minute_candles:
        return None
    lookback = minute_candles[-int(DIP_LOOKBACK_MIN):]
    if len(lookback) < 5:
        return None
    lows = [float(c.low) for c in lookback]
    highs = [float(c.high) for c in lookback]
    closes = [float(c.close) for c in lookback]
    trough_low = min(lows)
    trough_idx = lows.index(trough_low)
    current = closes[-1]
    pre_high = max(highs[:trough_idx + 1]) if trough_idx >= 0 else max(highs)
    if pre_high <= 0 or current <= 0:
        return None
    dip_pct = max(0.0, (pre_high - trough_low) / pre_high)
    trough_age_min = max(0, len(lookback) - 1 - trough_idx)
    dip_rate_bps_per_min = ((pre_high - trough_low) / pre_high) * 10_000.0 / max(1, trough_idx + 1)
    return {
        "dip_pct": float(dip_pct),
        "dip_rate_bps_per_min": float(dip_rate_bps_per_min),
        "trough_age_min": int(trough_age_min),
        "trough_low": float(trough_low),
    }


def _dip_reversal_ok(minute_candles: List['MinuteCandle'], trough_low: float) -> Tuple[bool, str]:
    if len(minute_candles) < max(REV_MIN_UP_CANDLES + 2, 5):
        return False, "not_enough_candles"
    closes = [float(c.close) for c in minute_candles]
    up_count = 0
    for i in range(len(closes) - REV_MIN_UP_CANDLES, len(closes)):
        if i <= 0:
            continue
        if closes[i] > closes[i - 1]:
            up_count += 1
    reclaim_level = trough_low * (1.0 + REV_RECLAIM_BPS / 10_000.0)
    if closes[-1] < reclaim_level:
        return False, f"no_reclaim last={closes[-1]:.6f} req={reclaim_level:.6f}"
    if up_count < REV_MIN_UP_CANDLES:
        return False, f"up_candles={up_count}"
    return True, f"up_candles={up_count} reclaim_ok"


def _room_to_target_pct(
    mid: float,
    day: Optional['MacroLevels'],
    week: Optional['MacroLevels'],
    target_pct: float,
    resist_buffer_bps: float,
) -> Tuple[bool, str]:
    if mid <= 0:
        return False, "bad_mid"
    target_px = mid * (1.0 + target_pct)
    levels = [x for x in [
        (day.resistance_zone_low if day else None),
        (day.resistance_zone_high if day else None),
        (day.prev_high if day else None),
        (week.resistance_zone_low if week else None),
        (week.resistance_zone_high if week else None),
        (week.prev_high if week else None),
    ] if x is not None and x > 0]
    if not levels:
        return True, "no_resistance_data"
    nearest_res = min(levels)
    buffer_px = mid * (resist_buffer_bps / 10_000.0)
    if target_px <= (nearest_res - buffer_px):
        return True, f"target_ok target={target_px:.6f} res={nearest_res:.6f}"
    return False, f"target_blocked target={target_px:.6f} res={nearest_res:.6f}"


def option1_room_to_target(mid: float, day: Optional['MacroLevels'], week: Optional['MacroLevels'], resist_buffer_bps: float) -> Tuple[bool, str]:
    return _room_to_target_pct(mid, day, week, target_pct=0.0080, resist_buffer_bps=resist_buffer_bps)


@dataclass
class EntryScore:
    ok: bool
    score: float
    tier: int
    reason: str
    dip_depth_score: float
    dip_speed_score: float
    reversal_score: float
    support_score: float
    room_score: float
    regime_score: float
    spread_penalty: float
    cost_penalty: float
    expected_net_edge_bps: float


def _score_to_tier(score: float) -> int:
    if score >= TIER_SCORE_BANDS[TIER_HIGH][0]:
        return TIER_HIGH
    if score >= TIER_SCORE_BANDS[TIER_MID][0]:
        return TIER_MID
    if score >= TIER_SCORE_BANDS[TIER_LOW][0]:
        return TIER_LOW
    return 0


def _clip_score(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def _support_proximity_score(mid: float, day: Optional['MacroLevels'], week: Optional['MacroLevels']) -> float:
    if mid <= 0:
        return 0.0

    candidates = []
    for levels in (day, week):
        if not levels:
            continue
        if levels.support_zone_low > 0 and levels.support_zone_high > 0:
            if levels.support_zone_low <= mid <= levels.support_zone_high:
                return 100.0
            zone_mid = (levels.support_zone_low + levels.support_zone_high) / 2.0
            dist_pct = abs(mid - zone_mid) / mid
            candidates.append(max(0.0, 100.0 - (dist_pct * 10000.0 * 4.0)))
        if levels.prev_low > 0:
            dist_pct = abs(mid - levels.prev_low) / mid
            candidates.append(max(0.0, 100.0 - (dist_pct * 10000.0 * 5.0)))
        if levels.val > 0:
            dist_pct = abs(mid - levels.val) / mid
            candidates.append(max(0.0, 100.0 - (dist_pct * 10000.0 * 5.0)))

    return float(max(candidates)) if candidates else 0.0


def _room_score(mid: float, day: Optional['MacroLevels'], week: Optional['MacroLevels'], resist_buffer_bps: float) -> Tuple[float, str]:
    room_ok, room_reason = option1_room_to_target(mid, day, week, resist_buffer_bps)
    if room_ok:
        return 100.0, room_reason

    low_room_ok, low_room_reason = _room_to_target_pct(
        mid, day, week, target_pct=0.0040, resist_buffer_bps=resist_buffer_bps
    )
    if low_room_ok:
        return 60.0, low_room_reason
    return 0.0, room_reason


def _estimate_net_edge_bps(
    *,
    score_room: float,
    spread_bps: float,
    tier_hint: int
) -> float:
    gross_target_bps = {
        TIER_LOW: 35.0,
        TIER_MID: 70.0,
        TIER_HIGH: 140.0,
    }.get(tier_hint, 35.0)

    friction = spread_bps + TAKER_FEE_BPS + EST_SLIPPAGE_BPS + EST_ADVERSE_FILL_BPS
    room_bonus = (score_room / 100.0) * 20.0
    return float(gross_target_bps + room_bonus - friction)


def score_entry_candidate(
    *,
    mid: float,
    spread_bps: float,
    levels_day: Optional['MacroLevels'],
    levels_week: Optional['MacroLevels'],
    minute_candles: List['MinuteCandle'],
    weekly_bias: Optional[float],
    trending_down: bool,
    resist_buffer_bps: float
) -> EntryScore:
    if mid <= 0:
        return EntryScore(False, 0.0, 0, "bad_mid", 0, 0, 0, 0, 0, 0, 0, 0, -999.0)

    dm = _dip_metrics(minute_candles)
    if not dm:
        return EntryScore(False, 0.0, 0, "dip_missing", 0, 0, 0, 0, 0, 0, 0, 0, -999.0)

    dip_pct = float(dm["dip_pct"])
    dip_rate = float(dm["dip_rate_bps_per_min"])
    trough_age = int(dm["trough_age_min"])
    trough_low = float(dm["trough_low"])

    if dip_pct < DIP_MIN_PCT:
        return EntryScore(False, 0.0, 0, f"dip_too_small={dip_pct:.4f}", 0, 0, 0, 0, 0, 0, 0, 0, -999.0)

    if trough_age > DIP_MAX_AGE_MIN:
        return EntryScore(False, 0.0, 0, f"dip_too_old age_min={trough_age}", 0, 0, 0, 0, 0, 0, 0, 0, -999.0)

    rev_ok, rev_reason = _dip_reversal_ok(minute_candles, trough_low)
    reversal_score = 100.0 if rev_ok else 0.0

    dip_depth_score = _clip_score((dip_pct / 0.0150) * 100.0)
    dip_speed_score = _clip_score((dip_rate / max(DIP_RATE_MIN_BPS_PER_MIN, 1e-9)) * 45.0)

    support_score = _support_proximity_score(mid, levels_day, levels_week)
    room_score, room_reason = _room_score(mid, levels_day, levels_week, resist_buffer_bps)

    if weekly_bias is None:
        regime_score = 50.0
    else:
        regime_score = _clip_score((weekly_bias + 1.0) * 50.0)

    if trending_down:
        regime_score = min(regime_score, 25.0)

    spread_penalty = max(0.0, (spread_bps - 5.0)) * (SCORE_SPREAD_PENALTY_W / 20.0)
    cost_penalty = (TAKER_FEE_BPS + EST_SLIPPAGE_BPS + EST_ADVERSE_FILL_BPS) * (SCORE_COST_PENALTY_W / 25.0)

    raw_score = (
        (dip_depth_score / 100.0) * SCORE_DIP_DEPTH_W
        + (dip_speed_score / 100.0) * SCORE_DIP_SPEED_W
        + (reversal_score / 100.0) * SCORE_REVERSAL_W
        + (support_score / 100.0) * SCORE_SUPPORT_W
        + (room_score / 100.0) * SCORE_ROOM_W
        + (regime_score / 100.0) * SCORE_REGIME_W
        - spread_penalty
        - cost_penalty
    )

    final_score = _clip_score(raw_score)
    tier = _score_to_tier(final_score)
    expected_net_edge_bps = _estimate_net_edge_bps(
        score_room=room_score,
        spread_bps=spread_bps,
        tier_hint=max(tier, TIER_LOW),
    )

    if not rev_ok:
        return EntryScore(False, final_score, tier, f"reversal_fail {rev_reason}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, expected_net_edge_bps)

    if support_score <= 0.0:
        return EntryScore(False, final_score, tier, "support_fail", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, expected_net_edge_bps)

    if room_score <= 0.0:
        return EntryScore(False, final_score, tier, f"room_fail {room_reason}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, expected_net_edge_bps)

    if spread_bps > MAX_SPREAD_BPS:
        return EntryScore(False, final_score, tier, f"spread_high={spread_bps:.1f}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, expected_net_edge_bps)

    if tier == TIER_LOW and spread_bps > SCALP_MAX_SPREAD_BPS:
        return EntryScore(False, final_score, tier, f"spread_high_low_tier={spread_bps:.1f}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, expected_net_edge_bps)

    if final_score < TIER_SCORE_BANDS[TIER_LOW][0]:
        return EntryScore(False, final_score, 0, f"score_too_low={final_score:.1f}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, expected_net_edge_bps)

    if expected_net_edge_bps <= 0.0:
        return EntryScore(False, final_score, 0, f"net_edge_nonpositive={expected_net_edge_bps:.1f}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, expected_net_edge_bps)

    return EntryScore(
        True,
        final_score,
        tier,
        f"score_ok={final_score:.1f} room={room_reason} edge_bps={expected_net_edge_bps:.1f}",
        dip_depth_score,
        dip_speed_score,
        reversal_score,
        support_score,
        room_score,
        regime_score,
        spread_penalty,
        cost_penalty,
        expected_net_edge_bps,
    )


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
    scored = score_entry_candidate(
        mid=mid,
        spread_bps=spread_bps,
        levels_day=levels_day,
        levels_week=levels_week,
        minute_candles=minute_candles,
        weekly_bias=weekly_bias,
        trending_down=trending_down,
        resist_buffer_bps=resist_buffer_bps,
    )
    return scored.ok, scored.tier, scored.reason

def _sigma_target_price(entry_price: float, sigma_bps: float, mult: float) -> float:
    if entry_price <= 0:
        return entry_price
    move_pct = (sigma_bps / 10000.0) * mult
    return entry_price * (1.0 + move_pct)


def get_exit_plan_for_tier(tier: int) -> Dict[str, float]:
    return EXIT_PLAN.get(tier, EXIT_PLAN[TIER_LOW])


def get_exit_targets(entry_price: float, sigma_bps: float, tier: int) -> Dict[str, float]:
    scalp_target = _sigma_target_price(entry_price, sigma_bps, SCALP_SIGMA_MULT[tier])
    core_target = _sigma_target_price(entry_price, sigma_bps, CORE_SIGMA_MULT[tier])
    return {
        "scalp_target": scalp_target,
        "core_target": core_target,
    }


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
        """
        Parse order fields into (filled_qty_base, avg_price, fee_usd, filled_notional_usd).
        MUST be base qty for filled_qty. If ambiguous, derive qty = notional/avg_price.
        """
        od = (order_d.get("order") if isinstance(order_d, dict) else None) or order_d
        side = str(od.get("side") or "").upper()

        # Common fields across SDK versions
        filled_size = safe_float(od.get("filled_size") or od.get("filledSize") or od.get("filled_base_size") or od.get("filledBaseSize"))
        filled_value = safe_float(od.get("filled_value") or od.get("filledValue") or od.get("filled_quote_size") or od.get("filledQuoteSize"))
        avg_price = safe_float(od.get("average_filled_price") or od.get("averageFilledPrice") or od.get("avg_price") or od.get("avgPrice"))
        fee = safe_float(od.get("total_fees") or od.get("totalFees") or od.get("fee") or od.get("fees")) or 0.0

        # Decide what is base qty vs notional
        filled_qty = float(filled_size or 0.0)
        filled_notional = filled_value

        # If avg_price missing but we have size/value, derive it
        if (avg_price is None or avg_price <= 0) and filled_qty > 0 and filled_notional is not None and filled_notional > 0:
            avg_price = float(filled_notional) / float(filled_qty)

        # If qty looks wrong (e.g., is actually quote), repair from notional/price
        if avg_price is not None and avg_price > 0 and filled_notional is not None and filled_notional > 0:
            expected_qty = float(filled_notional) / float(avg_price)
            if filled_qty <= 0 or (abs(filled_qty - expected_qty) / max(expected_qty, 1e-12)) > 0.10:
                filled_qty = float(expected_qty)

        return float(filled_qty), (float(avg_price) if avg_price else None), float(fee), (float(filled_notional) if filled_notional else None)


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

        # -------------------------------
        # FILL UNIT RECONCILIATION (CRITICAL)
        # Ensure filled_qty is BASE units and consistent with notional/price.
        # This prevents "qty looks like USD" bugs that create impossible positions/logs.
        # -------------------------------
        try:
            side_u2 = side_u  # BUY or SELL
            q = float(qty_f) if qty_f is not None else 0.0
            px = float(avg_px_f) if avg_px_f is not None else None
            notion = float(notional_f) if notional_f is not None else None

            if px is not None and px > 0 and notion is not None and notion > 0:
                expected_base = notion / px

                if side_u2 == "BUY":
                    qty_f = float(expected_base)
                # If qty is wildly inconsistent, replace qty with expected_base.
                # (Most common failure: BUY returns qty field in QUOTE units by mistake.)
                elif q <= 0 or (abs(q - expected_base) / max(expected_base, 1e-12)) > 0.10:
                    # prefer expected_base, but preserve the idea of "some fill happened"
                    qty_f = float(expected_base)

            # If we still don't have a usable avg_px but have notional and qty, derive it.
            if (avg_px_f is None or float(avg_px_f) <= 0) and notion is not None and float(qty_f) > 0:
                avg_px_f = float(notion) / float(qty_f)

        except Exception:
            pass

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

    def buy_limit_post_only(self, product_id: str, quote_usd: float, limit_price: float) -> dict:
        # Place a post-only limit buy using generic create order endpoint
        client_order_id = str(uuid.uuid4())
        payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {
                "limit_limit_gtc": {
                    "quote_size": str(round(float(quote_usd), 2)),
                    "limit_price": format(float(limit_price), ".8f").rstrip("0").rstrip("."),
                    "post_only": True,
                }
            },
        }
        return self.rest.post("/api/v3/brokerage/orders", data=payload)

    def sell_limit_post_only(self, product_id: str, base_qty: float, limit_price: float) -> dict:
        client_order_id = str(uuid.uuid4())
        payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": format(float(base_qty), ".10f").rstrip("0").rstrip("."),
                    "limit_price": format(float(limit_price), ".8f").rstrip("0").rstrip("."),
                    "post_only": True,
                }
            },
        }
        return self.rest.post("/api/v3/brokerage/orders", data=payload)

    def place_maker_with_reprice(
        self,
        *,
        side: str,
        product_id: str,
        quote_usd: Optional[float] = None,
        base_qty: Optional[float] = None,
        start_price: float,
        max_wait_sec: float = 6.0,
        reprice_every_sec: float = 2.0,
    ) -> dict:
        """
        Places post-only limit order and reprices a few times to improve fill odds,
        without ever crossing the spread (so it stays maker).
        Returns an ExecutionResult-like dict via existing fill parsing pipeline.
        """
        side_u = side.upper()
        assert side_u in ("BUY", "SELL")

        deadline = time.time() + float(max_wait_sec)
        limit_px = float(start_price)

        last_order_id = None

        while time.time() < deadline:
            try:
                if side_u == "BUY":
                    resp = self.buy_limit_post_only(product_id=product_id, quote_usd=float(quote_usd or 0.0), limit_price=limit_px)
                else:
                    resp = self.sell_limit_post_only(product_id=product_id, base_qty=float(base_qty or 0.0), limit_price=limit_px)
            except Exception as e:
                return ExecutionResult(
                    ok=False,
                    order_id=last_order_id,
                    client_order_id=None,
                    product_id=product_id,
                    side=side_u,
                    filled_qty=0.0,
                    avg_price=None,
                    fee_usd=0.0,
                    filled_notional_usd=None,
                    status="ERROR",
                    error=str(e),
                ).to_dict()

            order_id = self._extract_order_id(resp)
            last_order_id = order_id
            if not order_id:
                break

            t0 = time.time()
            while time.time() - t0 < float(reprice_every_sec) and time.time() < deadline:
                od = self.rest.get_order(order_id=order_id)
                od_d = od.to_dict() if hasattr(od, "to_dict") else od
                status = str(((od_d.get("order") or {}).get("status")) or "").upper()

                if "FILLED" in status:
                    fills = self._fetch_fills_for_order(order_id=order_id, product_id=product_id)
                    qty_f, avg_px_f, fee_f, notional_f = self._aggregate_fills(fills)

                    if qty_f <= 0:
                        qty_o, avg_px_o, fee_o, notional_o = self._parse_order_fill_fields(od_d)
                        qty_f, avg_px_f, fee_f, notional_f = qty_o, avg_px_o, fee_o, notional_o

                    try:
                        q = float(qty_f) if qty_f is not None else 0.0
                        px = float(avg_px_f) if avg_px_f is not None else None
                        notion = float(notional_f) if notional_f is not None else None

                        if px is not None and px > 0 and notion is not None and notion > 0:
                            expected_base = notion / px
                            if q <= 0 or (abs(q - expected_base) / max(expected_base, 1e-12)) > 0.10:
                                qty_f = float(expected_base)

                        if (avg_px_f is None or float(avg_px_f) <= 0) and notion is not None and float(qty_f) > 0:
                            avg_px_f = float(notion) / float(qty_f)
                    except Exception:
                        pass

                    try:
                        self.sync_after_trade(attempts=6, sleep_sec=0.5)
                    except Exception:
                        pass

                    return ExecutionResult(
                        ok=True,
                        order_id=order_id,
                        client_order_id=(od_d.get("order") or {}).get("client_order_id"),
                        product_id=product_id,
                        side=side_u,
                        filled_qty=float(qty_f),
                        avg_price=(float(avg_px_f) if avg_px_f else None),
                        fee_usd=float(fee_f or 0.0),
                        filled_notional_usd=(float(notional_f) if notional_f else None),
                        status="FILLED",
                        error=None,
                    ).to_dict()

                time.sleep(0.25)

            try:
                self.rest.cancel_orders(order_ids=[order_id])
            except Exception:
                pass

            if side_u == "BUY":
                limit_px *= 1.0002
            else:
                limit_px *= 0.9998

        return ExecutionResult(
            ok=False,
            order_id=last_order_id,
            client_order_id=None,
            product_id=product_id,
            side=side_u,
            filled_qty=0.0,
            avg_price=None,
            fee_usd=0.0,
            filled_notional_usd=None,
            status="NO_FILL",
            error="maker_no_fill",
        ).to_dict()


class TradingBot:
    """
    A structurally mean‑reverting trading bot.  Uses a three‑layer model:
    - Weekly macro to compute bias and support/resistance zones.
    - Daily macro to refine support and compute approximate activity-weighted value area for fair value.
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
        self.levels_writer = MacroLevelsCSVWriter(MACRO_LEVELS_CSV)
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
        self.entry_notional_usd: Dict[str, float] = {p: 0.0 for p in PRODUCTS}
        self.entry_buy_fee_usd: Dict[str, float] = {p: 0.0 for p in PRODUCTS}
        self.entry_buy_fee_bps: Dict[str, float] = {p: 0.0 for p in PRODUCTS}
        self.trailing_active: Dict[str, bool] = {p: False for p in PRODUCTS}
        # Tier tracking for the active position
        self.position_tier: Dict[str, int] = {p: 0 for p in PRODUCTS}
        self.last_tier_tp_ts: Dict[str, float] = {p: 0.0 for p in PRODUCTS}
        self.position_entry_price: Dict[str, Optional[float]] = {p: None for p in PRODUCTS}
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
                new_lots.append(PositionLot(qty=left, price=float(lot.price), tier=lot.tier, score=lot.score, meta=dict(lot.meta)))
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
        ask = tob.ask

        qty_to_sell = float(qty_to_sell)
        if qty_to_sell <= 1e-12:
            return None

        exec_price = float(bid)
        fee = 0.0
        sold_qty = qty_to_sell

        if isinstance(self.portfolio, LivePortfolio):
            r = await self._live_sell_maker(product_id=product, base_qty=qty_to_sell, ask=ask)
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
            r = await self._live_sell_maker(product_id=product, base_qty=float(qty), ask=tob.ask)
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
            filled_notional_usd=(float(filled_notional) if isinstance(self.portfolio, LivePortfolio) and filled_notional is not None else None),
        )

        ts_now = now_ts()
        self.positions[product] = []
        self.lot_tags[product] = []
        self.ladder_plan[product] = None
        self.peak_bid[product] = None
        self.trailing_active[product] = False
        self.position_tier[product] = 0
        self.position_entry_price[product] = None
        self.entry_notional_usd[product] = 0.0
        self.entry_buy_fee_usd[product] = 0.0
        self.entry_buy_fee_bps[product] = 0.0
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
        day = self.macro.get_levels(product_id, "day")
        week = self.macro.get_levels(product_id, "week")
        minute_candles = list(self.live_1m.get(product_id).candles) if self.live_1m.get(product_id) else []
        scored = score_entry_candidate(
            mid=float(mid),
            spread_bps=float(self.tob[product_id].spread_bps) if product_id in self.tob else 0.0,
            levels_day=day,
            levels_week=week,
            minute_candles=minute_candles,
            weekly_bias=weekly_bias,
            trending_down=trending_down,
            resist_buffer_bps=float(RESIST_BUFFER_BPS),
        )
        return scored.ok, scored.reason, {"score": scored.score}



    # --------------------------------------------------------
    # Live execution helpers (avoid blocking asyncio loop)
    # --------------------------------------------------------
    async def _live_buy_maker(self, *, product_id: str, quote_usd: float, bid: float) -> Any:
        if not isinstance(self.portfolio, LivePortfolio):
            raise RuntimeError("live buy called without LivePortfolio")
        # buy maker at bid (doesn't cross ask)
        return await asyncio.to_thread(
            self.portfolio.place_maker_with_reprice,
            side="BUY",
            product_id=product_id,
            quote_usd=float(quote_usd),
            start_price=float(bid),
            max_wait_sec=6.0,
            reprice_every_sec=2.0,
        )


    async def _live_sell_maker(self, *, product_id: str, base_qty: float, ask: float) -> Any:
        if not isinstance(self.portfolio, LivePortfolio):
            raise RuntimeError("live sell called without LivePortfolio")
        # sell maker at ask (doesn't cross bid)
        return await asyncio.to_thread(
            self.portfolio.place_maker_with_reprice,
            side="SELL",
            product_id=product_id,
            base_qty=float(base_qty),
            start_price=float(ask),
            max_wait_sec=6.0,
            reprice_every_sec=2.0,
        )


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
            levels_rows: List[Dict[str, Any]] = []
            for product in PRODUCTS:
                # Weekly 15‑minute candles
                candles_week = await self.fetcher.fetch_chunked(product, start_week, end_ts, "FIFTEEN_MINUTE")
                if candles_week:
                    levels_week = compute_macro_levels(candles_week)
                    if levels_week:
                        self.macro.set_levels(product, "week", levels_week)
                        levels_rows.append({"ts": end_ts, "product_id": product, "timeframe": "week", **levels_week.__dict__})
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
                        levels_rows.append({"ts": end_ts, "product_id": product, "timeframe": "day", **levels_day.__dict__})
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
                await self.levels_writer.write(levels_rows)
            except Exception as e:
                print("[macro] write failed:", e)
            # update last macro time
            self.last_macro_update = now_ts_i()
            await asyncio.sleep(MACRO_REFRESH_EVERY_SEC)

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------
    def _current_product_exposure_usd(self, product_id: str) -> float:
        tob = self.tob.get(product_id)
        if not tob:
            return 0.0
        return float(sum(l.qty for l in self.positions.get(product_id, [])) * tob.mid)

    def _current_total_exposure_usd(self) -> float:
        total = 0.0
        for product_id in PRODUCTS:
            total += self._current_product_exposure_usd(product_id)
        return float(total)

    def _open_position_count(self) -> int:
        return sum(1 for lots in self.positions.values() if sum(l.qty for l in lots) > 0)

    def compute_entry_notional(
        self,
        *,
        available_cash_usd: float,
        current_total_exposure_usd: float,
        current_equity_usd: float,
        current_product_exposure_usd: float,
        candidate_score: float,
        open_position_count: int,
        strong_candidate_count: int,
    ) -> float:
        if available_cash_usd <= 0:
            return 0.0

        if candidate_score >= HIGH_SCORE_UTIL_THRESHOLD:
            util_target = TARGET_UTIL_MAX
        elif candidate_score >= MID_SCORE_UTIL_THRESHOLD:
            util_target = TARGET_UTIL_MID
        else:
            util_target = TARGET_UTIL_MIN

        if strong_candidate_count >= 4:
            util_target = min(TARGET_UTIL_MAX, util_target + 0.10)

        target_gross_exposure = current_equity_usd * util_target
        deployable_gap = max(0.0, target_gross_exposure - current_total_exposure_usd)

        score_weight = max(0.25, min(1.0, candidate_score / 100.0))

        if candidate_score >= 80.0:
            base_alloc_frac = 0.12
        elif candidate_score >= 65.0:
            base_alloc_frac = 0.08
        else:
            base_alloc_frac = 0.05

        slot_softener = 1.0
        if open_position_count >= 10:
            slot_softener = 0.90
        if open_position_count >= 15:
            slot_softener = 0.80

        proposed = available_cash_usd * base_alloc_frac * score_weight * slot_softener
        proposed = min(proposed, deployable_gap if deployable_gap > 0 else proposed)

        remaining_product_room = max(0.0, MAX_EXPOSURE_PER_PRODUCT_USD - current_product_exposure_usd)
        proposed = min(proposed, remaining_product_room)

        if proposed < MIN_ENTRY_USD:
            return 0.0

        return float(proposed)

    async def eval_loop(self) -> None:
        while not self._stop_event.is_set():
            ts_now = now_ts_i()
            warmup_done = (ts_now - self.bot_start_ts) >= FIRST_BUY_DELAY_SEC

            snap_live: Optional[Dict[str, Dict[str, float]]] = None
            if isinstance(self.portfolio, LivePortfolio):
                try:
                    snap_live = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                    cash_usd = float(self.portfolio.get_tradable_usd(snapshot=snap_live))
                except Exception:
                    cash_usd = float(self.portfolio.cash_usd)
            else:
                cash_usd = float(self.portfolio.cash_usd) if self.portfolio else 0.0

            total_exposure = self._current_total_exposure_usd()
            equity_usd = cash_usd + total_exposure

            candidates = []
            for product_id in PRODUCTS:
                tob = self.tob.get(product_id)
                if not tob:
                    continue
                bid, ask, mid, spread_bps = tob.bid, tob.ask, tob.mid, tob.spread_bps
                levels_day = self.macro.get_levels(product_id, "day")
                levels_week = self.macro.get_levels(product_id, "week")
                weekly_bias = self.macro.compute_weekly_bias(product_id, mid) if levels_week else None
                minute_candles = list(self.live_1m.get(product_id).candles) if self.live_1m.get(product_id) else []
                sigma_bps = self._compute_sigma_bps_from_1m(product_id)

                lots = self.positions.get(product_id, [])
                position_qty = sum(l.qty for l in lots)
                avg_entry_price = (sum(l.qty * l.price for l in lots) / position_qty) if position_qty > 0 else None

                if position_qty > 0 and avg_entry_price and avg_entry_price > 0:
                    lot = lots[0]
                    lot_tier = lot.tier if lot.tier in EXIT_PLAN else TIER_LOW
                    lot_meta = lot.meta
                    exit_plan = get_exit_plan_for_tier(lot_tier)
                    targets = get_exit_targets(entry_price=avg_entry_price, sigma_bps=(sigma_bps or 35.0), tier=lot_tier)

                    sell_qty = 0.0
                    exit_reason = None
                    exit_role = None
                    if bid >= targets["scalp_target"] and not lot_meta.get("scalp_done", False):
                        sell_qty = position_qty * exit_plan["scalp_frac"]
                        exit_reason = "scalp_target_hit"
                        exit_role = "scalp_release"
                        lot_meta["scalp_done"] = True
                    elif bid >= targets["core_target"] and not lot_meta.get("core_done", False):
                        sell_qty = position_qty * exit_plan["core_frac"]
                        exit_reason = "core_target_hit"
                        exit_role = "core_release"
                        lot_meta["core_done"] = True

                    remaining_qty = sum(l.qty for l in self.positions.get(product_id, []))
                    peak_price = float(self.peak_bid.get(product_id) or bid)
                    if peak_price <= 0:
                        peak_price = bid
                    if bid > peak_price:
                        peak_price = bid
                        self.peak_bid[product_id] = peak_price
                    drawdown_from_peak = max(0.0, (peak_price - bid) / peak_price)
                    peak_profit = max(0.0, (peak_price - avg_entry_price) / avg_entry_price)
                    if drawdown_from_peak >= HARD_PEAK_STOP_PCT:
                        sell_qty = remaining_qty
                        exit_reason = "hard_peak_stop"
                        exit_role = "hard_peak_stop"
                    elif peak_profit >= TRAIL_ARM_PCT and drawdown_from_peak >= TRAIL_DRAWDOWN_PCT:
                        sell_qty = remaining_qty
                        exit_reason = "armed_trailing_drawdown"
                        exit_role = "runner_trail_exit"

                    sell_qty = min(position_qty, max(0.0, sell_qty))
                    if sell_qty > 0:
                        notional_usd = sell_qty * bid
                        exec_price = bid
                        fee = 0.0
                        filled_notional = None
                        if isinstance(self.portfolio, LivePortfolio):
                            r = await self._live_sell_maker(product_id=product_id, base_qty=sell_qty, ask=ask)
                            fill = self._require_live_fill(r, product_id=product_id, side="SELL")
                            if fill is not None:
                                filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
                                sell_qty = min(float(sell_qty), float(filled_qty))
                                exec_price = float(avg_px)
                                fee = float(fee_val)
                                notional_usd = float(filled_notional) if filled_notional is not None else float(sell_qty) * float(avg_px)
                            else:
                                sell_qty = 0.0
                        else:
                            fee = self.portfolio.credit(notional_usd, TAKER_FEE_BPS)

                        if sell_qty > 0:
                            fifo_cost, fifo_avg_entry = self._fifo_cost_basis(list(lots), sell_qty)
                            pnl_gross = float(notional_usd) - float(fifo_cost)
                            self.tlog.log_trade(
                                event="SELL", product_id=product_id, side="SELL", qty=sell_qty, price=exec_price,
                                fee_usd_val=fee, gross_pnl_usd=pnl_gross, net_pnl_usd=pnl_gross - fee,
                                entry_price=(fifo_avg_entry if fifo_avg_entry is not None else avg_entry_price),
                                exit_price=exec_price, weekly_bias=weekly_bias, note=exit_reason or "sell",
                                filled_notional_usd=(float(filled_notional) if filled_notional is not None else None),
                                exit_role=exit_role or "risk_off", exit_reason=exit_reason or "sell",
                            )
                            self._fifo_reduce_lots(product_id, sell_qty)

                scored = score_entry_candidate(
                    mid=mid,
                    spread_bps=spread_bps,
                    levels_day=levels_day,
                    levels_week=levels_week,
                    minute_candles=minute_candles,
                    weekly_bias=weekly_bias,
                    trending_down=False,
                    resist_buffer_bps=RESIST_BUFFER_BPS,
                )

                if scored.ok and warmup_done and self._open_position_count() < MAX_OPEN_POSITIONS:
                    candidates.append({
                        "product_id": product_id,
                        "mid": mid,
                        "bid": bid,
                        "ask": ask,
                        "spread_bps": spread_bps,
                        "score": scored.score,
                        "tier": scored.tier,
                        "reason": scored.reason,
                        "expected_net_edge_bps": scored.expected_net_edge_bps,
                        "weekly_bias": weekly_bias,
                    })

                self.mlog.log_snapshot(
                    ts=ts_now,
                    product_id=product_id,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    spread_bps=spread_bps,
                    exposures_usd=self._current_product_exposure_usd(product_id),
                    position_qty=position_qty,
                    avg_entry_price=avg_entry_price,
                    anchored_vwap=self._compute_anchored_vwap_24h(product_id, ts_now),
                    fair_value=self._compute_fair_value(product_id, mid, self._compute_anchored_vwap_24h(product_id, ts_now)),
                    sigma_bps=sigma_bps,
                    weekly_bias=weekly_bias,
                    state=("HOLD" if position_qty > 0 else "WATCH"),
                    cash_usd=cash_usd,
                    equity_usd=equity_usd,
                    entry_score=scored.score,
                    entry_tier=scored.tier,
                    entry_reason=scored.reason,
                    expected_net_edge_bps=scored.expected_net_edge_bps,
                    dip_depth_score=scored.dip_depth_score,
                    dip_speed_score=scored.dip_speed_score,
                    reversal_score=scored.reversal_score,
                    support_score=scored.support_score,
                    room_score=scored.room_score,
                    regime_score=scored.regime_score,
                    spread_penalty=scored.spread_penalty,
                    cost_penalty=scored.cost_penalty,
                )

            candidates.sort(key=lambda x: (x["score"], x["expected_net_edge_bps"]), reverse=True)
            strong_candidate_count = sum(1 for c in candidates if c["score"] >= MID_SCORE_UTIL_THRESHOLD)

            for candidate in candidates[:MAX_NEW_ENTRIES_PER_EVAL]:
                product_id = candidate["product_id"]
                if sum(l.qty for l in self.positions.get(product_id, [])) > 0:
                    continue
                product_exposure = self._current_product_exposure_usd(product_id)
                total_exposure = self._current_total_exposure_usd()
                open_count = self._open_position_count()

                entry_notional = self.compute_entry_notional(
                    available_cash_usd=cash_usd,
                    current_total_exposure_usd=total_exposure,
                    current_equity_usd=equity_usd,
                    current_product_exposure_usd=product_exposure,
                    candidate_score=candidate["score"],
                    open_position_count=open_count,
                    strong_candidate_count=strong_candidate_count,
                )
                if entry_notional < MIN_ENTRY_USD or not await self._live_can_afford(entry_notional, TAKER_FEE_BPS):
                    continue

                bid, ask = candidate["bid"], candidate["ask"]
                fee1 = 0.0
                filled_notional = None
                if isinstance(self.portfolio, LivePortfolio):
                    r = await self._live_buy_maker(product_id=product_id, quote_usd=entry_notional, bid=bid)
                    fill = self._require_live_fill(r, product_id=product_id, side="BUY")
                    if fill is not None:
                        filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
                        qty1 = float(filled_qty)
                        buy_px1 = float(avg_px)
                        fee1 = float(fee_val)
                        eff_price1 = float((filled_notional + fee1) / qty1) if qty1 > 0 and filled_notional is not None else buy_px1
                    else:
                        continue
                else:
                    qty1 = entry_notional / ask
                    buy_px1 = ask
                    fee1 = self.portfolio.debit(entry_notional, TAKER_FEE_BPS)
                    eff_price1 = float((entry_notional + fee1) / qty1) if qty1 > 0 else float(ask)

                if qty1 > 0:
                    lot_meta = {"scalp_done": False, "core_done": False}
                    self.positions[product_id] = [PositionLot(qty=qty1, price=eff_price1, tier=int(candidate["tier"]), score=float(candidate["score"]), meta=lot_meta)]
                    self.position_start_ts[product_id] = ts_now
                    self.last_buy_ts[product_id] = ts_now
                    self.last_buy_price[product_id] = ask
                    self.anchor_ts[product_id] = ts_now
                    self.peak_bid[product_id] = bid
                    self.tlog.log_trade(
                        event="BUY", product_id=product_id, side="BUY", qty=qty1, price=buy_px1,
                        fee_usd_val=fee1, gross_pnl_usd=0.0, net_pnl_usd=-fee1,
                        entry_price=buy_px1, exit_price=None, weekly_bias=candidate.get("weekly_bias"),
                        note=candidate.get("reason", "score_entry"),
                        filled_notional_usd=(float(filled_notional) if filled_notional is not None else None),
                        entry_score=float(candidate["score"]), entry_tier=int(candidate["tier"]),
                        entry_reason=str(candidate.get("reason", "score_entry")),
                        expected_net_edge_bps=float(candidate.get("expected_net_edge_bps", 0.0)),
                        lot_role="runner",
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
