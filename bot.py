# STRATEGY SUMMARY
# - Entry model: support-zone proximity + reversal confirmation + room-to-target
# - Exit model: tiered targets plus trailing, time, no-progress, and invalidation protection
# - Sizing model: exposure-aware entries with one optional add into confirmed winners
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
TARGET_PRODUCT_COUNT: int = 8          # total products to trade (includes BTC if available)
CANDIDATE_TOP_BY_USD_VOL: int = 60      # only consider the top-N USD-volume products (liquidity filter)
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
ORDERS_CSV_PATH: str = os.path.join(BASE_DIR, "orders.csv")
MARKET_CSV_PATH: str = os.path.join(BASE_DIR, "market.csv")
MACRO_WEEK_CSV: str = os.path.join(BASE_DIR, "macro_week.csv")  # 15-minute candles (past week)
MACRO_DAY_CSV: str = os.path.join(BASE_DIR, "macro_day.csv")    # 1-minute candles (past day)
MACRO_LEVELS_CSV: str = os.path.join(BASE_DIR, "macro_levels.csv")
CALIBRATION_CSV_PATH: str = os.path.join(BASE_DIR, "calibration.csv")
MICRO_HISTORY_CSV_PATH: str = os.path.join(BASE_DIR, "micro_history.csv")
POSITION_TARGETS_CSV_PATH: str = os.path.join(BASE_DIR, "position_targets.csv")
DEBUG_LOG_PATH: str = os.path.join(BASE_DIR, "debug.log")
DEBUG_LOG_ENABLED: bool = True
DEBUG_LOG_MAX_BYTES: int = 5_000_000

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

# ============================================================
# HIGHER-FREQUENCY CANONICAL TRADE CONFIG
# ============================================================

# Warm-up / cadence
FIRST_BUY_DELAY_SEC: float = 0.0
BUY_COOLDOWN_SEC: float = 20.0
POST_EXIT_COOLDOWN_SEC: float = 120.0
MAX_NEW_ENTRIES_PER_EVAL: int = 1
EVAL_TICK_SEC: float = 2.0

# Allocation / exposure
MAX_OPEN_POSITIONS: int = 20

# Minimum Coinbase order size guard.
# This is a bot-side minimum. Coinbase may still enforce product-specific minimums.
MIN_ENTRY_USD: float = 1.0
MIN_LIVE_ORDER_USD: float = 5.0

# Old fixed-dollar sizing is disabled.
ENTRY_SIZE_USD: float = 8.0
USE_FIXED_ENTRY_SIZE_USD: bool = False

# New probability-weighted percentage-of-equity sizing.
USE_EQUITY_PERCENT_POSITION_SIZING: bool = True

# A single new buy can be 5%–20% of total account equity.
# Total equity means cash + positions valued using live mids.
MIN_POSITION_PCT_OF_EQUITY: float = 0.05
MAX_SINGLE_BUY_PCT_OF_EQUITY: float = 0.20

# Max total exposure per product can reach 50% of total equity through scale-ins.
MAX_EXPOSURE_PER_PRODUCT_PCT_OF_EQUITY: float = 0.50

# Probability mapping:
# estimated probability below 52% gets no size.
# estimated probability at 52% gets minimum size; 78% gets maximum size.
PROB_FOR_MIN_SIZE: float = 0.52
PROB_FOR_MAX_SIZE: float = 0.78

# Cash reserve.
# The bot will not intentionally spend this final amount.
RESERVE_USD: float = 2.00

# Capital rotation:
# If available cash is insufficient for a stronger setup, the bot may sell weaker
# existing positions only when they are net-profitable after estimated fees.
ENABLE_PROFITABLE_ROTATION: bool = True
ROTATION_MIN_NEW_PROB_ADVANTAGE: float = 0.08
ROTATION_MIN_NEW_SCORE_ADVANTAGE: float = 8.0
ROTATION_MIN_NET_PROFIT_BPS: float = 15.0
ROTATION_SELL_FRACTION: float = 1.0

# Coinbase fee-tier auto-refresh.
# This bot requires real Coinbase-provided maker/taker fee rates before trading.
AUTO_REFRESH_COINBASE_FEE_TIER: bool = True
FEE_TIER_REFRESH_SEC: float = 60 * 60
REQUIRE_COINBASE_FEE_TIER: bool = True

# Coinbase portfolio source-of-truth behavior.
# In live mode, Coinbase balances and fills should be treated as authoritative.
SOURCE_OF_TRUTH_COINBASE: bool = True

# Startup handling for existing Coinbase crypto balances in PRODUCTS.
# Options:
#   "LIQUIDATE_EXISTING" = sell existing available balances for configured products before new entries.
#   "ADOPT_EXISTING"    = treat existing balances as bot-managed positions using current mid as approximate entry.
#   "IGNORE_EXISTING"   = leave existing balances alone, but still include them in equity/exposure calculations.
LIVE_STARTUP_MODE: str = "LIQUIDATE_EXISTING"

# Use market sells for startup liquidation.
# This is more likely to exit than maker-only post-only sells, but may pay taker fees/slippage.
STARTUP_LIQUIDATION_USE_MARKET: bool = True

# Skip tiny balances whose estimated USD value is below this threshold.
# Your current $1–$2 holdings may be near exchange minimums, so some sells may fail or be skipped.
MIN_STARTUP_LIQUIDATION_USD: float = 0.01

# How long to wait for websocket top-of-book prices before startup reconciliation.
STARTUP_TOB_TIMEOUT_SEC: float = 30.0

TARGET_UTIL_MIN: float = 0.35
TARGET_UTIL_MID: float = 0.65
TARGET_UTIL_MAX: float = 0.90

HIGH_SCORE_UTIL_THRESHOLD: float = 78.0
MID_SCORE_UTIL_THRESHOLD: float = 60.0

# Execution mode
# MARKET = more reliable fills, higher taker fee/spread cost.
# MAKER = cheaper if filled, but can fail/no-fill often.
# LIMIT_THEN_MARKET = try maker briefly, then fall back to market.
ENTRY_EXECUTION_MODE: str = "MARKET"
EXIT_EXECUTION_MODE: str = "LIMIT_THEN_MARKET"

# Execution friction
MAX_SPREAD_BPS: float = 18.0
SCALP_MAX_SPREAD_BPS: float = 10.0
EST_SLIPPAGE_BPS: float = 6.0
EST_ADVERSE_FILL_BPS: float = 6.0

# Fee-aware edge requirements
MIN_REQUIRED_NET_EDGE_BPS: float = 35.0
MIN_TARGET_TO_COST_MULT: float = 2.75
ROUND_TRIP_SAFETY_BPS: float = 8.0

# Minimum realized net gain required for discretionary profit exits and the
# calibrated projected-forward-gain buy gate. 1 basis point = 0.01%.
MIN_NET_GAIN_AFTER_FEES_BPS: float = 1.0

# Calibrated buy gate behavior.
# The old target-to-cost gate used target_bps, which is often only a few bps.
# The new EV system should use calibrated projected forward gain instead.
USE_CALIBRATED_FORWARD_GAIN_FOR_TARGET_COST_GATE: bool = True

# EV-primary buy behavior.
# When projected EV and cost coverage pass, score/probability targets are treated
# as ideal targets, not absolute blockers.
USE_EV_PRIMARY_BUY_GATE: bool = True

# Hard minimums prevent extremely weak signals from buying.
EV_PRIMARY_MIN_SCORE_FLOOR: float = 25.0
EV_PRIMARY_MIN_PROB_FLOOR: float = 0.35

# Strong EV can override conservative fallback score/probability targets.
EV_PRIMARY_MIN_PROJECTED_NET_BPS: float = 35.0

# Require projected forward gain to cover modeled cost plus minimum gain.
# This is more appropriate than requiring 2.75x cost for a tiny scalping strategy.
MIN_PROJECTED_GAIN_OVER_COST_BPS: float = MIN_NET_GAIN_AFTER_FEES_BPS

# Allow calibrated high-EV setups to buy without the old strict dip setup.
REQUIRE_STRICT_DIP_GATE_FOR_BUY: bool = False

# Do not require the old dip to be fresh if calibrated EV/probability/score pass.
REQUIRE_BASIC_REVERSAL_CONFIRMATION_FOR_CALIBRATED_BUY: bool = False

# Still block clearly falling markets.
BLOCK_BUY_WHEN_MICRO_TRENDING_DOWN: bool = True

# Basic reversal fallback requirements when strict_entry.ok is false.
BASIC_REVERSAL_MIN_SCORE: float = 45.0
BASIC_REVERSAL_MIN_ROOM_SCORE: float = 35.0
BASIC_REVERSAL_MIN_SUPPORT_SCORE: float = 35.0

# Order logging / safety
LOG_ORDER_ATTEMPTS: bool = True
REQUIRE_CONFIRMED_FILL_FOR_TRADE_LOG: bool = True

# Dip / reversal detection
DIP_LOOKBACK_MIN: int = 75
DIP_MAX_AGE_MIN: int = 12
DIP_MIN_PCT: float = 0.0015
DIP_RATE_MIN_BPS_PER_MIN: float = 4.0
REV_MIN_UP_CANDLES: int = 2
REV_RECLAIM_BPS: float = 8.0
REQUIRE_HIGHER_LOW_CONFIRMATION: bool = True
REQUIRE_MICRO_VWAP_RECLAIM: bool = True
MICRO_TREND_LOOKBACK_MIN: int = 15
MICRO_TREND_DOWN_BPS: float = -18.0
VWAP_RECLAIM_BUFFER_BPS: float = 3.0

# Tier score bands
TIER_LOW = 1
TIER_MID = 2
TIER_HIGH = 3

TIER_SCORE_BANDS = {
    TIER_LOW: (48.0, 63.9999),
    TIER_MID: (64.0, 79.9999),
    TIER_HIGH: (80.0, 100.0),
}

# Support / room / regime
SUPPORT_BUFFER_BPS: float = 30.0
RESIST_BUFFER_BPS: float = 10.0
WEEKLY_BIAS_THRESHOLD: float = -0.70

# Score weights
SCORE_DIP_DEPTH_W: float = 22.0
SCORE_DIP_SPEED_W: float = 14.0
SCORE_REVERSAL_W: float = 18.0
SCORE_SUPPORT_W: float = 12.0
SCORE_ROOM_W: float = 16.0
SCORE_REGIME_W: float = 6.0
SCORE_SPREAD_PENALTY_W: float = 10.0
SCORE_COST_PENALTY_W: float = 10.0

# Exit plan by tier
EXIT_PLAN = {
    TIER_LOW:  {"scalp_frac": 1.00, "core_frac": 0.00, "runner_frac": 0.00},
    TIER_MID:  {"scalp_frac": 0.65, "core_frac": 0.35, "runner_frac": 0.00},
    TIER_HIGH: {"scalp_frac": 0.35, "core_frac": 0.40, "runner_frac": 0.25},
}

SCALP_SIGMA_MULT = {
    TIER_LOW: 0.55,
    TIER_MID: 0.75,
    TIER_HIGH: 0.95,
}

CORE_SIGMA_MULT = {
    TIER_LOW: 0.90,
    TIER_MID: 1.20,
    TIER_HIGH: 1.65,
}

# Protective exits
TRAIL_ARM_PCT: float = 0.0065
TRAIL_DRAWDOWN_PCT: float = 0.0020
HARD_PEAK_STOP_PCT: float = 0.0040

# Armed target exits.
# Targets do not immediately sell. They arm a trailing release.
SCALP_TARGET_ARM_DRAWDOWN_PCT: float = 0.0010  # 0.10%
CORE_TARGET_ARM_DRAWDOWN_PCT: float = 0.0020   # 0.20%
RUNNER_TARGET_ARM_DRAWDOWN_PCT: float = 0.0030 # 0.30%

# Walk-forward calibration
ENABLE_WALK_FORWARD_CALIBRATION: bool = True

# Past day calibration: 1-minute candles.
CALIB_DAY_LOOKBACK_MINUTES: int = 24 * 60
CALIB_DAY_GRANULARITY: str = "ONE_MINUTE"

# Past week calibration: 15-minute candles.
CALIB_WEEK_LOOKBACK_MINUTES: int = 7 * 24 * 60
CALIB_WEEK_GRANULARITY: str = "FIFTEEN_MINUTE"

# Minimum candle history needed before scoring a historical moment.
CALIB_MIN_PREFIX_CANDLES_1M: int = 90
CALIB_MIN_PREFIX_CANDLES_15M: int = 32

# Future windows used to judge whether the signal worked.
CALIB_FORWARD_MINUTES_1M: int = 60
CALIB_FORWARD_BARS_15M: int = 16

# Minimum product history required before calibration.
CALIB_MIN_PRODUCT_SAMPLES: int = 25

# Exact calibration search.
# Do not round score/probability targets into buckets.
# The chosen target should come from actual observed walk-forward values.
CALIB_USE_EXACT_THRESHOLDS: bool = True

# To avoid overfitting, each candidate threshold must have enough historical samples.
CALIB_EXACT_MIN_SAMPLES: int = 12

# Candidate pool limits keep startup fast without rounding the final chosen value.
CALIB_MAX_EXACT_SCORE_CANDIDATES: int = 80
CALIB_MAX_EXACT_PROB_CANDIDATES: int = 80

# Minimum acceptable historical performance for a buy threshold.
CALIB_MIN_WIN_RATE: float = 0.54
CALIB_MIN_EXPECTED_VALUE_BPS: float = 2.0

# Fallback behavior if no threshold passes.
# These are only fallbacks, not floors.
DEFAULT_CALIB_MIN_SCORE: float = 60.0
DEFAULT_CALIB_MIN_PROB: float = 0.58
DEFAULT_CALIB_MIN_EV_BPS: float = 2.0

# Calibration repair:
# These are only emergency safety floors, not default targets.
# They prevent absurd values while preserving product-specific calibration.
CALIB_ABSOLUTE_MIN_SCORE: float = 20.0
CALIB_ABSOLUTE_MIN_PROB: float = 0.20

# If fallback uses winning observations, use these quantiles from actual winners.
# This keeps targets based on setups that reached the minimum gain after fees.
CALIB_WINNER_SCORE_QUANTILE: float = 0.55
CALIB_WINNER_PROB_QUANTILE: float = 0.55

# If no exact positive-EV threshold is found, still choose product-specific
# thresholds from historical winners instead of defaulting every product.
ALLOW_WINNER_BASED_FALLBACK_THRESHOLDS: bool = True

# Live recalibration can be CPU-heavy. Do it less often and off the event loop.
LIVE_RECALIBRATION_EVERY_SEC: float = 5 * 60
LIVE_RECALIBRATION_MIN_ROWS: int = 240

# Event loop lag diagnostics.
EVENT_LOOP_LAG_WARN_SEC: float = 3.0

# Account snapshot cache for telemetry.
TELEMETRY_ACCOUNT_REFRESH_TTL_SEC: float = 5.0

# Candidate pullbacks tested during sell calibration.
CALIB_SCALP_PULLBACK_CANDIDATES: List[float] = [0.0005, 0.0010, 0.0015, 0.0020]
CALIB_CORE_PULLBACK_CANDIDATES: List[float] = [0.0010, 0.0020, 0.0030, 0.0040]

# Keep calibrated values inside safe bounds.
CALIB_MIN_SCALP_PULLBACK: float = 0.0005
CALIB_MAX_SCALP_PULLBACK: float = 0.0030
CALIB_MIN_CORE_PULLBACK: float = 0.0010
CALIB_MAX_CORE_PULLBACK: float = 0.0060

# Time-based exits are disabled for this strategy style.
ENABLE_NO_PROGRESS_STOP: bool = False
ENABLE_TIME_STOP: bool = False

# Keep structure/risk exits enabled.
ENABLE_INVALIDATION_STOP: bool = True
ENABLE_HARD_PEAK_STOP: bool = True

# Faster invalidation/time exits for day-trading/scalping behavior
TIME_STOP_SEC: int = 45 * 60
NO_PROGRESS_STOP_SEC: int = 12 * 60
MIN_PROGRESS_BPS_BEFORE_TIME_STOP: float = 8.0
INVALIDATION_BUFFER_BPS: float = 10.0

# Risk brakes
MAX_DAILY_LOSS_USD: float = 1.50
MAX_CONSECUTIVE_LOSSES: int = 2
COOLDOWN_AFTER_LOSS_SEC: float = 15 * 60
MAX_TRADES_PER_HOUR: int = 4
MAX_TRADES_PER_PRODUCT_PER_HOUR: int = 1
PAUSE_AFTER_DAILY_LOSS_SEC: float = 6 * 60 * 60

# Scaling
ALLOW_SCALE_INTO_WINNERS: bool = True
MAX_SCALE_ADDS_PER_POSITION: int = 1
SCALE_ONLY_IF_UNREALIZED_NET_BPS_ABOVE: float = 22.0
SCALE_ADD_FRACTION_OF_ENTRY: float = 0.50

# Safety exits
RISK_OFF_REDUCTION_FRAC: float = 0.05
RISK_OFF_COOLDOWN_SEC: float = 60.0
RISK_OFF_MIN_NOTIONAL_USD: float = 1.0

# Universe / selection
AUTO_SELECT_PRODUCTS: bool = False
TARGET_PRODUCT_COUNT: int = 8
CANDIDATE_TOP_BY_USD_VOL: int = 60
MIN_DAILY_RANGE_PCT: float = 0.06

# Fair-value smoothing
FAIR_VALUE_MEDIAN_WINDOW: int = 9
FAIR_VALUE_SMOOTH_ALPHA: float = 0.18
FAIR_VALUE_MAX_STEP_BPS: float = 24.0
FAIR_VALUE_MAX_STEP_DOWN_BPS: float = 16.0

# Sell floor
MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT: float = 8.0

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

def _rotate_debug_log_if_needed() -> None:
    """Rotate an oversized debug log without ever interrupting trading."""
    if not DEBUG_LOG_ENABLED:
        return

    try:
        if not os.path.exists(DEBUG_LOG_PATH):
            return
        if os.path.getsize(DEBUG_LOG_PATH) <= int(DEBUG_LOG_MAX_BYTES):
            return

        old_path = DEBUG_LOG_PATH + ".old"
        try:
            if os.path.exists(old_path):
                os.remove(old_path)
        except Exception:
            pass
        os.replace(DEBUG_LOG_PATH, old_path)
    except Exception:
        # Never let logging break trading.
        pass


def log(msg: str) -> None:
    """Write a timestamped bot log line to both the terminal and debug.log."""
    try:
        ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except Exception:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    line = f"{ts} {msg}"
    print(line, flush=True)

    if not DEBUG_LOG_ENABLED:
        return

    try:
        _rotate_debug_log_if_needed()
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Logging should never crash the bot.
        pass


def log_exception(context: str, exc: Exception) -> None:
    """Write a compact exception line to debug.log and the terminal."""
    log(f"[exception] {context}: {type(exc).__name__}: {exc}")

def now_ts() -> float:
    """Return current UNIX timestamp as float."""
    return time.time()


def now_ts_i() -> int:
    """Return current UNIX timestamp as int seconds."""
    return int(time.time())


def bps_to_mult(bps: float) -> float:
    """Convert basis points to multiplicative factor (e.g. 50 bps → 1.005)."""
    return 1.0 + (bps / 10_000.0)


def clamp_float(x: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(x))))


def lerp_float(a: float, b: float, t: float) -> float:
    return float(float(a) + (float(b) - float(a)) * float(t))


def fee_usd(notional_usd: float, fee_bps: float) -> float:
    """Return fee in USD for a given notional and fee rate."""
    return notional_usd * (fee_bps / 10_000.0)


def product_base_asset(product_id: str) -> str:
    """
    Convert a Coinbase product like 'BTC-USD' into its base asset, e.g. 'BTC'.
    """
    try:
        return str(product_id).split("-")[0].strip().upper()
    except Exception:
        return ""


def safe_float(x: Any) -> Optional[float]:
    """Convert to float if possible, else None."""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into the inclusive range [lo, hi]."""
    return float(max(lo, min(hi, x)))


def can_exit_net_positive(
    *,
    entry_price: float,
    exit_price: float,
    taker_fee_bps: float,
    est_slippage_bps: float,
    est_adverse_fill_bps: float,
    min_net_profit_bps: float = 0.0,
) -> bool:
    """
    Estimate whether an exit is net-positive after exit-side costs.

    entry_price should already include buy-side fee when using PositionLot.price.
    That means the buy fee is already embedded in cost basis.

    This check estimates:
    exit proceeds after exit fee/slippage/adverse buffer
    versus effective entry cost basis.
    """
    if entry_price <= 0 or exit_price <= 0:
        return False

    exit_cost_bps = (
        float(taker_fee_bps)
        + float(est_slippage_bps)
        + float(est_adverse_fill_bps)
    )

    gross_move_bps = ((float(exit_price) / float(entry_price)) - 1.0) * 10000.0
    estimated_net_bps = gross_move_bps - exit_cost_bps

    return estimated_net_bps >= float(min_net_profit_bps)


def required_exit_price_for_net_gain(
    *,
    effective_entry_price: float,
    exit_fee_bps: float,
    est_slippage_bps: float,
    est_adverse_fill_bps: float,
    min_net_gain_bps: float,
) -> float:
    """
    Calculate the minimum exit price needed to clear a desired net gain.

    effective_entry_price should include the buy-side fee.
    """
    total_required_bps = (
        float(exit_fee_bps)
        + float(est_slippage_bps)
        + float(est_adverse_fill_bps)
        + float(min_net_gain_bps)
    )
    return float(effective_entry_price) * bps_to_mult(total_required_bps)

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
        log("[selection] strict daily-range filter returned no products; using fallback candidate list")

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
        # Finalise previous synthetic activity candle. Replace a seeded candle
        # for the same minute so startup history transitions cleanly into live data.
        closed_candle = MinuteCandle(
            minute_start_ts=self._cur_minute,
            open=float(self._o),
            high=float(self._h),
            low=float(self._l),
            close=float(self._c),
            volume=float(self._v),
        )
        if self.candles and self.candles[-1].minute_start_ts == self._cur_minute:
            self.candles[-1] = closed_candle
        else:
            self.candles.append(closed_candle)
        # Start new synthetic activity candle
        self._cur_minute = m
        self._o = self._h = self._l = self._c = mid
        self._v = 1.0

    def append_closed_candle(
        self,
        *,
        minute_start_ts: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0.0,
    ) -> None:
        """Seed a fully formed historical 1-minute candle with true OHLCV."""
        if close_price <= 0:
            return

        candle = MinuteCandle(
            minute_start_ts=int(minute_start_ts),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(volume or 0.0),
        )

        if self.candles and int(self.candles[-1].minute_start_ts) == int(minute_start_ts):
            self.candles[-1] = candle
        else:
            self.candles.append(candle)

        # Keep the current-minute builder aligned after the seeded history.
        self._cur_minute = int(minute_start_ts)
        self._o = float(open_price)
        self._h = float(high_price)
        self._l = float(low_price)
        self._c = float(close_price)
        self._v = float(volume or 1.0)

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
                "entry_price", "exit_price", "weekly_bias",
                "entry_score", "entry_tier", "expected_net_edge_bps", "exit_role", "note"
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
        expected_net_edge_bps: Optional[float] = None,
        exit_role: str = "",
    ) -> None:
        qty_val = float(qty)
        price_val = float(price)
        notional = float(filled_notional_usd) if filled_notional_usd is not None else (qty_val * price_val)

        # Final defensive normalization:
        # If qty looks wrong (for example, quote/notional USD was passed into qty),
        # repair it from notional / price so trades.csv always stores base quantity.
        if price_val > 0 and notional > 0:
            expected_qty = notional / price_val
            if qty_val <= 0:
                qty_val = expected_qty
            else:
                rel_err = abs(qty_val - expected_qty) / max(expected_qty, 1e-12)
                if rel_err > 0.10:
                    qty_val = expected_qty

        self.cum_pnl_usd += float(net_pnl_usd)

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            tsv = now_ts()
            dt_mst = datetime.fromtimestamp(tsv, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([
                f"{tsv:.6f}", dt_mst, event, product_id, side,
                f"{qty_val:.10f}", f"{price_val:.10f}", f"{notional:.10f}",
                f"{float(fee_usd_val):.10f}", f"{float(gross_pnl_usd):.10f}", f"{float(net_pnl_usd):.10f}",
                f"{self.cum_pnl_usd:.10f}",
                "" if entry_price is None else f"{float(entry_price):.10f}",
                "" if exit_price is None else f"{float(exit_price):.10f}",
                "" if weekly_bias is None else f"{float(weekly_bias):.6f}",
                "" if entry_score is None else f"{float(entry_score):.6f}",
                "" if entry_tier is None else str(entry_tier),
                "" if expected_net_edge_bps is None else f"{float(expected_net_edge_bps):.6f}",
                exit_role,
                note,
            ])


class OrderLogger:
    """
    Writes every order attempt to orders.csv.
    This is separate from trades.csv.

    trades.csv = confirmed fills only.
    orders.csv = attempts, failed buys, no-fills, rejects, errors, partials.
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
                "ts", "dt_mst", "event", "product_id", "side", "mode",
                "requested_quote_usd", "requested_base_qty",
                "ok", "status", "order_id", "client_order_id",
                "filled_qty", "avg_price", "filled_notional_usd", "fee_usd",
                "reason", "raw_error"
            ])

    def log_order(
        self,
        *,
        event: str,
        product_id: str,
        side: str,
        mode: str,
        requested_quote_usd: Optional[float] = None,
        requested_base_qty: Optional[float] = None,
        result: Optional[Any] = None,
        reason: str = "",
        raw_error: str = "",
    ) -> None:
        tsv = now_ts()
        dt_mst = datetime.fromtimestamp(tsv, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")

        ok = False
        status = ""
        order_id = ""
        client_order_id = ""
        filled_qty = 0.0
        avg_price = ""
        filled_notional = ""
        fee_usd_val = 0.0
        err = raw_error or ""

        try:
            d = result.to_dict() if hasattr(result, "to_dict") else result
            if isinstance(d, dict):
                ok = bool(d.get("ok", False))
                status = str(d.get("status", ""))
                order_id = str(d.get("order_id") or "")
                client_order_id = str(d.get("client_order_id") or "")
                filled_qty = float(d.get("filled_qty") or 0.0)
                if d.get("avg_price") is not None:
                    avg_price = f"{float(d.get('avg_price')):.10f}"
                if d.get("filled_notional_usd") is not None:
                    filled_notional = f"{float(d.get('filled_notional_usd')):.10f}"
                fee_usd_val = float(d.get("fee_usd") or 0.0)
                if d.get("error"):
                    err = str(d.get("error"))
        except Exception as e:
            err = err or str(e)

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                f"{tsv:.6f}", dt_mst, event, product_id, side, mode,
                "" if requested_quote_usd is None else f"{float(requested_quote_usd):.10f}",
                "" if requested_base_qty is None else f"{float(requested_base_qty):.10f}",
                str(bool(ok)), status, order_id, client_order_id,
                f"{float(filled_qty):.10f}",
                avg_price,
                filled_notional,
                f"{float(fee_usd_val):.10f}",
                reason,
                err,
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
                "estimated_prob_up", "position_pct",
                "target_bps", "projected_forward_gain_bps", "cost_bps",
                "calibrated_time_to_min_profit_minutes", "calibrated_forward_window_minutes",
                "current_maker_fee_bps", "current_taker_fee_bps", "fee_tier_reason",
                "dip_depth_score", "dip_speed_score", "reversal_score", "support_score",
                "room_score", "regime_score", "spread_penalty", "cost_penalty",
                "buy_gate_score_ok", "buy_gate_prob_ok", "buy_gate_ev_ok",
                "buy_gate_fee_ok", "buy_gate_strict_ok", "buy_gate_target_cost_ok",
                "buy_gate_spread_ok", "buy_gate_calibrated_ok",
                "buy_gate_tradeable", "buy_gate_blocker"
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
        estimated_prob_up: Optional[float] = None,
        position_pct: Optional[float] = None,
        target_bps: Optional[float] = None,
        projected_forward_gain_bps: Optional[float] = None,
        cost_bps: Optional[float] = None,
        calibrated_time_to_min_profit_minutes: Optional[float] = None,
        calibrated_forward_window_minutes: Optional[float] = None,
        current_maker_fee_bps: Optional[float] = None,
        current_taker_fee_bps: Optional[float] = None,
        fee_tier_reason: str = "",
        dip_depth_score: Optional[float] = None,
        dip_speed_score: Optional[float] = None,
        reversal_score: Optional[float] = None,
        support_score: Optional[float] = None,
        room_score: Optional[float] = None,
        regime_score: Optional[float] = None,
        spread_penalty: Optional[float] = None,
        cost_penalty: Optional[float] = None,
        buy_gate_score_ok: Optional[bool] = None,
        buy_gate_prob_ok: Optional[bool] = None,
        buy_gate_ev_ok: Optional[bool] = None,
        buy_gate_fee_ok: Optional[bool] = None,
        buy_gate_strict_ok: Optional[bool] = None,
        buy_gate_target_cost_ok: Optional[bool] = None,
        buy_gate_spread_ok: Optional[bool] = None,
        buy_gate_calibrated_ok: Optional[bool] = None,
        buy_gate_tradeable: Optional[bool] = None,
        buy_gate_blocker: str = "",
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
                "" if estimated_prob_up is None else f"{estimated_prob_up:.6f}",
                "" if position_pct is None else f"{position_pct:.6f}",
                "" if target_bps is None else f"{target_bps:.6f}",
                "" if projected_forward_gain_bps is None else f"{projected_forward_gain_bps:.6f}",
                "" if cost_bps is None else f"{cost_bps:.6f}",
                "" if calibrated_time_to_min_profit_minutes is None else f"{calibrated_time_to_min_profit_minutes:.6f}",
                "" if calibrated_forward_window_minutes is None else f"{calibrated_forward_window_minutes:.6f}",
                "" if current_maker_fee_bps is None else f"{current_maker_fee_bps:.6f}",
                "" if current_taker_fee_bps is None else f"{current_taker_fee_bps:.6f}",
                fee_tier_reason,
                "" if dip_depth_score is None else f"{dip_depth_score:.6f}",
                "" if dip_speed_score is None else f"{dip_speed_score:.6f}",
                "" if reversal_score is None else f"{reversal_score:.6f}",
                "" if support_score is None else f"{support_score:.6f}",
                "" if room_score is None else f"{room_score:.6f}",
                "" if regime_score is None else f"{regime_score:.6f}",
                "" if spread_penalty is None else f"{spread_penalty:.6f}",
                "" if cost_penalty is None else f"{cost_penalty:.6f}",
                "" if buy_gate_score_ok is None else str(bool(buy_gate_score_ok)),
                "" if buy_gate_prob_ok is None else str(bool(buy_gate_prob_ok)),
                "" if buy_gate_ev_ok is None else str(bool(buy_gate_ev_ok)),
                "" if buy_gate_fee_ok is None else str(bool(buy_gate_fee_ok)),
                "" if buy_gate_strict_ok is None else str(bool(buy_gate_strict_ok)),
                "" if buy_gate_target_cost_ok is None else str(bool(buy_gate_target_cost_ok)),
                "" if buy_gate_spread_ok is None else str(bool(buy_gate_spread_ok)),
                "" if buy_gate_calibrated_ok is None else str(bool(buy_gate_calibrated_ok)),
                "" if buy_gate_tradeable is None else str(bool(buy_gate_tradeable)),
                buy_gate_blocker,
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
            log(f"[macro] get_candles {product_id} {granularity} start={int(start)} end={int(end)} span={(int(end)-int(start))}")
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
            log_exception(f"[macro] fetch failed for {product_id} {granularity}", e)
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


@dataclass
class LiveSignal:
    """
    Continuous display/trading signal.

    Unlike EntryScore, this should not collapse to zero just because the exact
    dip-entry pattern is not present. This is the always-on signal that powers
    the viewer overview, projected buy sizing, and candidate ranking.
    """
    ok_to_trade: bool
    score: float
    tier: int
    reason: str
    estimated_prob_up: float
    position_pct: float
    expected_net_edge_bps: float
    target_bps: float
    cost_bps: float
    projected_forward_gain_bps: float = 0.0
    calibrated_time_to_min_profit_minutes: float = 0.0
    calibrated_forward_window_minutes: float = 0.0
    dip_depth_score: float = 0.0
    dip_speed_score: float = 0.0
    reversal_score: float = 0.0
    support_score: float = 0.0
    room_score: float = 0.0
    regime_score: float = 0.0
    spread_penalty: float = 0.0
    cost_penalty: float = 0.0
    trend_reason: str = ""
    vwap_reason: str = ""
    higher_low_reason: str = ""
    buy_gate_score_ok: bool = False
    buy_gate_prob_ok: bool = False
    buy_gate_ev_ok: bool = False
    buy_gate_fee_ok: bool = False
    buy_gate_strict_ok: bool = False
    buy_gate_target_cost_ok: bool = False
    buy_gate_spread_ok: bool = False
    buy_gate_calibrated_ok: bool = False
    buy_gate_tradeable: bool = False
    buy_gate_blocker: str = ""


@dataclass
class CalibrationObservation:
    product_id: str
    timeframe: str
    ts: int
    score: float
    probability: float
    expected_net_edge_bps: float
    target_bps: float
    cost_bps: float
    spread_bps: float
    max_favorable_bps: float
    max_adverse_bps: float
    reached_min_profit: bool
    reached_target: bool
    expected_value_bps: float
    win_bps: float
    loss_bps: float

    # Forward-projection measurements.
    time_to_min_profit_bars: Optional[int] = None
    time_to_min_profit_minutes: Optional[float] = None
    forward_window_minutes: Optional[float] = None
    projected_forward_gain_bps: float = 0.0


@dataclass
class ProductCalibrationProfile:
    product_id: str
    min_score: float = DEFAULT_CALIB_MIN_SCORE
    min_probability: float = DEFAULT_CALIB_MIN_PROB
    min_expected_value_bps: float = DEFAULT_CALIB_MIN_EV_BPS
    scalp_pullback_pct: float = SCALP_TARGET_ARM_DRAWDOWN_PCT
    core_pullback_pct: float = CORE_TARGET_ARM_DRAWDOWN_PCT
    day_sample_count: int = 0
    week_sample_count: int = 0
    day_win_rate: float = 0.0
    week_win_rate: float = 0.0
    blended_win_rate: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    expected_value_bps: float = 0.0

    # Calibrated forward-projection values.
    calibrated_projected_gross_bps: float = 0.0
    calibrated_projected_net_bps: float = 0.0
    calibrated_time_to_min_profit_minutes: float = 0.0
    calibrated_forward_window_minutes: float = 0.0

    reason: str = "default_profile"


class CalibrationLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "dt_mst", "product_id",
                "min_score", "min_probability", "min_expected_value_bps",
                "scalp_pullback_pct", "core_pullback_pct",
                "day_sample_count", "week_sample_count",
                "day_win_rate", "week_win_rate", "blended_win_rate",
                "avg_win_bps", "avg_loss_bps", "expected_value_bps",
                "calibrated_projected_gross_bps",
                "calibrated_projected_net_bps",
                "calibrated_time_to_min_profit_minutes",
                "calibrated_forward_window_minutes",
                "reason",
            ])

    def log_profile(self, profile: ProductCalibrationProfile) -> None:
        tsv = now_ts()
        dt_mst = datetime.fromtimestamp(tsv, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                f"{tsv:.6f}", dt_mst, profile.product_id,
                f"{profile.min_score:.6f}",
                f"{profile.min_probability:.6f}",
                f"{profile.min_expected_value_bps:.6f}",
                f"{profile.scalp_pullback_pct:.8f}",
                f"{profile.core_pullback_pct:.8f}",
                profile.day_sample_count, profile.week_sample_count,
                f"{profile.day_win_rate:.6f}",
                f"{profile.week_win_rate:.6f}",
                f"{profile.blended_win_rate:.6f}",
                f"{profile.avg_win_bps:.6f}",
                f"{profile.avg_loss_bps:.6f}",
                f"{profile.expected_value_bps:.6f}",
                f"{profile.calibrated_projected_gross_bps:.6f}",
                f"{profile.calibrated_projected_net_bps:.6f}",
                f"{profile.calibrated_time_to_min_profit_minutes:.6f}",
                f"{profile.calibrated_forward_window_minutes:.6f}",
                profile.reason,
            ])


class MicroHistoryLogger:
    """Atomically writes startup 1-minute historical candles for the viewer."""

    columns = [
        "ts", "dt_mst", "product_id", "open", "high", "low", "close", "volume",
    ]

    def __init__(self, path: str) -> None:
        self.path = path

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.columns)
            for r in rows:
                ts_val = float(r.get("ts", 0.0) or 0.0)
                dt_mst = datetime.fromtimestamp(ts_val, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
                w.writerow([
                    int(ts_val),
                    dt_mst,
                    r.get("product_id", ""),
                    f"{float(r.get('open', 0.0)):.10f}",
                    f"{float(r.get('high', 0.0)):.10f}",
                    f"{float(r.get('low', 0.0)):.10f}",
                    f"{float(r.get('close', 0.0)):.10f}",
                    f"{float(r.get('volume', 0.0)):.10f}",
                ])
        os.replace(tmp, self.path)


class PositionTargetsLogger:
    """Atomically writes the current open-position sell plan for the viewer."""

    columns = [
        "ts", "dt_mst", "product_id",
        "has_position", "position_qty", "avg_entry_price",
        "current_bid", "current_ask",
        "min_profitable_exit_price",
        "scalp_target_price", "core_target_price",
        "scalp_armed", "core_armed",
        "scalp_arm_peak", "core_arm_peak",
        "scalp_pullback_pct", "core_pullback_pct",
        "scalp_pullback_trigger_price", "core_pullback_trigger_price",
        "distance_to_min_profit_bps",
        "distance_to_scalp_bps",
        "distance_to_core_bps",
        "exit_plan_note",
    ]

    def __init__(self, path: str) -> None:
        self.path = path

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.columns)
            for r in rows:
                ts_val = float(r.get("ts", now_ts()) or now_ts())
                dt_mst = datetime.fromtimestamp(ts_val, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
                w.writerow([
                    f"{ts_val:.6f}",
                    dt_mst,
                    r.get("product_id", ""),
                    bool(r.get("has_position", False)),
                    f"{float(r.get('position_qty', 0.0)):.12f}",
                    "" if r.get("avg_entry_price") is None else f"{float(r.get('avg_entry_price')):.10f}",
                    "" if r.get("current_bid") is None else f"{float(r.get('current_bid')):.10f}",
                    "" if r.get("current_ask") is None else f"{float(r.get('current_ask')):.10f}",
                    "" if r.get("min_profitable_exit_price") is None else f"{float(r.get('min_profitable_exit_price')):.10f}",
                    "" if r.get("scalp_target_price") is None else f"{float(r.get('scalp_target_price')):.10f}",
                    "" if r.get("core_target_price") is None else f"{float(r.get('core_target_price')):.10f}",
                    bool(r.get("scalp_armed", False)),
                    bool(r.get("core_armed", False)),
                    "" if r.get("scalp_arm_peak") is None else f"{float(r.get('scalp_arm_peak')):.10f}",
                    "" if r.get("core_arm_peak") is None else f"{float(r.get('core_arm_peak')):.10f}",
                    f"{float(r.get('scalp_pullback_pct', 0.0)):.8f}",
                    f"{float(r.get('core_pullback_pct', 0.0)):.8f}",
                    "" if r.get("scalp_pullback_trigger_price") is None else f"{float(r.get('scalp_pullback_trigger_price')):.10f}",
                    "" if r.get("core_pullback_trigger_price") is None else f"{float(r.get('core_pullback_trigger_price')):.10f}",
                    "" if r.get("distance_to_min_profit_bps") is None else f"{float(r.get('distance_to_min_profit_bps')):.6f}",
                    "" if r.get("distance_to_scalp_bps") is None else f"{float(r.get('distance_to_scalp_bps')):.6f}",
                    "" if r.get("distance_to_core_bps") is None else f"{float(r.get('distance_to_core_bps')):.6f}",
                    r.get("exit_plan_note", ""),
                ])
        os.replace(tmp, self.path)


def _clip_score(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def _score_from_bps(value_bps: float, center_bps: float = 0.0, width_bps: float = 50.0) -> float:
    """
    Convert a bps value into a 0-100 score.
    Around center_bps = 50.
    Higher is better.
    """
    if width_bps <= 0:
        width_bps = 50.0
    return _clip_score(50.0 + ((float(value_bps) - float(center_bps)) / float(width_bps)) * 50.0)


def _recent_close_momentum_bps(minute_candles: List['MinuteCandle'], lookback: int = 5) -> float:
    """
    Recent close-to-close move in basis points.
    Positive = short-term upward pressure.
    """
    if not minute_candles or len(minute_candles) < 2:
        return 0.0

    candles = list(minute_candles)
    lookback = max(1, min(int(lookback), len(candles) - 1))
    first = float(candles[-lookback - 1].close)
    last = float(candles[-1].close)

    if first <= 0 or last <= 0:
        return 0.0

    return float((last / first - 1.0) * 10000.0)


def _recent_range_position_score(minute_candles: List['MinuteCandle'], lookback: int = 20) -> float:
    """
    Score where the latest close sits inside the recent range.
    0 = near range low.
    100 = near range high.
    """
    if not minute_candles or len(minute_candles) < 3:
        return 50.0

    candles = list(minute_candles)[-int(lookback):]
    lows = [float(c.low) for c in candles if float(c.low) > 0]
    highs = [float(c.high) for c in candles if float(c.high) > 0]
    close = float(candles[-1].close)

    if not lows or not highs or close <= 0:
        return 50.0

    lo = min(lows)
    hi = max(highs)

    if hi <= lo:
        return 50.0

    return _clip_score(((close - lo) / (hi - lo)) * 100.0)


def _score_to_tier(score: float) -> int:
    if score >= TIER_SCORE_BANDS[TIER_HIGH][0]:
        return TIER_HIGH
    if score >= TIER_SCORE_BANDS[TIER_MID][0]:
        return TIER_MID
    if score >= TIER_SCORE_BANDS[TIER_LOW][0]:
        return TIER_LOW
    return 0


def _support_proximity_score(mid: float, day: Optional['MacroLevels'], week: Optional['MacroLevels']) -> float:
    if mid <= 0:
        return 0.0

    vals = []
    for lv in (day, week):
        if not lv:
            continue

        if getattr(lv, "support_zone_low", 0) > 0 and getattr(lv, "support_zone_high", 0) > 0:
            if lv.support_zone_low <= mid <= lv.support_zone_high:
                return 100.0
            zone_mid = (lv.support_zone_low + lv.support_zone_high) / 2.0
            dist_pct = abs(mid - zone_mid) / mid
            vals.append(max(0.0, 100.0 - dist_pct * 10000.0 * 3.0))

        if getattr(lv, "prev_low", 0) > 0:
            dist_pct = abs(mid - lv.prev_low) / mid
            vals.append(max(0.0, 100.0 - dist_pct * 10000.0 * 4.0))

        if getattr(lv, "val", 0) > 0:
            dist_pct = abs(mid - lv.val) / mid
            vals.append(max(0.0, 100.0 - dist_pct * 10000.0 * 4.0))

    return float(max(vals)) if vals else 0.0


def _room_score(mid: float, day: Optional['MacroLevels'], week: Optional['MacroLevels'], resist_buffer_bps: float) -> Tuple[float, str]:
    ok, reason = option1_room_to_target(mid, day, week, resist_buffer_bps)
    if ok:
        return 100.0, reason

    ok2, reason2 = _room_to_target_pct(
        mid, day, week,
        target_pct=0.0035,
        resist_buffer_bps=resist_buffer_bps,
    )
    if ok2:
        return 60.0, reason2

    return 0.0, reason


def _estimate_net_edge_bps(
    score_room: float,
    spread_bps: float,
    tier_hint: int,
    round_trip_cost_bps: float,
) -> float:
    """
    Estimate edge using a caller-provided round-trip cost.
    The caller must pass a cost built from real Coinbase fee-tier values.
    """
    gross_target_bps = {
        TIER_LOW: 24.0,
        TIER_MID: 45.0,
        TIER_HIGH: 85.0,
    }.get(tier_hint, 24.0)

    room_bonus = (score_room / 100.0) * 12.0
    return float(gross_target_bps + room_bonus - float(round_trip_cost_bps))

def score_entry_candidate(
    *,
    mid: float,
    spread_bps: float,
    levels_day: Optional['MacroLevels'],
    levels_week: Optional['MacroLevels'],
    minute_candles: List['MinuteCandle'],
    weekly_bias: Optional[float],
    trending_down: bool,
    resist_buffer_bps: float,
    round_trip_cost_bps: float,
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

    dip_depth_score = _clip_score((dip_pct / 0.0100) * 100.0)
    dip_speed_score = _clip_score((dip_rate / max(DIP_RATE_MIN_BPS_PER_MIN, 1e-9)) * 50.0)
    support_score = _support_proximity_score(mid, levels_day, levels_week)
    room_score, room_reason = _room_score(mid, levels_day, levels_week, resist_buffer_bps)

    if weekly_bias is None:
        regime_score = 55.0
    else:
        regime_score = _clip_score((weekly_bias + 1.0) * 50.0)

    if trending_down:
        regime_score = min(regime_score, 30.0)

    spread_penalty = max(0.0, spread_bps - 6.0) * (SCORE_SPREAD_PENALTY_W / 20.0)
    cost_penalty = float(round_trip_cost_bps) * (SCORE_COST_PENALTY_W / 25.0)

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
    edge_bps = _estimate_net_edge_bps(
        room_score,
        spread_bps,
        max(tier, TIER_LOW),
        round_trip_cost_bps=float(round_trip_cost_bps),
    )

    if not rev_ok:
        return EntryScore(False, final_score, tier, f"reversal_fail {rev_reason}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, edge_bps)

    if support_score <= 0.0:
        return EntryScore(False, final_score, tier, "support_fail", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, edge_bps)

    if room_score <= 0.0:
        return EntryScore(False, final_score, tier, f"room_fail {room_reason}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, edge_bps)

    if spread_bps > MAX_SPREAD_BPS:
        return EntryScore(False, final_score, tier, f"spread_high={spread_bps:.1f}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, edge_bps)

    if tier == TIER_LOW and spread_bps > SCALP_MAX_SPREAD_BPS:
        return EntryScore(False, final_score, tier, f"spread_high_low_tier={spread_bps:.1f}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, edge_bps)

    if final_score < TIER_SCORE_BANDS[TIER_LOW][0]:
        return EntryScore(False, final_score, 0, f"score_too_low={final_score:.1f}", dip_depth_score, dip_speed_score, reversal_score, support_score, room_score, regime_score, spread_penalty, cost_penalty, edge_bps)

    if edge_bps < MIN_REQUIRED_NET_EDGE_BPS:
        return EntryScore(
            False,
            final_score,
            0,
            f"net_edge_too_low={edge_bps:.1f}<required={MIN_REQUIRED_NET_EDGE_BPS:.1f}",
            dip_depth_score,
            dip_speed_score,
            reversal_score,
            support_score,
            room_score,
            regime_score,
            spread_penalty,
            cost_penalty,
            edge_bps,
        )

    return EntryScore(
        True,
        final_score,
        tier,
        f"score_ok={final_score:.1f} room={room_reason} edge_bps={edge_bps:.1f}",
        dip_depth_score,
        dip_speed_score,
        reversal_score,
        support_score,
        room_score,
        regime_score,
        spread_penalty,
        cost_penalty,
        edge_bps,
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
    round_trip_cost_bps: float,
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
        round_trip_cost_bps=round_trip_cost_bps,
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

    def get_available_asset(self, asset: str, snapshot: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """
        Return available balance for an asset from Coinbase.
        Use this for sell sizing because held/locked balances may not be tradable.
        """
        snap = snapshot if snapshot is not None else self.refresh_snapshot()
        d = snap.get(str(asset).upper(), {})
        return float(max(0.0, float(d.get("available", 0.0))))

    def get_product_total_qty(self, product_id: str, snapshot: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """
        Return Coinbase total balance for the base asset in a product like BTC-USD.
        """
        asset = product_base_asset(product_id)
        if not asset:
            return 0.0
        return self.get_total_asset(asset, snapshot=snapshot)

    def get_product_available_qty(self, product_id: str, snapshot: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """
        Return Coinbase available/sellable balance for the base asset in a product like BTC-USD.
        """
        asset = product_base_asset(product_id)
        if not asset:
            return 0.0
        return self.get_available_asset(asset, snapshot=snapshot)

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

    def get_fee_tier_bps(self) -> Tuple[Optional[float], Optional[float], str]:
        """
        Try to read the account's Coinbase Advanced fee tier.

        Returns:
            (maker_fee_bps, taker_fee_bps, reason)

        Coinbase typically returns decimal rates:
            "0.0060" = 0.60% = 60 bps

        This method is intentionally defensive because Coinbase SDK method
        signatures can differ by installed SDK version.
        """
        resp = None
        last_err = ""

        call_attempts = [
            lambda: self.rest.get_transaction_summary(),
            lambda: self.rest.get_transaction_summary(product_type="SPOT"),
            lambda: self.rest.get_transaction_summary(product_type="UNKNOWN_PRODUCT_TYPE"),
        ]

        for fn in call_attempts:
            try:
                resp = fn()
                break
            except Exception as e:
                last_err = str(e)
                resp = None

        if resp is None:
            return None, None, f"fee_tier_unavailable: {last_err}"

        d = self._to_dict(resp)

        # Coinbase SDKs may wrap the payload differently.
        candidates = [
            d,
            d.get("transaction_summary") if isinstance(d, dict) else None,
            d.get("summary") if isinstance(d, dict) else None,
        ]

        fee_tier = None
        for c in candidates:
            if isinstance(c, dict) and isinstance(c.get("fee_tier"), dict):
                fee_tier = c.get("fee_tier")
                break

        if not isinstance(fee_tier, dict):
            # Sometimes nested response objects are lists or under data.
            data = d.get("data") if isinstance(d, dict) else None
            if isinstance(data, dict) and isinstance(data.get("fee_tier"), dict):
                fee_tier = data.get("fee_tier")

        if not isinstance(fee_tier, dict):
            keys = list(d.keys()) if isinstance(d, dict) else type(d)
            return None, None, f"fee_tier_missing_in_response keys={keys}"

        maker_raw = (
            fee_tier.get("maker_fee_rate")
            or fee_tier.get("maker_fee")
            or fee_tier.get("maker_rate")
        )
        taker_raw = (
            fee_tier.get("taker_fee_rate")
            or fee_tier.get("taker_fee")
            or fee_tier.get("taker_rate")
        )

        try:
            maker_bps = float(maker_raw) * 10000.0
            taker_bps = float(taker_raw) * 10000.0
        except Exception as e:
            return None, None, f"fee_tier_parse_error: {e}; fee_tier={fee_tier}"

        if maker_bps < 0 or taker_bps < 0 or maker_bps > 500 or taker_bps > 500:
            return None, None, f"fee_tier_out_of_range maker={maker_bps} taker={taker_bps}"

        pricing_tier = str(fee_tier.get("pricing_tier") or fee_tier.get("tier") or "")
        return (
            float(maker_bps),
            float(taker_bps),
            f"fee_tier_ok {pricing_tier} maker={maker_bps:.2f}bps taker={taker_bps:.2f}bps",
        )

    def sync_after_trade(self, attempts: int = 8, sleep_sec: float = 0.5) -> None:
        """Force-refresh balances a few times to let Coinbase settle post-trade."""
        for _ in range(max(1, int(attempts))):
            self.refresh_snapshot(force=True, ttl_sec=0.0)
            time.sleep(float(sleep_sec))

    # ---------------------------
    # Live balance refresh helpers
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
        self.olog = OrderLogger(ORDERS_CSV_PATH)
        self.clog = CalibrationLogger(CALIBRATION_CSV_PATH)
        self.micro_history_log = MicroHistoryLogger(MICRO_HISTORY_CSV_PATH)
        self.position_targets_log = PositionTargetsLogger(POSITION_TARGETS_CSV_PATH)
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
        # Per-product walk-forward calibration profiles.
        self.calibration_profiles: Dict[str, ProductCalibrationProfile] = {
            p: ProductCalibrationProfile(product_id=p) for p in PRODUCTS
        }
        self.last_live_calibration_ts: float = 0.0
        self.live_recalibration_running: bool = False
        self.last_loop_lag_check_ts: float = now_ts()
        self.cached_account_snapshot: Optional[Dict[str, Dict[str, float]]] = None
        self.cached_account_snapshot_ts: float = 0.0
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
        self.portfolio = LivePortfolio(rest)
        # last macro update time
        self.last_macro_update: float = 0.0
        # stop event
        self._stop_event = asyncio.Event()
        # startup timestamp (used for FIRST_BUY_DELAY_SEC warm-up)
        self.bot_start_ts: float = now_ts()
        self.last_heartbeat_ts: float = 0.0

        # Risk/session state
        self.daily_pnl_date: Optional[str] = None
        self.daily_realized_pnl_usd: float = 0.0
        self.consecutive_losses: int = 0
        self.paused_until_ts: float = 0.0
        self.trade_timestamps: Deque[float] = deque(maxlen=500)
        self.product_trade_timestamps: Dict[str, Deque[float]] = {p: deque(maxlen=200) for p in PRODUCTS}
        self.scale_add_count: Dict[str, int] = {p: 0 for p in PRODUCTS}

        # Dynamic Coinbase fee state.
        # Defaults are conservative until Coinbase fee tier is successfully detected.
        # Dynamic Coinbase fee state.
        # None means the bot is not allowed to trade yet.
        self.current_maker_fee_bps: Optional[float] = None
        self.current_taker_fee_bps: Optional[float] = None
        self.last_fee_tier_refresh_ts: float = 0.0
        self.last_fee_tier_reason: str = "not_refreshed_yet"

    async def preload_micro_history(self) -> None:
        """Preload true 1-minute OHLCV into bot buffers and viewer history."""
        history_rows: List[Dict[str, Any]] = []

        try:
            end_ts = int(now_ts())
            start_ts = end_ts - int(MICRO_PRELOAD_MINUTES) * 60

            for product in PRODUCTS:
                candles = await self.fetcher.fetch_chunked(product, start_ts, end_ts, "ONE_MINUTE")
                if not candles:
                    log(f"[startup] no micro preload candles for {product}")
                    continue

                for c in candles:
                    minute_ts = int(c.ts)
                    open_price = float(c.open)
                    high_price = float(c.high)
                    low_price = float(c.low)
                    close_price = float(c.close)
                    volume = float(c.volume or 0.0)

                    if close_price <= 0:
                        continue

                    self.live_1m[product].append_closed_candle(
                        minute_start_ts=minute_ts,
                        open_price=open_price,
                        high_price=high_price,
                        low_price=low_price,
                        close_price=close_price,
                        volume=volume,
                    )
                    self.mid_series[product].push(float(minute_ts) + 30.0, close_price)
                    history_rows.append({
                        "ts": minute_ts,
                        "product_id": product,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                    })

            self.micro_history_log.write_rows(history_rows)
            log(f"[startup] preloaded {MICRO_PRELOAD_MINUTES}m true micro candles; rows={len(history_rows)}")
        except Exception as e:
            log(f"[startup] micro preload failed: {e}")


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
            log(f"[{side_u.lower()}] non-dict execution result for {product_id}: {type(r)}")
            return None
        if r.get("ok") is not True:
            err = r.get("error") or "exec_not_ok"
            log(f"[{side_u.lower()}] execution failed for {product_id}: {err}")
            return None

        order_id = r.get("order_id")
        if not isinstance(order_id, str) or not order_id.strip():
            # LivePortfolio should already fail without an order_id, but we double-enforce here.
            log(f"[{side_u.lower()}] missing order_id; refusing to mutate local state for {product_id}")
            return None

        filled_qty = safe_float(r.get("filled_qty")) or 0.0
        if filled_qty <= 1e-12:
            log(f"[{side_u.lower()}] zero fill for {product_id} (order_id={order_id})")
            return None

        fee_val = safe_float(r.get("fee_usd")) or 0.0
        filled_notional = safe_float(r.get("filled_notional_usd"))
        avg_px = safe_float(r.get("avg_price"))

        # If avg_price missing, derive it from notional/qty (still fill-truth, not a quote fallback).
        if (avg_px is None or avg_px <= 0) and filled_notional is not None and filled_notional > 0:
            avg_px = float(filled_notional) / float(filled_qty)

        if avg_px is None or avg_px <= 0:
            # Do NOT fall back to bid/ask. Without a fill price we cannot log truthfully.
            log(
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
        if not new_lots:
            self.position_start_ts[product] = None
            self.position_entry_price[product] = None
            self.peak_bid[product] = None
            self.scale_add_count[product] = 0
        avg_entry = (removed_cost / removed_qty) if removed_qty > 1e-12 else None
        return removed_qty, avg_entry

    async def _sell_partial(self, product: str, qty_to_sell: float, note: str) -> Optional[Tuple[float, float, float]]:
        """Sell qty_to_sell on Coinbase and return (sold_qty, exec_price, fee_usd) if filled."""
        tob = self.tob.get(product)
        if not tob:
            return None

        qty_to_sell = float(qty_to_sell)
        if qty_to_sell <= 1e-12:
            return None

        fill = await self._execute_live_sell(
            product_id=product,
            base_qty=qty_to_sell,
            bid=float(tob.bid),
            ask=float(tob.ask),
            reason=note or "partial_sell",
        )
        if fill is None:
            return None

        filled_qty, avg_px, fee_val, _filled_notional, _order_id = fill
        sold_qty = float(min(qty_to_sell, filled_qty))
        return sold_qty, float(avg_px), float(fee_val)

    async def _force_sell_product(self, product: str, note: str = "") -> None:
        tob = self.tob.get(product)
        if not tob:
            return

        lots = self.positions.get(product, [])
        qty = sum(l.qty for l in lots) if lots else 0.0
        if qty <= 0:
            return

        fill = await self._execute_live_sell(
            product_id=product,
            base_qty=float(qty),
            bid=float(tob.bid),
            ask=float(tob.ask),
            reason=note or "force_sell",
        )
        if fill is None:
            return

        filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
        exec_price = float(avg_px)
        fee = float(fee_val)
        qty_sold = float(min(qty, filled_qty))

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
            filled_notional_usd=(float(filled_notional) if filled_notional is not None else None),
        )

        ts_now = now_ts()
        self._fifo_reduce_lots(product, qty_sold)
        if self.positions.get(product):
            return
        self.ladder_plan[product] = None
        self.peak_bid[product] = None
        self.trailing_active[product] = False
        self.position_tier[product] = 0
        self.position_entry_price[product] = None
        self.scale_add_count[product] = 0
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


    async def _refresh_coinbase_fee_tier_if_needed(self, *, force: bool = False) -> None:
        """
        Refresh Coinbase maker/taker fee bps from transaction_summary.

        This bot is live-only and fee-strict:
        - If Coinbase fee tier cannot be detected, trading is blocked.
        - No hardcoded maker/taker fallback is used.
        """
        if not AUTO_REFRESH_COINBASE_FEE_TIER:
            raise RuntimeError("AUTO_REFRESH_COINBASE_FEE_TIER must remain True for live-only fee-strict mode.")

        if not REQUIRE_COINBASE_FEE_TIER:
            raise RuntimeError("REQUIRE_COINBASE_FEE_TIER must remain True for live-only fee-strict mode.")

        if not isinstance(self.portfolio, LivePortfolio):
            raise RuntimeError("Live-only bot requires LivePortfolio.")

        t = now_ts()
        if (
            not force
            and self.current_maker_fee_bps is not None
            and self.current_taker_fee_bps is not None
            and (t - float(self.last_fee_tier_refresh_ts or 0.0)) < float(FEE_TIER_REFRESH_SEC)
        ):
            return

        maker_bps, taker_bps, reason = await asyncio.to_thread(self.portfolio.get_fee_tier_bps)

        if maker_bps is None or taker_bps is None:
            log(
                f"[fee-tier] unavailable maker={maker_bps} taker={taker_bps} "
                f"reason={reason}"
            )
            self.current_maker_fee_bps = None
            self.current_taker_fee_bps = None
            self.last_fee_tier_reason = reason
            self.last_fee_tier_refresh_ts = t
            raise RuntimeError(f"Coinbase fee tier is required but unavailable: {reason}")

        self.current_maker_fee_bps = float(maker_bps)
        self.current_taker_fee_bps = float(taker_bps)
        self.last_fee_tier_reason = reason
        self.last_fee_tier_refresh_ts = t
        log(
            f"[fee-tier] refreshed from Coinbase: {reason} "
            f"maker_bps={self.current_maker_fee_bps} "
            f"taker_bps={self.current_taker_fee_bps}"
        )

    def _entry_fee_bps_for_mode(self) -> float:
        if self.current_maker_fee_bps is None or self.current_taker_fee_bps is None:
            raise RuntimeError("Coinbase fee tier has not been loaded; refusing to estimate entry fees.")

        mode = str(ENTRY_EXECUTION_MODE).upper().strip()
        if mode in ("MARKET", "LIMIT_THEN_MARKET"):
            return float(self.current_taker_fee_bps)
        return float(self.current_maker_fee_bps)

    def _exit_fee_bps_for_mode(self) -> float:
        if self.current_maker_fee_bps is None or self.current_taker_fee_bps is None:
            raise RuntimeError("Coinbase fee tier has not been loaded; refusing to estimate exit fees.")

        mode = str(EXIT_EXECUTION_MODE).upper().strip()
        if mode in ("MARKET", "LIMIT_THEN_MARKET"):
            return float(self.current_taker_fee_bps)
        return float(self.current_maker_fee_bps)

    def _round_trip_cost_bps(
        self,
        *,
        entry_mode: Optional[str] = None,
        exit_mode: Optional[str] = None,
        spread_bps: float = 0.0,
    ) -> float:
        """
        Compute the round-trip cost using real Coinbase-provided fee tier values.

        Note:
        - Coinbase fee tier is real/account-provided.
        - Spread is real from top-of-book.
        - Slippage/adverse movement buffers are still risk buffers, not Coinbase fee data.
        """
        if self.current_maker_fee_bps is None or self.current_taker_fee_bps is None:
            raise RuntimeError("Coinbase fee tier has not been loaded; refusing to compute cost.")

        entry_mode = str(entry_mode or ENTRY_EXECUTION_MODE).upper().strip()
        exit_mode = str(exit_mode or EXIT_EXECUTION_MODE).upper().strip()

        entry_fee = (
            float(self.current_taker_fee_bps)
            if entry_mode in ("MARKET", "LIMIT_THEN_MARKET")
            else float(self.current_maker_fee_bps)
        )
        exit_fee = (
            float(self.current_taker_fee_bps)
            if exit_mode in ("MARKET", "LIMIT_THEN_MARKET")
            else float(self.current_maker_fee_bps)
        )

        return float(
            entry_fee
            + exit_fee
            + float(spread_bps)
            + EST_SLIPPAGE_BPS
            + EST_ADVERSE_FILL_BPS
            + ROUND_TRIP_SAFETY_BPS
        )

    def _target_move_bps_from_room_and_sigma(
        self,
        *,
        mid: float,
        product_id: str,
        levels_day: Optional['MacroLevels'],
        levels_week: Optional['MacroLevels'],
        sigma_bps: Optional[float],
    ) -> float:
        candidates: List[float] = []

        for lv in (levels_day, levels_week):
            if not lv:
                continue
            for attr in ("resistance_zone_low", "resistance_zone_high", "prev_high", "vah", "breakout"):
                v = getattr(lv, attr, None)
                try:
                    val = float(v)
                except Exception:
                    continue
                if val > mid:
                    candidates.append(((val / mid) - 1.0) * 10000.0)

        if sigma_bps is not None and sigma_bps > 0:
            candidates.append(float(sigma_bps) * 1.15)

        if not candidates:
            return 0.0

        # Use the nearest realistic upside target rather than an unrealistic far target.
        return float(max(0.0, min(candidates)))

    def _fee_aware_expected_edge_bps(
        self,
        *,
        product_id: str,
        mid: float,
        spread_bps: float,
        levels_day: Optional['MacroLevels'],
        levels_week: Optional['MacroLevels'],
        sigma_bps: Optional[float],
    ) -> Tuple[float, float, float]:
        target_bps = self._target_move_bps_from_room_and_sigma(
            mid=mid,
            product_id=product_id,
            levels_day=levels_day,
            levels_week=levels_week,
            sigma_bps=sigma_bps,
        )
        cost_bps = self._round_trip_cost_bps(spread_bps=spread_bps)
        net_bps = float(target_bps - cost_bps)
        return net_bps, target_bps, cost_bps

    def _estimate_display_prob_up(
        self,
        *,
        score: float,
        spread_bps: float,
        momentum_5_bps: float,
        momentum_15_bps: float,
        support_score: float,
        room_score: float,
        regime_score: float,
        vwap_ok: bool,
        higher_low_ok: bool,
        trending_down: bool,
        target_bps: Optional[float] = None,
        cost_bps: Optional[float] = None,
        fee_available: bool = False,
    ) -> float:
        """
        Display probability for live monitoring.

        This should move continuously for every product.
        It is not a guarantee and not a standalone permission to buy.

        If real Coinbase fees are available, target/cost improves the probability.
        If real fees are unavailable, this still displays a live price-action probability,
        but the trade gate remains closed.
        """
        s = clamp_float(float(score), 0.0, 100.0)

        # Base probability from live score.
        # score 0 -> 32%, score 50 -> 50%, score 100 -> 68%
        prob = 0.32 + (s / 100.0) * 0.36

        # Momentum contribution.
        prob += clamp_float(float(momentum_5_bps) / 80.0, -0.08, 0.08)
        prob += clamp_float(float(momentum_15_bps) / 140.0, -0.06, 0.06)

        # Structure contribution.
        prob += ((float(support_score) - 50.0) / 100.0) * 0.045
        prob += ((float(room_score) - 50.0) / 100.0) * 0.040
        prob += ((float(regime_score) - 50.0) / 100.0) * 0.035

        # Confirmation contribution.
        prob += 0.030 if vwap_ok else -0.030
        prob += 0.030 if higher_low_ok else -0.030

        # Trend and spread penalties.
        if trending_down:
            prob -= 0.070

        if spread_bps > SCALP_MAX_SPREAD_BPS:
            prob -= 0.025
        if spread_bps > MAX_SPREAD_BPS:
            prob -= 0.080

        # Fee-aware contribution only when real Coinbase fee data is available.
        if fee_available and target_bps is not None and cost_bps is not None and float(cost_bps) > 0:
            ratio = float(target_bps) / float(cost_bps)
            if ratio >= 4.0:
                prob += 0.050
            elif ratio >= 3.0:
                prob += 0.032
            elif ratio >= MIN_TARGET_TO_COST_MULT:
                prob += 0.018
            else:
                prob -= clamp_float((MIN_TARGET_TO_COST_MULT - ratio) * 0.040, 0.0, 0.09)

        return clamp_float(prob, 0.20, 0.88)

    def _estimate_prob_up_from_candidate(
        self,
        *,
        score: float,
        expected_net_edge_bps: float,
        spread_bps: float,
        target_bps: Optional[float] = None,
        cost_bps: Optional[float] = None,
        trending_down: bool = False,
        vwap_ok: bool = False,
        higher_low_ok: bool = False,
    ) -> float:
        """
        Estimate probability that price reaches the intended profitable move from the buy point.

        This is a dynamic signal-confidence model, not a guarantee.
        It intentionally does not clamp weak setups to 50%, because that made the
        viewer look stagnant and hid meaningful differences between products.
        """
        s = clamp_float(float(score), 0.0, 100.0)
        edge = float(expected_net_edge_bps)
        spr = float(spread_bps)

        # Base from score. Allows weak setups below 50 and strong setups above 50.
        # score 0   => 0.35
        # score 50  => 0.50
        # score 80  => 0.59 before boosts
        # score 100 => 0.65 before boosts
        prob = 0.35 + (s / 100.0) * 0.30

        # Fee-adjusted edge contribution.
        # Strong edge should visibly raise probability; negative edge should lower it.
        if MIN_REQUIRED_NET_EDGE_BPS > 0:
            edge_ratio = edge / float(MIN_REQUIRED_NET_EDGE_BPS)
            prob += clamp_float(edge_ratio, -2.0, 2.5) * 0.045

        # Target-to-cost quality.
        if target_bps is not None and cost_bps is not None and float(cost_bps) > 0:
            ratio = float(target_bps) / float(cost_bps)
            if ratio >= 4.0:
                prob += 0.055
            elif ratio >= 3.0:
                prob += 0.035
            elif ratio >= MIN_TARGET_TO_COST_MULT:
                prob += 0.018
            else:
                prob -= clamp_float((float(MIN_TARGET_TO_COST_MULT) - ratio) * 0.06, 0.0, 0.12)

        # Confirmation boosts.
        if vwap_ok:
            prob += 0.030
        else:
            prob -= 0.030

        if higher_low_ok:
            prob += 0.030
        else:
            prob -= 0.030

        # Trend penalty.
        if trending_down:
            prob -= 0.080

        # Spread penalty. Wide spreads are hostile to small trades.
        if spr > SCALP_MAX_SPREAD_BPS:
            prob -= 0.030
        if spr > MAX_SPREAD_BPS:
            prob -= 0.100

        # Keep the output bounded but visibly dynamic.
        return clamp_float(prob, 0.25, 0.88)

    def _build_historical_signal_from_candles(
        self,
        *,
        product_id: str,
        candles: List[Candle],
        weekly_candles: Optional[List[Candle]],
        spread_bps: float,
    ) -> LiveSignal:
        """Build a signal using only candles available at a replay point."""
        if not candles:
            raise ValueError("No candles supplied for historical signal.")

        mid = float(candles[-1].close)
        levels_day = compute_macro_levels(candles)
        levels_week = compute_macro_levels(weekly_candles or candles)

        weekly_bias = None
        if levels_week and levels_week.range_low > 0 and levels_week.range_high > levels_week.range_low:
            weekly_bias = (
                (mid - levels_week.range_low)
                / (levels_week.range_high - levels_week.range_low)
            ) * 2.0 - 1.0

        closes = [float(c.close) for c in candles if float(c.close) > 0]
        sigma_bps = None
        if len(closes) >= 20:
            rets = [
                (closes[i] / closes[i - 1] - 1.0) * 10000.0
                for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]
            if rets:
                sigma_bps = float(np.std(rets[-60:])) if len(rets) >= 2 else 0.0

        target_bps = self._target_move_bps_from_room_and_sigma(
            mid=mid,
            product_id=product_id,
            levels_day=levels_day,
            levels_week=levels_week,
            sigma_bps=sigma_bps,
        )
        fee_available = (
            self.current_maker_fee_bps is not None
            and self.current_taker_fee_bps is not None
        )
        if fee_available:
            try:
                cost_bps = self._round_trip_cost_bps(spread_bps=spread_bps)
            except Exception:
                cost_bps = 0.0
        else:
            cost_bps = 0.0
        expected_net_edge_bps = float(target_bps - cost_bps) if fee_available else 0.0

        support_score = _support_proximity_score(mid, levels_day, levels_week)
        room_score, room_reason = _room_score(mid, levels_day, levels_week, RESIST_BUFFER_BPS)
        regime_score = (
            55.0
            if weekly_bias is None
            else _clip_score((float(weekly_bias) + 1.0) * 50.0)
        )

        momentum_5_bps = _recent_close_momentum_bps(candles, lookback=5)
        momentum_15_bps = _recent_close_momentum_bps(candles, lookback=15)
        trending_down = False
        trend_reason = "hist_trend_ok"
        if len(closes) >= MICRO_TREND_LOOKBACK_MIN:
            first = closes[-MICRO_TREND_LOOKBACK_MIN]
            last = closes[-1]
            move_bps = ((last / first) - 1.0) * 10000.0 if first > 0 else 0.0
            trending_down = move_bps <= MICRO_TREND_DOWN_BPS
            trend_reason = f"hist_trend move_bps={move_bps:.1f}"
            if trending_down:
                regime_score = min(regime_score, 35.0)

        total_pv = 0.0
        total_v = 0.0
        for candle in candles[-1440:]:
            volume = max(0.0, float(candle.volume))
            typical = (float(candle.high) + float(candle.low) + float(candle.close)) / 3.0
            total_pv += typical * volume
            total_v += volume
        hist_vwap = total_pv / total_v if total_v > 0 else None
        if hist_vwap is None or hist_vwap <= 0:
            vwap_ok = True
            vwap_reason = "hist_vwap_unknown"
        else:
            required_vwap = float(hist_vwap) * bps_to_mult(VWAP_RECLAIM_BUFFER_BPS)
            vwap_ok = mid >= required_vwap
            vwap_reason = "hist_vwap_reclaim" if vwap_ok else "hist_below_vwap"

        higher_low_ok = False
        higher_low_reason = "hist_higher_low_unknown"
        if len(candles) >= 8:
            recent = candles[-8:]
            lows = [float(c.low) for c in recent]
            recent_closes = [float(c.close) for c in recent]
            recent_low = min(lows[-3:])
            prior_low = min(lows[:5])
            close_strength = recent_closes[-1] > recent_closes[-2] >= recent_closes[-3]
            higher_low_ok = recent_low > prior_low and close_strength
            higher_low_reason = (
                "hist_higher_low_confirmed"
                if higher_low_ok
                else "hist_higher_low_not_confirmed"
            )

        dip_metrics = _dip_metrics(candles)
        if dip_metrics:
            dip_pct = float(dip_metrics.get("dip_pct", 0.0))
            dip_rate = float(dip_metrics.get("dip_rate_bps_per_min", 0.0))
            trough_low = float(dip_metrics.get("trough_low", 0.0))
            dip_depth_score = _clip_score(
                (dip_pct / max(DIP_MIN_PCT * 4.0, 1e-9)) * 100.0
            )
            dip_speed_score = _clip_score(
                (dip_rate / max(DIP_RATE_MIN_BPS_PER_MIN, 1e-9)) * 50.0
            )
            reversal_ok, reversal_reason = _dip_reversal_ok(candles, trough_low)
            reversal_score = 100.0 if reversal_ok else 35.0
        else:
            dip_depth_score = 35.0
            dip_speed_score = 35.0
            reversal_score = 35.0
            reversal_reason = "hist_dip_neutral"

        momentum_score = (
            _score_from_bps(momentum_5_bps, center_bps=0.0, width_bps=35.0) * 0.60
            + _score_from_bps(momentum_15_bps, center_bps=0.0, width_bps=65.0) * 0.40
        )
        range_position_score = _recent_range_position_score(candles, lookback=20)
        vwap_score = 72.0 if vwap_ok else 38.0
        higher_low_score = 72.0 if higher_low_ok else 38.0
        spread_penalty = max(0.0, float(spread_bps) - 6.0) * 0.80
        cost_penalty = max(0.0, float(cost_bps) - 50.0) * 0.10
        edge_score = _score_from_bps(
            expected_net_edge_bps,
            center_bps=0.0,
            width_bps=max(35.0, MIN_REQUIRED_NET_EDGE_BPS),
        )
        raw_score = (
            support_score * 0.14
            + room_score * 0.14
            + regime_score * 0.10
            + reversal_score * 0.11
            + dip_depth_score * 0.08
            + dip_speed_score * 0.05
            + momentum_score * 0.18
            + range_position_score * 0.05
            + vwap_score * 0.07
            + higher_low_score * 0.06
            + edge_score * 0.12
            - spread_penalty
            - cost_penalty
        )
        score = _clip_score(raw_score)
        tier = _score_to_tier(score)
        estimated_prob_up = self._estimate_display_prob_up(
            score=score,
            spread_bps=spread_bps,
            momentum_5_bps=momentum_5_bps,
            momentum_15_bps=momentum_15_bps,
            support_score=support_score,
            room_score=room_score,
            regime_score=regime_score,
            vwap_ok=bool(vwap_ok),
            higher_low_ok=bool(higher_low_ok),
            trending_down=bool(trending_down),
            target_bps=target_bps,
            cost_bps=cost_bps,
            fee_available=fee_available,
        )
        position_pct = self._position_pct_from_probability(estimated_prob_up)
        reason = (
            f"historical_score={score:.1f}; prob={estimated_prob_up:.3f}; "
            f"edge={expected_net_edge_bps:.1f}; target={target_bps:.1f}; cost={cost_bps:.1f}; "
            f"mom5={momentum_5_bps:.1f}; mom15={momentum_15_bps:.1f}; "
            f"room={room_reason}; {vwap_reason}; {higher_low_reason}; "
            f"{trend_reason}; {reversal_reason}"
        )
        return LiveSignal(
            ok_to_trade=False,
            score=float(score),
            tier=int(tier),
            reason=reason,
            estimated_prob_up=float(estimated_prob_up),
            position_pct=float(position_pct),
            expected_net_edge_bps=float(expected_net_edge_bps),
            target_bps=float(target_bps),
            cost_bps=float(cost_bps),
            projected_forward_gain_bps=0.0,
            calibrated_time_to_min_profit_minutes=0.0,
            calibrated_forward_window_minutes=0.0,
            dip_depth_score=float(dip_depth_score),
            dip_speed_score=float(dip_speed_score),
            reversal_score=float(reversal_score),
            support_score=float(support_score),
            room_score=float(room_score),
            regime_score=float(regime_score),
            spread_penalty=float(spread_penalty),
            cost_penalty=float(cost_penalty),
            trend_reason=trend_reason,
            vwap_reason=vwap_reason,
            higher_low_reason=higher_low_reason,
        )

    def _evaluate_forward_outcome(
        self,
        *,
        entry_price: float,
        future_candles: List[Candle],
        target_bps: float,
        cost_bps: float,
        min_net_gain_bps: float,
        bar_minutes: float,
    ) -> Tuple[float, float, bool, bool, float, float, float, Optional[int], Optional[float], float]:
        """
        Look forward after a historical signal and determine what happened.

        Also measure how many bars/minutes it took to reach minimum profit and
        the total forward window represented by the replay.
        """
        if entry_price <= 0 or not future_candles:
            return 0.0, 0.0, False, False, 0.0, 0.0, 0.0, None, None, 0.0

        highs = [float(c.high) for c in future_candles if float(c.high) > 0]
        lows = [float(c.low) for c in future_candles if float(c.low) > 0]

        if not highs or not lows:
            return 0.0, 0.0, False, False, 0.0, 0.0, 0.0, None, None, 0.0

        max_high = max(highs)
        min_low = min(lows)
        max_favorable_bps = ((max_high / entry_price) - 1.0) * 10000.0
        max_adverse_bps = ((entry_price / min_low) - 1.0) * 10000.0
        required_profit_bps = float(cost_bps) + float(min_net_gain_bps)
        reached_min_profit = max_favorable_bps >= required_profit_bps
        reached_target = max_favorable_bps >= max(float(target_bps), required_profit_bps)
        win_bps = max(0.0, max_favorable_bps - float(cost_bps))
        loss_bps = max(0.0, max_adverse_bps)

        time_to_min_profit_bars = None
        time_to_min_profit_minutes = None
        if reached_min_profit:
            for idx, candle in enumerate(future_candles, start=1):
                if float(candle.high) <= 0:
                    continue
                move_bps = ((float(candle.high) / entry_price) - 1.0) * 10000.0
                if move_bps >= required_profit_bps:
                    time_to_min_profit_bars = int(idx)
                    time_to_min_profit_minutes = float(idx) * float(bar_minutes)
                    break

        forward_window_minutes = float(len(future_candles)) * float(bar_minutes)
        return (
            float(max_favorable_bps),
            float(max_adverse_bps),
            bool(reached_min_profit),
            bool(reached_target),
            float(win_bps),
            float(loss_bps),
            0.0,
            time_to_min_profit_bars,
            time_to_min_profit_minutes,
            forward_window_minutes,
        )

    def _walk_forward_observations(
        self,
        *,
        product_id: str,
        candles: List[Candle],
        weekly_candles: Optional[List[Candle]],
        timeframe: str,
        min_prefix: int,
        forward_bars: int,
        spread_bps: float,
    ) -> List[CalibrationObservation]:
        """Replay historical candles while keeping future bars out of each signal."""
        observations: List[CalibrationObservation] = []
        if not candles or len(candles) < min_prefix + forward_bars + 1:
            return observations

        for i in range(min_prefix, len(candles) - forward_bars):
            prefix = candles[:i]
            future = candles[i:i + forward_bars]
            if not prefix or not future:
                continue
            entry_price = float(prefix[-1].close)
            if entry_price <= 0:
                continue

            replay_ts = int(prefix[-1].ts)
            available_weekly = None
            if weekly_candles:
                available_weekly = [c for c in weekly_candles if int(c.ts) <= replay_ts]
            try:
                signal = self._build_historical_signal_from_candles(
                    product_id=product_id,
                    candles=prefix,
                    weekly_candles=available_weekly,
                    spread_bps=spread_bps,
                )
            except Exception:
                continue
            bar_minutes = 1.0 if timeframe in ("day_1m", "live_rolling_1m") else 15.0
            (
                max_favorable_bps,
                max_adverse_bps,
                reached_min_profit,
                reached_target,
                win_bps,
                loss_bps,
                _,
                time_to_min_profit_bars,
                time_to_min_profit_minutes,
                forward_window_minutes,
            ) = self._evaluate_forward_outcome(
                entry_price=entry_price,
                future_candles=future,
                target_bps=signal.target_bps,
                cost_bps=signal.cost_bps,
                min_net_gain_bps=MIN_NET_GAIN_AFTER_FEES_BPS,
                bar_minutes=bar_minutes,
            )
            expected_value_bps = win_bps if reached_min_profit else -loss_bps
            observations.append(CalibrationObservation(
                product_id=product_id,
                timeframe=timeframe,
                ts=replay_ts,
                score=float(signal.score),
                probability=float(signal.estimated_prob_up),
                expected_net_edge_bps=float(signal.expected_net_edge_bps),
                target_bps=float(signal.target_bps),
                cost_bps=float(signal.cost_bps),
                spread_bps=float(spread_bps),
                max_favorable_bps=float(max_favorable_bps),
                max_adverse_bps=float(max_adverse_bps),
                reached_min_profit=bool(reached_min_profit),
                reached_target=bool(reached_target),
                expected_value_bps=float(expected_value_bps),
                win_bps=float(win_bps),
                loss_bps=float(loss_bps),
                time_to_min_profit_bars=time_to_min_profit_bars,
                time_to_min_profit_minutes=time_to_min_profit_minutes,
                forward_window_minutes=forward_window_minutes,
                projected_forward_gain_bps=float(max_favorable_bps),
            ))
        return observations

    def _win_rate(self, observations: List[CalibrationObservation]) -> float:
        if not observations:
            return 0.0
        wins = sum(1 for observation in observations if observation.reached_min_profit)
        return float(wins / len(observations))

    def _safe_quantile(self, values: List[float], q: float, default: float) -> float:
        clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        if not clean:
            return float(default)
        return float(np.quantile(clean, clamp_float(float(q), 0.0, 1.0)))

    def _exact_candidate_values(
        self,
        values: List[float],
        *,
        max_candidates: int,
    ) -> List[float]:
        """
        Return exact observed values to use as threshold candidates.

        This does not round the chosen target. If there are too many values,
        it samples exact observed values across the distribution.
        """
        clean = sorted(set(
            float(v) for v in values
            if v is not None and np.isfinite(float(v))
        ))

        if not clean:
            return []

        if len(clean) <= int(max_candidates):
            return clean

        idxs = np.linspace(0, len(clean) - 1, int(max_candidates))
        out = []
        for idx in idxs:
            out.append(clean[int(round(float(idx)))])
        return sorted(set(out))

    def _observation_ev_stats(
        self,
        observations: List[CalibrationObservation],
    ) -> Tuple[float, float, float, float, int]:
        """
        Return:
            win_rate
            avg_win_bps
            avg_loss_bps
            expected_value_bps
            sample_count

        A win means the setup reached the minimum required net gain after fees.
        """
        n = len(observations)
        if n <= 0:
            return 0.0, 0.0, 0.0, -9999.0, 0

        wins = [o for o in observations if o.reached_min_profit]
        losses = [o for o in observations if not o.reached_min_profit]

        win_rate = len(wins) / max(1, n)
        avg_win = float(np.mean([o.win_bps for o in wins])) if wins else 0.0
        avg_loss = float(np.mean([o.loss_bps for o in losses])) if losses else 0.0
        ev = win_rate * avg_win - (1.0 - win_rate) * avg_loss

        return float(win_rate), float(avg_win), float(avg_loss), float(ev), int(n)

    def _projection_stats_from_observations(
        self,
        observations: List[CalibrationObservation],
    ) -> Tuple[float, float, float]:
        """Return median gross movement, time-to-profit, and forward window.

        Winning observations are preferred because this projection describes
        how far and how fast a similar setup usually moves when it works.
        """
        if not observations:
            return 0.0, 0.0, 0.0

        winners = [o for o in observations if o.reached_min_profit]
        source = winners if winners else observations
        favorable = [
            float(o.max_favorable_bps)
            for o in source
            if o.max_favorable_bps is not None
            and np.isfinite(float(o.max_favorable_bps))
        ]
        times = [
            float(o.time_to_min_profit_minutes)
            for o in winners
            if o.time_to_min_profit_minutes is not None
            and np.isfinite(float(o.time_to_min_profit_minutes))
        ]
        windows = [
            float(o.forward_window_minutes)
            for o in source
            if o.forward_window_minutes is not None
            and np.isfinite(float(o.forward_window_minutes))
        ]

        projected_gross_bps = float(np.median(favorable)) if favorable else 0.0
        median_time_to_min_profit_minutes = float(np.median(times)) if times else 0.0
        median_forward_window_minutes = float(np.median(windows)) if windows else 0.0
        return (
            projected_gross_bps,
            median_time_to_min_profit_minutes,
            median_forward_window_minutes,
        )

    def _build_calibration_profile(
        self,
        *,
        product_id: str,
        day_obs: List[CalibrationObservation],
        week_obs: List[CalibrationObservation],
    ) -> ProductCalibrationProfile:
        """
        Build exact per-product buy thresholds from walk-forward observations.

        This version does NOT round into buckets and does NOT force every product
        to the same 45 / 50% floor.

        It chooses the score/probability pair that historically had the best
        likelihood of reaching the minimum required gain after fees.
        """
        all_obs = list(day_obs) + list(week_obs)

        if len(all_obs) < CALIB_MIN_PRODUCT_SAMPLES:
            return ProductCalibrationProfile(
                product_id=product_id,
                min_score=DEFAULT_CALIB_MIN_SCORE,
                min_probability=DEFAULT_CALIB_MIN_PROB,
                min_expected_value_bps=DEFAULT_CALIB_MIN_EV_BPS,
                day_sample_count=len(day_obs),
                week_sample_count=len(week_obs),
                day_win_rate=self._win_rate(day_obs),
                week_win_rate=self._win_rate(week_obs),
                calibrated_projected_gross_bps=0.0,
                calibrated_projected_net_bps=0.0,
                calibrated_time_to_min_profit_minutes=0.0,
                calibrated_forward_window_minutes=0.0,
                reason=f"insufficient_samples total={len(all_obs)} using_defaults",
            )

        # Overall product stats.
        blended_wr, blended_avg_win, blended_avg_loss, blended_ev, _ = self._observation_ev_stats(all_obs)

        # Candidate thresholds come from exact observed values.
        score_candidates = self._exact_candidate_values(
            [o.score for o in all_obs],
            max_candidates=CALIB_MAX_EXACT_SCORE_CANDIDATES,
        )
        prob_candidates = self._exact_candidate_values(
            [o.probability for o in all_obs],
            max_candidates=CALIB_MAX_EXACT_PROB_CANDIDATES,
        )

        best: Optional[Dict[str, Any]] = None

        for score_threshold in score_candidates:
            for prob_threshold in prob_candidates:
                selected = [
                    o for o in all_obs
                    if float(o.score) >= float(score_threshold)
                    and float(o.probability) >= float(prob_threshold)
                ]

                if len(selected) < CALIB_EXACT_MIN_SAMPLES:
                    continue

                win_rate, avg_win, avg_loss, ev, n = self._observation_ev_stats(selected)
                (
                    projected_gross_bps,
                    median_time_to_min_profit,
                    median_forward_window,
                ) = self._projection_stats_from_observations(selected)

                if win_rate < CALIB_MIN_WIN_RATE:
                    continue

                if ev < CALIB_MIN_EXPECTED_VALUE_BPS:
                    continue

                # Tradeoff:
                # - prioritize expected value
                # - then win rate
                # - then sample count
                # - slightly prefer lower thresholds only after quality is proven
                opportunity_bonus = min(10.0, n / 25.0)
                quality_score = (
                    ev * 1.00
                    + win_rate * 10.0
                    + opportunity_bonus
                    - float(score_threshold) * 0.015
                    - float(prob_threshold) * 1.50
                )

                candidate = {
                    "score_threshold": float(score_threshold),
                    "prob_threshold": float(prob_threshold),
                    "win_rate": float(win_rate),
                    "avg_win": float(avg_win),
                    "avg_loss": float(avg_loss),
                    "ev": float(ev),
                    "projected_gross_bps": float(projected_gross_bps),
                    "median_time_to_min_profit": float(median_time_to_min_profit),
                    "median_forward_window": float(median_forward_window),
                    "n": int(n),
                    "quality_score": float(quality_score),
                }

                if best is None or candidate["quality_score"] > best["quality_score"]:
                    best = candidate

        if best is not None:
            return ProductCalibrationProfile(
                product_id=product_id,
                min_score=float(best["score_threshold"]),
                min_probability=float(best["prob_threshold"]),
                min_expected_value_bps=max(float(best["ev"]) * 0.35, CALIB_MIN_EXPECTED_VALUE_BPS),
                day_sample_count=len(day_obs),
                week_sample_count=len(week_obs),
                day_win_rate=self._win_rate(day_obs),
                week_win_rate=self._win_rate(week_obs),
                blended_win_rate=float(best["win_rate"]),
                avg_win_bps=float(best["avg_win"]),
                avg_loss_bps=float(best["avg_loss"]),
                expected_value_bps=float(best["ev"]),
                calibrated_projected_gross_bps=float(best["projected_gross_bps"]),
                calibrated_projected_net_bps=float(best["ev"]),
                calibrated_time_to_min_profit_minutes=float(best["median_time_to_min_profit"]),
                calibrated_forward_window_minutes=float(best["median_forward_window"]),
                reason=(
                    f"exact_threshold product={product_id} "
                    f"score>={best['score_threshold']:.6f} "
                    f"prob>={best['prob_threshold']:.6f} "
                    f"samples={best['n']} "
                    f"win_rate={best['win_rate']:.6f} "
                    f"ev={best['ev']:.6f} "
                    f"avg_win={best['avg_win']:.6f} "
                    f"avg_loss={best['avg_loss']:.6f}"
                ),
            )

        winning_obs = [o for o in all_obs if o.reached_min_profit]

        if winning_obs and ALLOW_WINNER_BASED_FALLBACK_THRESHOLDS:
            # Product-specific fallback:
            # Use this product's actual historical winners.
            # A "winner" means the future candles reached the minimum required
            # net gain after fees/costs.
            fallback_score = self._safe_quantile(
                [o.score for o in winning_obs],
                CALIB_WINNER_SCORE_QUANTILE,
                DEFAULT_CALIB_MIN_SCORE,
            )
            fallback_prob = self._safe_quantile(
                [o.probability for o in winning_obs],
                CALIB_WINNER_PROB_QUANTILE,
                DEFAULT_CALIB_MIN_PROB,
            )

            # Safety only. Do not force back to 60 / 58%.
            fallback_score = max(float(fallback_score), float(CALIB_ABSOLUTE_MIN_SCORE))
            fallback_prob = max(float(fallback_prob), float(CALIB_ABSOLUTE_MIN_PROB))

            fallback_wr, fallback_avg_win, fallback_avg_loss, fallback_ev, fallback_n = (
                self._observation_ev_stats(winning_obs)
            )

            (
                fallback_projected_gross,
                fallback_time_to_profit,
                fallback_window,
            ) = self._projection_stats_from_observations(winning_obs)

            return ProductCalibrationProfile(
                product_id=product_id,
                min_score=float(fallback_score),
                min_probability=float(fallback_prob),
                min_expected_value_bps=max(
                    DEFAULT_CALIB_MIN_EV_BPS,
                    CALIB_MIN_EXPECTED_VALUE_BPS,
                ),
                day_sample_count=len(day_obs),
                week_sample_count=len(week_obs),
                day_win_rate=self._win_rate(day_obs),
                week_win_rate=self._win_rate(week_obs),
                blended_win_rate=blended_wr,
                avg_win_bps=fallback_avg_win,
                avg_loss_bps=fallback_avg_loss,
                expected_value_bps=blended_ev,
                calibrated_projected_gross_bps=float(fallback_projected_gross),
                calibrated_projected_net_bps=float(blended_ev),
                calibrated_time_to_min_profit_minutes=float(fallback_time_to_profit),
                calibrated_forward_window_minutes=float(fallback_window),
                reason=(
                    f"winner_based_product_fallback product={product_id} "
                    f"winning_samples={len(winning_obs)} "
                    f"score_q={fallback_score:.6f} "
                    f"prob_q={fallback_prob:.6f} "
                    f"overall_ev={blended_ev:.6f} "
                    f"note=targets_from_actual_min_gain_winners"
                ),
            )

        return ProductCalibrationProfile(
            product_id=product_id,
            min_score=DEFAULT_CALIB_MIN_SCORE,
            min_probability=DEFAULT_CALIB_MIN_PROB,
            min_expected_value_bps=max(DEFAULT_CALIB_MIN_EV_BPS, CALIB_MIN_EXPECTED_VALUE_BPS),
            day_sample_count=len(day_obs),
            week_sample_count=len(week_obs),
            day_win_rate=self._win_rate(day_obs),
            week_win_rate=self._win_rate(week_obs),
            blended_win_rate=blended_wr,
            avg_win_bps=blended_avg_win,
            avg_loss_bps=blended_avg_loss,
            expected_value_bps=blended_ev,
            calibrated_projected_gross_bps=0.0,
            calibrated_projected_net_bps=float(blended_ev),
            calibrated_time_to_min_profit_minutes=0.0,
            calibrated_forward_window_minutes=0.0,
            reason=(
                f"no_winning_observations product={product_id} "
                f"using_defaults total={len(all_obs)} "
                f"overall_ev={blended_ev:.6f}"
            ),
        )

    def _simulate_armed_exit_net_bps(
        self,
        *,
        entry_price: float,
        future_candles: List[Candle],
        target_bps: float,
        cost_bps: float,
        pullback_pct: float,
    ) -> Optional[float]:
        """Simulate a target-arm and pullback exit over historical candles."""
        if entry_price <= 0 or not future_candles:
            return None
        target_price = entry_price * bps_to_mult(float(target_bps))
        armed = False
        peak = 0.0
        for candle in future_candles:
            high = float(candle.high)
            low = float(candle.low)
            close = float(candle.close)
            if not armed:
                if high >= target_price:
                    armed = True
                    peak = max(high, target_price)
                continue
            peak = max(peak, high)
            trigger_price = peak * (1.0 - float(pullback_pct))
            if low <= trigger_price:
                gross_bps = ((trigger_price / entry_price) - 1.0) * 10000.0
                return float(gross_bps - float(cost_bps))
            peak = max(peak, close)
        if armed and peak > 0:
            gross_bps = ((float(future_candles[-1].close) / entry_price) - 1.0) * 10000.0
            return float(gross_bps - float(cost_bps))
        return None

    def _calibrate_sell_pullbacks(
        self,
        *,
        product_id: str,
        candles: List[Candle],
        weekly_candles: Optional[List[Candle]],
        profile: ProductCalibrationProfile,
        spread_bps: float,
    ) -> ProductCalibrationProfile:
        """Choose scalp/core pullbacks from recent target-arm simulations."""
        required = CALIB_MIN_PREFIX_CANDLES_1M + CALIB_FORWARD_MINUTES_1M + 1
        if not candles or len(candles) < required:
            return profile
        scalp_results: Dict[float, List[float]] = {
            pullback: [] for pullback in CALIB_SCALP_PULLBACK_CANDIDATES
        }
        core_results: Dict[float, List[float]] = {
            pullback: [] for pullback in CALIB_CORE_PULLBACK_CANDIDATES
        }
        end_i = len(candles) - CALIB_FORWARD_MINUTES_1M
        for i in range(CALIB_MIN_PREFIX_CANDLES_1M, end_i):
            prefix = candles[:i]
            future = candles[i:i + CALIB_FORWARD_MINUTES_1M]
            entry_price = float(prefix[-1].close)
            if entry_price <= 0:
                continue
            replay_ts = int(prefix[-1].ts)
            available_weekly = None
            if weekly_candles:
                available_weekly = [c for c in weekly_candles if int(c.ts) <= replay_ts]
            try:
                signal = self._build_historical_signal_from_candles(
                    product_id=product_id,
                    candles=prefix,
                    weekly_candles=available_weekly,
                    spread_bps=spread_bps,
                )
            except Exception:
                continue
            if (
                signal.score < profile.min_score
                or signal.estimated_prob_up < profile.min_probability
            ):
                continue
            scalp_target_bps = max(
                signal.cost_bps + MIN_NET_GAIN_AFTER_FEES_BPS,
                signal.target_bps * 0.55,
            )
            core_target_bps = max(
                signal.cost_bps + MIN_NET_GAIN_AFTER_FEES_BPS,
                signal.target_bps,
            )
            for pullback in CALIB_SCALP_PULLBACK_CANDIDATES:
                result = self._simulate_armed_exit_net_bps(
                    entry_price=entry_price,
                    future_candles=future,
                    target_bps=scalp_target_bps,
                    cost_bps=signal.cost_bps,
                    pullback_pct=pullback,
                )
                if result is not None:
                    scalp_results[pullback].append(result)
            for pullback in CALIB_CORE_PULLBACK_CANDIDATES:
                result = self._simulate_armed_exit_net_bps(
                    entry_price=entry_price,
                    future_candles=future,
                    target_bps=core_target_bps,
                    cost_bps=signal.cost_bps,
                    pullback_pct=pullback,
                )
                if result is not None:
                    core_results[pullback].append(result)

        def choose_best(results: Dict[float, List[float]], default: float) -> float:
            best_pullback = default
            best_score = -1e9
            for pullback, values in results.items():
                if len(values) < 5:
                    continue
                win_rate = sum(
                    1 for value in values if value >= MIN_NET_GAIN_AFTER_FEES_BPS
                ) / len(values)
                candidate_score = float(np.mean(values)) + win_rate * 10.0
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pullback = pullback
            return float(best_pullback)

        profile.scalp_pullback_pct = clamp_float(
            choose_best(scalp_results, profile.scalp_pullback_pct),
            CALIB_MIN_SCALP_PULLBACK,
            CALIB_MAX_SCALP_PULLBACK,
        )
        profile.core_pullback_pct = clamp_float(
            choose_best(core_results, profile.core_pullback_pct),
            CALIB_MIN_CORE_PULLBACK,
            CALIB_MAX_CORE_PULLBACK,
        )
        profile.reason += (
            f"; sell_calibrated scalp={profile.scalp_pullback_pct:.4%} "
            f"core={profile.core_pullback_pct:.4%}"
        )
        return profile

    async def calibrate_products_on_startup(self) -> None:
        """Fetch recent history and build per-product calibration profiles."""
        if not ENABLE_WALK_FORWARD_CALIBRATION:
            log("[calibration] disabled")
            return
        log("[calibration] startup walk-forward calibration started")
        end_ts = int(now_ts_i())
        start_day = end_ts - CALIB_DAY_LOOKBACK_MINUTES * 60
        start_week = end_ts - CALIB_WEEK_LOOKBACK_MINUTES * 60
        for product in PRODUCTS:
            try:
                log(f"[calibration] fetching history for {product}")
                day_candles = await self.fetcher.fetch_chunked(
                    product, start_day, end_ts, CALIB_DAY_GRANULARITY
                )
                week_candles = await self.fetcher.fetch_chunked(
                    product, start_week, end_ts, CALIB_WEEK_GRANULARITY
                )
                if not day_candles:
                    log(f"[calibration] no day candles for {product}; using default profile")
                    profile = ProductCalibrationProfile(
                        product_id=product,
                        reason="no_day_candles_default",
                    )
                    self.calibration_profiles[product] = profile
                    self.clog.log_profile(profile)
                    continue
                tob = self.tob.get(product)
                spread_bps = (
                    float(tob.spread_bps)
                    if tob and tob.spread_bps > 0
                    else float(MAX_SPREAD_BPS)
                )
                day_obs = self._walk_forward_observations(
                    product_id=product,
                    candles=day_candles,
                    weekly_candles=week_candles,
                    timeframe="day_1m",
                    min_prefix=CALIB_MIN_PREFIX_CANDLES_1M,
                    forward_bars=CALIB_FORWARD_MINUTES_1M,
                    spread_bps=spread_bps,
                )
                week_obs = self._walk_forward_observations(
                    product_id=product,
                    candles=week_candles,
                    weekly_candles=week_candles,
                    timeframe="week_15m",
                    min_prefix=CALIB_MIN_PREFIX_CANDLES_15M,
                    forward_bars=CALIB_FORWARD_BARS_15M,
                    spread_bps=spread_bps,
                ) if week_candles else []
                log(
                    f"[calibration-debug] {product} "
                    f"day_obs={len(day_obs)} week_obs={len(week_obs)} "
                    f"spread_bps={spread_bps:.3f}"
                )
                profile = self._build_calibration_profile(
                    product_id=product,
                    day_obs=day_obs,
                    week_obs=week_obs,
                )
                profile = self._calibrate_sell_pullbacks(
                    product_id=product,
                    candles=day_candles,
                    weekly_candles=week_candles,
                    profile=profile,
                    spread_bps=spread_bps,
                )
                self.calibration_profiles[product] = profile
                self.clog.log_profile(profile)
                log(
                    f"[calibration-debug] {product} profile "
                    f"min_score={profile.min_score:.6f} "
                    f"min_prob={profile.min_probability:.6f} "
                    f"min_ev={profile.min_expected_value_bps:.6f} "
                    f"projected_gross={profile.calibrated_projected_gross_bps:.6f} "
                    f"projected_net={profile.calibrated_projected_net_bps:.6f} "
                    f"time_to_profit={profile.calibrated_time_to_min_profit_minutes:.3f} "
                    f"scalp_pb={profile.scalp_pullback_pct:.4%} "
                    f"core_pb={profile.core_pullback_pct:.4%} "
                    f"reason={profile.reason}"
                )
            except Exception as exc:
                log_exception(f"[calibration] failed for {product}", exc)
                profile = ProductCalibrationProfile(
                    product_id=product,
                    reason=f"calibration_error={exc}",
                )
                self.calibration_profiles[product] = profile
                self.clog.log_profile(profile)
        log("[calibration] startup walk-forward calibration finished")

    def _run_live_recalibration(self) -> None:
        """Smooth buy thresholds using completed rolling one-minute candles."""
        for product in PRODUCTS:
            live_rows = self.live_1m[product].export_rows(product)
            minimum_rows = max(
                LIVE_RECALIBRATION_MIN_ROWS,
                CALIB_MIN_PREFIX_CANDLES_1M + CALIB_FORWARD_MINUTES_1M,
            )
            if len(live_rows) < minimum_rows:
                continue
            live_candles = [
                Candle(
                    ts=int(row["ts"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                )
                for row in live_rows
            ]
            old_profile = self.calibration_profiles.get(
                product, ProductCalibrationProfile(product_id=product)
            )
            tob = self.tob.get(product)
            spread_bps = (
                float(tob.spread_bps)
                if tob and tob.spread_bps > 0
                else float(MAX_SPREAD_BPS)
            )
            forward_bars = min(
                CALIB_FORWARD_MINUTES_1M,
                max(5, len(live_candles) // 4),
            )
            day_obs = self._walk_forward_observations(
                product_id=product,
                candles=live_candles,
                weekly_candles=live_candles,
                timeframe="live_rolling_1m",
                min_prefix=CALIB_MIN_PREFIX_CANDLES_1M,
                forward_bars=forward_bars,
                spread_bps=spread_bps,
            )
            if len(day_obs) < CALIB_MIN_PRODUCT_SAMPLES:
                continue
            new_profile = self._build_calibration_profile(
                product_id=product,
                day_obs=day_obs,
                week_obs=[],
            )
            # Smooth each product against its own previous target only.
            # Do not clamp or bucket; preserve exact product-specific values.
            old_profile.min_score = (
                float(old_profile.min_score) * 0.80
                + float(new_profile.min_score) * 0.20
            )
            old_profile.min_probability = (
                float(old_profile.min_probability) * 0.80
                + float(new_profile.min_probability) * 0.20
            )
            old_profile.min_expected_value_bps = (
                float(old_profile.min_expected_value_bps) * 0.80
                + float(new_profile.min_expected_value_bps) * 0.20
            )
            old_profile.day_sample_count = new_profile.day_sample_count
            old_profile.day_win_rate = new_profile.day_win_rate
            old_profile.blended_win_rate = new_profile.blended_win_rate
            old_profile.avg_win_bps = new_profile.avg_win_bps
            old_profile.avg_loss_bps = new_profile.avg_loss_bps
            old_profile.expected_value_bps = new_profile.expected_value_bps
            old_profile.calibrated_projected_gross_bps = (
                float(old_profile.calibrated_projected_gross_bps) * 0.80
                + float(new_profile.calibrated_projected_gross_bps) * 0.20
            )
            old_profile.calibrated_projected_net_bps = (
                float(old_profile.calibrated_projected_net_bps) * 0.80
                + float(new_profile.calibrated_projected_net_bps) * 0.20
            )
            old_profile.calibrated_time_to_min_profit_minutes = (
                float(old_profile.calibrated_time_to_min_profit_minutes) * 0.80
                + float(new_profile.calibrated_time_to_min_profit_minutes) * 0.20
            )
            old_profile.calibrated_forward_window_minutes = (
                float(old_profile.calibrated_forward_window_minutes) * 0.80
                + float(new_profile.calibrated_forward_window_minutes) * 0.20
            )
            old_profile.reason = "smoothed_live_recalibration"
            self.calibration_profiles[product] = old_profile
            self.clog.log_profile(old_profile)


    def _build_live_signal(
        self,
        *,
        product_id: str,
        mid: float,
        spread_bps: float,
        levels_day: Optional['MacroLevels'],
        levels_week: Optional['MacroLevels'],
        minute_candles: List['MinuteCandle'],
        weekly_bias: Optional[float],
        sigma_bps: Optional[float],
    ) -> LiveSignal:
        """
        Build a continuous live signal for every product.

        This is intentionally separate from score_entry_candidate().
        score_entry_candidate() answers: "Is the strict dip setup tradeable?"
        _build_live_signal() answers: "How attractive is this product right now?"
        """
        fee_available = (
            self.current_maker_fee_bps is not None
            and self.current_taker_fee_bps is not None
        )

        target_bps = self._target_move_bps_from_room_and_sigma(
            mid=mid,
            product_id=product_id,
            levels_day=levels_day,
            levels_week=levels_week,
            sigma_bps=sigma_bps,
        )

        if fee_available:
            try:
                round_trip_cost_bps = self._round_trip_cost_bps(spread_bps=spread_bps)
            except Exception as cost_err:
                log(f"[signal] fee cost unavailable for {product_id}: {cost_err}")
                round_trip_cost_bps = None
        else:
            round_trip_cost_bps = None

        cost_bps = float(round_trip_cost_bps) if round_trip_cost_bps is not None else 0.0
        profile = self.calibration_profiles.get(
            product_id,
            ProductCalibrationProfile(product_id=product_id),
        )
        calibrated_forward_gain_bps = float(profile.calibrated_projected_gross_bps or 0.0)
        # Before calibration is available, fall back to the structure target.
        if calibrated_forward_gain_bps <= 0:
            calibrated_forward_gain_bps = float(target_bps)
        expected_net_edge_bps = (
            float(calibrated_forward_gain_bps - cost_bps) if fee_available else 0.0
        )

        support_score = _support_proximity_score(mid, levels_day, levels_week)
        room_score, room_reason = _room_score(mid, levels_day, levels_week, RESIST_BUFFER_BPS)

        if weekly_bias is None:
            regime_score = 55.0
        else:
            regime_score = _clip_score((float(weekly_bias) + 1.0) * 50.0)

        trending_down, trend_reason = self._micro_trending_down(product_id)
        vwap_ok, vwap_reason = self._micro_vwap_reclaimed(product_id, mid)
        higher_low_ok, higher_low_reason = self._higher_low_confirmed(product_id)

        if trending_down:
            regime_score = min(regime_score, 35.0)

        dm = _dip_metrics(minute_candles)
        if dm:
            dip_pct = float(dm.get("dip_pct", 0.0))
            dip_rate = float(dm.get("dip_rate_bps_per_min", 0.0))
            trough_low = float(dm.get("trough_low", 0.0))
            dip_depth_score = _clip_score((dip_pct / max(DIP_MIN_PCT * 4.0, 1e-9)) * 100.0)
            dip_speed_score = _clip_score((dip_rate / max(DIP_RATE_MIN_BPS_PER_MIN, 1e-9)) * 50.0)
            rev_ok, _rev_reason = _dip_reversal_ok(minute_candles, trough_low)
            reversal_score = 100.0 if rev_ok else 35.0
        else:
            dip_depth_score = 35.0
            dip_speed_score = 35.0
            reversal_score = 35.0

        momentum_5_bps = _recent_close_momentum_bps(minute_candles, lookback=5)
        momentum_15_bps = _recent_close_momentum_bps(minute_candles, lookback=15)
        momentum_score = (
            _score_from_bps(momentum_5_bps, center_bps=0.0, width_bps=35.0) * 0.60
            + _score_from_bps(momentum_15_bps, center_bps=0.0, width_bps=65.0) * 0.40
        )
        range_position_score = _recent_range_position_score(minute_candles, lookback=20)

        vwap_score = 72.0 if vwap_ok else 38.0
        higher_low_score = 72.0 if higher_low_ok else 38.0
        spread_penalty = max(0.0, float(spread_bps) - 6.0) * 0.80
        cost_penalty = max(0.0, float(cost_bps) - 50.0) * 0.10
        edge_score = _score_from_bps(
            expected_net_edge_bps,
            center_bps=0.0,
            width_bps=max(35.0, MIN_REQUIRED_NET_EDGE_BPS),
        )

        raw_score = (
            support_score * 0.14
            + room_score * 0.14
            + regime_score * 0.10
            + reversal_score * 0.11
            + dip_depth_score * 0.08
            + dip_speed_score * 0.05
            + momentum_score * 0.18
            + range_position_score * 0.05
            + vwap_score * 0.07
            + higher_low_score * 0.06
            + edge_score * 0.12
            - spread_penalty
            - cost_penalty
        )

        score = _clip_score(raw_score)
        tier = _score_to_tier(score)
        estimated_prob_up = self._estimate_display_prob_up(
            score=score,
            spread_bps=spread_bps,
            momentum_5_bps=momentum_5_bps,
            momentum_15_bps=momentum_15_bps,
            support_score=support_score,
            room_score=room_score,
            regime_score=regime_score,
            vwap_ok=bool(vwap_ok),
            higher_low_ok=bool(higher_low_ok),
            trending_down=bool(trending_down),
            target_bps=target_bps,
            cost_bps=cost_bps,
            fee_available=fee_available,
        )

        position_pct = self._position_pct_from_probability(estimated_prob_up)

        if fee_available and round_trip_cost_bps is not None:
            strict_entry = score_entry_candidate(
                mid=mid,
                spread_bps=spread_bps,
                levels_day=levels_day,
                levels_week=levels_week,
                minute_candles=minute_candles,
                weekly_bias=weekly_bias,
                trending_down=trending_down,
                resist_buffer_bps=RESIST_BUFFER_BPS,
                round_trip_cost_bps=float(round_trip_cost_bps),
            )
        else:
            strict_entry = EntryScore(
                False,
                score,
                tier,
                "fee_tier_unavailable_trade_gate_closed",
                dip_depth_score,
                dip_speed_score,
                reversal_score,
                support_score,
                room_score,
                regime_score,
                spread_penalty,
                cost_penalty,
                0.0,
            )

        calib_min_score = float(profile.min_score)
        calib_min_probability = float(profile.min_probability)
        calib_min_ev = float(profile.min_expected_value_bps)

        # Individual buy-gate checks.
        buy_gate_fee_ok = bool(fee_available and round_trip_cost_bps is not None)
        buy_gate_score_target_ok = bool(score >= calib_min_score)
        buy_gate_prob_target_ok = bool(estimated_prob_up >= calib_min_probability)

        buy_gate_score_floor_ok = bool(score >= float(EV_PRIMARY_MIN_SCORE_FLOOR))
        buy_gate_prob_floor_ok = bool(
            estimated_prob_up >= float(EV_PRIMARY_MIN_PROB_FLOOR)
        )

        buy_gate_ev_ok = bool(
            expected_net_edge_bps >= max(
                float(MIN_REQUIRED_NET_EDGE_BPS),
                calib_min_ev,
                float(EV_PRIMARY_MIN_PROJECTED_NET_BPS),
            )
        )

        # The calibrated targets should remain the real displayed target.
        # EV-primary mode may be used as a secondary permissive mode,
        # but it should not hide broken calibration or replace the repaired target.
        if USE_EV_PRIMARY_BUY_GATE and buy_gate_ev_ok:
            buy_gate_score_ok = bool(
                buy_gate_score_target_ok or buy_gate_score_floor_ok
            )
            buy_gate_prob_ok = bool(
                buy_gate_prob_target_ok or buy_gate_prob_floor_ok
            )
        else:
            buy_gate_score_ok = buy_gate_score_target_ok
            buy_gate_prob_ok = buy_gate_prob_target_ok
        # Target/cost gate:
        # Use calibrated projected forward gain, not the small structural target_bps.
        # The structural target is often only a few bps and was blocking every buy.
        if USE_CALIBRATED_FORWARD_GAIN_FOR_TARGET_COST_GATE:
            buy_gate_target_cost_ok = bool(
                cost_bps > 0
                and calibrated_forward_gain_bps >= (
                    cost_bps + float(MIN_PROJECTED_GAIN_OVER_COST_BPS)
                )
            )
        else:
            buy_gate_target_cost_ok = bool(
                cost_bps > 0
                and target_bps >= cost_bps * float(MIN_TARGET_TO_COST_MULT)
            )
        buy_gate_spread_ok = bool(spread_bps <= float(MAX_SPREAD_BPS))
        buy_gate_strict_ok = bool(strict_entry.ok)

        # Basic market-safety check:
        # Do not force the old exact dip/reversal pattern unless configured.
        # For calibrated buys, the key safety check is avoiding obviously falling microtrends.
        basic_reversal_ok = True

        if BLOCK_BUY_WHEN_MICRO_TRENDING_DOWN and bool(trending_down):
            basic_reversal_ok = False

        if REQUIRE_BASIC_REVERSAL_CONFIRMATION_FOR_CALIBRATED_BUY:
            basic_reversal_ok = bool(
                basic_reversal_ok
                and score >= float(BASIC_REVERSAL_MIN_SCORE)
                and room_score >= float(BASIC_REVERSAL_MIN_ROOM_SCORE)
                and support_score >= float(BASIC_REVERSAL_MIN_SUPPORT_SCORE)
            )

        if REQUIRE_STRICT_DIP_GATE_FOR_BUY:
            buy_gate_setup_ok = buy_gate_strict_ok
            setup_blocker = f"strict_entry_blocked:{strict_entry.reason}"
        else:
            if buy_gate_strict_ok:
                buy_gate_setup_ok = True
                setup_blocker = "strict_entry_passed"
            else:
                buy_gate_setup_ok = bool(basic_reversal_ok)
                setup_blocker = (
                    "calibrated_setup_allowed"
                    if basic_reversal_ok
                    else f"micro_trend_blocked:{trend_reason}"
                )

        buy_gate_calibrated_ok = bool(
            buy_gate_score_ok
            and buy_gate_prob_ok
            and buy_gate_ev_ok
        )

        ok_to_trade = bool(
            buy_gate_fee_ok
            and buy_gate_setup_ok
            and buy_gate_calibrated_ok
            and buy_gate_target_cost_ok
            and buy_gate_spread_ok
        )

        blockers = []
        if not buy_gate_fee_ok:
            blockers.append("fee_not_ready")
        if not buy_gate_score_ok:
            if USE_EV_PRIMARY_BUY_GATE and buy_gate_ev_ok:
                blockers.append("score_below_ev_primary_floor")
            else:
                blockers.append("score_below_target")

        if not buy_gate_prob_ok:
            if USE_EV_PRIMARY_BUY_GATE and buy_gate_ev_ok:
                blockers.append("probability_below_ev_primary_floor")
            else:
                blockers.append("probability_below_target")
        if not buy_gate_ev_ok:
            blockers.append("ev_below_target")
        if not buy_gate_target_cost_ok:
            if USE_CALIBRATED_FORWARD_GAIN_FOR_TARGET_COST_GATE:
                blockers.append("projected_gain_does_not_cover_cost")
            else:
                blockers.append("target_to_cost_failed")
        if not buy_gate_spread_ok:
            blockers.append("spread_too_wide")
        if not buy_gate_setup_ok:
            blockers.append(setup_blocker)

        buy_gate_blocker = "BUY_READY" if ok_to_trade else ";".join(blockers)

        if ok_to_trade:
            log(
                f"[buy-gate] {product_id} BUY_READY "
                f"score={score:.3f} min_score={calib_min_score:.3f} "
                f"score_floor={EV_PRIMARY_MIN_SCORE_FLOOR:.3f} score_ok={buy_gate_score_ok} "
                f"score_target_ok={buy_gate_score_target_ok} "
                f"prob={estimated_prob_up:.6f} min_prob={calib_min_probability:.6f} "
                f"prob_floor={EV_PRIMARY_MIN_PROB_FLOOR:.6f} prob_ok={buy_gate_prob_ok} "
                f"prob_target_ok={buy_gate_prob_target_ok} "
                f"ev_primary={USE_EV_PRIMARY_BUY_GATE} "
                f"ev={expected_net_edge_bps:.3f} "
                f"min_ev={max(float(MIN_REQUIRED_NET_EDGE_BPS), calib_min_ev, float(EV_PRIMARY_MIN_PROJECTED_NET_BPS)):.3f} "
                f"target={target_bps:.3f} "
                f"projected_forward={calibrated_forward_gain_bps:.3f} "
                f"cost={cost_bps:.3f} "
                f"spread={spread_bps:.3f}"
            )
        else:
            log(
                f"[buy-gate] {product_id} BLOCKED "
                f"blocker={buy_gate_blocker} "
                f"score={score:.3f} min_score={calib_min_score:.3f} "
                f"score_floor={EV_PRIMARY_MIN_SCORE_FLOOR:.3f} score_ok={buy_gate_score_ok} "
                f"score_target_ok={buy_gate_score_target_ok} "
                f"prob={estimated_prob_up:.6f} min_prob={calib_min_probability:.6f} "
                f"prob_floor={EV_PRIMARY_MIN_PROB_FLOOR:.6f} prob_ok={buy_gate_prob_ok} "
                f"prob_target_ok={buy_gate_prob_target_ok} "
                f"ev_primary={USE_EV_PRIMARY_BUY_GATE} "
                f"ev={expected_net_edge_bps:.3f} "
                f"min_ev={max(float(MIN_REQUIRED_NET_EDGE_BPS), calib_min_ev, float(EV_PRIMARY_MIN_PROJECTED_NET_BPS)):.3f} ev_ok={buy_gate_ev_ok} "
                f"target={target_bps:.3f} "
                f"projected_forward={calibrated_forward_gain_bps:.3f} "
                f"cost={cost_bps:.3f} "
                f"target_cost_ok={buy_gate_target_cost_ok} "
                f"spread={spread_bps:.3f} spread_ok={buy_gate_spread_ok} "
                f"fee_ok={buy_gate_fee_ok} setup_ok={buy_gate_setup_ok} strict_ok={buy_gate_strict_ok} "
                f"calib_reason={profile.reason}"
            )

        if not ok_to_trade:
            position_pct = 0.0

        fee_state = "fee_ok" if fee_available else "fee_missing_trade_closed"

        reason = (
            f"live_score={score:.1f}; display_prob={estimated_prob_up:.3f}; "
            f"calib_min_score={calib_min_score:.1f}; "
            f"calib_min_prob={calib_min_probability:.3f}; "
            f"calib_ev={calib_min_ev:.2f}; "
            f"buy_gate={buy_gate_blocker}; "
            f"{fee_state}; strict={strict_entry.reason}; edge={expected_net_edge_bps:.1f}; "
            f"target={target_bps:.1f}; projected_forward={calibrated_forward_gain_bps:.1f}; "
            f"cost={cost_bps:.1f}; projected_net={expected_net_edge_bps:.1f}; "
            f"time_to_min_profit={profile.calibrated_time_to_min_profit_minutes:.1f}m; "
            f"mom5={momentum_5_bps:.1f}; mom15={momentum_15_bps:.1f}; "
            f"room={room_reason}; {vwap_reason}; {higher_low_reason}; {trend_reason}"
        )

        return LiveSignal(
            ok_to_trade=bool(ok_to_trade),
            score=float(score),
            tier=int(tier),
            reason=reason,
            estimated_prob_up=float(estimated_prob_up),
            position_pct=float(position_pct),
            expected_net_edge_bps=float(expected_net_edge_bps),
            target_bps=float(target_bps),
            cost_bps=float(cost_bps),
            projected_forward_gain_bps=float(calibrated_forward_gain_bps),
            calibrated_time_to_min_profit_minutes=float(profile.calibrated_time_to_min_profit_minutes),
            calibrated_forward_window_minutes=float(profile.calibrated_forward_window_minutes),
            dip_depth_score=float(dip_depth_score),
            dip_speed_score=float(dip_speed_score),
            reversal_score=float(reversal_score),
            support_score=float(support_score),
            room_score=float(room_score),
            regime_score=float(regime_score),
            spread_penalty=float(spread_penalty),
            cost_penalty=float(cost_penalty),
            trend_reason=trend_reason,
            vwap_reason=vwap_reason,
            higher_low_reason=higher_low_reason,
            buy_gate_score_ok=buy_gate_score_ok,
            buy_gate_prob_ok=buy_gate_prob_ok,
            buy_gate_ev_ok=buy_gate_ev_ok,
            buy_gate_fee_ok=buy_gate_fee_ok,
            buy_gate_strict_ok=buy_gate_strict_ok,
            buy_gate_target_cost_ok=buy_gate_target_cost_ok,
            buy_gate_spread_ok=buy_gate_spread_ok,
            buy_gate_calibrated_ok=buy_gate_calibrated_ok,
            buy_gate_tradeable=bool(ok_to_trade),
            buy_gate_blocker=buy_gate_blocker,
        )

    def _position_pct_from_probability(self, estimated_prob_up: float) -> float:
        """
        Map estimated probability to a single-buy percentage of total equity.

        Below PROB_FOR_MIN_SIZE: 0%.
        At PROB_FOR_MIN_SIZE: minimum size.
        At PROB_FOR_MAX_SIZE or higher: maximum single-buy size.
        """
        p = clamp_float(float(estimated_prob_up), 0.0, 1.0)

        if p < float(PROB_FOR_MIN_SIZE):
            return 0.0

        denom = max(1e-9, float(PROB_FOR_MAX_SIZE) - float(PROB_FOR_MIN_SIZE))
        t = clamp_float((p - float(PROB_FOR_MIN_SIZE)) / denom, 0.0, 1.0)
        return lerp_float(MIN_POSITION_PCT_OF_EQUITY, MAX_SINGLE_BUY_PCT_OF_EQUITY, t)

    def _micro_trending_down(self, product_id: str) -> Tuple[bool, str]:
        series = self.live_1m.get(product_id)
        if not series or len(series.candles) < max(5, MICRO_TREND_LOOKBACK_MIN):
            return False, "trend_unknown"

        candles = list(series.candles)
        closes = [float(c.close) for c in candles[-MICRO_TREND_LOOKBACK_MIN:] if float(c.close) > 0]
        if len(closes) < 5:
            return False, "trend_unknown"

        first = closes[0]
        last = closes[-1]
        move_bps = ((last / first) - 1.0) * 10000.0 if first > 0 else 0.0

        lower_highs = 0
        for i in range(2, len(closes)):
            if closes[i] < closes[i - 1] < closes[i - 2]:
                lower_highs += 1

        if move_bps <= MICRO_TREND_DOWN_BPS and lower_highs >= 2:
            return True, f"micro_down move_bps={move_bps:.1f} lower_seq={lower_highs}"

        return False, f"micro_ok move_bps={move_bps:.1f}"

    def _micro_vwap_reclaimed(self, product_id: str, mid: float) -> Tuple[bool, str]:
        avwap = self._compute_anchored_vwap_24h(product_id, now_ts())
        if avwap is None or avwap <= 0:
            return True, "vwap_unknown"

        required = float(avwap) * bps_to_mult(VWAP_RECLAIM_BUFFER_BPS)
        if mid >= required:
            return True, f"vwap_reclaim mid>=avwap+{VWAP_RECLAIM_BUFFER_BPS:.1f}bps"

        return False, f"below_vwap mid={mid:.8f} avwap={avwap:.8f}"

    def _higher_low_confirmed(self, product_id: str) -> Tuple[bool, str]:
        series = self.live_1m.get(product_id)
        if not series or len(series.candles) < 8:
            return False, "higher_low_unknown"

        candles = list(series.candles)[-8:]
        lows = [float(c.low) for c in candles]
        closes = [float(c.close) for c in candles]

        recent_low = min(lows[-3:])
        prior_low = min(lows[:5])
        close_strength = closes[-1] > closes[-2] >= closes[-3]

        if recent_low > prior_low and close_strength:
            return True, "higher_low_confirmed"

        return False, "higher_low_not_confirmed"

    def _risk_pause_active(self) -> bool:
        return now_ts() < float(self.paused_until_ts or 0.0)

    def _reset_daily_pnl_if_needed(self) -> None:
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        if self.daily_pnl_date != today:
            self.daily_pnl_date = today
            self.daily_realized_pnl_usd = 0.0
            self.consecutive_losses = 0

    def _record_realized_trade_result(self, net_pnl_usd: float) -> None:
        self._reset_daily_pnl_if_needed()
        self.daily_realized_pnl_usd += float(net_pnl_usd)

        if net_pnl_usd < 0:
            self.consecutive_losses += 1
            self.paused_until_ts = max(self.paused_until_ts, now_ts() + COOLDOWN_AFTER_LOSS_SEC)
        elif net_pnl_usd > 0:
            self.consecutive_losses = 0

        if self.daily_realized_pnl_usd <= -abs(MAX_DAILY_LOSS_USD):
            self.paused_until_ts = max(self.paused_until_ts, now_ts() + PAUSE_AFTER_DAILY_LOSS_SEC)
            log(f"[risk-pause] daily loss reached ${self.daily_realized_pnl_usd:.2f}; paused until {self.paused_until_ts}")

        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self.paused_until_ts = max(self.paused_until_ts, now_ts() + PAUSE_AFTER_DAILY_LOSS_SEC)
            log(f"[risk-pause] consecutive losses={self.consecutive_losses}; paused until {self.paused_until_ts}")

    def _trade_rate_ok(self, product_id: str) -> bool:
        t = now_ts()
        one_hour_ago = t - 3600.0

        recent_global = [x for x in self.trade_timestamps if x >= one_hour_ago]
        if len(recent_global) >= MAX_TRADES_PER_HOUR:
            return False

        dq = self.product_trade_timestamps.get(product_id)
        if dq is None:
            self.product_trade_timestamps[product_id] = deque(maxlen=200)
            dq = self.product_trade_timestamps[product_id]

        recent_product = [x for x in dq if x >= one_hour_ago]
        if len(recent_product) >= MAX_TRADES_PER_PRODUCT_PER_HOUR:
            return False

        return True

    def _record_trade_timestamp(self, product_id: str) -> None:
        t = now_ts()
        self.trade_timestamps.append(t)
        if product_id not in self.product_trade_timestamps:
            self.product_trade_timestamps[product_id] = deque(maxlen=200)
        self.product_trade_timestamps[product_id].append(t)

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
            round_trip_cost_bps=self._round_trip_cost_bps(
                spread_bps=float(self.tob[product_id].spread_bps) if product_id in self.tob else 0.0
            ),
        )
        return scored.ok, scored.reason, {"score": scored.score}



    # --------------------------------------------------------
    # Live execution helpers (avoid blocking asyncio loop)
    # --------------------------------------------------------
    async def _live_buy_market(self, *, product_id: str, quote_usd: float) -> Any:
        if not isinstance(self.portfolio, LivePortfolio):
            raise RuntimeError("live market buy called without LivePortfolio")
        return await asyncio.to_thread(
            self.portfolio.buy_market,
            product_id,
            float(quote_usd),
        )

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

    async def _live_sell_market(self, *, product_id: str, base_qty: float) -> Any:
        if not isinstance(self.portfolio, LivePortfolio):
            raise RuntimeError("live market sell called without LivePortfolio")
        return await asyncio.to_thread(
            self.portfolio.sell_market,
            product_id,
            float(base_qty),
        )

    async def _execute_live_buy(
        self,
        *,
        product_id: str,
        quote_usd: float,
        bid: float,
        ask: float,
        reason: str,
    ) -> Optional[Tuple[float, float, float, Optional[float], Optional[str]]]:
        """Execute a live buy and return only a Coinbase-confirmed fill."""
        mode = str(ENTRY_EXECUTION_MODE).upper().strip()
        result = None

        try:
            if mode == "MARKET":
                result = await self._live_buy_market(product_id=product_id, quote_usd=quote_usd)
            elif mode == "MAKER":
                result = await self._live_buy_maker(product_id=product_id, quote_usd=quote_usd, bid=bid)
            elif mode == "LIMIT_THEN_MARKET":
                result = await self._live_buy_maker(product_id=product_id, quote_usd=quote_usd, bid=bid)
                fill = self._require_live_fill(result, product_id=product_id, side="BUY")
                if LOG_ORDER_ATTEMPTS:
                    self.olog.log_order(
                        event="BUY_ATTEMPT", product_id=product_id, side="BUY", mode="MAKER_FIRST",
                        requested_quote_usd=quote_usd, result=result, reason=reason,
                    )
                if fill is not None:
                    return fill
                result = await self._live_buy_market(product_id=product_id, quote_usd=quote_usd)
            else:
                raise RuntimeError(f"Invalid ENTRY_EXECUTION_MODE={ENTRY_EXECUTION_MODE}")

            fill = self._require_live_fill(result, product_id=product_id, side="BUY")
            if LOG_ORDER_ATTEMPTS:
                self.olog.log_order(
                    event="BUY_ATTEMPT", product_id=product_id, side="BUY", mode=mode,
                    requested_quote_usd=quote_usd, result=result, reason=reason,
                )
            return fill

        except Exception as e:
            if LOG_ORDER_ATTEMPTS:
                self.olog.log_order(
                    event="BUY_ATTEMPT", product_id=product_id, side="BUY", mode=mode,
                    requested_quote_usd=quote_usd, result=result, reason=reason, raw_error=str(e),
                )
            log(f"[buy-error] {product_id} mode={mode} quote=${quote_usd:.2f}: {e}")
            return None

    async def _execute_live_sell(
        self,
        *,
        product_id: str,
        base_qty: float,
        bid: float,
        ask: float,
        reason: str,
        mode_override: Optional[str] = None,
    ) -> Optional[Tuple[float, float, float, Optional[float], Optional[str]]]:
        """Execute a live sell and return only a Coinbase-confirmed fill."""
        mode = str(mode_override or EXIT_EXECUTION_MODE).upper().strip()
        result = None

        try:
            if mode == "MARKET":
                result = await self._live_sell_market(product_id=product_id, base_qty=base_qty)
            elif mode == "MAKER":
                result = await self._live_sell_maker(product_id=product_id, base_qty=base_qty, ask=ask)
            elif mode == "LIMIT_THEN_MARKET":
                result = await self._live_sell_maker(product_id=product_id, base_qty=base_qty, ask=ask)
                fill = self._require_live_fill(result, product_id=product_id, side="SELL")
                if LOG_ORDER_ATTEMPTS:
                    self.olog.log_order(
                        event="SELL_ATTEMPT", product_id=product_id, side="SELL", mode="MAKER_FIRST",
                        requested_base_qty=base_qty, result=result, reason=reason,
                    )
                if fill is not None:
                    return fill
                result = await self._live_sell_market(product_id=product_id, base_qty=base_qty)
            else:
                raise RuntimeError(f"Invalid live sell execution mode={mode}")

            fill = self._require_live_fill(result, product_id=product_id, side="SELL")
            if LOG_ORDER_ATTEMPTS:
                self.olog.log_order(
                    event="SELL_ATTEMPT", product_id=product_id, side="SELL", mode=mode,
                    requested_base_qty=base_qty, result=result, reason=reason,
                )
            return fill

        except Exception as e:
            if LOG_ORDER_ATTEMPTS:
                self.olog.log_order(
                    event="SELL_ATTEMPT", product_id=product_id, side="SELL", mode=mode,
                    requested_base_qty=base_qty, result=result, reason=reason, raw_error=str(e),
                )
            log(f"[sell-error] {product_id} mode={mode} qty={base_qty:.12f}: {e}")
            return None


    async def _live_refresh_cash(self) -> float:
        """Refresh live cash snapshot in a thread (API calls can block)."""
        if not isinstance(self.portfolio, LivePortfolio):
            raise RuntimeError("Live-only bot requires LivePortfolio.")
        return await asyncio.to_thread(self.portfolio.refresh_cash)



    async def _live_can_afford(self, notional_usd: float, fee_bps: Optional[float] = None) -> bool:
        """
        Confirm there is enough available Coinbase USD for the requested buy.
        Uses Coinbase available USD as the authority.
        """
        if fee_bps is None:
            fee_bps = self._entry_fee_bps_for_mode()

        notional_usd = float(max(0.0, notional_usd))
        if notional_usd <= 0:
            return False

        if not isinstance(self.portfolio, LivePortfolio):
            return False

        def _check() -> bool:
            snap = self.portfolio.refresh_snapshot(force=True, ttl_sec=0.0)
            available = self.portfolio.get_tradable_usd(snapshot=snap)
            required = notional_usd * (1.0 + float(fee_bps) / 10000.0)
            required += float(RESERVE_USD)
            return available >= required

        return bool(await asyncio.to_thread(_check))

    async def _live_refresh_snapshot(self, *, force: bool = True, ttl_sec: float = 0.0) -> Optional[Dict[str, Dict[str, float]]]:
        """Refresh live balances snapshot from Coinbase in a worker thread (non-blocking for event loop)."""
        if not isinstance(self.portfolio, LivePortfolio):
            return None
        return await asyncio.to_thread(self.portfolio.refresh_snapshot, force=bool(force), ttl_sec=float(ttl_sec))

    async def _wait_for_tob_ready(self, timeout_sec: float = STARTUP_TOB_TIMEOUT_SEC) -> None:
        """
        Wait for top-of-book prices for all configured products before startup reconciliation.
        This prevents startup liquidation from skipping products simply because the websocket
        has not received their first bid/ask yet.
        """
        t0 = now_ts()
        last_log = 0.0

        while now_ts() - t0 < float(timeout_sec):
            ready = [
                p for p in PRODUCTS
                if self.tob.get(p) is not None
                and self.tob[p].bid > 0
                and self.tob[p].ask > 0
            ]

            if len(ready) >= len(PRODUCTS):
                log(f"[startup] top-of-book ready for all {len(ready)}/{len(PRODUCTS)} products")
                return

            if now_ts() - last_log >= 5.0:
                missing = [p for p in PRODUCTS if p not in ready]
                log(f"[startup] waiting for top-of-book | ready={len(ready)}/{len(PRODUCTS)} missing={missing}")
                last_log = now_ts()

            await asyncio.sleep(0.25)

        missing = [
            p for p in PRODUCTS
            if self.tob.get(p) is None
            or self.tob[p].bid <= 0
            or self.tob[p].ask <= 0
        ]
        log(f"[startup] top-of-book wait timed out; missing={missing}; liquidation will skip products without valid bid/ask")

    def _live_mid_by_product(self) -> Dict[str, float]:
        mids: Dict[str, float] = {}
        for p in PRODUCTS:
            tob = self.tob.get(p)
            if tob and tob.mid > 0:
                mids[p] = float(tob.mid)
        return mids

    async def _adopt_existing_coinbase_holdings(self) -> None:
        """
        Convert existing Coinbase balances into local PositionLot objects.
        This gives the bot the ability to manage/sell them.

        Important:
        - The entry price is approximated using current mid.
        - This is not true historical cost basis.
        - For accurate tax/P&L, Coinbase fills/history are still the authority.
        """
        if not isinstance(self.portfolio, LivePortfolio):
            return

        snap = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
        if not snap:
            return

        adopted = 0
        for product_id in PRODUCTS:
            tob = self.tob.get(product_id)
            if not tob or tob.mid <= 0:
                continue

            qty = self.portfolio.get_product_total_qty(product_id, snapshot=snap)
            if qty <= 1e-12:
                self.positions[product_id] = []
                continue

            approx_entry = float(tob.mid)
            self.positions[product_id] = [
                PositionLot(
                    qty=float(qty),
                    price=approx_entry,
                    tier=TIER_LOW,
                    score=0.0,
                    meta={
                        "coinbase_existing": True,
                        "scalp_done": False,
                        "core_done": False,
                    },
                )
            ]
            self.position_start_ts[product_id] = now_ts()
            self.position_entry_price[product_id] = approx_entry
            self.peak_bid[product_id] = float(tob.bid)
            adopted += 1

        log(f"[startup] adopted {adopted} existing Coinbase holdings as managed positions")

    async def _liquidate_existing_coinbase_holdings(self) -> None:
        """
        Sell existing available balances for configured PRODUCTS on startup.

        This intentionally uses Coinbase available balance as the sellable source of truth.
        Total balance may include hold/locked amounts that cannot be sold.
        """
        if not isinstance(self.portfolio, LivePortfolio):
            return

        snap = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
        if not snap:
            log("[startup-liquidation] no Coinbase snapshot available")
            return

        sold_count = 0
        skipped_count = 0

        for product_id in PRODUCTS:
            tob = self.tob.get(product_id)
            if not tob or tob.bid <= 0:
                log(f"[startup-liquidation] skipping {product_id}: no bid available")
                skipped_count += 1
                continue

            available_qty = self.portfolio.get_product_available_qty(product_id, snapshot=snap)
            total_qty = self.portfolio.get_product_total_qty(product_id, snapshot=snap)
            if available_qty <= 1e-12 and total_qty <= 1e-12:
                continue

            est_usd = float(available_qty) * float(tob.bid)

            if available_qty <= 1e-12:
                log(f"[startup-liquidation] {product_id}: total exists but available=0; cannot sell held/locked balance")
                skipped_count += 1
                continue

            if est_usd < MIN_STARTUP_LIQUIDATION_USD:
                log(
                    f"[startup-liquidation] skipping {product_id}: estimated value ${est_usd:.4f} "
                    f"is below MIN_STARTUP_LIQUIDATION_USD=${MIN_STARTUP_LIQUIDATION_USD:.2f}; adopting as managed holding"
                )

                if total_qty > 1e-12 and tob and tob.mid > 0:
                    approx_entry = float(tob.mid)
                    self.positions[product_id] = [
                        PositionLot(
                            qty=float(total_qty),
                            price=approx_entry,
                            tier=TIER_LOW,
                            score=0.0,
                            meta={
                                "coinbase_existing_tiny_balance": True,
                                "scalp_done": False,
                                "core_done": False,
                            },
                        )
                    ]
                    self.position_start_ts[product_id] = now_ts()
                    self.position_entry_price[product_id] = approx_entry
                    self.peak_bid[product_id] = float(tob.bid)

                skipped_count += 1
                continue

            log(
                f"[startup-liquidation] selling {product_id}: "
                f"available_qty={available_qty:.12f}, est_usd=${est_usd:.4f}"
            )

            try:
                startup_exit_mode = "MARKET" if STARTUP_LIQUIDATION_USE_MARKET else EXIT_EXECUTION_MODE
                fill = await self._execute_live_sell(
                    product_id=product_id,
                    base_qty=float(available_qty),
                    bid=float(tob.bid),
                    ask=float(tob.ask),
                    reason="startup_liquidate_existing_coinbase_balance",
                    mode_override=startup_exit_mode,
                )
                if fill is None:
                    log(f"[startup-liquidation] sell did not fill for {product_id}")
                    skipped_count += 1

                    # If the sell failed, adopt the remaining Coinbase balance so the normal
                    # exit logic can still see and manage it instead of leaving it invisible
                    # to self.positions.
                    try:
                        snap_after_fail = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                        remaining_qty = self.portfolio.get_product_total_qty(product_id, snapshot=snap_after_fail or {})
                        tob_after_fail = self.tob.get(product_id)
                        if remaining_qty > 1e-12 and tob_after_fail and tob_after_fail.mid > 0:
                            approx_entry = float(tob_after_fail.mid)
                            self.positions[product_id] = [
                                PositionLot(
                                    qty=float(remaining_qty),
                                    price=approx_entry,
                                    tier=TIER_LOW,
                                    score=0.0,
                                    meta={
                                        "coinbase_existing_after_failed_liquidation": True,
                                        "scalp_done": False,
                                        "core_done": False,
                                    },
                                )
                            ]
                            self.position_start_ts[product_id] = now_ts()
                            self.position_entry_price[product_id] = approx_entry
                            self.peak_bid[product_id] = float(tob_after_fail.bid)
                            log(f"[startup-liquidation] adopted remaining {product_id} qty={remaining_qty:.12f} after failed sell")
                    except Exception as adopt_err:
                        log(f"[startup-liquidation] could not adopt remaining {product_id} after failed sell: {adopt_err}")

                    continue

                filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
                self.tlog.log_trade(
                    event="STARTUP_LIQUIDATION",
                    product_id=product_id,
                    side="SELL",
                    qty=float(filled_qty),
                    price=float(avg_px),
                    fee_usd_val=float(fee_val),
                    gross_pnl_usd=0.0,
                    net_pnl_usd=-float(fee_val),
                    entry_price=None,
                    exit_price=float(avg_px),
                    weekly_bias=None,
                    note="startup_liquidate_existing_coinbase_balance",
                    filled_notional_usd=float(filled_notional),
                    exit_role="startup_liquidation",
                )

                self.positions[product_id] = []
                self.position_start_ts[product_id] = None
                self.position_entry_price[product_id] = None
                self.peak_bid[product_id] = None
                self.scale_add_count[product_id] = 0
                sold_count += 1

                snap = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)

            except Exception as e:
                log(f"[startup-liquidation] error selling {product_id}: {e}")
                skipped_count += 1

        final_snap = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
        final_cash = self.portfolio.get_tradable_usd(snapshot=final_snap or {})
        final_equity = self.portfolio.compute_equity_usd(
            mid_by_product=self._live_mid_by_product(),
            snapshot=final_snap,
        )

        log(
            f"[startup-liquidation] complete | sold={sold_count} skipped={skipped_count} "
            f"| tradable_usd=${final_cash:.2f} | equity_usd≈${final_equity:.2f}"
        )

    async def _startup_portfolio_reconcile(self) -> None:
        """One-time live startup portfolio handling."""
        if not isinstance(self.portfolio, LivePortfolio):
            return

        await self._wait_for_tob_ready(timeout_sec=STARTUP_TOB_TIMEOUT_SEC)

        snap = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
        if not snap:
            log("[startup] unable to read Coinbase balances")
            return

        cash = self.portfolio.get_tradable_usd(snapshot=snap)
        equity = self.portfolio.compute_equity_usd(
            mid_by_product=self._live_mid_by_product(),
            snapshot=snap,
        )
        log(f"[startup] Coinbase snapshot before mode={LIVE_STARTUP_MODE}: tradable_usd=${cash:.2f}, equity_usd≈${equity:.2f}")

        mode = str(LIVE_STARTUP_MODE).upper().strip()

        if mode == "LIQUIDATE_EXISTING":
            await self._liquidate_existing_coinbase_holdings()
        elif mode == "ADOPT_EXISTING":
            await self._adopt_existing_coinbase_holdings()
        elif mode == "IGNORE_EXISTING":
            log("[startup] ignoring existing holdings for management, but still using Coinbase for cash/equity")
        else:
            raise RuntimeError(
                "Invalid LIVE_STARTUP_MODE. Use 'LIQUIDATE_EXISTING', 'ADOPT_EXISTING', or 'IGNORE_EXISTING'."
            )

    async def run(self) -> None:
        """Launch websocket first, reconcile Coinbase portfolio, then start trading loops."""
        log("[run] TradingBot.run() started")
        log(f"[run] mode=LIVE_ONLY products={PRODUCTS}")

        log("[run] preloading micro history")
        await self.preload_micro_history()

        log("[run] starting websocket task first")
        ws_task = asyncio.create_task(self.ws_loop())

        log("[run] waiting for initial top-of-book data")
        await self._wait_for_tob_ready(timeout_sec=20.0)

        await self._refresh_coinbase_fee_tier_if_needed(force=True)

        log("[run] calibrating products before live trading")
        await self.calibrate_products_on_startup()

        log("[run] reconciling live Coinbase portfolio before trading")
        await self._startup_portfolio_reconcile()

        log("[run] starting macro / evaluation / telemetry tasks")
        tasks = [
            ws_task,
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
                    close_timeout=5,
                    max_queue=1024,
                ) as ws:
                    log(f"[ws] connected url={WS_MARKET_URL} products={PRODUCTS}")
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
                    log("[ws] subscribed successfully")
                    last_msg_ts = now_ts()
                    async for message in ws:
                        last_msg_ts = now_ts()
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
                log_exception("[ws] websocket loop error/reconnect", e)
                log(f"[ws] reconnecting in {WS_RECONNECT_DELAY_SEC}s")
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
                    log(f"[macro] week empty for {product}")
                if not candles_day:
                    log(f"[macro] day empty for {product} (live_rows={len(live_rows)})")
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
                log_exception("[macro] write failed", e)
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

        if SOURCE_OF_TRUTH_COINBASE and isinstance(self.portfolio, LivePortfolio):
            try:
                snap = self.portfolio.refresh_snapshot(force=False, ttl_sec=1.25)
                qty = self.portfolio.get_product_total_qty(product_id, snapshot=snap)
                return float(qty) * float(tob.mid)
            except Exception:
                pass

        return float(sum(l.qty for l in self.positions.get(product_id, [])) * tob.mid)

    def _current_total_exposure_usd(self) -> float:
        total = 0.0
        for product_id in PRODUCTS:
            total += self._current_product_exposure_usd(product_id)
        return float(total)

    def _open_position_count(self) -> int:
        if SOURCE_OF_TRUTH_COINBASE and isinstance(self.portfolio, LivePortfolio):
            try:
                snap = self.portfolio.refresh_snapshot(force=False, ttl_sec=1.25)
                count = 0
                for product_id in PRODUCTS:
                    qty = self.portfolio.get_product_total_qty(product_id, snapshot=snap)
                    if qty > 1e-12:
                        count += 1
                return count
            except Exception:
                pass

        return sum(1 for lots in self.positions.values() if sum(l.qty for l in lots) > 0)

    def _estimate_position_unrealized_net_bps(self, product_id: str, bid: float) -> Optional[float]:
        lots = self.positions.get(product_id, [])
        qty = sum(l.qty for l in lots)
        if qty <= 0:
            return None
        cost = sum(l.qty * l.price for l in lots)
        if cost <= 0:
            return None
        avg_entry = cost / qty
        spread_bps = 0.0
        tob = self.tob.get(product_id)
        if tob and tob.mid > 0:
            spread_bps = tob.spread_bps
        gross_bps = ((float(bid) / float(avg_entry)) - 1.0) * 10000.0
        exit_cost_bps = (
            self._exit_fee_bps_for_mode()
            + EST_SLIPPAGE_BPS
            + ROUND_TRIP_SAFETY_BPS
            + float(spread_bps)
        )
        return float(gross_bps - exit_cost_bps)

    def _position_score_for_rotation(self, product_id: str) -> float:
        lots = self.positions.get(product_id, [])
        if not lots:
            return 0.0
        try:
            return float(max(l.score for l in lots))
        except Exception:
            return 0.0

    def _position_prob_for_rotation(self, product_id: str) -> float:
        lots = self.positions.get(product_id, [])
        if not lots:
            return 0.0
        # Existing lots may not have probability in meta yet, so fall back from score.
        probs = []
        for lot in lots:
            try:
                probability = float(lot.meta.get("estimated_prob_up"))
            except Exception:
                probability = self._estimate_prob_up_from_candidate(
                    score=float(lot.score),
                    expected_net_edge_bps=MIN_REQUIRED_NET_EDGE_BPS,
                    spread_bps=0.0,
                )
            probs.append(probability)
        return float(max(probs)) if probs else 0.0

    async def _try_rotate_capital_for_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        needed_cash_usd: float,
        equity_usd: float,
    ) -> float:
        """
        If a stronger setup appears but available cash is insufficient, sell weaker
        existing positions only if they are net-positive after fees.

        Returns additional cash likely freed.
        """
        if not ENABLE_PROFITABLE_ROTATION:
            return 0.0
        if needed_cash_usd <= 0:
            return 0.0
        if not isinstance(self.portfolio, LivePortfolio):
            return 0.0

        new_product = str(candidate["product_id"])
        new_prob = float(candidate.get("estimated_prob_up", 0.0))
        new_score = float(candidate.get("score", 0.0))

        rotation_candidates: List[Tuple[float, str, float, float]] = []
        snapshot: Optional[Dict[str, Dict[str, float]]] = None
        if SOURCE_OF_TRUTH_COINBASE:
            try:
                snapshot = self.portfolio.refresh_snapshot(force=False, ttl_sec=1.25)
            except Exception:
                snapshot = None

        for held_product in PRODUCTS:
            if held_product == new_product:
                continue

            tob = self.tob.get(held_product)
            if not tob or tob.bid <= 0 or tob.ask <= 0:
                continue

            held_qty = sum(l.qty for l in self.positions.get(held_product, []))
            if SOURCE_OF_TRUTH_COINBASE and snapshot is not None:
                held_qty = self.portfolio.get_product_available_qty(held_product, snapshot=snapshot)

            if held_qty <= 1e-12:
                continue

            held_prob = self._position_prob_for_rotation(held_product)
            held_score = self._position_score_for_rotation(held_product)
            net_bps = self._estimate_position_unrealized_net_bps(held_product, tob.bid)

            if net_bps is None or net_bps < ROTATION_MIN_NET_PROFIT_BPS:
                continue

            probability_advantage = new_prob - held_prob
            score_advantage = new_score - held_score
            if (
                probability_advantage < ROTATION_MIN_NEW_PROB_ADVANTAGE
                and score_advantage < ROTATION_MIN_NEW_SCORE_ADVANTAGE
            ):
                continue

            est_value = float(held_qty) * float(tob.bid)
            # Lower priority number sells first: weakest probability, then lower score.
            priority = held_prob + (held_score / 200.0)
            rotation_candidates.append((priority, held_product, held_qty, est_value))

        rotation_candidates.sort(key=lambda item: item[0])

        freed = 0.0
        for _priority, held_product, held_qty, est_value in rotation_candidates:
            if freed >= needed_cash_usd:
                break

            tob = self.tob.get(held_product)
            if not tob:
                continue

            sell_qty = float(held_qty) * float(ROTATION_SELL_FRACTION)
            if sell_qty <= 0:
                continue

            log(
                f"[rotation] selling weaker profitable {held_product} to fund {new_product} "
                f"sell_qty={sell_qty:.12f} est_value=${est_value:.2f} "
                f"new_prob={new_prob:.3f} equity=${equity_usd:.2f}"
            )

            fill = await self._execute_live_sell(
                product_id=held_product,
                base_qty=sell_qty,
                bid=float(tob.bid),
                ask=float(tob.ask),
                reason=f"profitable_rotation_to_{new_product}",
            )

            if fill is None:
                continue

            filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
            filled_qty = float(filled_qty)
            avg_px = float(avg_px)
            fee = float(fee_val)
            notional_usd = (
                float(filled_notional)
                if filled_notional is not None
                else filled_qty * avg_px
            )

            lots = self.positions.get(held_product, [])
            fifo_cost, fifo_avg_entry = self._fifo_cost_basis(list(lots), filled_qty)
            pnl_gross = float(notional_usd) - float(fifo_cost)

            self.tlog.log_trade(
                event="SELL",
                product_id=held_product,
                side="SELL",
                qty=filled_qty,
                price=avg_px,
                fee_usd_val=fee,
                gross_pnl_usd=pnl_gross,
                net_pnl_usd=pnl_gross - fee,
                entry_price=(fifo_avg_entry if fifo_avg_entry is not None else None),
                exit_price=avg_px,
                weekly_bias=None,
                note=f"profitable_rotation_to_{new_product}",
                filled_notional_usd=notional_usd,
                exit_role="profitable_rotation",
            )

            self._record_trade_timestamp(held_product)
            self._record_realized_trade_result(pnl_gross - fee)
            self._fifo_reduce_lots(held_product, filled_qty)

            freed += max(0.0, notional_usd - fee)

            if sum(l.qty for l in self.positions.get(held_product, [])) <= 1e-12:
                self.positions[held_product] = []
                self.position_start_ts[held_product] = None
                self.position_entry_price[held_product] = None
                self.peak_bid[held_product] = None
                self.scale_add_count[held_product] = 0

            await self._live_refresh_snapshot(force=True, ttl_sec=0.0)

        return float(freed)

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
        estimated_prob_up: Optional[float] = None,
    ) -> float:
        """
        Compute buy size.

        New preferred behavior:
        - total equity = cash + positions
        - single buy = 5%–20% of total equity based on estimated probability
        - max exposure per product = 50% of total equity
        - never spend unavailable cash
        """
        available_cash_usd = float(max(0.0, available_cash_usd))
        current_equity_usd = float(max(0.0, current_equity_usd))
        current_product_exposure_usd = float(max(0.0, current_product_exposure_usd))

        if available_cash_usd <= 0 or current_equity_usd <= 0:
            return 0.0

        spendable_cash = max(0.0, available_cash_usd - float(RESERVE_USD))
        if spendable_cash <= 0:
            return 0.0

        if USE_EQUITY_PERCENT_POSITION_SIZING:
            prob = estimated_prob_up
            if prob is None:
                # Fallback from score only.
                prob = self._estimate_prob_up_from_candidate(
                    score=float(candidate_score),
                    expected_net_edge_bps=MIN_REQUIRED_NET_EDGE_BPS,
                    spread_bps=0.0,
                )

            single_buy_pct = self._position_pct_from_probability(float(prob))
            proposed = current_equity_usd * float(single_buy_pct)

            max_product_exposure = current_equity_usd * float(MAX_EXPOSURE_PER_PRODUCT_PCT_OF_EQUITY)
            remaining_product_room = max(0.0, max_product_exposure - current_product_exposure_usd)

            proposed = min(proposed, remaining_product_room, spendable_cash)

            if proposed < max(float(MIN_ENTRY_USD), float(MIN_LIVE_ORDER_USD)):
                return 0.0

            return float(proposed)

        if USE_FIXED_ENTRY_SIZE_USD:
            max_product_exposure = current_equity_usd * float(MAX_EXPOSURE_PER_PRODUCT_PCT_OF_EQUITY)
            remaining_product_room = max(0.0, max_product_exposure - current_product_exposure_usd)
            proposed = min(float(ENTRY_SIZE_USD), spendable_cash, remaining_product_room)
            if proposed < max(float(MIN_ENTRY_USD), float(MIN_LIVE_ORDER_USD)):
                return 0.0
            return float(proposed)

        # Legacy dynamic fallback. Kept only as backup.
        if candidate_score >= HIGH_SCORE_UTIL_THRESHOLD:
            util_target = TARGET_UTIL_MAX
        elif candidate_score >= MID_SCORE_UTIL_THRESHOLD:
            util_target = TARGET_UTIL_MID
        else:
            util_target = TARGET_UTIL_MIN

        if strong_candidate_count >= 5:
            util_target = min(TARGET_UTIL_MAX, util_target + 0.05)

        target_gross_exposure = current_equity_usd * util_target
        deployable_gap = max(0.0, target_gross_exposure - current_total_exposure_usd)
        score_weight = max(0.35, min(1.0, candidate_score / 100.0))

        if candidate_score >= 80.0:
            base_alloc_frac = 0.10
        elif candidate_score >= 64.0:
            base_alloc_frac = 0.07
        else:
            base_alloc_frac = 0.045

        proposed = available_cash_usd * base_alloc_frac * score_weight
        if deployable_gap > 0:
            proposed = min(proposed, deployable_gap)

        max_product_exposure = current_equity_usd * float(MAX_EXPOSURE_PER_PRODUCT_PCT_OF_EQUITY)
        remaining_room = max(0.0, max_product_exposure - current_product_exposure_usd)
        proposed = min(proposed, remaining_room, spendable_cash)

        if proposed < max(float(MIN_ENTRY_USD), float(MIN_LIVE_ORDER_USD)):
            return 0.0

        return float(proposed)

    async def eval_loop(self) -> None:
        while not self._stop_event.is_set():
            ts_now = now_ts()
            loop_gap = ts_now - float(self.last_loop_lag_check_ts or ts_now)
            if loop_gap > EVENT_LOOP_LAG_WARN_SEC:
                log(f"[lag] eval_loop gap={loop_gap:.2f}s; possible blocking work or REST delay")
            self.last_loop_lag_check_ts = ts_now
            try:
                await self._refresh_coinbase_fee_tier_if_needed(force=False)
            except Exception as e:
                log_exception("[fee-tier] trading paused because real Coinbase fees are unavailable", e)
                await asyncio.sleep(EVAL_TICK_SEC)
                continue

            if (
                ENABLE_WALK_FORWARD_CALIBRATION
                and not self.live_recalibration_running
                and ts_now - self.last_live_calibration_ts >= LIVE_RECALIBRATION_EVERY_SEC
            ):
                self.last_live_calibration_ts = ts_now
                self.live_recalibration_running = True

                async def _recalibrate_in_thread() -> None:
                    try:
                        await asyncio.to_thread(self._run_live_recalibration)
                    except Exception as e:
                        log(f"[calibration] live recalibration failed: {e}")
                    finally:
                        self.live_recalibration_running = False

                asyncio.create_task(_recalibrate_in_thread())

            if ts_now - self.last_heartbeat_ts >= 30.0:
                try:
                    cash_usd = float(self.portfolio.cash_usd)
                except Exception:
                    cash_usd = float("nan")

                log(f"[heartbeat] running | products={len(PRODUCTS)} | cash_usd={cash_usd:.2f}")
                self.last_heartbeat_ts = ts_now

            warmup_done = (ts_now - self.bot_start_ts) >= FIRST_BUY_DELAY_SEC

            snap_live: Optional[Dict[str, Dict[str, float]]] = None
            try:
                snap_live = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                cash_usd = float(self.portfolio.get_tradable_usd(snapshot=snap_live))
                equity_usd = float(
                    self.portfolio.compute_equity_usd(
                        mid_by_product=self._live_mid_by_product(),
                        snapshot=snap_live,
                    )
                )
            except Exception as e:
                log(f"[eval] Coinbase account refresh failed; skipping evaluation: {e}")
                await asyncio.sleep(EVAL_TICK_SEC)
                continue

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
                    required_net_exit_px = required_exit_price_for_net_gain(
                        effective_entry_price=avg_entry_price,
                        exit_fee_bps=self._exit_fee_bps_for_mode(),
                        est_slippage_bps=EST_SLIPPAGE_BPS,
                        est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                        min_net_gain_bps=max(
                            MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                            MIN_NET_GAIN_AFTER_FEES_BPS,
                        ),
                    )

                    # Arm scalp target instead of instantly selling it.
                    if (
                        bid >= max(targets["scalp_target"], required_net_exit_px)
                        and not lot_meta.get("scalp_done", False)
                        and not lot_meta.get("scalp_armed", False)
                    ):
                        if can_exit_net_positive(
                            entry_price=avg_entry_price,
                            exit_price=bid,
                            taker_fee_bps=self._exit_fee_bps_for_mode(),
                            est_slippage_bps=EST_SLIPPAGE_BPS,
                            est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                            min_net_profit_bps=max(
                                MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                                MIN_NET_GAIN_AFTER_FEES_BPS,
                            ),
                        ):
                            lot_meta["scalp_armed"] = True
                            lot_meta["scalp_arm_price"] = float(bid)
                            lot_meta["scalp_arm_peak"] = float(bid)
                            log(f"[exit-arm] {product_id} scalp armed at {bid:.8f}")

                    # Arm core target instead of instantly selling it.
                    if (
                        bid >= max(targets["core_target"], required_net_exit_px)
                        and not lot_meta.get("core_done", False)
                        and not lot_meta.get("core_armed", False)
                    ):
                        if can_exit_net_positive(
                            entry_price=avg_entry_price,
                            exit_price=bid,
                            taker_fee_bps=self._exit_fee_bps_for_mode(),
                            est_slippage_bps=EST_SLIPPAGE_BPS,
                            est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                            min_net_profit_bps=max(
                                MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                                MIN_NET_GAIN_AFTER_FEES_BPS,
                            ),
                        ):
                            lot_meta["core_armed"] = True
                            lot_meta["core_arm_price"] = float(bid)
                            lot_meta["core_arm_peak"] = float(bid)
                            log(f"[exit-arm] {product_id} core armed at {bid:.8f}")

                    # If scalp is armed, keep tracking the highest bid after arming.
                    # Sell only after a pullback from that post-arm high.
                    if lot_meta.get("scalp_armed", False) and not lot_meta.get("scalp_done", False):
                        scalp_peak = float(lot_meta.get("scalp_arm_peak") or bid)
                        if bid > scalp_peak:
                            scalp_peak = float(bid)
                            lot_meta["scalp_arm_peak"] = scalp_peak

                        scalp_drawdown = max(0.0, (scalp_peak - bid) / scalp_peak) if scalp_peak > 0 else 0.0

                        profile = self.calibration_profiles.get(
                            product_id,
                            ProductCalibrationProfile(product_id=product_id),
                        )
                        scalp_pullback_pct = float(profile.scalp_pullback_pct)

                        if scalp_drawdown >= scalp_pullback_pct:
                            if can_exit_net_positive(
                                entry_price=avg_entry_price,
                                exit_price=bid,
                                taker_fee_bps=self._exit_fee_bps_for_mode(),
                                est_slippage_bps=EST_SLIPPAGE_BPS,
                                est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                                min_net_profit_bps=max(
                                    MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                                    MIN_NET_GAIN_AFTER_FEES_BPS,
                                ),
                            ):
                                sell_qty = max(sell_qty, position_qty * exit_plan["scalp_frac"])
                                exit_reason = (
                                    f"scalp_armed_pullback peak={scalp_peak:.8f} "
                                    f"drawdown={scalp_drawdown:.4%}"
                                )
                                exit_role = "scalp_armed_release"
                                lot_meta["scalp_done"] = True
                                lot_meta["scalp_armed"] = False

                    # If core is armed, keep tracking the highest bid after arming.
                    # Sell only after a larger pullback from that post-arm high.
                    if lot_meta.get("core_armed", False) and not lot_meta.get("core_done", False):
                        core_peak = float(lot_meta.get("core_arm_peak") or bid)
                        if bid > core_peak:
                            core_peak = float(bid)
                            lot_meta["core_arm_peak"] = core_peak

                        core_drawdown = max(0.0, (core_peak - bid) / core_peak) if core_peak > 0 else 0.0

                        profile = self.calibration_profiles.get(
                            product_id,
                            ProductCalibrationProfile(product_id=product_id),
                        )
                        core_pullback_pct = float(profile.core_pullback_pct)

                        if core_drawdown >= core_pullback_pct:
                            if can_exit_net_positive(
                                entry_price=avg_entry_price,
                                exit_price=bid,
                                taker_fee_bps=self._exit_fee_bps_for_mode(),
                                est_slippage_bps=EST_SLIPPAGE_BPS,
                                est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                                min_net_profit_bps=max(
                                    MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                                    MIN_NET_GAIN_AFTER_FEES_BPS,
                                ),
                            ):
                                sell_qty = max(sell_qty, position_qty * exit_plan["core_frac"])
                                exit_reason = (
                                    f"core_armed_pullback peak={core_peak:.8f} "
                                    f"drawdown={core_drawdown:.4%}"
                                )
                                exit_role = "core_armed_release"
                                lot_meta["core_done"] = True
                                lot_meta["core_armed"] = False

                    remaining_qty = sum(l.qty for l in self.positions.get(product_id, []))
                    peak_bid = float(self.peak_bid.get(product_id) or bid)
                    if peak_bid <= 0:
                        peak_bid = bid
                    if bid > peak_bid:
                        peak_bid = bid
                        self.peak_bid[product_id] = peak_bid
                    drawdown_from_peak = max(0.0, (peak_bid - bid) / peak_bid) if peak_bid and peak_bid > 0 else 0.0
                    peak_profit = max(0.0, (peak_bid - avg_entry_price) / avg_entry_price) if peak_bid and avg_entry_price > 0 else 0.0

                    # True stop-loss / protective exit: always allowed
                    if ENABLE_HARD_PEAK_STOP and drawdown_from_peak >= HARD_PEAK_STOP_PCT:
                        sell_qty = remaining_qty
                        exit_reason = "hard_peak_stop"
                        exit_role = "hard_peak_stop"

                    # Discretionary trailing profit exit: only allowed if net positive after costs
                    elif peak_profit >= TRAIL_ARM_PCT and drawdown_from_peak >= TRAIL_DRAWDOWN_PCT:
                        if can_exit_net_positive(
                            entry_price=avg_entry_price,
                            exit_price=bid,
                            taker_fee_bps=self._exit_fee_bps_for_mode(),
                            est_slippage_bps=EST_SLIPPAGE_BPS,
                            est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                            min_net_profit_bps=MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                        ):
                            sell_qty = remaining_qty
                            exit_reason = "armed_trailing_drawdown"
                            exit_role = "runner_trail_exit"

                    # Time stop / no-progress stop / invalidation stop.
                    pos_start = self.position_start_ts.get(product_id)
                    if pos_start is not None and position_qty > 0 and avg_entry_price and avg_entry_price > 0:
                        age_sec = ts_now - float(pos_start)
                        unrealized_bps = ((bid / avg_entry_price) - 1.0) * 10000.0

                        if (
                            ENABLE_NO_PROGRESS_STOP
                            and age_sec >= NO_PROGRESS_STOP_SEC
                            and unrealized_bps < MIN_PROGRESS_BPS_BEFORE_TIME_STOP
                        ):
                            exit_reason = f"no_progress_stop age_sec={age_sec:.0f} unrealized_bps={unrealized_bps:.1f}"
                            exit_role = "no_progress_stop"
                            sell_qty = position_qty

                        if ENABLE_TIME_STOP and age_sec >= TIME_STOP_SEC:
                            exit_reason = f"time_stop age_sec={age_sec:.0f} unrealized_bps={unrealized_bps:.1f}"
                            exit_role = "time_stop"
                            sell_qty = position_qty

                        if ENABLE_INVALIDATION_STOP and levels_day and getattr(levels_day, "support_zone_low", 0) > 0:
                            invalidation_px = float(levels_day.support_zone_low) * bps_to_mult(-INVALIDATION_BUFFER_BPS)
                            if bid < invalidation_px:
                                exit_reason = f"invalidation_stop bid<{invalidation_px:.8f}"
                                exit_role = "invalidation_stop"
                                sell_qty = position_qty

                    sell_qty = min(position_qty, max(0.0, sell_qty))
                    if sell_qty > 0:
                        log(
                            f"[sell-attempt] {product_id} "
                            f"qty={sell_qty:.12f} reason={exit_reason} role={exit_role} "
                            f"bid={bid:.8f} ask={ask:.8f} "
                            f"avg_entry={avg_entry_price:.8f}"
                        )
                        notional_usd = sell_qty * bid
                        exec_price = bid
                        fee = 0.0
                        filled_notional = None
                        fill = await self._execute_live_sell(
                            product_id=product_id,
                            base_qty=sell_qty,
                            bid=bid,
                            ask=ask,
                            reason=exit_reason or "sell",
                        )
                        if fill is not None:
                            filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
                            log(
                                f"[sell-success] {product_id} "
                                f"qty={float(filled_qty):.12f} avg_px={float(avg_px):.8f} "
                                f"fee={float(fee_val):.6f} role={exit_role} reason={exit_reason}"
                            )
                            sell_qty = min(float(sell_qty), float(filled_qty))
                            exec_price = float(avg_px)
                            fee = float(fee_val)
                            notional_usd = float(filled_notional) if filled_notional is not None else float(sell_qty) * float(avg_px)
                        else:
                            sell_qty = 0.0

                        if sell_qty > 0:
                            fifo_cost, fifo_avg_entry = self._fifo_cost_basis(list(lots), sell_qty)
                            pnl_gross = float(notional_usd) - float(fifo_cost)
                            self.tlog.log_trade(
                                event="SELL", product_id=product_id, side="SELL", qty=sell_qty, price=exec_price,
                                fee_usd_val=fee, gross_pnl_usd=pnl_gross, net_pnl_usd=pnl_gross - fee,
                                entry_price=(fifo_avg_entry if fifo_avg_entry is not None else avg_entry_price),
                                exit_price=exec_price, weekly_bias=weekly_bias, note=exit_reason or "sell",
                                filled_notional_usd=(float(filled_notional) if filled_notional is not None else None),
                                exit_role=exit_role or "risk_off",
                            )
                            self._record_trade_timestamp(product_id)
                            self._record_realized_trade_result(pnl_gross - fee)
                            self._fifo_reduce_lots(product_id, sell_qty)

                live_signal = self._build_live_signal(
                    product_id=product_id,
                    mid=mid,
                    spread_bps=spread_bps,
                    levels_day=levels_day,
                    levels_week=levels_week,
                    minute_candles=minute_candles,
                    weekly_bias=weekly_bias,
                    sigma_bps=sigma_bps,
                )

                scored = EntryScore(
                    ok=live_signal.ok_to_trade,
                    score=live_signal.score,
                    tier=live_signal.tier,
                    reason=live_signal.reason,
                    dip_depth_score=live_signal.dip_depth_score,
                    dip_speed_score=live_signal.dip_speed_score,
                    reversal_score=live_signal.reversal_score,
                    support_score=live_signal.support_score,
                    room_score=live_signal.room_score,
                    regime_score=live_signal.regime_score,
                    spread_penalty=live_signal.spread_penalty,
                    cost_penalty=live_signal.cost_penalty,
                    expected_net_edge_bps=live_signal.expected_net_edge_bps,
                )

                estimated_prob_up = live_signal.estimated_prob_up
                position_pct = live_signal.position_pct
                target_bps = live_signal.target_bps
                cost_bps = live_signal.cost_bps

                if (
                    live_signal.ok_to_trade
                    and warmup_done
                    and not self._risk_pause_active()
                    and self._trade_rate_ok(product_id)
                    and self._open_position_count() < MAX_OPEN_POSITIONS
                ):
                    candidates.append({
                        "product_id": product_id,
                        "mid": mid,
                        "bid": bid,
                        "ask": ask,
                        "spread_bps": spread_bps,
                        "score": scored.score,
                        "tier": scored.tier,
                        "entry_reason": scored.reason,
                        "entry_score_obj": scored,
                        "expected_net_edge_bps": scored.expected_net_edge_bps,
                        "estimated_prob_up": float(estimated_prob_up),
                        "position_pct": float(position_pct),
                        "target_bps": float(target_bps),
                        "cost_bps": float(cost_bps),
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
                    estimated_prob_up=float(estimated_prob_up),
                    position_pct=float(position_pct),
                    target_bps=float(target_bps),
                    projected_forward_gain_bps=live_signal.projected_forward_gain_bps,
                    cost_bps=float(cost_bps),
                    calibrated_time_to_min_profit_minutes=live_signal.calibrated_time_to_min_profit_minutes,
                    calibrated_forward_window_minutes=live_signal.calibrated_forward_window_minutes,
                    current_maker_fee_bps=self.current_maker_fee_bps,
                    current_taker_fee_bps=self.current_taker_fee_bps,
                    fee_tier_reason=self.last_fee_tier_reason,
                    dip_depth_score=scored.dip_depth_score,
                    dip_speed_score=scored.dip_speed_score,
                    reversal_score=scored.reversal_score,
                    support_score=scored.support_score,
                    room_score=scored.room_score,
                    regime_score=scored.regime_score,
                    spread_penalty=scored.spread_penalty,
                    cost_penalty=scored.cost_penalty,
                    buy_gate_score_ok=live_signal.buy_gate_score_ok,
                    buy_gate_prob_ok=live_signal.buy_gate_prob_ok,
                    buy_gate_ev_ok=live_signal.buy_gate_ev_ok,
                    buy_gate_fee_ok=live_signal.buy_gate_fee_ok,
                    buy_gate_strict_ok=live_signal.buy_gate_strict_ok,
                    buy_gate_target_cost_ok=live_signal.buy_gate_target_cost_ok,
                    buy_gate_spread_ok=live_signal.buy_gate_spread_ok,
                    buy_gate_calibrated_ok=live_signal.buy_gate_calibrated_ok,
                    buy_gate_tradeable=live_signal.buy_gate_tradeable,
                    buy_gate_blocker=live_signal.buy_gate_blocker,
                )

            candidates.sort(
                key=lambda x: (
                    float(x.get("estimated_prob_up", 0.0)),
                    float(x.get("expected_net_edge_bps", 0.0)),
                    float(x.get("score", 0.0)),
                ),
                reverse=True,
            )
            if candidates:
                top = candidates[0]
                log(
                    f"[buy-candidates] count={len(candidates)} "
                    f"top={top.get('product_id')} "
                    f"score={float(top.get('score', 0.0)):.3f} "
                    f"prob={float(top.get('estimated_prob_up', 0.0)):.6f} "
                    f"ev={float(top.get('expected_net_edge_bps', 0.0)):.3f} "
                    f"position_pct={float(top.get('position_pct', 0.0)):.6f}"
                )
            else:
                log("[buy-candidates] count=0")

            strong_candidate_count = sum(1 for c in candidates if c["score"] >= MID_SCORE_UTIL_THRESHOLD)

            for candidate in candidates[:MAX_NEW_ENTRIES_PER_EVAL]:
                product_id = candidate["product_id"]
                existing_qty = sum(l.qty for l in self.positions.get(product_id, []))
                if SOURCE_OF_TRUTH_COINBASE and isinstance(self.portfolio, LivePortfolio):
                    try:
                        snap_check = self.portfolio.refresh_snapshot(force=False, ttl_sec=1.25)
                        existing_qty = self.portfolio.get_product_total_qty(product_id, snapshot=snap_check)
                    except Exception:
                        existing_qty = sum(l.qty for l in self.positions.get(product_id, []))
                product_exposure = self._current_product_exposure_usd(product_id)
                total_exposure = self._current_total_exposure_usd()
                open_count = self._open_position_count()

                entry_notional = self.compute_entry_notional(
                    available_cash_usd=cash_usd,
                    current_total_exposure_usd=total_exposure,
                    current_equity_usd=equity_usd,
                    current_product_exposure_usd=product_exposure,
                    candidate_score=float(candidate["score"]),
                    open_position_count=open_count,
                    strong_candidate_count=strong_candidate_count,
                    estimated_prob_up=float(candidate.get("estimated_prob_up", 0.0)),
                )

                if existing_qty > 1e-12:
                    if not ALLOW_SCALE_INTO_WINNERS:
                        continue

                    current_adds = int(self.scale_add_count.get(product_id, 0))
                    if current_adds >= MAX_SCALE_ADDS_PER_POSITION:
                        continue

                    local_lots = self.positions.get(product_id, [])
                    local_qty = sum(l.qty for l in local_lots)
                    local_cost = sum(l.qty * l.price for l in local_lots)
                    if local_qty <= 0 or local_cost <= 0:
                        continue

                    local_avg = local_cost / local_qty
                    unrealized_net_bps = (
                        ((candidate["bid"] / local_avg) - 1.0) * 10000.0
                        - self._round_trip_cost_bps(spread_bps=candidate["spread_bps"])
                    )
                    if unrealized_net_bps < SCALE_ONLY_IF_UNREALIZED_NET_BPS_ABOVE:
                        continue

                    # Scale-ins are still probability/equity based, but smaller than initial entries.
                    entry_notional = min(
                        entry_notional,
                        float(equity_usd)
                        * float(MAX_SINGLE_BUY_PCT_OF_EQUITY)
                        * float(SCALE_ADD_FRACTION_OF_ENTRY),
                    )

                    # Never allow total product exposure above 50% of equity.
                    max_product_exposure = (
                        float(equity_usd) * float(MAX_EXPOSURE_PER_PRODUCT_PCT_OF_EQUITY)
                    )
                    remaining_product_room = max(
                        0.0,
                        max_product_exposure - float(product_exposure),
                    )
                    entry_notional = min(entry_notional, remaining_product_room)

                min_order = max(float(MIN_ENTRY_USD), float(MIN_LIVE_ORDER_USD))

                if entry_notional < min_order:
                    log(
                        f"[buy-skip] {product_id} below_min_order "
                        f"entry_notional={entry_notional:.2f} min_order={min_order:.2f} "
                        f"cash={cash_usd:.2f} equity={equity_usd:.2f}"
                    )
                    continue

                entry_fee_bps = self._entry_fee_bps_for_mode()
                can_afford = await self._live_can_afford(entry_notional, entry_fee_bps)

                if not can_afford and ENABLE_PROFITABLE_ROTATION:
                    # Refresh real cash before deciding how much is missing.
                    snap_before_rotation = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                    live_cash_before = self.portfolio.get_tradable_usd(
                        snapshot=snap_before_rotation or {}
                    )

                    required_with_fee = (
                        entry_notional * (1.0 + entry_fee_bps / 10000.0)
                        + float(RESERVE_USD)
                    )
                    needed_cash = max(0.0, required_with_fee - float(live_cash_before))

                    if needed_cash > 0:
                        await self._try_rotate_capital_for_candidate(
                            candidate=candidate,
                            needed_cash_usd=needed_cash,
                            equity_usd=equity_usd,
                        )

                    # Re-check affordability after possible profitable rotation.
                    snap_after_rotation = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                    cash_usd = self.portfolio.get_tradable_usd(
                        snapshot=snap_after_rotation or {}
                    )
                    can_afford = await self._live_can_afford(entry_notional, entry_fee_bps)

                if not can_afford:
                    log(
                        f"[buy-skip] {product_id} cannot_afford "
                        f"entry_notional={entry_notional:.2f} "
                        f"cash={cash_usd:.2f} "
                        f"entry_fee_bps={entry_fee_bps:.3f}"
                    )
                    continue

                bid, ask = candidate["bid"], candidate["ask"]
                log(
                    f"[buy-attempt] {product_id} "
                    f"quote_usd={entry_notional:.2f} "
                    f"score={float(candidate.get('score', 0.0)):.3f} "
                    f"prob={float(candidate.get('estimated_prob_up', 0.0)):.6f} "
                    f"ev={float(candidate.get('expected_net_edge_bps', 0.0)):.3f} "
                    f"bid={bid:.8f} ask={ask:.8f}"
                )
                fill = await self._execute_live_buy(
                    product_id=product_id,
                    quote_usd=entry_notional,
                    bid=bid,
                    ask=ask,
                    reason=candidate.get("entry_reason", "score_entry"),
                )

                if fill is None:
                    log(f"[buy-failed] {product_id} live buy returned no confirmed fill")
                    continue

                filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
                log(
                    f"[buy-success] {product_id} "
                    f"qty={float(filled_qty):.12f} avg_px={float(avg_px):.8f} "
                    f"fee={float(fee_val):.6f} "
                    f"filled_notional={float(filled_notional or 0.0):.6f} "
                    f"order_id={_order_id}"
                )
                qty1 = float(filled_qty)
                buy_px1 = float(avg_px)
                fee1 = float(fee_val)
                eff_price1 = float((filled_notional + fee1) / qty1) if qty1 > 0 and filled_notional is not None else buy_px1

                if qty1 > 0:
                    lot_meta = {
                        "scalp_done": False,
                        "core_done": False,

                        # Armed target state.
                        "scalp_armed": False,
                        "core_armed": False,
                        "scalp_arm_price": None,
                        "core_arm_price": None,
                        "scalp_arm_peak": None,
                        "core_arm_peak": None,

                        "estimated_prob_up": float(candidate.get("estimated_prob_up", 0.0)),
                        "position_pct": float(candidate.get("position_pct", 0.0)),
                        "target_bps": float(candidate.get("target_bps", 0.0)),
                        "cost_bps": float(candidate.get("cost_bps", 0.0)),
                    }
                    existing_lots = self.positions.get(product_id, [])
                    if existing_lots:
                        existing_lots.append(
                            PositionLot(
                                qty=qty1,
                                price=eff_price1,
                                tier=int(candidate["tier"]),
                                score=float(candidate["score"]),
                                meta=lot_meta,
                            )
                        )
                        self.positions[product_id] = existing_lots
                        self.scale_add_count[product_id] = int(self.scale_add_count.get(product_id, 0)) + 1
                    else:
                        self.positions[product_id] = [
                            PositionLot(
                                qty=qty1,
                                price=eff_price1,
                                tier=int(candidate["tier"]),
                                score=float(candidate["score"]),
                                meta=lot_meta,
                            )
                        ]
                        self.scale_add_count[product_id] = 0
                        self.position_start_ts[product_id] = ts_now
                    self.last_buy_ts[product_id] = ts_now
                    self.last_buy_price[product_id] = ask
                    self.anchor_ts[product_id] = ts_now
                    self.peak_bid[product_id] = bid
                    self.tlog.log_trade(
                        event="BUY", product_id=product_id, side="BUY", qty=qty1, price=buy_px1,
                        fee_usd_val=fee1, gross_pnl_usd=0.0, net_pnl_usd=-fee1,
                        entry_price=buy_px1, exit_price=None, weekly_bias=candidate.get("weekly_bias"),
                        note=(
                            f"{candidate.get('entry_reason', 'score_entry')} "
                            f"prob={float(candidate.get('estimated_prob_up', 0.0)):.3f} "
                            f"pos_pct={float(candidate.get('position_pct', 0.0)):.3f}"
                        ),
                        filled_notional_usd=(float(filled_notional) if filled_notional is not None else None),
                        entry_score=float(candidate["score"]), entry_tier=int(candidate["tier"]),
                        expected_net_edge_bps=float(candidate.get("expected_net_edge_bps", 0.0)),
                    )
                    self._record_trade_timestamp(product_id)

            log(f"[loop] sleeping {EVAL_TICK_SEC:.1f}s until next evaluation")
            await asyncio.sleep(EVAL_TICK_SEC)


    def _write_position_targets_snapshot(self) -> None:
        """Write current open-position sell targets for viewer monitoring."""
        rows: List[Dict[str, Any]] = []
        ts_now = now_ts()

        for product_id in PRODUCTS:
            tob = self.tob.get(product_id)
            bid = float(tob.bid) if tob and tob.bid > 0 else None
            ask = float(tob.ask) if tob and tob.ask > 0 else None

            lots = self.positions.get(product_id, [])
            position_qty = sum(float(lot.qty) for lot in lots)
            has_position = position_qty > 1e-12

            row: Dict[str, Any] = {
                "ts": ts_now,
                "product_id": product_id,
                "has_position": has_position,
                "position_qty": position_qty,
                "avg_entry_price": None,
                "current_bid": bid,
                "current_ask": ask,
                "min_profitable_exit_price": None,
                "scalp_target_price": None,
                "core_target_price": None,
                "scalp_armed": False,
                "core_armed": False,
                "scalp_arm_peak": None,
                "core_arm_peak": None,
                "scalp_pullback_pct": 0.0,
                "core_pullback_pct": 0.0,
                "scalp_pullback_trigger_price": None,
                "core_pullback_trigger_price": None,
                "distance_to_min_profit_bps": None,
                "distance_to_scalp_bps": None,
                "distance_to_core_bps": None,
                "exit_plan_note": "no open position",
            }

            if has_position:
                avg_entry_price = sum(float(lot.qty) * float(lot.price) for lot in lots) / position_qty
                lot = lots[0]
                lot_tier = lot.tier if lot.tier in EXIT_PLAN else TIER_LOW
                lot_meta = lot.meta
                sigma_bps = self._compute_sigma_bps_from_1m(product_id) or 35.0
                targets = get_exit_targets(
                    entry_price=avg_entry_price,
                    sigma_bps=sigma_bps,
                    tier=lot_tier,
                )

                profile = self.calibration_profiles.get(
                    product_id,
                    ProductCalibrationProfile(product_id=product_id),
                )
                scalp_pullback_pct = float(profile.scalp_pullback_pct)
                core_pullback_pct = float(profile.core_pullback_pct)

                try:
                    min_exit_px = required_exit_price_for_net_gain(
                        effective_entry_price=avg_entry_price,
                        exit_fee_bps=self._exit_fee_bps_for_mode(),
                        est_slippage_bps=EST_SLIPPAGE_BPS,
                        est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                        min_net_gain_bps=max(
                            MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                            MIN_NET_GAIN_AFTER_FEES_BPS,
                        ),
                    )
                except Exception:
                    min_exit_px = None

                scalp_target = max(float(targets["scalp_target"]), float(min_exit_px or 0.0))
                core_target = max(float(targets["core_target"]), float(min_exit_px or 0.0))
                scalp_peak = lot_meta.get("scalp_arm_peak")
                core_peak = lot_meta.get("core_arm_peak")
                scalp_trigger = (
                    float(scalp_peak) * (1.0 - scalp_pullback_pct)
                    if scalp_peak is not None and float(scalp_peak) > 0
                    else None
                )
                core_trigger = (
                    float(core_peak) * (1.0 - core_pullback_pct)
                    if core_peak is not None and float(core_peak) > 0
                    else None
                )

                def dist_bps(target_px: Optional[float]) -> Optional[float]:
                    if bid is None or target_px is None or bid <= 0 or target_px <= 0:
                        return None
                    return ((float(target_px) / bid) - 1.0) * 10000.0

                scalp_armed = bool(lot_meta.get("scalp_armed", False))
                core_armed = bool(lot_meta.get("core_armed", False))
                row.update({
                    "avg_entry_price": avg_entry_price,
                    "min_profitable_exit_price": min_exit_px,
                    "scalp_target_price": scalp_target,
                    "core_target_price": core_target,
                    "scalp_armed": scalp_armed,
                    "core_armed": core_armed,
                    "scalp_arm_peak": scalp_peak,
                    "core_arm_peak": core_peak,
                    "scalp_pullback_pct": scalp_pullback_pct,
                    "core_pullback_pct": core_pullback_pct,
                    "scalp_pullback_trigger_price": scalp_trigger,
                    "core_pullback_trigger_price": core_trigger,
                    "distance_to_min_profit_bps": dist_bps(min_exit_px),
                    "distance_to_scalp_bps": dist_bps(scalp_target),
                    "distance_to_core_bps": dist_bps(core_target),
                    "exit_plan_note": (
                        "scalp/core armed trailing active"
                        if scalp_armed or core_armed
                        else "waiting for min-profit/scalp/core target"
                    ),
                })
                log(
                    f"[sell-plan] {product_id} "
                    f"qty={position_qty:.12f} entry={avg_entry_price:.8f} bid={bid} "
                    f"min_exit={min_exit_px} "
                    f"scalp_target={row.get('scalp_target_price')} "
                    f"core_target={row.get('core_target_price')} "
                    f"scalp_armed={row.get('scalp_armed')} "
                    f"core_armed={row.get('core_armed')} "
                    f"note={row.get('exit_plan_note')}"
                )

            rows.append(row)

        self.position_targets_log.write_rows(rows)


    async def telemetry_loop(self) -> None:
        """
        Periodically log market snapshots for viewer.  Includes exposures,
        volatility, anchored VWAP and fair value.
        """
        while not self._stop_event.is_set():
            ts_now = now_ts_i()
            try:
                if (
                    self.cached_account_snapshot is None
                    or now_ts() - float(self.cached_account_snapshot_ts or 0.0)
                    >= TELEMETRY_ACCOUNT_REFRESH_TTL_SEC
                ):
                    self.cached_account_snapshot = await asyncio.to_thread(
                        self.portfolio.refresh_snapshot,
                        True,
                        0.0,
                    )
                    self.cached_account_snapshot_ts = now_ts()

                snap_live = self.cached_account_snapshot
                cash_usd = self.portfolio.get_tradable_usd(snapshot=snap_live)
                equity_usd = self.portfolio.compute_equity_usd(
                    mid_by_product=self._live_mid_by_product(),
                    snapshot=snap_live,
                )
            except Exception as e:
                log(f"[telemetry] Coinbase equity refresh failed: {e}")
                snap_live = None
                cash_usd = 0.0
                equity_usd = 0.0
            # Log per product snapshot
            for product in PRODUCTS:
                tob = self.tob.get(product)
                if not tob:
                    continue
                mid = (tob.bid + tob.ask) / 2.0
                spread_bps = ((tob.ask - tob.bid) / mid) * 10_000.0 if mid > 0 else 0.0
                positions = self.positions[product]

                try:
                    position_qty = self.portfolio.get_product_total_qty(
                        product, snapshot=snap_live or {}
                    )
                    exposures_usd = float(position_qty) * float(mid)

                    local_qty = sum(lot.qty for lot in positions)
                    local_cost = sum(lot.qty * lot.price for lot in positions)
                    avg_entry_price = (local_cost / local_qty) if local_qty > 0 else None
                except Exception as e:
                    log(f"[telemetry] Coinbase position read failed for {product}: {e}")
                    exposures_usd = 0.0
                    position_qty = 0.0
                    avg_entry_price = None
                # anchored vwap (24h anchored, always-on)
                avwap = self._compute_anchored_vwap_24h(product, ts_now)

                levels_day = self.macro.get_levels(product, "day")
                levels_week = self.macro.get_levels(product, "week")
                # 24h-aware fair value blend + smoothing
                fair_value = self._compute_fair_value(product, mid, avwap)

                sigma_bps = self._compute_sigma_bps_from_1m(product)
                weekly_bias = self.macro.compute_weekly_bias(product, mid)
                state = "long" if position_qty > 0 else "flat"
                # Compute continuous live signal fields for every telemetry row.
                try:
                    minute_candles = list(self.live_1m.get(product).candles) if self.live_1m.get(product) else []

                    live_signal = self._build_live_signal(
                        product_id=product,
                        mid=mid,
                        spread_bps=spread_bps,
                        levels_day=levels_day,
                        levels_week=levels_week,
                        minute_candles=minute_candles,
                        weekly_bias=weekly_bias,
                        sigma_bps=sigma_bps,
                    )

                except Exception as sig_err:
                    log(f"[telemetry] live signal failed for {product}: {sig_err}")
                    live_signal = LiveSignal(
                        ok_to_trade=False,
                        score=0.0,
                        tier=0,
                        reason=f"live_signal_error={sig_err}",
                        estimated_prob_up=0.0,
                        position_pct=0.0,
                        expected_net_edge_bps=0.0,
                        target_bps=0.0,
                        cost_bps=0.0,
                        projected_forward_gain_bps=0.0,
                        calibrated_time_to_min_profit_minutes=0.0,
                        calibrated_forward_window_minutes=0.0,
                        dip_depth_score=0.0,
                        dip_speed_score=0.0,
                        reversal_score=0.0,
                        support_score=0.0,
                        room_score=0.0,
                        regime_score=0.0,
                        spread_penalty=0.0,
                        cost_penalty=0.0,
                        buy_gate_score_ok=False,
                        buy_gate_prob_ok=False,
                        buy_gate_ev_ok=False,
                        buy_gate_fee_ok=False,
                        buy_gate_strict_ok=False,
                        buy_gate_target_cost_ok=False,
                        buy_gate_spread_ok=False,
                        buy_gate_calibrated_ok=False,
                        buy_gate_tradeable=False,
                        buy_gate_blocker="signal_error_or_fallback",
                    )

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
                    entry_score=live_signal.score,
                    entry_tier=live_signal.tier,
                    entry_reason=live_signal.reason,
                    expected_net_edge_bps=live_signal.expected_net_edge_bps,
                    estimated_prob_up=live_signal.estimated_prob_up,
                    position_pct=live_signal.position_pct,
                    target_bps=live_signal.target_bps,
                    projected_forward_gain_bps=live_signal.projected_forward_gain_bps,
                    cost_bps=live_signal.cost_bps,
                    calibrated_time_to_min_profit_minutes=live_signal.calibrated_time_to_min_profit_minutes,
                    calibrated_forward_window_minutes=live_signal.calibrated_forward_window_minutes,
                    current_maker_fee_bps=self.current_maker_fee_bps,
                    current_taker_fee_bps=self.current_taker_fee_bps,
                    fee_tier_reason=self.last_fee_tier_reason,
                    dip_depth_score=live_signal.dip_depth_score,
                    dip_speed_score=live_signal.dip_speed_score,
                    reversal_score=live_signal.reversal_score,
                    support_score=live_signal.support_score,
                    room_score=live_signal.room_score,
                    regime_score=live_signal.regime_score,
                    spread_penalty=live_signal.spread_penalty,
                    cost_penalty=live_signal.cost_penalty,
                    buy_gate_score_ok=live_signal.buy_gate_score_ok,
                    buy_gate_prob_ok=live_signal.buy_gate_prob_ok,
                    buy_gate_ev_ok=live_signal.buy_gate_ev_ok,
                    buy_gate_fee_ok=live_signal.buy_gate_fee_ok,
                    buy_gate_strict_ok=live_signal.buy_gate_strict_ok,
                    buy_gate_target_cost_ok=live_signal.buy_gate_target_cost_ok,
                    buy_gate_spread_ok=live_signal.buy_gate_spread_ok,
                    buy_gate_calibrated_ok=live_signal.buy_gate_calibrated_ok,
                    buy_gate_tradeable=live_signal.buy_gate_tradeable,
                    buy_gate_blocker=live_signal.buy_gate_blocker,
                )
            try:
                self._write_position_targets_snapshot()
            except Exception as e:
                log(f"[telemetry] position target snapshot failed: {e}")

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

    log(f"[debug] writing debug log to {DEBUG_LOG_PATH}")
    log("[startup] bot.py launching")
    log(f"[startup] file={os.path.abspath(__file__)}")
    log("[startup] loading Coinbase client")
    rest = load_coinbase_client()

    log("[startup] loading environment")
    load_dotenv()
    api_key = (os.environ.get("COINBASE_API_KEY") or "").strip()
    pem = load_pem_secret_from_env()

    log("[startup] selecting products")
    if AUTO_SELECT_PRODUCTS:
        try:
            PRODUCTS = await asyncio.to_thread(select_diversified_products)
            if not PRODUCTS:
                log("[select] auto-selection returned no products; using defaults")
                PRODUCTS = list(PRODUCTS_DEFAULT)
        except Exception as e:
            log(f"[select] failed, using default products: {e}")
            PRODUCTS = list(PRODUCTS_DEFAULT)
    else:
        PRODUCTS = list(PRODUCTS_DEFAULT)

    # Currency safety: enforce USD quote pairs.
    PRODUCTS = [p for p in PRODUCTS if p.endswith("-USD")]

    log(f"[config] Trading products: {PRODUCTS}")
    log("[startup] creating TradingBot instance")
    bot = TradingBot(rest=rest, api_key=api_key, pem_secret=pem)
    log("[startup] LIVE-ONLY MODE: this bot can place real Coinbase orders.")

    if not hasattr(bot, "run"):
        raise RuntimeError("TradingBot instance has no run(); ensure you are running the updated bot.py file.")

    log("[startup] entering TradingBot.run()")
    await bot.run()
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("[shutdown] Bot interrupted by user.")
    except Exception as exc:
        log_exception("unhandled bot error", exc)
        raise
