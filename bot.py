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
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Deque, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd
import requests
import websockets
from dotenv import load_dotenv

from coinbase.rest import RESTClient
from coinbase import jwt_generator

try:
    from ai_brain import LocalAIBrain
except Exception:
    LocalAIBrain = None

try:
    from level8_council import Level8Council
except Exception:
    Level8Council = None


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
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "BNB-USD",
    "DOGE-USD",
    "ADA-USD",
    "LINK-USD",
    "AVAX-USD",
    "XLM-USD",
    "LTC-USD",
    "BCH-USD",
    "SHIB-USD",
    "DOT-USD",
    "SUI-USD",
]

# Auto-selection (diversify volatility + liquidity):
# We pick a set of USD pairs that (a) are liquid on Coinbase Exchange and
# (b) tend to have higher realized volatility when BTC is quiet, while keeping
# correlations in the basket lower.
AUTO_SELECT_PRODUCTS: bool = False
TARGET_PRODUCT_COUNT: int = 15         # total products to trade (includes BTC if available)
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
CANDIDATE_REPLAY_CSV_PATH: str = os.path.join(BASE_DIR, "candidate_replay.csv")
PRODUCTS_ACTIVE_CSV_PATH: str = os.path.join(BASE_DIR, "products_active.csv")
SIGNAL_EVENTS_CSV_PATH: str = os.path.join(BASE_DIR, "signal_events.csv")
LEVEL8_COUNCIL_DECISIONS_CSV_PATH: str = os.path.join(
    BASE_DIR, "council_decisions.csv"
)
TRADE_OUTCOMES_CSV_PATH: str = os.path.join(BASE_DIR, "trade_outcomes.csv")
MISSED_OPPORTUNITIES_CSV_PATH: str = os.path.join(BASE_DIR, "missed_opportunities.csv")
COUNCIL_OBSERVATION_OUTCOMES_CSV_PATH: str = os.path.join(
    BASE_DIR, "council_observation_outcomes.csv"
)
RECONCILIATION_CSV_PATH: str = os.path.join(BASE_DIR, "reconciliation.csv")
AGENT_PERFORMANCE_CSV_PATH: str = os.path.join(BASE_DIR, "agent_performance.csv")
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
MAX_NEW_ENTRIES_PER_EVAL: int = 3
EVAL_TICK_SEC: float = 2.0

# Bound aggregate deployment when several candidates pass in the same cycle.
MAX_CASH_DEPLOYED_PER_EVAL_PCT_OF_EQUITY: float = 0.40
ENABLE_MULTI_CANDIDATE_BUYS: bool = True

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

# ============================================================
# LEVEL 5 TRADING MANAGER — REMOVED FROM DECISION CHAIN
# ============================================================
#
# Level 5 is intentionally disabled and no longer participates in:
# - buy filtering
# - sell filtering
# - sizing
# - session pausing
# - strategy decisions
#
# Level 8 is the active intelligence layer.
ENABLE_LEVEL5_MANAGER: bool = False
LEVEL5_DISABLE_INVERTED_CYCLE: bool = False
LEVEL5_MODE: str = "DISABLED"
LEVEL5_MIN_POSITION_PCT: float = 0.0
LEVEL5_MAX_POSITION_PCT: float = 1.0

# ============================================================
# LEVEL 8 EVIDENCE-WEIGHTED COUNCIL
# ============================================================

ENABLE_LEVEL8_COUNCIL: bool = True

# Always-on council commentary.
# This makes the Level 8 council evaluate current market conditions even when
# nothing passes the normal buy gate, so the viewer always has council dialogue.
LEVEL8_ENABLE_COUNCIL_HEARTBEAT: bool = True
LEVEL8_COUNCIL_HEARTBEAT_EVERY_SEC: float = 8.0
LEVEL8_COUNCIL_HEARTBEAT_MAX_PRODUCTS: int = 15

# Active mode. No observe-only staged approach.
LEVEL8_MODE: str = "FILTER_AND_SIZE"

LEVEL8_ALLOW_TEST_BUCKET_LIVE_TRADES: bool = True
LEVEL8_ALLOW_CORE_BUCKET_LIVE_TRADES: bool = True

# Only hard spending ceiling: max 80% deployed, 20% reserve.
LEVEL8_RESERVE_CASH_PCT: float = 0.20
LEVEL8_MAX_SINGLE_TRADE_PCT: float = 0.80
LEVEL8_MAX_PRODUCT_EXPOSURE_PCT: float = 0.80
LEVEL8_MAX_TOTAL_EXPOSURE_PCT: float = 0.80

LEVEL8_MIN_TEST_TRADE_USD: float = MIN_LIVE_ORDER_USD
LEVEL8_SUPERSEDES_LEVEL5: bool = True

# ============================================================
# LEVEL 8 AGGRESSIVE LEARNING MODE
# ============================================================
#
# This mode deliberately feeds Level 8 more imperfect candidates so the bot can
# learn from real buys and missed opportunities.
#
# It does NOT disable mechanical protections:
# - valid bid/ask
# - fresh quote
# - available cash
# - Coinbase minimum order
# - 20% reserve

ENABLE_LEVEL8_LEARNING_MODE: bool = True

# Let moderately imperfect market states reach Level 8.
LEVEL8_LEARNING_MIN_SCORE: float = 22.0
LEVEL8_LEARNING_MIN_PROB: float = 0.22
LEVEL8_LEARNING_MIN_EV_BPS: float = -180.0

# Do not buy truly absurd spread conditions, but make this much looser.
LEVEL8_LEARNING_MAX_SPREAD_BPS: float = 80.0

# How many watch candidates can be sent into the real buy pipeline.
LEVEL8_LEARNING_MAX_EXTRA_CANDIDATES: int = 15

# Let the bot buy more than one learning candidate per evaluation if cash allows.
LEVEL8_LEARNING_MAX_NEW_ENTRIES_PER_EVAL: int = 5

# ============================================================
# LEVEL 8 CHART-ONLY OPPORTUNITY LEARNING
# ============================================================

ENABLE_LEVEL8_MISSED_OPPORTUNITY_LEARNING: bool = True
LEVEL8_OBSERVATION_REVIEW_WINDOWS_MIN: List[int] = [5, 15, 30, 60]
LEVEL8_MISSED_BIG_MOVE_BPS: float = 120.0
LEVEL8_MISSED_HUGE_MOVE_BPS: float = 250.0
LEVEL8_MISSED_REVIEW_EVERY_SEC: float = 60.0
LEVEL8_MISSED_MOVE_THRESHOLD_RELIEF: float = 0.04
LEVEL8_MISSED_MOVE_MAX_RELIEF: float = 0.12

# Max total exposure per product can reach 80% of total equity through scale-ins.
MAX_EXPOSURE_PER_PRODUCT_PCT_OF_EQUITY: float = 0.80

# Probability mapping for aggressive Level 8 learning-mode sizing.
PROB_FOR_MIN_SIZE: float = 0.25
PROB_FOR_MAX_SIZE: float = 0.70

# Dollar reserve is no longer the primary guardrail.
# Level 8's 20% reserve is the real reserve model.
RESERVE_USD: float = 0.00

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

# Top-of-book readiness.
# Websocket can be slow or sparse for some products. Use REST fallback quotes
# so the bot can monitor all configured coins.
ENABLE_REST_TOP_OF_BOOK_FALLBACK: bool = True

# Require nearly all configured products to have quotes before startup proceeds.
TOP_OF_BOOK_READY_MIN_PRODUCTS_PCT: float = 0.95
TOP_OF_BOOK_WAIT_SEC: float = 90.0

# Backfill quotes faster so the viewer does not bounce live → delayed → stale.
TOP_OF_BOOK_REST_FALLBACK_EVERY_SEC: float = 1.25

# Treat a quote as stale quickly enough that REST refreshes before the viewer
# spends a long time showing delayed/stale.
TOP_OF_BOOK_MAX_STALE_SEC: float = 6.0

# Still do not buy with stale or missing bid/ask.
REQUIRE_FRESH_TOP_OF_BOOK_FOR_BUY: bool = True

# ============================================================
# LOCAL AI BRAIN
# ============================================================

ENABLE_LOCAL_AI_BRAIN: bool = True
AI_MODE: str = "FILTER"  # OFF, OBSERVE, FILTER, CONTROL
AI_MIN_TRAINING_ROWS: int = 30
AI_RETRAIN_EVERY_SEC: float = 30 * 60
AI_ALLOW_BUY_ACTIONS: Set[str] = {"ALLOW_BUY"}
AI_BLOCK_BUY_ACTIONS: Set[str] = {"BLOCK_BUY"}
AI_MIN_CONFIDENCE_TO_BLOCK: float = 0.20
AI_MIN_CONFIDENCE_TO_ALLOW: float = 0.20

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
EXIT_EXECUTION_MODE: str = "MARKET"

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

# Simplified buy gate.
# The strategy buy decision should be based on only the three intended requirements.
SIMPLIFY_BUY_GATE_TO_THREE_REQUIREMENTS: bool = True

# Keep spread, setup/reversal, and target-cost as diagnostics only.
SPREAD_GATE_BLOCKS_BUY: bool = False
SETUP_REVERSAL_GATE_BLOCKS_BUY: bool = False
TARGET_COST_GATE_BLOCKS_BUY: bool = False

# Fee data is operationally required because the bot must calculate real costs.
FEE_DATA_REQUIRED_FOR_LIVE_BUY: bool = True

# Calibrated buy gate behavior.
# The old target-to-cost gate used target_bps, which is often only a few bps.
# The new EV system should use calibrated projected forward gain instead.
USE_CALIBRATED_FORWARD_GAIN_FOR_TARGET_COST_GATE: bool = True

# EV-primary buy behavior is disabled. Calibrated score, probability, and EV
# targets are all mandatory for actual buy permission.
USE_EV_PRIMARY_BUY_GATE: bool = False

# Legacy EV-primary floors are retained for diagnostic logging only.
EV_PRIMARY_MIN_SCORE_FLOOR: float = 25.0
EV_PRIMARY_MIN_PROB_FLOOR: float = 0.35
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

# Multi-window calibration and post-profit breathing-room analysis.
CALIB_FORWARD_WINDOWS_1M: List[int] = [15, 30, 60, 120, 180, 240]
CALIB_FORWARD_WINDOWS_15M: List[int] = [4, 8, 12, 16, 24]
CALIB_POST_PROFIT_BREATHING_MINUTES: int = 60
MAX_ADVERSE_BEFORE_PROFIT_BPS: float = 140.0
PREFERRED_TIME_TO_MIN_PROFIT_MINUTES: float = 180.0

# Similar historical setup matching.
SIMILAR_SCORE_BAND: float = 8.0
SIMILAR_PROB_BAND: float = 0.08
SIMILAR_COST_BAND_BPS: float = 80.0
SIMILAR_SPREAD_BAND_BPS: float = 8.0

# Live recalibration protection.
REQUIRE_VALID_LIVE_RECALIBRATION_PROFILE: bool = True
LIVE_RECALIBRATION_PROJECTED_GROSS_MIN_BPS: float = 1.0

# Edge-aware entry execution.
USE_EDGE_AWARE_ENTRY_EXECUTION: bool = True

# If the bot has decided to buy live, prioritize reliable execution.
# MARKET removes maker-no-fill failures.
# LIMIT_THEN_MARKET is also acceptable, but MARKET is the most reliable.
ENTRY_LOW_EDGE_MODE: str = "MARKET"
ENTRY_MEDIUM_EDGE_MODE: str = "MARKET"
ENTRY_HIGH_EDGE_MODE: str = "MARKET"

LOW_EDGE_MIN_PROJECTED_NET_BPS: float = 35.0
MEDIUM_EDGE_MIN_PROJECTED_NET_BPS: float = 75.0
HIGH_EDGE_MIN_PROJECTED_NET_BPS: float = 120.0

# Maker orders may save fees but can fail/no-fill.
# For launch reliability, do not use maker-only entries.
MAKER_ENTRY_TIMEOUT_SEC: float = 4.0

# Keep entry timing enabled, but make it permissive.
# This aims for "moderately bad learning buys," not catastrophic falling-knife buys.
REQUIRE_ENTRY_TIMING_CONFIRMATION: bool = True

# Do not hard-block every micro downtrend. Level 8 needs learning room.
BLOCK_BUY_WHILE_MICROTREND_DOWN: bool = False
REQUIRE_MICRO_UPTURN_FOR_BUY: bool = False
REQUIRE_PRICE_ABOVE_MICRO_VWAP_FOR_BUY: bool = False

# Keep a very loose higher-low / green-candle preference.
REQUIRE_HIGHER_LOW_OR_GREEN_SEQUENCE_FOR_BUY: bool = True
REQUIRE_NO_LOWER_LOW_SEQUENCE_FOR_BUY: bool = False

# Very permissive momentum stack.
MIN_ENTRY_MOMENTUM_1_BPS: float = -8.0
MIN_ENTRY_MOMENTUM_3_BPS: float = -14.0
MIN_ENTRY_MOMENTUM_5_BPS: float = -20.0
MIN_ENTRY_MOMENTUM_15_BPS: float = -45.0

ENTRY_GREEN_CANDLE_LOOKBACK: int = 5
ENTRY_MIN_GREEN_CANDLES: int = 1
ENTRY_TIMING_FAIL_COOLDOWN_SEC: float = 8.0

# Level 8 is now the main determinant.
# Do not let the old selectivity layer starve Level 8 of learning candidates.
ENABLE_RELATIVE_CANDIDATE_SELECTIVITY: bool = False
MAX_BUYABLE_RANKED_CANDIDATES: int = 15
MIN_RANK_ADVANTAGE_OVER_MEDIAN: float = -999.0
MIN_CANDIDATE_RANK_SCORE_TO_BUY: float = 0.0
MIN_LIVE_EV_BPS_FOR_ACTUAL_BUY: float = -180.0
LOW_CONFIDENCE_EV_BONUS_REQUIREMENT_BPS: float = 0.0
MAX_SIMULTANEOUS_BUY_READY_WITHOUT_RANK_EDGE: int = 15

# Spread still affects EV/cost math, but it should not prevent Level 8 from
# learning unless the spread is truly unreasonable.
ENABLE_HARD_SPREAD_FILTER_FOR_BUYS: bool = False
HARD_MAX_BUY_SPREAD_BPS: float = 80.0
PRODUCT_MAX_BUY_SPREAD_BPS: Dict[str, float] = {}
SPREAD_RANK_PENALTY_MULT: float = 0.35

# Post-buy outcome research windows.
POST_BUY_REVIEW_WINDOWS_MINUTES: List[int] = [5, 15, 30, 60, 120]
ENABLE_TRADE_OUTCOME_RESEARCH_LOG: bool = True

# If a signal stops qualifying, retain its armed state for this long so a brief
# evaluation gap does not immediately reset it.
BUY_ARMED_SIGNAL_TTL_SEC: float = 180.0

# Do not buy if the signal has been armed too long without confirming.
BUY_ARMED_SIGNAL_STALE_SEC: float = 300.0

# Profit lock and stale-position review.
ENABLE_PROFIT_LOCK: bool = True
PROFIT_LOCK_BUFFER_BPS: float = 2.0
PROFIT_LOCK_SELL_FRACTION: float = 1.0
ENABLE_STALE_POSITION_REVIEW: bool = True
STALE_POSITION_MIN_SCORE_TO_KEEP: float = 35.0
STALE_POSITION_MIN_PROB_TO_KEEP: float = 0.45
STALE_POSITION_MIN_EV_TO_KEEP_BPS: float = 0.0

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

# Calibration bank behavior.
# Startup builds the base bank. Hourly updates append new observations.
# Profiles are rebuilt from the accumulated bank, not from a tiny live-only window.
ENABLE_CALIBRATION_BANK: bool = True
CALIBRATION_UPDATE_EVERY_SEC: float = 60 * 60
CALIBRATION_BANK_MAX_OBSERVATIONS_PER_PRODUCT: int = 8000

# A profile should always be learned from data if any observations exist.
# Exact threshold is preferred, but best-available learned targets are used if exact fails.
ALLOW_BEST_AVAILABLE_LEARNED_PROFILE: bool = True

# Best-available learned targets use real observation distributions, not static defaults.
BEST_AVAILABLE_SCORE_QUANTILE: float = 0.55
BEST_AVAILABLE_PROB_QUANTILE: float = 0.55
BEST_AVAILABLE_EV_QUANTILE: float = 0.55

# If winners exist, use winners first. If no winners exist, use top-motion observations.
BEST_AVAILABLE_TOP_MOTION_FRACTION: float = 0.25

# Avoid zero targets while still not using static score/probability defaults.
MIN_LEARNED_TARGET_EPSILON: float = 1e-9

# Outcome-calibrated probability behavior.
# The old probability model was frequently pinned at 20%, which made winners look weak.
USE_OUTCOME_CALIBRATED_PROBABILITY: bool = True

# Do not clamp live probability to 20%. Let weak setups be low and strong setups rise.
DISPLAY_PROB_MIN: float = 0.01
DISPLAY_PROB_MAX: float = 0.92

# Probability should care about projected forward gain versus real modeled cost.
PROB_PROJECTED_FORWARD_RATIO_WEIGHT: float = 0.10
PROB_EXPECTED_EDGE_WEIGHT: float = 0.12
PROB_PRICE_ACTION_WEIGHT: float = 0.55
PROB_STRUCTURE_WEIGHT: float = 0.20

# Learned EV targets are minimum acceptable thresholds, not full winner EV.
BEST_AVAILABLE_EV_TARGET_FRACTION: float = 0.65
EXACT_THRESHOLD_EV_TARGET_FRACTION: float = 0.65

# Avoid probability targets sitting at the old artificial 20% floor.
MIN_LEARNED_PROB_TARGET: float = 0.05

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

# Loss exit behavior.
# Normal sells should be net-positive after fees/buffers.
# Loss sells are allowed only if the position is down at least this much from entry.
ALLOW_LOSS_SELL_ONLY_AT_POSITION_LOSS_PCT: bool = True
MAX_POSITION_LOSS_BEFORE_FORCED_SELL_PCT: float = 0.01

# ============================================================
# INVERTED STOP-LOSS CYCLE STRATEGY
# ============================================================
#
# Normal behavior:
#   BUY_READY -> buy
#   loss stop -> sell
#
# Inverted behavior:
#   BUY_READY -> set sell marker / virtual peak
#   old loss-stop point -> buy
#   next old loss-stop point while holding -> sell old position and buy larger
#   return to sell marker -> sell
ENABLE_INVERTED_STOPLOSS_CYCLE: bool = False

# Use the existing 1% loss threshold as the inverted buy trigger.
INVERTED_BUY_DROP_PCT: float = MAX_POSITION_LOSS_BEFORE_FORCED_SELL_PCT

# If another stop-loss point is reached while holding, rotate into a larger position.
INVERTED_ENABLE_LOSS_ROTATION: bool = True
INVERTED_REBUY_SIZE_MULTIPLIER: float = 1.35

# Hard caps so the larger rebuy does not overrun the account.
INVERTED_MAX_SINGLE_BUY_PCT_OF_EQUITY: float = 0.25
INVERTED_MAX_PRODUCT_EXPOSURE_PCT_OF_EQUITY: float = 0.60

# Sell when price revisits the old buy marker. If fee protection is enabled,
# the actual trigger is max(old_buy_marker_price, min_profitable_exit_price).
INVERTED_REQUIRE_FEE_POSITIVE_SELL: bool = True

# If price continues falling after an inverted buy, this is the next rotation trigger.
INVERTED_NEXT_STOP_FROM_ENTRY_PCT: float = MAX_POSITION_LOSS_BEFORE_FORCED_SELL_PCT

# Expire old markers if price never drops into the inverted buy zone.
INVERTED_MARKER_TTL_SEC: float = 4 * 60 * 60

# Cooldown after completing an inverted cycle.
INVERTED_POST_CYCLE_COOLDOWN_SEC: float = 60.0

# Logging.
INVERTED_LOG_PREFIX: str = "[inverted-cycle]"

# ============================================================
# INVERTED STRATEGY REPAIR / SAFETY CONFIG
# ============================================================

# Fill sanity:
# Reject fills whose average price is too far away from the current bid/ask.
# This prevents impossible local entries like SUI at ~0.38 while live market is ~0.75.
ENABLE_FILL_PRICE_SANITY_CHECK: bool = True
MAX_FILL_PRICE_DEVIATION_FROM_TOB_PCT: float = 0.02

# Dust cleanup:
# If a local position is too tiny to sell, clear it locally instead of repeatedly
# submitting invalid Coinbase orders.
ENABLE_LOCAL_DUST_POSITION_CLEANUP: bool = True
LOCAL_DUST_USD_THRESHOLD: float = 0.25
LOCAL_DUST_QTY_EPSILON: float = 1e-8

# Inverted buy stabilization:
# Old stop-loss touched = arm the buy.
# Actual buy happens only after price stops falling / stabilizes.
INVERTED_REQUIRE_TRIGGER_STABILIZATION: bool = True
INVERTED_BUY_TRIGGER_MIN_AGE_SEC: float = 20.0
INVERTED_BUY_STABILIZATION_LOOKBACK_CANDLES: int = 3
INVERTED_BUY_STABILIZATION_MAX_ADVERSE_BPS: float = 12.0
INVERTED_BUY_STABILIZATION_REQUIRE_NON_NEGATIVE_1M: bool = True

# Inverted rebuy rotation delay:
# Do not sell old + buy larger instantly. Require the deeper loss condition
# to persist briefly.
INVERTED_REBUY_ROTATION_MIN_HOLD_SEC: float = 180.0
INVERTED_REBUY_TRIGGER_CONFIRM_SEC: float = 45.0

# Unbuyable marker handling:
# If a marker keeps hitting its buy trigger but cannot buy due to zero notional,
# cash, exposure, or minimum size, expire or cool it down.
INVERTED_MAX_ZERO_NOTIONAL_TRIGGER_HITS: int = 5
INVERTED_UNBUYABLE_MARKER_COOLDOWN_SEC: float = 10 * 60

# When true, no-progress/time/invalidation/hard-peak exits cannot sell at a loss
# unless the position has crossed the 1% loss threshold.
BLOCK_NON_PROFIT_LOSS_EXITS_UNTIL_MAX_LOSS: bool = True

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
TARGET_PRODUCT_COUNT: int = 15
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


def candidate_rank_score(candidate: Dict[str, Any]) -> float:
    """Rank passing candidates by net opportunity, timing, and friction."""
    probability = float(candidate.get("estimated_prob_up", 0.0))
    expected_edge_bps = float(candidate.get("expected_net_edge_bps", 0.0))
    score = float(candidate.get("score", 0.0))
    spread_bps = float(candidate.get("spread_bps", 0.0))
    projected_bps = float(candidate.get("projected_forward_gain_bps", 0.0))
    cost_bps = float(candidate.get("cost_bps", 0.0))
    timing_reason = str(candidate.get("entry_timing_reason", ""))

    timing_bonus = 0.0
    if "entry_confirmed" in timing_reason:
        timing_bonus += 35.0
    if "hl=True" in timing_reason:
        timing_bonus += 10.0
    if "vwap=True" in timing_reason:
        timing_bonus += 8.0
    if "lower_low_seq=False" in timing_reason:
        timing_bonus += 8.0

    return (
        expected_edge_bps
        + probability * 80.0
        + score * 0.45
        + max(0.0, projected_bps - cost_bps) * 0.20
        + timing_bonus
        - spread_bps * float(SPREAD_RANK_PENALTY_MULT)
        - max(0.0, cost_bps - 260.0) * 0.08
    )


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


class _ResearchCSVLogger:
    """Append-only CSV logger with a stable schema and local-time timestamp."""

    columns: List[str] = []

    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.columns)

    def _write(self, kwargs: Dict[str, Any]) -> None:
        tsv = float(kwargs.get("ts", now_ts()))
        dt_mst = datetime.fromtimestamp(tsv, tz=timezone.utc).astimezone(TZ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        row: List[Any] = []
        for column in self.columns:
            if column == "ts":
                row.append(f"{tsv:.6f}")
            elif column == "dt_mst":
                row.append(dt_mst)
            else:
                row.append(kwargs.get(column, ""))
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


class SignalEventsLogger(_ResearchCSVLogger):
    """Log signal qualification, timing, and execution decisions."""

    columns = [
        "ts", "dt_mst", "event_type", "trade_id", "product_id",
        "rank", "rank_score", "buy_ready_count",
        "score", "score_target", "score_ok",
        "probability", "probability_target", "probability_ok",
        "ev_bps", "ev_target_bps", "ev_ok",
        "projected_forward_bps", "cost_bps", "spread_bps",
        "entry_timing_ok", "entry_timing_reason",
        "momentum_1_bps", "momentum_3_bps", "momentum_5_bps",
        "momentum_15_bps", "vwap_ok", "higher_low_ok", "green_candles",
        "candidate_rank_reason", "action", "reason",
    ]

    def log_event(self, **kwargs: Any) -> None:
        self._write(kwargs)


class ManagerStatusLogger(_ResearchCSVLogger):
    """Log the latest Level 5 session-risk posture for the live viewer."""

    columns = [
        "ts", "dt_mst", "risk_mode", "session_net", "loss_streak",
        "closed_count", "active_strategy_note", "reason",
    ]

    def log_status(self, status: Dict[str, Any]) -> None:
        self._write(status)


class TradeOutcomeLogger(_ResearchCSVLogger):
    """Log fixed-window price outcomes after each confirmed or adopted buy."""

    columns = [
        "ts", "dt_mst", "trade_id", "product_id",
        "review_minutes", "entry_ts", "entry_price", "review_price",
        "move_bps", "max_favorable_bps", "max_adverse_bps",
        "score_at_entry", "prob_at_entry", "ev_at_entry",
        "spread_at_entry", "timing_reason_at_entry",
        "position_open", "closed", "closed_reason", "closed_net_pnl_usd",
    ]

    def log_outcome(self, **kwargs: Any) -> None:
        self._write(kwargs)


class ReconciliationLogger(_ResearchCSVLogger):
    """Log uncertain Coinbase fill states and their final resolution."""

    columns = [
        "ts", "dt_mst", "event_type", "product_id", "side",
        "client_order_id", "order_id", "requested_quote_usd",
        "expected_base_delta", "actual_base_delta",
        "before_base", "after_base", "before_cash", "after_cash",
        "status", "error", "action_taken",
    ]

    def log_reconciliation(self, **kwargs: Any) -> None:
        self._write(kwargs)


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
                "ts", "dt_mst", "product_id", "source", "bid", "ask", "mid", "spread_bps",
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
        source: str = "unknown",
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
                f"{ts:.6f}", dt_mst, product_id, source, f"{bid:.10f}", f"{ask:.10f}", f"{mid:.10f}", f"{spread_bps:.6f}",
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
    selected_forward_window_minutes: float = 0.0
    post_profit_max_favorable_bps: float = 0.0
    post_profit_extra_gain_bps: float = 0.0
    adverse_before_profit_bps: float = 0.0
    survived_to_profit: bool = False
    accepted_by_calibration: bool = False


@dataclass
class ProductCalibrationProfile:
    product_id: str
    is_calibrated: bool = False
    calibration_status: str = "not_calibrated"
    min_score: float = 0.0
    min_probability: float = 0.0
    min_expected_value_bps: float = 0.0
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
    calibrated_selected_window_minutes: float = 0.0
    calibrated_post_profit_breathing_minutes: float = 0.0
    calibrated_post_profit_extra_gain_bps: float = 0.0
    calibrated_max_adverse_before_profit_bps: float = 0.0
    calibrated_expected_bps_per_minute: float = 0.0
    calibrated_raw_probability_median: float = 0.0
    calibrated_empirical_win_rate: float = 0.0
    calibrated_probability_model_note: str = ""

    reason: str = "not_calibrated"


class CandidateReplayLogger:
    """Persist walk-forward candidates so calibration decisions are inspectable."""

    columns = [
        "ts", "dt_mst", "product_id", "timeframe",
        "score", "probability", "expected_net_edge_bps",
        "target_bps", "cost_bps", "spread_bps",
        "selected_forward_window_minutes",
        "max_favorable_bps", "max_adverse_bps",
        "adverse_before_profit_bps", "time_to_min_profit_minutes",
        "forward_window_minutes", "post_profit_max_favorable_bps",
        "post_profit_extra_gain_bps", "reached_min_profit",
        "survived_to_profit", "accepted_by_calibration",
    ]

    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, newline="", encoding="utf-8") as f:
                    if next(csv.reader(f), []) == self.columns:
                        return
            except (OSError, csv.Error):
                pass
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.columns)

    def log_observations(self, observations: List[CalibrationObservation]) -> None:
        if not observations:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for o in observations:
                dt_mst = datetime.fromtimestamp(float(o.ts), tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
                w.writerow([
                    int(o.ts), dt_mst, o.product_id, o.timeframe,
                    f"{o.score:.6f}", f"{o.probability:.6f}",
                    f"{o.expected_net_edge_bps:.6f}", f"{o.target_bps:.6f}",
                    f"{o.cost_bps:.6f}", f"{o.spread_bps:.6f}",
                    f"{o.selected_forward_window_minutes:.6f}",
                    f"{o.max_favorable_bps:.6f}", f"{o.max_adverse_bps:.6f}",
                    f"{o.adverse_before_profit_bps:.6f}",
                    "" if o.time_to_min_profit_minutes is None else f"{o.time_to_min_profit_minutes:.6f}",
                    "" if o.forward_window_minutes is None else f"{o.forward_window_minutes:.6f}",
                    f"{o.post_profit_max_favorable_bps:.6f}",
                    f"{o.post_profit_extra_gain_bps:.6f}", bool(o.reached_min_profit),
                    bool(o.survived_to_profit), bool(o.accepted_by_calibration),
                ])


class CalibrationLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        header = [
            "ts", "dt_mst", "product_id",
            "is_calibrated", "calibration_status",
            "min_score", "min_probability", "min_expected_value_bps",
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
            "calibrated_probability_model_note",
            "reason",
        ]
        if os.path.exists(self.path):
            try:
                with open(self.path, newline="", encoding="utf-8") as f:
                    existing_header = next(csv.reader(f), [])
                if existing_header == header:
                    return
            except (OSError, csv.Error):
                pass
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def log_profile(self, profile: ProductCalibrationProfile) -> None:
        tsv = now_ts()
        dt_mst = datetime.fromtimestamp(tsv, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                f"{tsv:.6f}", dt_mst, profile.product_id,
                profile.is_calibrated, profile.calibration_status,
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
                f"{profile.calibrated_selected_window_minutes:.6f}",
                f"{profile.calibrated_post_profit_breathing_minutes:.6f}",
                f"{profile.calibrated_post_profit_extra_gain_bps:.6f}",
                f"{profile.calibrated_max_adverse_before_profit_bps:.6f}",
                f"{profile.calibrated_expected_bps_per_minute:.6f}",
                f"{profile.calibrated_raw_probability_median:.6f}",
                f"{profile.calibrated_empirical_win_rate:.6f}",
                profile.calibrated_probability_model_note,
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


class ActiveProductsLogger:
    """Atomically publishes the products monitored by the running bot."""

    columns = ["product_id"]

    def __init__(self, path: str) -> None:
        self.path = path

    def write_products(self, products: List[str]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.columns)
            for product in products:
                w.writerow([product])
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
        "profit_lock_armed", "profit_lock_price",
        "min_profitable_exit_price_from_lot",
        "calibrated_forward_window_minutes",
        "calibrated_post_profit_breathing_minutes",
        "inverted_mode",
        "inverted_marker_price",
        "inverted_buy_trigger_price",
        "inverted_target_sell_price",
        "inverted_next_loss_trigger_price",
        "inverted_rebuy_count",
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
                    bool(r.get("profit_lock_armed", False)),
                    "" if r.get("profit_lock_price") is None else f"{float(r.get('profit_lock_price')):.10f}",
                    "" if r.get("min_profitable_exit_price_from_lot") is None else f"{float(r.get('min_profitable_exit_price_from_lot')):.10f}",
                    "" if r.get("calibrated_forward_window_minutes") is None else f"{float(r.get('calibrated_forward_window_minutes')):.6f}",
                    "" if r.get("calibrated_post_profit_breathing_minutes") is None else f"{float(r.get('calibrated_post_profit_breathing_minutes')):.6f}",
                    bool(r.get("inverted_mode", False)),
                    "" if r.get("inverted_marker_price") is None else f"{float(r.get('inverted_marker_price')):.10f}",
                    "" if r.get("inverted_buy_trigger_price") is None else f"{float(r.get('inverted_buy_trigger_price')):.10f}",
                    "" if r.get("inverted_target_sell_price") is None else f"{float(r.get('inverted_target_sell_price')):.10f}",
                    "" if r.get("inverted_next_loss_trigger_price") is None else f"{float(r.get('inverted_next_loss_trigger_price')):.10f}",
                    int(r.get("inverted_rebuy_count", 0) or 0),
                    r.get("exit_plan_note", ""),
                ])
        for attempt in range(5):
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                if attempt >= 4:
                    log(
                        f"[position-targets] replace failed after retries; "
                        f"path={self.path}"
                    )
                    break
                time.sleep(0.15)


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
        self.product_meta_cache: Dict[str, Dict[str, Any]] = {}

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

    def get_best_bid_ask(self, product_id: str) -> Tuple[Optional[float], Optional[float]]:
        """Return a best-effort REST quote when websocket data is unavailable."""
        product_id = str(product_id).strip().upper()

        try:
            fn = getattr(self.rest, "get_best_bid_ask", None)
            if callable(fn):
                try:
                    resp = fn(product_ids=[product_id])
                except TypeError:
                    resp = fn(product_id=product_id)

                data = self._to_dict(resp)
                pricebooks = (
                    data.get("pricebooks")
                    or data.get("price_books")
                    or data.get("data")
                    or []
                )
                if isinstance(pricebooks, dict):
                    pricebooks = (
                        pricebooks.get("pricebooks")
                        or pricebooks.get("price_books")
                        or [pricebooks]
                    )

                if isinstance(pricebooks, list):
                    for pricebook in pricebooks:
                        pricebook_data = self._to_dict(pricebook)
                        if str(pricebook_data.get("product_id", "")).upper() not in ("", product_id):
                            continue
                        bids = pricebook_data.get("bids") or []
                        asks = pricebook_data.get("asks") or []
                        bid = safe_float(self._to_dict(bids[0]).get("price") or self._to_dict(bids[0]).get("bid_price")) if bids else None
                        ask = safe_float(self._to_dict(asks[0]).get("price") or self._to_dict(asks[0]).get("ask_price")) if asks else None
                        if bid is not None and ask is not None and bid > 0 and ask > 0:
                            return float(bid), float(ask)
        except Exception:
            pass

        try:
            fn = getattr(self.rest, "get_product_book", None)
            if callable(fn):
                try:
                    resp = fn(product_id=product_id, limit=1)
                except TypeError:
                    resp = fn(product_id)

                data = self._to_dict(resp)
                pricebook = self._to_dict(data.get("pricebook"))
                bids = data.get("bids") or pricebook.get("bids") or []
                asks = data.get("asks") or pricebook.get("asks") or []
                bid = safe_float(self._to_dict(bids[0]).get("price")) if bids else None
                ask = safe_float(self._to_dict(asks[0]).get("price")) if asks else None
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    return float(bid), float(ask)
        except Exception:
            pass

        return None, None

    def get_product_meta(self, product_id: str) -> Dict[str, Any]:
        """Return Coinbase product metadata used to format order sizes."""
        product_id = str(product_id).strip().upper()

        if product_id in self.product_meta_cache:
            return self.product_meta_cache[product_id]

        product: Dict[str, Any] = {}

        try:
            response = self.rest.get_products(
                product_ids=[product_id],
                get_tradability_status=True,
            )
            data = self._to_dict(response)
            products = data.get("products", [])

            if isinstance(products, list) and products:
                product = self._to_dict(products[0])

        except TypeError:
            try:
                response = self.rest.get_products(limit=1000)
                data = self._to_dict(response)
                products = data.get("products", [])

                if isinstance(products, list):
                    for item in products:
                        candidate = self._to_dict(item)
                        if str(candidate.get("product_id", "")).strip().upper() == product_id:
                            product = candidate
                            break

            except Exception:
                product = {}

        except Exception:
            product = {}

        self.product_meta_cache[product_id] = product
        return product

    def _decimal_places_from_increment(self, increment: str) -> int:
        try:
            d = Decimal(str(increment))
            return max(0, -d.as_tuple().exponent)
        except Exception:
            return 8

    def format_base_size_for_product(self, product_id: str, base_qty: float) -> str:
        """Format a base quantity to its Coinbase increment, always rounding down."""
        product = self.get_product_meta(product_id)

        base_increment = (
            product.get("base_increment")
            or product.get("baseIncrement")
            or product.get("base_increment_size")
            or product.get("base_increment_amount")
            or "0.00000001"
        )

        try:
            increment = Decimal(str(base_increment))
            qty = Decimal(str(float(base_qty)))

            if qty <= 0:
                return "0"
            if increment <= 0:
                raise InvalidOperation("base_increment must be positive")

            floored = (qty / increment).to_integral_value(rounding=ROUND_DOWN) * increment

            places = self._decimal_places_from_increment(str(base_increment))
            formatted = f"{floored:.{places}f}".rstrip("0").rstrip(".")

            return formatted if formatted else "0"

        except (InvalidOperation, ValueError, TypeError):
            qty = Decimal(str(float(base_qty)))
            floored = qty.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            return f"{floored:.8f}".rstrip("0").rstrip(".")

    def product_base_min_size(self, product_id: str) -> float:
        """Return Coinbase's minimum base order size for a product, if available."""
        product = self.get_product_meta(product_id)

        raw = (
            product.get("base_min_size")
            or product.get("baseMinSize")
            or product.get("base_min_order_size")
            or product.get("min_market_funds")
            or 0
        )

        try:
            return float(raw)
        except Exception:
            return 0.0

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

        before_snapshot: Dict[str, Dict[str, float]] = {}
        before_cash = 0.0
        before_base_total = 0.0
        before_snapshot_ok = False
        base_asset = product_base_asset(product_id)

        try:
            before_snapshot = self.refresh_snapshot(force=True, ttl_sec=0.0)
            before_snapshot_ok = bool(before_snapshot)
            before_cash = self.get_tradable_usd(snapshot=before_snapshot)
            before_base_total = (
                self.get_total_asset(base_asset, snapshot=before_snapshot)
                if base_asset
                else 0.0
            )
        except Exception as exc:
            log(f"[fill-reconcile] pre-trade balance snapshot failed for {product_id}: {exc}")

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
                base_size = self.format_base_size_for_product(product_id, float(base_qty))
                base_min_size = self.product_base_min_size(product_id)

                try:
                    base_size_float = float(base_size)
                except Exception:
                    base_size_float = 0.0

                if base_size_float <= 0:
                    return ExecutionResult(
                        ok=False,
                        order_id=None,
                        client_order_id=client_order_id,
                        product_id=product_id,
                        side=side_u,
                        filled_qty=0.0,
                        avg_price=None,
                        fee_usd=0.0,
                        filled_notional_usd=None,
                        status="INVALID",
                        error=f"formatted_base_size<=0 raw_qty={base_qty} formatted={base_size}",
                    ).to_dict()

                if base_min_size > 0 and base_size_float < base_min_size:
                    return ExecutionResult(
                        ok=False,
                        order_id=None,
                        client_order_id=client_order_id,
                        product_id=product_id,
                        side=side_u,
                        filled_qty=0.0,
                        avg_price=None,
                        fee_usd=0.0,
                        filled_notional_usd=None,
                        status="INVALID",
                        error=(
                            f"base_size_below_min raw_qty={base_qty} "
                            f"formatted={base_size} min={base_min_size}"
                        ),
                    ).to_dict()

                resp = self.rest.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=base_size,
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

        after_snapshot: Dict[str, Dict[str, float]] = {}
        after_cash = 0.0
        after_base_total = 0.0
        after_snapshot_ok = False

        try:
            self.sync_after_trade(attempts=8, sleep_sec=0.5)
            after_snapshot = self.refresh_snapshot(force=True, ttl_sec=0.0)
            after_snapshot_ok = bool(after_snapshot)
            after_cash = self.get_tradable_usd(snapshot=after_snapshot)
            after_base_total = (
                self.get_total_asset(base_asset, snapshot=after_snapshot)
                if base_asset
                else 0.0
            )
        except Exception as exc:
            log(f"[fill-reconcile] post-trade balance snapshot failed for {product_id}: {exc}")

        # Coinbase balance deltas are authoritative for BUY quantity. This avoids
        # treating quote-denominated SDK fields as base-asset fills.
        if side_u == "BUY":
            requested_quote = float(quote_usd or 0.0)
            if not before_snapshot_ok or not after_snapshot_ok:
                ok_final = False
                err = err or "buy_balance_snapshot_unavailable"
            else:
                try:
                    base_delta = float(after_base_total) - float(before_base_total)
                    cash_delta = float(before_cash) - float(after_cash)

                    if base_delta > 1e-12:
                        qty_f = float(base_delta)
                        if cash_delta > 0 and cash_delta <= requested_quote * 1.25:
                            notional_f = float(cash_delta)
                        else:
                            notional_f = float(requested_quote)

                        if notional_f > 0:
                            avg_px_f = float(notional_f) / float(qty_f)
                        ok_final = True
                        status = status or "FILLED_BALANCE_DELTA"
                        err = None
                    else:
                        ok_final = False
                        err = err or "buy_no_base_balance_delta"
                except Exception as exc:
                    ok_final = False
                    err = err or "buy_balance_delta_reconcile_failed"
                    log(f"[fill-reconcile] BUY balance-delta failed for {product_id}: {exc}")

        # A small quote order must never create an implausibly large local position.
        if side_u == "BUY" and ok_final:
            requested_quote = float(quote_usd or 0.0)
            try:
                if (
                    requested_quote > 0
                    and notional_f is not None
                    and float(notional_f) > requested_quote * 1.25
                ):
                    log(
                        f"[fill-reconcile] {product_id} impossible BUY notional; "
                        f"requested_quote={requested_quote:.6f} "
                        f"notional={float(notional_f):.6f}; forcing failure"
                    )
                    ok_final = False
                    err = "impossible_buy_notional_vs_requested_quote"

                if requested_quote > 0 and avg_px_f is not None and qty_f > 0:
                    max_possible_base = (requested_quote * 1.25) / float(avg_px_f)
                    if float(qty_f) > max_possible_base:
                        log(
                            f"[fill-reconcile] {product_id} impossible BUY qty; "
                            f"requested_quote={requested_quote:.6f} "
                            f"qty={float(qty_f):.12f} "
                            f"avg_px={float(avg_px_f):.8f}; forcing failure"
                        )
                        ok_final = False
                        err = "impossible_buy_qty_vs_requested_quote"
            except Exception as exc:
                ok_final = False
                err = err or "buy_sanity_check_failed"
                log(f"[fill-reconcile] BUY sanity check failed for {product_id}: {exc}")

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
        formatted = self.format_base_size_for_product(product_id, float(base_qty))
        log(
            f"[sell-format] {product_id} market raw_qty={float(base_qty):.12f} "
            f"formatted_base_size={formatted}"
        )
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

        formatted_base_size = self.format_base_size_for_product(product_id, float(base_qty))

        log(
            f"[sell-format] {product_id} limit raw_qty={float(base_qty):.12f} "
            f"formatted_base_size={formatted_base_size} limit_price={float(limit_price):.8f}"
        )

        payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": formatted_base_size,
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
        self.candidate_replay_log = CandidateReplayLogger(CANDIDATE_REPLAY_CSV_PATH)
        self.active_products_log = ActiveProductsLogger(PRODUCTS_ACTIVE_CSV_PATH)
        self.signal_events_log = SignalEventsLogger(SIGNAL_EVENTS_CSV_PATH)
        self.trade_outcomes_log = TradeOutcomeLogger(TRADE_OUTCOMES_CSV_PATH)
        self.reconciliation_log = ReconciliationLogger(RECONCILIATION_CSV_PATH)
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
        self.calibration_observation_bank: Dict[str, List[CalibrationObservation]] = {
            product: [] for product in PRODUCTS
        }
        self.last_hourly_calibration_update_ts: float = 0.0
        self.last_buy_gate_log_ts_by_product: Dict[str, float] = {}
        self.last_sell_failure_ts_by_product: Dict[str, float] = {}
        self.armed_buy_signals: Dict[str, Dict[str, Any]] = {}
        self.last_entry_timing_fail_ts_by_product: Dict[str, float] = {}
        self.pending_buy_reconciliations: Dict[str, Dict[str, Any]] = {}
        self.last_buy_execution_result: Dict[str, Dict[str, Any]] = {}
        self.post_buy_review_queue: List[Dict[str, Any]] = []
        self.live_recalibration_running: bool = False
        self.last_loop_lag_check_ts: float = now_ts()
        self.cached_account_snapshot: Optional[Dict[str, Dict[str, float]]] = None
        self.cached_account_snapshot_ts: float = 0.0
        self.ai_brain = None
        self.level8_council = None
        self.last_level8_council_heartbeat_ts: float = 0.0
        self.last_ai_train_ts: float = 0.0
        self.last_agent_performance_update_ts: float = 0.0
        self.last_level8_missed_opportunity_review_ts: float = 0.0
        if ENABLE_LOCAL_AI_BRAIN and LocalAIBrain is not None:
            try:
                self.ai_brain = LocalAIBrain(
                    min_training_rows=AI_MIN_TRAINING_ROWS
                )
                log("[ai] LocalAIBrain initialized")
            except Exception as exc:
                self.ai_brain = None
                log(f"[ai] failed to initialize LocalAIBrain: {exc}")
        if ENABLE_LEVEL8_COUNCIL and Level8Council is not None:
            try:
                self.level8_council = Level8Council()
                log("[level8] council initialized")
            except Exception as exc:
                self.level8_council = None
                log(f"[level8] council initialization failed: {exc}")
        # positions per product: list of PositionLot
        self.positions: Dict[str, List[PositionLot]] = {p: [] for p in PRODUCTS}
        self.inverted_markers: Dict[str, Dict[str, Any]] = {}
        self.inverted_cycle_cooldown_until: Dict[str, float] = {}
        self.inverted_cycle_index_by_product: Dict[str, int] = {p: 0 for p in PRODUCTS}
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


    def _fill_price_near_live_market(
        self,
        *,
        product_id: str,
        side: str,
        avg_price: float,
    ) -> Tuple[bool, str]:
        """Reject impossible fill prices that are far from the live market."""
        if not ENABLE_FILL_PRICE_SANITY_CHECK:
            return True, "fill_price_sanity_disabled"

        tob = self.tob.get(product_id)
        if not tob or tob.bid <= 0 or tob.ask <= 0:
            return True, "no_tob_for_sanity_check"

        side_u = str(side).upper().strip()
        px = float(avg_price)
        if px <= 0:
            return False, "avg_price<=0"

        # BUY should be near ask. SELL should be near bid.
        ref = float(tob.ask) if side_u == "BUY" else float(tob.bid)
        if ref <= 0:
            return True, "invalid_ref_price"

        deviation = abs(px - ref) / ref
        if deviation > float(MAX_FILL_PRICE_DEVIATION_FROM_TOB_PCT):
            return (
                False,
                f"fill_price_too_far_from_tob side={side_u} "
                f"avg_price={px:.8f} ref={ref:.8f} "
                f"deviation_pct={deviation * 100:.3f}% "
                f"max_pct={MAX_FILL_PRICE_DEVIATION_FROM_TOB_PCT * 100:.3f}%",
            )

        return True, (
            f"fill_price_ok side={side_u} avg_price={px:.8f} "
            f"ref={ref:.8f} deviation_pct={deviation * 100:.3f}%"
        )

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

        if side_u == "BUY":
            log(
                f"[buy-fill-check] {product_id} "
                f"filled_qty={filled_qty:.12f} "
                f"avg_px={0.0 if avg_px is None else float(avg_px):.8f} "
                f"filled_notional={0.0 if filled_notional is None else float(filled_notional):.6f}"
            )

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

        sanity_ok, sanity_reason = self._fill_price_near_live_market(
            product_id=product_id,
            side=side_u,
            avg_price=float(avg_px),
        )
        if not sanity_ok:
            log(
                f"[fill-sanity] {product_id} rejected {side_u} fill: "
                f"{sanity_reason}"
            )
            try:
                self.reconciliation_log.log_reconciliation(
                    event_type="fill_price_sanity_rejected",
                    product_id=product_id,
                    side=side_u,
                    order_id=order_id,
                    status="rejected",
                    error=sanity_reason,
                    action_taken="rejected_impossible_fill_price",
                )
            except Exception:
                pass
            return None

        log(
            f"[fill-sanity] {product_id} accepted {side_u} fill: "
            f"{sanity_reason}"
        )
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

    def _entry_fee_bps_for_mode(self, execution_mode: Optional[str] = None) -> float:
        if self.current_maker_fee_bps is None or self.current_taker_fee_bps is None:
            raise RuntimeError("Coinbase fee tier has not been loaded; refusing to estimate entry fees.")

        mode = str(execution_mode or ENTRY_EXECUTION_MODE).upper().strip()

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

    def _position_gross_loss_pct(self, *, entry_price: float, exit_price: float) -> float:
        """Return the gross price loss from entry to the current exit price."""
        try:
            entry = float(entry_price)
            exit_px = float(exit_price)
            if entry <= 0 or exit_px <= 0:
                return 0.0
            return max(0.0, (entry - exit_px) / entry)
        except Exception:
            return 0.0

    def _loss_stop_triggered(self, *, entry_price: float, exit_price: float) -> bool:
        """Return true only after the configured maximum-loss threshold is reached."""
        if not BLOCK_NON_PROFIT_LOSS_EXITS_UNTIL_MAX_LOSS:
            return True
        if not ALLOW_LOSS_SELL_ONLY_AT_POSITION_LOSS_PCT:
            return True

        loss_pct = self._position_gross_loss_pct(
            entry_price=entry_price,
            exit_price=exit_price,
        )

        return bool(loss_pct >= float(MAX_POSITION_LOSS_BEFORE_FORCED_SELL_PCT))

    def _exit_is_net_positive(
        self,
        *,
        entry_price: float,
        exit_price: float,
        min_net_profit_bps: float = 0.0,
    ) -> bool:
        """Return whether an exit remains net-positive after fees and buffers."""
        return can_exit_net_positive(
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            taker_fee_bps=self._exit_fee_bps_for_mode(),
            est_slippage_bps=EST_SLIPPAGE_BPS,
            est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
            min_net_profit_bps=float(min_net_profit_bps),
        )

    def _allow_exit_for_role(
        self,
        *,
        product_id: str,
        role: str,
        entry_price: float,
        exit_price: float,
        min_net_profit_bps: float = 0.0,
    ) -> bool:
        """Allow fee-aware profitable exits or a forced exit at the 1% loss stop."""
        net_positive = self._exit_is_net_positive(
            entry_price=entry_price,
            exit_price=exit_price,
            min_net_profit_bps=min_net_profit_bps,
        )

        if net_positive:
            return True

        loss_stop = self._loss_stop_triggered(
            entry_price=entry_price,
            exit_price=exit_price,
        )

        if loss_stop:
            loss_pct = self._position_gross_loss_pct(
                entry_price=entry_price,
                exit_price=exit_price,
            )
            log(
                f"[loss-stop] {product_id} allowing loss exit role={role} "
                f"loss_pct={loss_pct * 100:.3f}% "
                f"threshold={MAX_POSITION_LOSS_BEFORE_FORCED_SELL_PCT * 100:.3f}% "
                f"entry={entry_price:.8f} bid={exit_price:.8f}"
            )
            return True

        log(
            f"[sell-block] {product_id} blocked loss exit role={role} "
            f"entry={entry_price:.8f} bid={exit_price:.8f} "
            f"loss_pct={self._position_gross_loss_pct(entry_price=entry_price, exit_price=exit_price) * 100:.3f}% "
            f"threshold={MAX_POSITION_LOSS_BEFORE_FORCED_SELL_PCT * 100:.3f}%"
        )

        return False

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
        projected_forward_gain_bps: Optional[float] = None,
        expected_net_edge_bps: Optional[float] = None,
        cost_bps: Optional[float] = None,
        fee_available: bool = False,
    ) -> float:
        """
        Outcome-calibrated live probability.

        The older version was often pinned at 20%, which caused the bot to
        under-score probable winners. This version uses:
        - live score / price action
        - projected forward gain versus real cost
        - projected net edge
        - structure and confirmation
        """
        s = clamp_float(float(score), 0.0, 100.0)

        # Price-action base.
        # score 0 -> 18%, score 50 -> 48%, score 100 -> 78% before modifiers.
        price_action_prob = 0.18 + (s / 100.0) * 0.60

        # Momentum contribution.
        momentum_adj = 0.0
        momentum_adj += clamp_float(float(momentum_5_bps) / 90.0, -0.08, 0.08)
        momentum_adj += clamp_float(float(momentum_15_bps) / 160.0, -0.06, 0.06)

        # Structure contribution.
        structure_adj = 0.0
        structure_adj += ((float(support_score) - 50.0) / 100.0) * 0.055
        structure_adj += ((float(room_score) - 50.0) / 100.0) * 0.050
        structure_adj += ((float(regime_score) - 50.0) / 100.0) * 0.040

        # Confirmation contribution.
        confirmation_adj = 0.0
        confirmation_adj += 0.035 if vwap_ok else -0.025
        confirmation_adj += 0.035 if higher_low_ok else -0.025

        # Trend and spread penalties.
        risk_adj = 0.0
        if trending_down:
            risk_adj -= 0.065

        if spread_bps > SCALP_MAX_SPREAD_BPS:
            risk_adj -= 0.020
        if spread_bps > MAX_SPREAD_BPS:
            risk_adj -= 0.070

        # Fee-aware projection contribution.
        projection_adj = 0.0

        if (
            fee_available
            and cost_bps is not None
            and float(cost_bps) > 0
            and projected_forward_gain_bps is not None
        ):
            projected_ratio = float(projected_forward_gain_bps) / float(cost_bps)

            # Ratio around 1.0 means projected gross only covers cost.
            # Ratio above 1.0 means there is room for actual net profit.
            projection_adj += clamp_float((projected_ratio - 1.0) * 0.16, -0.10, 0.16)

        if expected_net_edge_bps is not None:
            # Positive edge matters because it already includes projected forward
            # gain minus real modeled cost.
            edge_ratio = float(expected_net_edge_bps) / max(
                1.0, float(MIN_REQUIRED_NET_EDGE_BPS)
            )
            projection_adj += clamp_float(edge_ratio * 0.045, -0.09, 0.14)

        prob = (
            price_action_prob * PROB_PRICE_ACTION_WEIGHT
            + (0.50 + structure_adj + confirmation_adj) * PROB_STRUCTURE_WEIGHT
            + (0.50 + projection_adj)
            * (
                PROB_PROJECTED_FORWARD_RATIO_WEIGHT
                + PROB_EXPECTED_EDGE_WEIGHT
            )
            + momentum_adj
            + risk_adj
        )

        return clamp_float(prob, DISPLAY_PROB_MIN, DISPLAY_PROB_MAX)

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
            projected_forward_gain_bps=target_bps,
            expected_net_edge_bps=expected_net_edge_bps,
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
    ) -> Tuple[
        float, float, bool, bool, float, float, float,
        Optional[int], Optional[float], float,
        float, float, float, bool,
    ]:
        """Measure profitability, timing, continuation, and pre-profit adversity."""
        if entry_price <= 0 or not future_candles:
            return 0.0, 0.0, False, False, 0.0, 0.0, 0.0, None, None, 0.0, 0.0, 0.0, 0.0, False

        required_profit_bps = float(cost_bps) + float(min_net_gain_bps)
        max_high = 0.0
        min_low = float("inf")
        max_favorable_bps = 0.0
        max_adverse_bps = 0.0
        time_to_min_profit_bars: Optional[int] = None
        time_to_min_profit_minutes: Optional[float] = None
        adverse_before_profit_bps = 0.0
        low_before_profit = entry_price
        profit_hit_index: Optional[int] = None

        for idx, candle in enumerate(future_candles, start=1):
            high = float(candle.high)
            low = float(candle.low)
            if high <= 0 or low <= 0:
                continue
            max_high = max(max_high, high)
            min_low = min(min_low, low)
            max_favorable_bps = max(max_favorable_bps, ((max_high / entry_price) - 1.0) * 10000.0)
            max_adverse_bps = max(max_adverse_bps, ((entry_price / min_low) - 1.0) * 10000.0)
            if time_to_min_profit_bars is None:
                low_before_profit = min(low_before_profit, low)
                if ((high / entry_price) - 1.0) * 10000.0 >= required_profit_bps:
                    time_to_min_profit_bars = idx
                    time_to_min_profit_minutes = float(idx) * float(bar_minutes)
                    profit_hit_index = idx - 1
                    adverse_before_profit_bps = ((entry_price / low_before_profit) - 1.0) * 10000.0

        reached_min_profit = time_to_min_profit_bars is not None
        reached_target = max_favorable_bps >= max(float(target_bps), required_profit_bps)
        win_bps = max(0.0, max_favorable_bps - float(cost_bps))
        loss_bps = max(0.0, max_adverse_bps)
        forward_window_minutes = float(len(future_candles)) * float(bar_minutes)
        post_profit_max_favorable_bps = 0.0
        post_profit_extra_gain_bps = 0.0
        if reached_min_profit and profit_hit_index is not None:
            post_profit_highs = [float(c.high) for c in future_candles[profit_hit_index:] if float(c.high) > 0]
            if post_profit_highs:
                post_profit_max_favorable_bps = ((max(post_profit_highs) / entry_price) - 1.0) * 10000.0
                post_profit_extra_gain_bps = max(0.0, post_profit_max_favorable_bps - required_profit_bps)
        survived_to_profit = bool(reached_min_profit and adverse_before_profit_bps <= MAX_ADVERSE_BEFORE_PROFIT_BPS)
        return (
            float(max_favorable_bps), float(max_adverse_bps), bool(reached_min_profit),
            bool(reached_target), float(win_bps), float(loss_bps), 0.0,
            time_to_min_profit_bars, time_to_min_profit_minutes,
            float(forward_window_minutes), float(post_profit_max_favorable_bps),
            float(post_profit_extra_gain_bps), float(adverse_before_profit_bps),
            bool(survived_to_profit),
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
                post_profit_max_favorable_bps,
                post_profit_extra_gain_bps,
                adverse_before_profit_bps,
                survived_to_profit,
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
                selected_forward_window_minutes=float(forward_window_minutes),
                post_profit_max_favorable_bps=float(post_profit_max_favorable_bps),
                post_profit_extra_gain_bps=float(post_profit_extra_gain_bps),
                adverse_before_profit_bps=float(adverse_before_profit_bps),
                survived_to_profit=bool(survived_to_profit),
            ))
        return observations

    def _walk_forward_observations_multi_window(
        self,
        *,
        product_id: str,
        candles: List[Candle],
        weekly_candles: Optional[List[Candle]],
        timeframe: str,
        min_prefix: int,
        forward_windows: List[int],
        spread_bps: float,
    ) -> List[CalibrationObservation]:
        """Replay each signal across several horizons and retain its best survivable path."""
        observations: List[CalibrationObservation] = []
        if not candles or not forward_windows:
            return observations
        windows = sorted({int(x) for x in forward_windows if int(x) > 0})
        if not windows:
            return observations
        max_forward = max(windows)
        if len(candles) < min_prefix + max_forward + 1:
            return observations

        for i in range(min_prefix, len(candles) - max_forward):
            prefix = candles[:i]
            entry_price = float(prefix[-1].close) if prefix else 0.0
            if entry_price <= 0:
                continue
            replay_ts = int(prefix[-1].ts)
            available_weekly = [c for c in weekly_candles if int(c.ts) <= replay_ts] if weekly_candles else None
            try:
                signal = self._build_historical_signal_from_candles(
                    product_id=product_id, candles=prefix,
                    weekly_candles=available_weekly, spread_bps=spread_bps,
                )
            except Exception:
                continue
            bar_minutes = 1.0 if timeframe in ("day_1m", "live_rolling_1m") else 15.0
            best_obs: Optional[CalibrationObservation] = None
            best_quality = -float("inf")
            for forward_bars in windows:
                future = candles[i:i + forward_bars]
                if len(future) < forward_bars:
                    continue
                (
                    max_favorable_bps, max_adverse_bps, reached_min_profit,
                    reached_target, win_bps, loss_bps, _, time_to_min_profit_bars,
                    time_to_min_profit_minutes, forward_window_minutes,
                    post_profit_max_favorable_bps, post_profit_extra_gain_bps,
                    adverse_before_profit_bps, survived_to_profit,
                ) = self._evaluate_forward_outcome(
                    entry_price=entry_price, future_candles=future,
                    target_bps=signal.target_bps, cost_bps=signal.cost_bps,
                    min_net_gain_bps=MIN_NET_GAIN_AFTER_FEES_BPS,
                    bar_minutes=bar_minutes,
                )
                expected_value_bps = win_bps if reached_min_profit else -loss_bps
                time_penalty = 0.0 if time_to_min_profit_minutes is None else max(
                    0.0, time_to_min_profit_minutes - PREFERRED_TIME_TO_MIN_PROFIT_MINUTES
                ) * 0.05
                quality = (
                    expected_value_bps + post_profit_extra_gain_bps * 0.35
                    - adverse_before_profit_bps * 0.45 - time_penalty
                    - (0.0 if survived_to_profit else 1000.0)
                )
                obs = CalibrationObservation(
                    product_id=product_id, timeframe=timeframe, ts=replay_ts,
                    score=float(signal.score), probability=float(signal.estimated_prob_up),
                    expected_net_edge_bps=float(signal.expected_net_edge_bps),
                    target_bps=float(signal.target_bps), cost_bps=float(signal.cost_bps),
                    spread_bps=float(spread_bps), max_favorable_bps=float(max_favorable_bps),
                    max_adverse_bps=float(max_adverse_bps), reached_min_profit=bool(reached_min_profit),
                    reached_target=bool(reached_target), expected_value_bps=float(expected_value_bps),
                    win_bps=float(win_bps), loss_bps=float(loss_bps),
                    time_to_min_profit_bars=time_to_min_profit_bars,
                    time_to_min_profit_minutes=time_to_min_profit_minutes,
                    forward_window_minutes=forward_window_minutes,
                    projected_forward_gain_bps=float(max_favorable_bps),
                    selected_forward_window_minutes=float(forward_window_minutes),
                    post_profit_max_favorable_bps=float(post_profit_max_favorable_bps),
                    post_profit_extra_gain_bps=float(post_profit_extra_gain_bps),
                    adverse_before_profit_bps=float(adverse_before_profit_bps),
                    survived_to_profit=bool(survived_to_profit),
                )
                if quality > best_quality:
                    best_quality, best_obs = quality, obs
            if best_obs is not None:
                observations.append(best_obs)
        return observations

    def _win_rate(self, observations: List[CalibrationObservation]) -> float:
        if not observations:
            return 0.0
        wins = sum(1 for observation in observations if observation.reached_min_profit)
        return float(wins / len(observations))

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
    ) -> Tuple[float, float, float, float, float, float]:
        """Return gross, timing, selected-window, continuation, and adversity medians."""
        if not observations:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        winners = [o for o in observations if o.reached_min_profit and o.survived_to_profit]
        source = winners if winners else observations
        def finite(values: List[float]) -> List[float]:
            return [float(v) for v in values if v is not None and np.isfinite(float(v))]
        favorable = finite([o.max_favorable_bps for o in source])
        times = finite([o.time_to_min_profit_minutes for o in winners])
        windows = finite([o.forward_window_minutes for o in source])
        selected_windows = finite([o.selected_forward_window_minutes for o in source])
        extra = finite([o.post_profit_extra_gain_bps for o in source])
        adverse = finite([o.adverse_before_profit_bps for o in source])
        return tuple(float(np.median(v)) if v else 0.0 for v in (
            favorable, times, windows, selected_windows, extra, adverse,
        ))

    def _append_calibration_observations(
        self,
        *,
        product_id: str,
        observations: List[CalibrationObservation],
    ) -> None:
        """
        Append observations to the product's calibration bank.

        Startup observations remain the base while hourly updates add new
        information instead of replacing the profile with a tiny live window.
        """
        if not ENABLE_CALIBRATION_BANK:
            return

        if product_id not in self.calibration_observation_bank:
            self.calibration_observation_bank[product_id] = []

        self.calibration_observation_bank[product_id].extend(observations)

        max_n = int(CALIBRATION_BANK_MAX_OBSERVATIONS_PER_PRODUCT)
        if len(self.calibration_observation_bank[product_id]) > max_n:
            self.calibration_observation_bank[product_id] = (
                self.calibration_observation_bank[product_id][-max_n:]
            )

    def _observations_for_profile(
        self,
        *,
        product_id: str,
        fallback: List[CalibrationObservation],
    ) -> List[CalibrationObservation]:
        """Return accumulated observations, or the provided fallback."""
        banked = self.calibration_observation_bank.get(product_id, [])
        return list(banked) if banked else list(fallback)

    def _build_best_available_learned_profile(
        self,
        *,
        product_id: str,
        observations: List[CalibrationObservation],
        day_obs: List[CalibrationObservation],
        week_obs: List[CalibrationObservation],
        status: str,
    ) -> ProductCalibrationProfile:
        """Build data-derived targets when no exact threshold qualifies."""
        observations = list(observations)

        if not observations:
            return ProductCalibrationProfile(
                product_id=product_id,
                is_calibrated=False,
                calibration_status="no_observations",
                min_score=float("nan"),
                min_probability=float("nan"),
                min_expected_value_bps=float("nan"),
                day_sample_count=len(day_obs),
                week_sample_count=len(week_obs),
                reason="NO LEARNED TARGETS: no observations available",
            )

        winners = [
            observation
            for observation in observations
            if bool(observation.reached_min_profit)
            and bool(observation.survived_to_profit)
        ]

        if winners:
            source = winners
            source_label = "survived_winners"
        else:
            sorted_by_motion = sorted(
                observations,
                key=lambda observation: float(
                    observation.max_favorable_bps or 0.0
                ),
                reverse=True,
            )
            take_n = max(
                1,
                int(
                    len(sorted_by_motion)
                    * float(BEST_AVAILABLE_TOP_MOTION_FRACTION)
                ),
            )
            source = sorted_by_motion[:take_n]
            source_label = "top_motion_no_winners"

        source_probabilities = [
            float(observation.probability)
            for observation in source
            if observation.probability is not None
            and np.isfinite(float(observation.probability))
        ]
        source_raw_prob_median = (
            float(np.median(source_probabilities))
            if source_probabilities
            else 0.0
        )
        source_empirical_win_rate = (
            sum(
                1
                for observation in source
                if observation.reached_min_profit
                and observation.survived_to_profit
            )
            / max(1, len(source))
        )

        def q(
            values: List[float],
            quantile: float,
            fallback: float = MIN_LEARNED_TARGET_EPSILON,
        ) -> float:
            clean = [
                float(value)
                for value in values
                if value is not None and np.isfinite(float(value))
            ]
            value = (
                float(np.quantile(clean, float(quantile)))
                if clean
                else float(fallback)
            )
            if value <= MIN_LEARNED_TARGET_EPSILON:
                value = MIN_LEARNED_TARGET_EPSILON
            return value

        learned_score = q(
            [observation.score for observation in source],
            BEST_AVAILABLE_SCORE_QUANTILE,
        )
        learned_prob = max(
            q(
                [observation.probability for observation in source],
                BEST_AVAILABLE_PROB_QUANTILE,
            ),
            float(MIN_LEARNED_PROB_TARGET),
        )
        learned_ev = q(
            [
                float(observation.expected_value_bps)
                for observation in source
                if np.isfinite(float(observation.expected_value_bps))
            ],
            BEST_AVAILABLE_EV_QUANTILE,
        )

        # The EV target is the minimum acceptable edge required for a new trade.
        # It should not require the live setup to match the full winner EV.
        learned_ev_target = max(
            float(CALIB_MIN_EXPECTED_VALUE_BPS),
            float(learned_ev) * float(BEST_AVAILABLE_EV_TARGET_FRACTION),
        )

        win_rate, avg_win, avg_loss, ev, _ = self._observation_ev_stats(
            observations
        )
        (
            projected_gross_bps,
            median_time_to_min_profit,
            median_forward_window,
            median_selected_window,
            median_post_profit_extra_gain,
            median_adverse_before_profit,
        ) = self._projection_stats_from_observations(source)

        return ProductCalibrationProfile(
            product_id=product_id,
            is_calibrated=True,
            calibration_status=f"best_available_{source_label}",
            min_score=float(learned_score),
            min_probability=float(learned_prob),
            min_expected_value_bps=float(learned_ev_target),
            day_sample_count=len(day_obs),
            week_sample_count=len(week_obs),
            day_win_rate=self._win_rate(day_obs),
            week_win_rate=self._win_rate(week_obs),
            blended_win_rate=float(win_rate),
            avg_win_bps=float(avg_win),
            avg_loss_bps=float(avg_loss),
            expected_value_bps=float(ev),
            calibrated_projected_gross_bps=float(projected_gross_bps),
            calibrated_projected_net_bps=float(ev),
            calibrated_time_to_min_profit_minutes=float(
                median_time_to_min_profit
            ),
            calibrated_forward_window_minutes=float(median_forward_window),
            calibrated_selected_window_minutes=float(median_selected_window),
            calibrated_post_profit_breathing_minutes=float(
                CALIB_POST_PROFIT_BREATHING_MINUTES
            ),
            calibrated_post_profit_extra_gain_bps=float(
                median_post_profit_extra_gain
            ),
            calibrated_max_adverse_before_profit_bps=float(
                median_adverse_before_profit
            ),
            calibrated_expected_bps_per_minute=(
                float(ev)
                / max(1.0, float(median_time_to_min_profit or 1.0))
            ),
            calibrated_raw_probability_median=float(
                source_raw_prob_median
            ),
            calibrated_empirical_win_rate=float(
                source_empirical_win_rate
            ),
            calibrated_probability_model_note=(
                f"source={source_label}; "
                f"raw_prob_median={source_raw_prob_median:.6f}; "
                f"source_empirical_win_rate={source_empirical_win_rate:.6f}"
            ),
            reason=(
                f"learned_profile product={product_id} status={status} "
                f"source={source_label} source_n={len(source)} "
                f"total_n={len(observations)} score_q={learned_score:.6f} "
                f"prob_q={learned_prob:.6f} "
                f"ev_target={learned_ev_target:.6f} "
                f"learned_ev={learned_ev:.6f} "
                f"projected_gross={projected_gross_bps:.6f} "
                f"time_to_profit={median_time_to_min_profit:.3f} "
                f"window={median_forward_window:.3f}"
            ),
        )

    def _uncalibrated_profile(
        self,
        *,
        product_id: str,
        status: str,
        day_obs: Optional[List[CalibrationObservation]] = None,
        week_obs: Optional[List[CalibrationObservation]] = None,
        blended_ev: float = 0.0,
        avg_win_bps: float = 0.0,
        avg_loss_bps: float = 0.0,
    ) -> ProductCalibrationProfile:
        """
        Return a non-tradeable calibration profile.

        This replaces all previous default/fallback calibration behavior.
        If the bot cannot produce real calibrated targets, the product should
        not trade until recalibration succeeds.
        """
        day_obs = day_obs or []
        week_obs = week_obs or []

        return ProductCalibrationProfile(
            product_id=product_id,
            is_calibrated=False,
            calibration_status=status,
            min_score=0.0,
            min_probability=0.0,
            min_expected_value_bps=0.0,
            day_sample_count=len(day_obs),
            week_sample_count=len(week_obs),
            day_win_rate=self._win_rate(day_obs),
            week_win_rate=self._win_rate(week_obs),
            blended_win_rate=0.0,
            avg_win_bps=float(avg_win_bps),
            avg_loss_bps=float(avg_loss_bps),
            expected_value_bps=float(blended_ev),
            calibrated_projected_gross_bps=0.0,
            calibrated_projected_net_bps=float(blended_ev),
            calibrated_time_to_min_profit_minutes=0.0,
            calibrated_forward_window_minutes=0.0,
            reason=f"UNCALIBRATED: {status}",
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
            if ALLOW_BEST_AVAILABLE_LEARNED_PROFILE and all_obs:
                return self._build_best_available_learned_profile(
                    product_id=product_id,
                    observations=all_obs,
                    day_obs=day_obs,
                    week_obs=week_obs,
                    status=(
                        f"insufficient_samples total={len(all_obs)} "
                        f"required={CALIB_MIN_PRODUCT_SAMPLES}"
                    ),
                )

            return self._uncalibrated_profile(
                product_id=product_id,
                status=(
                    f"no_observations total={len(all_obs)} "
                    f"required={CALIB_MIN_PRODUCT_SAMPLES}"
                ),
                day_obs=day_obs,
                week_obs=week_obs,
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

                if selected:
                    reference_cost = float(np.median([o.cost_bps for o in selected]))
                    reference_spread = float(np.median([o.spread_bps for o in selected]))
                    similar_selected = [
                        o for o in selected
                        if abs(float(o.score) - float(score_threshold)) <= SIMILAR_SCORE_BAND
                        and abs(float(o.probability) - float(prob_threshold)) <= SIMILAR_PROB_BAND
                        and abs(float(o.cost_bps) - reference_cost) <= SIMILAR_COST_BAND_BPS
                        and abs(float(o.spread_bps) - reference_spread) <= SIMILAR_SPREAD_BAND_BPS
                    ]
                    if len(similar_selected) >= CALIB_EXACT_MIN_SAMPLES:
                        selected = similar_selected

                selected_for_stats = [
                    o for o in selected
                    if (not o.reached_min_profit) or o.survived_to_profit
                ]
                if len(selected_for_stats) < CALIB_EXACT_MIN_SAMPLES:
                    continue

                win_rate, avg_win, avg_loss, ev, n = self._observation_ev_stats(selected_for_stats)
                (
                    projected_gross_bps,
                    median_time_to_min_profit,
                    median_forward_window,
                    median_selected_window,
                    median_post_profit_extra_gain,
                    median_adverse_before_profit,
                ) = self._projection_stats_from_observations(selected_for_stats)

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
                    + win_rate * 12.0
                    + opportunity_bonus
                    + median_post_profit_extra_gain * 0.20
                    - median_adverse_before_profit * 0.25
                    - max(0.0, median_time_to_min_profit - PREFERRED_TIME_TO_MIN_PROFIT_MINUTES) * 0.05
                    - float(score_threshold) * 0.010
                    - float(prob_threshold) * 1.00
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
                    "median_selected_window": float(median_selected_window),
                    "median_post_profit_extra_gain": float(median_post_profit_extra_gain),
                    "median_adverse_before_profit": float(median_adverse_before_profit),
                    "selected_observations": selected_for_stats,
                    "n": int(n),
                    "quality_score": float(quality_score),
                }

                if best is None or candidate["quality_score"] > best["quality_score"]:
                    best = candidate

        if best is not None:
            for observation in best["selected_observations"]:
                observation.accepted_by_calibration = True

            selected_for_stats = best["selected_observations"]
            selected_probabilities = [
                float(observation.probability)
                for observation in selected_for_stats
                if observation.probability is not None
                and np.isfinite(float(observation.probability))
            ]
            selected_raw_prob_median = (
                float(np.median(selected_probabilities))
                if selected_probabilities
                else 0.0
            )
            selected_empirical_win_rate = (
                sum(
                    1
                    for observation in selected_for_stats
                    if observation.reached_min_profit
                    and observation.survived_to_profit
                )
                / max(1, len(selected_for_stats))
            )
            calibrated_prob_threshold = max(
                float(best["prob_threshold"]),
                float(MIN_LEARNED_PROB_TARGET),
            )

            return ProductCalibrationProfile(
                product_id=product_id,
                is_calibrated=True,
                calibration_status="exact_threshold",
                min_score=max(
                    float(best["score_threshold"]),
                    float(MIN_LEARNED_TARGET_EPSILON),
                ),
                min_probability=float(calibrated_prob_threshold),
                min_expected_value_bps=max(
                    float(CALIB_MIN_EXPECTED_VALUE_BPS),
                    float(best["ev"])
                    * float(EXACT_THRESHOLD_EV_TARGET_FRACTION),
                ),
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
                calibrated_selected_window_minutes=float(best["median_selected_window"]),
                calibrated_post_profit_breathing_minutes=float(CALIB_POST_PROFIT_BREATHING_MINUTES),
                calibrated_post_profit_extra_gain_bps=float(best["median_post_profit_extra_gain"]),
                calibrated_max_adverse_before_profit_bps=float(best["median_adverse_before_profit"]),
                calibrated_expected_bps_per_minute=(
                    float(best["ev"])
                    / max(
                        1.0,
                        float(best["median_time_to_min_profit"] or 1.0),
                    )
                ),
                calibrated_raw_probability_median=float(
                    selected_raw_prob_median
                ),
                calibrated_empirical_win_rate=float(
                    selected_empirical_win_rate
                ),
                calibrated_probability_model_note=(
                    "source=exact_threshold; "
                    f"raw_prob_median={selected_raw_prob_median:.6f}; "
                    "selected_empirical_win_rate="
                    f"{selected_empirical_win_rate:.6f}"
                ),
                reason=(
                    f"exact_threshold product={product_id} "
                    f"score>={best['score_threshold']:.6f} "
                    f"prob>={best['prob_threshold']:.6f} "
                    f"samples={best['n']} "
                    f"win_rate={best['win_rate']:.6f} "
                    f"ev={best['ev']:.6f} "
                    f"avg_win={best['avg_win']:.6f} "
                    f"avg_loss={best['avg_loss']:.6f} "
                    f"window={best['median_selected_window']:.1f}m "
                    f"post_profit_extra={best['median_post_profit_extra_gain']:.2f} "
                    f"adverse_before_profit={best['median_adverse_before_profit']:.2f}"
                ),
            )

        winning_obs = [o for o in all_obs if o.reached_min_profit]

        if ALLOW_BEST_AVAILABLE_LEARNED_PROFILE and all_obs:
            status = (
                f"exact_threshold_not_found winners_exist={len(winning_obs)} "
                f"using_best_available_learned_profile"
                if winning_obs
                else (
                    f"no_winning_observations total={len(all_obs)} "
                    f"using_top_motion_learned_profile"
                )
            )
            return self._build_best_available_learned_profile(
                product_id=product_id,
                observations=all_obs,
                day_obs=day_obs,
                week_obs=week_obs,
                status=status,
            )

        return self._uncalibrated_profile(
            product_id=product_id,
            status=(
                f"no_observations_available total={len(all_obs)} "
                f"overall_ev={blended_ev:.6f}"
            ),
            day_obs=day_obs,
            week_obs=week_obs,
            blended_ev=blended_ev,
            avg_win_bps=blended_avg_win,
            avg_loss_bps=blended_avg_loss,
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
        if not profile.is_calibrated:
            return profile
        required = CALIB_MIN_PREFIX_CANDLES_1M + max(CALIB_FORWARD_WINDOWS_1M) + 1
        if not candles or len(candles) < required:
            return profile
        scalp_results: Dict[float, List[float]] = {
            pullback: [] for pullback in CALIB_SCALP_PULLBACK_CANDIDATES
        }
        core_results: Dict[float, List[float]] = {
            pullback: [] for pullback in CALIB_CORE_PULLBACK_CANDIDATES
        }
        end_i = len(candles) - max(CALIB_FORWARD_WINDOWS_1M)
        for i in range(CALIB_MIN_PREFIX_CANDLES_1M, end_i):
            prefix = candles[:i]
            future = candles[i:i + max(CALIB_FORWARD_WINDOWS_1M)]
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
            if not profile.is_calibrated:
                continue
            if signal.score < profile.min_score:
                continue
            if signal.estimated_prob_up < profile.min_probability:
                continue
            if signal.expected_net_edge_bps < profile.min_expected_value_bps:
                continue
            outcome = self._evaluate_forward_outcome(
                entry_price=entry_price,
                future_candles=future,
                target_bps=signal.target_bps,
                cost_bps=signal.cost_bps,
                min_net_gain_bps=MIN_NET_GAIN_AFTER_FEES_BPS,
                bar_minutes=1.0,
            )
            if not outcome[2] or not outcome[-1]:
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
                if not day_candles and not week_candles:
                    log(
                        f"[calibration] no historical candles for {product}; "
                        f"product remains uncalibrated"
                    )
                    profile = self._uncalibrated_profile(
                        product_id=product,
                        status="no_historical_candles",
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
                day_obs = self._walk_forward_observations_multi_window(
                    product_id=product,
                    candles=day_candles,
                    weekly_candles=week_candles,
                    timeframe="day_1m",
                    min_prefix=CALIB_MIN_PREFIX_CANDLES_1M,
                    forward_windows=CALIB_FORWARD_WINDOWS_1M,
                    spread_bps=spread_bps,
                ) if day_candles else []
                week_obs = self._walk_forward_observations_multi_window(
                    product_id=product,
                    candles=week_candles,
                    weekly_candles=week_candles,
                    timeframe="week_15m",
                    min_prefix=CALIB_MIN_PREFIX_CANDLES_15M,
                    forward_windows=CALIB_FORWARD_WINDOWS_15M,
                    spread_bps=spread_bps,
                ) if week_candles else []
                log(
                    f"[calibration-debug] {product} "
                    f"day_obs={len(day_obs)} week_obs={len(week_obs)} "
                    f"spread_bps={spread_bps:.3f}"
                )
                self._append_calibration_observations(
                    product_id=product,
                    observations=day_obs + week_obs,
                )
                profile = self._build_calibration_profile(
                    product_id=product,
                    day_obs=day_obs,
                    week_obs=week_obs,
                )
                try:
                    self.candidate_replay_log.log_observations(day_obs + week_obs)
                except Exception as replay_error:
                    log(f"[candidate-replay] failed to log startup replay for {product}: {replay_error}")
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
                profile = self._uncalibrated_profile(
                    product_id=product,
                    status=f"calibration_error={exc}",
                )
                self.calibration_profiles[product] = profile
                self.clog.log_profile(profile)
        log("[calibration] startup walk-forward calibration finished")

    def _run_hourly_banked_recalibration(self) -> None:
        """Append recent observations and rebuild profiles from the full bank."""
        for product in PRODUCTS:
            try:
                live_rows = self.live_1m[product].export_rows(product)
                minimum_rows = (
                    CALIB_MIN_PREFIX_CANDLES_1M
                    + max(CALIB_FORWARD_WINDOWS_1M)
                    + 5
                )
                if len(live_rows) < minimum_rows:
                    log(
                        f"[calibration-hourly] {product} skipped: "
                        f"live_rows={len(live_rows)} required={minimum_rows}; "
                        f"keeping existing profile"
                    )
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

                tob = self.tob.get(product)
                spread_bps = (
                    float(tob.spread_bps)
                    if tob and tob.spread_bps > 0
                    else float(MAX_SPREAD_BPS)
                )

                recent_candles = live_candles[-max(360, minimum_rows):]
                forward_windows = [
                    window
                    for window in CALIB_FORWARD_WINDOWS_1M
                    if window <= max(5, len(recent_candles) // 3)
                ]
                new_obs = self._walk_forward_observations_multi_window(
                    product_id=product,
                    candles=recent_candles,
                    weekly_candles=live_candles,
                    timeframe="live_rolling_1m",
                    min_prefix=CALIB_MIN_PREFIX_CANDLES_1M,
                    forward_windows=forward_windows,
                    spread_bps=spread_bps,
                )

                if not new_obs:
                    log(
                        f"[calibration-hourly] {product} no new observations; "
                        f"keeping existing profile"
                    )
                    continue

                self._append_calibration_observations(
                    product_id=product,
                    observations=new_obs,
                )

                try:
                    self.candidate_replay_log.log_observations(new_obs)
                except Exception as replay_error:
                    log(
                        f"[candidate-replay] failed to log hourly replay for "
                        f"{product}: {replay_error}"
                    )

                bank_obs = self._observations_for_profile(
                    product_id=product,
                    fallback=new_obs,
                )
                if not bank_obs:
                    log(
                        f"[calibration-hourly] {product} bank empty after append; "
                        f"keeping existing profile"
                    )
                    continue

                profile_new = self._build_calibration_profile(
                    product_id=product,
                    day_obs=bank_obs,
                    week_obs=[],
                )

                if not profile_new.is_calibrated:
                    log(
                        f"[calibration-hourly] {product} rebuilt profile not "
                        f"calibrated; keeping existing profile "
                        f"status={profile_new.calibration_status}"
                    )
                    continue

                self.calibration_profiles[product] = profile_new
                self.clog.log_profile(profile_new)

                log(
                    f"[calibration-hourly] {product} updated "
                    f"score={profile_new.min_score:.6f} "
                    f"prob={profile_new.min_probability:.6f} "
                    f"ev={profile_new.min_expected_value_bps:.6f} "
                    f"status={profile_new.calibration_status} "
                    f"bank_n={len(bank_obs)}"
                )
            except Exception as exc:
                log_exception(
                    f"[calibration-hourly] failed for {product}",
                    exc,
                )


    def _latest_live_signal_for_product(self, product_id: str) -> Optional[LiveSignal]:
        tob = self.tob.get(product_id)
        series = self.live_1m.get(product_id)
        if not tob or tob.mid <= 0 or series is None:
            return None
        minute_candles = list(series.candles)
        if not minute_candles:
            return None
        levels_day = self.macro.get_levels(product_id, "day")
        levels_week = self.macro.get_levels(product_id, "week")
        weekly_bias = self.macro.compute_weekly_bias(product_id, tob.mid) if levels_week else None
        return self._build_live_signal(
            product_id=product_id,
            mid=float(tob.mid),
            spread_bps=float(tob.spread_bps),
            levels_day=levels_day,
            levels_week=levels_week,
            minute_candles=minute_candles,
            weekly_bias=weekly_bias,
            sigma_bps=self._compute_sigma_bps_from_1m(product_id),
        )

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
        profile_is_calibrated = bool(getattr(profile, "is_calibrated", False))
        raw_calib_min_score = float(profile.min_score)
        raw_calib_min_probability = float(profile.min_probability)
        raw_calib_min_ev = float(profile.min_expected_value_bps)
        calibration_targets_valid = bool(
            np.isfinite(raw_calib_min_score)
            and raw_calib_min_score > 0.0
            and np.isfinite(raw_calib_min_probability)
            and raw_calib_min_probability > 0.0
            and np.isfinite(raw_calib_min_ev)
            and raw_calib_min_ev > 0.0
        )
        buy_gate_calibration_ready = bool(
            profile_is_calibrated and calibration_targets_valid
        )

        if not buy_gate_calibration_ready:
            log(
                f"[calibration-status] {product_id} AWAITING_CALIBRATION "
                f"status={profile.calibration_status} reason={profile.reason}"
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
            projected_forward_gain_bps=calibrated_forward_gain_bps,
            expected_net_edge_bps=expected_net_edge_bps,
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

        if buy_gate_calibration_ready:
            calib_min_score = raw_calib_min_score
            calib_min_probability = raw_calib_min_probability
            calib_min_ev = raw_calib_min_ev
        else:
            calib_min_score = math.nan
            calib_min_probability = math.nan
            calib_min_ev = math.nan

        # Individual buy-gate checks.
        buy_gate_fee_ok = bool(fee_available and round_trip_cost_bps is not None)
        buy_gate_score_target_ok = bool(
            buy_gate_calibration_ready
            and np.isfinite(calib_min_score)
            and score >= calib_min_score
        )
        buy_gate_prob_target_ok = bool(
            buy_gate_calibration_ready
            and np.isfinite(calib_min_probability)
            and estimated_prob_up >= calib_min_probability
        )

        # Floors are diagnostic only. They do not authorize buys.
        buy_gate_score_floor_ok = bool(score >= float(EV_PRIMARY_MIN_SCORE_FLOOR))
        buy_gate_prob_floor_ok = bool(
            estimated_prob_up >= float(EV_PRIMARY_MIN_PROB_FLOOR)
        )

        required_min_ev = (
            max(float(MIN_REQUIRED_NET_EDGE_BPS), calib_min_ev)
            if buy_gate_calibration_ready and np.isfinite(calib_min_ev)
            else math.nan
        )
        buy_gate_ev_ok = bool(
            buy_gate_calibration_ready
            and np.isfinite(calib_min_ev)
            and expected_net_edge_bps >= required_min_ev
        )

        # Actual buy permission must use calibrated targets only.
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
            buy_gate_score_target_ok
            and buy_gate_prob_target_ok
            and buy_gate_ev_ok
        )

        if SIMPLIFY_BUY_GATE_TO_THREE_REQUIREMENTS:
            # The actual strategy decision:
            # Buy when the three core calibrated requirements pass.
            three_requirement_signal_ok = bool(
                buy_gate_score_target_ok
                and buy_gate_prob_target_ok
                and buy_gate_ev_ok
            )

            # Operational readiness:
            # Fee data is required so EV/cost math is grounded in real Coinbase fees.
            operational_ok = bool(
                buy_gate_calibration_ready
                and (buy_gate_fee_ok if FEE_DATA_REQUIRED_FOR_LIVE_BUY else True)
            )

            ok_to_trade = bool(
                three_requirement_signal_ok
                and operational_ok
            )
        else:
            # Old multi-gate behavior, kept only as fallback.
            ok_to_trade = bool(
                buy_gate_calibration_ready
                and buy_gate_fee_ok
                and buy_gate_setup_ok
                and buy_gate_calibrated_ok
                and buy_gate_target_cost_ok
                and buy_gate_spread_ok
            )

        blockers = []

        if SIMPLIFY_BUY_GATE_TO_THREE_REQUIREMENTS:
            if not buy_gate_calibration_ready:
                blockers.append(
                    f"calibration_not_ready:{profile.calibration_status}"
                )
            if FEE_DATA_REQUIRED_FOR_LIVE_BUY and not buy_gate_fee_ok:
                blockers.append("fee_data_not_ready")

            if not buy_gate_score_target_ok:
                blockers.append("score_below_calibrated_target")

            if not buy_gate_prob_target_ok:
                blockers.append("probability_below_calibrated_target")

            if not buy_gate_ev_ok:
                blockers.append("ev_below_calibrated_target")

            # Diagnostic-only notes. These should not block buys in simplified mode.
            diagnostics = []
            if not buy_gate_spread_ok:
                diagnostics.append("spread_would_have_blocked_old_gate")
            if not buy_gate_target_cost_ok:
                diagnostics.append("target_cost_would_have_blocked_old_gate")
            if not buy_gate_setup_ok:
                diagnostics.append(
                    f"setup_would_have_blocked_old_gate:{setup_blocker}"
                )

            diagnostic_note = (
                ",".join(diagnostics) if diagnostics else "diagnostics_clear"
            )
            buy_gate_mode = "simplified_three_requirements"
        else:
            if not buy_gate_calibration_ready:
                blockers.append(
                    f"calibration_not_ready:{profile.calibration_status}"
                )
            if not buy_gate_fee_ok:
                blockers.append("fee_not_ready")
            if not buy_gate_score_ok:
                blockers.append("score_below_target")
            if not buy_gate_prob_ok:
                blockers.append("probability_below_target")
            if not buy_gate_ev_ok:
                blockers.append("ev_below_target")
            if not buy_gate_target_cost_ok:
                blockers.append("target_to_cost_failed")
            if not buy_gate_spread_ok:
                blockers.append("spread_too_wide")
            if not buy_gate_setup_ok:
                blockers.append(setup_blocker)

            diagnostic_note = "old_gate_mode"
            buy_gate_mode = "legacy_multi_gate"

        buy_gate_blocker = "BUY_READY" if ok_to_trade else ";".join(blockers)

        last_logged = self.last_buy_gate_log_ts_by_product.get(
            product_id, 0.0
        )
        log_ts = now_ts()
        should_log_buy_gate = bool(
            ok_to_trade or log_ts - float(last_logged) >= 15.0
        )
        if should_log_buy_gate:
            self.last_buy_gate_log_ts_by_product[product_id] = log_ts

        if should_log_buy_gate and ok_to_trade:
            log(
                f"[buy-gate] {product_id} BUY_READY "
                f"mode={buy_gate_mode} "
                f"diagnostics={diagnostic_note} "
                f"calibrated={buy_gate_calibration_ready} "
                f"calibration_status={profile.calibration_status} "
                f"score={score:.3f} min_score={calib_min_score:.3f} "
                f"score_floor={EV_PRIMARY_MIN_SCORE_FLOOR:.3f} score_floor_ok={buy_gate_score_floor_ok} "
                f"score_ok={buy_gate_score_ok} score_target_ok={buy_gate_score_target_ok} "
                f"prob={estimated_prob_up:.6f} min_prob={calib_min_probability:.6f} "
                f"raw_prob_median={profile.calibrated_raw_probability_median:.6f} "
                f"empirical_wr={profile.calibrated_empirical_win_rate:.6f} "
                f"prob_floor={EV_PRIMARY_MIN_PROB_FLOOR:.6f} prob_floor_ok={buy_gate_prob_floor_ok} "
                f"prob_ok={buy_gate_prob_ok} prob_target_ok={buy_gate_prob_target_ok} "
                f"ev_primary={USE_EV_PRIMARY_BUY_GATE} "
                f"ev_primary_floor={EV_PRIMARY_MIN_PROJECTED_NET_BPS:.3f} "
                f"ev={expected_net_edge_bps:.3f} "
                f"min_ev={required_min_ev:.3f} "
                f"ev_ok={buy_gate_ev_ok} "
                f"target={target_bps:.3f} "
                f"projected_forward={calibrated_forward_gain_bps:.3f} "
                f"cost={cost_bps:.3f} "
                f"spread={spread_bps:.3f}"
            )
        elif should_log_buy_gate:
            log(
                f"[buy-gate] {product_id} BLOCKED "
                f"mode={buy_gate_mode} "
                f"blocker={buy_gate_blocker} "
                f"diagnostics={diagnostic_note} "
                f"calibrated={buy_gate_calibration_ready} "
                f"calibration_status={profile.calibration_status} "
                f"score={score:.3f} min_score={calib_min_score:.3f} "
                f"score_floor={EV_PRIMARY_MIN_SCORE_FLOOR:.3f} score_floor_ok={buy_gate_score_floor_ok} "
                f"score_ok={buy_gate_score_ok} score_target_ok={buy_gate_score_target_ok} "
                f"prob={estimated_prob_up:.6f} min_prob={calib_min_probability:.6f} "
                f"raw_prob_median={profile.calibrated_raw_probability_median:.6f} "
                f"empirical_wr={profile.calibrated_empirical_win_rate:.6f} "
                f"prob_floor={EV_PRIMARY_MIN_PROB_FLOOR:.6f} prob_floor_ok={buy_gate_prob_floor_ok} "
                f"prob_ok={buy_gate_prob_ok} prob_target_ok={buy_gate_prob_target_ok} "
                f"ev_primary={USE_EV_PRIMARY_BUY_GATE} "
                f"ev_primary_floor={EV_PRIMARY_MIN_PROJECTED_NET_BPS:.3f} "
                f"ev={expected_net_edge_bps:.3f} "
                f"min_ev={required_min_ev:.3f} ev_ok={buy_gate_ev_ok} "
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
        """Convert a positive, buy-ready probability into an equity allocation."""
        p = clamp_float(float(estimated_prob_up), 0.0, 1.0)

        if p <= 0.0:
            return 0.0

        # The entry gate decides whether a signal is tradable. A passing signal
        # below the preferred sizing range still receives the minimum allocation.
        if p <= float(PROB_FOR_MIN_SIZE):
            return float(MIN_POSITION_PCT_OF_EQUITY)

        span = max(1e-9, float(PROB_FOR_MAX_SIZE) - float(PROB_FOR_MIN_SIZE))
        t = clamp_float((p - float(PROB_FOR_MIN_SIZE)) / span, 0.0, 1.0)
        return float(MIN_POSITION_PCT_OF_EQUITY) + t * (
            float(MAX_SINGLE_BUY_PCT_OF_EQUITY)
            - float(MIN_POSITION_PCT_OF_EQUITY)
        )

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

    def _recent_momentum_bps_for_product(self, product_id: str, lookback: int) -> float:
        """Return recent close-to-close momentum in bps from live 1m candles."""
        try:
            candles = list(self.live_1m[product_id].candles)
        except Exception:
            candles = []

        if not candles or len(candles) <= int(lookback):
            return 0.0

        try:
            old_px = float(candles[-int(lookback) - 1].close)
            new_px = float(candles[-1].close)
            if old_px <= 0 or new_px <= 0:
                return 0.0
            return ((new_px / old_px) - 1.0) * 10000.0
        except Exception:
            return 0.0

    def _recent_green_candle_count(self, product_id: str, lookback: int) -> int:
        try:
            candles = list(self.live_1m[product_id].candles)
        except Exception:
            candles = []
        count = 0
        for candle in candles[-max(1, int(lookback)):]:
            try:
                if float(candle.close) > float(candle.open):
                    count += 1
            except Exception:
                pass
        return count

    def _recent_lower_low_sequence(self, product_id: str, lookback: int = 4) -> bool:
        try:
            candles = list(self.live_1m[product_id].candles)
        except Exception:
            candles = []
        if len(candles) < max(2, int(lookback)):
            return False
        try:
            lows = [float(c.low) for c in candles[-int(lookback):]]
            return all(lows[i] < lows[i - 1] for i in range(1, len(lows)))
        except Exception:
            return False

    def _entry_momentum_snapshot(self, product_id: str) -> Dict[str, float]:
        return {
            "mom1": self._recent_momentum_bps_for_product(product_id, 1),
            "mom3": self._recent_momentum_bps_for_product(product_id, 3),
            "mom5": self._recent_momentum_bps_for_product(product_id, 5),
            "mom15": self._recent_momentum_bps_for_product(product_id, 15),
        }

    def _max_buy_spread_for_product(self, product_id: str) -> float:
        normalized = str(product_id).upper().strip()
        return float(PRODUCT_MAX_BUY_SPREAD_BPS.get(normalized, HARD_MAX_BUY_SPREAD_BPS))

    def _spread_allows_buy(self, product_id: str, spread_bps: float) -> Tuple[bool, str]:
        if not ENABLE_HARD_SPREAD_FILTER_FOR_BUYS:
            return True, "spread_filter_disabled"
        max_spread = self._max_buy_spread_for_product(product_id)
        if float(spread_bps) > max_spread:
            return False, f"spread_too_high spread={float(spread_bps):.3f} max={max_spread:.3f}"
        return True, f"spread_ok spread={float(spread_bps):.3f} max={max_spread:.3f}"

    def _level8_decision_for_candidate(
        self,
        *,
        candidate: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        product_id = str(candidate.get("product_id", ""))
        fallback = {
            "action": "WAIT",
            "strategy": candidate.get("manager_strategy", "LEVEL8_UNAVAILABLE"),
            "bucket": "BLOCKED",
            "risk_mode": "NORMAL",
            "recommended_position_pct": 0.0,
            "decision_id": "",
            "truth_score": 0.0,
            "final_buy_score": 0.0,
            "buy_threshold": 0.0,
            "confidence": 0.0,
            "reason": "level8_unavailable_fail_closed",
        }
        if not ENABLE_LEVEL8_COUNCIL or self.level8_council is None:
            return False, fallback

        try:
            try:
                decision = self.level8_council.decide_buy(
                    product_id=product_id,
                    candidate=candidate,
                )
            except TypeError:
                probability = clamp_float(float(candidate.get("estimated_prob_up", 0.5)), 0.0, 1.0)
                score = clamp_float(float(candidate.get("score", 0.0)) / 100.0, 0.0, 1.0)
                expected_edge = float(candidate.get("expected_net_edge_bps", 0.0))
                edge_score = clamp_float(0.5 + expected_edge / 240.0, 0.0, 1.0)
                spread_bps = float(candidate.get("spread_bps", 0.0))
                spread_quality = clamp_float(1.0 - max(0.0, spread_bps) / 100.0, 0.0, 1.0)
                projected_forward = float(candidate.get("projected_forward_gain_bps", 0.0))
                forward_score = clamp_float(0.5 + projected_forward / 320.0, 0.0, 1.0)
                cost_bps = float(candidate.get("cost_bps", 0.0))
                cost_score = clamp_float(1.0 - max(0.0, cost_bps) / 400.0, 0.0, 1.0)

                moms = self._entry_momentum_snapshot(product_id)
                mom1 = float(moms.get("mom1", 0.0))
                mom3 = float(moms.get("mom3", 0.0))
                mom5 = float(moms.get("mom5", 0.0))
                mom15 = float(moms.get("mom15", 0.0))
                trend_score = clamp_float(0.50 + mom5 / 180.0 + mom15 / 300.0, 0.0, 1.0)
                mean_reversion_score = clamp_float(0.45 + max(0.0, -mom5) / 180.0 + max(0.0, mom1) / 160.0, 0.0, 1.0)
                breakout_score = clamp_float(0.45 + max(0.0, mom3) / 150.0 + max(0.0, mom5) / 220.0, 0.0, 1.0)
                risk_vote = self.level8_council.risk_agent()
                votes = [
                    {"agent": "trend", "buy": trend_score, "sell": clamp_float(1.0-trend_score,0.0,1.0), "hold": clamp_float(0.40+trend_score*0.30,0.0,1.0), "wait": clamp_float(0.60-trend_score*0.35,0.0,1.0), "confidence": clamp_float(0.35+abs(mom5)/180.0,0.20,0.90)},
                    {"agent": "mean_reversion", "buy": mean_reversion_score, "sell": clamp_float(1.0-mean_reversion_score,0.0,1.0), "hold": 0.45, "wait": clamp_float(0.55-mean_reversion_score*0.25,0.0,1.0), "confidence": clamp_float(0.35+abs(mom5)/220.0,0.20,0.85)},
                    {"agent": "breakout", "buy": breakout_score, "sell": clamp_float(1.0-breakout_score,0.0,1.0), "hold": clamp_float(0.40+breakout_score*0.20,0.0,1.0), "wait": clamp_float(0.65-breakout_score*0.30,0.0,1.0), "confidence": clamp_float(0.30+max(0.0,mom3)/160.0,0.20,0.85)},
                    {"agent": "ai_outcome", "buy": probability, "sell": clamp_float(1.0-probability,0.0,1.0), "hold": clamp_float(0.35+probability*0.35,0.0,1.0), "wait": clamp_float(0.70-probability*0.40,0.0,1.0), "confidence": clamp_float(0.25+abs(probability-0.5)*1.20,0.20,0.85)},
                    {"agent": "execution", "buy": clamp_float(spread_quality*0.45+cost_score*0.35+edge_score*0.20,0.0,1.0), "sell": clamp_float(1.0-spread_quality*0.60,0.0,1.0), "hold": 0.45, "wait": clamp_float(1.0-spread_quality,0.0,1.0), "confidence": clamp_float(0.30+spread_quality*0.55,0.20,0.95)},
                    {"agent": "product_health", "buy": clamp_float(score*0.35+edge_score*0.30+forward_score*0.35,0.0,1.0), "sell": clamp_float(1.0-forward_score,0.0,1.0), "hold": clamp_float(0.40+forward_score*0.30,0.0,1.0), "wait": clamp_float(0.65-forward_score*0.35,0.0,1.0), "confidence": clamp_float(0.30+score*0.50,0.20,0.85)},
                    {"agent": "risk", "buy": float(risk_vote.get("buy",0.55)), "sell": float(risk_vote.get("sell",0.42)), "hold": float(risk_vote.get("hold",0.55)), "wait": float(risk_vote.get("wait",0.40)), "confidence": float(risk_vote.get("confidence",0.50))},
                ]
                truth_buy = clamp_float(spread_quality*0.25+cost_score*0.15+probability*0.25+score*0.20+forward_score*0.15,0.0,1.0)
                truth_vote = {"agent":"truth", "buy":truth_buy, "sell":clamp_float(1.0-truth_buy,0.0,1.0), "hold":clamp_float(0.35+truth_buy*0.30,0.0,1.0), "wait":clamp_float(0.80-truth_buy*0.55,0.0,1.0), "confidence":clamp_float(0.30+truth_buy*0.55,0.20,0.90)}
                strategy = str(candidate.get("manager_strategy", "LEVEL8_DIRECT"))
                decision = self.level8_council.decide_buy(product_id=product_id, strategy=strategy, votes=votes, truth_vote=truth_vote)
        except Exception as exc:
            log(f"[level8] decision failed for {product_id}: {exc}")
            return False, {**fallback, "reason": f"level8_decision_failed_fail_closed:{exc}"}

        if isinstance(decision, dict):
            action = decision.get("action", "WAIT")
            strategy = candidate.get("manager_strategy", "LEGACY")
            bucket = decision.get("bucket", "SHADOW")
            risk_mode = decision.get("risk_mode", "NORMAL")
            recommended_position_pct = decision.get("position_pct", 0.0)
            decision_id = decision.get("decision_id", str(uuid.uuid4()))
            truth_score = decision.get("truth_score", 0.0)
            final_buy_score = decision.get("final_buy", 0.0)
            buy_threshold = decision.get("buy_threshold", 0.0)
            confidence = decision.get("confidence", truth_score)
            reason = decision.get("sizing_reason", decision.get("reason", ""))
        else:
            action = decision.action
            strategy = decision.strategy
            bucket = decision.bucket
            risk_mode = decision.risk_mode
            recommended_position_pct = decision.recommended_position_pct
            decision_id = decision.decision_id
            truth_score = decision.truth_score
            final_buy_score = decision.final_buy_score
            buy_threshold = decision.buy_threshold
            confidence = decision.confidence
            reason = decision.reason

        info = {
            "action": action,
            "strategy": strategy,
            "bucket": bucket,
            "risk_mode": risk_mode,
            "recommended_position_pct": float(recommended_position_pct),
            "decision_id": decision_id,
            "truth_score": float(truth_score),
            "final_buy_score": float(final_buy_score),
            "buy_threshold": float(buy_threshold),
            "confidence": float(confidence),
            "reason": reason,
        }
        try:
            if isinstance(decision, dict):
                self._append_level8_vote_snapshots(
                    product_id=product_id,
                    decision_id=str(info.get("decision_id", "")),
                    strategy=str(info.get("strategy", candidate.get("manager_strategy", "LEVEL8_DIRECT"))),
                    votes=decision.get("votes", []),
                    truth_vote=decision.get("truth_vote", {}),
                    reason=str(info.get("reason", "")),
                )
        except Exception as exc:
            log(f"[level8] vote snapshot write failed {product_id}: {exc}")

        allow = str(action).upper() == "ALLOW_BUY"
        if str(action).upper() == "SHADOW":
            allow = False
        return bool(allow), info

    def _run_level8_council_heartbeat(
        self,
        *,
        watch_candidates: List[Dict[str, Any]],
    ) -> None:
        """
        Runs Level 8 council commentary even when no live buy candidate passes.

        This keeps:
        - council_votes.csv
        - council_decisions.csv
        - agent_adjustments.csv
        - adaptive_thresholds.csv
        - shadow_trades.csv

        actively updating so the viewer can show the council discussing both
        positive and negative market conditions.
        """
        if not ENABLE_LEVEL8_COUNCIL:
            return

        if not LEVEL8_ENABLE_COUNCIL_HEARTBEAT:
            return

        if self.level8_council is None:
            return

        nowv = now_ts()
        if (
            nowv - float(self.last_level8_council_heartbeat_ts)
            < float(LEVEL8_COUNCIL_HEARTBEAT_EVERY_SEC)
        ):
            return

        self.last_level8_council_heartbeat_ts = nowv

        usable = [
            dict(c)
            for c in watch_candidates
            if str(c.get("product_id", "")).strip()
        ]

        if not usable:
            return

        usable.sort(
            key=lambda c: (
                float(c.get("expected_net_edge_bps", 0.0)),
                float(c.get("score", 0.0)),
                float(c.get("estimated_prob_up", 0.0)),
            ),
            reverse=True,
        )

        for candidate in usable[: int(LEVEL8_COUNCIL_HEARTBEAT_MAX_PRODUCTS)]:
            product_id = str(candidate.get("product_id", ""))

            candidate["heartbeat_only"] = True
            candidate["manager_strategy"] = (
                candidate.get("manager_strategy") or "COUNCIL_HEARTBEAT"
            )

            try:
                level8_ok, level8_info = self._level8_decision_for_candidate(
                    candidate=candidate,
                )

                decision_id = str(level8_info.get("decision_id", ""))
                action = str(level8_info.get("action", "WAIT"))
                strategy = str(
                    level8_info.get(
                        "strategy",
                        candidate.get(
                            "manager_strategy", "COUNCIL_HEARTBEAT"
                        ),
                    )
                )
                bucket = str(level8_info.get("bucket", "SHADOW"))
                truth_score = float(
                    level8_info.get("truth_score", 0.0) or 0.0
                )
                final_buy = float(
                    level8_info.get("final_buy_score", 0.0) or 0.0
                )
                buy_threshold = float(
                    level8_info.get("buy_threshold", 0.0) or 0.0
                )
                recommended_pct = float(
                    level8_info.get("recommended_position_pct", 0.0) or 0.0
                )
                confidence = float(
                    level8_info.get("confidence", truth_score) or 0.0
                )
                reason = str(level8_info.get("reason", ""))

                self._append_level8_decision_snapshot(
                    product_id=product_id,
                    decision_id=decision_id,
                    action=action,
                    strategy=strategy,
                    bucket=bucket,
                    risk_mode=str(level8_info.get("risk_mode", "NORMAL")),
                    truth_score=truth_score,
                    final_buy_score=final_buy,
                    final_sell_score=0.0,
                    buy_threshold=buy_threshold,
                    sell_threshold=0.0,
                    recommended_position_pct=recommended_pct,
                    confidence=confidence,
                    reason=(
                        f"heartbeat_only=True;"
                        f"why_not_ready={candidate.get('why_not_ready', '')};"
                        f"{reason}"
                    ),
                )

                self.signal_events_log.log_event(
                    event_type="level8_council_heartbeat",
                    trade_id=decision_id,
                    product_id=product_id,
                    rank_score=(
                        f"{float(candidate.get('rank_score', 0.0)):.6f}"
                    ),
                    buy_ready_count=0,
                    score=f"{float(candidate.get('score', 0.0)):.6f}",
                    probability=(
                        f"{float(candidate.get('estimated_prob_up', 0.0)):.6f}"
                    ),
                    ev_bps=(
                        f"{float(candidate.get('expected_net_edge_bps', 0.0)):.6f}"
                    ),
                    projected_forward_bps=(
                        f"{float(candidate.get('projected_forward_gain_bps', 0.0)):.6f}"
                    ),
                    cost_bps=f"{float(candidate.get('cost_bps', 0.0)):.6f}",
                    spread_bps=(
                        f"{float(candidate.get('spread_bps', 0.0)):.6f}"
                    ),
                    action="commentary",
                    reason=(
                        f"heartbeat_only=True;"
                        f"decision={action};"
                        f"strategy={strategy};"
                        f"bucket={bucket};"
                        f"truth={truth_score:.3f};"
                        f"final_buy={final_buy:.3f};"
                        f"threshold={buy_threshold:.3f};"
                        f"recommended_pct={recommended_pct:.3f};"
                        f"why_not_ready={candidate.get('why_not_ready', '')};"
                        f"{reason}"
                    ),
                )

            except Exception as exc:
                log(f"[level8] council heartbeat failed {product_id}: {exc}")

    def _append_level8_vote_snapshots(
        self, *, product_id: str, decision_id: str, strategy: str,
        votes: List[Dict[str, Any]], truth_vote: Dict[str, Any], reason: str,
    ) -> None:
        """Write council votes so each member can be graded against outcomes."""
        try:
            path = os.path.join(BASE_DIR, "council_votes.csv")
            columns = ["ts", "dt_mst", "decision_id", "product_id", "agent", "strategy", "raw_buy_score", "raw_sell_score", "raw_hold_score", "raw_wait_score", "adjusted_buy_score", "adjusted_sell_score", "adjusted_hold_score", "adjusted_wait_score", "confidence", "reliability", "product_adjustment", "strategy_adjustment", "recent_performance_adjustment", "weight", "reason"]
            write_header = not os.path.exists(path) or os.path.getsize(path) == 0
            ts_val = now_ts()
            dt_mst = datetime.fromtimestamp(ts_val, tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
            rows = list(votes or [])
            if truth_vote:
                rows.append(dict(truth_vote))
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(columns)
                for vote in rows:
                    confidence = float(vote.get("confidence", 0.0) or 0.0)
                    reliability = float(vote.get("reliability", 1.0) or 1.0)
                    writer.writerow([f"{ts_val:.6f}", dt_mst, decision_id, product_id, str(vote.get("agent", "unknown")), strategy,
                        f"{float(vote.get('buy', vote.get('raw_buy_score', 0.0)) or 0.0):.6f}", f"{float(vote.get('sell', vote.get('raw_sell_score', 0.0)) or 0.0):.6f}", f"{float(vote.get('hold', vote.get('raw_hold_score', 0.0)) or 0.0):.6f}", f"{float(vote.get('wait', vote.get('raw_wait_score', 0.0)) or 0.0):.6f}",
                        f"{float(vote.get('adjusted_buy_score', vote.get('buy', 0.0)) or 0.0):.6f}", f"{float(vote.get('adjusted_sell_score', vote.get('sell', 0.0)) or 0.0):.6f}", f"{float(vote.get('adjusted_hold_score', vote.get('hold', 0.0)) or 0.0):.6f}", f"{float(vote.get('adjusted_wait_score', vote.get('wait', 0.0)) or 0.0):.6f}", f"{confidence:.6f}", f"{reliability:.6f}", "0.000000", "0.000000", "0.000000", f"{max(0.0, confidence * reliability):.6f}", reason])
        except Exception as exc:
            log(f"[level8] vote snapshot append failed: {exc}")

    def _append_level8_decision_snapshot(
        self,
        *,
        product_id: str,
        decision_id: str,
        action: str,
        strategy: str,
        bucket: str,
        risk_mode: str,
        truth_score: float,
        final_buy_score: float,
        final_sell_score: float,
        buy_threshold: float,
        sell_threshold: float,
        recommended_position_pct: float,
        confidence: float,
        reason: str,
    ) -> None:
        """
        Write a normalized Level 8 decision row for live and heartbeat decisions.
        """
        try:
            columns = [
                "ts",
                "dt_mst",
                "decision_id",
                "product_id",
                "action",
                "strategy",
                "bucket",
                "risk_mode",
                "truth_score",
                "final_buy_score",
                "final_sell_score",
                "buy_threshold",
                "sell_threshold",
                "recommended_position_pct",
                "confidence",
                "reason",
            ]
            write_header = not os.path.exists(
                LEVEL8_COUNCIL_DECISIONS_CSV_PATH
            )
            ts_val = now_ts()
            dt_mst = (
                datetime.fromtimestamp(ts_val, tz=timezone.utc)
                .astimezone(TZ)
                .strftime("%Y-%m-%d %H:%M:%S")
            )

            with open(
                LEVEL8_COUNCIL_DECISIONS_CSV_PATH,
                "a",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(columns)
                writer.writerow([
                    f"{ts_val:.6f}",
                    dt_mst,
                    decision_id,
                    product_id,
                    action,
                    strategy,
                    bucket,
                    risk_mode,
                    f"{float(truth_score):.6f}",
                    f"{float(final_buy_score):.6f}",
                    f"{float(final_sell_score):.6f}",
                    f"{float(buy_threshold):.6f}",
                    f"{float(sell_threshold):.6f}",
                    f"{float(recommended_position_pct):.6f}",
                    f"{float(confidence):.6f}",
                    reason,
                ])
        except Exception as exc:
            log(f"[level8] decision snapshot write failed: {exc}")

    def _level8_should_hold_or_exit(
        self,
        *,
        product_id: str,
        entry_price: float,
        current_price: float,
        unrealized_bps: float,
        spread_bps: float,
        cost_bps: float,
        default_exit_reason: str,
        hard_exit: bool = False,
    ) -> Tuple[bool, str]:
        if hard_exit:
            return True, f"level8_hard_exit_bypass;{default_exit_reason}"
        if not ENABLE_LEVEL8_COUNCIL or self.level8_council is None:
            return True, f"level8_disabled;{default_exit_reason}"

        try:
            context = {
                "entry_price": float(entry_price),
                "current_price": float(current_price),
                "unrealized_bps": float(unrealized_bps),
                "spread_bps": float(spread_bps),
                "cost_bps": float(cost_bps),
                "hard_exit": bool(hard_exit),
                "strategy": "EXIT_REVIEW",
            }
            if hasattr(self.level8_council, "decide_exit"):
                decision = self.level8_council.decide_exit(
                    product_id=product_id,
                    context=context,
                )
            else:
                sell_score = clamp_float(
                    0.42
                    + max(0.0, float(unrealized_bps)) / 260.0
                    + max(0.0, -float(unrealized_bps)) / 420.0
                    + float(cost_bps) / 900.0,
                    0.0,
                    1.0,
                )
                vote = {
                    "agent": "exit_review",
                    "buy": 1.0 - sell_score,
                    "sell": sell_score,
                    "hold": 1.0 - sell_score,
                    "wait": 0.0,
                    "confidence": 0.80,
                }
                decision = self.level8_council.decide_buy(
                    product_id=product_id,
                    strategy="EXIT_REVIEW",
                    votes=[vote],
                    truth_vote={**vote, "agent": "exit_truth"},
                )
        except Exception as exc:
            log(f"[level8] exit decision failed {product_id}: {exc}")
            return True, f"level8_exit_failed_allow_legacy;{default_exit_reason}"

        if isinstance(decision, dict):
            final_sell_score = float(decision.get("final_sell", 0.0))
            sell_threshold = float(decision.get("sell_threshold", 0.0))
            truth_score = float(decision.get("truth_score", 0.0))
            reason = str(decision.get("reason", "legacy_council_exit_review"))
            if final_sell_score >= sell_threshold:
                action = "ALLOW_SELL"
            elif abs(float(unrealized_bps)) >= 90.0 and final_sell_score >= sell_threshold - 0.08:
                action = "ALLOW_SELL"
            else:
                action = "HOLD"
        else:
            final_sell_score = float(decision.final_sell_score)
            sell_threshold = float(decision.sell_threshold)
            truth_score = float(decision.truth_score)
            reason = str(decision.reason)
            action = str(decision.action).upper()

        if action == "ALLOW_SELL":
            return True, (
                f"level8_sell_allowed final_sell={final_sell_score:.3f};"
                f"threshold={sell_threshold:.3f};"
                f"truth={truth_score:.3f};"
                f"{reason};{default_exit_reason}"
            )
        return False, (
            f"level8_hold final_sell={final_sell_score:.3f};"
            f"threshold={sell_threshold:.3f};"
            f"truth={truth_score:.3f};"
            f"{reason};{default_exit_reason}"
        )

    def _entry_timing_confirmation(
        self,
        *,
        product_id: str,
        signal: Optional["LiveSignal"],
    ) -> Tuple[bool, str]:
        """Confirm that price is actually turning upward before buying."""
        trending_down, trend_reason = self._micro_trending_down(product_id)
        moms = self._entry_momentum_snapshot(product_id)
        mom1, mom3 = moms["mom1"], moms["mom3"]
        mom5, mom15 = moms["mom5"], moms["mom15"]
        vwap_ok = False
        higher_low_ok = False
        try:
            tob = self.tob.get(product_id)
            if tob and tob.mid > 0:
                vwap_ok, _ = self._micro_vwap_reclaimed(product_id, float(tob.mid))
                higher_low_ok, _ = self._higher_low_confirmed(product_id)
        except Exception:
            pass
        green_count = self._recent_green_candle_count(product_id, ENTRY_GREEN_CANDLE_LOOKBACK)
        lower_low_seq = self._recent_lower_low_sequence(product_id, lookback=4)

        detail = (
            f"mom1={mom1:.2f};mom3={mom3:.2f};mom5={mom5:.2f};mom15={mom15:.2f};"
            f"vwap={vwap_ok};hl={higher_low_ok};green={green_count};"
            f"lower_low_seq={lower_low_seq}"
        )
        if BLOCK_BUY_WHILE_MICROTREND_DOWN and trending_down:
            return False, f"microtrend_down:{trend_reason};{detail}"
        if REQUIRE_NO_LOWER_LOW_SEQUENCE_FOR_BUY and lower_low_seq:
            return False, f"lower_low_sequence;{detail}"

        momentum_confirmed = bool(
            mom1 >= float(MIN_ENTRY_MOMENTUM_1_BPS)
            and mom15 >= float(MIN_ENTRY_MOMENTUM_15_BPS)
            and (
                mom3 >= float(MIN_ENTRY_MOMENTUM_3_BPS)
                or mom5 >= float(MIN_ENTRY_MOMENTUM_5_BPS)
            )
        )
        vwap_confirmed = bool(vwap_ok) if REQUIRE_PRICE_ABOVE_MICRO_VWAP_FOR_BUY else True
        structure_confirmed = (
            bool(higher_low_ok or green_count >= int(ENTRY_MIN_GREEN_CANDLES))
            if REQUIRE_HIGHER_LOW_OR_GREEN_SEQUENCE_FOR_BUY
            else True
        )
        upturn_ok = bool(momentum_confirmed and vwap_confirmed and structure_confirmed)
        if REQUIRE_MICRO_UPTURN_FOR_BUY and not upturn_ok:
            return (
                False,
                f"no_confirmed_upturn;{detail};momentum_confirmed={momentum_confirmed};"
                f"vwap_confirmed={vwap_confirmed};structure_confirmed={structure_confirmed}",
            )
        return True, f"entry_confirmed;{detail}"

    def _queue_post_buy_reviews(
        self, *, trade_id: str, product_id: str, entry_ts: float, entry_price: float,
        candidate: Dict[str, Any],
    ) -> None:
        for minutes in POST_BUY_REVIEW_WINDOWS_MINUTES:
            self.post_buy_review_queue.append({
                "review_ts": float(entry_ts) + float(minutes) * 60.0,
                "review_minutes": int(minutes),
                "trade_id": trade_id,
                "product_id": product_id,
                "entry_ts": float(entry_ts),
                "entry_price": float(entry_price),
                "score_at_entry": float(candidate.get("score", 0.0)),
                "prob_at_entry": float(candidate.get("estimated_prob_up", 0.0)),
                "ev_at_entry": float(candidate.get("expected_net_edge_bps", 0.0)),
                "spread_at_entry": float(candidate.get("spread_bps", 0.0)),
                "timing_reason_at_entry": str(candidate.get("entry_timing_reason", "")),
            })

    def _adopt_live_position_after_uncertain_buy(
        self, *, product_id: str, qty: float, entry_price: float, pending: Dict[str, Any],
    ) -> None:
        """Adopt only the reconciled balance delta as a managed position lot."""
        qty = float(qty)
        entry_price = float(entry_price)
        if qty <= 0 or entry_price <= 0:
            return
        adopted_ts = now_ts()
        trade_id = f"adopted-{product_id}-{int(adopted_ts)}-{uuid.uuid4().hex[:8]}"
        candidate = dict(pending.get("candidate") or {})
        lot_meta = {
            "trade_id": trade_id,
            "entry_ts": adopted_ts,
            "entry_reason": "adopted_after_uncertain_buy",
            "source": "coinbase_balance_reconciliation",
            "pending_reason": pending.get("reason", ""),
            "estimated_prob_up": float(candidate.get("estimated_prob_up", 0.0)),
            "position_pct": float(candidate.get("position_pct", 0.0)),
            "target_bps": float(candidate.get("target_bps", 0.0)),
            "cost_bps": float(candidate.get("cost_bps", 0.0)),
            "scalp_done": False, "core_done": False,
            "scalp_armed": False, "core_armed": False,
            "profit_lock_armed": False,
        }
        self.positions.setdefault(product_id, []).append(PositionLot(
            qty=qty, price=entry_price, tier=int(candidate.get("tier", TIER_LOW)),
            score=float(candidate.get("score", 0.0)), meta=lot_meta,
        ))
        self.lot_tags.setdefault(product_id, []).append("RECONCILED")
        self.position_start_ts[product_id] = self.position_start_ts.get(product_id) or adopted_ts
        self.position_entry_price[product_id] = entry_price
        self.last_buy_ts[product_id] = adopted_ts
        self.last_buy_price[product_id] = entry_price
        self.anchor_ts[product_id] = adopted_ts
        self._queue_post_buy_reviews(
            trade_id=trade_id, product_id=product_id, entry_ts=adopted_ts,
            entry_price=entry_price, candidate=candidate,
        )
        log(
            f"[reconcile] adopted uncertain buy {product_id} qty={qty:.12f} "
            f"entry={entry_price:.8f} trade_id={trade_id}"
        )

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
            max_wait_sec=MAKER_ENTRY_TIMEOUT_SEC,
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
        execution_mode: Optional[str] = None,
    ) -> Optional[Tuple[float, float, float, Optional[float], Optional[str]]]:
        """Execute a live buy and return only a Coinbase-confirmed fill."""
        mode = str(execution_mode or ENTRY_EXECUTION_MODE).upper().strip()
        result = None

        try:
            if mode == "MARKET":
                result = await self._live_buy_market(product_id=product_id, quote_usd=quote_usd)
            elif mode == "MAKER":
                result = await self._live_buy_maker(product_id=product_id, quote_usd=quote_usd, bid=bid)
            elif mode == "LIMIT_THEN_MARKET":
                result = await self._live_buy_maker(
                    product_id=product_id,
                    quote_usd=quote_usd,
                    bid=bid,
                )

                fill = self._require_live_fill(
                    result, product_id=product_id, side="BUY"
                )

                if LOG_ORDER_ATTEMPTS:
                    self.olog.log_order(
                        event="BUY_ATTEMPT",
                        product_id=product_id,
                        side="BUY",
                        mode="MAKER_FIRST",
                        requested_quote_usd=quote_usd,
                        result=result,
                        reason=reason,
                    )

                if fill is not None:
                    self.last_buy_execution_result[product_id] = (
                        dict(result) if isinstance(result, dict) else {}
                    )
                    return fill

                maker_error = ""
                try:
                    if isinstance(result, dict):
                        maker_error = str(
                            result.get("error") or result.get("status") or ""
                        )
                except Exception:
                    maker_error = ""

                log(
                    f"[buy-fallback] {product_id} maker did not fill; "
                    f"maker_error={maker_error}; falling back to market"
                )

                result = await self._live_buy_market(
                    product_id=product_id, quote_usd=quote_usd
                )
            else:
                raise RuntimeError(f"Invalid live buy execution mode={mode}")

            self.last_buy_execution_result[product_id] = dict(result) if isinstance(result, dict) else {}
            fill = self._require_live_fill(result, product_id=product_id, side="BUY")
            if LOG_ORDER_ATTEMPTS:
                self.olog.log_order(
                    event="BUY_ATTEMPT", product_id=product_id, side="BUY", mode=mode,
                    requested_quote_usd=quote_usd, result=result, reason=reason,
                )
            return fill

        except Exception as e:
            self.last_buy_execution_result[product_id] = {"error": str(e), "status": "EXCEPTION"}
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
        log(
            f"[sell-attempt] {product_id} mode={mode} "
            f"qty={base_qty:.12f} reason={reason}"
        )

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
                        event="SELL_ATTEMPT",
                        product_id=product_id,
                        side="SELL",
                        mode="MAKER_FIRST",
                        requested_base_qty=base_qty,
                        result=result,
                        reason=reason,
                    )

                if fill is not None:
                    return fill

                maker_error = ""
                try:
                    if isinstance(result, dict):
                        maker_error = str(result.get("error") or result.get("status") or "")
                except Exception:
                    maker_error = ""

                log(
                    f"[sell-fallback] {product_id} maker did not fill; "
                    f"maker_error={maker_error}; falling back to market"
                )

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

    def _rest_backfill_top_of_book(self, product_ids: Optional[List[str]] = None) -> None:
        """Fill missing or stale top-of-book data using Coinbase REST quotes."""
        if not ENABLE_REST_TOP_OF_BOOK_FALLBACK:
            return
        if not isinstance(self.portfolio, LivePortfolio):
            return

        requested_products = product_ids or list(PRODUCTS)
        now_value = now_ts()
        for product_id in requested_products:
            tob = self.tob.get(product_id)
            is_missing = tob is None or tob.bid <= 0 or tob.ask <= 0
            try:
                is_stale = tob is not None and (now_value - float(tob.ts)) > float(TOP_OF_BOOK_MAX_STALE_SEC)
            except Exception:
                is_stale = True

            if not is_missing and not is_stale:
                continue

            bid, ask = self.portfolio.get_best_bid_ask(product_id)
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                continue

            quote = TopOfBook(bid=float(bid), ask=float(ask), ts=now_value)
            self.tob[product_id] = quote
            log(
                f"[tob-rest] {product_id} bid={bid:.8f} ask={ask:.8f} "
                f"spread_bps={quote.spread_bps:.3f}"
            )

    async def _wait_for_tob_ready(self, timeout_sec: float = TOP_OF_BOOK_WAIT_SEC) -> None:
        """Wait until the configured percentage of products have valid quotes."""
        started_at = now_ts()
        last_log = 0.0
        last_rest_tob_backfill = 0.0
        required_ready = max(
            1,
            int(math.ceil(len(PRODUCTS) * float(TOP_OF_BOOK_READY_MIN_PRODUCTS_PCT))),
        )

        while now_ts() - started_at < float(timeout_sec):
            if (
                ENABLE_REST_TOP_OF_BOOK_FALLBACK
                and now_ts() - last_rest_tob_backfill >= TOP_OF_BOOK_REST_FALLBACK_EVERY_SEC
            ):
                missing_or_stale = []
                current_time = now_ts()
                for product_id in PRODUCTS:
                    quote = self.tob.get(product_id)
                    if (
                        quote is None
                        or quote.bid <= 0
                        or quote.ask <= 0
                        or current_time - float(quote.ts) > float(TOP_OF_BOOK_MAX_STALE_SEC)
                    ):
                        missing_or_stale.append(product_id)
                self._rest_backfill_top_of_book(missing_or_stale)
                last_rest_tob_backfill = now_ts()

            ready = [
                product_id for product_id in PRODUCTS
                if self.tob.get(product_id) is not None
                and self.tob[product_id].bid > 0
                and self.tob[product_id].ask > 0
            ]
            if len(ready) >= required_ready:
                log(f"[startup] top-of-book ready for {len(ready)}/{len(PRODUCTS)} products")
                return

            if now_ts() - last_log >= 5.0:
                missing = [product_id for product_id in PRODUCTS if product_id not in ready]
                log(
                    f"[startup] waiting for top-of-book | "
                    f"ready={len(ready)}/{len(PRODUCTS)} required={required_ready} missing={missing}"
                )
                last_log = now_ts()

            await asyncio.sleep(0.25)

        missing = [
            product_id for product_id in PRODUCTS
            if self.tob.get(product_id) is None
            or self.tob[product_id].bid <= 0
            or self.tob[product_id].ask <= 0
        ]
        log(
            f"[startup] top-of-book wait timed out; "
            f"ready={len(PRODUCTS) - len(missing)}/{len(PRODUCTS)} "
            f"required={required_ready} missing={missing}; "
            f"trading will skip products without fresh bid/ask"
        )

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

        await self._wait_for_tob_ready(timeout_sec=TOP_OF_BOOK_WAIT_SEC)

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
        self.active_products_log.write_products(PRODUCTS)
        log(f"[startup] active products written: {PRODUCTS}")

        log("[run] preloading micro history")
        await self.preload_micro_history()

        log("[run] starting websocket task first")
        ws_task = asyncio.create_task(self.ws_loop())

        log("[run] waiting for initial top-of-book data")
        await self._wait_for_tob_ready(timeout_sec=TOP_OF_BOOK_WAIT_SEC)

        await self._refresh_coinbase_fee_tier_if_needed(force=True)

        self._rest_backfill_top_of_book(PRODUCTS)
        log("[run] calibrating products before live trading")
        await self.calibrate_products_on_startup()
        self.last_hourly_calibration_update_ts = now_ts()
        log(
            f"[calibration-hourly] next hourly update scheduled in "
            f"{CALIBRATION_UPDATE_EVERY_SEC:.0f}s"
        )

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

    def _inverted_marker_is_active(self, product_id: str) -> bool:
        marker = self.inverted_markers.get(product_id)
        if not marker:
            return False

        ts = float(marker.get("ts", 0.0))
        if ts <= 0:
            return False

        age_sec = now_ts() - ts
        if age_sec > float(INVERTED_MARKER_TTL_SEC):
            log(
                f"{INVERTED_LOG_PREFIX} {product_id} marker expired "
                f"age_sec={age_sec:.1f}"
            )
            self.inverted_markers.pop(product_id, None)
            return False

        return True

    def _set_inverted_marker_from_candidate(
        self,
        *,
        candidate: Dict[str, Any],
    ) -> None:
        """Turn a qualified normal buy signal into a future sell marker."""
        product_id = str(candidate.get("product_id", ""))
        if not product_id:
            return

        bid = float(candidate.get("bid", 0.0) or 0.0)
        ask = float(candidate.get("ask", 0.0) or 0.0)
        marker_price = ask if ask > 0 else bid
        if marker_price <= 0:
            return

        cycle_index = int(self.inverted_cycle_index_by_product.get(product_id, 0)) + 1
        self.inverted_cycle_index_by_product[product_id] = cycle_index
        buy_trigger_price = marker_price * (1.0 - float(INVERTED_BUY_DROP_PCT))

        self.inverted_markers[product_id] = {
            "ts": now_ts(),
            "cycle_index": cycle_index,
            "product_id": product_id,
            "marker_price": float(marker_price),
            "sell_marker_price": float(marker_price),
            "buy_trigger_price": float(buy_trigger_price),
            "source": "old_buy_signal",
            "candidate": dict(candidate),
            "score": float(candidate.get("score", 0.0)),
            "probability": float(candidate.get("estimated_prob_up", 0.0)),
            "ev_bps": float(candidate.get("expected_net_edge_bps", 0.0)),
            "rank_score": float(candidate.get("rank_score", 0.0)),
            "armed_buy": False,
            "bought": False,
            "rebuy_count": 0,
            "last_buy_notional": 0.0,
        }

        log(
            f"{INVERTED_LOG_PREFIX} {product_id} marker_set "
            f"cycle={cycle_index} marker={marker_price:.8f} "
            f"buy_trigger={buy_trigger_price:.8f} "
            f"score={float(candidate.get('score', 0.0)):.3f} "
            f"prob={float(candidate.get('estimated_prob_up', 0.0)):.6f} "
            f"ev={float(candidate.get('expected_net_edge_bps', 0.0)):.3f}"
        )
        self.signal_events_log.log_event(
            event_type="inverted_marker_set",
            product_id=product_id,
            rank_score=f"{float(candidate.get('rank_score', 0.0)):.6f}",
            score=f"{float(candidate.get('score', 0.0)):.6f}",
            probability=f"{float(candidate.get('estimated_prob_up', 0.0)):.6f}",
            ev_bps=f"{float(candidate.get('expected_net_edge_bps', 0.0)):.6f}",
            projected_forward_bps=f"{float(candidate.get('projected_forward_gain_bps', 0.0)):.6f}",
            cost_bps=f"{float(candidate.get('cost_bps', 0.0)):.6f}",
            spread_bps=f"{float(candidate.get('spread_bps', 0.0)):.6f}",
            action="set_marker",
            reason=f"sell_marker={marker_price:.8f};buy_trigger={buy_trigger_price:.8f}",
        )

    def _inverted_has_open_position(self, product_id: str) -> bool:
        return self._inverted_current_position_qty(product_id) > 1e-12

    def _inverted_avg_entry(self, product_id: str) -> Optional[float]:
        lots = self.positions.get(product_id, [])
        qty = sum(float(lot.qty) for lot in lots)
        if qty <= 1e-12:
            return None
        return sum(float(lot.qty) * float(lot.price) for lot in lots) / qty

    def _inverted_next_loss_trigger_price(self, product_id: str) -> Optional[float]:
        avg_entry = self._inverted_avg_entry(product_id)
        if avg_entry is None or avg_entry <= 0:
            return None
        return float(avg_entry) * (1.0 - float(INVERTED_NEXT_STOP_FROM_ENTRY_PCT))

    def _inverted_current_position_qty(self, product_id: str) -> float:
        return sum(float(lot.qty) for lot in self.positions.get(product_id, []))

    def _position_estimated_usd_value(self, product_id: str, qty: float) -> float:
        tob = self.tob.get(product_id)
        if not tob or tob.mid <= 0:
            return 0.0
        return max(0.0, float(qty) * float(tob.mid))

    def _is_local_dust_position(self, product_id: str, qty: float) -> bool:
        if not ENABLE_LOCAL_DUST_POSITION_CLEANUP:
            return False

        qty = float(qty)
        if qty <= float(LOCAL_DUST_QTY_EPSILON):
            return True

        usd_value = self._position_estimated_usd_value(product_id, qty)
        return bool(usd_value > 0 and usd_value < float(LOCAL_DUST_USD_THRESHOLD))

    def _clear_local_dust_position(
        self,
        *,
        product_id: str,
        reason: str,
    ) -> None:
        qty = self._inverted_current_position_qty(product_id)
        usd_value = self._position_estimated_usd_value(product_id, qty)
        log(
            f"[dust-cleanup] {product_id} clearing local dust "
            f"qty={qty:.12f} usd_value={usd_value:.6f} reason={reason}"
        )

        try:
            self.signal_events_log.log_event(
                event_type="dust_position_cleared",
                product_id=product_id,
                action="clear_local_dust",
                reason=f"qty={qty:.12f};usd_value={usd_value:.6f};{reason}",
            )
        except Exception:
            pass

        self.positions[product_id] = []
        self.lot_tags[product_id] = []
        self.position_start_ts[product_id] = None
        self.position_entry_price[product_id] = None
        self.peak_bid[product_id] = None
        self.scale_add_count[product_id] = 0
        self.trailing_active[product_id] = False

    def _inverted_marker_trigger_key(self, product_id: str) -> str:
        return f"{product_id}:trigger"

    def _inverted_get_marker_state(self, marker: Dict[str, Any]) -> Dict[str, Any]:
        state = marker.get("state")
        if not isinstance(state, dict):
            state = {}
            marker["state"] = state
        return state

    def _inverted_buy_trigger_stabilized(
        self,
        *,
        product_id: str,
        marker: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Arm at the old stop-loss and buy only after price stabilizes."""
        if not INVERTED_REQUIRE_TRIGGER_STABILIZATION:
            return True, "stabilization_disabled"

        tob = self.tob.get(product_id)
        if not tob or tob.ask <= 0:
            return False, "no_tob"

        state = self._inverted_get_marker_state(marker)
        nowv = now_ts()
        buy_trigger = float(marker.get("buy_trigger_price") or 0.0)
        if buy_trigger <= 0:
            return False, "invalid_buy_trigger"

        if float(tob.ask) > buy_trigger:
            state.pop("trigger_first_seen_ts", None)
            state.pop("trigger_low_ask", None)
            return False, "not_in_trigger_zone"

        first_seen = float(state.get("trigger_first_seen_ts") or 0.0)
        if first_seen <= 0:
            state["trigger_first_seen_ts"] = nowv
            state["trigger_low_ask"] = float(tob.ask)
            return (
                False,
                f"trigger_armed_waiting_age ask={float(tob.ask):.8f} "
                f"trigger={buy_trigger:.8f}",
            )

        trigger_age = nowv - first_seen
        state["trigger_low_ask"] = min(
            float(state.get("trigger_low_ask") or float(tob.ask)),
            float(tob.ask),
        )
        if trigger_age < float(INVERTED_BUY_TRIGGER_MIN_AGE_SEC):
            return (
                False,
                f"trigger_too_new age={trigger_age:.1f}s "
                f"required={INVERTED_BUY_TRIGGER_MIN_AGE_SEC:.1f}s",
            )

        try:
            candles = list(self.live_1m[product_id].candles)
        except Exception:
            candles = []

        lookback = int(INVERTED_BUY_STABILIZATION_LOOKBACK_CANDLES)
        if lookback > 0 and len(candles) >= lookback:
            recent = candles[-lookback:]
            first_close = float(recent[0].close)
            last_close = float(recent[-1].close)
            low = min(float(c.low) for c in recent)
            adverse_bps = (
                ((first_close / low) - 1.0) * 10000.0
                if first_close > 0 and low > 0
                else 0.0
            )
            still_sliding = bool(
                last_close < first_close
                and adverse_bps > float(INVERTED_BUY_STABILIZATION_MAX_ADVERSE_BPS)
            )
            if still_sliding:
                return (
                    False,
                    f"still_sliding adverse_bps={adverse_bps:.2f} "
                    f"first_close={first_close:.8f} last_close={last_close:.8f}",
                )

        if INVERTED_BUY_STABILIZATION_REQUIRE_NON_NEGATIVE_1M:
            mom1 = self._recent_momentum_bps_for_product(product_id, 1)
            if mom1 < 0:
                return False, f"mom1_negative {mom1:.2f}bps"

        return (
            True,
            f"stabilized age={trigger_age:.1f}s ask={float(tob.ask):.8f} "
            f"trigger={buy_trigger:.8f}",
        )

    def _inverted_loss_rotation_confirmed(
        self,
        *,
        product_id: str,
        marker: Dict[str, Any],
        trigger_price: float,
    ) -> Tuple[bool, str]:
        """Require position age and persistent loss before a larger rebuy."""
        tob = self.tob.get(product_id)
        if not tob or tob.bid <= 0:
            return False, "no_tob"

        avg_entry = self._inverted_avg_entry(product_id)
        if avg_entry is None:
            return False, "no_avg_entry"

        entry_ts = float(self.position_start_ts.get(product_id) or 0.0)
        age = now_ts() - entry_ts if entry_ts > 0 else 0.0
        if age < float(INVERTED_REBUY_ROTATION_MIN_HOLD_SEC):
            return (
                False,
                f"position_too_new_for_rotation age={age:.1f}s "
                f"required={INVERTED_REBUY_ROTATION_MIN_HOLD_SEC:.1f}s",
            )

        state = self._inverted_get_marker_state(marker)
        nowv = now_ts()
        if float(tob.bid) > float(trigger_price):
            state.pop("loss_rotation_first_seen_ts", None)
            return False, "not_below_loss_rotation_trigger"

        first_seen = float(state.get("loss_rotation_first_seen_ts") or 0.0)
        if first_seen <= 0:
            state["loss_rotation_first_seen_ts"] = nowv
            return (
                False,
                f"loss_rotation_armed_waiting_confirm bid={float(tob.bid):.8f} "
                f"trigger={float(trigger_price):.8f}",
            )

        confirm_age = nowv - first_seen
        if confirm_age < float(INVERTED_REBUY_TRIGGER_CONFIRM_SEC):
            return (
                False,
                f"loss_rotation_confirming age={confirm_age:.1f}s "
                f"required={INVERTED_REBUY_TRIGGER_CONFIRM_SEC:.1f}s",
            )

        return (
            True,
            f"loss_rotation_confirmed age={age:.1f}s "
            f"confirm_age={confirm_age:.1f}s bid={float(tob.bid):.8f} "
            f"trigger={float(trigger_price):.8f}",
        )

    def _inverted_target_sell_price(self, product_id: str) -> Optional[float]:
        marker = self.inverted_markers.get(product_id)
        if not marker:
            return None

        marker_sell = float(marker.get("sell_marker_price") or 0.0)
        if marker_sell <= 0:
            return None
        if not INVERTED_REQUIRE_FEE_POSITIVE_SELL:
            return marker_sell

        avg_entry = self._inverted_avg_entry(product_id)
        if avg_entry is None or avg_entry <= 0:
            return marker_sell

        try:
            min_exit = required_exit_price_for_net_gain(
                effective_entry_price=float(avg_entry),
                exit_fee_bps=self._exit_fee_bps_for_mode(),
                est_slippage_bps=EST_SLIPPAGE_BPS,
                est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                min_net_gain_bps=max(
                    MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                    MIN_NET_GAIN_AFTER_FEES_BPS,
                ),
            )
            return max(float(marker_sell), float(min_exit))
        except Exception:
            return marker_sell

    def _inverted_compute_buy_notional(
        self,
        *,
        product_id: str,
        equity_usd: float,
        marker: Dict[str, Any],
        is_rebuy: bool,
    ) -> float:
        """Size an inverted entry, increasing but capping loss-rotation rebuys."""
        candidate = dict(marker.get("candidate") or {})
        base_pct = float(
            candidate.get("position_pct", MIN_POSITION_PCT_OF_EQUITY)
            or MIN_POSITION_PCT_OF_EQUITY
        )
        base_pct = max(float(MIN_POSITION_PCT_OF_EQUITY), base_pct)

        if is_rebuy:
            previous_notional = float(marker.get("last_buy_notional") or 0.0)
            if previous_notional > 0:
                notional = previous_notional * float(INVERTED_REBUY_SIZE_MULTIPLIER)
            else:
                notional = (
                    float(equity_usd)
                    * base_pct
                    * float(INVERTED_REBUY_SIZE_MULTIPLIER)
                )
        else:
            notional = float(equity_usd) * base_pct

        max_single = float(equity_usd) * float(INVERTED_MAX_SINGLE_BUY_PCT_OF_EQUITY)
        max_product = float(equity_usd) * float(
            INVERTED_MAX_PRODUCT_EXPOSURE_PCT_OF_EQUITY
        )
        product_room = max(
            0.0,
            max_product - float(self._current_product_exposure_usd(product_id)),
        )
        notional = min(float(notional), float(max_single), float(product_room))

        min_viable = max(float(MIN_ENTRY_USD), float(MIN_LIVE_ORDER_USD))
        if 0 < notional < min_viable:
            log(
                f"{INVERTED_LOG_PREFIX} {product_id} notional_below_min "
                f"notional={notional:.6f} min_viable={min_viable:.6f} "
                f"equity={float(equity_usd):.6f} product_room={product_room:.6f}"
            )
            return 0.0

        if notional <= 0:
            log(
                f"{INVERTED_LOG_PREFIX} {product_id} notional_zero "
                f"equity={float(equity_usd):.6f} product_room={product_room:.6f} "
                f"current_exposure={float(self._current_product_exposure_usd(product_id)):.6f} "
                f"max_product={max_product:.6f}"
            )
            return 0.0

        return float(notional)

    async def _execute_inverted_buy(
        self,
        *,
        product_id: str,
        marker: Dict[str, Any],
        equity_usd: float,
        is_rebuy: bool,
        reason: str,
    ) -> bool:
        tob = self.tob.get(product_id)
        if not tob or tob.ask <= 0 or tob.bid <= 0:
            log(f"{INVERTED_LOG_PREFIX} {product_id} buy_skip no_tob")
            return False

        quote_usd = self._inverted_compute_buy_notional(
            product_id=product_id,
            equity_usd=equity_usd,
            marker=marker,
            is_rebuy=is_rebuy,
        )
        if quote_usd <= 0:
            state = self._inverted_get_marker_state(marker)
            state["zero_notional_count"] = int(state.get("zero_notional_count", 0)) + 1
            log(
                f"{INVERTED_LOG_PREFIX} {product_id} buy_skip zero_notional "
                f"count={state['zero_notional_count']}"
            )
            if state["zero_notional_count"] >= int(INVERTED_MAX_ZERO_NOTIONAL_TRIGGER_HITS):
                marker["cooldown_until"] = (
                    now_ts() + float(INVERTED_UNBUYABLE_MARKER_COOLDOWN_SEC)
                )
                log(
                    f"{INVERTED_LOG_PREFIX} {product_id} marker_unbuyable_cooldown "
                    f"cooldown_sec={INVERTED_UNBUYABLE_MARKER_COOLDOWN_SEC:.1f}"
                )
            return False

        entry_fee_bps = self._entry_fee_bps_for_mode(
            execution_mode=ENTRY_EXECUTION_MODE
        )
        if not await self._live_can_afford(quote_usd, entry_fee_bps):
            log(
                f"{INVERTED_LOG_PREFIX} {product_id} buy_skip cannot_afford "
                f"quote_usd={quote_usd:.2f}"
            )
            return False

        trade_id = f"inverted-{product_id}-{int(now_ts())}-{uuid.uuid4().hex[:8]}"
        log(
            f"{INVERTED_LOG_PREFIX} {product_id} buy_attempt "
            f"trade_id={trade_id} quote_usd={quote_usd:.2f} "
            f"is_rebuy={is_rebuy} reason={reason} "
            f"bid={float(tob.bid):.8f} ask={float(tob.ask):.8f}"
        )
        self.signal_events_log.log_event(
            event_type="inverted_buy_attempt",
            trade_id=trade_id,
            product_id=product_id,
            action="attempt_buy",
            reason=reason,
        )

        fill = await self._execute_live_buy(
            product_id=product_id,
            quote_usd=float(quote_usd),
            bid=float(tob.bid),
            ask=float(tob.ask),
            reason=reason,
            execution_mode=ENTRY_EXECUTION_MODE,
        )
        if fill is None:
            log(f"{INVERTED_LOG_PREFIX} {product_id} buy_failed reason={reason}")
            return False

        qty, avg_px, fee, filled_notional, order_id = fill
        qty = float(qty)
        avg_px = float(avg_px)
        fee = float(fee)
        filled_notional_f = float(filled_notional or quote_usd)
        candidate = dict(marker.get("candidate") or {})
        lot_meta = {
            "trade_id": trade_id,
            "order_id": order_id,
            "strategy_mode": "inverted_stoploss_cycle",
            "cycle_index": int(marker.get("cycle_index", 0)),
            "entry_reason": reason,
            "old_buy_marker_price": float(marker.get("sell_marker_price", 0.0)),
            "inverted_buy_trigger_price": float(marker.get("buy_trigger_price", 0.0)),
            "is_rebuy": bool(is_rebuy),
            "rebuy_count": int(marker.get("rebuy_count", 0)),
            "estimated_prob_up": float(candidate.get("estimated_prob_up", 0.0)),
            "position_pct": float(candidate.get("position_pct", 0.0)),
            "target_bps": float(candidate.get("target_bps", 0.0)),
            "cost_bps": float(candidate.get("cost_bps", 0.0)),
            "min_profitable_exit_price": None,
        }
        try:
            lot_meta["min_profitable_exit_price"] = float(
                required_exit_price_for_net_gain(
                    effective_entry_price=avg_px,
                    exit_fee_bps=self._exit_fee_bps_for_mode(),
                    est_slippage_bps=EST_SLIPPAGE_BPS,
                    est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                    min_net_gain_bps=max(
                        MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                        MIN_NET_GAIN_AFTER_FEES_BPS,
                    ),
                )
            )
        except Exception:
            pass

        self.positions.setdefault(product_id, []).append(
            PositionLot(
                qty=qty,
                price=avg_px,
                tier=int(candidate.get("tier", TIER_LOW)),
                score=float(candidate.get("score", 0.0)),
                meta=lot_meta,
            )
        )
        self.lot_tags.setdefault(product_id, []).append("INVERTED")
        entry_ts = now_ts()
        self.position_start_ts[product_id] = entry_ts
        self.position_entry_price[product_id] = avg_px
        self.last_buy_ts[product_id] = entry_ts
        self.last_buy_price[product_id] = avg_px
        self.peak_bid[product_id] = float(tob.bid)
        self.anchor_ts[product_id] = entry_ts

        marker["bought"] = True
        marker["last_buy_notional"] = filled_notional_f
        marker["last_entry_price"] = avg_px
        marker["last_trade_id"] = trade_id
        if is_rebuy:
            marker["rebuy_count"] = int(marker.get("rebuy_count", 0)) + 1

        self.tlog.log_trade(
            event="BUY",
            product_id=product_id,
            side="BUY",
            qty=qty,
            price=avg_px,
            fee_usd_val=fee,
            gross_pnl_usd=0.0,
            net_pnl_usd=-fee,
            entry_price=avg_px,
            exit_price=None,
            weekly_bias=None,
            note=(
                f"inverted_stoploss_buy reason={reason} "
                f"marker={float(marker.get('sell_marker_price', 0.0)):.8f} "
                f"trigger={float(marker.get('buy_trigger_price', 0.0)):.8f} "
                f"is_rebuy={is_rebuy}"
            ),
            filled_notional_usd=filled_notional_f,
            entry_score=float(candidate.get("score", 0.0)),
            entry_tier=int(candidate.get("tier", TIER_LOW)),
            expected_net_edge_bps=float(candidate.get("expected_net_edge_bps", 0.0)),
        )
        self._record_trade_timestamp(product_id)
        self._queue_post_buy_reviews(
            trade_id=trade_id,
            product_id=product_id,
            entry_ts=entry_ts,
            entry_price=avg_px,
            candidate=candidate,
        )
        self.signal_events_log.log_event(
            event_type="inverted_buy_fill",
            trade_id=trade_id,
            product_id=product_id,
            action="buy_filled",
            reason=(
                f"qty={qty:.12f};avg_px={avg_px:.8f};"
                f"notional={filled_notional_f:.6f}"
            ),
        )
        log(
            f"{INVERTED_LOG_PREFIX} {product_id} buy_success "
            f"trade_id={trade_id} qty={qty:.12f} avg_px={avg_px:.8f} "
            f"notional={filled_notional_f:.6f} fee={fee:.6f}"
        )
        return True

    async def _execute_inverted_full_sell(
        self,
        *,
        product_id: str,
        reason: str,
    ) -> bool:
        tob = self.tob.get(product_id)
        if not tob or tob.bid <= 0:
            return False

        qty = self._inverted_current_position_qty(product_id)
        if qty <= 1e-12:
            return False

        if self._is_local_dust_position(product_id, qty):
            self._clear_local_dust_position(
                product_id=product_id,
                reason=f"inverted_sell_requested_but_position_is_dust;{reason}",
            )
            return True

        # Coinbase available balance is the source of truth for what can actually be sold.
        try:
            if isinstance(self.portfolio, LivePortfolio):
                snap = self.portfolio.refresh_snapshot(force=True, ttl_sec=0.0)
                base_asset = product_base_asset(product_id)
                available_qty = self.portfolio.get_available_asset(
                    base_asset,
                    snapshot=snap,
                )
                if available_qty <= 0:
                    self._clear_local_dust_position(
                        product_id=product_id,
                        reason=(
                            "coinbase_available_base_is_zero_before_sell;"
                            f"local_qty={qty:.12f}"
                        ),
                    )
                    return True

                if available_qty < qty:
                    log(
                        f"{INVERTED_LOG_PREFIX} {product_id} sell_qty_clamped_to_coinbase_available "
                        f"local_qty={qty:.12f} available={available_qty:.12f}"
                    )
                    qty = float(available_qty)

                if self._is_local_dust_position(product_id, qty):
                    self._clear_local_dust_position(
                        product_id=product_id,
                        reason=(
                            "available_qty_is_dust_before_sell;"
                            f"available={available_qty:.12f}"
                        ),
                    )
                    return True
        except Exception as e:
            log(f"{INVERTED_LOG_PREFIX} {product_id} sell preflight failed: {e}")

        log(
            f"{INVERTED_LOG_PREFIX} {product_id} sell_attempt "
            f"mode={EXIT_EXECUTION_MODE} qty={qty:.12f} "
            f"reason={reason} bid={float(tob.bid):.8f}"
        )
        fill = await self._execute_live_sell(
            product_id=product_id,
            base_qty=float(qty),
            bid=float(tob.bid),
            ask=float(tob.ask),
            reason=reason,
        )
        if fill is None:
            self.last_sell_failure_ts_by_product[product_id] = now_ts()
            log(f"{INVERTED_LOG_PREFIX} {product_id} sell_failed reason={reason}")
            return False

        filled_qty, avg_px, fee, filled_notional, order_id = fill
        filled_qty = min(float(qty), float(filled_qty))
        avg_px = float(avg_px)
        fee = float(fee)
        filled_notional_f = float(filled_notional or filled_qty * avg_px)
        fifo_cost, fifo_avg_entry = self._fifo_cost_basis(
            list(self.positions.get(product_id, [])),
            filled_qty,
        )
        gross_pnl = filled_notional_f - float(fifo_cost)
        net_pnl = gross_pnl - fee

        self.tlog.log_trade(
            event="SELL",
            product_id=product_id,
            side="SELL",
            qty=filled_qty,
            price=avg_px,
            fee_usd_val=fee,
            gross_pnl_usd=gross_pnl,
            net_pnl_usd=net_pnl,
            entry_price=fifo_avg_entry,
            exit_price=avg_px,
            weekly_bias=None,
            note=f"inverted_cycle_sell reason={reason} order_id={order_id}",
            filled_notional_usd=filled_notional_f,
        )
        self._record_trade_timestamp(product_id)
        self._record_realized_trade_result(net_pnl)
        self._fifo_reduce_lots(product_id, filled_qty)

        remaining_qty_after_fifo = self._inverted_current_position_qty(product_id)
        if remaining_qty_after_fifo > 0 and self._is_local_dust_position(
            product_id,
            remaining_qty_after_fifo,
        ):
            self._clear_local_dust_position(
                product_id=product_id,
                reason=(
                    "remaining_after_fifo_sell_is_dust;"
                    f"remaining={remaining_qty_after_fifo:.12f}"
                ),
            )

        fully_closed = self._inverted_current_position_qty(product_id) <= 1e-12
        if fully_closed:
            self.positions[product_id] = []
            self.lot_tags[product_id] = []
            self.position_start_ts[product_id] = None
            self.position_entry_price[product_id] = None
            self.peak_bid[product_id] = None
            self.scale_add_count[product_id] = 0
            self.trailing_active[product_id] = False
            self.last_exit_ts = now_ts()
            self.inverted_cycle_cooldown_until[product_id] = (
                now_ts() + float(INVERTED_POST_CYCLE_COOLDOWN_SEC)
            )
        else:
            log(
                f"{INVERTED_LOG_PREFIX} {product_id} sell_partial "
                f"remaining_qty={self._inverted_current_position_qty(product_id):.12f}"
            )

        log(
            f"{INVERTED_LOG_PREFIX} {product_id} sell_success "
            f"qty={filled_qty:.12f} avg_px={avg_px:.8f} "
            f"net_pnl={net_pnl:.6f} reason={reason} fully_closed={fully_closed}"
        )
        return fully_closed

    async def _process_inverted_stoploss_cycle(
        self,
        *,
        equity_usd: float,
    ) -> None:
        """Process marker buys, marker exits, and larger loss-rotation rebuys."""
        for product_id in PRODUCTS:
            tob = self.tob.get(product_id)
            if not tob or tob.bid <= 0 or tob.ask <= 0:
                continue

            cooldown_until = float(
                self.inverted_cycle_cooldown_until.get(product_id, 0.0)
            )
            if cooldown_until > now_ts():
                continue

            marker = self.inverted_markers.get(product_id)
            has_position = self._inverted_has_open_position(product_id)
            if not marker and not has_position:
                continue
            if marker and not has_position and not self._inverted_marker_is_active(product_id):
                continue

            marker = self.inverted_markers.get(product_id)
            if marker:
                marker_cooldown_until = float(marker.get("cooldown_until") or 0.0)
                if marker_cooldown_until > now_ts():
                    continue

            if marker and not has_position:
                buy_trigger = float(marker.get("buy_trigger_price") or 0.0)
                if buy_trigger > 0 and float(tob.ask) <= buy_trigger:
                    stable_ok, stable_reason = self._inverted_buy_trigger_stabilized(
                        product_id=product_id,
                        marker=marker,
                    )
                    self.signal_events_log.log_event(
                        event_type=(
                            "inverted_buy_trigger_confirmed"
                            if stable_ok
                            else "inverted_buy_trigger_waiting"
                        ),
                        product_id=product_id,
                        action="buy_trigger",
                        reason=(
                            f"ask={float(tob.ask):.8f};"
                            f"trigger={buy_trigger:.8f};"
                            f"{stable_reason}"
                        ),
                    )
                    if stable_ok:
                        ok = await self._execute_inverted_buy(
                            product_id=product_id,
                            marker=marker,
                            equity_usd=equity_usd,
                            is_rebuy=False,
                            reason=(
                                "inverted_old_stoploss_buy_stabilized "
                                f"ask<={buy_trigger:.8f};{stable_reason}"
                            ),
                        )
                        if not ok:
                            state = self._inverted_get_marker_state(marker)
                            state["failed_buy_trigger_hits"] = (
                                int(state.get("failed_buy_trigger_hits", 0)) + 1
                            )
                            if (
                                int(state.get("failed_buy_trigger_hits", 0))
                                >= int(INVERTED_MAX_ZERO_NOTIONAL_TRIGGER_HITS)
                            ):
                                marker["cooldown_until"] = (
                                    now_ts()
                                    + float(INVERTED_UNBUYABLE_MARKER_COOLDOWN_SEC)
                                )
                                log(
                                    f"{INVERTED_LOG_PREFIX} {product_id} marker_cooldown "
                                    f"failed_buy_trigger_hits={state.get('failed_buy_trigger_hits')}"
                                )
                continue

            if not has_position:
                continue

            if not marker:
                avg_entry = self._inverted_avg_entry(product_id)
                if avg_entry:
                    cycle_index = int(
                        self.inverted_cycle_index_by_product.get(product_id, 0)
                    ) + 1
                    self.inverted_cycle_index_by_product[product_id] = cycle_index
                    self.inverted_markers[product_id] = {
                        "ts": now_ts(),
                        "cycle_index": cycle_index,
                        "product_id": product_id,
                        "marker_price": float(avg_entry)
                        * (1.0 + float(INVERTED_BUY_DROP_PCT)),
                        "sell_marker_price": float(avg_entry)
                        * (1.0 + float(INVERTED_BUY_DROP_PCT)),
                        "buy_trigger_price": float(avg_entry),
                        "source": "recovered_from_existing_position",
                        "candidate": {},
                        "bought": True,
                        "rebuy_count": 0,
                        "last_buy_notional": abs(
                            float(avg_entry)
                            * self._inverted_current_position_qty(product_id)
                        ),
                    }
                    for recovered_lot in self.positions.get(product_id, []):
                        recovered_lot.meta["strategy_mode"] = (
                            "inverted_stoploss_cycle"
                        )
                        recovered_lot.meta["cycle_index"] = cycle_index
                        recovered_lot.meta["entry_reason"] = (
                            "recovered_from_existing_position"
                        )
                    marker = self.inverted_markers[product_id]
            if not marker:
                continue

            target_sell = self._inverted_target_sell_price(product_id)
            if target_sell is not None and float(tob.bid) >= float(target_sell):
                self.signal_events_log.log_event(
                    event_type="inverted_sell_marker_hit",
                    product_id=product_id,
                    action="sell_trigger",
                    reason=(
                        f"bid={float(tob.bid):.8f};"
                        f"target={float(target_sell):.8f}"
                    ),
                )
                sold = await self._execute_inverted_full_sell(
                    product_id=product_id,
                    reason=f"inverted_sell_marker_hit bid>={float(target_sell):.8f}",
                )
                if sold:
                    self.inverted_markers.pop(product_id, None)
                continue

            next_loss_trigger = self._inverted_next_loss_trigger_price(product_id)
            if (
                INVERTED_ENABLE_LOSS_ROTATION
                and next_loss_trigger is not None
                and float(tob.bid) <= float(next_loss_trigger)
            ):
                rotation_ok, rotation_reason = self._inverted_loss_rotation_confirmed(
                    product_id=product_id,
                    marker=marker,
                    trigger_price=float(next_loss_trigger),
                )
                log(
                    f"{INVERTED_LOG_PREFIX} {product_id} rebuy_rotation_check "
                    f"ok={rotation_ok} reason={rotation_reason}"
                )
                self.signal_events_log.log_event(
                    event_type=(
                        "inverted_loss_rotation_confirmed"
                        if rotation_ok
                        else "inverted_loss_rotation_waiting"
                    ),
                    product_id=product_id,
                    action="sell_old_buy_larger" if rotation_ok else "wait",
                    reason=rotation_reason,
                )
                if not rotation_ok:
                    continue

                sold = await self._execute_inverted_full_sell(
                    product_id=product_id,
                    reason=(
                        "inverted_rebuy_rotation_sell_old_confirmed "
                        f"bid<={next_loss_trigger:.8f};{rotation_reason}"
                    ),
                )
                if sold:
                    new_marker_price = float(tob.ask)
                    new_buy_trigger = new_marker_price * (
                        1.0 - float(INVERTED_BUY_DROP_PCT)
                    )
                    marker["ts"] = now_ts()
                    marker["sell_marker_price"] = float(new_marker_price)
                    marker["marker_price"] = float(new_marker_price)
                    marker["buy_trigger_price"] = float(new_buy_trigger)

                    state = self._inverted_get_marker_state(marker)
                    state.pop("trigger_first_seen_ts", None)
                    state.pop("trigger_low_ask", None)
                    state.pop("loss_rotation_first_seen_ts", None)
                    state["failed_buy_trigger_hits"] = 0

                    await self._execute_inverted_buy(
                        product_id=product_id,
                        marker=marker,
                        equity_usd=equity_usd,
                        is_rebuy=True,
                        reason=(
                            "inverted_rebuy_larger_after_confirmed_loss "
                            f"new_marker={new_marker_price:.8f} "
                            f"new_trigger={new_buy_trigger:.8f}"
                        ),
                    )

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
            min_viable = max(
                float(MIN_ENTRY_USD),
                float(MIN_LIVE_ORDER_USD),
                current_equity_usd * float(MIN_POSITION_PCT_OF_EQUITY),
            )
            if proposed > 0.0:
                proposed = max(float(proposed), float(min_viable))

            max_product_exposure = current_equity_usd * float(MAX_EXPOSURE_PER_PRODUCT_PCT_OF_EQUITY)
            remaining_product_room = max(0.0, max_product_exposure - current_product_exposure_usd)

            # Cash and exposure caps remain authoritative. If either cap makes the
            # minimum viable order impossible, skip rather than overspend.
            proposed = min(proposed, remaining_product_room, spendable_cash)

            if proposed < float(min_viable):
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

    def _maybe_train_ai_brain(self) -> None:
        if not ENABLE_LOCAL_AI_BRAIN or self.ai_brain is None:
            return
        current_ts = now_ts()
        if current_ts - self.last_ai_train_ts < AI_RETRAIN_EVERY_SEC:
            return
        self.last_ai_train_ts = current_ts
        try:
            result = self.ai_brain.train()
            log(f"[ai] train_result={result}")
        except Exception as exc:
            log(f"[ai] training failed: {exc}")

    def _ai_context_from_candidate(
        self, candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        product_id = str(candidate.get("product_id", ""))
        moms = {"mom1": 0.0, "mom3": 0.0, "mom5": 0.0, "mom15": 0.0}
        try:
            moms = self._entry_momentum_snapshot(product_id)
        except Exception:
            pass

        green = 0
        try:
            green = self._recent_green_candle_count(
                product_id, ENTRY_GREEN_CANDLE_LOOKBACK
            )
        except Exception:
            pass

        return {
            "ts": now_ts(),
            "score": float(candidate.get("score", 0.0)),
            "probability": float(candidate.get("estimated_prob_up", 0.0)),
            "ev_bps": float(candidate.get("expected_net_edge_bps", 0.0)),
            "projected_forward_bps": float(
                candidate.get("projected_forward_gain_bps", 0.0)
            ),
            "cost_bps": float(candidate.get("cost_bps", 0.0)),
            "spread_bps": float(candidate.get("spread_bps", 0.0)),
            "momentum_1_bps": float(moms.get("mom1", 0.0)),
            "momentum_3_bps": float(moms.get("mom3", 0.0)),
            "momentum_5_bps": float(moms.get("mom5", 0.0)),
            "momentum_15_bps": float(moms.get("mom15", 0.0)),
            "green_candles": float(green),
            "rank_score": float(candidate.get("rank_score", 0.0)),
            "buy_ready_count": float(candidate.get("buy_ready_count", 0.0)),
        }

    def _ai_allows_candidate(
        self, *, candidate: Dict[str, Any]
    ) -> Tuple[bool, str]:
        if not ENABLE_LOCAL_AI_BRAIN:
            return True, "ai_disabled"
        if self.ai_brain is None:
            return True, "ai_unavailable"

        mode = str(AI_MODE).upper()
        if mode == "OFF":
            return True, "ai_mode_off"

        product_id = str(candidate.get("product_id", ""))
        try:
            decision = self.ai_brain.predict(
                product_id, self._ai_context_from_candidate(candidate)
            )
        except Exception as exc:
            return True, f"ai_prediction_failed:{exc}"

        reason = (
            f"ai_action={decision.action};"
            f"confidence={decision.confidence:.3f};"
            f"prob_up_30m={decision.prob_up_30m:.3f};"
            f"expected_move={decision.expected_move_30m_bps:.2f};"
            f"expected_adverse={decision.expected_adverse_bps:.2f};"
            f"{decision.reason}"
        )
        if mode == "OBSERVE":
            return True, f"observe_only;{reason}"
        if mode == "FILTER":
            if (
                decision.action in AI_BLOCK_BUY_ACTIONS
                and decision.confidence >= AI_MIN_CONFIDENCE_TO_BLOCK
            ):
                return False, reason
            return True, reason
        return True, f"control_mode_not_enabled_for_direct_trading;{reason}"

    def _market_price_near_ts(
        self,
        *,
        product_id: str,
        target_ts: float,
        max_age_sec: float = 90.0,
    ) -> Optional[float]:
        """Return the closest recorded market mid near a timestamp."""
        try:
            if not os.path.exists(MARKET_CSV_PATH):
                return None
            frame = pd.read_csv(
                MARKET_CSV_PATH,
                usecols=lambda column: column in {"ts", "product_id", "mid"},
            )
            if frame.empty or not {"ts", "product_id", "mid"}.issubset(frame.columns):
                return None
            frame = frame[
                frame["product_id"].astype(str) == str(product_id)
            ].copy()
            frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
            frame["mid"] = pd.to_numeric(frame["mid"], errors="coerce")
            frame = frame.dropna(subset=["ts", "mid"])
            if frame.empty:
                return None
            frame["delta"] = (frame["ts"] - float(target_ts)).abs()
            row = frame.sort_values("delta").iloc[0]
            price = float(row["mid"])
            if float(row["delta"]) > float(max_age_sec) or price <= 0:
                return None
            return price
        except Exception as exc:
            log(f"[level8-learning] market price lookup failed {product_id}: {exc}")
            return None

    def _append_missed_opportunity_row(
        self,
        *,
        decision_row: Dict[str, Any],
        review_minutes: int,
        entry_price: float,
        review_price: float,
        move_bps: float,
        missed_type: str,
    ) -> None:
        """Append a chart move the council missed by choosing WAIT or SHADOW."""
        try:
            columns = [
                "ts", "dt_mst", "decision_id", "product_id", "review_minutes",
                "decision_action", "decision_bucket", "decision_strategy",
                "final_buy_score", "buy_threshold", "truth_score",
                "recommended_position_pct", "entry_price", "review_price",
                "move_bps", "missed_type", "reason",
            ]
            write_header = (
                not os.path.exists(MISSED_OPPORTUNITIES_CSV_PATH)
                or os.path.getsize(MISSED_OPPORTUNITIES_CSV_PATH) == 0
            )
            ts_val = now_ts()
            dt_mst = (
                datetime.fromtimestamp(ts_val, tz=timezone.utc)
                .astimezone(TZ)
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            with open(
                MISSED_OPPORTUNITIES_CSV_PATH, "a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(columns)
                writer.writerow([
                    f"{ts_val:.6f}", dt_mst, decision_row.get("decision_id", ""),
                    decision_row.get("product_id", ""), int(review_minutes),
                    decision_row.get("action", ""), decision_row.get("bucket", ""),
                    decision_row.get("strategy", ""),
                    f"{float(decision_row.get('final_buy_score', 0.0) or 0.0):.6f}",
                    f"{float(decision_row.get('buy_threshold', 0.0) or 0.0):.6f}",
                    f"{float(decision_row.get('truth_score', 0.0) or 0.0):.6f}",
                    f"{float(decision_row.get('recommended_position_pct', 0.0) or 0.0):.6f}",
                    f"{float(entry_price):.12f}", f"{float(review_price):.12f}",
                    f"{float(move_bps):.6f}", missed_type,
                    (
                        f"chart_only_review;decision={decision_row.get('action', '')};"
                        f"bucket={decision_row.get('bucket', '')};"
                        f"move_bps={float(move_bps):.2f};"
                        f"review_minutes={int(review_minutes)}"
                    ),
                ])
        except Exception as exc:
            log(f"[level8-learning] missed opportunity append failed: {exc}")

    def _append_council_observation_outcome(
        self,
        *,
        decision_row: Dict[str, Any],
        review_minutes: int,
        entry_price: float,
        review_price: float,
        move_bps: float,
    ) -> None:
        """Log chart outcomes for Level 8 decisions even without a fill."""
        try:
            columns = [
                "ts", "dt_mst", "decision_id", "product_id", "review_minutes",
                "decision_action", "decision_bucket", "decision_strategy",
                "final_buy_score", "buy_threshold", "truth_score",
                "recommended_position_pct", "entry_price", "review_price",
                "move_bps", "would_have_won", "missed_big_move", "reason",
            ]
            write_header = (
                not os.path.exists(COUNCIL_OBSERVATION_OUTCOMES_CSV_PATH)
                or os.path.getsize(COUNCIL_OBSERVATION_OUTCOMES_CSV_PATH) == 0
            )
            ts_val = now_ts()
            dt_mst = (
                datetime.fromtimestamp(ts_val, tz=timezone.utc)
                .astimezone(TZ)
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            with open(
                COUNCIL_OBSERVATION_OUTCOMES_CSV_PATH,
                "a",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(columns)
                writer.writerow([
                    f"{ts_val:.6f}", dt_mst, decision_row.get("decision_id", ""),
                    decision_row.get("product_id", ""), int(review_minutes),
                    decision_row.get("action", ""), decision_row.get("bucket", ""),
                    decision_row.get("strategy", ""),
                    f"{float(decision_row.get('final_buy_score', 0.0) or 0.0):.6f}",
                    f"{float(decision_row.get('buy_threshold', 0.0) or 0.0):.6f}",
                    f"{float(decision_row.get('truth_score', 0.0) or 0.0):.6f}",
                    f"{float(decision_row.get('recommended_position_pct', 0.0) or 0.0):.6f}",
                    f"{float(entry_price):.12f}", f"{float(review_price):.12f}",
                    f"{float(move_bps):.6f}", int(float(move_bps) > 0),
                    int(float(move_bps) >= float(LEVEL8_MISSED_BIG_MOVE_BPS)),
                    (
                        f"observation_outcome;move_bps={float(move_bps):.2f};"
                        f"review_minutes={int(review_minutes)}"
                    ),
                ])
        except Exception as exc:
            log(f"[level8-learning] observation outcome append failed: {exc}")

    def _review_level8_missed_opportunities(self) -> None:
        """Review recent council decisions against subsequent chart movement."""
        if not ENABLE_LEVEL8_MISSED_OPPORTUNITY_LEARNING:
            return
        nowv = now_ts()
        if (
            nowv - float(self.last_level8_missed_opportunity_review_ts)
            < float(LEVEL8_MISSED_REVIEW_EVERY_SEC)
        ):
            return
        self.last_level8_missed_opportunity_review_ts = nowv
        try:
            if not os.path.exists(LEVEL8_COUNCIL_DECISIONS_CSV_PATH):
                return
            decisions = pd.read_csv(LEVEL8_COUNCIL_DECISIONS_CSV_PATH)
            required = {
                "ts", "decision_id", "product_id", "action", "bucket",
                "final_buy_score", "buy_threshold", "truth_score",
                "recommended_position_pct",
            }
            if decisions.empty or not required.issubset(decisions.columns):
                return
            decisions["ts"] = pd.to_numeric(decisions["ts"], errors="coerce")
            decisions = decisions.dropna(subset=["ts"])

            already_reviewed = set()
            if os.path.exists(COUNCIL_OBSERVATION_OUTCOMES_CSV_PATH):
                try:
                    existing = pd.read_csv(COUNCIL_OBSERVATION_OUTCOMES_CSV_PATH)
                    for _, row in existing.iterrows():
                        minutes = pd.to_numeric(
                            row.get("review_minutes", ""), errors="coerce"
                        )
                        if pd.notna(minutes):
                            already_reviewed.add(
                                f"{row.get('decision_id', '')}|{int(minutes)}"
                            )
                except Exception:
                    already_reviewed = set()

            min_window = min(LEVEL8_OBSERVATION_REVIEW_WINDOWS_MIN)
            max_window = max(LEVEL8_OBSERVATION_REVIEW_WINDOWS_MIN)
            reviewable = decisions[
                (decisions["ts"] <= nowv - min_window * 60.0)
                & (decisions["ts"] >= nowv - (max_window + 30) * 60.0)
            ]
            for _, decision in reviewable.tail(800).iterrows():
                product_id = str(decision.get("product_id", ""))
                decision_id = str(decision.get("decision_id", ""))
                if not product_id or not decision_id:
                    continue
                entry_ts = float(decision["ts"])
                entry_price = self._market_price_near_ts(
                    product_id=product_id, target_ts=entry_ts, max_age_sec=120.0
                )
                if entry_price is None:
                    continue
                decision_dict = decision.to_dict()
                for review_minutes in LEVEL8_OBSERVATION_REVIEW_WINDOWS_MIN:
                    key = f"{decision_id}|{int(review_minutes)}"
                    review_ts = entry_ts + float(review_minutes) * 60.0
                    if key in already_reviewed or review_ts > nowv - 10.0:
                        continue
                    review_price = self._market_price_near_ts(
                        product_id=product_id,
                        target_ts=review_ts,
                        max_age_sec=120.0,
                    )
                    if review_price is None:
                        continue
                    move_bps = ((review_price / entry_price) - 1.0) * 10000.0
                    self._append_council_observation_outcome(
                        decision_row=decision_dict,
                        review_minutes=int(review_minutes),
                        entry_price=entry_price,
                        review_price=review_price,
                        move_bps=move_bps,
                    )
                    already_reviewed.add(key)
                    if (
                        str(decision.get("action", "")).upper() in {"WAIT", "SHADOW"}
                        and move_bps >= float(LEVEL8_MISSED_BIG_MOVE_BPS)
                    ):
                        missed_type = (
                            "huge_missed_jump"
                            if move_bps >= float(LEVEL8_MISSED_HUGE_MOVE_BPS)
                            else "missed_big_jump"
                        )
                        self._append_missed_opportunity_row(
                            decision_row=decision_dict,
                            review_minutes=int(review_minutes),
                            entry_price=entry_price,
                            review_price=review_price,
                            move_bps=move_bps,
                            missed_type=missed_type,
                        )
        except Exception as exc:
            log(f"[level8-learning] missed opportunity review failed: {exc}")

    def _append_agent_performance_from_outcomes(self) -> None:
        """
        Connect council votes to trade and chart-only outcomes for learning.
        """
        try:
            votes_path = os.path.join(BASE_DIR, "council_votes.csv")
            trade_outcomes_path = TRADE_OUTCOMES_CSV_PATH
            observation_outcomes_path = COUNCIL_OBSERVATION_OUTCOMES_CSV_PATH
            out_path = AGENT_PERFORMANCE_CSV_PATH

            if not os.path.exists(votes_path):
                return
            if (
                not os.path.exists(trade_outcomes_path)
                and not os.path.exists(observation_outcomes_path)
            ):
                return

            votes = pd.read_csv(votes_path)
            frames = []
            if os.path.exists(trade_outcomes_path):
                trade_outcomes = pd.read_csv(trade_outcomes_path)
                if not trade_outcomes.empty:
                    trade_outcomes["source"] = "trade_outcome"
                    frames.append(trade_outcomes)
            if os.path.exists(observation_outcomes_path):
                observation_outcomes = pd.read_csv(observation_outcomes_path)
                if not observation_outcomes.empty:
                    observation_outcomes["source"] = "observation_outcome"
                    frames.append(observation_outcomes)
            if not frames:
                return
            outcomes = pd.concat(frames, ignore_index=True, sort=False)
            required_vote_cols = {"decision_id", "ts", "product_id", "agent"}
            required_outcome_cols = {"ts", "product_id", "move_bps"}
            if (
                votes.empty
                or outcomes.empty
                or not required_vote_cols.issubset(votes.columns)
                or not required_outcome_cols.issubset(outcomes.columns)
            ):
                return

            votes["ts"] = pd.to_numeric(votes["ts"], errors="coerce")
            outcomes["ts"] = pd.to_numeric(outcomes["ts"], errors="coerce")
            outcomes["move_bps"] = pd.to_numeric(
                outcomes["move_bps"], errors="coerce"
            ).fillna(0.0)

            if "review_minutes" in outcomes.columns:
                outcomes["review_minutes"] = pd.to_numeric(
                    outcomes["review_minutes"], errors="coerce"
                )
                outcomes = outcomes[
                    outcomes["review_minutes"].isin([15, 30, 60])
                ].copy()

            outcomes = outcomes.dropna(subset=["ts"])
            if outcomes.empty:
                return

            existing_keys = set()
            if os.path.exists(out_path):
                try:
                    existing = pd.read_csv(out_path)
                    if not existing.empty and "perf_key" in existing.columns:
                        existing_keys = set(existing["perf_key"].astype(str).tolist())
                except Exception:
                    existing_keys = set()

            write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "ts", "dt_mst", "perf_key", "decision_id", "product_id",
                        "agent", "strategy", "agent_buy_score", "agent_sell_score",
                        "agent_hold_score", "agent_wait_score", "confidence",
                        "reliability", "outcome_move_bps", "outcome_success",
                        "reason",
                    ])

                for _, vote in votes.tail(2000).iterrows():
                    product_id = str(vote.get("product_id", ""))
                    vote_ts = float(vote.get("ts", 0.0) or 0.0)
                    agent = str(vote.get("agent", ""))
                    if not product_id or not math.isfinite(vote_ts) or vote_ts <= 0 or not agent:
                        continue

                    decision_id = str(vote.get("decision_id", ""))
                    product_outcomes = outcomes[
                        outcomes["product_id"].astype(str) == product_id
                    ].copy()
                    if decision_id and "decision_id" in product_outcomes.columns:
                        decision_matches = product_outcomes[
                            product_outcomes["decision_id"].astype(str)
                            == decision_id
                        ].copy()
                        if not decision_matches.empty:
                            product_outcomes = decision_matches
                    product_outcomes = product_outcomes[
                        product_outcomes["ts"] >= vote_ts
                    ].copy()
                    if product_outcomes.empty:
                        continue

                    product_outcomes["time_delta"] = product_outcomes["ts"] - vote_ts
                    outcome = product_outcomes.sort_values("time_delta").iloc[0]
                    move_bps = float(outcome.get("move_bps", 0.0))
                    decision_action = str(
                        outcome.get("decision_action", outcome.get("action", ""))
                    ).upper()
                    if (
                        decision_action in {"WAIT", "SHADOW"}
                        and move_bps >= float(LEVEL8_MISSED_BIG_MOVE_BPS)
                    ):
                        success = 0
                    elif move_bps > 0:
                        success = 1
                    else:
                        success = 0
                    perf_key = (
                        f"{str(vote.get('decision_id', ''))}|"
                        f"{product_id}|{agent}|"
                        f"{int(float(outcome.get('ts', 0.0) or 0.0))}"
                    )
                    if perf_key in existing_keys:
                        continue
                    existing_keys.add(perf_key)

                    ts_val = now_ts()
                    dt_mst = (
                        datetime.fromtimestamp(ts_val, tz=timezone.utc)
                        .astimezone(TZ)
                        .strftime("%Y-%m-%d %H:%M:%S")
                    )
                    writer.writerow([
                        f"{ts_val:.6f}", dt_mst, perf_key,
                        vote.get("decision_id", ""), product_id, agent,
                        vote.get("strategy", ""),
                        vote.get("adjusted_buy_score", ""),
                        vote.get("adjusted_sell_score", ""),
                        vote.get("adjusted_hold_score", ""),
                        vote.get("adjusted_wait_score", ""),
                        vote.get("confidence", ""), vote.get("reliability", ""),
                        f"{move_bps:.6f}", success,
                        f"matched_30m_outcome move_bps={move_bps:.3f}",
                    ])
        except Exception as exc:
            log(f"[level8] agent performance update failed: {exc}")

    async def eval_loop(self) -> None:
        while not self._stop_event.is_set():
            ts_now = now_ts()
            if ENABLE_LEVEL8_MISSED_OPPORTUNITY_LEARNING:
                self._review_level8_missed_opportunities()
            if (
                ENABLE_LEVEL8_COUNCIL
                and ts_now - float(self.last_agent_performance_update_ts) >= 60.0
            ):
                self.last_agent_performance_update_ts = ts_now
                self._append_agent_performance_from_outcomes()
            self._maybe_train_ai_brain()
            loop_gap = ts_now - float(self.last_loop_lag_check_ts or ts_now)
            if loop_gap > EVENT_LOOP_LAG_WARN_SEC:
                log(f"[lag] eval_loop gap={loop_gap:.2f}s; possible blocking work or REST delay")
            self.last_loop_lag_check_ts = ts_now

            if self.pending_buy_reconciliations and isinstance(self.portfolio, LivePortfolio):
                for product_id_r, pending in list(self.pending_buy_reconciliations.items()):
                    age = ts_now - float(pending.get("ts", ts_now))
                    if age < 10.0:
                        continue
                    try:
                        snapshot = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                        base_asset = product_base_asset(product_id_r)
                        qty_now = self.portfolio.get_total_asset(base_asset, snapshot=snapshot or {})
                        before_base = float(pending.get("before_base", 0.0))
                        actual_delta = max(0.0, float(qty_now) - before_base)
                        if actual_delta > 1e-12:
                            tob = self.tob.get(product_id_r)
                            fallback_ask = float(pending.get("ask", 0.0))
                            mid = float(tob.mid) if tob and tob.mid > 0 else fallback_ask
                            entry_price = max(fallback_ask, mid)
                            self._adopt_live_position_after_uncertain_buy(
                                product_id=product_id_r, qty=actual_delta,
                                entry_price=entry_price, pending=pending,
                            )
                            self.reconciliation_log.log_reconciliation(
                                event_type="delayed_buy_adopted", product_id=product_id_r, side="BUY",
                                requested_quote_usd=f"{float(pending.get('requested_quote_usd', 0.0)):.6f}",
                                actual_base_delta=f"{actual_delta:.12f}", before_base=f"{before_base:.12f}",
                                after_base=f"{float(qty_now):.12f}", status="adopted",
                                error=pending.get("reason", ""),
                                action_taken="adopted_coinbase_balance_delta_as_position",
                            )
                            self.pending_buy_reconciliations.pop(product_id_r, None)
                        elif age > 60.0:
                            self.reconciliation_log.log_reconciliation(
                                event_type="delayed_buy_rejected", product_id=product_id_r, side="BUY",
                                requested_quote_usd=f"{float(pending.get('requested_quote_usd', 0.0)):.6f}",
                                before_base=f"{before_base:.12f}", after_base=f"{float(qty_now):.12f}",
                                status="rejected", error="no_base_balance_delta_after_60s",
                                action_taken="dropped_pending_reconciliation",
                            )
                            self.pending_buy_reconciliations.pop(product_id_r, None)
                    except Exception as exc:
                        log(f"[reconcile] delayed buy reconcile failed for {product_id_r}: {exc}")

            if self.post_buy_review_queue and ENABLE_TRADE_OUTCOME_RESEARCH_LOG:
                remaining_reviews: List[Dict[str, Any]] = []
                for review in self.post_buy_review_queue:
                    if ts_now < float(review.get("review_ts", 0.0)):
                        remaining_reviews.append(review)
                        continue
                    product_r = str(review.get("product_id", ""))
                    entry_r = float(review.get("entry_price", 0.0))
                    tob_r = self.tob.get(product_r)
                    if not tob_r or entry_r <= 0:
                        remaining_reviews.append(review)
                        continue
                    current_mid = float(tob_r.mid)
                    move_bps = ((current_mid / entry_r) - 1.0) * 10000.0
                    max_fav: Any = ""
                    max_adv: Any = ""
                    try:
                        candles = list(self.live_1m[product_r].candles)
                        entry_ts = float(review.get("entry_ts", 0.0))
                        path = [c for c in candles if float(c.ts) >= entry_ts]
                        if path:
                            high = max(float(c.high) for c in path)
                            low = min(float(c.low) for c in path)
                            max_fav = f"{((high / entry_r) - 1.0) * 10000.0:.6f}"
                            max_adv = f"{((entry_r / low) - 1.0) * 10000.0:.6f}"
                    except Exception:
                        pass
                    position_open = sum(float(lot.qty) for lot in self.positions.get(product_r, [])) > 1e-12
                    self.trade_outcomes_log.log_outcome(
                        trade_id=review.get("trade_id", ""), product_id=product_r,
                        review_minutes=review.get("review_minutes", ""),
                        entry_ts=f"{float(review.get('entry_ts', 0.0)):.6f}",
                        entry_price=f"{entry_r:.8f}", review_price=f"{current_mid:.8f}",
                        move_bps=f"{move_bps:.6f}", max_favorable_bps=max_fav,
                        max_adverse_bps=max_adv,
                        score_at_entry=f"{float(review.get('score_at_entry', 0.0)):.6f}",
                        prob_at_entry=f"{float(review.get('prob_at_entry', 0.0)):.6f}",
                        ev_at_entry=f"{float(review.get('ev_at_entry', 0.0)):.6f}",
                        spread_at_entry=f"{float(review.get('spread_at_entry', 0.0)):.6f}",
                        timing_reason_at_entry=review.get("timing_reason_at_entry", ""),
                        position_open=position_open, closed=not position_open,
                    )
                    log(
                        f"[post-buy-review] {product_r} trade_id={review.get('trade_id')} "
                        f"window={review.get('review_minutes')}m move_bps={move_bps:.2f} "
                        f"entry={entry_r:.8f} mid={current_mid:.8f}"
                    )
                self.post_buy_review_queue = remaining_reviews

            try:
                await self._refresh_coinbase_fee_tier_if_needed(force=False)
            except Exception as e:
                log_exception("[fee-tier] trading paused because real Coinbase fees are unavailable", e)
                await asyncio.sleep(EVAL_TICK_SEC)
                continue

            if (
                ENABLE_WALK_FORWARD_CALIBRATION
                and not self.live_recalibration_running
                and (
                    ts_now - self.last_hourly_calibration_update_ts
                    >= CALIBRATION_UPDATE_EVERY_SEC
                )
            ):
                self.last_hourly_calibration_update_ts = ts_now
                self.live_recalibration_running = True

                async def _hourly_calibrate_in_thread() -> None:
                    try:
                        await asyncio.to_thread(
                            self._run_hourly_banked_recalibration
                        )
                    except Exception as e:
                        log(
                            f"[calibration] hourly banked recalibration "
                            f"failed: {e}"
                        )
                    finally:
                        self.live_recalibration_running = False

                asyncio.create_task(_hourly_calibrate_in_thread())

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

            skip_new_buys_this_loop = False

            candidates = []
            council_watch_candidates: List[Dict[str, Any]] = []
            for product_id in PRODUCTS:
                tob = self.tob.get(product_id)
                if REQUIRE_FRESH_TOP_OF_BOOK_FOR_BUY:
                    if tob is None or tob.bid <= 0 or tob.ask <= 0:
                        self._rest_backfill_top_of_book([product_id])
                        tob = self.tob.get(product_id)

                    if tob is None or tob.bid <= 0 or tob.ask <= 0:
                        log(f"[buy-skip] {product_id} no fresh top-of-book")
                        continue

                    if now_ts() - float(tob.ts) > float(TOP_OF_BOOK_MAX_STALE_SEC):
                        self._rest_backfill_top_of_book([product_id])
                        tob = self.tob.get(product_id)

                    if (
                        tob is None
                        or now_ts() - float(tob.ts) > float(TOP_OF_BOOK_MAX_STALE_SEC)
                    ):
                        log(f"[buy-skip] {product_id} stale top-of-book")
                        continue

                if not tob:
                    continue
                bid, ask, mid, spread_bps = tob.bid, tob.ask, tob.mid, tob.spread_bps
                levels_day = self.macro.get_levels(product_id, "day")
                levels_week = self.macro.get_levels(product_id, "week")
                weekly_bias = self.macro.compute_weekly_bias(product_id, mid) if levels_week else None
                minute_candles = list(self.live_1m.get(product_id).candles) if self.live_1m.get(product_id) else []
                sigma_bps = self._compute_sigma_bps_from_1m(product_id)

                lots = self.positions.get(product_id, [])
                if ENABLE_INVERTED_STOPLOSS_CYCLE:
                    inverted_lot = any(
                        isinstance(lot.meta, dict)
                        and lot.meta.get("strategy_mode") == "inverted_stoploss_cycle"
                        for lot in lots
                    )
                    if inverted_lot:
                        # Inverted positions are managed by the cycle processor.
                        continue

                position_qty = sum(l.qty for l in lots)
                avg_entry_price = (sum(l.qty * l.price for l in lots) / position_qty) if position_qty > 0 else None

                if position_qty > 0 and avg_entry_price and avg_entry_price > 0:
                    lot = lots[0]
                    lot_tier = lot.tier if lot.tier in EXIT_PLAN else TIER_LOW
                    lot_meta = lot.meta
                    exit_plan = get_exit_plan_for_tier(lot_tier)
                    targets = get_exit_targets(entry_price=avg_entry_price, sigma_bps=(sigma_bps or 35.0), tier=lot_tier)

                    sell_qty = 0.0
                    exit_reason = ""
                    exit_role = "level8_direct_exit_review"
                    unrealized_bps = ((float(bid) / float(avg_entry_price)) - 1.0) * 10000.0
                    if position_qty > 1e-12 and bid > 0 and ask > 0:
                        try:
                            should_exit_l8, level8_exit_reason = self._level8_should_hold_or_exit(
                                product_id=product_id, entry_price=float(avg_entry_price),
                                current_price=float(bid), unrealized_bps=float(unrealized_bps),
                                spread_bps=float(tob.spread_bps if tob else 0.0),
                                cost_bps=float(self._round_trip_cost_bps(spread_bps=spread_bps)),
                                default_exit_reason="level8_direct_exit_review", hard_exit=False,
                            )
                            if should_exit_l8:
                                sell_qty, exit_reason, exit_role = position_qty, level8_exit_reason, "level8_direct_exit"
                            else:
                                log(f"[level8] hold {product_id}: {level8_exit_reason}")
                        except Exception as exc:
                            log(f"[level8] direct exit review failed {product_id}: {exc}")
                    if sell_qty > 0:
                        last_sell_fail = self.last_sell_failure_ts_by_product.get(product_id, 0.0)
                        if ts_now - float(last_sell_fail) < 5.0:
                            log(f"[sell-skip] {product_id} recent sell failure cooldown")
                            continue
                        sell_trade_id = next((str(l.meta.get("trade_id")) for l in lots if isinstance(l.meta, dict) and l.meta.get("trade_id")), "")
                        self.signal_events_log.log_event(event_type="sell_attempt", trade_id=sell_trade_id, product_id=product_id, action="attempt_sell", reason=str(exit_reason or "level8_direct_exit"))
                        log(f"[sell-attempt] {product_id} qty={sell_qty:.12f} reason={exit_reason} role={exit_role} bid={bid:.8f} ask={ask:.8f} avg_entry={avg_entry_price:.8f}")
                        notional_usd, exec_price, fee, filled_notional = sell_qty * bid, bid, 0.0, None
                        fill = await self._execute_live_sell(product_id=product_id, base_qty=sell_qty, bid=bid, ask=ask, reason=exit_reason or "level8_direct_exit")
                        if fill is not None:
                            filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
                            self.signal_events_log.log_event(event_type="sell_fill", trade_id=sell_trade_id, product_id=product_id, action="sell_filled", reason=f"exit_role={exit_role};exit_reason={exit_reason}")
                            sell_qty, exec_price, fee = min(float(sell_qty), float(filled_qty)), float(avg_px), float(fee_val)
                            notional_usd = float(filled_notional) if filled_notional is not None else sell_qty * exec_price
                        else:
                            self.last_sell_failure_ts_by_product[product_id] = ts_now
                            sell_qty = 0.0
                        if sell_qty > 0:
                            fifo_cost, fifo_avg_entry = self._fifo_cost_basis(list(lots), sell_qty)
                            pnl_gross = float(notional_usd) - float(fifo_cost)
                            self.tlog.log_trade(event="SELL", product_id=product_id, side="SELL", qty=sell_qty, price=exec_price, fee_usd_val=fee, gross_pnl_usd=pnl_gross, net_pnl_usd=pnl_gross-fee, entry_price=fifo_avg_entry if fifo_avg_entry is not None else avg_entry_price, exit_price=exec_price, weekly_bias=weekly_bias, note=exit_reason or "level8_direct_exit", filled_notional_usd=float(filled_notional) if filled_notional is not None else None, exit_role=exit_role)
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

                profile = self.calibration_profiles.get(
                    product_id,
                    ProductCalibrationProfile(product_id=product_id),
                )

                council_watch_candidates.append({
                    "product_id": product_id,
                    "mid": float(mid),
                    "bid": float(bid),
                    "ask": float(ask),
                    "spread_bps": float(spread_bps),
                    "score": float(scored.score),
                    "tier": int(scored.tier),
                    "entry_reason": str(scored.reason),
                    "entry_score_obj": scored,
                    "signal": live_signal,
                    "entry_timing_ok": False,
                    "entry_timing_reason": "heartbeat_market_watch",
                    "expected_net_edge_bps": float(scored.expected_net_edge_bps),
                    "estimated_prob_up": float(estimated_prob_up),
                    "position_pct": float(position_pct),
                    "target_bps": float(target_bps),
                    "cost_bps": float(cost_bps),
                    "projected_forward_gain_bps": float(
                        live_signal.projected_forward_gain_bps
                    ),
                    "min_score": float(profile.min_score),
                    "min_probability": float(profile.min_probability),
                    "min_expected_value_bps": float(
                        profile.min_expected_value_bps
                    ),
                    "calibrated_time_to_min_profit_minutes": float(
                        live_signal.calibrated_time_to_min_profit_minutes
                    ),
                    "calibrated_forward_window_minutes": float(
                        live_signal.calibrated_forward_window_minutes
                    ),
                    "rank_score": (
                        float(scored.score)
                        + float(scored.expected_net_edge_bps) * 0.10
                    ),
                    "buy_ready_count": 0,
                    "manager_strategy": "COUNCIL_HEARTBEAT",
                    "heartbeat_only": True,
                    "ok_to_trade": bool(live_signal.ok_to_trade),
                    "why_not_ready": (
                        "passes_basic_live_signal"
                        if bool(live_signal.ok_to_trade)
                        else str(live_signal.reason)
                    ),
                })

            if ENABLE_LEVEL8_COUNCIL and LEVEL8_ENABLE_COUNCIL_HEARTBEAT:
                self._run_level8_council_heartbeat(
                    watch_candidates=council_watch_candidates,
                )

            if ENABLE_LEVEL8_COUNCIL:
                candidates = []
                for watch_candidate in council_watch_candidates:
                    product_id_l8 = str(watch_candidate.get("product_id", ""))
                    if not product_id_l8:
                        continue
                    bid_l8 = float(watch_candidate.get("bid", 0.0) or 0.0)
                    ask_l8 = float(watch_candidate.get("ask", 0.0) or 0.0)
                    mid_l8 = float(watch_candidate.get("mid", 0.0) or 0.0)
                    spread_l8 = float(watch_candidate.get("spread_bps", 999.0) or 999.0)
                    if bid_l8 <= 0 or ask_l8 <= 0 or mid_l8 <= 0 or ask_l8 < bid_l8 or spread_l8 > 250.0:
                        continue
                    c = dict(watch_candidate)
                    c["manager_strategy"] = "LEVEL8_DIRECT"
                    c["entry_reason"] = f"level8_direct_market_candidate;score={float(c.get('score',0.0)):.2f};prob={float(c.get('estimated_prob_up',0.0)):.3f};ev={float(c.get('expected_net_edge_bps',0.0)):.2f};spread={spread_l8:.2f}"
                    c["heartbeat_only"], c["learning_candidate"] = False, True
                    c["entry_timing_ok"], c["entry_timing_reason"] = True, "level8_direct_no_pre_l8_timing_gate"
                    c["rank_score"] = float(c.get("score",0.0)) + float(c.get("expected_net_edge_bps",0.0))*0.03 + float(c.get("estimated_prob_up",0.0))*20.0
                    candidates.append(c)
                log(f"[level8-direct] direct market candidates={len(candidates)} from_watch={len(council_watch_candidates)}")

            if skip_new_buys_this_loop:
                candidates = []

            if ENABLE_LEVEL8_COUNCIL and candidates:
                level8_filtered_candidates: List[Dict[str, Any]] = []
                for candidate in candidates:
                    product_id_l8 = str(candidate.get("product_id", ""))
                    candidate["manager_strategy"] = str(
                        candidate.get("level8_strategy")
                        or candidate.get("manager_strategy")
                        or candidate.get("entry_reason")
                        or "LEVEL8_DIRECT"
                    )
                    level8_ok, level8_info = self._level8_decision_for_candidate(
                        candidate=candidate
                    )
                    candidate["level8_ok"] = bool(level8_ok)
                    candidate["level8_action"] = level8_info.get("action", "")
                    candidate["level8_strategy"] = level8_info.get("strategy", "")
                    candidate["level8_bucket"] = level8_info.get("bucket", "")
                    candidate["level8_risk_mode"] = level8_info.get("risk_mode", "")
                    candidate["level8_decision_id"] = level8_info.get("decision_id", "")
                    candidate["level8_truth_score"] = float(
                        level8_info.get("truth_score", 0.0) or 0.0
                    )
                    candidate["level8_final_buy_score"] = float(
                        level8_info.get("final_buy_score", 0.0) or 0.0
                    )
                    candidate["level8_buy_threshold"] = float(
                        level8_info.get("buy_threshold", 0.0) or 0.0
                    )
                    candidate["level8_recommended_position_pct"] = float(
                        level8_info.get("recommended_position_pct", 0.0) or 0.0
                    )
                    candidate["level8_reason"] = str(level8_info.get("reason", ""))
                    self._append_level8_decision_snapshot(
                        product_id=product_id_l8,
                        decision_id=str(level8_info.get("decision_id", "")),
                        action=str(level8_info.get("action", "WAIT")),
                        strategy=str(level8_info.get("strategy", "")),
                        bucket=str(level8_info.get("bucket", "")),
                        risk_mode=str(
                            level8_info.get("risk_mode", "NORMAL")
                        ),
                        truth_score=float(
                            level8_info.get("truth_score", 0.0) or 0.0
                        ),
                        final_buy_score=float(
                            level8_info.get("final_buy_score", 0.0) or 0.0
                        ),
                        final_sell_score=0.0,
                        buy_threshold=float(
                            level8_info.get("buy_threshold", 0.0) or 0.0
                        ),
                        sell_threshold=0.0,
                        recommended_position_pct=float(
                            level8_info.get(
                                "recommended_position_pct", 0.0
                            )
                            or 0.0
                        ),
                        confidence=float(
                            level8_info.get("confidence", 0.0) or 0.0
                        ),
                        reason=str(level8_info.get("reason", "")),
                    )
                    try:
                        self.signal_events_log.log_event(
                            event_type="level8_council_decision",
                            trade_id=level8_info.get("decision_id", ""),
                            product_id=product_id_l8,
                            rank_score=f"{float(candidate.get('rank_score', 0.0)):.6f}",
                            buy_ready_count=len(candidates),
                            score=f"{float(candidate.get('score', 0.0)):.6f}",
                            probability=f"{float(candidate.get('estimated_prob_up', 0.0)):.6f}",
                            ev_bps=f"{float(candidate.get('expected_net_edge_bps', 0.0)):.6f}",
                            projected_forward_bps=f"{float(candidate.get('projected_forward_gain_bps', 0.0)):.6f}",
                            cost_bps=f"{float(candidate.get('cost_bps', 0.0)):.6f}",
                            spread_bps=f"{float(candidate.get('spread_bps', 0.0)):.6f}",
                            action="keep" if level8_ok else "reject",
                            reason=level8_info.get("reason", ""),
                        )
                    except Exception:
                        pass
                    if level8_ok:
                        level8_filtered_candidates.append(candidate)
                    else:
                        log(
                            f"[level8] blocked {product_id_l8}: "
                            f"action={level8_info.get('action')} "
                            f"bucket={level8_info.get('bucket')} "
                            f"reason={level8_info.get('reason')}"
                        )
                candidates = level8_filtered_candidates

            if candidates:
                top_preview = ", ".join(
                    f"{candidate.get('product_id')}(rank={float(candidate.get('rank_score', 0.0)):.2f},"
                    f"score={float(candidate.get('score', 0.0)):.1f},"
                    f"prob={float(candidate.get('estimated_prob_up', 0.0)):.3f},"
                    f"ev={float(candidate.get('expected_net_edge_bps', 0.0)):.1f},"
                    f"pct={float(candidate.get('position_pct', 0.0)):.3f})"
                    for candidate in candidates[:5]
                )
                log(f"[buy-candidates] buy_ready={buy_ready_count} selectable={len(candidates)} top={top_preview}")
            else:
                log(f"[buy-candidates] buy_ready={buy_ready_count} selectable=0")

            strong_candidate_count = sum(1 for c in candidates if c["score"] >= MID_SCORE_UTIL_THRESHOLD)
            max_deploy_this_eval = (
                float(equity_usd)
                * (float(LEVEL8_MAX_TOTAL_EXPOSURE_PCT) if ENABLE_LEVEL8_COUNCIL else float(MAX_CASH_DEPLOYED_PER_EVAL_PCT_OF_EQUITY))
            )
            deployed_this_eval = 0.0

            timed_candidates = []
            for candidate in candidates:
                candidate["entry_timing_ok"] = True
                candidate["entry_timing_reason"] = "level8_direct_timing_gate_removed"
                timed_candidates.append(candidate)
                try:
                    self.signal_events_log.log_event(event_type="entry_timing_check", product_id=str(candidate.get("product_id", "")), rank_score=f"{float(candidate.get('rank_score',0.0)):.6f}", buy_ready_count=len(candidates), score=f"{float(candidate.get('score',0.0)):.6f}", probability=f"{float(candidate.get('estimated_prob_up',0.0)):.6f}", ev_bps=f"{float(candidate.get('expected_net_edge_bps',0.0)):.6f}", projected_forward_bps=f"{float(candidate.get('projected_forward_gain_bps',0.0)):.6f}", cost_bps=f"{float(candidate.get('cost_bps',0.0)):.6f}", spread_bps=f"{float(candidate.get('spread_bps',0.0)):.6f}", entry_timing_ok=True, entry_timing_reason="level8_direct_timing_gate_removed", action="keep", reason="timing is now a Level 8 council input, not a pre-Level-8 blocker")
                except Exception:
                    pass

            ai_filtered_candidates = []
            for candidate in timed_candidates:
                product_id_for_ai = str(candidate.get("product_id", ""))
                ai_reason = "ai_pre_l8_veto_removed_ai_is_council_input"
                candidate["ai_ok"], candidate["ai_reason"] = True, ai_reason
                try:
                    self.signal_events_log.log_event(event_type="ai_candidate_filter", product_id=product_id_for_ai, rank_score=f"{float(candidate.get('rank_score',0.0)):.6f}", score=f"{float(candidate.get('score',0.0)):.6f}", probability=f"{float(candidate.get('estimated_prob_up',0.0)):.6f}", ev_bps=f"{float(candidate.get('expected_net_edge_bps',0.0)):.6f}", spread_bps=f"{float(candidate.get('spread_bps',0.0)):.6f}", action="keep", reason=ai_reason)
                except Exception:
                    pass
                ai_filtered_candidates.append(candidate)

            if ENABLE_LEVEL8_LEARNING_MODE:
                candidate_slice = ai_filtered_candidates[
                    : int(LEVEL8_LEARNING_MAX_NEW_ENTRIES_PER_EVAL)
                ]
            else:
                candidate_slice = (
                    ai_filtered_candidates[:MAX_NEW_ENTRIES_PER_EVAL]
                    if ENABLE_MULTI_CANDIDATE_BUYS
                    else ai_filtered_candidates[:1]
                )

            if ENABLE_INVERTED_STOPLOSS_CYCLE:
                for candidate in ai_filtered_candidates:
                    product_id_for_marker = str(candidate.get("product_id", ""))
                    if not product_id_for_marker:
                        continue
                    if self._inverted_has_open_position(product_id_for_marker):
                        continue
                    if self._inverted_marker_is_active(product_id_for_marker):
                        continue
                    self._set_inverted_marker_from_candidate(candidate=candidate)

                # This runs every evaluation, including evaluations with no fresh
                # candidate, so existing markers and positions remain managed.
                await self._process_inverted_stoploss_cycle(equity_usd=equity_usd)
                log(f"[loop] sleeping {EVAL_TICK_SEC:.1f}s until next evaluation")
                await asyncio.sleep(EVAL_TICK_SEC)
                continue

            for candidate in candidate_slice:
                product_id = candidate["product_id"]
                if product_id in self.pending_buy_reconciliations:
                    log(f"[buy-skip] {product_id} pending delayed buy reconciliation")
                    continue
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
                    current_adds = int(self.scale_add_count.get(product_id, 0))
                    local_lots = self.positions.get(product_id, [])
                    local_qty = sum(l.qty for l in local_lots)
                    local_cost = sum(l.qty * l.price for l in local_lots)
                    local_avg = local_cost / local_qty if local_qty > 0 and local_cost > 0 else 0.0
                    max_product_exposure = float(equity_usd) * float(MAX_EXPOSURE_PER_PRODUCT_PCT_OF_EQUITY)
                    remaining_product_room = max(0.0, max_product_exposure - float(product_exposure))
                    entry_notional = min(entry_notional, remaining_product_room)

                remaining_eval_budget = max(
                    0.0,
                    float(max_deploy_this_eval) - float(deployed_this_eval),
                )
                entry_notional = min(float(entry_notional), remaining_eval_budget)

                if ENABLE_LEVEL8_COUNCIL and str(LEVEL8_MODE).upper() in {
                    "FILTER_AND_SIZE",
                    "COUNCIL_CONTROL",
                }:
                    l8_pct = float(
                        candidate.get(
                            "level8_recommended_position_pct",
                            MAX_SINGLE_BUY_PCT_OF_EQUITY,
                        )
                        or 0.0
                    )
                    if l8_pct > 0:
                        level8_cap = float(equity_usd) * min(
                            float(l8_pct),
                            float(LEVEL8_MAX_SINGLE_TRADE_PCT),
                        )
                        entry_notional = min(float(entry_notional), level8_cap)

                    reserve_cash_required = (
                        float(equity_usd) * float(LEVEL8_RESERVE_CASH_PCT)
                    )
                    spendable_cash = max(
                        0.0, float(cash_usd) - reserve_cash_required
                    )
                    entry_notional = min(float(entry_notional), spendable_cash)
                    if entry_notional <= 0:
                        log(
                            f"[level8] {product_id} entry_notional blocked by "
                            f"20pct reserve cash={cash_usd:.6f} "
                            f"equity={equity_usd:.6f} "
                            f"reserve={reserve_cash_required:.6f}"
                        )
                        continue

                min_order = max(float(MIN_ENTRY_USD), float(MIN_LIVE_ORDER_USD))

                if entry_notional < min_order:
                    reserve_cash_required = (
                        float(equity_usd) * float(LEVEL8_RESERVE_CASH_PCT)
                        if ENABLE_LEVEL8_COUNCIL
                        else float(RESERVE_USD)
                    )
                    spendable_cash = max(
                        0.0, float(cash_usd) - reserve_cash_required
                    )
                    if (
                        ENABLE_LEVEL8_COUNCIL
                        and str(candidate.get("level8_action", "")).upper()
                        == "ALLOW_BUY"
                        and spendable_cash >= min_order
                        and remaining_eval_budget >= min_order
                    ):
                        log(
                            f"[level8] {product_id} raising ALLOW_BUY notional "
                            f"to min live order old={entry_notional:.2f} "
                            f"new={min_order:.2f}"
                        )
                        entry_notional = min_order
                    else:
                        log(
                            f"[buy-skip] {product_id} below_min_order "
                            f"entry_notional={entry_notional:.2f} "
                            f"min_order={min_order:.2f} cash={cash_usd:.2f} "
                            f"equity={equity_usd:.2f}"
                        )
                        continue

                bid, ask = candidate["bid"], candidate["ask"]

                entry_mode_for_this_trade = ENTRY_EXECUTION_MODE

                if USE_EDGE_AWARE_ENTRY_EXECUTION:
                    projected_net = float(candidate.get("expected_net_edge_bps", 0.0))

                    if projected_net >= float(HIGH_EDGE_MIN_PROJECTED_NET_BPS):
                        entry_mode_for_this_trade = ENTRY_HIGH_EDGE_MODE
                    elif projected_net >= float(MEDIUM_EDGE_MIN_PROJECTED_NET_BPS):
                        entry_mode_for_this_trade = ENTRY_MEDIUM_EDGE_MODE
                    else:
                        entry_mode_for_this_trade = ENTRY_LOW_EDGE_MODE

                entry_fee_bps = self._entry_fee_bps_for_mode(
                    execution_mode=entry_mode_for_this_trade
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

                trade_id = f"{product_id}-{int(now_ts())}-{uuid.uuid4().hex[:8]}"
                before_buy_base = 0.0
                before_buy_cash = float(cash_usd)
                try:
                    if isinstance(self.portfolio, LivePortfolio):
                        before_buy_snapshot = await self._live_refresh_snapshot(force=True, ttl_sec=0.0)
                        before_buy_base = self.portfolio.get_product_total_qty(
                            product_id, snapshot=before_buy_snapshot or {}
                        )
                        before_buy_cash = self.portfolio.get_tradable_usd(
                            snapshot=before_buy_snapshot or {}
                        )
                except Exception as exc:
                    log(f"[reconcile] pre-buy snapshot failed for {product_id}: {exc}")

                self.signal_events_log.log_event(
                    event_type="buy_attempt", trade_id=trade_id, product_id=product_id,
                    rank_score=f"{float(candidate.get('rank_score', 0.0)):.6f}",
                    buy_ready_count=buy_ready_count, score=f"{float(candidate.get('score', 0.0)):.6f}",
                    probability=f"{float(candidate.get('estimated_prob_up', 0.0)):.6f}",
                    ev_bps=f"{float(candidate.get('expected_net_edge_bps', 0.0)):.6f}",
                    projected_forward_bps=f"{float(candidate.get('projected_forward_gain_bps', 0.0)):.6f}",
                    cost_bps=f"{float(candidate.get('cost_bps', 0.0)):.6f}",
                    spread_bps=f"{float(candidate.get('spread_bps', 0.0)):.6f}",
                    entry_timing_ok=candidate.get("entry_timing_ok", ""),
                    entry_timing_reason=candidate.get("entry_timing_reason", ""),
                    action="attempt_buy", reason="selected_after_rank_and_timing",
                )

                log(
                    f"[buy-attempt] {product_id} "
                    f"mode={entry_mode_for_this_trade} "
                    f"quote_usd={entry_notional:.2f} "
                    f"entry_fee_bps={entry_fee_bps:.3f} "
                    f"timing={candidate.get('entry_timing_ok')} "
                    f"timing_reason={candidate.get('entry_timing_reason', '')} "
                    f"score={float(candidate.get('score', 0.0)):.3f} "
                    f"prob={float(candidate.get('estimated_prob_up', 0.0)):.6f} "
                    f"ev={float(candidate.get('expected_net_edge_bps', 0.0)):.3f} "
                    f"level8_action={candidate.get('level8_action', '')} "
                    f"level8_strategy={candidate.get('level8_strategy', '')} "
                    f"level8_bucket={candidate.get('level8_bucket', '')} "
                    f"level8_truth={float(candidate.get('level8_truth_score', 0.0)):.3f} "
                    f"level8_buy_score={float(candidate.get('level8_final_buy_score', 0.0)):.3f} "
                    f"level8_threshold={float(candidate.get('level8_buy_threshold', 0.0)):.3f} "
                    f"bid={bid:.8f} ask={ask:.8f}"
                )
                fill = await self._execute_live_buy(
                    product_id=product_id,
                    quote_usd=entry_notional,
                    bid=bid,
                    ask=ask,
                    reason=candidate.get("entry_reason", "score_entry"),
                    execution_mode=entry_mode_for_this_trade,
                )

                if fill is None:
                    result = dict(self.last_buy_execution_result.get(product_id) or {})
                    error_text = str(result.get("error") or result.get("status") or "")
                    uncertain = any(token in error_text.lower() for token in (
                        "buy_no_base_balance_delta", "balance_snapshot",
                        "ambiguous_fill", "balance_delta_reconcile",
                    ))
                    if uncertain:
                        self.pending_buy_reconciliations[product_id] = {
                            "ts": now_ts(), "product_id": product_id,
                            "requested_quote_usd": float(entry_notional),
                            "candidate": dict(candidate), "bid": bid, "ask": ask,
                            "before_base": float(before_buy_base),
                            "before_cash": float(before_buy_cash),
                            "trade_id": trade_id, "reason": error_text,
                        }
                        self.reconciliation_log.log_reconciliation(
                            event_type="pending_buy_reconciliation", product_id=product_id, side="BUY",
                            client_order_id=result.get("client_order_id", ""),
                            order_id=result.get("order_id", ""),
                            requested_quote_usd=f"{float(entry_notional):.6f}",
                            before_base=f"{float(before_buy_base):.12f}",
                            before_cash=f"{float(before_buy_cash):.6f}",
                            status="pending", error=error_text,
                            action_taken="queued_for_delayed_reconcile",
                        )
                    log(
                        f"[buy-failed] {product_id} live buy returned no confirmed fill "
                        f"error={error_text}"
                    )
                    continue

                filled_qty, avg_px, fee_val, filled_notional, _order_id = fill
                actual_deployment = float(filled_notional or entry_notional)
                deployed_this_eval += actual_deployment
                cash_usd = max(0.0, float(cash_usd) - actual_deployment - float(fee_val))
                log(
                    f"[buy-success] {product_id} "
                    f"qty={float(filled_qty):.12f} avg_px={float(avg_px):.8f} "
                    f"fee={float(fee_val):.6f} "
                    f"filled_notional={float(filled_notional or 0.0):.6f} "
                    f"order_id={_order_id}"
                )
                self.armed_buy_signals.pop(product_id, None)
                entry_ts = now_ts()
                self._queue_post_buy_reviews(
                    trade_id=trade_id, product_id=product_id, entry_ts=entry_ts,
                    entry_price=float(avg_px), candidate=candidate,
                )
                self.signal_events_log.log_event(
                    event_type="buy_fill", trade_id=trade_id, product_id=product_id,
                    action="buy_filled",
                    reason=(
                        f"avg_px={float(avg_px):.8f};qty={float(filled_qty):.12f};"
                        f"notional={float(filled_notional or entry_notional):.6f}"
                    ),
                )
                qty1 = float(filled_qty)
                buy_px1 = float(avg_px)
                fee1 = float(fee_val)
                eff_price1 = float((filled_notional + fee1) / qty1) if qty1 > 0 and filled_notional is not None else buy_px1

                if qty1 > 0:
                    lot_meta = {
                        "trade_id": trade_id,
                        "entry_ts": entry_ts,
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
                        "profit_lock_armed": False,
                        "profit_lock_price": None,
                        "min_profitable_exit_price": None,
                        "calibrated_time_to_min_profit_minutes": float(candidate.get("calibrated_time_to_min_profit_minutes", 0.0)),
                        "calibrated_forward_window_minutes": float(candidate.get("calibrated_forward_window_minutes", 0.0)),
                        "calibrated_post_profit_breathing_minutes": float(candidate.get("calibrated_post_profit_breathing_minutes", CALIB_POST_PROFIT_BREATHING_MINUTES)),
                        "level8_decision_id": candidate.get("level8_decision_id", ""),
                        "level8_action": candidate.get("level8_action", ""),
                        "level8_strategy": candidate.get("level8_strategy", ""),
                        "level8_bucket": candidate.get("level8_bucket", ""),
                        "level8_truth_score": float(candidate.get("level8_truth_score", 0.0)),
                        "level8_final_buy_score": float(candidate.get("level8_final_buy_score", 0.0)),
                        "level8_buy_threshold": float(candidate.get("level8_buy_threshold", 0.0)),
                        "level8_reason": candidate.get("level8_reason", ""),
                    }
                    lot_meta["min_profitable_exit_price"] = float(required_exit_price_for_net_gain(
                        effective_entry_price=eff_price1,
                        exit_fee_bps=self._exit_fee_bps_for_mode(),
                        est_slippage_bps=EST_SLIPPAGE_BPS,
                        est_adverse_fill_bps=EST_ADVERSE_FILL_BPS,
                        min_net_gain_bps=max(
                            MIN_NET_PROFIT_BPS_FOR_DISCRETIONARY_EXIT,
                            MIN_NET_GAIN_AFTER_FEES_BPS,
                        ),
                    ))
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
                "profit_lock_armed": False,
                "profit_lock_price": None,
                "min_profitable_exit_price_from_lot": None,
                "calibrated_forward_window_minutes": None,
                "calibrated_post_profit_breathing_minutes": None,
                "inverted_mode": bool(ENABLE_INVERTED_STOPLOSS_CYCLE),
                "inverted_marker_price": None,
                "inverted_buy_trigger_price": None,
                "inverted_target_sell_price": None,
                "inverted_next_loss_trigger_price": None,
                "inverted_rebuy_count": 0,
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
                    "profit_lock_armed": bool(lot_meta.get("profit_lock_armed", False)),
                    "profit_lock_price": lot_meta.get("profit_lock_price"),
                    "min_profitable_exit_price_from_lot": lot_meta.get("min_profitable_exit_price"),
                    "calibrated_forward_window_minutes": lot_meta.get("calibrated_forward_window_minutes"),
                    "calibrated_post_profit_breathing_minutes": lot_meta.get("calibrated_post_profit_breathing_minutes"),
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

            marker = self.inverted_markers.get(product_id)
            if marker:
                row["inverted_marker_price"] = marker.get("marker_price")
                row["inverted_buy_trigger_price"] = marker.get("buy_trigger_price")
                row["inverted_target_sell_price"] = self._inverted_target_sell_price(
                    product_id
                )
                row["inverted_next_loss_trigger_price"] = (
                    self._inverted_next_loss_trigger_price(product_id)
                )
                row["inverted_rebuy_count"] = int(marker.get("rebuy_count", 0))

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
                    source="telemetry",
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


def _coinbase_response_dict(response: Any) -> Dict[str, Any]:
    """Convert a Coinbase SDK response into a plain dictionary."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        data = response.to_dict()
        return data if isinstance(data, dict) else {}
    data = getattr(response, "__dict__", None)
    return data if isinstance(data, dict) else {}


def _coinbase_flag(value: Any) -> bool:
    """Interpret Coinbase boolean fields without treating the string 'false' as true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def get_available_product_ids(
    client: RESTClient,
    configured_products: Optional[List[str]] = None,
) -> Set[str]:
    """Return product ids that Coinbase currently reports as tradable."""
    requested = list(configured_products or [])
    try:
        response = (
            client.get_products(
                product_ids=requested,
                get_tradability_status=True,
            )
            if requested
            else client.get_products(get_tradability_status=True)
        )
    except TypeError:
        # Support older coinbase-advanced-py releases without these filters.
        response = client.get_products(limit=1000)

    data = _coinbase_response_dict(response)
    products: Any = data.get("products", [])
    if not isinstance(products, list):
        products = getattr(response, "products", [])
    if not isinstance(products, list):
        products = []

    available: Set[str] = set()
    allowed_statuses = {"online", "active", "tradable"}
    for product in products:
        item = _coinbase_response_dict(product)
        product_id = str(item.get("product_id", "")).strip()
        status = str(item.get("status", "")).strip().lower()
        disabled = any(
            _coinbase_flag(item.get(field, False))
            for field in ("trading_disabled", "is_disabled", "view_only", "cancel_only")
        )

        if not product_id or disabled:
            continue
        if status and status not in allowed_statuses:
            continue
        available.add(product_id)

    return available


def validate_configured_products_with_coinbase(
    products: List[str],
    client: RESTClient,
) -> List[str]:
    """Keep only configured products Coinbase currently reports as tradable."""
    try:
        available = get_available_product_ids(client, configured_products=products)
    except Exception as exc:
        log(
            "[products] could not validate product list with Coinbase; "
            f"using configured list: {exc}"
        )
        return list(products)

    valid = [product for product in products if product in available]
    removed = [product for product in products if product not in available]

    if removed:
        log(f"[products] removed unavailable Coinbase products: {removed}")
    if not valid:
        raise RuntimeError("No configured products are currently available on Coinbase.")

    return valid


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

    # Currency safety: enforce USD quote pairs, then confirm Coinbase currently
    # permits them before creating subscriptions or evaluating orders.
    PRODUCTS = [p for p in PRODUCTS if p.endswith("-USD")]
    PRODUCTS = await asyncio.to_thread(
        validate_configured_products_with_coinbase,
        PRODUCTS,
        rest,
    )

    log(f"[config] product_count={len(PRODUCTS)} products={PRODUCTS}")
    if len(PRODUCTS) < 15:
        log(
            "[config] warning: fewer than 15 products active after Coinbase "
            f"validation; active_count={len(PRODUCTS)}"
        )
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
