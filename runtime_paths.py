import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_ROOT_DIR = os.path.join(BASE_DIR, "CSVs")
RAW_CACHE_DIR = os.path.join(CSV_ROOT_DIR, "01_raw_cache")
REPLAY_DIR = os.path.join(CSV_ROOT_DIR, "02_replay")
BACKTEST_DIR = os.path.join(CSV_ROOT_DIR, "03_backtest")
RISK_DIR = os.path.join(CSV_ROOT_DIR, "04_risk")
QUANT_DIR = os.path.join(CSV_ROOT_DIR, "05_quant_state")
LIVE_DIR = os.path.join(CSV_ROOT_DIR, "06_live_trading")
RUNTIME_DIR = os.path.join(CSV_ROOT_DIR, "07_runtime_state")
VIEWER_DIR = os.path.join(CSV_ROOT_DIR, "08_viewer")
RESEARCH_DIR = os.path.join(CSV_ROOT_DIR, "09_continuous_research")
ARCHIVE_DIR = os.path.join(CSV_ROOT_DIR, "99_archive")
DEBUG_DIR = os.path.join(BASE_DIR, "debug")

RAW_CACHE_FILES = {"historical_replay_15m_90d.csv","historical_replay_1h_365d.csv","historical_replay_1d_2y.csv","historical_shadow_replay.csv","micro_history.csv","macro_day.csv","macro_week.csv","macro_levels.csv","market.csv","trade_outcomes.csv","feature_store_summary.csv"}
REPLAY_FILES = {"candidate_replay.csv","historical_replay_summary.csv","replay_fee_comparison_summary.csv","strategy_variant_replay_summary.csv","shadow_sell_replay.csv","shadow_trades.csv","fifth_pass_live_style_replay.csv","fifth_pass_live_style_summary.csv","fifth_pass_product_contribution.csv","fifth_pass_blockers.csv"}
BACKTEST_FILES = {"backtest_recommendations.csv","backtest_sell_recommendations.csv","backtest_agent_priors.csv","backtest_setup_performance.csv","backtest_summary.csv","walk_forward_validation.csv","agent_ablation.csv","agent_decision_influence.csv","product_agent_influence.csv","four_pass_agent_buy_timing.csv","four_pass_agent_sell_timing.csv","four_pass_council_buy_timing.csv","four_pass_council_sell_timing.csv","four_pass_agent_context_ratings.csv","four_pass_final_agent_ratings.csv","four_pass_product_live_gate.csv","four_pass_profitability_summary.csv","four_pass_purged_walk_forward.csv","four_pass_sell_path_replay.csv","trade_frequency_estimate.csv","calibration.csv","adaptive_thresholds.csv"}
RISK_FILES = {"risk_live_gate.csv","risk_ev_confidence.csv","risk_monte_carlo_summary.csv","risk_context_performance.csv"}
QUANT_FILES = {"feature_outcome_correlation.csv","feature_correlation_matrix.csv","markov_regime_policy.csv","markov_regime_transitions.csv","kalman_filter_policy.csv","kalman_live_state.csv","quant_state_summary.csv"}
LIVE_FILES = {"trades.csv","orders.csv","live_trade_blockers.csv","approved_but_shadowed.csv","position_targets.csv","account_balance_diagnostics.csv","reconciliation.csv","council_decisions.csv","council_votes.csv","council_observation_outcomes.csv","decision_audit.csv","signal_events.csv","ai_predictions.csv","ai_feature_importance.csv","product_cooldowns.csv","agent_performance.csv","agent_leaderboard.csv","agent_adjustments.csv","agent_component_replay_attribution.csv","agent_side_ratings.csv","agent_trade_policy.csv"}
RUNTIME_FILES = {"calculation_status.json","calculation_complete_latch.json","viewer_snapshot.json","historical_replay_manifest.json","startup_runtime_inventory.csv","post_patch_audit.csv","exchange_product_map.csv","products_active.csv","products_selected.json"}
RESEARCH_FILES = {
    "continuous_research_status.json",
    "continuous_research_history.csv",
    "market_state_analog_matches.csv",
    "market_state_analog_summary.csv",
    "cross_asset_analog_matches.csv",
    "cross_asset_analog_summary.csv",
    "sell_model_ratio_grid.csv",
    "cross_asset_sell_model_ratio_grid.csv",
    "adaptive_sell_model_policy.csv",
    "adaptive_decision_policy.csv",
    "cross_asset_adaptive_decision_policy.csv",
    "incremental_strategy_simulation_summary.csv",
    "background_replay_expansion_summary.csv",
    "research_file_health.csv",
    "research_backfill_plan.csv",
}

PARQUET_ACCELERATED_FILES = set()
for _bucket in [RAW_CACHE_FILES, REPLAY_FILES, BACKTEST_FILES, RISK_FILES, QUANT_FILES, RESEARCH_FILES]:
    PARQUET_ACCELERATED_FILES.update(_bucket)

CSV_APPEND_ONLY_FILES = {
    "trades.csv", "orders.csv", "live_trade_blockers.csv", "approved_but_shadowed.csv",
    "position_targets.csv", "account_balance_diagnostics.csv", "reconciliation.csv",
    "signal_events.csv", "decision_audit.csv", "council_votes.csv", "council_decisions.csv",
}

def parquet_runtime_path(filename: str) -> str:
    path = runtime_path(filename)
    base, _ = os.path.splitext(path)
    return base + ".parquet"

def is_parquet_accelerated(filename: str) -> bool:
    return os.path.basename(str(filename)) in PARQUET_ACCELERATED_FILES

def is_csv_append_only(filename: str) -> bool:
    return os.path.basename(str(filename)) in CSV_APPEND_ONLY_FILES

def ensure_runtime_dirs() -> None:
    for path in [CSV_ROOT_DIR, RAW_CACHE_DIR, REPLAY_DIR, BACKTEST_DIR, RISK_DIR, QUANT_DIR, LIVE_DIR, RUNTIME_DIR, VIEWER_DIR, RESEARCH_DIR, ARCHIVE_DIR, DEBUG_DIR]:
        os.makedirs(path, exist_ok=True)

def runtime_dir_for_filename(filename: str) -> str:
    name = os.path.basename(str(filename))
    if name.endswith(".meta.json"):
        return runtime_dir_for_filename(name[:-10])
    for files, directory in [(RAW_CACHE_FILES, RAW_CACHE_DIR),(REPLAY_FILES, REPLAY_DIR),(BACKTEST_FILES, BACKTEST_DIR),(RISK_FILES, RISK_DIR),(QUANT_FILES, QUANT_DIR),(LIVE_FILES, LIVE_DIR),(RUNTIME_FILES, RUNTIME_DIR),(RESEARCH_FILES, RESEARCH_DIR)]:
        if name in files:
            return directory
    if name.endswith(".csv"):
        return CSV_ROOT_DIR
    if name.endswith(".json"):
        return RUNTIME_DIR
    if name.endswith(".log"):
        return DEBUG_DIR
    return BASE_DIR

def runtime_path(filename: str) -> str:
    ensure_runtime_dirs()
    return os.path.join(runtime_dir_for_filename(filename), os.path.basename(str(filename)))

def sidecar_meta_path(path: str) -> str:
    return f"{path}.meta.json"

def runtime_meta_path(path_or_filename: str) -> str:
    if os.path.isabs(path_or_filename):
        return sidecar_meta_path(path_or_filename)
    return runtime_path(f"{os.path.basename(path_or_filename)}.meta.json")

def migrate_root_runtime_files_to_csv_tree() -> Dict[str, object]:
    ensure_runtime_dirs()
    moved, skipped, errors = [], [], []
    known_files = set()
    for bucket in [RAW_CACHE_FILES, REPLAY_FILES, BACKTEST_FILES, RISK_FILES, QUANT_FILES, LIVE_FILES, RUNTIME_FILES, RESEARCH_FILES]:
        known_files.update(bucket)
        known_files.update({f"{name}.meta.json" for name in bucket})
    for filename in sorted(known_files):
        src = os.path.join(BASE_DIR, filename)
        dst = runtime_path(filename)
        try:
            if not os.path.exists(src):
                continue
            if os.path.abspath(src) == os.path.abspath(dst):
                skipped.append({"filename": filename, "reason": "already_in_runtime_path"}); continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                if os.path.getsize(src) > os.path.getsize(dst):
                    shutil.move(dst, os.path.join(ARCHIVE_DIR, f"{filename}.{int(time.time())}.old")); shutil.move(src, dst); moved.append({"filename": filename, "action": "replaced_smaller_existing"})
                else:
                    shutil.move(src, os.path.join(ARCHIVE_DIR, f"{filename}.{int(time.time())}.root_duplicate")); skipped.append({"filename": filename, "reason": "runtime_path_already_has_equal_or_larger_file"})
            else:
                shutil.move(src, dst); moved.append({"filename": filename, "action": "moved_to_runtime_tree"})
        except Exception as exc:
            errors.append({"filename": filename, "error": str(exc)})
    return {"moved": moved, "skipped": skipped, "errors": errors, "csv_root": CSV_ROOT_DIR}

def generated_file_version() -> str:
    return "persistent_csv_data_lake_v1_2026_06_21"

def write_generated_file_meta(path: str, reason: str = "") -> None:
    try:
        with open(sidecar_meta_path(path), "w", encoding="utf-8") as file:
            json.dump({"generation_version": generated_file_version(), "generated_at_ts": time.time(), "generated_at_iso": datetime.now(timezone.utc).isoformat(), "reason": str(reason or "")}, file, indent=2, sort_keys=True)
    except Exception:
        pass
