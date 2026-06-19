"""Historical replay worker process scaffolding."""

import csv
import os
import time
from typing import Any, Dict

from historical_replay_engine import run_replay_job_from_cache


def count_worker_output_rows(path: str) -> int:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def worker_health_check(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight process-pool health check."""
    return {"ok": True, "worker": "historical_replay_worker", "received": payload, "ts": time.time()}


def process_worker_output_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Worker-safe summary of a product/timeframe output file."""
    path = str(payload.get("output_path") or "")
    product_id = str(payload.get("product_id") or "")
    timeframe = str(payload.get("timeframe") or "")
    rows = count_worker_output_rows(path)
    summary = {"ok": True, "product_id": product_id, "timeframe": timeframe, "output_path": path, "rows": rows, "ts": time.time()}
    try:
        import pandas as pd

        if rows > 0:
            frame = pd.read_csv(path)
            for col in ["primary_net_pnl_bps", "comparison_net_pnl_bps", "comparison_net_improvement_bps"]:
                if col in frame.columns:
                    series = pd.to_numeric(frame[col], errors="coerce").dropna()
                    if not series.empty:
                        summary[f"{col}_avg"] = float(series.mean())
                        summary[f"{col}_median"] = float(series.median())
            if "comparison_would_have_won" in frame.columns:
                wins = pd.to_numeric(frame["comparison_would_have_won"], errors="coerce").fillna(0)
                summary["comparison_win_rate"] = float((wins > 0).mean())
            if "primary_would_have_won" in frame.columns:
                wins = pd.to_numeric(frame["primary_would_have_won"], errors="coerce").fillna(0)
                summary["primary_win_rate"] = float((wins > 0).mean())
    except Exception as exc:
        summary["summary_error"] = str(exc)
    return summary


def run_full_replay_worker_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """True CPU worker entrypoint for ProcessPoolExecutor replay jobs."""
    try:
        return run_replay_job_from_cache(payload)
    except Exception as exc:
        return {
            "ok": False,
            "worker_mode": "process",
            "product_id": str(payload.get("product_id") or ""),
            "timeframe": str(payload.get("timeframe") or ""),
            "output_path": str(payload.get("output_path") or ""),
            "error": str(exc),
            "ts": time.time(),
        }
