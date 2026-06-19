"""Historical replay worker process scaffolding."""

import csv
import os
import time
from typing import Any, Dict


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
    return {"ok": True, "product_id": product_id, "timeframe": timeframe, "output_path": path, "rows": rows, "ts": time.time()}
