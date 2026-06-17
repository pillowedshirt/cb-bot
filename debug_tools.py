from __future__ import annotations

import json
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

BASE_DIR = Path(__file__).resolve().parent
DEBUG_DIR = BASE_DIR / "debug"
OVERALL_DEBUG_PATH = BASE_DIR / "debug.log"
MAX_DEBUG_BYTES = 5_000_000
_LOCK = threading.RLock()
_THROTTLE: Dict[str, float] = {}


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _safe_name(module_name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(module_name or "unknown"))
    return out.strip("._") or "unknown"


def _ensure_debug_dir() -> None:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _rotate_if_needed(path: Path) -> None:
    try:
        if not path.exists():
            return
        if path.stat().st_size <= MAX_DEBUG_BYTES:
            return
        old_path = path.with_suffix(path.suffix + ".old")
        try:
            if old_path.exists():
                old_path.unlink()
        except Exception:
            pass
        path.replace(old_path)
    except Exception:
        pass


def _json_safe(value: Any, max_chars: int = 5000) -> str:
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = repr(value)
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def module_log_path(module_name: str) -> Path:
    _ensure_debug_dir()
    return DEBUG_DIR / f"{_safe_name(module_name)}.debug.log"


def write_debug(
    module_name: str,
    message: str,
    *,
    level: str = "INFO",
    data: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
    also_overall: bool = True,
) -> None:
    """
    Safe debug writer. Writes:
    - debug/<module>.debug.log
    - debug.log if also_overall=True
    This function never raises.
    """
    try:
        _ensure_debug_dir()
        module_name = _safe_name(module_name)
        level = str(level or "INFO").upper()
        line = f"{_timestamp()} [{level}] [{module_name}] {message}"
        if data:
            line += " | data=" + _json_safe(data)
        if exc is not None:
            line += (
                "\n"
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
            )
        with _LOCK:
            module_path = module_log_path(module_name)
            _rotate_if_needed(module_path)
            with module_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
            if also_overall:
                _rotate_if_needed(OVERALL_DEBUG_PATH)
                with OVERALL_DEBUG_PATH.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
    except Exception:
        pass


def module_debug(
    module_name: str,
    message: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    level: str = "INFO",
    also_overall: bool = True,
) -> None:
    write_debug(module_name, message, level=level, data=data, also_overall=also_overall)


def module_exception(
    module_name: str,
    context: str,
    exc: BaseException,
    *,
    data: Optional[Dict[str, Any]] = None,
    also_overall: bool = True,
) -> None:
    write_debug(module_name, context, level="ERROR", data=data, exc=exc, also_overall=also_overall)


def debug_every(
    module_name: str,
    key: str,
    seconds: float,
    message: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    level: str = "INFO",
    also_overall: bool = True,
) -> None:
    """Throttled logging helper. Useful for viewer reruns and fast bot loops."""
    try:
        throttle_key = f"{_safe_name(module_name)}::{key}"
        now = time.time()
        last = float(_THROTTLE.get(throttle_key, 0.0))
        if now - last < float(seconds):
            return
        _THROTTLE[throttle_key] = now
        write_debug(module_name, message, level=level, data=data, also_overall=also_overall)
    except Exception:
        pass


@contextmanager
def debug_timer(
    module_name: str,
    name: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    also_overall: bool = False,
):
    start = time.time()
    try:
        yield
        elapsed = time.time() - start
        module_debug(
            module_name,
            f"{name} completed",
            data={**(data or {}), "elapsed_sec": round(elapsed, 4)},
            level="DEBUG",
            also_overall=also_overall,
        )
    except Exception as exc:
        elapsed = time.time() - start
        module_exception(module_name, f"{name} failed after {elapsed:.4f}s", exc, data=data, also_overall=True)
        raise


def initialize_all_module_debug_logs(base_dir: Optional[str | Path] = None) -> None:
    """
    Creates a debug log file for every .py file in the project folder.
    """
    try:
        folder = Path(base_dir) if base_dir else BASE_DIR
        for path in sorted(folder.glob("*.py")):
            module_debug(
                path.stem,
                "debug_file_initialized",
                data={"file": str(path), "size_bytes": path.stat().st_size if path.exists() else 0},
                level="DEBUG",
                also_overall=False,
            )
    except Exception:
        pass


def file_status(path: str | Path) -> Dict[str, Any]:
    try:
        p = Path(path)
        return {
            "path": str(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "mtime": p.stat().st_mtime if p.exists() else 0.0,
            "age_sec": max(0.0, time.time() - p.stat().st_mtime) if p.exists() else None,
        }
    except Exception as exc:
        return {"path": str(path), "exists": False, "error": f"{type(exc).__name__}: {exc}"}


def dataframe_debug_summary(
    frame: Any,
    *,
    required_columns: Optional[Iterable[str]] = None,
    name: str = "dataframe",
) -> Dict[str, Any]:
    """Lightweight pandas dataframe summary."""
    try:
        required = list(required_columns or [])
        columns = list(getattr(frame, "columns", []) or [])
        missing = [c for c in required if c not in columns]
        return {
            "name": name,
            "empty": bool(getattr(frame, "empty", True)),
            "rows": int(len(frame)) if hasattr(frame, "__len__") else 0,
            "columns": columns[:120],
            "column_count": len(columns),
            "required_columns": required,
            "missing_required_columns": missing,
        }
    except Exception as exc:
        return {"name": name, "error": f"{type(exc).__name__}: {exc}"}


def viewer_snapshot_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    try:
        coins = dict(snapshot.get("coins", {}) or {})
        updated_ts = float(snapshot.get("updated_ts", 0.0) or 0.0)
        required_coin_fields = [
            "product_id",
            "price",
            "truth_score",
            "final_buy_score",
            "expected_utility_bps",
            "buy_vs_wait_edge_bps",
            "volume_profile_leader_buy_score",
            "value_acceptance_state",
            "volume_node_state",
            "order_book_imbalance",
            "liquidity_risk_score",
        ]
        missing_by_coin = {}
        for product_id, row in coins.items():
            row = dict(row or {})
            missing = [field for field in required_coin_fields if field not in row]
            if missing:
                missing_by_coin[str(product_id)] = missing
        return {
            "updated_ts": updated_ts,
            "snapshot_age_sec": max(0.0, time.time() - updated_ts) if updated_ts > 0 else None,
            "coin_count": len(coins),
            "top_products": snapshot.get("top_products", []),
            "live_positions": snapshot.get("live_positions", []),
            "missing_required_fields_by_coin": missing_by_coin,
            "readiness": snapshot.get("readiness", {}),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def csv_debug_summary(
    path: str | Path,
    *,
    required_columns: Optional[Iterable[str]] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Reads only enough CSV data to summarize the file."""
    try:
        import pandas as pd

        status = file_status(path)
        if not status.get("exists"):
            return {
                **status,
                "name": name or Path(path).name,
                "rows": 0,
                "columns": [],
                "missing_required_columns": list(required_columns or []),
            }
        frame = pd.read_csv(path, nrows=250)
        return {
            **status,
            **dataframe_debug_summary(frame, required_columns=required_columns, name=name or Path(path).name),
        }
    except Exception as exc:
        return {**file_status(path), "name": name or Path(path).name, "error": f"{type(exc).__name__}: {exc}"}


def csv_runtime_status(path: str | Path, required_columns: Optional[Iterable[str]] = None, name: Optional[str] = None) -> Dict[str, Any]:
    status = csv_debug_summary(path, required_columns=required_columns, name=name)
    try:
        p = Path(path)
        status["age_sec"] = max(0.0, time.time() - p.stat().st_mtime) if p.exists() else None
    except Exception:
        status["age_sec"] = None
    return status


def normal_early_state(module_name: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
    payload = dict(data or {})
    payload["state"] = "normal_early_learning"
    write_debug(module_name, message, level="INFO", data=payload, also_overall=False)
