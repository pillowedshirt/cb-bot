import os
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
except Exception:
    pa = None
    pq = None

try:
    import polars as pl
except Exception:
    pl = None

PARQUET_COMPRESSION = os.getenv("PARQUET_COMPRESSION", "zstd")


def parquet_path_for(csv_path: str) -> str:
    base, _ = os.path.splitext(str(csv_path))
    return base + ".parquet"


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def _exists_nonempty(path: str) -> bool:
    try:
        return bool(path and os.path.exists(path) and os.path.getsize(path) > 0)
    except Exception:
        return False


def read_table(csv_path: str, *, prefer_parquet: bool = True, columns: Optional[List[str]] = None, on_bad_lines: str = "skip") -> pd.DataFrame:
    csv_path = str(csv_path)
    pq_path = parquet_path_for(csv_path)
    if prefer_parquet and _exists_nonempty(pq_path) and (not _exists_nonempty(csv_path) or _mtime(pq_path) >= _mtime(csv_path)):
        try:
            return pd.read_parquet(pq_path, columns=columns)
        except Exception:
            pass
    if not _exists_nonempty(csv_path):
        if _exists_nonempty(pq_path):
            try:
                return pd.read_parquet(pq_path, columns=columns)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
    try:
        try:
            frame = pd.read_csv(csv_path, on_bad_lines=on_bad_lines, engine="python", usecols=columns)
        except TypeError:
            frame = pd.read_csv(csv_path, on_bad_lines=on_bad_lines, engine="python")
            if columns:
                frame = frame[[c for c in columns if c in frame.columns]]
    except Exception:
        return pd.DataFrame()
    try:
        write_table(csv_path, frame, write_csv=False, write_parquet=True)
    except Exception:
        pass
    return frame


def scan_table(csv_path: str):
    if pl is None:
        return None
    pq_path = parquet_path_for(csv_path)
    try:
        if _exists_nonempty(pq_path) and (not _exists_nonempty(csv_path) or _mtime(pq_path) >= _mtime(csv_path)):
            return pl.scan_parquet(pq_path)
    except Exception:
        pass
    try:
        frame = read_table(csv_path)
        if frame.empty:
            return None
        return pl.from_pandas(frame).lazy()
    except Exception:
        return None


def write_table(
    csv_path: str,
    frame: pd.DataFrame,
    *,
    write_csv: bool = True,
    write_parquet: bool = True,
    compression: Optional[str] = None,
) -> None:
    """
    Atomic table writer.

    Large research/backtest files should write Parquet sidecars when possible.
    If Parquet is unavailable, never break the bot; fall back to CSV.
    """
    csv_path = str(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    if frame is None:
        frame = pd.DataFrame()

    parquet_written = False

    if write_parquet and pa is not None:
        pq_path = parquet_path_for(csv_path)
        tmp_pq = pq_path + ".tmp"

        try:
            frame.to_parquet(
                tmp_pq,
                index=False,
                compression=compression or PARQUET_COMPRESSION,
            )
            os.replace(tmp_pq, pq_path)
            parquet_written = True
        except Exception:
            try:
                if os.path.exists(tmp_pq):
                    os.remove(tmp_pq)
            except Exception:
                pass

            # Retry with snappy because it is usually the safest pyarrow compression.
            try:
                frame.to_parquet(
                    tmp_pq,
                    index=False,
                    compression="snappy",
                )
                os.replace(tmp_pq, pq_path)
                parquet_written = True
            except Exception:
                try:
                    if os.path.exists(tmp_pq):
                        os.remove(tmp_pq)
                except Exception:
                    pass
                parquet_written = False

    if write_csv or not parquet_written:
        tmp_csv = csv_path + ".tmp"
        frame.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, csv_path)


def write_rows_table(csv_path: str, columns: List[str], rows: List[List[Any]], *, write_csv: bool = True, write_parquet: bool = True) -> None:
    write_table(csv_path, pd.DataFrame(rows, columns=columns), write_csv=write_csv, write_parquet=write_parquet)


def append_csv_and_refresh_parquet_periodically(csv_path: str, row: Dict[str, Any], columns: List[str], *, refresh_every_rows: int = 250) -> None:
    import csv
    csv_path = str(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    existed = _exists_nonempty(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not existed:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as file:
            line_count = sum(1 for _ in file)
        if line_count > 1 and (line_count - 1) % int(refresh_every_rows) == 0:
            write_table(csv_path, pd.read_csv(csv_path, on_bad_lines="skip", engine="python"), write_csv=False, write_parquet=True)
    except Exception:
        pass


def ensure_parquet_sidecar(csv_path: str) -> bool:
    try:
        csv_path = str(csv_path)
        pq_path = parquet_path_for(csv_path)
        if not _exists_nonempty(csv_path):
            return False
        if _exists_nonempty(pq_path) and _mtime(pq_path) >= _mtime(csv_path):
            return True
        write_table(csv_path, pd.read_csv(csv_path, on_bad_lines="skip", engine="python"), write_csv=False, write_parquet=True)
        return True
    except Exception:
        return False


def bulk_ensure_parquet_sidecars(paths: Iterable[str]) -> Dict[str, Any]:
    ok = failed = 0
    files = []
    started = time.time()
    for path in paths:
        success = ensure_parquet_sidecar(str(path))
        ok += int(success)
        failed += int(not success)
        files.append({"path": str(path), "ok": bool(success)})
    return {"ok": ok, "failed": failed, "files": files, "elapsed_sec": round(time.time() - started, 3)}
