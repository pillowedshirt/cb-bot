import csv
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import requests

from exchange_catalog import coinbase_to_binance_symbol

BINANCE_BULK_BASE_URL = "https://data.binance.vision/data/spot"
BINANCE_BULK_CACHE_DIR_NAME = "binance_bulk_cache"
TZ_NAME = "America/Phoenix"
TZ = ZoneInfo(TZ_NAME)

@dataclass
class NormalizedCandle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    source_exchange: str
    source_symbol: str
    source_interval: str

def _month_iter(start_ts: int, end_ts: int) -> List[Tuple[int, int]]:
    start_dt = datetime.fromtimestamp(int(start_ts), tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_dt = datetime.fromtimestamp(int(end_ts), tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        out.append((y, m))
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
    return out

def _binance_interval_for_timeframe(timeframe: str) -> str:
    tf = str(timeframe or "").lower().strip()
    if tf == "primary_15m_90d": return "15m"
    if tf == "regime_1h_365d": return "1h"
    if tf == "daily_1d_2y": return "1d"
    raise ValueError(f"Unsupported Binance historical timeframe: {timeframe}")

def _safe_ts_from_binance(raw_value: str) -> int:
    value = int(float(str(raw_value).strip()))
    if value > 10**14: return int(value / 1_000_000)
    if value > 10**11: return int(value / 1_000)
    return int(value)

def _dt_mst(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")

class BinanceBulkHistoricalProvider:
    """Public historical Binance spot kline provider for replay/backlog only."""
    def __init__(self, *, base_dir: str, timeout_sec: float = 30.0, prefer_binance_us: bool = False):
        self.base_dir = base_dir
        self.timeout_sec = float(timeout_sec)
        self.prefer_binance_us = bool(prefer_binance_us)
        self.cache_dir = os.path.join(base_dir, BINANCE_BULK_CACHE_DIR_NAME)
        os.makedirs(self.cache_dir, exist_ok=True)

    def binance_symbol_for_product(self, product_id: str) -> Optional[str]:
        return coinbase_to_binance_symbol(product_id, prefer_us=self.prefer_binance_us)

    def monthly_zip_url(self, *, symbol: str, interval: str, year: int, month: int) -> str:
        ym = f"{int(year):04d}-{int(month):02d}"
        filename = f"{symbol}-{interval}-{ym}.zip"
        return f"{BINANCE_BULK_BASE_URL}/monthly/klines/{symbol}/{interval}/{filename}"

    def local_zip_path(self, *, symbol: str, interval: str, year: int, month: int) -> str:
        folder = os.path.join(self.cache_dir, "spot", "monthly", "klines", symbol, interval)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{symbol}-{interval}-{int(year):04d}-{int(month):02d}.zip")

    def daily_zip_url(self, *, symbol: str, interval: str, year: int, month: int, day: int) -> str:
        ymd = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        filename = f"{symbol}-{interval}-{ymd}.zip"
        return f"{BINANCE_BULK_BASE_URL}/daily/klines/{symbol}/{interval}/{filename}"

    def local_daily_zip_path(self, *, symbol: str, interval: str, year: int, month: int, day: int) -> str:
        folder = os.path.join(self.cache_dir, "spot", "daily", "klines", symbol, interval)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{symbol}-{interval}-{int(year):04d}-{int(month):02d}-{int(day):02d}.zip")

    def _day_iter(self, start_ts: int, end_ts: int) -> List[Tuple[int, int, int]]:
        start_dt = datetime.fromtimestamp(int(start_ts), tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = datetime.fromtimestamp(int(end_ts), tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        days = []
        cur = start_dt
        while cur <= end_dt:
            days.append((cur.year, cur.month, cur.day))
            cur = cur + timedelta(days=1)
        return days

    def _download_file(self, *, url: str, local_path: str) -> bool:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return True
        tmp_path = local_path + ".tmp"
        try:
            response = requests.get(url, timeout=self.timeout_sec, stream=True)
            if response.status_code == 404:
                return False
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk: f.write(chunk)
            if os.path.getsize(tmp_path) <= 0:
                try: os.remove(tmp_path)
                except Exception: pass
                return False
            os.replace(tmp_path, local_path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp_path): os.remove(tmp_path)
            except Exception:
                pass
            return False

    def _read_monthly_zip(self, *, zip_path: str, product_id: str, symbol: str, interval: str, start_ts: int, end_ts: int) -> List[NormalizedCandle]:
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) <= 0:
            return []
        candles: List[NormalizedCandle] = []
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csv_names: return []
                with z.open(csv_names[0], "r") as raw:
                    reader = csv.reader(raw.read().decode("utf-8", errors="replace").splitlines())
                    for row in reader:
                        if not row or len(row) < 6: continue
                        if str(row[0]).strip().lower() in {"open_time", "open time"}: continue
                        try:
                            ts = _safe_ts_from_binance(row[0])
                            if ts < int(start_ts) or ts > int(end_ts): continue
                            candles.append(NormalizedCandle(ts=ts, open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]), volume=float(row[5]), source="binance_bulk", source_exchange="binance", source_symbol=symbol, source_interval=interval))
                        except Exception:
                            continue
            return candles
        except Exception:
            return []

    def _read_daily_zip(self, *, zip_path: str, product_id: str, symbol: str, interval: str, start_ts: int, end_ts: int) -> List[NormalizedCandle]:
        return self._read_monthly_zip(zip_path=zip_path, product_id=product_id, symbol=symbol, interval=interval, start_ts=start_ts, end_ts=end_ts)

    def fetch_bulk_candles(self, *, product_id: str, timeframe: str, start_ts: int, end_ts: int) -> Tuple[List[NormalizedCandle], Dict[str, object]]:
        symbol = self.binance_symbol_for_product(product_id)
        if not symbol:
            return [], {"ok": False, "reason": "no_binance_symbol_mapping", "product_id": product_id, "timeframe": timeframe}
        interval = _binance_interval_for_timeframe(timeframe)
        candles_by_ts: Dict[int, NormalizedCandle] = {}
        attempted = downloaded = missing = 0
        for year, month in _month_iter(start_ts, end_ts):
            attempted += 1
            url = self.monthly_zip_url(symbol=symbol, interval=interval, year=year, month=month)
            local_path = self.local_zip_path(symbol=symbol, interval=interval, year=year, month=month)
            ok = self._download_file(url=url, local_path=local_path)
            if not ok:
                missing += 1; continue
            downloaded += 1
            for candle in self._read_monthly_zip(zip_path=local_path, product_id=product_id, symbol=symbol, interval=interval, start_ts=start_ts, end_ts=end_ts):
                candles_by_ts[int(candle.ts)] = candle
        daily_attempted = daily_downloaded = daily_missing = 0
        expected_interval_sec = {"15m": 15 * 60, "1h": 60 * 60, "1d": 24 * 60 * 60}.get(interval, 60 * 60)
        existing_ts = set(candles_by_ts.keys())
        for year, month, day in self._day_iter(start_ts, end_ts):
            day_start = int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())
            day_end = day_start + 86400 - 1
            if day_end < int(start_ts) or day_start > int(end_ts):
                continue
            expected_points = max(1, int(86400 / max(1, expected_interval_sec)))
            covered_points = sum(1 for ts in existing_ts if day_start <= int(ts) <= day_end)
            if covered_points >= max(1, int(expected_points * 0.80)):
                continue
            daily_attempted += 1
            url = self.daily_zip_url(symbol=symbol, interval=interval, year=year, month=month, day=day)
            local_path = self.local_daily_zip_path(symbol=symbol, interval=interval, year=year, month=month, day=day)
            ok = self._download_file(url=url, local_path=local_path)
            if not ok:
                daily_missing += 1
                continue
            daily_downloaded += 1
            for candle in self._read_daily_zip(zip_path=local_path, product_id=product_id, symbol=symbol, interval=interval, start_ts=start_ts, end_ts=end_ts):
                candles_by_ts[int(candle.ts)] = candle
                existing_ts.add(int(candle.ts))
        candles = sorted(candles_by_ts.values(), key=lambda c: int(c.ts))
        return candles, {"daily_files_attempted": daily_attempted, "daily_files_downloaded_or_cached": daily_downloaded, "daily_files_missing": daily_missing, "ok": bool(candles), "source": "binance_bulk", "source_exchange": "binance", "product_id": product_id, "symbol": symbol, "timeframe": timeframe, "interval": interval, "start_ts": int(start_ts), "end_ts": int(end_ts), "months_attempted": attempted, "monthly_files_downloaded_or_cached": downloaded, "monthly_files_missing": missing, "candles": len(candles)}

def write_normalized_candles_to_bot_cache(*, path: str, product_id: str, candles: List[NormalizedCandle], min_ts: int) -> int:
    if not candles: return 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    columns = ["ts", "dt_mst", "product_id", "open", "high", "low", "close", "volume", "historical_source", "source_exchange", "source_symbol", "source_interval"]
    if file_exists:
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                existing_header = next(csv.reader(f), [])
            if existing_header and any(col not in existing_header for col in columns):
                existing_rows = []
                with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    for row in csv.DictReader(f):
                        existing_rows.append({col: row.get(col, "") for col in columns})
                tmp_path = path + ".source-columns.tmp"
                with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    for row in existing_rows:
                        writer.writerow(row)
                os.replace(tmp_path, path)
        except Exception:
            pass
    existing_ts = set()
    if file_exists:
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("product_id") or "") != str(product_id): continue
                    try:
                        ts = int(float(row.get("ts") or 0))
                        if ts >= int(min_ts): existing_ts.add(ts)
                    except Exception: continue
        except Exception:
            existing_ts = set()
    rows_written = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists: writer.writeheader()
        for candle in candles:
            ts = int(candle.ts)
            if ts < int(min_ts) or ts in existing_ts: continue
            writer.writerow({"ts": ts, "dt_mst": _dt_mst(ts), "product_id": product_id, "open": float(candle.open), "high": float(candle.high), "low": float(candle.low), "close": float(candle.close), "volume": float(candle.volume), "historical_source": candle.source, "source_exchange": candle.source_exchange, "source_symbol": candle.source_symbol, "source_interval": candle.source_interval})
            existing_ts.add(ts); rows_written += 1
    return rows_written
