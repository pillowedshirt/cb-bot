import hashlib, hmac, os, time, urllib.parse
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv
class BinanceUSClient:
    def __init__(self, *, api_key: Optional[str]=None, api_secret: Optional[str]=None, base_url: Optional[str]=None, recv_window: int=5000, timeout_sec: float=20.0):
        load_dotenv(); self.api_key=api_key or os.getenv("BINANCE_US_API_KEY", "").strip(); self.api_secret=api_secret or os.getenv("BINANCE_US_API_SECRET", "").strip(); self.base_url=(base_url or os.getenv("BINANCE_US_REST_BASE_URL", "https://api.binance.us")).rstrip("/"); self.recv_window=int(recv_window); self.timeout_sec=float(timeout_sec); self._time_offset_ms=0
        if not self.api_key: raise RuntimeError("Missing BINANCE_US_API_KEY in .env")
        if not self.api_secret: raise RuntimeError("Missing BINANCE_US_API_SECRET in .env")
    def _headers(self): return {"X-MBX-APIKEY": self.api_key}
    def _timestamp_ms(self): return int(time.time()*1000)+int(self._time_offset_ms)
    def _sign(self, params): return hmac.new(self.api_secret.encode(), urllib.parse.urlencode(params, doseq=True).encode(), hashlib.sha256).hexdigest()
    def _request(self, method, path, *, params=None, signed=False):
        params=dict(params or {})
        if signed: params["timestamp"]=self._timestamp_ms(); params.setdefault("recvWindow", self.recv_window); params["signature"]=self._sign(params)
        url=self.base_url+path; method=method.upper(); headers=self._headers() if signed or method in {"POST","DELETE","PUT"} else None
        resp = requests.request(method, url, params=params if method in {"GET","PUT"} else None, data=params if method in {"POST","DELETE"} else None, headers=headers, timeout=self.timeout_sec)
        if resp.status_code >= 400: raise RuntimeError(f"Binance.US {method} {path} failed status={resp.status_code} body={resp.text[:1000]}")
        return {} if not resp.text else resp.json()
    def sync_time(self):
        data=self._request("GET","/api/v3/time"); server_ms=int(data.get("serverTime")); local_ms=int(time.time()*1000); self._time_offset_ms=server_ms-local_ms; return {"serverTime":server_ms,"localTime":local_ms,"offsetMs":self._time_offset_ms}
    def ping(self): return self._request("GET","/api/v3/ping")
    def exchange_info(self, symbol=None): return self._request("GET","/api/v3/exchangeInfo", params={"symbol": symbol} if symbol else {})
    def account(self): return self._request("GET","/api/v3/account", signed=True)
    def trading_fee(self, symbol=None): return self._request("GET","/sapi/v1/asset/query/trading-fee", params={"symbol": symbol} if symbol else {}, signed=True)
    def book_ticker(self, symbol=None): return self._request("GET","/api/v3/ticker/bookTicker", params={"symbol": symbol} if symbol else {})
    def klines(self, symbol, interval, start_ms=None, end_ms=None, limit=1000):
        p={"symbol":symbol,"interval":interval,"limit":int(limit)}; 
        if start_ms is not None: p["startTime"]=int(start_ms)
        if end_ms is not None: p["endTime"]=int(end_ms)
        return self._request("GET","/api/v3/klines", params=p)
    def new_order(self, **params): return self._request("POST","/api/v3/order", params=params, signed=True)
    def test_order(self, **params): return self._request("POST","/api/v3/order/test", params=params, signed=True)
    def get_order(self, symbol, order_id=None, client_order_id=None):
        p={"symbol":symbol};
        if order_id: p["orderId"]=order_id
        if client_order_id: p["origClientOrderId"]=client_order_id
        return self._request("GET","/api/v3/order", params=p, signed=True)
    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        p={"symbol":symbol};
        if order_id: p["orderId"]=order_id
        if client_order_id: p["origClientOrderId"]=client_order_id
        return self._request("DELETE","/api/v3/order", params=p, signed=True)
    def open_orders(self, symbol=None): return self._request("GET","/api/v3/openOrders", params={"symbol":symbol} if symbol else {}, signed=True)
    def my_trades(self, symbol, limit=500): return self._request("GET","/api/v3/myTrades", params={"symbol":symbol,"limit":int(limit)}, signed=True)
    def create_listen_key(self):
        data=self._request("POST","/api/v3/userDataStream", params={}, signed=False); lk=data.get("listenKey")
        if not lk: raise RuntimeError(f"Binance.US did not return listenKey: {data}")
        return str(lk)
    def keepalive_listen_key(self, listen_key): return self._request("PUT","/api/v3/userDataStream", params={"listenKey": listen_key}, signed=False)
    def close_listen_key(self, listen_key): return self._request("DELETE","/api/v3/userDataStream", params={"listenKey": listen_key}, signed=False)
