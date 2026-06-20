import asyncio, json, os, time
from typing import Any, Callable, Dict, List
import websockets
from exchange_catalog import EXCHANGE_BINANCE_US, execution_symbol_to_product, product_to_execution_symbol
class BinanceUSBookTickerStream:
    def __init__(self, products: List[str], on_book_ticker: Callable[[Dict[str, Any]], None]):
        self.products=list(products); self.on_book_ticker=on_book_ticker; self.base_url=os.getenv("BINANCE_US_WS_BASE_URL", "wss://stream.binance.us:9443").rstrip("/"); self._stop=False
    def url(self):
        streams=[f"{s.lower()}@bookTicker" for p in self.products for s in [product_to_execution_symbol(p, exchange=EXCHANGE_BINANCE_US)] if s]
        return f"{self.base_url}/stream?streams={'/'.join(streams)}"
    async def run_forever(self):
        while not self._stop:
            try:
                async with websockets.connect(self.url(), ping_interval=180, ping_timeout=600) as ws:
                    async for raw in ws:
                        msg=json.loads(raw); data=msg.get("data", msg); symbol=str(data.get("s") or data.get("symbol") or "").upper(); product_id=execution_symbol_to_product(symbol, exchange=EXCHANGE_BINANCE_US) or symbol
                        bid=float(data.get("b") or data.get("bidPrice") or 0.0); ask=float(data.get("a") or data.get("askPrice") or 0.0); bid_qty=float(data.get("B") or data.get("bidQty") or 0.0); ask_qty=float(data.get("A") or data.get("askQty") or 0.0); mid=(bid+ask)/2.0 if bid>0 and ask>0 else 0.0; spread=((ask-bid)/mid*10000.0) if mid>0 else 0.0
                        self.on_book_ticker({"exchange":EXCHANGE_BINANCE_US,"symbol":symbol,"product_id":product_id,"bid":bid,"ask":ask,"bid_qty":bid_qty,"ask_qty":ask_qty,"mid":mid,"spread_bps":spread,"ts":time.time(),"raw":data})
            except Exception:
                await asyncio.sleep(5.0)
    def stop(self): self._stop=True
