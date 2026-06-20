from dataclasses import dataclass
from typing import Any, Dict, Optional
@dataclass
class ExchangeOrderResult:
    ok: bool; exchange: str; product_id: str; side: str; requested_quote_usd: float=0.0; filled_qty: float=0.0; avg_price: float=0.0; fee_usd: float=0.0; order_id: str=""; status: str=""; raw: Optional[Dict[str, Any]]=None; error: str=""
class BaseExchangeAdapter:
    exchange_id="base"
    def get_account_snapshot(self)->Dict[str,Any]: raise NotImplementedError
    def get_top_of_book(self, product_id: str)->Dict[str,Any]: raise NotImplementedError
    def place_market_buy(self, product_id: str, quote_usd: float)->ExchangeOrderResult: raise NotImplementedError
    def place_limit_buy(self, product_id: str, quote_usd: float, limit_price: float)->ExchangeOrderResult: raise NotImplementedError
    def place_market_sell(self, product_id: str, base_qty: float)->ExchangeOrderResult: raise NotImplementedError
    def place_limit_sell(self, product_id: str, base_qty: float, limit_price: float)->ExchangeOrderResult: raise NotImplementedError
    def cancel_order(self, order_id: str, product_id: Optional[str]=None, symbol: Optional[str]=None)->bool: raise NotImplementedError
import os, time, uuid
from binance_us_client import BinanceUSClient
from binance_symbol_filters import parse_symbol_rules, quantity_from_quote, format_price, format_quantity, order_meets_minimums
from exchange_catalog import product_to_execution_symbol, execution_symbol_to_product, EXCHANGE_BINANCE_US
class BinanceUSAdapter(BaseExchangeAdapter):
    exchange_id=EXCHANGE_BINANCE_US
    def __init__(self, *, dry_run: Optional[bool]=None, allow_real_orders: Optional[bool]=None):
        self.client=BinanceUSClient(); self.client.sync_time(); self.dry_run=(str(os.getenv("BINANCE_US_DRY_RUN","true")).strip().lower() in {"1","true","yes","on"} if dry_run is None else bool(dry_run)); self.allow_real_orders=(str(os.getenv("BINANCE_US_ALLOW_REAL_ORDERS","false")).strip().lower() in {"1","true","yes","on"} if allow_real_orders is None else bool(allow_real_orders)); self.symbol_rules_cache={}; self.fee_cache={}
    def product_to_symbol(self, product_id):
        symbol=product_to_execution_symbol(product_id, exchange=EXCHANGE_BINANCE_US)
        if not symbol: raise RuntimeError(f"No Binance.US symbol mapping for {product_id}")
        return symbol
    def symbol_to_product(self, symbol): return execution_symbol_to_product(symbol, exchange=EXCHANGE_BINANCE_US) or symbol
    def symbol_rules(self, symbol):
        symbol=str(symbol).upper()
        if symbol not in self.symbol_rules_cache:
            info=self.client.exchange_info(symbol=symbol); symbols=info.get("symbols", [])
            if not symbols: raise RuntimeError(f"No exchangeInfo returned for {symbol}")
            self.symbol_rules_cache[symbol]=parse_symbol_rules(symbols[0])
        return self.symbol_rules_cache[symbol]
    def get_account_snapshot(self): return self.client.account()
    def balances_by_asset(self):
        out={}
        for b in self.client.account().get("balances", []):
            asset=str(b.get("asset") or ""); free=float(b.get("free") or 0.0); locked=float(b.get("locked") or 0.0)
            if asset: out[asset]={"free":free,"locked":locked,"total":free+locked}
        return out
    def get_available_asset(self, asset): return float(self.balances_by_asset().get(str(asset).upper(), {}).get("free", 0.0))
    def get_total_asset(self, asset): return float(self.balances_by_asset().get(str(asset).upper(), {}).get("total", 0.0))
    def get_top_of_book(self, product_id):
        symbol=self.product_to_symbol(product_id); raw=self.client.book_ticker(symbol=symbol); bid=float(raw.get("bidPrice") or 0.0); ask=float(raw.get("askPrice") or 0.0); mid=(bid+ask)/2 if bid>0 and ask>0 else 0.0
        return {"exchange":self.exchange_id,"product_id":product_id,"symbol":symbol,"bid":bid,"ask":ask,"mid":mid,"spread_bps":((ask-bid)/mid*10000 if mid>0 else 0.0),"bid_qty":float(raw.get("bidQty") or 0.0),"ask_qty":float(raw.get("askQty") or 0.0),"ts":time.time(),"raw":raw}
    def get_order_book_snapshot(self, product_id: str, limit: int = 25):
        symbol=self.product_to_symbol(product_id); raw=self.client.depth(symbol=symbol, limit=int(limit))
        bids=[[float(price), float(qty)] for price, qty in raw.get("bids", []) if float(price)>0 and float(qty)>0]
        asks=[[float(price), float(qty)] for price, qty in raw.get("asks", []) if float(price)>0 and float(qty)>0]
        bid_notional=sum(price*qty for price, qty in bids); ask_notional=sum(price*qty for price, qty in asks); denom=bid_notional+ask_notional
        imbalance=((bid_notional-ask_notional)/denom) if denom>0 else 0.0
        return {"exchange":self.exchange_id,"product_id":product_id,"symbol":symbol,"bids":bids,"asks":asks,"bid_notional":float(bid_notional),"ask_notional":float(ask_notional),"imbalance":float(imbalance),"ts":time.time(),"raw":raw}
    def get_order(self, product_id: str, order_id: str):
        symbol=self.product_to_symbol(product_id); return self.client.get_order(symbol=symbol, order_id=str(order_id))
    def cancel_product_order(self, product_id: str, order_id: str) -> bool:
        symbol=self.product_to_symbol(product_id); raw=self.client.cancel_order(symbol=symbol, order_id=str(order_id)); return bool(raw)
    def fee_bps_for_symbol(self, symbol):
        symbol=str(symbol).upper()
        if symbol not in self.fee_cache:
            try:
                row=self.client.trading_fee(symbol=symbol); row=row[0] if isinstance(row, list) else row; maker=float(row.get("makerCommission") or 0.0)*10000; taker=float(row.get("takerCommission") or 0.0)*10000
                self.fee_cache[symbol]={"maker_bps":maker,"taker_bps":taker,"source":"api"}
            except Exception as exc:
                maker=float(os.getenv("BINANCE_US_FALLBACK_MAKER_FEE_BPS","0.0")); taker=float(os.getenv("BINANCE_US_FALLBACK_TAKER_FEE_BPS","2.0"))
                self.fee_cache[symbol]={"maker_bps":maker,"taker_bps":taker,"source":"fallback","error":str(exc)}
        return self.fee_cache[symbol]
    def _result_from_order(self, *, product_id, side, requested_quote_usd=0.0, raw):
        fills=raw.get("fills") or []; qty=float(raw.get("executedQty") or 0.0); quote=float(raw.get("cummulativeQuoteQty") or raw.get("cumQuote") or 0.0); fee=0.0; fee_assets=[]
        for f in fills:
            commission=float(f.get("commission") or 0.0); asset=str(f.get("commissionAsset") or "")
            fee_assets.append({"asset":asset,"amount":commission})
            if asset in {"USDT","USD","USDC"}: fee += commission
        raw["fee_assets"] = fee_assets
        return ExchangeOrderResult(str(raw.get("status","")).upper() in {"FILLED","PARTIALLY_FILLED","NEW"}, self.exchange_id, product_id, side, float(requested_quote_usd), qty, quote/qty if qty>0 else 0.0, fee, str(raw.get("orderId") or ""), str(raw.get("status") or ""), raw, "")
    def place_market_buy(self, product_id, quote_usd):
        symbol=self.product_to_symbol(product_id); rules=self.symbol_rules(symbol); tob=self.get_top_of_book(product_id)
        if float(quote_usd) < float(rules.min_notional): raise RuntimeError(f"Order below Binance.US minNotional for {symbol}: quote={quote_usd} min={rules.min_notional}")
        params={"symbol":symbol,"side":"BUY","type":"MARKET","quoteOrderQty":str(round(float(quote_usd),2)),"newOrderRespType":"FULL","newClientOrderId":f"bot-buy-{uuid.uuid4().hex[:20]}"}
        if self.dry_run or not self.allow_real_orders: return ExchangeOrderResult(True,self.exchange_id,product_id,"BUY",float(quote_usd),status="DRY_RUN_TEST_OK",raw={"test_order":self.client.test_order(**params),"params":params,"top_of_book":tob})
        return self._result_from_order(product_id=product_id, side="BUY", requested_quote_usd=float(quote_usd), raw=self.client.new_order(**params))
    def place_market_sell(self, product_id, base_qty):
        symbol=self.product_to_symbol(product_id); rules=self.symbol_rules(symbol); qty=format_quantity(base_qty, rules, market=True); tob=self.get_top_of_book(product_id); price=float(tob.get("bid",0.0) or 0.0)
        if not order_meets_minimums(quote_usd=float(qty)*price, qty=qty, price=price, rules=rules): raise RuntimeError(f"Market sell does not meet Binance.US filters symbol={symbol} qty={qty} price={price}")
        params={"symbol":symbol,"side":"SELL","type":"MARKET","quantity":qty,"newOrderRespType":"FULL","newClientOrderId":f"bot-sell-{uuid.uuid4().hex[:20]}"}
        if self.dry_run or not self.allow_real_orders: return ExchangeOrderResult(True,self.exchange_id,product_id,"SELL",status="DRY_RUN_TEST_OK",raw={"test_order":self.client.test_order(**params),"params":params,"top_of_book":tob})
        return self._result_from_order(product_id=product_id, side="SELL", raw=self.client.new_order(**params))
    def place_limit_buy(self, product_id, quote_usd, limit_price):
        symbol=self.product_to_symbol(product_id); rules=self.symbol_rules(symbol); price=format_price(limit_price, rules); qty=quantity_from_quote(float(quote_usd), float(limit_price), rules)
        if not order_meets_minimums(quote_usd=float(quote_usd), qty=qty, price=float(limit_price), rules=rules): raise RuntimeError(f"Limit buy does not meet Binance.US filters symbol={symbol} qty={qty} price={price}")
        params={"symbol":symbol,"side":"BUY","type":"LIMIT_MAKER","quantity":qty,"price":price,"newOrderRespType":"FULL","newClientOrderId":f"bot-lbuy-{uuid.uuid4().hex[:20]}"}
        if self.dry_run or not self.allow_real_orders: return ExchangeOrderResult(True,self.exchange_id,product_id,"BUY",float(quote_usd),status="DRY_RUN_TEST_OK",raw={"test_order":self.client.test_order(**params),"params":params})
        return self._result_from_order(product_id=product_id, side="BUY", requested_quote_usd=float(quote_usd), raw=self.client.new_order(**params))
    def place_limit_sell(self, product_id, base_qty, limit_price):
        symbol=self.product_to_symbol(product_id); rules=self.symbol_rules(symbol); params={"symbol":symbol,"side":"SELL","type":"LIMIT_MAKER","quantity":format_quantity(base_qty, rules),"price":format_price(limit_price, rules),"newOrderRespType":"FULL","newClientOrderId":f"bot-lsell-{uuid.uuid4().hex[:20]}"}
        if self.dry_run or not self.allow_real_orders: return ExchangeOrderResult(True,self.exchange_id,product_id,"SELL",status="DRY_RUN_TEST_OK",raw={"test_order":self.client.test_order(**params),"params":params})
        return self._result_from_order(product_id=product_id, side="SELL", raw=self.client.new_order(**params))
    def cancel_order(self, order_id, product_id=None, symbol=None):
        if not symbol:
            if not product_id: raise RuntimeError("cancel_order requires product_id or symbol")
            symbol=self.product_to_symbol(product_id)
        return bool(self.client.cancel_order(symbol=symbol, order_id=str(order_id)))

# Binance.US order status helper
def _binance_adapter_get_order(self, product_id: str, order_id: str) -> Dict[str, Any]:
    symbol = self.product_to_symbol(product_id)
    return self.client.get_order(symbol=symbol, order_id=str(order_id))
BinanceUSAdapter.get_order = _binance_adapter_get_order
