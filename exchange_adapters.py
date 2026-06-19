from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class ExchangeOrderResult:
    ok: bool
    exchange: str
    product_id: str
    side: str
    requested_quote_usd: float = 0.0
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fee_usd: float = 0.0
    order_id: str = ""
    status: str = ""
    raw: Optional[Dict[str, Any]] = None
    error: str = ""

class BaseExchangeAdapter:
    """Future live-execution abstraction."""
    exchange_id = "base"
    def get_account_snapshot(self) -> Dict[str, Any]: raise NotImplementedError
    def get_top_of_book(self, product_id: str) -> Dict[str, Any]: raise NotImplementedError
    def place_market_buy(self, product_id: str, quote_usd: float) -> ExchangeOrderResult: raise NotImplementedError
    def place_limit_buy(self, product_id: str, quote_usd: float, limit_price: float) -> ExchangeOrderResult: raise NotImplementedError
    def place_market_sell(self, product_id: str, base_qty: float) -> ExchangeOrderResult: raise NotImplementedError
    def cancel_order(self, order_id: str) -> bool: raise NotImplementedError

class CoinbaseAdapterPlaceholder(BaseExchangeAdapter):
    """Placeholder for gradually wrapping existing Coinbase code."""
    exchange_id = "coinbase"
    def __init__(self, live_portfolio: Any): self.live_portfolio = live_portfolio

class BinanceAdapterNotImplemented(BaseExchangeAdapter):
    """Future Binance live execution adapter. This intentionally does not trade yet."""
    exchange_id = "binance"
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key; self.api_secret = api_secret; self.testnet = bool(testnet)
    def get_account_snapshot(self) -> Dict[str, Any]: raise NotImplementedError("Binance live account support is not implemented yet.")
    def get_top_of_book(self, product_id: str) -> Dict[str, Any]: raise NotImplementedError("Binance live top-of-book support is not implemented yet.")
    def place_market_buy(self, product_id: str, quote_usd: float) -> ExchangeOrderResult: raise NotImplementedError("Binance live buying is not implemented yet.")
    def place_limit_buy(self, product_id: str, quote_usd: float, limit_price: float) -> ExchangeOrderResult: raise NotImplementedError("Binance live buying is not implemented yet.")
    def place_market_sell(self, product_id: str, base_qty: float) -> ExchangeOrderResult: raise NotImplementedError("Binance live selling is not implemented yet.")
    def cancel_order(self, order_id: str) -> bool: raise NotImplementedError("Binance live canceling is not implemented yet.")
