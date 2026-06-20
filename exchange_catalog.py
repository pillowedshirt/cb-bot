import os
from dataclasses import dataclass
from typing import Dict, Optional, List

EXCHANGE_COINBASE = "coinbase"
EXCHANGE_BINANCE = "binance"
EXCHANGE_BINANCE_US = "binance_us"
LIVE_EXECUTION_EXCHANGE = os.getenv("LIVE_EXECUTION_EXCHANGE", EXCHANGE_BINANCE_US).strip().lower()
HISTORICAL_DATA_SOURCE_PRIORITY: List[str] = ["local_cache", "binance_bulk", "coinbase_fallback"]

@dataclass(frozen=True)
class ProductSymbolMap:
    canonical_product_id: str
    coinbase_product_id: str
    binance_symbol: Optional[str]
    binance_us_symbol: Optional[str] = None
    base_asset: str = ""
    quote_asset_coinbase: str = "USD"
    quote_asset_binance: str = "USDT"
    historical_source_note: str = ""

PRODUCT_SYMBOL_MAP: Dict[str, ProductSymbolMap] = {
    "BTC-USD": ProductSymbolMap("BTC-USD", "BTC-USD", "BTCUSDT", "BTCUSDT", "BTC", historical_source_note="Binance BTCUSDT used as historical proxy for Coinbase BTC-USD."),
    "ETH-USD": ProductSymbolMap("ETH-USD", "ETH-USD", "ETHUSDT", "ETHUSDT", "ETH", historical_source_note="Binance ETHUSDT used as historical proxy for Coinbase ETH-USD."),
    "SOL-USD": ProductSymbolMap("SOL-USD", "SOL-USD", "SOLUSDT", "SOLUSDT", "SOL", historical_source_note="Binance SOLUSDT used as historical proxy for Coinbase SOL-USD."),
    "XRP-USD": ProductSymbolMap("XRP-USD", "XRP-USD", "XRPUSDT", "XRPUSDT", "XRP", historical_source_note="Binance XRPUSDT used as historical proxy for Coinbase XRP-USD."),
    "BNB-USD": ProductSymbolMap("BNB-USD", "BNB-USD", "BNBUSDT", "BNBUSDT", "BNB", historical_source_note="Binance BNBUSDT used as historical proxy for Coinbase BNB-USD."),
    "DOGE-USD": ProductSymbolMap("DOGE-USD", "DOGE-USD", "DOGEUSDT", "DOGEUSDT", "DOGE", historical_source_note="Binance DOGEUSDT used as historical proxy for Coinbase DOGE-USD."),
    "ADA-USD": ProductSymbolMap("ADA-USD", "ADA-USD", "ADAUSDT", "ADAUSDT", "ADA", historical_source_note="Binance ADAUSDT used as historical proxy for Coinbase ADA-USD."),
    "LINK-USD": ProductSymbolMap("LINK-USD", "LINK-USD", "LINKUSDT", "LINKUSDT", "LINK", historical_source_note="Binance LINKUSDT used as historical proxy for Coinbase LINK-USD."),
    "AVAX-USD": ProductSymbolMap("AVAX-USD", "AVAX-USD", "AVAXUSDT", "AVAXUSDT", "AVAX", historical_source_note="Binance AVAXUSDT used as historical proxy for Coinbase AVAX-USD."),
    "XLM-USD": ProductSymbolMap("XLM-USD", "XLM-USD", "XLMUSDT", "XLMUSDT", "XLM", historical_source_note="Binance XLMUSDT used as historical proxy for Coinbase XLM-USD."),
    "LTC-USD": ProductSymbolMap("LTC-USD", "LTC-USD", "LTCUSDT", "LTCUSDT", "LTC", historical_source_note="Binance LTCUSDT used as historical proxy for Coinbase LTC-USD."),
    "BCH-USD": ProductSymbolMap("BCH-USD", "BCH-USD", "BCHUSDT", "BCHUSDT", "BCH", historical_source_note="Binance BCHUSDT used as historical proxy for Coinbase BCH-USD."),
    "SHIB-USD": ProductSymbolMap("SHIB-USD", "SHIB-USD", "SHIBUSDT", "SHIBUSDT", "SHIB", historical_source_note="Binance SHIBUSDT used as historical proxy for Coinbase SHIB-USD."),
    "DOT-USD": ProductSymbolMap("DOT-USD", "DOT-USD", "DOTUSDT", "DOTUSDT", "DOT", historical_source_note="Binance DOTUSDT used as historical proxy for Coinbase DOT-USD."),
    "SUI-USD": ProductSymbolMap("SUI-USD", "SUI-USD", "SUIUSDT", "SUIUSDT", "SUI", historical_source_note="Binance SUIUSDT used as historical proxy for Coinbase SUI-USD."),
}

def get_symbol_map(product_id: str) -> Optional[ProductSymbolMap]:
    return PRODUCT_SYMBOL_MAP.get(str(product_id).strip().upper())

def coinbase_to_binance_symbol(product_id: str, *, prefer_us: bool = False) -> Optional[str]:
    mapping = get_symbol_map(product_id)
    if not mapping:
        return None
    if prefer_us and mapping.binance_us_symbol:
        return mapping.binance_us_symbol
    return mapping.binance_symbol

def binance_to_coinbase_product(symbol: str) -> Optional[str]:
    symbol = str(symbol or "").strip().upper()
    for product_id, mapping in PRODUCT_SYMBOL_MAP.items():
        if symbol in {mapping.binance_symbol, mapping.binance_us_symbol}:
            return product_id
    return None

def product_mapping_rows() -> List[Dict[str, str]]:
    rows = []
    for product_id, mapping in PRODUCT_SYMBOL_MAP.items():
        rows.append({"canonical_product_id": mapping.canonical_product_id, "coinbase_product_id": mapping.coinbase_product_id, "binance_symbol": mapping.binance_symbol or "", "binance_us_symbol": mapping.binance_us_symbol or "", "base_asset": mapping.base_asset, "coinbase_quote": mapping.quote_asset_coinbase, "binance_quote": mapping.quote_asset_binance, "note": mapping.historical_source_note})
    return rows


def canonical_product_id(product_id: str) -> str:
    value = str(product_id or "").strip().upper()
    if value in PRODUCT_SYMBOL_MAP:
        return value
    mapped = binance_to_coinbase_product(value)
    return mapped or value

def product_to_execution_symbol(product_id: str, exchange: str = EXCHANGE_BINANCE_US) -> Optional[str]:
    mapping = get_symbol_map(product_id)
    if not mapping:
        return None
    if str(exchange).lower() == EXCHANGE_BINANCE_US:
        return mapping.binance_us_symbol or mapping.binance_symbol
    if str(exchange).lower() == EXCHANGE_BINANCE:
        return mapping.binance_symbol
    return mapping.coinbase_product_id

def execution_symbol_to_product(symbol: str, exchange: str = EXCHANGE_BINANCE_US) -> Optional[str]:
    if str(exchange).lower() in {EXCHANGE_BINANCE_US, EXCHANGE_BINANCE}:
        return binance_to_coinbase_product(symbol)
    return str(symbol or "").strip().upper()
