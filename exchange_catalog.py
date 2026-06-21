import os
from dataclasses import dataclass
from typing import Dict, Optional, List

EXCHANGE_COINBASE = "coinbase"
EXCHANGE_BINANCE = "binance"
EXCHANGE_BINANCE_US = "binance_us"
LIVE_EXECUTION_EXCHANGE = os.getenv("LIVE_EXECUTION_EXCHANGE", EXCHANGE_BINANCE_US).strip().lower()
HISTORICAL_DATA_SOURCE_PRIORITY: List[str] = ["local_cache", "binance_bulk", "legacy_removed_fallback"]

@dataclass(frozen=True)
# NOTE:
# coinbase_product_id is legacy field naming only.
# The runtime bot is Binance.US-only.
# Internal product IDs like BTC-USD are canonical display IDs.
# Execution symbols are Binance.US symbols selected dynamically by quote asset.
class ProductSymbolMap:
    canonical_product_id: str
    coinbase_product_id: str
    binance_symbol: Optional[str]
    binance_us_symbol: Optional[str] = None
    base_asset: str = ""
    quote_asset_coinbase: str = "USD"
    quote_asset_binance: str = "USDT"
    binance_us_symbol_usd: Optional[str] = None
    binance_us_symbol_usdt: Optional[str] = None
    historical_source_note: str = ""

PRODUCT_SYMBOL_MAP: Dict[str, ProductSymbolMap] = {
    "BTC-USD": ProductSymbolMap("BTC-USD", "BTC-USD", "BTCUSDT", "BTCUSDT", "BTC", binance_us_symbol_usd="BTCUSD", binance_us_symbol_usdt="BTCUSDT", historical_source_note="Binance.US live symbol selected dynamically: BTCUSD if USD quote exists, otherwise BTCUSDT."),
    "ETH-USD": ProductSymbolMap("ETH-USD", "ETH-USD", "ETHUSDT", "ETHUSDT", "ETH", binance_us_symbol_usd="ETHUSD", binance_us_symbol_usdt="ETHUSDT", historical_source_note="Binance.US live symbol selected dynamically: ETHUSD if USD quote exists, otherwise ETHUSDT."),
    "SOL-USD": ProductSymbolMap("SOL-USD", "SOL-USD", "SOLUSDT", "SOLUSDT", "SOL", binance_us_symbol_usd="SOLUSD", binance_us_symbol_usdt="SOLUSDT", historical_source_note="Binance.US live symbol selected dynamically: SOLUSD if USD quote exists, otherwise SOLUSDT."),
    "XRP-USD": ProductSymbolMap("XRP-USD", "XRP-USD", "XRPUSDT", "XRPUSDT", "XRP", binance_us_symbol_usd="XRPUSD", binance_us_symbol_usdt="XRPUSDT", historical_source_note="Binance.US live symbol selected dynamically: XRPUSD if USD quote exists, otherwise XRPUSDT."),
    "BNB-USD": ProductSymbolMap("BNB-USD", "BNB-USD", "BNBUSDT", "BNBUSDT", "BNB", binance_us_symbol_usd="BNBUSD", binance_us_symbol_usdt="BNBUSDT", historical_source_note="Binance.US live symbol selected dynamically: BNBUSD if USD quote exists, otherwise BNBUSDT."),
    "DOGE-USD": ProductSymbolMap("DOGE-USD", "DOGE-USD", "DOGEUSDT", "DOGEUSDT", "DOGE", binance_us_symbol_usd="DOGEUSD", binance_us_symbol_usdt="DOGEUSDT", historical_source_note="Binance.US live symbol selected dynamically: DOGEUSD if USD quote exists, otherwise DOGEUSDT."),
    "ADA-USD": ProductSymbolMap("ADA-USD", "ADA-USD", "ADAUSDT", "ADAUSDT", "ADA", binance_us_symbol_usd="ADAUSD", binance_us_symbol_usdt="ADAUSDT", historical_source_note="Binance.US live symbol selected dynamically: ADAUSD if USD quote exists, otherwise ADAUSDT."),
    "LINK-USD": ProductSymbolMap("LINK-USD", "LINK-USD", "LINKUSDT", "LINKUSDT", "LINK", binance_us_symbol_usd="LINKUSD", binance_us_symbol_usdt="LINKUSDT", historical_source_note="Binance.US live symbol selected dynamically: LINKUSD if USD quote exists, otherwise LINKUSDT."),
    "AVAX-USD": ProductSymbolMap("AVAX-USD", "AVAX-USD", "AVAXUSDT", "AVAXUSDT", "AVAX", binance_us_symbol_usd="AVAXUSD", binance_us_symbol_usdt="AVAXUSDT", historical_source_note="Binance.US live symbol selected dynamically: AVAXUSD if USD quote exists, otherwise AVAXUSDT."),
    "XLM-USD": ProductSymbolMap("XLM-USD", "XLM-USD", "XLMUSDT", "XLMUSDT", "XLM", binance_us_symbol_usd="XLMUSD", binance_us_symbol_usdt="XLMUSDT", historical_source_note="Binance.US live symbol selected dynamically: XLMUSD if USD quote exists, otherwise XLMUSDT."),
    "LTC-USD": ProductSymbolMap("LTC-USD", "LTC-USD", "LTCUSDT", "LTCUSDT", "LTC", binance_us_symbol_usd="LTCUSD", binance_us_symbol_usdt="LTCUSDT", historical_source_note="Binance.US live symbol selected dynamically: LTCUSD if USD quote exists, otherwise LTCUSDT."),
    "BCH-USD": ProductSymbolMap("BCH-USD", "BCH-USD", "BCHUSDT", "BCHUSDT", "BCH", binance_us_symbol_usd="BCHUSD", binance_us_symbol_usdt="BCHUSDT", historical_source_note="Binance.US live symbol selected dynamically: BCHUSD if USD quote exists, otherwise BCHUSDT."),
    "SHIB-USD": ProductSymbolMap("SHIB-USD", "SHIB-USD", "SHIBUSDT", "SHIBUSDT", "SHIB", binance_us_symbol_usd="SHIBUSD", binance_us_symbol_usdt="SHIBUSDT", historical_source_note="Binance.US live symbol selected dynamically: SHIBUSD if USD quote exists, otherwise SHIBUSDT."),
    "DOT-USD": ProductSymbolMap("DOT-USD", "DOT-USD", "DOTUSDT", "DOTUSDT", "DOT", binance_us_symbol_usd="DOTUSD", binance_us_symbol_usdt="DOTUSDT", historical_source_note="Binance.US live symbol selected dynamically: DOTUSD if USD quote exists, otherwise DOTUSDT."),
    "SUI-USD": ProductSymbolMap("SUI-USD", "SUI-USD", "SUIUSDT", "SUIUSDT", "SUI", binance_us_symbol_usd="SUIUSD", binance_us_symbol_usdt="SUIUSDT", historical_source_note="Binance.US live symbol selected dynamically: SUIUSD if USD quote exists, otherwise SUIUSDT."),
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
        if symbol in {mapping.binance_symbol, mapping.binance_us_symbol, mapping.binance_us_symbol_usd, mapping.binance_us_symbol_usdt}:
            return product_id
    return None

def product_mapping_rows() -> List[Dict[str, str]]:
    rows = []
    for product_id, mapping in PRODUCT_SYMBOL_MAP.items():
        rows.append({
            "canonical_product_id": mapping.canonical_product_id,
            "coinbase_product_id": mapping.coinbase_product_id,
            "binance_symbol": mapping.binance_symbol or "",
            "binance_us_symbol": mapping.binance_us_symbol or "",
            "binance_us_symbol_usd": mapping.binance_us_symbol_usd or "",
            "binance_us_symbol_usdt": mapping.binance_us_symbol_usdt or "",
            "base_asset": mapping.base_asset,
            "canonical_quote": mapping.quote_asset_coinbase,
            "binance_quote": mapping.quote_asset_binance,
            "note": mapping.historical_source_note,
        })
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
