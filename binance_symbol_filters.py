from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional
@dataclass
class BinanceSymbolRules:
    symbol: str; base_asset: str; quote_asset: str; status: str; min_qty: Decimal; max_qty: Decimal; step_size: Decimal; min_notional: Decimal; max_notional: Optional[Decimal]; tick_size: Decimal; market_min_qty: Optional[Decimal]=None; market_max_qty: Optional[Decimal]=None; market_step_size: Optional[Decimal]=None
def _dec(value: Any, default: str = "0") -> Decimal:
    try: return Decimal(str(value))
    except Exception: return Decimal(default)
def _filter_map(symbol_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(f.get("filterType")): f for f in symbol_info.get("filters", []) if isinstance(f, dict)}
def parse_symbol_rules(symbol_info: Dict[str, Any]) -> BinanceSymbolRules:
    filters = _filter_map(symbol_info); lot=filters.get("LOT_SIZE", {}); market_lot=filters.get("MARKET_LOT_SIZE", {}); price_filter=filters.get("PRICE_FILTER", {}); min_notional_filter=filters.get("MIN_NOTIONAL", {}); notional_filter=filters.get("NOTIONAL", {})
    min_notional=_dec(notional_filter.get("minNotional") or min_notional_filter.get("minNotional") or "0"); max_raw=notional_filter.get("maxNotional"); max_notional=_dec(max_raw) if max_raw is not None else None
    return BinanceSymbolRules(str(symbol_info.get("symbol") or ""), str(symbol_info.get("baseAsset") or ""), str(symbol_info.get("quoteAsset") or ""), str(symbol_info.get("status") or ""), _dec(lot.get("minQty"), "0"), _dec(lot.get("maxQty"), "0"), _dec(lot.get("stepSize"), "0"), min_notional, max_notional, _dec(price_filter.get("tickSize"), "0"), _dec(market_lot.get("minQty")) if market_lot else None, _dec(market_lot.get("maxQty")) if market_lot else None, _dec(market_lot.get("stepSize")) if market_lot else None)
def floor_to_step(value: Any, step: Decimal) -> Decimal:
    d=_dec(value); return d if step <= 0 else (d / step).to_integral_value(rounding=ROUND_DOWN) * step
def format_decimal(value: Decimal) -> str:
    s=format(value.normalize(), "f"); s=s.rstrip("0").rstrip(".") if "." in s else s; return s if s else "0"
def format_price(price: Any, rules: BinanceSymbolRules) -> str: return format_decimal(floor_to_step(price, rules.tick_size))
def format_quantity(qty: Any, rules: BinanceSymbolRules, *, market: bool=False) -> str:
    step=rules.market_step_size if market and rules.market_step_size else rules.step_size; return format_decimal(floor_to_step(qty, step))
def quantity_from_quote(quote_usd: float, price: float, rules: BinanceSymbolRules, *, market: bool=False) -> str:
    if price <= 0: return "0"
    return format_quantity(Decimal(str(float(quote_usd))) / Decimal(str(float(price))), rules, market=market)
def order_meets_minimums(*, quote_usd: float, qty: str, price: float, rules: BinanceSymbolRules) -> bool:
    qty_dec=_dec(qty); notional=qty_dec * Decimal(str(float(price)))
    return not (rules.min_qty > 0 and qty_dec < rules.min_qty) and not (rules.min_notional > 0 and notional < rules.min_notional) and not (rules.max_notional is not None and rules.max_notional > 0 and notional > rules.max_notional) and rules.status.upper() == "TRADING"
