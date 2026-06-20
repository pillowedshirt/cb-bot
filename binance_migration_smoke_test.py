from dotenv import load_dotenv
from binance_us_client import BinanceUSClient
from binance_symbol_filters import parse_symbol_rules
load_dotenv(); client=BinanceUSClient()
print("sync_time:", client.sync_time()); print("ping:", client.ping())
account=client.account(); print("canTrade:", account.get("canTrade")); print("accountType:", account.get("accountType")); print("permissions:", account.get("permissions")); print("balances_count:", len(account.get("balances", [])))
symbol="BTCUSDT"; info=client.exchange_info(symbol=symbol); rules=parse_symbol_rules(info["symbols"][0]); print("symbol:", rules.symbol, rules.status, rules.base_asset, rules.quote_asset); print("min_notional:", rules.min_notional); print("step_size:", rules.step_size); print("tick_size:", rules.tick_size); print("bookTicker:", client.book_ticker(symbol=symbol)); print("trading_fee:", client.trading_fee(symbol=symbol)); print("Testing order validation only; this does not place a live order.")
try: print("test_order:", client.test_order(symbol=symbol, side="BUY", type="MARKET", quoteOrderQty="5.00", newOrderRespType="FULL"))
except Exception as exc: print("test_order_failed:", exc)
