"""CLI del bot: python scripts/run_bot.py --mode step|backtest [--online]"""
import argparse, json, logging, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from topbot.bot import TopBot, BotConfig
from topbot.backtest import run_backtest, BacktestConfig
from topbot.data import DataConfig, load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["step", "backtest", "train"], default="step")
ap.add_argument("--symbol", default="BTCUSDT")
ap.add_argument("--interval", default="1h")
ap.add_argument("--days", type=int, default=730)
ap.add_argument("--online", action="store_true", help="descargar datos reales de Binance")
args = ap.parse_args()

cfg = BotConfig(symbol=args.symbol, interval=args.interval, days=args.days,
                use_network=args.online)
if args.mode == "step":
    bot = TopBot(cfg)
    print(json.dumps(bot.step(), indent=2, default=str))
elif args.mode == "train":
    bot = TopBot(cfg)
    ens = bot.refresh_model()
    print(json.dumps(ens.val_stats, indent=2))
else:
    df = load_dataset(args.symbol, args.interval, args.days,
                      DataConfig(symbol=args.symbol, interval=args.interval,
                                 use_network=args.online))
    res = run_backtest(df, BacktestConfig())
    print(json.dumps(res.metrics, indent=2))
    res.trades.to_csv("backtest_trades.csv", index=False)
    res.equity_curve.to_csv("backtest_equity.csv")
