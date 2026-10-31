"""Orquestador del bot: une datos -> features -> modelo -> riesgo -> ordenes.

Modos:
  * backtest : evaluation honesta con walk-forward (ver backtest.py).
  * live     : emite decisiones sobre la ultima vela cerrada; por defecto en modo
    paper-trading (registro en JSONL) para poder auditar antes de conectar una
    API real de exchange. La integracion de broker queda detras de `execute_order`.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    # Ejecución directa del archivo (`python bot.py`). Convertimos el módulo en
    # paquete para que los imports relativos funcionen sin instalar nada.
    import types, os, sys
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if "topbot" not in sys.modules:
        _pkg = types.ModuleType("topbot")
        _pkg.__path__ = [_pkg_dir]
        sys.modules["topbot"] = _pkg
    __package__ = "topbot"

from .data import DataConfig, load_dataset
from .features import build_features
from .model import ModelConfig, train, signal as make_signal
from .risk import RiskConfig, position_size, stop_targets

log = logging.getLogger("topbot.bot")


@dataclass
class BotConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    days: int = 730
    use_network: bool = True
    log_file: str = "logs/decisions.jsonl"
    dry_run: bool = True            # nunca enviar ordenes reales sin cambiar a False
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


class TopBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.ensemble = None
        self.last_bar_ts = None
        os.makedirs(os.path.dirname(cfg.log_file) or ".", exist_ok=True)

    # ---------------- ciclo de vida ----------------
    def refresh_model(self, df: pd.DataFrame | None = None):
        """Reentrena el ensemble con todo el historico disponible."""
        df = df if df is not None else self._load()
        from .features import make_labels
        y = make_labels(df, horizons=(self.cfg.model.horizon,),
                        k_atr=self.cfg.model.k_atr)
        X = build_features(df).loc[y.index[:-self.cfg.model.horizon]]
        y = y.iloc[:-self.cfg.model.horizon]
        self.ensemble = train(X, y[f"y_tri_{self.cfg.model.horizon}"], self.cfg.model)
        return self.ensemble

    def step(self) -> dict | None:
        """Procesa la ultima vela cerrada y devuelve la decision (o None)."""
        df = self._load()
        if self.ensemble is None:
            self.refresh_model(df)
        X_last = build_features(df).iloc[[-1]]
        ts = df.index[-1]
        if self.last_bar_ts == ts:
            return None  # esta vela ya fue procesada
        self.last_bar_ts = ts

        p_up = float(self.ensemble.predict_proba_up(X_last)[0])
        sig = int(make_signal(np.array([p_up]), self.cfg.model.prob_threshold)[0])
        px = float(df["close"].iloc[-1])
        atr = float(X_last["atr_rel"].iloc[0]) * px
        equity = self._equity()
        qty = position_size(equity, px, atr, p_up if sig >= 0 else 1 - p_up,
                            self.cfg.risk) if sig != 0 else 0.0
        sl_tp = stop_targets(px, atr, sig, self.cfg.risk) if sig != 0 else (None, None)

        decision = {
            "ts": str(ts), "symbol": self.cfg.symbol, "price": round(px, 2),
            "prob_up": round(p_up, 4), "signal": sig,
            "qty": round(qty, 6), "stop": sl_tp[0], "take_profit": sl_tp[1],
            "model_stats": self.ensemble.val_stats if self.ensemble else None,
            "dry_run": self.cfg.dry_run,
        }
        self._log(decision)
        if sig != 0 and not self.cfg.dry_run:
            self.execute_order(decision)
        return decision

    # ---------------- hooks ----------------
    def execute_order(self, decision: dict):
        """Punto de integracion con el exchange (CCXT/binance-connector).

        Se deja deliberadamente vacio hasta validar la estrategia en paper
        trading; al implementarlo respetar idempotencia (client_order_id = ts).
        """
        raise NotImplementedError("Conecta aqui la API real del exchange")

    # ---------------- internos ----------------
    def _load(self) -> pd.DataFrame:
        dcfg = DataConfig(symbol=self.cfg.symbol, interval=self.cfg.interval,
                          use_network=self.cfg.use_network)
        return load_dataset(self.cfg.symbol, self.cfg.interval, self.cfg.days, dcfg)

    def _equity(self) -> float:
        path = os.path.splitext(self.cfg.log_file)[0] + "_state.json"
        if os.path.exists(path):
            with open(path) as f:
                return float(json.load(f).get("equity", 10_000.0))
        return 10_000.0

    def _log(self, decision: dict):
        with open(self.cfg.log_file, "a") as f:
            f.write(json.dumps(decision, default=str) + "\n")


# ---------------- CLI: python bot.py (ejecución directa, sin instalar el paquete) ----------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="TopBot: orquestador datos -> features -> modelo -> riesgo.")
    ap.add_argument("--mode", choices=["step", "backtest", "train"], default="step")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h", choices=["15m", "1h", "4h", "1d"])
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--online", action="store_true",
                    help="descargar datos reales de Binance (por defecto usa caché/offline)")
    args = ap.parse_args()

    cfg = BotConfig(symbol=args.symbol, interval=args.interval, days=args.days,
                    use_network=args.online)
    bot = TopBot(cfg)
    if args.mode == "backtest":
        from .backtest import BacktestConfig, run_backtest
        dcfg = DataConfig(symbol=cfg.symbol, interval=cfg.interval,
                          use_network=cfg.use_network)
        df = load_dataset(cfg.symbol, cfg.interval, cfg.days, dcfg)
        print(run_backtest(df, BacktestConfig(), ModelConfig(), RiskConfig()))
    else:
        bot.train() if args.mode == "train" else None
        decision = bot.step()
        print(json.dumps(decision, indent=2, default=str))
