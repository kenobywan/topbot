"""Motor de backtesting realista (y eficiente).

Diferencias frente a un backtest ingenuo:
  * Sin look-ahead: la senal en t usa features hasta t; entrada al cierre t,
    salida evaluada con high/low de velas t+1..t+h.
  * Costes reales por lado: fee + slippage configurables.
  * Stops intra-vela conservadores (si SL y TP caben en la misma vela -> SL).
  * Walk-forward con PURGE: el modelo en cada tramo se entrena solo con datos
    anteriores separados por `purge_gap` velas (evita solapamiento de etiquetas).
  * Ventana rodante (`train_window`) + batch prediction -> coste O(n) razonable.
  * Metricas completas: retorno, CAGR, Sharpe, Sortino, max DD, hit rate,
    avg pnl neto por trade, profit factor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import build_dataset
from .model import ModelConfig, train, signal as make_signal
from .risk import RiskConfig, position_size, stop_targets

log = logging.getLogger("topbot.backtest")


@dataclass
class BacktestConfig:
    start_frac: float = 0.5      # primera mitad: warmup + primer entrenamiento
    horizon: int = 4
    prob_threshold: float = 0.55
    fee: float = 0.0005          # 5 bps por lado
    slippage: float = 0.0003     # 3 bps por lado
    train_window: int = 3000     # velas de historico para cada entrenamiento
    retrain_every: int = 750     # reentrena cada N velas del periodo de test
    allow_shorts: bool = True
    initial_equity: float = 10_000.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def _metrics(eq: pd.Series, trades: pd.DataFrame, bars_per_year: float) -> dict:
    ret = eq.pct_change().dropna()
    years = max(len(eq) / bars_per_year, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    sharpe = ret.mean() / ret.std(ddof=0) * np.sqrt(bars_per_year) if ret.std(ddof=0) > 0 else 0.0
    downside = ret[ret < 0].std(ddof=0)
    sortino = ret.mean() / downside * np.sqrt(bars_per_year) if downside > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    t = trades
    if len(t):
        gains = t.loc[t.pnl > 0, "pnl"].sum()
        losses = abs(t.loc[t.pnl < 0, "pnl"].sum())
        pf = round(float(gains / losses), 2) if losses > 0 else float("inf")
    else:
        pf = 0.0
    return {
        "total_return": round(float(total), 4),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
        "max_drawdown": round(dd, 4),
        "n_trades": int(len(t)),
        "hit_rate": round(float((t["pnl"] > 0).mean()), 3) if len(t) else 0.0,
        "avg_pnl_bps": round(float(t["pnl_pct"].mean() * 1e4), 1) if len(t) else 0.0,
        "profit_factor": pf,
    }


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig | None = None,
                 risk: RiskConfig | None = None, mcfg: ModelConfig | None = None,
                 verbose: bool = True) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    risk = risk or RiskConfig()
    mcfg = mcfg or ModelConfig(horizon=cfg.horizon, n_splits=2,
                               prob_threshold=cfg.prob_threshold)

    X, y = build_dataset(df, horizons=(cfg.horizon,))
    close = df["close"].loc[X.index]
    high, low = df["high"].loc[X.index], df["low"].loc[X.index]
    atr_rel = X["atr_rel"].values

    n = len(X)
    first_test = int(n * cfg.start_frac)
    purge = max(cfg.horizon * 2, mcfg.purge_gap)
    bars_per_year = 365 * 24 / max(cfg.horizon, 1)

    # ---- Predicciones out-of-sample por tramos walk-forward (batch, sin leakage).
    # IMPORTANTE: se purgan las ultimas `purge` velas de cada train porque sus
    # etiquetas (retorno a h velas) aun miran dentro del tramo de test.
    proba = np.full(n, np.nan)
    i = first_test
    while i < n - cfg.horizon:
        tr_end = max(i - purge, purge)
        tr_start = max(tr_end - cfg.train_window, 0)
        ens = train(X.iloc[tr_start:tr_end],
                    y[f"y_tri_{cfg.horizon}"].iloc[tr_start:tr_end], mcfg)
        if ens is None:
            break
        te_end = min(i + cfg.retrain_every, n - cfg.horizon)
        proba[i:te_end] = ens.predict_proba_up(X.iloc[i:te_end])
        if verbose:
            log.info("Walk-forward [%d:%d] entrenado en [%d:%d] acc=%.3f",
                     i, te_end, tr_start, tr_end, ens.val_stats["val_acc_ens"])
        i = te_end

    # ---- Simulacion de cartera SIN look-ahead:
    # La senal se toma al cierre de t; la entrada se ejecuta al APERTURA de t+1
    # (con slippage). Los stops se revisan con high/low de t+1..t+horizon y, si
    # nada toca un nivel, se cierra al cierre de t+horizon. Las etiquetas de
    # entrenamiento en el borde final se purgan (`purge` velas) para que ninguna
    # etiqueta del train observe precios del tramo testeado.
    equity = cfg.initial_equity
    eq_curve, trades = [], []
    pending_sig, pending_p = 0, 0.5          # senal tomada en el cierre anterior
    pos_side, pos_qty, entry_px, sl_px, tp_px, entry_i = 0, 0.0, 0.0, 0.0, 0.0, -1
    cost_rt = 2 * (cfg.fee + cfg.slippage)

    for i in range(first_test, n - cfg.horizon):
        px_open = df["open"].iloc[i]
        px_close = close.iloc[i]
        atr = atr_rel[i] * px_close

        # 1) liquidaciones de la posicion abierta (vela actual)
        if pos_side != 0:
            hit_sl = (low.iloc[i] <= sl_px) if pos_side > 0 else (high.iloc[i] >= sl_px)
            hit_tp = (high.iloc[i] >= tp_px) if pos_side > 0 else (low.iloc[i] <= tp_px)
            exit_px = None
            if hit_sl:
                exit_px = sl_px
            elif hit_tp:
                exit_px = tp_px
            elif i - entry_i >= cfg.horizon:
                exit_px = px_close
            if exit_px is not None:
                exit_px -= pos_side * cfg.slippage * exit_px   # slippage de salida
                pnl = (exit_px - entry_px) * pos_qty * pos_side
                pnl -= 2 * cfg.fee * exit_px * abs(pos_qty)
                pnl_pct = (exit_px / entry_px - 1) * pos_side - cost_rt
                equity += pnl
                trades.append({"ts": X.index[i], "side": pos_side, "entry": entry_px,
                               "exit": exit_px, "qty": pos_qty, "pnl": pnl,
                               "pnl_pct": pnl_pct,
                               "reason": "sl" if hit_sl else ("tp" if hit_tp else "time")})
                pos_side, pos_qty = 0, 0.0

        # 2) ejecutar al apertura la senal generada al cierre anterior
        if pos_side == 0 and pending_sig != 0:
            qty = position_size(equity, px_open, atr, pending_p, risk,
                                realized_vol=max(atr_rel[i], 1e-6) * 0.6)
            if qty > 0:
                pos_side, pos_qty, entry_i = pending_sig, qty, i
                entry_px = px_open * (1 + pending_sig * cfg.slippage)
                sl_px, tp_px = stop_targets(entry_px, atr, pending_sig, risk)
        pending_sig = 0

        # 3) tomar nueva senal al CIERRE de esta vela (para la apertura siguiente)
        if pos_side == 0 and not np.isnan(proba[i]):
            p_up = proba[i]
            sig = make_signal(np.array([p_up]), cfg.prob_threshold)[0]
            if sig != 0 and (sig > 0 or cfg.allow_shorts):
                pending_sig = sig
                pending_p = p_up if sig > 0 else 1 - p_up

        mark = equity + ((px_close - entry_px) * pos_qty * pos_side if pos_side else 0.0)
        eq_curve.append((X.index[i], mark))

    eq = pd.Series(dict(eq_curve)).sort_index()
    tr = pd.DataFrame(trades)
    metrics = _metrics(eq, tr, bars_per_year) if len(eq) > 2 else {}
    if verbose:
        log.info("Backtest: %s", metrics)
    return BacktestResult(eq, tr, metrics)
