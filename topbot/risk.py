"""Gestion de riesgo: tamano de posicion, stops dinamicos y limites de cartera.

Reglas (aplicadas por el bot/backtest):
  * Riesgo fijo por operacion (% del equity), con stop a k*ATR -> tamano = riesgo/stop.
  * Kelly fraccionario limitado por probabilidad calibrada y ratio TP/SL.
  * Reduccion de riesgo cuando la volatilidad realizada sube (target vol).
  * Tope de exposicion simultanea, numero maximo de trades/dia y drawdown kill-switch.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005      # 0.5% del equity por operacion
    max_position_frac: float = 0.25    # tope: 25% del equity por posicion
    max_gross_exposure: float = 0.60   # suma |posiciones| <= 60% del equity
    atr_sl_mult: float = 1.5           # stop-loss a 1.5 ATR
    atr_tp_mult: float = 2.5           # take-profit a 2.5 ATR (R:R ~ 1.67)
    target_vol_annual: float = 0.30    # objetivo vol anualizada p.ej. BTC
    vol_lookback: int = 24             # velas para estimar vol realizada
    kelly_fraction: float = 0.25       # usar solo 1/4 de Kelly
    max_trades_per_day: int = 8
    max_drawdown_stop: float = 0.15    # kill-switch: cerrar todo si DD > 15%
    min_edge: float = 0.02             # margen minimo de prob. sobre umbral


def position_size(equity: float, price: float, atr: float, prob_up: float,
                  cfg: RiskConfig, realized_vol: float = 0.0) -> float:
    """Devuelve tamano en unidades del activo (0 si no hay trade viable)."""
    if equity <= 0 or price <= 0 or atr <= 0:
        return 0.0
    sl_dist = cfg.atr_sl_mult * atr
    if sl_dist <= 0:
        return 0.0

    # 1) Base: riesgo fijo sobre distancia del stop
    qty_risk = (equity * cfg.risk_per_trade) / sl_dist

    # 2) Kelly fraccionario como techo suave (long; simetrico para short)
    tp_dist = cfg.atr_tp_mult * atr
    b = tp_dist / sl_dist
    p = max(min(prob_up, 0.99), 0.01)
    kelly = (b * p - (1 - p)) / b if b > 0 else 0.0
    kelly_frac = max(kelly, 0.0) * cfg.kelly_fraction
    if kelly_frac > 0:
        qty_kelly = (equity * kelly_frac) / price
        qty_risk = min(qty_risk, max(qty_kelly, qty_risk * 0.25))

    # 3) Target volatility: escalar hacia abajo si la vol supera el objetivo
    if realized_vol > 0:
        hv_annual = realized_vol * (365 * 24) ** 0.5  # para velas horarias
        scale = min(1.0, cfg.target_vol_annual / max(hv_annual, 1e-9))
        qty_risk *= scale

    # 4) Topes duros
    qty_cap = (equity * cfg.max_position_frac) / price
    return float(max(0.0, min(qty_risk, qty_cap)))


def stop_targets(entry: float, atr: float, side: int, cfg: RiskConfig):
    """(stop_loss, take_profit) segun direccion."""
    if side > 0:
        return entry - cfg.atr_sl_mult * atr, entry + cfg.atr_tp_mult * atr
    return entry + cfg.atr_sl_mult * atr, entry - cfg.atr_tp_mult * atr


def expected_value(prob: float, reward_r: float, risk_r: float = 1.0,
                   fee_roundtrip: float = 0.0, slippage_roundtrip: float = 0.0) -> float:
    """EV en multiples de R neto de costes (costes expresados en fraction del precio;
    se aproximan restando su equivalente en R asumiendo stop ~ entrada)."""
    gross = prob * reward_r - (1 - prob) * risk_r
    cost_r = (fee_roundtrip + slippage_roundtrip) / max(risk_r * 0.01, 1e-9)
    return gross - cost_r
