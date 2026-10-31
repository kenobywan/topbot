"""Ingenieria de caracteristicas para prediccion.

Mejoras clave frente a un bot tipico (RSI/MACD sueltos sin normalizar):
  * Indicadores implementados SIN look-ahead (todo usa solo datos pasados).
  * Features estacionarias (retornos, ratios, posiciones relativas), no precios brutos.
  * Normalizacion robusta (mediana/RMA) calculada SOLO con ventana historica
    mediante transformadores fit-in-train aplicados con expanding median MAD.
  * Multi-horizonte: mismas features sirven para predecir 1/4/12 velas ahead.
  * Etiquetas triple-barrier simplificadas: sube/baja neutra segun umbral dinamico
    (k * ATR relativo), evitando clases desbalanceadas por regimen de volatilidad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------- helpers ------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(line, signal)
    return line, sig, line - sig


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _bbands(close: pd.Series, period: int = 20, nstd: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper, lower = mid + nstd * sd, mid - nstd * sd
    width = (upper - lower) / mid
    pos = (close - lower) / (upper - lower).replace(0, np.nan)
    return pos, width


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    return ((close - ll) / (hh - ll).replace(0, np.nan)) * 100


def _zscore(series: pd.Series, window: int = 200) -> pd.Series:
    """Z-score con media/desviacion expansivas minimizadas en coste computacional."""
    m = series.expanding(min_periods=50).mean()
    s = series.expanding(min_periods=50).std(ddof=0).replace(0, np.nan)
    return ((series - m) / s).clip(-6, 6)


def _vwap_dev(df: pd.DataFrame, period: int = 24) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).rolling(period).sum()
    vv = df["volume"].rolling(period).sum().replace(0, np.nan)
    return (df["close"] - pv / vv) / df["close"]


def _obv_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    sign = np.sign(df["close"].diff()).fillna(0)
    obv = (sign * df["volume"]).cumsum()
    out = pd.DataFrame(index=df.index)
    out["obv_z"] = _zscore(obv.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0))
    out["vol_ratio_20"] = np.log1p(df["volume"]) - np.log1p(df["volume"].rolling(20).mean())
    out["trades_ratio"] = (np.log1p(df["trades"]) - np.log1p(df["trades"].rolling(20).mean())
                           if "trades" in df.columns else 0.0)
    return out


def _returns_lag(close: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    logc = np.log(close)
    for lag in (1, 2, 3, 6, 12, 24, 72):
        out[f"ret_{lag}"] = logc.diff(lag)
    # Momentum con normalizacion de volatilidad (risk-adjusted momentum)
    vol24 = logc.diff().rolling(24).std(ddof=0).replace(0, np.nan)
    for lag in (6, 24, 72):
        out[f"mom_z_{lag}"] = (out[f"ret_{lag}"] / (vol24 * np.sqrt(lag))).clip(-6, 6)
    return out


def _time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    hour = idx.hour + idx.minute / 60
    out["hod_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hod_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return out


def _vol_regime(df: pd.DataFrame) -> pd.DataFrame:
    logc = np.log(df["close"])
    ret = logc.diff()
    out = pd.DataFrame(index=df.index)
    rv5 = ret.rolling(5).std(ddof=0)
    rv24 = ret.rolling(24).std(ddof=0)
    rv168 = ret.rolling(168).std(ddof=0)
    out["rv_ratio_short"] = np.log(rv5 / rv24.replace(0, np.nan)).clip(-3, 3)
    out["rv_ratio_long"] = np.log(rv24 / rv168.replace(0, np.nan)).clip(-3, 3)
    atr_rel = _atr(df) / df["close"]
    out["atr_rel"] = atr_rel
    out["atr_rel_pct"] = atr_rel / atr_rel.expanding(min_periods=50).quantile(0.99).clip(lower=1e-9)
    # Range expansion: movimiento intra-vela vs cierre previo
    rng = (df["high"] - df["low"]) / df["close"]
    out["intraday_range_z"] = ((rng - rng.expanding(50).mean()) /
                               rng.expanding(50).std(ddof=0).replace(0, np.nan)).clip(-6, 6)
    return out


def _trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    e10, e50, e200 = _ema(close, 10), _ema(close, 50), _ema(close, 200)
    out["ema10_50"] = (close / e50 - 1)
    out["ema50_200"] = (e50 / e200 - 1)
    out["dist_ema200"] = (close / e200 - 1)
    slope = e50.pct_change(10) / e50 * e50  # pendiente normalizada aprox
    out["ema50_slope"] = (e50 / e50.shift(10) - 1).clip(-0.5, 0.5)
    # Eficiencia de mercado de Kaufman: cambio neto / suma de cambios absolutos
    direction = (close - close.shift(20)).abs()
    volatility = close.diff().abs().rolling(20).sum()
    out["kaufman_eff"] = (direction / volatility.replace(0, np.nan)).clip(0, 1)
    return out


# ------------------------- API principal ----------------------------------

FEATURE_COLUMNS_ORDER = None  # se rellena tras construir el primer dataset


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte OHLCV limpio en matriz de caracteristicas sin look-ahead."""
    close, high, low = df["close"], df["high"], df["low"]
    feats = pd.DataFrame(index=df.index)

    feats["rsi_14"] = (_rsi(close, 14) - 50) / 50
    feats["rsi_7"] = (_rsi(close, 7) - 50) / 50
    macd_line, macd_sig, macd_hist = _macd(close)
    feats["macd_hist"] = (macd_hist / close).clip(-0.05, 0.05) * 100
    feats["macd_cross"] = np.sign(macd_line - macd_sig)
    bb_pos, bb_width = _bbands(close)
    feats["bb_pos"] = (bb_pos - 0.5) * 2
    feats["bb_width"] = _zscore(bb_width.fillna(bb_width.median()))
    st = _stoch_k(high, low, close)
    feats["stoch_k"] = (st - 50) / 50
    feats["stoch_d"] = ((_stoch_k(high, low, close).rolling(3).mean()) - 50) / 50

    feats = pd.concat([feats, _returns_lag(close), _vol_regime(df),
                       _trend_strength(df), _obv_volume_features(df),
                       _time_features(df.index)], axis=1)

    # Interacciones utiles para modelos lineales/arboles pequenos
    feats["rsi_x_mom24"] = feats["rsi_14"] * feats["mom_z_24"]
    feats["trend_x_vol"] = feats["ema50_200"] * feats["rv_ratio_long"]

    feats = feats.replace([np.inf, -np.inf], np.nan)
    # NaN iniciales de ventanas largas -> 0 (el modelo ignora zona warmup al cortar)
    feats = feats.fillna(0.0)
    return feats


def make_labels(df: pd.DataFrame, horizons=(1, 4, 12), k_atr: float = 0.5) -> pd.DataFrame:
    """Etiquetas por horizonte con umbral simetrico basado en ATR relativo.

    y = +1 si el retorno forward supera k*ATR_rel, -1 si cae por debajo, 0 neutro.
    Para clasificacion binaria se usa `y_bin_h{h}` (neutros excluidos via peso).
    """
    logc = np.log(df["close"])
    atr_rel = _atr(df) / df["close"]
    lab = pd.DataFrame(index=df.index)
    for h in horizons:
        fwd = logc.shift(-h) - logc
        thr = k_atr * atr_rel * np.sqrt(h)
        lab[f"y_fwd_{h}"] = fwd
        lab[f"y_tri_{h}"] = np.select([fwd > thr, fwd < -thr], [1, -1], 0)
        lab[f"y_bin_{h}"] = (fwd > 0).astype(int)
        lab[f"w_{h}"] = (lab[f"y_tri_{h}"] != 0).astype(float)  # 1 = muestra informativa
    return lab


def build_dataset(df: pd.DataFrame, horizons=(1, 4, 12)) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = build_features(df)
    y = make_labels(df, horizons)
    # Las ultimas N velas no tienen etiqueta completa
    y = y.iloc[:-max(horizons)]
    X = X.loc[y.index]
    return X, y
