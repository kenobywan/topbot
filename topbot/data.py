"""Módulo de datos: descarga, caché local y validación de OHLCV.

Mejoras frente a una versión ingenua (una sola llamada sin control):
  * Descarga por paginación (1000 velas por petición) hasta cubrir el rango pedido.
  * Caché en disco (parquet/csv) con revalidación incremental: sólo se descargan
    las velas que faltan desde el último timestamp cacheado.
  * Validación estricta: duplicados, huecos temporales, ohlc incoherente,
    valores no positivos, outliers extremos de retorno.
  * Relleno de huecos para que la serie sea continua (necesario para indicadores).
  * Modo offline con generador sintético realista (caminaleatorio con volatilidad
    agrupada y saltos) para pruebas deterministas sin conexión.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

try:  #网络 disponible?
    import requests
except ImportError:  # pragma: no cover
    requests = None

log = logging.getLogger("topbot.data")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
VALID_INTERVALS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
                   "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
                   "1d": 86_400_000}


@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    cache_dir: str = "data_cache"
    use_network: bool = True
    max_requests_per_minute: int = 40
    _last_request_ts: float = field(default=0.0, repr=False)

    def throttle(self) -> None:
        """Respeta el límite de peticiones del exchange."""
        if requests is None:
            return
        min_gap = 60.0 / self.max_requests_per_minute
        wait = min_gap - (time.time() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.time()


def _cache_path(cfg: DataConfig) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    base = os.path.join(cfg.cache_dir, f"{cfg.symbol}_{cfg.interval}")
    return base + ".parquet" if _parquet_available() else base + ".csv"


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def _read_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()
    except Exception as exc:  # caché corrupta -> se ignora
        log.warning("Caché ilegible (%s): %s", path, exc)
        return None


def _write_cache(df: pd.DataFrame, path: str) -> None:
    try:
        if path.endswith(".parquet"):
            df.to_parquet(path)
        else:
            df.to_csv(path)
    except Exception as exc:
        log.warning("No se pudo escribir caché: %s", exc)


def fetch_binance(symbol: str, interval: str, start_ms: int, end_ms: int,
                  cfg: Optional[DataConfig] = None) -> pd.DataFrame:
    """Descarga velas de Binance con paginación y throttling."""
    if requests is None:
        raise RuntimeError("La librería `requests` no está disponible.")
    cfg = cfg or DataConfig(symbol=symbol, interval=interval)
    step = VALID_INTERVALS[interval]
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        cfg.throttle()
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": symbol, "interval": interval,
                    "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][6] + 1  # close_time + 1ms
        if len(batch) < 1000:
            break
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return _empty_frame()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out = df.set_index("ts")[["open", "high", "low", "close", "volume", "trades"]].astype(float)
    return out


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades"],
                        index=pd.DatetimeIndex([], tz="UTC", name="ts"))


def synthetic_ohlcv(n: int = 6000, interval: str = "1h", seed: int = 42,
                    start_price: float = 30_000.0) -> pd.DataFrame:
    """Serie OHLCV sintética realista (GARCH-like vol clustering + jumps).

    Pensada para tests offline deterministas; NO usar para conclusiones de negocio.
    """
    rng = np.random.default_rng(seed)
    step = VALID_INTERVALS[interval] / 3_600_000  # horas
    # Volatilidad con clusters: AR(1) sobre log-vol
    log_vol = np.zeros(n)
    for i in range(1, n):
        shock = abs(rng.normal(0, 0.08)) if rng.random() < 0.02 else 0.0
        log_vol[i] = 0.94 * log_vol[i - 1] + 0.06 * (-2.6) + rng.normal(0, 0.05) + shock
    vol = np.exp(log_vol) * 0.02
    drift = 0.00005 + 0.0004 * np.sin(np.arange(n) / 500.0)  # regimes suaves
    rets = rng.normal(drift, vol)
    jumps = rng.random(n) < 0.005
    rets[jumps] += rng.normal(0, 0.03, jumps.sum())
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0, vol)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.lognormal(10, 0.5, n) * (1 + 5 * vol)
    trades = rng.poisson(2000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq=f"{step}h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume, "trades": trades},
                        index=idx).rename_axis("ts")


def validate(df: pd.DataFrame, tolerance_ret: float = 0.5) -> list[str]:
    """Devuelve lista de problemas detectados (vacía == datos sanos)."""
    issues = []
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        issues.append(f"Faltan columnas: {required - set(df.columns)}")
        return issues
    if df.index.has_duplicates:
        issues.append(f"{df.index.duplicated().sum()} índices duplicados")
    if df.isna().any().any():
        issues.append(f"{int(df.isna().sum().sum())} celdas NaN")
    bad_ohlc = ((df["high"] < df[["open", "close"]].max(axis=1)) |
                (df["low"] > df[["open", "close"]].min(axis=1)))
    if bad_ohlc.any():
        issues.append(f"{int(bad_ohlc.sum())} velas con high/low incoherente")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append("Precios no positivos")
    r = np.log(df["close"]).diff()
    extreme = (r.abs() > tolerance_ret).sum()
    if extreme:
        issues.append(f"{int(extreme)} retornos extremos (|logret|>{tolerance_ret})")
    return issues


def fill_gaps(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Reindexa a la rejilla temporal completa; huecos se rellenan con última
   .close y volumen 0 (marcado con columna '_filled')."""
    freq = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1D"}[interval]
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    had_gaps = not df.index.equals(full)
    out = df.reindex(full)
    filled = out["close"].isna()
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].ffill()
    out["volume"] = out["volume"].fillna(0.0)
    if "trades" in out.columns:
        out["trades"] = out["trades"].fillna(0.0)
    out["_filled"] = filled
    if had_gaps:
        log.info("Se rellenaron %d huecos temporales", int(filled.sum()))
    return out.rename_axis("ts")


def load_dataset(symbol: str = "BTCUSDT", interval: str = "1h",
                 days: int = 730, cfg: Optional[DataConfig] = None,
                 force_refresh: bool = False) -> pd.DataFrame:
    """API principal: devuelve OHLCV limpio, continuo y validado.

    Estrategia: caché + actualización incremental; si no hay red, usa la caché;
    si tampoco hay caché y use_network es False, genera datos sintéticos.
    """
    cfg = cfg or DataConfig(symbol=symbol, interval=interval)
    now = pd.Timestamp.utcnow()
    start = now - pd.Timedelta(days=days)

    if not cfg.use_network:
        cache = _read_cache(_cache_path(cfg))
        if cache is not None and len(cache) > 500:
            log.info("Modo offline: usando caché (%d velas)", len(cache))
            return prepare(cache, interval)
        log.info("Modo offline sin caché: generando datos sintéticos")
        return prepare(synthetic_ohlcv(n=min(days * 24, 8000), interval=interval),
                       interval)

    path = _cache_path(cfg)
    cache = None if force_refresh else _read_cache(path)
    fetch_from = start
    if cache is not None and len(cache) > 0:
        last = cache.index.max()
        if (now - last.to_pydatetime()).total_seconds() < VALID_INTERVALS[interval] / 1000:
            log.info("Caché al día (%d velas)", len(cache))
            return prepare(cache, interval)
        fetch_from = last + pd.Timedelta(milliseconds=1)

    try:
        new = fetch_binance(symbol, interval,
                            int(fetch_from.timestamp() * 1000),
                            int(now.timestamp() * 1000), cfg)
        df = pd.concat([c for c in [cache, new] if c is not None and len(c)]) \
             if (cache is not None and len(new)) else new
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df.index >= start]
        _write_cache(df, path)
    except Exception as exc:
        log.warning("Fallo de red (%s); usando caché si existe", exc)
        if cache is None:
            raise
        df = cache

    issues = validate(df)
    if issues:
        log.warning("Datos con incidencias: %s", "; ".join(issues))
    return prepare(df, interval)


def prepare(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Limpieza final: quita última vela incompleta, rellena huecos, dedupe."""
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if "_filled" not in df.columns:
        now = pd.Timestamp.now(tz="UTC")
        partial = df.index.max() + pd.Timedelta(milliseconds=VALID_INTERVALS[interval])
        if partial > now:  # última vela aún no cierra
            df = df.iloc[:-1]
    df = fill_gaps(df, interval)
    return df.dropna(subset=["open", "high", "low", "close"])
