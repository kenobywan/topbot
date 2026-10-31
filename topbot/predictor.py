"""Predictor de precio multi-moneda: semáforo COMPRAR / ESPERAR / VENDER para spot.

Diseñado para que el usuario decida MANUALMENTE qué moneda comprar o vender:

  * Por cada moneda entrena un clasificador calibrado (Logit + HistGBM) que
    estima P(precio sube en H horas), validado walk-forward CON purge+embargo
    (sin fuga de datos) => la probabilidad es fiable, no un número inflado.
  * Predice además un RANGO de precio probable (regresión por cuantiles 20/50/80)
    en lugar de un único "precio futuro" falso-preciso.
  * Convierte P(alza) + retorno esperado neto de comisiones spot en una acción
    clara con motivo legible y nivel de confianza según su validez histórica OOS.
  * Ranking multi-moneda: ordena por ventaja esperada neta para que sepas DÓNDE
    poner el dinero primero.

Uso CLI:  python scripts/predict.py --online
Uso API:  from topbot.predictor import PredictorConfig, predict_watchlist
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler

from .data import DataConfig, load_dataset
from .features import build_features, make_labels, _atr
from .model import PurgedWalkForward, _platt

log = logging.getLogger("topbot.predictor")

# Comisión spot típica Binance: 0.10% taker por lado => ~0.20% ida y vuelta
SPOT_FEE_ROUNDTRIP = 0.002

_PRETTY = {
    "ret_1": "retorno 1 vela", "ret_2": "retorno 2 velas", "ret_3": "retorno 3 velas",
    "ret_6": "retorno 6 velas", "ret_12": "retorno 12 velas", "ret_24": "retorno 24h",
    "ret_48": "retorno 48h", "ret_72": "retorno 72h", "ret_168": "retorno 7d",
    "mom_ema_9_21": "cruce EMA 9/21", "mom_ema_21_50": "cruce EMA 21/50",
    "mom_ema_50_200": "tendencia EMA 50/200", "rsi_14": "RSI(14)",
    "rsi_7": "RSI(7)", "macd_hist": "histograma MACD", "bb_pos": "posición Bandas Bollinger",
    "bb_width": "anchura Bollinger", "atr_rel": "volatilidad ATR relativa",
    "stoch_k": "Estocástico %K", "zscore_200": "Z-score 200", "vwap_dev": "desvío VWAP",
    "vol_ratio": "ratio de volumen", "obv_slope": "pendiente OBV",
    "vol_regime": "régimen de volatilidad", "hl_range": "rango alta-baja",
    "close_loc": "cierre en el rango", "hour_sin": "hora (seno)", "hour_cos": "hora (coseno)",
    "dow": "día de semana",
}


def _pretty_feature(name: str) -> str:
    return _PRETTY.get(name, name.replace("_", " "))


# Features cíclicas/de calendario: su signo lineal no es interpretable; se
# excluyen de los drivers para no engañar con falsos "alcista/bajista".
_NON_DIRECTIONAL = {"hod_sin", "hod_cos", "dow_sin", "dow_cos", "month", "is_weekend"}


@dataclass
class PredictorConfig:
    symbols: list[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "LINKUSDT"])
    interval: str = "1h"
    days: int = 730
    horizons_hours: tuple[int, ...] = (4, 24, 72, 168)   # 4h, 1d, 3d, 7d
    primary_horizon: int = 24                            # horizonte que manda la señal
    use_network: bool = True
    cache_dir: str = "data_cache"
    fee_roundtrip: float = SPOT_FEE_ROUNDTRIP
    buy_prob: float = 0.58           # p_up mínima para recomendar COMPRAR
    sell_prob: float = 0.42          # p_up máxima para recomendar VENDER/EVITAR
    min_acc_oos: float = 0.52        # exactitud OOS mínima para confiar
    quantiles: tuple[float, float] = (0.20, 0.80)
    random_state: int = 7
    min_bars: int = 2000             # velas mínimas de histórico por moneda

    def bars_per_hour(self) -> float:
        return {"1m": 60, "5m": 12, "15m": 4, "30m": 2, "1h": 1,
                "4h": 0.25, "1d": 1 / 24}[self.interval]


# ---------------------------------------------------------------------------
# Regresor de cuantiles (rango de precio previsto) con scaler integrado
# ---------------------------------------------------------------------------

class _QuantileModel:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.scaler = None
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> bool:
        try:
            self.scaler = StandardScaler().fit(X.values)
            self.model = QuantileRegressor(quantile=self.alpha, alpha=0.1,
                                           solver="highs", fit_intercept=True)
            self.model.fit(self.scaler.transform(X.values), y)
            return True
        except Exception as exc:
            log.debug("QuantileRegressor(alpha=%s) falló: %s", self.alpha, exc)
            return False

    def predict(self, Xv: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(Xv))


# ---------------------------------------------------------------------------
# Modelo entrenado por moneda/horizonte
# ---------------------------------------------------------------------------

@dataclass
class CoinPredictor:
    """P(alza) calibrada + rango de retorno log forward para 1 moneda/horizonte."""
    logit: LogisticRegression
    gbm: HistGradientBoostingClassifier
    w_gbm: float
    platt: tuple[float, float]
    q_med: Optional[_QuantileModel]
    q_lo: Optional[_QuantileModel]
    q_hi: Optional[_QuantileModel]
    feature_names: list[str]
    horizon: int
    atr_rel_now: float          # ATR relativo actual (para banda heurística)
    gbm_imp: Optional[np.ndarray] = None   # permutación de importancia (escala z)
    logit_coefs: Optional[pd.Series] = None  # coeficientes estandarizados del Logit
    stats: dict = field(default_factory=dict)

    def drivers(self, X_last: pd.DataFrame, top_n: int = 3) -> list[tuple[str, str]]:
        """Explica QUÉ empuja la señal: contribución lineal del Logit (signo
        exacto) escalada por la importancia de cada feature en el ensemble.
        Devuelve [(feature legible, 'alcista/bajista')]."""
        if self.logit_coefs is None:
            return []
        row = X_last[self.feature_names]
        contrib = self.logit_coefs.reindex(self.feature_names).fillna(0) * row.iloc[0]
        contrib = contrib.drop(index=[f for f in _NON_DIRECTIONAL
                                      if f in contrib.index], errors="ignore")
        if self.gbm_imp is not None:
            imp = pd.Series(self.gbm_imp, index=self.feature_names)
            scale = (imp / max(float(imp.max()), 1e-9)).clip(0.25, 1.0)
            contrib = contrib * scale.reindex(contrib.index).fillna(1.0)
        ranked = contrib.abs().sort_values(ascending=False).head(top_n)
        return [(_pretty_feature(f), "alcista" if contrib[f] > 0 else "bajista")
                for f in ranked.index if abs(contrib[f]) > 1e-6]

    def predict(self, X: pd.DataFrame) -> dict:
        """X: DataFrame de features (normalmente la última fila = 'ahora').

        Devuelve arrays: p_up calibrada y retornos log forward (mediana y
        percentiles 20/80). Si los cuantiles lineales fallan o se cruzan, se
        usa una banda heurística simétrica basada en el ATR actual.
        """
        Xdf = X[self.feature_names]
        Xv = Xdf.values
        p_lg = self.logit.predict_proba(Xv)[:, 1]
        p_gb = self.gbm.predict_proba(Xdf)[:, 1]  # gbm entrenado con nombres
        p = np.clip((1 - self.w_gbm) * p_lg + self.w_gbm * p_gb, 1e-6, 1 - 1e-6)
        a, b = self.platt
        p_up = 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))
        out = {"p_up": p_up}
        use_q = (self.q_med is not None and self.q_med.model is not None)
        if use_q:
            r_med = self.q_med.predict(Xv)
            r_lo = self.q_lo.predict(Xv)
            r_hi = self.q_hi.predict(Xv)
            # si un cuantil lineal se cruza de forma absurda, caemos a la banda ATR
            use_q = bool(np.all(r_lo <= r_med) and np.all(r_med <= r_hi))
        if use_q:
            out["ret_med"], out["ret_lo"], out["ret_hi"] = r_med, r_lo, r_hi
        else:  # heurística sin cuantiles: banda +/- ~1 sigma estimada por ATR
            half = self.atr_rel_now * np.sqrt(self.horizon)
            out["ret_med"] = (p_up - 0.5) * 2 * half
            out["ret_lo"] = -half
            out["ret_hi"] = +half
        return out


def train_coin_predictor(X: pd.DataFrame, y_tri: pd.Series, y_fwd: pd.Series,
                         horizon: int, atr_rel_now: float,
                         cfg: PredictorConfig) -> Optional[CoinPredictor]:
    """Entrena clasificador calibrado + cuantiles para una moneda.

    Validación honesta: walk-forward purgado (etiquetas miran `horizon` velas
    ahead; se elimina todo solapamiento train/test).
    """
    mask_inf = (y_tri != 0).values
    y_bin = (y_tri > 0).astype(int).values
    tr_all = np.arange(len(X))[mask_inf]
    if len(tr_all) < 800:
        log.warning("Historial insuficiente (%d muestras informativas)", len(tr_all))
        return None

    gap = max(horizon * 2, 12)
    wf = PurgedWalkForward(n_splits=3, gap=gap)
    accs_logit, accs_gbm, ens_p, ens_y = [], [], [], []
    for tr_i, va_i, _te in wf.split(len(X)):
        tr_i = tr_i[mask_inf[tr_i]]
        va_i = va_i[mask_inf[va_i]]
        if len(tr_i) < 400 or len(va_i) < 80:
            continue
        Xtr, ytr = X.iloc[tr_i], y_bin[tr_i]
        Xva, yva = X.iloc[va_i], y_bin[va_i]
        lg = LogisticRegression(max_iter=2000, C=0.2, random_state=cfg.random_state)
        lg.fit(Xtr.values, ytr)
        gb = HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                            learning_rate=0.05,
                                            random_state=cfg.random_state)
        gb.fit(Xtr, ytr)
        p_lg = lg.predict_proba(Xva.values)[:, 1]
        p_gb = gb.predict_proba(Xva)[:, 1]
        accs_logit.append(float(((p_lg > 0.5) == (yva == 1)).mean()))
        accs_gbm.append(float(((p_gb > 0.5) == (yva == 1)).mean()))
        ens_p.append(0.5 * (p_lg + p_gb))
        ens_y.append(yva)

    if not ens_p:
        return None
    ens_p_cat = np.concatenate(ens_p)
    ens_y_cat = np.concatenate(ens_y)
    acc_oos = float(((ens_p_cat > 0.5) == (ens_y_cat == 1)).mean())
    a, b = _platt(ens_p_cat, ens_y_cat, prior=float(ens_y_cat.mean()))

    # Ajuste final sobre todo el histórico informativo
    lg_f = LogisticRegression(max_iter=2000, C=0.2, random_state=cfg.random_state)
    lg_f.fit(X.iloc[tr_all].values, y_bin[tr_all])
    gb_f = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05,
                                          random_state=cfg.random_state)
    gb_f.fit(X.iloc[tr_all], y_bin[tr_all])
    ml, mg = float(np.mean(accs_logit)), float(np.mean(accs_gbm))
    denom = (ml - 0.48) + (mg - 0.48)
    w_gbm = float(np.clip((mg - 0.48) / denom, 0.25, 0.75)) if denom > 0 else 0.5

    # --- Explicabilidad: coefs del Logit (escala z) + importancia permutacional GBM
    logit_coefs = pd.Series(lg_f.coef_[0], index=list(X.columns))
    gbm_imp = None
    try:
        from sklearn.inspection import permutation_importance
        va_i = tr_all[-min(600, len(tr_all) // 4):]  # holdout temporal reciente
        if len(va_i) >= 100:
            res = permutation_importance(
                gb_f, X.iloc[va_i], y_bin[va_i], n_repeats=4, random_state=cfg.random_state,
                scoring="neg_log_loss", n_jobs=1)
            gbm_imp = np.clip(res.importances_mean, 0, None)
    except Exception as exc:
        log.debug("permutation_importance no disponible: %s", exc)

    # Cuantiles del retorno forward (solo en tramos informativos, sin neutros raros)
    Xtr_q, yf_tr = X.iloc[tr_all], y_fwd.values[tr_all]
    q_med, q_lo, q_hi = (_QuantileModel(0.5), _QuantileModel(cfg.quantiles[0]),
                         _QuantileModel(cfg.quantiles[1]))
    ok = q_med.fit(Xtr_q, yf_tr)
    if ok:
        q_lo.fit(Xtr_q, yf_tr)
        q_hi.fit(Xtr_q, yf_tr)
    else:
        q_med = q_lo = q_hi = None

    stats = {
        "acc_oos": round(acc_oos, 4),
        "acc_logit": round(ml, 4),
        "acc_gbm": round(mg, 4),
        "n_train": int(len(tr_all)),
        "w_gbm": round(w_gbm, 3),
        "base_rate_up": round(float(ens_y_cat.mean()), 3),
        "quantiles_ok": bool(ok),
    }
    return CoinPredictor(lg_f, gb_f, w_gbm, (a, b), q_med, q_lo, q_hi,
                         list(X.columns), horizon, atr_rel_now,
                         gbm_imp=gbm_imp, logit_coefs=logit_coefs, stats=stats)


# ---------------------------------------------------------------------------
# Decisión humana: COMPRAR / ESPERAR / VENDER (spot, sin apalancamiento)
# ---------------------------------------------------------------------------

def confidence_label(acc_oos: float, n_train: int) -> str:
    """Qué tan confiable es esta señal según su pasado (out-of-sample)."""
    if acc_oos >= 0.56 and n_train >= 3000:
        return "ALTA"
    if acc_oos >= 0.54 and n_train >= 1500:
        return "MEDIA"
    if acc_oos >= 0.52:
        return "BAJA"
    return "MUY BAJA"


def decide(p_up: float, ret_net: float, acc_oos: float,
           cfg: PredictorConfig) -> tuple[str, str]:
    """Acción + motivo legible. VENDER = reduce/sale si la tienes (spot real)."""
    reasons = []
    if p_up >= cfg.buy_prob and ret_net > 0:
        action = "COMPRAR"
        reasons.append(f"P(alza {cfg.primary_horizon}h)={p_up:.0%}, "
                       f"retorno esperado neto +{ret_net:.2%}")
    elif p_up <= cfg.sell_prob and ret_net < 0:
        action = "VENDER"
        reasons.append(f"P(alza {cfg.primary_horizon}h)={p_up:.0%}, "
                       f"retorno esperado neto {ret_net:.2%}")
    else:
        action = "ESPERAR"
        reasons.append(f"señal débil (P(alza)={p_up:.0%} cerca de 50%)")
    if acc_oos < cfg.min_acc_oos:
        reasons.append(f"exactitud histórica OOS pobre ({acc_oos:.1%}): "
                       f"no conviene operar esta moneda")
    return action, " · ".join(reasons)


# ---------------------------------------------------------------------------
# Pipeline por moneda y watchlist completa
# ---------------------------------------------------------------------------

def predict_symbol(symbol: str, cfg: PredictorConfig,
                   df: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """Predicción completa para UNA moneda en todos los horizontes.

    Devuelve dict con: precio actual, por horizonte {p_up, retorno mediana,
    rango de precio}, y decisión del horizonte primario. None si hay pocos datos.
    Si `df` se proporciona, se reutiliza (evita recargas desde el dashboard).
    """
    if df is None:
        df = load_dataset(symbol=symbol, interval=cfg.interval, days=cfg.days,
                          cfg=DataConfig(symbol=symbol, interval=cfg.interval,
                                         cache_dir=cfg.cache_dir,
                                         use_network=cfg.use_network))
    if len(df) < cfg.min_bars:
        log.warning("%s: histórico insuficiente (%d velas)", symbol, len(df))
        return None

    X_full = build_features(df)
    bph = cfg.bars_per_hour()
    horizons_bars = sorted({max(1, int(round(h * bph))) for h in cfg.horizons_hours})
    primary_bars = max(1, int(round(cfg.primary_horizon * bph)))
    y_full = make_labels(df, horizons=horizons_bars)
    max_h = max(horizons_bars)
    y_cut = y_full.iloc[:-max_h]
    X_cut = X_full.loc[y_cut.index]
    atr_rel_now = float((_atr(df) / df["close"]).iloc[-1])

    price_now = float(df["close"].iloc[-1])
    result = {"symbol": symbol, "price_now": price_now, "interval": cfg.interval,
              "n_bars": int(len(df)), "horizons": {}}

    for h in horizons_bars:
        pred = train_coin_predictor(X_cut, y_cut[f"y_tri_{h}"], y_cut[f"y_fwd_{h}"],
                                    h, atr_rel_now, cfg)
        if pred is None:
            continue
        # Predice sobre la última vela CERRADA (X_full incluye esa fila extra
        # que no tiene etiqueta; sus features solo usan datos pasados).
        X_last = X_full.iloc[[-1]]
        out = pred.predict(X_last)
        p_up = float(np.asarray(out["p_up"]).ravel()[0])
        r_med = float(np.asarray(out["ret_med"]).ravel()[0])
        r_lo = float(np.asarray(out["ret_lo"]).ravel()[0])
        r_hi = float(np.asarray(out["ret_hi"]).ravel()[0])
        # ordenamos por si los cuantiles/heurística quedaran invertidos
        r_lo, r_hi = min(r_lo, r_hi), max(r_lo, r_hi)
        # retorno esperado aproximado: media de la banda ponderada por p_up
        exp_ret = float(r_lo * (1 - p_up) + r_hi * p_up)
        net = exp_ret - cfg.fee_roundtrip
        result["horizons"][h] = {
            "horizon_hours": int(round(h / bph)),
            "p_up": round(p_up, 4),
            "exp_return": round(exp_ret, 5),
            "exp_return_net_fee": round(net, 5),
            "price_range": [round(price_now * np.exp(r_lo), 8),
                            round(price_now * np.exp(r_hi), 8)],
            "price_median": round(price_now * np.exp(r_med), 8),
            "acc_oos": pred.stats["acc_oos"],
            "confidence": confidence_label(pred.stats["acc_oos"], pred.stats["n_train"]),
            "drivers": pred.drivers(X_last),
            "stats": pred.stats,
        }

    if not result["horizons"]:
        return None
    h_used = primary_bars if primary_bars in result["horizons"] \
        else min(result["horizons"], key=lambda hh: abs(hh - primary_bars))
    result["primary"] = {"horizon_used": h_used, **result["horizons"][h_used]}
    prim = result["primary"]
    action, reason = decide(prim["p_up"], prim["exp_return_net_fee"],
                            prim["acc_oos"], cfg)
    result["action"] = action
    result["reason"] = reason
    # drivers legibles del horizonte principal: {"RSI(14)": "bajista", ...}
    result["highlights"] = {f: d for f, d in prim.get("drivers", [])}
    # score para ranking: ventaja neta ajustada por confianza histórica
    conf_w = {"ALTA": 1.0, "MEDIA": 0.7, "BAJA": 0.4, "MUY BAJA": 0.1}[prim["confidence"]]
    result["score"] = round(prim["exp_return_net_fee"] * conf_w, 5)
    return result


def predict_watchlist(cfg: Optional[PredictorConfig] = None,
                      with_frames: bool = False):
    """Tabla resumen de todas las monedas, ordenada por atractivo de compra.

    Con with_frames=True devuelve también (tabla, resultados_completos, dfs)
    para que el dashboard no vuelva a descargar/entrenar."""
    cfg = cfg or PredictorConfig()
    rows, results, frames = [], [], {}
    for sym in cfg.symbols:
        t0 = time.time()
        try:
            df = load_dataset(symbol=sym, interval=cfg.interval, days=cfg.days,
                              cfg=DataConfig(symbol=sym, interval=cfg.interval,
                                             cache_dir=cfg.cache_dir,
                                             use_network=cfg.use_network))
            r = predict_symbol(sym, cfg, df=df)
        except Exception as exc:
            log.error("%s: fallo (%s)", sym, exc)
            continue
        if r is None:
            continue
        frames[sym] = df
        results.append(r)
        prim = r["primary"]
        drivers_txt = ", ".join(f"{f} {d}" for f, d in prim.get("drivers", []))
        rows.append({
            "moneda": sym,
            "precio_actual": r["price_now"],
            "accion": r["action"],
            "p_alza": prim["p_up"],
            "horizon_h": prim["horizon_hours"],
            "retorno_esperado_%": round(prim["exp_return"] * 100, 2),
            "retorno_neto_%": round(prim["exp_return_net_fee"] * 100, 2),
            "precio_estimado": prim["price_median"],
            "rango_p20_p80": f"{prim['price_range'][0]:.6g} – {prim['price_range'][1]:.6g}",
            "confianza": prim["confidence"],
            "acc_oos_%": round(prim["acc_oos"] * 100, 1),
            "drivers": drivers_txt,
            "score": r["score"],
            "motivo": r["reason"],
        })
        log.info("%s procesada en %.1fs", sym, time.time() - t0)
    df_out = pd.DataFrame(rows)
    if len(df_out):
        df_out = df_out.sort_values("score", ascending=False).reset_index(drop=True)
    if with_frames:
        return df_out, results, frames
    return df_out
