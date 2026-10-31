"""Modelo de prediccion: ensemble calibrado con validacion purgada (walk-forward).

Mejoras frente a un clasificador unico entrenado al azar sobre todo el historico:
  * Walk-forward con PURGE y EMBARGO entre train/test: las etiquetas miran h velas
    hacia el futuro, por lo que las muestras cercanas a la frontera contaminan
    el test si no se purgan ( fuga de datos sutil y muy comun en bots ).
  * Ensemble de modelos diversos: LogisticRegression (senales lineales suaves),
    GradientBoosting y RandomForest (no lineal). Promedio ponderado por precision
    estimada en validacion interna.
  * Calibracion de probabilidades (Platt scaling) en el conjunto de validacion:
    una senal "0.9" solo se usa para dimensionar riesgo si realmente implica ~90%.
  * Metrica de seleccion honesta: accuracy sobre muestras informativas (triple
    barrier != 0) + log-loss calibrado + ganancia bruta esperada por trade.
  * Reentrenamiento periodico automatico (cada `retrain_every` velas en backtest,
    o bajo demanda en vivo) -> adapta el modelo a cambios de regimen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

log = logging.getLogger("topbot.model")


@dataclass
class ModelConfig:
    horizon: int = 4              # velas ahead que se predice
    k_atr: float = 0.5            # umbral triple-barrier (en etiquetas)
    purge_gap: int = 12           # velas de separacion train->valid->test
    n_splits: int = 4             # folds walk-forward para evaluar/purgar
    prob_threshold: float = 0.55  # umbral minimo de probabilidad para operar
    retrain_every: int = 1000     # velas entre reentrenamientos en produccion
    random_state: int = 7


def _base_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logit": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.2, random_state=seed)),
        ]),
        "gbm": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=seed),
        "rf": RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=40,
            n_jobs=-1, random_state=seed),
    }


def _platt(p: np.ndarray, y: np.ndarray, prior: float) -> tuple[float, float]:
    """Regresion logistica 1D (escala+desplazamiento) sobre logits -> calibracion."""
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(x, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


class PurgedWalkForward:
    """Divide [start:end] en folds temporales con purge+embargo.

    Cada fold: train = todo lo anterior a (valid_test_inicio - gap),
    despues valida y finalmente testea; nunca hay solapamiento de etiquetas.
    """

    def __init__(self, n_splits: int, gap: int):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, n: int):
        usable = n // 2  # primera mitad: warmup de indicadores/ventanas
        span = n - usable
        fold = span // (self.n_splits + 1)
        for i in range(self.n_splits):
            te0 = usable + (i + 1) * fold
            te1 = te0 + fold if i < self.n_splits - 1 else n
            va0 = te0 - fold // 2 - self.gap
            va1 = te0 - self.gap
            tr1 = va0 - self.gap
            if tr1 < 200 or va1 - va0 < 50:
                continue
            yield np.arange(0, tr1), np.arange(va0, va1), np.arange(te0, te1)


@dataclass
class TrainedEnsemble:
    models: dict
    weights: dict
    platt: tuple[float, float]
    feature_names: list[str]
    horizon: int
    val_stats: dict = field(default_factory=dict)

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X[self.feature_names].values
        p = np.zeros(len(Xv))
        wsum = 0.0
        for name, mdl in self.models.items():
            w = self.weights[name]
            if w <= 0:
                continue
            p += w * mdl.predict_proba(Xv)[:, 1]
            wsum += w
        p /= max(wsum, 1e-9)
        a, b = self.platt
        z = np.clip(p, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(a * np.log(z / (1 - z)) + b)))


def train(X: pd.DataFrame, y_tri: pd.Series, cfg: ModelConfig,
          final_fit: bool = True) -> Optional[TrainedEnsemble]:
    """Entrena el ensemble con validacion purgada; devuelve modelo calibrado.

    y_tri: etiquetas triple-barrier (-1,0,1). Se convierten a binaria usando solo
    muestras informativas (|y|=1); el neutro se descarta del entrenamiento.
    """
    mask_inf = (y_tri != 0).values
    y_bin = (y_tri > 0).astype(int)
    idx_all = np.arange(len(X))

    scores: dict[str, list[float]] = {k: [] for k in _base_models(cfg.random_state)}
    weights_default = {"logit": 1.0, "gbm": 1.5, "rf": 1.2}
    p_va_all, y_va_all = [], []

    wf = PurgedWalkForward(cfg.n_splits, gap=max(cfg.horizon * 2, cfg.purge_gap))
    for tr_i, va_i, _te in wf.split(len(X)):
        tr_i = tr_i[mask_inf[tr_i]]
        va_i = va_i[mask_inf[va_i]]
        if len(tr_i) < 300 or len(va_i) < 60:
            continue
        Xtr, ytr = X.iloc[tr_i], y_bin.iloc[tr_i]
        Xva, yva = X.iloc[va_i], y_bin.iloc[va_i]
        pv_this = []
        for name, mdl in _base_models(cfg.random_state).items():
            mdl.fit(Xtr, ytr)
            p = mdl.predict_proba(Xva)[:, 1]
            acc = float(((p > 0.5) == (yva.values == 1)).mean())
            scores[name].append(acc)
            pv_this.append(p)
        # ensemble promedio simple en validacion para medir su accuracy
        ens = np.mean(pv_this, axis=0)
        scores.setdefault("ens", []).append(float(((ens > 0.5) == (yva.values == 1)).mean()))
        p_va_all.append((ens, yva.values))

    if not p_va_all:
        log.warning("Validacion insuficiente; no se entrena modelo")
        return None

    weights = {}
    for name in scores:
        if name == "ens" or not scores[name]:
            continue
        acc = float(np.mean(scores[name]))
        # peso proporcional a la ventaja sobre moneda (0.5), acotado
        weights[name] = max(0.0, (acc - 0.48)) * weights_default[name]
    total = sum(weights.values())
    if total <= 0:
        weights = {k: weights_default[k] / 3 for k in weights_default}
    else:
        weights = {k: v / total for k, v in weights.items()}

    ens_acc = float(np.mean(scores.get("ens", [0.5])))
    ens_p = np.concatenate([p for p, _ in p_va_all])
    ens_y = np.concatenate([yv for _, yv in p_va_all])
    try:
        ll = float(log_loss(ens_y, np.clip(ens_p, 1e-6, 1 - 1e-6)))
    except Exception:
        ll = float("nan")
    a, b = _platt(ens_p, ens_y, prior=float(ens_y.mean()))

    # Ajuste final sobre TODAS las muestras informativas disponibles
    trained = {}
    if final_fit:
        all_tr = idx_all[mask_inf]
        for name in weights:
            mdl = _base_models(cfg.random_state)[name]
            mdl.fit(X.iloc[all_tr], y_bin.iloc[all_tr])
            trained[name] = mdl
    else:
        trained = _base_models(cfg.random_state)

    stats = {"val_acc_ens": ens_acc, "val_logloss": ll,
             "per_model_acc": {k: round(float(np.mean(v)), 4)
                               for k, v in scores.items() if v and k != "ens"},
             "weights": {k: round(v, 3) for k, v in weights.items()},
             "n_train": int(mask_inf.sum())}
    log.info("Entrenado: acc=%.3f logloss=%.3f pesos=%s",
             ens_acc, ll, {k: round(v, 2) for k, v in weights.items()})
    return TrainedEnsemble(trained, weights, (a, b), list(X.columns),
                           cfg.horizon, stats)


def signal(prob_up: np.ndarray, threshold: float) -> np.ndarray:
    """Convierte probabilidad calibrada en senal {-1, 0, +1}.

    Usa un doble umbral simetrico: operar solo cuando la probabilidad es
    suficientemente alejada de 0.5 (edge esperado > costes).
    """
    sig = np.zeros(len(prob_up), dtype=int)
    sig[prob_up >= threshold] = 1
    sig[prob_up <= 1 - threshold] = -1
    return sig
