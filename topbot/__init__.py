"""TopBot - Bot de trading cripto con predicción mejorada.

Módulos:
    data       -> Descarga, caché y validación de datos OHLCV (Binance).
    features   -> Ingeniería de características (indicadores técnicos robustos).
    model      -> Modelo de predicción ensemble + calibración de probabilidades.
    backtest   -> Motor de backtesting con costes realistas y métricas.
    risk       -> Gestión de riesgo (tamaño de posición, SL/TP dinámicos).
    bot        -> Orquestador del bot (señales -> decisiones -> registro).
"""

__version__ = "2.0.0"
