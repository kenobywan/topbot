# TOPBOT Predictor 📈

Predictor de precio multi-moneda para **decisión manual en Spot** (no ejecuta órdenes).
Por cada moneda devuelve: probabilidad calibrada de alza, precio estimado, rango
p20–p80, retorno neto de comisiones y una acción clara: 🟢 COMPRAR / 🟡 ESPERAR / 🔴 VENDER.

## Uso rápido

```bash
pip install -r requirements.txt            # pandas, scikit-learn, matplotlib, plotly...
python scripts/predict.py --online --dashboard   # datos reales + tabla + gráficos
python scripts/predict.py --symbols BTCUSDT ETHUSDT --online
python scripts/predict.py --interval 4h --horizon 24 --online
```

Salidas:
* Consola: ranking por ventaja esperada neta + motivos + drivers (RSI, MACD, momentum…)
* `predictions.csv` / `logs/predictions.jsonl`: histórico auditable
* `reports/predictions.html`: dashboard interactivo (tabla clicable + velas + banda prevista)
* `reports/prices.png`: panel de gráficos por moneda (velas recientes, proyección y rango)

### Requisitos en Windows (PowerShell)

```powershell
py -m pip install -r requirements.txt      # si "python" no funciona, usa "py"
cd topbot                                  # carpeta que contiene bot.py
py bot.py                                  # ya funciona la ejecución directa
# o desde la raíz del proyecto:
py scripts\predict.py --online --dashboard
```

Nota: `bot.py` está dentro de la carpeta `topbot/`. Si lo ejecutas directamente
(`python bot.py`) ahora funciona sin errores de import; también puedes usar
`python -m topbot.bot` o `python scripts/run_bot.py` desde la raíz.

## Cómo decide (mejoras frente a la versión anterior)

1. **Sin look-ahead**: walk-forward purgado con gap ≥ 2×horizonte; predice solo sobre vela cerrada.
2. **Probabilidades calibradas**: ensemble Logit+HistGBM ponderado por exactitud OOS + Platt.
3. **Rango honesto**: cuantiles 20/50/80 del retorno forward (fallback: banda ±ATR·√t).
4. **Neto de fees**: comisión spot ida/vuelta (0.20%) restada antes de recomendar comprar.
5. **Confianza = validez histórica**: señal ALTA/MEDIA/BAJA según acc OOS real de esa moneda.
6. **Explicación**: top-3 drivers con dirección (excluye features cíclicas no direccionales).

⚠️ No es asesoramiento financiero: las probabilidades se validan solo en el pasado.
