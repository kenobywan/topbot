"""CLI del predictor multi-moneda: qué comprar/vender hoy en spot.

Ejemplos:
  python scripts/predict.py --online                       # watchlist completa, datos reales
  python scripts/predict.py --symbols BTCUSDT ETHUSDT --online
  python scripts/predict.py --interval 4h --horizon 24     # señal a 24h con velas 4h
  python scripts/predict.py                                # modo offline (sintético/caché)
Salidas: tabla por consola + predictions.csv + predictions.json (histórico auditable).
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from topbot.predictor import PredictorConfig, predict_symbol, predict_watchlist

ap = argparse.ArgumentParser(description="Predictor de precio para compra/venta manual en spot")
ap.add_argument("--symbols", nargs="*", default=None,
                help="monedas (ej: BTCUSDT ETHUSDT). Por defecto watchlist completa")
ap.add_argument("--interval", default="1h", choices=["15m", "1h", "4h", "1d"])
ap.add_argument("--days", type=int, default=730)
ap.add_argument("--horizon", type=int, default=24,
                help="horizonte principal en HORAS que manda la señal (4, 24, 72, 168)")
ap.add_argument("--online", action="store_true", help="descargar datos reales de Binance")
ap.add_argument("--detail", action="store_true",
                help="guardar además el detalle multi-horizonte (4h/1d/3d/7d) en JSONL")
ap.add_argument("--csv", default="predictions.csv")
ap.add_argument("--json-out", default="logs/predictions.jsonl")
args = ap.parse_args()

cfg = PredictorConfig(interval=args.interval, days=args.days,
                      primary_horizon=args.horizon, use_network=args.online)
if args.symbols:
    cfg.symbols = [s.upper() for s in args.symbols]

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
print(f"\n=== PREDICTOR TOPBOT — {now} | velas {cfg.interval} | "
      f"señal principal: {cfg.primary_horizon}h ===\n")

tabla = predict_watchlist(cfg)
if len(tabla) == 0:
    print("Sin resultados (¿historial insuficiente o sin red? prueba --online).")
    sys.exit(1)

EMOJIS = {"COMPRAR": "🟢", "ESPERAR": "🟡", "VENDER": "🔴"}

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
view = tabla.copy()
view["accion"] = view["accion"].map(lambda a: f"{EMOJIS.get(a, '')} {a}")
print(view[["moneda", "precio_actual", "accion", "p_alza", "horizon_h",
            "retorno_esperado_%", "retorno_neto_%", "precio_estimado",
            "rango_p20_p80", "confianza", "acc_oos_%"]].to_string(index=False))
print("\nMotivos:")
for _, r in tabla.iterrows():
    print(f"  {r['moneda']:<10} {r['accion']:<8} {r['motivo']}")

# Persistencia: CSV (repaso rápido) + JSONL append (histórico auditable)
tabla.to_csv(args.csv, index=False)
os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
detail = []
if args.detail:  # recalcula con todos los horizontes para el registro completo
    for sym in cfg.symbols:
        try:
            r = predict_symbol(sym, cfg)
            if r:
                detail.append(r)
        except Exception as exc:
            logging.getLogger("predict").error("%s: %s", sym, exc)
with open(args.json_out, "a") as fh:
    fh.write(json.dumps({"ts": now, "interval": cfg.interval,
                         "primary_horizon": cfg.primary_horizon,
                         "summary": tabla.to_dict(orient="records"),
                         "detail": detail}, default=str) + "\n")
print(f"\nGuardado: {args.csv} y {args.json_out}")
print("\n⚠️  No es asesoramiento financiero. Las probabilidades están validadas "
      "solo en el pasado;\nel mercado puede cambiar de régimen. Empieza por las de confianza ALTA/MEDIA.")
