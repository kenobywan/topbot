# -*- coding: utf-8 -*-
"""
CryptoBot Top 100 — v2
Analiza el top 100 de criptomonedas con datos en vivo y genera señales de
COMPRA / MANTENER / VENTA.

Fuentes de datos:
  - CoinGecko  : precios, volumen, sparkline 7d, histórico 30d (precio+volumen)
  - OKX        : derivados estilo CoinGlass -> funding rate, open interest,
                 ratio long/short, volumen comprador/vendedor (taker)

Indicadores:
  - RSI(14) horario, SMA 24h/72h, posición en rango 7d, volatilidad
  - Soportes y resistencias por pivotes (con nº de toques)
  - Flujo de capital 30d: OBV, volumen bruto y divergencias precio/volumen
  - Funding rate y long/short como medida de sentimiento (contrarian)

NO es asesoramiento financiero. Herramienta educativa.
"""

import json
import math
import os
import sys

# Windows: fuerza UTF-8 en la consola para evitar errores de codificación
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

from flask import Flask, jsonify, send_file

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "analisis.json")

CG_MARKETS = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=150&page=1"
    "&sparkline=true&price_change_percentage=1h%2C24h%2C7d%2C30d"
)
COINBASE = "https://api.exchange.coinbase.com"

# ⭐ FAVORITAS: monedas que se analizan SIEMPRE aunque no estén en el top 100.
# Añade aquí los ids de CoinGecko que quieras vigilar (el id sale en la URL
# de la moneda en coingecko.com, p.ej. coingecko.com/es/monedas/virtual-protocol)
WATCHLIST = ["virtual-protocol", "raydium"]

CG_MARKETS_IDS = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&ids={ids}"
    "&sparkline=true&price_change_percentage=1h%2C24h%2C7d%2C30d"
)
CG_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart?vs_currency=usd&days=90"

OKX = "https://www.okx.com"

REFRESH_SECONDS = 300          # ciclo principal (top 100)
TOPE_BONO_CAIDA = 22           # tope conjunto de los bonos "por caer" (anti doble conteo)
DETALLE_CACHE_SEG = 900        # caché del análisis profundo por moneda (15 min)

STABLECOINS = {
    "tether", "usd-coin", "dai", "binance-usd", "true-usd", "usdd",
    "frax", "paxos-standard", "gemini-dollar", "first-digital-usd",
    "ethena-usde", "usds", "paypal-usd", "tether-gold", "pax-gold",
    "usd1-wlfi", "falcon-finance", "susds",
}

# Tokens que NO son monedas operables: wrapped, staking líquido, fondos tokenizados
_PATRONES_DERIVADO = ("wrapped", "staked", "restaked", "bridged", "-peg", "tokenized", "usyc")
_NOMBRES_DERIVADO = ("fund", "wrapped", "staked", "etf", "treasury", "t-bill", "money market", "yield", "heloc")
_IDS_DERIVADO = {"weth", "solv-btc", "binance-bitcoin", "kelp-dao-restaked-eth", "rocket-pool-eth"}

# Monedas cuyo rank de CoinGecko está inflado respecto al top 100 real de
# CoinMarketCap (capitalización dudosa / poco float). Nunca se analizan.
EXCLUIDAS = {"htx-dao"}


def es_derivado(m):
    """True si el token es una versión wrapped/staked de otro activo o un fondo tokenizado."""
    cid, nombre = m.get("id", ""), (m.get("name") or "").lower()
    return (cid in _IDS_DERIVADO
            or any(p in cid for p in _PATRONES_DERIVADO)
            or any(p in nombre for p in _NOMBRES_DERIVADO))


app = Flask(__name__)
VERSION = "8.5"
_estado = {"monedas": [], "actualizado": None, "error": None, "version": VERSION}
_derivados = {}                 # simbolo -> {funding, oi_usd, long_short}
_detalle_cache = {}             # coin_id -> (timestamp, dict)
_lock = threading.Lock()


def http_json(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "CryptoBot/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ----------------------------- Indicadores ------------------------------

def rsi(precios, periodo=14, ventana=None):
    """
    RSI clasico de Wilder calculado sobre una VENTANA acotada.

    FALLO CORREGIDO (el mas grave del predictor): el cribado y el refinado
    llamaban a rsi(cierres) con las 200-300 velas horarias completas. El RSI es
    un oscilador de 14 periodos; darle 300 velas no lo hace mas fiable, lo
    aplana: la media de ganancias/perdedores queda dominada por el promedio de
    dos semanas y se queda pegada a un valor fijo sea cual sea la accion
    reciente. Comprobado con datos sinteticos: una moneda lateral perfecta daba
    RSI 4 y nota de VENTA, y una que sube sin parar tambien puntuaba como VENTA.
    Con ventana = 4*periodo el indicador mira lo que debe mirar (~56 horas) y
    deja de inventar señales.
    """
    if not precios or len(precios) < periodo + 1:
        return None
    n = ventana if ventana else 4 * periodo
    serie = precios[-(n + 1):] if len(precios) > n + 1 else precios
    deltas = [serie[i] - serie[i - 1] for i in range(1, len(serie))]
    g = [max(d, 0) for d in deltas]
    p = [max(-d, 0) for d in deltas]
    ag = sum(g[:periodo]) / periodo
    ap = sum(p[:periodo]) / periodo
    for i in range(periodo, len(deltas)):
        ag = (ag * (periodo - 1) + g[i]) / periodo
        ap = (ap * (periodo - 1) + p[i]) / periodo
    if ap <= 1e-12:
        return 100.0
    if ag <= 1e-12:
        return 0.0
    return 100 - 100 / (1 + ag / ap)


def sma(precios, n):
    return sum(precios[-n:]) / n if len(precios) >= n else None


def macd_hist(precios):
    """
    Histograma MACD(12,26,9) sobre la serie horaria, normalizado como % del precio.
    Devuelve (valor_actual, valor_hace_6h) para saber si el momentum crece o decae.
    """
    if len(precios) < 45:
        return None
    k12, k26, k9 = 2 / 13, 2 / 27, 2 / 10
    e12 = e26 = None
    macds = []
    for i, p in enumerate(precios):
        e12 = p if e12 is None else p * k12 + e12 * (1 - k12)
        e26 = p if e26 is None else p * k26 + e26 * (1 - k26)
        if i >= 26:
            macds.append(e12 - e26)
    sig, hists = None, []
    for mv in macds:
        sig = mv if sig is None else mv * k9 + sig * (1 - k9)
        hists.append(mv - sig)
    if len(hists) < 8:
        return None
    # FALLO CORREGIDO: se comparaba la barra del histograma contra la de hace 6h
    # SIN suavizar. En velas horarias esa barra salta de signo de una hora a otra
    # (ruido), asi que "MACD creciendo/decaiendo" era casi una moneda al aire y
    # corria el riesgo de comprar/venir en contra de momentum real. Se compara
    # ahora una media movil de 3 barras vs la de hace 6h: misma idea, mucho mas
    # estable (menos señales falsas de giro).
    def _m3(i):
        seg = hists[max(0, i - 2):i + 1]
        return sum(seg) / len(seg)
    sm_act = _m3(len(hists) - 1)
    sm_ant = _m3(len(hists) - 7)
    return sm_act / precios[-1] * 100, sm_ant / precios[-1] * 100


def velas_1h_okx(simbolo, n=120):
    """
    Cierres horarios EN TIEMPO REAL desde OKX (spot; si no existe, futuros).
    A diferencia del sparkline de CoinGecko (que llega con ~1h de retraso),
    estas velas incluyen el movimiento de la última hora.
    Devuelve lista de cierres de antigua -> reciente, o None si no hay par.
    """
    for inst in (f"{simbolo}-USDT", f"{simbolo}-USDT-SWAP"):
        try:
            d = http_json(f"{OKX}/api/v5/market/candles?instId={inst}&bar=1H&limit={n}", timeout=10)
            velas = d.get("data") or []
            if len(velas) >= 50:
                return [float(c[4]) for c in reversed(velas)]
        except Exception:
            pass
    return None


def velas_diarias_okx(simbolo, n=220):
    """Cierres DIARIOS desde OKX (para la SMA200: CoinGecko gratis solo da 90 días)."""
    for inst in (f"{simbolo}-USDT", f"{simbolo}-USDT-SWAP"):
        try:
            d = http_json(f"{OKX}/api/v5/market/candles?instId={inst}&bar=1D&limit={n}", timeout=10)
            velas = d.get("data") or []
            if len(velas) >= 60:
                return [float(c[4]) for c in reversed(velas)]
        except Exception:
            pass
    return None


BINANCE_SPOT = "https://api.binance.com"
_binance_spot_ok = [True]      # se desactiva solo si Binance está geobloqueado


def _binance_get(path):
    """GET a Binance spot. Si el bloqueo es geográfico (451/403/418), deja de
    intentarlo el resto de la sesión y todo cae a OKX automáticamente."""
    if not _binance_spot_ok[0]:
        return None
    try:
        return http_json(f"{BINANCE_SPOT}{path}", timeout=10)
    except urllib.error.HTTPError as e:
        if e.code in (451, 403, 418, 429):
            _binance_spot_ok[0] = False
            print(f"Binance spot no disponible (HTTP {e.code}) -> velas/libro de OKX", flush=True)
        return None                      # 400 = el par no cotiza en Binance
    except Exception:
        return None


def velas_ohlc_vivo(simbolo, n=300):
    """Velas 1H EN VIVO con mechas y volumen USD, de la fuente con MÁS liquidez
    disponible: Binance primero, OKX de respaldo.
    Devuelve (máximos, mínimos, cierres, aperturas, vol_usd, fuente).
    300 velas para que el RSI tenga el mismo 'calentamiento' que TradingView."""
    k = _binance_klines(simbolo, "1h", n)
    if k and len(k) >= 50:
        return ([float(c[2]) for c in k], [float(c[3]) for c in k],
                [float(c[4]) for c in k], [float(c[1]) for c in k],
                [float(c[7]) for c in k], "Binance")
    v = velas_ohlc_okx(simbolo, n)
    return (*v, "OKX") if v else None


def _binance_klines(simbolo, interval, limit):
    k = _binance_get(f"/api/v3/klines?symbol={simbolo}USDT&interval={interval}&limit={limit}")
    return k if isinstance(k, list) and k else None


def velas_ohlc_okx(simbolo, n=300):
    for inst in (f"{simbolo}-USDT", f"{simbolo}-USDT-SWAP"):
        try:
            d = http_json(f"{OKX}/api/v5/market/candles?instId={inst}&bar=1H&limit={n}", timeout=10)
            velas = d.get("data") or []
            if len(velas) >= 50:
                velas = list(reversed(velas))
                return ([float(c[2]) for c in velas],
                        [float(c[3]) for c in velas],
                        [float(c[4]) for c in velas],
                        [float(c[1]) for c in velas],
                        [float(c[7]) if len(c) > 7 and float(c[7]) > 0
                         else float(c[5]) * float(c[4]) for c in velas])
        except Exception:
            pass
    return None


def micro_vivo(simbolo):
    """Libro de órdenes + velas 15m de Binance (más volumen = libro más
    representativo); si no está disponible o el par no cotiza, OKX."""
    d = _binance_get(f"/api/v3/depth?symbol={simbolo}USDT&limit=100")
    try:
        if d and d.get("bids") and d.get("asks"):
            bids = [(float(p), float(q)) for p, q in d["bids"]]
            asks = [(float(p), float(q)) for p, q in d["asks"]]
            mid = (bids[0][0] + asks[0][0]) / 2
            usd_b = sum(p * q for p, q in bids if p >= mid * 0.98)
            usd_a = sum(p * q for p, q in asks if p <= mid * 1.02)
            v15 = _binance_klines(simbolo, "15m", 6)
            imp = ((float(v15[-1][4]) / float(v15[-5][4]) - 1) * 100) if v15 and len(v15) >= 5 else None
            return usd_b, usd_a, imp
    except Exception:
        pass
    return micro_okx(simbolo)


def micro_okx(simbolo):
    """
    Micro-marco EN VIVO para el empuje inmediato (se recalcula cada ciclo):
      - Libro de órdenes spot de OKX: USD en muros de COMPRA vs VENTA a ±2%
        del precio (los muros de compra sostienen/empujan, los de venta frenan).
      - Velas de 15 min: ¿el precio está subiendo AHORA (última hora)?
    Devuelve (usd_compra, usd_venta, impulso_15m_pct) o None si no hay spot.
    """
    inst = f"{simbolo}-USDT"
    try:
        d = http_json(f"{OKX}/api/v5/market/books?instId={inst}&sz=100", timeout=10)
        data = (d.get("data") or [None])[0]
        if not data:
            return None
        bids = [(float(f[0]), float(f[1])) for f in data.get("bids", [])]
        asks = [(float(f[0]), float(f[1])) for f in data.get("asks", [])]
        if not bids or not asks:
            return None
        mid = (bids[0][0] + asks[0][0]) / 2
        usd_b = sum(p * q for p, q in bids if p >= mid * 0.98)
        usd_a = sum(p * q for p, q in asks if p <= mid * 1.02)
        d2 = http_json(f"{OKX}/api/v5/market/candles?instId={inst}&bar=15m&limit=6", timeout=10)
        v15 = list(reversed(d2.get("data") or []))
        imp = ((float(v15[-1][4]) / float(v15[-5][4]) - 1) * 100) if len(v15) >= 5 else None
        return usd_b, usd_a, imp
    except Exception:
        return None


def _flujo_capital(abres, cierres, vols):
    """Flujo neto de capital aproximado con velas 1h: el volumen en USD de las
    velas verdes cuenta como dinero ENTRANDO y el de las rojas como SALIENDO.
    Devuelve (neto_24h, total_24h, neto_ultima_hora)."""
    datos = list(zip(abres, cierres, vols))
    neto24 = sum(v if c >= o else -v for o, c, v in datos[-24:])
    tot24 = sum(v for _, _, v in datos[-24:])
    o1, c1, v1 = datos[-1]
    neto1h = v1 if c1 >= o1 else -v1
    return neto24, tot24, neto1h


def _fmt_musd(x):
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.2f}B$"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M$"
    return f"{x/1e3:.0f}k$"


def _pts_flujo(neto24, tot24, neto1h):
    """Si el capital ENTRA con claridad en 24h y la última hora confirma,
    anticipa presión alcista (acumulación); si SALE en ambos marcos, bajista."""
    if tot24 < 1_000_000:                      # volumen ínfimo: sin señal fiable
        return 0, None
    r = neto24 / tot24
    if r >= 0.25 and neto1h > 0:
        return 6, (f"💰 Entra capital: +{_fmt_musd(neto24)} netos comprados en 24h "
                   f"({r*100:.0f}% del volumen) y la última hora también compradora")
    if r <= -0.25 and neto1h < 0:
        return -6, (f"💸 Sale capital: {_fmt_musd(neto24)} netos vendidos en 24h "
                    f"({-r*100:.0f}% del volumen) y la última hora también vendedora")
    return 0, None


def _pts_libro(micro):
    """Muro de compras dominante + velas 15m al alza = empuje inmediato real
    (el ejemplo clásico: precio en soporte y el libro lleno de compras).
    Al revés: pared de ventas encima y 15m cayendo = freno inmediato."""
    if not micro:
        return 0, None
    usd_b, usd_a, imp = micro
    if usd_b + usd_a < 50_000 or imp is None:   # libro demasiado fino: sin señal
        return 0, None
    if usd_b >= 1.8 * usd_a and imp > 0.15:
        return 6, (f"🧲 Libro de órdenes: muros de compra {usd_b/max(usd_a,1):.1f}× mayores que las ventas "
                   f"(±2%) y velas 15m subiendo ({imp:+.1f}% en 1h): empuje inmediato al alza")
    if usd_a >= 1.8 * usd_b and imp < -0.15:
        return -6, (f"🧲 Libro de órdenes: muros de venta {usd_a/max(usd_b,1):.1f}× mayores que las compras "
                    f"(±2%) y velas 15m cayendo ({imp:+.1f}% en 1h): presión vendedora inmediata")
    return 0, None


def sr_por_mechas(highs, lows, precio_ref, ventana=2, max_n=3):
    """
    Soportes y resistencias REALES: las resistencias se sacan de los MÁXIMOS
    (mechas superiores) y los soportes de los MÍNIMOS (mechas inferiores),
    igual que se ven en un gráfico de velas. Los extremos del periodo cuentan
    aunque sean demasiado recientes para ser pivote.
    """
    if len(highs) < 2 * ventana + 3 or not precio_ref:
        return [], []
    maxs = pivotes(highs, ventana)[0]
    mins = pivotes(lows, ventana)[1]
    mx, mn = max(highs), min(lows)
    if not any(abs(x - mx) <= mx * 0.004 for x in maxs):
        maxs.append(mx)
    if not any(abs(x - mn) <= mn * 0.004 for x in mins):
        mins.append(mn)
    tol = min(0.03, max(0.004, (mx - mn) / precio_ref * 0.10))
    niv_r = agrupar_niveles(maxs, tol)
    niv_s = agrupar_niveles(mins, tol)
    sop = [x for x in niv_s if x["nivel"] < precio_ref * 0.998]
    res = [x for x in niv_r if x["nivel"] > precio_ref * 1.002]
    sop.sort(key=lambda x: -x["nivel"])
    res.sort(key=lambda x: x["nivel"])
    clave = lambda x: {"nivel": round(x["nivel"], 8), "toques": x["toques"],
                       "dist": round((x["nivel"] / precio_ref - 1) * 100, 2)}
    return [clave(x) for x in sop[:max_n]], [clave(x) for x in res[:max_n]]


def refinar_con_okx(m):
    """
    Sustituye los datos horarios RETRASADOS (sparkline de CoinGecko) por velas
    1H EN VIVO de OKX: RSI 1h, MACD, columna Hoy, soportes/resistencias con
    mechas, objetivos, stop y nota se recalculan con datos exactos del gráfico.
    Devuelve True si se pudo refinar (par en OKX y precios coherentes).
    """
    if m.get("senal") == "STABLE":
        return False
    ohlc = velas_ohlc_vivo(m["simbolo"])
    if not ohlc:
        return False
    highs, lows, cierres, abres, vols_usd, fuente = ohlc
    precio = m.get("precio")
    if not precio or abs(cierres[-1] / precio - 1) > 0.02:
        return False
    r_new = rsi(cierres)
    r4_new = rsi(cierres[3::4])
    mh_new = macd_hist(cierres)
    mh_old = tuple(m["_mh"]) if m.get("_mh") else None
    # FALLO CORREGIDO (doble conteo): el termino "_pts_rsi4h(r_new)" de la
    # version anterior usaba el RSI 1h como si fuera 4h, inflando la correccion.
    # Se usa SIEMPRE el RSI 4h real (m["rsi4h"] en el lado del cribado) y solo
    # entra en el lado nuevo si existe; si falta, aporta 0 (no se duplica 1h).
    def _bonus_4h(r4, r1h):
        if r4 is None or (r1h is not None and abs(r4 - r1h) < 1e-9):
            return 0
        return 8 if r4 <= 30 else (-8 if r4 >= 70 else 0)
    delta = ((_pts_rsi1h(r_new) + _bonus_4h(r4_new, r_new) + _pts_macd1h(mh_new, r_new))
             - (_pts_rsi1h(m.get("rsi")) + m.get("_b_rsi4", 0)
                + _pts_macd1h(mh_old, m.get("rsi"))))
    # re-aplica el tope anti doble conteo con los datos EN VIVO.
    # FALLO CORREGIDO: "_b_caida" no incluia el bono del RSI, asi que al refinar
    # se devolvan hasta 20 pts que el cribado ya habia recortado. Ahora se
    # recalcula el bono completo (RSI + 4h + rango + dip) con valores en vivo.
    b_rsi_new = max(0, _pts_rsi1h(r_new)) + max(0, _bonus_4h(r4_new, r_new))
    bono_caida_new = b_rsi_new + m.get("_b_rango_dip", 0)
    exceso_new = max(0, bono_caida_new - TOPE_BONO_CAIDA)
    delta += m.get("_exceso", 0) - exceso_new
    sop, res = sr_por_mechas(highs[-49:], lows[-49:], precio)
    # re-evalúa las reglas de proximidad S/R con los niveles EN VIVO
    pts_sr_new, raz_sr_new = _pts_proximidad_sr(sop, res)
    delta += pts_sr_new - m.get("_pts_sr", 0)
    # re-evalúa la regla anti-persecución con el cambio 3d EN VIVO
    ch3d_new = (((cierres[-1] / cierres[-73] - 1) * 100)
                if len(cierres) >= 73 and cierres[-73] else m.get("ch3d"))
    pts3d_new, raz3d_new = _pts_persecucion(ch3d_new, cierres)
    delta += pts3d_new - m.get("_pts_3d", 0)
    # 🧲 micro-marco 15m + libro de órdenes: empuje/freno inmediato EN VIVO
    pts_libro_new, raz_libro_new = _pts_libro(micro_vivo(m["simbolo"]))
    delta += pts_libro_new - m.get("_pts_libro", 0)
    # 💰 flujo de capital 24h + última hora: ¿entra o sale dinero de la moneda?
    flujo24, flujo_tot24, flujo1h = _flujo_capital(abres, cierres, vols_usd)
    pts_flujo_new, raz_flujo_new = _pts_flujo(flujo24, flujo_tot24, flujo1h)
    delta += pts_flujo_new - m.get("_pts_flujo", 0)
    obj_c = sop[0]["nivel"] if sop else None
    if obj_c and obj_c >= precio * 0.998:
        obj_c = None
    obj_v = res[0]["nivel"] if res else None
    if obj_v and obj_v <= precio * 1.002:
        obj_v = None
    stop = sop[1]["nivel"] * 0.98 if len(sop) > 1 else min(lows[-49:]) * 0.97
    # FALLO CORREGIDO: si el segundo soporte quedaba POR ENCIMA del precio (puede
    # pasar al re-medir con velas en vivo), el "stop loss" quedaba arriba: una
    # orden imposible que distorsionaba el riesgo/beneficio del detalle.
    if stop and stop >= precio:
        stop = min(lows[-49:]) * 0.97
    puntos = max(0, min(100, m["puntos"] + delta))
    if m.get("liquidez_baja"):
        puntos = min(puntos, 52.0)
    senal = senal_por_puntos(puntos)
    desc = f"📡 Velas 1h EN VIVO de {fuente} (con mechas)"
    if r_new is not None:
        desc += f": RSI 1h {r_new:.0f}"
    if r4_new is not None:
        desc += f", RSI 4h {r4_new:.0f}"
    if mh_new is not None:
        desc += ", MACD " + ("subiendo" if mh_new[0] >= mh_new[1] else "bajando")
    if delta:
        desc += f" → corrige {delta:+d} pts al dato retrasado de CoinGecko"
    razones = [z for z in m["razones"] if not _es_bullet_intradia(z)
               and not _es_bullet_sr(z) and not z.startswith("⚖️")
               and not z.startswith("🏃") and not z.startswith("🧲")
               and not z.startswith("💰") and not z.startswith("💸")]
    razones += raz_sr_new
    if raz3d_new:
        razones.append(raz3d_new)
    if raz_libro_new:
        razones.append(raz_libro_new)
    if raz_flujo_new:
        razones.append(raz_flujo_new)
    if exceso_new:
        razones.insert(0, f"⚖️ Sobreventa + rango bajo + dip cuentan la misma caída: "
                          f"bono limitado a +22 ({-exceso_new} pts)")
    razones.insert(0, desc)
    m.update({"rsi": r_new, "rsi4h": r4_new, "_exceso": exceso_new, "_pts_sr": pts_sr_new,
              "ch3d": ch3d_new, "_pts_3d": pts3d_new, "_pts_libro": pts_libro_new,
              "_pts_flujo": pts_flujo_new,
              "flujo24": round(flujo24), "flujo_tot24": round(flujo_tot24),
              "flujo1h": round(flujo1h),
              "_mh": list(mh_new) if mh_new else None,
              "hoy": calc_hoy(mh_new, r_new),
              "soportes": sop, "resistencias": res,
              "objetivo_compra": obj_c, "objetivo_venta": obj_v, "stop_loss": stop,
              "puntos": round(puntos, 1), "senal": senal,
              "accion": accion_sugerida(senal, precio, obj_c, obj_v),
              "razones": razones, "okx_vivo": True, "fuente_vivo": fuente})
    return True


def _pts_rsi1h(r):
    """Puntos que el cribado asigna al RSI 1h (para poder corregirlos en vivo)."""
    if r is None:
        return 0
    if r <= 30:
        return 20
    if r <= 40:
        return 12
    if r >= 75:
        return -20
    if r >= 65:
        return -12
    return 0


def _bonus_4h_real(r4, r1h):
    """Bono/penalizacion del RSI 4h, SOLO si es un marco realmente distinto del 1h."""
    if r4 is None or (r1h is not None and abs(r4 - r1h) < 1e-9):
        return 0
    return 8 if r4 <= 30 else (-8 if r4 >= 70 else 0)


def _pts_rsi4h(r4, r1h=None):
    """
    Puntos que el cribado asigna al RSI 4h (para corregirlos en vivo).

    FALLO CORREGIDO: antes se llamaba a esta funcion pasando el RSI 1h cuando
    no habia RSI 4h. Con RSI 1h <= 30 eso sumaba +8 EXTRA, de modo que la
    misma sobreventa se comptaba DOS VECES (+20 del RSI 1h y +8 "del 4h") y,
    al refinar con velas en vivo, el delta devolvia esos puntos fantasma a la
    nota (sesgo alcista sistematico en monedas cayendo). Ahora, si falta el
    RSI 4h, se evalua contra el RSI 1h pero SE DESCUENTAN los puntos que ese
    RSI 1h ya aporta por su propia regla: el resultado neto nunca duplica.
    """
    if r4 is None:
        return 0
    bruto = 8 if r4 <= 30 else (-8 if r4 >= 70 else 0)
    if r1h is not None and abs(r4 - r1h) < 1e-9:
        # mismo numero medido dos veces: solo cuenta lo que excede a _pts_rsi1h
        bruto -= _pts_rsi1h(r1h) if bruto > 0 else 0
        bruto = max(0, bruto) if r4 <= 30 else min(0, bruto)
    return bruto


def _pts_macd1h(mh, r=None):
    """Puntos que el cribado asigna al MACD horario (para corregirlos en vivo)."""
    if mh is None:
        return 0
    h_act, h_ant = mh
    if h_act > 0 and h_act > h_ant:
        return 6
    if h_act < 0 and h_act < h_ant:
        return -6
    if h_act > 0 and h_act < h_ant:
        return -5 if (r is not None and r >= 65) else -2
    return 0


def calc_hoy(mh, r):
    """Columna 'Hoy': ¿el intradía acompaña para ejecutar la señal hoy?"""
    hoy = "neutral"
    if mh is not None:
        if mh[0] > 0 and mh[0] >= mh[1] and (r is None or r < 70):
            hoy = "favorable"
        elif (mh[0] < 0 and mh[0] <= mh[1]) or (r is not None and r >= 75):
            hoy = "desfavorable"
    return hoy


def _pts_proximidad_sr(sops, ress):
    """
    Reglas de cercanía a soportes/resistencias y riesgo/beneficio inmediato.
    Devuelve (puntos, razones). Se usa en el cribado y se RE-EVALÚA en el
    refinado con los niveles en vivo, para que puntos y niveles siempre cuadren.
    """
    pts, raz = 0, []
    # FALLO CORREGIDO (sesgo): antes se premiaba "apoyado en soporte" sin exigir
    # que el suelo estuviera CERCA (-2%) y sin penalizar un soporte lejano.
    # Resultado: una moneda con soporte a -5% y resistencia a +3% recibia +8 por
    # el soporte y -6 por riesgo/beneficio, es decir COMPRA neta estando colgada
    # a mitad de rango. Ahora la regla es simétrica: sólo puntúa el nivel que
    # está pegado (<=2%), y un suelo lejos resta (hay recorrido de caída).
    if sops and abs(sops[0]["dist"]) <= 2 and sops[0]["toques"] >= 3:
        pts += 8; raz.append(f"Apoyado en soporte fuerte ({sops[0]['toques']} toques) a {sops[0]['dist']}%")
    elif sops and abs(sops[0]["dist"]) > 4:
        pts -= 4; raz.append(f"Suelo cercano lejano ({sops[0]['dist']}%): hay recorrido de caída hasta el primer soporte")
    if ress:
        if ress[0]["dist"] <= 2 and ress[0]["toques"] >= 3:
            pts -= 8; raz.append(f"Pegado a resistencia fuerte ({ress[0]['toques']} toques) a +{ress[0]['dist']}%")
        elif ress[0]["dist"] <= 1.5:
            pts -= 5; raz.append(f"Resistencia inmediata a +{ress[0]['dist']}%: poco margen de subida antes de chocar")
    if sops and ress:
        arriba, abajo = ress[0]["dist"], abs(sops[0]["dist"])
        if arriba > 0 and abajo > 1 and arriba < abajo * 0.7:
            pts -= 6; raz.append(f"Mal riesgo/beneficio hoy: +{arriba:.1f}% hasta la resistencia "
                                 f"frente a -{abajo:.1f}% hasta el soporte")
    return pts, raz


def _es_bullet_sr(z):
    """Detecta los bullets de proximidad S/R (para reemplazarlos en el refinado)."""
    return (z.startswith("Apoyado en soporte fuerte") or z.startswith("Pegado a resistencia fuerte")
            or z.startswith("Resistencia inmediata") or z.startswith("Mal riesgo/beneficio hoy"))


def _pendiente_cierres(precios, n=24):
    """Pendiente normalizada (%) de las ultimas n velas: sirve para saber si la
    tendencia sigue viva o ya afloja."""
    try:
        serie = precios[-n:] if len(precios) >= n else precios
        if len(serie) < 8 or not serie[0]:
            return None
        x = list(range(len(serie)))
        mx, my = sum(x) / len(x), sum(serie) / len(serie)
        num = sum((p_ - mx) * (v - my) for p_, v in zip(x, serie))
        den = sum((p_ - mx) ** 2 for p_ in x) or 1
        return (num / den) * len(serie) / my * 100
    except Exception:
        return None


def _pts_persecucion(ch3d, precios=None):
    """
    Anti-persecución: los tirones suelen durar 1-3 días sin retroceso.
    Si la moneda YA lleva 3 días subiendo fuerte Y ADEMÁS se está frenando,
    entrar ahora es llegar tarde.

    FALLO CORREGIDO: antes bastaba con que el acumulado de 3 días fuera >=15%
    para restar 8 puntos, sin mirar si el precio sigue ganando o ya afloja. El
    acumulado a 3 días se mantiene alto muchas horas después del pico, así que
    la regla castigaba de forma permanente monedas en tendencia sana (y el
    backtest lo confirmó: restaba justo cuando más probabilidades había de que
    siguiera subiendo). Ahora solo resta si el movimiento está AGOTADO: último
    día plano/bajo o pendiente claramente cediendo.
    """
    if ch3d is None:
        return 0, None
    agotado = True
    if precios and len(precios) >= 97:
        base = precios[-25]
        ult_24 = (precios[-1] / base - 1) if base else 0
        pen_hoy = _pendiente_cierres(precios, 24)
        pen_ayer = _pendiente_cierres(precios[:-24], 24)
        cede = (pen_ayer is not None and pen_hoy is not None and pen_ayer > 0
                and pen_hoy < pen_ayer * 0.6)
        agotado = (ult_24 <= 0.005) or cede
    if ch3d >= 15 and not agotado:
        return 0, (f"+{ch3d:.1f}% en 3 días con la subida aún ganando velocidad: "
                   f"tendencia viva, no se penaliza entrar")
    if ch3d >= 15:
        return -8, (f"🏃 Ya sube +{ch3d:.1f}% en 3 días y se está frenando: "
                    f"perseguir aquí es tarde, mejor esperar el retroceso")
    if ch3d >= 10:
        return -4, f"🏃 Lleva +{ch3d:.1f}% en 3 días: cuidado con perseguir la subida"
    # FALLO CORREGIDO (asimetría): no había regla espejo. Una moneda que cae
    # -15% en 3 días recibía 0 aquí mientras que una que sube +15% perdía 8 pts,
    # de modo que el marcador empujaba a comprar cuchillos en caída libre.
    if ch3d <= -15:
        rebota = False
        if precios and len(precios) >= 25:
            pen_hoy = _pendiente_cierres(precios, 24)
            rebota = pen_hoy is not None and pen_hoy > 0
        if not rebota:
            return -6, (f"🩸 Cae {ch3d:.1f}% en 3 días y sin señal de suelo: "
                        f"la inercia bajista sigue viva, no adivines el mínimo")
        return 0, f"Cae {ch3d:.1f}% en 3 días pero ya está rebotando: posible dip comprable"
    if ch3d <= -10:
        return -3, f"🩸 Lleva {ch3d:.1f}% en 3 días: posible cuchillo cayendo, espera estabilización"
    return 0, None
def _es_bullet_intradia(z):
    """Detecta los bullets de RSI 1h y MACD horario del cribado (para reemplazarlos)."""
    return (z.startswith("MACD") or z.startswith("📡")
            or z.startswith("Las barras del MACD")
            or (z.startswith("RSI ") and z[4:5].isdigit()))


def posicion_en_rango(precios):
    lo, hi = min(precios), max(precios)
    return 50.0 if hi == lo else (precios[-1] - lo) / (hi - lo) * 100


def volatilidad(precios):
    rets = [precios[i] / precios[i - 1] - 1 for i in range(1, len(precios)) if precios[i - 1] > 0]
    if len(rets) < 2:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(24 * 365) * 100


def pivotes(precios, ventana=3):
    """Detecta máximos y mínimos locales (pivotes) en la serie."""
    maxs, mins = [], []
    for i in range(ventana, len(precios) - ventana):
        seg = precios[i - ventana:i + ventana + 1]
        if precios[i] == max(seg):
            maxs.append(precios[i])
        if precios[i] == min(seg):
            mins.append(precios[i])
    return maxs, mins


def agrupar_niveles(valores, tolerancia=0.012):
    """Agrupa pivotes cercanos (±1.2%) en niveles con nº de toques."""
    niveles = []
    for v in sorted(valores):
        if niveles and abs(v - niveles[-1]["nivel"]) / niveles[-1]["nivel"] <= tolerancia:
            n = niveles[-1]
            n["nivel"] = (n["nivel"] * n["toques"] + v) / (n["toques"] + 1)
            n["toques"] += 1
        else:
            niveles.append({"nivel": v, "toques": 1})
    return niveles


def soportes_resistencias(precios, ventana=3, max_n=3, precio_ref=None):
    """
    Soportes (bajo el precio) y resistencias (sobre el precio) por pivotes.
    - Tolerancia de agrupación adaptada a la volatilidad de la moneda.
    - precio_ref: precio ACTUAL de mercado (el histórico llega con ~1h de
      retraso; sin esto, un soporte podía quedar mal clasificado).
    """
    if len(precios) < 2 * ventana + 3:
        return [], []
    maxs, mins = pivotes(precios, ventana)
    # Los extremos absolutos del periodo también son niveles reales aunque sean
    # demasiado recientes para contar como pivote (p. ej. el máximo de hoy):
    # se añaden si ningún pivote los representa ya
    mx, mn = max(precios), min(precios)
    if not any(abs(x - mx) <= mx * 0.004 for x in maxs):
        maxs.append(mx)
    if not any(abs(x - mn) <= mn * 0.004 for x in mins):
        mins.append(mn)
    rango_rel = (max(precios) - min(precios)) / precios[-1] if precios[-1] else 0
    tol = min(0.03, max(0.005, rango_rel * 0.12))
    niveles = agrupar_niveles(maxs + mins, tol)
    actual = precio_ref if precio_ref else precios[-1]
    sop = [n for n in niveles if n["nivel"] < actual * 0.998]
    res = [n for n in niveles if n["nivel"] > actual * 1.002]
    sop.sort(key=lambda n: (-n["nivel"]))                      # más cercanos primero
    res.sort(key=lambda n: (n["nivel"]))
    clave = lambda n: {"nivel": round(n["nivel"], 8), "toques": n["toques"],
                       "dist": round((n["nivel"] / actual - 1) * 100, 2)}
    return [clave(n) for n in sop[:max_n]], [clave(n) for n in res[:max_n]]


def obv(precios, volumenes):
    """On-Balance Volume: acumula volumen según la dirección del precio."""
    serie = [0.0]
    for i in range(1, len(precios)):
        if precios[i] > precios[i - 1]:
            serie.append(serie[-1] + volumenes[i])
        elif precios[i] < precios[i - 1]:
            serie.append(serie[-1] - volumenes[i])
        else:
            serie.append(serie[-1])
    return serie


def pendiente_normalizada(serie):
    """Pendiente por regresión lineal, normalizada al rango de la serie (-1..1 aprox)."""
    n = len(serie)
    if n < 3:
        return 0.0
    xs = list(range(n))
    mx, my = (n - 1) / 2, sum(serie) / n
    num = sum((xs[i] - mx) * (serie[i] - my) for i in range(n))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    pend = num / den * n
    rango = max(serie) - min(serie)
    return pend / rango if rango else 0.0


def flujo_30d(precios, volumenes):
    """
    Flujo de capital de los últimos 30 días, sin etiquetas interpretativas:
      - pendiente del precio (~10 días)
      - pendiente del OBV (¿entra o sale dinero?)
      - pendiente del volumen bruto (media 24h, ~10 días)
      - posición del precio en el rango del mes
    """
    if len(precios) < 100 or len(volumenes) != len(precios):
        return None
    reciente_p = precios[-240:]
    serie_obv = obv(precios, volumenes)[-240:]
    pp = pendiente_normalizada(reciente_p)
    po = pendiente_normalizada(serie_obv)
    pv = 0.0
    if len(volumenes) >= 264:
        suav = [sum(volumenes[i - 24:i]) / 24 for i in range(len(volumenes) - 240, len(volumenes))]
        pv = pendiente_normalizada(suav)
    rango_mes = max(precios) - min(precios)
    pos_mes = (precios[-1] - min(precios)) / rango_mes * 100 if rango_mes else 50
    return {"pend_precio": round(pp, 2), "pend_obv": round(po, 2),
            "pend_vol": round(pv, 2), "pos_mes": round(pos_mes, 1)}


def ema_serie(vals, n):
    if len(vals) < n:
        return None
    k = 2 / (n + 1)
    e = sum(vals[:n]) / n
    out = [None] * (n - 1) + [e]
    for v in vals[n:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def sma_serie(vals, n):
    if len(vals) < n:
        return None
    return [None] * (n - 1) + [sum(vals[i - n + 1:i + 1]) / n for i in range(n - 1, len(vals))]


def _estado_cruce(rapida, lenta):
    """Estado (alcista/bajista) del par de medias y hace cuántos días cruzaron."""
    if rapida is None or lenta is None:
        return None
    difs = [f - s for f, s in zip(rapida, lenta) if f is not None and s is not None]
    if len(difs) < 2:
        return None
    estado = "alcista" if difs[-1] > 0 else "bajista"
    hace = None
    for i in range(len(difs) - 1, 0, -1):
        if (difs[i] > 0) != (difs[i - 1] > 0):
            hace = len(difs) - 1 - i
            break
    return {"estado": estado, "cruce_hace_dias": hace,
            "rapida": round(rapida[-1], 8), "lenta": round(lenta[-1], 8)}


def cruces_diarios(diarios):
    """Cruce de medias sobre velas DIARIAS: EMA20/50 (tendencia)."""
    if len(diarios) < 51:
        return None
    res = {
        "ema": _estado_cruce(ema_serie(diarios, 20), ema_serie(diarios, 50)),
    }
    s20 = sma_serie(diarios, 20)
    if s20 and s20[-1]:
        res["precio_vs_sma20"] = round((diarios[-1] / s20[-1] - 1) * 100, 2)
    return res


def tendencia_diaria(diarios):
    """
    Compara los últimos días entre sí (velas diarias del último mes):
    pendiente de la última semana vs la semana anterior -> ¿la tendencia
    acelera, se frena o está girando?
    """
    if len(diarios) < 15:
        return None
    pr = pendiente_normalizada(diarios[-7:])        # última semana
    pa = pendiente_normalizada(diarios[-14:-7])     # semana anterior
    verdes = sum(1 for i in range(len(diarios) - 7, len(diarios))
                 if diarios[i] > diarios[i - 1])

    if pr > 0.15:
        if pr > pa + 0.1:
            clase, desc = "ALCISTA Y ACELERANDO", "La subida diaria es más fuerte esta semana que la anterior: tendencia ganando fuerza."
        elif pr < pa - 0.1:
            clase, desc = "ALCISTA PERO FRENÁNDOSE", "Sigue subiendo pero más despacio que la semana pasada: la tendencia pierde fuelle."
        else:
            clase, desc = "ALCISTA ESTABLE", "Subida diaria sostenida al mismo ritmo que la semana anterior."
    elif pr < -0.15:
        if pr < pa - 0.1:
            clase, desc = "BAJISTA Y ACELERANDO", "La caída diaria se está acelerando respecto a la semana anterior: no intentar adivinar el suelo."
        elif pr > pa + 0.1:
            clase, desc = "BAJISTA PERO FRENÁNDOSE", "Sigue cayendo pero cada vez menos: la presión vendedora se agota, vigilar giro."
        else:
            clase, desc = "BAJISTA ESTABLE", "Caída diaria sostenida."
    else:
        if pa < -0.15:
            clase, desc = "LATERAL TRAS CAÍDA (posible suelo)", "Dejó de caer y se estabiliza: los suelos suelen formarse así."
        elif pa > 0.15:
            clase, desc = "LATERAL TRAS SUBIDA (pausa o techo)", "Dejó de subir y lateraliza: pausa sana o inicio de techo."
        else:
            clase, desc = "LATERAL", "Sin dirección diaria clara en las dos últimas semanas."
    return {"clase": clase, "descripcion": desc, "pend_semana": round(pr, 2),
            "pend_semana_previa": round(pa, 2), "dias_verdes_7": verdes}


AJUSTE_TENDENCIA_D = {
    "ALCISTA Y ACELERANDO": +6,
    "ALCISTA ESTABLE": +3,
    "ALCISTA PERO FRENÁNDOSE": -3,
    "LATERAL TRAS CAÍDA (posible suelo)": +3,
    "LATERAL TRAS SUBIDA (pausa o techo)": -3,
    "LATERAL": 0,
    "BAJISTA PERO FRENÁNDOSE": +3,
    "BAJISTA ESTABLE": -4,
    "BAJISTA Y ACELERANDO": -7,
}


# ------------------------- Derivados (estilo CoinGlass) ------------------------

BINANCE = "https://fapi.binance.com"


def derivados_binance(simbolos):
    """
    Derivados desde Binance Futures (mejor cobertura del top 100).
    El funding de TODOS los pares llega en una sola llamada.
    Lanza excepción si Binance no está disponible (p.ej. geobloqueo) -> se usa OKX.
    """
    data = http_json(f"{BINANCE}/fapi/v1/premiumIndex", timeout=15)
    if not isinstance(data, list):
        raise RuntimeError("respuesta inválida (posible geobloqueo)")
    deriv, marks = {}, {}
    quiero = {s + "USDT": s for s in simbolos}
    for it in data:
        s = quiero.get(it.get("symbol"))
        if s:
            deriv.setdefault(s, {})["funding"] = float(it.get("lastFundingRate") or 0) * 100
            marks[s] = float(it.get("markPrice") or 0)
    if not deriv:
        raise RuntimeError("sin pares coincidentes")
    for s in list(marks):
        try:
            oi = http_json(f"{BINANCE}/fapi/v1/openInterest?symbol={s}USDT", timeout=10)
            deriv[s]["oi_usd"] = float(oi["openInterest"]) * marks[s]
        except Exception:
            pass
        time.sleep(0.06)
        try:
            ls = http_json(f"{BINANCE}/futures/data/globalLongShortAccountRatio"
                           f"?symbol={s}USDT&period=5m&limit=1", timeout=10)
            if ls:
                deriv[s]["long_short"] = float(ls[0]["longShortRatio"])
        except Exception:
            pass
        time.sleep(0.2)
    return deriv


def actualizar_derivados(simbolos):
    """Intenta Binance primero (más pares); si está bloqueado, usa OKX."""
    if simbolos:
        try:
            d = derivados_binance(simbolos)
            print(f"Derivados: Binance ({sum(1 for x in d.values() if 'funding' in x)} pares)", flush=True)
            return d
        except Exception as e:
            print(f"Binance no disponible ({e}) -> usando OKX", flush=True)
    return derivados_okx(simbolos)


def derivados_okx(simbolos):
    """Funding rate, open interest y long/short de OKX para los símbolos dados."""
    deriv = {}
    try:  # Open interest de TODOS los swaps en una llamada
        d = http_json(f"{OKX}/api/v5/public/open-interest?instType=SWAP")
        for it in d.get("data", []):
            inst = it.get("instId", "")
            if inst.endswith("-USDT-SWAP"):
                base = inst[: -len("-USDT-SWAP")]
                deriv.setdefault(base, {})["oi_usd"] = float(it.get("oiUsd") or 0)
    except Exception as e:
        print(f"OI error: {e}", flush=True)

    for s in simbolos:
        try:  # funding rate por instrumento
            d = http_json(f"{OKX}/api/v5/public/funding-rate?instId={s}-USDT-SWAP", timeout=10)
            data = d.get("data") or []
            if data:
                deriv.setdefault(s, {})["funding"] = float(data[0]["fundingRate"]) * 100
        except Exception:
            pass
        time.sleep(0.12)

    for s in simbolos:
        try:  # ratio de cuentas long/short
            d = http_json(f"{OKX}/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={s}&period=5m", timeout=10)
            data = d.get("data") or []
            if data:
                deriv.setdefault(s, {})["long_short"] = float(data[0][1])
        except Exception:
            pass
        time.sleep(0.45)

    return deriv


def muros_coinbase(simbolo):
    """
    Muros de órdenes reales del libro de Coinbase (top 50 niveles por lado):
    los mayores bloques de compra actúan como soporte inmediato y los de
    venta como resistencia inmediata.
    """
    try:
        d = http_json(f"{COINBASE}/products/{simbolo}-USD/book?level=2", timeout=10)
        bids = [(float(p), float(q)) for p, q, _ in d.get("bids", [])]
        asks = [(float(p), float(q)) for p, q, _ in d.get("asks", [])]
        if not bids or not asks:
            return None
        mid = (bids[0][0] + asks[0][0]) / 2

        def top_muros(lado):
            # solo órdenes a ±10% del precio actual (las lejanas no son relevantes)
            muros = [{"precio": p, "usd": p * q, "dist": round((p / mid - 1) * 100, 2)}
                     for p, q in lado if abs(p / mid - 1) <= 0.10]
            muros.sort(key=lambda x: -x["usd"])
            return [{**m, "usd": round(m["usd"])} for m in muros[:3]]

        return {"compra": top_muros(bids), "venta": top_muros(asks), "mid": mid}
    except Exception:
        return None  # no listada en Coinbase o error puntual


def taker_ratio(simbolo):
    """Ratio volumen comprador / vendedor (taker) de la última hora en OKX."""
    try:
        d = http_json(f"{OKX}/api/v5/rubik/stat/taker-volume?ccy={simbolo}&instType=CONTRACTS&period=5m", timeout=10)
        filas = (d.get("data") or [])[:12]
        compra = sum(float(f[1]) for f in filas)
        venta = sum(float(f[2]) for f in filas)
        return round(compra / venta, 2) if venta > 0 else None
    except Exception:
        return None


# ----------------------------- Puntuación -------------------------------

UMBRAL_COMPRA = 56              # >= -> COMPRA (y >=68 COMPRA FUERTE)
UMBRAL_VENTA = 44               # <= -> VENTA  (y <=32 VENTA FUERTE)


def senal_por_puntos(p):
    """
    Umbrales CONTIGUOS. FALLO CORREGIDO: la version anterior usaba
    COMPRA >= 58 y VENTA <= 42, dejando una franja muerta (42 < p < 58) en la
    que ni compradores ni vendedores eran suficientemente fuertes: cualquier
    moneda con 43-57 puntos salia como "MANTENER" aunque el marcador dijera lo
    contrario. Ahora los tramos cubren [0,100] sin huecos y simetricos
    respecto del neutral (50).
    """
    if p >= 68:
        return "COMPRA FUERTE"
    if p >= UMBRAL_COMPRA:
        return "COMPRA"
    if p <= 32:
        return "VENTA FUERTE"
    if p <= UMBRAL_VENTA:
        return "VENTA"
    return "MANTENER"


def fmt_precio(p):
    if p is None:
        return "—"
    if p >= 1000:
        return f"${p:,.0f}"
    if p >= 1:
        return f"${p:,.2f}"
    return f"${p:.6f}"


def accion_sugerida(senal, precio, obj_c, obj_v):
    """Traduce la señal a una acción concreta HOY según dónde está el precio."""
    if senal in ("COMPRA", "COMPRA FUERTE") and precio:
        # techo pegado por encima: entrar aquí no tiene recorrido
        if obj_v and obj_v / precio - 1 < 0.015:
            return (f"⛔ Resistencia a solo +{(obj_v / precio - 1) * 100:.1f}%: esperar a que "
                    f"rompa {fmt_precio(obj_v)} (o un retroceso), no entrar aquí")
        if not obj_c:
            return "⚠️ Sin soporte claro por debajo: si entras, hazlo pequeño y con stop ajustado"
        if precio > obj_c * 1.015:
            return f"⏳ No perseguir: esperar retroceso a la zona de compra ({fmt_precio(obj_c)})"
        return "✅ Precio en zona de compra: entrada razonable hoy"
    if senal in ("VENTA", "VENTA FUERTE") and precio:
        if not obj_v:
            return "🚨 En máximos sin resistencia por encima (subida vertical): tomar ganancias / asegurar stop, no comprar"
        if precio < obj_v * 0.985:
            return f"⏳ Si es posible, esperar rebote hacia {fmt_precio(obj_v)} para vender mejor"
        return "✅ Precio en zona de venta: tomar ganancias hoy"
    return None


def _razones_cribo(precios, precio, r, r4h, sma24, sma72, mh, pos_rango, sops, ress,
                   ch3d, ch24, ch7d, ch30, ath_dist, vol_mcap, funding, long_short,
                   btc_ch24, coin_id):
    """Explica en texto los puntos que asigna _puntua_cribo (mismas condiciones)."""
    raz = []
    if r is not None:
        if r <= 30:
            raz.append(f"RSI {r:.0f}: sobreventa fuerte → zona de compra")
        elif r <= 40:
            raz.append(f"RSI {r:.0f}: sobreventa moderada")
        elif r >= 75:
            raz.append(f"RSI {r:.0f}: sobrecompra fuerte → zona de venta")
        elif r >= 65:
            raz.append(f"RSI {r:.0f}: sobrecompra moderada")
        else:
            raz.append(f"RSI {r:.0f}: zona neutral")
    if r4h is not None and abs(r4h - (r or -99)) > 1e-9:
        if r4h >= 70:
            raz.append(f"RSI 4h en {r4h:.0f}: sobrecompra en marco mayor")
        elif r4h <= 30:
            raz.append(f"RSI 4h en {r4h:.0f}: sobreventa en marco mayor")
    if sma24 and sma72 and precio:
        if precio > sma24 > sma72:
            raz.append("Tendencia alcista (precio > SMA24h > SMA72h)")
        elif precio < sma24 < sma72:
            raz.append("Tendencia bajista (precio < SMA24h < SMA72h)")
        if len(precios) >= 96:
            s24a, s72a = sma(precios[:-24], 24), sma(precios[:-24], 72)
            if s24a and s72a:
                if s24a <= s72a and sma24 > sma72:
                    raz.append("🚀 Cruce alcista de medias en las últimas 24h: la moneda está despertando")
                elif s24a >= s72a and sma24 < sma72:
                    raz.append("Cruce bajista de medias en las últimas 24h")
    if mh is not None:
        h_act, h_ant = mh
        if h_act > 0 and h_act > h_ant:
            raz.append("MACD horario positivo y creciendo: momentum a favor")
        elif h_act < 0 and h_act < h_ant:
            raz.append("MACD horario negativo y cayendo: momentum en contra")
        elif h_act > 0 and h_act < h_ant:
            raz.append("Las barras del MACD decrecen en las alturas (RSI alto): toma de ganancias"
                       if (r or 0) >= 65 else "MACD positivo pero perdiendo fuerza")
    if len(precios) >= 60:
        prev = precios[:-24]
        hi_p, lo_p = max(prev), min(prev)
        comp = (hi_p - lo_p) / precio if precio else 1
        if precio > hi_p * 1.005 and comp < 0.12 and (ch24 or 0) > 1 and (mh is None or mh[0] > 0):
            raz.append("🚀 Despertar: rompe HOY al alza un rango comprimido de varios días")
        elif precio < lo_p * 0.995 and comp < 0.12 and (ch24 or 0) < -1:
            raz.append("Ruptura bajista HOY del rango de varios días")
    if pos_rango is not None:
        if pos_rango <= 20:
            raz.append(f"Precio en el {pos_rango:.0f}% inferior del rango 7d (barato)")
        elif pos_rango >= 85:
            raz.append(f"Precio en el {pos_rango:.0f}% del rango 7d (cerca del máximo)")
        if pos_rango >= 85 and mh is not None and mh[0] < 0:
            raz.append("⛔ Contradicción alcista: precio pegado al máximo semanal pero "
                       "MACD negativo → alto riesgo de comprar el techo, mejor esperar retroceso")
    raz += _pts_proximidad_sr(sops, ress)[1]
    if ch30 is not None and ch24 is not None:
        if ch30 > 5 and ch24 < -3:
            raz.append(f"Caída de {ch24:.1f}% en 24h dentro de tendencia mensual +{ch30:.1f}% (dip)")
        elif ch30 < -20 and ch24 < -5:
            raz.append("Caída persistente: cuchillo cayendo, mejor esperar")
        elif ch30 > 100:
            raz.append(f"Subida parabólica (+{ch30:.0f}% en 30d): alto riesgo de corrección fuerte")
        elif ch30 > 40 and (ch7d or 0) > 20:
            raz.append(f"Subida vertical (+{ch30:.0f}% en 30d): riesgo de corrección")
    if ch7d is not None:
        if 2 < ch7d <= 15:
            raz.append(f"Momentum semanal sano (+{ch7d:.1f}%)")
        elif ch7d < -15:
            raz.append(f"Corrección semanal fuerte ({ch7d:.1f}%): posible rebote")
    p3, r3 = _pts_persecucion(ch3d, precios)
    if r3:
        raz.append(r3)
    bono = max(0, _pts_rsi1h(r)) + max(0, _bonus_4h_real(r4h, r))
    bono += 12 if (pos_rango is not None and pos_rango <= 20) else 0
    bono += 10 if (ch30 is not None and ch24 is not None and ch30 > 5 and ch24 < -3) else 0
    exc = max(0, bono - TOPE_BONO_CAIDA)
    if exc:
        raz.append(f"⚖️ Sobreventa + rango bajo + dip cuentan la misma caída: "
                   f"bono limitado a +{TOPE_BONO_CAIDA} ({-exc} pts)")
    if vol_mcap is not None:
        if vol_mcap >= 0.15:
            raz.append(f"Volumen alto ({vol_mcap*100:.0f}% del market cap)")
        elif vol_mcap < 0.01:
            raz.append("Volumen muy bajo: poca liquidez")
    if ath_dist is not None:
        if -25 <= ath_dist <= -5 and (ch30 or 0) > 0:
            raz.append(f"A {abs(ath_dist):.0f}% del máximo histórico con tendencia positiva")
        elif ath_dist > -3:
            raz.append("En máximos históricos: riesgo de toma de ganancias")
    return raz


def analizar_moneda(m, deriv, btc_ch24=None):
    precios = (m.get("sparkline_in_7d") or {}).get("price") or []
    precios = [p for p in precios if p and p > 0]
    # El sparkline de CoinGecko llega con ~1h de retraso: empalmamos el precio
    # actual como última "vela" para que RSI/MACD/rango vean el último movimiento
    _pa = m.get("current_price")
    if precios and _pa and 0.5 < _pa / precios[-1] < 2:
        precios.append(_pa)

    es_stable = m["id"] in STABLECOINS or (
        m.get("current_price") and 0.97 <= m["current_price"] <= 1.03
        and abs(m.get("price_change_percentage_7d_in_currency") or 99) < 0.6
    )

    r = rsi(precios) if precios else None
    r4h = rsi(precios[3::4]) if precios else None      # velas de 4h (marco mayor)
    sma24 = sma(precios, 24)
    sma72 = sma(precios, 72)
    pos_rango = posicion_en_rango(precios) if precios else None
    vol = volatilidad(precios) if precios else None
    # S/R del marco operativo de HOY: pivotes de las últimas 48 horas.
    # Niveles cercanos y accionables en 24h (la escalera del mes se ve en el
    # detalle y entra en el veredicto vía confirmación 🧱).
    vent24 = precios[-49:] if precios else []
    sops, ress = (soportes_resistencias(vent24, ventana=2, precio_ref=m.get("current_price"))
                  if precios else ([], []))

    ch1h = m.get("price_change_percentage_1h_in_currency")
    ch24 = m.get("price_change_percentage_24h_in_currency")
    ch7d = m.get("price_change_percentage_7d_in_currency")
    ch30 = m.get("price_change_percentage_30d_in_currency")
    # Cambio a 3 días: CoinGecko no lo da directo -> lo calculamos con las
    # velas horarias (72h atrás). Se recalcula con velas OKX EN VIVO al refinar.
    ch3d = ((precios[-1] / precios[-73] - 1) * 100) if len(precios) >= 73 and precios[-73] else None
    precio = m.get("current_price") or 0
    vol_mcap = (m.get("total_volume") or 0) / m["market_cap"] if m.get("market_cap") else 0
    ath_dist = m.get("ath_change_percentage")

    simbolo = m["symbol"].upper()
    dd = deriv.get(simbolo, {})
    funding = dd.get("funding")
    oi_usd = dd.get("oi_usd")
    long_short = dd.get("long_short")

    if vol is not None and vol < 8:   # fondos tokenizados / renta fija
        es_stable = True

    base = {
        "id": m["id"], "rank": m.get("market_cap_rank"), "simbolo": simbolo,
        "nombre": m["name"], "imagen": m.get("image"), "precio": precio,
        "market_cap": m.get("market_cap"), "ch1h": ch1h, "ch24": ch24, "ch3d": ch3d,
        "ch7d": ch7d,
        "ch30": ch30, "rsi": r, "rsi4h": r4h, "pos_rango": pos_rango, "volatilidad": vol,
        "watch": bool(m.get("_watch")),
        "vol_mcap": vol_mcap, "ath_dist": ath_dist,
        "soportes": sops, "resistencias": ress,
        "funding": funding, "oi_usd": oi_usd, "long_short": long_short,
    }

    if es_stable:
        base.update({"puntos": 50.0, "senal": "STABLE",
                     "razones": ["Stablecoin / fondo tokenizado: no se opera."],
                     "objetivo_compra": None, "objetivo_venta": None, "stop_loss": None})
        return base

    # PUNTUACION DE PRECIO: delegada en _puntua_cribo(), que es el MISMO codigo
    # que usa el backtest. FALLO CORREGIDO: antes estos pesos estaban duplicados
    # a mano en dos sitios (cribado y refinado/backtest imposible), y las dos
    # copias se habian ido separando (doble conteo del RSI, tope anti-caida con
    # componentes distintos, umbrales con hueco). Aqui solo se anaden los
    # factores que no dependen solo del precio (derivados, BTC, libro, flujo).
    puntos, _senal_prev = _puntua_cribo(precios, precio_actual=precio, ch24=ch24,
                                        ch7d=ch7d, ch30=ch30, ath_dist=ath_dist,
                                        vol_mcap=vol_mcap)
    razones = _razones_cribo(precios, precio, r, r4h, sma24, sma72, mh, pos_rango,
                             sops, ress, ch3d, ch24, ch7d, ch30, ath_dist, vol_mcap,
                             funding, long_short, btc_ch24, m["id"])
    b_rsi = max(0, _pts_rsi1h(r))
    b_rsi4_pos = 8 if (r4h is not None and abs(r4h - (r or -99)) > 1e-9 and r4h <= 30) else 0
    b_rango = 12 if (pos_rango is not None and pos_rango <= 20) else 0
    b_dip = 10 if (ch30 is not None and ch24 is not None and ch30 > 5 and ch24 < -3) else 0
    bono_caida = b_rsi + b_rsi4_pos + b_rango + b_dip
    exceso_caida = max(0, bono_caida - TOPE_BONO_CAIDA)
    pts_sr = _pts_proximidad_sr(sops, ress)[0]
    pts_3d = _pts_persecucion(ch3d, precios)[0]
    b_rsi4 = _bonus_4h_real(r4h, r)

    # --- factores que NO dependen solo del precio (no entran en el backtest) ---
    if funding is not None:
        if funding >= 0.05:
            puntos -= 8; razones.append(f"Funding muy positivo ({funding:.3f}%/8h): longs sobreapalancados, riesgo de long squeeze")
        elif funding >= 0.03:
            puntos -= 4; razones.append(f"Funding elevado ({funding:.3f}%/8h): mercado calentándose")
        elif funding <= -0.03:
            puntos += 8; razones.append(f"Funding negativo ({funding:.3f}%/8h): shorts pagan, posible short squeeze alcista")
        elif funding < 0:
            puntos += 3; razones.append(f"Funding ligeramente negativo ({funding:.3f}%/8h)")
    if long_short is not None:
        if long_short >= 3:
            puntos -= 5; razones.append(f"Ratio long/short {long_short:.2f}: exceso de longs (señal contraria)")
        elif long_short <= 0.7:
            puntos += 5; razones.append(f"Ratio long/short {long_short:.2f}: exceso de shorts (combustible para subida)")
    if btc_ch24 is not None and m["id"] != "bitcoin":
        if btc_ch24 <= -2.5:
            puntos -= 5; razones.append(f"BTC cae {btc_ch24:.1f}% hoy: mercado de riesgo, prudencia con las compras")
        elif btc_ch24 >= 2.5:
            puntos += 2; razones.append(f"BTC sube {btc_ch24:.1f}% hoy: viento de cola para el mercado")
    puntos = max(0, min(100, puntos))

    # Filtro de calidad: liquidez ínfima = market cap poco fiable, NO operable
    liquidez_baja = vol_mcap < 0.02 or (m.get("total_volume") or 0) < 5_000_000
    if liquidez_baja and puntos > 52:
        puntos = 52
        razones.append("⚠️ Liquidez muy baja (volumen <2% del market cap): posible capitalización "
                       "inflada, señal de compra bloqueada por seguridad")

    puntos = max(0, min(100, puntos))
    senal = senal_por_puntos(puntos)

    obj_c = sops[0]["nivel"] if sops else (min(vent24) * 1.01 if vent24 else None)
    if obj_c and precio and obj_c >= precio * 0.998:
        obj_c = None                        # sin zona de compra clara por debajo
    obj_v = ress[0]["nivel"] if ress else (max(vent24) * 0.99 if vent24 else None)
    if obj_v and precio and obj_v <= precio * 1.002:
        obj_v = None                        # en máximos: no hay resistencia por encima
    stop = (sops[1]["nivel"] * 0.98 if len(sops) > 1 else (min(vent24) * 0.95 if vent24 else None))
    # mismo guardabolera que en el refinado: un stop por encima del precio es
    # una orden imposible y falsea el riesgo/beneficio mostrado
    if stop and precio and stop >= precio:
        stop = (min(vent24) * 0.95 if vent24 else None)

    # Indicador auxiliar "HOY": ¿el intradía acompaña para ejecutar la señal hoy?
    hoy = calc_hoy(mh, r)

    base.update({"puntos": round(puntos, 1), "senal": senal, "razones": razones,
                 "liquidez_baja": liquidez_baja, "hoy": hoy,
                 "_mh": list(mh) if mh else None,
                 "_b_rsi": b_rsi, "_b_rsi4": b_rsi4, "_b_rango_dip": b_rango + b_dip,
                 "_exceso": exceso_caida, "_pts_sr": pts_sr,
                 "_pts_3d": pts_3d, "_pts_libro": 0, "_pts_flujo": 0,
                 "accion": accion_sugerida(senal, precio, obj_c, obj_v),
                 "objetivo_compra": obj_c, "objetivo_venta": obj_v, "stop_loss": stop})
    return base


def _puntua_cribo(precios, precio_actual=None, ch24=None, ch7d=None, ch30=None,
                  ath_dist=None, vol_mcap=None):
    """
    Version AISLADA y reutilizable del bloque de puntuacion del cribado.
    FALLO CORREGIDO: antes la logica de puntos vivia DENTRO de analizar_moneda
    mezclada con las llamadas de red, asi que era imposible probarla o calibrarla
    (de ahi que los umbrales estuvieran mal ajustados durante versiones). Ahora
    el cribado y el backtest llaman al MISMO codigo: lo que se valida es lo que
    se publica.

    Solo usa indicadores de PRECIO (los unicos con historico gratuito suficiente);
    funding/libro/tendencia-diaria se anaden aparte en analizar_moneda.
    Devuelve (puntos, senal).
    """
    precios = [p for p in precios if p and p > 0]
    if len(precios) < 60:
        return 50.0, "MANTENER"
    precio = precio_actual or precios[-1]
    r = rsi(precios)
    r4h = rsi(precios[3::4])
    sma24, sma72 = sma(precios, 24), sma(precios, 72)
    pos_rango = posicion_en_rango(precios)
    mh = macd_hist(precios)
    vent = precios[-49:]
    sops, ress = soportes_resistencias(vent, ventana=2, precio_ref=precio)
    ch3d = ((precios[-1] / precios[-73] - 1) * 100) if len(precios) >= 73 and precios[-73] else None

    puntos = 50.0
    b_rsi = b_rango = b_dip = b_rsi4_pos = 0
    puntos += _pts_rsi1h(r)
    b_rsi = max(0, _pts_rsi1h(r))
    if r4h is not None and abs(r4h - (r or -99)) > 1e-9:
        e4 = _bonus_4h_real(r4h, r)
        puntos += e4
        b_rsi4_pos = max(0, e4)
    if sma24 and sma72:
        if precio > sma24 > sma72:
            puntos += 8
        elif precio < sma24 < sma72:
            puntos -= 6
        if len(precios) >= 96:
            s24a, s72a = sma(precios[:-24], 24), sma(precios[:-24], 72)
            if s24a and s72a:
                if s24a <= s72a and sma24 > sma72:
                    puntos += 7
                elif s24a >= s72a and sma24 < sma72:
                    puntos -= 6
    puntos += _pts_macd1h(mh, r)
    if len(precios) >= 60:
        prev = precios[:-24]
        hi_prev, lo_prev = max(prev), min(prev)
        compresion = (hi_prev - lo_prev) / precio if precio else 1
        if precio > hi_prev * 1.005 and compresion < 0.12 and (ch24 or 0) > 1 and (mh is None or mh[0] > 0):
            puntos += 10
        elif precio < lo_prev * 0.995 and compresion < 0.12 and (ch24 or 0) < -1:
            puntos -= 8
    if pos_rango is not None:
        if pos_rango <= 20:
            puntos += 12; b_rango = 12
        elif pos_rango >= 85:
            puntos -= 12
    if pos_rango is not None and pos_rango >= 85 and mh is not None and mh[0] < 0:
        puntos -= 10
    puntos += _pts_proximidad_sr(sops, ress)[0]
    if ch30 is not None and ch24 is not None:
        if ch30 > 5 and ch24 < -3:
            puntos += 10; b_dip = 10
        elif ch30 < -20 and ch24 < -5:
            puntos -= 8
        elif ch30 > 100:
            puntos -= 10
        elif ch30 > 40 and (ch7d or 0) > 20:
            puntos -= 8
    if ch7d is not None:
        if 2 < ch7d <= 15:
            puntos += 6
        elif ch7d < -15:
            puntos += 4
    puntos += _pts_persecucion(ch3d, precios)[0]
    puntos -= max(0, b_rsi + b_rsi4_pos + b_rango + b_dip - TOPE_BONO_CAIDA)
    if ath_dist is not None:
        if -25 <= ath_dist <= -5 and (ch30 or 0) > 0:
            puntos += 4
        elif ath_dist > -3:
            puntos -= 5
    if vol_mcap is not None:
        if vol_mcap >= 0.15:
            puntos += 4
        elif vol_mcap < 0.01:
            puntos -= 4
    puntos = max(0, min(100, puntos))
    return round(puntos, 1), senal_por_puntos(puntos)


# --------------------------- Backtest (validación) ------------------------

def _serie_horaria(simbolo, n=720):
    """Cierres horarios para el backtest (Binance primero, OKX de respaldo)."""
    k = _binance_klines(simbolo, "1h", min(n, 1000))
    if k and len(k) >= 100:
        return [float(c[4]) for c in k]
    v = velas_ohlc_okx(simbolo, min(n, 300))
    return v[2] if v else None


def backtest_senal(simbolos=("BTC", "ETH", "SOL"), horizonte=24, paso=24):
    """
    ¿Funciona realmente el marcador? Esta funcion responde con datos, no con
    opiniones: recorre el historico hacia atras, calcula la nota COMO LA CALCULA
    EL CRIBADO (50 + bonos/penalizaciones) en cada punto, y mide despues lo que
    hizo el precio a +horizonte horas.

    Devuelve, por simbolo y por tramo de puntos:
      - acierto_compra : % de veces que el precio subio cuando la nota decia COMPRA
      - acierto_venta  : % de veces que el precio bajo cuando la nota decia VENTA
      - edge           : correlacion simple (puntos altos -> retorno alto)
    Es un backtest SENCILLO (solo precio, sin derivados ni libro, porque esos
    datos historicos no estan disponibles gratis): sirve para detectar sesgos y
    umbrales mal calibrados, no como garantia de rentabilidad.
    """
    salida = {}
    for s in simbolos:
        cierres = _serie_horaria(s)
        if not cierres or len(cierres) < 200 + horizonte + paso:
            salida[s] = {"error": "historico insuficiente"}
            continue
        registros = []
        i = len(cierres) - horizonte
        while i >= 200:
            ventana = cierres[i - 199:i + 1]
            p = _puntua_cribo(ventana, cierres[i])
            ret = (cierres[i + horizonte] / cierres[i] - 1) * 100
            registros.append((p, ret))
            i -= paso
        if not registros:
            continue
        compras = [(pp, rr) for pp, rr in registros if pp >= UMBRAL_COMPRA]
        ventas = [(pp, rr) for pp, rr in registros if pp <= UMBRAL_VENTA]
        media_p = sum(x for x, _ in registros) / len(registros)
        media_r = sum(x for _, x in registros) / len(registros)
        cov = sum((pp - media_p) * (rr - media_r) for pp, rr in registros)
        den = math.sqrt(sum((pp - media_p) ** 2 for pp, _ in registros)
                        * sum((rr - media_r) ** 2 for _, rr in registros)) or 1
        salida[s] = {
            "muestras": len(registros),
            "horizonte_h": horizonte,
            "edge_corr": round(cov / den, 3),
            "compra_n": len(compras),
            "acierto_compra": round(sum(1 for _, r in compras if r > 0) / len(compras) * 100, 1) if compras else None,
            "ret_medio_compra": round(sum(r for _, r in compras) / len(compras), 2) if compras else None,
            "venta_n": len(ventas),
            "acierto_venta": round(sum(1 for _, r in ventas if r < 0) / len(ventas) * 100, 1) if ventas else None,
            "ret_medio_venta": round(sum(r for _, r in ventas) / len(ventas), 2) if ventas else None,
        }
    return salida


# ------------------------------- Motor ----------------------------------

def confirmar_candidatas(max_n=16):
    """
    Segunda etapa: para las señales de compra/venta más fuertes, descarga el
    histórico de 90 días y CONFIRMA o REBAJA la señal según divergencias de volumen,
    tendencia diaria, cruces de medias diarios, RSI diario y presión compradora real.
    Espaciado para respetar el límite gratuito de CoinGecko.
    """
    with _lock:
        monedas = list(_estado["monedas"])
    cands = [m for m in monedas if m["senal"] not in ("MANTENER", "STABLE")]
    cands.sort(key=lambda m: -abs(m["puntos"] - 50))

    nuevas_descargas = 0
    for m in cands:
        # si ya hay análisis 30d en caché lo aplica gratis; si no, descarga (máx. max_n por ciclo)
        en_cache = _detalle_cache.get(m["id"])
        fresco = en_cache and (time.time() - en_cache[0] < DETALLE_CACHE_SEG)
        if not fresco:
            if nuevas_descargas >= max_n:
                continue
            nuevas_descargas += 1
        det = analisis_profundo(m["id"], m["simbolo"], m["precio"])
        if not fresco:
            time.sleep(7)
        if not det.get("ok"):
            continue
        ajuste = 0
        nota_extra = ""

        # 0) INTRADÍA EN TIEMPO REAL (velas 1h de OKX, sin retraso).
        #    El cribado midió RSI 1h y MACD horario sobre el sparkline de CoinGecko,
        #    que llega con ~1h de retraso: si en la última hora hubo un giro
        #    (p. ej. un retroceso fuerte), aquí se re-mide y se corrige la nota.
        cierres = None if m.get("okx_vivo") else velas_1h_okx(m["simbolo"])
        if cierres:
            r_rt = rsi(cierres)
            mh_rt = macd_hist(cierres)
            mh_old = tuple(m["_mh"]) if m.get("_mh") else None
            # FALLO CORREGIDO (inestabilidad / señal que oscila): este ajuste se
            # aplicaba SOBRE la nota ya refinada en vivo, comparando velas OKX
            # contra un RSI/MACD que ya podian ser de otra fuente y de otra hora.
            # El resultado era un delta casi aleatorio (+/-30 pts) que cambiaba la
            # senal de un ciclo al siguiente. Ahora: si la nota ya esta refinada
            # EN VIVO (okx_vivo), NO se vuelve a corregir; solo se corrige cuando
            # el cribado sigue con los datos retrasados de CoinGecko.
            delta_rt = ((_pts_rsi1h(r_rt) + _pts_macd1h(mh_rt, r_rt))
                        - (_pts_rsi1h(m.get("rsi")) + _pts_macd1h(mh_old, m.get("rsi"))))
            if m.get("okx_vivo"):
                delta_rt = 0
            if mh_rt is not None and mh_rt[0] > 0 and mh_rt[0] >= mh_rt[1]:
                desc_macd = "positivo y creciendo"
            elif mh_rt is not None and mh_rt[0] > 0:
                desc_macd = "positivo pero perdiendo fuerza"
            elif mh_rt is not None and mh_rt[0] <= mh_rt[1]:
                desc_macd = "negativo y cayendo"
            elif mh_rt is not None:
                desc_macd = "negativo pero recuperándose"
            else:
                desc_macd = "sin datos"
            bullet = ("📡 Intradía en TIEMPO REAL (velas 1h OKX): "
                      + (f"RSI {r_rt:.0f}" if r_rt is not None else "RSI n/d")
                      + f", MACD horario {desc_macd}")
            if delta_rt:
                bullet += f" → corrige el cribado {delta_rt:+d} pts (el sparkline llegaba ~1h retrasado)"
                ajuste += delta_rt
                nota_extra += f"; intradía en vivo corrige {delta_rt:+d}"
            with _lock:
                if r_rt is not None:
                    m["rsi"] = r_rt
                m["hoy"] = calc_hoy(mh_rt, r_rt)
                m["_mh"] = list(mh_rt) if mh_rt else None
                m["razones"] = [z for z in m["razones"] if not _es_bullet_intradia(z)]
                m["razones"].insert(0, bullet)
            # S/R de 24h re-medidos con las velas EN VIVO (solo si OKX cotiza
            # prácticamente igual que CoinGecko, para no mezclar escalas)
            if abs(cierres[-1] / m["precio"] - 1) < 0.02:
                sr_s, sr_r = soportes_resistencias(cierres[-49:], ventana=2,
                                                   precio_ref=m["precio"])
                if sr_s or sr_r:
                    n_obj_c = sr_s[0]["nivel"] if sr_s else None
                    if n_obj_c and n_obj_c >= m["precio"] * 0.998:
                        n_obj_c = None
                    n_obj_v = sr_r[0]["nivel"] if sr_r else None
                    if n_obj_v and n_obj_v <= m["precio"] * 1.002:
                        n_obj_v = None
                    with _lock:
                        m["soportes"], m["resistencias"] = sr_s, sr_r
                        m["objetivo_compra"], m["objetivo_venta"] = n_obj_c, n_obj_v
                        if len(sr_s) > 1:
                            m["stop_loss"] = sr_s[1]["nivel"] * 0.98
        # 📐 CHECKLIST CLÁSICO DIARIO
        # 1) SMA200 diaria (velas de OKX: CoinGecko gratis solo llega a 90 días)
        d200 = velas_diarias_okx(m["simbolo"])
        if d200 and len(d200) >= 205 and m["precio"] and abs(d200[-1] / m["precio"] - 1) < 0.02:
            sma200 = sum(d200[-200:]) / 200
            rel200 = (m["precio"] / sma200 - 1) * 100
            if rel200 > 0:
                ajuste += 4
                nota_extra += f"; precio {rel200:+.0f}% sobre la SMA200 diaria (fondo alcista, +4)"
            else:
                ajuste -= 4
                nota_extra += f"; precio {rel200:+.0f}% BAJO la SMA200 diaria (fondo bajista, -4)"
        # 2) RSI diario rebotando desde la zona de 30
        if det.get("rsi_d_rebote"):
            ajuste += 5
            nota_extra += "; RSI diario rebotando desde la zona de 30 (+5)"
        # 3) ¿El volumen acompaña los días verdes o los rojos? (últimos 14 días)
        vvr = det.get("vol_verde_ratio")
        if vvr is not None:
            if vvr >= 1.3:
                ajuste += 4
                nota_extra += f"; volumen alto en los días verdes (x{vvr} vs días rojos, +4)"
            elif vvr <= 0.7:
                ajuste -= 4
                nota_extra += f"; el volumen carga los días ROJOS (x{vvr}, -4)"
        fl = det.get("flujo_30d") or {}
        if fl:
            # divergencias precio/volumen (OBV y volumen bruto)
            if fl.get("pend_precio", 0) > 0.25 and fl.get("pend_obv", 0) < 0:
                ajuste -= 10
                nota_extra += "; divergencia bajista: el precio sube pero el OBV cae"
            elif fl.get("pend_precio", 0) > 0.25 and fl.get("pend_vol", 0) < -0.15:
                ajuste -= 5
                nota_extra += "; el volumen decrece mientras el precio sube"
            elif fl.get("pend_precio", 0) > 0.25 and fl.get("pend_obv", 0) > 0.1:
                ajuste += 4
                nota_extra += "; subida respaldada por volumen (OBV acompaña)"
            if fl.get("pend_precio", 0) < -0.25 and fl.get("pend_obv", 0) > 0.1:
                ajuste += 7
                nota_extra += "; absorción: el precio cae pero el OBV sube (compradores aguantando)"
            # techo mensual sin respaldo de flujo comprador
            if fl.get("pos_mes", 50) >= 90 and fl.get("pend_obv", 0) <= 0:
                ajuste -= 6
                nota_extra += "; en el máximo del mes SIN flujo comprador (OBV plano)"
            # ⛔ COHERENCIA: señal de COMPRA con el dinero SALIENDO a 10 días
            #    (el caso ARB: dip atractivo pero el OBV cae = distribución)
            if (m["senal"].startswith("COMPRA") and fl.get("pend_obv", 0) < -0.1
                    and fl.get("pend_precio", 0) <= 0.25):
                ajuste -= 8
                nota_extra += ("; ⛔ contradicción: señal de compra pero el dinero "
                               "SALE a 10 días (OBV cayendo, -8)")
            # y al revés: señal de VENTA con el dinero ENTRANDO
            if (m["senal"].startswith("VENTA") and fl.get("pend_obv", 0) > 0.1
                    and fl.get("pend_precio", 0) >= -0.25):
                ajuste += 6
                nota_extra += ("; ⚠️ ojo: señal de venta pero el dinero ENTRA "
                               "a 10 días (OBV subiendo, +6)")
        # Cruces de medias en marco DIARIO
        cd = det.get("cruces_diarios")
        if cd:
            e = cd.get("ema")
            if e:
                if e["cruce_hace_dias"] is not None and e["cruce_hace_dias"] <= 5:
                    if e["estado"] == "alcista":
                        ajuste += 7
                        nota_extra += f"; cruce ALCISTA EMA20/50 diario hace {e['cruce_hace_dias']}d (+7)"
                    else:
                        ajuste -= 7
                        nota_extra += f"; cruce BAJISTA EMA20/50 diario hace {e['cruce_hace_dias']}d (-7)"
                elif e["estado"] == "alcista":
                    ajuste += 3
                else:
                    ajuste -= 3
        # Tendencia diaria: comparación de la última semana con la anterior
        td = det.get("tendencia_diaria")
        if td:
            aj_td = AJUSTE_TENDENCIA_D.get(td["clase"], 0)
            if aj_td:
                ajuste += aj_td
                nota_extra += f"; tendencia diaria {td['clase']} ({aj_td:+d})"
        # RSI diario extremo: el marco grande manda
        rd = det.get("rsi_diario")
        if rd is not None:
            if rd >= 68 and m["senal"].startswith("COMPRA"):
                ajuste -= 7
                nota_extra += f"; RSI diario en sobrecompra ({rd:.0f})"
            elif rd <= 32 and m["senal"].startswith("VENTA"):
                ajuste += 7
                nota_extra += f"; RSI diario en sobreventa ({rd:.0f})"
        # resistencia mensual pegada por encima -> poco recorrido para una compra
        res30 = det.get("resistencias_30d") or []
        if res30 and res30[0]["dist"] <= 2.5 and m["senal"].startswith("COMPRA"):
            fuerte = res30[0].get("toques", 1) >= 3
            ajuste -= 6 if fuerte else 3
            nota_extra += (f"; resistencia mensual {'FUERTE (' + str(res30[0]['toques']) + ' toques) ' if fuerte else ''}"
                           f"muy cerca (+{res30[0]['dist']}%): esperar a que la rompa")
        # 🧱 ESCALERA 30d EN EL VEREDICTO: los techos/suelos del mes mandan en el swing
        sop30 = det.get("soportes_30d") or []
        if m["senal"].startswith("COMPRA") and res30:
            # techo relevante: primera resistencia FUERTE (>=3 toques) o, si no hay, la más cercana
            techo = next((r for r in res30 if r.get("toques", 1) >= 3), res30[0])
            with _lock:
                m["razones"].append(f"🧱 Techo del mes más cercano: {fmt_precio(techo['nivel'])} "
                                    f"({techo['toques']} toque{'s' if techo['toques'] != 1 else ''}, a +{techo['dist']}%)")
                # objetivo de venta realista: no apuntar más allá del primer techo fuerte del mes
                if (techo.get("toques", 1) >= 3 and m.get("objetivo_venta")
                        and m["precio"] * 1.002 < techo["nivel"] < m["objetivo_venta"] * 0.995):
                    m["objetivo_venta"] = techo["nivel"]
                    nota_extra += f"; objetivo recortado al techo del mes ({techo['toques']} toques)"
            # riesgo/beneficio del swing: recorrido hasta el techo vs caída hasta el stop
            if m.get("stop_loss") and m["precio"]:
                riesgo = (1 - m["stop_loss"] / m["precio"]) * 100
                if riesgo > 1 and 2.5 < techo["dist"] < riesgo * 0.7:
                    ajuste -= 5
                    nota_extra += (f"; 🧱 mal riesgo/beneficio semanal: +{techo['dist']}% hasta el techo "
                                   f"({techo['toques']} toques) vs -{riesgo:.1f}% hasta el stop")
        elif m["senal"].startswith("VENTA") and sop30:
            suelo = next((s for s in sop30 if s.get("toques", 1) >= 3), sop30[0])
            with _lock:
                m["razones"].append(f"🧱 Suelo del mes más cercano: {fmt_precio(suelo['nivel'])} "
                                    f"({suelo['toques']} toque{'s' if suelo['toques'] != 1 else ''}, a {suelo['dist']}%)")
            if suelo.get("toques", 1) >= 3 and abs(suelo["dist"]) <= 3.0:
                ajuste += 4
                nota_extra += (f"; 🧱 soporte fuerte del mes a {suelo['dist']}%: la caída puede "
                               "frenarse ahí (venta con poco recorrido)")
        tk = det.get("taker_ratio")
        if tk is not None:
            if tk >= 1.25:
                ajuste += 3
            elif tk <= 0.8:
                ajuste -= 3
        nuevos = max(0, min(100, m["puntos"] + ajuste))
        nueva_senal = senal_por_puntos(nuevos)
        td_det = det.get("tendencia_diaria")
        with _lock:
            m["puntos"] = round(nuevos, 1)
            m["tendencia_d"] = td_det["clase"] if td_det else None
            m["confirmada"] = True
            m["razones"].append(
                f"Confirmación 30d{nota_extra if nota_extra else ': sin divergencias relevantes'}"
                + (f", presión taker {tk}" if tk is not None else "")
                + f" → ajuste {ajuste:+d} puntos"
                + (f" (señal rebajada de {m['senal']} a {nueva_senal})"
                   if nueva_senal != m["senal"] else " (señal confirmada)"))
            m["senal"] = nueva_senal
            m["accion"] = accion_sugerida(nueva_senal, m["precio"],
                                          m["objetivo_compra"], m["objetivo_venta"])

    with _lock:
        _estado["monedas"].sort(key=lambda x: -(x["puntos"] if x["senal"] != "STABLE" else -1))
        try:
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump({"actualizado": _estado["actualizado"],
                           "monedas": _estado["monedas"]}, f, ensure_ascii=False, indent=1)
        except Exception:
            pass
    print(f"Confirmación 30d aplicada a {min(len(cands), max_n)} candidatas", flush=True)


def ciclo_actualizacion():
    primera = True
    ultimo_crudo = None
    while True:
        try:
            try:
                crudo = http_json(CG_MARKETS)
                # descarta wrapped/staked/fondos y excluidas, rellena hasta 100 reales
                crudo = [m for m in crudo if not es_derivado(m)
                         and m.get("id") not in EXCLUIDAS][:100]
                # ⭐ añade las favoritas que no estén ya en la lista
                ids_lista = {m["id"] for m in crudo}
                faltan = [w for w in WATCHLIST if w not in ids_lista]
                if faltan:
                    try:
                        extra = http_json(CG_MARKETS_IDS.format(ids="%2C".join(faltan)))
                        ids_ok = {e["id"] for e in extra}
                        for w in faltan:
                            if w not in ids_ok:
                                print(f"⚠️ WATCHLIST: el id '{w}' no existe en CoinGecko — "
                                      f"revisa comas y el id exacto de la URL de coingecko.com", flush=True)
                        for e in extra:
                            e["_watch"] = True
                        crudo += [e for e in extra if e["id"] not in ids_lista]
                    except Exception as e:
                        print(f"Watchlist no disponible este ciclo: {e}", flush=True)
                ultimo_crudo = [dict(m) for m in crudo]
            except Exception as e_cg:
                # 🔁 CoinGecko caído o con 429: NO se aborta el ciclo. Se
                # reutilizan los últimos precios y se refina con velas EN VIVO
                # (Binance/OKX), que es de donde salen flujo 💵, RSI 📡, etc.
                if not ultimo_crudo:
                    raise
                print(f"CoinGecko no disponible ({e_cg}) -> ciclo con los últimos "
                      f"precios + velas EN VIVO de Binance/OKX", flush=True)
                crudo = [dict(m) for m in ultimo_crudo]
            simbolos = [m["symbol"].upper() for m in crudo
                        if m["id"] not in STABLECOINS][:90]
            if primera:
                # primer arranque: publica rápido solo con OI (1 llamada)
                deriv = actualizar_derivados([])
            else:
                deriv = actualizar_derivados(simbolos)

            btc = next((x for x in crudo if x["id"] == "bitcoin"), None)
            btc_ch24 = btc.get("price_change_percentage_24h_in_currency") if btc else None
            monedas = [analizar_moneda(m, deriv, btc_ch24) for m in crudo]
            # 📡 REFINADO EN VIVO: para cada moneda con par en OKX, se sustituyen
            # los datos horarios retrasados por velas 1H reales (con mechas)
            if not primera:
                n_vivo = 0
                for mo in monedas:
                    try:
                        if refinar_con_okx(mo):
                            n_vivo += 1
                            time.sleep(0.12)
                    except Exception:
                        pass
                print(f"Velas 1h EN VIVO aplicadas a {n_vivo} monedas", flush=True)
            monedas.sort(key=lambda x: -(x["puntos"] if x["senal"] != "STABLE" else -1))
            ahora = datetime.now(timezone.utc).isoformat()
            with _lock:
                _estado["monedas"] = monedas
                _estado["actualizado"] = ahora
                _estado["error"] = None
                _derivados.clear()
                _derivados.update(deriv)
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump({"actualizado": ahora, "monedas": monedas}, f, ensure_ascii=False, indent=1)
            n_der = sum(1 for x in deriv.values() if "funding" in x)
            print(f"[{ahora}] OK: {len(monedas)} monedas, derivados de {n_der}", flush=True)
            if primera:
                primera = False
                continue  # sin esperar: siguiente ciclo ya con derivados completos
            try:
                confirmar_candidatas()
            except Exception as e:
                print(f"Error confirmación: {e}", flush=True)
        except Exception as e:
            with _lock:
                _estado["error"] = str(e)
            print(f"Error ciclo: {e}", flush=True)
        time.sleep(REFRESH_SECONDS)


def analisis_profundo(coin_id, simbolo, precio_ref=None):
    """Análisis profundo bajo demanda: S/R mensuales, flujo de capital, marcos diarios."""
    ahora = time.time()
    en_cache = _detalle_cache.get(coin_id)
    if en_cache and ahora - en_cache[0] < DETALLE_CACHE_SEG:
        return en_cache[1]
    try:
        d = http_json(CG_CHART.format(id=coin_id))
        precios_90 = [p[1] for p in d.get("prices", []) if p[1]]
        vols_90 = [v[1] for v in d.get("total_volumes", [])]
        if len(vols_90) != len(precios_90):
            vols_90 = vols_90[:len(precios_90)] + [0] * max(0, len(precios_90) - len(vols_90))
        precios = precios_90[-720:]                # últimos 30 días (horario)
        vols = vols_90[-720:]
        diarios = precios_90[23::24]               # ~90 velas diarias
        sops, ress = soportes_resistencias(precios, ventana=5, max_n=3, precio_ref=precio_ref)
        rsi_diario = rsi(diarios)
        # Checklist clásico: ¿el RSI diario viene REBOTANDO desde la zona de 30?
        rsi_d_prev = rsi(diarios[:-5]) if len(diarios) > 25 else None
        rsi_d_rebote = bool(rsi_diario is not None and rsi_d_prev is not None
                            and rsi_d_prev <= 35 and rsi_diario > rsi_d_prev + 5
                            and rsi_diario < 60)
        # Checklist clásico: ¿el volumen es alto en los días verdes? (últimos 14 días)
        vol_verde_ratio = None
        cierres_d = precios[23::24]
        vols_d = [sum(vols[i:i + 24]) for i in range(0, len(vols) - 23, 24)]
        if len(cierres_d) >= 15 and len(vols_d) >= len(cierres_d):
            verdes, rojos = [], []
            for i in range(len(cierres_d) - 14, len(cierres_d)):
                if i <= 0:
                    continue
                (verdes if cierres_d[i] >= cierres_d[i - 1] else rojos).append(vols_d[i])
            if verdes and rojos and sum(rojos) > 0:
                vol_verde_ratio = round((sum(verdes) / len(verdes)) / (sum(rojos) / len(rojos)), 2)
        det = {
            "ok": True,
            "rsi_diario": round(rsi_diario, 1) if rsi_diario is not None else None,
            "rsi_d_rebote": rsi_d_rebote,
            "vol_verde_ratio": vol_verde_ratio,
            "tendencia_diaria": tendencia_diaria(diarios),
            "cruces_diarios": cruces_diarios(diarios),
            "soportes_30d": sops, "resistencias_30d": ress,
            "flujo_30d": flujo_30d(precios, vols),
            "taker_ratio": taker_ratio(simbolo),
            "muros": muros_coinbase(simbolo),
            "max_30d": max(precios) if precios else None,
            "min_30d": min(precios) if precios else None,
        }
    except urllib.error.HTTPError as e:
        det = {"ok": False, "error": "Límite de la API alcanzado, reintenta en un minuto"
               if e.code == 429 else f"Error HTTP {e.code}"}
        return det  # no cachear errores
    except Exception as e:
        return {"ok": False, "error": str(e)}
    _detalle_cache[coin_id] = (ahora, det)
    return det


# -------------------------------- Web -----------------------------------

@app.route("/api/analisis")
def api_analisis():
    with _lock:
        return jsonify(_estado)


@app.route("/api/detalle/<coin_id>")
def api_detalle(coin_id):
    with _lock:
        moneda = next((m for m in _estado["monedas"] if m["id"] == coin_id), None)
    if not moneda:
        return jsonify({"ok": False, "error": "Moneda no encontrada"}), 404
    return jsonify(analisis_profundo(coin_id, moneda["simbolo"], moneda["precio"]))


@app.route("/api/backtest")
def api_backtest():
    """Validacion del marcador: {"BTC": {...}, ...} con aciertos reales medidos."""
    horizonte = 24
    try:
        horizonte = max(1, min(168, int(os.environ.get("BT_HORIZONTE", "24"))))
    except Exception:
        pass
    simbolos = tuple(os.environ.get("BT_SIMBOLOS", "BTC,ETH,SOL").split(","))
    return jsonify(backtest_senal(simbolos=simbolos, horizonte=horizonte))


@app.route("/")
def index():
    resp = send_file(os.path.join(BASE, "index.html"))
    # Evita que el navegador guarde en caché una versión antigua de la interfaz
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


if __name__ == "__main__" and "--backtest" in sys.argv:
    # python bot.py --backtest   ->  informa sin arrancar el servidor
    import json as _json
    print(_json.dumps(backtest_senal(), ensure_ascii=False, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    print(f"⭐ WATCHLIST: {WATCHLIST}")
    print(f"=== CryptoBot v{VERSION} (avisa si un id de la WATCHLIST no existe) ===")
    threading.Thread(target=ciclo_actualizacion, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, debug=False)
