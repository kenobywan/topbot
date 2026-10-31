"""Dashboard visual del predictor: misma idea de layout (tabla + gráfico por moneda),
mejorado en cálculos y decisiones.

Salidas:
  * reports/predictions.html : dashboard interactivo (Plotly, si está instalado)
  * reports/prices.png       : panel PNG con velas recientes + banda prevista
                               + proyección al horizonte principal (matplotlib)

Los cálculos provienen siempre de topbot.predictor (modelo calibrado, sin
look-ahead, neto de comisiones); este módulo solo dibuja, no decide.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("topbot.dashboard")

ACTION_COLORS = {"COMPRAR": "#16c784", "ESPERAR": "#f5a623", "VENDER": "#ea3943"}
ACTION_EMOJI = {"COMPRAR": "🟢", "ESPERAR": "🟡", "VENDER": "🔴"}
CONF_COLOR = {"ALTA": "#16c784", "MEDIA": "#9ac8ff", "BAJA": "#f5a623",
              "MUY BAJA": "#ea3943"}


def _fmt_price(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.3f}"
    if v >= 0.01:
        return f"{v:.5f}"
    return f"{v:.7f}"


# ---------------------------------------------------------------------------
# Panel PNG (matplotlib) — gráfico por moneda
# ---------------------------------------------------------------------------

def plot_coins_png(results: list[dict], dfs: dict[str, pd.DataFrame],
                   path: str, lookback_bars: int = 240, cols: int = 2) -> bool:
    """Panel con un subgráfico por moneda: velas recientes, banda p20-p80 y
    línea de precio estimado al horizonte principal. Devuelve True si guardó."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        from matplotlib.lines import Line2D
    except ImportError:
        log.warning("matplotlib no instalado; sin PNG")
        return False

    results = [r for r in results if r]
    if not results:
        return False
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.0 * rows),
                             squeeze=False, dpi=110)
    for ax in np.asarray(axes).ravel():
        ax.set_visible(False)

    for ax, r in zip(np.asarray(axes).ravel(), results):
        sym = r["symbol"]
        df = dfs.get(sym)
        prim = r["primary"]
        action = r["action"]
        color = ACTION_COLORS.get(action, "#888")
        if df is None or len(df) == 0:
            ax.set_title(f"{sym} — sin datos", color="red")
            continue
        d = df.tail(lookback_bars)
        x = np.asarray(pd.DatetimeIndex(d.index).to_pydatetime(), dtype="datetime64[s]")
        n = len(d)
        # eje extendido hacia el futuro para la proyección
        step = (x[1] - x[0]) if n > 1 else pd.Timedelta(hours=1).to_pytimedelta()
        h_used = prim["horizon_used"]
        bph = {"1m": 60, "5m": 12, "15m": 4, "30m": 2, "1h": 1,
               "4h": 0.25, "1d": 1 / 24}.get(r.get("interval", "1h"), 1)
        fut_bars = max(2, int(round(h_used * bph)))
        xf = [x[-1]]
        for i in range(1, fut_bars + 1):
            xf.append(x[-1] + i * step)
        xf = np.array(xf)

        # banda: percentil 20/80 del horizonte principal proyectado hacia el
        # futuro desde el presente (crecimiento de incertidumbre ~ sqrt(t))
        p_lo, p_hi = prim["price_range"]
        p_med = prim["price_median"]
        p_now = r["price_now"]
        tfrac = (np.arange(len(xf)) - (len(x) - 1)) / max(1, len(xf) - 1)
        tfrac = np.clip(tfrac, 0, 1)
        band_lo = p_now * np.exp(np.log(max(p_lo, 1e-12) / max(p_now, 1e-12)) * np.sqrt(tfrac))
        band_hi = p_now * np.exp(np.log(max(p_hi, 1e-12) / max(p_now, 1e-12)) * np.sqrt(tfrac))
        med_line = p_now * np.exp(np.log(max(p_med, 1e-12) / max(p_now, 1e-12)) * tfrac)

        ax.fill_between(x, d["low"], d["high"], color="#dfe6ee", alpha=.5, zorder=1)
        up = d["close"] >= d["open"]
        ax.vlines(x, d["open"], d["close"], color=np.where(up, "#16c784", "#ea3943"),
                  linewidth=.8, zorder=2)
        ax.plot(x, d["close"], color="#2b3a55", linewidth=1.0, alpha=.7, zorder=3)
        ax.fill_between(xf[len(x) - 1:], band_lo[len(x) - 1:], band_hi[len(x) - 1:],
                        color=color, alpha=.18, label="rango p20–p80", zorder=2)
        ax.plot(xf[len(x) - 1:], med_line[len(x) - 1:], "--", color=color,
                linewidth=1.8, label=f"estimado {prim['horizon_hours']}h", zorder=4)

        title = (f"{sym}  {_fmt_price(p_now)}  →  {_fmt_price(p_med)} "
                 f"({prim['exp_return_net_fee'] * 100:+.2f}% neto)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(.02, .96, f"{ACTION_EMOJI.get(action, '')} {action}",
                transform=ax.transAxes, va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=.9))
        ax.text(.98, .96,
                f"P(alza)={prim['p_up']:.0%} · conf. {prim['confidence']} · OOS {prim['acc_oos'] * 100:.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=8, color="#555")
        ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.25)
        ax.set_ylim(min(float(np.nanmin(band_lo)), float(d["low"].min())) * .995,
                    max(float(np.nanmax(band_hi)), float(d["high"].max())) * 1.005)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles = [Line2D([], [], marker="s", ls="", color=c, label=a)
               for a, c in ACTION_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    ts = results[0].get("generated_at", "")
    fig.suptitle(f"TOPBOT Predictor — señales spot ({ts})", fontsize=13,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(path)
    plt.close(fig)
    log.info("PNG guardado: %s", path)
    return True


# ---------------------------------------------------------------------------
# Dashboard HTML interactivo (Plotly)
# ---------------------------------------------------------------------------

def render_html(results: list[dict], dfs: dict[str, pd.DataFrame],
                path: str, cfg_summary: str = "") -> bool:
    """Dashboard autocontenido: tabla ranking + pestañas con gráfico por moneda.
    Usa Plotly si está disponible; si no, devuelve False (el llamador puede
    pedir --png como alternativa)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly no instalado; sin HTML interactivo")
        return False

    results = [r for r in results if r]
    if not results:
        return False

    # --- Tabla ranking -----------------------------------------------------
    rows_html = []
    for i, r in enumerate(results, 1):
        p = r["primary"]
        color = ACTION_COLORS.get(r["action"], "#888")
        cc = CONF_COLOR.get(p["confidence"], "#888")
        ret = p["exp_return_net_fee"] * 100
        hl = [f"<b>{k}</b>: {v}" for k, v in r.get("highlights", {}).items()]
        rows_html.append(f"""
        <tr data-target="chart-{i}" onclick="document.getElementById('chart-{i}').scrollIntoView({{behavior:'smooth'}})"
            style="--ac:{color}">
          <td>{i}</td><td class="sym">{r['symbol']}</td>
          <td>{_fmt_price(r['price_now'])}</td>
          <td><span class="badge" style="background:{color}">{r['action']}</span></td>
          <td>{p['p_up']:.0%}</td>
          <td class="{'pos' if ret >= 0 else 'neg'}">{ret:+.2f}%</td>
          <td>{_fmt_price(p['price_median'])}</td>
          <td>{_fmt_price(p['price_range'][0])} – {_fmt_price(p['price_range'][1])}</td>
          <td><span class="conf" style="color:{cc}">● {p['confidence']}</span></td>
          <td>{p['acc_oos'] * 100:.1f}%</td>
          <td class="why">{r['reason']}<br><span class="hl">✦ {' · '.join(hl)}</span></td>
        </tr>""")

    # --- Figuras por moneda -------------------------------------------------
    figs_html = []
    for i, r in enumerate(results, 1):
        sym = r["symbol"]
        df = dfs.get(sym)
        if df is None or len(df) == 0:
            continue
        d = df.tail(240)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                            vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"],
                                     low=d["low"], close=d["close"], name=sym,
                                     increasing_line_color="#16c784",
                                     decreasing_line_color="#ea3943"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["volume"], name="Volumen",
                                 marker_color="#8fa7c9", opacity=.55), row=2, col=1)
        fig.update_layout(yaxis2=dict(range=[0, float(d["volume"].max()) * 3]),
                          legend=dict(orientation="h", y=1.08))

        prim = r["primary"]
        step = (d.index[-1] - d.index[-2]) if len(d) > 1 else pd.Timedelta(hours=1)
        bph = {"1m": 60, "5m": 12, "15m": 4, "30m": 2, "1h": 1,
               "4h": 0.25, "1d": 1 / 24}.get(r.get("interval", "1h"), 1)
        fut = max(2, int(round(prim["horizon_used"] * bph)))
        xf = pd.date_range(d.index[-1], periods=fut + 1, freq=step)
        p_now, p_med = r["price_now"], prim["price_median"]
        p_lo, p_hi = prim["price_range"]
        t = np.linspace(0, 1, fut + 1)
        med = p_now * np.exp(np.log(max(p_med, 1e-12) / max(p_now, 1e-12)) * t)
        hi_curve = p_now * np.exp(np.log(max(p_hi, 1e-12) / max(p_now, 1e-12)) * np.sqrt(t))
        lo_curve = p_now * np.exp(np.log(max(p_lo, 1e-12) / max(p_now, 1e-12)) * np.sqrt(t))
        color = ACTION_COLORS.get(r["action"], "#888")
        fig.add_trace(go.Scatter(x=list(xf) + list(xf)[::-1],
                                 y=list(hi_curve) + list(lo_curve)[::-1],
                                 fill="toself", line=dict(width=0),
                                 fillcolor=color + "33", name="rango p20–p80",
                                 hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xf, y=med, mode="lines", name="precio estimado",
                                 line=dict(color=color, dash="dash", width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[xf[-1]], y=[p_med], mode="markers+text",
                                 marker=dict(color=color, size=9),
                                 text=[_fmt_price(p_med)], textposition="top right",
                                 showlegend=False), row=1, col=1)

        # --- Niveles de decisión para compra/venta MANUAL en spot ---------
        # soportes/resistencias recientes (pivot highs/lows de las últimas velas)
        win = min(len(d), 120)
        dd = d.iloc[-win:]
        highs_l, lows_l = dd["high"].values, dd["low"].values
        piv_hi = [highs_l[j] for j in range(2, win - 2)
                  if highs_l[j] == max(highs_l[max(0, j - 2):j + 3])]
        piv_lo = [lows_l[j] for j in range(2, win - 2)
                  if lows_l[j] == min(lows_l[max(0, j - 2):j + 3])]
        res = float(np.median(piv_hi)) if piv_hi else float(dd["high"].max())
        sup = float(np.median(piv_lo)) if piv_lo else float(dd["low"].min())
        atr_h = float(abs(np.log(max(p_hi, 1e-12) / max(p_now, 1e-12))
                          - np.log(max(p_lo, 1e-12) / max(p_now, 1e-12))) / 2.0
                      * p_now / max(fut, 1))  # ancho medio de banda por vela
        buy_zone = p_med - 0.5 * atr_h * np.sqrt(fut) * (1 - prim["p_up"])
        sell_stop = sup - 1.5 * atr_h if p_now <= res else p_now * (1 - 0.03)
        take_profit = p_med + 1.0 * atr_h * np.sqrt(fut)
        lvl_styles = [
            (res,  "#8a6df0", "R resistencia"),
            (sup,  "#f0a202", "S soporte"),
            (take_profit, "#16c784", "TP objetivo"),
            (sell_stop,   "#ea3943", "SL stop"),
        ]
        for val, ccol, lbl in lvl_styles:
            fig.add_hline(y=float(val), line=dict(color=ccol, width=1.2, dash="dot"),
                          annotation_text=f"{lbl} {_fmt_price(float(val))}",
                          annotation_position="right",
                          annotation_font=dict(size=10, color=ccol), row=1, col=1)

        fig.update_layout(template="plotly_white", height=460, margin=dict(l=40, r=110, t=60, b=30),
                          xaxis_rangeslider_visible=False,
                          title=dict(text=(
                              f"<b>{sym}</b> — {r['action']} · P(alza {prim['horizon_hours']}h)="
                              f"{prim['p_up']:.0%} · neto {prim['exp_return_net_fee'] * 100:+.2f}% "
                              f"· confianza {prim['confidence']}"), font=dict(size=14)))
        div = fig.to_html(full_html=False, include_plotlyjs=False,
                         config={"displaylogo": False})
        # Tarjeta de niveles accionables bajo el gráfico
        card = (
            f'<div class="levels">'
            f'<span><b>Zona de compra</b>: {_fmt_price(float(buy_zone))} – {_fmt_price(float(p_med))}</span>'
            f'<span><b>Objetivo (TP)</b>: {_fmt_price(float(take_profit))}</span>'
            f'<span><b>Stop (SL)</b>: {_fmt_price(float(sell_stop))}</span>'
            f'<span><b>Soporte</b>: {_fmt_price(sup)} · <b>Resistencia</b>: {_fmt_price(res)}</span>'
            f'</div>'
        )
        figs_html.append(f'<div class="coin" id="chart-{i}">{div}{card}'
                         f'<p class="reason">{r["reason"]}</p></div>')

    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TOPBOT Predictor</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
 body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f4f7fb;margin:0;color:#1c2733}}
 header{{background:linear-gradient(90deg,#0b1e3a,#123a6b);color:#fff;padding:18px 28px}}
 header h1{{margin:0;font-size:20px}} header p{{margin:4px 0 0;opacity:.75;font-size:12px}}
 main{{max-width:1280px;margin:0 auto;padding:20px}}
 table{{border-collapse:collapse;width:100%;background:#fff;border-radius:12px;overflow:hidden;
        box-shadow:0 2px 10px rgba(16,40,80,.08);font-size:13px}}
 th{{background:#0b1e3a;color:#cfe3ff;text-align:left;padding:9px 10px;font-weight:600}}
 td{{padding:9px 10px;border-top:1px solid #edf1f6;vertical-align:top}}
 tr:hover td{{background:#f0f6ff}} tr{{cursor:pointer;border-left:4px solid var(--ac)}}
 .sym{{font-weight:700}} .pos{{color:#0a8f5b;font-weight:600}} .neg{{color:#c62828;font-weight:600}}
 .badge{{color:#fff;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700}}
 .conf{{font-weight:600;font-size:12px}} .why{{max-width:420px;color:#41506a;font-size:12px}}
 .hl{{color:#7a5cff;font-size:11px}}
 .coin{{background:#fff;border-radius:12px;margin-top:22px;padding:8px 12px;
        box-shadow:0 2px 10px rgba(16,40,80,.08)}}
 .levels{{display:flex;flex-wrap:wrap;gap:8px 18px;padding:6px 12px 2px;font-size:12px;
        color:#1c2733;background:#f0f6ff;border-radius:8px;margin:0 4px 6px}}
 .levels span{{white-space:nowrap}}
 .reason{{color:#41506a;font-size:12px;padding:0 12px 8px}}
 footer{{text-align:center;color:#8a97ab;font-size:11px;padding:26px}}
</style></head><body>
<header><h1>📈 TOPBOT Predictor — decisión manual en Spot</h1>
<p>{cfg_summary}</p></header>
<main>
<h2>Ranking por ventaja esperada neta</h2>
<table><thead><tr><th>#</th><th>Moneda</th><th>Precio</th><th>Acción</th><th>P(alza)</th>
<th>Neto %</th><th>Estimado</th><th>Rango p20–p80</th><th>Confianza</th><th>OOS %</th>
<th>Motivo &amp; drivers</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table>
<h2 style="margin-top:28px">Gráficos por moneda</h2>
{''.join(figs_html)}
</main>
<footer>No es asesoramiento financiero · probabilidades validadas walk-forward purgado ·
prioriza señales ALTA/MEDIA</footer>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("HTML guardado: %s", path)
    return True
