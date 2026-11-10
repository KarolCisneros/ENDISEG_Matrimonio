"""
graficos.py — Figuras del estudio (matplotlib, sin dependencias extra)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AZUL, GRIS, ROJO = "#1f5fa8", "#6b7280", "#b4361d"

# Tipografía y tamaños pensados para diapositivas (16:9). Cada figura se guarda en PNG y en SVG (vectorial).
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
                     "svg.fonttype": "none"})
FIG_16_9 = (10.0, 5.6)


def _guardar(fig, ruta: Path):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200)
    fig.savefig(ruta.with_suffix(".svg"))
    plt.close(fig)


def _estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)


def grafica_evento(es: pd.DataFrame, ruta: Path, titulo: str, ylabel: str = "Efecto en puntos porcentuales"):
    """Estudio de eventos θ(e): IC puntual (barras) y banda sup-t (sombra)."""
    fig, ax = plt.subplots(figsize=FIG_16_9, dpi=150)
    e = es["e"].values
    if es["supt_lo_pp"].notna().any():
        ax.fill_between(e, es["supt_lo_pp"], es["supt_hi_pp"], color=AZUL, alpha=0.12, lw=0, label="Banda sup-t (uniforme)")
    ax.errorbar(e, es["theta_pp"], yerr=[es["theta_pp"] - es["lo_pp"], es["hi_pp"] - es["theta_pp"]],
                fmt="none", ecolor=AZUL, elinewidth=1.2, capsize=2.5)
    pre, post = es["e"] < 0, es["e"] >= 0
    ax.scatter(e[pre], es.loc[pre, "theta_pp"], color=GRIS, s=26, zorder=3, label="Antes de la ley (tendencias previas)")
    ax.scatter(e[post], es.loc[post, "theta_pp"], color=AZUL, s=30, zorder=3, label="Después de la ley")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color=ROJO, lw=1, ls="--")
    ax.set_xlabel("Años desde la legalización (e = t − g); e = −1 es la referencia")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _estilo(ax)
    fig.tight_layout()
    _guardar(fig, ruta)


def grafica_bosque(tabla: pd.DataFrame, ruta: Path, titulo: str, col_nombre="especificacion"):
    """Gráfica de bosque (forest plot) de estimaciones con su IC. tabla: nombre, est_pp, lo_pp, hi_pp."""
    t = tabla.dropna(subset=["est_pp"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.0, max(3.2, 0.42 * len(t) + 1.6)), dpi=150)
    y = np.arange(len(t))[::-1]
    ax.errorbar(t["est_pp"], y, xerr=[t["est_pp"] - t["lo_pp"], t["hi_pp"] - t["est_pp"]],
                fmt="o", color=AZUL, ecolor=AZUL, elinewidth=1.3, capsize=2.5, ms=5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(t[col_nombre], fontsize=8)
    ax.set_xlabel("Efecto promedio posterior a la ley (puntos porcentuales)")
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold")
    _estilo(ax)
    ax.grid(axis="x", color="#e5e7eb", lw=0.8)
    fig.tight_layout()
    _guardar(fig, ruta)


def grafica_tasas(tasas: pd.DataFrame, ruta: Path, titulo: str):
    """Tasa ponderada de primera pareja del mismo sexo por año de inicio, por grupo de legalización."""
    fig, ax = plt.subplots(figsize=FIG_16_9, dpi=150)
    for g, sub in tasas.groupby("grupo"):
        ax.plot(sub["anio"], sub["tasa_pct"], lw=1.6 if g != "No tratadas" else 2.2,
                color=GRIS if g == "No tratadas" else None, label=str(g))
    ax.set_xlabel("Año de inicio de vida sexual")
    ax.set_ylabel("% con primera pareja del mismo sexo (ponderado)")
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, title="Año de legalización")
    _estilo(ax)
    fig.tight_layout()
    _guardar(fig, ruta)


def grafica_sensibilidad(df: pd.DataFrame, ruta: Path, titulo: str, xlabel: str, punto_quiebre: float | None):
    fig, ax = plt.subplots(figsize=FIG_16_9, dpi=150)
    ax.fill_between(df["M"], df["lo_pp"], df["hi_pp"], color=AZUL, alpha=0.18, lw=0, label="Conjunto identificado + EE")
    ax.plot(df["M"], df["cota_inf_pp"], color=AZUL, lw=1.2)
    ax.plot(df["M"], df["cota_sup_pp"], color=AZUL, lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    if punto_quiebre is not None and np.isfinite(punto_quiebre):
        ax.axvline(punto_quiebre, color=ROJO, ls="--", lw=1, label=f"Punto de quiebre ≈ {punto_quiebre:.3g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Efecto en puntos porcentuales")
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    _estilo(ax)
    fig.tight_layout()
    _guardar(fig, ruta)


def grafica_evento_panel(paneles: list, ruta: Path, titulo: str, ncols: int = 3, ylabel: str = "pp"):
    """Varios estudios de eventos en una sola figura. paneles = [(subtitulo, es_dataframe), ...]."""
    n = len(paneles)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), dpi=150, squeeze=False, sharex=False)
    for ax, (sub, es) in zip(axes.ravel(), paneles):
        if es is None or len(es) == 0:
            ax.text(0.5, 0.5, "sin celdas estimables", ha="center", va="center", transform=ax.transAxes, color=GRIS)
            ax.set_title(sub, fontsize=9, loc="left")
            ax.set_axis_off()
            continue
        e = es["e"].values
        ax.errorbar(e, es["theta_pp"], yerr=[es["theta_pp"] - es["lo_pp"], es["hi_pp"] - es["theta_pp"]],
                    fmt="none", ecolor=AZUL, elinewidth=1, capsize=2)
        pre = es["e"] < 0
        ax.scatter(e[pre], es.loc[pre, "theta_pp"], color=GRIS, s=18, zorder=3)
        ax.scatter(e[~pre], es.loc[~pre, "theta_pp"], color=AZUL, s=20, zorder=3)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(-0.5, color=ROJO, lw=0.9, ls="--")
        ax.set_title(sub, fontsize=9, loc="left", fontweight="bold")
        ax.set_xlabel("e = t − g", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        _estilo(ax)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, ruta)


def grafica_anio(anio: pd.DataFrame, ruta: Path, titulo: str):
    """Efecto por AÑO CALENDARIO de inicio sexual (solo periodos posteriores a la ley)."""
    fig, ax = plt.subplots(figsize=FIG_16_9, dpi=150)
    a = anio.sort_values("anio")
    ax.errorbar(a["anio"], a["theta_pp"], yerr=[a["theta_pp"] - a["lo_pp"], a["hi_pp"] - a["theta_pp"]],
                fmt="o", color=AZUL, ecolor=AZUL, elinewidth=1.2, capsize=2.5, ms=5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Año calendario de inicio de la vida sexual (solo entidades ya con matrimonio igualitario)")
    ax.set_ylabel("Efecto en puntos porcentuales")
    ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(a["anio"].astype(int))
    ax.tick_params(axis="x", labelrotation=45)
    _estilo(ax)
    fig.tight_layout()
    _guardar(fig, ruta)


def grafica_loo(tabla: pd.DataFrame, ruta: Path, titulo: str, ref_pp: float):
    """Efecto global dejando fuera una entidad a la vez (leave-one-out)."""
    t = tabla.dropna(subset=["est_pp"]).sort_values("est_pp").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.28 * len(t) + 1.5)), dpi=150)
    y = np.arange(len(t))
    col = np.where(t["amparo_temprano"].fillna(0).astype(int) == 1, ROJO, AZUL)
    ax.scatter(t["est_pp"], y, c=col, s=28, zorder=3)
    ax.axvline(ref_pp, color="black", lw=0.9, ls="--", label=f"Con todas las entidades ({ref_pp:.2f} pp)")
    ax.axvline(0, color=GRIS, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.entidad} (G={r.G})" for r in t.itertuples()], fontsize=8)
    ax.set_xlabel("Efecto promedio posterior a la ley al excluir la entidad (pp). Rojo = amparos tempranos")
    ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    _estilo(ax)
    ax.grid(axis="x", color="#e5e7eb", lw=0.8)
    fig.tight_layout()
    _guardar(fig, ruta)
