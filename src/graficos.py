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


def _estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)


def grafica_evento(es: pd.DataFrame, ruta: Path, titulo: str, ylabel: str = "Efecto en puntos porcentuales"):
    """Estudio de eventos θ(e): IC puntual (barras) y banda sup-t (sombra)."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
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
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def grafica_bosque(tabla: pd.DataFrame, ruta: Path, titulo: str, col_nombre="especificacion"):
    """Gráfica de bosque (forest plot) de estimaciones con su IC. tabla: nombre, est_pp, lo_pp, hi_pp."""
    t = tabla.dropna(subset=["est_pp"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(t) + 1.6), dpi=150)
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
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def grafica_tasas(tasas: pd.DataFrame, ruta: Path, titulo: str):
    """Tasa ponderada de primera pareja del mismo sexo por año de inicio, por grupo de legalización."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    for g, sub in tasas.groupby("grupo"):
        ax.plot(sub["anio"], sub["tasa_pct"], lw=1.6 if g != "No tratadas" else 2.2,
                color=GRIS if g == "No tratadas" else None, label=str(g))
    ax.set_xlabel("Año de inicio de vida sexual")
    ax.set_ylabel("% con primera pareja del mismo sexo (ponderado)")
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, title="Año de legalización")
    _estilo(ax)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def grafica_sensibilidad(df: pd.DataFrame, ruta: Path, titulo: str, xlabel: str, punto_quiebre: float | None):
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
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
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)
