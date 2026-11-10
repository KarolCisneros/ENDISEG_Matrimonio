"""
05_heterogeneidad.py — Heterogeneidad del efecto (dimensiones definidas de antemano)
====================================================================================
Metodología, "Heterogeneidad": se exploran cuatro dimensiones fijadas de antemano:
  (1) orientación sexual autoidentificada — solo como heterogeneidad, con la cautela de que la
      legalización pudo modificar quién se identifica como no heterosexual;
  (2) sexo;
  (3) nivel socioeconómico (terciles del índice de bienes y vivienda);
  (4) cohorte — para distinguir a quien inició su vida sexual poco después de la ley de quien lo
      hizo mucho después (resumen dinámico corto / mediano / largo plazo en cada submuestra) y por
      cohorte de nacimiento.
Más la nota de la metodología "tomar en cuenta: cuándo se dieron cuenta": dentro de LGBTI+, se separa a
quienes ya sabían su orientación al momento del debut de quienes se dieron cuenta después.

Cada submuestra se estima con el mismo estimador que el modelo principal (cada ATT(g,t) se calcula
DENTRO de la submuestra), con las mismas remuestras bootstrap por entidad (XI compartida), lo que
permite probar diferencias entre grupos con remuestras pareadas.

Entradas : base_analitica.csv.gz
Salidas  : resultados/tablas/heterogeneidad_{resumen,diferencias}.csv ; figuras/heterogeneidad.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from did_cs import Espec, estimar_cs, fila_resultado, generar_xi
from graficos import grafica_bosque

N_MIN_SUBMUESTRA = 150


def definir_submuestras(m: pd.DataFrame):
    """(dimensión, grupo, máscara). El primer grupo de cada dimensión es la referencia en las diferencias."""
    s = []
    s.append(("Todas", "Todas", m.index == m.index))
    s += [("Orientación (OSIG)", "Normativa (heterosexual y cis)", m["grupo_osig"] == "normativa"),
          ("Orientación (OSIG)", "LGBTI+", m["grupo_osig"] == "LGBTI+")]
    s += [("Orientación detallada", "Heterosexual", m["orient_cat"] == "heterosexual"),
          ("Orientación detallada", "Gay / lesbiana", m["orient_cat"].isin(["gay", "lesbiana"])),
          ("Orientación detallada", "Bisexual", m["orient_cat"] == "bisexual")]
    s += [("Sexo al nacer", "Hombres", m["mujer"] == 0), ("Sexo al nacer", "Mujeres", m["mujer"] == 1)]
    for t, nom in [(1, "Bajo"), (2, "Medio"), (3, "Alto")]:
        s.append(("Nivel socioeconómico", nom, m["ses_tercil"] == t))
    for c in ["<1970", "1970-79", "1980-89", "1990-99", "2000+"]:
        s.append(("Cohorte de nacimiento", c, m["cohorte_grupo"].astype(str) == c))
    lg = m["grupo_osig"] == "LGBTI+"
    s += [("LGBTI+ según momento de darse cuenta", "Ya sabía su orientación al debut", lg & (m["consciente_al_debut"] == 1)),
          ("LGBTI+ según momento de darse cuenta", "Se dio cuenta después del debut", lg & (m["consciente_al_debut"] == 0))]
    return s


def main(B: int):
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.BASE_ANALITICA, dtype={"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str})
    m = df[df["muestra_principal"] == 1].reset_index(drop=True)
    XI = generar_xi(B)
    filas, draws = [], {}
    for dim, nombre, mask in definir_submuestras(m):
        sub = m[mask]
        if len(sub) < N_MIN_SUBMUESTRA:
            print(f"   [{dim} | {nombre}] N={len(sub)} < {N_MIN_SUBMUESTRA}: se omite")
            continue
        esp = Espec(nombre=f"{dim} | {nombre}", min_obs=C.MIN_OBS_CELDA if nombre == "Todas" else C.MIN_OBS_CELDA_SUBMUESTRAS)
        res = estimar_cs(sub, esp, B=B, XI=XI, n_jobs=C.N_JOBS)
        filas.append(fila_resultado(res, dimension=dim, grupo=nombre))
        if res is not None and res.boot is not None:
            draws[(dim, nombre)] = res.draws(("resumen", "global_post"))
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(C.DIR_TABLAS / "heterogeneidad_resumen.csv", index=False)

    # diferencias contra el primer grupo de cada dimensión (remuestras pareadas por entidad)
    dif = []
    for dim, g in tabla.groupby("dimension", sort=False):
        if dim == "Todas" or len(g) < 2:
            continue
        ref = g.iloc[0]
        for _, r in g.iloc[1:].iterrows():
            a, b = draws.get((dim, r["grupo"])), draws.get((dim, ref["grupo"]))
            if a is None or b is None or len(a) != len(b):
                continue
            d = (r["est_pp"] - ref["est_pp"]) / 100
            se = float(np.std(a - b, ddof=1))
            dif.append(dict(dimension=dim, grupo=r["grupo"], referencia=ref["grupo"], dif_pp=100 * d, se_pp=100 * se,
                            p=float(2 * (1 - stats.norm.cdf(abs(d / se)))) if se > 0 else np.nan))
    pd.DataFrame(dif).round(4).to_csv(C.DIR_TABLAS / "heterogeneidad_diferencias.csv", index=False)

    print("\n== Efecto promedio posterior a la ley por submuestra (pp)")
    cols = ["dimension", "grupo", "n", "celdas", "est_pp", "se_pp", "lo_pp", "hi_pp", "p", "corto_est_pp", "largo_est_pp"]
    print(tabla[[c for c in cols if c in tabla.columns]].round(3).to_string(index=False))
    if dif:
        print("\n== Diferencias contra el grupo de referencia (remuestras pareadas)")
        print(pd.DataFrame(dif).round(3).to_string(index=False))
    t = tabla.dropna(subset=["est_pp"]).copy()
    t["especificacion"] = t["dimension"].str.slice(0, 22) + " · " + t["grupo"]
    if len(t):
        grafica_bosque(t, C.DIR_FIGURAS / "heterogeneidad.png", "Heterogeneidad del efecto de la legalización")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=C.B_BOOT_SECUNDARIO)
    main(ap.parse_args().B)
