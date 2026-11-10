"""
09_diagnostico_entidades.py — ¿Qué entidades mueven el resultado?
==================================================================
Motivo: en 06_robustez.py, excluir las entidades con amparos tempranos (especificación 3a) cambió el
efecto de +1.3 a +3.1 pp. Con solo 32 entidades (21 tratadas en 10 grupos) un estimado puede depender de
unas cuantas. Este script lo descompone:

  1. Descripción de las entidades con amparo temprano (grupo, fecha, personas, casos Y = 1).
  2. Leave-one-out: el efecto promedio posterior a la ley dejando fuera UNA entidad a la vez (32 estimaciones
     puntuales, sin bootstrap) y cuánto cambia respecto al modelo con todas.
  3. Leave-one-group-out: lo mismo dejando fuera un GRUPO de legalización completo.
  4. Efecto excluyendo cada entidad con amparo temprano por separado y todas juntas (confirma 3a).

Entradas : base_analitica.csv.gz
Salidas  : resultados/tablas/diag_{amparos,loo_entidad,loo_grupo}.csv ; figuras/loo_entidades.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
from did_cs import Espec, estimar_cs
from graficos import grafica_loo


def _efecto(base):
    res = estimar_cs(base, Espec(), B=0, verbose=False)
    if res is None:
        return np.nan, 0
    r = res.get("global_post")
    return (np.nan if r is None else float(r["est_pp"])), len(res.celdas)


def main():
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.BASE_ANALITICA, dtype={"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str, "ENT": str})
    m = df[df["muestra_principal"] == 1].reset_index(drop=True)
    ref, ncel = _efecto(m)
    print(f">> Efecto promedio posterior con todas las entidades: {ref:.3f} pp ({ncel} celdas)")

    # --- 1. descripción por entidad
    d = (m.groupby(["ENT", "entidad"], as_index=False)
           .agg(G=("G", "first"), amparo_temprano=("amparo_temprano", "first"),
                discrepancia_fuentes=("discrepancia_fuentes", "first"),
                n=("FACTOR", "size"), n_Y1=("Y_mismo_sexo", "sum"), poblacion=("FACTOR", "sum")))
    d["G"] = d["G"].astype(int)
    d["tratada"] = (d["G"] != C.ANIO_NO_TRATADO).astype(int)
    amp = d[d["amparo_temprano"] == 1].sort_values("G")
    amp.round(1).to_csv(C.DIR_TABLAS / "diag_amparos.csv", index=False)
    print("\n== Entidades con amparo temprano")
    print(amp[["ENT", "entidad", "G", "n", "n_Y1", "discrepancia_fuentes"]].to_string(index=False))

    # --- 2. leave-one-out por entidad
    filas = []
    for r in d.itertuples():
        est, nc = _efecto(m[m["ENT"] != r.ENT])
        filas.append(dict(ENT=r.ENT, entidad=r.entidad, G=r.G, amparo_temprano=r.amparo_temprano, n=r.n, n_Y1=r.n_Y1,
                          est_pp=est, celdas=nc, cambio_pp=est - ref))
    loo = pd.DataFrame(filas)
    loo.round(4).to_csv(C.DIR_TABLAS / "diag_loo_entidad.csv", index=False)
    print("\n== Leave-one-out por entidad (efecto promedio posterior sin esa entidad, pp)")
    print(loo.sort_values("cambio_pp")[["entidad", "G", "amparo_temprano", "est_pp", "cambio_pp"]].round(3).to_string(index=False))
    grafica_loo(loo, C.DIR_FIGURAS / "loo_entidades.png", "Sensibilidad a excluir una entidad (leave-one-out)", ref)

    # --- 3. leave-one-group-out
    g_filas = []
    for g in sorted(m["G"].unique()):
        est, nc = _efecto(m[m["G"] != g])
        g_filas.append(dict(G_excluido=int(g), n_excluidos=int((m["G"] == g).sum()), est_pp=est, celdas=nc, cambio_pp=est - ref))
    pd.DataFrame(g_filas).round(4).to_csv(C.DIR_TABLAS / "diag_loo_grupo.csv", index=False)
    print("\n== Leave-one-group-out (grupo de legalización completo; 9999 = no tratadas)")
    print(pd.DataFrame(g_filas).round(3).to_string(index=False))

    # --- 4. excluir amparos: uno por uno y juntas
    a_filas = []
    for r in amp.itertuples():
        est, nc = _efecto(m[(m["ENT"] != r.ENT)])
        a_filas.append(dict(excluye=r.entidad, est_pp=est, celdas=nc))
    est, nc = _efecto(m[m["amparo_temprano"] == 0])
    a_filas.append(dict(excluye="TODAS las de amparo temprano", est_pp=est, celdas=nc))
    pd.DataFrame(a_filas).round(4).to_csv(C.DIR_TABLAS / "diag_excluir_amparos.csv", index=False)
    print("\n== Excluir entidades con amparo temprano (individual y en conjunto)")
    print(pd.DataFrame(a_filas).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
