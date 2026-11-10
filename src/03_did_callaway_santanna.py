"""
03_did_callaway_santanna.py — MODELO PRINCIPAL (Callaway y Sant'Anna)
======================================================================
Metodología: "Efecto por grupo y año (modelo principal)" + "Juntar todo en una curva (estudio de eventos)"
             + "Pruebas de hipótesis e inferencia".

Qué hace
  1. Estima ATT(g,t) para cada grupo de legalización g y año de inicio sexual t con controles
     "aún no tratados", periodo base universal g−1 y estimador doblemente robusto con X_i =
     (cohorte de nacimiento, sexo, índice socioeconómico) y ponderadores FACTOR.
  2. Agrega a la curva θ(e) con pesos por personas tratadas en cada celda.
  3. H0: ATT = 0 → efecto promedio posterior a la ley (global), y a corto, mediano y largo plazo.
  4. Prueba conjunta de Wald de tendencias paralelas previas (θ(e) = 0 para e < −1).
  5. Errores por bootstrap multiplicador de dos niveles (entidad y UPM; ver did_cs.py), B_BOOT remuestras.
  6. Efecto por AÑO CALENDARIO de inicio sexual (celdas posteriores a la ley), con IC.

Entradas : base_analitica.csv.gz
Salidas  : resultados/tablas/cs_{att_gt,evento,resumen,anio,wald}.csv ; resultados/figuras/cs_{evento,anio}.png
           datos/procesados/cs_boot_evento.csv  (remuestras θ(e), insumo de 07_rambachan_roth.py)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
from did_cs import Espec, estimar_cs, generar_xi
from graficos import grafica_anio, grafica_evento


def main(B: int):
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    C.DIR_PROC.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.BASE_ANALITICA, dtype={"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str})
    m = df[df["muestra_principal"] == 1]

    print(f">> Callaway–Sant'Anna | N = {len(m):,} | bootstrap ({C.INFERENCIA}) B = {B} | procesos = {C.N_JOBS}")
    res = estimar_cs(m, Espec(), B=B, XI=generar_xi(B), n_jobs=C.N_JOBS)
    if res is None:
        sys.exit("No hay celdas estimables: revisa 02_diagnosticos.py")

    res.celdas.round(4).to_csv(C.DIR_TABLAS / "cs_att_gt.csv", index=False)
    res.es.round(4).to_csv(C.DIR_TABLAS / "cs_evento.csv", index=False)
    res.resumen.round(4).to_csv(C.DIR_TABLAS / "cs_resumen.csv", index=False)
    res.omitidas.to_csv(C.DIR_TABLAS / "cs_celdas_omitidas.csv", index=False)
    pd.DataFrame([res.wald]).to_csv(C.DIR_TABLAS / "cs_wald_pretendencias.csv", index=False)
    if res.anio is not None and len(res.anio):
        res.anio.round(4).to_csv(C.DIR_TABLAS / "cs_anio_calendario.csv", index=False)

    # remuestras de θ(e) para la sensibilidad Rambachan–Roth (matriz B × e)
    ie = [i for i, et in enumerate(res.etiquetas) if et[0] == "es"]
    pd.DataFrame(res.boot[:, ie], columns=[f"e{res.etiquetas[i][1]}" for i in ie]).to_csv(
        C.DIR_PROC / "cs_boot_evento.csv", index=False)

    grafica_evento(res.es, C.DIR_FIGURAS / "cs_evento.png",
                   "Matrimonio igualitario y primera pareja del mismo sexo: estudio de eventos (C&S)")

    if res.anio is not None and len(res.anio):
        grafica_anio(res.anio, C.DIR_FIGURAS / "cs_anio.png",
                     "Efecto por año calendario de inicio sexual (C&S, solo periodos posteriores a la ley)")

    print("\n== Efecto promedio (puntos porcentuales; IC {:.0f}%)".format(100 * C.NIVEL_CONFIANZA))
    print(res.resumen.round(3).to_string(index=False))
    print("\n== Estudio de eventos θ(e), pp")
    print(res.es[["e", "theta_pp", "se_pp", "lo_pp", "hi_pp", "n_celdas"]].round(3).to_string(index=False))
    if res.anio is not None and len(res.anio):
        print("\n== Efecto por año calendario de inicio sexual, pp")
        print(res.anio[["anio", "theta_pp", "se_pp", "lo_pp", "hi_pp", "p", "n_celdas", "n_trat"]].round(3).to_string(index=False))
    if res.wald:
        w = res.wald
        print(f"\n== Prueba de Wald de tendencias previas: W = {w['W']:.2f} (gl = {w['gl']}), "
              f"p χ² = {w['p_chi2']:.3f}, p bootstrap = {w['p_boot']:.3f}")
    print(f"\n   Remuestras válidas: {res.B_validas}/{B} | celdas estimables: {len(res.celdas)} | omitidas: {len(res.omitidas)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=C.B_BOOT, help="remuestras del bootstrap")
    main(ap.parse_args().B)
