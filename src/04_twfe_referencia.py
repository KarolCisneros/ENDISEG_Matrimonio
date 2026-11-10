"""
04_twfe_referencia.py — Modelo clásico TWFE (solo como referencia)
===================================================================
Metodología: "Modelo clásico, solo como referencia (TWFE)":

    Y_ist = α_s + λ_t + β D_st + X_i'γ + ε_ist

  α_s  efecto fijo de entidad (nivel base de primeras veces entre personas del mismo sexo)
  λ_t  efecto fijo del año de inicio sexual (absorbe cambios nacionales de ese año)
  D_st = 1 si en la entidad s ya estaba legalizado el matrimonio en el año t
  β    efecto de la ley según la regresión tradicional.

Se estima por mínimos cuadrados ponderados con FACTOR. Errores: agrupados por entidad (CR1,
t con 31 gl) y bootstrap multiplicador por entidad. Se compara con el efecto global del modelo
principal (03): si difieren, el TWFE está sesgado por comparaciones "prohibidas" (tratados vs.
ya tratados) cuando el efecto cambia con el tiempo.

Entradas : base_analitica.csv.gz ; resultados/tablas/cs_resumen.csv (si existe, para la comparación)
Salidas  : resultados/tablas/twfe_vs_cs.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as C
from did_cs import Espec, generar_xi, twfe


def main(B: int):
    df = pd.read_csv(C.BASE_ANALITICA, dtype={"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str})
    m = df[df["muestra_principal"] == 1]
    r = twfe(m, Espec(), B=B, XI=generar_xi(B))
    print(f">> TWFE: β = {r['beta_pp']:.3f} pp | EE CR1 = {r['se_cr1_pp']:.3f} | EE bootstrap = {r.get('se_boot_pp', float('nan')):.3f}"
          f" | IC {100 * C.NIVEL_CONFIANZA:.0f}% = [{r['lo_pp']:.3f}, {r['hi_pp']:.3f}] | N = {r['n']:,}")
    filas = [dict(modelo="TWFE (referencia)", est_pp=r["beta_pp"], se_pp=r.get("se_boot_pp", r["se_cr1_pp"]),
                  lo_pp=r["lo_pp"], hi_pp=r["hi_pp"], p=r.get("p_boot", r["p_cr1"]))]
    ruta = C.DIR_TABLAS / "cs_resumen.csv"
    if ruta.exists():
        cs = pd.read_csv(ruta)
        g = cs[cs["nombre"] == "global_post"].iloc[0]
        filas.insert(0, dict(modelo="Callaway–Sant'Anna (global posterior)", est_pp=g["est_pp"], se_pp=g["se_pp"],
                             lo_pp=g["lo_pp"], hi_pp=g["hi_pp"], p=g["p"]))
    out = pd.DataFrame(filas)
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(C.DIR_TABLAS / "twfe_vs_cs.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=C.B_BOOT)
    main(ap.parse_args().B)
