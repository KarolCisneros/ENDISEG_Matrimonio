"""
10_multiplicidad.py — Ajuste de p-valores por comparaciones múltiples
=====================================================================
Se probaron muchas hipótesis (11 grupos de legalización, ~15 submuestras, 17 especificaciones de robustez,
9 resultados de mecanismos). Con tantas pruebas, algún p < 0.05 aparece por azar. Este paso agrega a cada
tabla las columnas p_holm (FWER) y p_bh (FDR), calculadas DENTRO de cada familia de pruebas:

  * efectos por grupo de legalización (cs_resumen.csv)
  * heterogeneidad: submuestras (heterogeneidad_resumen.csv) y diferencias entre submuestras
  * robustez: las 16 especificaciones distintas de la principal (robustez_resumen.csv)
  * mecanismos: los resultados por población (mecanismos_resumen.csv)
  * estudio de eventos: coeficientes previos (cs_evento.csv, e < −1)

Entradas : tablas de los pasos 03, 05, 06 y 08     Salidas: resultados/tablas/*_ajustado.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as C
from multiplicidad import bh, holm


def _ajusta(df, p="p", mascara=None):
    df = df.copy()
    mascara = df[p].notna() if mascara is None else (mascara & df[p].notna())
    df["p_holm"], df["p_bh"] = float("nan"), float("nan")
    if mascara.sum() > 0:
        df.loc[mascara, "p_holm"] = holm(df.loc[mascara, p].values)
        df.loc[mascara, "p_bh"] = bh(df.loc[mascara, p].values)
    df["n_pruebas_familia"] = int(mascara.sum())
    return df


def _leer(nombre):
    ruta = C.DIR_TABLAS / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


def main():
    familias = [
        ("cs_resumen.csv", "cs_resumen_ajustado.csv", lambda d: d["nombre"].astype(str).str.startswith("grupo_"), "p",
         "Efectos por grupo de legalización"),
        ("heterogeneidad_resumen.csv", "heterogeneidad_resumen_ajustado.csv", lambda d: d["grupo"] != "Todas", "p",
         "Submuestras (heterogeneidad)"),
        ("heterogeneidad_diferencias.csv", "heterogeneidad_diferencias_ajustado.csv", None, "p",
         "Diferencias entre submuestras"),
        ("robustez_resumen.csv", "robustez_resumen_ajustado.csv", lambda d: ~d["especificacion"].astype(str).str.startswith("0."), "p",
         "Especificaciones de robustez"),
        ("mecanismos_resumen.csv", "mecanismos_resumen_ajustado.csv", None, "p", "Mecanismos"),
        ("cs_evento.csv", "cs_evento_ajustado.csv", lambda d: d["e"] < -1, "p", "Coeficientes previos"),
    ]
    for entrada, salida, filtro, p, titulo in familias:
        d = _leer(entrada)
        if d is None:
            print(f"   (se omite {entrada}: no existe)")
            continue
        out = _ajusta(d, p, None if filtro is None else filtro(d))
        out.round(4).to_csv(C.DIR_TABLAS / salida, index=False)
        k = out["p_holm"].notna().sum()
        print(f"\n== {titulo}: {k} pruebas | p<0.05 sin ajustar: {int((out.loc[out['p_holm'].notna(), p] < 0.05).sum())} | "
              f"con Holm: {int((out['p_holm'] < 0.05).sum())} | con BH: {int((out['p_bh'] < 0.05).sum())}")
        sig = out[(out[p] < 0.05) & out["p_holm"].notna()]
        if len(sig):
            cols = [c for c in ["nombre", "grupo", "especificacion", "poblacion", "resultado", "e", "est_pp", p, "p_holm", "p_bh"] if c in sig.columns]
            print(sig[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
