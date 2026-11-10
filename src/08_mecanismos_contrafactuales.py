"""
08_mecanismos_contrafactuales.py — ¿Se adelanta o se cambia la primera pareja?
==============================================================================
Metodología, "Contrafactuales": un aumento de Y (primera pareja del mismo sexo) puede deberse a
  1. que la persona ADELANTA una relación con el mismo sexo que de todos modos habría tenido, o
  2. que la persona CAMBIA a una relación con el mismo sexo que nunca habría tenido.
Operacionalización (propuesta; confirmar con el equipo de investigación):
  * "Se cambia"  (margen extensivo): Z = 1 si ALGUNA VEZ en la vida tuvo relaciones con alguien del mismo
        sexo (P7_8), en toda la muestra. Un aumento de Z significa que más personas llegan a tener una
        pareja del mismo sexo.
  * "Se adelanta" (margen de calendario): Y (primera pareja del mismo sexo) SOLO entre quienes alguna vez
        tuvieron pareja del mismo sexo. Un aumento de Y condicionando a Z = 1 significa que, entre quienes
        la tendrían de todos modos, esa relación pasa a ser la primera.
Cautela: condicionar en Z = 1 (resultado posterior) es una descomposición descriptiva, no un efecto causal
puro; Z observa la vida entera, así que quienes iniciaron más recientemente tienen menos exposición
(el efecto fijo de periodo lo absorbe porque afecta igual a tratados y controles).
Se corre en toda la muestra, en LGBTI+ y en OSIG normativa (placebo).

Entradas : base_analitica.csv.gz   Salidas: resultados/tablas/mecanismos_resumen.csv ; figuras/mecanismos.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as C
from did_cs import Espec, estimar_cs, fila_resultado, generar_xi
from graficos import grafica_bosque


def main(B: int):
    df = pd.read_csv(C.BASE_ANALITICA, dtype={"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str})
    m = df[df["muestra_principal"] == 1].reset_index(drop=True)
    XI = generar_xi(B)
    pobl = [("Toda la muestra", m), ("LGBTI+", m[m["grupo_osig"] == "LGBTI+"]),
            ("OSIG normativa (placebo)", m[m["grupo_osig"] == "normativa"])]
    filas = []
    for nombre_p, base in pobl:
        mo = C.MIN_OBS_CELDA if nombre_p == "Toda la muestra" else C.MIN_OBS_CELDA_SUBMUESTRAS
        # Y: referencia
        res = estimar_cs(base, Espec(nombre=f"Y primera pareja | {nombre_p}", min_obs=mo), B=B, XI=XI, n_jobs=C.N_JOBS)
        filas.append(fila_resultado(res, poblacion=nombre_p, resultado="Y: primera pareja del mismo sexo (principal)"))
        # "Se cambia": Z alguna vez
        res = estimar_cs(base, Espec(y="alguna_vez_mismo_sexo", nombre=f"Z alguna vez | {nombre_p}", min_obs=mo),
                         B=B, XI=XI, n_jobs=C.N_JOBS)
        filas.append(fila_resultado(res, poblacion=nombre_p, resultado="Z: alguna vez pareja del mismo sexo ('se cambia')"))
        # "Se adelanta": Y | Z = 1
        sub = base[base["alguna_vez_mismo_sexo"] == 1]
        res = estimar_cs(sub, Espec(nombre=f"Y | Z=1 | {nombre_p}", min_obs=C.MIN_OBS_CELDA_SUBMUESTRAS),
                         B=B, XI=XI, n_jobs=C.N_JOBS)
        filas.append(fila_resultado(res, poblacion=nombre_p, resultado="Y entre quienes alguna vez tuvieron pareja del mismo sexo ('se adelanta')"))
    t = pd.DataFrame(filas)
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(C.DIR_TABLAS / "mecanismos_resumen.csv", index=False)
    print("\n== Mecanismos (efecto promedio posterior, pp)")
    print(t[["poblacion", "resultado", "n", "est_pp", "se_pp", "lo_pp", "hi_pp", "p"]].round(3).to_string(index=False))
    t["especificacion"] = t["poblacion"].str.slice(0, 18) + " · " + t["resultado"].str.slice(0, 22)
    grafica_bosque(t, C.DIR_FIGURAS / "mecanismos.png", "Mecanismos: ¿se adelanta o se cambia?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=C.B_BOOT_SECUNDARIO)
    main(ap.parse_args().B)
