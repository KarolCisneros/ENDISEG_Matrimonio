"""
06_robustez.py — Pruebas de robustez y sensibilidad
====================================================
Metodología, "Como robustez se prueban…":
  * distintas definiciones de la ventana de evento             → ventana ±5, ±8, pre 15 y periodos de 2 y 3 años
  * un margen de ±1 año en la fecha de inicio sexual           → desplazamiento −1 / +1 del año de inicio
  * exclusión de entidades con jurisprudencia o amparos tempranos → columna amparo_temprano de legalizacion_entidades.csv
  * definición alternativa del resultado: primer encuentro erótico → P7_3 / P7_4_* (año = nacimiento + edad)
Además (metodología, "casos de personas trans… prueba de sensibilidad" y nota "usar heterosexuales como placebo"):
  * incluir personas trans/no binarias, y estimar solo con ellas;
  * placebo: solo personas con OSIG normativa (heterosexuales y cis);
y variantes de la fecha de tratamiento (aprobación en vez de entrada en vigor; vigor formal) y sin covariables.

Cada especificación usa las MISMAS remuestras bootstrap por entidad que el modelo principal.

Entradas : base_analitica.csv.gz
Salidas  : resultados/tablas/robustez_resumen.csv ; resultados/figuras/robustez.png
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
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.BASE_ANALITICA, dtype={"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str})
    m = df[df["muestra_principal"] == 1].reset_index(drop=True)
    con_trans = df[df["muestra_con_trans"] == 1].reset_index(drop=True)
    solo_trans = con_trans[con_trans["trans_nobin"] == 1]
    XI = generar_xi(B)

    # (nombre, base de datos, kwargs de Espec)
    esp = [
        ("0. Principal", m, {}),
        ("1a. Ventana de evento ±5 años", m, dict(pre=5, post=5)),
        ("1b. Ventana de evento ±8 años", m, dict(pre=8, post=8)),
        ("1c. Ventana pre 15 / post 10", m, dict(pre=15, post=10)),
        ("1d. Periodos de 2 años", m, dict(bin=2)),
        ("1e. Periodos de 3 años", m, dict(bin=3)),
        ("2a. Año de inicio sexual −1", m, dict(desplaz=-1)),
        ("2b. Año de inicio sexual +1", m, dict(desplaz=+1)),
        ("3a. Sin entidades con amparos tempranos", m[m["amparo_temprano"] == 0], {}),
        ("3b. Sin el grupo 2021 (cobertura posterior mínima)", m[m["G"] != 2021], {}),
        ("4. Resultado: primer encuentro erótico", m, dict(y="Y_erotico_mismo_sexo", t="anio_erotico")),
        ("5a. Tratamiento = año de aprobación", m, dict(g="G_aprob")),
        ("5b. Tratamiento = entrada en vigor formal", m, dict(g="G_vigor")),
        ("6. Sin covariables", m, dict(covars=[])),
        ("7a. Incluye personas trans / no binarias", con_trans, {}),
        ("7b. Solo personas trans / no binarias", solo_trans, dict(min_obs=C.MIN_OBS_CELDA_SUBMUESTRAS)),
        ("8. Placebo: solo OSIG normativa (heterosexuales)", m[m["grupo_osig"] == "normativa"], {}),
    ]
    filas = []
    for nombre, base, kw in esp:
        kw = dict(kw)
        e = Espec(nombre=nombre, **kw)
        print(f">> {nombre}")
        res = estimar_cs(base, e, B=B, XI=XI, n_jobs=C.N_JOBS)
        fila = fila_resultado(res, especificacion=nombre)
        if res is None:
            fila["n"] = len(base)      # tamaño de la submuestra aunque no haya celdas estimables
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(C.DIR_TABLAS / "robustez_resumen.csv", index=False)
    cols = ["especificacion", "n", "celdas", "est_pp", "se_pp", "lo_pp", "hi_pp", "p", "wald_p_boot"]
    print("\n== Robustez: efecto promedio posterior a la ley (pp)")
    print(tabla[cols].round(3).to_string(index=False))
    grafica_bosque(tabla, C.DIR_FIGURAS / "robustez.png", "Robustez del efecto promedio posterior a la ley")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=C.B_BOOT_SECUNDARIO)
    main(ap.parse_args().B)
