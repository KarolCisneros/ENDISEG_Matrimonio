"""
generar_tmodulo_sintetico.py — Datos de prueba con un efecto CONOCIDO
======================================================================
Sirve para verificar que el pipeline recupera un efecto plantado antes de usar el TMODULO real.
Usa las llaves, EDAD, EST_DIS, UPM_DIS y FACTOR REALES (de TSDEM y TAPART_A/B) y simula las
respuestas de TMODULO (P7_*, P8_1, P9_1, …). NO son datos reales: no se interpretan.

Efecto plantado (puntos porcentuales sobre P(primera pareja del mismo sexo)):
    LGBTI+ (orientación LGB+): +TAU_LGB tras la legalización de su entidad
    heterosexuales            : +TAU_HET (0 por defecto)
Tendencia nacional común: sube 0.15 pp por año (la absorbe el efecto de periodo).
Además hay confusión observable: el NSE reduce la probabilidad de debutar con pareja del mismo sexo.

Uso:
    python pruebas/generar_tmodulo_sintetico.py --origen <carpeta con TSDEM/TAPART/TVIVIENDA> --destino <carpeta>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from utils_encuesta import buscar_csv, leer_csv

TAU_LGB, TAU_HET = 0.12, 0.0


def main(origen: Path, destino: Path, semilla: int = 7):
    rng = np.random.default_rng(semilla)
    llaves = ["FOLIO", "VIV_SEL", "HOGAR", "N_REN"]
    ab = pd.concat([leer_csv(buscar_csv(origen, k), llaves + ["EST_DIS", "UPM_DIS", "FACTOR"])
                    for k in ("tapart_a", "tapart_b")], ignore_index=True)
    sd = leer_csv(buscar_csv(origen, "tsdem"), llaves + ["EDAD", "SEXO"])
    d = ab.merge(sd, on=llaves, how="left")
    d["edad"] = pd.to_numeric(d["EDAD"], errors="coerce")
    d["sexo"] = pd.to_numeric(d["SEXO"], errors="coerce")
    d = d[d["edad"].between(15, 96) & d["sexo"].isin([1, 2])].reset_index(drop=True)
    n = len(d)
    ent = d["FOLIO"].str[:2].astype(int)

    # año de legalización (mismo criterio que config: principal < 2021-12-01 → año; si no, no tratada)
    leg = pd.read_csv(Path(__file__).resolve().parents[1] / "datos" / "legalizacion_entidades.csv", dtype={"ent": str})
    f = pd.to_datetime(leg["fecha_principal"])
    G = pd.Series(np.where(f < pd.Timestamp("2021-12-01"), f.dt.year, 9999), index=leg["ent"].astype(int))
    g_i = ent.map(G).values

    # índice SES latente (se reflejará en las variables de vivienda solo vía la base sintética de TVIVIENDA)
    ses = rng.normal(size=n)
    lgb = rng.random(n) < (0.06 + 0.01 * (ses > 0))
    trans = rng.random(n) < 0.008
    nunca = rng.random(n) < 0.08
    edad_debut = np.clip(np.round(rng.gamma(30, 0.58, n)).astype(int), 11, 40)
    edad_debut = np.minimum(edad_debut, d["edad"].values.astype(int))
    anio_nac = 2021 - d["edad"].values
    anio_ini = anio_nac + edad_debut
    post = anio_ini >= g_i
    base_h = 0.006 + 0.0015 * (anio_ini - 2000)           # tendencia nacional
    base_l = 0.22 + 0.0015 * (anio_ini - 2000)
    p = np.where(lgb, base_l, base_h) - 0.03 * np.where(lgb, ses, 0.0) * 0.5 - 0.002 * ses
    p = p + post * np.where(lgb, TAU_LGB, TAU_HET)
    p = np.clip(p, 0.001, 0.95)
    y = rng.random(n) < p
    sexo = d["sexo"].values.astype(int)
    pareja_h = np.where(sexo == 1, y, ~y)        # hombre-pareja si (hombre y mismo sexo) o (mujer y distinto sexo)
    p76_1 = np.where(pareja_h, 1, 2)
    p76_2 = np.where(~pareja_h, 1, 2)
    # primer encuentro erótico: 0-2 años antes, coincide con Y el 85% de las veces
    edad_er = np.maximum(edad_debut - rng.integers(0, 3, n), 8)
    y_er = np.where(rng.random(n) < 0.85, y, ~y)
    pareja_h_er = np.where(sexo == 1, y_er, ~y_er)
    # P7_8 (vida entera)
    ever_mismo = y | (lgb & (rng.random(n) < 0.55)) | (~lgb & (rng.random(n) < 0.02))
    p78 = np.where(ever_mismo,
                   np.where(sexo == 1, rng.choice([3, 2, 4], n, p=[0.3, 0.2, 0.5]), rng.choice([1, 2, 5], n, p=[0.3, 0.2, 0.5])),
                   np.where(sexo == 1, 5, 4))
    orient = np.where(lgb, np.where(sexo == 2, rng.choice([1, 3, 6], n, p=[0.5, 0.45, 0.05]),
                                    rng.choice([2, 3, 6], n, p=[0.5, 0.45, 0.05])), np.where(sexo == 1, 5, 4))
    p91 = np.where(trans, np.where(sexo == 1, 2, 1), sexo)
    out = pd.DataFrame({
        "FOLIO": d["FOLIO"], "VIV_SEL": d["VIV_SEL"], "HOGAR": d["HOGAR"], "N_REN": d["N_REN"],
        "P4_1": d["edad"].astype(int).astype(str).str.zfill(2),
        "P7_1": sexo.astype(str), "P7_1A": np.where(rng.random(n) < 0.003, "1", "2"),
        "P7_3": np.where(nunca, "00", pd.Series(edad_er).astype(str).str.zfill(2)),
        "P7_4_1": np.where(nunca, "", np.where(pareja_h_er, "1", "2")),
        "P7_4_2": np.where(nunca, "", np.where(~pareja_h_er, "1", "2")),
        "P7_4_3": np.where(nunca, "", "2"),
        "P7_5": np.where(nunca, "00", pd.Series(edad_debut).astype(str).str.zfill(2)),
        "P7_6_1": np.where(nunca, "", pd.Series(p76_1).astype(str)),
        "P7_6_2": np.where(nunca, "", pd.Series(p76_2).astype(str)),
        "P7_6_3": np.where(nunca, "", "2"),
        "P7_8": np.where(nunca, "", pd.Series(p78).astype(str)),
        "P8_1": pd.Series(orient).astype(str), "P8_1A": "", "P8_2": np.where(lgb, pd.Series(rng.integers(8, 26, n)).astype(str).str.zfill(2), ""),
        "P9_1": pd.Series(p91).astype(str), "FILTRO_9_2": np.where(p91 != sexo, "1", "2"),
        "ENT": ent.astype(str).str.zfill(2), "EST_DIS": d["EST_DIS"], "UPM_DIS": d["UPM_DIS"], "FACTOR": d["FACTOR"],
    })
    destino.mkdir(parents=True, exist_ok=True)
    out.to_csv(destino / "conjunto_de_datos_tmodulo_endiseg_2021.csv", index=False)
    print(f"TMODULO sintético: {len(out):,} filas | LGB+ {lgb.mean():.1%} | efecto plantado LGB+ = {TAU_LGB * 100:.1f} pp, "
          f"heterosexuales = {TAU_HET * 100:.1f} pp")
    print("Efecto plantado en toda la muestra ≈ {:.2f} pp (promedio ponderado por la proporción de LGB+)".format(
        100 * (TAU_LGB * lgb.mean() + TAU_HET * (1 - lgb.mean()))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", type=Path, required=True)
    ap.add_argument("--destino", type=Path, required=True)
    a = ap.parse_args()
    main(a.origen, a.destino)
