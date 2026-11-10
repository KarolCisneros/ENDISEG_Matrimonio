"""
validacion_monte_carlo.py — ¿El estimador recupera un efecto conocido?
=======================================================================
Toma la base analítica (idealmente la construida con el TMODULO SINTÉTICO) y, en cada réplica,
vuelve a simular Y con un efecto verdadero conocido:
    personas LGB+ : +12 pp tras la legalización de su entidad
    heterosexuales:   0 pp (placebo)
y estima el efecto global posterior con Callaway–Sant'Anna (sin bootstrap, solo el estimador puntual).
Reporta sesgo y desviación estándar entre réplicas (que sirve para comparar con el EE del bootstrap).

Uso (después de correr 01_construir_base.py con datos sintéticos):
    python pruebas/validacion_monte_carlo.py [--reps 60]
NO lo corras sobre la base real: sobrescribe Y en memoria con datos simulados (no toca el archivo).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import config as C
from did_cs import Espec, estimar_cs

TAU = 0.12


def main(reps: int):
    df = pd.read_csv(C.BASE_ANALITICA, dtype={"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str})
    m = df[df["muestra_principal"] == 1].reset_index(drop=True)
    t = np.floor(m["anio_inicio"]).astype(int)
    post = (t >= m["G"]).values
    lgb = (m["lgb_orient"] == 1).values
    rng = np.random.default_rng(1)

    def sim():
        p = np.where(lgb, 0.22 + 0.0015 * (t - 2000), 0.006 + 0.0015 * (t - 2000)) - 0.002 * m["ses_z"].values
        p = np.clip(p + post * np.where(lgb, TAU, 0.0), 0.001, 0.95)
        d = m.copy()
        d["Y_mismo_sexo"] = (rng.random(len(m)) < p).astype(float)
        return d

    casos = {"LGB+ (periodos de 1 año)": (1, dict(min_obs=5)),
             "LGB+ (periodos de 3 años)": (1, dict(min_obs=5, bin=3)),
             "Heterosexuales (placebo)": (0, {})}
    res = {k: [] for k in casos}
    for _ in range(reps):
        d = sim()
        for k, (es_lgb, kw) in casos.items():
            sub = d[d["lgb_orient"] == es_lgb]
            o = estimar_cs(sub, Espec(**kw), B=0, verbose=False)
            res[k].append(o.get("global_post")["est_pp"] if o is not None else np.nan)
    print(f"Efecto verdadero: LGB+ = {TAU * 100:.1f} pp | heterosexuales = 0 pp | {reps} réplicas")
    for k, v in res.items():
        v = np.array(v)
        verdad = TAU * 100 if "LGB" in k else 0.0
        print(f"{k:28s} media = {np.nanmean(v):6.2f} | sesgo = {np.nanmean(v) - verdad:6.2f} | "
              f"desv. est. entre réplicas = {np.nanstd(v, ddof=1):5.2f} | válidas = {int(np.isfinite(v).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    main(ap.parse_args().reps)
