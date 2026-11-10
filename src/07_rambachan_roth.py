"""
07_rambachan_roth.py — Sensibilidad a violaciones de tendencias paralelas
==========================================================================
Metodología, "Pruebas de hipótesis e inferencia": las tendencias previas no prueban que las
tendencias paralelas valgan después de la ley; este script mide cuánta violación toleraría el
resultado (Rambachan y Roth, 2023). Ver honest_rr.py para la formulación.

Objetivos θ:  (a) efecto inmediato (e = 0)   (b) promedio de e = 0 … K−1.
Entradas : resultados/tablas/cs_evento.csv y datos/procesados/cs_boot_evento.csv (de 03)
Salidas  : resultados/tablas/rr_{sd,rm,quiebre}.csv ; figuras/rr_*.png
           resultados/tablas/honestdid_{betahat,sigma}.csv  → insumo para el paquete R HonestDiD
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from graficos import grafica_sensibilidad
from honest_rr import conjunto_identificado, punto_quiebre


def _tramo_contiguo(es_e: set, inicio: int, paso: int, maximo: int):
    out, e = [], inicio
    while e in es_e and len(out) < maximo:
        out.append(e)
        e += paso
    return out


def main(k_post: int):
    es = pd.read_csv(C.DIR_TABLAS / "cs_evento.csv")
    boot = pd.read_csv(C.DIR_PROC / "cs_boot_evento.csv")
    b = C.ANIOS_POR_PERIODO
    es_e = set(es["e"].astype(int))
    pre_e = sorted(_tramo_contiguo(es_e, -2 * b, -b, 50))          # e = -2b, -3b, ... hacia atrás
    post_e = _tramo_contiguo(es_e, 0, b, k_post)
    if len(pre_e) < 1 or len(post_e) < 1:
        sys.exit("No hay suficientes coeficientes contiguos pre/post para la sensibilidad.")
    pre_e = pre_e[::-1] if pre_e[0] > pre_e[-1] else pre_e
    pre_e = sorted(pre_e)
    th = es.set_index("e")["theta_pp"] / 100
    beta_pre = th.loc[pre_e].values
    beta_post = th.loc[post_e].values
    K = len(post_e)
    cols = [f"e{e}" for e in pre_e + post_e]
    Bm = boot[cols].values
    Sigma = np.cov(Bm, rowvar=False)
    z = stats.norm.ppf(0.5 + C.NIVEL_CONFIANZA / 2)
    print(f">> Pre: e = {pre_e} | Post: e = {post_e}")

    pd.DataFrame([np.concatenate([beta_pre, beta_post])], columns=cols).to_csv(C.DIR_TABLAS / "honestdid_betahat.csv", index=False)
    pd.DataFrame(Sigma, columns=cols).to_csv(C.DIR_TABLAS / "honestdid_sigma.csv", index=False)

    objetivos = {"Efecto inmediato (e=0)": np.eye(K)[0], f"Promedio e=0..{post_e[-1]}": np.ones(K) / K}
    cambio2 = np.abs(np.diff(np.concatenate([beta_pre, [0.0]]), n=2)) if len(beta_pre) >= 1 else np.array([0.0])
    escala_sd = max(float(cambio2.max()) if len(cambio2) else 0.0, 1e-4)
    filas_sd, filas_rm, quiebre = [], [], []
    for nombre, l in objetivos.items():
        idx = [cols.index(c) for c in cols[len(pre_e):]]
        se_l = float(np.sqrt(l @ Sigma[np.ix_(idx, idx)] @ l))
        est = float(l @ beta_post)
        grilla_sd = np.linspace(0, 3 * escala_sd, 31)
        for M in grilla_sd:
            lo, hi = conjunto_identificado(beta_pre, beta_post, l, "SD", M)
            filas_sd.append(dict(objetivo=nombre, M=M, cota_inf_pp=100 * lo, cota_sup_pp=100 * hi,
                                 lo_pp=100 * (lo - z * se_l), hi_pp=100 * (hi + z * se_l)))
        bq_sd = punto_quiebre(beta_pre, beta_post, l, "SD", np.linspace(0, 6 * escala_sd, 241))
        if len(beta_pre) >= 1:
            grilla_rm = np.linspace(0, 3, 31)
            for Mb in grilla_rm:
                lo, hi = conjunto_identificado(beta_pre, beta_post, l, "RM", Mb)
                filas_rm.append(dict(objetivo=nombre, M=Mb, cota_inf_pp=100 * lo, cota_sup_pp=100 * hi,
                                     lo_pp=100 * (lo - z * se_l), hi_pp=100 * (hi + z * se_l)))
            bq_rm = punto_quiebre(beta_pre, beta_post, l, "RM", np.linspace(0, 10, 201))
        else:
            bq_rm = np.nan
        quiebre.append(dict(objetivo=nombre, estimacion_pp=100 * est, se_pp=100 * se_l,
                            quiebre_SD_M=bq_sd, quiebre_RM_Mbar=bq_rm))
    sd, rm, q = pd.DataFrame(filas_sd), pd.DataFrame(filas_rm), pd.DataFrame(quiebre)
    sd.round(4).to_csv(C.DIR_TABLAS / "rr_sd.csv", index=False)
    rm.round(4).to_csv(C.DIR_TABLAS / "rr_rm.csv", index=False)
    q.round(4).to_csv(C.DIR_TABLAS / "rr_quiebre.csv", index=False)
    print("\n== Puntos de quiebre (valor del parámetro con el que el IC deja de excluir al cero)")
    print(q.round(4).to_string(index=False))
    print("   M = cambio máximo permitido en la pendiente de la tendencia diferencial (en proporción);"
          "\n   M̄ = veces el mayor cambio de tendencia observado antes de la ley.")
    for i, (nombre, _) in enumerate(objetivos.items()):
        s = sd[sd["objetivo"] == nombre]
        grafica_sensibilidad(s, C.DIR_FIGURAS / f"rr_sd_{i}.png", f"Sensibilidad (suavidad) — {nombre}",
                             "M: cambio máximo de pendiente de la tendencia diferencial (proporción)",
                             q.iloc[i]["quiebre_SD_M"])
        if len(rm):
            r = rm[rm["objetivo"] == nombre]
            grafica_sensibilidad(r, C.DIR_FIGURAS / f"rr_rm_{i}.png", f"Sensibilidad (magnitudes relativas) — {nombre}",
                                 "M̄: veces el mayor cambio de tendencia previo", q.iloc[i]["quiebre_RM_Mbar"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_post", type=int, default=4, help="número de coeficientes posteriores (e=0..k-1)")
    main(ap.parse_args().k_post)
