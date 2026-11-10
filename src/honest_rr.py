"""
honest_rr.py — Sensibilidad a violaciones de tendencias paralelas (Rambachan y Roth, 2023)
===========================================================================================
Idea (metodología: "cuánta violación de tendencias paralelas toleraría el resultado"):
los coeficientes del estudio de eventos son  β_e = τ_e + δ_e , donde τ_e es el efecto causal y δ_e
la diferencia de tendencias que hubiera existido sin la ley. Antes de la ley (e < 0) δ_e = β_e se
observa; después de la ley no. Se restringe δ_post a un conjunto Δ y se calcula el CONJUNTO
IDENTIFICADO de θ = l'τ_post:

  Δ^SD(M)  : "suavidad". La segunda diferencia de δ no puede exceder M en valor absoluto
             (M = 0: la tendencia previa se extrapola linealmente).
  Δ^RM(M̄)  : "magnitudes relativas". Cambios consecutivos de δ después de la ley ≤ M̄ veces el
             mayor cambio consecutivo observado antes de la ley.

El punto de quiebre (breakdown) es el M (o M̄) más pequeño con el que el conjunto identificado
incluye al cero: hasta ahí el resultado "aguanta".

IMPORTANTE: aquí se calcula el conjunto identificado por programación lineal y se le suma ±z·EE
(intervalo externo aproximado). NO es el intervalo condicional / FLCI del paquete R HonestDiD.
Para el artículo final use HonestDiD con los archivos que exporta 07_rambachan_roth.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def _prep(beta_pre: np.ndarray, K: int):
    """Vector de referencia: [pre (e=-m..-2), 0 (e=-1)] y número de coeficientes post K."""
    return np.concatenate([np.asarray(beta_pre, float), [0.0]]), K


def _restricciones_sd(v_pre: np.ndarray, K: int, M: float):
    """Segundas diferencias a lo largo de [pre..., 0, post...] con δ_post incógnita."""
    m1 = len(v_pre)                       # incluye el cero de e=-1
    n = m1 + K
    A, b = [], []
    for j in range(1, n - 1):
        fila = np.zeros(K)
        const = 0.0
        for coef, pos in ((1.0, j + 1), (-2.0, j), (1.0, j - 1)):
            if pos < m1:
                const += coef * v_pre[pos]
            else:
                fila[pos - m1] += coef
        if not fila.any():
            continue                       # tripleta totalmente en el pasado: no restringe
        A.append(fila);  b.append(M - const)       #  fila·δ + const ≤ M
        A.append(-fila); b.append(M + const)       # -fila·δ - const ≤ M
    return np.array(A), np.array(b)


def _restricciones_rm(v_pre: np.ndarray, K: int, Mbar: float):
    S = np.max(np.abs(np.diff(v_pre))) if len(v_pre) > 1 else np.nan
    if not np.isfinite(S):
        raise ValueError("Se necesitan al menos 2 coeficientes pre (e=-2 y la referencia) para Δ^RM.")
    lim = Mbar * S
    A, b = [], []
    # |δ_0 - 0| ≤ lim ;  |δ_{k+1} - δ_k| ≤ lim
    for k in range(K):
        fila = np.zeros(K)
        fila[k] = 1.0
        if k > 0:
            fila[k - 1] = -1.0
        A.append(fila);  b.append(lim)
        A.append(-fila); b.append(lim)
    return np.array(A), np.array(b)


def conjunto_identificado(beta_pre, beta_post, l, tipo: str, M: float):
    """
    Cotas [inferior, superior] de θ = l'τ_post = l'(β_post − δ_post) bajo Δ^SD(M) o Δ^RM(M̄).
    beta_pre : e = -m … -2 (orden creciente) ; beta_post : e = 0 … K-1 ; l : pesos (K,)
    """
    beta_post = np.asarray(beta_post, float)
    l = np.asarray(l, float)
    K = len(beta_post)
    v_pre, _ = _prep(beta_pre, K)
    A, b = (_restricciones_sd(v_pre, K, M) if tipo == "SD" else _restricciones_rm(v_pre, K, M))
    base = float(l @ beta_post)
    lim = [(None, None)] * K
    r_min = linprog(c=l, A_ub=A, b_ub=b, bounds=lim, method="highs")   # min l'δ  → cota superior de θ
    r_max = linprog(c=-l, A_ub=A, b_ub=b, bounds=lim, method="highs")  # max l'δ  → cota inferior de θ
    if r_min.status != 0 or r_max.status != 0:
        return np.nan, np.nan
    return base - (-r_max.fun), base - r_min.fun


def punto_quiebre(beta_pre, beta_post, l, tipo: str, grilla: np.ndarray):
    """Menor valor de la grilla para el que el conjunto identificado contiene al cero (np.inf si ninguno)."""
    for M in grilla:
        lo, hi = conjunto_identificado(beta_pre, beta_post, l, tipo, M)
        if np.isfinite(lo) and lo <= 0 <= hi:
            return float(M)
    return np.inf
