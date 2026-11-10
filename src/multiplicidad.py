"""
multiplicidad.py — Corrección por comparaciones múltiples (Holm y Benjamini–Hochberg)
=====================================================================================
Holm      : controla la probabilidad de AL MENOS un falso positivo en la familia (FWER); más estricto.
BH        : controla la proporción esperada de falsos positivos entre los rechazos (FDR); más potente.
"""
import numpy as np


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    orden = np.argsort(pv)
    m = len(pv)
    aj = np.maximum.accumulate((m - np.arange(m)) * pv[orden])
    res = np.empty(m)
    res[orden] = np.minimum(aj, 1.0)
    out[ok] = res
    return out


def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    m = len(pv)
    orden = np.argsort(pv)
    aj = pv[orden] * m / (np.arange(m) + 1)
    aj = np.minimum.accumulate(aj[::-1])[::-1]
    res = np.empty(m)
    res[orden] = np.minimum(aj, 1.0)
    out[ok] = res
    return out
