"""
utils_encuesta.py — Utilidades comunes
=======================================
* Lectura de las tablas CSV de INEGI (todo como texto para no perder ceros a la izquierda).
* Álgebra básica ponderada: mínimos cuadrados (WLS), logit ponderado (IRLS), cuantiles.
* Estimadores de diseño muestral con las variables de diseño de la ENDISEG:
      FACTOR   = factor de expansión (ponderador)
      EST_DIS  = estrato de diseño
      UPM_DIS  = unidad primaria de muestreo
* Índice socioeconómico (componente principal ponderado) con las variables de TVIVIENDA.

No depende de statsmodels: solo numpy, pandas y scipy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------------
def buscar_csv(directorio: Path, fragmento: str) -> Path:
    """Busca recursivamente un .csv cuyo nombre contenga `fragmento` (sin distinguir mayúsculas)."""
    directorio = Path(directorio)
    if not directorio.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {directorio}")
    frag = fragmento.lower()
    candidatos = [p for p in directorio.rglob("*")
                  if p.is_file() and p.suffix.lower() == ".csv" and frag in p.name.lower()]
    if not candidatos:
        raise FileNotFoundError(
            f"No encontré ningún .csv con '{fragmento}' en el nombre dentro de {directorio}. "
            "Revisa DIR_DATOS en config.py o la variable de entorno ENDISEG_DIR_DATOS."
        )
    candidatos.sort(key=lambda p: -p.stat().st_size)
    return candidatos[0]


def leer_csv(ruta: Path, columnas: list[str] | None = None) -> pd.DataFrame:
    """Lee un CSV de INEGI como texto. Los vacíos ("") quedan como NaN."""
    usecols = None if columnas is None else (lambda c: c in set(columnas))
    ultimo_error = None
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(ruta, dtype=str, encoding=enc, na_values=[""],
                               keep_default_na=False, usecols=usecols)
        except UnicodeDecodeError as e:  # pragma: no cover
            ultimo_error = e
    raise ultimo_error  # type: ignore[misc]


def num(df: pd.DataFrame, col: str) -> pd.Series:
    """Columna como número (códigos no numéricos, p. ej. 'b' = blanco por secuencia, pasan a NaN)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


# ----------------------------------------------------------------------------
# Álgebra ponderada
# ----------------------------------------------------------------------------
def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mínimos cuadrados ponderados (ecuaciones normales con una pizca de ridge por estabilidad)."""
    Xw = X * w[:, None]
    A = Xw.T @ X
    b = Xw.T @ y
    A.flat[:: A.shape[0] + 1] += 1e-10
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]


def logit_w(X: np.ndarray, y: np.ndarray, w: np.ndarray, beta0: np.ndarray | None = None,
            max_iter: int = 30, tol: float = 1e-8, ridge: float = 1e-6) -> np.ndarray:
    """Regresión logística ponderada por IRLS (Newton-Raphson). `beta0` permite arrancar en caliente."""
    p = X.shape[1]
    beta = np.zeros(p) if beta0 is None else beta0.astype(float).copy()
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        v = np.maximum(mu * (1 - mu), 1e-10)
        W = w * v
        z = eta + (y - mu) / v
        XW = X * W[:, None]
        A = XW.T @ X
        A.flat[:: p + 1] += ridge
        nuevo = np.linalg.solve(A, XW.T @ z)
        if np.max(np.abs(nuevo - beta)) < tol:
            return nuevo
        beta = nuevo
    return beta


def cuantil_ponderado(x: np.ndarray, w: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    """Cuantil(es) de x con pesos w."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cw, x)


# ----------------------------------------------------------------------------
# Diseño muestral: media ponderada con error estándar por linealización
# ----------------------------------------------------------------------------
def media_svy(y, w, estrato, upm) -> dict:
    """
    Media ponderada y su error estándar con el diseño de la ENDISEG
    (estratos EST_DIS, conglomerados UPM_DIS, ponderador FACTOR).

    Linealización de Taylor del estimador de razón  ȳ = Σ w·y / Σ w , con UPM
    seleccionadas con reemplazo dentro de cada estrato:
        u_i  = w_i (y_i - ȳ) / Σ w
        Var  = Σ_h  n_h/(n_h-1) · Σ_c (u_hc - ū_h)²     (u_hc = suma de u_i en la UPM c)
    Estratos con una sola UPM no aportan varianza (tratamiento "certainty").
    """
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    y, w = y[ok], w[ok]
    est = np.asarray(estrato)[ok]
    up = np.asarray(upm)[ok]
    if len(y) == 0:
        return dict(media=np.nan, se=np.nan, n=0, gl=0)
    W = w.sum()
    ybar = (w * y).sum() / W
    u = w * (y - ybar) / W
    t = pd.DataFrame({"h": est, "c": up, "u": u}).groupby(["h", "c"], sort=False)["u"].sum().reset_index()
    var = 0.0
    n_upm = len(t)
    n_est = t["h"].nunique()
    for _, g in t.groupby("h", sort=False):
        nh = len(g)
        if nh > 1:
            var += nh / (nh - 1) * ((g["u"] - g["u"].mean()) ** 2).sum()
    return dict(media=ybar, se=float(np.sqrt(var)), n=int(len(y)), gl=int(max(n_upm - n_est, 1)))


def ic_normal(est: float, se: float, nivel: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + nivel / 2)
    return est - z * se, est + z * se


# ----------------------------------------------------------------------------
# Índice socioeconómico (componente principal ponderado, estilo Filmer-Pritchett)
# ----------------------------------------------------------------------------
def indicadores_vivienda(v: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de bienestar de la vivienda (mayor = mejor) a partir de TVIVIENDA."""
    p = lambda c: num(v, c)  # noqa: E731
    ind = pd.DataFrame(index=v.index)
    ind["piso_firme"] = (p("P1_1") >= 2).astype(float).where(p("P1_1").notna())
    ind["hacinamiento_inv"] = -(p("P2_1") / p("P1_2").where(p("P1_2") > 0))
    ind["agua_entubada"] = p("P1_3").map({1: 2.0, 2: 1.0, 3: 0.0})
    ind["drenaje_red"] = p("P1_4").map({1: 2.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0})
    ind["taza_bano"] = (p("P1_5") == 1).astype(float).where(p("P1_5").notna())
    ind["luz"] = (p("P1_6") == 1).astype(float).where(p("P1_6").notna())
    nombres = {"P1_7_1": "refrigerador", "P1_7_2": "lavadora", "P1_7_3": "automovil",
               "P1_7_4": "pantalla_plana", "P1_7_5": "computadora", "P1_7_6": "internet",
               "P1_7_7": "streaming"}
    for c, n in nombres.items():
        # 'b' (blanco por secuencia, p. ej. sin luz eléctrica) equivale a no tener el bien
        ind[n] = (p(c) == 1).astype(float)
    return ind


def indice_ses(ind: pd.DataFrame, w: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Primer componente principal ponderado de los indicadores (estandarizados con pesos).
    Devuelve el índice estandarizado (media 0, desv. 1 ponderadas) y un dict con cargas
    y proporción de varianza explicada. El signo se orienta para que más bienes = mayor índice.
    """
    X = ind.copy()
    X = X.fillna(X.median())
    w = np.asarray(w, float)
    w = w / w.sum()
    mu = (X.values * w[:, None]).sum(0)
    sd = np.sqrt((((X.values - mu) ** 2) * w[:, None]).sum(0))
    sd[sd == 0] = 1.0
    Z = (X.values - mu) / sd
    C = (Z * w[:, None]).T @ Z
    val, vec = np.linalg.eigh(C)
    o = np.argsort(val)[::-1]
    val, vec = val[o], vec[:, o]
    carga = vec[:, 0]
    score = Z @ carga
    n_bienes = ind[["refrigerador", "lavadora", "automovil", "pantalla_plana", "computadora",
                    "internet", "streaming"]].sum(axis=1).values
    if np.corrcoef(score, n_bienes)[0, 1] < 0:
        carga, score = -carga, -score
    m = (score * w).sum()
    s = np.sqrt((((score - m) ** 2) * w).sum())
    score = (score - m) / s
    info = dict(var_explicada=float(val[0] / val.sum()),
                cargas=dict(zip(ind.columns, np.round(carga, 4))))
    return score, info
