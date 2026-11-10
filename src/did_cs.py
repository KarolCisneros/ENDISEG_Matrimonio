"""
did_cs.py — Diferencias en diferencias escalonado (Callaway y Sant'Anna, 2021)
================================================================================
Implementación propia (numpy) para SECCIONES TRANSVERSALES REPETIDAS con ponderadores de encuesta.

Mapa con la metodología
-----------------------
* Unidad de tratamiento : entidad de residencia; grupo g = año en que la entidad legalizó.
* "Periodo"             : año estimado de inicio de la vida sexual (cada persona se observa una vez).
* Efecto por grupo y año: ATT(g,t) = E[Y(g) − Y(∞) | G = g]. Se compara el grupo g solo con
                          entidades AÚN NO TRATADAS (G > max(t, g−1)), con periodo base universal g−1.
* Dos ajustes a la vez (doblemente robusto, Sant'Anna y Zhao 2020, estimador rc1):
      (i)  reponderación: más peso a controles parecidos al grupo tratado según X_i
           (propensity score logístico con ponderadores de la encuesta);
      (ii) regresión de resultado: predice Y de los controles según X_i en cada periodo.
  Ambos se calculan dentro de cada ATT(g,t), con las personas del grupo g y de las entidades
  aún no tratadas.
* Estudio de eventos    : θ(e) = Σ_g w_g ATT(g, g+e), e = t − g, con w_g ∝ personas tratadas que
                          aporta cada grupo en esa celda.
* Inferencia            : bootstrap multiplicador en dos niveles que respeta el diseño de la encuesta:
                          (1) cada ENTIDAD (32 clústeres) recibe un peso aleatorio Exp(1) que multiplica
                              los ponderadores de sus personas (variabilidad entre entidades), y
                          (2) cada UPM (UPM_DIS) recibe otro peso Exp(1) independiente (variabilidad
                              muestral dentro de la entidad).
                          Con cada juego de pesos se re-estima todo (propensity score y regresiones).
                          El nivel (2) es necesario porque, si un grupo de legalización es UNA sola
                          entidad, multiplicar todos sus pesos por una constante no cambia su media
                          ponderada y el nivel (1) por sí solo no vería su ruido muestral.
                          EE = desv. est. de las remuestras; banda sup-t; prueba de Wald conjunta.

El estimador puntual replica la fórmula del paquete R `DRDID::drdid_rc1` con pesos muestrales;
la inferencia es la descrita arriba (no usa la función de influencia analítica).
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from utils_encuesta import logit_w, wls

N_CLUSTERS = 32
N_UPM = 6375          # UPM_DIS va de 0000001 a 0006374


# ----------------------------------------------------------------------------
# Especificación
# ----------------------------------------------------------------------------
@dataclass
class Espec:
    y: str = "Y_mismo_sexo"
    t: str = "anio_inicio"          # año de inicio (periodo)
    g: str = "G"                    # año de legalización de la entidad (ANIO_NO_TRATADO si no tratada)
    covars: list = field(default_factory=lambda: list(C.COVARIABLES))
    peso: str = "FACTOR"
    cluster: str = "ENT"
    pre: int = C.VENTANA_PRE
    post: int = C.VENTANA_POST
    bin: int = C.ANIOS_POR_PERIODO
    min_obs: int = C.MIN_OBS_CELDA
    anio_max: int = C.ANIO_MAX_INICIO
    desplaz: int = 0                # años sumados al periodo (robustez ±1)
    trim: float = C.TRIM_PS
    nombre: str = "principal"


def generar_xi(B: int, semilla: int = C.SEMILLA) -> tuple[np.ndarray, np.ndarray]:
    """
    Pesos aleatorios Exp(1): (B × 32 entidades, B × 6375 UPM). Se comparten entre especificaciones
    (remuestras pareadas), así que se pueden comparar estimaciones de distintas submuestras.
    """
    rng = np.random.default_rng(semilla)
    return rng.exponential(1.0, size=(B, N_CLUSTERS)), rng.exponential(1.0, size=(B, N_UPM))


def _pesos_boot(c, xi_e, xi_u, modo):
    w = c.s
    if modo in ("entidad", "ambos"):
        w = w * xi_e[c.cl]
    if modo in ("upm", "ambos"):
        w = w * xi_u[c.up]
    return w


# ----------------------------------------------------------------------------
# Preparación de datos y celdas (g, t)
# ----------------------------------------------------------------------------
def _preparar(df: pd.DataFrame, esp: Espec) -> dict:
    cols = [esp.y, esp.t, esp.g, esp.peso, esp.cluster] + list(esp.covars)
    d = df.dropna(subset=cols)
    d = d[d[esp.peso] > 0]
    t = np.floor(d[esp.t].values.astype(float) + esp.desplaz).astype(int)
    d = d[t <= esp.anio_max]
    t = t[t <= esp.anio_max]
    b = esp.bin
    per = (t // b) * b
    G = d[esp.g].astype(int).values
    Gb = np.where(G == C.ANIO_NO_TRATADO, C.ANIO_NO_TRATADO, (G // b) * b)
    X = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in esp.covars])
    return dict(y=d[esp.y].values.astype(float), X=X, s=d[esp.peso].values.astype(float),
                per=per, G=Gb, cl=d[esp.cluster].astype(int).values - 1,
                up=d["UPM_DIS"].astype(int).values, n=len(d))


class Celda:
    __slots__ = ("g", "t", "e", "idx", "y", "X", "D", "T", "s", "cl", "i0", "X0", "y0", "iT", "XT",
                 "d1t1", "d1t0", "d0t1", "d0t0", "nw", "up", "n_d1_post", "n_d0_post", "n_d1_base", "n_d0_base", "b_ps0")


def _construir_celdas(P: dict, esp: Espec) -> tuple[list[Celda], pd.DataFrame]:
    per, G = P["per"], P["G"]
    b = esp.bin
    por_per = {p: np.where(per == p)[0] for p in np.unique(per)}
    anio_max_b = (esp.anio_max // b) * b
    grupos = sorted(g for g in np.unique(G) if g != C.ANIO_NO_TRATADO and g <= anio_max_b)
    celdas, omitidas = [], []
    k_min, k_max = -int(np.ceil(esp.pre / b)), int(np.floor(esp.post / b))
    for g in grupos:
        base = g - b
        if base not in por_per:
            omitidas.append((g, np.nan, "sin datos en el periodo base"))
            continue
        for k in range(k_min, k_max + 1):
            t = g + k * b
            if t == base or t not in por_per or t > anio_max_b:
                continue
            lim = max(t, base)
            filas, Dflag, Tflag = [], [], []
            for ix, es_post in ((por_per[base], 0.0), (por_per[t], 1.0)):
                Gi = G[ix]
                d = Gi == g
                c = (Gi > lim) & ~d
                sel = d | c
                filas.append(ix[sel])
                Dflag.append(d[sel].astype(float))
                Tflag.append(np.full(sel.sum(), es_post))
            idx = np.concatenate(filas)
            D = np.concatenate(Dflag)
            T = np.concatenate(Tflag)
            n11, n10 = int(((D == 1) & (T == 1)).sum()), int(((D == 1) & (T == 0)).sum())
            n01, n00 = int(((D == 0) & (T == 1)).sum()), int(((D == 0) & (T == 0)).sum())
            if min(n11, n10, n01, n00) < esp.min_obs:
                omitidas.append((g, t, f"celda chica (trat. post {n11}, trat. base {n10}, ctrl. post {n01}, ctrl. base {n00})"))
                continue
            c_ = Celda()
            c_.g, c_.t, c_.e, c_.idx = g, t, t - g, idx
            c_.y, c_.X, c_.D, c_.T = P["y"][idx], P["X"][idx], D, T
            c_.s, c_.cl, c_.up = P["s"][idx], P["cl"][idx], P["up"][idx]
            c_.i0 = [np.where((D == 0) & (T == tt))[0] for tt in (0, 1)]
            c_.X0 = [c_.X[i] for i in c_.i0]
            c_.y0 = [c_.y[i] for i in c_.i0]
            c_.iT = [np.where(T == tt)[0] for tt in (0, 1)]
            c_.XT = [c_.X[i] for i in c_.iT]
            c_.d1t1, c_.d1t0 = D * T, D * (1 - T)
            c_.d0t1, c_.d0t0 = (1 - D) * T, (1 - D) * (1 - T)
            c_.nw = float((c_.s * c_.d1t1).sum())
            c_.n_d1_post, c_.n_d0_post, c_.n_d1_base, c_.n_d0_base = n11, n01, n10, n00
            c_.b_ps0 = None
            celdas.append(c_)
    return celdas, pd.DataFrame(omitidas, columns=["g", "t", "motivo"])


# ----------------------------------------------------------------------------
# ATT(g,t) doblemente robusto para secciones transversales repetidas
# ----------------------------------------------------------------------------
def _att_celda(c: Celda, w: np.ndarray, trim: float, devolver_b: bool = False):
    try:
        b_ps = logit_w(c.X, c.D, w, c.b_ps0)
        ps = 1.0 / (1.0 + np.exp(-np.clip(c.X @ b_ps, -30, 30)))
        mu = np.empty_like(c.y)
        for tt in (0, 1):
            beta = wls(c.X0[tt], c.y0[tt], w[c.i0[tt]])
            mu[c.iT[tt]] = c.XT[tt] @ beta
        odds = ps / (1.0 - ps)
        ok = ps < trim
        w11, w10 = w * c.d1t1, w * c.d1t0
        w01, w00 = w * odds * c.d0t1 * ok, w * odds * c.d0t0 * ok
        s11, s10, s01, s00 = w11.sum(), w10.sum(), w01.sum(), w00.sum()
        if min(s11, s10, s01, s00) <= 0:
            return (np.nan, b_ps) if devolver_b else np.nan
        a = w11 / s11 - w10 / s10 - w01 / s01 + w00 / s00
        att = float(a @ (c.y - mu))
        return (att, b_ps) if devolver_b else att
    except np.linalg.LinAlgError:
        return (np.nan, None) if devolver_b else np.nan


# ----------------------------------------------------------------------------
# Agregación (estudio de eventos, resúmenes, por grupo)
# ----------------------------------------------------------------------------
def _matriz_agregacion(celdas: list[Celda]) -> tuple[np.ndarray, list[tuple]]:
    n = len(celdas)
    filas, etiquetas = [], []

    def agrega(sel, etiqueta):
        w = np.array([c.nw if sel(c) else 0.0 for c in celdas])
        if w.sum() > 0:
            filas.append(w / w.sum())
            etiquetas.append(etiqueta)

    for e in sorted({c.e for c in celdas}):
        agrega(lambda c, e=e: c.e == e, ("es", e))
    agrega(lambda c: c.e >= 0, ("resumen", "global_post"))
    for nom, (lo, hi) in [("corto", C.RANGO_CORTO), ("mediano", C.RANGO_MEDIANO), ("largo", C.RANGO_LARGO)]:
        agrega(lambda c, lo=lo, hi=hi: lo <= c.e <= hi, ("resumen", nom))
    for g in sorted({c.g for c in celdas}):
        agrega(lambda c, g=g: c.g == g and c.e >= 0, ("grupo", g))
    # año calendario de inicio sexual (solo celdas posteriores a la ley)
    for t in sorted({c.t for c in celdas if c.e >= 0}):
        agrega(lambda c, t=t: c.t == t and c.e >= 0, ("anio", t))
    return (np.vstack(filas) if filas else np.zeros((0, n))), etiquetas


# ----------------------------------------------------------------------------
# Bootstrap multiplicador por entidad (paralelizable)
# ----------------------------------------------------------------------------
_CTX: dict = {}


def _trabajo(rango):
    celdas, XI, trim, modo = _CTX["celdas"], _CTX["XI"], _CTX["trim"], _CTX["modo"]
    out = np.empty((len(rango), len(celdas)))
    for k, b in enumerate(rango):
        xi_e, xi_u = XI[0][b], XI[1][b]
        for j, c in enumerate(celdas):
            out[k, j] = _att_celda(c, _pesos_boot(c, xi_e, xi_u, modo), trim)
    return out


def _bootstrap(celdas, XI, trim, n_jobs, modo):
    B = XI[0].shape[0]
    _CTX.update(celdas=celdas, XI=XI, trim=trim, modo=modo)
    if n_jobs <= 1 or B < 20 or "fork" not in mp.get_all_start_methods():
        return _trabajo(range(B))
    trozos = [r for r in np.array_split(np.arange(B), n_jobs * 2) if len(r)]
    with cf.ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as ex:
        partes = list(ex.map(_trabajo, trozos))
    return np.vstack(partes)


# ----------------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------------
@dataclass
class ResultadoCS:
    esp: Espec
    celdas: pd.DataFrame
    omitidas: pd.DataFrame
    es: pd.DataFrame                 # estudio de eventos (en puntos porcentuales)
    resumen: pd.DataFrame            # global / corto / mediano / largo / por grupo (pp)
    n_obs: int
    n_trat: int
    boot: np.ndarray | None          # B × n_agregados (en proporción, no pp)
    etiquetas: list
    B_validas: int = 0
    wald: dict = field(default_factory=dict)
    anio: pd.DataFrame | None = None  # efecto por año calendario de inicio sexual (celdas posteriores a la ley)

    def get(self, clave: str):
        """Fila de resumen (est, se, lo, hi, p) en pp."""
        r = self.resumen[self.resumen["nombre"] == clave]
        return None if r.empty else r.iloc[0]

    def draws(self, etiqueta: tuple) -> np.ndarray | None:
        if self.boot is None:
            return None
        return self.boot[:, self.etiquetas.index(etiqueta)]


def estimar_cs(df: pd.DataFrame, esp: Espec, B: int = C.B_BOOT, XI=None,
               n_jobs: int = C.N_JOBS, verbose: bool = True, modo: str = C.INFERENCIA) -> ResultadoCS | None:
    P = _preparar(df, esp)
    celdas, omitidas = _construir_celdas(P, esp)
    if verbose:
        print(f"   [{esp.nombre}] N={P['n']:,} | celdas estimables={len(celdas)} | omitidas={len(omitidas)}")
    if not celdas:
        return None

    # --- estimación puntual
    att = np.empty(len(celdas))
    for j, c in enumerate(celdas):
        att[j], c.b_ps0 = _att_celda(c, c.s, esp.trim, devolver_b=True)
    bien = np.isfinite(att)
    if not bien.all():
        celdas = [c for c, ok in zip(celdas, bien) if ok]
        att = att[bien]
    Agg, etiquetas = _matriz_agregacion(celdas)
    theta = Agg @ att

    # --- bootstrap multiplicador por entidad
    boot, B_ok = None, 0
    if B and B > 0:
        if XI is None or XI[0].shape[0] < B:
            XI = generar_xi(B)
        XIb = (XI[0][:B], XI[1][:B])
        attb = _bootstrap(celdas, XIb, esp.trim, n_jobs, modo)
        buenas = np.isfinite(attb).all(axis=1)
        attb = attb[buenas]
        B_ok = int(buenas.sum())
        boot = attb @ Agg.T if B_ok > 5 else None
    se = np.nanstd(boot, axis=0, ddof=1) if boot is not None else np.full(len(theta), np.nan)
    z = stats.norm.ppf(0.5 + C.NIVEL_CONFIANZA / 2)

    def fila(i):
        est, s_ = theta[i], se[i]
        p = 2 * (1 - stats.norm.cdf(abs(est / s_))) if (s_ and np.isfinite(s_) and s_ > 0) else np.nan
        return est * 100, s_ * 100, (est - z * s_) * 100, (est + z * s_) * 100, p

    # tabla ATT(g,t)
    se_c = None
    if B_ok > 5:
        se_c = np.nanstd(attb, axis=0, ddof=1)
    tc = pd.DataFrame({
        "g": [c.g for c in celdas], "t": [c.t for c in celdas], "e": [c.e for c in celdas],
        "att_pp": att * 100, "se_pp": (se_c * 100 if se_c is not None else np.nan),
        "n_trat_post": [c.n_d1_post for c in celdas], "n_ctrl_post": [c.n_d0_post for c in celdas],
        "n_trat_base": [c.n_d1_base for c in celdas], "n_ctrl_base": [c.n_d0_base for c in celdas],
    })

    # estudio de eventos (incluye e = -1 normalizado a 0)
    filas_es = []
    for i, et in enumerate(etiquetas):
        if et[0] == "es":
            est, s_, lo, hi, p = fila(i)
            filas_es.append(dict(e=et[1], theta_pp=est, se_pp=s_, lo_pp=lo, hi_pp=hi, p=p,
                                 n_celdas=int(sum(1 for c in celdas if c.e == et[1]))))
    es = pd.DataFrame(filas_es)
    if not (es["e"] == -esp.bin).any():
        es = pd.concat([es, pd.DataFrame([dict(e=-esp.bin, theta_pp=0.0, se_pp=0.0, lo_pp=0.0, hi_pp=0.0, p=np.nan, n_celdas=0)])])
    es = es.sort_values("e").reset_index(drop=True)

    # banda sup-t (uniforme) sobre e != base
    es["supt_lo_pp"], es["supt_hi_pp"] = np.nan, np.nan
    if B_ok > 5:
        ie = [i for i, et in enumerate(etiquetas) if et[0] == "es"]
        sd = se[ie]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tt = np.abs(boot[:, ie] - theta[ie]) / np.where(sd > 0, sd, np.nan)
            ok_fila = np.isfinite(tt).any(axis=1)
            tmax = np.nanmax(tt[ok_fila], axis=1) if ok_fila.any() else np.array([])
        if len(tmax) > 5:
            crit = float(np.quantile(tmax, C.NIVEL_CONFIANZA))
            for i in ie:
                m = es["e"] == etiquetas[i][1]
                es.loc[m, "supt_lo_pp"] = (theta[i] - crit * se[i]) * 100
                es.loc[m, "supt_hi_pp"] = (theta[i] + crit * se[i]) * 100

    # resumen
    filas_r = []
    for i, et in enumerate(etiquetas):
        if et[0] in ("resumen", "grupo"):
            est, s_, lo, hi, p = fila(i)
            nombre = et[1] if et[0] == "resumen" else f"grupo_{et[1]}"
            filas_r.append(dict(nombre=nombre, est_pp=est, se_pp=s_, lo_pp=lo, hi_pp=hi, p=p))
    resumen = pd.DataFrame(filas_r)

    # efecto por año calendario de inicio sexual (con IC puntual)
    filas_a = []
    for i, et in enumerate(etiquetas):
        if et[0] == "anio":
            est, s_, lo, hi, p = fila(i)
            filas_a.append(dict(anio=et[1], theta_pp=est, se_pp=s_, lo_pp=lo, hi_pp=hi, p=p,
                                n_celdas=int(sum(1 for c in celdas if c.t == et[1] and c.e >= 0)),
                                n_trat=int(sum(c.n_d1_post for c in celdas if c.t == et[1] and c.e >= 0))))
    anio_df = pd.DataFrame(filas_a)

    res = ResultadoCS(esp=esp, celdas=tc, omitidas=omitidas, es=es, resumen=resumen, n_obs=P["n"],
                      n_trat=int(sum(c.n_d1_post for c in celdas)), boot=boot, etiquetas=etiquetas, B_validas=B_ok,
                      anio=anio_df)
    if boot is not None:
        res.wald = wald_pretrend(res)
    return res


def fila_resultado(res: ResultadoCS | None, **extra) -> dict:
    """Una fila de tabla con el efecto global, corto/mediano/largo plazo y la prueba de tendencias previas."""
    out = dict(extra)
    if res is None:
        out.update(n=0, celdas=0, est_pp=np.nan)
        return out
    out.update(n=res.n_obs, celdas=len(res.celdas))
    for clave, pref in [("global_post", ""), ("corto", "corto_"), ("mediano", "mediano_"), ("largo", "largo_")]:
        r = res.get(clave)
        for k in ("est_pp", "se_pp", "lo_pp", "hi_pp", "p"):
            out[pref + k] = np.nan if r is None else r[k]
    out["wald_p_boot"] = res.wald.get("p_boot", np.nan) if res.wald else np.nan
    out["wald_p_chi2"] = res.wald.get("p_chi2", np.nan) if res.wald else np.nan
    return out


# ----------------------------------------------------------------------------
# Prueba conjunta de tendencias previas (Wald)
# ----------------------------------------------------------------------------
def wald_pretrend(res: ResultadoCS) -> dict:
    """
    H0: θ(e) = 0 para todos los e < -1 (e = -1 es la referencia y vale 0 por construcción).
    Matriz de varianzas = covarianza de las remuestras del bootstrap multiplicador.
    Se reportan el p-valor χ² asintótico y el p-valor bootstrap (más fiable con solo 32 entidades).
    """
    ie = [i for i, et in enumerate(res.etiquetas) if et[0] == "es" and et[1] < -res.esp.bin]
    if len(ie) == 0 or res.boot is None:
        return {}
    th = np.array([res.es.loc[res.es["e"] == res.etiquetas[i][1], "theta_pp"].iloc[0] / 100 for i in ie])
    Bm = res.boot[:, ie]
    V = np.cov(Bm, rowvar=False).reshape(len(ie), len(ie))
    Vi = np.linalg.pinv(V)
    W = float(th @ Vi @ th)
    k = int(np.linalg.matrix_rank(V))
    Wb = np.einsum("bi,ij,bj->b", Bm - th, Vi, Bm - th)
    return dict(W=W, gl=k, p_chi2=float(1 - stats.chi2.cdf(W, k)), p_boot=float((Wb >= W).mean()), n_pre=len(ie))


# ----------------------------------------------------------------------------
# TWFE (solo como referencia)
# ----------------------------------------------------------------------------
def twfe(df: pd.DataFrame, esp: Espec, B: int = C.B_BOOT, XI=None, modo: str = C.INFERENCIA) -> dict:
    """
    Y_ist = α_s + λ_t + β D_st + X_i'γ + ε_ist   (mínimos cuadrados ponderados con FACTOR)
    D_st = 1 si en la entidad s ya estaba legalizado el matrimonio en el año t.
    EE: (a) agrupados por entidad (CR1) y (b) bootstrap multiplicador por entidad.
    """
    cols = [esp.y, esp.t, esp.g, esp.peso, esp.cluster] + list(esp.covars)
    d = df.dropna(subset=cols)
    d = d[d[esp.peso] > 0]
    t = np.floor(d[esp.t].values.astype(float) + esp.desplaz).astype(int)
    G = d[esp.g].astype(int).values
    tratados = G[G != C.ANIO_NO_TRATADO]
    t_min = (tratados.min() if len(tratados) else t.min()) - esp.pre
    m = (t <= esp.anio_max) & (t >= t_min)
    d, t, G = d[m], t[m], G[m]
    per = (t // esp.bin) * esp.bin
    Gb = np.where(G == C.ANIO_NO_TRATADO, C.ANIO_NO_TRATADO, (G // esp.bin) * esp.bin)
    D = ((per >= Gb) & (Gb != C.ANIO_NO_TRATADO)).astype(float)
    cl = d[esp.cluster].astype(int).values
    cols_X = [D] + [d[c].values.astype(float) for c in esp.covars]
    for e in np.unique(cl)[1:]:
        cols_X.append((cl == e).astype(float))
    for p in np.unique(per)[1:]:
        cols_X.append((per == p).astype(float))
    X = np.column_stack([np.ones(len(d))] + cols_X)
    y = d[esp.y].values.astype(float)
    s = d[esp.peso].values.astype(float)
    beta = wls(X, y, s)
    resid = y - X @ beta
    bread = np.linalg.inv((X * s[:, None]).T @ X)
    meat = np.zeros_like(bread)
    for c in np.unique(cl):
        mk = cl == c
        sc = (X[mk] * (s[mk] * resid[mk])[:, None]).sum(0)
        meat += np.outer(sc, sc)
    Gc, N, K = len(np.unique(cl)), len(y), X.shape[1]
    V = bread @ meat @ bread * (Gc / (Gc - 1)) * ((N - 1) / (N - K))
    out = dict(beta_pp=beta[1] * 100, se_cr1_pp=float(np.sqrt(V[1, 1])) * 100, n=N, n_clusters=Gc,
               p_cr1=float(2 * (1 - stats.t.cdf(abs(beta[1] / np.sqrt(V[1, 1])), df=Gc - 1))))
    if B and B > 0:
        if XI is None or XI[0].shape[0] < B:
            XI = generar_xi(B)
        idx_cl = cl - 1
        idx_up = d["UPM_DIS"].astype(int).values
        bs = np.empty(B)
        for b in range(B):
            w = s
            if modo in ("entidad", "ambos"):
                w = w * XI[0][b][idx_cl]
            if modo in ("upm", "ambos"):
                w = w * XI[1][b][idx_up]
            bs[b] = wls(X, y, w)[1]
        out["se_boot_pp"] = float(np.std(bs, ddof=1)) * 100
        out["p_boot"] = float(2 * (1 - stats.norm.cdf(abs(beta[1] / np.std(bs, ddof=1)))))
    z = stats.norm.ppf(0.5 + C.NIVEL_CONFIANZA / 2)
    se_ref = out.get("se_boot_pp", out["se_cr1_pp"])
    out["lo_pp"], out["hi_pp"] = out["beta_pp"] - z * se_ref, out["beta_pp"] + z * se_ref
    return out
