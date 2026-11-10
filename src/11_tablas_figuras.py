"""
11_tablas_figuras.py — Paquete de tablas y figuras listo para presentación
==========================================================================
Reúne en  resultados/presentacion/  todo lo que se necesita para armar diapositivas o un artículo:

  tablas/  T01…T10 en CSV con formato de publicación ("estimado (EE)", "[IC 95%]", p, p ajustado) y UN solo
           libro  tablas_presentacion.xlsx  (una hoja por tabla, si está instalado openpyxl) + tablas.md
  figuras/ F01…F12 (PNG a 200 dpi y SVG vectorial, proporción 16:9) con nombre ordenado
  indice_presentacion.md   qué es cada tabla/figura y qué mensaje sostiene

Es un paso de LECTURA: no estima nada. Si falta alguna tabla (porque no corriste ese paso), se omite.
Entradas : tablas y figuras de los pasos 02–10
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
from graficos import grafica_bosque

DIR_OUT = C.DIR_TRABAJO / "resultados" / "presentacion"
T = C.DIR_TABLAS
F = C.DIR_FIGURAS


def _leer(nombre):
    r = T / nombre
    return pd.read_csv(r) if r.exists() else None


def _primero(*nombres):
    for n in nombres:
        d = _leer(n)
        if d is not None:
            return d
    return None


def _num(x, d=2):
    return "" if pd.isna(x) else f"{x:.{d}f}".replace("-", "−")


def _est_ee(est, se):
    return "" if pd.isna(est) else (f"{_num(est)} ({_num(se)})" if pd.notna(se) else _num(est))


def _ic(lo, hi):
    return "" if pd.isna(lo) or pd.isna(hi) else f"[{_num(lo)}, {_num(hi)}]"


def _p(x):
    return "" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}")


def _estrellas(p):
    return "" if pd.isna(p) else ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")


def _formatea(df, nombre_col, extra=()):
    """est_pp/se_pp/lo_pp/hi_pp/p (+ p_holm, p_bh) → columnas de publicación."""
    if "est_pp" not in df.columns and "theta_pp" in df.columns:
        df = df.rename(columns={"theta_pp": "est_pp"})
    out = pd.DataFrame({nombre_col[0]: df[nombre_col[1]].values})
    for c, ncol in extra:
        out[ncol] = df[c].values
    out["Efecto en pp (EE)"] = [_est_ee(a, b) + _estrellas(p) for a, b, p in zip(df["est_pp"], df["se_pp"], df["p"])]
    out["IC 95%"] = [_ic(a, b) for a, b in zip(df["lo_pp"], df["hi_pp"])]
    out["p"] = df["p"].map(_p).values
    if "p_holm" in df:
        out["p Holm"] = df["p_holm"].map(_p).values
        out["p BH"] = df["p_bh"].map(_p).values
    return out


def main():
    (DIR_OUT / "tablas").mkdir(parents=True, exist_ok=True)
    (DIR_OUT / "figuras").mkdir(parents=True, exist_ok=True)
    tablas, indice_t, indice_f = {}, [], []

    def agrega(clave, titulo, df, mensaje):
        if df is None or len(df) == 0:
            return
        tablas[clave] = df
        df.to_csv(DIR_OUT / "tablas" / f"{clave}.csv", index=False, encoding="utf-8-sig")
        indice_t.append((clave, titulo, mensaje))

    # T01: modelo principal + TWFE
    cs = _leer("cs_resumen.csv")
    tw = _leer("twfe_vs_cs.csv")
    if cs is not None:
        nom = {"global_post": "Efecto promedio posterior a la ley", "corto": "Corto plazo (0–2 años)",
               "mediano": "Mediano plazo (3–5 años)", "largo": "Largo plazo (6+ años)"}
        t1 = cs[cs["nombre"].isin(nom)].copy()
        t1["Estimación"] = t1["nombre"].map(nom)
        t1 = _formatea(t1, ("Estimación", "Estimación"))
        t1.insert(1, "Modelo", "Callaway–Sant'Anna")
        if tw is not None:
            r = tw[tw["modelo"].astype(str).str.contains("TWFE")].copy()
            if len(r):
                f = _formatea(r.assign(Estimación="Efecto promedio (referencia)"), ("Estimación", "Estimación"))
                f.insert(1, "Modelo", "TWFE")
                t1 = pd.concat([t1, f], ignore_index=True)
        agrega("T01_modelo_principal", "Modelo principal y TWFE", t1,
               "El efecto promedio sobre la primera pareja del mismo sexo no se distingue de cero con C&S ni con TWFE.")

    # T02: efectos por grupo
    cs_a = _leer("cs_resumen_ajustado.csv")
    if cs_a is not None:
        g = cs_a[cs_a["nombre"].astype(str).str.startswith("grupo_")].copy()
        g["Grupo"] = g["nombre"].str.replace("grupo_", "Legalizó en ", regex=False)
        agrega("T02_efecto_por_grupo", "Efecto por grupo de legalización", _formatea(g, ("Grupo", "Grupo")),
               "Los grupos son ruidosos; ajustar por comparaciones múltiples elimina los p < 0.05 aislados.")

    # T03: estudio de eventos
    es = _primero("cs_evento_ajustado.csv", "cs_evento.csv")
    if es is not None:
        t3 = _formatea(es.assign(e_=es["e"]), ("e (años desde la ley)", "e_"), extra=[("n_celdas", "Celdas")])
        agrega("T03_estudio_de_eventos", "Estudio de eventos θ(e)", t3,
               "Coeficientes previos ≈ 0 respaldan tendencias paralelas; los de e ≥ 6 descansan en muy pocas celdas.")

    # T04: año calendario
    an = _leer("cs_anio_calendario.csv")
    if an is not None:
        t4 = _formatea(an, ("Año de inicio sexual", "anio"), extra=[("n_celdas", "Celdas"), ("n_trat", "Tratados")])
        agrega("T04_anio_calendario", "Efecto por año calendario de inicio sexual", t4,
               "Efecto promedio por año de debut (solo entidades ya con ley).")

    # T05: heterogeneidad
    h = _primero("heterogeneidad_resumen_ajustado.csv", "heterogeneidad_resumen.csv")
    if h is not None:
        t5 = _formatea(h, ("Grupo", "grupo"), extra=[("dimension", "Dimensión"), ("n", "N"), ("celdas", "Celdas")])
        t5 = t5[["Dimensión", "Grupo", "N", "Celdas"] + [c for c in t5.columns if c not in ("Dimensión", "Grupo", "N", "Celdas")]]
        agrega("T05_heterogeneidad", "Heterogeneidad del efecto", t5,
               "Ninguna diferencia entre submuestras es robusta tras ajustar por comparaciones múltiples.")
    hd = _leer("heterogeneidad_diferencias_ajustado.csv")
    if hd is not None:
        hd = hd.rename(columns={"dif_pp": "est_pp"})
        hd["lo_pp"], hd["hi_pp"] = np.nan, np.nan
        agrega("T05b_diferencias", "Diferencias entre submuestras",
               _formatea(hd, ("Grupo", "grupo"), extra=[("dimension", "Dimensión"), ("referencia", "Referencia")]),
               "Diferencia contra el primer grupo de cada dimensión, con remuestras pareadas.")

    # T06: robustez
    r = _primero("robustez_resumen_ajustado.csv", "robustez_resumen.csv")
    if r is not None:
        agrega("T06_robustez", "Robustez y sensibilidad", _formatea(r, ("Especificación", "especificacion"), extra=[("n", "N"), ("celdas", "Celdas")]),
               "El signo y el tamaño se mantienen en la mayoría; el placebo en heterosexuales es ≈ 0.")

    # T07: mecanismos
    me = _primero("mecanismos_resumen_ajustado.csv", "mecanismos_resumen.csv")
    if me is not None:
        t7 = _formatea(me, ("Resultado", "resultado"), extra=[("poblacion", "Población"), ("n", "N")])
        if "wald_p_boot" in me:
            t7["p tendencias previas (Wald)"] = me["wald_p_boot"].map(_p).values
        t7 = t7[["Población"] + [c for c in t7.columns if c != "Población"]]
        agrega("T07_mecanismos", "Mecanismos: se adelanta / se cambia", t7,
               "El efecto sobre 'alguna vez pareja del mismo sexo' (Z) es el candidato a hallazgo; exige tendencias previas planas.")

    # T08: leave-one-out
    lo = _leer("diag_loo_entidad.csv")
    if lo is not None:
        t8 = lo.sort_values("cambio_pp").copy()
        t8["Efecto sin la entidad (pp)"] = t8["est_pp"].map(_num)
        t8["Cambio vs. todas (pp)"] = t8["cambio_pp"].map(_num)
        t8 = t8.rename(columns={"entidad": "Entidad", "G": "Año de legalización", "amparo_temprano": "Amparo temprano"})
        agrega("T08_leave_one_out", "Sensibilidad a excluir una entidad",
               t8[["Entidad", "Año de legalización", "Amparo temprano", "Efecto sin la entidad (pp)", "Cambio vs. todas (pp)"]],
               "Muestra si el resultado depende de una entidad o de las que tienen amparos tempranos.")

    # T09/T10: descriptivos y flujo de muestra
    for arch, clave, titulo, msg in [("flujo_muestra.csv", "T09_flujo_muestra", "Flujo de la muestra", "Cómo se llega de 44,189 personas a la muestra principal."),
                                     ("descriptivos_Y.csv", "T10_descriptivos", "Descriptivos del resultado", "Tasas ponderadas con EE de diseño.")]:
        d = _leer(arch)
        if d is not None:
            agrega(clave, titulo, d.round(4), msg)

    # --- libro de Excel y markdown
    try:
        with pd.ExcelWriter(DIR_OUT / "tablas_presentacion.xlsx", engine="openpyxl") as xw:
            for k, d in tablas.items():
                d.to_excel(xw, sheet_name=k[:31], index=False)
    except Exception as ex:
        print(f"   (no se generó el .xlsx: {ex}; los CSV sí están)")
    with open(DIR_OUT / "tablas.md", "w", encoding="utf-8") as fh:
        for k, d in tablas.items():
            fh.write(f"### {k}\n\n")
            try:
                fh.write(d.to_markdown(index=False) + "\n\n")
            except Exception:
                fh.write(d.to_string(index=False) + "\n\n")

    # --- figura-resumen (forest) con los resultados clave
    clave = []
    if cs is not None:
        for n, et in [("global_post", "Modelo principal: efecto promedio"), ("corto", "Corto plazo (0–2 años)"),
                      ("mediano", "Mediano plazo (3–5)"), ("largo", "Largo plazo (6+)")]:
            f = cs[cs["nombre"] == n]
            if len(f):
                clave.append(dict(especificacion=et, **f.iloc[0][["est_pp", "lo_pp", "hi_pp"]].to_dict()))
    if tw is not None:
        f = tw[tw["modelo"].astype(str).str.contains("TWFE")]
        if len(f):
            clave.append(dict(especificacion="TWFE (referencia)", **f.iloc[0][["est_pp", "lo_pp", "hi_pp"]].to_dict()))
    if r is not None:
        for pref, et in [("8.", "Placebo: heterosexuales"), ("3a", "Sin entidades con amparos tempranos"), ("6.", "Sin covariables")]:
            f = r[r["especificacion"].astype(str).str.startswith(pref)]
            if len(f) and pd.notna(f.iloc[0]["est_pp"]):
                clave.append(dict(especificacion=et, **f.iloc[0][["est_pp", "lo_pp", "hi_pp"]].to_dict()))
    if me is not None:
        f = me[(me["poblacion"] == "Toda la muestra") & me["resultado"].astype(str).str.startswith("Z:")]
        if len(f):
            clave.append(dict(especificacion="Alguna vez pareja del mismo sexo (Z)", **f.iloc[0][["est_pp", "lo_pp", "hi_pp"]].to_dict()))
    if clave:
        grafica_bosque(pd.DataFrame(clave), F / "resumen_resultados.png", "Resumen: efecto de la legalización según especificación")

    # --- copiar figuras con nombre ordenado
    figs = [("cs_tasas", "tasas_por_grupo", "Tasas por grupo de legalización y año de inicio (descriptivo)"),
            ("cs_evento", "cs_evento", "Estudio de eventos del modelo principal"),
            ("cs_anio", "cs_anio", "Efecto por año calendario de inicio sexual"),
            ("resumen", "resumen_resultados", "Resumen de resultados clave"),
            ("robustez", "robustez", "Robustez (17 especificaciones)"),
            ("heterogeneidad", "heterogeneidad", "Heterogeneidad por submuestra"),
            ("mecanismos", "mecanismos", "Mecanismos: efecto promedio"),
            ("mecanismos_evento", "mecanismos_evento", "Mecanismos: estudio de eventos y tendencias previas"),
            ("loo_entidades", "loo_entidades", "Sensibilidad a excluir una entidad"),
            ("rr_sd", "rr_sd_0", "Sensibilidad Rambachan–Roth (suavidad), efecto inmediato"),
            ("rr_rm", "rr_rm_0", "Sensibilidad Rambachan–Roth (magnitudes relativas), efecto inmediato")]
    for i, (clave_f, arch, desc) in enumerate(figs, start=1):
        for ext in (".png", ".svg"):
            src = F / f"{arch}{ext}"
            if src.exists():
                shutil.copy(src, DIR_OUT / "figuras" / f"F{i:02d}_{clave_f}{ext}")
                if ext == ".png":
                    indice_f.append((f"F{i:02d}_{clave_f}", desc))

    with open(DIR_OUT / "indice_presentacion.md", "w", encoding="utf-8") as fh:
        fh.write("# Índice de tablas y figuras para la presentación\n\n## Tablas\n\n")
        for k, t, msg in indice_t:
            fh.write(f"- **{k}** — {t}. _{msg}_\n")
        fh.write("\n## Figuras (PNG y SVG)\n\n")
        for k, d in indice_f:
            fh.write(f"- **{k}** — {d}\n")
        fh.write("\nLos mensajes son hipótesis de lectura: confírmalos con los números reales antes de presentarlos.\n")
    print(f">> {len(tablas)} tablas y {len(indice_f)} figuras en {DIR_OUT}")
    for k, t, _ in indice_t:
        print(f"   {k}: {t}")


if __name__ == "__main__":
    main()
