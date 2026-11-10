"""
02_diagnosticos.py — Validaciones y diagnóstico de la muestra (ANTES de estimar)
=================================================================================
Responde cuatro preguntas que condicionan todo lo demás:
  1. ¿Mi base reproduce lo que publica INEGI?  (totales de los tabulados; FACTOR vs. TAPART A/B)
  2. ¿Cómo se anidan las variables de diseño? (UPM dentro de entidad; estratos)
  3. ¿Cuántos datos hay por grupo de legalización y año?  (es la restricción binding del DiD)
  4. ¿Qué celdas ATT(g,t) son estimables y cuáles se omiten por muestra insuficiente?

Entradas : base_analitica.csv.gz (01), TAPART_A / TAPART_B (opcional, solo para validar FACTOR)
Salidas  : resultados/tablas/{validacion_tabulados,descriptivos_Y,conteos_grupo_anio,celdas_omitidas,
           grupos_legalizacion}.csv ; resultados/figuras/tasas_por_grupo.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
from did_cs import Espec, _preparar, _construir_celdas
from graficos import grafica_tasas
from utils_encuesta import buscar_csv, leer_csv, media_svy, ic_normal


TXT = {"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str}


def main():
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    C.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.BASE_ANALITICA, dtype=TXT)
    m = df[df["muestra_principal"] == 1].copy()

    # ------------------------------------------------------------------ 1. validación contra INEGI
    print("== 1. Validación contra los tabulados básicos de INEGI (observaciones muestrales)")
    T = C.TOTALES_TABULADOS
    filas = [
        ("Total de personas", len(df), T["total"]),
        ("OSIG LGBTI+ (v1: orientación LGB+ o identidad no normativa)", int(df["lgbti_v1"].sum()), T["lgbti"]),
        ("OSIG LGBTI+ (v2: v1 + variación corporal/intersex)", int(df["lgbti_v2"].sum()), T["lgbti"]),
        ("OSIG normativa (v1)", int((1 - df["lgbti_v1"]).sum()), T["normativa"]),
        ("OSIG normativa (v2)", int((1 - df["lgbti_v2"]).sum()), T["normativa"]),
    ]
    val = pd.DataFrame(filas, columns=["concepto", "esta_base", "tabulado_INEGI"])
    val["diferencia"] = val["esta_base"] - val["tabulado_INEGI"]
    print(val.to_string(index=False))
    val.to_csv(C.DIR_TABLAS / "validacion_tabulados.csv", index=False)
    mejor = "v1" if abs(val.iloc[1]["diferencia"]) <= abs(val.iloc[2]["diferencia"]) else "v2"
    print(f"   -> La definición que más se acerca a INEGI es {mejor}  (DEF_LGBTI en config.py = {C.DEF_LGBTI})")
    print(f"   Población 15+ expandida (Σ FACTOR): {df['FACTOR'].sum():,.0f}")

    # FACTOR de TMODULO vs. TAPART_A/B: son las mismas personas, el ponderador debe coincidir
    try:
        ab = []
        for k in ("tapart_a", "tapart_b"):
            ab.append(leer_csv(buscar_csv(C.DIR_DATOS, C.PATRONES_ARCHIVO[k]),
                               ["FOLIO", "VIV_SEL", "HOGAR", "N_REN", "FACTOR"]))
        ab = pd.concat(ab)
        ab["FACTOR_ab"] = pd.to_numeric(ab["FACTOR"])
        chk = df.merge(ab[["FOLIO", "VIV_SEL", "HOGAR", "N_REN", "FACTOR_ab"]],
                       on=["FOLIO", "VIV_SEL", "HOGAR", "N_REN"], how="left")
        sin = int(chk["FACTOR_ab"].isna().sum())
        dif = int((chk["FACTOR"] != chk["FACTOR_ab"]).sum() - sin)
        print(f"   TMODULO vs TAPART_A ∪ TAPART_B: personas sin par = {sin}; FACTOR distinto = {dif}")
    except FileNotFoundError:
        print("   (TAPART_A/TAPART_B no disponibles: se omite la verificación del FACTOR)")

    # ------------------------------------------------------------------ 2. diseño muestral
    print("\n== 2. Variables de diseño")
    print(f"   Estratos (EST_DIS): {df['EST_DIS'].nunique()} | UPM (UPM_DIS): {df['UPM_DIS'].nunique()} | "
          f"entidades: {df['ENT'].nunique()}")
    print(f"   Rango de FACTOR: {df['FACTOR'].min():,.0f} – {df['FACTOR'].max():,.0f}")
    print(f"   UPM que cruzan entidades: {int((df.groupby('UPM_DIS')['ENT'].nunique() > 1).sum())}  "
          "(0 = las UPM están anidadas en entidad, lo que justifica agrupar errores por entidad)")

    # ------------------------------------------------------------------ 3. grupos de legalización
    print("\n== 3. Grupos de legalización (G) en la muestra principal")
    gr = (m.groupby("G").agg(n_entidades=("ENT", "nunique"), n_personas=("ENT", "size"),
                              n_Y1=("Y_mismo_sexo", "sum"),
                              entidades=("entidad", lambda s: ", ".join(sorted(set(s))))).reset_index())
    gr["G"] = gr["G"].replace(C.ANIO_NO_TRATADO, "No tratada en la muestra")
    gr.to_csv(C.DIR_TABLAS / "grupos_legalizacion.csv", index=False)
    print(gr[["G", "n_entidades", "n_personas", "n_Y1"]].to_string(index=False))

    # descriptivos con diseño muestral
    print("\n== Descriptivos de Y (primera pareja del mismo sexo) con diseño muestral (linealización)")
    filas = []
    for nombre, sel in [("Todas", m.index == m.index),
                        ("OSIG LGBTI+", m["grupo_osig"] == "LGBTI+"),
                        ("OSIG normativa", m["grupo_osig"] == "normativa"),
                        ("Hombres (sexo al nacer)", m["mujer"] == 0), ("Mujeres (sexo al nacer)", m["mujer"] == 1)]:
        s = m[sel]
        r = media_svy(s["Y_mismo_sexo"], s["FACTOR"], s["EST_DIS"], s["UPM_DIS"])
        lo, hi = ic_normal(r["media"], r["se"], C.NIVEL_CONFIANZA)
        filas.append(dict(grupo=nombre, n=r["n"], pct=100 * r["media"], se_pct=100 * r["se"],
                          lo_pct=100 * lo, hi_pct=100 * hi, gl_diseno=r["gl"]))
    desc = pd.DataFrame(filas)
    print(desc.round(2).to_string(index=False))
    desc.to_csv(C.DIR_TABLAS / "descriptivos_Y.csv", index=False)

    # ------------------------------------------------------------------ 4. conteos grupo × año
    d = m.copy()
    d["anio"] = np.floor(d["anio_inicio"]).astype(int)
    d["grupo"] = np.where(d["G"] == C.ANIO_NO_TRATADO, "No tratadas", d["G"].astype(str))
    cnt = d.groupby(["grupo", "anio"]).agg(n=("Y_mismo_sexo", "size"), n_Y1=("Y_mismo_sexo", "sum"),
                                           n_pond=("FACTOR", "sum")).reset_index()
    cnt.to_csv(C.DIR_TABLAS / "conteos_grupo_anio.csv", index=False)

    def tasa(s):
        return 100 * np.average(s["Y_mismo_sexo"], weights=s["FACTOR"])

    t = (d[d["anio"] >= 2000].groupby(["grupo", "anio"]).apply(lambda s: pd.Series(
        {"tasa_pct": tasa(s), "n": len(s)}), include_groups=False).reset_index())
    t["tasa_pct"] = t.groupby("grupo")["tasa_pct"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    grafica_tasas(t, C.DIR_FIGURAS / "tasas_por_grupo.png",
                  "Primera pareja del mismo sexo por año de inicio y grupo de legalización (media móvil de 3 años)")

    # ------------------------------------------------------------------ celdas estimables
    print("\n== 4. Celdas ATT(g,t) estimables con la especificación principal")
    esp = Espec()
    P = _preparar(m, esp)
    celdas, omit = _construir_celdas(P, esp)
    omit.to_csv(C.DIR_TABLAS / "celdas_omitidas.csv", index=False)
    por_g = pd.DataFrame({"g": [c.g for c in celdas], "e": [c.e for c in celdas]})
    resumen = por_g.groupby("g").agg(celdas=("e", "size"), pre=("e", lambda s: int((s < 0).sum())),
                                     post=("e", lambda s: int((s >= 0).sum())))
    print(f"   Estimables: {len(celdas)} | omitidas: {len(omit)}  (MIN_OBS_CELDA = {C.MIN_OBS_CELDA})")
    print(resumen.to_string())
    sin_post = [g for g in sorted(set(P['G'])) if g != C.ANIO_NO_TRATADO and g not in resumen.index]
    if sin_post:
        print(f"   ⚠ Grupos sin ninguna celda estimable (se pierden del análisis): {sin_post}")
        print("     Opciones: subir ANIOS_POR_PERIODO a 2-3 o bajar MIN_OBS_CELDA en config.py")
    print("\n>> Tablas en", C.DIR_TABLAS)


if __name__ == "__main__":
    main()
