"""
01_construir_base.py — Construcción de la base analítica
=========================================================
Metodología (sección "Preparar los datos"):
  * personas de 15 años o más que ya tuvieron relaciones sexuales y dieron respuestas válidas;
  * se excluyen "otro", "no recuerda" y "no especificado";
  * las personas trans se separan del análisis principal (prueba de sensibilidad aparte);
  * Y_ist = 1 si la primera pareja fue del mismo sexo (sexo asignado al nacer vs. sexo de la
    pareja), 0 si fue de distinto sexo;
  * G_s  = año en que la entidad s legalizó el matrimonio igualitario;
  * X_i  = cohorte de nacimiento, sexo e índice socioeconómico (bienes y características de la
    vivienda). La orientación autoidentificada NO entra como control (se mide en 2021, después
    del tratamiento): solo se usa para heterogeneidad y placebo.

Entradas : TMODULO, TSDEM, TVIVIENDA (CSV de INEGI), datos/legalizacion_entidades.csv
Salidas  : datos/procesados/base_analitica.csv.gz
           resultados/tablas/flujo_muestra.csv
Estructura relacional (Modelo Relacional de INEGI):
    TVIVIENDA (FOLIO, VIV_SEL) 1─N TSDEM (FOLIO, VIV_SEL, HOGAR, N_REN) 1─1 TMODULO (mismas 4 llaves)
TMODULO contiene solo a la persona elegida de cada hogar; su FACTOR es el único ponderador válido
para las variables de TMODULO (P7_*, P8_*, P9_*): no se mezcla con el FACTOR de TSDEM/TVIVIENDA.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
from utils_encuesta import buscar_csv, leer_csv, num, indicadores_vivienda, indice_ses, cuantil_ponderado

LLAVES_P = ["FOLIO", "VIV_SEL", "HOGAR", "N_REN"]
COLS_TMODULO = LLAVES_P + [
    "P4_1", "P7_1", "P7_1A", "P7_2", "P7_3", "P7_4_1", "P7_4_2", "P7_4_3", "P7_5",
    "P7_6_1", "P7_6_2", "P7_6_3", "P7_8", "P8_1", "P8_1A", "P8_2", "P9_1", "FILTRO_9_2",
    "P12_1_2", "ENT", "EST_DIS", "UPM_DIS", "FACTOR",
]
OBLIGATORIAS = ["P4_1", "P7_1", "P7_3", "P7_4_1", "P7_4_2", "P7_5", "P7_6_1", "P7_6_2", "P7_6_3",
                "P8_1", "P9_1", "ENT", "EST_DIS", "UPM_DIS", "FACTOR"]
COLS_TSDEM = LLAVES_P + ["EDAD", "SEXO", "P3_6_1", "P3_6_2", "EST_DIS", "UPM_DIS"]
COLS_TVIV = ["FOLIO", "VIV_SEL", "P1_1", "P1_2", "P1_3", "P1_4", "P1_5", "P1_6", "P1_7_1", "P1_7_2",
             "P1_7_3", "P1_7_4", "P1_7_5", "P1_7_6", "P1_7_7", "P2_1"]


def pareja_mismo_sexo(sexo_nac, es_hombre, es_mujer, es_otro):
    """
    Y = 1 si la pareja es del mismo sexo asignado al nacer de la persona, 0 si es de distinto sexo.
    Respuestas válidas: exactamente una de {hombre, mujer} y no "otro". Cualquier otra combinación
    (otro, ambas, ninguna) → NaN (se excluye).
    """
    hom, muj, otro = (es_hombre == 1), (es_mujer == 1), (es_otro == 1)
    valido = (hom ^ muj) & ~otro
    mismo = np.where(sexo_nac == 1, hom, muj)
    return pd.Series(np.where(valido, mismo.astype(float), np.nan), index=sexo_nac.index)


def construir():
    C.DIR_PROC.mkdir(parents=True, exist_ok=True)
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ lectura
    print(">> Leyendo tablas de INEGI desde", C.DIR_DATOS)
    r_tm = buscar_csv(C.DIR_DATOS, C.PATRONES_ARCHIVO["tmodulo"])
    tm = leer_csv(r_tm, COLS_TMODULO)
    faltan = [c for c in OBLIGATORIAS if c not in tm.columns]
    if faltan:
        raise KeyError(f"En {r_tm.name} faltan columnas obligatorias: {faltan}")
    sd = leer_csv(buscar_csv(C.DIR_DATOS, C.PATRONES_ARCHIVO["tsdem"]), COLS_TSDEM)
    vv = leer_csv(buscar_csv(C.DIR_DATOS, C.PATRONES_ARCHIVO["tvivienda"]), COLS_TVIV)
    print(f"   TMODULO {len(tm):,} filas | TSDEM {len(sd):,} | TVIVIENDA {len(vv):,}")

    # ------------------------------------------------------------------ uniones
    assert not tm.duplicated(LLAVES_P).any(), "TMODULO: llave duplicada"
    assert not sd.duplicated(LLAVES_P).any(), "TSDEM: llave duplicada"
    assert not vv.duplicated(["FOLIO", "VIV_SEL"]).any(), "TVIVIENDA: llave duplicada"

    sd_ = sd.rename(columns={"EST_DIS": "EST_DIS_sd", "UPM_DIS": "UPM_DIS_sd"})
    d = tm.merge(sd_, on=LLAVES_P, how="left", validate="one_to_one", indicator="_m_sd")
    n_sin_sd = int((d["_m_sd"] != "both").sum())
    d = d.drop(columns="_m_sd")
    d = d.merge(vv, on=["FOLIO", "VIV_SEL"], how="left", validate="many_to_one", indicator="_m_v")
    n_sin_viv = int((d["_m_v"] != "both").sum())
    d = d.drop(columns="_m_v")
    print(f"   Personas de TMODULO sin renglón en TSDEM: {n_sin_sd} | sin vivienda en TVIVIENDA: {n_sin_viv}")

    inc = ((d["EST_DIS"] != d["EST_DIS_sd"]) | (d["UPM_DIS"] != d["UPM_DIS_sd"])).sum()
    print(f"   Inconsistencias EST_DIS/UPM_DIS entre TMODULO y TSDEM: {int(inc)}")

    # ------------------------------------------------------------------ numéricas básicas
    for c in ["P4_1", "P7_1", "P7_1A", "P7_2", "P7_3", "P7_4_1", "P7_4_2", "P7_4_3", "P7_5",
              "P7_6_1", "P7_6_2", "P7_6_3", "P7_8", "P8_1", "P8_1A", "P8_2", "P9_1", "FILTRO_9_2",
              "P12_1_2", "ENT", "FACTOR"]:
        d[c] = num(d, c)
    d["EDAD_sd"] = num(d, "EDAD")
    d["ENT"] = d["ENT"].astype("Int64")

    # Verificación: los dos primeros dígitos del FOLIO son la entidad
    ent_folio = d["FOLIO"].str[:2].astype(int)
    n_ent_inc = int((ent_folio != d["ENT"]).sum())
    print(f"   Discrepancias ENT vs. FOLIO[:2]: {n_ent_inc}")

    # UPM anidada en una sola entidad (necesario para agrupar errores por entidad)
    upm_multi = (d.groupby("UPM_DIS")["ENT"].nunique() > 1).sum()
    print(f"   UPM que cruzan entidades: {int(upm_multi)}")

    # ------------------------------------------------------------------ demografía
    edad = d["P4_1"].where(d["P4_1"].between(15, 96))
    edad = edad.fillna(d["EDAD_sd"].where(d["EDAD_sd"].between(15, 96)))
    d["edad"] = edad
    d["anio_nac"] = C.ANIO_REF - d["edad"]
    d["mujer"] = (d["P7_1"] == 2).astype(float).where(d["P7_1"].isin([1, 2]))
    d["cohorte_c"] = (d["anio_nac"] - 1980) / 10.0
    d["cohorte_grupo"] = pd.cut(d["anio_nac"], [0, 1969, 1979, 1989, 1999, 3000],
                                labels=["<1970", "1970-79", "1980-89", "1990-99", "2000+"])

    # ------------------------------------------------------------------ orientación e identidad
    d["orient_cat"] = d["P8_1"].map({1: "lesbiana", 2: "gay", 3: "bisexual", 4: "heterosexual",
                                     5: "heterosexual", 6: "otra"})
    lgb = d["P8_1"].isin([1, 2, 3, 6])
    # Identidad de género no normativa: sexo al nacer distinto de identidad (FILTRO_9_2 == 1),
    # o "tanto hombre como mujer", "ni hombre ni mujer", "otro género" (P9_1 ∈ {3,4,5}).
    trans = (d["FILTRO_9_2"] == 1) | d["P9_1"].isin([3, 4, 5]) | (
        d["P9_1"].isin([1, 2]) & d["P7_1"].isin([1, 2]) & (d["P9_1"] != d["P7_1"]))
    d["trans_nobin"] = trans.astype(int)
    d["lgbti_v1"] = (lgb | trans).astype(int)
    d["lgbti_v2"] = (lgb | trans | (d["P7_1A"] == 1)).astype(int)
    d["lgbti"] = d["lgbti_v1"] if C.DEF_LGBTI == "v1" else d["lgbti_v2"]
    d["grupo_osig"] = np.where(d["lgbti"] == 1, "LGBTI+", "normativa")
    d["lgb_orient"] = lgb.astype(int)

    # ------------------------------------------------------------------ inicio de vida sexual
    p75 = d["P7_5"]
    d["nunca_rs"] = (p75 == 0).astype(int)
    d["edad_debut"] = p75.where(p75.between(1, 84))
    d["debut_nr_ne"] = p75.isin([98, 99]).astype(int)
    if C.EDAD_DEBUT_MIN is not None:
        d["edad_debut"] = d["edad_debut"].where(d["edad_debut"] >= C.EDAD_DEBUT_MIN)
    d["debut_gt_edad"] = (d["edad_debut"] > d["edad"]).astype(int)
    d["anio_inicio"] = d["anio_nac"] + d["edad_debut"] + C.DESPLAZAMIENTO_ANIO_INICIO

    # Resultado principal: primera pareja sexual del mismo sexo
    d["Y_mismo_sexo"] = pareja_mismo_sexo(d["P7_1"], d["P7_6_1"], d["P7_6_2"], d["P7_6_3"])
    d["pareja_otro"] = (d["P7_6_3"] == 1).astype(int)

    # Resultado alternativo: primer encuentro erótico (besos/caricias consentidos), P7_3 y P7_4_*
    p73 = d["P7_3"]
    d["edad_erotico"] = p73.where(p73.between(1, 63))
    d["anio_erotico"] = d["anio_nac"] + d["edad_erotico"] + C.DESPLAZAMIENTO_ANIO_INICIO
    d["Y_erotico_mismo_sexo"] = pareja_mismo_sexo(d["P7_1"], d["P7_4_1"], d["P7_4_2"], d["P7_4_3"])

    # Para el análisis de contrafactuales: alguna vez en la vida relaciones con alguien del mismo sexo (P7_8)
    #   P7_8: 1 casi siempre con mujeres y algunas con hombres | 2 con mujeres y hombres por igual |
    #         3 casi siempre con hombres y algunas con mujeres | 4 únicamente con hombres | 5 únicamente con mujeres
    p78 = d["P7_8"]
    ever_h = p78.isin([1, 2, 3, 4])   # tuvo parejas hombres
    ever_m = p78.isin([1, 2, 3, 5])   # tuvo parejas mujeres
    ever_mismo = np.where(d["P7_1"] == 1, ever_h, ever_m).astype(float)
    d["alguna_vez_mismo_sexo"] = pd.Series(ever_mismo, index=d.index).where(p78.between(1, 5))

    # "Cuándo se dieron cuenta" de su orientación (P8_2: 01 = siempre lo supo; 02-59 = edad)
    p82 = d["P8_2"]
    edad_cuenta = pd.Series(np.where(p82 == 1, 0.0, np.where(p82.between(2, 59), p82, np.nan)), index=d.index)
    d["anio_darse_cuenta"] = d["anio_nac"] + edad_cuenta
    d["consciente_al_debut"] = (d["anio_inicio"] >= d["anio_darse_cuenta"]).astype(float).where(
        d["anio_darse_cuenta"].notna() & d["anio_inicio"].notna())

    # ------------------------------------------------------------------ índice socioeconómico
    ind = indicadores_vivienda(d)
    ok_viv = d["P1_1"].notna()
    ses = np.full(len(d), np.nan)
    if ok_viv.sum() > 50:
        s, info = indice_ses(ind[ok_viv], d.loc[ok_viv, "FACTOR"].values)
        ses[ok_viv.values] = s
        print(f"   Índice SES: 1.er componente explica {info['var_explicada']:.1%} de la varianza")
        pd.Series(info["cargas"], name="carga").to_csv(C.DIR_TABLAS / "ses_cargas.csv")
    d["ses_z"] = ses
    ok = d["ses_z"].notna()
    if ok.sum() > 0:
        q = cuantil_ponderado(d.loc[ok, "ses_z"].values, d.loc[ok, "FACTOR"].values, [1 / 3, 2 / 3])
        d["ses_tercil"] = np.where(d["ses_z"] <= q[0], 1, np.where(d["ses_z"] <= q[1], 2, 3))
        d["ses_tercil"] = pd.Series(d["ses_tercil"], index=d.index).where(ok)

    # ------------------------------------------------------------------ tratamiento
    leg = pd.read_csv(C.ARCHIVO_LEGALIZACION, dtype={"ent": str})
    leg["ENT"] = leg["ent"].astype(int)
    lim = pd.Timestamp(C.FECHA_LIMITE_TRATAMIENTO)
    for col, nom in [("fecha_principal", "G"), ("fecha_aprobacion", "G_aprob"), ("fecha_formal_vigor", "G_vigor")]:
        f = pd.to_datetime(leg[col])
        leg[nom] = np.where(f < lim, f.dt.year, C.ANIO_NO_TRATADO)
    leg["G_nota_dias"] = (lim - pd.to_datetime(leg["fecha_principal"])).dt.days
    d = d.merge(leg[["ENT", "entidad", "G", "G_aprob", "G_vigor", "amparo_temprano",
                     "anio_amparo_temprano", "discrepancia_fuentes"]], on="ENT", how="left", validate="many_to_one")
    d["G"] = d["G"].astype("Int64")

    # ------------------------------------------------------------------ muestra principal y flujo
    d["w"] = d["FACTOR"]
    base = d["edad"].notna() & (d["edad"] >= C.EDAD_MIN)
    tuvo = base & d["edad_debut"].notna()
    sin_debut_valido = base & ~tuvo
    valido_y = tuvo & d["Y_mismo_sexo"].notna()
    no_inconsistente = valido_y & (d["debut_gt_edad"] == 0) & (d["anio_inicio"] <= C.ANIO_MAX_INICIO)
    cov_ok = no_inconsistente & d[C.COVARIABLES].notna().all(axis=1)
    principal = cov_ok & (d["trans_nobin"] == 0)
    d["muestra_principal"] = principal.astype(int)
    d["muestra_con_trans"] = cov_ok.astype(int)

    wsum = lambda m: float(d.loc[m, "FACTOR"].sum())  # noqa: E731
    pasos = [
        ("Personas en TMODULO", d.index == d.index, None),
        ("Con edad válida de 15 años o más", base, None),
        ("  (-) Nunca han tenido relaciones sexuales (P7_5 = 00)", base & (d["nunca_rs"] == 1), "resta"),
        ("  (-) Edad de primera relación 'no recuerda' / 'no especificado' (98/99)", base & (d["debut_nr_ne"] == 1), "resta"),
        ("Con edad de primera relación válida", tuvo, None),
        ("  (-) Primera pareja 'otro', ambos sexos o sin respuesta", tuvo & d["Y_mismo_sexo"].isna(), "resta"),
        ("Con resultado Y válido", valido_y, None),
        ("  (-) Edad de debut > edad actual o año de inicio posterior al levantamiento", valido_y & ~no_inconsistente, "resta"),
        ("  (-) Falta alguna covariable (cohorte, sexo, índice SES)", no_inconsistente & ~cov_ok, "resta"),
        ("Muestra con trans/no binarias incluidas (sensibilidad)", cov_ok, None),
        ("  (-) Personas trans / no binarias (se analizan aparte)", cov_ok & (d["trans_nobin"] == 1), "resta"),
        ("MUESTRA PRINCIPAL", principal, None),
    ]
    flujo = pd.DataFrame({"paso": [p[0] for p in pasos],
                          "n_muestra": [int(p[1].sum()) for p in pasos],
                          "n_expandido": [round(wsum(p[1])) for p in pasos]})
    flujo.to_csv(C.DIR_TABLAS / "flujo_muestra.csv", index=False)
    print("\n" + flujo.to_string(index=False))

    cols_salida = (LLAVES_P + ["ENT", "entidad", "EST_DIS", "UPM_DIS", "FACTOR", "w", "edad", "anio_nac", "cohorte_c",
                   "cohorte_grupo", "mujer", "P7_1", "P9_1", "orient_cat", "lgb_orient", "trans_nobin", "lgbti_v1",
                   "lgbti_v2", "lgbti", "grupo_osig", "edad_debut", "anio_inicio", "edad_erotico", "anio_erotico",
                   "Y_mismo_sexo", "Y_erotico_mismo_sexo", "alguna_vez_mismo_sexo", "anio_darse_cuenta",
                   "consciente_al_debut", "ses_z", "ses_tercil", "G", "G_aprob", "G_vigor", "amparo_temprano",
                   "anio_amparo_temprano", "discrepancia_fuentes", "P12_1_2", "muestra_principal", "muestra_con_trans"])
    cols_salida = [c for c in cols_salida if c in d.columns]
    d[cols_salida].to_csv(C.BASE_ANALITICA, index=False)
    print(f"\n>> Base analítica guardada en {C.BASE_ANALITICA}  ({len(d):,} filas)")
    return d


if __name__ == "__main__":
    construir()
