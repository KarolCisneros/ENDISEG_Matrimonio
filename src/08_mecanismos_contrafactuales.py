"""
08_mecanismos_contrafactuales.py — ¿Se adelanta o se cambia la primera pareja?
==============================================================================
Metodología, "Contrafactuales": un aumento de Y (primera pareja del mismo sexo) puede deberse a
  1. que la persona ADELANTA una relación con el mismo sexo que de todos modos habría tenido, o
  2. que la persona CAMBIA a una relación con el mismo sexo que nunca habría tenido.
Operacionalización (propuesta; confirmar con el equipo de investigación):
  * "Se cambia"  (margen extensivo): Z = 1 si ALGUNA VEZ en la vida tuvo relaciones con alguien del mismo
        sexo (P7_8), en toda la muestra. Un aumento de Z significa que más personas llegan a tener una
        pareja del mismo sexo.
  * "Se adelanta" (margen de calendario): Y (primera pareja del mismo sexo) SOLO entre quienes alguna vez
        tuvieron pareja del mismo sexo. Un aumento de Y condicionando a Z = 1 significa que, entre quienes
        la tendrían de todos modos, esa relación pasa a ser la primera.
Cautela: condicionar en Z = 1 (resultado posterior) es una descomposición descriptiva, no un efecto causal
puro; Z observa la vida entera, así que quienes iniciaron más recientemente tienen menos exposición
(el efecto fijo de periodo lo absorbe porque afecta igual a tratados y controles).
Se corre en toda la muestra, en LGBTI+ y en OSIG normativa (placebo).
Para cada resultado y población se guarda además el ESTUDIO DE EVENTOS completo y la prueba de Wald de
tendencias previas: el efecto sobre Z solo es creíble si θ(e) previo a la ley es plano.

Entradas : base_analitica.csv.gz
Salidas  : resultados/tablas/mecanismos_{resumen,evento}.csv ; figuras/mecanismos.png ; figuras/mecanismos_evento.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as C
from did_cs import Espec, estimar_cs, fila_resultado, generar_xi
from graficos import grafica_bosque, grafica_evento_panel


def _corto(txt: str) -> str:
    """Etiquetas breves para los paneles de la figura."""
    if txt.startswith("Y:"):
        return "Y (primera pareja)"
    if txt.startswith("Z:"):
        return "Z (alguna vez)"
    if txt.startswith("Y entre"):
        return "Y | Z=1 (se adelanta)"
    return txt.replace("OSIG normativa (placebo)", "Normativa (placebo)")


def main(B: int):
    df = pd.read_csv(C.BASE_ANALITICA, dtype={"FOLIO": str, "VIV_SEL": str, "HOGAR": str, "N_REN": str})
    m = df[df["muestra_principal"] == 1].reset_index(drop=True)
    XI = generar_xi(B)
    pobl = [("Toda la muestra", m), ("LGBTI+", m[m["grupo_osig"] == "LGBTI+"]),
            ("OSIG normativa (placebo)", m[m["grupo_osig"] == "normativa"])]
    filas, eventos, paneles = [], [], []

    def correr(base, nombre_p, resultado, esp):
        res = estimar_cs(base, esp, B=B, XI=XI, n_jobs=C.N_JOBS)
        filas.append(fila_resultado(res, poblacion=nombre_p, resultado=resultado))
        if res is not None:
            ev = res.es.copy()
            ev.insert(0, "resultado", resultado)
            ev.insert(0, "poblacion", nombre_p)
            eventos.append(ev)
            w = res.wald.get("p_boot", float("nan")) if res.wald else float("nan")
            paneles.append((f"{_corto(nombre_p)} · {_corto(resultado)}\nWald tendencias previas p = {w:.2f}", res.es))
        else:
            paneles.append((f"{_corto(nombre_p)} · {_corto(resultado)}", None))

    for nombre_p, base in pobl:
        mo = C.MIN_OBS_CELDA if nombre_p == "Toda la muestra" else C.MIN_OBS_CELDA_SUBMUESTRAS
        correr(base, nombre_p, "Y: primera pareja del mismo sexo (principal)",
               Espec(nombre=f"Y primera pareja | {nombre_p}", min_obs=mo))
        correr(base, nombre_p, "Z: alguna vez pareja del mismo sexo ('se cambia')",
               Espec(y="alguna_vez_mismo_sexo", nombre=f"Z alguna vez | {nombre_p}", min_obs=mo))
        sub = base[base["alguna_vez_mismo_sexo"] == 1]
        correr(sub, nombre_p, "Y entre quienes alguna vez tuvieron pareja del mismo sexo ('se adelanta')",
               Espec(nombre=f"Y | Z=1 | {nombre_p}", min_obs=C.MIN_OBS_CELDA_SUBMUESTRAS))
    t = pd.DataFrame(filas)
    C.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(C.DIR_TABLAS / "mecanismos_resumen.csv", index=False)
    if eventos:
        pd.concat(eventos).round(4).to_csv(C.DIR_TABLAS / "mecanismos_evento.csv", index=False)
    grafica_evento_panel(paneles, C.DIR_FIGURAS / "mecanismos_evento.png",
                         "Mecanismos: estudio de eventos por resultado y población (pp)")
    print("\n== Mecanismos (efecto promedio posterior, pp; wald_p_boot = tendencias previas)")
    print(t[["poblacion", "resultado", "n", "est_pp", "se_pp", "lo_pp", "hi_pp", "p", "wald_p_boot"]].round(3).to_string(index=False))
    t["especificacion"] = t["poblacion"].str.slice(0, 18) + " · " + t["resultado"].str.slice(0, 22)
    grafica_bosque(t, C.DIR_FIGURAS / "mecanismos.png", "Mecanismos: ¿se adelanta o se cambia?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=C.B_BOOT_SECUNDARIO)
    main(ap.parse_args().B)
