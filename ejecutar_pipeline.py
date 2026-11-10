"""
ejecutar_pipeline.py — Corre todo el estudio en orden
======================================================
    python ejecutar_pipeline.py             # análisis completo (B = 999 principal, 299 secundarias)
    python ejecutar_pipeline.py --rapido    # prueba de humo (B = 49 / 29)
    python ejecutar_pipeline.py --desde 3   # reanuda desde el paso 3
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PASOS = [
    ("01_construir_base.py", "Construcción de la base analítica"),
    ("02_diagnosticos.py", "Validaciones y diagnóstico de la muestra"),
    ("03_did_callaway_santanna.py", "Modelo principal (Callaway y Sant'Anna)"),
    ("04_twfe_referencia.py", "TWFE de referencia"),
    ("05_heterogeneidad.py", "Heterogeneidad"),
    ("06_robustez.py", "Robustez"),
    ("07_rambachan_roth.py", "Sensibilidad Rambachan-Roth"),
    ("08_mecanismos_contrafactuales.py", "Mecanismos (se adelanta / se cambia)"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--desde", type=int, default=1)
    a = ap.parse_args()
    env = dict(os.environ)
    if a.rapido:
        env["ENDISEG_B_BOOT"], env["ENDISEG_B_SEC"] = "49", "29"
    for i, (script, titulo) in enumerate(PASOS, start=1):
        if i < a.desde:
            continue
        print(f"\n{'=' * 78}\nPASO {i}: {titulo}  ({script})\n{'=' * 78}", flush=True)
        r = subprocess.run([sys.executable, str(RAIZ / "src" / script)], env=env, cwd=RAIZ)
        if r.returncode != 0:
            sys.exit(f"El paso {i} ({script}) falló. Corrige y reanuda con --desde {i}")
    print("\nListo. Resultados en resultados/tablas y resultados/figuras")
