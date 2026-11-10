"""
config.py — Parámetros centrales del estudio
=============================================
Matrimonio igualitario e inicio de vida sexual en México:
diferencias en diferencias escalonado con microdatos de la ENDISEG 2021.

TODO lo que sea una decisión del investigador (ventanas, covariables, umbrales de
muestra, definición de LGBTI+, etc.) vive aquí, para que cada script sea reproducible
y para que cambiar una decisión no obligue a tocar el código de estimación.

Rutas: se pueden sobrescribir con variables de entorno
    ENDISEG_DIR_DATOS  carpeta donde están los CSV de INEGI (se busca recursivamente)
    ENDISEG_SALIDAS    carpeta donde se escriben datos procesados y resultados
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. Rutas
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DIR_DATOS = Path(os.environ.get("ENDISEG_DIR_DATOS", RAIZ / "documentacion" / "endiseg_bd"))
DIR_TRABAJO = Path(os.environ.get("ENDISEG_SALIDAS", RAIZ))
DIR_PROC = DIR_TRABAJO / "datos" / "procesados"
DIR_TABLAS = DIR_TRABAJO / "resultados" / "tablas"
DIR_FIGURAS = DIR_TRABAJO / "resultados" / "figuras"
ARCHIVO_LEGALIZACION = RAIZ / "datos" / "legalizacion_entidades.csv"
BASE_ANALITICA = DIR_PROC / "base_analitica.csv.gz"

# Fragmento del nombre de archivo (sin importar mayúsculas) con el que se localiza cada
# tabla de INEGI dentro de DIR_DATOS. Ej.: conjunto_de_datos_tmodulo_endiseg_2021.csv
PATRONES_ARCHIVO = {
    "tmodulo": "tmodulo",
    "tsdem": "tsdem",
    "tvivienda": "tvivienda",
    "tapart_a": "tapart_a",
    "tapart_b": "tapart_b",
}

# ----------------------------------------------------------------------------
# 2. Levantamiento y construcción del "periodo"
# ----------------------------------------------------------------------------
# El levantamiento de la ENDISEG fue del 23-ago-2021 al 21-ene-2022 (comunicado INEGI 492/21).
# EDAD se mide en años cumplidos a la fecha de entrevista, que casi siempre cae en 2021.
# Año de nacimiento aproximado = ANIO_REF - EDAD (error de ±1 año). Esa incertidumbre es la
# que se evalúa en 06_robustez.py con el margen de ±1 año en el inicio sexual.
ANIO_REF = 2021
ANIO_MAX_INICIO = 2021          # no hay inicios sexuales observables después del año de campo
EDAD_MIN = 15                   # población objetivo de la ENDISEG
EDAD_DEBUT_MIN = None           # p. ej. 10 para descartar edades de debut implausibles; None = no excluir

# Desplazamiento (años) aplicado al año de inicio sexual en la estimación principal.
DESPLAZAMIENTO_ANIO_INICIO = 0

# ----------------------------------------------------------------------------
# 3. Tratamiento: año de legalización por entidad (datos/legalizacion_entidades.csv)
# ----------------------------------------------------------------------------
# Una entidad cuya fecha_principal cae en o después de esta fecha se codifica como
# "aún no tratada" (G = ANIO_NO_TRATADO): prácticamente no hay entrevistas posteriores
# al tratamiento (campo termina 21-ene-2022) y tratarla como grupo 2021 solo añadiría ruido.
FECHA_LIMITE_TRATAMIENTO = "2021-12-01"
ANIO_NO_TRATADO = 9999

# ----------------------------------------------------------------------------
# 4. Especificación del estimador (Callaway y Sant'Anna, 2021)
# ----------------------------------------------------------------------------
VENTANA_PRE = 10                # años antes de la ley que entran como periodos de evento (e = -10 … -2)
VENTANA_POST = 10               # años después de la ley (e = 0 … 10); lo limita el año de campo
ANIOS_POR_PERIODO = 1           # 1 = periodo anual (metodología). Subirlo (2-3) agrupa años si las celdas son pequeñas
MIN_OBS_CELDA = 10              # mínimo de observaciones (sin ponderar) por celda grupo×periodo para estimar ATT(g,t)
TRIM_PS = 0.995                 # se descartan controles con propensity score > TRIM_PS (como DRDID)
COVARIABLES = ["cohorte_c", "mujer", "ses_z"]   # X_i: cohorte de nacimiento, sexo, índice socioeconómico

# Resúmenes dinámicos (en años desde la ley)
RANGO_CORTO = (0, 2)
RANGO_MEDIANO = (3, 5)
RANGO_LARGO = (6, 99)

# ----------------------------------------------------------------------------
# 5. Inferencia
# ----------------------------------------------------------------------------
# Bootstrap multiplicador: a cada entidad (32 clústeres) se le asigna un peso aleatorio Exp(1) que
# multiplica los ponderadores de sus personas y se re-estima todo; además, cada UPM recibe otro peso
# Exp(1) independiente para reflejar el muestreo dentro de la entidad (ver did_cs.py).
# INFERENCIA: "ambos" (entidad + UPM, recomendado), "entidad" (solo la metodología a nivel entidad) o "upm".
INFERENCIA = os.environ.get("ENDISEG_INFERENCIA", "ambos")
B_BOOT = int(os.environ.get("ENDISEG_B_BOOT", 999))              # remuestras del análisis principal
B_BOOT_SECUNDARIO = int(os.environ.get("ENDISEG_B_SEC", 299))    # remuestras en heterogeneidad, robustez y mecanismos
NIVEL_CONFIANZA = 0.95
SEMILLA = 20211108
N_JOBS = max(1, (os.cpu_count() or 1) - 1)

# ----------------------------------------------------------------------------
# 6. Definiciones de variables
# ----------------------------------------------------------------------------
# Definición operativa de OSIG LGBTI+ (se valida contra los tabulados de INEGI en 02_diagnosticos.py):
#   "v1": orientación LGB+ (P8_1 ∈ {1,2,3,6}) o identidad de género no normativa
#   "v2": v1 o variación del cuerpo/intersex (P7_1A == 1)
DEF_LGBTI = "v1"

# Totales publicados en los tabulados básicos de INEGI (Tema 6), observaciones muestrales.
TOTALES_TABULADOS = {"total": 44189, "normativa": 41827, "lgbti": 2362}

# Estimación con muestras chicas: si una celda no llega a MIN_OBS_CELDA se omite.
MIN_OBS_CELDA_SUBMUESTRAS = 5
