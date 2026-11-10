# Matrimonio igualitario e inicio de vida sexual — ENDISEG 2021

Pipeline en Python para estimar si la legalización del matrimonio igualitario, que ocurrió en años distintos según la entidad, cambió la probabilidad de que la **primera pareja sexual sea del mismo sexo**. Usa los microdatos de la ENDISEG 2021 (INEGI) y un diseño de **diferencias en diferencias escalonado** (Callaway y Sant'Anna, 2021).

> **Estado del repo (importante).**
> - Los **ocho pasos corren de principio a fin** con un `TMODULO` **sintético** (claves, edades, estratos, UPM y ponderadores reales de tus bases; respuestas simuladas con un efecto conocido).
> - **Validado con datos sintéticos:**
>   - El estimador puntual es insesgado (Monte Carlo, 60 réplicas, `pruebas/validacion_monte_carlo.py`): sesgo de −0.19 pp con periodos de 3 años (efecto verdadero 12 pp) y de +0.13 pp en el placebo.
>   - El error estándar del bootstrap con `INFERENCIA = "ambos"` queda bien calibrado frente a la dispersión real entre réplicas (placebo: 1.49 real vs. 1.69; LGBTI+: 16.1 real vs. unos 15–16).
>   - Con `INFERENCIA = "entidad"` (solo por entidad) la prueba de tendencias previas rechazó con p ≈ 0.03 en datos sin violación; con `"ambos"` da p = 0.77.
> - **Nunca se ha corrido con el `TMODULO` real.** Las definiciones de LGBTI+, la cobertura de P8_2/P7_8 y la tabla de legalización siguen pendientes de validar con tus datos (ver [Limitaciones y pendientes](#7-limitaciones-y-pendientes)).
> - **Poder estadístico bajo.** Con tamaños de muestra como los de la prueba, el efecto global tiene un error estándar de unos 2.5 pp y el de LGBTI+ de unos 16 pp (8.6 pp con periodos de 3 años). Detalle en la sección 7.

---

## 1. Estructura del repositorio

```
ENDISEG_Matrimonio/
├── README.md
├── requirements.txt
├── config.py                          # TODAS las decisiones del investigador
├── ejecutar_pipeline.py               # corre los pasos 01–11 en orden
├── datos/
│   ├── legalizacion_entidades.csv     # año de legalización por entidad (VERIFICAR)
│   ├── checklist_verificacion_fechas.csv  # plantilla para verificar cada fecha con la gaceta oficial
│   └── procesados/                    # (se crea) base_analitica.csv.gz y remuestras
├── documentacion/
│   └── endiseg_bd/                    # AQUÍ van los CSV de INEGI (ver sección 2)
├── src/
│   ├── utils_encuesta.py              # lectura, WLS/logit ponderados, diseño muestral, índice SES
│   ├── did_cs.py                      # estimador Callaway–Sant'Anna, bootstrap, TWFE
│   ├── honest_rr.py                   # cotas de Rambachan–Roth (programación lineal)
│   ├── graficos.py                    # figuras
│   ├── 01_construir_base.py
│   ├── 02_diagnosticos.py
│   ├── 03_did_callaway_santanna.py
│   ├── 04_twfe_referencia.py
│   ├── 05_heterogeneidad.py
│   ├── 06_robustez.py
│   ├── 07_rambachan_roth.py
│   ├── 08_mecanismos_contrafactuales.py
│   ├── 09_diagnostico_entidades.py    # leave-one-out por entidad y por grupo; entidades con amparo temprano
│   ├── 10_multiplicidad.py            # p-valores ajustados (Holm, Benjamini–Hochberg) por familia de pruebas
│   ├── 11_tablas_figuras.py           # paquete de tablas y figuras para la presentación
│   └── multiplicidad.py               # funciones holm() y bh()
├── r/
│   └── honestdid.R                    # sensibilidad Rambachan–Roth con el paquete oficial HonestDiD
├── pruebas/
│   ├── generar_tmodulo_sintetico.py   # datos de prueba con efecto conocido
│   └── validacion_monte_carlo.py      # ¿el estimador recupera el efecto verdadero? (sesgo y dispersión)
└── resultados/                        # (se crea)
    ├── tablas/                        # CSV
    ├── figuras/                       # PNG + SVG
    └── presentacion/                  # tablas formateadas (T01…), figuras (F01…), tablas_presentacion.xlsx e índice
```

## 2. Datos de entrada

Coloca los CSV de INEGI dentro de `documentacion/endiseg_bd/` (se buscan recursivamente, sin importar mayúsculas ni subcarpetas). Cada archivo se localiza por un fragmento de su nombre (`PATRONES_ARCHIVO` en `config.py`):

| Tabla | Fragmento en el nombre | Obligatoria | Para qué |
|---|---|---|---|
| `TMODULO` | `tmodulo` | **Sí** | Persona elegida de cada hogar: sexo al nacer (P7_1), primera relación (P7_5), sexo de la pareja (P7_6_*), orientación (P8_1), identidad (P9_1), entidad, diseño y FACTOR. **Es la base del estudio.** |
| `TSDEM` | `tsdem` | **Sí** | Residentes del hogar: edad de respaldo. |
| `TVIVIENDA` | `tvivienda` | **Sí** | Características y bienes de la vivienda para el índice socioeconómico. |
| `TAPART_A`, `TAPART_B` | `tapart_a`, `tapart_b` | No | Solo para validar que el FACTOR de `TMODULO` coincida (paso 02). No entran al análisis. |

Ejemplo: `conjunto_de_datos_tmodulo_endiseg_2021.csv`.

**Relación entre tablas** (Modelo Relacional de INEGI, que ya tienes):

```
TVIVIENDA (FOLIO, VIV_SEL)  ──1:N──  TSDEM (FOLIO, VIV_SEL, HOGAR, N_REN)  ──1:1──  TMODULO (mismas 4 llaves)
```

Las llaves se leen como **texto** para no perder ceros a la izquierda. Los vacíos quedan como NaN.

**Columnas que lee de `TMODULO`:** `FOLIO, VIV_SEL, HOGAR, N_REN, P4_1, P7_1, P7_1A, P7_3, P7_4_1, P7_4_2, P7_4_3, P7_5, P7_6_1, P7_6_2, P7_6_3, P7_8, P8_1, P8_2, P9_1, FILTRO_9_2, ENT, EST_DIS, UPM_DIS, FACTOR`. Si falta una obligatoria, el paso 01 se detiene y dice cuál.

**Dónde viven las variables de diseño:**
- `EST_DIS` (estrato), `UPM_DIS` (unidad primaria de muestreo) y `FACTOR` (factor de expansión) aparecen en todas las tablas.
- Para variables de `TMODULO` se usa **solo el `FACTOR` de `TMODULO`**. No se mezcla con el de `TSDEM` ni `TVIVIENDA`.
- `ENT` coincide con los dos primeros dígitos de `FOLIO`. Esto se verificó con tus bases reales (0 discrepancias).

## 3. Cómo correrlo

```bash
pip install -r requirements.txt          # numpy, pandas, scipy, matplotlib; Python ≥ 3.10

python ejecutar_pipeline.py --rapido     # prueba de humo (B = 49 / 29 remuestras)
python ejecutar_pipeline.py              # análisis completo (B = 999 / 299)
python ejecutar_pipeline.py --desde 3    # reanuda desde el paso 3
```

O cada paso por separado, en este orden (cada uno lee lo que escribió el anterior):

```bash
python src/01_construir_base.py
python src/02_diagnosticos.py
python src/03_did_callaway_santanna.py [--B 999]
python src/04_twfe_referencia.py       [--B 999]
python src/05_heterogeneidad.py        [--B 299]
python src/06_robustez.py              [--B 299]
python src/07_rambachan_roth.py        [--k_post 4]
python src/08_mecanismos_contrafactuales.py [--B 299]
python src/09_diagnostico_entidades.py
python src/10_multiplicidad.py
python src/11_tablas_figuras.py
Rscript r/honestdid.R                  # opcional; requiere R y el paquete HonestDiD
```

**Rutas alternativas** (por ejemplo, para probar con datos sintéticos sin mezclar salidas):

```bash
export ENDISEG_DIR_DATOS=/ruta/a/los/csv
export ENDISEG_SALIDAS=/ruta/de/salidas
```

**Prueba con datos sintéticos** (recomendada antes de usar los reales):

```bash
python pruebas/generar_tmodulo_sintetico.py --origen documentacion/endiseg_bd --destino /ruta/sintetica
# la carpeta de destino debe contener también TSDEM, TVIVIENDA y TAPART_A/B (enlaces simbólicos sirven)
export ENDISEG_DIR_DATOS=/ruta/sintetica ENDISEG_SALIDAS=/ruta/salidas_sinteticas
python ejecutar_pipeline.py --rapido
```
El generador planta un efecto de +12 pp en personas con orientación LGB+ y 0 en heterosexuales, y una tendencia nacional común. El pipeline debe recuperar algo cercano a eso en la submuestra LGBTI+ (paso 05) y cercano a 0 en el placebo.

## 4. Qué hace cada paso y cómo se relaciona con la metodología

El mapeo siguiente es la lectura que hice del PDF de metodología. Si algo no coincide con lo que querías, dímelo y lo ajusto.

| Idea de la metodología | Dónde está |
|---|---|
| Población: 15 años o más con primera relación sexual válida; se excluyen "otro", "no recuerda", "no especificado" | `01` |
| Resultado Y = 1 si la primera pareja fue del mismo sexo (sexo al nacer vs. sexo de la pareja) | `01` |
| Tratamiento: año de legalización de la entidad de residencia; "periodo" = año de inicio sexual | `01`, `datos/legalizacion_entidades.csv` |
| Controles: cohorte de nacimiento, sexo, índice socioeconómico; la orientación NO es control | `01`, `config.COVARIABLES` |
| Personas trans analizadas aparte | `01` (se excluyen del principal), `06` (sensibilidad) |
| Modelo principal Callaway–Sant'Anna, controles "aún no tratados", base g−1 | `03`, `did_cs.py` |
| Estudio de eventos, efecto global y por plazo | `03` |
| Prueba de tendencias paralelas previas | `03` (Wald) |
| Inferencia con bootstrap multiplicador por entidad | `did_cs.py` |
| TWFE solo como referencia | `04` |
| Heterogeneidad (orientación, sexo, nivel socioeconómico, cohorte/plazo; momento de darse cuenta) | `05` |
| Robustez (ventanas, ±1 año, amparos tempranos, resultado alternativo) | `06` |
| Placebo con heterosexuales | `06` (spec 8), `08` |
| Sensibilidad a violaciones de tendencias paralelas (Rambachan–Roth) | `07`, `honest_rr.py` |
| Contrafactuales: se adelanta vs. se cambia | `08` (incluye estudio de eventos y Wald de tendencias previas de cada resultado) |
| Efecto por año calendario de debut (con IC) | `03` → `cs_anio_calendario.csv`, `cs_anio.png` |
| ¿Depende el resultado de unas pocas entidades? (amparos tempranos) | `09` |
| Muchas pruebas → algunos p < 0.05 por azar | `10` (columnas `p_holm`, `p_bh`) |
| Tablas y figuras de presentación | `11` |

### Paso 01 — `01_construir_base.py` (base analítica)

**Entra:** `TMODULO`, `TSDEM`, `TVIVIENDA`, `datos/legalizacion_entidades.csv`.
**Sale:** `datos/procesados/base_analitica.csv.gz`, `resultados/tablas/flujo_muestra.csv`, `resultados/tablas/ses_cargas.csv`.

1. Une las tres tablas (valida 1:1 y muchos:1), revisa que `EST_DIS`/`UPM_DIS` coincidan, que `ENT` = `FOLIO[:2]` y que las UPM no crucen entidades.
2. **Edad y cohorte:** `anio_nac = 2021 − edad` (error de ±1 año, porque la edad es en años cumplidos a la fecha de entrevista y el campo fue del 23-ago-2021 al 21-ene-2022). `cohorte_c = (anio_nac − 1980)/10`.
3. **Orientación e identidad:**
   - `P8_1`: 1 lesbiana, 2 gay, 3 bisexual, 4 y 5 heterosexual, 6 otra.
   - Trans/no binaria = `FILTRO_9_2 == 1` o `P9_1 ∈ {3,4,5}`.
   - OSIG LGBTI+ = orientación LGB+ o identidad no normativa (`v1`), o además variación corporal (`v2`). El paso 02 te dice cuál reproduce los totales de INEGI.
4. **Inicio sexual:** `edad_debut = P7_5` (válida 1–84; 00 = nunca; 98/99 se excluyen). `anio_inicio = anio_nac + edad_debut`.
5. **Resultado Y** (`Y_mismo_sexo`): se toma `P7_6_1` (hombre) y `P7_6_2` (mujer) y el sexo al nacer. Es válido solo si exactamente una de las dos parejas está marcada y no es "otro". Y = 1 si el sexo de la pareja coincide con el sexo al nacer.
6. **Otras variables:**
   - `Y_erotico_mismo_sexo` (primer encuentro erótico, `P7_3`/`P7_4_*`).
   - `alguna_vez_mismo_sexo` (`P7_8`).
   - `consciente_al_debut` (`P8_2`: si ya sabía su orientación al debutar).
7. **Índice socioeconómico:** primer componente principal ponderado (estilo Filmer–Pritchett) de piso, hacinamiento, agua, drenaje, baño, luz y siete bienes. Se guardan los terciles.
8. **Tratamiento:** une `G` (año de legalización) por entidad. Si la fecha principal cae en o después del 2021-12-01, la entidad queda como "no tratada" (`ANIO_NO_TRATADO = 9999`): el campo terminó el 21-ene-2022 y casi no hay entrevistas posteriores a esa fecha.
9. **Flujo de muestra:** `flujo_muestra.csv` cuenta cuántas personas se pierden en cada exclusión. `muestra_principal` = válidos, sin trans, con covariables completas. `muestra_con_trans` los incluye.

### Paso 02 — `02_diagnosticos.py` (validaciones antes de estimar)

**Sale:** `validacion_tabulados.csv`, `descriptivos_Y.csv`, `grupos_legalizacion.csv`, `conteos_grupo_anio.csv`, `celdas_omitidas.csv`, `figuras/tasas_por_grupo.png`.

1. Compara con los tabulados de INEGI: total 44,189; OSIG normativa 41,827; OSIG LGBTI+ 2,362. Es tu validación de que las definiciones de orientación e identidad están bien.
2. Con `TAPART_A/B`, verifica que el `FACTOR` de `TMODULO` sea el mismo para las mismas personas.
3. Resume las variables de diseño (estratos, UPM, entidades, rango de FACTOR).
4. Descriptivos de Y con error estándar **por linealización** usando estrato, UPM y FACTOR.
5. Cuenta personas y casos con Y = 1 por grupo de legalización y año.
6. Lista qué celdas ATT(g,t) son estimables con `MIN_OBS_CELDA`. **Si un grupo se queda sin celdas, lo avisa**; la solución es subir `ANIOS_POR_PERIODO` a 2 o 3.

### Paso 03 — `03_did_callaway_santanna.py` (modelo principal)

**Sale:** `cs_att_gt.csv`, `cs_evento.csv`, `cs_resumen.csv`, `cs_anio_calendario.csv`, `cs_wald_pretendencias.csv`, `cs_celdas_omitidas.csv`, `figuras/cs_evento.png`, `figuras/cs_anio.png`, `datos/procesados/cs_boot_evento.csv`.

- **ATT(g,t):** efecto en el grupo que legalizó en el año g, medido en el año de inicio t. El grupo g se compara solo con entidades aún no tratadas (G > máx(t, g−1)), con periodo base universal g−1.
- **Estimador:** doblemente robusto para secciones transversales repetidas (Sant'Anna y Zhao 2020, equivalente a `DRDID::drdid_rc1`), con ponderadores FACTOR. Combina:
  - reponderación por propensity score logístico en X = (cohorte, sexo, índice socioeconómico), y
  - regresión de resultado en los controles, por periodo.
- **Agregación:** `θ(e) = Σ_g w_g ATT(g, g+e)`, con w_g proporcional a las personas tratadas que aporta cada grupo. Salen:
  - el efecto global posterior a la ley (`global_post`),
  - corto plazo (e = 0–2), mediano (3–5) y largo (6+) y por grupo,
  - e = −1 normalizado a 0.
- **Prueba de Wald** conjunta de θ(e) = 0 para e < −1, con p-valor χ² y p-valor bootstrap.
- **Banda sup-t** uniforme para el estudio de eventos.
- **Año calendario:** además de agregar por e = t − g, se agrega por el año calendario de inicio sexual t (solo celdas posteriores a la ley), con IC puntual.

### Paso 04 — `04_twfe_referencia.py`

`Y = α_entidad + λ_año + β·D + X'γ`, por mínimos cuadrados ponderados. Errores agrupados por entidad (CR1) y por bootstrap. Escribe `twfe_vs_cs.csv` para compararlo con el efecto global del paso 03. Es solo referencia.

### Paso 05 — `05_heterogeneidad.py`

Corre el mismo estimador **dentro de cada submuestra**, con las mismas remuestras que el principal, lo que permite comparar grupos con remuestras pareadas. Dimensiones:
- orientación (normativa / LGBTI+; heterosexual / gay-lesbiana / bisexual),
- sexo,
- terciles socioeconómicos,
- cohorte de nacimiento,
- LGBTI+ según ya supiera su orientación al debutar o no.

Omite submuestras de menos de 150 personas. Sale: `heterogeneidad_resumen.csv`, `heterogeneidad_diferencias.csv`, `figuras/heterogeneidad.png`.

### Paso 06 — `06_robustez.py`

17 especificaciones con las mismas remuestras. Sale: `robustez_resumen.csv`, `figuras/robustez.png`.
- Ventanas de ±5, ±8 y pre 15; periodos de 2 y 3 años.
- Inicio sexual −1 y +1 año.
- Sin entidades con amparos tempranos; sin el grupo 2021.
- Resultado alternativo: primer encuentro erótico.
- Tratamiento = año de aprobación, o año de entrada en vigor formal.
- Sin covariables.
- Incluir trans; solo trans.
- Placebo con OSIG normativa.

### Paso 07 — `07_rambachan_roth.py`

Mide cuánta violación de tendencias paralelas toleraría el resultado, bajo suavidad (Δ^SD(M)) y magnitudes relativas (Δ^RM(M̄)), con punto de quiebre. Escribe `rr_sd.csv`, `rr_rm.csv`, `rr_quiebre.csv` y los archivos `honestdid_betahat.csv` / `honestdid_sigma.csv`.

> **Esto NO es el intervalo condicional / FLCI del paquete R `HonestDiD`.** Es el conjunto identificado por programación lineal más ±1.96·EE (aproximación). Para el artículo final, usa `HonestDiD` con los archivos exportados (`numPrePeriods` = número de coeficientes pre; `numPostPeriods` = `k_post`).

### Paso 08 — `08_mecanismos_contrafactuales.py`

Mi operacionalización de los dos contrafactuales (confírmala):
- **"Se cambia":** Z = alguna vez tuvo pareja del mismo sexo (`P7_8`), en toda la muestra.
- **"Se adelanta":** Y solo entre quienes tienen Z = 1.

Se corre para toda la muestra, LGBTI+ y OSIG normativa (placebo). Condicionar en Z es descriptivo, no causal.
Para cada resultado y población guarda el estudio de eventos (`mecanismos_evento.csv`, `figuras/mecanismos_evento.png`) y el p-valor de la prueba de Wald de tendencias previas: un efecto sobre Z solo es creíble si los coeficientes previos a la ley son planos.

### Paso 09 — `09_diagnostico_entidades.py`
Descompone de qué entidades depende el efecto promedio: describe las entidades con amparo temprano (`diag_amparos.csv`), recalcula el efecto dejando fuera una entidad a la vez (`diag_loo_entidad.csv`, `figuras/loo_entidades.png`), un grupo de legalización a la vez (`diag_loo_grupo.csv`) y las entidades de amparo temprano una por una y todas juntas (`diag_excluir_amparos.csv`). Solo estimaciones puntuales (sin bootstrap): sirve para ver el cambio, no para inferir.

### Paso 10 — `10_multiplicidad.py`
Agrega `p_holm` (controla la probabilidad de al menos un falso positivo) y `p_bh` (controla la proporción de falsos positivos entre los rechazos) a las tablas de grupos, heterogeneidad, diferencias, robustez, mecanismos y coeficientes previos, dentro de cada familia. Escribe `*_ajustado.csv` y reporta cuántos p < 0.05 sobreviven. **Reporta el p ajustado junto al p original.**

### Paso 11 — `11_tablas_figuras.py`
No estima nada: lee las salidas y arma `resultados/presentacion/` con tablas T01…T10 (formato "estimado (EE)", IC, p, p ajustado; CSV, un libro `.xlsx` y `tablas.md`), figuras F01…F11 (PNG a 200 dpi y SVG, proporción 16:9) y un `indice_presentacion.md` que dice qué sostiene cada una. Si falta alguna salida (porque no corriste ese paso), la omite.

### Sensibilidad con el paquete oficial — `r/honestdid.R`
Lee `honestdid_betahat.csv` y `honestdid_sigma.csv` (los exporta el paso 07) y corre `HonestDiD` (magnitudes relativas y suavidad) para el efecto inmediato y el promedio posterior. Escribe `honestdid_{rm,sd}_*.csv` y las figuras correspondientes. Es la versión que debes citar en el artículo; el paso 07 es una aproximación.

## 5. Parámetros que puedes cambiar (`config.py`)

| Parámetro | Valor por defecto | Significado |
|---|---|---|
| `VENTANA_PRE`, `VENTANA_POST` | 10, 10 | Años antes y después de la ley en el estudio de eventos |
| `ANIOS_POR_PERIODO` | 1 | Agrupa años si las celdas son chicas |
| `MIN_OBS_CELDA` | 10 | Mínimo de personas por celda para estimar ATT(g,t) |
| `COVARIABLES` | cohorte, sexo, SES | X del estimador |
| `DEF_LGBTI` | `v1` | Definición de LGBTI+ (validar en paso 02) |
| `FECHA_LIMITE_TRATAMIENTO` | 2021-12-01 | Después de esta fecha, la entidad se considera no tratada |
| `B_BOOT`, `B_BOOT_SECUNDARIO` | 999, 299 | Remuestras del bootstrap |
| `INFERENCIA` | `ambos` | `entidad`, `upm` o `ambos` |
| `NIVEL_CONFIANZA` | 0.95 | |
| `DESPLAZAMIENTO_ANIO_INICIO` | 0 | Desplazamiento del año de inicio |

Todas las estimaciones se reportan en **puntos porcentuales**.

## 6. Tabla de legalización (`datos/legalizacion_entidades.csv`)

| Columna | Qué es |
|---|---|
| `fecha_principal` | Fecha en que el Registro Civil ya podía casar a parejas del mismo sexo sin amparo; define `G` |
| `fecha_formal_vigor`, `fecha_aprobacion` | Para las pruebas de robustez 5a y 5b |
| `amparo_temprano`, `anio_amparo_temprano` | Para la robustez 3a |
| `discrepancia_fuentes` | 1 donde las fuentes no coinciden |

**Verifícala contra los periódicos oficiales estatales antes de publicar.** Las fechas salen de Wikipedia, Gaceta UNAM y notas de prensa, no de fuentes oficiales. Seis entidades tienen discrepancia entre fuentes: Baja California, Chiapas, Jalisco, Oaxaca, Puebla y Yucatán. En varios casos el año de amparos tempranos no está precisado.

Con estos datos, 21 entidades quedan tratadas en 10 grupos (2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021) y 11 como aún no tratadas durante el campo.

## 7. Limitaciones y pendientes

1. **Inferencia.**
   - Si un grupo de legalización es una sola entidad (CDMX, Quintana Roo), multiplicar todos sus pesos por una constante no cambia su media ponderada. Por eso el bootstrap solo por entidad no ve su ruido muestral y subestima el error (en la prueba, el error del efecto global fue 0.9 pp con `"entidad"` y 2.4 pp con `"ambos"`).
   - Por eso el valor por defecto es `INFERENCIA = "ambos"` (entidad + UPM), que se validó con datos sintéticos.
   - Aun así, con un solo clúster tratado la inferencia a nivel entidad es frágil; no se puede salir de esa limitación con estos datos.
2. **Poder estadístico (la limitación más importante).**
   - En la prueba, el efecto global tuvo un error estándar de unos 2.5 pp, mientras que un efecto plantado de 12 pp sobre las personas LGB+ equivale a menos de 1 pp sobre toda la muestra: **no se distingue de cero**.
   - En LGBTI+ (unos 2,600 casos) solo 27 de 147 celdas ATT(g,t) son estimables con `MIN_OBS_CELDA_SUBMUESTRAS = 5`, y el error estándar es de unos 16 pp. Con periodos de 3 años baja a unos 8.6 pp.
   - En subgrupos más finos (gay/lesbiana, "ya sabía su orientación al debut") las estimaciones son inestables (por ejemplo, 77 pp con error de 35 pp en la prueba). **No las interpretes.**
   - Recomendación: para LGBTI+ y subgrupos usa `ANIOS_POR_PERIODO = 3`, y trata el modelo de toda la muestra como el análisis de referencia.
3. **Las cohortes anteriores a 1990 no tienen periodo posterior a la ley.** Debutaron casi todas antes de 2010 (primera legalización), así que solo aportan periodos previos y las submuestras por cohorte quedan vacías. Las cohortes que identifican el efecto son las nacidas desde 1990 (la heterogeneidad usa <1990, 1990-94, 1995-99, 2000-04 y 2005+).
4. **Año de inicio aproximado.** `anio_nac = 2021 − edad` (±1 año). El margen se prueba en la robustez 2a/2b.
5. **Entidad de residencia ≠ entidad del debut.** Quien migró queda mal asignado.
6. **Jurisprudencia nacional de 2015** abrió el amparo en todo el país. Es un choque común que absorbe el efecto fijo de año, pero matiza qué significa "legalización".
7. **Sun y Abraham** no está implementado (solo Callaway–Sant'Anna y TWFE). Si lo quieres como estimador secundario, hay que escribirlo.
8. **Rambachan–Roth** es aproximado (ver paso 07); para el artículo usa `r/honestdid.R`.
8b. **Fechas de legalización sin verificar.** Usa `datos/checklist_verificacion_fechas.csv` (seis entidades con discrepancia entre fuentes van primero). Si cambias una fecha, vuelve a correr desde el paso 01.
8c. **Comparaciones múltiples.** Hay ~60 pruebas entre grupos, submuestras, robustez y mecanismos. Un p < 0.05 aislado no es evidencia; mira `p_holm` y `p_bh` (paso 10).
8d. **Personas trans/no binarias solas** (robustez 7b): con ~335 personas no hay celdas estimables. Se reporta como limitación.
9. **Definición de LGBTI+ y cobertura de P8_2/P7_8:** se validan en el paso 02 cuando corras con el `TMODULO` real. Hasta entonces son supuestos míos a partir del descriptor de archivos.

## 8. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `No encontré ningún .csv con 'tmodulo'…` | Falta `TMODULO` o está fuera de `DIR_DATOS`; revisa la ruta y el nombre del archivo |
| `faltan columnas obligatorias` | El CSV no trae alguna columna listada en la sección 2 |
| Grupos "sin ninguna celda estimable" (02) | Sube `ANIOS_POR_PERIODO` a 2–3 o baja `MIN_OBS_CELDA` |
| Pasos 05–08 muy lentos | Baja `B_BOOT_SECUNDARIO`; con `N_JOBS > 1` usa varios procesos (solo en Linux/macOS) |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| LGBTI+ no coincide con 2,362 (02) | Prueba `DEF_LGBTI = "v2"` o revisa la construcción de orientación e identidad |

## 9. Referencias

- Callaway, B. y Sant'Anna, P. (2021). Difference-in-Differences with Multiple Time Periods. *Journal of Econometrics*.
- Sant'Anna, P. y Zhao, J. (2020). Doubly Robust Difference-in-Differences Estimators. *Journal of Econometrics*.
- Rambachan, A. y Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*.
- Filmer, D. y Pritchett, L. (2001). Estimating Wealth Effects without Expenditure Data. *Demography*.
- INEGI. ENDISEG 2021: Descriptor de archivos, Modelo Lógico y Modelo Relacional.
