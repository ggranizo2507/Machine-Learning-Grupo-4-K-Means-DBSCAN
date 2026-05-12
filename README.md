# 🔥 Developer Burnout — Perfiles Latentes con K-Means & DBSCAN

> **Aprendizaje No Supervisado | Detección de Anomalías**  
> Universidad UEES · Machine Learning · Grupo 4 · Semana 3

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-7%2C000%20registros-informational)
![License](https://img.shields.io/badge/Licencia-MIT-green)

</div>

---

## 🎯 Pregunta central del análisis

> **"Queremos descubrir perfiles de burnout de desarrolladores sin decirle al modelo quién tiene burnout alto, medio o bajo."**

El objetivo es descubrir **perfiles latentes** a partir de las *causas* del burnout (horas de trabajo, carga operativa, índice de recuperación), **no** de la etiqueta resultante. Por eso `burnout_level` y `stress_level` quedan fuera del modelo durante todo el entrenamiento, y solo se usan al final como **validación externa**.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Dataset](#-dataset)
- [Pipeline del Análisis](#-pipeline-del-análisis)
- [Modelos Aplicados](#-modelos-aplicados)
- [Resultados Principales](#-resultados-principales)
- [Instalación y Uso](#-instalación-y-uso)
- [Wiki](#-wiki)
- [Créditos](#-créditos)

---

## 📌 Descripción del Proyecto

Este notebook aplica **aprendizaje no supervisado** sobre un dataset de 7,000 desarrolladores de software para descubrir perfiles de riesgo de burnout sin utilizar ninguna etiqueta supervisada.

### ¿Qué hace este análisis, paso a paso?

| Etapa | Técnica | Objetivo |
|:---:|:---|:---|
| 1–4B | EDA + Feature Engineering | Entender datos, eliminar leakage, construir features compuestas sin redundancia |
| 4C | Correlación espacio final | Verificar que r < 0.70 entre todos los pares de variables |
| 5 | Preprocesamiento con split | Imputar y escalar **solo con train** — sin contaminar validación |
| 6 | Selección de k | Elegir k óptimo con 4 métricas + ranking combinado |
| 7–8 | **K-Means** | Segmentar perfiles + validar generalización en validación |
| 9 | **DBSCAN** | Clustering por densidad + detectar puntos de ruido |
| 10 | **PCA + t-SNE** | Visualizar la estructura de los clusters en 2D |
| 11 | **IF + LOF** | Detectar anomalías desde perspectivas complementarias |
| 12–14 | Conclusiones | Comparativa global + hallazgos de negocio + reflexión metodológica |

---

## 🗂️ Estructura del Repositorio

```
developer-burnout-clustering/
│
├── 📓 KMeans_DBSCAN_anomalias_Grupo_4_1205.ipynb   ← Notebook principal
├── 📊 developer_burnout_dataset.csv                 ← Dataset fuente
│
├── 📄 README.md                                     ← Este archivo
│
└── 📚 wiki/
    ├── Home.md                                      ← Portada del Wiki
    ├── 01_Dataset.md                                ← Descripción del dataset
    ├── 02_EDA_y_Preprocesamiento.md                 ← EDA y decisiones de limpieza
    ├── 03_Feature_Engineering.md                    ← Diseño del espacio del modelo
    ├── 04_KMeans.md                                 ← Modelo K-Means detallado
    ├── 05_DBSCAN.md                                 ← Modelo DBSCAN y detección de ruido
    ├── 06_PCA_tSNE.md                               ← Reducción de dimensionalidad
    ├── 07_Anomalias.md                              ← Isolation Forest y LOF
    └── 08_Conclusiones.md                           ← Resultados y reflexión final
```

---

## 📊 Dataset

**Fuente:** [Developer Burnout Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)

| Campo | Tipo | Descripción |
|:---|:---:|:---|
| `age` | float | Edad del desarrollador (años) |
| `experience_years` | float | Años de experiencia en programación |
| `daily_work_hours` | float | Horas de trabajo promedio por día |
| `sleep_hours` | float | Horas de sueño promedio por día |
| `caffeine_intake` | float | Bebidas con cafeína consumidas por día |
| `bugs_per_day` | float | Bugs producidos por día (promedio) |
| `commits_per_day` | float | Commits de código por día |
| `meetings_per_day` | float | Reuniones diarias |
| `screen_time` | float | Horas totales frente a pantalla por día |
| `exercise_hours` | float | Horas de ejercicio físico por día |
| `stress_level` | float | Puntuación de estrés calculada (0–100) |
| `burnout_level` | str | **Target:** Low / Medium / High *(excluida del modelo)* |

> ⚠️ **Nota crítica:** `stress_level` predice `burnout_level` con **99.99% de precisión** mediante una regla de tres umbrales. Incluirla equivale a filtrar el target al modelo. **Ambas variables fueron excluidas del entrenamiento.**

---

## 🔬 Pipeline del Análisis

### Decisiones de exclusión de variables

| Variable | Razón de exclusión |
|:---|:---|
| `burnout_level` | TARGET — etiqueta supervisada, no debe entrar al modelo |
| `stress_level` | Codificación determinista de `burnout_level` (r = 0.91, precisión 99.99%) |
| `age` | Sin poder discriminatorio (r < 0.02 con todo) |
| `experience_years` | Sin poder discriminatorio (r < 0.02 con todo) |
| `screen_time` | Redundante con `daily_work_hours` (r = 0.93) |
| `commits_per_day` | No discrimina burnout en ningún cuartil (r = −0.01) |

### Espacio final del modelo — 5 variables

| Variable | Origen | Fórmula | Rol |
|:---|:---:|:---|:---|
| `caffeine_intake` | 📌 Original | — | Compensación energética |
| `bugs_per_day` | 📌 Original | — | Carga técnica directa |
| `ratio_trabajo_descanso` | 🆕 Nueva | `daily_work_hours / sleep_hours` | Desequilibrio jornada/descanso |
| `carga_operativa` | 🆕 Nueva | `bugs_per_day + meetings_per_day` | Presión técnica + social |
| `indice_recuperacion` | 🆕 Nueva | `sleep_hours + exercise_hours` | Capacidad de recuperación física |

**Impacto del Feature Engineering:**

| Espacio | Silhouette | Davies-Bouldin | Calinski-H |
|:---|:---:|:---:|:---:|
| 7 variables con doble conteo | 0.147 | 2.250 | 1,032 |
| **5 variables sin doble conteo** | **0.252** | **1.541** | **2,131** |

---

## 🤖 Modelos Aplicados

### K-Means — Segmentación de perfiles

- **k óptimo:** k=2, seleccionado por votación de 4 métricas (Silhouette, Davies-Bouldin, Calinski-Harabasz, Codo)
- **Generalización:** diferencia Silhouette train/val < 0.05 ✅
- **Validación:** Test Chi² confirma diferencias significativas (p < 0.05) entre clusters en train **y** validación

| Perfil | Señal principal | Relación con burnout |
|:---|:---|:---:|
| 🔴 **Alto Riesgo — Sobrecargado** | Alto ratio trabajo/descanso, alta carga operativa, bajo índice de recuperación | Mayoría `burnout_level = High` |
| 🟢 **Bajo Riesgo — Equilibrado** | Jornadas acotadas, buena recuperación física, carga moderada | Mayoría `burnout_level = Low` |

### DBSCAN — Detección por densidad

- **Resultado:** 1 cluster denso + ~5% de puntos de ruido
- **Diagnóstico:** el dataset tiene densidad uniforme en 7D (distribución sintética). DBSCAN no segmenta pero identifica **desarrolladores con perfiles estadísticamente atípicos**
- **Uso recomendado:** detector de outliers para intervención individual urgente

### Isolation Forest + LOF — Detección de anomalías

| Método | Escala | Anomalías |
|:---|:---:|:---:|
| Isolation Forest | Global | ~5% del train |
| Local Outlier Factor | Local | ~5% del train |
| **Consenso IF + LOF** | **Ambas** | **~2.5% — máxima confianza** |

### PCA + t-SNE — Reducción de dimensionalidad

- **PCA:** captura ~35% de varianza en 2 componentes (resultado esperado: 7 variables poco correlacionadas)
- **t-SNE:** mayor fidelidad al clustering original (preserva vecindad local)

---

## 📈 Resultados Principales

```
╔══════════════════════════════════════════════════════════════╗
║   HALLAZGO CENTRAL                                          ║
╠══════════════════════════════════════════════════════════════╣
║   Sin usar burnout_level ni stress_level, K-Means           ║
║   identificó dos perfiles con diferencias                   ║
║   estadísticamente significativas en su distribución        ║
║   de burnout real (Chi², p < 0.05 en train y val).         ║
║                                                             ║
║   El modelo descubrió estructura a partir de CAUSAS         ║
║   de comportamiento, no de la etiqueta.                     ║
╚══════════════════════════════════════════════════════════════╝
```

**Aplicación práctica:**

| Modelo | Pregunta que responde | Uso organizacional |
|:---|:---|:---|
| K-Means | ¿A qué perfil pertenece este dev? | Políticas grupales diferenciadas |
| DBSCAN | ¿Quién no encaja en ningún patrón? | Alerta de casos atípicos |
| IF + LOF | ¿Quién es estadísticamente inusual? | Lista de prioridad para RRHH |
| PCA | ¿Qué variables concentran más información? | Reducir encuestas a variables clave |

---

## 🚀 Instalación y Uso

### Opción A — Google Colab (recomendado)

1. Abrir el notebook en Google Colab
2. Montar Google Drive y ajustar `FILE_PATH` en el Paso 2:
   ```python
   FILE_PATH = '/content/drive/My Drive/ruta/developer_burnout_dataset.csv'
   ```
3. Ejecutar todas las celdas en orden (Runtime → Run all)

### Opción B — Entorno local

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/developer-burnout-clustering.git
cd developer-burnout-clustering

# Instalar dependencias
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy missingno

# Abrir el notebook
jupyter notebook KMeans_DBSCAN_anomalias_Grupo_4_1205.ipynb
```

> 💡 **Ajuste de ruta local:** en el Paso 2, reemplazar el bloque de Google Drive:
> ```python
> df_raw = pd.read_csv('developer_burnout_dataset.csv')
> ```

### Dependencias principales

| Librería | Versión mínima | Uso |
|:---|:---:|:---|
| `pandas` | 2.0 | Manipulación de datos |
| `numpy` | 1.24 | Cálculo numérico |
| `scikit-learn` | 1.3 | Modelos ML |
| `matplotlib` / `seaborn` | — | Visualización estática |
| `plotly` | — | Visualización interactiva |
| `missingno` | — | Diagnóstico de NaN |
| `scipy` | — | Test estadísticos |

---

## 📚 Wiki

La documentación extendida está disponible en la carpeta `wiki/`:

| Página | Contenido |
|:---|:---|
| [Home](wiki/Home.md) | Portada y guía de navegación |
| [Dataset](wiki/01_Dataset.md) | Estructura, campos y naturaleza sintética |
| [EDA y Preprocesamiento](wiki/02_EDA_y_Preprocesamiento.md) | Análisis exploratorio, NaN, distribuciones |
| [Feature Engineering](wiki/03_Feature_Engineering.md) | Decisiones de exclusión y variables compuestas |
| [K-Means](wiki/04_KMeans.md) | Selección de k, perfiles, validación |
| [DBSCAN](wiki/05_DBSCAN.md) | Parámetros, maldición dimensionalidad, ruido |
| [PCA y t-SNE](wiki/06_PCA_tSNE.md) | Reducción dimensional, loadings, fidelidad |
| [Anomalías](wiki/07_Anomalias.md) | Isolation Forest, LOF, consenso |
| [Conclusiones](wiki/08_Conclusiones.md) | Hallazgos, limitaciones, reflexión |

---

## 👥 Créditos

| | |
|:---|:---|
| **Institución** | Universidad UEES — Ecuador |
| **Curso** | Machine Learning |
| **Grupo** | Grupo 4 |
| **Semana** | Semana 3 |
| **Dataset** | [Kaggle — Developer Burnout Prediction Dataset](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples) |

---

<div align="center">

*Análisis desarrollado como ejercicio académico de aprendizaje no supervisado*

</div>


## 👥 Autores

**Grupo 4 — Machine Learning**
Universidad de Especialidades Espíritu Santo (UEES)

| # | Nombre |
|---|--------|
| 1 | Eduardo Alejandro Ceballos Jijón |
| 2 | Guillermo Leonidas Granizo Veintimilla |
| 3 | José Farid Ulloa Manzur |
| 4 | Christian Xavier Valle Maridueña |

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.
