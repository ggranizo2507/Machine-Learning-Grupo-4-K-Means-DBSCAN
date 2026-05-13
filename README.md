# 🔥 Clasificación de Perfiles de Burnout en Desarrolladores
## Aprendizaje No Supervisado | K-Means · DBSCAN · Detección de Anomalías

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

---

## 🎯 Pregunta central del proyecto

> **"Queremos descubrir perfiles de burnout de desarrolladores *sin* decirle al modelo quién tiene burnout alto, medio o bajo."**

El objetivo es descubrir **perfiles latentes** a partir de las *causas* del burnout —horas de trabajo, carga operativa, calidad del sueño, hábitos de recuperación— y **no** de la etiqueta resultado (`burnout_level`). El modelo aprende patrones de comportamiento; la etiqueta se usa solo al final como validación externa.

---

## 📁 Estructura del repositorio

```
📦 burnout-clustering/
│
├── 📓 KMeans_DBSCAN_anomalias_Grupo_4.ipynb   ← Notebook principal (11 pasos)
├── 📊 developer_burnout_dataset.csv            ← Dataset (7,000 registros · 12 variables)
├── 📄 README.md                                ← Este archivo
│
└── 📚 wiki/
    ├── Home.md                                 ← Índice y guía de navegación
    ├── 01_Dataset.md                           ← Fuente, variables, naturaleza sintética
    ├── 02_Seleccion_Variables_Leakage.md       ← Exclusiones y análisis de leakage
    ├── 03_Feature_Engineering.md               ← Features construidas y espacio final
    ├── 04_Preprocesamiento.md                  ← Split, imputación, escalado adaptativo
    ├── 05_KMeans_Seleccion_k.md                ← 4 métricas, ranking combinado, k=2
    ├── 06_KMeans_Perfiles.md                   ← Centroides, etiquetas, validación Chi²
    ├── 07_DBSCAN.md                            ← Clustering por densidad, maldición dim.
    ├── 08_Reduccion_Dimensional.md             ← PCA vs t-SNE, loadings, fidelidad
    ├── 09_Deteccion_Anomalias.md               ← IF + LOF, consenso, cruce con DBSCAN
    └── 10_Conclusiones.md                      ← Perfiles, comparativa, limitaciones
```

---

## 🗂️ Dataset

**Fuente:** [Developer Burnout Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)
**Registros:** 7,000 · **Variables:** 12 · **NaN:** ~2% por columna (patrón MCAR)

| Variable | Tipo | Descripción |
|---|---|---|
| `age` | Continua | Edad del desarrollador (años) |
| `experience_years` | Continua | Años de experiencia en programación |
| `daily_work_hours` | Continua | Horas promedio de trabajo por día |
| `sleep_hours` | Continua | Horas promedio de sueño diario |
| `caffeine_intake` | Discreta | Bebidas con cafeína consumidas al día |
| `bugs_per_day` | Discreta | Bugs producidos diariamente |
| `commits_per_day` | Discreta | Commits de código por día |
| `meetings_per_day` | Discreta | Reuniones diarias |
| `screen_time` | Continua | Tiempo total frente a pantalla (horas/día) |
| `exercise_hours` | Continua | Tiempo de ejercicio físico diario (horas) |
| `stress_level` | Continua | Puntuación de estrés calculada (0–100) |
| `burnout_level` | Categórica | **TARGET:** Low / Medium / High *(excluida del modelo)* |

> ⚠️ El dataset presenta señales claras de ser sintético: skewness ≈ 0 en todas las variables, cero outliers IQR y NaN distribuidos uniformemente al 2% por columna. Esto no invalida el análisis metodológico pero implica que las fronteras de cluster son más difusas que en datos reales de campo.

---

## 🔬 Pipeline del análisis (11 pasos)

```
RAW DATA (7,000 × 12 variables)
    │
    ▼
 PASO 3 — EDA
    Distribuciones · Correlaciones Pearson/Spearman
    Diagnóstico sintético · Poder discriminatorio visual
    │
    ▼
 PASO 4 — SELECCIÓN + FEATURE ENGINEERING
    Exclusión de burnout_level (TARGET)
    Exclusión de stress_level  (leakage 99.99%)
    Exclusión de redundantes y sin poder discriminatorio
    Construcción de 3 features compuestas
    │
    ▼  [4 variables limpias]
 PASO 5 — PREPROCESAMIENTO
    Split 80/20 · Imputación mediana · StandardScaler adaptativo
    │
    ├──────────────────────────┐
    ▼                          ▼
 PASO 6-8                   PASO 9
 K-MEANS                    DBSCAN
 k=2 óptimo                 eps por K-Distance Graph
 2 perfiles                 1 bloque denso + ~5% ruido
    │
    ▼
 PASO 10 — REDUCCIÓN DIMENSIONAL
    PCA (~35% varianza) · t-SNE (vecindad local) · Comparativa
    │
    ▼
 PASO 11 — DETECCIÓN DE ANOMALÍAS
    Isolation Forest + LOF · Consenso · Cruce con DBSCAN ruido
    │
    ▼
 PASO 12-15 — RESUMEN + CONCLUSIONES
    Panel visual · Tabla comparativa · Interpretación de negocio
```

---

## ⚙️ Espacio final del modelo (4 variables)

Después del análisis de leakage, redundancias y feature engineering, el modelo opera con:

| Variable | Origen | Fórmula | Qué captura |
|---|---|---|---|
| `caffeine_intake` | 📌 Original retenida | — | Compensación energética |
| `ratio_trabajo_descanso` | 🆕 Construida | `daily_work_hours / sleep_hours` | Desequilibrio vida-trabajo |
| `carga_operativa` | 🆕 Construida | `bugs_per_day + meetings_per_day` | Presión técnica + social |
| `indice_recuperacion` | 🆕 Construida | `sleep_hours + exercise_hours` | Capacidad de recuperación física |

### Variables excluidas y su razón

| Variable | Razón de exclusión |
|---|---|
| `burnout_level` | TARGET — incluirla sería hacer trampa |
| `stress_level` | Codificación determinista del target (precisión 99.99%, regla de 3 condiciones) |
| `screen_time` | Redundante con `daily_work_hours` (r = 0.93) |
| `age` | Sin poder discriminatorio (r < 0.02 con todo) |
| `experience_years` | Sin poder discriminatorio (r < 0.02 con todo) |
| `commits_per_day` | No discrimina burnout en ningún cuartil (r = −0.01) |

---

## 📊 Resultados principales

### K-Means (k = 2, modelo principal de segmentación)

| Métrica | Valor (train) | Valor (val) |
|---|---|---|
| Silhouette Score | ~0.25 | Diferencia < 0.05 ✅ |
| Davies-Bouldin | ~1.54 | — |
| Calinski-Harabasz | ~2,131 | — |

**Perfiles descubiertos:**

| Perfil | Etiqueta | Características principales |
|---|---|---|
| 🔴 Cluster A | **Alto Riesgo — Sobrecargado** | Alto `ratio_trabajo_descanso`, alta `carga_operativa`, bajo `indice_recuperacion`, alto `caffeine` |
| 🟢 Cluster B | **Bajo Riesgo — Equilibrado** | Bajo `ratio_trabajo_descanso`, baja `carga_operativa`, alto `indice_recuperacion`, cafeína moderada |

La validación con `burnout_level` como referencia externa confirma diferencias estadísticamente significativas entre clusters (test Chi², p < 0.05 en train y val).

### DBSCAN (detección de outliers)

En el espacio de 4 dimensiones con distribución uniforme, DBSCAN detecta un único bloque denso más ~5% de puntos de ruido. Su valor en este proyecto no es la segmentación sino la **detección de outliers**: desarrolladores con perfiles estadísticamente atípicos que no encajan en ningún patrón esperado.

### Detección de anomalías (IF + LOF)

| Método | Anomalías detectadas | Perspectiva |
|---|---|---|
| Isolation Forest | ~5% del train | Outliers globales |
| LOF | ~5% del train | Outliers locales (relativos al vecindario) |
| **Consenso IF+LOF** | **~1-2% del train** | **Mayor confianza — intervención prioritaria** |

### Reducción dimensional

| Técnica | Varianza capturada (2D) | Ejes interpretables |
|---|---|---|
| PCA | ~35% (PC1: intensidad laboral, PC2: recuperación) | ✅ Sí |
| t-SNE | N/A (no lineal) | ❌ No |

---

## 🛠️ Requisitos e instalación

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy missingno
```

### Ejecución en Google Colab

1. Subir `developer_burnout_dataset.csv` a Google Drive
2. Ajustar `FILE_PATH` en el **Paso 2** del notebook:
   ```python
   FILE_PATH = '/content/drive/My Drive/ruta/al/developer_burnout_dataset.csv'
   ```
3. Ejecutar todas las celdas **en orden secuencial** (cada paso depende del anterior)

> El notebook incluye verificaciones de dependencias explícitas entre pasos. Si se ejecutan celdas fuera de orden, se lanzará un `RuntimeError` con el mensaje correspondiente.

---

## 📚 Wiki del proyecto

La documentación detallada está organizada en la carpeta `wiki/`. Cada archivo cubre un aspecto específico del análisis:

- **[Home](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki)** — Índice completo y mapa conceptual
- **[Dataset](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/01_Dataset)** — Descripción de variables, estadísticas, diagnóstico sintético
- **[Selección de variables y Leakage](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/02_Seleccion_Variables_Leakage)** — Por qué se excluyen `burnout_level` y `stress_level`
- **[Feature Engineering](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/03_Feature_Engineering)** — Construcción de las 3 features compuestas
- **[Preprocesamiento](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/04_Preprocesamiento)** — Split, imputación honesta, escalado adaptativo
- **[K-Means: selección de k](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/05_KMeans_Seleccion_k)** — 4 métricas, ranking combinado, k=2
- **[K-Means: perfiles](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/06_KMeans_Perfiles)** — Centroides, etiquetado, validación Chi²
- **[DBSCAN](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/07_DBSCAN)** — Clustering por densidad y maldición de la dimensionalidad
- **[Reducción dimensional](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/08_Reduccion_Dimensional)** — PCA vs t-SNE
- **[Detección de anomalías](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/09_Deteccion_Anomalias)** — IF + LOF + consenso
- **[Conclusiones](https://github.com/ggranizo2507/Machine-Learning-Grupo-4-K-Means-DBSCAN/wiki/10_Conclusiones)** — Perfiles, tabla comparativa, limitaciones

---

## 👥 Autoría

**Grupo 4** — Asignatura: Machine Learning · Aprendizaje No Supervisado
Universidad UEES

| # | Nombre |
|---|--------|
| 1 | Eduardo Alejandro Ceballos Jijón |
| 2 | Guillermo Leonidas Granizo Veintimilla |
| 3 | José Farid Ulloa Manzur |
| 4 | Christian Xavier Valle Maridueña |
---

## 📄 Licencia
Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

Dataset original: Kaggle (uso académico).

Código del notebook: uso académico — Grupo 4, UEES.
