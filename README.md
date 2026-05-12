<div align="center">

# 🔥 Developer Burnout — Perfiles Latentes con Modelos No Supervisados

**K-Means · DBSCAN · Isolation Forest · LOF · PCA · t-SNE**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)
[![License](https://img.shields.io/badge/Licencia-MIT-22C55E?style=flat-square)](LICENSE)

*Universidad de Especialidades Espíritu Santo (UEES) · Grupo 4 · Machine Learning*

</div>

---

## 🎯 Pregunta central del proyecto

> **"Queremos descubrir perfiles de burnout de desarrolladores sin decirle al modelo quién tiene burnout alto, medio o bajo."**

El objetivo es revelar **perfiles latentes** a partir de las *causas* del burnout (horas de trabajo, carga operativa, hábitos de sueño y recuperación), **no** de la etiqueta resultante.

`burnout_level` se excluye del entrenamiento y se usa únicamente al final como **validación externa** para confirmar coherencia con la realidad.

---

## 📂 Estructura del repositorio

```
📦 developer-burnout-unsupervised/
├── 📓 KMeans_DBSCAN_anomalias_Grupo_4_1205_FINAL.ipynb
├── 📊 developer_burnout_dataset.csv
├── 📄 README.md
└── 📚 wiki/
    ├── Home.md
    ├── 01-Dataset-y-Variables.md
    ├── 02-Analisis-Exploratorio-EDA.md
    ├── 03-Seleccion-Variables-y-Leakage.md
    ├── 04-Feature-Engineering.md
    ├── 05-Preprocesamiento.md
    ├── 06-Seleccion-k-optimo.md
    ├── 07-KMeans-Entrenamiento-y-Perfiles.md
    ├── 08-DBSCAN.md
    ├── 09-PCA-y-tSNE.md
    ├── 10-Deteccion-de-Anomalias.md
    ├── 11-Resultados-y-Comparativa.md
    └── 12-Limitaciones-y-Trabajo-Futuro.md
```

---

## 📊 Dataset

| Atributo | Detalle |
|----------|---------|
| **Fuente** | [Kaggle — Developer Burnout Prediction Dataset](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples) |
| **Registros** | 7,000 desarrolladores |
| **Variables** | 12 (11 numéricas + 1 categórica target) |
| **Valores faltantes** | ~2% por columna — patrón MCAR uniforme |
| **Naturaleza** | Sintético: skewness máximo 0.06, cero outliers IQR |

### Descripción de variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | Numérica discreta | Edad del desarrollador (años) |
| `experience_years` | Numérica discreta | Años totales de experiencia |
| `daily_work_hours` | Numérica continua | Horas promedio de trabajo diario |
| `sleep_hours` | Numérica continua | Horas de sueño por día |
| `caffeine_intake` | Numérica discreta | Bebidas con cafeína por día |
| `bugs_per_day` | Numérica discreta | Bugs producidos por día |
| `commits_per_day` | Numérica discreta | Commits por día |
| `meetings_per_day` | Numérica discreta | Reuniones por día |
| `screen_time` | Numérica continua | Tiempo total frente a pantalla (horas/día) |
| `exercise_hours` | Numérica continua | Horas de ejercicio físico diario |
| `stress_level` | Numérica continua | Score de estrés calculado (0–100) |
| `burnout_level` | **Categórica — TARGET** | Low / Medium / High |

---

## 🏗️ Arquitectura del análisis

```
┌──────────────────────────────────────────────────────────────┐
│  FASE 1 — COMPRENSIÓN DE DATOS                               │
│  Paso 1 · Importación de librerías                           │
│  Paso 2 · Carga del dataset desde Google Drive               │
│  Paso 3 · EDA: NaN, distribuciones, correlaciones            │
├──────────────────────────────────────────────────────────────┤
│  FASE 2 — INGENIERÍA DE DATOS                                │
│  Paso 4 · Selección de variables + Feature Engineering       │
│  Paso 5 · Preprocesamiento: imputación, escalado, split      │
├──────────────────────────────────────────────────────────────┤
│  FASE 3 — MODELADO                                           │
│  Paso 6 · Selección de k óptimo (4 métricas + ranking)      │
│  Paso 7 · Entrenamiento final K-Means + heatmap centroides   │
│  Paso 8 · Etiquetado y validación cruzada con burnout_level  │
│  Paso 9 · DBSCAN: K-Distance Graph + clustering densidad     │
├──────────────────────────────────────────────────────────────┤
│  FASE 4 — VISUALIZACIÓN Y ANOMALÍAS                          │
│  Paso 10 · PCA + t-SNE — reducción y comparativa            │
│  Paso 11 · Isolation Forest + LOF — detección de anomalías   │
├──────────────────────────────────────────────────────────────┤
│  FASE 5 — COMUNICACIÓN DE RESULTADOS                         │
│  Paso 12 · Panel visual resumen comparativo                  │
│  Paso 13 · Conclusiones finales y tabla comparativa          │
│  Paso 14 · Guía de lectura del panel visual                  │
│  Paso 15 · Reflexión: perfiles, limitaciones, comunicación   │
└──────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Decisiones metodológicas clave

### Variables excluidas del modelo (con justificación cuantitativa)

| Variable | Razón | Evidencia |
|----------|-------|-----------|
| `burnout_level` | TARGET — leakage directo | — |
| `stress_level` | Codificación determinista del target | Predice burnout_level con 99.99% precisión; r=0.91 con target |
| `screen_time` | Redundante con `daily_work_hours` | r=0.93 |
| `age` | Sin poder discriminatorio | r<0.02 con todas las variables |
| `experience_years` | Sin poder discriminatorio | r<0.02 con todas las variables |
| `commits_per_day` | Sin correlación con burnout | r=−0.01; % burnout idéntico en Q1–Q4 |

### Espacio final del modelo — 4 variables limpias

| Variable | Origen | Fórmula | Concepto |
|----------|--------|---------|---------|
| `caffeine_intake` | 📌 Original | — | Compensación energética |
| `ratio_trabajo_descanso` | 🆕 Construida | `daily_work_hours / sleep_hours` | Desequilibrio vida-trabajo |
| `carga_operativa` | 🆕 Construida | `bugs_per_day + meetings_per_day` | Presión técnica + social |
| `indice_recuperacion` | 🆕 Construida | `sleep_hours + exercise_hours` | Capacidad de recuperación |

> **Verificación anti-leakage:** R² del espacio combinado con `burnout_level` ≈ 0.45.
> Muy por debajo del umbral de `stress_level` (R²=0.83). No hay reconstrucción del target.

**Impacto del feature engineering medido:**

| Espacio | Silhouette | Davies-Bouldin | Calinski-H |
|---------|-----------|----------------|-----------|
| 7 variables con doble conteo | 0.147 | 2.250 | 1,032 |
| **4 variables sin doble conteo** | **0.252** | **1.541** | **2,131** |

---

## 📈 Resultados

### K-Means (k=2)

| Métrica | Valor |
|---------|-------|
| Silhouette (train) | ~0.252 |
| Silhouette (val) — diferencia vs train | < 0.05 ✅ |
| Davies-Bouldin | ~1.541 |
| Calinski-Harabasz | ~2,131 |

### Tres perfiles identificados

#### 🔴 Perfil 1 — Desarrollador Sobrecargado (Alto riesgo)
- `ratio_trabajo_descanso` **alto** — jornadas >10h con sueño <6h
- `carga_operativa` **alta** — muchos bugs + >5 reuniones diarias
- `indice_recuperacion` **bajo** — poco sueño, sin ejercicio
- `caffeine_intake` **alto** — compensación energética artificial
- **Validación:** concentra la mayoría de casos `burnout_level = High`

#### 🟢 Perfil 2 — Desarrollador Equilibrado (Bajo riesgo)
- `ratio_trabajo_descanso` **bajo** — jornadas ~6-8h con sueño >7h
- `carga_operativa` **baja** — pocos bugs, <4 reuniones
- `indice_recuperacion` **alto** — buen sueño + ejercicio regular
- `caffeine_intake` **bajo-moderado**
- **Validación:** concentra los casos `burnout_level = Low/Medium`

#### 🚨 Perfil 3 — Desarrollador Atípico (Caso urgente individual)
- ~5% del dataset — detectado exclusivamente por Isolation Forest y LOF
- Combinaciones extremas de variables que no encajan en ningún cluster
- Consenso IF+LOF (~1-2%): candidatos prioritarios para intervención individual de RRHH

### Comparativa de todos los modelos

| Modelo | Tipo | Resultado | Uso recomendado |
|--------|------|-----------|-----------------|
| **K-Means (k=2)** | Partitivo | 2 perfiles con Silhouette~0.252 | Políticas grupales de bienestar |
| **DBSCAN** | Densidad | 1 cluster + ~5% ruido | Confirmar outliers globales; valida K-Means |
| **Isolation Forest** | Anomalías globales | ~5% anomalías | Priorización rápida de casos urgentes |
| **LOF** | Anomalías locales | ~5% anomalías | Revisión contextual por equipo |
| **PCA** | Reducción lineal | ~35% varianza en 2D | Análisis e interpretación de ejes |
| **t-SNE** | Reducción no lineal | Alta fidelidad local | Visualización de estructura |

---

## 🚀 Cómo ejecutar

### Google Colab (recomendado)

```
1. Subir developer_burnout_dataset.csv a:
   My Drive/MachineLearning/Semana3/

2. Abrir el notebook en Colab

3. Ejecutar celdas en orden (Paso 1 → Paso 15)
   Si la ruta difiere, ajustar FILE_PATH en el Paso 2
```

### Entorno local

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn missingno scipy

jupyter notebook KMeans_DBSCAN_anomalias_Grupo_4_1205_FINAL.ipynb
```

> Para entorno local, comentar las líneas de `google.colab` en el Paso 2 y ajustar `FILE_PATH`.

---

## 🧰 Stack tecnológico

| Categoría | Librerías |
|-----------|-----------|
| Datos | `pandas`, `numpy` |
| Visualización | `matplotlib`, `seaborn`, `plotly`, `missingno` |
| Preprocesamiento | `SimpleImputer`, `ColumnTransformer`, `StandardScaler`, `RobustScaler`, `OneHotEncoder`, `Pipeline` |
| Clustering | `KMeans`, `DBSCAN` |
| Anomalías | `IsolationForest`, `LocalOutlierFactor` |
| Reducción dimensional | `PCA`, `TSNE` |
| Evaluación | `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` |
| Estadística | `scipy.stats.shapiro`, `sklearn.preprocessing.MinMaxScaler` |

---

## 📚 Documentación completa

Consulta la [Wiki del proyecto](wiki/Home.md) para la documentación completa, pedagógica y detallada de cada etapa del análisis.

---

<div align="center">
<sub>Dataset: <a href="https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples">Developer Burnout Prediction Dataset — Kaggle</a> · UEES Grupo 4</sub>
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
