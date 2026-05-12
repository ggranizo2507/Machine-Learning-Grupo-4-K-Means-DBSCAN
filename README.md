# 🔥 Developer Burnout Profiling — Aprendizaje NO Supervisado

> **Universidad UEES · Machine Learning · Grupo 4 · Semana 3**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)

---

## 📋 Descripción del Proyecto

Este proyecto aplica **aprendizaje no supervisado** para descubrir perfiles latentes de burnout en desarrolladores de software, sin utilizar la etiqueta `burnout_level` durante el entrenamiento.

> **Pregunta central:**
> *"¿Podemos descubrir perfiles de burnout de desarrolladores sin decirle al modelo quién tiene burnout alto, medio o bajo?"*

El enfoque identifica patrones a partir de **causas observables** —horas de trabajo, carga operativa, calidad del sueño, hábitos de ejercicio— y valida si los grupos encontrados coinciden con los niveles reales de burnout como verificación externa.

---

## 🎯 Objetivo

Descubrir perfiles latentes a partir de las causas del burnout (horas, estrés, satisfacción), no de la etiqueta resultante, utilizando cuatro familias de modelos no supervisados:

| Modelo | Rol en el análisis |
|--------|-------------------|
| **K-Means** | Segmentación principal de perfiles |
| **DBSCAN** | Detección de estructura por densidad y outliers |
| **PCA + t-SNE** | Reducción de dimensionalidad y visualización |
| **Isolation Forest + LOF** | Detección de anomalías individuales |

---

## 📂 Estructura del Repositorio

```
developer-burnout-profiling/
│
├── 📓 KMeans_DBSCAN_anomalias_Grupo_4.ipynb   # Notebook principal (Google Colab)
├── 📊 developer_burnout_dataset.csv            # Dataset (7,000 registros)
├── 📄 README.md                                # Este archivo
├── 📜 LICENSE                                  # Licencia MIT
│
└── 📚 wiki/                                    # Documentación extendida
    ├── 01_contexto_y_objetivo.md
    ├── 02_estructura_del_dataset.md
    ├── 03_metodologia.md
    ├── 04_feature_engineering.md
    ├── 05_resultados_kmeans.md
    ├── 06_resultados_dbscan.md
    ├── 07_reduccion_dimensionalidad.md
    ├── 08_deteccion_anomalias.md
    └── 09_conclusiones.md
```

---

## 🗂️ Dataset

**Fuente:** [Developer Burnout Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `age` | float | Edad del desarrollador (años) |
| `experience_years` | float | Años totales de experiencia en programación |
| `daily_work_hours` | float | Promedio de horas de trabajo por día |
| `sleep_hours` | float | Promedio de horas de sueño por día |
| `caffeine_intake` | float | Bebidas con cafeína consumidas por día |
| `bugs_per_day` | float | Promedio de bugs producidos por día |
| `commits_per_day` | float | Commits de código por día |
| `meetings_per_day` | float | Reuniones diarias |
| `screen_time` | float | Tiempo total frente a pantalla por día (horas) |
| `exercise_hours` | float | Tiempo diario de ejercicio físico (horas) |
| `stress_level` | float | Puntuación de estrés calculada (0–100) |
| `burnout_level` | str | **Variable objetivo** — Low / Medium / High |

> ⚠️ `burnout_level` y `stress_level` fueron **excluidas del entrenamiento**. `stress_level` predice `burnout_level` con 99.99% de precisión mediante umbrales simples (data leakage confirmado). Ambas se usan únicamente como validación externa.

---

## ⚙️ Decisiones Metodológicas Clave

### Variables excluidas del modelo

| Variable | Razón |
|----------|-------|
| `burnout_level` | TARGET — incluirla sería hacer trampa |
| `stress_level` | Codificación determinista de `burnout_level` (r = 0.91, regla de umbrales: 99.99% precisión) |
| `age` | Sin poder discriminatorio (r < 0.02 con todas las variables) |
| `experience_years` | Sin poder discriminatorio (r < 0.02) |
| `screen_time` | Redundante con `daily_work_hours` (r = 0.93) |
| `commits_per_day` | Sin poder discriminatorio (r = −0.01 con burnout en todos los cuartiles) |

### Feature Engineering — Espacio final de 5 variables

| Variable compuesta | Fórmula | Qué captura |
|-------------------|---------|-------------|
| `caffeine_intake` | Original | Compensación energética |
| `bugs_per_day` | Original | Carga técnica directa (r = +0.46 con burnout) |
| `ratio_trabajo_descanso` | `daily_work_hours / sleep_hours` | Desequilibrio jornada/sueño |
| `carga_operativa` | `bugs_per_day + meetings_per_day` | Presión técnica + social combinada |
| `indice_recuperacion` | `sleep_hours + exercise_hours` | Capacidad de recuperación física |

**Impacto medido del feature engineering:**

| Espacio | Silhouette | Davies-Bouldin | Calinski-H |
|---------|-----------|----------------|------------|
| 7 vars con doble conteo | 0.147 | 2.250 | 1,032 |
| **5 vars sin doble conteo** | **0.252** | **1.541** | **2,131** |

> ✅ Mejora del **+71%** en Silhouette y del **+106%** en Calinski-Harabasz.

---

## 📊 Resultados Principales

### K-Means: k = 2 (óptimo)

| Métrica | k=2 | k=3 | Veredicto |
|---------|-----|-----|-----------|
| Silhouette | **0.1451** | 0.1399 | ✅ k=2 |
| Davies-Bouldin | 2.2649 | 1.9140 | ⚠️ diferencia < 0.05 |
| Calinski-Harabasz | **1,032.3** | 986.8 | ✅ k=2 |

### Perfiles Descubiertos

```
🔴 PERFIL SOBRECARGADO — Alto Riesgo
   • ratio_trabajo_descanso   → ALTO (jornadas largas vs poco sueño)
   • carga_operativa          → ALTA (bugs + reuniones elevados)
   • caffeine_intake          → ALTO (compensación energética)
   • indice_recuperacion      → BAJO (poco sueño y ejercicio)
   → Concentra la mayor proporción de burnout_level = "High"

🟢 PERFIL EQUILIBRADO — Bajo Riesgo
   • ratio_trabajo_descanso   → MODERADO (jornadas acotadas)
   • carga_operativa          → BAJA (menos bugs y reuniones)
   • caffeine_intake          → MODERADO
   • indice_recuperacion      → ALTO (sueño y ejercicio regulares)
   → Concentra la mayor proporción de burnout_level = "Low"
```

### Validación

- **Generalización:** Diferencia Silhouette train/val < 0.05 ✅
- **Test Chi²:** p < 0.05 en train y validación → los clusters capturan diferencias reales de burnout
- **Conclusión:** El modelo reproducción la separación de burnout **sin haber visto la etiqueta**

### Detección de Anomalías

| Método | Anomalías detectadas | Enfoque |
|--------|---------------------|---------|
| Isolation Forest | ~5% del train | Outliers globales |
| LOF | ~5% del train | Outliers locales |
| **Consenso IF+LOF** | **~2.5%** | **Máxima confianza diagnóstica** |

---

## 🛠️ Stack Tecnológico

```python
pandas          # Manipulación de datos
numpy           # Operaciones numéricas
matplotlib      # Visualizaciones estáticas
seaborn         # Visualizaciones estadísticas
plotly          # Visualizaciones interactivas
missingno       # Análisis de valores faltantes

sklearn.cluster         # KMeans, DBSCAN
sklearn.preprocessing   # StandardScaler, OneHotEncoder
sklearn.impute          # SimpleImputer
sklearn.compose         # ColumnTransformer, Pipeline
sklearn.decomposition   # PCA
sklearn.manifold        # t-SNE
sklearn.ensemble        # IsolationForest
sklearn.neighbors       # LocalOutlierFactor, NearestNeighbors
sklearn.metrics         # silhouette_score, davies_bouldin_score, calinski_harabasz_score

scipy.stats     # Test de normalidad Shapiro-Wilk
```

---

## 🚀 Cómo Ejecutar

### En Google Colab (recomendado)

1. Abrir el notebook `KMeans_DBSCAN_anomalias_Grupo_4.ipynb` en Google Colab
2. Subir el archivo `developer_burnout_dataset.csv` a Google Drive en la ruta:
   ```
   /My Drive/MachineLearning/Semana3/developer_burnout_dataset.csv
   ```
3. Ejecutar todas las celdas en orden (Runtime → Run All)

### En entorno local

```bash
# Clonar el repositorio
git clone https://github.com/[usuario]/developer-burnout-profiling.git
cd developer-burnout-profiling

# Instalar dependencias
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy missingno

# Ajustar la ruta del dataset en el Paso 2 del notebook:
# FILE_PATH = './developer_burnout_dataset.csv'

# Abrir el notebook
jupyter notebook KMeans_DBSCAN_anomalias_Grupo_4.ipynb
```

> ⚠️ Si se ejecuta en local, cambiar la celda de `google.colab.drive.mount` en el Paso 2 y ajustar `FILE_PATH` a la ruta local del CSV.

---

## 📈 Flujo del Análisis

```
Paso 1  → Importación de librerías
Paso 2  → Carga del dataset (7,000 registros × 12 columnas)
Paso 3  → EDA: distribuciones, NaN, correlaciones Pearson/Spearman
Paso 4  → Selección de variables + Feature Engineering (5 vars finales)
Paso 5  → Preprocesamiento: split 80/20, imputación, escalado adaptativo
Paso 6  → Selección del k óptimo (rango 2–12, 4 métricas + ranking)
Paso 7  → K-Means final (k=2) + validación train/val
Paso 8  → Etiquetado de perfiles + análisis de interpretabilidad
Paso 9  → DBSCAN: parámetros por K-Distance Graph + comparativa
Paso 10 → Reducción dimensionalidad: PCA + t-SNE + comparativa
Paso 11 → Detección de anomalías: Isolation Forest + LOF + consenso
Paso 12 → Resumen comparativo visual de todos los modelos
Paso 13 → Conclusiones finales y hallazgos de negocio
Paso 14 → Reflexión metodológica: limitaciones y mejoras propuestas
```

---

## 👥 Autores

**Grupo 4 — Universidad UEES**
Machine Learning · Semana 3

---

## 📚 Referencias

- Scikit-learn: Machine Learning in Python — Pedregosa et al. (2011)
- Dataset: [Developer Burnout Prediction Dataset](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples) — Kaggle
- Ester, M. et al. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise.* KDD-96.
- Liu, F.T. et al. (2008). *Isolation Forest.* ICDM.
- Breunig, M. et al. (2000). *LOF: Identifying Density-Based Local Outliers.* SIGMOD.

---

## 👥 Autores

**Grupo 4 — Machine Learning**
Universidad de Especialidades Espíritu Santo (UEES)

| # | Nombre |
|---|--------|
| 1 | Eduardo Alejandro Ceballos Jijón |
| 2 | Guillermo Leónidas Granizo Veintimilla |
| 3 | José Farid Ulloa Manzur |
| 4 | Christian Xavier Valle Maridueña |

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.
