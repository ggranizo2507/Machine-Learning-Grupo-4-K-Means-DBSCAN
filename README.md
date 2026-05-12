<div align="center">

# 🔥 Clasificación de Perfiles de Burnout en Desarrolladores
### Aprendizaje No Supervisado · K-Means · DBSCAN · Isolation Forest · LOF

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)](https://colab.research.google.com)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)

**Universidad de Especialidades Espíritu Santo (UEES) — Grupo 4**

</div>

---

## 🎯 Pregunta central del proyecto

> **"Queremos descubrir perfiles de burnout de desarrolladores sin decirle al modelo quién tiene burnout alto, medio o bajo."**

El objetivo es revelar **perfiles latentes** a partir de las *causas* del burnout — horas de trabajo, carga operativa, calidad del sueño, recuperación física — **sin usar la etiqueta resultante**. La variable `burnout_level` se consulta únicamente al final como validación externa posterior; nunca durante el entrenamiento.

---

## 📁 Estructura del repositorio

```
📦 burnout-developer-clustering
 ┣ 📓 KMeans_DBSCAN_anomalias_Grupo_4_1205_FINAL.ipynb
 ┣ 📊 developer_burnout_dataset.csv
 ┣ 📄 README.md
 ┗ 📚 wiki/
    ┣ 🏠 Home.md
    ┣ 01_Dataset_y_Variables.md
    ┣ 02_EDA_Analisis_Exploratorio.md
    ┣ 03_Seleccion_Variables_Feature_Engineering.md
    ┣ 04_Preprocesamiento.md
    ┣ 05_KMeans_Seleccion_k_y_Entrenamiento.md
    ┣ 06_DBSCAN_Clustering_por_Densidad.md
    ┣ 07_Reduccion_Dimensional_PCA_tSNE.md
    ┣ 08_Deteccion_Anomalias_IF_LOF.md
    ┣ 09_Perfiles_Identificados_y_Resultados.md
    ┗ 10_Limitaciones_y_Recomendaciones.md
```

---

## 📊 Sobre el dataset

| Atributo | Detalle |
|----------|---------|
| **Fuente** | [Developer Burnout Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples) |
| **Registros** | 7,000 desarrolladores |
| **Variables** | 12 (11 numéricas + 1 categórica target) |
| **Valores faltantes** | ~140 NaN por columna (~2%, patrón MCAR uniforme) |
| **Target** | `burnout_level`: Low / Medium / High — **NO entra al modelo** |
| **Naturaleza** | Sintético (skewness ≈ 0, cero outliers IQR, NaN uniformes) |

### Variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | Numérica | Edad del desarrollador (años) |
| `experience_years` | Numérica | Años de experiencia total |
| `daily_work_hours` | Numérica | Promedio horas de trabajo/día |
| `sleep_hours` | Numérica | Promedio horas de sueño/día |
| `caffeine_intake` | Numérica discreta | Bebidas con cafeína por día |
| `bugs_per_day` | Numérica discreta | Bugs producidos diariamente |
| `commits_per_day` | Numérica discreta | Commits realizados por día |
| `meetings_per_day` | Numérica discreta | Reuniones atendidas por día |
| `screen_time` | Numérica | Tiempo frente a pantalla (h/día) |
| `exercise_hours` | Numérica | Horas de ejercicio físico/día |
| `stress_level` | Numérica | Puntuación de estrés (0–100) |
| `burnout_level` | Categórica | **TARGET** — Low / Medium / High |

---

## 🏗️ Arquitectura del análisis — 15 Pasos

```
PASOS 1–2  │ Importación de librerías · Carga del dataset
───────────┼──────────────────────────────────────────────────────────────
PASO  3    │ EDA completo (3A→3H)
           │  3A · Info general y estadísticas descriptivas
           │  3B · Diagnóstico de NaN + visualización missingno
           │  3C · Distribuciones de variables numéricas
           │  3D · Diagnóstico: ¿real o sintético?
           │  3E · Distribución del target + boxplots discriminatorios
           │  3F · Heatmap correlaciones Pearson (dataset completo)
           │  3G · Diagnóstico específico commits_per_day
           │  3H · Correlación Spearman vs Pearson
───────────┼──────────────────────────────────────────────────────────────
PASO  4    │ Selección de variables + Feature Engineering (4A→4C)
           │  4A · Resumen exclusiones del EDA
           │  4B · Construcción de 3 features compuestas
           │  4C · Correlación del espacio final del modelo
───────────┼──────────────────────────────────────────────────────────────
PASO  5    │ Preprocesamiento con split correcto (5A→5D)
           │  5A · Split train/val + ColumnTransformer + imputación
           │  5B · Diagnóstico automático de Scaler (Shapiro-Wilk)
           │  5C · Escalado adaptativo (StandardScaler elegido)
           │  5D · Verificación visual antes/después del escalado
───────────┼──────────────────────────────────────────────────────────────
PASO  6    │ Selección k óptimo (rango 2–12, 4 métricas)
           │  6A · Evaluación Inercia, Silhouette, DB, Calinski-H
           │  6B · Panel visual de 4 métricas + ranking combinado
           │  6C · Selección automática: k = 2
───────────┼──────────────────────────────────────────────────────────────
PASO  7    │ Entrenamiento K-Means final (7A→7B)
PASO  8    │ Validación externa con burnout_level (referencia)
PASO  9    │ DBSCAN: K-Distance Graph, eps, min_samples, resultado
───────────┼──────────────────────────────────────────────────────────────
PASO 10    │ Reducción dimensional (10.1→10.11)
           │  PCA: varianza, loadings, scatter, comparativa PCA vs train
           │  t-SNE: proyección, comparativa fidelidad, panel visual
───────────┼──────────────────────────────────────────────────────────────
PASO 11    │ Detección de anomalías (11.1→11.4)
           │  11.1  · Isolation Forest
           │  11.2  · LOF — Local Outlier Factor
           │  11.2B · Cruce DBSCAN ruido vs IF/LOF
           │  11.3  · Análisis grupos de anomalías
           │  11.4  · Panel visual LOF completo
───────────┼──────────────────────────────────────────────────────────────
PASOS 12–15│ Panel resumen · Conclusiones · Perfiles · Limitaciones
```

---

## ⚙️ Decisiones metodológicas clave

### Variables excluidas del modelo y sus razones

| Variable | Razón | Evidencia cuantitativa |
|----------|-------|------------------------|
| `burnout_level` | **TARGET** — leakage directo | — |
| `stress_level` | Codificación determinista del target | 99.99% precisión con regla de 3 umbrales; r = 0.91 con burnout |
| `screen_time` | Redundante con `daily_work_hours` | r = 0.93 |
| `age` | Sin poder discriminatorio | r < 0.02 con todas las demás |
| `experience_years` | Sin poder discriminatorio | r < 0.02 con todas las demás |
| `commits_per_day` | Sin correlación con burnout | r = −0.01; % burnout alto idéntico en Q1, Q2, Q3 y Q4 |

### Espacio final del modelo — 4 variables limpias

| Variable | Tipo | Fórmula | Qué captura |
|----------|------|---------|-------------|
| `caffeine_intake` | 📌 Original | Variable directa | Compensación energética artificial |
| `ratio_trabajo_descanso` | 🆕 Construida | `daily_work_hours / sleep_hours` | Desequilibrio vida-trabajo |
| `carga_operativa` | 🆕 Construida | `bugs_per_day + meetings_per_day` | Presión técnica y social |
| `indice_recuperacion` | 🆕 Construida | `sleep_hours + exercise_hours` | Capacidad de recuperación física |

**Verificación anti-leakage:** R² features nuevas → `burnout_level` ≈ 0.45 (muy por debajo del umbral de `stress_level`: R² = 0.83).

**Impacto medido del feature engineering (k=2):**

| Espacio | Silhouette | Davies-Bouldin | Calinski-H |
|---------|-----------|----------------|------------|
| 7 variables con doble conteo | 0.147 | 2.250 | 1,032 |
| **4 variables sin doble conteo (final)** | **0.252** | **1.541** | **2,131** |

---

## 📈 Resultados

### K-Means — Métricas obtenidas

| Métrica | Valor |
|---------|-------|
| **k óptimo** | **2** (ranking combinado de 4 métricas) |
| **Silhouette train** | ~0.252 |
| **Silhouette val** | Diferencia < 0.05 ✅ (no sobreajustado) |
| **Davies-Bouldin** | ~1.541 (menor = mejor) |
| **Calinski-Harabasz** | ~2,131 (mayor = mejor) |

### Tres perfiles identificados

**🔴 Perfil 1 — Desarrollador Sobrecargado (Alto Riesgo)**

| Variable | Posición | Señal |
|----------|----------|-------|
| `ratio_trabajo_descanso` | 🔺 Por encima de la media | Jornadas largas con poco sueño |
| `carga_operativa` | 🔺 Por encima de la media | Muchos bugs + reuniones frecuentes |
| `indice_recuperacion` | 🔻 Por debajo de la media | Poco sueño y sin ejercicio |
| `caffeine_intake` | 🔺 Por encima de la media | Compensación energética elevada |

Trabaja >10–11 h/día, asiste a >5 reuniones, duerme <6 h. **Validación:** concentra casos `burnout_level = High`.

---

**🟢 Perfil 2 — Desarrollador Equilibrado (Bajo Riesgo)**

| Variable | Posición | Factor protector |
|----------|----------|-----------------|
| `ratio_trabajo_descanso` | 🔻 Por debajo de la media | Jornadas razonables con buen descanso |
| `carga_operativa` | 🔻 Por debajo de la media | Poca presión técnica y social |
| `indice_recuperacion` | 🔺 Por encima de la media | Sueño adecuado + ejercicio regular |
| `caffeine_intake` | Bajo-moderado | Sin compensación artificial |

Trabaja ~6–8 h/día, <4 reuniones, duerme >7 h. **Validación:** concentra casos `Low` y `Medium burnout`.

---

**🚨 Perfil 3 — Desarrollador Atípico (Urgente Individual)**

Detectado por Isolation Forest + LOF. ~5% del dataset. Combinaciones extremas de variables que no encajan en ninguno de los dos perfiles. El **consenso IF+LOF** (~1–2%) identifica los candidatos prioritarios para revisión individual de RRHH.

---

### DBSCAN — resultado diagnóstico

DBSCAN encontró **1 solo cluster + ~5% de ruido**. No es un fallo del algoritmo: en un espacio de 4 dimensiones con distribución sintética uniforme no existen zonas de baja densidad que delimiten grupos naturales. Confirma que los 2 perfiles de K-Means son **particiones métricas válidas** y valida K-Means como el modelo correcto para segmentación.

---

### Reducción dimensional

| Técnica | Varianza 2D | Ejes | Uso recomendado |
|---------|-------------|------|-----------------|
| **PCA** | ~35% | ✅ PC1 = intensidad laboral · PC2 = recuperación vs presión | Analizar y comunicar |
| **t-SNE** | N/A (no lineal) | ❌ Sin interpretación directa | Visualizar estructura local |

---

## 🚀 Cómo ejecutar

### Google Colab

```
1. Sube developer_burnout_dataset.csv a Google Drive:
   My Drive/MachineLearning/Semana3/developer_burnout_dataset.csv

2. Abre el notebook en Colab

3. Ejecuta las celdas en orden: Paso 1 → Paso 15
```

> ⚠️ Si el archivo está en otra ruta, modifica `FILE_PATH` en el **Paso 2** antes de ejecutar.

### Entorno local

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn missingno scipy
jupyter notebook KMeans_DBSCAN_anomalias_Grupo_4_1205_FINAL.ipynb
```

---

## 🧰 Stack tecnológico

| Categoría | Librerías |
|-----------|-----------|
| **Datos** | `pandas`, `numpy` |
| **Visualización** | `matplotlib`, `seaborn`, `plotly` |
| **Diagnóstico NaN** | `missingno` |
| **Preprocesamiento** | `SimpleImputer`, `ColumnTransformer`, `StandardScaler`, `RobustScaler`, `OneHotEncoder` |
| **Clustering** | `KMeans`, `DBSCAN` |
| **Anomalías** | `IsolationForest`, `LocalOutlierFactor` |
| **Reducción dimensional** | `PCA`, `TSNE` |
| **Evaluación clustering** | `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` |
| **Estadística** | `scipy.stats.shapiro` |

---

## ⚠️ Advertencia sobre los datos

El análisis detectó señales de **dataset sintético**:

- Skewness máximo 0.06 en todas las variables (datos reales → |skew| > 0.5)
- **Cero outliers IQR** en las 11 columnas numéricas
- NaN en patrón MCAR uniforme (exactamente 2% por columna)
- Fronteras de `stress_level` perfectamente deterministas (99.99% precisión)

Los resultados son válidos como exploración metodológica. Con datos reales de campo, los clusters serán menos nítidos y los parámetros óptimos requieren recalibración.

---

## 📚 Wiki del proyecto

La documentación detallada de cada etapa está disponible en la carpeta `wiki/`:

| Página | Contenido |
|--------|-----------|
| [Home](wiki/Home.md) | Presentación, navegación y principios metodológicos |
| [Dataset](wiki/01_Dataset_y_Variables.md) | Variables, naturaleza sintética e implicaciones |
| [EDA](wiki/02_EDA_Analisis_Exploratorio.md) | Distribuciones, NaN, correlaciones Pearson y Spearman |
| [Feature Engineering](wiki/03_Seleccion_Variables_Feature_Engineering.md) | Exclusiones justificadas, features construidas, anti-leakage |
| [Preprocesamiento](wiki/04_Preprocesamiento.md) | Split, imputación, diagnóstico de scaler, escalado |
| [K-Means](wiki/05_KMeans_Seleccion_k_y_Entrenamiento.md) | Selección de k, entrenamiento, heatmap de centroides, validación |
| [DBSCAN](wiki/06_DBSCAN_Clustering_por_Densidad.md) | K-Distance Graph, resultado, interpretación del 1 cluster |
| [PCA y t-SNE](wiki/07_Reduccion_Dimensional_PCA_tSNE.md) | Varianza, loadings, fidelidad, cuándo usar cada técnica |
| [Anomalías](wiki/08_Deteccion_Anomalias_IF_LOF.md) | IF vs LOF, perspectiva global vs local, consenso |
| [Resultados](wiki/09_Perfiles_Identificados_y_Resultados.md) | Tres perfiles, tabla comparativa, resumen ejecutivo |
| [Limitaciones](wiki/10_Limitaciones_y_Recomendaciones.md) | 5 limitaciones documentadas con propuestas de solución |

---

## 👥 Equipo

**Grupo 4 — Modelos No Supervisados**
Universidad de Especialidades Espíritu Santo (UEES)

---

<div align="center">
<sub>Fuente: <a href="https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples">Developer Burnout Prediction Dataset — Kaggle</a></sub>
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
