# 🔥 Clasificación de Perfiles de Burnout en Desarrolladores
## Modelos No Supervisados | K-Means & DBSCAN | Detección de Anomalías

> **"Queremos descubrir perfiles de burnout de desarrolladores sin decirle al modelo quién tiene burnout alto, medio o bajo."**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Abrir%20en-Colab-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Objetivo y Pregunta Central](#-objetivo-y-pregunta-central)
- [Dataset](#-dataset)
- [Pipeline Metodológico](#-pipeline-metodológico)
- [Modelos Aplicados](#-modelos-aplicados)
- [Resultados Principales](#-resultados-principales)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Instalación y Uso](#-instalación-y-uso)
- [Hallazgos Clave](#-hallazgos-clave)
- [Integrantes del Grupo](#-integrantes-del-grupo)

---

## 🎯 Descripción del Proyecto

Este proyecto aplica **aprendizaje no supervisado** para descubrir perfiles latentes de burnout en desarrolladores de software, partiendo exclusivamente de variables de comportamiento y hábitos (carga laboral, calidad del sueño, consumo de cafeína, carga operativa), sin utilizar la etiqueta `burnout_level` durante el entrenamiento.

El enfoque es metodológicamente riguroso: `burnout_level` se usa **únicamente al final**, como referencia de validación externa, para verificar que los patrones encontrados tienen coherencia con la realidad.

---

## 🔬 Objetivo y Pregunta Central

**Objetivo:** Descubrir perfiles latentes a partir de las *causas* del burnout (horas, estrés, satisfacción), **no** de la etiqueta resultante.

### ¿Por qué excluir `burnout_level` del entrenamiento?

| Razón | Explicación |
|-------|-------------|
| **Es el TARGET** | Incluirla sería hacer trampa — el modelo re-descubriría la etiqueta |
| **Perfiles causales** | El objetivo es encontrar patrones en las *causas*, no en el resultado |
| **Validez del clustering** | Los clusters deben reflejar estructura real de comportamiento, no segmentar por etiqueta |
| **Uso en producción** | En entornos reales, esta etiqueta puede no estar disponible |

> ⚠️ `stress_level` también se excluye: es una **codificación determinista** de `burnout_level`. Una regla de tres condiciones la predice con precisión del 99.99%.

---

## 📊 Dataset

| Campo | Valor |
|-------|-------|
| **Fuente** | [Kaggle — Developer Burnout Prediction Dataset](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples) |
| **Registros** | 7,000 |
| **Variables** | 12 |
| **Naturaleza** | Sintético (distribuciones uniformes, skewness ≈ 0, sin outliers IQR) |
| **Valores faltantes** | ~140 por columna (2% MCAR) |

### Variables del Dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | Numérica continua | Edad del desarrollador (años) |
| `experience_years` | Numérica | Años totales de experiencia en programación |
| `daily_work_hours` | Numérica continua | Promedio de horas de trabajo diarias |
| `sleep_hours` | Numérica continua | Promedio de horas de sueño diarias |
| `caffeine_intake` | Numérica discreta | Bebidas con cafeína por día |
| `bugs_per_day` | Numérica discreta | Bugs producidos por día (promedio) |
| `commits_per_day` | Numérica discreta | Commits de código por día |
| `meetings_per_day` | Numérica discreta | Reuniones diarias |
| `screen_time` | Numérica continua | Tiempo total de exposición a pantalla (horas/día) |
| `exercise_hours` | Numérica continua | Tiempo diario de ejercicio físico (horas) |
| `stress_level` | Numérica continua | Score calculado de estrés (0–100) |
| `burnout_level` | Categórica | **TARGET** — Low / Medium / High |

### Distribución del Target

| Clase | Registros | Porcentaje |
|-------|-----------|------------|
| Medium | 3,485 | 49.8% |
| High | 1,782 | 25.5% |
| Low | 1,593 | 22.8% |

---

## 🔄 Pipeline Metodológico

```
Datos Raw (7,000 registros × 12 variables)
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 1: Importación de librerías       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 2: Carga del dataset              │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 3: EDA completo                   │
│  • Estadísticas descriptivas            │
│  • Diagnóstico NaN                      │
│  • Diagnóstico sintético vs real        │
│  • Distribuciones + boxplots            │
│  • Correlaciones Pearson + Spearman     │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 4: Selección de variables         │
│  • Exclusión de burnout_level y         │
│    stress_level (leakage)               │
│  • Feature Engineering:                 │
│    - ratio_trabajo_descanso             │
│    - carga_operativa                    │
│    - indice_recuperacion               │
│  • Verificación anti-leakage           │
│  • Correlación del espacio final        │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 5: Preprocesamiento               │
│  • Split Train/Val (80/20)              │
│  • Imputación por mediana (solo train)  │
│  • Diagnóstico automático de Scaler     │
│  • StandardScaler o RobustScaler        │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 6: Selección k óptimo (K-Means)   │
│  • Rango k=2 a k=12                     │
│  • 4 métricas: Inercia, Silhouette,     │
│    Davies-Bouldin, Calinski-Harabasz     │
│  • Ranking combinado                    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 7: K-Means final                  │
│  • Entrenamiento con k óptimo           │
│  • Heatmap de centroides                │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 8: Etiquetado y validación        │
│  • Etiquetas automáticas por centroides │
│  • Validación con burnout_level         │
│  • Test Chi² de independencia           │
│  • Radar charts + boxplots              │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 9: DBSCAN                         │
│  • K-Distance Graph                     │
│  • Selección de eps y min_samples       │
│  • Detección de ruido/outliers          │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 10: Reducción de dimensionalidad  │
│  • PCA: varianza acumulada + loadings   │
│  • t-SNE: vecindad local                │
│  • Comparativa PCA vs t-SNE             │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 11: Detección de anomalías        │
│  • Isolation Forest                     │
│  • LOF (Local Outlier Factor)           │
│  • Cruce DBSCAN vs IF+LOF              │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PASO 12-14: Resumen + Reflexión        │
│  • Panel visual comparativo             │
│  • Conclusiones integradas              │
│  • Limitaciones y mejoras propuestas    │
└─────────────────────────────────────────┘
```

---

## 🤖 Modelos Aplicados

### K-Means (Clustering Partitivo)
- **Rol:** Segmentación principal de perfiles de burnout
- **k óptimo:** Seleccionado por ranking combinado de 4 métricas (rango k=2–12)
- **Ventaja:** Clusters interpretables para políticas organizacionales

### DBSCAN (Clustering por Densidad)
- **Rol:** Detección de outliers y validación de estructura
- **Parámetros:** `eps` seleccionado por K-Distance Graph; `min_samples = n_dimensiones`
- **Hallazgo clave:** Maldición de la dimensionalidad → DBSCAN útil para outliers, no para segmentación

### Isolation Forest + LOF
- **Rol:** Detección individual de anomalías
- **Estrategia:** Consenso IF+LOF = mayor confianza en la detección
- **`contamination=0.05`** (ajustable según criterio de negocio)

### PCA + t-SNE
- **Rol:** Reducción de dimensionalidad y visualización
- **PCA:** Ejes interpretables (loadings) — varianza ~35% en 2D
- **t-SNE:** Preserva vecindad local — solo para visualización

---

## 📈 Resultados Principales

> Los valores exactos dependen de la ejecución; la estructura del modelo es determinista con `random_state=42`.

### Espacio Final del Modelo (4 variables sin leakage)

| Variable | Tipo | Rol |
|----------|------|-----|
| `caffeine_intake` | Original | Indicador de compensación energética |
| `ratio_trabajo_descanso` | Nueva | `daily_work_hours / sleep_hours` |
| `carga_operativa` | Nueva | `bugs_per_day + meetings_per_day` |
| `indice_recuperacion` | Nueva | `sleep_hours + exercise_hours` |

### Variables Excluidas del Modelo

| Variable | Razón |
|----------|-------|
| `burnout_level` | TARGET — no usar en clustering |
| `stress_level` | Codificación determinista de burnout_level (99.99% accuracy) |
| `age` | Sin poder discriminatorio (r < 0.02) |
| `experience_years` | Sin poder discriminatorio (r < 0.02) |
| `screen_time` | Redundante con `daily_work_hours` (r = 0.93) |
| `commits_per_day` | Sin poder discriminatorio (r = −0.01 con burnout) |

---

## 📁 Estructura del Repositorio

```
📦 burnout-developer-clustering/
├── 📓 KMeans_DBSCAN_anomalias_Grupo_4.ipynb   # Notebook principal
├── 📊 developer_burnout_dataset.csv            # Dataset fuente
├── 📄 README.md                                # Este archivo
│
├── 📚 wiki/
│   ├── 00_Inicio.md                           # Página principal Wiki
│   ├── 01_EDA.md                              # Análisis exploratorio
│   ├── 02_Feature_Engineering.md              # Ingeniería de variables
│   ├── 03_Preprocesamiento.md                 # Pipeline de preprocesamiento
│   ├── 04_KMeans.md                           # Modelo K-Means
│   ├── 05_DBSCAN.md                           # Modelo DBSCAN
│   ├── 06_Dimensionalidad.md                  # PCA y t-SNE
│   ├── 07_Anomalias.md                        # Detección de anomalías
│   └── 08_Conclusiones.md                     # Resultados finales
│
└── 📋 LICENSE
```

---

## ⚙️ Instalación y Uso

### Opción 1: Google Colab (recomendado)

1. Subir el notebook `KMeans_DBSCAN_anomalias_Grupo_4.ipynb` a Google Drive
2. Subir `developer_burnout_dataset.csv` a la misma carpeta
3. Ajustar `FILE_PATH` en el Paso 2 según la ruta de tu Drive
4. Ejecutar todas las celdas en orden (`Runtime > Run all`)

### Opción 2: Entorno local

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/burnout-developer-clustering.git
cd burnout-developer-clustering

# Instalar dependencias
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy missingno

# Abrir el notebook
jupyter notebook KMeans_DBSCAN_anomalias_Grupo_4.ipynb
```

### Dependencias principales

```python
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
plotly>=5.0
scikit-learn>=1.2
scipy>=1.9
missingno>=0.5
```

> **Nota:** El notebook instala `missingno` automáticamente si no está disponible.

---

## 💡 Hallazgos Clave

### 1. El burnout no es lineal — es multidimensional
El análisis confirmó que no existe una única variable causal. El **desequilibrio entre carga y recuperación** (`ratio_trabajo_descanso`) es el factor más discriminativo, no la carga laboral absoluta.

### 2. `stress_level` es un proxy del target
Con una regla de tres umbrales simples (`≤35` → Low, `<70` → Medium, `≥70` → High), `stress_level` predice `burnout_level` con **99.99% de precisión**. Incluirla habría convertido el clustering no supervisado en supervisado encubierto.

### 3. DBSCAN + maldición de la dimensionalidad
En el espacio de 4 variables del modelo, DBSCAN detecta una sola región densa. Esto no es un error — es información: el dataset sintético tiene densidad uniforme y no presenta separaciones naturales detectables por densidad.

### 4. Los factores protectores importan tanto como los de riesgo
`sleep_hours` y `exercise_hours` son los únicos dos factores con correlación **negativa** con burnout. Son los "escudos" del dataset y aparecen en dos variables compuestas del modelo.

### 5. Complementariedad de modelos
| Nivel de análisis | Herramienta | Salida |
|-------------------|-------------|--------|
| Macro (grupos) | K-Means | *k* perfiles para políticas organizacionales |
| Micro (individuos) | IF + LOF | Lista de desarrolladores para revisión urgente |
| Validación estructura | DBSCAN | Outliers de densidad que no encajan en ningún perfil |
| Comunicación | PCA | Visualización e interpretación de ejes causales |

---

## 👥 Integrantes del Grupo

| Nombre | Universidad |
|--------|-------------|
| **Eduardo Alejandro Ceballos Jijón** | UEES |
| **Guillermo Leonidas Granizo Veintimilla** | UEES |
| **José Farid Ulloa Manzur** | UEES |
| **Christian Xavier Valle Maridueña** | UEES |

---

## 📚 Referencias

- Kaggle Dataset: [Developer Burnout Prediction Dataset — 7000 Samples](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)
- Scikit-learn Documentation: [Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- Ester et al. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise.* KDD-96.
- Breunig et al. (2000). *LOF: Identifying density-based local outliers.* ACM SIGMOD.
- Liu et al. (2008). *Isolation Forest.* IEEE ICDM.

---

<div align="center">
  <sub>Proyecto académico desarrollado para la Universidad de Especialidades Espíritu Santo (UEES)</sub><br>
  <sub>Curso: Aprendizaje No Supervisado y Detección de Anomalías</sub>
</div>

