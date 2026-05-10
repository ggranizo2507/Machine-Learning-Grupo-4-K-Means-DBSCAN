# 🔥 Clasificación de Perfiles de Burnout en Desarrolladores
### Aprendizaje No Supervisado · K-Means · DBSCAN · PCA · t-SNE · Isolation Forest · LOF

> **Universidad:** UEES — Universidad de Especialidades Espíritu Santo  
> **Asignatura:** Machine Learning — Semana 3  
> **Grupo:** 4  
> **Dataset:** [Developer Burnout Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)

---

## 📋 Tabla de Contenidos

- [Contexto del Problema](#-contexto-del-problema)
- [Pregunta de Investigación](#-pregunta-de-investigación)
- [Dataset](#-dataset)
- [Decisiones de Diseño y Justificaciones](#-decisiones-de-diseño-y-justificaciones)
- [Pipeline del Análisis](#-pipeline-del-análisis)
- [Modelos Implementados y Justificación](#-modelos-implementados-y-justificación)
- [Resultados Principales](#-resultados-principales)
- [Perfiles Identificados](#-perfiles-identificados)
- [Comparativa de Modelos](#-comparativa-de-modelos)
- [Limitaciones y Propuestas de Mejora](#-limitaciones-y-propuestas-de-mejora)
- [Recomendaciones de Negocio](#-recomendaciones-de-negocio)
- [Cómo Ejecutar](#-cómo-ejecutar)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Requerimientos](#-requerimientos)
- [Referencias](#-referencias)

---

## 🎯 Contexto del Problema

El **burnout laboral** en equipos de desarrollo de software es una problemática creciente con impacto medible en las organizaciones:

- 📉 Reduce la velocidad de entrega entre un 30 y 50 %
- 🔄 Eleva la rotación de talento (costo de reemplazo: 6–9 meses de salario)
- 🐛 Degrada la calidad del código y aumenta la deuda técnica
- 💬 Deteriora el clima de equipo y la colaboración

Las organizaciones necesitan **detectar el riesgo de forma proactiva**, antes de que el agotamiento se manifieste. Este proyecto aplica algoritmos de **aprendizaje no supervisado** para segmentar a 7.000 desarrolladores en perfiles de riesgo a partir de sus hábitos laborales, sin usar la etiqueta de clasificación, permitiendo diseñar intervenciones diferenciadas por grupo.

> **¿Por qué aprendizaje no supervisado?**  
> En producción, la etiqueta `burnout_level` no estará disponible para empleados nuevos ni en tiempo real. El clustering permite identificar perfiles de riesgo basándose únicamente en variables de comportamiento observable y actuar *antes* de que el burnout se declare.

---

## ❓ Pregunta de Investigación

> *«¿Es posible descubrir automáticamente perfiles latentes de riesgo de burnout —usando horas de trabajo, nivel de estrés, horas de sueño y carga operativa— sin utilizar la etiqueta de clasificación supervisada?»*

**Hipótesis:** Los patrones de intensidad laboral y carga operativa son suficientes para separar grupos de desarrolladores con distinto nivel de riesgo, incluso sin información explícita sobre el burnout declarado.

**Validación:** La etiqueta `burnout_level` se reserva para **verificación posterior**. Si los clusters descubiertos coinciden con los niveles reales, el modelo es coherente con la realidad. Esto no es entrenamiento supervisado; es validación.

---

## 🗂️ Dataset

| Atributo | Valor |
|---|---|
| **Fuente** | Kaggle — Developer Burnout Prediction Dataset |
| **Registros** | 7.000 desarrolladores |
| **Variables totales** | 12 |
| **Variables usadas en el modelo** | 11 (todas numéricas) |
| **Variable excluida** | `burnout_level` — target reservado para validación |
| **Valores nulos** | 140 filas (2 %) en todas las columnas — patrón MCAR |
| **Estrategia de imputación** | Mediana para numéricas, moda para categóricas |

### Descripción de variables

| Variable | Descripción | Rol en el modelo |
|---|---|---|
| `age` | Edad del desarrollador | Feature |
| `experience_years` | Años de experiencia laboral | Feature |
| `daily_work_hours` | Horas de trabajo diarias | Feature |
| `sleep_hours` | Horas de sueño por noche | Feature |
| `caffeine_intake` | Consumo de cafeína (tazas/día) | Feature |
| `bugs_per_day` | Bugs reportados o resueltos por día | Feature |
| `commits_per_day` | Commits al repositorio por día | Feature |
| `meetings_per_day` | Reuniones por día | Feature |
| `screen_time` | Horas frente a pantalla | Feature |
| `exercise_hours` | Horas de ejercicio semanal | Feature |
| `stress_level` | Nivel de estrés (escala 0–100) | Feature |
| `burnout_level` | Low / Medium / High | **TARGET — excluida del entrenamiento** |

### Hallazgos clave del EDA

| Par de variables | Correlación r | Implicación para el modelo |
|---|---|---|
| `daily_work_hours` ↔ `screen_time` | **+0.93** | Redundancia casi perfecta — candidata a eliminación |
| `daily_work_hours` ↔ `stress_level` | **+0.60** | Horas = predictor más fuerte de estrés |
| `screen_time` ↔ `stress_level` | **+0.55** | Confirma cadena: pantalla → estrés |
| `bugs_per_day` ↔ `stress_level` | **+0.49** | Carga operativa eleva el estrés |
| `meetings_per_day` ↔ `stress_level` | **+0.35** | Interrupciones elevan el estrés |
| `sleep_hours` ↔ `stress_level` | **−0.25** | Sueño tiene efecto protector |
| `age` y `experience_years` | **< 0.03** con todo | No predicen burnout en este dataset |

---

## 🧠 Decisiones de Diseño y Justificaciones

Cada decisión metodológica tiene una justificación técnica explícita:

| Decisión | Alternativa descartada | Justificación |
|---|---|---|
| **Excluir `burnout_level`** | Incluirla como feature | Sería _data leakage_: el modelo re-descubriría la etiqueta en lugar de encontrar patrones latentes |
| **`OneHotEncoder`** en lugar de `LabelEncoder` | LabelEncoder | LabelEncoder impone orden artificial entre categorías, distorsionando las distancias euclidianas de K-Means y DBSCAN |
| **Diagnóstico automático de Scaler** | Aplicar siempre StandardScaler | Con outliers o asimetría, RobustScaler es más apropiado; el diagnóstico con IQR + Shapiro decide objetivamente |
| **4 métricas para elegir k** | Solo el método del codo | El codo puede ser ambiguo; el ranking combinado de 4 métricas reduce el sesgo de depender de una sola |
| **DBSCAN después de K-Means** | Solo uno de los dos | Son complementarios: K-Means segmenta toda la población; DBSCAN detecta perfiles atípicos que no encajan en ningún grupo |
| **Consenso IF + LOF** | Solo un detector de anomalías | Un punto marcado por ambos métodos tiene mayor confianza de ser una anomalía real y no un falso positivo |

---

## 🗺️ Pipeline del Análisis

```
developer_burnout_dataset.csv  (7.000 × 12)
              │
              ▼
  ┌───────────────────────────┐
  │  1. EDA Completo          │  Estadísticas · NaN · Distribuciones · Correlaciones
  └─────────────┬─────────────┘
                │
                ▼
  ┌───────────────────────────┐
  │  2. Preprocesamiento      │  Imputación mediana · OneHotEncoder · Scaler adaptativo
  └─────────────┬─────────────┘
                │
                ▼
  ┌───────────────────────────┐
  │  3. Selección de k        │  Codo · Silhouette · Davies-Bouldin · Calinski-Harabasz
  └─────────────┬─────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
    ┌─────────┐   ┌─────────┐
    │ K-Means │   │  DBSCAN │   Clustering
    └────┬────┘   └────┬────┘
         └──────┬──────┘
                ▼
  ┌───────────────────────────┐
  │  4. Reducción Dimensional │  PCA (lineal) · t-SNE (no lineal) · Comparativa
  └─────────────┬─────────────┘
                │
                ▼
  ┌───────────────────────────┐
  │  5. Detección Anomalías   │  Isolation Forest · LOF · Consenso IF+LOF
  └─────────────┬─────────────┘
                │
                ▼
  ┌───────────────────────────┐
  │  6. Perfiles + Reflexión  │  Etiquetado · Validación · Recomendaciones · Limitaciones
  └───────────────────────────┘
```

---

## ⚙️ Modelos Implementados y Justificación

### K-Means
Algoritmo de partición que asigna cada punto al centroide más cercano de forma iterativa. Elegido como modelo principal de segmentación porque produce clusters interpretables con perfiles bien definidos y escala eficientemente con 7.000 registros.

- `init='k-means++'` → inicialización mejorada que evita mínimos locales
- `n_init=20` → 20 reiniciaciones para robustez estadística
- `k óptimo = 2` → seleccionado por ranking combinado de 4 métricas en rango 2–12

### DBSCAN
Algoritmo de clustering por densidad. No requiere especificar k y detecta outliers de forma nativa. Complementa a K-Means: donde K-Means segmenta, DBSCAN detecta los perfiles que no encajan en ningún grupo.

- `eps` → determinado automáticamente por K-Distance Graph
- `min_samples = 5` → heurística estándar para datasets medianos
- **Resultado:** 1 cluster denso + ~5 % de ruido (maldición de la dimensionalidad en 11D)

### PCA
Reducción lineal que maximiza la varianza capturada. Permite visualizar clusters en 2D e interpretar qué variables dominan cada eje mediante los loadings.

- **PC1 (~22 %)** = Eje de Intensidad Laboral (`daily_work_hours`, `screen_time`, `stress_level`)
- **PC2 (~12 %)** = Eje de Carga Operativa (`bugs_per_day`, `meetings_per_day`)
- **Total 2D: ~34.5 %** de varianza → resultado esperado con variables independientes

### t-SNE
Reducción no lineal que preserva la vecindad local. Mayor fidelidad al clustering 11D que PCA. Usada para visualización en presentaciones y comparativa cuantitativa de fidelidad.

- `perplexity=30`, `learning_rate=200`, `random_state=42`

### Isolation Forest + LOF
Dos métodos complementarios de detección de anomalías con enfoques distintos:

| Método | Enfoque | Detecta |
|---|---|---|
| **Isolation Forest** | Árboles aleatorios: los puntos fáciles de aislar son anómalos | Anomalías globales |
| **LOF** | Densidad local: menor densidad que los vecinos = anómalo | Anomalías locales relativas al entorno |

**Consenso IF+LOF:** ~2.5 % del dataset (≈175 desarrolladores) → máxima prioridad de revisión.

---

## 📊 Resultados Principales

| Modelo / Métrica | Resultado | Interpretación |
|---|---|---|
| K-Means — k óptimo | **2** | Dos perfiles principales: equilibrado y sobrecargado |
| K-Means — Silhouette | **~0.14** | Solapamiento esperado con variables independientes |
| K-Means — Davies-Bouldin | mínimo en k=2 | Clusters más distintos entre sí con k=2 |
| DBSCAN — clusters | **1 + ruido** | Densidad uniforme en 11D — útil solo para outliers |
| PCA — varianza 2D | **~34.5 %** | Se necesitan 6–7 PCs para el 80 % |
| t-SNE — fidelidad | **mayor que PCA** | Preserva mejor la vecindad local |
| Isolation Forest | **~5 %** ≈ 350 devs | Anomalías globales detectadas |
| LOF | **~5 %** ≈ 350 devs | Anomalías locales detectadas |
| **Consenso IF+LOF** | **~2.5 %** ≈ 175 devs | **Máxima prioridad de atención** |

---

## 👥 Perfiles Identificados

### 🔴 Perfil Sobrecargado — Alto Riesgo de Burnout

| Variable | Valor típico |
|---|---|
| `daily_work_hours` | Alto ≈ 10–11 h/día |
| `screen_time` | Alto ≈ 10–11 h/día |
| `stress_level` | Alto ≈ 70–80 / 100 |
| `sleep_hours` | Bajo ≈ 5–6 h/noche |
| `exercise_hours` | Bajo |
| `bugs_per_day` | Alto |
| `meetings_per_day` | Alto |

Jornadas extendidas, alta carga operativa, estrés elevado, hábitos de descanso deteriorados. Concentran la mayoría de registros con `burnout_level = 'High'`.

### 🟢 Perfil Equilibrado — Bajo Riesgo de Burnout

| Variable | Valor típico |
|---|---|
| `daily_work_hours` | Moderado-bajo ≈ 6–7 h/día |
| `stress_level` | Bajo ≈ 30–40 / 100 |
| `sleep_hours` | Alto ≈ 7–8 h/noche |
| `exercise_hours` | Alto |
| `commits_per_day` | Moderado-alto |

Carga laboral sostenible, buenos hábitos de descanso, estrés controlado. Alta productividad técnica sin sacrificar bienestar.

### 🟡 Perfil Intermedio — si k=3

Carga media-alta, estrés moderado, mejor gestión del descanso que el perfil sobrecargado. Posibles desarrolladores con mayor resiliencia o mejores condiciones laborales. Perfil tentativo: *"Señales tempranas — monitoreo preventivo recomendado"*.

---

## ⚖️ Comparativa de Modelos

| Modelo | Propósito en este análisis | Fortaleza | Limitación |
|---|---|---|---|
| **K-Means** | Segmentar toda la población | Clusters interpretables y accionables | Silhouette bajo — solapamiento entre grupos |
| **DBSCAN** | Detectar perfiles atípicos | Nativo para outliers, sin k predefinido | 1 solo cluster en 11D — densidad uniforme |
| **PCA** | Entender qué variables dominan | Loadings interpretables | Solo 34.5 % varianza en 2 componentes |
| **t-SNE** | Visualizar agrupamientos complejos | Mayor fidelidad al clustering 11D | Ejes sin interpretación directa |
| **Isolation Forest** | Anomalías globales | Rápido, escala con 7.000 registros | Solo detecta extremos globales |
| **LOF** | Anomalías locales relativas | Detecta lo que IF no ve | Sensible al parámetro `n_neighbors` |

> **K-Means y DBSCAN son complementarios, no competidores. PCA y t-SNE también. IF y LOF se validan mutuamente.**

---

## 🛠️ Limitaciones y Propuestas de Mejora

### 1 — Redundancia `daily_work_hours` ↔ `screen_time` (r = 0.93)
**Problema:** Duplica el peso del eje de horas en K-Means.  
**Mejora:** Eliminar `screen_time` o crear `intensidad_laboral = (daily_work_hours + screen_time) / 2`.

### 2 — Silhouette Score bajo (~0.14)
**Problema:** Variables estadísticamente independientes → sin fronteras nítidas.  
**Mejora:** Pipeline PCA (6 componentes = ~80 % varianza) → KMeans. Mejora esperada: 0.14 → 0.20–0.25.

### 3 — DBSCAN con 1 solo cluster en 11 dimensiones
**Problema:** Maldición de la dimensionalidad homogeneiza las distancias.  
**Mejora:** Reducir a 3–4 PCs antes de DBSCAN para distancias más discriminativas.

### 4 — Validación solo descriptiva
**Problema:** Sin prueba estadística de que los clusters diferencian significativamente el burnout.  
**Mejora:** Test Chi² entre `Cluster_KMeans` y `burnout_level` (p < 0.05 = diferencia real).

### 5 — Dataset sintético
**Problema:** Distribuciones uniformes y ausencia de outliers sugieren datos simulados.  
**Mejora:** Validar en encuestas reales con ruido natural y sesgo de respuesta.

---

## 💼 Recomendaciones de Negocio

| Acción | Basada en | Grupo objetivo |
|---|---|---|
| Monitoreo preventivo continuo | K-Means | Cluster con mayor `stress_level` promedio |
| Revisión de distribución de carga | K-Means + LOF | Cluster con mayor `daily_work_hours` |
| Entrevista 1:1 urgente con RR.HH. | Consenso IF+LOF | ~175 developers marcados por ambos métodos |
| Investigar posible sub-reporte de estrés | LOF solo | Anomalías con bajo estrés pero muchas horas |
| Replicar condiciones del perfil equilibrado | K-Means | Cluster de bienestar como política de referencia |

---

## ▶️ Cómo Ejecutar

### Opción A — Google Colab (recomendado)

```
1. Subir developer_burnout_dataset.csv a My Drive/MachineLearning/Semana3/
2. Abrir KMeans_DBSCAN_anomalias_Grupo_4.ipynb en Google Colab
3. Runtime > Run all
```

### Opción B — Entorno local

```bash
git clone https://github.com/[tu-usuario]/burnout-clustering-grupo4.git
cd burnout-clustering-grupo4

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook KMeans_DBSCAN_anomalias_Grupo_4.ipynb
```

> ⚠️ Ajustar `FILE_PATH` en la celda del Paso 2 si se ejecuta localmente.

---

## 📁 Estructura del Repositorio

```
burnout-clustering-grupo4/
│
├── KMeans_DBSCAN_anomalias_Grupo_4.ipynb   ← Notebook principal (81 celdas)
├── developer_burnout_dataset.csv            ← Dataset (7.000 × 12)
├── requirements.txt                         ← Dependencias Python
├── README.md                                ← Este archivo
│
└── wiki/
    ├── 01-contexto-y-dataset.md
    ├── 02-eda-y-preprocesamiento.md
    ├── 03-kmeans.md
    ├── 04-dbscan.md
    ├── 05-pca-tsne.md
    ├── 06-anomalias.md
    ├── 07-perfiles-y-conclusiones.md
    └── 08-limitaciones-y-mejoras.md
```

---

## 📦 Requerimientos

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
scikit-learn>=1.1.0
scipy>=1.9.0
missingno>=0.5.1
```

---

## 📚 Referencias

- Kaggle: [Developer Burnout Prediction Dataset](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)
- Scikit-learn: [Clustering User Guide](https://scikit-learn.org/stable/modules/clustering.html)
- Van der Maaten, L. & Hinton, G. (2008). *Visualizing Data using t-SNE*. JMLR.
- Liu, F.T., Ting, K.M. & Zhou, Z.H. (2008). *Isolation Forest*. ICDM.
- Breunig, M.M. et al. (2000). *LOF: Identifying Density-Based Local Outliers*. SIGMOD.
- Rousseeuw, P.J. (1987). *Silhouettes: A graphical aid to interpretation and validation of cluster analysis*.

---

> *«Ningún modelo cuenta la historia completa por sí solo. El analista conecta todos los fragmentos con criterio de negocio y propone acciones concretas.»*
