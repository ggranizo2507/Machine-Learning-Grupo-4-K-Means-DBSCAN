# README.md — Proyecto de Machine Learning No Supervisado

# 🔥 Clasificación de perfiles de Burnout del Desarrollador

## Clustering, Reducción de Dimensionalidad y Detección de Anomalías con K-Means, DBSCAN, PCA, t-SNE, Isolation Forest y LOF

---

## 📌 Descripción del Proyecto

Este proyecto implementa un pipeline completo de análisis de datos y Machine Learning no supervisado aplicado al estudio del burnout en desarrolladores de software.

El objetivo principal es identificar perfiles de comportamiento, agrupaciones naturales y patrones atípicos dentro de un conjunto de datos relacionado con condiciones laborales, estrés, satisfacción y hábitos de trabajo.

El notebook integra:

* Exploración y análisis estadístico de datos (EDA)
* Preprocesamiento avanzado
* Clustering con K-Means
* Clustering por densidad con DBSCAN
* Reducción de dimensionalidad con PCA y t-SNE
* Detección de anomalías con Isolation Forest y Local Outlier Factor (LOF)
* Interpretabilidad de clusters
* Visualización avanzada
* Validación analítica de resultados

---

# 🎯 Objetivos del Proyecto

## Objetivo General

Identificar perfiles de burnout y patrones anómalos en desarrolladores de software mediante técnicas de aprendizaje no supervisado.

## Objetivos Específicos

* Analizar relaciones entre variables asociadas al burnout.
* Identificar agrupaciones naturales de desarrolladores.
* Detectar perfiles de alto riesgo.
* Evaluar diferencias entre algoritmos de clustering.
* Reducir dimensionalidad para visualización e interpretación.
* Detectar anomalías relevantes mediante técnicas de outlier detection.
* Interpretar resultados desde una perspectiva analítica y de negocio.

---

# 🧠 Justificación del Proyecto

El burnout en desarrolladores de software representa un problema crítico dentro de la industria tecnológica debido a factores como:

* Sobrecarga laboral
* Jornadas extensas
* Fatiga cognitiva
* Alta presión operativa
* Estrés continuo
* Baja satisfacción laboral

Muchas organizaciones poseen grandes volúmenes de datos relacionados con desempeño y bienestar, pero no cuentan con mecanismos analíticos para descubrir patrones ocultos.

El aprendizaje no supervisado permite:

* Detectar grupos de comportamiento sin etiquetas previas
* Identificar perfiles de riesgo
* Descubrir estructuras ocultas en los datos
* Encontrar anomalías difíciles de detectar manualmente

Este proyecto demuestra cómo aplicar técnicas modernas de ciencia de datos para transformar datos operativos en conocimiento útil para la toma de decisiones.

---

# 🗂 Dataset

## Fuente

Dataset utilizado:

`developer_burnout_dataset.csv`

El conjunto de datos contiene variables relacionadas con:

* Estrés laboral
* Horas de trabajo
* Horas frente a pantalla
* Bugs resueltos
* Reuniones
* Satisfacción laboral
* Balance vida/trabajo
* Fatiga
* Carga operativa
* Otros indicadores asociados al burnout

---

# ⚙️ Tecnologías Utilizadas

| Tecnología       | Uso                       |
| ---------------- | ------------------------- |
| Python           | Lenguaje principal        |
| Pandas           | Manipulación de datos     |
| NumPy            | Operaciones numéricas     |
| Matplotlib       | Visualización             |
| Seaborn          | Visualización estadística |
| Scikit-Learn     | Modelado ML               |
| PCA              | Reducción dimensional     |
| t-SNE            | Visualización no lineal   |
| K-Means          | Clustering                |
| DBSCAN           | Clustering por densidad   |
| Isolation Forest | Detección de anomalías    |
| LOF              | Local Outlier Factor      |
| Google Colab     | Desarrollo del notebook   |

---

# 🏗 Estructura del Proyecto

```text
📦 burnout-clustering-project
 ┣ 📂 data
 ┃ ┗ 📄 developer_burnout_dataset.csv
 ┣ 📂 notebooks
 ┃ ┗ 📄 KMeans_DBSCAN_anomalias_Grupo_4.ipynb
 ┣ 📂 images
 ┃ ┣ 📄 elbow_method.png
 ┃ ┣ 📄 silhouette_analysis.png
 ┃ ┣ 📄 heatmap_centroids.png
 ┃ ┣ 📄 pca_projection.png
 ┃ ┣ 📄 tsne_projection.png
 ┃ ┗ 📄 anomaly_detection.png
 ┣ 📂 wiki
 ┃ ┣ 📄 Home.md
 ┃ ┣ 📄 EDA.md
 ┃ ┣ 📄 Clustering.md
 ┃ ┣ 📄 DBSCAN.md
 ┃ ┣ 📄 PCA_TSNE.md
 ┃ ┣ 📄 Anomaly_Detection.md
 ┃ ┗ 📄 Conclusions.md
 ┣ 📄 README.md
 ┣ 📄 requirements.txt
 ┗ 📄 LICENSE
```

---

# 🔍 Metodología Implementada

## 1️⃣ Análisis Exploratorio de Datos (EDA)

Se realiza:

* Revisión de tipos de datos
* Análisis descriptivo
* Distribuciones
* Correlaciones
* Identificación de valores nulos
* Identificación de outliers

### Hallazgo relevante

El dataset presenta correlaciones relativamente bajas entre variables, indicando independencia parcial entre factores asociados al burnout.

Esto influye directamente en:

* Separabilidad de clusters
* Comportamiento de PCA
* Eficiencia de DBSCAN

---

## 2️⃣ Preprocesamiento

### Técnicas aplicadas

* OneHotEncoder
* ColumnTransformer
* Imputación de valores
* Escalado adaptativo
* StandardScaler
* RobustScaler

### Justificación

El escalado es fundamental debido a:

* Diferencias de magnitud entre variables
* Sensibilidad de K-Means a distancias
* Sensibilidad de DBSCAN a densidades
* Sensibilidad de PCA a varianza

---

## 3️⃣ Clustering con K-Means

### Selección de k óptimo

El notebook evalúa valores de:

```python
k = 2 → 12
```

### Métricas utilizadas

| Métrica           | Objetivo                           |
| ----------------- | ---------------------------------- |
| Inercia           | Método del codo                    |
| Silhouette Score  | Separación entre clusters          |
| Davies-Bouldin    | Minimizar similitud entre clusters |
| Calinski-Harabasz | Maximizar separación               |

### Resultado

Se construye un ranking combinado para seleccionar automáticamente el mejor valor de k.

---

## 4️⃣ Interpretación de Clusters

Los centroides son interpretados mediante heatmaps.

Ejemplo de perfiles:

| Cluster   | Interpretación                  |
| --------- | ------------------------------- |
| Cluster 0 | Bajo estrés y alta satisfacción |
| Cluster 1 | Riesgo moderado                 |
| Cluster 2 | Alto burnout                    |
| Cluster 3 | Alta carga operativa            |

### Importancia

El algoritmo genera grupos matemáticos, pero el analista les otorga significado de negocio.

---

## 5️⃣ DBSCAN

### Objetivo

Detectar estructuras densas y ruido sin necesidad de definir k.

### Parámetros principales

```python
DBSCAN(eps=?, min_samples=?)
```

### Hallazgo importante

Debido a la alta dimensionalidad del dataset:

* DBSCAN tiende a detectar una nube densa uniforme
* Se observa el efecto de la maldición de la dimensionalidad
* K-Means ofrece mejor segmentación práctica para este dataset

### Conclusión metodológica

El comportamiento de DBSCAN es un resultado válido y aporta valor analítico.

---

## 6️⃣ PCA y t-SNE

### PCA

Se utiliza para:

* Reducir dimensionalidad
* Interpretar ejes principales
* Analizar loadings

### Hallazgo relevante

PCA captura aproximadamente:

```text
34.5% de la varianza total
```

Esto indica:

* Variables relativamente independientes
* Baja redundancia estructural
* Alta complejidad multivariable

### Interpretación de componentes

| Componente | Interpretación     |
| ---------- | ------------------ |
| PC1        | Intensidad laboral |
| PC2        | Carga operativa    |

---

### t-SNE

Se utiliza para:

* Visualización no lineal
* Preservar vecindad local
* Analizar separación visual de clusters

### Diferencia clave

| Técnica | Preserva        |
| ------- | --------------- |
| PCA     | Varianza global |
| t-SNE   | Vecindad local  |

---

## 7️⃣ Detección de Anomalías

### Algoritmos utilizados

* Isolation Forest
* Local Outlier Factor (LOF)

### Objetivo

Identificar:

* Desarrolladores con comportamiento extremo
* Posibles perfiles críticos
* Casos de burnout atípico

### Análisis realizado

* Comparación entre normales y anomalías
* Desviación respecto a medias
* Consenso IF + LOF
* Distribución de burnout_level

---

# 📊 Principales Hallazgos

## Insights relevantes

### 🔹 Los clusters identifican perfiles diferenciados de burnout

Los modelos permiten segmentar desarrolladores según:

* Estrés
* Carga laboral
* Satisfacción
* Fatiga
* Actividad operativa

---

### 🔹 K-Means funciona mejor que DBSCAN en este dataset

Debido a:

* Alta dimensionalidad
* Baja densidad diferenciada
* Variables parcialmente independientes

---

### 🔹 PCA confirma complejidad estructural

La baja varianza explicada indica:

* Burnout multifactorial
* No existe una sola dimensión dominante

---

### 🔹 Las anomalías representan perfiles críticos

Los outliers detectados pueden representar:

* Casos de agotamiento extremo
* Sobrecarga laboral severa
* Conductas operativas inusuales

---

# 📈 Aplicaciones Reales

Este proyecto puede utilizarse en:

* Recursos Humanos
* People Analytics
* Gestión del Talento
* Prevención de burnout
* Salud ocupacional
* Gestión de productividad
* Analítica organizacional

---

# 🚀 Cómo Ejecutar el Proyecto

## 1️⃣ Clonar repositorio

```bash
git clone https://github.com/usuario/burnout-clustering-project.git
```

## 2️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

## 3️⃣ Ejecutar notebook

Abrir:

```text
KMeans_DBSCAN_anomalias_Grupo_4.ipynb
```

En:

* Google Colab
* Jupyter Notebook
* VSCode

---

# 📦 requirements.txt

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
notebook
```

---

# 📚 Propuesta de Wiki para GitHub

---

# 📄 Home.md

# Bienvenido a la Wiki del Proyecto

Esta wiki documenta el desarrollo completo del proyecto de clustering y detección de anomalías aplicado al burnout de desarrolladores.

## Contenido

* EDA
* Preprocesamiento
* K-Means
* DBSCAN
* PCA y t-SNE
* Detección de anomalías
* Resultados
* Conclusiones

---

# 📄 EDA.md

# Exploratory Data Analysis (EDA)

## Objetivo

Comprender la estructura del dataset antes del modelado.

## Actividades realizadas

* Estadística descriptiva
* Valores faltantes
* Correlaciones
* Distribuciones
* Detección de outliers

## Hallazgos

* Correlaciones bajas
* Variables relativamente independientes
* Dataset apto para clustering

---

# 📄 Clustering.md

# Clustering con K-Means

## Selección de k

Se evaluaron métricas:

* Silhouette
* Inercia
* Davies-Bouldin
* Calinski-Harabasz

## Resultado

El mejor k se seleccionó automáticamente mediante ranking combinado.

## Interpretación

Los clusters representan perfiles diferenciados de burnout.

---

# 📄 DBSCAN.md

# DBSCAN

## Objetivo

Detectar agrupaciones por densidad y ruido.

## Hallazgo

DBSCAN identificó una nube densa uniforme.

## Interpretación

La alta dimensionalidad dificulta la separación por densidad.

---

# 📄 PCA_TSNE.md

# PCA y t-SNE

## PCA

Reduce dimensionalidad preservando varianza.

## t-SNE

Preserva vecindad local para visualización.

## Hallazgo

PCA explicó aproximadamente 34.5% de la varianza.

---

# 📄 Anomaly_Detection.md

# Detección de Anomalías

## Modelos

* Isolation Forest
* LOF

## Objetivo

Detectar perfiles extremos de burnout.

## Hallazgo

Las anomalías presentan diferencias importantes respecto a los perfiles normales.

---

# 📄 Conclusions.md

# Conclusiones

## Conclusiones técnicas

* K-Means fue el modelo más útil.
* DBSCAN evidenció limitaciones por dimensionalidad.
* PCA mostró baja redundancia estructural.
* Las anomalías representan perfiles críticos.

## Conclusiones de negocio

* El burnout puede analizarse mediante ML no supervisado.
* Existen perfiles diferenciados de desarrolladores.
* La analítica puede apoyar decisiones de bienestar organizacional.

---

# 🔮 Mejoras Futuras

## Posibles extensiones

* Modelos de Deep Learning
* Autoencoders para anomalías
* Clustering jerárquico
* UMAP
* Dashboards interactivos
* Integración con Power BI
* Predicción supervisada de burnout

---

# 👨‍💻 Autor

Proyecto académico de Machine Learning No Supervisado aplicado al análisis de burnout en desarrolladores.

---

# 📄 Licencia

Este proyecto puede distribuirse bajo licencia MIT.

