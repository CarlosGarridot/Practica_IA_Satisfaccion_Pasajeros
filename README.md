# Predicción de Satisfacción de Pasajeros de Aerolíneas

**Práctica III — Inteligencia Artificial**  
Carlos Garrido del Toro & Elena del Pilar Fernández Wyzynska  
Enero 2026

---

## Descripción

Proyecto de aprendizaje automático supervisado que predice si un pasajero de aerolínea quedará **satisfecho o insatisfecho** con su experiencia de vuelo, a partir de datos demográficos, características del viaje y valoraciones de servicios (wifi, comida, comodidad, etc.).

Se comparan cinco modelos de clasificación evaluados con múltiples métricas, incluyendo análisis de sesgo, overfitting y eficiencia computacional.

---

## Dataset

**Airline Passenger Satisfaction Dataset**  
Fuente: [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)  
- 25.976 filas × 25 columnas  
- Variable objetivo: `satisfaction` (Satisfecho / Neutral o Insatisfecho)

---

## Pipeline del Proyecto

### 1. Preprocesamiento
- Eliminación de columnas irrelevantes (`id`, `Unnamed`)
- Gestión de valores nulos: imputación por media en variables numéricas
- Codificación de variables categóricas con `LabelEncoder`
- Normalización con `StandardScaler`

### 2. Análisis Exploratorio (EDA)
- Clustering no supervisado con **K-Means** para descubrir perfiles de pasajeros
- Selección de `k=3` mediante el **Método del Codo**
- Identificación de grupos: viajeros de negocios exigentes vs. turistas relajados

### 3. Modelos Supervisados
Comparativa de cinco algoritmos de clasificación: Perceptrón, Regresión Logística, SVM, Árbol de Decisión y Random Forest.

### 4. Análisis Comparativo
- **Sesgo por clase**: F1-Score separado para clientes satisfechos e insatisfechos
- **Tipos de error**: Falsos Positivos vs. Falsos Negativos con interpretación de negocio
- **Overfitting**: Comparativa accuracy train vs. test (gap)
- **Eficiencia**: Tiempo de entrenamiento y latencia de predicción por muestra

---

## Conclusión

El **Random Forest** es el modelo ganador: mayor accuracy, menor número de Falsos Positivos (139 casos, 1.78% de riesgo de fuga) y comportamiento equilibrado entre clases. El SVM queda como segunda opción con un rendimiento muy similar pero mayor coste computacional.

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualización-4c72b0)

---

## Cómo ejecutarlo

```bash
# 1. Clona el repositorio
git clone https://github.com/tu-usuario/nombre-repo.git

# 2. Instala las dependencias
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Descarga el dataset de Kaggle y colócalo en la raíz del proyecto como:
#    airline_passenger_satisfaction.csv

# 4. Abre el notebook
jupyter notebook airline_satisfaction.ipynb
```

---
