# 🎓 Sistema Predictivo de Retención Estudiantil

> **Ciclo de vida completo del dato**: ETL → EDA → Machine Learning → Despliegue Web

Sistema de alerta temprana que predice si un estudiante está en riesgo académico, permitiendo a coordinadores intervenir a tiempo. Construido con Python, scikit-learn y Streamlit.

---

## 📁 Estructura del Proyecto

```
student_retention/
│
├── data/
│   ├── student_data.csv          # Dataset original (395 estudiantes, UCI)
│   ├── student_clean.csv         # Dataset limpio post-ETL
│   └── student_encoded.csv       # Dataset codificado para ML
│
├── models/
│   ├── random_forest.pkl         # Modelo Random Forest entrenado
│   ├── regresión_logística.pkl   # Modelo Regresión Logística
│   ├── feature_names.pkl         # Nombres de features del modelo
│   └── *_metrics.json            # Métricas de evaluación
│
├── notebooks/
│   ├── eda_plots/                # Visualizaciones del EDA (6 gráficos)
│   └── ml_plots/                 # Gráficos de evaluación de modelos
│
├── src/
│   ├── etl.py                    # Fase 1: Extracción, Transformación y Carga
│   ├── eda.py                    # Fase 2: Análisis Exploratorio de Datos
│   ├── modeling.py               # Fase 3: Entrenamiento y Evaluación ML
│   ├── predictor.py              # Servicio de predicción (inference)
│   └── train_pipeline.py         # Script maestro (ejecuta Fases 1-3)
│
├── app/
│   └── streamlit_app.py          # Fase 4: Aplicación Web (Streamlit)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/student-retention.git
cd student-retention
```

### 2. Crear entorno virtual

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Entrenar el modelo (Fases 1-3: ETL + EDA + ML)

```bash
python src/train_pipeline.py
```

Este comando ejecuta automáticamente:
- **ETL**: limpieza, codificación y creación de la variable objetivo
- **EDA**: generación de 6 gráficos de análisis exploratorio
- **ML**: entrena Regresión Logística y Random Forest, evalúa y guarda el mejor

### 5. Lanzar la aplicación web (Fase 4)

```bash
streamlit run app/streamlit_app.py
```

Abre tu navegador en `http://localhost:8501`

---

## 🔬 Fases del Proyecto

### Fase 1 — ETL y Limpieza (`src/etl.py`)
- Carga del dataset CSV con pandas
- Reporte de calidad de datos (nulos, duplicados, outliers)
- Codificación de variables binarias (yes/no → 0/1)
- One-Hot Encoding para variables categóricas (school, Mjob, etc.)
- Winsorización de outliers en variable `absences`
- Creación de variable objetivo: `at_risk = 1 si G3 < 10`

### Fase 2 — EDA (`src/eda.py`)
Genera 6 visualizaciones clave:
1. Distribución de la variable objetivo (balanceo de clases)
2. Distribución de notas G1/G2/G3 por nivel de riesgo
3. Ausencias vs riesgo (boxplot + scatter)
4. Factores socioeconómicos (educación parental)
5. Mapa de correlaciones con la variable objetivo
6. Hábitos de estudio y reprobaciones previas

**Hallazgos principales:**
- G1 y G2 son los predictores más fuertes (correlación > 0.85 con at_risk)
- Estudiantes con ≥1 reprobación tienen 70%+ de probabilidad de riesgo
- Mayor educación de los padres reduce el riesgo hasta en 30%
- Ausencias > 10 son un indicador temprano de riesgo

### Fase 3 — Modelado ML (`src/modeling.py`)
| Métrica | Regresión Logística | Random Forest |
|---------|---------------------|---------------|
| Accuracy | 88.6% | 91.1% |
| Precision (Riesgo) | 79.3% | 85.2% |
| **Recall (Riesgo)** | **88.5%** | **88.5%** |
| F1-Score (Riesgo) | 83.6% | 86.8% |
| **ROC-AUC** | **0.979** | 0.968 |
| CV F1-macro | 0.886 ± 0.038 | 0.902 ± 0.036 |

> **Modelo seleccionado: Regresión Logística** por mejor ROC-AUC (0.979).
> En sistemas de alerta temprana, el Recall alto es crítico: mejor identificar más
> estudiantes en riesgo (incluso con algunos falsos positivos) que pasar por alto
> casos reales.

### Fase 4 — Despliegue (`app/streamlit_app.py`)
Aplicación web con 3 módulos:
- **Predicción Individual**: formulario con 20+ variables → resultado en tiempo real
- **Predicción Masiva**: sube un CSV y descarga las predicciones
- **Análisis Exploratorio**: galería interactiva de todos los gráficos generados

---

## 📊 Dataset

**Fuente**: [Student Performance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

| Característica | Detalle |
|---|---|
| Registros | 395 estudiantes |
| Variables | 33 (académicas, familiares, sociales) |
| Target | `G3 < 10` → En Riesgo (32.9% de los estudiantes) |
| Nulos | 0 (dataset de alta calidad) |

---

## 🛠 Stack Tecnológico

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.10+ |
| Data Wrangling | pandas, numpy |
| Visualización | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Persistencia | joblib |
| Web App | Streamlit |
| Control de versiones | Git + GitHub |

---

## 🧪 Uso Programático (sin interfaz)

```python
from src.predictor import load_artifacts, predict_single

model, features, metrics = load_artifacts()

student = {
    "age": 17, "Medu": 2, "Fedu": 1, "studytime": 1,
    "failures": 1, "absences": 12, "G1": 7, "G2": 8,
    "internet": "no", "higher": "no", "schoolsup": "no",
    # ... resto de variables
}

result = predict_single(student, model, features)
print(result)
# {'at_risk': True, 'probability': 0.82, 'risk_level': 'Alto', ...}
```

---

## 📌 Estructura de Ramas (Git)

```
main              ← versión estable, lista para producción
├── feature/etl   ← desarrollo Fase 1
├── feature/eda   ← desarrollo Fase 2
├── feature/ml    ← desarrollo Fase 3
└── feature/app   ← desarrollo Fase 4 (Streamlit)
```

---

