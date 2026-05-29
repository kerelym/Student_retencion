"""
Fase 4: Servicio de Predicción
Carga el modelo entrenado y expone funciones de inferencia para la app web.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "models"
RISK_THRESHOLD = 10
BINARY_MAP = {"yes": 1, "no": 0}


def load_artifacts(models_dir: Path = MODELS_DIR):
    """Carga el mejor modelo y la lista de features."""
    # Prefiere Random Forest, si no existe usa Regresión Logística
    for name in ["random_forest", "regresión_logística", "regresi_n_log_stica"]:
        pkl = models_dir / f"{name}.pkl"
        if pkl.exists():
            model = joblib.load(pkl)
            features = joblib.load(models_dir / "feature_names.pkl")
            metrics_path = models_dir / f"{name}_metrics.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            return model, features, metrics

    raise FileNotFoundError(
        f"No se encontró ningún modelo en {models_dir}. "
        "Ejecuta primero `python src/train_pipeline.py`."
    )


def _preprocess_input(student_dict: dict, feature_names: list) -> pd.DataFrame:
    """
    Convierte un diccionario con los datos de un estudiante
    al DataFrame con el mismo formato que el modelo espera.
    """
    df = pd.DataFrame([student_dict])

    # Codificación binaria yes/no
    bool_cols = [c for c in df.columns
                 if df[c].iloc[0] in ("yes", "no", 1, 0)]
    for col in bool_cols:
        if df[col].iloc[0] in ("yes", "no"):
            df[col] = df[col].map(BINARY_MAP)

    # One-Hot Encoding para categóricas restantes
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Alinear columnas con el modelo
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def predict_single(student_dict: dict, model=None, feature_names=None) -> dict:
    """
    Predice el riesgo de un único estudiante.

    Args:
        student_dict: dict con los campos del formulario
        model: modelo ya cargado (opcional, lo carga si no se pasa)
        feature_names: lista de features (opcional)

    Returns:
        dict con:
          - at_risk (bool)
          - probability (float 0-1, probabilidad de estar en riesgo)
          - risk_level ("Alto" | "Medio" | "Bajo")
          - recommendation (str)
    """
    if model is None:
        model, feature_names, _ = load_artifacts()

    X = _preprocess_input(student_dict, feature_names)
    prob = model.predict_proba(X)[0][1]
    at_risk = bool(prob >= 0.5)

    if prob >= 0.70:
        risk_level = "Alto"
        recommendation = (
            "⚠️ Intervención inmediata recomendada. "
            "Coordinar con tutores y servicios de apoyo académico."
        )
    elif prob >= 0.45:
        risk_level = "Medio"
        recommendation = (
            "📋 Monitorear de cerca. "
            "Programar reunión de seguimiento con el estudiante."
        )
    else:
        risk_level = "Bajo"
        recommendation = (
            "✅ Estudiante en buen camino. "
            "Mantener el acompañamiento regular."
        )

    return {
        "at_risk": at_risk,
        "probability": round(float(prob), 4),
        "probability_pct": round(float(prob) * 100, 1),
        "risk_level": risk_level,
        "recommendation": recommendation,
    }


def predict_batch(df_input: pd.DataFrame, model=None, feature_names=None) -> pd.DataFrame:
    """
    Predice el riesgo para un DataFrame completo (carga CSV masivo).

    Returns:
        DataFrame original + columnas: probability, risk_level, recommendation
    """
    if model is None:
        model, feature_names, _ = load_artifacts()

    results = []
    for _, row in df_input.iterrows():
        try:
            res = predict_single(row.to_dict(), model, feature_names)
        except Exception as e:
            res = {
                "at_risk": None,
                "probability": None,
                "probability_pct": None,
                "risk_level": "Error",
                "recommendation": str(e),
            }
        results.append(res)

    results_df = pd.DataFrame(results)
    return pd.concat([df_input.reset_index(drop=True), results_df], axis=1)
