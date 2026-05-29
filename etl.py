"""
Fase 3: Modelado Predictivo (Machine Learning)
Entrena, evalúa y persiste dos modelos: Regresión Logística y Random Forest.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "at_risk"
# Excluir notas finales para no hacer trampa (G3 es lo que predecimos indirectamente)
EXCLUDE_COLS = [TARGET, "G3"]


# ──────────────────────────────────────────────
# Preparación de datos
# ──────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) y variable objetivo (y)."""
    drop = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET]
    # Conservar solo columnas numéricas (después del OHE en ETL)
    X = X.select_dtypes(include=[np.number])
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ──────────────────────────────────────────────
# Definición de modelos
# ──────────────────────────────────────────────

def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            C=1.0,
        )),
    ])


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# ──────────────────────────────────────────────
# Entrenamiento y evaluación
# ──────────────────────────────────────────────

def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Entrena el modelo y devuelve un dict con todas las métricas."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Cross-validation (F1 macro, 5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")

    metrics = {
        "model_name": name,
        "accuracy": report["accuracy"],
        "precision_risk": report["1"]["precision"],
        "recall_risk": report["1"]["recall"],
        "f1_risk": report["1"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "roc_auc": roc_auc,
        "cv_f1_macro_mean": cv_scores.mean(),
        "cv_f1_macro_std": cv_scores.std(),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision_risk']:.4f}  (clase En Riesgo)")
    print(f"  Recall     : {metrics['recall_risk']:.4f}  (clase En Riesgo)")
    print(f"  F1-Score   : {metrics['f1_risk']:.4f}  (clase En Riesgo)")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  CV F1-macro: {metrics['cv_f1_macro_mean']:.4f} ± {metrics['cv_f1_macro_std']:.4f}")

    return metrics, y_pred, y_proba


def plot_evaluation(name: str, model, X_test, y_test, y_pred, y_proba, out_dir: Path):
    """Genera confusion matrix + ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Evaluación: {name}", fontsize=14, fontweight="bold")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sin Riesgo", "En Riesgo"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Matriz de Confusión")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="#E74C3C", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_xlabel("Tasa de Falsos Positivos")
    axes[1].set_ylabel("Tasa de Verdaderos Positivos")
    axes[1].set_title("Curva ROC")
    axes[1].legend()

    fig.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    path = out_dir / f"eval_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ML] Gráfico guardado: {path.name}")


def plot_feature_importance(model_rf: Pipeline, feature_names: list, out_dir: Path):
    """Top-15 features más importantes del Random Forest."""
    importances = model_rf.named_steps["clf"].feature_importances_
    indices = np.argsort(importances)[-15:][::-1]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top_features)), top_values[::-1],
            color="#3498DB", edgecolor="white")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Top 15 Variables Más Importantes (Random Forest)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance_rf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[ML] Feature importance guardada.")


def plot_model_comparison(results: list[dict], out_dir: Path):
    """Comparativa de métricas entre modelos."""
    metrics = ["accuracy", "precision_risk", "recall_risk", "f1_risk", "roc_auc"]
    labels = ["Accuracy", "Precision\n(Riesgo)", "Recall\n(Riesgo)", "F1\n(Riesgo)", "ROC-AUC"]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, res in enumerate(results):
        values = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=res["model_name"],
                      edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Métrica")
    ax.set_title("Comparativa de Modelos", fontsize=14, fontweight="bold")
    ax.legend()
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="Umbral 0.8")
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[ML] Comparativa de modelos guardada.")


# ──────────────────────────────────────────────
# Persistencia
# ──────────────────────────────────────────────

def save_model(model, feature_names: list, metrics: dict, path: Path):
    """Guarda el modelo, los nombres de features y las métricas."""
    path.mkdir(parents=True, exist_ok=True)
    safe = metrics["model_name"].lower().replace(" ", "_")
    joblib.dump(model, path / f"{safe}.pkl")
    with open(path / f"{safe}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(feature_names, path / "feature_names.pkl")
    print(f"[ML] Modelo guardado: {safe}.pkl")


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────

def run_modeling(df_encoded: pd.DataFrame, models_dir: Path, plots_dir: Path) -> Pipeline:
    """
    Ejecuta el pipeline completo de ML y devuelve el mejor modelo.
    """
    X, y = prepare_features(df_encoded)
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_names = list(X.columns)

    print(f"[ML] Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[ML] Prevalencia en riesgo (train): {y_train.mean()*100:.1f}%")

    models = {
        "Regresión Logística": build_logistic_regression(),
        "Random Forest": build_random_forest(),
    }

    all_results = []
    trained_models = {}

    plots_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        plot_evaluation(name, model, X_test, y_test, y_pred, y_proba, plots_dir)
        save_model(model, feature_names, metrics, models_dir)
        all_results.append(metrics)
        trained_models[name] = model

    # Feature importance (solo RF)
    plot_feature_importance(
        trained_models["Random Forest"], feature_names, plots_dir
    )
    plot_model_comparison(all_results, plots_dir)

    # Seleccionar mejor modelo por ROC-AUC
    best = max(all_results, key=lambda x: x["roc_auc"])
    print(f"\n[ML] ✅ Mejor modelo: {best['model_name']} (ROC-AUC={best['roc_auc']:.4f})")

    return trained_models[best["model_name"]], feature_names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from etl import run_pipeline

    base = Path(__file__).parent.parent
    _, df_enc = run_pipeline(base / "data" / "student_data.csv")
    run_modeling(
        df_enc,
        models_dir=base / "models",
        plots_dir=base / "notebooks" / "ml_plots",
    )
