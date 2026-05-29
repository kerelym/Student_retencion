"""
Fase 2: Análisis Exploratorio de Datos (EDA)
Genera visualizaciones clave para entender los factores de riesgo estudiantil.
"""

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Paleta institucional
PALETTE = {"En Riesgo": "#E74C3C", "Sin Riesgo": "#2ECC71"}
RISK_COLORS = ["#2ECC71", "#E74C3C"]
FIG_DPI = 150


def _save(fig: plt.Figure, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Guardado: {name}")


def plot_target_distribution(df: pd.DataFrame, out: Path):
    """Distribución de la variable objetivo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribución de Riesgo Estudiantil", fontsize=14, fontweight="bold")

    counts = df["at_risk"].value_counts().sort_index()
    labels = ["Sin Riesgo (G3≥10)", "En Riesgo (G3<10)"]
    colors = RISK_COLORS

    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Conteo de Estudiantes")
    axes[0].set_ylabel("Cantidad")
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(val), ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Proporción de Riesgo")

    fig.tight_layout()
    _save(fig, out, "01_target_distribution.png")


def plot_grades_vs_risk(df: pd.DataFrame, out: Path):
    """Distribución de notas G1, G2, G3 por nivel de riesgo."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Notas por Trimestre vs Riesgo", fontsize=14, fontweight="bold")

    for ax, grade in zip(axes, ["G1", "G2", "G3"]):
        for risk, color in zip([0, 1], RISK_COLORS):
            label = "Sin Riesgo" if risk == 0 else "En Riesgo"
            subset = df[df["at_risk"] == risk][grade]
            ax.hist(subset, bins=15, alpha=0.7, color=color, label=label, edgecolor="white")
        ax.set_title(f"Nota {grade}")
        ax.set_xlabel("Calificación (0-20)")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        ax.axvline(10, color="black", linestyle="--", alpha=0.5, label="Umbral")

    fig.tight_layout()
    _save(fig, out, "02_grades_vs_risk.png")


def plot_absences_vs_risk(df: pd.DataFrame, out: Path):
    """¿Las ausencias predicen el riesgo?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ausencias vs Riesgo Estudiantil", fontsize=14, fontweight="bold")

    risk_labels = {0: "Sin Riesgo", 1: "En Riesgo"}
    data_groups = [df[df["at_risk"] == k]["absences"] for k in [0, 1]]

    axes[0].boxplot(data_groups, labels=["Sin Riesgo", "En Riesgo"],
                    patch_artist=True,
                    boxprops=dict(facecolor="lightblue"),
                    medianprops=dict(color="red", linewidth=2))
    axes[0].set_title("Distribución de Ausencias")
    axes[0].set_ylabel("Número de Ausencias")

    # Scatter: ausencias vs G3
    scatter_colors = df["at_risk"].map({0: RISK_COLORS[0], 1: RISK_COLORS[1]})
    axes[1].scatter(df["absences"], df["G3"], c=scatter_colors, alpha=0.5, s=30)
    axes[1].axhline(10, color="black", linestyle="--", alpha=0.6, label="Umbral riesgo")
    axes[1].set_xlabel("Ausencias")
    axes[1].set_ylabel("Nota Final (G3)")
    axes[1].set_title("Ausencias vs Nota Final")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, out, "03_absences_vs_risk.png")


def plot_socioeconomic_factors(df: pd.DataFrame, out: Path):
    """Factores socioeconómicos: educación de padres y trabajo."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Factores Socioeconómicos vs Riesgo", fontsize=14, fontweight="bold")

    for ax, col, title in zip(
        axes,
        ["Medu", "Fedu"],
        ["Educación de la Madre (0-4)", "Educación del Padre (0-4)"],
    ):
        risk_rate = df.groupby(col)["at_risk"].mean() * 100
        bars = ax.bar(risk_rate.index, risk_rate.values,
                      color=[RISK_COLORS[1] if v > 35 else RISK_COLORS[0]
                             for v in risk_rate.values],
                      edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Nivel Educativo")
        ax.set_ylabel("% en Riesgo")
        ax.set_ylim(0, 80)
        for bar, val in zip(bars, risk_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", fontsize=9)

    fig.tight_layout()
    _save(fig, out, "04_socioeconomic_factors.png")


def plot_correlation_heatmap(df: pd.DataFrame, out: Path):
    """Mapa de correlaciones con la variable objetivo."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()[["at_risk"]].drop("at_risk").sort_values("at_risk")

    fig, ax = plt.subplots(figsize=(7, 10))
    fig.suptitle("Correlación de Variables con Riesgo", fontsize=14, fontweight="bold")

    colors = [RISK_COLORS[1] if v > 0 else RISK_COLORS[0] for v in corr["at_risk"]]
    bars = ax.barh(corr.index, corr["at_risk"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlación de Pearson con `at_risk`")
    ax.set_title("Variables más correlacionadas con el riesgo")

    fig.tight_layout()
    _save(fig, out, "05_correlation_heatmap.png")


def plot_study_time_vs_risk(df: pd.DataFrame, out: Path):
    """Tiempo de estudio y actividades extracurriculares."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Hábitos de Estudio vs Riesgo", fontsize=14, fontweight="bold")

    # Tiempo de estudio
    study_risk = df.groupby("studytime")["at_risk"].mean() * 100
    axes[0].bar(study_risk.index, study_risk.values, color="#3498DB", edgecolor="white")
    axes[0].set_title("Tiempo de Estudio vs % en Riesgo")
    axes[0].set_xlabel("Tiempo de estudio (1=<2h, 4=>10h)")
    axes[0].set_ylabel("% en Riesgo")

    # Historial de reprobaciones
    fail_risk = df.groupby("failures")["at_risk"].mean() * 100
    axes[1].bar(fail_risk.index, fail_risk.values, color="#E67E22", edgecolor="white")
    axes[1].set_title("Reprobaciones Previas vs % en Riesgo")
    axes[1].set_xlabel("Número de materias reprobadas")
    axes[1].set_ylabel("% en Riesgo")
    for bar, val in zip(axes[1].patches, fail_risk.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", fontweight="bold")

    fig.tight_layout()
    _save(fig, out, "06_study_habits.png")


def run_eda(df: pd.DataFrame, out_dir: Path):
    """Ejecuta todas las visualizaciones EDA."""
    print("[EDA] Iniciando análisis exploratorio...")
    plot_target_distribution(df, out_dir)
    plot_grades_vs_risk(df, out_dir)
    plot_absences_vs_risk(df, out_dir)
    plot_socioeconomic_factors(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_study_time_vs_risk(df, out_dir)
    print(f"[EDA] {6} gráficos generados en {out_dir}")


if __name__ == "__main__":
    from etl import run_pipeline
    base = Path(__file__).parent.parent
    df_clean, _ = run_pipeline(base / "data" / "student_data.csv")
    run_eda(df_clean, base / "notebooks" / "eda_plots")
