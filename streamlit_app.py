"""
Pipeline de Entrenamiento Completo (Fases 1-3)
Ejecutar una sola vez para generar modelos y visualizaciones.

Uso:
    python src/train_pipeline.py
"""

import sys
from pathlib import Path

# Asegurar imports locales
sys.path.insert(0, str(Path(__file__).parent))

from etl import run_pipeline
from eda import run_eda
from modeling import run_modeling


def main():
    base = Path(__file__).parent.parent
    data_path = base / "data" / "student_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {data_path}.\n"
            "Asegúrate de colocar 'student_data.csv' en la carpeta data/."
        )

    print("=" * 60)
    print("  SISTEMA DE RETENCIÓN ESTUDIANTIL — PIPELINE DE ML")
    print("=" * 60)

    # Fase 1: ETL
    print("\n[FASE 1] ETL y Limpieza de Datos")
    df_clean, df_encoded = run_pipeline(data_path)
    df_clean.to_csv(base / "data" / "student_clean.csv", index=False)
    df_encoded.to_csv(base / "data" / "student_encoded.csv", index=False)

    # Fase 2: EDA
    print("\n[FASE 2] Análisis Exploratorio de Datos")
    eda_dir = base / "notebooks" / "eda_plots"
    run_eda(df_clean, eda_dir)

    # Fase 3: Modelado
    print("\n[FASE 3] Entrenamiento y Evaluación de Modelos")
    ml_plots_dir = base / "notebooks" / "ml_plots"
    best_model, features = run_modeling(
        df_encoded,
        models_dir=base / "models",
        plots_dir=ml_plots_dir,
    )

    print("\n" + "=" * 60)
    print("  ✅ Pipeline completado exitosamente.")
    print(f"  Modelos en       : {base / 'models'}")
    print(f"  Gráficos EDA     : {eda_dir}")
    print(f"  Gráficos ML      : {ml_plots_dir}")
    print("  Siguiente paso   : streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
