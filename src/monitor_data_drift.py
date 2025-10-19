import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

def monitor_drift():
    # ===============================
    # 1️⃣ Cargar datasets de referencia y actual
    # ===============================
    reference_path = "data/processed/train.csv"
    current_path = "data/processed/test.csv"

    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        raise FileNotFoundError("No se encontraron los archivos de train/test procesados.")

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # ===============================
    # 2️⃣ Verificar que tengan columnas compatibles
    # ===============================
    common_cols = list(set(reference.columns) & set(current.columns))
    if not common_cols:
        raise ValueError("No hay columnas comunes entre los datasets para comparar drift.")
    
    reference = reference[common_cols]
    current = current[common_cols]

    print(f"📊 Comparando {len(common_cols)} columnas entre train y test...")

    # ===============================
    # 3️⃣ Generar reporte de drift
    # ===============================
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    os.makedirs("reports", exist_ok=True)
    output_path = "reports/data_drift_report.html"
    report.save_html(output_path)

    print(f"✅ Reporte de data drift generado en: {output_path}")

if __name__ == "__main__":
    monitor_drift()
