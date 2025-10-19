# =====================================================
# 🧠 ENTRENAMIENTO DEL MODELO CON PIPELINE AUTOMÁTICO
# =====================================================
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # habilita el imputador
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# =====================================================
# 🔧 Función principal de entrenamiento
# =====================================================
def train_model():
    # --- Cargar datos
    data = pd.read_csv("data/processed/train.csv")
    print(f"📦 Dataset cargado: {data.shape[0]} filas, {data.shape[1]} columnas")

    # --- Eliminar columnas identificadoras (no predictivas)
    id_cols = [c for c in data.columns if "key" in c.lower() or "id" in c.lower()]
    if id_cols:
        data = data.drop(columns=id_cols)
        print(f"🧹 Columnas ID eliminadas: {id_cols}")

    # --- Separar variables predictoras y target
    X = data.drop("target", axis=1)
    y = data["target"]

    # --- Detectar columnas numéricas y categóricas
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    print(f"🔢 Columnas numéricas: {len(num_cols)}")
    print(f"🔠 Columnas categóricas: {len(cat_cols)}")

    # =====================================================
    # 🧩 Construir el pipeline de preprocesamiento
    # =====================================================
    numeric_pipeline = Pipeline([
        ("imputer", IterativeImputer(max_iter=30, random_state=42)),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    # =====================================================
    # ⚙️ Modelo principal
    # =====================================================
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=300, solver="saga", n_jobs=-1))
    ])

    # =====================================================
    # 🧪 Entrenamiento con MLflow Tracking
    # =====================================================
    mlflow.set_experiment("mlops_demo")

    with mlflow.start_run(run_name="logistic_regression_run"):
        print("🔧 Entrenando modelo...")
        pipeline.fit(X, y)

        # --- Métricas básicas sobre entrenamiento
        y_pred = pipeline.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

        # --- Log en consola
        print(f"✅ Modelo entrenado correctamente con {X.shape[1]} variables y {X.shape[0]} registros.")
        print(f"📈 Métricas (train): acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")

        # --- Log de métricas en MLflow
        mlflow.log_metric("accuracy_train", acc)
        mlflow.log_metric("precision_train", prec)
        mlflow.log_metric("recall_train", rec)
        mlflow.log_metric("f1_train", f1)

        # --- Guardar modelo localmente
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(pipeline, model_path)
        print(f"💾 Guardado en: {model_path}")

        # --- Registrar modelo en MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_artifact(model_path)
        print("📊 Modelo registrado en MLflow exitosamente.")


# =====================================================
# 🚀 Ejecutar
# =====================================================
if __name__ == "__main__":
    train_model()
