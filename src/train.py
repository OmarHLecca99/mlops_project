import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import joblib
import os

def train_model():
    # ===============================
    # 1️⃣ Configurar experimento MLflow
    # ===============================
    mlflow.set_experiment("mlops_demo")

    with mlflow.start_run():
        # ===============================
        # 2️⃣ Cargar dataset procesado
        # ===============================
        data = pd.read_csv('data/processed/train.csv')

        # Muestra reducida (opcional para evitar exceso de RAM)
        if len(data) > 10000:
            data = data.sample(n=10000, random_state=42)

        # ===============================
        # 3️⃣ Separar features y target
        # ===============================
        if 'target' not in data.columns:
            raise ValueError("❌ No se encontró la columna 'target' en el dataset.")

        X = data.drop('target', axis=1)
        y = data['target']

        # ===============================
        # 4️⃣ Eliminar columnas tipo ID
        # ===============================
        id_like = [c for c in X.columns if 'id' in c.lower() or 'key' in c.lower()]
        if id_like:
            X = X.drop(columns=id_like)
            print(f"🧹 Columnas ID eliminadas: {id_like}")

        # ===============================
        # 5️⃣ Codificar columnas categóricas
        # ===============================
        for col in X.columns:
            if X[col].dtype == 'object' or str(X[col].dtype).startswith('category'):
                if X[col].nunique() <= 50:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                else:
                    print(f"⚠️ Columna '{col}' eliminada (demasiadas categorías o texto libre).")
                    X = X.drop(columns=[col])

        # ===============================
        # 6️⃣ Imputar valores faltantes
        # ===============================
        if X.isnull().any().any():
            print("🔧 Imputando valores faltantes...")
            num_cols = X.select_dtypes(include=['float64', 'int64']).columns
            cat_cols = X.select_dtypes(exclude=['float64', 'int64']).columns

            # Imputadores
            num_imputer = SimpleImputer(strategy='median')
            cat_imputer = SimpleImputer(strategy='most_frequent')

            # Aplicar
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
            if len(cat_cols) > 0:
                X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        # ===============================
        # 7️⃣ Crear pipeline (escalado + modelo)
        # ===============================
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                max_iter=300,
                solver='saga',
                penalty='l2',
                n_jobs=-1,
                verbose=1
            ))
        ])

        # ===============================
        # 8️⃣ Entrenar modelo
        # ===============================
        pipeline.fit(X, y)

        print(f"✅ Modelo entrenado correctamente con {X.shape[1]} variables y {len(X)} registros.")

        # ===============================
        # 9️⃣ Guardar modelo
        # ===============================
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/model.pkl")
        print("💾 Guardado en: models/model.pkl")

        # ===============================
        # 🔟 Registrar en MLflow
        # ===============================
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("solver", "saga")

        print("📊 Modelo registrado en MLflow exitosamente.")

if __name__ == "__main__":
    train_model()
