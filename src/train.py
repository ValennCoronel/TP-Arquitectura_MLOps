"""
Script de Entrenamiento y Tracking con MLflow.
Caso de uso: Predicción de Fuga de Clientes (Customer Churn).

Este módulo genera un dataset sintético realista, entrena dos algoritmos
(Logistic Regression y Random Forest), compara sus métricas y registra
hiperparámetros, métricas y artefactos en MLflow.
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_churn_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético tabular de churn con características representativas.

    En un entorno productivo real:
    - Esta función sería reemplazada por una conexión a un Feature Store (ej. Feast, Hopsworks,
      AWS SageMaker Feature Store) o un Data Warehouse (ej. Snowflake, Google BigQuery, AWS Redshift).
    - Los datos de clientes vendrían enriquecidos con agregaciones de streaming (ej. Kafka/Flink)
      y tablas batch históricas.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        weights=[0.75, 0.25],  # 25% tasa de churn simulada (desbalance de clases realista)
        flip_y=0.03,
        random_state=random_state,
    )

    # Escalamos a valores realistas de negocio
    df = pd.DataFrame(
        {
            "tenure_months": np.clip(np.round(np.abs(X[:, 0]) * 15 + 1), 1, 72).astype(int),
            "monthly_charges": np.round(np.abs(X[:, 1]) * 35 + 20, 2),
            "total_charges": np.round(np.abs(X[:, 2]) * 1500 + 50, 2),
            "num_support_tickets": np.clip(np.round(np.abs(X[:, 3]) * 3), 0, 10).astype(int),
            "online_security": (X[:, 4] > 0).astype(int),
            "churn": y,
        }
    )
    return df


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula las métricas clave de clasificación."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics


def train_and_track():
    """
    Pipeline principal de orquestación de experimentos con MLflow.
    """
    # =========================================================================
    # PRODUCCIÓN: Configuración de Tracking Server Remoto
    # =========================================================================
    # En producción real:
    # 1. Definir URI del servidor central (ej. desplegado en ECS con backend PostgreSQL en RDS):
    #    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5000"))
    # 2. Los artefactos se almacenan directamente en S3:
    #    os.environ["MLFLOW_S3_ENDPOINT_URL"] = ... o permisos IAM vía IAM Roles for Service Accounts.
    # Para esta PoC, se utiliza el almacenamiento local predeterminado (./mlruns).

    experiment_name = "customer-churn-prediction"
    mlflow.set_experiment(experiment_name)

    print(f"[*] Iniciando ejecución de experimento en MLflow: '{experiment_name}'")

    # 1. Preparación de datos
    df = generate_churn_dataset(n_samples=3000, random_state=42)
    feature_cols = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_tickets",
        "online_security",
    ]
    target_col = "churn"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models_to_evaluate = {
        "Logistic_Regression": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
                ]
            ),
            "params": {
                "model_type": "LogisticRegression",
                "C": 1.0,
                "max_iter": 1000,
                "scaler": "StandardScaler",
                "class_weight": "none",
            },
        },
        "Random_Forest": {
            "pipeline": Pipeline(
                [
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=120,
                            max_depth=6,
                            min_samples_split=5,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 120,
                "max_depth": 6,
                "min_samples_split": 5,
            },
        },
    }

    best_model_name = None
    best_f1_score = -1.0
    best_pipeline = None
    best_run_id = None

    # 2. Entrenamiento iterativo y tracking
    for run_name, config in models_to_evaluate.items():
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"\n[+] Entrenando modelo: {run_name} (Run ID: {run_id})")

            # Registro de hiperparámetros y metadata de datos
            mlflow.log_params(config["params"])
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.set_tag("pipeline.environment", "poc")
            mlflow.set_tag("algorithm.family", config["params"]["model_type"])

            # Entrenamiento
            pipeline = config["pipeline"]
            pipeline.fit(X_train, y_train)

            # Evaluación
            metrics = evaluate_model(pipeline, X_test, y_test)
            mlflow.log_metrics(metrics)

            print(
                f"    Métricas -> Accuracy: {metrics['accuracy']:.4f} | "
                f"F1-Score: {metrics['f1_score']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f}"
            )

            # Logueo del artefacto en MLflow
            # En producción, esto sube los binarios directamente a S3 / GCS
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=X_train.head(3),
            )

            # Selección del campeón
            if metrics["f1_score"] > best_f1_score:
                best_f1_score = metrics["f1_score"]
                best_model_name = run_name
                best_pipeline = pipeline
                best_run_id = run_id

    print("\n" + "=" * 60)
    print(f"[!] MODELO CAMPEÓN SELECCIONADO: {best_model_name}")
    print(f"[!] Mejor F1-Score: {best_f1_score:.4f} | Run ID: {best_run_id}")
    print("=" * 60)

    # 3. Exportación de artefacto para Serving
    # =========================================================================
    # PRODUCCIÓN: MLflow Model Registry
    # =========================================================================
    # En producción con MLflow Model Registry:
    #   model_uri = f"runs:/{best_run_id}/model"
    #   mlflow.register_model(model_uri, "customer-churn-production")
    #   client = mlflow.tracking.MlflowClient()
    #   client.set_registered_model_alias("customer-churn-production", "champion", version)
    #
    # Para garantizar que el Docker de FastAPI funcione de inmediato sin dependencias
    # de red externas en la PoC, serializamos el pipeline ganador en 'models/best_model.pkl'.
    os.makedirs("models", exist_ok=True)
    export_path = os.path.join("models", "best_model.pkl")
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_names": feature_cols,
            "run_id": best_run_id,
            "model_name": best_model_name,
            "f1_score": best_f1_score,
        },
        export_path,
    )
    print(f"[✓] Artefacto exportado exitosamente para el servicio API en: '{export_path}'")
    print("[✓] Ejecución de MLflow completada. Ejecuta 'mlflow ui' para explorar la interfaz gráfica.")


if __name__ == "__main__":
    train_and_track()
