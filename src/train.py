"""
Script de Entrenamiento y Tracking con MLflow.
Caso de uso: Predicción de Fuga de Clientes (Customer Churn).

Este módulo genera un dataset sintético realista, entrena dos algoritmos
(Logistic Regression y Random Forest), compara sus métricas y registra
hiperparámetros, métricas y artefactos en MLflow.
"""

import os
import json
import joblib
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cargar automáticamente variables de entorno desde el archivo .env si existe
load_dotenv()


def generate_churn_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético tabular de churn con características representativas.

    En un entorno productivo real:
    - Esta función sería reemplazada por una conexión a un Feature Store (ej. Feast, Hopsworks,
      AWS SageMaker Feature Store) o un Data Warehouse (Snowflake, BigQuery, Redshift).
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
    # CONFIGURACIÓN DE TRACKING SERVER (Local o Docker Compose / Remoto)
    # =========================================================================
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[*] MLflow Tracking Server configurado en: {tracking_uri}")
    else:
        print("[*] MLflow utilizando almacenamiento local predeterminado (./mlruns)")

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
                    # Hiperparámetros originales:
                    #(
                    #    "classifier",
                    #    LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                    #),
                    # Nuevos hiperparámetros para prueba (comentados):
                     (
                         "classifier",
                         LogisticRegression(
                             C=0.5,
                             class_weight="balanced",
                             max_iter=1000,
                             random_state=42,
                         ),
                     ),
                ]
            ),
            # Solo metadatos: los hiperparámetros se leen del modelo al registrar la corrida
            "params": {
                "model_type": "LogisticRegression",
                "scaler": "StandardScaler",
            },
        },
        "Random_Forest": {
            "pipeline": Pipeline(
                [
                    (
                        "classifier",
                        # Hiperparámetros originales:
                        #RandomForestClassifier(
                        #    n_estimators=120,
                        #    max_depth=6,
                        #    min_samples_split=5,
                        #    random_state=42,
                        #),
                        # Nuevos hiperparámetros para prueba (comentados):
                        RandomForestClassifier(
                             n_estimators=200,
                             max_depth=10,
                             min_samples_split=2,
                             min_samples_leaf=2,
                             criterion="entropy",
                             random_state=42,
                         ),
                    ),
                ]
            ),
            # Solo metadatos: los hiperparámetros se leen del modelo al registrar la corrida
            "params": {
                "model_type": "RandomForestClassifier",
            },
        },
    }

    best_model_name = None
    best_f1_score = -1.0
    best_pipeline = None
    best_run_id = None
    best_metrics = {}
    all_runs_summary = {}

    # 2. Entrenamiento iterativo y tracking
    for run_name, config in models_to_evaluate.items():
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"\n[+] Entrenando modelo: {run_name} (Run ID: {run_id})")

            pipeline = config["pipeline"]

            # Registro de hiperparámetros y metadata de datos.
            # Los hiperparámetros se leen del propio modelo (último paso del pipeline),
            # así lo registrado en MLflow siempre coincide con lo que se entrena.
            run_params = {**config["params"], **pipeline[-1].get_params()}
            mlflow.log_params(run_params)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.set_tag("pipeline.environment", "poc")
            mlflow.set_tag("algorithm.family", config["params"]["model_type"])

            # Entrenamiento
            pipeline.fit(X_train, y_train)

            # Evaluación
            metrics = evaluate_model(pipeline, X_test, y_test)
            mlflow.log_metrics(metrics)

            all_runs_summary[run_name] = {
                "run_id": run_id,
                "params": run_params,
                "metrics": metrics,
            }

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
                best_metrics = metrics

    print("\n" + "=" * 60)
    print(f"[!] MODELO CAMPEÓN SELECCIONADO: {best_model_name}")
    print(f"[!] Mejor F1-Score: {best_f1_score:.4f} | Run ID: {best_run_id}")
    print("=" * 60)

    # 3. Etiquetado de Gobernanza en MLflow (CHAMPION / CHALLENGER / DISCARDED)
    model_uri = f"runs:/{best_run_id}/model"
    print(f"\n[*] Model URI del ganador: {model_uri}")

    client = MlflowClient()
    promote_env = os.getenv("PROMOTE_TO_CHAMPION", "false").lower() == "true"

    # Etiquetar las corridas descartadas en el experimento
    for r_name, r_info in all_runs_summary.items():
        r_id = r_info["run_id"]
        if r_id != best_run_id:
            client.set_tag(r_id, "mlops.role", "DISCARDED")
            client.set_tag(r_id, "mlops.tournament_result", "loser")
            r_info["mlops_role"] = "DISCARDED"

    registered_version = None
    assigned_role = "CHALLENGER"

    # Si hay un servidor de tracking configurado (ej. DagsHub), registramos en el Model Registry
    if tracking_uri and not tracking_uri.startswith("file:"):
        try:
            print("[*] Registrando modelo en el Model Registry de MLflow...")
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="customer_churn_model",
            )
            registered_version = str(registered_model.version)
            reg_uri = f"models:/customer_churn_model/{registered_version}"

            prev_versions = client.search_model_versions("name='customer_churn_model'")
            other_versions = [
                pv for pv in prev_versions if str(pv.version) != registered_version
            ]

            # Verificar si ya existe algún CHAMPION previo en el Registry
            existing_champions = [
                pv
                for pv in other_versions
                if getattr(pv, "current_stage", "") == "Production"
                or (getattr(pv, "tags", {}) or {}).get("mlops.role") == "CHAMPION"
            ]

            # Si hay versiones previas pero ninguna tenía el tag CHAMPION aún,
            # promovemos la versión previa más reciente como CHAMPION de referencia
            if other_versions and not existing_champions and not promote_env:
                latest_prev = max(other_versions, key=lambda v: int(v.version))
                client.set_model_version_tag(
                    "customer_churn_model", str(latest_prev.version), "mlops.role", "CHAMPION"
                )
                if latest_prev.run_id:
                    client.set_tag(latest_prev.run_id, "mlops.role", "CHAMPION")
                try:
                    client.transition_model_version_stage(
                        name="customer_churn_model",
                        version=str(latest_prev.version),
                        stage="Production",
                    )
                except Exception:
                    pass
                existing_champions = [latest_prev]

            # Es CHAMPION si es la única versión (v1) o si estamos desplegando a main
            is_champion = (not other_versions) or promote_env
            assigned_role = "CHAMPION" if is_champion else "CHALLENGER"
            target_stage = "Production" if is_champion else "Staging"

            if is_champion and existing_champions:
                # Archivar versiones previas que fueran CHAMPION
                for pv in existing_champions:
                    client.set_model_version_tag(
                        "customer_churn_model", str(pv.version), "mlops.role", "ARCHIVED_CHAMPION"
                    )
                    if pv.run_id:
                        client.set_tag(pv.run_id, "mlops.role", "ARCHIVED_CHAMPION")

            # Etiquetar la versión en el Model Registry y transicionar su Stage
            client.set_model_version_tag(
                "customer_churn_model", registered_version, "mlops.role", assigned_role
            )
            try:
                client.transition_model_version_stage(
                    name="customer_churn_model",
                    version=registered_version,
                    stage=target_stage,
                    archive_existing_versions=is_champion,
                )
            except Exception:
                pass

            print("[✓] Modelo registrado y etiquetado exitosamente en el Registry!")
            print("    - Nombre: 'customer_churn_model'")
            print(f"    - Versión: {registered_version} | Stage: {target_stage}")
            print(f"    - Tag mlops.role: {assigned_role}")
            print(f"    - Registry URI para Serving: '{reg_uri}'")
        except Exception as err:
            print(f"[!] Nota sobre Model Registry: {err}")
    else:
        assigned_role = "CHAMPION" if promote_env else "CHALLENGER"

    # Etiquetar la corrida ganadora en la tabla de Experimentos de MLflow
    client.set_tag(best_run_id, "mlops.role", assigned_role)
    client.set_tag(best_run_id, "mlops.tournament_result", "winner")
    all_runs_summary[best_model_name]["mlops_role"] = assigned_role

    # 4. Exportación local para serving offline, Docker y comparación en PR
    os.makedirs("models", exist_ok=True)
    export_path = os.path.join("models", "best_model.pkl")
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_names": feature_cols,
            "run_id": best_run_id,
            "model_name": best_model_name,
            "f1_score": best_f1_score,
            "metrics": best_metrics,
            "mlops_role": assigned_role,
        },
        export_path,
    )

    summary_path = os.path.join("models", "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "best_model_name": best_model_name,
                "best_run_id": best_run_id,
                "registered_version": registered_version,
                "mlops_role": assigned_role,
                "best_metrics": best_metrics,
                "runs": all_runs_summary,
            },
            f,
            indent=2,
        )

    print(f"[✓] Artefacto exportado localmente en: '{export_path}'")
    print("[✓] Pipeline de entrenamiento finalizado exitosamente.")


if __name__ == "__main__":
    train_and_track()
