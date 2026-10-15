"""
Servicio de Inferencia FastAPI para Predicción de Fuga de Clientes (Customer Churn).

Expone endpoints para validación de salud (/health) y predicción en tiempo real (/predict),
integrándose con el modelo entrenado y registrado vía MLflow.
"""

import os
import time
from typing import Literal, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# =========================================================================
# CONFIGURACIÓN Y CARGA DEL MODELO
# =========================================================================
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "best_model.pkl"))
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", None)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Microservicio MLOps para inferencia en tiempo real de abandono de clientes.",
    version="1.0.0",
)

# Estado global del modelo en memoria
model_bundle = None


def load_model():
    """
    Carga el artefacto serializado del modelo.

    En un entorno productivo real:
    - Se cargaría directamente desde MLflow Model Registry utilizando la URI:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
      o bien descargando el artefacto desde el Bucket S3 configurado por Terraform.
    - Se implementaría un webhook o polling para recarga en caliente (zero-downtime)
      cuando se promocione un nuevo modelo a producción sin reiniciar el pod/contenedor.
    """
    global model_bundle

    # Opción 1: Carga desde URI de MLflow si está especificado
    if MLFLOW_MODEL_URI:
        try:
            import mlflow.sklearn

            pipeline = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
            model_bundle = {
                "pipeline": pipeline,
                "feature_names": [
                    "tenure_months",
                    "monthly_charges",
                    "total_charges",
                    "num_support_tickets",
                    "online_security",
                ],
                "run_id": "loaded_from_mlflow_uri",
                "model_name": "mlflow_registry_model",
                "f1_score": None,
            }
            return
        except Exception as e:
            print(f"[!] Advertencia: Fallo al cargar desde MLFLOW_MODEL_URI ({e}). Intentando local.")

    # Opción 2: Carga desde archivo local preempaquetado o montado por volumen
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)
        print(f"[✓] Modelo cargado exitosamente desde '{MODEL_PATH}'")
    else:
        print(f"[!] ADVERTENCIA: No se encontró artefacto en '{MODEL_PATH}'.")
        print("[!] Ejecute 'python -m src.train' primero para generar el artefacto.")


@app.on_event("startup")
def startup_event():
    """Inicializa el modelo al arrancar la aplicación."""
    load_model()


# =========================================================================
# SCHEMAS DE PYDANTIC (VALIDACIÓN Y CONTRATO DE DATOS)
# =========================================================================
class CustomerFeatures(BaseModel):
    """Esquema de entrada con los atributos del cliente para inferencia."""

    tenure_months: int = Field(
        ...,
        ge=0,
        le=120,
        description="Meses de antigüedad del cliente en la compañía.",
        example=14,
    )
    monthly_charges: float = Field(
        ...,
        ge=0.0,
        description="Cargo mensual regular del cliente en USD.",
        example=65.50,
    )
    total_charges: float = Field(
        ...,
        ge=0.0,
        description="Total histórico facturado acumulado en USD.",
        example=917.00,
    )
    num_support_tickets: int = Field(
        ...,
        ge=0,
        le=50,
        description="Cantidad de reclamos o tickets de soporte técnico abiertos.",
        example=2,
    )
    online_security: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 si posee paquete de seguridad online activo, 0 en caso contrario.",
        example=1,
    )


class PredictionResponse(BaseModel):
    """Esquema de salida con la probabilidad y clasificación de riesgo."""

    churn_prediction: int = Field(..., description="1 indica propensión a fuga, 0 permanencia.")
    churn_probability: float = Field(..., description="Probabilidad estimada de fuga (0.0 a 1.0).")
    risk_level: Literal["Bajo", "Medio", "Alto"] = Field(
        ..., description="Nivel categórico de riesgo de churn para acciones de retención."
    )
    model_name: str
    run_id: Optional[str] = None
    latency_ms: float


class HealthResponse(BaseModel):
    """Esquema de respuesta de comprobación de salud."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    run_id: Optional[str] = None



# =========================================================================
# ENDPOINTS DE LA API
# =========================================================================
@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check del Microservicio",
)
def health_check():
    """
    Endpoint para probes de liveness y readiness de Kubernetes / AWS ALB / ECS.
    """
    is_ready = model_bundle is not None
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_name=model_bundle.get("model_name") if is_ready else None,
        run_id=model_bundle.get("run_id") if is_ready else None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predicción de Fuga de Cliente",
)
def predict_churn(customer: CustomerFeatures):
    """
    Recibe los atributos de un cliente y retorna la probabilidad de fuga estimada.
    """
    if model_bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo predictivo no se encuentra cargado en el servicio.",
        )

    start_time = time.perf_counter()

    # Construir DataFrame con el orden de columnas esperado por el pipeline
    data_dict = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    input_df = pd.DataFrame([data_dict])

    try:
        pipeline = model_bundle["pipeline"]

        # Predicción de clases y probabilidades
        prediction = int(pipeline.predict(input_df)[0])

        if hasattr(pipeline, "predict_proba"):
            probability = float(pipeline.predict_proba(input_df)[0][1])
        else:
            probability = float(prediction)

        # Categorización de riesgo para el equipo comercial / CRM
        if probability < 0.35:
            risk = "Bajo"
        elif probability < 0.65:
            risk = "Medio"
        else:
            risk = "Alto"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # =====================================================================
        # PRODUCCIÓN: Monitoreo, Data Drift y Observabilidad
        # =====================================================================
        # En producción aquí se integraría:
        # 1. Enviar métricas a Prometheus (tiempo de inferencia, recuento por riesgo).
        # 2. Asíncronamente (vía Kafka o SQS/Firehose) enviar payload y predicción
        #    a un bucket de telemetría para evaluar Data Drift con Evidently AI o WhyLogs.

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk,
            model_name=model_bundle.get("model_name", "unknown"),
            run_id=model_bundle.get("run_id"),
            latency_ms=latency_ms,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el cálculo de la inferencia: {str(exc)}",
        )
