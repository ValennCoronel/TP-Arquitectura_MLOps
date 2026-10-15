"""
Pruebas automatizadas con pytest para el microservicio de Serving FastAPI.
Verifica endpoints, contratos de datos de Pydantic y lógica de predicción.
"""

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.app import app
import src.app as app_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_mock_model_if_empty():
    """
    Fixture que garantiza la disponibilidad de un modelo en memoria
    para pruebas unitarias en entornos CI donde aún no se ejecutó train.py.
    """
    if app_module.model_bundle is None:
        # Entrenar un mini-modelo sintético para propósitos de test
        mock_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ])
        # 5 features coincidentes con CustomerFeatures
        X_dummy = np.array([
            [12, 50.0, 600.0, 1, 1],
            [2, 90.0, 180.0, 4, 0],
            [24, 40.0, 960.0, 0, 1],
            [1, 100.0, 100.0, 5, 0],
        ])
        y_dummy = np.array([0, 1, 0, 1])
        mock_pipeline.fit(X_dummy, y_dummy)

        app_module.model_bundle = {
            "pipeline": mock_pipeline,
            "feature_names": [
                "tenure_months",
                "monthly_charges",
                "total_charges",
                "num_support_tickets",
                "online_security",
            ],
            "run_id": "test-mock-run-id",
            "model_name": "Test_Logistic_Regression",
            "f1_score": 1.0,
        }


def test_health_endpoint():
    """Verifica que el endpoint /health responda HTTP 200 y reporte el modelo activo."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_name" in data


def test_predict_endpoint_valid_payload():
    """Verifica una inferencia exitosa con datos válidos de cliente."""
    payload = {
        "tenure_months": 12,
        "monthly_charges": 65.50,
        "total_charges": 786.00,
        "num_support_tickets": 2,
        "online_security": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "churn_prediction" in data
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Bajo", "Medio", "Alto"]
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0


def test_predict_endpoint_invalid_payload():
    """Verifica que Pydantic capture violaciones de esquema (ej. valores negativos)."""
    invalid_payload = {
        "tenure_months": -5,  # Invalido: ge=0 requerido
        "monthly_charges": 65.50,
        "total_charges": 786.00,
        "num_support_tickets": 2,
        "online_security": 1,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
