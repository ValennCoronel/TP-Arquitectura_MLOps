# 🚀 MLOps Proof of Concept (PoC): Customer Churn LifeCycle

Prueba de concepto end-to-end de un ciclo de vida completo de Machine Learning (MLOps) aplicada al caso de uso de **Predicción de Fuga de Clientes (Customer Churn)**.

Diseñado bajo estándares de ingeniería de software y MLOps para demostraciones técnicas y bases arquitectónicas de producción.

---

## 🏛️ Arquitectura del Sistema

```
  +--------------------------------------------------------------------------+
  |                             FASE 1: TRAINING                             |
  |                                                                          |
  |  [Synthetic Data / Feature Store]                                        |
  |                 │                                                        |
  |                 ▼                                                        |
  |          [src/train.py]                                                  |
  |                 │                                                        |
  |         ┌───────┴────────┐                                               |
  |         ▼                ▼                                               |
  |  [Logistic Reg.]   [Random Forest]                                       |
  |         │                │                                               |
  |         └───────┬────────┘                                               |
  |                 ▼                                                        |
  |          [MLflow Tracking] ────► Local (./mlruns) / AWS S3 Artifacts     |
  |                 │                                                        |
  |                 ▼ (Best Model Export)                                    |
  |       [models/best_model.pkl]                                            |
  +----------------─┼────────────────────────────────────────────────────────+
                    │
                    ▼
  +--------------------------------------------------------------------------+
  |                             FASE 2: SERVING                              |
  |                                                                          |
  |   [FastAPI Microservice (src/app.py)]                                    |
  |     ├── GET  /health   -> Status & Model Metadata                        |
  |     └── POST /predict  -> Churn Probability & Risk Rating                |
  |                                                                          |
  |   [Containerization: Dockerfile] (Python 3.11-slim, Non-root appuser)   |
  +----------------─┼────────────────────────────────────────────────────────+
                    │
                    ▼
  +--------------------------------------------------------------------------+
  |                             FASE 3: CI/CD                                |
  |                                                                          |
  |   [GitHub Actions: .github/workflows/mlops-pipeline.yml]                 |
  |     ├── Job 1: CI -> Flake8 (Linting) + Pytest (Unit Tests)             |
  |     └── Job 2: CD -> Docker Build + Push ECR + Deploy (EC2 / ECS / Run)  |
  +----------------─┼────────────────────────────────────────────────────────+
                    │
                    ▼
  +--------------------------------------------------------------------------+
  |                             FASE 4: INFRAESTRUCTURA                      |
  |                                                                          |
  |   [Terraform (IaC): terraform/main.tf]                                   |
  |     ├── AWS S3 Bucket (Cifrado + Versionado para artefactos MLflow)      |
  |     ├── AWS IAM Role + Instance Profile (Sin credenciales hardcodeadas)  |
  |     ├── AWS Security Group (Puertos 80, 8000, 22)                        |
  |     └── AWS EC2 t2.micro (Free Tier con Docker & AWS CLI preinstalado)   |
  +--------------------------------------------------------------------------+
```

---

## 📂 Estructura de Directorios

```text
arqMlOps/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml   # Pipeline CI/CD automatizado en GitHub Actions
├── models/
│   └── .gitkeep                 # Directorio destino para artefactos exportados
├── src/
│   ├── __init__.py
│   ├── app.py                   # Microservicio FastAPI con endpoints /health y /predict
│   └── train.py                 # Pipeline de entrenamiento, tracking y selección en MLflow
├── terraform/
│   ├── main.tf                  # Definición de recursos AWS (S3, IAM, Security Group, EC2)
│   ├── outputs.tf               # Outputs de infraestructura (IPs, Bucket Names, URLs)
│   └── variables.tf             # Variables parametrizables de Terraform
├── tests/
│   ├── __init__.py
│   └── test_api.py              # Pruebas unitarias y de validación con pytest
├── .dockerignore                # Optimización del contexto de construcción Docker
├── .flake8                      # Reglas de estilo y calidad de código
├── .gitignore                   # Exclusión de artefactos pesados y secretos
├── Dockerfile                   # Imagen Docker de producción optimizada y segura
├── docker-compose.yml           # Entorno de desarrollo local (MLflow + FastAPI + Train)
├── README.md                    # Documentación técnica completa
└── requirements.txt             # Dependencias del proyecto fijadas
```

---

## ⚡ Inicio Rápido con Docker Compose (Recomendado para Devs)

Si tienes Docker Desktop instalado, no necesitas instalar Python ni dependencias en tu máquina local:

```bash
# 1. Levantar la API FastAPI (puerto 8000)
docker compose up -d

# 2. (Opcional) Ejecutar reentrenamiento de modelos en contenedor hacia DagsHub MLflow
docker compose run --rm train

# 3. Ver experimentos y modelos en MLflow Web (DagsHub):
#    https://dagshub.com/coronelvalentino/arquitectura-mlops.mlflow
# 4. Probar inferencia en FastAPI Swagger: http://localhost:8000/docs
```

---

## 🛠️ Guía Rápida de Ejecución Local (Alternativa sin Docker)

### 1. Preparar Entorno Local

```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows:
.\venv\Scripts\activate
# Activar en Linux/macOS:
# source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Entrenar y Trackear Experimentos con MLflow

Ejecuta el script de entrenamiento para generar los datos sintéticos, entrenar ambos modelos (Logistic Regression vs Random Forest) y exportar el mejor:

```bash
python -m src.train
```

Para visualizar las métricas y artefactos en la interfaz gráfica de MLflow:

```bash
mlflow ui
# Abrir en el navegador: http://127.0.0.1:5000
```

### 3. Ejecutar Pruebas y Linting

```bash
# Verificación de sintaxis y estándares PEP8
flake8 src tests

# Ejecución de pruebas unitarias
pytest tests/ -v
```

### 4. Ejecutar el Servicio de Inferencia (FastAPI)

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Acceder a la documentación Swagger interactiva: `http://localhost:8000/docs`

#### Ejemplo de Inferencia:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure_months": 12,
       "monthly_charges": 65.50,
       "total_charges": 786.00,
       "num_support_tickets": 2,
       "online_security": 1
     }'
```

Respuesta esperada:
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.1842,
  "risk_level": "Bajo",
  "model_name": "Random_Forest",
  "run_id": "...",
  "latency_ms": 3.45
}
```

### 5. Construcción y Ejecución con Docker

```bash
# Construir la imagen
docker build -t customer-churn-api:latest .

# Ejecutar el contenedor
docker run -d -p 8000:8000 --name churn-api customer-churn-api:latest

# Probar estado
curl http://localhost:8000/health
```

### 6. Despliegue de Infraestructura con Terraform (AWS)

```bash
cd terraform

# Inicializar proveedores
terraform init

# Planificar cambios
terraform plan

# Aplicar (provisionar S3 y EC2)
terraform apply
```

---

## 🏭 De PoC a Producción Real: Consideraciones Senior

1. **Feature Store Centralizado**:
   - Reemplazar la generación sintética por Feast o AWS SageMaker Feature Store para evitar el *train-serve skew* y compartir features offline y online.

2. **Model Registry & Zero-Downtime Serving**:
   - Centralizar MLflow Tracking en un cluster ECS con RDS PostgreSQL como backend store y S3 como artifact store.
   - Usar tags o alias `@champion` en el Model Registry para desacoplar el ciclo de reentrenamiento del despliegue del contenedor.

3. **Monitoreo Continuo & Drift Detection**:
   - Instrumentar la API con `prometheus-fastapi-instrumentator`.
   - Transmitir asíncronamente las entradas y predicciones hacia un bucket o bus Kafka para calcular Data Drift y Concept Drift (ej. con Evidently AI).

4. **Escalado de Infraestructura**:
   - Migrar de la instancia única EC2 a Amazon ECS (Fargate) o Amazon EKS detrás de un Application Load Balancer con auto-scaling horizontal basado en concurrencia / latencia.
