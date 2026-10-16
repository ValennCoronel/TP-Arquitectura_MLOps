# 🎤 Plan de Demo y Guion de Presentación en Vivo: Suite MLOps End-to-End

Este documento es tu **guía paso a paso para presentar la arquitectura MLOps en vivo** ante una audiencia técnica o de negocio. Está estructurada para durar entre **15 y 20 minutos**, mostrando el problema que resuelve, las ventajas de la suite, la experimentación en MLflow, el Quality Gate en Pull Requests y el despliegue automatizado en AWS.

---

## 📋 Checklist Pre-Demo (10 minutos antes de empezar)

1. **Encender o verificar la infraestructura en AWS:**
   ```powershell
   .\aws-control.ps1 status
   # Si está detenida: .\aws-control.ps1 start
   # Si fue destruida: .\aws-control.ps1 up
   ```
2. **Tener abiertas estas 4 pestañas en el navegador:**
   - **Pestaña 1:** El repositorio en **GitHub** (vista del código y pestaña *Actions*).
   - **Pestaña 2:** **DagsHub / MLflow UI** (pestaña *Experiments* y *Models / Registry*).
   - **Pestaña 3:** **Swagger UI de Producción en AWS** (`http://<IP_EC2>:8000/docs`).
   - **Pestaña 4:** Consola de **AWS EC2 / ECR** (opcional, para mostrar los recursos reales).
3. **Tener abierta tu terminal en VS Code** en la rama `development`:
   ```powershell
   git checkout development
   ```

---

## 🎬 Estructura de la Demo (Paso a Paso)

### Bloque 1: El Problema y la Propuesta de Valor (3 min)
**Qué explicar al público:**
- *"En la mayoría de los equipos de Data Science, los modelos viven en notebooks de Jupyter aislados. Pasar un modelo a producción suele tardar semanas, se pierden los hiperparámetros con los que fue entrenado y nadie sabe con certeza si el nuevo modelo es mejor que el que está corriendo hoy en producción."*
- *"Esta suite MLOps resuelve ese puente de punta a punta aplicando cuatro pilares:"*
  1. **Gobernanza y Linaje (MLflow + DagsHub):** Ningún modelo es una caja negra; cada entrenamiento registra métricas, parámetros y binarios versionados.
  2. **Calidad y Evaluación Automática (CI + Champion vs. Challenger):** Cada Pull Request entrena, testea la API y comenta automáticamente si el nuevo modelo supera al de producción.
  3. **Infraestructura Reproducible y Efímera (Terraform):** Todo AWS (EC2, ECR, S3, IAM) se levanta o destruye con un solo comando manteniendo costo $0 cuando no se usa.
  4. **Despliegue Seguro Zero-Touch (CD con AWS SSM):** Sin claves SSH expuestas ni pasos manuales al hacer merge a `main`.

---

### Bloque 2: Experimentación y Comparación de Modelos en MLflow (4 min)
**Qué hacer en vivo:**
1. En tu terminal local, ejecuta el pipeline de entrenamiento:
   ```powershell
   python -m src.train
   ```
2. Muestra en la consola cómo el script entrena dos familias de algoritmos (**Logistic Regression** vs. **Random Forest**), evalúa las 5 métricas (`Accuracy`, `F1-Score`, `Precision`, `Recall`, `ROC-AUC`) y elige automáticamente al **Modelo Campeón** por `F1-Score`.
3. **Pasa a la Pestaña 2 (DagsHub / MLflow UI):**
   - Entra al experimento `customer-churn-prediction`.
   - Selecciona las dos últimas corridas (`Logistic_Regression` y `Random_Forest`) y haz clic en **"Compare"**.
   - **Qué destacar al público:**
     - En la tabla de corridas de MLflow, haz clic en **"Columns"** -> activa el tag **`mlops.role`** para mostrar en pantalla cómo MLflow identifica automáticamente quién es **`CHAMPION`**, quién es **`CHALLENGER`** y quién fue **`DISCARDED`**.
     - Muestra la gráfica de coordenadas paralelas / barras comparando el `F1-Score` y `ROC-AUC`.
     - Entra a los artefactos de la corrida ganadora y muestra cómo MLflow guardó automáticamente el binario `model.pkl`, el esquema de entrada (`input_example.json`) y las versiones exactas de las librerías (`requirements.txt`).
     - Muestra la sección **Model Registry (`customer_churn_model`)** donde la versión en producción aparece en Stage **`Production` (`mlops.role = CHAMPION`)** y la versión en prueba en Stage **`Staging` (`mlops.role = CHALLENGER`)**.

---

### Bloque 3: El "Efecto Wow" — Cambio de Hiperparámetros y Reporte en Pull Request (5 min)
**Qué explicar al público:**
- *"¿Qué pasa cuando un Data Scientist quiere proponer una mejora al modelo? En lugar de mandar un archivo `.pkl` por Slack o correo, hace un cambio en el código y abre un Pull Request."*

**Qué hacer en vivo:**
1. Abre [`src/train.py`](src/train.py#L138-L143) en VS Code y cambia un hiperparámetro del `RandomForestClassifier`, por ejemplo:
   - Cambia `n_estimators=120` a `n_estimators=150` o `max_depth=6` a `max_depth=8` (líneas 138-140).
2. Haz commit y push a la rama `development`:
   ```powershell
   git add src/train.py
   git commit -m "feat(model): ajustar profundidad de Random Forest para mejorar F1"
   git push origin development
   ```
3. **Ve a la Pestaña 1 (GitHub)** y abre un **Pull Request** desde `development` hacia `main`.
4. Muestra cómo se dispara automáticamente el workflow de GitHub Actions:
   - Valida el estilo de código con `flake8`.
   - Ejecuta los 5 tests unitarios de la API y del modelo con `pytest`.
   - Ejecuta la evaluación **Champion vs. Challenger**.
5. **Muestra el comentario automático que aparece en la conversación del Pull Request:**
   - **Qué destacar al público:**
     - *"Miren cómo el bot de MLOps acaba de comentar en el Pull Request sin intervención humana."*
     - Señala el veredicto (`✅ APROBADO: El modelo Challenger SUPERA/MANTIENE el rendimiento actual`).
     - Señala la tabla comparativa donde el revisor del PR puede ver exactamente el `F1-Score` y `ROC-AUC` de **Producción (Champion)** frente al **Candidato del PR (Challenger)** y el torneo interno de algoritmos.

---

### Bloque 4: Pase a Producción (Merge a `main`) e Inferencia en Vivo en AWS (4 min)
**Qué hacer en vivo:**
1. En el Pull Request de GitHub, haz clic en **"Merge pull request"** -> **"Confirm merge"**.
2. Entra a la pestaña **Actions** y muestra cómo ahora se desbloquea el segundo trabajo: **`Build & Deploy to AWS Free Tier (ECR + EC2)`**:
   - Explica mientras corre (~60 seg):
     - *"GitHub Actions empaqueta el microservicio FastAPI en una imagen Docker inmutable, la sube a Amazon ECR con el hash del commit, descubre automáticamente qué instancia EC2 está encendida mediante su etiqueta de Terraform y le ordena por AWS Systems Manager actualizar el contenedor."*
3. **Pasa a la Pestaña 3 (`http://<IP_EC2>:8000/docs` - Swagger UI en AWS):**
   - Ejecuta primero `GET /health` -> **"Try it out"** -> **"Execute"** para mostrar que el servicio responde `"status": "healthy"` y `"model_loaded": true`.
   - Abre `POST /predict` -> **"Try it out"** y prueba dos escenarios de negocio en vivo:
     - **Caso 1: Cliente Fidelizado (Bajo Riesgo):**
       ```json
       {
         "tenure_months": 48,
         "monthly_charges": 35.0,
         "total_charges": 1680.0,
         "num_support_tickets": 0,
         "online_security": 1
       }
       ```
       *(Muestra la respuesta en `< 15 ms` con `risk_level: "Bajo"`).*
     - **Caso 2: Cliente en Peligro de Fuga (Alto Riesgo):**
       ```json
       {
         "tenure_months": 2,
         "monthly_charges": 115.0,
         "total_charges": 230.0,
         "num_support_tickets": 7,
         "online_security": 0
       }
       ```
       *(Muestra cómo cambia la probabilidad y el `risk_level` para que el CRM dispare una acción de retención).*

---

### Bloque 5: Cierre — Eficiencia de Costos y FinOps con Terraform (2 min)
**Qué explicar al público para cerrar:**
- *"Finalmente, una arquitectura moderna también debe cuidar el presupuesto de nube (FinOps). Con nuestro script de control sobre Terraform, podemos pausar el cómputo en segundos o destruir toda la infraestructura al terminar la jornada y volver a levantarla mañana sin reconfigurar ni un solo Secret en GitHub."*
- Muestra en la terminal:
  ```powershell
  .\aws-control.ps1 status
  ```

---

## 💡 Preguntas Frecuentes del Público (Q&A Cheat Sheet)

- **¿Qué pasa si envío datos corruptos o tipos incorrectos a la API?**
  - *Respuesta:* FastAPI + Pydantic validan el contrato de entrada antes de tocar el modelo. *(Puedes probar en vivo en `/predict` poniendo `"tenure_months": -5` y mostrar que devuelve `422 Unprocessable Entity` automáticamente).*
- **¿Cómo harían rollback si un modelo en producción empieza a fallar?**
  - *Respuesta:* Tenemos doble capacidad de rollback:
    1. A nivel de **Modelo**: cambiar la variable `MLFLOW_MODEL_URI` a la versión anterior en el Model Registry (`models:/customer_churn_model/1`) y llamar a `POST /reload-model` sin siquiera reiniciar el contenedor.
    2. A nivel de **Contenedor**: en Amazon ECR cada imagen queda guardada con el SHA del commit de Git, permitiendo volver a la imagen exacta anterior en segundos.
- **¿Cómo escalaría esta PoC para soportar 10,000 peticiones por segundo?**
  - *Respuesta:* Como el servicio ya está contenerizado en ECR y desacoplado de MLflow, en [`terraform/main.tf`](terraform/main.tf) simplemente se cambia el recurso `aws_instance` por **Amazon ECS Fargate** o **AWS App Runner** detrás de un **Application Load Balancer (ALB)** con autoescalado horizontal, sin tocar ni una sola línea del código Python de la API.
