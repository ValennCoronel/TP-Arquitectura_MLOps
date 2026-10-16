# =========================================================================
# DOCKERFILE - FASTAPI MLOPS SERVING
# =========================================================================
# Imagen base ligera oficial de Python sobre Debian slim
FROM python:3.11-slim AS runtime

# Variables de entorno para optimización de Python en contenedores:
# - Evitar la generación de archivos .pyc en disco
# - Salida estándar sin buffer (los logs se emiten directamente a stdout/stderr)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Directorio de trabajo
WORKDIR /app

# Buenas prácticas de seguridad en Producción:
# Instalar dependencias del sistema mínimas necesarias y limpiar caché de apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario sin privilegios root (Principio de menor privilegio)
RUN useradd -m -u 1000 appuser

# Copiar requirements primero para aprovechar la caché de capas de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente y directorio de modelos
COPY src/ ./src/
COPY models/ ./models/

# Crear y ajustar permisos para directorios de trabajo y almacenamiento
RUN mkdir -p /mlflow /app/models && \
    chown -R appuser:appuser /app /mlflow

# Cambiar al usuario seguro
USER appuser

# Exponer el puerto de la API
EXPOSE 8000

# Healthcheck nativo de Docker para orquestadores (Docker Swarm, AWS ECS, Kubernetes)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# =========================================================================
# NOTA DE PRODUCCIÓN:
# En entornos productivos de alta concurrencia:
# 1. Se puede usar Gunicorn con UvicornWorker para escalado vertical por procesos:
#    CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.app:app", "--bind", "0.0.0.0:8000"]
# 2. El modelo no se empaqueta dentro de la imagen; en su lugar, un script de entrypoint
#    o un Kubernetes Init-Container descarga el modelo del Registry/S3 al iniciar el Pod.
# =========================================================================
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
