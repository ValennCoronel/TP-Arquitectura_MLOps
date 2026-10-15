"""
Punto de entrada raíz para el servicio FastAPI de Serving.
Expone la aplicación FastAPI desde src.app.
"""

from src.app import app

__all__ = ["app"]
