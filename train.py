"""
Punto de entrada raíz para el entrenamiento y tracking con MLflow.
Redirige a src.train.train_and_track().
"""

from src.train import train_and_track

if __name__ == "__main__":
    train_and_track()
