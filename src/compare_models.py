"""
Módulo de Comparación Automática de Modelos (Champion vs. Challenger) para Pull Requests.

Genera un reporte en Markdown comparando el modelo candidato entrenado en la rama actual
(Challenger) contra el modelo en producción en MLflow Registry o el modelo base (Champion).
"""

import json
import os
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()


def fetch_mlflow_champion_metrics(
    challenger_run_id: str,
) -> Optional[Tuple[str, str, Dict[str, float]]]:
    """
    Intenta obtener las métricas del modelo actual en Producción desde MLflow / DagsHub.
    Retorna (nombre_modelo, identificador, diccionario_metricas) o None si no está disponible.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri or tracking_uri.startswith("file:"):
        return None

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # 1. Revisar si MLFLOW_MODEL_URI apunta a una versión o run específico en Producción
        prod_uri = os.getenv("MLFLOW_MODEL_URI", "")
        if prod_uri.startswith("models:/"):
            parts = prod_uri.replace("models:/", "").split("/")
            if len(parts) == 2:
                model_name, version = parts[0], parts[1]
                mv = client.get_model_version(name=model_name, version=version)
                if mv.run_id and mv.run_id != challenger_run_id:
                    run = client.get_run(mv.run_id)
                    algo = run.data.tags.get("mlflow.runName", f"{model_name} v{version}")
                    return (
                        f"Producción ({algo})",
                        f"v{version} (`{mv.run_id[:8]}`)",
                        dict(run.data.metrics),
                    )

        # 2. Buscar versiones previas registradas en Model Registry
        versions = client.search_model_versions("name='customer_churn_model'")
        sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        for mv in sorted_versions:
            if mv.run_id and mv.run_id != challenger_run_id:
                run = client.get_run(mv.run_id)
                algo = run.data.tags.get("mlflow.runName", "RegisteredModel")
                return (
                    f"Registry Champion ({algo})",
                    f"v{mv.version} (`{mv.run_id[:8]}`)",
                    dict(run.data.metrics),
                )
    except Exception as exc:
        print(f"[!] Aviso al consultar Champion en MLflow remoto: {exc}")

    return None


def generate_comparison_report(
    summary_path: str = os.path.join("models", "training_summary.json"),
    output_md_path: str = "model_comparison_report.md",
) -> str:
    """
    Construye el reporte Markdown comparando el Challenger contra el Champion/Baseline.
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"No se encontró '{summary_path}'. Ejecuta 'python -m src.train' primero."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    challenger_name = summary["best_model_name"]
    challenger_run_id = summary["best_run_id"]
    challenger_metrics = summary["best_metrics"]
    challenger_version = summary.get("registered_version")
    runs = summary.get("runs", {})

    # Intentar obtener el Champion desde MLflow remoto; si no, usar el otro modelo del benchmark
    remote_champion = fetch_mlflow_champion_metrics(challenger_run_id)

    if remote_champion:
        champ_label, champ_id, champ_metrics = remote_champion
        comparison_source = "MLflow Model Registry (DagsHub Producción)"
    else:
        # Usar el modelo alternativo entrenado (ej. Logistic_Regression) como Baseline de referencia
        baseline_candidates = [k for k in runs.keys() if k != challenger_name]
        if baseline_candidates:
            base_key = baseline_candidates[0]
            base_run = runs[base_key]
            champ_label = f"Baseline ({base_key})"
            champ_id = f"`{base_run['run_id'][:8]}`"
            champ_metrics = base_run["metrics"]
        else:
            champ_label = "Baseline Actual"
            champ_id = "N/A"
            champ_metrics = challenger_metrics
        comparison_source = "Benchmark Interno del Pipeline (Baseline vs. Challenger)"

    f1_challenger = challenger_metrics.get("f1_score", 0.0)
    f1_champion = champ_metrics.get("f1_score", 0.0)
    f1_diff = f1_challenger - f1_champion

    is_better_or_equal = f1_challenger >= f1_champion

    if f1_diff > 0.0001:
        verdict_badge = "✅ **APROBADO: El modelo Challenger SUPERA al modelo de referencia**"
        verdict_detail = (
            f"El candidato **{challenger_name}** mejora el **F1-Score** en "
            f"**`+{f1_diff:.4f}`** (`+{(f1_diff * 100):.2f}%`) respecto a **{champ_label}**."
        )
    elif abs(f1_diff) <= 0.0001:
        verdict_badge = "✅ **APROBADO: El modelo Challenger MANTIENE el rendimiento actual**"
        verdict_detail = (
            f"El candidato **{challenger_name}** iguala el **F1-Score** (`{f1_challenger:.4f}`) "
            f"sin degradación estadística frente a **{champ_label}**."
        )
    else:
        verdict_badge = "⚠️ **ATENCIÓN: El modelo Challenger presenta menor F1-Score**"
        verdict_detail = (
            f"El candidato **{challenger_name}** tiene una diferencia de **`{f1_diff:.4f}`** "
            f"en **F1-Score** frente a **{champ_label}**. Revisar hiperparámetros antes del merge."
        )

    challenger_id_str = f"`{challenger_run_id[:8]}`"
    if challenger_version:
        challenger_id_str = f"v{challenger_version} ({challenger_id_str})"

    lines = [
        "## 🤖 Reporte MLOps: Evaluación Champion vs. Challenger",
        "",
        f"> {verdict_badge}",
        f"> {verdict_detail}",
        "",
        f"**Fuente de comparación:** {comparison_source}",
        "",
        "### 📊 1. Comparación Directa (Producción/Referencia vs. Candidato PR)",
        "",
        "| Métrica | 🏆 Referencia / Prod | 🚀 Challenger (PR) | Delta (Diferencia) | Estado |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    metric_labels = [
        ("f1_score", "F1-Score (Métrica Principal)"),
        ("roc_auc", "ROC-AUC"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]

    for key, label in metric_labels:
        c_val = champ_metrics.get(key, 0.0)
        n_val = challenger_metrics.get(key, 0.0)
        diff = n_val - c_val
        sign = "+" if diff >= 0 else ""
        if diff > 0.0001:
            icon = "🟢 Mejora"
        elif abs(diff) <= 0.0001:
            icon = "🔵 Igual"
        else:
            icon = "🟡 Menor"
        lines.append(
            f"| **{label}** | `{c_val:.4f}` | **`{n_val:.4f}`** | `{sign}{diff:.4f}` | {icon} |"
        )

    lines.extend(
        [
            "",
            "### 🧪 2. Torneo de Algoritmos en esta Ejecución (MLflow Experiment)",
            "",
            "| Algoritmo Evaluado | Run ID | F1-Score | ROC-AUC | Accuracy | Selección |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for run_name, rdata in runs.items():
        rmetrics = rdata["metrics"]
        rid = rdata["run_id"][:8]
        is_winner = "🏆 **Ganador**" if run_name == challenger_name else "Descartado"
        lines.append(
            f"| `{run_name}` | `{rid}` | `{rmetrics['f1_score']:.4f}` | "
            f"`{rmetrics['roc_auc']:.4f}` | `{rmetrics['accuracy']:.4f}` | {is_winner} |"
        )

    lines.extend(
        [
            "",
            "### 🔗 3. Metadatos de Gobernanza y Linaje",
            f"- **Experimento MLflow:** `{summary['experiment_name']}`",
            f"- **Modelo Seleccionado (Challenger):** `{challenger_name}` ({challenger_id_str})",
            f"- **Modelo de Referencia (Champion):** `{champ_label}` ({champ_id})",
            f"- **Decisión del Quality Gate:** {'APROBADO' if is_better_or_equal else 'REVISAR'}",
            "",
        ]
    )

    report_content = "\n".join(lines)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[✓] Reporte comparativo generado en '{output_md_path}'")
    return report_content


if __name__ == "__main__":
    report = generate_comparison_report()
    print("\n" + report)
