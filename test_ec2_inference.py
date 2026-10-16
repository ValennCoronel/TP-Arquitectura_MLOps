"""
Script de prueba de inferencia remota para la API desplegada en AWS EC2.

Diseñado para ejecutarse de forma independiente sin dependencias externas
(utiliza únicamente la librería estándar de Python: urllib y json).

Uso:
    python test_ec2_inference.py
    python test_ec2_inference.py <IP_O_HOST>
    python test_ec2_inference.py 52.90.232.241:8000
    python test_ec2_inference.py localhost:8000
"""

import json
import sys
import time
import urllib.error
import urllib.request

# IP pública de tu instancia EC2 por defecto
DEFAULT_HOST = "52.90.232.241:8000"


def get_base_url():
    """Determina la URL base a partir de argumentos o valor por defecto."""
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host.rstrip("/")


def check_health(base_url: str) -> bool:
    """Verifica el estado del servicio /health."""
    url = f"{base_url}/health"
    print(f"\n[1] Verificando Health Check en: {url} ...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EC2-Inference-Test"})
        with urllib.request.urlopen(req, timeout=15) as response:
            status_code = response.getcode()
            data = json.loads(response.read().decode("utf-8"))
            print(f"    -> Código HTTP: {status_code}")
            print(f"    -> Estado API : {data.get('status')}")
            print(f"    -> Modelo cargado: {data.get('model_loaded')}")
            print(f"    -> Modelo actual : {data.get('model_name') or 'Ninguno'}")
            print(f"    -> Run ID actual : {data.get('run_id') or 'N/A'}")

            if not data.get("model_loaded"):
                print("    [!] AVISO: El modelo aún no está cargado en la instancia.")
                print("        (Si recién hiciste merge a main, espera a que el CD finalice el deploy).")
                return False
            return True
    except urllib.error.URLError as e:
        print(f"    [X] Error de conexión: {e}")
        return False
    except Exception as e:
        print(f"    [X] Error inesperado: {e}")
        return False


def send_prediction(base_url: str, case_name: str, payload: dict):
    """Envía un payload de cliente al endpoint /predict y muestra el resultado."""
    url = f"{base_url}/predict"
    print(f"\n[*] Probando inferencia: '{case_name}'")
    print("    Atributos del cliente:")
    for k, v in payload.items():
        print(f"      - {k}: {v}")

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "EC2-Inference-Test",
        },
    )

    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            latency_total = round((time.perf_counter() - start_time) * 1000, 2)
            result = json.loads(response.read().decode("utf-8"))
            pred = result.get("churn_prediction")
            prob = result.get("churn_probability", 0.0)
            risk = result.get("risk_level", "Desconocido")
            model = result.get("model_name", "N/A")
            latency_api = result.get("latency_ms", "N/A")

            print("    Resultado de Inferencia:")
            print(f"      [OK] Predicción Churn   : {pred} ({'Se fuga' if pred == 1 else 'Permanece'})")
            print(f"      [OK] Probabilidad Fuga  : {prob:.2%}")
            print(f"      [OK] Nivel de Riesgo    : {risk}")
            print(f"      [OK] Modelo Servidor    : {model}")
            print(f"      [OK] Latencia Servidor  : {latency_api} ms")
            print(f"      [OK] Latencia Roundtrip : {latency_total} ms")
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        print(f"    [X] Error HTTP {e.code}: {error_msg}")
    except urllib.error.URLError as e:
        print(f"    [X] Error de conexión hacia EC2: {e.reason}")
    except Exception as e:
        print(f"    [X] Error: {e}")


def main():
    base_url = get_base_url()
    print("=" * 65)
    print("   TEST DE INFERENCIA EN VIVO - CLIENTE EC2 (MLOps POC)")
    print(f"   Target URL: {base_url}")
    print("=" * 65)

    is_healthy = check_health(base_url)

    # Escenarios de prueba con datos representativos
    test_cases = [
        (
            "Cliente 1: Riesgo Alto (Poca antigüedad, muchos reclamos, sin seguridad)",
            {
                "tenure_months": 2,
                "monthly_charges": 89.9,
                "total_charges": 179.8,
                "num_support_tickets": 6,
                "online_security": 0,
            },
        ),
        (
            "Cliente 2: Riesgo Bajo / Cliente Fiel (Alta antigüedad, sin reclamos)",
            {
                "tenure_months": 48,
                "monthly_charges": 45.0,
                "total_charges": 2160.0,
                "num_support_tickets": 0,
                "online_security": 1,
            },
        ),
        (
            "Cliente 3: Caso Promedio / Moderado",
            {
                "tenure_months": 14,
                "monthly_charges": 65.5,
                "total_charges": 917.0,
                "num_support_tickets": 2,
                "online_security": 1,
            },
        ),
    ]

    print("\n" + "=" * 65)
    print("   EJECUTANDO CASOS DE PRUEBA DE INFERENCIA (/predict)")
    print("=" * 65)

    for case_name, payload in test_cases:
        send_prediction(base_url, case_name, payload)

    print("\n" + "=" * 65)
    print(f"[OK] Pruebas finalizadas. Documentacion Swagger: {base_url}/docs")
    print("=" * 65)


if __name__ == "__main__":
    main()
