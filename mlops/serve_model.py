import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

# Connect to MLflow
mlflow.set_tracking_uri("http://localhost:8083")

# Load best model (version 1 — highest accuracy)
model = mlflow.sklearn.load_model(
    "models:/deploy-risk-classifier/1"
)

def predict_deployment_risk(
    test_coverage,
    cpu_load,
    memory_load,
    pr_size,
    critical_cves=0,
    high_cves=0,
    pod_restarts=0,
    hour_of_day=12
):
    features = pd.DataFrame([{
        "test_coverage":  test_coverage,
        "cpu_load":       cpu_load,
        "memory_load":    memory_load,
        "pr_size":        pr_size,
        "critical_cves":  critical_cves,
        "high_cves":      high_cves,
        "pod_restarts":   pod_restarts,
        "hour_of_day":    hour_of_day
    }])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    risk_map = {0: "HIGH", 1: "LOW", 2: "MEDIUM"}
    risk = risk_map.get(prediction, "MEDIUM")
    confidence = round(float(max(probability)) * 100, 1)

    return risk, confidence

if __name__ == "__main__":
    print("🤖 Deploy Risk Classifier — ML Model Predictions\n")
    print("="*55)

    test_cases = [
        {
            "name": "Safe deployment",
            "test_coverage": 85, "cpu_load": 40,
            "memory_load": 50, "pr_size": 100,
            "critical_cves": 0, "high_cves": 1,
            "pod_restarts": 0, "hour_of_day": 14
        },
        {
            "name": "Risky deployment",
            "test_coverage": 55, "cpu_load": 85,
            "memory_load": 90, "pr_size": 800,
            "critical_cves": 2, "high_cves": 10,
            "pod_restarts": 8, "hour_of_day": 23
        },
        {
            "name": "Medium risk",
            "test_coverage": 72, "cpu_load": 65,
            "memory_load": 70, "pr_size": 300,
            "critical_cves": 0, "high_cves": 4,
            "pod_restarts": 3, "hour_of_day": 10
        },
    ]

    for case in test_cases:
        name = case.pop("name")
        risk, confidence = predict_deployment_risk(**case)
        emoji = "✅" if risk == "LOW" else "⚠️" if risk == "MEDIUM" else "🚫"
        print(f"{emoji} {name}")
        print(f"   Risk Level:  {risk}")
        print(f"   Confidence:  {confidence}%")
        print()
