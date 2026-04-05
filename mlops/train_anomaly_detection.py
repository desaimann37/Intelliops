import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import json

mlflow.set_tracking_uri("http://localhost:8083")
mlflow.set_experiment("anomaly-detection")

print("📊 Generating time series metrics data...")

np.random.seed(42)
n_normal = 900
n_anomaly = 100

# Normal metrics
normal_data = {
    "cpu_usage":        np.random.normal(35, 10, n_normal).clip(5, 70),
    "memory_usage":     np.random.normal(45, 12, n_normal).clip(10, 75),
    "error_rate":       np.random.normal(0.5, 0.3, n_normal).clip(0, 2),
    "latency_ms":       np.random.normal(120, 30, n_normal).clip(50, 250),
    "pod_restarts":     np.random.poisson(0.5, n_normal),
    "request_rate":     np.random.normal(500, 100, n_normal).clip(200, 900),
    "network_io_mb":    np.random.normal(50, 15, n_normal).clip(10, 100),
}

# Anomalous metrics
anomaly_data = {
    "cpu_usage":        np.random.normal(88, 5, n_anomaly).clip(75, 100),
    "memory_usage":     np.random.normal(92, 4, n_anomaly).clip(80, 100),
    "error_rate":       np.random.normal(15, 5, n_anomaly).clip(5, 30),
    "latency_ms":       np.random.normal(800, 100, n_anomaly).clip(500, 1200),
    "pod_restarts":     np.random.poisson(5, n_anomaly),
    "request_rate":     np.random.normal(50, 20, n_anomaly).clip(0, 150),
    "network_io_mb":    np.random.normal(200, 30, n_anomaly).clip(150, 300),
}

normal_df  = pd.DataFrame(normal_data)
anomaly_df = pd.DataFrame(anomaly_data)
normal_df["label"]  = 1   # normal
anomaly_df["label"] = -1  # anomaly

df = pd.concat([normal_df, anomaly_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Normal samples:  {n_normal}")
print(f"   Anomaly samples: {n_anomaly}")

feature_cols = [
    "cpu_usage", "memory_usage", "error_rate",
    "latency_ms", "pod_restarts", "request_rate", "network_io_mb"
]

X = df[feature_cols]
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── EXPERIMENT 1: Default contamination ──
print("\n🧪 Experiment 1: contamination=0.1")
with mlflow.start_run(run_name="contamination-0.1"):
    model1 = IsolationForest(contamination=0.1, random_state=42)
    model1.fit(X_scaled)
    preds1 = model1.predict(X_scaled)

    correct = (preds1 == y.values).sum()
    accuracy = correct / len(y)

    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("n_samples", len(df))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("anomalies_detected", (preds1 == -1).sum())

    mlflow.sklearn.log_model(
        model1, "model",
        registered_model_name="anomaly-detector"
    )

    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Anomalies detected: {(preds1==-1).sum()}")

# ── EXPERIMENT 2: Higher contamination ──
print("\n🧪 Experiment 2: contamination=0.15")
with mlflow.start_run(run_name="contamination-0.15"):
    model2 = IsolationForest(contamination=0.15, random_state=42)
    model2.fit(X_scaled)
    preds2 = model2.predict(X_scaled)

    correct2 = (preds2 == y.values).sum()
    accuracy2 = correct2 / len(y)

    mlflow.log_param("contamination", 0.15)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("n_samples", len(df))
    mlflow.log_metric("accuracy", accuracy2)
    mlflow.log_metric("anomalies_detected", (preds2 == -1).sum())

    mlflow.sklearn.log_model(
        model2, "model",
        registered_model_name="anomaly-detector"
    )

    print(f"   Accuracy: {accuracy2:.3f}")
    print(f"   Anomalies detected: {(preds2==-1).sum()}")

# ── EXPERIMENT 3: More estimators ──
print("\n🧪 Experiment 3: n_estimators=200")
with mlflow.start_run(run_name="n-estimators-200"):
    model3 = IsolationForest(
        contamination=0.1,
        n_estimators=200,
        random_state=42
    )
    model3.fit(X_scaled)
    preds3 = model3.predict(X_scaled)

    correct3 = (preds3 == y.values).sum()
    accuracy3 = correct3 / len(y)

    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("n_samples", len(df))
    mlflow.log_metric("accuracy", accuracy3)
    mlflow.log_metric("anomalies_detected", (preds3 == -1).sum())

    mlflow.sklearn.log_model(
        model3, "model",
        registered_model_name="anomaly-detector"
    )

    print(f"   Accuracy: {accuracy3:.3f}")
    print(f"   Anomalies detected: {(preds3==-1).sum()}")

# Save scaler for inference
import joblib
joblib.dump(scaler, "mlops/anomaly_scaler.pkl")
print("\n✅ Scaler saved to mlops/anomaly_scaler.pkl")

print("\n" + "="*50)
print("📊 ANOMALY DETECTION EXPERIMENT SUMMARY")
print("="*50)
print(f"Experiment 1 (cont=0.10):  accuracy={accuracy:.3f}")
print(f"Experiment 2 (cont=0.15):  accuracy={accuracy2:.3f}")
print(f"Experiment 3 (est=200):    accuracy={accuracy3:.3f}")

best = max(accuracy, accuracy2, accuracy3)
print(f"\n🏆 Best accuracy: {best:.3f}")
print(f"✅ Models registered in MLflow as 'anomaly-detector'")
print("="*50)
