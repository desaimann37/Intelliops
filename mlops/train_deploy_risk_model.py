import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json

# Connect to MLflow
mlflow.set_tracking_uri("http://localhost:8083")
mlflow.set_experiment("deploy-risk-classifier")

# ── GENERATE TRAINING DATA ──
# Simulates historical deployment outcomes
np.random.seed(42)
n_samples = 1000

print("📊 Generating training data...")

data = {
    "test_coverage":  np.random.uniform(40, 100, n_samples),
    "cpu_load":       np.random.uniform(10, 95, n_samples),
    "memory_load":    np.random.uniform(10, 95, n_samples),
    "pr_size":        np.random.randint(10, 1000, n_samples),
    "critical_cves":  np.random.randint(0, 5, n_samples),
    "high_cves":      np.random.randint(0, 20, n_samples),
    "pod_restarts":   np.random.randint(0, 15, n_samples),
    "hour_of_day":    np.random.randint(0, 24, n_samples),
}

df = pd.DataFrame(data)

# Generate risk labels based on rules
def calculate_risk(row):
    score = 0
    if row["test_coverage"] < 70:  score += 3
    if row["cpu_load"] > 80:       score += 3
    if row["memory_load"] > 85:    score += 3
    if row["critical_cves"] > 0:   score += 5
    if row["high_cves"] > 5:       score += 2
    if row["pr_size"] > 500:       score += 1
    if row["pod_restarts"] > 5:    score += 2
    if row["hour_of_day"] >= 22 or row["hour_of_day"] <= 6:
        score += 1

    if score >= 6:   return "HIGH"
    elif score >= 3: return "MEDIUM"
    else:            return "LOW"

df["risk_level"] = df.apply(calculate_risk, axis=1)

print(f"   Total samples: {n_samples}")
print(f"   LOW risk:    {(df['risk_level']=='LOW').sum()}")
print(f"   MEDIUM risk: {(df['risk_level']=='MEDIUM').sum()}")
print(f"   HIGH risk:   {(df['risk_level']=='HIGH').sum()}")

# Encode labels
le = LabelEncoder()
df["risk_encoded"] = le.fit_transform(df["risk_level"])

# Features and target
feature_cols = [
    "test_coverage", "cpu_load", "memory_load",
    "pr_size", "critical_cves", "high_cves",
    "pod_restarts", "hour_of_day"
]

X = df[feature_cols]
y = df["risk_encoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── EXPERIMENT 1: Default parameters ──
print("\n🧪 Running Experiment 1: Default parameters...")
with mlflow.start_run(run_name="default-params"):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("train_samples", len(X_train))

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name="deploy-risk-classifier"
    )

    print(f"   Accuracy: {acc:.3f}")
    print(f"   F1 Score: {f1:.3f}")

# ── EXPERIMENT 2: Tuned parameters ──
print("\n🧪 Running Experiment 2: Tuned parameters...")
with mlflow.start_run(run_name="tuned-params"):
    model2 = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)

    acc2 = accuracy_score(y_test, y_pred2)
    f1_2 = f1_score(y_test, y_pred2, average="weighted")

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("train_samples", len(X_train))

    mlflow.log_metric("accuracy", acc2)
    mlflow.log_metric("f1_score", f1_2)

    mlflow.sklearn.log_model(
        model2, "model",
        registered_model_name="deploy-risk-classifier"
    )

    print(f"   Accuracy: {acc2:.3f}")
    print(f"   F1 Score: {f1_2:.3f}")

# ── EXPERIMENT 3: More estimators ──
print("\n🧪 Running Experiment 3: More estimators...")
with mlflow.start_run(run_name="high-estimators"):
    model3 = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=4,
        random_state=42
    )
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)

    acc3 = accuracy_score(y_test, y_pred3)
    f1_3 = f1_score(y_test, y_pred3, average="weighted")

    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("train_samples", len(X_train))

    mlflow.log_metric("accuracy", acc3)
    mlflow.log_metric("f1_score", f1_3)

    mlflow.sklearn.log_model(
        model3, "model",
        registered_model_name="deploy-risk-classifier"
    )

    print(f"   Accuracy: {acc3:.3f}")
    print(f"   F1 Score: {f1_3:.3f}")

# ── SUMMARY ──
print("\n" + "="*50)
print("📊 EXPERIMENT SUMMARY")
print("="*50)
print(f"Experiment 1 (default):  accuracy={acc:.3f}  f1={f1:.3f}")
print(f"Experiment 2 (tuned):    accuracy={acc2:.3f}  f1={f1_2:.3f}")
print(f"Experiment 3 (high-est): accuracy={acc3:.3f}  f1={f1_3:.3f}")

best_acc = max(acc, acc2, acc3)
best_exp = ["default", "tuned", "high-estimators"][[acc, acc2, acc3].index(best_acc)]
print(f"\n🏆 Best model: {best_exp} with accuracy {best_acc:.3f}")
print(f"✅ All experiments logged to MLflow at http://localhost:8083")
print("="*50)
