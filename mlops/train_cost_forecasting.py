import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

mlflow.set_tracking_uri("http://localhost:8083")
mlflow.set_experiment("cost-forecasting")

print("📊 Generating historical resource usage data...")

np.random.seed(42)
n_days = 365

# Simulate 1 year of daily cluster metrics
days = np.arange(n_days)

# Seasonal patterns
seasonal = 20 * np.sin(2 * np.pi * days / 365)
weekly   = 10 * np.sin(2 * np.pi * days / 7)
trend    = 0.1 * days

data = {
    "day":              days,
    "cpu_avg":          (35 + trend * 0.05 + seasonal * 0.3
                         + weekly * 0.2
                         + np.random.normal(0, 5, n_days)).clip(10, 90),
    "memory_avg":       (45 + trend * 0.03 + seasonal * 0.2
                         + np.random.normal(0, 4, n_days)).clip(15, 85),
    "pod_count":        (20 + trend * 0.02 + weekly
                         + np.random.normal(0, 2, n_days)).clip(5, 50).astype(int),
    "storage_gb":       (100 + trend * 0.5
                         + np.random.normal(0, 10, n_days)).clip(50, 300),
    "network_gb":       (50 + seasonal * 0.5
                         + np.random.normal(0, 8, n_days)).clip(10, 150),
    "deployments":      np.random.poisson(3, n_days),
    "incidents":        np.random.poisson(0.5, n_days),
}

df = pd.DataFrame(data)

# Calculate daily cost
df["daily_cost"] = (
    df["cpu_avg"] * 0.05 +
    df["memory_avg"] * 0.03 +
    df["pod_count"] * 0.10 +
    df["storage_gb"] * 0.02 +
    df["network_gb"] * 0.01 +
    df["deployments"] * 0.50 +
    np.random.normal(0, 2, n_days)
).clip(5, 50)

# Monthly cost
df["monthly_cost"] = df["daily_cost"] * 30

print(f"   Days simulated: {n_days}")
print(f"   Avg daily cost:   ${df['daily_cost'].mean():.2f}")
print(f"   Avg monthly cost: ${df['monthly_cost'].mean():.2f}")

feature_cols = [
    "cpu_avg", "memory_avg", "pod_count",
    "storage_gb", "network_gb", "deployments", "incidents"
]

X = df[feature_cols]
y = df["monthly_cost"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── EXPERIMENT 1: Linear Regression ──
print("\n🧪 Experiment 1: Linear Regression")
with mlflow.start_run(run_name="linear-regression"):
    model1 = LinearRegression()
    model1.fit(X_train_scaled, y_train)
    preds1 = model1.predict(X_test_scaled)

    mae1 = mean_absolute_error(y_test, preds1)
    r2_1 = r2_score(y_test, preds1)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_metric("mae", mae1)
    mlflow.log_metric("r2_score", r2_1)

    mlflow.sklearn.log_model(
        model1, "model",
        registered_model_name="cost-forecaster"
    )

    print(f"   MAE:      ${mae1:.2f}")
    print(f"   R2 Score: {r2_1:.3f}")

# ── EXPERIMENT 2: Gradient Boosting ──
print("\n🧪 Experiment 2: Gradient Boosting Regressor")
with mlflow.start_run(run_name="gradient-boosting"):
    model2 = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model2.fit(X_train_scaled, y_train)
    preds2 = model2.predict(X_test_scaled)

    mae2 = mean_absolute_error(y_test, preds2)
    r2_2 = r2_score(y_test, preds2)

    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("mae", mae2)
    mlflow.log_metric("r2_score", r2_2)

    mlflow.sklearn.log_model(
        model2, "model",
        registered_model_name="cost-forecaster"
    )

    print(f"   MAE:      ${mae2:.2f}")
    print(f"   R2 Score: {r2_2:.3f}")

# ── EXPERIMENT 3: Tuned Gradient Boosting ──
print("\n🧪 Experiment 3: Tuned Gradient Boosting")
with mlflow.start_run(run_name="tuned-gradient-boosting"):
    model3 = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model3.fit(X_train_scaled, y_train)
    preds3 = model3.predict(X_test_scaled)

    mae3 = mean_absolute_error(y_test, preds3)
    r2_3 = r2_score(y_test, preds3)

    mlflow.log_param("model_type", "GradientBoosting-tuned")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_metric("mae", mae3)
    mlflow.log_metric("r2_score", r2_3)

    mlflow.sklearn.log_model(
        model3, "model",
        registered_model_name="cost-forecaster"
    )

    print(f"   MAE:      ${mae3:.2f}")
    print(f"   R2 Score: {r2_3:.3f}")

# Save scaler
joblib.dump(scaler, "mlops/cost_scaler.pkl")
print("\n✅ Scaler saved to mlops/cost_scaler.pkl")

print("\n" + "="*50)
print("📊 COST FORECASTING EXPERIMENT SUMMARY")
print("="*50)
print(f"Experiment 1 (Linear):          MAE=${mae1:.2f}  R2={r2_1:.3f}")
print(f"Experiment 2 (GBR):             MAE=${mae2:.2f}  R2={r2_2:.3f}")
print(f"Experiment 3 (GBR-tuned):       MAE=${mae3:.2f}  R2={r2_3:.3f}")

best_mae = min(mae1, mae2, mae3)
best_exp = ["linear", "gbr", "gbr-tuned"][[mae1,mae2,mae3].index(best_mae)]
print(f"\n🏆 Best model: {best_exp} with MAE=${best_mae:.2f}")
print(f"✅ Models registered as 'cost-forecaster' in MLflow")
print("="*50)
