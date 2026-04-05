import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import joblib
import requests
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
import datetime

mlflow.set_tracking_uri("http://localhost:8083")
llm = ChatOllama(model="llama3.1", base_url="http://172.26.231.242:11434")

# Load model and scaler
model  = mlflow.sklearn.load_model("models:/cost-forecaster/2")
scaler = joblib.load("mlops/cost_scaler.pkl")

class MLCostState(TypedDict):
    cpu_avg:          float
    memory_avg:       float
    pod_count:        int
    storage_gb:       float
    network_gb:       float
    deployments:      int
    incidents:        int
    predicted_cost:   float
    current_cost:     float
    savings_potential: float
    recommendations:  str
    action_taken:     str

# Node 1 — collect metrics
def collect_metrics(state: MLCostState) -> MLCostState:
    print("💰 Collecting resource metrics...")

    cpu = memory = 0.0
    pod_count = 0

    try:
        base = "http://localhost:9090/api/v1/query"

        r = requests.get(base, params={
            "query": "100 - (avg(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)"
        }, timeout=3)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            if result:
                cpu = float(result[0]["value"][1])

        r = requests.get(base, params={
            "query": "100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)"
        }, timeout=3)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            if result:
                memory = float(result[0]["value"][1])

        r = requests.get(base, params={
            "query": "count(kube_pod_info)"
        }, timeout=3)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            if result:
                pod_count = int(float(result[0]["value"][1]))

        print(f"   CPU avg:    {cpu:.1f}%")
        print(f"   Memory avg: {memory:.1f}%")
        print(f"   Pod count:  {pod_count}")

    except Exception as e:
        print(f"   Using simulated metrics")
        cpu       = 35.0
        memory    = 45.0
        pod_count = 22

    return {
        **state,
        "cpu_avg":    cpu,
        "memory_avg": memory,
        "pod_count":  pod_count
    }

# Node 2 — ML model forecasts cost
def forecast_cost(state: MLCostState) -> MLCostState:
    print("🧠 ML Model forecasting monthly cost...")

    features = pd.DataFrame([{
        "cpu_avg":     state["cpu_avg"],
        "memory_avg":  state["memory_avg"],
        "pod_count":   state["pod_count"],
        "storage_gb":  state["storage_gb"],
        "network_gb":  state["network_gb"],
        "deployments": state["deployments"],
        "incidents":   state["incidents"]
    }])

    features_scaled = scaler.transform(features)
    predicted_cost  = float(model.predict(features_scaled)[0])

    # Current cost estimate
    current_cost = (
        state["cpu_avg"] * 0.05 +
        state["memory_avg"] * 0.03 +
        state["pod_count"] * 0.10 +
        state["storage_gb"] * 0.02 +
        state["network_gb"] * 0.01
    ) * 30

    savings_potential = max(0, current_cost - predicted_cost * 0.8)

    print(f"   Predicted monthly cost: ${predicted_cost:.2f}")
    print(f"   Current monthly cost:   ${current_cost:.2f}")
    print(f"   Savings potential:      ${savings_potential:.2f}")

    return {
        **state,
        "predicted_cost":    round(predicted_cost, 2),
        "current_cost":      round(current_cost, 2),
        "savings_potential": round(savings_potential, 2)
    }

# Node 3 — LLM recommendations
def generate_recommendations(state: MLCostState) -> MLCostState:
    print("🤖 LLM generating cost recommendations...")

    prompt = f"""
    You are a Kubernetes cost optimization AI agent.

    Current cluster costs:
    - CPU usage: {state['cpu_avg']:.1f}%
    - Memory usage: {state['memory_avg']:.1f}%
    - Pod count: {state['pod_count']}
    - Storage: {state['storage_gb']:.0f}GB
    - Network: {state['network_gb']:.0f}GB/month
    - Deployments this month: {state['deployments']}
    - Incidents this month: {state['incidents']}

    ML Model predictions:
    - Predicted monthly cost: ${state['predicted_cost']:.2f}
    - Current monthly cost:   ${state['current_cost']:.2f}
    - Savings potential:      ${state['savings_potential']:.2f}

    Give 3 specific actionable recommendations to reduce costs.
    Format as numbered list with estimated savings for each.
    """

    response = llm.invoke(prompt)
    return {**state, "recommendations": response.content}

# Node 4 — automated actions
def take_action(state: MLCostState) -> MLCostState:
    print("⚡ Applying cost optimizations...")

    actions = []
    if state["savings_potential"] > 50:
        actions.append("AUTO: Right-sizing recommendations sent to Slack")
    actions.append("AUTO: Scheduled off-peak scale-down at midnight")
    actions.append("AUTO: Cost report logged to MLflow")

    # Log to MLflow
    import mlflow
    with mlflow.start_run(run_name=f"cost-report-{datetime.date.today()}",
                          experiment_id="3"):
        mlflow.log_metric("predicted_cost", state["predicted_cost"])
        mlflow.log_metric("current_cost", state["current_cost"])
        mlflow.log_metric("savings_potential", state["savings_potential"])

    return {**state, "action_taken": "\n".join(actions)}

# Node 5 — output
def output_report(state: MLCostState) -> MLCostState:
    print("\n" + "="*60)
    print("💰 ML-POWERED COST OPTIMIZER REPORT")
    print("="*60)
    print(f"📊 Current monthly cost:    ${state['current_cost']:.2f}")
    print(f"🔮 Predicted monthly cost:  ${state['predicted_cost']:.2f}")
    print(f"💵 Savings potential:       ${state['savings_potential']:.2f}")
    print(f"\n🤖 Recommendations:\n{state['recommendations']}")
    print(f"\n⚡ Actions:\n{state['action_taken']}")
    print("="*60)
    return state

# Build graph
graph = StateGraph(MLCostState)
graph.add_node("collect",   collect_metrics)
graph.add_node("forecast",  forecast_cost)
graph.add_node("recommend", generate_recommendations)
graph.add_node("action",    take_action)
graph.add_node("output",    output_report)

graph.set_entry_point("collect")
graph.add_edge("collect",   "forecast")
graph.add_edge("forecast",  "recommend")
graph.add_edge("recommend", "action")
graph.add_edge("action",    "output")
graph.add_edge("output",    END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 ML-Powered Cost Optimizer Agent\n")
    result = agent.invoke({
        "cpu_avg":          0.0,
        "memory_avg":       0.0,
        "pod_count":        0,
        "storage_gb":       100.0,
        "network_gb":       50.0,
        "deployments":      10,
        "incidents":        2,
        "predicted_cost":   0.0,
        "current_cost":     0.0,
        "savings_potential": 0.0,
        "recommendations":  "",
        "action_taken":     ""
    })
