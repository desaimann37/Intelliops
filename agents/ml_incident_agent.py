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

# Load ML model and scaler
model  = mlflow.sklearn.load_model("models:/anomaly-detector/1")
scaler = joblib.load("mlops/anomaly_scaler.pkl")

class MLIncidentState(TypedDict):
    cpu_usage:        float
    memory_usage:     float
    error_rate:       float
    latency_ms:       float
    pod_restarts:     int
    request_rate:     float
    network_io_mb:    float
    anomaly_detected: bool
    anomaly_score:    float
    severity:         str
    root_cause:       str
    action_taken:     str
    incident_report:  str

# Node 1 — collect real metrics
def collect_metrics(state: MLIncidentState) -> MLIncidentState:
    print("📊 Collecting live metrics from Prometheus...")

    cpu = memory = error_rate = latency = request_rate = network = 0.0
    pod_restarts = 0

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
            "query": "sum(kube_pod_container_status_restarts_total)"
        }, timeout=3)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            if result:
                pod_restarts = int(float(result[0]["value"][1]))

        print(f"   CPU:          {cpu:.1f}%")
        print(f"   Memory:       {memory:.1f}%")
        print(f"   Pod Restarts: {pod_restarts}")

    except Exception as e:
        print(f"   Using simulated metrics")
        cpu          = 87.5
        memory       = 91.2
        error_rate   = 15.3
        latency      = 850.0
        pod_restarts = 8
        request_rate = 45.0
        network      = 210.0

    return {
        **state,
        "cpu_usage":     cpu,
        "memory_usage":  memory,
        "error_rate":    error_rate,
        "latency_ms":    latency,
        "pod_restarts":  pod_restarts,
        "request_rate":  request_rate,
        "network_io_mb": network
    }

# Node 2 — ML model detects anomaly
def detect_anomaly(state: MLIncidentState) -> MLIncidentState:
    print("🧠 ML Model detecting anomalies...")

    features = pd.DataFrame([{
        "cpu_usage":     state["cpu_usage"],
        "memory_usage":  state["memory_usage"],
        "error_rate":    state["error_rate"],
        "latency_ms":    state["latency_ms"],
        "pod_restarts":  state["pod_restarts"],
        "request_rate":  state["request_rate"],
        "network_io_mb": state["network_io_mb"]
    }])

    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    anomaly_score   = float(model.score_samples(features_scaled)[0])

    anomaly_detected = prediction == -1

    # Severity based on score
    if anomaly_score < -0.2:
        severity = "CRITICAL"
    elif anomaly_score < -0.1:
        severity = "HIGH"
    elif anomaly_detected:
        severity = "MEDIUM"
    else:
        severity = "NONE"

    print(f"   Anomaly detected: {anomaly_detected}")
    print(f"   Anomaly score:    {anomaly_score:.3f}")
    print(f"   Severity:         {severity}")

    return {
        **state,
        "anomaly_detected": anomaly_detected,
        "anomaly_score":    round(anomaly_score, 3),
        "severity":         severity
    }

# Node 3 — LLM root cause analysis
def analyze_root_cause(state: MLIncidentState) -> MLIncidentState:
    if not state["anomaly_detected"]:
        return {**state, "root_cause": "System healthy. No anomaly detected."}

    print("🤖 LLM analyzing root cause...")

    prompt = f"""
    You are an incident response AI agent.

    ML Model detected an anomaly with severity: {state['severity']}
    Anomaly score: {state['anomaly_score']} (more negative = more anomalous)

    Current metrics:
    - CPU: {state['cpu_usage']:.1f}%
    - Memory: {state['memory_usage']:.1f}%
    - Error rate: {state['error_rate']:.1f}%
    - Latency: {state['latency_ms']:.0f}ms
    - Pod restarts: {state['pod_restarts']}
    - Request rate: {state['request_rate']:.0f}/s

    Identify the most likely root cause and
    recommend immediate actions in 3 bullet points.
    """

    response = llm.invoke(prompt)
    return {**state, "root_cause": response.content}

# Node 4 — automated action
def take_action(state: MLIncidentState) -> MLIncidentState:
    if not state["anomaly_detected"]:
        return {**state, "action_taken": "No action needed"}

    print("⚡ Taking automated action...")

    if state["severity"] == "CRITICAL":
        action = "ROLLBACK triggered via ArgoCD — reverting to last stable version"
    elif state["severity"] == "HIGH":
        action = "SCALING UP — adding 2 more pod replicas"
    else:
        action = "ALERTING on-call engineer via PagerDuty"

    print(f"   Action: {action}")
    return {**state, "action_taken": action}

# Node 5 — generate report
def generate_report(state: MLIncidentState) -> MLIncidentState:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not state["anomaly_detected"]:
        report = f"[{timestamp}] ✅ System healthy. ML model score: {state['anomaly_score']:.3f}"
    else:
        report = f"""
INCIDENT REPORT — {timestamp}
{'='*50}
Severity:      {state['severity']}
Anomaly Score: {state['anomaly_score']}
CPU:           {state['cpu_usage']:.1f}%
Memory:        {state['memory_usage']:.1f}%
Pod Restarts:  {state['pod_restarts']}

ROOT CAUSE:
{state['root_cause']}

ACTION TAKEN:
{state['action_taken']}
{'='*50}
        """

    print("\n" + report)
    return {**state, "incident_report": report}

# Routing
def route_after_detection(state: MLIncidentState) -> str:
    return "analyze" if state["anomaly_detected"] else "report"

# Build graph
graph = StateGraph(MLIncidentState)
graph.add_node("collect",  collect_metrics)
graph.add_node("detect",   detect_anomaly)
graph.add_node("analyze",  analyze_root_cause)
graph.add_node("action",   take_action)
graph.add_node("report",   generate_report)

graph.set_entry_point("collect")
graph.add_edge("collect", "detect")
graph.add_conditional_edges(
    "detect",
    route_after_detection,
    {"analyze": "analyze", "report": "report"}
)
graph.add_edge("analyze", "action")
graph.add_edge("action",  "report")
graph.add_edge("report",  END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 ML-Powered Incident Response Agent\n")
    result = agent.invoke({
        "cpu_usage":        0.0,
        "memory_usage":     0.0,
        "error_rate":       0.0,
        "latency_ms":       0.0,
        "pod_restarts":     0,
        "request_rate":     0.0,
        "network_io_mb":    0.0,
        "anomaly_detected": False,
        "anomaly_score":    0.0,
        "severity":         "NONE",
        "root_cause":       "",
        "action_taken":     "",
        "incident_report":  ""
    })
