from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
import requests
import datetime

llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

class IncidentState(TypedDict):
    # Metrics
    cpu_usage: float
    memory_usage: float
    error_rate: float
    pod_restarts: int
    # Analysis
    anomaly_detected: bool
    severity: str
    root_cause: str
    # Response
    action_taken: str
    incident_report: str

# Node 1 — collect live metrics from Prometheus
def collect_metrics(state: IncidentState) -> IncidentState:
    print("📊 Collecting metrics from Prometheus...")

    cpu_usage = 0.0
    memory_usage = 0.0
    error_rate = 0.0
    pod_restarts = 0

    try:
        # Query Prometheus via port-forward
        base = "http://localhost:9090/api/v1/query"

        # CPU usage
        r = requests.get(base, params={
            "query": "100 - (avg(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)"
        }, timeout=3)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            if result:
                cpu_usage = float(result[0]["value"][1])

        # Memory usage
        r = requests.get(base, params={
            "query": "100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)"
        }, timeout=3)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            if result:
                memory_usage = float(result[0]["value"][1])

        # Pod restarts
        r = requests.get(base, params={
            "query": "sum(kube_pod_container_status_restarts_total)"
        }, timeout=3)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            if result:
                pod_restarts = int(float(result[0]["value"][1]))

        print(f"   CPU: {cpu_usage:.1f}%")
        print(f"   Memory: {memory_usage:.1f}%")
        print(f"   Pod Restarts: {pod_restarts}")

    except Exception as e:
        print(f"   Prometheus not reachable, using simulated metrics")
        # Simulate metrics for testing
        cpu_usage = 87.5
        memory_usage = 91.2
        error_rate = 15.3
        pod_restarts = 5

    return {
        **state,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "error_rate": error_rate,
        "pod_restarts": pod_restarts
    }

# Node 2 — detect anomalies
def detect_anomaly(state: IncidentState) -> IncidentState:
    print("🔍 Analyzing metrics for anomalies...")

    anomaly = False
    severity = "NONE"

    if state["cpu_usage"] > 85 or state["memory_usage"] > 90:
        anomaly = True
        severity = "CRITICAL"
    elif state["cpu_usage"] > 70 or state["memory_usage"] > 80:
        anomaly = True
        severity = "HIGH"
    elif state["pod_restarts"] > 3:
        anomaly = True
        severity = "MEDIUM"

    print(f"   Anomaly detected: {anomaly}")
    print(f"   Severity: {severity}")

    return {
        **state,
        "anomaly_detected": anomaly,
        "severity": severity
    }

# Node 3 — LLM analyzes root cause
def analyze_root_cause(state: IncidentState) -> IncidentState:
    if not state["anomaly_detected"]:
        return {**state, "root_cause": "No anomaly detected. System healthy."}

    print("🤖 Agent analyzing root cause...")

    prompt = f"""
    You are an incident response AI agent.

    Current system metrics:
    - CPU usage: {state['cpu_usage']:.1f}%
    - Memory usage: {state['memory_usage']:.1f}%
    - Error rate: {state['error_rate']:.1f}%
    - Pod restarts: {state['pod_restarts']}
    - Severity: {state['severity']}

    Analyze the most likely root cause and recommend
    immediate actions to resolve the incident.

    Format:
    ROOT_CAUSE: [most likely cause]
    IMMEDIATE_ACTION: [what to do right now]
    PREVENTION: [how to prevent this in future]
    """

    response = llm.invoke(prompt)
    return {**state, "root_cause": response.content}

# Node 4 — take automated action
def take_action(state: IncidentState) -> IncidentState:
    if not state["anomaly_detected"]:
        return {**state, "action_taken": "No action needed"}

    print("⚡ Taking automated action...")

    action = ""
    if state["severity"] == "CRITICAL":
        action = "ROLLBACK initiated — reverting to last stable deployment via ArgoCD"
        print("   🔄 Triggering ArgoCD rollback...")
        # In production: requests.post("http://argocd-server/api/v1/applications/nginx-app/sync")
    elif state["severity"] == "HIGH":
        action = "SCALING UP — increasing pod replicas to handle load"
        print("   📈 Scaling up deployment...")
        # In production: subprocess.run(["kubectl", "scale", "deployment/nginx", "--replicas=3"])
    elif state["severity"] == "MEDIUM":
        action = "RESTARTING unhealthy pods"
        print("   🔄 Restarting pods...")

    return {**state, "action_taken": action}

# Node 5 — generate incident report
def generate_report(state: IncidentState) -> IncidentState:
    print("📋 Generating incident report...")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not state["anomaly_detected"]:
        report = f"[{timestamp}] System check passed. All metrics normal."
    else:
        report = f"""
INCIDENT REPORT — {timestamp}
{'='*50}
Severity:     {state['severity']}
CPU Usage:    {state['cpu_usage']:.1f}%
Memory Usage: {state['memory_usage']:.1f}%
Pod Restarts: {state['pod_restarts']}

ROOT CAUSE ANALYSIS:
{state['root_cause']}

ACTION TAKEN:
{state['action_taken']}
{'='*50}
        """

    print("\n" + report)
    return {**state, "incident_report": report}

# Routing function
def route_after_detection(state: IncidentState) -> str:
    if state["anomaly_detected"]:
        return "analyze"
    return "report"

# Build graph
graph = StateGraph(IncidentState)
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
    {
        "analyze": "analyze",
        "report":  "report"
    }
)

graph.add_edge("analyze", "action")
graph.add_edge("action",  "report")
graph.add_edge("report",  END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 Running Incident Response Agent...\n")
    result = agent.invoke({
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "error_rate": 0.0,
        "pod_restarts": 0,
        "anomaly_detected": False,
        "severity": "NONE",
        "root_cause": "",
        "action_taken": "",
        "incident_report": ""
    })
