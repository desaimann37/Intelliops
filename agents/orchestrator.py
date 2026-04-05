from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
import subprocess
import requests
import json
import datetime

llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

class OrchestratorState(TypedDict):
    # Input
    image: str
    test_coverage: float
    cpu_load: float
    memory_load: float
    pr_size: int
    # Code review results
    code_quality_score: int
    code_review_decision: str
    # Security results
    critical_count: int
    high_count: int
    security_decision: str
    # Deploy results
    deploy_decision: str
    # Incident monitoring
    anomaly_detected: bool
    severity: str
    incident_action: str
    # Final
    final_decision: str
    summary: str

# ── NODE 1: CODE REVIEW ──
def run_code_review(state: OrchestratorState) -> OrchestratorState:
    print("\n📝 STEP 1: Code Review Agent running...")

    # Dynamically detect repo path
    repo_path = os.environ.get("WORKSPACE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--stat"],
        capture_output=True, text=True,
        cwd=repo_path
    )

    diff_result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD"],
        capture_output=True, text=True,
        cwd=repo_path
    )

    diff = diff_result.stdout[:2000]
    lines_added = diff.count("\n+")
    lines_removed = diff.count("\n-")

    prompt = f"""
    You are a code review AI agent.
    Lines added: {lines_added}
    Lines removed: {lines_removed}
    Diff: {diff[:1000]}

    Score 0-100 and decide APPROVE or REQUEST_CHANGES.
    Rules: REQUEST_CHANGES if score below 60.
    Format: SCORE: [0-100] | DECISION: [APPROVE/REQUEST_CHANGES] | REASON: [reason]
    """

    response = llm.invoke(prompt)
    output = response.content

    score = 70
    for line in output.split("\n"):
        if "SCORE:" in line:
            try:
                score = int(''.join(filter(str.isdigit, line.split(":")[1][:4])))
            except:
                pass

    # Deterministic rule
    decision = "REQUEST_CHANGES" if score < 60 else "APPROVE"
    print(f"   Quality Score: {score}/100")
    print(f"   Decision: {decision}")

    return {
        **state,
        "code_quality_score": score,
        "code_review_decision": decision
    }

# ── NODE 2: SECURITY SCAN ──
def run_security_scan(state: OrchestratorState) -> OrchestratorState:
    print("\n🔒 STEP 2: Security Scan Agent running...")

    result = subprocess.run(
        ["trivy", "image", "--format", "json",
         "--severity", "CRITICAL,HIGH", "--quiet", state["image"]],
        capture_output=True, text=True
    )

    critical_count = 0
    high_count = 0

    try:
        data = json.loads(result.stdout)
        for r in data.get("Results", []):
            for vuln in r.get("Vulnerabilities", []):
                if vuln.get("Severity") == "CRITICAL":
                    critical_count += 1
                elif vuln.get("Severity") == "HIGH":
                    high_count += 1
    except:
        pass

    prompt = f"""
    Security scan results:
    Critical: {critical_count}, High: {high_count}
    Rules: BLOCK only if CRITICAL count is greater than 0 OR high count is greater than 5. ALLOW if high count is 5 or fewer and no CRITICAL.
    Format: DECISION: [BLOCK/ALLOW] | REASON: [reason]
    """

    # Deterministic rule — no LLM ambiguity
    if critical_count > 0:
        decision = "BLOCK"
    elif high_count > 5:
        decision = "BLOCK"
    else:
        decision = "ALLOW"

    print(f"   Critical: {critical_count} | High: {high_count}")
    print(f"   Decision: {decision}")

    return {
        **state,
        "critical_count": critical_count,
        "high_count": high_count,
        "security_decision": decision
    }

# ── NODE 3: DEPLOY DECISION ──
def run_deploy_decision(state: OrchestratorState) -> OrchestratorState:
    print("\n🚀 STEP 3: Deploy Decision Agent running...")

    prompt = f"""
    Deployment metrics:
    Test coverage: {state['test_coverage']}%
    CPU load: {state['cpu_load']}%
    Memory load: {state['memory_load']}%
    PR size: {state['pr_size']} lines
    Rules: NO-GO if coverage<70%, CPU>80%, memory>85%
    Format: DECISION: [GO/NO-GO] | REASON: [reason]
    """

    # Deterministic rule
    if state["test_coverage"] < 70:
        decision = "NO-GO"
    elif state["cpu_load"] > 80:
        decision = "NO-GO"
    elif state["memory_load"] > 85:
        decision = "NO-GO"
    else:
        decision = "GO"
    print(f"   Decision: {decision}")

    return {**state, "deploy_decision": decision}

# ── NODE 4: INCIDENT MONITOR ──
def run_incident_monitor(state: OrchestratorState) -> OrchestratorState:
    print("\n🔍 STEP 4: Incident Response Agent monitoring...")

    cpu = 0.0
    memory = 0.0
    restarts = 0

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
            "query": "sum(kube_pod_container_status_restarts_total)"
        }, timeout=3)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            if result:
                restarts = int(float(result[0]["value"][1]))
    except:
        cpu = state["cpu_load"]
        memory = state["memory_load"]

    anomaly = cpu > 85 or memory > 90 or restarts > 10
    severity = "CRITICAL" if (cpu > 85 or memory > 90) else "MEDIUM" if restarts > 10 else "NONE"
    action = "ROLLBACK initiated" if severity == "CRITICAL" else "Monitoring closely" if anomaly else "System healthy"

    print(f"   CPU: {cpu:.1f}% | Restarts: {restarts}")
    print(f"   Anomaly: {anomaly} | Action: {action}")

    return {
        **state,
        "anomaly_detected": anomaly,
        "severity": severity,
        "incident_action": action
    }

# ── ROUTING FUNCTIONS ──
def route_after_code_review(state: OrchestratorState) -> str:
    if state["code_review_decision"] == "REQUEST_CHANGES":
        return "rejected"
    return "security_scan"

def route_after_security(state: OrchestratorState) -> str:
    if state["security_decision"] == "BLOCK":
        return "rejected"
    return "deploy_decision"

def route_after_deploy(state: OrchestratorState) -> str:
    if state["deploy_decision"] == "NO-GO":
        return "rejected"
    return "incident_monitor"

# ── NODE 5: FINAL OUTPUT ──
def output_result(state: OrchestratorState) -> OrchestratorState:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if state["final_decision"] == "APPROVED":
        emoji = "✅"
    else:
        emoji = "🚫"

    print("\n" + "="*60)
    print(f"{emoji} INTELLIOPS PIPELINE DECISION: {state['final_decision']}")
    print(f"🕐 Timestamp: {timestamp}")
    print(f"📊 Code Quality:  {state['code_quality_score']}/100 → {state['code_review_decision']}")
    print(f"🔒 Security:      Critical={state['critical_count']} High={state['high_count']} → {state['security_decision']}")
    print(f"🚀 Deploy:        {state['deploy_decision']}")
    print(f"🔍 Incident:      Severity={state['severity']} → {state['incident_action']}")
    print(f"📝 Summary:       {state['summary']}")
    print("="*60)
    return state

def handle_approved(state: OrchestratorState) -> OrchestratorState:
    return {
        **state,
        "final_decision": "APPROVED",
        "summary": "All checks passed. Deploying via ArgoCD."
    }

def handle_rejected(state: OrchestratorState) -> OrchestratorState:
    reasons = []
    if state["code_review_decision"] == "REQUEST_CHANGES":
        reasons.append(f"Code quality {state['code_quality_score']}/100")
    if state["security_decision"] == "BLOCK":
        reasons.append(f"Security: {state['critical_count']} critical, {state['high_count']} high CVEs")
    if state["deploy_decision"] == "NO-GO":
        reasons.append("Deploy metrics failed")

    return {
        **state,
        "final_decision": "REJECTED",
        "summary": "Rejected: " + " | ".join(reasons)
    }

# ── BUILD GRAPH ──
graph = StateGraph(OrchestratorState)

graph.add_node("code_review",     run_code_review)
graph.add_node("security_scan",   run_security_scan)
graph.add_node("deploy_decision", run_deploy_decision)
graph.add_node("incident_monitor",run_incident_monitor)
graph.add_node("approved",        handle_approved)
graph.add_node("rejected",        handle_rejected)
graph.add_node("output",          output_result)

graph.set_entry_point("code_review")

graph.add_conditional_edges(
    "code_review",
    route_after_code_review,
    {
        "rejected":     "rejected",
        "security_scan":"security_scan"
    }
)

graph.add_conditional_edges(
    "security_scan",
    route_after_security,
    {
        "rejected":       "rejected",
        "deploy_decision":"deploy_decision"
    }
)

graph.add_conditional_edges(
    "deploy_decision",
    route_after_deploy,
    {
        "rejected":        "rejected",
        "incident_monitor":"incident_monitor"
    }
)

graph.add_edge("incident_monitor", "approved")
graph.add_edge("approved",         "output")
graph.add_edge("rejected",         "output")
graph.add_edge("output",           END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 IntelliOps Full Pipeline Starting...\n")
    print("━"*60)

    agent.invoke({
        "image": "alpine:latest",
        "test_coverage": 85.0,
        "cpu_load": 40.0,
        "memory_load": 50.0,
        "pr_size": 100,
        "code_quality_score": 0,
        "code_review_decision": "",
        "critical_count": 0,
        "high_count": 0,
        "security_decision": "",
        "deploy_decision": "",
        "anomaly_detected": False,
        "severity": "NONE",
        "incident_action": "",
        "final_decision": "",
        "summary": ""
    })
