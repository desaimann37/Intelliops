from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
import subprocess
import json

llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Combined state for all agents
class OrchestratorState(TypedDict):
    # Input
    image: str
    test_coverage: float
    cpu_load: float
    memory_load: float
    pr_size: int
    # Security agent results
    critical_count: int
    high_count: int
    security_decision: str
    # Deploy agent results
    deploy_decision: str
    # Final
    final_decision: str
    summary: str

# ── SECURITY SCAN NODE ──
def run_security_scan(state: OrchestratorState) -> OrchestratorState:
    print("\n🔒 STEP 1: Security Scan Agent running...")

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

    print(f"   Found: {critical_count} CRITICAL, {high_count} HIGH")

    prompt = f"""
    You are a security AI agent.
    Image: {state['image']}
    Critical vulnerabilities: {critical_count}
    High vulnerabilities: {high_count}
    Rules:
    - BLOCK if any CRITICAL found
    - BLOCK if more than 5 HIGH found
    - ALLOW otherwise
    Format: DECISION: [BLOCK/ALLOW] | REASON: [reason]
    """
    response = llm.invoke(prompt)
    security_decision = "BLOCK" if "BLOCK" in response.content.upper() else "ALLOW"
    print(f"   Security Decision: {security_decision}")

    return {
        **state,
        "critical_count": critical_count,
        "high_count": high_count,
        "security_decision": security_decision
    }

# ── DEPLOY DECISION NODE ──
def run_deploy_decision(state: OrchestratorState) -> OrchestratorState:
    print("\n🚀 STEP 2: Deploy Decision Agent running...")

    prompt = f"""
    You are a DevOps AI agent making a deployment decision.
    Test coverage: {state['test_coverage']}%
    CPU load: {state['cpu_load']}%
    Memory load: {state['memory_load']}%
    PR size: {state['pr_size']} lines
    Rules:
    - NO-GO if test coverage below 70%
    - NO-GO if CPU load above 80%
    - NO-GO if memory load above 85%
    - CAUTION if PR size above 500 lines
    Format: DECISION: [GO/NO-GO] | REASON: [reason]
    """
    response = llm.invoke(prompt)
    deploy_decision = "NO-GO" if "NO-GO" in response.content.upper() else "GO"
    print(f"   Deploy Decision: {deploy_decision}")

    return {
        **state,
        "deploy_decision": deploy_decision
    }

# ── ROUTING FUNCTION ──
def route_after_security(state: OrchestratorState) -> str:
    if state["security_decision"] == "BLOCK":
        return "blocked"
    return "deploy_decision"

# ── FINAL DECISION NODE ──
def make_final_decision(state: OrchestratorState) -> OrchestratorState:
    print("\n🤖 STEP 3: Orchestrator making final decision...")

    if state["security_decision"] == "BLOCK":
        final = "REJECTED"
        summary = f"Pipeline BLOCKED by Security Agent. Critical: {state['critical_count']}, High: {state['high_count']}"
    elif state["deploy_decision"] == "NO-GO":
        final = "REJECTED"
        summary = f"Pipeline REJECTED by Deploy Decision Agent."
    else:
        final = "APPROVED"
        summary = "All checks passed. Deployment APPROVED. Triggering ArgoCD sync."

    return {
        **state,
        "final_decision": final,
        "summary": summary
    }

# ── BLOCKED NODE ──
def handle_blocked(state: OrchestratorState) -> OrchestratorState:
    return make_final_decision(state)

# ── OUTPUT NODE ──
def output_result(state: OrchestratorState) -> OrchestratorState:
    emoji = "✅" if state["final_decision"] == "APPROVED" else "🚫"
    print("\n" + "="*60)
    print(f"{emoji}  FINAL PIPELINE DECISION: {state['final_decision']}")
    print(f"📝 {state['summary']}")
    print("="*60)
    return state

# ── BUILD GRAPH ──
graph = StateGraph(OrchestratorState)

graph.add_node("security_scan",    run_security_scan)
graph.add_node("deploy_decision",  run_deploy_decision)
graph.add_node("blocked",          handle_blocked)
graph.add_node("final_decision",   make_final_decision)
graph.add_node("output",           output_result)

graph.set_entry_point("security_scan")

graph.add_conditional_edges(
    "security_scan",
    route_after_security,
    {
        "blocked":        "blocked",
        "deploy_decision": "deploy_decision"
    }
)

graph.add_edge("blocked",         "output")
graph.add_edge("deploy_decision", "final_decision")
graph.add_edge("final_decision",  "output")
graph.add_edge("output",          END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 IntelliOps Orchestrator Agent Starting...\n")

    print("━"*60)
    print("SCENARIO 1: Good image, good metrics")
    print("━"*60)
    agent.invoke({
        "image": "alpine:latest",
        "test_coverage": 85.0,
        "cpu_load": 40.0,
        "memory_load": 50.0,
        "pr_size": 100,
        "critical_count": 0,
        "high_count": 0,
        "security_decision": "",
        "deploy_decision": "",
        "final_decision": "",
        "summary": ""
    })

    print("\n" + "━"*60)
    print("SCENARIO 2: Bad image, bad metrics")
    print("━"*60)
    agent.invoke({
        "image": "nginx:latest",
        "test_coverage": 55.0,
        "cpu_load": 85.0,
        "memory_load": 90.0,
        "pr_size": 800,
        "critical_count": 0,
        "high_count": 0,
        "security_decision": "",
        "deploy_decision": "",
        "final_decision": "",
        "summary": ""
    })
