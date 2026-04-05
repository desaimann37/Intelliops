from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
import subprocess
import json

# Connect to local Ollama
llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Define agent state
class SecurityState(TypedDict):
    image: str
    scan_output: str
    critical_count: int
    high_count: int
    decision: str
    reasoning: str

# Node 1 — run Trivy scan
def run_trivy_scan(state: SecurityState) -> SecurityState:
    print(f"🔍 Scanning image: {state['image']}")
    
    result = subprocess.run(
        [
            "trivy", "image",
            "--format", "json",
            "--severity", "CRITICAL,HIGH",
            "--quiet",
            state["image"]
        ],
        capture_output=True,
        text=True
    )
    
    # Parse results
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
    
    print(f"Found: {critical_count} CRITICAL, {high_count} HIGH vulnerabilities")
    
    return {
        **state,
        "scan_output": result.stdout[:2000],
        "critical_count": critical_count,
        "high_count": high_count
    }

# Node 2 — LLM analyzes results
def analyze_vulnerabilities(state: SecurityState) -> SecurityState:
    print("🤖 Agent analyzing vulnerabilities...")
    
    prompt = f"""
    You are a security AI agent reviewing a Docker image scan.
    
    Image scanned: {state['image']}
    Critical vulnerabilities: {state['critical_count']}
    High vulnerabilities: {state['high_count']}
    
    Rules:
    - BLOCK if any CRITICAL vulnerabilities found
    - WARN if more than 5 HIGH vulnerabilities found
    - ALLOW if no CRITICAL and fewer than 5 HIGH
    
    Give a clear BLOCK or ALLOW decision with reason.
    Format: DECISION: [BLOCK/ALLOW] | REASON: [your reason]
    """
    
    response = llm.invoke(prompt)
    output = response.content
    
    if "BLOCK" in output.upper():
        decision = "BLOCK"
    else:
        decision = "ALLOW"
    
    return {
        **state,
        "decision": decision,
        "reasoning": output
    }

# Node 3 — output result
def output_result(state: SecurityState) -> SecurityState:
    emoji = "🚫" if state["decision"] == "BLOCK" else "✅"
    print("\n" + "="*50)
    print(f"{emoji} SECURITY DECISION: {state['decision']}")
    print(f"📊 Critical: {state['critical_count']} | High: {state['high_count']}")
    print(f"📝 REASONING: {state['reasoning']}")
    print("="*50)
    return state

# Build the agent graph
graph = StateGraph(SecurityState)
graph.add_node("scan",     run_trivy_scan)
graph.add_node("analyze",  analyze_vulnerabilities)
graph.add_node("output",   output_result)

graph.set_entry_point("scan")
graph.add_edge("scan",    "analyze")
graph.add_edge("analyze", "output")
graph.add_edge("output",  END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 Running Security Scan Agent...\n")
    
    print("TEST: Scanning nginx image")
    result = agent.invoke({
        "image": "nginx:latest",
        "scan_output": "",
        "critical_count": 0,
        "high_count": 0,
        "decision": "",
        "reasoning": ""
    })
