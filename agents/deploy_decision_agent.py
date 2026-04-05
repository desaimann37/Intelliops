from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
import requests

# Connect to local Ollama
llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Define agent state
class DeployState(TypedDict):
    test_coverage: float
    cpu_load: float
    memory_load: float
    pr_size: int
    decision: str
    reasoning: str

# Node 1 — collect metrics
def collect_metrics(state: DeployState) -> DeployState:
    print("📊 Collecting metrics...")
    return state

# Node 2 — analyze with LLM
def analyze_deployment(state: DeployState) -> DeployState:
    print("🤖 Agent analyzing deployment risk...")
    
    prompt = f"""
    You are a DevOps AI agent making a deployment decision.
    
    Current metrics:
    - Test coverage: {state['test_coverage']}%
    - CPU load: {state['cpu_load']}%
    - Memory load: {state['memory_load']}%
    - PR size (lines changed): {state['pr_size']}
    
    Rules:
    - REJECT if test coverage below 70%
    - REJECT if CPU load above 80%
    - REJECT if memory load above 85%
    - CAUTION if PR size above 500 lines
    
    Give a GO or NO-GO decision with a brief reason.
    Format: DECISION: [GO/NO-GO] | REASON: [your reason]
    """
    
    response = llm.invoke(prompt)
    output = response.content
    
    if "NO-GO" in output.upper():
        decision = "NO-GO"
    else:
        decision = "GO"
    
    return {
        **state,
        "decision": decision,
        "reasoning": output
    }

# Node 3 — output result
def output_result(state: DeployState) -> DeployState:
    print("\n" + "="*50)
    print(f"🚀 DEPLOY DECISION: {state['decision']}")
    print(f"📝 REASONING: {state['reasoning']}")
    print("="*50)
    return state

# Build the agent graph
graph = StateGraph(DeployState)
graph.add_node("collect_metrics", collect_metrics)
graph.add_node("analyze", analyze_deployment)
graph.add_node("output", output_result)

graph.set_entry_point("collect_metrics")
graph.add_edge("collect_metrics", "analyze")
graph.add_edge("analyze", "output")
graph.add_edge("output", END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 Running Deploy Decision Agent...\n")
    
    print("TEST 1: Safe deployment")
    result = agent.invoke({
        "test_coverage": 85.0,
        "cpu_load": 45.0,
        "memory_load": 50.0,
        "pr_size": 120,
        "decision": "",
        "reasoning": ""
    })
    
    print("\nTEST 2: Risky deployment")
    result = agent.invoke({
        "test_coverage": 55.0,
        "cpu_load": 85.0,
        "memory_load": 90.0,
        "pr_size": 800,
        "decision": "",
        "reasoning": ""
    })
