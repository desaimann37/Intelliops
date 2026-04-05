import mlflow
import mlflow.sklearn
import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict

mlflow.set_tracking_uri("http://localhost:8083")
llm = ChatOllama(model="llama3.1", base_url="http://172.26.231.242:11434")

# Load ML model from registry
model = mlflow.sklearn.load_model("models:/deploy-risk-classifier/1")

class MLDeployState(TypedDict):
    test_coverage: float
    cpu_load: float
    memory_load: float
    pr_size: int
    critical_cves: int
    high_cves: int
    pod_restarts: int
    hour_of_day: int
    risk_level: str
    confidence: float
    decision: str
    explanation: str

# Node 1 — ML model predicts risk
def predict_risk(state: MLDeployState) -> MLDeployState:
    print("🧠 ML Model predicting deployment risk...")

    features = pd.DataFrame([{
        "test_coverage":  state["test_coverage"],
        "cpu_load":       state["cpu_load"],
        "memory_load":    state["memory_load"],
        "pr_size":        state["pr_size"],
        "critical_cves":  state["critical_cves"],
        "high_cves":      state["high_cves"],
        "pod_restarts":   state["pod_restarts"],
        "hour_of_day":    state["hour_of_day"]
    }])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    risk_map = {0: "HIGH", 1: "LOW", 2: "MEDIUM"}
    risk = risk_map.get(prediction, "MEDIUM")
    confidence = round(float(max(probability)) * 100, 1)

    print(f"   Risk Level:  {risk}")
    print(f"   Confidence:  {confidence}%")

    return {**state, "risk_level": risk, "confidence": confidence}

# Node 2 — LLM explains the decision
def explain_decision(state: MLDeployState) -> MLDeployState:
    print("🤖 LLM generating explanation...")

    decision = "GO" if state["risk_level"] == "LOW" else "NO-GO"

    prompt = f"""
    You are a DevOps AI agent explaining a deployment decision.

    ML Model Assessment:
    - Risk Level: {state['risk_level']}
    - Confidence: {state['confidence']}%

    Current metrics:
    - Test coverage: {state['test_coverage']}%
    - CPU load: {state['cpu_load']}%
    - Memory load: {state['memory_load']}%
    - PR size: {state['pr_size']} lines
    - Critical CVEs: {state['critical_cves']}
    - High CVEs: {state['high_cves']}
    - Pod restarts: {state['pod_restarts']}
    - Hour of day: {state['hour_of_day']}

    Decision: {decision}

    Explain this decision in 2-3 sentences for the engineering team.
    """

    response = llm.invoke(prompt)

    return {
        **state,
        "decision": decision,
        "explanation": response.content
    }

# Node 3 — output
def output_result(state: MLDeployState) -> MLDeployState:
    emoji = "✅" if state["decision"] == "GO" else "🚫"
    print("\n" + "="*55)
    print(f"{emoji} ML-POWERED DEPLOY DECISION: {state['decision']}")
    print(f"🎯 Risk: {state['risk_level']} | Confidence: {state['confidence']}%")
    print(f"📝 {state['explanation']}")
    print("="*55)
    return state

# Build graph
graph = StateGraph(MLDeployState)
graph.add_node("predict",  predict_risk)
graph.add_node("explain",  explain_decision)
graph.add_node("output",   output_result)

graph.set_entry_point("predict")
graph.add_edge("predict", "explain")
graph.add_edge("explain", "output")
graph.add_edge("output",  END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 ML-Powered Deploy Decision Agent\n")

    print("TEST: Current deployment")
    agent.invoke({
        "test_coverage": 85.0,
        "cpu_load": 40.0,
        "memory_load": 50.0,
        "pr_size": 120,
        "critical_cves": 0,
        "high_cves": 1,
        "pod_restarts": 2,
        "hour_of_day": 14,
        "risk_level": "",
        "confidence": 0.0,
        "decision": "",
        "explanation": ""
    })
