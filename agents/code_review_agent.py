from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
import subprocess

llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

class CodeReviewState(TypedDict):
    diff: str
    file_count: int
    lines_added: int
    lines_removed: int
    quality_score: int
    issues: str
    decision: str
    reasoning: str

# Node 1 — get git diff
def get_git_diff(state: CodeReviewState) -> CodeReviewState:
    print("📂 Getting git diff...")

    result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--stat"],
        capture_output=True,
        text=True,
        cwd="/home/linux_admin/Intelliops"
    )

    diff_result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD"],
        capture_output=True,
        text=True,
        cwd="/home/linux_admin/Intelliops"
    )

    diff = diff_result.stdout[:3000]
    stat = result.stdout

    # Count lines added/removed
    lines_added = diff.count("\n+")
    lines_removed = diff.count("\n-")
    file_count = stat.count("|")

    print(f"   Files changed: {file_count}")
    print(f"   Lines added: {lines_added}")
    print(f"   Lines removed: {lines_removed}")

    return {
        **state,
        "diff": diff if diff else "No diff available",
        "file_count": file_count,
        "lines_added": lines_added,
        "lines_removed": lines_removed
    }

# Node 2 — LLM reviews the code
def review_code(state: CodeReviewState) -> CodeReviewState:
    print("🤖 Agent reviewing code quality...")

    prompt = f"""
    You are an expert code review AI agent.

    PR Statistics:
    - Files changed: {state['file_count']}
    - Lines added: {state['lines_added']}
    - Lines removed: {state['lines_removed']}

    Code diff (truncated):
    {state['diff'][:2000]}

    Review the code and provide:
    1. Quality score (0-100)
    2. Key issues found (if any)
    3. APPROVE or REQUEST_CHANGES decision

    Rules:
    - REQUEST_CHANGES if score below 60
    - REQUEST_CHANGES if security anti-patterns found
    - APPROVE if score 60 or above

    Format your response exactly as:
    SCORE: [0-100]
    ISSUES: [list main issues or "None"]
    DECISION: [APPROVE/REQUEST_CHANGES]
    REASON: [brief explanation]
    """

    response = llm.invoke(prompt)
    output = response.content

    # Parse score
    quality_score = 70
    for line in output.split("\n"):
        if line.startswith("SCORE:"):
            try:
                quality_score = int(line.split(":")[1].strip())
            except:
                pass

    # Parse decision
    decision = "REQUEST_CHANGES" if "REQUEST_CHANGES" in output.upper() else "APPROVE"

    return {
        **state,
        "quality_score": quality_score,
        "issues": output,
        "decision": decision,
        "reasoning": output
    }

# Node 3 — output result
def output_result(state: CodeReviewState) -> CodeReviewState:
    emoji = "✅" if state["decision"] == "APPROVE" else "🔄"
    print("\n" + "="*60)
    print(f"{emoji} CODE REVIEW DECISION: {state['decision']}")
    print(f"📊 Quality Score: {state['quality_score']}/100")
    print(f"📝 Review:\n{state['reasoning']}")
    print("="*60)
    return state

# Build graph
graph = StateGraph(CodeReviewState)
graph.add_node("get_diff", get_git_diff)
graph.add_node("review",   review_code)
graph.add_node("output",   output_result)

graph.set_entry_point("get_diff")
graph.add_edge("get_diff", "review")
graph.add_edge("review",   "output")
graph.add_edge("output",   END)

agent = graph.compile()

if __name__ == "__main__":
    print("🔄 Running Code Review Agent...\n")
    result = agent.invoke({
        "diff": "",
        "file_count": 0,
        "lines_added": 0,
        "lines_removed": 0,
        "quality_score": 0,
        "issues": "",
        "decision": "",
        "reasoning": ""
    })
