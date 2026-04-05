with open("agents/orchestrator.py", "r") as f:
    content = f.read()

# Fix 1 — deterministic code review decision
old = '''    decision = "REQUEST_CHANGES" if "REQUEST_CHANGES" in output.upper() else "APPROVE"
    print(f"   Quality Score: {score}/100")
    print(f"   Decision: {decision}")'''

new = '''    # Deterministic rule
    decision = "REQUEST_CHANGES" if score < 60 else "APPROVE"
    print(f"   Quality Score: {score}/100")
    print(f"   Decision: {decision}")'''

content = content.replace(old, new)

# Fix 2 — deterministic deploy decision
old = '''    response = llm.invoke(prompt)
    decision = "NO-GO" if "NO-GO" in response.content.upper() else "GO"
    print(f"   Decision: {decision}")

    return {**state, "deploy_decision": decision}'''

new = '''    # Deterministic rule
    if state["test_coverage"] < 70:
        decision = "NO-GO"
    elif state["cpu_load"] > 80:
        decision = "NO-GO"
    elif state["memory_load"] > 85:
        decision = "NO-GO"
    else:
        decision = "GO"
    print(f"   Decision: {decision}")

    return {**state, "deploy_decision": decision}'''

content = content.replace(old, new)

with open("agents/orchestrator.py", "w") as f:
    f.write(content)

print("Both rules fixed!")
