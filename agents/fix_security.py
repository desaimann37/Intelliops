# Read the file
with open("agents/orchestrator.py", "r") as f:
    content = f.read()

# Replace the security decision logic
old = '''    response = llm.invoke(prompt)
    decision = "BLOCK" if "BLOCK" in response.content.upper() else "ALLOW"'''

new = '''    # Deterministic rule — no LLM ambiguity
    if critical_count > 0:
        decision = "BLOCK"
    elif high_count > 5:
        decision = "BLOCK"
    else:
        decision = "ALLOW"'''

content = content.replace(old, new)

with open("agents/orchestrator.py", "w") as f:
    f.write(content)

print("Fixed!")
