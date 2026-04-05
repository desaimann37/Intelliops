import os

with open("agents/orchestrator.py", "r") as f:
    content = f.read()

# Replace hardcoded path with dynamic workspace detection
old = '''    result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--stat"],
        capture_output=True, text=True,
        cwd="/home/linux_admin/Intelliops"
    )

    diff_result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD"],
        capture_output=True, text=True,
        cwd="/home/linux_admin/Intelliops"
    )'''

new = '''    # Dynamically detect repo path
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
    )'''

content = content.replace(old, new)

with open("agents/orchestrator.py", "w") as f:
    f.write(content)

print("Path fixed!")
