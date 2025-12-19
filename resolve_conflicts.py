import subprocess
import os

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

status = run_command("git status --porcelain")
lines = status.splitlines()

for line in lines:
    if not line: continue
    code = line[:2]
    filename = line[3:].strip('"')
    
    if code == "UU" or code == "AA":
        # Both modified or both added
        run_command(f'git checkout --ours "{filename}"')
        run_command(f'git add "{filename}"')
    elif code == "DU":
        # Deleted by us, but exists in theirs
        # We want it deleted
        run_command(f'git rm "{filename}"')
    elif code == "UD":
        # Deleted by them, but exists in us
        # We want to keep it
        run_command(f'git add "{filename}"')

run_command("git add .")
print("Done resolving conflicts.")
