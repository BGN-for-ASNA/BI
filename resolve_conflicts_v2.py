import subprocess
import os

def run_command(cmd, input=None):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, input=input)
    return result

# Find all files with conflict markers
result = run_command('grep -rl "<<<<<<< HEAD" .')
files = result.stdout.splitlines()

for f in files:
    if "resolve_conflicts.py" in f or ".git/" in f:
        continue
    print(f"Resolving: {f}")
    # Try to checkout ours
    res = run_command(f'git checkout --ours "{f}"')
    if res.returncode != 0:
        print(f"Failed to checkout ours for {f}, trying to keep as is but staging.")
    run_command(f'git add "{f}"')

# Also handle files that were deleted in ours but exist in theirs (DU)
# We need to find them from git status if possible, or just assume we want them deleted if they were DU.
status = run_command("git status --porcelain")
for line in status.stdout.splitlines():
    if line.startswith("DU "):
        filename = line[3:].strip('"')
        print(f"Removing DU file: {filename}")
        run_command(f'git rm "{filename}"')

print("Done.")
