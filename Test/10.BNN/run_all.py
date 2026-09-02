import subprocess
import sys
import os

def run_script(script_name):
    print(f"Running {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"Successfully finished {script_name}.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}. Exit code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    scripts_to_run = [
        "code/BNN covariance.py",
        "code/BNN regressions.py"
    ]
    
    for script in scripts_to_run:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Warning: {script} not found in the current directory.")
            
    print("All tests completed.")
