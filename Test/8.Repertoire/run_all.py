import os
import subprocess
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, "scripts")
    log_file = os.path.join(base_dir, "results", "run_all_log.txt")
    
    # Scripts to run for the BF vs Stan comparison
    scripts = [
        "benchmark.py",
        "compare_moments_p.py",
        "compare_summary.py"
    ]
    
    env = os.environ.copy()
    
    # Make sure results directory exists
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    
    with open(log_file, "w") as f_log:
        f_log.write("Execution Log - Repertoire Comparison\n")
        f_log.write("="*50 + "\n")
        
        for script in scripts:
            print(f"\n{'='*50}")
            print(f"Running {script}...")
            print(f"{'='*50}")
            
            try:
                result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, check=True, cwd=scripts_dir)
                print(f"Successfully completed {script}")
                f_log.write(f"[SUCCESS] {script}\n")
                if result.stdout:
                    f_log.write(result.stdout + "\n")
                f_log.flush()
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}:")
                print(e.stdout)
                print(e.stderr)
                f_log.write(f"[FAILURE] {script}\n")
                f_log.write(f"Stdout:\n{e.stdout}\n")
                f_log.write(f"Stderr:\n{e.stderr}\n")
                f_log.flush()
                
    print("\nAll scripts finished. Check results/run_all_log.txt for details.")

if __name__ == "__main__":
    main()
