import os
import subprocess
import glob
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(base_dir, "run_all_log.txt")
    
    # Find all Model_ subdirectories
    subdirs = sorted([d for d in os.listdir(base_dir) if d.startswith("Model_") and os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"Found {len(subdirs)} subdirectories: {', '.join(subdirs)}")
    
    env = os.environ.copy()
    
    with open(log_file, "w") as f_log:
        f_log.write(f"Execution Log - Phylogenetic Analysis Models\n")
        f_log.write("="*50 + "\n")
        
        for subdir in subdirs:
            work_dir = os.path.join(base_dir, subdir)
            print(f"\n{'='*50}")
            print(f"Processing {subdir}...")
            print(f"{'='*50}")
            f_log.write(f"\n--- {subdir} ---\n")
            
            # Find python scripts (fit_BF_*.py)
            py_scripts = sorted(glob.glob(os.path.join(work_dir, "fit_BF_*.py")))
            for py_script in py_scripts:
                script_name = os.path.basename(py_script)
                print(f"Running {script_name}...")
                try:
                    result = subprocess.run([sys.executable, script_name], env=env, capture_output=True, text=True, check=True, cwd=work_dir)
                    print(f"Successfully completed {script_name}")
                    f_log.write(f"[SUCCESS] {script_name}\n")
                    f_log.flush()
                except subprocess.CalledProcessError as e:
                    print(f"Error running {script_name}:")
                    print(e.stdout)
                    print(e.stderr)
                    f_log.write(f"[FAILURE] {script_name}\n")
                    f_log.write(f"Stdout:\n{e.stdout}\n")
                    f_log.write(f"Stderr:\n{e.stderr}\n")
                    f_log.write("-" * 30 + "\n")
                    f_log.flush()
            
            # Find R compare scripts (compare_*.R)
            r_scripts = sorted(glob.glob(os.path.join(work_dir, "compare_*.R")))
            for r_script in r_scripts:
                script_name = os.path.basename(r_script)
                print(f"Running {script_name}...")
                try:
                    result = subprocess.run(["Rscript", script_name], env=env, capture_output=True, text=True, check=True, cwd=work_dir)
                    print(f"Successfully completed {script_name}")
                    f_log.write(f"[SUCCESS] {script_name}\n")
                    f_log.flush()
                except subprocess.CalledProcessError as e:
                    print(f"Error running {script_name}:")
                    print(e.stdout)
                    print(e.stderr)
                    f_log.write(f"[FAILURE] {script_name}\n")
                    f_log.write(f"Stdout:\n{e.stdout}\n")
                    f_log.write(f"Stderr:\n{e.stderr}\n")
                    f_log.write("-" * 30 + "\n")
                    f_log.flush()

if __name__ == "__main__":
    main()
