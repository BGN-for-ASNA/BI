import os
import subprocess
import sys

# Get the python executable from the BayesInference virtualenv
python_exe = "/home/sebastian_sosa/.virtualenvs/BayesInference/bin/python3"

models = [
    "1.Continuous variable.py",
    "2.Categorical variable.py",
    "3.Continuous interactions.py",
    "4.Binomial.py",
    "5.Binomial with indices.py",
    "6.Poisson.py",
    "7.Negative binomial.py",
    "8.Multinomial.py",
    "9.Beta binomial.py",
    "10.Zero inflated.py",
    "11.Varying intercepts.py",
    "12.Varying effects.py",
    # "13.Gaussian processes.py"  # Excluded as per user request (too long)
]

script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure plots directory exists
plots_dir = os.path.join(script_dir, "plots")
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

print(f"Starting execution of {len(models)} models...")

env = os.environ.copy()
env['BI_NSIM'] = os.environ.get('BI_NSIM', '10')

log_file = os.path.join(script_dir, "log.txt")
with open(log_file, "w") as f_log:
    f_log.write(f"Execution Log - {len(models)} models\n")
    f_log.write("="*50 + "\n")

    for model in models:
        print(f"\n{'='*50}")
        print(f"Running {model}...")
        print(f"{'='*50}")
        
        try:
            # Run the model script from its own directory
            model_path = os.path.join(script_dir, model)
            result = subprocess.run([python_exe, model_path], env=env, capture_output=True, text=True, check=True, cwd=script_dir)
            print(f"Successfully completed {model}")
            f_log.write(f"[SUCCESS] {model}\n")
            f_log.flush()
        except subprocess.CalledProcessError as e:
            print(f"Error running {model}:")
            print(e.stderr)
            f_log.write(f"[FAILURE] {model}\n")
            f_log.write(f"Error:\n{e.stderr}\n")
            f_log.write("-" * 30 + "\n")
            f_log.flush()
        except Exception as e:
            print(f"Unexpected error: {e}")
            f_log.write(f"[ERROR] {model}: {e}\n")
            f_log.flush()

print("\nAll models finished.")
print(f"Check {log_file} for details.")
