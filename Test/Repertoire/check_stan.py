import cmdstanpy
try:
    print(f"CmdStan path: {cmdstanpy.cmdstan_path()}")
except Exception as e:
    print(f"Error: {e}")
