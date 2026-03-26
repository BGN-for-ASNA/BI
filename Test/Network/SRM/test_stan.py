from cmdstanpy import CmdStanModel
import os

print("Testing cmdstanpy...")
try:
    if os.path.exists('SRM.exe'):
        print("Found SRM.exe, loading...")
        sm = CmdStanModel(stan_file='SRM.stan', exe_file='SRM.exe')
    else:
        print("SRM.exe not found, compiling SRM.stan...")
        sm = CmdStanModel(stan_file='SRM.stan')
    print("Model initialized successfully.")
    print(f"Model name: {sm.name}")
except Exception as e:
    print(f"Failed to initialize model: {e}")
