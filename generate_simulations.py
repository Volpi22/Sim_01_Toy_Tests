import subprocess
import os

# --- CONFIGURATION ---
# Set this to the total number of CSV files you need. 
N_SIMS = 5

# Ensure to build ns-3 once before running this script to save time
NS3_DIRECTORY = os.path.expanduser("~/ns-allinone-3.36.1/ns-3.36.1")
OUTPUT_DIR = os.path.expanduser("~/EXPERIMENTS/Sim_01_Toy_Tests/raw_ns3_traces")

print(f" Starting ns-3 Trace Generation Pipeline ({N_SIMS} simulations)...")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(1, N_SIMS + 1):
    print(f"\n[ns-3 RUN] Executing simulation {i}/{N_SIMS}...")
    
    # Define the final destination for this specific run
    final_csv_path = os.path.join(OUTPUT_DIR, f"{i}_pH.csv")
    
    # Clean up any leftover default csvNano.csv to prevent stale reads
    stale_csv = os.path.join(NS3_DIRECTORY, "csvNano.csv")
    if os.path.exists(stale_csv):
        os.remove(stale_csv)

    # The command runs ns-3, then moves the output to our specific Output Directory with the new name
    ns3_command = f"./ns3 run toy_simulation > ~/EXPERIMENTS/Sim_01_Toy_Tests/log_output.txt 2>&1 && mv csvNano.csv {final_csv_path}"
    
    result = subprocess.run(ns3_command, shell=True, cwd=NS3_DIRECTORY, executable='/bin/bash')
    
    if result.returncode != 0:
        raise RuntimeError(f"ns-3 simulation {i} failed with return code {result.returncode}.")
    
    if not os.path.exists(final_csv_path):
        raise FileNotFoundError(f"ns-3 finished, but {final_csv_path} was not created!")
        
    print(f"  -> Successfully saved: {i}_pH.csv")

print("\n DONE! All ns-3 traces have been generated.")