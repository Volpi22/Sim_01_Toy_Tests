"""
NS-3 Trace Generation Pipeline

This script automates the execution of the ns-3 network simulator to generate 
baseline spatial discrete-event traces. It runs a specified ns-3 simulation 
multiple times, captures the output, handles file renaming, and routes the 
resulting CSVs to a centralized dataset directory for downstream disease simulation.
"""

import subprocess
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Ensure to build ns-3 once before running this script to save time
NS3_DIRECTORY = os.path.expanduser("~/ns-allinone-3.36.1/ns-3.36.1")
OUTPUT_DIR = os.path.expanduser("~/EXPERIMENTS/Sim_01_Toy_Tests/raw_ns3_traces")

# Define the total number of baseline CSV traces required
N_SIMS = 5

def generate_ns3_traces(num_sims, ns3_dir, output_dir):
    """
    Executes the ns-3 simulator sequentially to generate baseline spatial traces.
    
    @param num_sims (int): The number of independent simulation iterations to run.
    @param ns3_dir (str): The absolute path to the compiled ns-3 root directory.
    @param output_dir (str): The target directory where renamed CSV traces will be saved.
    
    @return: None. (Raises exceptions on failure).
    """
    print(f" Starting ns-3 Trace Generation Pipeline ({num_sims} simulations)...")
    
    # Ensure the target dataset directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the log file for capturing ns-3 stdout/stderr
    log_file = os.path.join(output_dir, "..", "log_output.txt")

    for i in range(1, num_sims + 1):
        print(f"\n[ns-3 RUN] Executing simulation {i}/{num_sims}...")
        
        # Define the final destination and standard naming convention for this run
        # Note: Saved simply as "{i}.csv" to match the DiseaseSimulator ingestion logic
        final_csv_path = os.path.join(output_dir, f"{i}.csv")
        
        # Clean up any leftover default trace files to prevent stale data reads
        stale_csv = os.path.join(ns3_dir, "csvNano.csv")
        if os.path.exists(stale_csv):
            os.remove(stale_csv)

        # Build the execution command:
        # 1. Run the specific toy simulation.
        # 2. Redirect stdout and stderr (2>&1) to the log file to keep the terminal clean.
        # 3. Upon success (&&), move and rename the generated csvNano.csv to our dataset folder.
        ns3_command = f"./ns3 run toy_simulation > {log_file} 2>&1 && mv csvNano.csv {final_csv_path}"
        
        # Execute the bash command synchronously
        result = subprocess.run(ns3_command, shell=True, cwd=ns3_dir, executable='/bin/bash')
        
        # --- ERROR HANDLING ---
        if result.returncode != 0:
            print(f"CRITICAL ERROR: ns-3 simulation {i} failed.")
            print(f"Check {log_file} for detailed C++ error logs.")
            raise RuntimeError(f"Return code: {result.returncode}")
        
        if not os.path.exists(final_csv_path):
            raise FileNotFoundError(f"ns-3 finished successfully, but {final_csv_path} was not created!")
            
        print(f"  -> Successfully generated and saved: {i}.csv")

    print("\n DONE! All ns-3 traces have been generated.")


if __name__ == '__main__':
    # Validate the ns-3 directory before starting
    if not os.path.exists(NS3_DIRECTORY):
        print(f"Error: ns-3 directory not found at {NS3_DIRECTORY}")
        sys.exit(1)
        
    try:
        generate_ns3_traces(
            num_sims=N_SIMS, 
            ns3_dir=NS3_DIRECTORY, 
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        print(f"\nPipeline halted due to error: {e}")