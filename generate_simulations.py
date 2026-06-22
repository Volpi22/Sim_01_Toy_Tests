"""
NS-3 Trace Generation Pipeline
Executes the ns-3 simulator to generate the baseline 1-week spatial trace.
"""
import subprocess
import os

NS3_DIRECTORY = os.path.expanduser("~/ns-allinone-3.36.1/ns-3.36.1")
OUTPUT_DIR = os.path.expanduser("~/EXPERIMENTS/Sim_01_Toy_Tests/raw_ns3_traces")

def generate_1_week_trace():
    print("Starting 1-Week NS-3 Trace Generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    final_csv_path = os.path.join(OUTPUT_DIR, "1_week_trace.csv")
    log_file = os.path.join(OUTPUT_DIR, "..", "ns3_1_week_log.txt")

    # IMPORTANT: Ensure your 'toy_simulation' C++ file has the SimulationTime set to 604800 (1 week)
    ns3_command = f"./ns3 run toy_simulation > {log_file} 2>&1 && mv csvNano.csv {final_csv_path}"
    
    result = subprocess.run(ns3_command, shell=True, cwd=NS3_DIRECTORY, executable='/bin/bash')
    
    if result.returncode != 0:
        raise RuntimeError(f"NS-3 failed. Check {log_file} for C++ errors.")
        
    print(f"Successfully generated 1-week trace: {final_csv_path}")

if __name__ == '__main__':
    generate_1_week_trace()