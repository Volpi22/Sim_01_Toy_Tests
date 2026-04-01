"""
Combined Dataset Assembly Module (Stability-Optimized)

This script is a specialized, memory-optimized orchestrator for generating the 
multimodal (combined pH and Temperature) dataset. Because combined simulations 
generate significantly larger DataFrames, this script prioritizes RAM stability 
through aggressive garbage collection, immediate datatype downcasting, and a 
constrained multiprocessing pool.
"""

import pandas as pd
import os
import multiprocessing
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from diseases import DiseaseSimulator

# ==========================================
# --- 1. DATASET REGULATION PARAMETERS ---
# ==========================================
# Resolve the path to the raw NS-3/BVS baseline traces dynamically
TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_ns3_traces")

N_CSVS_GENERATED = 5  
RUNS_PER_DISEASE_PER_CSV = 4

# Target strictly the combined multimodal diseases for this pipeline
COMBINED_DISEASES = [
    'septic_shock_combined', 'malignant_hyperthermia_combined', 
    'ischemic_bowel_combined', 'pyelo_combined', 'ischemia_combined'
]

# Error probabilities. Beta anchors to the worst-case sensor (pH) for confidence calculations.
BETA_VAL = 0.10 
ALPHA_VAL = 0.30

# Calculate a highly conservative core count to prevent out-of-memory (OOM) crashes
STABLE_CORES = max(1, multiprocessing.cpu_count() // 3)
# ==========================================

def process_patient(task_params):
    """
    Worker function to simulate a single multimodal patient scenario.
    
    @param task_params (tuple): A 4-element tuple containing:
                                - csv_path (str): Path to the base NS-3 trace CSV.
                                - disease_key (str): Identifier for the specific combined pathology.
                                - is_healthy (bool): Flag indicating if the patient is a healthy control.
                                - sim_id_str (str): Unique string identifier for the simulation.
                                
    @return tuple or None: Returns a tuple (raw_df, agg_df, sim_id_str) upon successful 
                           simulation. Returns None if an exception occurs to prevent pipeline crashes.
    """
    csv_path, disease_key, is_healthy, sim_id_str = task_params
    
    try:
        simulator = DiseaseSimulator(df_path=csv_path)
        
        # Route to the combined simulation engine
        if is_healthy:
            # Push the safe start out of bounds (24h) to guarantee healthy data
            raw_df, agg_df, _ = simulator.run_combined_disease(
                alpha=ALPHA_VAL, beta=BETA_VAL, disease_key=disease_key, safe_start_time=86400
            )
        else:
            raw_df, agg_df, raw_t_start = simulator.run_combined_disease(
                alpha=ALPHA_VAL, beta=BETA_VAL, disease_key=disease_key
            )

        # ------------------------------------------
        # Label Extraction & Ground Truth Alignment
        # ------------------------------------------
        label = 'healthy' if is_healthy else disease_key
        
        # Complex combined diseases often span multiple zones; default to -1 for broad tracking
        target_zone = -1 if is_healthy else (simulator.diseases[disease_key]['zones'][0] if len(simulator.diseases[disease_key]['zones']) == 1 else -1)
        
        t_start = 86400.00 if is_healthy else round(raw_t_start, 2)
        t_peak = 86400.00 if is_healthy else round(t_start + simulator.diseases[disease_key]['progression_time'], 2)

        # ------------------------------------------
        # Metadata Stamping & Memory Optimization
        # ------------------------------------------
        for df in [raw_df, agg_df]:
            df['Simulation_ID'] = sim_id_str
            df['Disease_Label'] = label
            df['Target_Zone'] = target_zone
            df['True_t_start'] = t_start
            df['True_t_peak'] = t_peak
            
            # MEMORY OPTIMIZATION: Downcast float64 to float32 immediately before returning 
            # to the master process to drastically reduce IPC overhead and RAM usage.
            fcols = df.select_dtypes(include=['float64']).columns
            df[fcols] = df[fcols].astype('float32')

        return raw_df, agg_df, sim_id_str
        
    except Exception as e:
        print(f"Error in {sim_id_str}: {e}")
        return None


if __name__ == '__main__':
    print(f"Starting STABILITY-FIRST Combined Dataset Assembly ({STABLE_CORES} cores)...")
    os.makedirs('datasets', exist_ok=True)

    tasks = []
    sim_counter = 1
    
    # 1. Task Queue Construction
    for csv_idx in range(1, N_CSVS_GENERATED + 1):
        csv_path = os.path.join(TRACES_DIR, f"{csv_idx}.csv")
        if not os.path.exists(csv_path): 
            continue
            
        # Queue pathological scenarios
        for disease in COMBINED_DISEASES:
            for _ in range(RUNS_PER_DISEASE_PER_CSV):
                tasks.append((csv_path, disease, False, f"sim_combined_{sim_counter:04d}"))
                sim_counter += 1
        
        # Queue healthy controls to maintain class balance
        healthy_runs = len(COMBINED_DISEASES) * RUNS_PER_DISEASE_PER_CSV
        for _ in range(healthy_runs):
            tasks.append((csv_path, COMBINED_DISEASES[0], True, f"sim_combined_{sim_counter:04d}"))
            sim_counter += 1

    master_raw_list = []
    master_agg_list = []
    total_tasks = len(tasks)
    
    # 2. Parallel Execution
    with ProcessPoolExecutor(max_workers=STABLE_CORES) as executor:
        futures = [executor.submit(process_patient, t) for t in tasks]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            
            # Unpack safely to avoid appending None if a worker failed
            if result:
                raw_df, agg_df, _ = result
                master_raw_list.append(raw_df)
                master_agg_list.append(agg_df)
            
            # Periodic status update and forced garbage collection
            if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                print(f"  -> Progress: {i+1}/{total_tasks} processed. RAM check encouraged.")
                gc.collect() 

    # 3. Concatenation and Serialization
    print(f"\nFINAL STEP: Concatenating {len(master_raw_list)} simulations...")
    
    if master_raw_list:
        # Process and save the raw dataset
        final_raw = pd.concat(master_raw_list, ignore_index=True).sort_values(by='Simulation_ID', kind='stable')
        final_raw.to_parquet('datasets/tuning_dataset_combined_raw.parquet', index=False)
        
        # Aggressively delete the raw memory block before processing the aggregated block
        del final_raw
        gc.collect()

        # Process and save the aggregated dataset
        final_agg = pd.concat(master_agg_list, ignore_index=True).sort_values(by='Simulation_ID', kind='stable')
        final_agg.to_parquet('datasets/tuning_dataset_combined_agg.parquet', index=False)
        
        print("COMBINED datasets saved successfully.")
    else:
        print("No data was generated. Check input trace files.")