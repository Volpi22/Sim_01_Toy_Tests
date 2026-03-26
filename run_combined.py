import pandas as pd
import os
import multiprocessing
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from diseases import DiseaseSimulator

# ==========================================
# --- 1. DATASET REGULATION PARAMETERS ---
# ==========================================
TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_ns3_traces")

N_CSVS_GENERATED = 5  
RUNS_PER_DISEASE_PER_CSV = 4

# Target only the combined diseases
COMBINED_DISEASES = [
    'septic_shock_combined', 'malignant_hyperthermia_combined', 
    'ischemic_bowel_combined', 'pyelo_combined', 'ischemia_combined'
]

BETA_VAL = 0.1 # Using pH beta for combined worst-case
ALPHA_VAL = 0.3

STABLE_CORES = max(1, multiprocessing.cpu_count() // 3)
# ==========================================

def process_patient(task_params):
    """Worker function to simulate a single patient."""
    csv_path, disease_key, is_healthy, sim_id_str = task_params
    
    try:
        simulator = DiseaseSimulator(df_path=csv_path)
        
        if is_healthy:
            raw_df, agg_df, _ = simulator.run_combined_disease(
                alpha=ALPHA_VAL, beta=BETA_VAL, disease_key=disease_key, safe_start_time=86400
            )
        else:
            raw_df, agg_df, raw_t_start = simulator.run_combined_disease(
                alpha=ALPHA_VAL, beta=BETA_VAL, disease_key=disease_key
            )

        # Labels
        label = 'healthy' if is_healthy else disease_key
        target_zone = -1 if is_healthy else (simulator.diseases[disease_key]['zones'][0] if len(simulator.diseases[disease_key]['zones']) == 1 else -1)
        t_start = 86400.00 if is_healthy else round(raw_t_start, 2)
        t_peak = 86400.00 if is_healthy else round(t_start + simulator.diseases[disease_key]['progression_time'], 2)

        for df in [raw_df, agg_df]:
            df['Simulation_ID'] = sim_id_str
            df['Disease_Label'] = label
            df['Target_Zone'] = target_zone
            df['True_t_start'] = t_start
            df['True_t_peak'] = t_peak
            # Optimize memory immediately
            fcols = df.select_dtypes(include=['float64']).columns
            df[fcols] = df[fcols].astype('float32')

        return raw_df, agg_df, sim_id_str
    except Exception as e:
        print(f"Error in {sim_id_str}: {e}")
        return None

if __name__ == '__main__':
    print(f" Starting STABILITY-FIRST Combined Dataset Assembly ({STABLE_CORES} cores)...")
    os.makedirs('datasets', exist_ok=True)

    tasks = []
    sim_counter = 1
    for csv_idx in range(1, N_CSVS_GENERATED + 1):
        csv_path = os.path.join(TRACES_DIR, f"{csv_idx}.csv")
        if not os.path.exists(csv_path): continue
            
        for disease in COMBINED_DISEASES:
            for _ in range(RUNS_PER_DISEASE_PER_CSV):
                tasks.append((csv_path, disease, False, f"sim_combined_{sim_counter:04d}"))
                sim_counter += 1
        
        healthy_runs = len(COMBINED_DISEASES) * RUNS_PER_DISEASE_PER_CSV
        for _ in range(healthy_runs):
            tasks.append((csv_path, COMBINED_DISEASES[0], True, f"sim_combined_{sim_counter:04d}"))
            sim_counter += 1

    master_raw_list = []
    master_agg_list = []
    total_tasks = len(tasks)
    
    with ProcessPoolExecutor(max_workers=STABLE_CORES) as executor:
        futures = [executor.submit(process_patient, t) for t in tasks]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                raw_df, agg_df, _ = result
                master_raw_list.append(raw_df)
                master_agg_list.append(agg_df)
            
            if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                print(f"  -> Progress: {i+1}/{total_tasks} processed. RAM check encouraged.")
                gc.collect() # Force garbage collection

    print(f"\n FINAL STEP: Concatenating {len(master_raw_list)} simulations...")
    
    if master_raw_list:
        final_raw = pd.concat(master_raw_list, ignore_index=True).sort_values(by='Simulation_ID', kind='stable')
        final_raw.to_parquet('datasets/tuning_dataset_combined_raw.parquet', index=False)
        del final_raw
        gc.collect()

        final_agg = pd.concat(master_agg_list, ignore_index=True).sort_values(by='Simulation_ID', kind='stable')
        final_agg.to_parquet('datasets/tuning_dataset_combined_agg.parquet', index=False)
        
        print(" COMBINED datasets saved successfully.")
    else:
        print(" No data was generated. Check input trace files.")