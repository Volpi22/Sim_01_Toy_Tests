import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from diseases import DiseaseSimulator

# ==========================================
# --- 1. DATASET REGULATION PARAMETERS ---
# ==========================================
# TRACES_DIR = os.path.expanduser("~/EXPERIMENTS/Sim_01_Toy_Tests/raw_ns3_traces")
TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_ns3_traces")

N_CSVS_GENERATED = 5  
RUNS_PER_DISEASE_PER_CSV = 4

# Map the exact keys from your DiseaseSimulator to their respective datasets
DISEASES = {
    'ph': [
        'systemic_alkalosis_ph', 'dka_ph', 'stroke_ph', 'nec_ph', 'rta_ph'
    ],
    'temp': [
        'systemic_fever_temp', 'endocarditis_temp', 'meningitis_temp', 'arterial_occlusion_temp', 'dvt_temp'
    ],
    'combined': [
        'septic_shock_combined', 'malignant_hyperthermia_combined', 'ischemic_bowel_combined', 'pyelo_combined', 'ischemia_combined'
    ]
}

# Define beta (measurement error) for each sensor type. 
# Alpha is kept constant at 0.3 for all.
BETA_VALUES = {
    'ph': 0.1,
    'temp': 0.01,
    'combined': 0.1 # Using the higher beta as a worst-case scenario for the combined model
}

# ==========================================

def process_patient(task_params):
    """Worker function to simulate a single patient and return the DataFrames."""
    csv_path, param_type, disease_key, is_healthy, sim_id_str = task_params
    
    # Grab the specific beta for this sensor type. Alpha is always 0.3.
    current_beta = BETA_VALUES[param_type]
    current_alpha = 0.3
    
    simulator = DiseaseSimulator(df_path=csv_path)
    
    # Select the correct simulation function based on the parameter type
    if param_type == 'combined':
        if is_healthy:
            raw_df, agg_df, _ = simulator.run_combined_disease(
                alpha=current_alpha, beta=current_beta, disease_key=disease_key, safe_start_time=86400
            )
        else:
            raw_df, agg_df, raw_t_start = simulator.run_combined_disease(
                alpha=current_alpha, beta=current_beta, disease_key=disease_key
            )
    else:
        if is_healthy:
            raw_df, agg_df, _ = simulator.run_single_disease(
                alpha=current_alpha, beta=current_beta, disease_key=disease_key, safe_start_time=86400
            )
        else:
            raw_df, agg_df, raw_t_start = simulator.run_single_disease(
                alpha=current_alpha, beta=current_beta, disease_key=disease_key
            )

    # Calculate Ground Truth Labels
    label = 'healthy' if is_healthy else disease_key
    target_zone = -1 if is_healthy else (simulator.diseases[disease_key]['zones'][0] if len(simulator.diseases[disease_key]['zones']) == 1 else -1)
    t_start = 86400.00 if is_healthy else round(raw_t_start, 2)
    t_peak = 86400.00 if is_healthy else round(t_start + simulator.diseases[disease_key]['progression_time'], 2)

    # Stamp DataFrames
    for df in [raw_df, agg_df]:
        df['Simulation_ID'] = sim_id_str
        df['Disease_Label'] = label
        df['Target_Zone'] = target_zone
        df['True_t_start'] = t_start
        df['True_t_peak'] = t_peak

    return raw_df, agg_df, sim_id_str


if __name__ == '__main__':
    SAFE_CORES = max(1, multiprocessing.cpu_count() // 3)
    
    print(f" Starting Parallel Dataset Assembly on {SAFE_CORES} cores...")
    os.makedirs('datasets', exist_ok=True)

    for param_type, disease_list in DISEASES.items():
        print(f"\n{'='*60}\n ASSEMBLING {param_type.upper()} DATASETS \n{'='*60}")

        tasks = []
        sim_counter = 1
        for csv_idx in range(1, N_CSVS_GENERATED + 1):
            csv_path = os.path.join(TRACES_DIR, f"{csv_idx}.csv")
            if not os.path.exists(csv_path): raise FileNotFoundError(f"Missing: {csv_path}")
                
            for disease in disease_list:
                for _ in range(RUNS_PER_DISEASE_PER_CSV):
                    tasks.append((csv_path, param_type, disease, False, f"sim_{param_type}_{sim_counter:04d}"))
                    sim_counter += 1
            
            healthy_runs = len(disease_list) * RUNS_PER_DISEASE_PER_CSV
            for _ in range(healthy_runs):
                tasks.append((csv_path, param_type, disease_list[0], True, f"sim_{param_type}_{sim_counter:04d}"))
                sim_counter += 1

        master_raw_list = []
        master_agg_list = []
        total_tasks = len(tasks)
        
        # Use a more memory-efficient execution pattern
        with ProcessPoolExecutor(max_workers=SAFE_CORES) as executor:
            futures = [executor.submit(process_patient, t) for t in tasks]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    raw_df, agg_df, sim_id = future.result()
                    
                    # MEMORY OPTIMIZATION: Convert to Float32 to save 50% RAM
                    for df in [raw_df, agg_df]:
                        float_cols = df.select_dtypes(include=['float64']).columns
                        df[float_cols] = df[float_cols].astype('float32')
                    
                    master_raw_list.append(raw_df)
                    master_agg_list.append(agg_df)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                        print(f"  -> Progress: {i+1}/{total_tasks} processed...")
                except Exception as e:
                    print(f"CRITICAL ERROR in worker: {e}")

        print(f"\n Concatenating and saving {param_type.upper()}...")
        
        # Concatenate and sort
        final_raw = pd.concat(master_raw_list, ignore_index=True).sort_values(by='Simulation_ID', kind='stable')
        final_agg = pd.concat(master_agg_list, ignore_index=True).sort_values(by='Simulation_ID', kind='stable')
        
        # Write to Parquet (much faster and smaller than CSV)
        final_raw.to_parquet(f'datasets/tuning_dataset_{param_type}_raw.parquet', index=False)
        final_agg.to_parquet(f'datasets/tuning_dataset_{param_type}_agg.parquet', index=False)
        
        # CLEAR MEMORY: Explicitly delete the lists before starting the next parameter type
        del master_raw_list
        del master_agg_list
        print(f" {param_type.upper()} saved. Memory cleared.")

    print("\n ALL DATASETS GENERATED SUCCESSFULLY!")