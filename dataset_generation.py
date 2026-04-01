"""
Dataset Assembly Module

This script orchestrates the parallel generation of synthetic physiological
data (healthy and pathological) used for fine-tuning the AnomalyDetector's
Kalman filters. It utilizes baseline discrete-event traces and overlays
specific disease profiles using the DiseaseSimulator. 

To manage large memory footprints, it employs asynchronous parallel processing 
and downcasts data types on the fly before saving to compressed Parquet formats.
"""

import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from diseases import DiseaseSimulator

# ==========================================
# --- 1. DATASET REGULATION PARAMETERS ---
# ==========================================
# Resolve the path to the raw NS-3/BVS baseline traces dynamically
TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_ns3_traces")

# Define the scale of the dataset corpus
N_CSVS_GENERATED = 5  
RUNS_PER_DISEASE_PER_CSV = 4

# Map the exact keys from the DiseaseSimulator to their respective datasets
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

# Define beta (measurement error probability) for each sensor type. 
# Note: Alpha (localization error) is kept constant at 0.3 for all simulations.
BETA_VALUES = {
    'ph': 0.10,
    'temp': 0.01,
    'combined': 0.10  # Anchoring to the highest beta as a conservative worst-case scenario
}

# ==========================================
# --- 2. WORKER FUNCTION ---
# ==========================================

def process_patient(task_params):
    """
    Worker function to simulate a single patient scenario and return the resulting DataFrames.

    @param task_params: A tuple containing the parameters for a single simulation run:
                        (csv_path, param_type, disease_key, is_healthy, sim_id_str)
                        - csv_path (str): Path to the base NS-3/BVS trace CSV.
                        - param_type (str): Type of sensor ('ph', 'temp', or 'combined').
                        - disease_key (str): Identifier for the specific pathology.
                        - is_healthy (bool): Flag indicating if the patient is healthy (control group).
                        - sim_id_str (str): Unique string identifier for the simulation.
                        
    @return: A tuple containing:
             - raw_df (pd.DataFrame): Time-series data of asynchronous nanobot pings.
             - agg_df (pd.DataFrame): Time-series data of statistically aggregated packets.
             - sim_id_str (str): The unique simulation identifier.
    """
    csv_path, param_type, disease_key, is_healthy, sim_id_str = task_params
    
    # Extract the precise hardware error profile for this simulation
    current_beta = BETA_VALUES[param_type]
    current_alpha = 0.3
    
    simulator = DiseaseSimulator(df_path=csv_path)
    
    # Route to the appropriate simulation method based on modality
    if param_type == 'combined':
        if is_healthy:
            # Safe start time pushed beyond 24h to ensure the patient remains healthy
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

    # ------------------------------------------
    # Calculate Ground Truth Labels for the Optimizer
    # ------------------------------------------
    label = 'healthy' if is_healthy else disease_key
    
    # If healthy, target zone is -1. If diseased, grab the primary target zone.
    # Note: Complex diseases with multiple zones default to -1 here for strict patient-level tuning.
    target_zone = -1 if is_healthy else (simulator.diseases[disease_key]['zones'][0] if len(simulator.diseases[disease_key]['zones']) == 1 else -1)
    
    # Establish ground-truth progression timelines
    t_start = 86400.00 if is_healthy else round(raw_t_start, 2)
    t_peak = 86400.00 if is_healthy else round(t_start + simulator.diseases[disease_key]['progression_time'], 2)

    # ------------------------------------------
    # Stamp DataFrames with Ground Truth Metadata
    # ------------------------------------------
    for df in [raw_df, agg_df]:
        df['Simulation_ID'] = sim_id_str
        df['Disease_Label'] = label
        df['Target_Zone'] = target_zone
        df['True_t_start'] = t_start
        df['True_t_peak'] = t_peak

    return raw_df, agg_df, sim_id_str


# ==========================================
# --- 3. MAIN EXECUTION PIPELINE ---
# ==========================================

if __name__ == '__main__':
    # Reserve cores to prevent system locking during massive I/O operations
    SAFE_CORES = max(1, multiprocessing.cpu_count() // 3)
    
    print(f" Starting Parallel Dataset Assembly on {SAFE_CORES} cores...")
    os.makedirs('datasets', exist_ok=True)

    for param_type, disease_list in DISEASES.items():
        print(f"\n{'='*60}\n ASSEMBLING {param_type.upper()} DATASETS \n{'='*60}")

        tasks = []
        sim_counter = 1
        
        # Build the task queue
        for csv_idx in range(1, N_CSVS_GENERATED + 1):
            csv_path = os.path.join(TRACES_DIR, f"{csv_idx}.csv")
            if not os.path.exists(csv_path): 
                raise FileNotFoundError(f"Missing base trace: {csv_path}")
                
            # Queue diseased patients
            for disease in disease_list:
                for _ in range(RUNS_PER_DISEASE_PER_CSV):
                    tasks.append((csv_path, param_type, disease, False, f"sim_{param_type}_{sim_counter:04d}"))
                    sim_counter += 1
            
            # Queue a proportional number of healthy control patients to balance the dataset
            healthy_runs = len(disease_list) * RUNS_PER_DISEASE_PER_CSV
            for _ in range(healthy_runs):
                tasks.append((csv_path, param_type, disease_list[0], True, f"sim_{param_type}_{sim_counter:04d}"))
                sim_counter += 1

        master_raw_list = []
        master_agg_list = []
        total_tasks = len(tasks)
        
        # Execute the task queue in parallel
        with ProcessPoolExecutor(max_workers=SAFE_CORES) as executor:
            futures = [executor.submit(process_patient, t) for t in tasks]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    raw_df, agg_df, sim_id = future.result()
                    
                    # MEMORY OPTIMIZATION: Downcast float64 to float32 to cut RAM footprint by 50%
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
        
        # Concatenate and sort to ensure temporal consistency
        final_raw = pd.concat(master_raw_list, ignore_index=True).sort_values(by='Simulation_ID', kind='stable')
        final_agg = pd.concat(master_agg_list, ignore_index=True).sort_values(by='Simulation_ID', kind='stable')
        
        # Write to Parquet (Provides significant read/write speed improvements and compression over CSV)
        final_raw.to_parquet(f'datasets/tuning_dataset_{param_type}_raw.parquet', index=False)
        final_agg.to_parquet(f'datasets/tuning_dataset_{param_type}_agg.parquet', index=False)
        
        # CLEAR MEMORY: Explicitly delete large data structures before the next loop iteration
        del master_raw_list
        del master_agg_list
        print(f" {param_type.upper()} saved. Memory cleared.")

    print("\n ALL DATASETS GENERATED SUCCESSFULLY!")