import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from diseases import DiseaseSimulator

# ==========================================
# --- 1. DATASET REGULATION PARAMETERS ---
# ==========================================
TRACES_DIR = os.path.expanduser("~/EXPERIMENTS/Sim_01_Toy_Tests/raw_ns3_traces")

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

# Leave cores free for OS stability
NUM_CORES = max(1, multiprocessing.cpu_count() - 3)
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
    print(f" Starting Massive Dataset Assembly on {NUM_CORES} cores...")
    os.makedirs('datasets', exist_ok=True)

    # Loop through pH, Temp, and Combined
    for param_type, disease_list in DISEASES.items():
        print(f"\n{'='*60}")
        print(f" ASSEMBLING {param_type.upper()} DATASETS ")
        print(f"{'='*60}")

        healthy_runs_per_csv = len(disease_list) * RUNS_PER_DISEASE_PER_CSV
        total_diseased = N_CSVS_GENERATED * len(disease_list) * RUNS_PER_DISEASE_PER_CSV
        total_healthy = N_CSVS_GENERATED * healthy_runs_per_csv
        total_tasks = total_diseased + total_healthy
        
        print(f" Processing {total_tasks} patient records for {param_type.upper()}...")

        # 1. Build the list of tasks for this parameter type
        tasks = []
        sim_counter = 1
        
        for csv_idx in range(1, N_CSVS_GENERATED + 1):
            # Changed to match your generalized naming convention (e.g., 1.csv)
            csv_path = os.path.join(TRACES_DIR, f"{csv_idx}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing trace file: {csv_path}")
                
            # Add Diseased Tasks
            for disease in disease_list:
                for _ in range(RUNS_PER_DISEASE_PER_CSV):
                    sim_id = f"sim_{param_type}_{sim_counter:04d}"
                    tasks.append((csv_path, param_type, disease, False, sim_id))
                    sim_counter += 1
                    
            # Add Healthy Tasks
            for _ in range(healthy_runs_per_csv):
                sim_id = f"sim_{param_type}_{sim_counter:04d}"
                tasks.append((csv_path, param_type, disease_list[0], True, sim_id))
                sim_counter += 1

        # 2. Execute tasks in parallel
        master_raw_list = []
        master_agg_list = []
        
        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            futures = {executor.submit(process_patient, task): task for task in tasks}
            
            completed = 0
            for future in as_completed(futures):
                raw_df, agg_df, sim_id_str = future.result()
                master_raw_list.append(raw_df)
                master_agg_list.append(agg_df)
                
                completed += 1
                if completed % 10 == 0 or completed == total_tasks:
                    print(f"  -> Progress: {completed}/{total_tasks} patients processed...")

        # 3. Concatenate, Sort, and Save
        print(f"\n {param_type.upper()} tasks finished! Sorting and Concatenating...")
        final_raw_df = pd.concat(master_raw_list, ignore_index=True)
        final_agg_df = pd.concat(master_agg_list, ignore_index=True)

        final_raw_df.sort_values(by=['Simulation_ID'], kind='stable', inplace=True)
        final_agg_df.sort_values(by=['Simulation_ID'], kind='stable', inplace=True)

        # Dynamically name the output files based on the loop type
        raw_out_path = f'datasets/tuning_dataset_{param_type}_raw.parquet'
        agg_out_path = f'datasets/tuning_dataset_{param_type}_agg.parquet'
        
        final_raw_df.to_parquet(raw_out_path, index=False)
        final_agg_df.to_parquet(agg_out_path, index=False)

        print(f" {param_type.upper()} Datasets Saved Successfully!")

    print("\n ALL DATASETS GENERATED SUCCESSFULLY! You are ready for Optuna Tuning.")