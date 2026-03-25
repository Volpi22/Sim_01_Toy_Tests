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
# PH_DISEASES = ['systemic_alkalosis_ph', 'dka_ph']
PH_DISEASES = ['systemic_alkalosis_ph', 'dka_ph', 'stroke_ph', 'nec_ph', 'rta_ph']
RUNS_PER_DISEASE_PER_CSV = 4
HEALTHY_RUNS_PER_CSV = len(PH_DISEASES) * RUNS_PER_DISEASE_PER_CSV

NUM_CORES = multiprocessing.cpu_count()
# ==========================================

def process_patient(task_params):
    """Worker function to simulate a single patient and return the DataFrames."""
    csv_path, disease_key, is_healthy, sim_id_str = task_params
    
    simulator = DiseaseSimulator(df_path=csv_path)
    
    if is_healthy:
        raw_df, agg_df, _ = simulator.run_single_disease(
            alpha=0.3, beta=0.10, disease_key=disease_key, safe_start_time=86400
        )
        label = 'healthy'
        target_zone = -1
        t_start = 86400.00
        t_peak = 86400.00
    else:
        raw_df, agg_df, raw_t_start = simulator.run_single_disease(
            alpha=0.3, beta=0.10, disease_key=disease_key
        )
        label = disease_key
        target_zone = simulator.diseases[disease_key]['zones'][0] if len(simulator.diseases[disease_key]['zones']) == 1 else -1
        t_start = round(raw_t_start, 2)
        t_peak = round(t_start + simulator.diseases[disease_key]['progression_time'], 2)

    # Stamp DataFrames
    for df in [raw_df, agg_df]:
        df['Simulation_ID'] = sim_id_str
        df['Disease_Label'] = label
        df['Target_Zone'] = target_zone
        df['True_t_start'] = t_start
        df['True_t_peak'] = t_peak

    return raw_df, agg_df, sim_id_str


if __name__ == '__main__':
    total_diseased = N_CSVS_GENERATED * len(PH_DISEASES) * RUNS_PER_DISEASE_PER_CSV
    total_healthy = N_CSVS_GENERATED * HEALTHY_RUNS_PER_CSV
    total_tasks = total_diseased + total_healthy
    
    print(f" Starting PARALLEL Dataset Assembly on {NUM_CORES} cores...")
    print(f" Processing {total_tasks} total patient records...")

    # 1. Build the list of tasks
    tasks = []
    sim_counter = 1
    
    for csv_idx in range(1, N_CSVS_GENERATED + 1):
        csv_path = os.path.join(TRACES_DIR, f"{csv_idx}_pH.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing trace file: {csv_path}")
            
        # Add Diseased Tasks
        for disease in PH_DISEASES:
            for _ in range(RUNS_PER_DISEASE_PER_CSV):
                sim_id = f"sim_ph_{sim_counter:04d}"
                tasks.append((csv_path, disease, False, sim_id))
                sim_counter += 1
                
        # Add Healthy Tasks
        for _ in range(HEALTHY_RUNS_PER_CSV):
            sim_id = f"sim_ph_{sim_counter:04d}"
            # Pass the first disease key just to satisfy the simulator requirements
            tasks.append((csv_path, PH_DISEASES[0], True, sim_id))
            sim_counter += 1

    # 2. Execute tasks in parallel
    master_raw_list = []
    master_agg_list = []
    
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        # Submit all tasks and show progress
        futures = {executor.submit(process_patient, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            raw_df, agg_df, sim_id_str = future.result()
            master_raw_list.append(raw_df)
            master_agg_list.append(agg_df)
            
            completed += 1
            if completed % 10 == 0 or completed == total_tasks:
                print(f"  -> Progress: {completed}/{total_tasks} patients processed...")

    # 3. Concatenate and Save
    print("\nAll parallel tasks finished! Concatenating DataFrames...")
    final_raw_df = pd.concat(master_raw_list, ignore_index=True)
    final_agg_df = pd.concat(master_agg_list, ignore_index=True)

    print("Sorting data to restore sequential order...")
    # kind='stable' ensures that the perfect chronological order generated 
    # by the worker threads is preserved while grouping the IDs back together.
    final_raw_df.sort_values(by=['Simulation_ID'], kind='stable', inplace=True)
    final_agg_df.sort_values(by=['Simulation_ID'], kind='stable', inplace=True)

    print("Saving to Parquet files...")
    os.makedirs('datasets', exist_ok=True)
    final_raw_df.to_parquet('datasets/tuning_dataset_ph_raw.parquet', index=False)
    final_agg_df.to_parquet('datasets/tuning_dataset_ph_agg.parquet', index=False)

    print(f"DONE! Successfully compiled {total_tasks} records in parallel.")