import pandas as pd
import numpy as np
import optuna
import warnings
import joblib
import os
import gc
import glob
import multiprocessing

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) 

from drift_detection import AnomalyDetector

class ClinicalCostEvaluator:
    """
    Evaluates filter performance based on the custom clinical Cost Function:
    Cost = (w1 * FPR) + (w2 * FNR) + (w3 * NPL)
    """
    def __init__(self, w1=1000.0, w2=1000.0, w3=300.0, total_zones=20):
        self.w1 = w1  
        self.w2 = w2  
        self.w3 = w3  
        self.total_zones = total_zones

    def calculate_cost(self, results_list):
        total_fp = 0
        total_fn = 0
        total_tp = 0
        total_tn = 0
        npl_sum = 0.0

        for res in results_list:
            target_zone = res['Target_Zone']
            t_start = res['True_t_start']
            t_peak = res['True_t_peak']
            det_times = res['Detection_Times']  

            if target_zone == -1:  # Healthy Patient
                fp_in_sim = len(det_times)
                total_fp += fp_in_sim
                total_tn += (self.total_zones - fp_in_sim)
            else:  # Diseased Patient
                if target_zone in det_times:
                    # EXTRACT THE DETECTION TIME SAFELY
                    raw_det_val = det_times[target_zone]
                    
                    if isinstance(raw_det_val, dict):
                        valid_times = [t for t in raw_det_val.values() if t is not None]
                        if not valid_times:
                            total_fn += 1
                            continue
                        t_det = min(valid_times) 
                    else:
                        t_det = raw_det_val

                    # EVALUATE THE DETECTION
                    if t_det >= t_start:
                        total_tp += 1
                        progression_window = t_peak - t_start
                        npl = min(1.0, (t_det - t_start) / progression_window)
                        npl_sum += npl
                    else:
                        total_fp += 1
                        total_fn += 1 
                else:
                    total_fn += 1

                other_zones_flagged = len([z for z in det_times.keys() if z != target_zone])
                total_fp += other_zones_flagged
                total_tn += ((self.total_zones - 1) - other_zones_flagged)

        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
        
        total_diseased = total_tp + total_fn
        fnr = total_fn / total_diseased if total_diseased > 0 else 0.0
        
        mean_npl = npl_sum / total_tp if total_tp > 0 else 1.0 

        cost = (self.w1 * fpr) + (self.w2 * fnr) + (self.w3 * mean_npl)
        return cost


class FilterOptimizer:
    """Optuna objective classes for both Raw and Aggregated tuning."""
    def __init__(self, data_path, stream_type='raw', param_type='ph'):
        print(f"Loading {param_type.upper()} {stream_type.upper()} dataset into memory...")
        self.df = pd.read_parquet(data_path)
        self.stream_type = stream_type
        self.param_type = param_type
        self.evaluator = ClinicalCostEvaluator()

        # Match the alpha/beta to the dataset generation exact values
        self.alpha = 0.3
        self.beta_map = {
            'ph': 0.10,
            'temp': 0.01,
            'combined': 0.10
        }
        self.current_beta = self.beta_map[self.param_type]

        self.simulations = []
        for sim_id, group in self.df.groupby('Simulation_ID'):
            first_row = group.iloc[0]
            self.simulations.append({
                'df': group.copy(),
                'Target_Zone': first_row['Target_Zone'],
                'True_t_start': first_row['True_t_start'],
                'True_t_peak': first_row['True_t_peak']
            })
        print(f"Loaded {len(self.simulations)} unique patient scenarios.")

    def objective_raw(self, trial):
        """1D/2D Optimization for the Raw Stream Kalman Filter."""
        print(f"  [Worker {os.getpid()}] Starting Trial {trial.number}...")

        if self.param_type == 'ph':
            thresh = trial.suggest_float("raw_drift_threshold", 0.01, 2.0, step=0.05)
            detector = AnomalyDetector(alpha=self.alpha, beta=self.current_beta, raw_drift_threshold=thresh)
        elif self.param_type == 'temp':
            thresh = trial.suggest_float("raw_drift_threshold", 0.5, 5.0, step=0.1)
            detector = AnomalyDetector(alpha=self.alpha, beta=self.current_beta, raw_drift_threshold=thresh)
        elif self.param_type == 'combined':
            thresh_ph = trial.suggest_float("raw_drift_ph", 0.01, 2.0, step=0.05)
            thresh_temp = trial.suggest_float("raw_drift_temp", 0.5, 5.0, step=0.1)
            
            # ---> UPDATED: Passing the specific betas for the combined raw stream! <---
            detector = AnomalyDetector(
                alpha=self.alpha, 
                beta=self.current_beta, 
                beta_ph=self.beta_map['ph'], 
                beta_temp=self.beta_map['temp'],
                raw_drift_ph=thresh_ph, 
                raw_drift_temp=thresh_temp
            )

        results = []
        for sim in self.simulations:
            if self.param_type == 'combined':
                _, det_times = detector.detect_combined_raw_stream(sim['df'])
            else:
                _, det_times = detector.detect_raw_stream(sim['df'])
                
            results.append({
                'Target_Zone': sim['Target_Zone'],
                'True_t_start': sim['True_t_start'],
                'True_t_peak': sim['True_t_peak'],
                'Detection_Times': det_times
            })
            
        cost = self.evaluator.calculate_cost(results)
        print(f"  [Trial {trial.number}] Finished. Cost: {cost:.4f}")
        return cost

    def objective_agg(self, trial):
        """2D/3D Optimization for the Aggregated Stream Kalman Filter."""
        print(f"  [Worker {os.getpid()}] Starting Aggregated Trial {trial.number}...")

        persistence = trial.suggest_int("agg_persistence", 1, 3)

        if self.param_type == 'ph':
            abs_drift = trial.suggest_float("agg_abs_drift", 0.01, 0.20, step=0.01)
            detector = AnomalyDetector(alpha=self.alpha, beta=self.current_beta, agg_persistence=persistence, agg_abs_drift=abs_drift)
        elif self.param_type == 'temp':
            abs_drift = trial.suggest_float("agg_abs_drift", 0.1, 1.5, step=0.1)
            detector = AnomalyDetector(alpha=self.alpha, beta=self.current_beta, agg_persistence=persistence, agg_abs_drift=abs_drift)
        elif self.param_type == 'combined':
            drift_ph = trial.suggest_float("agg_abs_drift_ph", 0.01, 0.20, step=0.01)
            drift_temp = trial.suggest_float("agg_abs_drift_temp", 0.1, 1.5, step=0.1)
            
            # ---> UPDATED: Passing the specific betas for the combined agg stream! <---
            detector = AnomalyDetector(
                alpha=self.alpha, 
                beta=self.current_beta, 
                beta_ph=self.beta_map['ph'], 
                beta_temp=self.beta_map['temp'],
                agg_persistence=persistence, 
                agg_abs_drift_ph=drift_ph, 
                agg_abs_drift_temp=drift_temp
            )

        results = []
        for sim in self.simulations:
            if self.param_type == 'combined':
                _, det_times = detector.detect_combined_aggregated_stream(sim['df'])
            else:
                _, det_times = detector.detect_aggregated_stream(sim['df'])
                
            results.append({
                'Target_Zone': sim['Target_Zone'],
                'True_t_start': sim['True_t_start'],
                'True_t_peak': sim['True_t_peak'],
                'Detection_Times': det_times
            })

        cost = self.evaluator.calculate_cost(results)
        print(f"  [Trial {trial.number}] Finished Aggregated. Cost: {cost:.4f}")
        return cost


def run_all_optimizations(datasets_to_tune=['ph', 'temp', 'combined'], save_path='tuning_results/completed_studies.pkl'):
    """
    Executes the tuning pipeline and saves progress progressively to free memory.
    """
    completed_studies = {}

    for p_type in datasets_to_tune:
        print(f"\n{'='*60}")
        print(f" STARTING OPTIMIZATION FOR: {p_type.upper()}")
        print(f"{'='*60}\n")
        
        completed_studies[p_type] = {}
        db_url = f"sqlite:///optuna_study_{p_type}.db"
        SAFE_CORES = 5
        
        # ==========================================
        # 1. RAW STREAM TUNING
        # ==========================================
        raw_opt = FilterOptimizer(f'datasets/tuning_dataset_{p_type}_raw.parquet', stream_type='raw', param_type=p_type)
        study_raw = optuna.create_study(direction="minimize", study_name=f"Raw_{p_type}", storage=db_url, load_if_exists=True)
        
        n_trials_raw = 40 if p_type == 'combined' else 20
        print(f"-> Running {n_trials_raw} trials for {p_type.upper()} RAW...")
        
        study_raw.optimize(raw_opt.objective_raw, n_trials=n_trials_raw, n_jobs=SAFE_CORES, show_progress_bar=False) 
        completed_studies[p_type]['raw'] = study_raw
        
        print(f"[{p_type.upper()} RAW] Lowest Cost:   {study_raw.best_value:.4f}")

        # 🧹 FREE RAM BEFORE LOADING THE NEXT DATASET
        del raw_opt
        import gc
        gc.collect()

        # ==========================================
        # 2. AGGREGATED STREAM TUNING
        # ==========================================
        agg_opt = FilterOptimizer(f'datasets/tuning_dataset_{p_type}_agg.parquet', stream_type='agg', param_type=p_type)
        study_agg = optuna.create_study(direction="minimize", study_name=f"Agg_{p_type}", storage=db_url, load_if_exists=True)
        
        n_trials_agg = 60 if p_type == 'combined' else 30
        print(f"-> Running {n_trials_agg} trials for {p_type.upper()} AGGREGATED...")
        
        study_agg.optimize(agg_opt.objective_agg, n_trials=n_trials_agg, n_jobs=SAFE_CORES, show_progress_bar=False)
        completed_studies[p_type]['agg'] = study_agg

        print(f"[{p_type.upper()} AGG] Lowest Cost:   {study_agg.best_value:.4f}")

        # 🧹 FREE RAM AGAIN
        del agg_opt
        gc.collect()

        # ==========================================
        # 3. PROGRESSIVE SAVE (THE SAFETY NET)
        # ==========================================
        joblib.dump(completed_studies, save_path)
        print(f"\n[💾] PROGRESS SAVED: {p_type.upper()} safely written to {save_path}")
        
        # 🧹 FORCE A FINAL GARBAGE COLLECTION AT THE END OF THE LOOP
        gc.collect() 
        
    print("\n All parallel optimizations complete!")
    return completed_studies

if __name__ == '__main__':
    # --- 0. CLEANUP & SETUP ---
    print(" Cleaning up old optimization data...")
    for db_file in glob.glob("optuna_study_*.db"):
        try:
            os.remove(db_file)
        except Exception:
            pass

    os.makedirs('tuning_results', exist_ok=True)
    results_path = 'tuning_results/completed_studies.pkl'
    if os.path.exists(results_path):
        os.remove(results_path)

    print("\n" + "="*60)
    print(" STARTING TERMINAL OPTIMIZATION ENGINE")
    print("="*60 + "\n")

    # --- 1. EXECUTION & PROGRESSIVE SAVING ---
    run_all_optimizations(['ph', 'temp', 'combined'], save_path=results_path)
    
    print("\n" + "="*60)
    print(" SUCCESS: Entire Optimization complete.")
    print(" You can now switch to your Jupyter Notebook to visualize.")
    print("="*60)