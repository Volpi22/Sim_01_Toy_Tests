"""
Targeted Hyperparameter Tuning Engine (Static Boundaries) - COMBINED ONLY

Optimizes the AnomalyDetector strictly for specific clinical profiles 
(Post-Operative or Diabetes-Related) using the Dual-Sensor (Combined) architecture.
"""

import pandas as pd
import numpy as np
import optuna
import warnings
import joblib
import os
import gc
import glob

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) 

from drift_detection import AnomalyDetector
from parameter_tuning import ClinicalCostEvaluator

# ==========================================
# CLINICAL COHORT DEFINITIONS
# ==========================================
COHORTS = {
    'post_op': [
        'systemic_alkalosis_ph', 'dvt_temp', 'malignant_hyperthermia_combined', 
        'septic_shock_combined', 'ischemic_bowel_combined'
    ],
    'diabetes': [
        'dka_ph', 'stroke_ph', 'rta_ph', 
        'ischemia_combined', 'pyelo_combined'
    ]
}

# ==========================================
# OPTIMIZATION ENGINE
# ==========================================

class TargetedCombinedOptimizer:
    def __init__(self, data_path, cohort_name, stream_type='raw'):
        print(f"Loading COMBINED {stream_type.upper()} dataset...")
        df = pd.read_parquet(data_path)
        self.stream_type = stream_type
        self.evaluator = ClinicalCostEvaluator()

        self.alpha = 0.3
        self.beta = 0.10
        self.beta_temp = 0.01

        # ---------------------------------------------------------
        # TARGETED DATASET FILTERING & BALANCING
        # ---------------------------------------------------------
        target_diseases = COHORTS[cohort_name]
        
        diseased_sims = []
        healthy_sims = []
        
        for sim_id, group in df.groupby('Simulation_ID'):
            first_row = group.iloc[0]
            label = first_row.get('Disease_Label', 'healthy')
            
            sim_data = {
                'df': group.copy(),
                'Target_Zone': first_row['Target_Zone'],
                'True_t_start': first_row['True_t_start'],
                'True_t_peak': first_row['True_t_peak']
            }
            
            if label in target_diseases:
                diseased_sims.append(sim_data)
            elif label == 'healthy':
                healthy_sims.append(sim_data)

        num_diseased = len(diseased_sims)
        self.simulations = diseased_sims + healthy_sims[:num_diseased]
        
        print(f"[{cohort_name.upper()}] Filtered Corpus: {num_diseased} Diseased + {len(healthy_sims[:num_diseased])} Healthy Controls.")

    def objective_raw(self, trial):
        leakage = trial.suggest_float("raw_leakage", 0.85, 0.99)
        thresh_ph = trial.suggest_float("raw_drift_ph", 0.01, 2.0, step=0.05)
        thresh_temp = trial.suggest_float("raw_drift_temp", 0.5, 5.0, step=0.1)
        
        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, beta_ph=self.beta, beta_temp=self.beta_temp,
            raw_drift_ph=thresh_ph, raw_drift_temp=thresh_temp, raw_leakage=leakage
        )

        results = []
        for sim in self.simulations:
            _, det_times = detector.detect_combined_raw_stream(sim['df'])
            results.append({
                'Target_Zone': sim['Target_Zone'], 'True_t_start': sim['True_t_start'],
                'True_t_peak': sim['True_t_peak'], 'Detection_Times': det_times
            })
            
        return self.evaluator.calculate_cost(results)

    def objective_agg(self, trial):
        persistence = trial.suggest_int("agg_persistence", 1, 3)
        drift_ph = trial.suggest_float("agg_abs_drift_ph", 0.01, 0.20, step=0.01)
        drift_temp = trial.suggest_float("agg_abs_drift_temp", 0.1, 1.5, step=0.1)
        
        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, beta_ph=self.beta, beta_temp=self.beta_temp,
            agg_persistence=persistence, agg_abs_drift_ph=drift_ph, agg_abs_drift_temp=drift_temp
        )

        results = []
        for sim in self.simulations:
            _, det_times = detector.detect_combined_aggregated_stream(sim['df'])
            results.append({
                'Target_Zone': sim['Target_Zone'], 'True_t_start': sim['True_t_start'],
                'True_t_peak': sim['True_t_peak'], 'Detection_Times': det_times
            })

        return self.evaluator.calculate_cost(results)

# ==========================================
# PIPELINE ORCHESTRATION
# ==========================================

def run_targeted_optimizations(cohort_name):
    save_path = f'tuning_results/completed_static_{cohort_name}.pkl'
    completed_studies = {'combined': {}}

    print(f"\n{'='*60}\n STARTING COMBINED OPTIMIZATION FOR: {cohort_name.upper()}\n{'='*60}\n")
    db_url = f"sqlite:///optuna_study_target_{cohort_name}_combined.db"
    SAFE_CORES = 5
    
    # RAW
    raw_opt = TargetedCombinedOptimizer('datasets/tuning_dataset_combined_raw.parquet', cohort_name, 'raw')
    study_raw = optuna.create_study(direction="minimize", study_name="Raw_combined", storage=db_url, load_if_exists=True)
    study_raw.optimize(raw_opt.objective_raw, n_trials=40, n_jobs=SAFE_CORES, show_progress_bar=False) 
    completed_studies['combined']['raw'] = study_raw
    del raw_opt
    gc.collect()

    # AGGREGATED
    agg_opt = TargetedCombinedOptimizer('datasets/tuning_dataset_combined_agg.parquet', cohort_name, 'agg')
    study_agg = optuna.create_study(direction="minimize", study_name="Agg_combined", storage=db_url, load_if_exists=True)
    study_agg.optimize(agg_opt.objective_agg, n_trials=60, n_jobs=SAFE_CORES, show_progress_bar=False)
    completed_studies['combined']['agg'] = study_agg
    del agg_opt
    gc.collect()

    joblib.dump(completed_studies, save_path)
    return completed_studies

if __name__ == '__main__':
    for db_file in glob.glob("optuna_study_target_*.db"):
        try: os.remove(db_file)
        except: pass

    os.makedirs('tuning_results', exist_ok=True)
    
    print("\n" + "="*60 + "\n 1. TUNING POST-OPERATIVE PROFILE\n" + "="*60)
    run_targeted_optimizations('post_op')
    
    print("\n" + "="*60 + "\n 2. TUNING DIABETES PROFILE\n" + "="*60)
    run_targeted_optimizations('diabetes')