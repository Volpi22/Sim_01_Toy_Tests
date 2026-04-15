"""
Targeted Z-Score Tuning Engine - COMBINED ONLY

Optimizes the Adaptive Variance-Scaled Thresholds (Z-Scoring) strictly 
for specific clinical profiles using the Dual-Sensor architecture.
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

class TargetedZScoreCombinedOptimizer:
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
                'df': group.copy(), 'Target_Zone': first_row['Target_Zone'],
                'True_t_start': first_row['True_t_start'], 'True_t_peak': first_row['True_t_peak']
            }
            if label in target_diseases: diseased_sims.append(sim_data)
            elif label == 'healthy': healthy_sims.append(sim_data)

        num_diseased = len(diseased_sims)
        self.simulations = diseased_sims + healthy_sims[:num_diseased]
        print(f"[{cohort_name.upper()}] Filtered Corpus: {num_diseased} Diseased + {len(healthy_sims[:num_diseased])} Healthy Controls.")

    def objective_raw(self, trial):
        leakage = trial.suggest_float("raw_leakage", 0.85, 0.99)
        k_ph = trial.suggest_float("k_raw_ph", 1.0, 8.0, step=0.1)
        k_temp = trial.suggest_float("k_raw_temp", 1.0, 8.0, step=0.1)
        
        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, beta_ph=self.beta, beta_temp=self.beta_temp,
            use_z_score=True, k_raw_ph=k_ph, k_raw_temp=k_temp, raw_leakage=leakage
        )

        results = []
        for sim in self.simulations:
            _, det_times = detector.detect_combined_raw_stream(sim['df'])
            results.append({'Target_Zone': sim['Target_Zone'], 'True_t_start': sim['True_t_start'], 'True_t_peak': sim['True_t_peak'], 'Detection_Times': det_times})
        return self.evaluator.calculate_cost(results)

    def objective_agg(self, trial):
        persistence = trial.suggest_int("agg_persistence", 1, 3)
        k_ph = trial.suggest_float("k_agg_ph", 1.0, 6.0, step=0.1)
        k_temp = trial.suggest_float("k_agg_temp", 1.0, 6.0, step=0.1)
        
        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, beta_ph=self.beta, beta_temp=self.beta_temp,
            use_z_score=True, k_agg_ph=k_ph, k_agg_temp=k_temp, agg_persistence=persistence
        )

        results = []
        for sim in self.simulations:
            _, det_times = detector.detect_combined_aggregated_stream(sim['df'])
            results.append({'Target_Zone': sim['Target_Zone'], 'True_t_start': sim['True_t_start'], 'True_t_peak': sim['True_t_peak'], 'Detection_Times': det_times})
        return self.evaluator.calculate_cost(results)

def run_targeted_zscore_optimizations(cohort_name):
    save_path = f'tuning_results/completed_zscore_{cohort_name}.pkl'
    completed_studies = {'combined': {}}

    print(f"\n{'='*60}\n STARTING COMBINED Z-SCORE OPTIMIZATION FOR: {cohort_name.upper()}\n{'='*60}\n")
    db_url = f"sqlite:///optuna_study_target_zscore_{cohort_name}_combined.db"
    SAFE_CORES = 3
    
    # RAW
    raw_opt = TargetedZScoreCombinedOptimizer('datasets/tuning_dataset_combined_raw.parquet', cohort_name, 'raw')
    study_raw = optuna.create_study(direction="minimize", study_name="Z_Raw_combined", storage=db_url, load_if_exists=True)
    study_raw.optimize(raw_opt.objective_raw, n_trials=40, n_jobs=SAFE_CORES, show_progress_bar=False) 
    completed_studies['combined']['raw'] = study_raw
    del raw_opt
    gc.collect()

    # AGGREGATED
    agg_opt = TargetedZScoreCombinedOptimizer('datasets/tuning_dataset_combined_agg.parquet', cohort_name, 'agg')
    study_agg = optuna.create_study(direction="minimize", study_name="Z_Agg_combined", storage=db_url, load_if_exists=True)
    study_agg.optimize(agg_opt.objective_agg, n_trials=60, n_jobs=SAFE_CORES, show_progress_bar=False)
    completed_studies['combined']['agg'] = study_agg
    del agg_opt
    gc.collect()

    joblib.dump(completed_studies, save_path)
    return completed_studies

if __name__ == '__main__':
    for db_file in glob.glob("optuna_study_target_zscore_*.db"):
        try: os.remove(db_file)
        except: pass

    os.makedirs('tuning_results', exist_ok=True)
    
    print("\n" + "="*60 + "\n 1. TUNING Z-SCORE POST-OPERATIVE PROFILE\n" + "="*60)
    run_targeted_zscore_optimizations('post_op')
    
    print("\n" + "="*60 + "\n 2. TUNING Z-SCORE DIABETES PROFILE\n" + "="*60)
    run_targeted_zscore_optimizations('diabetes')