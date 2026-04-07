"""
Z-Score Hyperparameter Tuning Engine

This script leverages Optuna to optimize the ADAPTIVE VARIANCE-SCALED THRESHOLDS 
(Z-Scoring) for the AnomalyDetector. Instead of finding static boundaries, 
it finds the optimal standard deviation multipliers (k) for each zone.
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
# Z-SCORE OPTIMIZATION ENGINE
# ==========================================

class ZScoreFilterOptimizer:
    def __init__(self, data_path, stream_type='raw', param_type='ph'):
        print(f"Loading {param_type.upper()} {stream_type.upper()} dataset into memory...")
        self.df = pd.read_parquet(data_path)
        self.stream_type = stream_type
        self.param_type = param_type
        self.evaluator = ClinicalCostEvaluator()

        self.alpha = 0.3
        self.beta_map = {'ph': 0.10, 'temp': 0.01, 'combined': 0.10}
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
        print(f"  [Worker {os.getpid()}] Starting Z-Score Trial {trial.number}...")

        leakage = trial.suggest_float("raw_leakage", 0.85, 0.99)

        if self.param_type == 'ph' or self.param_type == 'temp':
            # Search between 1 and 8 standard deviations
            k_mult = trial.suggest_float("k_raw_multiplier", 1.0, 8.0, step=0.1)
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.current_beta, 
                use_z_score=True, 
                k_raw_multiplier=k_mult, 
                raw_leakage=leakage
            )
        elif self.param_type == 'combined':
            k_ph = trial.suggest_float("k_raw_ph", 1.0, 8.0, step=0.1)
            k_temp = trial.suggest_float("k_raw_temp", 1.0, 8.0, step=0.1)
            
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.current_beta, 
                beta_ph=self.beta_map['ph'], beta_temp=self.beta_map['temp'],
                use_z_score=True,
                k_raw_ph=k_ph, k_raw_temp=k_temp,
                raw_leakage=leakage
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
        print(f"  [Worker {os.getpid()}] Starting Z-Score Aggregated Trial {trial.number}...")

        persistence = trial.suggest_int("agg_persistence", 1, 3)

        if self.param_type == 'ph' or self.param_type == 'temp':
            # Aggregated streams are smoother, so they might need tighter sigmas (1.0 to 6.0)
            k_mult = trial.suggest_float("k_agg_multiplier", 1.0, 6.0, step=0.1)
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.current_beta, 
                use_z_score=True,
                k_agg_multiplier=k_mult,
                agg_persistence=persistence
            )
        elif self.param_type == 'combined':
            k_ph = trial.suggest_float("k_agg_ph", 1.0, 6.0, step=0.1)
            k_temp = trial.suggest_float("k_agg_temp", 1.0, 6.0, step=0.1)
            
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.current_beta, 
                beta_ph=self.beta_map['ph'], beta_temp=self.beta_map['temp'],
                use_z_score=True,
                k_agg_ph=k_ph, k_agg_temp=k_temp,
                agg_persistence=persistence
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

# ==========================================
# PIPELINE ORCHESTRATION
# ==========================================

def run_zscore_optimizations(datasets_to_tune=['ph', 'temp', 'combined'], save_path='tuning_results/completed_zscore_studies.pkl'):
    completed_studies = {}

    for p_type in datasets_to_tune:
        print(f"\n{'='*60}")
        print(f" STARTING Z-SCORE OPTIMIZATION FOR: {p_type.upper()}")
        print(f"{'='*60}\n")
        
        completed_studies[p_type] = {}
        # Changed DB name to prevent overwriting your old static runs
        db_url = f"sqlite:///optuna_study_zscore_{p_type}.db"
        SAFE_CORES = 3
        
        raw_opt = ZScoreFilterOptimizer(f'datasets/tuning_dataset_{p_type}_raw.parquet', stream_type='raw', param_type=p_type)
        study_raw = optuna.create_study(direction="minimize", study_name=f"Z_Raw_{p_type}", storage=db_url, load_if_exists=True)
        
        n_trials_raw = 40 if p_type == 'combined' else 20
        study_raw.optimize(raw_opt.objective_raw, n_trials=n_trials_raw, n_jobs=SAFE_CORES, show_progress_bar=False) 
        completed_studies[p_type]['raw'] = study_raw
        
        print(f"[{p_type.upper()} Z-RAW] Lowest Cost:   {study_raw.best_value:.4f}")

        del raw_opt
        gc.collect()

        agg_opt = ZScoreFilterOptimizer(f'datasets/tuning_dataset_{p_type}_agg.parquet', stream_type='agg', param_type=p_type)
        study_agg = optuna.create_study(direction="minimize", study_name=f"Z_Agg_{p_type}", storage=db_url, load_if_exists=True)
        
        n_trials_agg = 60 if p_type == 'combined' else 30
        study_agg.optimize(agg_opt.objective_agg, n_trials=n_trials_agg, n_jobs=SAFE_CORES, show_progress_bar=False)
        completed_studies[p_type]['agg'] = study_agg

        print(f"[{p_type.upper()} Z-AGG] Lowest Cost:   {study_agg.best_value:.4f}")

        del agg_opt
        gc.collect()

        joblib.dump(completed_studies, save_path)
        print(f"\n[PROGRESS SAVED] {p_type.upper()} Z-Scores safely written to {save_path}")
        
    print("\n All parallel Z-Score optimizations complete!")
    return completed_studies


if __name__ == '__main__':
    print(" Cleaning up old Z-Score optimization data...")
    for db_file in glob.glob("optuna_study_zscore_*.db"):
        try:
            os.remove(db_file)
        except Exception:
            pass

    os.makedirs('tuning_results', exist_ok=True)
    results_path = 'tuning_results/completed_zscore_studies.pkl'
    if os.path.exists(results_path):
        os.remove(results_path)

    print("\n" + "="*60)
    print(" STARTING Z-SCORE OPTIMIZATION ENGINE")
    print("="*60 + "\n")

    run_zscore_optimizations(['ph', 'temp', 'combined'], save_path=results_path)
    
    print("\n" + "="*60)
    print(" SUCCESS: Z-Score Optimization complete.")
    print("="*60)