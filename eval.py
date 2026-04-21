import os
import sys
import joblib
from evaluation import ComprehensiveTestSuite, TargetedCohortTestSuite

# ==========================================
# LOGGER SETUP
# ==========================================
class Logger(object):
    """
    Custom logger that duplicates sys.stdout to both the terminal 
    and a specified text file.
    """
    def __init__(self, filename="full_evaluation_results.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

sys.stdout = Logger("full_evaluation_results.txt")


# ==========================================
# 1. COMPREHENSIVE EVALUATIONS
# ==========================================
def run_comprehensive_static():
    results_path = 'tuning_results/completed_studies_BEST.pkl'
    if not os.path.exists(results_path): return
    
    print("\n\n" + "="*80)
    print(" === 1. COMPREHENSIVE GENERALIZED EVALUATION (STATIC THRESHOLDS) ===")
    print("="*80)
    
    st = joblib.load(results_path)
    params = {
        'ph_raw_drift': st['ph']['raw'].best_params['raw_drift_threshold'],
        'ph_raw_leakage': st['ph']['raw'].best_params['raw_leakage'],
        'ph_agg_pers': st['ph']['agg'].best_params['agg_persistence'],
        'ph_agg_drift': st['ph']['agg'].best_params['agg_abs_drift'],
        'temp_raw_drift': st['temp']['raw'].best_params['raw_drift_threshold'],
        'temp_raw_leakage': st['temp']['raw'].best_params['raw_leakage'],
        'temp_agg_pers': st['temp']['agg'].best_params['agg_persistence'],
        'temp_agg_drift': st['temp']['agg'].best_params['agg_abs_drift'],
        'comb_raw_drift_ph': st['combined']['raw'].best_params['raw_drift_ph'],
        'comb_raw_drift_temp': st['combined']['raw'].best_params['raw_drift_temp'],
        'comb_raw_leakage': st['combined']['raw'].best_params['raw_leakage'],
        'comb_agg_pers': st['combined']['agg'].best_params['agg_persistence'],
        'comb_agg_drift_ph': st['combined']['agg'].best_params['agg_abs_drift_ph'],
        'comb_agg_drift_temp': st['combined']['agg'].best_params['agg_abs_drift_temp']
    }
    
    tester = ComprehensiveTestSuite(params, use_z_score=False)
    tester.run_full_suite(num_iterations=10)

def run_comprehensive_zscore():
    results_path = 'tuning_results/completed_zscore_studies.pkl'
    if not os.path.exists(results_path): return
    
    print("\n\n" + "="*80)
    print(" === 2. COMPREHENSIVE GENERALIZED EVALUATION (Z-SCORE) ===")
    print("="*80)
    
    st = joblib.load(results_path)
    params = {
        'ph_k_raw': st['ph']['raw'].best_params['k_raw_multiplier'],
        'ph_k_agg': st['ph']['agg'].best_params['k_agg_multiplier'],
        'ph_raw_leakage': st['ph']['raw'].best_params['raw_leakage'],
        'ph_agg_pers': st['ph']['agg'].best_params['agg_persistence'],
        'temp_k_raw': st['temp']['raw'].best_params['k_raw_multiplier'],
        'temp_k_agg': st['temp']['agg'].best_params['k_agg_multiplier'],
        'temp_raw_leakage': st['temp']['raw'].best_params['raw_leakage'],
        'temp_agg_pers': st['temp']['agg'].best_params['agg_persistence'],
        'comb_k_raw_ph': st['combined']['raw'].best_params['k_raw_ph'],
        'comb_k_raw_temp': st['combined']['raw'].best_params['k_raw_temp'],
        'comb_k_agg_ph': st['combined']['agg'].best_params['k_agg_ph'],
        'comb_k_agg_temp': st['combined']['agg'].best_params['k_agg_temp'],
        'comb_raw_leakage': st['combined']['raw'].best_params['raw_leakage'],
        'comb_agg_pers': st['combined']['agg'].best_params['agg_persistence']
    }
    
    tester = ComprehensiveTestSuite(params, use_z_score=True)
    tester.run_full_suite(num_iterations=10)


# ==========================================
# 2. TARGETED EVALUATIONS
# ==========================================
def run_targeted_static(cohort_name):
    results_path = f'tuning_results/completed_static_{cohort_name}.pkl'
    if not os.path.exists(results_path): return
    
    print("\n\n" + "="*80)
    print(f" === TARGETED STATIC EVALUATION: {cohort_name.upper()} ===")
    print("="*80)
    
    st = joblib.load(results_path)
    params = {
        'comb_raw_drift_ph': st['combined']['raw'].best_params['raw_drift_ph'],
        'comb_raw_drift_temp': st['combined']['raw'].best_params['raw_drift_temp'],
        'comb_raw_leakage': st['combined']['raw'].best_params['raw_leakage'],
        'comb_agg_pers': st['combined']['agg'].best_params['agg_persistence'],
        'comb_agg_drift_ph': st['combined']['agg'].best_params['agg_abs_drift_ph'],
        'comb_agg_drift_temp': st['combined']['agg'].best_params['agg_abs_drift_temp']
    }
    
    tester = TargetedCohortTestSuite(cohort_name, params, use_z_score=False)
    tester.run_targeted_suite(num_iterations=10)

def run_targeted_zscore(cohort_name):
    results_path = f'tuning_results/completed_zscore_{cohort_name}.pkl'
    if not os.path.exists(results_path): return
    
    print("\n\n" + "="*80)
    print(f" === TARGETED Z-SCORE EVALUATION: {cohort_name.upper()} ===")
    print("="*80)
    
    st = joblib.load(results_path)
    params = {
        'comb_k_raw_ph': st['combined']['raw'].best_params['k_raw_ph'],
        'comb_k_raw_temp': st['combined']['raw'].best_params['k_raw_temp'],
        'comb_raw_leakage': st['combined']['raw'].best_params['raw_leakage'],
        'comb_agg_pers': st['combined']['agg'].best_params['agg_persistence'],
        'comb_k_agg_ph': st['combined']['agg'].best_params['k_agg_ph'],
        'comb_k_agg_temp': st['combined']['agg'].best_params['k_agg_temp']
    }
    
    tester = TargetedCohortTestSuite(cohort_name, params, use_z_score=True)
    tester.run_targeted_suite(num_iterations=10)


# ==========================================
# EXECUTE ALL
# ==========================================
try:
    run_comprehensive_static()
    run_comprehensive_zscore()
    
    run_targeted_static('post_op')
    run_targeted_static('diabetes')
    
    run_targeted_zscore('post_op')
    run_targeted_zscore('diabetes')

finally:
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print("\n[SUCCESS] All evaluations complete. Master log saved to 'full_evaluation_results.txt'")