"""
A-Posteriori Likelihood Extractor for Spatial Naive Bayes Classification

This independent script runs optimized Kalman filters over the TRAINING/TUNING dataset.
It automatically loads the optimized hyperparameters directly from the serialized
Optuna studies (.pkl files) generated during the tuning phase.
It extracts the empirical True Positive Rates (TPR) and False Positive Rates (FPR) 
for both Raw and Aggregated streams across all architectures and cohorts.
Laplace smoothing is applied to mathematically prevent Zero-Frequency singularities.
The final dictionary is saved directly to a .txt file.
"""

from diseases import DiseaseSimulator
from drift_detection import AnomalyDetector
from joblib import Parallel, delayed
import joblib
import multiprocessing
import pprint
import os

# ==========================================
# CONFIGURATION
# ==========================================

COHORTS = {
    'generalized': [
        'arterial_occlusion_temp', 'dka_ph', 'dvt_temp', 'endocarditis_temp', 
        'ischemia_combined', 'ischemic_bowel_combined', 'malignant_hyperthermia_combined', 
        'meningitis_temp', 'nec_ph', 'pyelo_combined', 'rta_ph', 
        'septic_shock_combined', 'stroke_ph', 'systemic_alkalosis_ph', 'systemic_fever_temp'
    ],
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
# OPTUNA PARAMETER LOADER
# ==========================================

def load_optuna_params(cohort_name, arch_type):
    """
    Dynamically loads the best parameters from the serialized Optuna studies.
    """
    if cohort_name == 'generalized':
        filepath = 'tuning_results/completed_studies.pkl' if arch_type == 'static' else 'tuning_results/completed_zscore_studies.pkl'
    else:
        filepath = f'tuning_results/completed_static_{cohort_name}.pkl' if arch_type == 'static' else f'tuning_results/completed_zscore_{cohort_name}.pkl'

    if not os.path.exists(filepath):
        print(f"[WARNING] Tuning file not found: {filepath}. Skipping {cohort_name.upper()} | {arch_type.upper()}.")
        return None

    studies = joblib.load(filepath)
    params = {}
    
    # 1. Load COMBINED params (used for targeted cohorts and dual-sensor diseases)
    if 'combined' in studies:
        params.update(studies['combined']['raw'].best_params)
        params.update(studies['combined']['agg'].best_params)
        
    # 2. If generalized, we also need single params for the single-parameter diseases
    if cohort_name == 'generalized':
        if 'ph' in studies:
            ph_raw = studies['ph']['raw'].best_params
            ph_agg = studies['ph']['agg'].best_params
            for k, v in ph_raw.items(): params[f"ph_{k}"] = v
            for k, v in ph_agg.items(): params[f"ph_{k}"] = v
        if 'temp' in studies:
            temp_raw = studies['temp']['raw'].best_params
            temp_agg = studies['temp']['agg'].best_params
            for k, v in temp_raw.items(): params[f"temp_{k}"] = v
            for k, v in temp_agg.items(): params[f"temp_{k}"] = v
            
    return params

# ==========================================
# EXTRACTION ENGINE
# ==========================================

class LikelihoodExtractor:
    def __init__(self, trace_path='tuning_data.csv', alpha=0.3, beta=0.10, beta_temp=0.01):
        self.trace_path = trace_path
        self.alpha = alpha
        self.beta = beta
        self.beta_temp = beta_temp
        
        # Extract disease metadata
        temp_sim = DiseaseSimulator(df_path=self.trace_path)
        self.disease_dict = temp_sim.diseases
        del temp_sim

    def count_hits_and_misses(self, det_dict, target_zones, t_start, is_healthy):
        """Evaluates raw/agg detections against ground truth to return TP, FN, FP, TN."""
        tp, fn, fp, tn = 0, 0, 0, 0
        triggered_zones = set(det_dict.keys())
        
        if is_healthy:
            fp = len(triggered_zones)
            tn = 20 - fp
        else:
            for tz in target_zones:
                if tz in triggered_zones:
                    raw_det = det_dict[tz]
                    t_det = min([t for t in raw_det.values() if t is not None]) if isinstance(raw_det, dict) else raw_det
                    if t_det >= t_start:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                else:
                    fn += 1
                    
            leakage = len([z for z in triggered_zones if z not in target_zones])
            fp += leakage
            tn = (20 - len(target_zones)) - leakage
            
        return tp, fn, fp, tn

    def worker(self, d_key, cohort_name, arch_type, params, is_healthy=False):
        """Runs a single simulation and extracts performance for BOTH raw and agg streams simultaneously."""
        simulator = DiseaseSimulator(df_path=self.trace_path)
        is_zscore = (arch_type == 'zscore')
        d_info = self.disease_dict.get(d_key, {})
        
        # 1. GENERATE DATA (Generalized vs Targeted routing)
        if cohort_name == 'generalized':
            if is_healthy:
                raw_df, agg_df, _ = simulator.run_single_disease(self.alpha, self.beta, 'systemic_fever_temp', safe_start_time=86400)
                t_start = 999999
                target_zones = []
                d_type = 'single'
                param = 'temp' 
            else:
                d_type = d_info['type']
                param = d_info.get('param', 'temp')
                if d_type == 'single':
                    raw_df, agg_df, t_start = simulator.run_single_disease(self.alpha, self.beta, d_key)
                else:
                    raw_df, agg_df, t_start = simulator.run_combined_disease(self.alpha, self.beta, d_key)
                target_zones = d_info['zones']
        else:
            # Targeted Cohorts force everything through Combined
            d_type = 'combined'
            if is_healthy:
                raw_df, agg_df, _ = simulator.run_combined_disease(self.alpha, self.beta, 'systemic_fever_temp', safe_start_time=86400)
                t_start = 999999
                target_zones = []
            else:
                raw_df, agg_df, t_start = simulator.run_combined_disease(self.alpha, self.beta, d_key)
                target_zones = d_info['zones']

        # 2. RUN DETECTOR
        if d_type == 'single':
            if is_zscore:
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, use_z_score=True,
                    k_raw_multiplier=params[f"{param}_k_raw_multiplier"],
                    k_agg_multiplier=params[f"{param}_k_agg_multiplier"],
                    raw_leakage=params[f"{param}_raw_leakage"],
                    agg_persistence=params[f"{param}_agg_persistence"]
                )
            else:
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, use_z_score=False,
                    raw_drift_threshold=params[f"{param}_raw_drift_threshold"],
                    raw_leakage=params[f"{param}_raw_leakage"],
                    agg_persistence=params[f"{param}_agg_persistence"],
                    agg_abs_drift=params[f"{param}_agg_abs_drift"]
                )
            _, det_raw = detector.detect_raw_stream(raw_df)
            _, det_agg = detector.detect_aggregated_stream(agg_df)
            
        else: # COMBINED
            if is_zscore:
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, beta_ph=self.beta, beta_temp=self.beta_temp, use_z_score=True,
                    k_raw_ph=params['k_raw_ph'], k_raw_temp=params['k_raw_temp'],
                    k_agg_ph=params['k_agg_ph'], k_agg_temp=params['k_agg_temp'],
                    raw_leakage=params['raw_leakage'], agg_persistence=params['agg_persistence']
                )
            else:
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, beta_ph=self.beta, beta_temp=self.beta_temp, use_z_score=False,
                    raw_drift_ph=params['raw_drift_ph'], raw_drift_temp=params['raw_drift_temp'],
                    raw_leakage=params['raw_leakage'], agg_persistence=params['agg_persistence'],
                    agg_abs_drift_ph=params['agg_abs_drift_ph'], agg_abs_drift_temp=params['agg_abs_drift_temp']
                )
            _, det_raw = detector.detect_combined_raw_stream(raw_df) 
            _, det_agg = detector.detect_combined_aggregated_stream(agg_df)

        # 3. COUNT HITS
        raw_tp, raw_fn, raw_fp, raw_tn = self.count_hits_and_misses(det_raw, target_zones, t_start, is_healthy)
        agg_tp, agg_fn, agg_fp, agg_tn = self.count_hits_and_misses(det_agg, target_zones, t_start, is_healthy)
        
        return (raw_tp, raw_fn, raw_fp, raw_tn, agg_tp, agg_fn, agg_fp, agg_tn)

    def extract_likelihoods(self, cohort_name, arch_type, iterations=3):
        diseases = COHORTS[cohort_name]
        params = load_optuna_params(cohort_name, arch_type)
        
        if not params:
            return None # Skip if tuning file is missing

        cores = max(1, multiprocessing.cpu_count() // 2)
        
        g_raw_tp, g_raw_fn, g_raw_fp, g_raw_tn = 0, 0, 0, 0
        g_agg_tp, g_agg_fn, g_agg_fp, g_agg_tn = 0, 0, 0, 0
        
        print(f"Extracting {cohort_name.upper()} | {arch_type.upper()}...")
        for i in range(iterations):
            res_diseased = Parallel(n_jobs=cores)(
                delayed(self.worker)(d, cohort_name, arch_type, params, is_healthy=False) for d in diseases
            )
            res_healthy = Parallel(n_jobs=cores)(
                delayed(self.worker)('healthy', cohort_name, arch_type, params, is_healthy=True) for _ in range(len(diseases))
            )
            
            for (r_tp, r_fn, r_fp, r_tn, a_tp, a_fn, a_fp, a_tn) in res_diseased + res_healthy:
                g_raw_tp += r_tp; g_raw_fn += r_fn; g_raw_fp += r_fp; g_raw_tn += r_tn
                g_agg_tp += a_tp; g_agg_fn += a_fn; g_agg_fp += a_fp; g_agg_tn += a_tn

        # Apply Laplace Smoothing
        # Raw
        raw_tpr = (g_raw_tp + 1) / ((g_raw_tp + g_raw_fn) + 2)
        raw_fpr = (g_raw_fp + 1) / ((g_raw_tn + g_raw_fp) + 2)
        # Aggregated
        agg_tpr = (g_agg_tp + 1) / ((g_agg_tp + g_agg_fn) + 2)
        agg_fpr = (g_agg_fp + 1) / ((g_agg_tn + g_agg_fp) + 2)
        
        return {
            'raw': {'tpr': round(raw_tpr, 4), 'fpr': round(raw_fpr, 4)},
            'agg': {'tpr': round(agg_tpr, 4), 'fpr': round(agg_fpr, 4)}
        }

if __name__ == "__main__":
    print("\n==========================================================================")
    print(" BAYESIAN CALIBRATION ROUTINE")
    print(" Dynamically loading Optuna results and extracting Priors from Tuning Data...")
    print("===========================================================================\n")
    
    # Use your specific tuning dataset csv here
    extractor = LikelihoodExtractor(trace_path='tuning_data.csv')
    
    final_likelihood_dict = {}
    
    for cohort in ['generalized', 'post_op', 'diabetes']:
        final_likelihood_dict[cohort] = {}
        for arch in ['static', 'zscore']:
            rates = extractor.extract_likelihoods(cohort, arch, iterations=3)
            if rates:
                final_likelihood_dict[cohort][arch] = rates
            
    # Write the dictionary to a text file
    output_filename = "calibrated_likelihoods.txt"
    
    with open(output_filename, "w") as f:
        f.write("CALIBRATED_LIKELIHOODS = \\\n")
        pprint.pprint(final_likelihood_dict, stream=f, sort_dicts=False)
        f.write("\n")
        
    print("\n==========================================================================")
    print(f" CALIBRATION COMPLETE! RESULTS SAVED TO: {os.path.abspath(output_filename)}")
    print("===========================================================================\n")