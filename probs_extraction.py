"""
A-Posteriori Likelihood Extractor for Spatial Naive Bayes Classification

This independent script evaluates the EXACT original Parquet TRAINING datasets 
used by Optuna. It does not generate new simulated data. It mathematically counts 
the exact hits, misses, and false alarms directly from the tuning dataset to extract 
empirical TPR and FPR, ensuring zero data leakage and perfect statistical alignment 
with the training phase.
"""

import pandas as pd
import joblib
import os
import multiprocessing
from joblib import Parallel, delayed
import pprint

from drift_detection import AnomalyDetector

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
# HELPER FUNCTIONS
# ==========================================

def load_optuna_params(cohort_name, arch_type):
    """Dynamically loads the best parameters from the serialized Optuna studies."""
    if cohort_name == 'generalized':
        if arch_type == 'static':
            filepath = 'tuning_results/completed_studies_BEST.pkl'
            if not os.path.exists(filepath): filepath = 'tuning_results/completed_studies.pkl'
        else:
            filepath = 'tuning_results/completed_zscore_studies.pkl'
    else:
        filepath = f'tuning_results/completed_static_{cohort_name}.pkl' if arch_type == 'static' else f'tuning_results/completed_zscore_{cohort_name}.pkl'

    if not os.path.exists(filepath):
        print(f"[WARNING] Tuning file not found: {filepath}. Skipping {cohort_name.upper()} | {arch_type.upper()}.")
        return None

    studies = joblib.load(filepath)
    params = {}
    
    if 'combined' in studies:
        params.update(studies['combined']['raw'].best_params)
        params.update(studies['combined']['agg'].best_params)
        
    if cohort_name == 'generalized':
        if 'ph' in studies:
            for k, v in studies['ph']['raw'].best_params.items(): params[f"ph_{k}"] = v
            for k, v in studies['ph']['agg'].best_params.items(): params[f"ph_{k}"] = v
        if 'temp' in studies:
            for k, v in studies['temp']['raw'].best_params.items(): params[f"temp_{k}"] = v
            for k, v in studies['temp']['agg'].best_params.items(): params[f"temp_{k}"] = v
            
    return params

def count_hits(det_times, target_zone, t_start):
    """Evaluates the footprint against ground-truth parameters from the Parquet row."""
    tp, fn, fp, tn = 0, 0, 0, 0
    triggered_zones = set(det_times.keys())
    
    is_healthy = (target_zone == -1)
    
    if is_healthy:
        fp = len(triggered_zones)
        tn = 20 - fp
    else:
        if target_zone in triggered_zones:
            raw_det = det_times[target_zone]
            t_det = min([t for t in raw_det.values() if t is not None]) if isinstance(raw_det, dict) else raw_det
            if t_det >= t_start:
                tp += 1
            else:
                fp += 1
                fn += 1
        else:
            fn += 1
            
        leakage = len([z for z in triggered_zones if z != target_zone])
        fp += leakage
        tn = 19 - leakage
        
    return tp, fn, fp, tn

def evaluate_group(group, arch_type, p_type, stream_type, params):
    """Standalone worker to run the AnomalyDetector on a single patient DataFrame slice."""
    first_row = group.iloc[0]
    target_zone = first_row['Target_Zone']
    t_start = first_row['True_t_start']
    
    alpha = 0.3
    beta_map = {'ph': 0.10, 'temp': 0.01, 'combined': 0.10}
    is_zscore = (arch_type == 'zscore')
    
    if p_type == 'ph' or p_type == 'temp':
        beta = beta_map[p_type]
        prefix = f"{p_type}_"
        raw_leakage = params[f"{prefix}raw_leakage"]
        agg_pers = params[f"{prefix}agg_persistence"]
        
        if is_zscore:
            detector = AnomalyDetector(
                alpha=alpha, beta=beta, use_z_score=True,
                k_raw_multiplier=params[f"{prefix}k_raw_multiplier"],
                k_agg_multiplier=params[f"{prefix}k_agg_multiplier"],
                raw_leakage=raw_leakage, agg_persistence=agg_pers
            )
        else:
            detector = AnomalyDetector(
                alpha=alpha, beta=beta, use_z_score=False,
                raw_drift_threshold=params[f"{prefix}raw_drift_threshold"],
                raw_leakage=raw_leakage, agg_persistence=agg_pers,
                agg_abs_drift=params[f"{prefix}agg_abs_drift"]
            )
            
        if stream_type == 'raw':
            _, det_times = detector.detect_raw_stream(group)
        else:
            _, det_times = detector.detect_aggregated_stream(group)
            
    else: # COMBINED
        beta = beta_map['combined']
        raw_leakage = params["raw_leakage"]
        agg_pers = params["agg_persistence"]
        
        if is_zscore:
            detector = AnomalyDetector(
                alpha=alpha, beta=beta, beta_ph=beta_map['ph'], beta_temp=beta_map['temp'],
                use_z_score=True,
                k_raw_ph=params["k_raw_ph"], k_raw_temp=params["k_raw_temp"],
                k_agg_ph=params["k_agg_ph"], k_agg_temp=params["k_agg_temp"],
                raw_leakage=raw_leakage, agg_persistence=agg_pers
            )
        else:
            detector = AnomalyDetector(
                alpha=alpha, beta=beta, beta_ph=beta_map['ph'], beta_temp=beta_map['temp'],
                use_z_score=False,
                raw_drift_ph=params["raw_drift_ph"], raw_drift_temp=params["raw_drift_temp"],
                raw_leakage=raw_leakage, agg_persistence=agg_pers,
                agg_abs_drift_ph=params["agg_abs_drift_ph"], agg_abs_drift_temp=params["agg_abs_drift_temp"]
            )
            
        if stream_type == 'raw':
            _, det_times = detector.detect_combined_raw_stream(group)
        else:
            _, det_times = detector.detect_combined_aggregated_stream(group)
            
    return count_hits(det_times, target_zone, t_start)

# ==========================================
# EXTRACTION ENGINE
# ==========================================

class DatasetLikelihoodExtractor:
    def __init__(self, data_dir='datasets'):
        self.data_dir = data_dir

    def extract_likelihoods(self, cohort_name, arch_type):
        params = load_optuna_params(cohort_name, arch_type)
        if not params: return None
        
        target_diseases = COHORTS[cohort_name]
        p_types = ['ph', 'temp', 'combined'] if cohort_name == 'generalized' else ['combined']
        streams = ['raw', 'agg']
        
        results = {'raw': {'tp':0, 'fn':0, 'fp':0, 'tn':0}, 'agg': {'tp':0, 'fn':0, 'fp':0, 'tn':0}}
        
        print(f"\nExtracting {cohort_name.upper()} | {arch_type.upper()}...")
        
        for stream in streams:
            for p_type in p_types:
                file_path = os.path.join(self.data_dir, f"tuning_dataset_{p_type}_{stream}.parquet")
                if not os.path.exists(file_path):
                    print(f"  [!] Missing dataset: {file_path}. Skipping.")
                    continue
                    
                df = pd.read_parquet(file_path)
                diseased_sims, healthy_sims = [], []
                
                # Filter rows by the current Cohort
                for sim_id, group in df.groupby('Simulation_ID'):
                    label = group.iloc[0].get('Disease_Label', 'healthy')
                    if label in target_diseases:
                        diseased_sims.append(group)
                    elif label == 'healthy':
                        healthy_sims.append(group)
                        
                # Replicate the exact balancing used in your Targeted Parameter Tuning script
                if cohort_name == 'generalized':
                    sims_to_run = diseased_sims + healthy_sims
                else:
                    num_diseased = len(diseased_sims)
                    sims_to_run = diseased_sims + healthy_sims[:num_diseased]
                    
                print(f"  -> Processing {len(sims_to_run)} simulations from {p_type.upper()} {stream.upper()}...")
                
                cores = max(1, multiprocessing.cpu_count() // 2)
                res = Parallel(n_jobs=cores)(
                    delayed(evaluate_group)(group, arch_type, p_type, stream, params) for group in sims_to_run
                )
                
                for tp, fn, fp, tn in res:
                    results[stream]['tp'] += tp
                    results[stream]['fn'] += fn
                    results[stream]['fp'] += fp
                    results[stream]['tn'] += tn
                    
        # Apply Laplace Smoothing to the Final Aggregated Counts
        final_rates = {}
        for stream in streams:
            tp, fn = results[stream]['tp'], results[stream]['fn']
            fp, tn = results[stream]['fp'], results[stream]['tn']
            
            actual_diseased = tp + fn
            actual_healthy = tn + fp
            
            if actual_diseased == 0: actual_diseased = 1
            if actual_healthy == 0: actual_healthy = 1
            
            tpr = (tp + 1) / (actual_diseased + 2)
            fpr = (fp + 1) / (actual_healthy + 2)
            
            final_rates[stream] = {'tpr': round(tpr, 5), 'fpr': round(fpr, 5)}
            
        return final_rates

if __name__ == "__main__":
    print("\n==========================================================================")
    print(" BAYESIAN CALIBRATION ROUTINE (NATIVE DATASET EXTRACTION)")
    print(" Dynamically loading Optuna results and grading original Parquet datasets...")
    print("===========================================================================\n")
    
    extractor = DatasetLikelihoodExtractor(data_dir='datasets')
    final_likelihood_dict = {}
    
    for cohort in ['generalized', 'post_op', 'diabetes']:
        final_likelihood_dict[cohort] = {}
        for arch in ['static', 'zscore']:
            rates = extractor.extract_likelihoods(cohort, arch)
            if rates:
                final_likelihood_dict[cohort][arch] = rates
            
    output_filename = "calibrated_likelihoods.txt"
    with open(output_filename, "w") as f:
        f.write("CALIBRATED_LIKELIHOODS = \\\n")
        pprint.pprint(final_likelihood_dict, stream=f, sort_dicts=False)
        f.write("\n")
        
    print("\n==========================================================================")
    print(f" CALIBRATION COMPLETE! RESULTS SAVED TO: {os.path.abspath(output_filename)}")
    print("===========================================================================\n")