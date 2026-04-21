"""
Clinical Evaluation and Testing Module (Multi-Run Parallelized)

This module provides a robust framework for validating optimized Kalman filters 
against stochastic physiological simulations. It supports:
1. Parallelized execution of diseased and healthy scenarios.
2. Multi-run statistical aggregation to calculate Mean +/- Standard Deviation.
3. Strict clinical categorization of alerts (True Positives, False Positives, Spatial Error).
4. Dual-Architecture Support: Evaluates both Static Thresholds and Adaptive Z-Scoring.
5. Disease-Level Granularity: Breaks down performance metrics by specific pathology.
"""

from diseases import DiseaseSimulator
from drift_detection import AnomalyDetector
from plots import AnomalyVisualizer
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

# ==========================================
# CLINICAL TRIAL RUNNER
# ==========================================

class ClinicalTrialRunner:
    """
    Executes head-to-head simulations of specific diseases or healthy baselines.
    Features 'Smart Reporting' to automatically categorize alerts as True Positives, 
    False Positives (premature/healthy), or Spatial Errors (leakage).
    """
    def __init__(self, alpha=0.3, beta=0.10, beta_ph=0.10, beta_temp=0.01, target_conf=70.0):
        """
        Initializes the ClinicalTrialRunner with hardware error profiles.
        
        @param alpha (float): Localization error probability.
        @param beta (float): Highest baseline measurement error (used for binomial confidence).
        @param beta_ph (float): Specific measurement error for the pH sensor.
        @param beta_temp (float): Specific measurement error for the Temperature sensor.
        @param target_conf (float): The statistical confidence threshold.
        """
        self.alpha = alpha
        self.beta = beta  
        self.beta_ph = beta_ph
        self.beta_temp = beta_temp
        self.target_conf = target_conf
        
        self.simulator = DiseaseSimulator()
        self.visualizer = AnomalyVisualizer()

    def _print_alerts(self, alert_dict, stream_name, t_start, is_healthy, target_zones=None):
        """
        Formats and prints detection results, applying clinical logic to categorize 
        each alert based on its spatial accuracy and temporal onset.
        
        @param alert_dict (dict): Dictionary mapping zones to detection timestamps.
        @param stream_name (str): Identifier for the data stream (e.g., 'RAW STREAM').
        @param t_start (float): The ground-truth start time of the disease (in seconds).
        @param is_healthy (bool): Flag indicating if the scenario is a healthy control.
        @param target_zones (list, optional): List of ground-truth diseased zones.
        """
        print(f"\n[{stream_name} ALERTS]")
        
        if not alert_dict:
            if is_healthy:
                print("  No anomalies detected. (Expected Healthy Behavior)")
            else:
                print("  No anomalies detected. (!! FALSE NEGATIVE / MISSED !!)")
            return
            
        for zone_id, det_val in alert_dict.items():
            # 1. Determine spatial validity
            is_target = (target_zones is not None and zone_id in target_zones)
            
            # 2. Extract detection time safely
            if isinstance(det_val, dict):
                valid_times = [t for t in det_val.values() if t is not None]
                t_det = min(valid_times) if valid_times else None
                trigger_param = "pH" if det_val.get('ph') == t_det else "Temp"
            else:
                t_det = det_val
                trigger_param = "Sensor"

            if t_det is None: 
                continue

            # 3. Apply temporal/clinical labeling
            time_str = f"Hour {t_det/3600:.2f}"
            
            if is_healthy:
                print(f"  --> ZONE {zone_id} ALERT: [FALSE POSITIVE] at {time_str} by {trigger_param}")
            elif is_target:
                delay = t_det - t_start
                if delay < 0:
                    status = f"FP/Early Detection ({-delay/60:.1f} min early)"
                else:
                    status = f"Delay: {delay/60:.1f} min"
                print(f"  --> ZONE {zone_id} ALERT: (TRUE POSITIVE) at {time_str} | {status}")
            else:
                print(f"  --> ZONE {zone_id} ALERT: [SPATIAL ERROR / LEAKAGE] at {time_str} by {trigger_param}")

    def run_single_parameter_trial(self, disease_key, is_healthy=False, 
                                   raw_drift=2.5, agg_pers=3, agg_drift=0.10, safe_start=1800):
        """
        Executes a localized trial for a single-parameter disease (pH or Temp).
        
        @param disease_key (str): The pathology to simulate.
        @param is_healthy (bool): Flag to run as a healthy baseline.
        @param raw_drift/agg_pers/agg_drift (float/int): Filter thresholds.
        @param safe_start (float): Minimum onset time.
        """
        target_zones = self.simulator.diseases[disease_key]['zones']
        actual_start = 86400 if is_healthy else safe_start
        title_prefix = "Healthy Baseline" if is_healthy else f"Detection Speed ({disease_key.upper()})"
        
        print(f"\n{'='*60}")
        print(f"--- RUNNING TRIAL: {disease_key.upper()} (Target Zones: {target_zones}) ---")
        print(f"{'='*60}")

        raw_df, agg_df, t_start = self.simulator.run_single_disease(
            alpha=self.alpha, beta=self.beta, disease_key=disease_key,
            target_confidence=self.target_conf, safe_start_time=actual_start
        )

        if agg_df is None or agg_df.empty:
            print("Simulation failed or returned empty data.")
            return

        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, target_confidence=self.target_conf,
            raw_drift_threshold=raw_drift, agg_persistence=agg_pers, agg_abs_drift=agg_drift
        )

        processed_raw_df, all_raw_alerts = detector.detect_raw_stream(raw_df)
        processed_agg_df, all_agg_alerts = detector.detect_aggregated_stream(agg_df)

        print("\n--- Detection Results ---")
        if not is_healthy: 
            print(f"Clinical Onset: Hour {t_start/3600:.2f}")
        
        self._print_alerts(all_raw_alerts, "RAW STREAM", t_start, is_healthy, target_zones=target_zones)
        self._print_alerts(all_agg_alerts, "AGGREGATED STREAM", t_start, is_healthy, target_zones=target_zones)

        self.visualizer.plot_comparison(
            raw_df=processed_raw_df, 
            agg_df=processed_agg_df, 
            det_raw=all_raw_alerts, 
            det_agg=all_agg_alerts, 
            target_zones=target_zones,
            title=f"{title_prefix}"
        )

    def run_combined_parameter_trial(self, disease_key, is_healthy=False, 
                                      raw_drift_ph=1.5, raw_drift_temp=3.5, 
                                      agg_pers=3, agg_drift_ph=0.10, agg_drift_temp=0.5, 
                                      safe_start=1800):
        """
        Executes a trial for dual-parameter diseases (pH AND Temp).
        
        @param disease_key (str): The combined pathology identifier.
        @param is_healthy (bool): Flag to run as a healthy control.
        """
        target_zones = self.simulator.diseases[disease_key]['zones']
        actual_start = 86400 if is_healthy else safe_start
        title_prefix = "Healthy Combined Baseline" if is_healthy else f"Detection Speed ({disease_key.upper()})"
        
        print(f"\n{'='*60}")
        print(f"--- RUNNING COMBINED TRIAL: {disease_key.upper()} (Target Zones: {target_zones}) ---")
        print(f"{'='*60}")

        raw_df, agg_df, t_start = self.simulator.run_combined_disease(
            alpha=self.alpha, beta=self.beta, disease_key=disease_key,
            target_confidence=self.target_conf, safe_start_time=actual_start
        )

        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, beta_ph=self.beta_ph, beta_temp=self.beta_temp, 
            target_confidence=self.target_conf,
            raw_drift_ph=raw_drift_ph, raw_drift_temp=raw_drift_temp,
            agg_persistence=agg_pers, agg_abs_drift_ph=agg_drift_ph, agg_abs_drift_temp=agg_drift_temp
        )

        processed_raw_df, all_raw_alerts = detector.detect_combined_raw_stream(raw_df)
        processed_agg_df, all_agg_alerts = detector.detect_combined_aggregated_stream(agg_df)

        print("\n--- Detection Results ---")
        if not is_healthy: 
            print(f"Clinical Onset: Hour {t_start/3600:.2f}")
        
        self._print_alerts(all_raw_alerts, "RAW COMBINED", t_start, is_healthy, target_zones=target_zones)
        self._print_alerts(all_agg_alerts, "AGGREGATED COMBINED", t_start, is_healthy, target_zones=target_zones)

        self.visualizer.plot_combined_comparison(
            raw_df=processed_raw_df, 
            agg_df=processed_agg_df, 
            det_raw=all_raw_alerts, 
            det_agg=all_agg_alerts, 
            target_zones=target_zones,
            title=f"{title_prefix}"
        )

# ==========================================
# PERFORMANCE EVALUATOR
# ==========================================

class FinalPerformanceEvaluator:
    """
    Orchestrates the clinical scoring of the nanonetwork. 
    Distinguishes between Patient-Level survival metrics and Zone-Level spatial accuracy.
    Now includes granular, pathology-specific evaluation while retaining single-run tests.
    """
    def __init__(self, total_zones=20):
        """
        @param total_zones (int): Total anatomical regions monitored.
        """
        self.total_zones = total_zones

    def evaluate_test_suite(self, results_list, stream_name):
        """
        Calculates and prints metrics for a single iteration of 30 patients.
        """
        stats = self._get_scalars(results_list)
        
        print(f"\n{'='*50}")
        print(f" {stream_name} - FINAL CLINICAL REPORT")
        print(f"{'='*50}")
        print(f" -> Patient-Level Sensitivity: {stats['patient_recall'] * 100:.1f}%")
        print(f" -> Zone-Level Precision: {stats['zone_precision'] * 100:.1f}%")
        print(f" -> Mean Detection Speed: {stats['mean_npl'] * 100:.1f}% of Disease Peak")
        print(f"{'='*50}")

    def evaluate_multi_run(self, all_runs_results, stream_name):
        """
        Calculates and prints cross-iteration statistics (Mean +/- Std Dev).
        
        @param all_runs_results (list): Collection of result lists from multiple iterations.
        @param stream_name (str): The identifier for the data stream evaluated.
        """
        metrics = {
            'patient_recall': [], 'zone_recall': [], 
            'zone_precision': [], 'mean_npl': [],
            'patient_fp': [], 'zone_fp': [], 'system_accuracy': []
        }

        for results_list in all_runs_results:
            stats = self._get_scalars(results_list)
            for k, v in stats.items():
                metrics[k].append(v)

        print(f"\n{'='*60}")
        print(f" {stream_name} - FINAL MULTI-RUN STATISTICAL REPORT")
        print(f"{'='*60}")
        
        for k, v in metrics.items():
            mean_val = np.mean(v)
            std_val = np.std(v)
            
            display_name = k.replace('_', ' ').title()
            unit = "%" if any(x in k for x in ["accuracy", "recall", "precision", "npl"]) else "counts"
            
            mult = 100.0 if unit == "%" else 1.0
            print(f" -> {display_name}: {mean_val*mult:.2f}{unit} (+/- {std_val*mult:.2f})")
        
        print(f"{'='*60}\n")
        
    def evaluate_disease_level_multi_run(self, all_runs_results, stream_name):
        """
        Groups results by specific disease key and calculates metrics per pathology.
        
        @param all_runs_results (list): Collection of result lists from multiple iterations.
        @param stream_name (str): The identifier for the data stream evaluated.
        """
        print(f"\n{'-'*70}")
        print(f" {stream_name} - DISEASE-LEVEL BREAKDOWN")
        print(f"{'-'*70}")

        # Extract all unique disease keys present in the results
        all_keys = set()
        for run in all_runs_results:
            for res in run:
                all_keys.add(res.get('Disease_Key', 'Unknown'))
        
        # Sort for clean display (put healthy at the end)
        disease_keys = sorted([k for k in all_keys if k != 'healthy'])
        if 'healthy' in all_keys:
            disease_keys.append('healthy')

        # Evaluate each disease independently across all runs
        for d_key in disease_keys:
            metrics = {
                'patient_recall': [], 'zone_recall': [], 
                'zone_precision': [], 'mean_npl': [],
                'patient_fp': [], 'zone_fp': [], 'system_accuracy': []
            }

            for run_results in all_runs_results:
                # Isolate the patients corresponding strictly to this disease
                disease_subset = [res for res in run_results if res.get('Disease_Key') == d_key]
                if disease_subset:
                    stats = self._get_scalars(disease_subset)
                    for k, v in stats.items():
                        metrics[k].append(v)

            print(f"\n   [ PATHOLOGY: {d_key.upper()} ]")
            
            # Healthy patients don't have Recall or NPL, they only have False Alarms
            if d_key == 'healthy':
                m_pat_fp, s_pat_fp = np.mean(metrics['patient_fp']), np.std(metrics['patient_fp'])
                m_zon_fp, s_zon_fp = np.mean(metrics['zone_fp']), np.std(metrics['zone_fp'])
                m_acc, s_acc = np.mean(metrics['system_accuracy']), np.std(metrics['system_accuracy'])
                
                print(f"    -> False Alarms (Patients) : {m_pat_fp:.2f} (+/- {s_pat_fp:.2f})")
                print(f"    -> Spatial Leakage (Zones) : {m_zon_fp:.2f} (+/- {s_zon_fp:.2f})")
                print(f"    -> System Accuracy         : {m_acc*100.0:>6.2f}% (+/- {s_acc*100.0:.2f})")
            else:
                # For diseases, print the full diagnostic breakdown
                for k, v in metrics.items():
                    # Skip printing FP for individual diseases unless desired, focus on detection efficacy
                    if k in ['patient_fp', 'zone_fp']: continue 
                    
                    mean_val = np.mean(v)
                    std_val = np.std(v)
                    display_name = k.replace('_', ' ').title()
                    
                    # Ensure 'accuracy' is recognized here as well
                    unit = "%" if any(x in k for x in ["recall", "precision", "npl", "accuracy"]) else " counts"
                    mult = 100.0 if unit == "%" else 1.0
                    
                    print(f"    -> {display_name:<25}: {mean_val*mult:>6.2f}{unit} (+/- {std_val*mult:.2f})")

    def _get_scalars(self, results_list):
        """
        Internal mathematical engine to extract scalar metrics from a list of results.
        """
        patient_tp, patient_fp, patient_fn = 0, 0, 0
        zone_tp, zone_fp, zone_fn = 0, 0, 0
        zone_tn = 0
        total_eval_zones = 0 
        npl_list = []

        for res in results_list:
            target_zones = res['Target_Zones']
            t_start = res['True_t_start']
            t_peak = res['True_t_peak']
            det_times = res['Detection_Times']  

            # Scenario 1: Healthy Controls
            if not target_zones or target_zones == [-1]:  
                fp_count = len(det_times)
                patient_fp += 1 if fp_count > 0 else 0
                zone_fp += fp_count
                zone_tn += (20 - fp_count)  # The rest were correctly identified as healthy
                total_eval_zones += 20
                continue

            # Scenario 2: Diseased Patients
            prog_window = max(t_peak - t_start, 1)
            patient_hit = False
            
            for tz in target_zones:
                if tz in det_times:
                    raw_det = det_times[tz]
                    t_det = min([t for t in raw_det.values() if t is not None]) if isinstance(raw_det, dict) else raw_det
                    
                    if t_det >= t_start:
                        zone_tp += 1
                        patient_hit = True
                        npl_list.append(min(1.0, (t_det - t_start) / prog_window))
                    else:
                        zone_fp += 1 
                        zone_fn += 1 
                else:
                    zone_fn += 1

            if patient_hit: 
                patient_tp += 1
            else: 
                patient_fn += 1

            # Calculate spatial leakage (False Positives in healthy zones)
            leakage = len([z for z in det_times.keys() if z not in target_zones])
            zone_fp += leakage
            
            # Calculate True Negatives (Healthy zones that were correctly ignored)
            # Total zones (20) minus the diseased zones, minus the ones we falsely triggered on
            zone_tn += (20 - len(target_zones)) - leakage
            total_eval_zones += 20
            
            if leakage > 0 and not patient_hit: 
                patient_fp += 1 

        return {
            'patient_recall': patient_tp / (patient_tp + patient_fn) if (patient_tp + patient_fn) > 0 else 0,
            'zone_recall': zone_tp / (zone_tp + zone_fn) if (zone_tp + zone_fn) > 0 else 0,
            'zone_precision': zone_tp / (zone_tp + zone_fp) if (zone_tp + zone_fp) > 0 else 0,
            'system_accuracy': (zone_tp + zone_tn) / total_eval_zones if total_eval_zones > 0 else 0, # NEW METRIC
            'mean_npl': np.mean(npl_list) if npl_list else 1.0,
            'patient_fp': patient_fp,
            'zone_fp': zone_fp
        }
# ==========================================
# COMPREHENSIVE TEST SUITE (PARALLELIZED)
# ==========================================

class ComprehensiveTestSuite:
    """
    Benchmark engine designed for rapid execution of the clinical hold-out test.
    Utilizes multiprocessing to run stochastic simulations in parallel and 
    aggregates multi-run statistics. Now supports Z-score architecture testing.
    """
    def __init__(self, optimized_params, use_z_score=False, test_trace_path='position_data.csv', alpha=0.3, beta=0.10):
        """
        Initializes the test suite.
        
        @param optimized_params (dict): Hyperparameters from Optuna.
        @param use_z_score (bool): If True, tests the Adaptive Variance-Scaled architecture.
        @param test_trace_path (str): The hold-out CSV trace.
        """
        self.alpha = alpha
        self.beta = beta
        self.params = optimized_params
        self.use_z_score = use_z_score
        self.test_trace_path = test_trace_path
        self.evaluator = FinalPerformanceEvaluator()
        
        # Meta-data extraction
        temp_sim = DiseaseSimulator(df_path=self.test_trace_path)
        self.diseases = list(temp_sim.diseases.keys())
        self.disease_dict = temp_sim.diseases
        del temp_sim

    def _run_one_full_iteration(self, iter_idx, cores):
        """
        Executes one full diagnostic cycle (30 patients).
        
        @param iter_idx (int): Current iteration counter.
        @param cores (int): CPU core count for worker allocation.
        @return tuple: (Raw iteration results, Aggregated iteration results).
        """
        print(f" --- Starting Iteration {iter_idx+1} ---")
        
        diseased_res = Parallel(n_jobs=cores)(delayed(self._worker_diseased)(d) for d in self.diseases)
        healthy_res = Parallel(n_jobs=cores)(delayed(self._worker_healthy)(i) for i in range(15))
        
        raw_iter = [r[0] for r in diseased_res] + [r[0] for r in healthy_res]
        agg_iter = [r[1] for r in diseased_res] + [r[1] for r in healthy_res]
        return raw_iter, agg_iter

    def _worker_diseased(self, d_key):
        """Isolated worker process for diseased simulations."""
        simulator = DiseaseSimulator(df_path=self.test_trace_path)
        d_info = self.disease_dict[d_key]
        
        if d_info['type'] == 'single':
            raw_df, agg_df, t_start = simulator.run_single_disease(self.alpha, self.beta, d_key)
            
            # --- Z-SCORE VS STATIC BRANCHING ---
            if self.use_z_score:
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, use_z_score=True,
                    k_raw_multiplier=self.params[f"{d_info['param']}_k_raw"],
                    k_agg_multiplier=self.params[f"{d_info['param']}_k_agg"],
                    raw_leakage=self.params[f"{d_info['param']}_raw_leakage"],
                    agg_persistence=self.params[f"{d_info['param']}_agg_pers"]
                )
            else:
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, use_z_score=False,
                    raw_drift_threshold=self.params[f"{d_info['param']}_raw_drift"],
                    raw_leakage=self.params[f"{d_info['param']}_raw_leakage"],
                    agg_persistence=self.params[f"{d_info['param']}_agg_pers"],
                    agg_abs_drift=self.params[f"{d_info['param']}_agg_drift"]
                )
                
            _, det_raw = detector.detect_raw_stream(raw_df)
            _, det_agg = detector.detect_aggregated_stream(agg_df)
            
        else: 
            raw_df, agg_df, t_start = simulator.run_combined_disease(self.alpha, self.beta, d_key)
            
            # --- Z-SCORE VS STATIC BRANCHING ---
            if self.use_z_score:
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, beta_ph=0.10, beta_temp=0.01, use_z_score=True,
                    k_raw_ph=self.params['comb_k_raw_ph'],
                    k_raw_temp=self.params['comb_k_raw_temp'],
                    k_agg_ph=self.params['comb_k_agg_ph'],
                    k_agg_temp=self.params['comb_k_agg_temp'],
                    raw_leakage=self.params['comb_raw_leakage'],
                    agg_persistence=self.params['comb_agg_pers']
                )
            else:
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, beta_ph=0.10, beta_temp=0.01, use_z_score=False,
                    raw_drift_ph=self.params['comb_raw_drift_ph'],
                    raw_drift_temp=self.params['comb_raw_drift_temp'],
                    raw_leakage=self.params['comb_raw_leakage'],
                    agg_persistence=self.params['comb_agg_pers'],
                    agg_abs_drift_ph=self.params['comb_agg_drift_ph'],
                    agg_abs_drift_temp=self.params['comb_agg_drift_temp']
                )
                
            _, det_raw = detector.detect_combined_raw_stream(raw_df) 
            _, det_agg = detector.detect_combined_aggregated_stream(agg_df)

        res_base = {'Disease_Key': d_key, 'Target_Zones': d_info['zones'], 'True_t_start': t_start, 'True_t_peak': t_start + d_info['progression_time']}
        return {**res_base, 'Detection_Times': det_raw}, {**res_base, 'Detection_Times': det_agg}

    def _worker_healthy(self, run_id):
        """Isolated worker process for healthy control simulations."""
        simulator = DiseaseSimulator(df_path=self.test_trace_path)
        raw_df, agg_df, _ = simulator.run_single_disease(self.alpha, self.beta, 'systemic_fever_temp', safe_start_time=86400)
        
        # We use the pH parameters to rigorously stress-test the healthy baseline
        if self.use_z_score:
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.beta, use_z_score=True,
                k_raw_multiplier=self.params['ph_k_raw'],
                k_agg_multiplier=self.params['ph_k_agg'],
                raw_leakage=self.params['ph_raw_leakage'],
                agg_persistence=self.params['ph_agg_pers']
            )
        else:
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.beta, use_z_score=False,
                raw_drift_threshold=self.params['ph_raw_drift'],
                raw_leakage=self.params['ph_raw_leakage'],
                agg_persistence=self.params['ph_agg_pers'],
                agg_abs_drift=self.params['ph_agg_drift']
            )
            
        _, det_raw = detector.detect_raw_stream(raw_df)
        _, det_agg = detector.detect_aggregated_stream(agg_df)

        res_base = {'Disease_Key': 'healthy', 'Target_Zones': [-1], 'True_t_start': 999999, 'True_t_peak': 999999}
        return {**res_base, 'Detection_Times': det_raw}, {**res_base, 'Detection_Times': det_agg}

    def run_full_suite(self, num_iterations=5):
        """
        Orchestrates the entire benchmarking pipeline across multiple stochastic iterations.
        
        @param num_iterations (int): Total number of suite cycles to execute.
        """
        cores = max(1, multiprocessing.cpu_count() // 2)
        arch_type = "Z-SCORE" if self.use_z_score else "STATIC THRESHOLD"
        print(f"Starting {arch_type} Multi-Run Parallel Test Suite ({num_iterations} iterations) on {cores} cores...")
        
        all_raw_runs, all_agg_runs = [], []

        for i in range(num_iterations):
            raw_iter, agg_iter = self._run_one_full_iteration(i, cores)
            all_raw_runs.append(raw_iter)
            all_agg_runs.append(agg_iter)
        
        print("\nAll iterations finished. Calculating cross-run statistics...")
        
        # 1. Global Metrics
        self.evaluator.evaluate_multi_run(all_raw_runs, f"RAW STREAM ({arch_type})")
        self.evaluator.evaluate_multi_run(all_agg_runs, f"AGGREGATED STREAM ({arch_type})")
        
        # 2. Disease-Level Metrics
        self.evaluator.evaluate_disease_level_multi_run(all_raw_runs, f"RAW STREAM ({arch_type})")
        self.evaluator.evaluate_disease_level_multi_run(all_agg_runs, f"AGGREGATED STREAM ({arch_type})")

# ==========================================
# TARGETED COHORT TEST SUITE (PARALLELIZED)
# ==========================================

# Define the clinical cohorts for targeted evaluation
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

class TargetedCohortTestSuite:
    """
    Benchmark engine designed specifically for Targeted Cohort evaluation 
    (Post-Operative or Diabetes). It forces all scenarios (single or dual parameter) 
    through the Dual-Sensor (Combined) Simulator and evaluates them strictly 
    against the targeted Combined Kalman Filter parameters.
    """
    def __init__(self, cohort_name, optimized_combined_params, use_z_score=False, test_trace_path='position_data.csv', alpha=0.3, beta=0.10, beta_temp=0.01):
        """
        Initializes the targeted suite.
        
        @param cohort_name (str): 'post_op' or 'diabetes'.
        @param optimized_combined_params (dict): The optimal parameters for the combined filter.
        @param use_z_score (bool): If True, tests the Adaptive Variance-Scaled architecture.
        """
        self.cohort_name = cohort_name
        self.target_diseases = COHORTS[cohort_name]
        self.params = optimized_combined_params
        self.use_z_score = use_z_score
        self.test_trace_path = test_trace_path
        self.alpha = alpha
        self.beta = beta
        self.beta_temp = beta_temp
        self.evaluator = FinalPerformanceEvaluator()
        
        temp_sim = DiseaseSimulator(df_path=self.test_trace_path)
        self.disease_dict = temp_sim.diseases
        del temp_sim

    def _run_one_targeted_iteration(self, iter_idx, cores):
        """Executes one full diagnostic cycle for the targeted cohort."""
        print(f" --- Starting Targeted Iteration {iter_idx+1} ---")
        
        diseased_res = Parallel(n_jobs=cores)(delayed(self._worker_scenario)(d) for d in self.target_diseases)
        
        # Spawn an EXACT EQUAL number of healthy controls to keep FP rates mathematically balanced
        num_controls = len(self.target_diseases)
        healthy_res = Parallel(n_jobs=cores)(delayed(self._worker_scenario)('healthy', is_healthy=True) for _ in range(num_controls))
        
        raw_iter = [r[0] for r in diseased_res] + [r[0] for r in healthy_res]
        agg_iter = [r[1] for r in diseased_res] + [r[1] for r in healthy_res]
        return raw_iter, agg_iter

    def _worker_scenario(self, d_key, is_healthy=False):
        simulator = DiseaseSimulator(df_path=self.test_trace_path)
        
        # 1. FORCE THROUGH THE COMBINED SIMULATOR (Even if it's a single-parameter disease)
        if is_healthy:
            raw_df, agg_df, _ = simulator.run_combined_disease(self.alpha, self.beta, 'systemic_fever_temp', safe_start_time=86400)
            t_start, t_peak = 999999, 999999
            target_zones = [-1]
        else:
            raw_df, agg_df, t_start = simulator.run_combined_disease(self.alpha, self.beta, d_key)
            t_peak = t_start + self.disease_dict[d_key]['progression_time']
            target_zones = self.disease_dict[d_key]['zones']

        # 2. INITIALIZE THE COMBINED FILTER
        if self.use_z_score:
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.beta, beta_ph=self.beta, beta_temp=self.beta_temp, 
                use_z_score=True,
                k_raw_ph=self.params['comb_k_raw_ph'],
                k_raw_temp=self.params['comb_k_raw_temp'],
                k_agg_ph=self.params['comb_k_agg_ph'],
                k_agg_temp=self.params['comb_k_agg_temp'],
                raw_leakage=self.params['comb_raw_leakage'],
                agg_persistence=self.params['comb_agg_pers']
            )
        else:
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.beta, beta_ph=self.beta, beta_temp=self.beta_temp, 
                use_z_score=False,
                raw_drift_ph=self.params['comb_raw_drift_ph'],
                raw_drift_temp=self.params['comb_raw_drift_temp'],
                raw_leakage=self.params['comb_raw_leakage'],
                agg_persistence=self.params['comb_agg_pers'],
                agg_abs_drift_ph=self.params['comb_agg_drift_ph'],
                agg_abs_drift_temp=self.params['comb_agg_drift_temp']
            )
            
        # 3. RUN DETECTION
        _, det_raw = detector.detect_combined_raw_stream(raw_df) 
        _, det_agg = detector.detect_combined_aggregated_stream(agg_df)

        label = 'healthy' if is_healthy else d_key
        res_base = {'Disease_Key': label, 'Target_Zones': target_zones, 'True_t_start': t_start, 'True_t_peak': t_peak}
        return {**res_base, 'Detection_Times': det_raw}, {**res_base, 'Detection_Times': det_agg}

    def run_targeted_suite(self, num_iterations=10):
        cores = max(1, multiprocessing.cpu_count() // 2)
        arch_type = "Z-SCORE" if self.use_z_score else "STATIC THRESHOLD"
        
        print(f"\nStarting {arch_type} TARGETED COMBINED Test Suite for {self.cohort_name.upper()}...")
        print(f" -> {len(self.target_diseases)} Targeted Pathologies + {len(self.target_diseases)} Matched Healthy Controls")
        print(f" -> Executing {num_iterations} Stochastic Iterations on {cores} cores...")
        
        all_raw_runs, all_agg_runs = [], []

        for i in range(num_iterations):
            raw_iter, agg_iter = self._run_one_targeted_iteration(i, cores)
            all_raw_runs.append(raw_iter)
            all_agg_runs.append(agg_iter)
        
        print(f"\nAll {num_iterations} iterations finished. Evaluating Data...")
        
        # 1. Output Global Summaries for the Cohort
        self.evaluator.evaluate_multi_run(all_raw_runs, f"TARGETED RAW ({self.cohort_name.upper()} | {arch_type})")
        self.evaluator.evaluate_multi_run(all_agg_runs, f"TARGETED AGGREGATED ({self.cohort_name.upper()} | {arch_type})")
        
        # 2. Output Specific Disease Breakdowns
        self.evaluator.evaluate_disease_level_multi_run(all_raw_runs, f"TARGETED RAW ({self.cohort_name.upper()} | {arch_type})")
        self.evaluator.evaluate_disease_level_multi_run(all_agg_runs, f"TARGETED AGGREGATED ({self.cohort_name.upper()} | {arch_type})")