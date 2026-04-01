"""
Clinical Evaluation and Testing Module

This module provides the framework for rigorously validating the optimized Kalman 
filters against hold-out datasets. It features tools for single head-to-head 
clinical trials, a comprehensive parallelized test suite for rapid benchmarking, 
and a stringent clinical evaluator that categorizes detections into Patient-Level 
(survival) and Zone-Level (spatial mapping) metrics.
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
        @param beta (float): The highest baseline measurement error (used for generalized confidence).
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
            # 1. Determine the spatial validity of the alert
            is_target = (target_zones is not None and zone_id in target_zones)
            
            # 2. Extract the detection time safely depending on the data structure
            if isinstance(det_val, dict):
                valid_times = [t for t in det_val.values() if t is not None]
                t_det = min(valid_times) if valid_times else None
                trigger_param = "pH" if det_val.get('ph') == t_det else "Temp"
            else:
                t_det = det_val
                trigger_param = "Sensor"

            if t_det is None: 
                continue

            # 3. Apply temporal logic to label the detection
            time_str = f"Hour {t_det/3600:.2f}"
            
            if is_healthy:
                print(f"  --> ZONE {zone_id} ALERT: [FALSE POSITIVE] at {time_str} by {trigger_param}")
            elif is_target:
                delay = t_det - t_start
                # If the alarm triggered before the disease even started, it is a false positive
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
        Executes a localized trial for a single-parameter disease (pH or Temp) and plots the results.
        
        @param disease_key (str): The specific pathology to simulate.
        @param is_healthy (bool): Flag to run the simulation as a healthy baseline.
        @param raw_drift (float): Raw stream CUSUM threshold.
        @param agg_pers (int): Aggregated stream persistence threshold.
        @param agg_drift (float): Aggregated stream absolute drift threshold.
        @param safe_start (float): Minimum time before the disease can manifest.
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
        Executes a localized trial for a dual-parameter disease (pH AND Temp) and plots the results.
        
        @param disease_key (str): The specific combined pathology to simulate.
        @param is_healthy (bool): Flag to run the simulation as a healthy baseline.
        @param raw_drift_ph/temp (float): Raw stream CUSUM thresholds.
        @param agg_pers (int): Aggregated stream persistence threshold.
        @param agg_drift_ph/temp (float): Aggregated stream absolute drift thresholds.
        @param safe_start (float): Minimum time before the disease can manifest.
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
            alpha=self.alpha, 
            beta=self.beta, 
            beta_ph=self.beta_ph,     
            beta_temp=self.beta_temp, 
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
    Calculates strict clinical metrics across a suite of simulations. 
    Distinguishes between Patient-Level accuracy (overall survival) and 
    Zone-Level accuracy (spatial mapping and false alarm penalties).
    """
    def __init__(self, total_zones=20):
        """
        Initializes the performance evaluator.
        
        @param total_zones (int): The total number of monitored anatomical regions.
        """
        self.total_zones = total_zones

    def evaluate_test_suite(self, results_list, stream_name):
        """
        Processes a list of simulation results and prints a formatted clinical report.
        
        @param results_list (list): Collection of dictionaries containing metadata and detection times.
        @param stream_name (str): Identifier for the evaluated data stream.
        """
        # Patient-Level Counters
        patient_tp, patient_fp, patient_fn = 0, 0, 0
        
        # Zone-Level Counters
        zone_tp, zone_fp, zone_fn = 0, 0, 0
        npl_list = []

        for res in results_list:
            target_zones = res['Target_Zones']
            t_start = res['True_t_start']
            t_peak = res['True_t_peak']
            det_times = res['Detection_Times']  

            # --- 1. HEALTHY PATIENT SCENARIO ---
            if not target_zones or target_zones == [-1]:  
                if len(det_times) > 0:
                    patient_fp += 1
                    zone_fp += len(det_times)
                continue

            # --- 2. DISEASED PATIENT SCENARIO ---
            # Guard against division by zero if onset and peak are instantaneous
            progression_window = max(t_peak - t_start, 1)
            patient_alarm_rung = False
            
            # Check target zones (Did we catch the actual disease?)
            for tz in target_zones:
                if tz in det_times:
                    raw_det = det_times[tz]
                    
                    # Safely extract the detection time if it is a multimodal dictionary
                    t_det = min([t for t in raw_det.values() if t is not None]) if isinstance(raw_det, dict) else raw_det
                    
                    if t_det >= t_start:
                        zone_tp += 1
                        patient_alarm_rung = True
                        npl_list.append(min(1.0, (t_det - t_start) / progression_window))
                    else:
                        # Alert triggered before the disease physically started
                        zone_fp += 1 
                        zone_fn += 1 
                else:
                    zone_fn += 1 # Missed this specific targeted zone entirely

            # Determine Patient-Level Survival (At least one target zone triggered a valid alert)
            if patient_alarm_rung:
                patient_tp += 1
            else:
                patient_fn += 1

            # Check for spatial leakage (Alerts triggered in physically healthy zones)
            spatial_fps = len([z for z in det_times.keys() if z not in target_zones])
            zone_fp += spatial_fps
            
            if spatial_fps > 0 and not patient_alarm_rung:
                # The alarm rang, but for the completely wrong body part
                patient_fp += 1 

        # --- Calculate Final Clinical Metrics ---
        patient_recall = patient_tp / (patient_tp + patient_fn) if (patient_tp + patient_fn) > 0 else 0.0
        zone_recall = zone_tp / (zone_tp + zone_fn) if (zone_tp + zone_fn) > 0 else 0.0
        zone_precision = zone_tp / (zone_tp + zone_fp) if (zone_tp + zone_fp) > 0 else 0.0
        mean_npl = np.mean(npl_list) if npl_list else 1.0

        print(f"\n{'='*50}")
        print(f" {stream_name} - FINAL CLINICAL REPORT")
        print(f"{'='*50}")
        print(f" TOTAL PATIENTS: {len(results_list)} (15 Diseased, 15 Healthy)")
        print(f" -> Patient-Level Sensitivity (Lives Saved): {patient_recall * 100:.1f}%")
        print(f"    (Patient TPs: {patient_tp}, Misses: {patient_fn}, False Alarms: {patient_fp})")
        print("-" * 50)
        print(f" -> Zone-Level Sensitivity (Mapping Acc):    {zone_recall * 100:.1f}%")
        print(f" -> Zone-Level Precision (Trust factor):     {zone_precision * 100:.1f}%")
        print(f"    (Zone TPs: {zone_tp}, Misses: {zone_fn}, False Alarms: {zone_fp})")
        print("-" * 50)
        print(f" Mean Detection Speed: {mean_npl * 100:.1f}% of Disease Peak")
        print(f"{'='*50}")

# ==========================================
# COMPREHENSIVE TEST SUITE (PARALLELIZED)
# ==========================================

class ComprehensiveTestSuite:
    """
    Executes the final validation benchmark across all diseases and healthy baselines.
    Utilizes joblib multiprocessing to execute simulations in parallel for maximum speed.
    """
    def __init__(self, optimized_params, test_trace_path='position_data.csv', alpha=0.3, beta=0.10):
        """
        Initializes the parallel test suite.
        
        @param optimized_params (dict): The final optimized Kalman thresholds.
        @param test_trace_path (str): Path to the hold-out test dataset trace.
        @param alpha (float): Localization error probability.
        @param beta (float): Measurement error probability.
        """
        self.alpha = alpha
        self.beta = beta
        self.params = optimized_params
        self.test_trace_path = test_trace_path
        
        self.evaluator = FinalPerformanceEvaluator()
        
        # Temporarily instantiate a simulator exclusively to extract the disease dictionary keys
        temp_sim = DiseaseSimulator(df_path=self.test_trace_path)
        self.diseases = list(temp_sim.diseases.keys())
        self.disease_dict = temp_sim.diseases
        del temp_sim # Clean up immediately to save memory

    def _worker_diseased(self, d_key):
        """
        Worker function to simulate a single diseased scenario in an isolated process.
        
        @param d_key (str): The specific disease identifier to simulate.
        @return tuple: Two dictionaries containing raw and aggregated detection results.
        """
        # Initialize a fresh simulator INSIDE the worker to ensure thread safety
        simulator = DiseaseSimulator(df_path=self.test_trace_path)
        d_info = self.disease_dict[d_key]
        
        if d_info['type'] == 'single':
            raw_df, agg_df, t_start = simulator.run_single_disease(
                alpha=self.alpha, beta=self.beta, disease_key=d_key, target_confidence=70.0
            )
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.beta,
                raw_drift_threshold=self.params[f"{d_info['param']}_raw_drift"],
                raw_leakage=self.params[f"{d_info['param']}_raw_leakage"],
                agg_persistence=self.params[f"{d_info['param']}_agg_pers"],
                agg_abs_drift=self.params[f"{d_info['param']}_agg_drift"]
            )
            _, det_raw = detector.detect_raw_stream(raw_df)
            _, det_agg = detector.detect_aggregated_stream(agg_df)
        else: 
            raw_df, agg_df, t_start = simulator.run_combined_disease(
                alpha=self.alpha, beta=self.beta, disease_key=d_key, target_confidence=70.0
            )
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.beta, beta_ph=0.10, beta_temp=0.01,
                raw_drift_ph=self.params['comb_raw_drift_ph'],
                raw_drift_temp=self.params['comb_raw_drift_temp'],
                raw_leakage=self.params['comb_raw_leakage'],
                agg_persistence=self.params['comb_agg_pers'],
                agg_abs_drift_ph=self.params['comb_agg_drift_ph'],
                agg_abs_drift_temp=self.params['comb_agg_drift_temp']
            )
            _, det_raw = detector.detect_combined_raw_stream(raw_df) 
            _, det_agg = detector.detect_combined_aggregated_stream(agg_df)

        res_base = {
            'Target_Zones': d_info['zones'], 
            'True_t_start': t_start, 
            'True_t_peak': t_start + d_info['progression_time']
        }
        
        # Return a tuple of (raw_results, agg_results) to the main orchestrator thread
        return {**res_base, 'Detection_Times': det_raw}, {**res_base, 'Detection_Times': det_agg}

    def _worker_healthy(self, run_id):
        """
        Worker function to run a single healthy baseline scenario in an isolated process.
        
        @param run_id (int): A dummy identifier for the parallel iteration.
        @return tuple: Two dictionaries containing raw and aggregated detection results.
        """
        simulator = DiseaseSimulator(df_path=self.test_trace_path)
        
        # Push the safe start out of bounds (86400s) to guarantee healthy data
        raw_df, agg_df, t_start = simulator.run_single_disease(
            alpha=self.alpha, beta=self.beta, disease_key='systemic_fever_temp', safe_start_time=86400
        )
        
        # Use the most sensitive detector parameters (pH) to aggressively test healthy resilience
        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta,
            raw_drift_threshold=self.params['ph_raw_drift'],
            raw_leakage=self.params['ph_raw_leakage'],
            agg_persistence=self.params['ph_agg_pers'],
            agg_abs_drift=self.params['ph_agg_drift']
        )
        
        _, det_raw = detector.detect_raw_stream(raw_df)
        _, det_agg = detector.detect_aggregated_stream(agg_df)

        res_base = {'Target_Zones': [-1], 'True_t_start': 999999, 'True_t_peak': 999999}
        return {**res_base, 'Detection_Times': det_raw}, {**res_base, 'Detection_Times': det_agg}

    def run_full_suite(self):
        """
        Orchestrates the parallel execution of all diseased and healthy scenarios 
        and routes the compiled results to the performance evaluator.
        """
        # Reserve one CPU core to maintain system stability during heavy execution
        cores = max(1, multiprocessing.cpu_count() // 2)
        print(f"Starting Parallel Test Suite on {cores} cores...")
        
        print(" -> Simulating 15 Diseased scenarios...")
        diseased_results = Parallel(n_jobs=cores)(delayed(self._worker_diseased)(d_key) for d_key in self.diseases)
        
        print(" -> Simulating 15 Healthy scenarios...")
        healthy_results = Parallel(n_jobs=cores)(delayed(self._worker_healthy)(i) for i in range(15))
        
        print("\nAggregating results and running Final Evaluator...")
        
        # Unpack the tuples returned by the parallel workers
        results_raw = [res[0] for res in diseased_results] + [res[0] for res in healthy_results]
        results_agg = [res[1] for res in diseased_results] + [res[1] for res in healthy_results]
        
        # Dispatch the aggregated results to the clinical evaluator
        self.evaluator.evaluate_test_suite(results_raw, "RAW STREAM")
        self.evaluator.evaluate_test_suite(results_agg, "AGGREGATED STREAM")