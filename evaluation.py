from diseases import DiseaseSimulator
from drift_detection import AnomalyDetector
from plots import AnomalyVisualizer
import numpy as np
import pandas as pd

class ClinicalTrialRunner:
    """
    Executes head-to-head simulations of specific diseases or healthy baselines.
    Updated with 'Smart Reporting' to distinguish True Positives from Spatial Errors.
    """
    def __init__(self, alpha=0.3, beta=0.10, beta_ph=0.10, beta_temp=0.01, target_conf=70.0):
        self.alpha = alpha
        self.beta = beta  # The highest beta, used for the binomial confidence
        self.beta_ph = beta_ph
        self.beta_temp = beta_temp
        self.target_conf = target_conf
        self.simulator = DiseaseSimulator()
        self.visualizer = AnomalyVisualizer()

    def _print_alerts(self, alert_dict, stream_name, t_start, is_healthy, target_zones=None):
        """
        Enhanced helper to format results. 
        Categorizes alerts as True Positive, False Positive, or Spatial Error.
        """
        print(f"\n[{stream_name} ALERTS]")
        if not alert_dict:
            if is_healthy:
                print("  No anomalies detected. (Expected Healthy Behavior)")
            else:
                print("  No anomalies detected. (!! FALSE NEGATIVE / MISSED !!)")
            return
            
        for zone_id, det_val in alert_dict.items():
            # 1. Determine the 'Type' of Alert (Now checking a list of zones)
            is_target = (target_zones is not None and zone_id in target_zones)
            
            # 2. Extract detection time safely
            if isinstance(det_val, dict):
                valid_times = [t for t in det_val.values() if t is not None]
                t_det = min(valid_times) if valid_times else None
                trigger_param = "pH" if det_val.get('ph') == t_det else "Temp"
            else:
                t_det = det_val
                trigger_param = "Sensor"

            if t_det is None: continue

            # 3. Logic-based Labeling
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
        """Executes a trial for a single-parameter disease (pH or Temp)."""
        
        # ---> CHANGED: Grab ALL target zones instead of just index 0
        target_zones = self.simulator.diseases[disease_key]['zones']
        actual_start = 86400 if is_healthy else safe_start
        title_prefix = "Healthy Baseline" if is_healthy else f"Detection Speed ({disease_key.upper()})"
        
        print(f"\n{'='*60}")
        print(f"--- RUNNING TRIAL: {disease_key.upper()} (Target Zones: {target_zones}) ---")
        print(f"{'='*60}")

        # ... [Keep your simulator and detector lines exactly the same] ...
        raw_df, agg_df, t_start = self.simulator.run_single_disease(
            alpha=self.alpha, beta=self.beta, disease_key=disease_key,
            target_confidence=self.target_conf, safe_start_time=actual_start
        )

        if agg_df is None or agg_df.empty:
            print("Simulation failed or returned empty data."); return

        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, target_confidence=self.target_conf,
            raw_drift_threshold=raw_drift, agg_persistence=agg_pers, agg_abs_drift=agg_drift
        )

        processed_raw_df, all_raw_alerts = detector.detect_raw_stream(raw_df)
        processed_agg_df, all_agg_alerts = detector.detect_aggregated_stream(agg_df)

        print("\n--- Detection Results ---")
        if not is_healthy: 
            print(f"Clinical Onset: Hour {t_start/3600:.2f}")
        
        # ---> CHANGED: Pass target_zones (plural)
        self._print_alerts(all_raw_alerts, "RAW STREAM", t_start, is_healthy, target_zones=target_zones)
        self._print_alerts(all_agg_alerts, "AGGREGATED STREAM", t_start, is_healthy, target_zones=target_zones)

        # ---> CHANGED: Pass entire DataFrames and the list of target zones
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
        """Executes a trial for a dual-parameter disease (pH AND Temp)."""
        
        # ---> CHANGED: Grab ALL target zones
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

        # ---> FIXED TYPO: detector = AnomalyDetector
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
        
        # ---> CHANGED: Pass target_zones (plural)
        self._print_alerts(all_raw_alerts, "RAW COMBINED", t_start, is_healthy, target_zones=target_zones)
        self._print_alerts(all_agg_alerts, "AGGREGATED COMBINED", t_start, is_healthy, target_zones=target_zones)

        # ---> CHANGED: Pass entire DataFrames and the list of target zones
        self.visualizer.plot_combined_comparison(
            raw_df=processed_raw_df, 
            agg_df=processed_agg_df, 
            det_raw=all_raw_alerts, 
            det_agg=all_agg_alerts, 
            target_zones=target_zones,
            title=f"{title_prefix}"
        )

class FinalPerformanceEvaluator:
    """
    Calculates clinical metrics. Distinguishes between Patient-Level accuracy 
    (did the alarm ring?) and Zone-Level accuracy (did we map the disease perfectly?).
    """
    def __init__(self, total_zones=20):
        self.total_zones = total_zones

    def evaluate_test_suite(self, results_list, stream_name):
        # Patient-Level
        patient_tp, patient_fp, patient_fn = 0, 0, 0
        # Zone-Level
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
            progression_window = t_peak - t_start
            patient_alarm_rung = False
            
            # Check target zones (Did we catch the actual disease?)
            for tz in target_zones:
                if tz in det_times:
                    raw_det = det_times[tz]
                    t_det = min([t for t in raw_det.values() if t is not None]) if isinstance(raw_det, dict) else raw_det
                    
                    if t_det >= t_start:
                        zone_tp += 1
                        patient_alarm_rung = True
                        npl_list.append(min(1.0, (t_det - t_start) / progression_window))
                    else:
                        zone_fp += 1 # Flagged before the disease even started
                        zone_fn += 1 
                else:
                    zone_fn += 1 # Missed this specific zone

            # Did the patient survive? (At least one target zone triggered an alert)
            if patient_alarm_rung:
                patient_tp += 1
            else:
                patient_fn += 1

            # Check for spatial leakage (Flagged zones that weren't diseased)
            spatial_fps = len([z for z in det_times.keys() if z not in target_zones])
            zone_fp += spatial_fps
            if spatial_fps > 0 and not patient_alarm_rung:
                patient_fp += 1 # The alarm rang, but for the completely wrong body part

        # --- Calculate Clinical Metrics ---
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


class ComprehensiveTestSuite:
    """Executes the final benchmark on the new Test Dataset."""
    def __init__(self, optimized_params, alpha=0.3, beta=0.10):
        self.alpha = alpha
        self.beta = beta
        self.params = optimized_params
        
        # NOTE: Make sure your DiseaseSimulator is loading the NEW test trace dataset here!
        self.simulator = DiseaseSimulator() 
        self.evaluator = FinalPerformanceEvaluator()

    def run_full_suite(self):
        print("Starting Full Test Suite...")
        diseases = list(self.simulator.diseases.keys())
        
        results_raw = []
        results_agg = []

        # 1. RUN ALL 15 DISEASES
        for d_key in diseases:
            d_info = self.simulator.diseases[d_key]
            
            if d_info['type'] == 'single':
                raw_df, agg_df, t_start = self.simulator.run_single_disease(
                    alpha=self.alpha, beta=self.beta, disease_key=d_key, target_confidence=70.0
                )
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta,
                    raw_drift_threshold=self.params[f"{d_info['param']}_raw_drift"],
                    agg_persistence=self.params[f"{d_info['param']}_agg_pers"],
                    agg_abs_drift=self.params[f"{d_info['param']}_agg_drift"]
                )
            else: # combined
                raw_df, agg_df, t_start = self.simulator.run_combined_disease(
                    alpha=self.alpha, beta=self.beta, disease_key=d_key, target_confidence=70.0
                )
                detector = AnomalyDetector(
                    alpha=self.alpha, beta=self.beta, beta_ph=0.10, beta_temp=0.01,
                    raw_drift_ph=self.params['comb_raw_drift_ph'],
                    raw_drift_temp=self.params['comb_raw_drift_temp'],
                    agg_persistence=self.params['comb_agg_pers'],
                    agg_abs_drift_ph=self.params['comb_agg_drift_ph'],
                    agg_abs_drift_temp=self.params['comb_agg_drift_temp']
                )

            if d_info['type'] == 'combined':
                _, det_raw = detector.detect_combined_raw_stream(raw_df) 
                _, det_agg = detector.detect_combined_aggregated_stream(agg_df)
            else:
                _, det_raw = detector.detect_raw_stream(raw_df)
                _, det_agg = detector.detect_aggregated_stream(agg_df)

            res_base = {'Target_Zones': d_info['zones'], 'True_t_start': t_start, 'True_t_peak': t_start + d_info['progression_time']}
            results_raw.append({**res_base, 'Detection_Times': det_raw})
            results_agg.append({**res_base, 'Detection_Times': det_agg})

        # 2. RUN 15 HEALTHY BASELINES
        for _ in range(15):
            # Using systemic_fever just as a dummy to get healthy data (passing safe_start way out of bounds)
            raw_df, agg_df, t_start = self.simulator.run_single_disease(
                alpha=self.alpha, beta=self.beta, disease_key='systemic_fever_temp', safe_start_time=86400
            )
            # Use the most sensitive detector (pH) to test healthy resilience
            detector = AnomalyDetector(
                alpha=self.alpha, beta=self.beta,
                raw_drift_threshold=self.params['ph_raw_drift'],
                agg_persistence=self.params['ph_agg_pers'],
                agg_abs_drift=self.params['ph_agg_drift']
            )
            _, det_raw = detector.detect_raw_stream(raw_df)
            _, det_agg = detector.detect_aggregated_stream(agg_df)

            res_base = {'Target_Zones': [-1], 'True_t_start': 999999, 'True_t_peak': 999999}
            results_raw.append({**res_base, 'Detection_Times': det_raw})
            results_agg.append({**res_base, 'Detection_Times': det_agg})

        # 3. EVALUATE
        self.evaluator.evaluate_test_suite(results_raw, "RAW STREAM")
        self.evaluator.evaluate_test_suite(results_agg, "AGGREGATED STREAM")