from diseases import DiseaseSimulator
from drift_detection import AnomalyDetector
from plots import AnomalyVisualizer

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