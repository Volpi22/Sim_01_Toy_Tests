from diseases import DiseaseSimulator
from drift_detection import AnomalyDetector
from plots import AnomalyVisualizer

class ClinicalTrialRunner:
    """
    Executes head-to-head simulations of specific diseases or healthy baselines.
    Handles the instantiation of the simulator, the anomaly detector, 
    terminal reporting, and final visual plotting.
    """
    def __init__(self, alpha=0.3, beta=0.10, target_conf=70.0):
        self.alpha = alpha
        self.beta = beta
        self.target_conf = target_conf
        self.simulator = DiseaseSimulator()
        self.visualizer = AnomalyVisualizer()

    def _print_alerts(self, alert_dict, stream_name, t_start, is_healthy):
        """Helper to format and print the detection results."""
        print(f"\n[{stream_name} ALERTS]")
        if not alert_dict:
            if is_healthy:
                print("  No anomalies detected in any zone. (Expected Healthy Behavior)")
            else:
                print("  No anomalies detected in any zone. (FALSE NEGATIVE / MISSED)")
            return
            
        for zone_id, det_val in alert_dict.items():
            print(f"  --> ZONE {zone_id} ALERT:")
            
            # Extract time safely whether it's a float or a dict (combined)
            if isinstance(det_val, dict):
                valid_times = [t for t in det_val.values() if t is not None]
                t_det = min(valid_times) if valid_times else None
                trigger_param = "pH" if det_val.get('ph') == t_det else "Temp"
            else:
                t_det = det_val
                trigger_param = "Sensor"
                
            if t_det is not None:
                if is_healthy:
                    print(f"      - FALSE POSITIVE: Flagged at Hour {t_det/3600:.2f} by {trigger_param}")
                else:
                    delay = t_det - t_start
                    status = f"False Positive ({-delay/60:.1f} min early)" if delay < 0 else f"Delay: {delay/60:.1f} min"
                    print(f"      - Flagged at Hour {t_det/3600:.2f} by {trigger_param} | {status}")

    def run_single_parameter_trial(self, disease_key, is_healthy=False, 
                                   raw_drift=2.5, agg_pers=3, agg_drift=0.10, safe_start=1800):
        """Executes a trial for a single-parameter disease (pH or Temp)."""
        
        target_zone = self.simulator.diseases[disease_key]['zones'][0]
        actual_start = 86400 if is_healthy else safe_start
        title_prefix = "Healthy Baseline" if is_healthy else f"Detection Speed ({disease_key.upper()})"
        
        print(f"\n{'='*50}")
        print(f"--- RUNNING TRIAL: {disease_key.upper()} (Zone {target_zone}) ---")
        print(f"{'='*50}")

        raw_df, agg_df, t_start = self.simulator.run_single_disease(
            alpha=self.alpha, beta=self.beta, disease_key=disease_key,
            target_confidence=self.target_conf, safe_start_time=actual_start
        )

        if agg_df is None or agg_df.empty:
            print("Simulation failed or returned empty data."); return

        print(f"\n--- Running Anomaly Detection ---")
        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, target_confidence=self.target_conf,
            raw_drift_threshold=raw_drift, agg_persistence=agg_pers, agg_abs_drift=agg_drift
        )

        processed_raw_df, all_raw_alerts = detector.detect_raw_stream(raw_df)
        processed_agg_df, all_agg_alerts = detector.detect_aggregated_stream(agg_df)

        print("\n--- Detection Results ---")
        if not is_healthy: print(f"Disease actually started at: Hour {t_start/3600:.2f}")
        
        self._print_alerts(all_raw_alerts, "RAW STREAM", t_start, is_healthy)
        self._print_alerts(all_agg_alerts, "AGGREGATED STREAM", t_start, is_healthy)

        print(f"\n--- Generating Visualization ---")
        plot_det_raw = all_raw_alerts.get(target_zone, None)
        plot_det_agg = all_agg_alerts.get(target_zone, None)
        
        plot_raw_df = processed_raw_df[processed_raw_df['Zone'] == target_zone]
        plot_agg_df = processed_agg_df[processed_agg_df['Zone'] == target_zone]

        self.visualizer.plot_comparison(
            raw_df=plot_raw_df, agg_df=plot_agg_df, 
            det_raw=plot_det_raw, det_agg=plot_det_agg, 
            title=f"{title_prefix} - Zone {target_zone}"
        )


    def run_combined_parameter_trial(self, disease_key, is_healthy=False, 
                                     raw_drift_ph=1.5, raw_drift_temp=3.5, 
                                     agg_pers=3, agg_drift_ph=0.10, agg_drift_temp=0.5, 
                                     safe_start=1800):
        """Executes a trial for a dual-parameter disease (pH AND Temp)."""
        
        target_zone = self.simulator.diseases[disease_key]['zones'][0]
        actual_start = 86400 if is_healthy else safe_start
        title_prefix = "Healthy Combined Baseline" if is_healthy else f"Detection Speed ({disease_key.upper()})"
        
        print(f"\n{'='*50}")
        print(f"--- RUNNING COMBINED TRIAL: {disease_key.upper()} (Zone {target_zone}) ---")
        print(f"{'='*50}")

        raw_df, agg_df, t_start = self.simulator.run_combined_disease(
            alpha=self.alpha, beta=self.beta, disease_key=disease_key,
            target_confidence=self.target_conf, safe_start_time=actual_start
        )

        if agg_df is None or agg_df.empty:
            print("Simulation failed or returned empty data."); return

        print(f"\n--- Running Combined Anomaly Detection ---")
        detector = AnomalyDetector(
            alpha=self.alpha, beta=self.beta, target_confidence=self.target_conf,
            raw_drift_ph=raw_drift_ph, raw_drift_temp=raw_drift_temp,
            agg_persistence=agg_pers, agg_abs_drift_ph=agg_drift_ph, agg_abs_drift_temp=agg_drift_temp
        )

        processed_raw_df, all_raw_alerts = detector.detect_combined_raw_stream(raw_df)
        processed_agg_df, all_agg_alerts = detector.detect_combined_aggregated_stream(agg_df)

        print("\n--- Detection Results ---")
        if not is_healthy: print(f"Disease actually started at: Hour {t_start/3600:.2f}")
        
        self._print_alerts(all_raw_alerts, "RAW COMBINED STREAM", t_start, is_healthy)
        self._print_alerts(all_agg_alerts, "AGGREGATED COMBINED STREAM", t_start, is_healthy)

        print(f"\n--- Generating Combined Visualization ---")
        
        # Safely extract dictionary alerts for plotting, or default to None
        det_raw_dict = all_raw_alerts.get(target_zone, {'ph': None, 'temp': None})
        det_agg_dict = all_agg_alerts.get(target_zone, {'ph': None, 'temp': None})
        
        plot_raw_df = processed_raw_df[processed_raw_df['Zone'] == target_zone]
        plot_agg_df = processed_agg_df[processed_agg_df['Zone'] == target_zone]

        self.visualizer.plot_combined_comparison(
            raw_df=plot_raw_df, agg_df=plot_agg_df, 
            det_raw=det_raw_dict, det_agg=det_agg_dict, 
            title=f"{title_prefix} - Zone {target_zone}"
        )