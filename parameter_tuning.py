import pandas as pd
import numpy as np
import optuna
import warnings

# Suppress pandas chained assignment warnings for cleaner output during tuning
warnings.filterwarnings('ignore')

from drift_detection import AnomalyDetector

class ClinicalCostEvaluator:
    """
    Evaluates filter performance based on the custom clinical Cost Function:
    Cost = (w1 * FPR) + (w2 * FNR) + (w3 * NPL)
    """
    def __init__(self, w1=0.4, w2=0.4, w3=0.2, total_zones=10):
        self.w1 = w1  # Weight for False Positive Rate (Alarm Fatigue)
        self.w2 = w2  # Weight for False Negative Rate (Missed Detections)
        self.w3 = w3  # Weight for Normalized Progression Latency (Speed)
        self.total_zones = total_zones

    def calculate_cost(self, results_list):
        """
        Calculates the aggregate clinical cost across all simulations.
        results_list: List of dicts containing ground truth and detection times.
        """
        total_fp = 0
        total_fn = 0
        total_tp = 0
        total_tn = 0
        npl_sum = 0.0

        for res in results_list:
            target_zone = res['Target_Zone']
            t_start = res['True_t_start']
            t_peak = res['True_t_peak']
            det_times = res['Detection_Times']  # Format: {zone_id: time_seconds}

            if target_zone == -1:  # Healthy Patient
                fp_in_sim = len(det_times)
                total_fp += fp_in_sim
                total_tn += (self.total_zones - fp_in_sim)
            else:  # Diseased Patient
                # Check target zone detection
                if target_zone in det_times:
                    t_det = det_times[target_zone]
                    if t_det >= t_start:
                        # True Positive: Caught during or after onset
                        total_tp += 1
                        # Calculate NPL: Cap at 1.0 (100% severity) if detected after peak
                        progression_window = t_peak - t_start
                        npl = min(1.0, (t_det - t_start) / progression_window)
                        npl_sum += npl
                    else:
                        # Premature detection: Flagged before disease started -> False Positive
                        total_fp += 1
                        total_fn += 1 # Missed the actual disease window
                else:
                    # Missed completely -> False Negative
                    total_fn += 1

                # Count spatial false positives (flagged the wrong zone)
                other_zones_flagged = len([z for z in det_times.keys() if z != target_zone])
                total_fp += other_zones_flagged
                total_tn += ((self.total_zones - 1) - other_zones_flagged)

        # Calculate Rates
        # FPR: Total False Positives / Total Negative Opportunities
        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
        
        # FNR: Total False Negatives / Total Positive Opportunities (Total diseased patients)
        total_diseased = total_tp + total_fn
        fnr = total_fn / total_diseased if total_diseased > 0 else 0.0
        
        # Mean NPL
        mean_npl = npl_sum / total_tp if total_tp > 0 else 1.0 # Max penalty if no TPs

        # Final Composite Cost
        cost = (self.w1 * fpr) + (self.w2 * fnr) + (self.w3 * mean_npl)
        return cost


class FilterOptimizer:
    """Optuna objective classes for both Raw and Aggregated tuning."""
    def __init__(self, data_path, stream_type='raw'):
        print(f"Loading {stream_type.upper()} dataset into memory...")
        self.df = pd.read_parquet(data_path)
        self.stream_type = stream_type
        self.evaluator = ClinicalCostEvaluator()

        # Pre-group data by simulation ID to save time during the Optuna loop
        self.simulations = []
        for sim_id, group in self.df.groupby('Simulation_ID'):
            # Extract ground truth from the first row of the simulation
            first_row = group.iloc[0]
            self.simulations.append({
                'df': group.copy(),
                'Target_Zone': first_row['Target_Zone'],
                'True_t_start': first_row['True_t_start'],
                'True_t_peak': first_row['True_t_peak']
            })
        print(f"Loaded {len(self.simulations)} unique patient scenarios.")

    def objective_raw(self, trial):
        """1D Optimization for the Raw Stream Kalman Filter."""
        # 1. Define Search Space
        drift_thresh = trial.suggest_float("raw_drift_threshold", 0.01, 5.0, step=0.1)

        # 2. Instantiate Detector with trial parameters
        detector = AnomalyDetector(
            alpha=0.3, beta=0.10, target_confidence=70.0,
            raw_drift_threshold=drift_thresh
        )

        # 3. Run all simulations
        results = []
        for sim in self.simulations:
            _, det_times = detector.detect_raw_stream(sim['df'])
            results.append({
                'Target_Zone': sim['Target_Zone'],
                'True_t_start': sim['True_t_start'],
                'True_t_peak': sim['True_t_peak'],
                'Detection_Times': det_times
            })

        # 4. Calculate and return Cost
        return self.evaluator.calculate_cost(results)

    def objective_agg(self, trial):
        """2D Optimization for the Aggregated Stream Kalman Filter."""
        # 1. Define Search Space
        persistence = trial.suggest_int("agg_persistence", 1, 6)
        abs_drift = trial.suggest_float("agg_abs_drift", 0.01, 0.20, step=0.01)

        # 2. Instantiate Detector
        detector = AnomalyDetector(
            alpha=0.3, beta=0.10, target_confidence=70.0,
            agg_persistence=persistence,
            agg_abs_drift=abs_drift
        )

        # 3. Run all simulations
        results = []
        for sim in self.simulations:
            _, det_times = detector.detect_aggregated_stream(sim['df'])
            results.append({
                'Target_Zone': sim['Target_Zone'],
                'True_t_start': sim['True_t_start'],
                'True_t_peak': sim['True_t_peak'],
                'Detection_Times': det_times
            })

        # 4. Calculate and return Cost
        return self.evaluator.calculate_cost(results)