import numpy as np
from utils import StatsUtils

# ==========================================
# KALMAN FILTERS FOR DRIFT DETECTION
# ==========================================

class RawStreamKalman:
    """
    1D Kalman Filter designed for raw, noisy nanobot pings.
    It uses a Cumulative Sum (CUSUM) approach to detect sustained drift,
    making it resilient to isolated, extreme outliers caused by sensing errors.
    """
    # ---> NEW: Added 'leakage' to the init arguments <---
    def __init__(self, alpha, beta, initial_val, q_process=1e-5, drift_threshold=2.5, leakage=0.98):
        """
        Args:
            alpha (float): Probability of localization error.
            beta (float): Probability of measurement error.
            initial_val (float): The starting baseline value.
            q_process (float): Process noise variance (how much we trust the model vs data).
            drift_threshold (float): The threshold for the cumulative residual to flag an anomaly.
            leakage (float): How much memory of previous errors to retain (0.0 to 1.0).
        """
        # Measurement uncertainty (R) combines sensor error and location mismatch
        self.r = (beta**2) + (alpha**2)
        self.q = q_process
        self.drift_threshold = drift_threshold
        self.leakage = leakage  # ---> NEW: Store it
        
        # Store state variables for multiple zones independently
        self.zones = {}
        self.initial_val = initial_val

    def check_zone(self, zone_id, raw_value):
        """Processes a single raw reading and updates the filter state."""
        if zone_id not in self.zones:
            self.zones[zone_id] = {
                'x': self.initial_val,  # Current estimated value
                'p': 1.0,               # Estimate uncertainty
                'cum_res': 0.0,         # Cumulative residual (for CUSUM drift detection)
                'detected_at': None
            }
        
        z = self.zones[zone_id]

        # 1. PREDICT STEP: Project the state and uncertainty forward
        z['p'] = z['p'] + self.q

        # 2. UPDATE STEP: Correct the estimate using the new measurement
        residual = raw_value - z['x']                  # Difference between measurement and prediction
        k_gain = z['p'] / (z['p'] + self.r)            # Kalman Gain (how much to trust the new measurement)
        
        z['x'] = z['x'] + k_gain * residual            # Update estimate
        z['p'] = (1 - k_gain) * z['p']                 # Update uncertainty

        # 3. DRIFT DETECTION: Accumulate sustained errors with a leakage factor to prevent over-sensitivity to constant noise
        # ---> NEW: Use self.leakage instead of the hardcoded 0.98 <---
        z['cum_res'] = (z['cum_res'] + residual) * self.leakage
        is_drift = abs(z['cum_res']) > self.drift_threshold
        
        if is_drift:
            # Dampen the cumulative residual instead of resetting it to 0.
            # This allows continuous tracking while preventing a single spike from triggering multiple alarms.
            z['cum_res'] *= 0.5 
            
        return is_drift, z['x']


class AggregatedKalman:
    """
    Adaptive 2nd-Order Kalman Filter for aggregated data packets.
    Tracks both the value (position) and the rate of change (velocity).
    Dynamically scales measurement noise based on the statistical confidence of the packet.
    """
    def __init__(self, initial_val, q_velocity=1e-6, persistence=3, velocity_threshold=0.001, abs_drift_threshold=0.05):
        """
        Args:
            initial_val (float): The starting baseline value.
            q_process (float): Process noise variance for velocity.
            persistence (int): Consecutive 'bad' packets required to trigger an anomaly.
            velocity_threshold (float): Minimum rate of change to be considered "moving".
            abs_drift_threshold (float): Minimum absolute deviation from baseline to trigger an anomaly.
        """
        self.q = q_velocity
        self.persistence = persistence
        self.velocity_threshold = velocity_threshold
        self.abs_drift_threshold = abs_drift_threshold
        self.zones = {}
        self.initial_val = initial_val

    def check_zone(self, zone_id, agg_value, binom_conf):
        """Processes an aggregated median value and its statistical confidence."""
        if zone_id not in self.zones:
            self.zones[zone_id] = {
                # x is a 2D state vector: [Current Value, Rate of Change (Velocity)]
                'x': np.array([[self.initial_val], [0.0]]), 
                'p': np.eye(2) * 10,  # Initial covariance matrix
                'hit_counter': 0      # Tracks consecutive anomaly conditions
            }
        
        z = self.zones[zone_id]
        
        # Adaptive Measurement Noise (R):
        # Higher binomial confidence (closer to 100) results in lower noise (R -> 0), meaning we trust the data more.
        # Squaring it penalizes low-confidence packets heavily.
        r_val = max(1e-6, ((100 - binom_conf) / 100)**2)
        
        # System Matrices
        F = np.array([[1, 1], [0, 1]]) # State Transition Matrix (Value += Velocity)
        H = np.array([[1, 0]])         # Observation Matrix (We only measure the Value, not Velocity)
        Q = np.eye(2) * self.q         # Process Noise Covariance
        R = np.array([[r_val]])        # Dynamic Measurement Noise

        # 1. PREDICT STEP
        z['x'] = F @ z['x']
        z['p'] = F @ z['p'] @ F.T + Q

        # 2. UPDATE STEP
        residual = agg_value - (H @ z['x'])[0,0]
        S = H @ z['p'] @ H.T + R                        # Innovation covariance
        k_gain = z['p'] @ H.T @ np.linalg.inv(S)        # Calculate Kalman Gain

        z['x'] = z['x'] + k_gain * residual             # Update State
        z['p'] = (np.eye(2) - k_gain @ H) @ z['p']      # Update Covariance

        # 3. ANOMALY DETECTION LOGIC
        velocity = z['x'][1, 0]
        current_estimate = z['x'][0, 0]
        
        # Condition A: Is the value consistently trending upward/downward?
        is_moving = abs(velocity) > self.velocity_threshold
        
        # Condition B: Has the value moved far enough from the healthy baseline to be clinically relevant?
        has_drifted_far_enough = abs(current_estimate - self.initial_val) > self.abs_drift_threshold
        
        # Both conditions must be met simultaneously to register a "hit"
        if is_moving and has_drifted_far_enough:
            z['hit_counter'] += 1
        else:
            z['hit_counter'] = 0

        # Trigger anomaly only if the bad conditions persist for 'n' consecutive packets
        return (z['hit_counter'] >= self.persistence), current_estimate

# ==========================================
# ANOMALY DETECTOR ROUTER
# ==========================================

class AnomalyDetector:
    """
    Whole-body Anomaly Detector (NumPy Optimized).
    Processes all zones independently, maintaining separate baselines and Kalman 
    filters for each anatomical region. Returns a dictionary of triggering zones.
    """
    # ---> NEW: Added raw_leakage parameter to the router <---
    def __init__(self, alpha=0.05, beta=0.1, target_confidence=70.0, 
                 raw_drift_threshold=2.5, raw_drift_ph=1.5, raw_drift_temp=3.5, 
                 agg_persistence=3, agg_abs_drift=0.05, agg_abs_drift_ph=0.10, agg_abs_drift_temp=0.5,
                 beta_ph=None, beta_temp=None, raw_leakage=0.98): 
        
        self.alpha = alpha
        self.beta = beta # Used for single streams and Binomial Confidence
        
        # If specific betas aren't provided, default to the main beta
        self.beta_ph = beta_ph if beta_ph is not None else beta
        self.beta_temp = beta_temp if beta_temp is not None else beta
        
        self.target_confidence = target_confidence
        
        self.raw_drift_threshold = raw_drift_threshold
        self.raw_drift_ph = raw_drift_ph
        self.raw_drift_temp = raw_drift_temp
        self.raw_leakage = raw_leakage  # ---> NEW: Store it
        
        self.agg_persistence = agg_persistence
        self.agg_abs_drift = agg_abs_drift
        self.agg_abs_drift_ph = agg_abs_drift_ph
        self.agg_abs_drift_temp = agg_abs_drift_temp

    # ==========================================
    # 1. RAW STREAM (SINGLE PARAMETER)
    # ==========================================
    def detect_raw_stream(self, raw_df):
        if raw_df.empty: return raw_df, {}
        
        df_out = raw_df.copy().sort_values(['Zone', 'Time_Seconds'])
        
        # Convert to fast NumPy arrays
        zones = df_out['Zone'].values
        times = df_out['Time_Seconds'].values
        values = df_out['Value'].values
        
        filtered_results = np.zeros(len(values))
        detection_times = {}

        for zone in np.unique(zones):
            mask = (zones == zone)
            z_vals, z_times = values[mask], times[mask]
            z_filt = np.zeros(len(z_vals))
            
            buffer = []
            baseline_established = False
            kf = None
            det_time = None

            for i in range(len(z_vals)):
                val = z_vals[i]
                if not baseline_established:
                    buffer.append(val)
                    if StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, len(buffer)) >= self.target_confidence:
                        dynamic_baseline = np.median(buffer)
                        # ---> NEW: Pass the leakage to the Kalman filter <---
                        kf = RawStreamKalman(alpha=self.alpha, beta=self.beta, initial_val=dynamic_baseline, drift_threshold=self.raw_drift_threshold, leakage=self.raw_leakage)
                        baseline_established = True
                        z_filt[:i+1] = dynamic_baseline # Backfill calibration phase
                else:
                    is_drift, x_val = kf.check_zone(zone, val)
                    z_filt[i] = x_val
                    if is_drift and det_time is None:
                        det_time = z_times[i]

            filtered_results[mask] = z_filt
            if det_time is not None:
                detection_times[zone] = det_time

        df_out['Filtered_Value'] = filtered_results
        return df_out, detection_times

    # ==========================================
    # 2. AGGREGATED STREAM (SINGLE PARAMETER)
    # ==========================================
    def detect_aggregated_stream(self, agg_df):
        if agg_df.empty: return agg_df, {}
        
        df_out = agg_df.copy().sort_values(['Zone', 'Time_Seconds_End'])
        if 'Binomial_%' not in df_out.columns:
            df_out['Binomial_%'] = df_out['Count'].apply(lambda n: StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, n))

        zones = df_out['Zone'].values
        times = df_out['Time_Seconds_End'].values
        values = df_out['Median_Val'].values
        confs = df_out['Binomial_%'].values
        
        filtered_results = np.zeros(len(values))
        detection_times = {}

        for zone in np.unique(zones):
            mask = (zones == zone)
            z_vals, z_times, z_confs = values[mask], times[mask], confs[mask]
            z_filt = np.zeros(len(z_vals))
            
            baseline_established = False
            kf = None
            det_time = None

            for i in range(len(z_vals)):
                val, conf = z_vals[i], z_confs[i]
                if not baseline_established:
                    if conf >= self.target_confidence:
                        dynamic_baseline = val
                        kf = AggregatedKalman(initial_val=dynamic_baseline, persistence=self.agg_persistence, abs_drift_threshold=self.agg_abs_drift)
                        baseline_established = True
                        z_filt[i] = dynamic_baseline
                else:
                    is_drift, x_val = kf.check_zone(zone, val, conf)
                    z_filt[i] = x_val
                    if is_drift and det_time is None:
                        det_time = z_times[i]

            filtered_results[mask] = z_filt
            if det_time is not None:
                detection_times[zone] = det_time

        df_out['Filtered_Value'] = filtered_results
        return df_out, detection_times

    # ==========================================
    # 3. RAW STREAM (COMBINED PARAMETERS)
    # ==========================================
    def detect_combined_raw_stream(self, raw_df):
        if raw_df.empty: return raw_df, {}
        
        df_out = raw_df.copy().sort_values(['Zone', 'Time_Seconds'])
        
        zones = df_out['Zone'].values
        times = df_out['Time_Seconds'].values
        val_ph = df_out['Value_pH'].values
        val_temp = df_out['Value_Temp'].values
        
        filt_ph = np.zeros(len(val_ph))
        filt_temp = np.zeros(len(val_temp))
        detection_times = {} 

        for zone in np.unique(zones):
            mask = (zones == zone)
            z_times = times[mask]
            z_ph, z_temp = val_ph[mask], val_temp[mask]
            
            out_ph, out_temp = np.zeros(len(z_ph)), np.zeros(len(z_temp))
            buf_ph, buf_temp = [], []
            
            baseline_established = False
            kf_ph, kf_temp = None, None
            det_ph, det_temp = None, None

            for i in range(len(z_ph)):
                v_ph, v_temp = z_ph[i], z_temp[i]
                
                if not baseline_established:
                    buf_ph.append(v_ph)
                    buf_temp.append(v_temp)
                    # Notice we still use self.beta (the highest one) to compute the combined binomial confidence
                    if StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, len(buf_ph)) >= self.target_confidence:
                        # ---> NEW: Pass the leakage to the Kalman filters <---
                        kf_ph = RawStreamKalman(alpha=self.alpha, beta=self.beta_ph, initial_val=np.median(buf_ph), drift_threshold=self.raw_drift_ph, leakage=self.raw_leakage)
                        kf_temp = RawStreamKalman(alpha=self.alpha, beta=self.beta_temp, initial_val=np.median(buf_temp), drift_threshold=self.raw_drift_temp, leakage=self.raw_leakage)
                        baseline_established = True
                        out_ph[:i+1] = np.median(buf_ph)
                        out_temp[:i+1] = np.median(buf_temp)
                else:
                    is_drift_ph, x_ph = kf_ph.check_zone(zone, v_ph)
                    is_drift_temp, x_temp = kf_temp.check_zone(zone, v_temp)
                    
                    out_ph[i], out_temp[i] = x_ph, x_temp
                    
                    if is_drift_ph and det_ph is None: det_ph = z_times[i]
                    if is_drift_temp and det_temp is None: det_temp = z_times[i]

            filt_ph[mask] = out_ph
            filt_temp[mask] = out_temp
            
            # Record detection if EITHER parameter drifted
            if det_ph is not None or det_temp is not None:
                detection_times[zone] = {'ph': det_ph, 'temp': det_temp}

        df_out['Filtered_pH'] = filt_ph
        df_out['Filtered_Temp'] = filt_temp
        return df_out, detection_times

    # ==========================================
    # 4. AGGREGATED STREAM (COMBINED PARAMETERS)
    # ==========================================
    def detect_combined_aggregated_stream(self, agg_df):
        if agg_df.empty: return agg_df, {}
        
        df_out = agg_df.copy().sort_values(['Zone', 'Time_Seconds_End'])
        if 'Binomial_%' not in df_out.columns:
            # Still using the highest self.beta for conservative binomial calculation
            df_out['Binomial_%'] = df_out['Count'].apply(lambda n: StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, n))

        zones = df_out['Zone'].values
        times = df_out['Time_Seconds_End'].values
        val_ph = df_out['Median_pH'].values
        val_temp = df_out['Median_Temp'].values
        confs = df_out['Binomial_%'].values
        
        filt_ph = np.zeros(len(val_ph))
        filt_temp = np.zeros(len(val_temp))
        detection_times = {}

        for zone in np.unique(zones):
            mask = (zones == zone)
            z_times, z_confs = times[mask], confs[mask]
            z_ph, z_temp = val_ph[mask], val_temp[mask]
            
            out_ph, out_temp = np.zeros(len(z_ph)), np.zeros(len(z_temp))
            
            baseline_established = False
            kf_ph, kf_temp = None, None
            det_ph, det_temp = None, None

            for i in range(len(z_ph)):
                v_ph, v_temp, conf = z_ph[i], z_temp[i], z_confs[i]
                
                if not baseline_established:
                    if conf >= self.target_confidence:
                        kf_ph = AggregatedKalman(initial_val=v_ph, persistence=self.agg_persistence, abs_drift_threshold=self.agg_abs_drift_ph)
                        kf_temp = AggregatedKalman(initial_val=v_temp, persistence=self.agg_persistence, abs_drift_threshold=self.agg_abs_drift_temp)
                        baseline_established = True
                        out_ph[i], out_temp[i] = v_ph, v_temp
                else:
                    is_drift_ph, x_ph = kf_ph.check_zone(zone, v_ph, conf)
                    is_drift_temp, x_temp = kf_temp.check_zone(zone, v_temp, conf)
                    
                    out_ph[i], out_temp[i] = x_ph, x_temp
                    
                    if is_drift_ph and det_ph is None: det_ph = z_times[i]
                    if is_drift_temp and det_temp is None: det_temp = z_times[i]

            filt_ph[mask] = out_ph
            filt_temp[mask] = out_temp
            
            if det_ph is not None or det_temp is not None:
                detection_times[zone] = {'ph': det_ph, 'temp': det_temp}

        df_out['Filtered_pH'] = filt_ph
        df_out['Filtered_Temp'] = filt_temp
        return df_out, detection_times