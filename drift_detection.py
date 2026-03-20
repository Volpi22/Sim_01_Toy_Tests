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
    def __init__(self, alpha, beta, initial_val, q_process=1e-5, drift_threshold=2.5):
        """
        Args:
            alpha (float): Probability of localization error.
            beta (float): Probability of measurement error.
            initial_val (float): The starting baseline value.
            q_process (float): Process noise variance (how much we trust the model vs data).
            drift_threshold (float): The threshold for the cumulative residual to flag an anomaly.
        """
        # Measurement uncertainty (R) combines sensor error and location mismatch
        self.r = (beta**2) + (alpha**2)
        self.q = q_process
        self.drift_threshold = drift_threshold
        
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

        # 3. DRIFT DETECTION: Accumulate sustained errors
        z['cum_res'] += residual
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
    High-level wrapper that manages data buffering, baseline calibration, 
    and routing data to the appropriate Kalman filters.
    """
    def __init__(self, target_zone, alpha=0.05, beta=0.1, target_confidence=70.0, 
                 raw_drift_threshold=2.5, agg_persistence=3, 
                 agg_abs_drift=0.05, agg_abs_drift_ph=0.10, agg_abs_drift_temp=0.5):
        
        # Core identification and error rates
        self.target_zone = target_zone
        self.alpha = alpha
        self.beta = beta
        self.target_confidence = target_confidence
        
        # Tuning levers for the Raw Filter (CUSUM)
        self.raw_drift_threshold = raw_drift_threshold
        
        # Tuning levers for the Aggregated Filter (Velocity + Drift)
        self.agg_persistence = agg_persistence
        
        # Absolute Drift Thresholds (Single parameter & Dual parameter limits)
        self.agg_abs_drift = agg_abs_drift
        self.agg_abs_drift_ph = agg_abs_drift_ph
        self.agg_abs_drift_temp = agg_abs_drift_temp

    def detect_raw_stream(self, raw_df):
        """Processes a single-parameter raw nanobot stream."""
        df_zone = raw_df[raw_df['Zone'] == self.target_zone].copy()
        if df_zone.empty:
            return df_zone, None
            
        df_zone = df_zone.sort_values('Time_Seconds')

        buffer = []
        baseline_established = False
        kf = None
        filtered_vals = []
        detection_time = None

        for _, row in df_zone.iterrows():
            # Phase 1: CALIBRATION
            # Buffer incoming data until the collective statistical confidence reaches the target.
            if not baseline_established:
                buffer.append(row['Value'])
                current_conf = StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, len(buffer))
                
                if current_conf >= self.target_confidence:
                    # Once confidence is met, calculate the median to establish a robust baseline
                    dynamic_baseline = np.median(buffer)
                    kf = RawStreamKalman(
                        alpha=self.alpha, 
                        beta=self.beta, 
                        initial_val=dynamic_baseline,
                        drift_threshold=self.raw_drift_threshold
                    )
                    baseline_established = True
                    # Backfill the filtered values array for seamless plotting
                    filtered_vals = [dynamic_baseline] * len(buffer)
                else:
                    filtered_vals.append(np.nan) # Keep blank while calibrating
                continue
                
            # Phase 2: ACTIVE DETECTION
            is_drift, x_val = kf.check_zone(self.target_zone, row['Value'])
            filtered_vals.append(x_val)
            
            # Record the precise second the first anomaly is triggered
            if is_drift and detection_time is None:
                detection_time = row['Time_Seconds']

        df_zone['Filtered_Value'] = filtered_vals
        return df_zone, detection_time

    def detect_aggregated_stream(self, agg_df):
        """Processes a single-parameter aggregated stream."""
        df_zone = agg_df[agg_df['Zone'] == self.target_zone].copy()
        if df_zone.empty:
            return df_zone, None
            
        df_zone = df_zone.sort_values('Time_Seconds_End')
        
        # Calculate statistical confidence if it isn't already in the dataframe
        if 'Binomial_%' not in df_zone.columns:
            df_zone['Binomial_%'] = df_zone['Count'].apply(
                lambda n: StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, n)
            )

        baseline_established = False
        kf = None
        filtered_vals = []
        detection_time = None

        for _, row in df_zone.iterrows():
            binom_conf = row['Binomial_%']
            
            # Phase 1: CALIBRATION
            if not baseline_established:
                if binom_conf >= self.target_confidence:
                    dynamic_baseline = row['Median_Val']
                    kf = AggregatedKalman(
                        initial_val=dynamic_baseline, 
                        persistence=self.agg_persistence,
                        abs_drift_threshold=self.agg_abs_drift
                    )
                    baseline_established = True
                    filtered_vals.append(dynamic_baseline)
                else:
                    filtered_vals.append(np.nan) 
                continue
                
            # Phase 2: ACTIVE DETECTION
            is_drift, x_val = kf.check_zone(self.target_zone, row['Median_Val'], binom_conf)
            filtered_vals.append(x_val)
            
            if is_drift and detection_time is None:
                detection_time = row['Time_Seconds_End']

        df_zone['Filtered_Value'] = filtered_vals
        return df_zone, detection_time
    
    def detect_combined_raw_stream(self, raw_df):
        """Processes a dual-parameter (pH & Temp) raw stream using independent filters."""
        df_zone = raw_df[raw_df['Zone'] == self.target_zone].copy()
        if df_zone.empty: return df_zone, {'ph': None, 'temp': None}
        df_zone = df_zone.sort_values('Time_Seconds')

        buf_ph, buf_temp = [], []
        baseline_established = False
        kf_ph, kf_temp = None, None
        filt_ph, filt_temp = [], []
        det_time = {'ph': None, 'temp': None}

        for _, row in df_zone.iterrows():
            # Phase 1: CALIBRATION
            if not baseline_established:
                buf_ph.append(row['Value_pH'])
                buf_temp.append(row['Value_Temp'])
                current_conf = StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, len(buf_ph))
                
                if current_conf >= self.target_confidence:
                    kf_ph = RawStreamKalman(
                        alpha=self.alpha, 
                        beta=self.beta, 
                        initial_val=np.median(buf_ph),
                        drift_threshold=self.raw_drift_threshold
                    )
                    kf_temp = RawStreamKalman(
                        alpha=self.alpha, 
                        beta=self.beta, 
                        initial_val=np.median(buf_temp),
                        drift_threshold=self.raw_drift_threshold
                    )
                    baseline_established = True
                    filt_ph = [np.median(buf_ph)] * len(buf_ph)
                    filt_temp = [np.median(buf_temp)] * len(buf_temp)
                else:
                    filt_ph.append(np.nan)
                    filt_temp.append(np.nan)
                continue
                
            # Phase 2: ACTIVE DETECTION
            is_drift_ph, x_ph = kf_ph.check_zone(self.target_zone, row['Value_pH'])
            is_drift_temp, x_temp = kf_temp.check_zone(self.target_zone, row['Value_Temp'])
            
            filt_ph.append(x_ph)
            filt_temp.append(x_temp)
            
            # Record detection times independently for pH and Temp
            if is_drift_ph and det_time['ph'] is None: det_time['ph'] = row['Time_Seconds']
            if is_drift_temp and det_time['temp'] is None: det_time['temp'] = row['Time_Seconds']

        df_zone['Filtered_pH'] = filt_ph
        df_zone['Filtered_Temp'] = filt_temp
        return df_zone, det_time

    def detect_combined_aggregated_stream(self, agg_df):
        """Processes a dual-parameter aggregated stream using independent adaptive filters."""
        df_zone = agg_df[agg_df['Zone'] == self.target_zone].copy()
        if df_zone.empty: return df_zone, {'ph': None, 'temp': None}
        df_zone = df_zone.sort_values('Time_Seconds_End')
        
        if 'Binomial_%' not in df_zone.columns:
            df_zone['Binomial_%'] = df_zone['Count'].apply(lambda n: StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, n))

        baseline_established = False
        kf_ph, kf_temp = None, None
        filt_ph, filt_temp = [], []
        det_time = {'ph': None, 'temp': None}

        for _, row in df_zone.iterrows():
            binom_conf = row['Binomial_%']
            
            # Phase 1: CALIBRATION
            if not baseline_established:
                if binom_conf >= self.target_confidence:
                    # Note the use of parameter-specific absolute drift thresholds
                    kf_ph = AggregatedKalman(
                        initial_val=row['Median_pH'], 
                        persistence=self.agg_persistence,
                        abs_drift_threshold=self.agg_abs_drift_ph
                    )
                    kf_temp = AggregatedKalman(
                        initial_val=row['Median_Temp'], 
                        persistence=self.agg_persistence,
                        abs_drift_threshold=self.agg_abs_drift_temp
                    )
                    
                    baseline_established = True
                    filt_ph.append(row['Median_pH'])
                    filt_temp.append(row['Median_Temp'])
                else:
                    filt_ph.append(np.nan)
                    filt_temp.append(np.nan)
                continue
                
            # Phase 2: ACTIVE DETECTION
            is_drift_ph, x_ph = kf_ph.check_zone(self.target_zone, row['Median_pH'], binom_conf)
            is_drift_temp, x_temp = kf_temp.check_zone(self.target_zone, row['Median_Temp'], binom_conf)
            
            filt_ph.append(x_ph)
            filt_temp.append(x_temp)
            
            if is_drift_ph and det_time['ph'] is None: det_time['ph'] = row['Time_Seconds_End']
            if is_drift_temp and det_time['temp'] is None: det_time['temp'] = row['Time_Seconds_End']

        df_zone['Filtered_pH'] = filt_ph
        df_zone['Filtered_Temp'] = filt_temp
        return df_zone, det_time