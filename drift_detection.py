import numpy as np

class RawStreamKalman:
    """
    Kalman Filter for RAW nanobot pings.
    Uses Alpha (Localization) and Beta (Measurement) for the noise model.
    """
    def __init__(self, alpha, beta, initial_val, q_process=1e-5, drift_threshold=2.5):
        # Combined uncertainty R = beta^2 (sensor) + alpha^2 (location mismatch)
        self.r = (beta**2) + (alpha**2)
        self.q = q_process
        self.drift_threshold = drift_threshold
        self.zones = {}
        self.initial_val = initial_val

    def check_zone(self, zone_id, raw_value):
        if zone_id not in self.zones:
            self.zones[zone_id] = {
                'x': self.initial_val, 
                'p': 1.0,              
                'cum_res': 0.0,
                'detected_at': None
            }
        
        z = self.zones[zone_id]

        # 1. Predict
        z['p'] = z['p'] + self.q

        # 2. Update
        residual = raw_value - z['x']
        k_gain = z['p'] / (z['p'] + self.r)
        
        z['x'] = z['x'] + k_gain * residual
        z['p'] = (1 - k_gain) * z['p']

        # 3. CUSUM-like Drift Detection
        z['cum_res'] += residual
        is_drift = abs(z['cum_res']) > self.drift_threshold
        
        if is_drift:
            # We don't reset cum_res completely to allow continuous tracking, 
            # but we damp it to prevent repeated triggers from one spike.
            z['cum_res'] *= 0.5 
            
        return is_drift, z['x']


class AggregatedKalman:
    """
    Adaptive 2nd-Order Kalman Filter for Aggregated Samples.
    Uses Binomial Confidence to dynamically weight the measurement.
    """
    def __init__(self, initial_val, q_velocity=1e-6, persistence=3):
        self.q = q_velocity
        self.persistence = persistence
        self.zones = {}
        self.initial_val = initial_val

    def check_zone(self, zone_id, agg_value, binom_conf):
        if zone_id not in self.zones:
            self.zones[zone_id] = {
                'x': np.array([[self.initial_val], [0.0]]), 
                'p': np.eye(2) * 10,
                'hit_counter': 0
            }
        
        z = self.zones[zone_id]
        
        # Mapping Binomial Confidence (0-100) to Measurement Noise R
        # Higher confidence -> Lower R (more trust)
        # We use a power of 2 to penalize low confidence heavily
        r_val = max(1e-6, ((100 - binom_conf) / 100)**2)
        
        # Matrices
        F = np.array([[1, 1], [0, 1]]) 
        H = np.array([[1, 0]])         
        Q = np.eye(2) * self.q         
        R = np.array([[r_val]])

        # 1. Predict
        z['x'] = F @ z['x']
        z['p'] = F @ z['p'] @ F.T + Q

        # 2. Update
        residual = agg_value - (H @ z['x'])[0,0]
        S = H @ z['p'] @ H.T + R
        k_gain = z['p'] @ H.T @ np.linalg.inv(S)

        z['x'] = z['x'] + k_gain * residual
        z['p'] = (np.eye(2) - k_gain @ H) @ z['p']

        # 3. Detection: Monitor the 'Velocity' state (x[1])
        # If the rate of change is consistently non-zero
        velocity = z['x'][1, 0]
        is_moving = abs(velocity) > 0.001 
        
        if is_moving:
            z['hit_counter'] += 1
        else:
            z['hit_counter'] = 0

        return (z['hit_counter'] >= self.persistence), z['x'][0, 0]