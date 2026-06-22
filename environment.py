import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from simulation import DigitalPatientEngine
import random

class PredictiveChemoEnv(gym.Env):
    """
    Real-Time Asynchronous Event-Driven DRQN Sensor Fusion Environment.
    Streams a single 43-D vector every minute. Recurrent memory layers (LSTM) 
    in the agent handle historical trajectories implicitly.
    """
    def __init__(self, schedule_parquet_path, drug_name='Doxorubicin', tumor_zone=14):
        super(PredictiveChemoEnv, self).__init__()
        
        self.schedule_df = pd.read_parquet(schedule_parquet_path)
        self.max_available_minute = len(self.schedule_df) - 255 
        
        self.drug_name = drug_name
        self.tumor_zone = tumor_zone
        
        # Action Space: [0.90, 0.95, 1.0, 1.05, 1.10] of nominal dose
        self.action_space = spaces.Discrete(5)
        self.action_mapping = {0: 0.90, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.10}
        
        # State Space: A single minute's telemetry packet (43-D array)
        # Features: [pH(20), Temp(20), Toxicity(1), Dose(1), Confidence_Flag(1)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32)
        
        self.max_steps = 240 
        self.critical_toxicity_threshold = 50.0

    def reset(self, seed=None):
        self.engine = DigitalPatientEngine(drug_name=self.drug_name)
        self.current_step = 0
        self.total_toxicity = 0.0
        self.current_dose = 1.0 
        
        # Track asynchronous aggregation intervals dynamically
        self.minutes_since_last_aggregation = 0
        self.next_aggregation_target = random.randint(10, 18)
        
        start_idx = random.randint(0, self.max_available_minute)
        self.session_data = self.schedule_df.iloc[start_idx : start_idx + 255].values
        
        baseline_ph_buffer = []
        baseline_temp_buffer = []
        
        for i in range(15):
            raw_ph = self.session_data[i, 0:20]
            raw_temp = self.session_data[i, 20:40]
            baseline_ph_buffer.append(raw_ph)
            baseline_temp_buffer.append(raw_temp)
            
        self.mu_B_ph = np.nanmedian(baseline_ph_buffer, axis=0)
        self.mu_B_temp = np.nanmedian(baseline_temp_buffer, axis=0)

        # --- NEW: Inject the Warburg Effect (Pathology) ---
        self.disease_ph_offset = np.zeros(20)
        self.disease_temp_offset = np.zeros(20)
        
        tz_idx = self.tumor_zone - 1
        # Tumor is highly acidic (-0.6 pH) and hyperthermic (+1.0 C)
        self.disease_ph_offset[tz_idx] = -0.6
        self.disease_temp_offset[tz_idx] = +1.0
        
        # Add the disease to the baseline for our starting anchors
        diseased_start_ph = self.mu_B_ph + self.disease_ph_offset
        diseased_start_temp = self.mu_B_temp + self.disease_temp_offset
        
        # Establish anchor for clinical forward-filling
        self.last_valid_ph = np.copy(diseased_start_ph)
        self.last_valid_temp = np.copy(diseased_start_temp)
        
        self.prev_distance = self._calc_tumor_distance(diseased_start_ph, diseased_start_temp)
        
        # Return initial diseased observation packet (Confidence_Flag = 1.0)
        initial_obs = np.concatenate([diseased_start_ph, diseased_start_temp, [0.0, 1.0, 1.0]])
        return initial_obs.astype(np.float32), {}

    def _calc_tumor_distance(self, ph_array, temp_array):
        tz_idx = self.tumor_zone - 1
        z_ph = (ph_array[tz_idx] - self.mu_B_ph[tz_idx]) / 0.05
        z_temp = (temp_array[tz_idx] - self.mu_B_temp[tz_idx]) / 0.2
        return np.sqrt(z_ph**2 + z_temp**2)

    def step(self, action):
        new_dose = self.action_mapping[action]
        
        # Traceable Explanatory Wrapper
        if new_dose < self.current_dose:
            print(f"[CLINICAL ALERT] Minute {self.current_step}: AI Controller reduced infusion to {new_dose*100}%.")
            print(f"-> Rationale: Sequential trajectory analysis indicates predictive acceleration toward toxicity bounds.")
        
        self.current_dose = new_dose
        
        # Toxicity-Induced Hemodynamic Shift
        stress_factor = min(0.30, 0.30 * (self.total_toxicity / self.critical_toxicity_threshold))
        cardiac_multiplier = 1.0 + stress_factor
        
        # Step Physics Engine (1 minute)
        self.engine.step_pharmacokinetics(self.current_dose, cardiac_multiplier=cardiac_multiplier)
        
        data_idx = 15 + self.current_step
        base_ph = self.session_data[data_idx, 0:20]
        base_temp = self.session_data[data_idx, 20:40]

        # Apply the underlying disease pathology to the background noise
        diseased_ph = base_ph + self.disease_ph_offset
        diseased_temp = base_temp + self.disease_temp_offset
        
        # Continuous Pharmacodynamics (Drug acts on the diseased tissue)
        true_ph, true_temp = self.engine.apply_pharmacodynamics(diseased_ph, diseased_temp)
        
        # Track Toxicity metrics
        c_safe = self.engine.drug['C_safe']
        clearance_zone_conc = self.engine.C[self.engine.drug['clearance_zone']]
        step_tox = max(0, clearance_zone_conc - c_safe)
        self.total_toxicity += step_tox
        
        # Dynamic Event-Triggered Evaluation Loop
        self.minutes_since_last_aggregation += 1
        is_aggregation_event = (self.minutes_since_last_aggregation >= self.next_aggregation_target)
        
        if is_aggregation_event:
            # Mode A: High-confidence consensus achieved
            confidence_flag = 1.0
            visible_ph = np.copy(true_ph)
            visible_temp = np.copy(true_temp)
            
            # Cross-check for transient dropouts using forward-filling
            for z in range(20):
                if np.isnan(visible_ph[z]): visible_ph[z] = self.last_valid_ph[z]
                if np.isnan(visible_temp[z]): visible_temp[z] = self.last_valid_temp[z]
                
            self.minutes_since_last_aggregation = 0
            self.next_aggregation_target = random.randint(10, 18)
            
            self.last_valid_ph = np.copy(visible_ph)
            self.last_valid_temp = np.copy(visible_temp)
        else:
            # Mode B: Swarm is accumulating packets. Stream raw 1-minute noisy telemetry.
            confidence_flag = 0.0
            
            raw_noise_ph = np.random.normal(0, 0.02, 20)
            raw_noise_temp = np.random.normal(0, 0.05, 20)
            visible_ph = true_ph + raw_noise_ph
            visible_temp = true_temp + raw_noise_temp
            
            # Forward-Filling for dropouts
            for z in range(20):
                if np.isnan(base_ph[z]) or np.isnan(visible_ph[z]):
                    visible_ph[z] = self.last_valid_ph[z]
                else:
                    self.last_valid_ph[z] = visible_ph[z]
                    
                if np.isnan(base_temp[z]) or np.isnan(visible_temp[z]):
                    visible_temp[z] = self.last_valid_temp[z]
                else:
                    self.last_valid_temp[z] = visible_temp[z]

        # Calculate Reward based on ground truth
        curr_dist = self._calc_tumor_distance(true_ph, true_temp)
        efficacy = -(curr_dist - self.prev_distance)
        self.prev_distance = curr_dist
        
        reward = (10.0 * efficacy) - (5.0 * step_tox)
        
        self.current_step += 1
        terminated = self.total_toxicity > self.critical_toxicity_threshold 
        truncated = self.current_step >= self.max_steps 
        
        if terminated: 
            reward -= 500
            
        current_obs = np.concatenate([visible_ph, visible_temp, [self.total_toxicity, self.current_dose, confidence_flag]])
        
        return current_obs.astype(np.float32), reward, terminated, truncated, {}