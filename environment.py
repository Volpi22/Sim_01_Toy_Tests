import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from simulation import DigitalPatientEngine
import random

class PredictiveChemoEnv(gym.Env):
    """
    Real-Time Asynchronous Event-Driven DRQN Sensor Fusion Environment.
    Streams a single 63-D vector every minute. Recurrent memory layers (LSTM) 
    in the agent handle historical trajectories implicitly.
    """
    def __init__(self, schedule_parquet_path, drug_name='Doxorubicin', tumor_zone=14):
        super(PredictiveChemoEnv, self).__init__()
        
        self.schedule_df = pd.read_parquet(schedule_parquet_path)
        self.max_available_minute = len(self.schedule_df) - 255 
        
        self.drug_name = drug_name
        self.tumor_zone = tumor_zone
        
        # 0-3: Broad strokes for emergencies and stabilization
        # 4-9: Fine-tuned increments for precision targeting
        self.action_space = spaces.Discrete(10)
        self.action_mapping = {
            0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 
            4: 0.85, 5: 0.90, 6: 0.95, 7: 1.00, 
            8: 1.05, 9: 1.10
        }
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(63,), dtype=np.float32
        )
        
        self.engine = DigitalPatientEngine()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        start_min = random.randint(0, self.max_available_minute)
        end_min = start_min + 255 
        
        self.session_data = self.schedule_df.iloc[start_min:end_min].values
        
        patient_clearance_variance = np.random.uniform(0.80, 1.20)
        self.patient_clearance_factor = patient_clearance_variance
        
        self.engine = DigitalPatientEngine()
        self.engine.drug['k_elim'] *= patient_clearance_variance
        
        self.current_step = 0
        self.max_steps = 240
        self.current_dose = 0.0
        
        self.mu_B_ph = np.mean(self.session_data[0:15, 0:20], axis=0)
        self.mu_B_temp = np.mean(self.session_data[0:15, 20:40], axis=0)
        
        self.total_toxicity = 0.0
        self.critical_toxicity_threshold = 50.0 
        
        tz_idx = self.tumor_zone - 1
        initial_ph = self.mu_B_ph[tz_idx] - 0.6
        initial_temp = self.mu_B_temp[tz_idx] + 1.0
        
        self.prev_distance = np.sqrt(
            ((initial_ph - self.mu_B_ph[tz_idx]) / 0.05)**2 + 
            ((initial_temp - self.mu_B_temp[tz_idx]) / 0.2)**2
        )
        
        self.last_valid_ph = np.copy(self.session_data[15, 0:20])
        self.last_valid_temp = np.copy(self.session_data[15, 20:40])
        self.last_valid_tox = np.zeros(20)
        
        self.minutes_since_last_aggregation = 0
        self.next_aggregation_target = random.randint(10, 18)
        
        # --- NEW: BIOLOGICAL RESISTANCE FLAG ---
        self.is_resistant = False 
        
        return self._get_obs(confidence_flag=1.0), {}

    def _calc_tumor_distance(self, current_ph, current_temp):
        tz_idx = self.tumor_zone - 1
        z_ph = (current_ph[tz_idx] - self.mu_B_ph[tz_idx]) / 0.05
        z_temp = (current_temp[tz_idx] - self.mu_B_temp[tz_idx]) / 0.2
        return np.sqrt(z_ph**2 + z_temp**2)

    def _get_obs(self, confidence_flag):
        z_ph = (self.last_valid_ph - self.mu_B_ph) / 0.05
        z_temp = (self.last_valid_temp - self.mu_B_temp) / 0.2
        norm_tox = self.last_valid_tox / self.engine.drug['C_safe']
        norm_total_tox = self.total_toxicity / self.critical_toxicity_threshold
        
        obs = np.concatenate([
            z_ph, z_temp, norm_tox, 
            [norm_total_tox, self.current_dose, confidence_flag]
        ]).astype(np.float32)
        return obs

    def step(self, action):
        # --- CRITICAL CARE OVERRIDE ---
        if self.total_toxicity > self.critical_toxicity_threshold:
            action = 0

        old_dose = self.current_dose
        new_dose = self.action_mapping[action]
        dose_diff = new_dose - old_dose
        
        # Traceable Explanatory Wrapper
        if dose_diff <= -0.50:
            print(f"    -> [ALERT] Minute {self.current_step}: Emergency dose reduction ({dose_diff*100:.0f}%)!")
        elif dose_diff >= 0.50:
            print(f"    -> [ALERT] Minute {self.current_step}: High-dose increase ({dose_diff*100:.0f}%)!")
            
        self.current_dose = new_dose
        
        stress_factor = min(0.30, 0.30 * (self.total_toxicity / self.critical_toxicity_threshold))
        cardiac_multiplier = 1.0 + stress_factor
        
        self.engine.step_pharmacokinetics(self.current_dose, cardiac_multiplier=cardiac_multiplier)
        
        data_idx = 15 + self.current_step
        base_ph = self.session_data[data_idx, 0:20]
        base_temp = self.session_data[data_idx, 20:40]

        true_ph, true_temp = self.engine.apply_pharmacodynamics(
            base_ph_array=base_ph, 
            base_temp_array=base_temp, 
            tz_idx=self.tumor_zone - 1,
            max_ph_offset=-0.6,
            max_temp_offset=+1.0
        )
        
        # --- NEW: ABSOLUTE RESISTANCE OVERRIDE ---
        if self.is_resistant:
            tz_idx = self.tumor_zone - 1
            true_ph[tz_idx] = self.mu_B_ph[tz_idx] - 0.6  # Freeze acidic state
            true_temp[tz_idx] = self.mu_B_temp[tz_idx] + 1.0  # Freeze fever state
        
        # --- NEW: CONTINUOUS AUC TOXICITY ACCUMULATION ---
        # Biological Model: dX/dt = k_damage * C - k_repair * X
        clearance_zone_conc = self.engine.C[self.engine.drug['clearance_zone']]
        k_damage = 0.5 
        step_tox = k_damage * clearance_zone_conc

        self.total_toxicity = (self.total_toxicity * 0.99) + step_tox        
        
        true_tox = self.engine.C[1:21]
        
        self.minutes_since_last_aggregation += 1
        is_aggregation_event = (self.minutes_since_last_aggregation >= self.next_aggregation_target)
        
        if is_aggregation_event:
            confidence_flag = 1.0
            visible_ph = np.copy(true_ph)
            visible_temp = np.copy(true_temp)
            visible_tox = np.copy(true_tox)
            
            for z in range(20):
                if np.isnan(visible_ph[z]): visible_ph[z] = self.last_valid_ph[z]
                if np.isnan(visible_temp[z]): visible_temp[z] = self.last_valid_temp[z]
                if np.isnan(visible_tox[z]): visible_tox[z] = self.last_valid_tox[z]
                
            self.minutes_since_last_aggregation = 0
            self.next_aggregation_target = random.randint(10, 18)
            
            self.last_valid_ph = np.copy(visible_ph)
            self.last_valid_temp = np.copy(visible_temp)
            self.last_valid_tox = np.copy(visible_tox)
        else:
            confidence_flag = 0.0
            
            raw_noise_ph = np.random.normal(0, 0.02, 20)
            raw_noise_temp = np.random.normal(0, 0.05, 20)
            raw_noise_tox = np.random.normal(0, 0.05, 20)
            
            visible_ph = true_ph + raw_noise_ph
            visible_temp = true_temp + raw_noise_temp
            visible_tox = true_tox + raw_noise_tox
            
            for z in range(20):
                if np.isnan(base_ph[z]) or np.isnan(visible_ph[z]):
                    visible_ph[z] = self.last_valid_ph[z]
                else:
                    self.last_valid_ph[z] = visible_ph[z]
                    
                if np.isnan(base_temp[z]) or np.isnan(visible_temp[z]):
                    visible_temp[z] = self.last_valid_temp[z]
                else:
                    self.last_valid_temp[z] = visible_temp[z]
                    
                if np.isnan(visible_tox[z]):
                    visible_tox[z] = self.last_valid_tox[z]
                else:
                    self.last_valid_tox[z] = visible_tox[z]

        curr_dist = self._calc_tumor_distance(true_ph, true_temp)
        distance_reduction = self.prev_distance - curr_dist
        self.prev_distance = curr_dist
        
        sickness_penalty = -5.0 * curr_dist
        efficacy_reward = 500.0 * distance_reduction
        
        warning_threshold = 0.75 * self.critical_toxicity_threshold
        risk_ratio = self.total_toxicity / warning_threshold
        toxicity_penalty = 150.0 * step_tox * (risk_ratio ** 1.5)
        
        if dose_diff > 0: smoothing_penalty = 5.0 * (dose_diff ** 2)
        else: smoothing_penalty = 2.5 * (dose_diff ** 2)
            
        reward = efficacy_reward + sickness_penalty - toxicity_penalty - smoothing_penalty
        reward = reward / 100.0
        
        terminated = False 
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps 
            
        z_ph_all = (visible_ph - self.mu_B_ph) / 0.05
        z_temp_all = (visible_temp - self.mu_B_temp) / 0.2
        norm_tox_all = visible_tox / self.engine.drug['C_safe'] 
        norm_total_tox = self.total_toxicity / self.critical_toxicity_threshold
        
        current_obs = np.concatenate([
            z_ph_all, z_temp_all, norm_tox_all, 
            [norm_total_tox, self.current_dose, confidence_flag]
        ]).astype(np.float32)
        
        tz_idx = self.tumor_zone - 1
        clearance_idx = self.engine.drug['clearance_zone'] - 1
        
        info = {
            'real_ph': true_ph[tz_idx], 
            'real_temp': true_temp[tz_idx],
            'clearance_temp': true_temp[clearance_idx],
            'clearance_ph': true_ph[clearance_idx],
            'drug_concentration': true_tox[tz_idx], 
            'efficacy_reward': efficacy_reward,       
            'sickness_penalty': sickness_penalty,     
            'toxicity_penalty': toxicity_penalty,     
            'smoothing_penalty': smoothing_penalty
        }
        
        return current_obs, reward, terminated, truncated, info