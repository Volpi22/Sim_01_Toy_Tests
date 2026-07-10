import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from simulation import DigitalPatientEngine
import random
from utils import debug

class PredictiveChemoEnv(gym.Env):
    def __init__(self, schedule_parquet_path, drug_name='Doxorubicin', tumor_zone=14):
        super(PredictiveChemoEnv, self).__init__()
        
        self.schedule_df = pd.read_parquet(schedule_parquet_path)
        self.max_available_minute = len(self.schedule_df) - 255 
        
        self.drug_name = drug_name
        self.tumor_zone = tumor_zone
        
        self.action_space = spaces.Discrete(10)
        self.action_mapping = {
            0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 
            4: 0.85, 5: 0.90, 6: 0.95, 7: 1.00, 
            8: 1.05, 9: 1.10
        }
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(64,), dtype=np.float32
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
        
        self.patient_clearance_factor = np.random.uniform(0.80, 1.20)
        
        self.engine = DigitalPatientEngine()
        self.engine.drug['k_elim'] *= self.patient_clearance_factor
        
        self.engine.tumor_growth_rate = np.random.uniform(0.015, 0.035)
        self.organ_recovery_rate = np.random.uniform(0.93, 0.97)
        
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
        
        self.is_resistant = False 
        return self._get_obs(), {}

    def _calc_tumor_distance(self, current_ph, current_temp):
        tz_idx = self.tumor_zone - 1
        z_ph = (current_ph[tz_idx] - self.mu_B_ph[tz_idx]) / 0.05
        z_temp = (current_temp[tz_idx] - self.mu_B_temp[tz_idx]) / 0.2
        return np.sqrt(z_ph**2 + z_temp**2)

    def _get_obs(self):
        z_ph = (self.last_valid_ph - self.mu_B_ph) / 0.05
        z_temp = (self.last_valid_temp - self.mu_B_temp) / 0.2
        norm_tox = self.last_valid_tox / self.engine.drug['C_safe']
        norm_total_tox = self.total_toxicity / self.critical_toxicity_threshold
        
        clearance_idx = self.engine.drug['clearance_zone'] - 1
        clearance_conc = self.engine.C[clearance_idx + 1] / self.engine.drug['C_safe']
        
        obs = np.concatenate([
            z_ph, z_temp, norm_tox, 
            [norm_total_tox, self.current_dose, clearance_conc, self.patient_clearance_factor]
        ]).astype(np.float32)
        return obs

    def step(self, action):
        old_dose = self.current_dose
        new_dose = self.action_mapping[action]
        dose_diff = new_dose - old_dose
        
        if debug:
            if dose_diff <= -0.50:
                print(f"    -> [ALERT] Minute {self.current_step}: Emergency dose reduction ({dose_diff*100:.0f}%) to prevent systemic toxicity crash!")
            elif dose_diff >= 0.50:
                print(f"    -> [ALERT] Minute {self.current_step}: High-dose aggressive escalation ({dose_diff*100:.0f}%) to suppress tumor regrowth!")
            
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
            mu_B_ph=self.mu_B_ph,
            mu_B_temp=self.mu_B_temp,
            tz_idx=self.tumor_zone - 1,
            max_ph_offset=-0.6,
            max_temp_offset=+1.0
        )
        
        if self.is_resistant:
            tz_idx = self.tumor_zone - 1
            true_ph[tz_idx] = self.mu_B_ph[tz_idx] - 0.6  
            true_temp[tz_idx] = self.mu_B_temp[tz_idx] + 1.0  
        
        clearance_idx = self.engine.drug['clearance_zone'] - 1
        clearance_zone_conc = self.engine.C[clearance_idx + 1]
        
        k_damage = 2.0 
        step_tox = k_damage * clearance_zone_conc
        self.total_toxicity = (self.total_toxicity * self.organ_recovery_rate) + step_tox      
        true_tox = self.engine.C[1:21]
        
        self.last_valid_ph = np.copy(true_ph)
        self.last_valid_temp = np.copy(true_temp)
        self.last_valid_tox = np.copy(true_tox)

        curr_dist = self._calc_tumor_distance(true_ph, true_temp)
        distance_reduction = self.prev_distance - curr_dist
        self.prev_distance = curr_dist
        
        risk_ratio = self.total_toxicity / self.critical_toxicity_threshold
        
        reward = (
            500.0 * distance_reduction 
            - 0.15 * curr_dist                           
            - 150.0 * step_tox * (risk_ratio ** 2.0)    
            - 2.5 * (dose_diff ** 2)                    
        )
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps 
        
        if self.total_toxicity > self.critical_toxicity_threshold:
            terminated = True
            reward -= 100.0  
            
        reward = reward / 100.0
        current_obs = self._get_obs()
        
        tz_idx = self.tumor_zone - 1
        info = {
            'real_ph': true_ph[tz_idx], 
            'real_temp': true_temp[tz_idx],
            'clearance_temp': true_temp[clearance_idx],
            'clearance_ph': true_ph[clearance_idx],
            'efficacy_reward': 500.0 * distance_reduction,       
            'toxicity_penalty': 150.0 * step_tox * (risk_ratio ** 2.0),     
            'smoothing_penalty': 2.5 * (dose_diff ** 2)
        }
        
        return current_obs, reward, terminated, truncated, info