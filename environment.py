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
        
        # 64 Dimensions: 20 pH, 20 Temp, 20 Tox, and 4 High-Value Globals
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(64,), dtype=np.float32
        )
        self.engine = DigitalPatientEngine()
        
        # ---------------------------------------------------------
        # CONFIGURABLE CLINICAL & REWARD CONSTANTS
        # ---------------------------------------------------------
        # Default placeholder (Randomized properly in reset)
        self.toxicity_retention = 0.95
        
        # Version A Reward Configuration
        self.safe_toxicity_threshold = 30.0
        self.toxicity_scale = 0.05 
        self.efficacy_scale = 500.0
        self.critical_toxicity_threshold = 50.0 
        
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
        
        # Renamed from organ_recovery_rate for mathematical clarity
        self.toxicity_retention = np.random.uniform(0.93, 0.97)
        
        self.current_step = 0
        self.max_steps = 240
        self.current_dose = 0.0
        
        self.mu_B_ph = np.mean(self.session_data[0:15, 0:20], axis=0)
        self.mu_B_temp = np.mean(self.session_data[0:15, 20:40], axis=0)
        
        self.total_toxicity = 0.0
        
        tz_idx = self.tumor_zone - 1
        
        # Initialization strictly maps to the engine's starting tumor mass (0.85)
        initial_ph = self.mu_B_ph[tz_idx] + (self.engine.tumor_mass * -0.6)
        initial_temp = self.mu_B_temp[tz_idx] + (self.engine.tumor_mass * 1.0)
        
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
        
        # Capture pre-step mass
        prev_mass = self.engine.tumor_mass
        
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
            
        # Capture post-step mass and true delta
        new_mass = self.engine.tumor_mass
        delta_mass = new_mass - prev_mass
        
        clearance_idx = self.engine.drug['clearance_zone'] - 1
        clearance_zone_conc = self.engine.C[clearance_idx + 1]
        
        k_damage = 2.0 
        step_tox = k_damage * clearance_zone_conc
        
        # Uses toxicity_retention for mathematically clean decay
        self.total_toxicity = (self.total_toxicity * self.toxicity_retention) + step_tox      
        true_tox = self.engine.C[1:21]
        
        self.last_valid_ph = np.copy(true_ph)
        self.last_valid_temp = np.copy(true_temp)
        self.last_valid_tox = np.copy(true_tox)

        # Distance bookkeeping
        curr_dist = self._calc_tumor_distance(true_ph, true_temp)
        distance_reduction = self.prev_distance - curr_dist
        self.prev_distance = curr_dist
        
        # ---------------------------------------------------------
        # VERSION A: QUADRATIC HINGE & STRICT ENCAPSULATION
        # ---------------------------------------------------------
        efficacy_reward = self.efficacy_scale * distance_reduction
        distance_penalty = 0.15 * curr_dist 
        
        # Toxicity Penalty: Quadratic Hinge
        tox_overage = max(0.0, self.total_toxicity - self.safe_toxicity_threshold)
        toxicity_penalty = self.toxicity_scale * (tox_overage ** 2.0)
        
        smoothing_penalty = 2.5 * (dose_diff ** 2)
        
        reward_raw = (efficacy_reward - distance_penalty - toxicity_penalty - smoothing_penalty)
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps 
        
        terminal_penalty = 0.0
        if self.total_toxicity > self.critical_toxicity_threshold:
            terminated = True
            terminal_penalty = 100.0
            reward_raw -= terminal_penalty  
            
        reward_scaled = reward_raw / 100.0
        current_obs = self._get_obs()
        
        # --- LATENT TELEMETRY FOR DEBUGGING ---
        tz_idx = self.tumor_zone - 1
        ce_tumor = self.engine.Ce[tz_idx + 1]
        growth_ratio = (self.engine.tumor_growth_rate - 0.005) / 0.045
        ec50_factor = 0.60 - (0.30 * growth_ratio)
        ec50 = ec50_factor * self.engine.drug['C_safe']
        
        hill_fraction = (ce_tumor ** self.engine.gamma) / ((ec50 ** self.engine.gamma) + (ce_tumor ** self.engine.gamma) + 1e-8)
        k_eff = 1.0 + 0.2 * np.tanh(prev_mass)
        growth = self.engine.tumor_growth_rate * prev_mass * np.log(k_eff / max(1e-8, prev_mass))
        
        e_max = self.engine.drug.get('Emax', 0.08) 
        stage_multiplier = (1.1 - prev_mass)
        kill = e_max * hill_fraction * max(0.01, prev_mass) * stage_multiplier
        net_growth = growth - kill
        # --------------------------------------

        info = {
            'real_ph': true_ph[tz_idx], 
            'real_temp': true_temp[tz_idx],
            'clearance_temp': true_temp[clearance_idx],
            'clearance_ph': true_ph[clearance_idx],
            
            # Centralized Bookkeeping
            'Ce': ce_tumor,
            'hill': hill_fraction,
            'growth': growth,
            'kill': kill,
            'net_growth': net_growth,
            'mass': new_mass,
            'delta_mass': delta_mass,
            'curr_dist': curr_dist,
            'distance_reduction': distance_reduction,
            'step_tox': step_tox,
            'total_toxicity': self.total_toxicity,
            
            # Explicit Reward Variables for Diagnostic
            'efficacy_scale': self.efficacy_scale,
            'efficacy_reward': efficacy_reward,
            'distance_penalty': distance_penalty,
            'tox_overage': tox_overage,               
            'toxicity_penalty': toxicity_penalty,     
            'smoothing_penalty': smoothing_penalty,
            'terminal_penalty': terminal_penalty,
            'reward_raw': reward_raw,
            'reward_scaled': reward_scaled,
            'survived': not terminated
        }
        
        return current_obs, reward_scaled, terminated, truncated, info