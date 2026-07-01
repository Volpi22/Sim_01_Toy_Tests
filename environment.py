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
        
        # 0-3: Broad strokes for emergencies and stabilization
        # 4-9: Fine-tuned increments for precision targeting
        self.action_space = spaces.Discrete(10)
        self.action_mapping = {
            0: 0.00,  # Emergency Stop (Pause Infusion)
            1: 0.25,  # Quarter Dose
            2: 0.50,  # Half Dose
            3: 0.75,  # Three-Quarter Dose
            4: 0.85,  # Precision adjustments:
            5: 0.90,
            6: 0.95,
            7: 1.00,  # Nominal Prescribed Dose
            8: 1.05,
            9: 1.10
        }
        
        # State Space: A single minute's telemetry packet (63-D array)
        # Features: [pH_1..20, Temp_1..20, Tox_1..20, Total_Sys_Tox(1), Dose(1), Confidence_Flag(1)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(63,), dtype=np.float32)
        
        self.max_steps = 240 
        self.critical_toxicity_threshold = 50.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomizes the patient's metabolic clearance rate by +/- 20%
        # A value of 0.80 means sluggish kidneys; 1.20 means highly efficient clearance.
        self.patient_clearance_factor = np.random.uniform(0.80, 1.20)
        
        self.engine = DigitalPatientEngine(drug_name=self.drug_name)
        
        # Mutates the engine's copy of the drug constants for this episode's specific patient.
        if hasattr(self.engine, 'drug') and 'k_elim' in self.engine.drug:
            self.engine.drug['k_elim'] *= self.patient_clearance_factor
        
        self.current_step = 0
        self.total_toxicity = 0.0
        self.current_dose = 0.0
        
        # Track asynchronous aggregation intervals dynamically
        self.minutes_since_last_aggregation = 0
        self.next_aggregation_target = random.randint(12, 16)
        
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

        # Inject the Warburg Effect (Pathology) ---
        self.disease_ph_offset = np.zeros(20)
        self.disease_temp_offset = np.zeros(20)
        
        tz_idx = self.tumor_zone - 1
        # Tumor is highly acidic (-0.6 pH) and hyperthermic (+1.0 C)
        self.disease_ph_offset[tz_idx] = -0.6
        self.disease_temp_offset[tz_idx] = +1.0
        
        # Add the disease to the baseline for our starting anchors
        diseased_start_ph = self.mu_B_ph + self.disease_ph_offset
        diseased_start_temp = self.mu_B_temp + self.disease_temp_offset
        
        # Establish anchors for clinical forward-filling
        self.last_valid_ph = np.copy(diseased_start_ph)
        self.last_valid_temp = np.copy(diseased_start_temp)
        self.last_valid_tox = np.zeros(20)
        
        self.prev_distance = self._calc_tumor_distance(diseased_start_ph, diseased_start_temp)
        
        # Whole-Body Normalization
        z_ph_all = (diseased_start_ph - self.mu_B_ph) / 0.05
        z_temp_all = (diseased_start_temp - self.mu_B_temp) / 0.2
        norm_tox_all = np.zeros(20) # 0 dose at start
        norm_total_tox = self.total_toxicity / self.critical_toxicity_threshold
        
        # 63-D Initial Observation
        initial_obs = np.concatenate([
            z_ph_all, z_temp_all, norm_tox_all, 
            [norm_total_tox, self.current_dose, 1.0]
        ]).astype(np.float32)
        
        return initial_obs, {}

    def _calc_tumor_distance(self, ph_array, temp_array):
        tz_idx = self.tumor_zone - 1
        z_ph = (ph_array[tz_idx] - self.mu_B_ph[tz_idx]) / 0.05
        z_temp = (temp_array[tz_idx] - self.mu_B_temp[tz_idx]) / 0.2
        return np.sqrt(z_ph**2 + z_temp**2)

    def step(self, action):
        old_dose = self.current_dose
        new_dose = self.action_mapping[action]
        dose_diff = new_dose - old_dose
        
        # Traceable Explanatory Wrapper
        if dose_diff <= -0.50:
            print(f"    -> [ALERT] Minute {self.current_step}: Emergency dose reduction ({dose_diff*100:.0f}%)!")
        elif dose_diff >= 0.50:
            print(f"    -> [ALERT] Minute {self.current_step}: High-dose increase ({dose_diff*100:.0f}%)!")
            
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
        
        # Track Toxicity metrics (WITH CLEARANCE DECAY)
        c_safe = self.engine.drug['C_safe']
        clearance_zone_conc = self.engine.C[self.engine.drug['clearance_zone']]
        step_tox = max(0, clearance_zone_conc - c_safe)
        
        # Ground Truth Instantaneous Toxicity (Concentration in all 20 zones)
        true_tox = self.engine.C[1:21]
        
        # Dynamic Event-Triggered Evaluation Loop (Heuristic Timer)
        self.minutes_since_last_aggregation += 1
        is_aggregation_event = (self.minutes_since_last_aggregation >= self.next_aggregation_target)
        
        if is_aggregation_event:
            # Mode A: High-confidence consensus achieved
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
            # Mode B: Swarm is accumulating packets. Stream raw noisy telemetry.
            confidence_flag = 0.0
            
            raw_noise_ph = np.random.normal(0, 0.02, 20)
            raw_noise_temp = np.random.normal(0, 0.05, 20)
            raw_noise_tox = np.random.normal(0, 0.05, 20) # Same hardware error profile as temp
            
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

        # --- KEEP REWARD CALCULATION EXACTLY THE SAME ---
        curr_dist = self._calc_tumor_distance(true_ph, true_temp)
        distance_reduction = self.prev_distance - curr_dist
        self.prev_distance = curr_dist
        efficacy_reward = 200.0 * distance_reduction
        
        warning_threshold = 0.75 * self.critical_toxicity_threshold
        risk_ratio = self.total_toxicity / warning_threshold
        toxicity_penalty = 200.0 * step_tox * (risk_ratio ** 3)
        
        if dose_diff > 0: smoothing_penalty = 50.0 * (dose_diff ** 2)
        else: smoothing_penalty = 25.0 * (dose_diff ** 2)
            
        reward = efficacy_reward - toxicity_penalty - smoothing_penalty
        
        self.current_step += 1
        terminated = self.total_toxicity > self.critical_toxicity_threshold 
        truncated = self.current_step >= self.max_steps 
        
        if terminated: reward -= 500.0 
            
        # Normalizing Whole-Body Inputs
        z_ph_all = (visible_ph - self.mu_B_ph) / 0.05
        z_temp_all = (visible_temp - self.mu_B_temp) / 0.2
        # Normalize local toxicity relative to the safe limit
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
            'clearance_temp': true_temp[clearance_idx]
        }
        
        return current_obs, reward, terminated, truncated, info