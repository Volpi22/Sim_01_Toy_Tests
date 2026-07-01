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
        
        # State Space: A single minute's telemetry packet (5-D array)
        # Features: [Tumor_pH(1), Tumor_Temp(1), Toxicity(1), Dose(1), Confidence_Flag(1)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
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
        
        # Establish anchor for clinical forward-filling
        self.last_valid_ph = np.copy(diseased_start_ph)
        self.last_valid_temp = np.copy(diseased_start_temp)
        
        self.prev_distance = self._calc_tumor_distance(diseased_start_ph, diseased_start_temp)
        
        # Normalizing the inputs (Z-scores for telemetry, fractions for dose/tox)
        z_ph = (diseased_start_ph[tz_idx] - self.mu_B_ph[tz_idx]) / 0.05
        z_temp = (diseased_start_temp[tz_idx] - self.mu_B_temp[tz_idx]) / 0.2
        norm_tox = self.total_toxicity / self.critical_toxicity_threshold
        
        initial_obs = np.array([z_ph, z_temp, norm_tox, self.current_dose, 1.0], dtype=np.float32)
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
            print(f"    -> [CLINICAL ALERT] Minute {self.current_step}: Massive emergency dose reduction ({dose_diff*100:.0f}%)!")
            
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
            self.next_aggregation_target = random.randint(12, 16)
            
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
        
        # 1. Therapeutic Efficacy (Velocity of healing)
        efficacy = -(curr_dist - self.prev_distance)
        self.prev_distance = curr_dist
        
        # 2. The "Pathological Footprint" Penalty
        disease_penalty = 1.0 * curr_dist
        
        # We set the reference denominator to 75% of the max threshold to account for drug inertia.
        # At exactly 75% toxicity, the ratio is 1.0, and the penalty multiplier (200.0) 
        # cancels out the maximum efficacy reward (200.0), creating a mathematical equilibrium.
        warning_threshold = 0.75 * self.critical_toxicity_threshold
        risk_ratio = self.total_toxicity / warning_threshold
        
        # If toxicity > 75%, risk_ratio > 1, and the cubic exponent makes the penalty explode.
        toxicity_penalty = 200.0 * step_tox * (risk_ratio ** 3)
        
        # 3. Asymmetric & Quadratic Shock Penalty
        if dose_diff > 0:
            smoothing_penalty = 50.0 * (dose_diff ** 2)
        else:
            smoothing_penalty = 25.0 * (dose_diff ** 2)
            
        # 4. Total Reward Calculation
        reward = (200.0 * efficacy) - disease_penalty - toxicity_penalty - smoothing_penalty
        
        self.current_step += 1
        terminated = self.total_toxicity > self.critical_toxicity_threshold 
        truncated = self.current_step >= self.max_steps 
        
        if terminated: 
            # Death penalty
            reward -= 10000 
            
        # Normalizing the inputs for the neural network
        tz_idx = self.tumor_zone - 1
        z_ph = (visible_ph[tz_idx] - self.mu_B_ph[tz_idx]) / 0.05
        z_temp = (visible_temp[tz_idx] - self.mu_B_temp[tz_idx]) / 0.2
        norm_tox = self.total_toxicity / self.critical_toxicity_threshold
        
        current_obs = np.array([z_ph, z_temp, norm_tox, self.current_dose, confidence_flag], dtype=np.float32)
        
        # Pass the real unnormalized values back so the evaluation dashboard can plot them
        info = {'real_ph': true_ph[tz_idx], 'real_temp': true_temp[tz_idx]}
        
        return current_obs, reward, terminated, truncated, info