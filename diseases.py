import pandas as pd
import numpy as np
import random
import os
from utils import DataManager, StatsUtils, EnergyModel

class DiseaseSimulator:
    """
    Simulates physiological changes over time due to localized pathological conditions.
    Applies a time-scaled linear progression to simulate disease onset and peak severity,
    while simultaneously modeling nanobot energy constraints and sampling errors.
    """
    
    # BUFFER_TIME ensures the simulation continues for at least 2 hours after the disease peaks
    BUFFER_TIME = 7200 

    def __init__(self, df_path='position_data.csv'):
        """Initializes the simulator with anatomical mappings and disease profiles."""
        self.dm = DataManager()
        self.df_path = df_path
        self.zone_to_vessels = self.dm.ZONE_TO_VESSELS
        
        # --- DISEASE PROFILES ---
        # Defines target zones, parameter shifts, and realistic progression velocities (in seconds)
        all_zones = list(range(1, 21))
        
        self.diseases = {
            # ==========================================
            # 1. pH DEVIATIONS
            # ==========================================
            'systemic_alkalosis_ph': {
                'type': 'single', 'param': 'ph',
                'zones': all_zones,                                 # All Zones
                'shifts': {'ph': +0.20},                            # Hyperventilation/Alkalosis
                'progression_time': 7200                            # 2 hours (Fast)
            },
            'dka_ph': {
                'type': 'single', 'param': 'ph',
                'zones': [1, 2, 3, 4, 5],                           # Core/Aorta
                'shifts': {'ph': -0.30},                            # Diabetic Ketoacidosis
                'progression_time': 21600                           # 6 hours (Moderate)
            },
            'stroke_ph': {
                'type': 'single', 'param': 'ph',
                'zones': [7],                                       # Head Microcirculation
                'shifts': {'ph': -0.25},                            # Severe localized acidosis
                'progression_time': 3600                            # 1 hour (Very Fast / Acute)
            },
            'nec_ph': {
                'type': 'single', 'param': 'ph',
                'zones': [12],                                      # Intestinal Microcirculation
                'shifts': {'ph': -0.25},                            # Necrotizing Enterocolitis
                'progression_time': 14400                           # 4 hours (Moderate-Fast)
            },
            'rta_ph': {
                'type': 'single', 'param': 'ph',
                'zones': [16],                                      # Renal
                'shifts': {'ph': -0.15},                            # Renal Tubular Acidosis
                'progression_time': 43200                           # 12 hours (Slow)
            },

            # ==========================================
            # 2. TEMPERATURE DEVIATIONS
            # ==========================================
            'systemic_fever_temp': {
                'type': 'single', 'param': 'temp',
                'zones': all_zones,                                 # All Zones
                'shifts': {'temp': 2.5},                            # Systemic Fever
                'progression_time': 14400                           # 4 hours (Moderate-Fast)
            },
            'endocarditis_temp': {
                'type': 'single', 'param': 'temp',
                'zones': [1, 2, 3],                                 # Heart
                'shifts': {'temp': 1.5},                            # Infective Endocarditis
                'progression_time': 43200                           # 12 hours (Slow)
            },
            'meningitis_temp': {
                'type': 'single', 'param': 'temp',
                'zones': [8],                                       # Head Venous
                'shifts': {'temp': 2.0},                            # Meningitis
                'progression_time': 10800                           # 3 hours (Fast)
            },
            'arterial_occlusion_temp': {
                'type': 'single', 'param': 'temp',
                'zones': [19],                                      # Lower Microcirculation
                'shifts': {'temp': -3.0},                           # Acute Arterial Occlusion (Cold Limb)
                'progression_time': 5400                            # 1.5 hours (Very Fast)
            },
            'dvt_temp': {
                'type': 'single', 'param': 'temp',
                'zones': [19],                                      # Lower Microcirculation
                'shifts': {'temp': 1.8},                            # Deep Vein Thrombosis
                'progression_time': 28800                           # 8 hours (Moderate-Slow)
            },

            # ==========================================
            # 3. COMBINED DEVIATIONS (pH + Temp)
            # ==========================================
            'septic_shock_combined': {
                'type': 'combined', 
                'zones': all_zones,                                 # All Zones
                'shifts': {'ph': -0.35, 'temp': 2.8},               # Severe Sepsis (Lactic Acid + Fever)
                'progression_time': 10800                           # 3 hours (Fast/Critical)
            },
            'malignant_hyperthermia_combined': {
                'type': 'combined', 
                'zones': [1, 2, 3, 4, 5],                           # Core/Aorta
                'shifts': {'ph': -0.40, 'temp': 4.0},               # Extreme Hypermetabolic State
                'progression_time': 1800                            # 30 mins (Extremely Fast/Lethal)
            },
            'ischemic_bowel_combined': {
                'type': 'combined', 
                'zones': [12],                                      # Intestinal Microcirculation
                'shifts': {'ph': -0.30, 'temp': 2.0},               # Bowel Necrosis + Peritonitis
                'progression_time': 14400                           # 4 hours (Moderate)
            },
            'pyelo_combined': {
                'type': 'combined', 
                'zones': [16],                                      # Renal
                'shifts': {'ph': -0.2, 'temp': 2.0},                # Acute Pyelonephritis (Acidosis + Fever)
                'progression_time': 21600                           # 6 hours (Moderate)
            },
            'ischemia_combined': {
                'type': 'combined', 
                'zones': [19],                                      # Lower Microcirculation
                'shifts': {'ph': -0.3, 'temp': -2.5},               # Advanced Acute Limb Ischemia
                'progression_time': 7200                            # 2 hours (Fast)
            }
        }

    # ==========================================
    # INTERNAL HELPERS
    # ==========================================

    def _load_data(self):
        if not os.path.exists(self.df_path):
            raise FileNotFoundError(f"Missing input: {self.df_path}")
        cols = ['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID']
        df = pd.read_csv(self.df_path, header=None, names=cols)
        df.dropna(inplace=True)
        df.sort_values(by='Time_Seconds', inplace=True)
        return df

    def _update_bot_energy(self, state, t, energy_model):
        delta_t = t - state['last_time']
        state['last_time'] = t
        
        if delta_t > 0:
            state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
            state['energy'] = max(0, state['energy'] - (energy_model.idle_drain_per_sec * delta_t))
            
        if state['energy'] <= energy_model.E_off: state['is_awake'] = False
        elif state['energy'] >= energy_model.E_on: state['is_awake'] = True
        return state

    def _get_affected_vessels(self, zones):
        vessels = []
        for z in zones:
            vessels.extend(self.zone_to_vessels.get(z, []))
        return vessels

    def _get_diseased_value(self, data_dict, vessel_id, beta, t, affected_vessels, shift_val, t_start, t_peak):
        if vessel_id not in data_dict:
            chosen_id = random.choice(list(data_dict.keys()))
        elif random.random() < beta:
            available_ids = list(data_dict.keys())
            available_ids.remove(vessel_id)
            chosen_id = random.choice(available_ids)
        else:
            chosen_id = vessel_id
            
        min_val, max_val = data_dict[chosen_id]
        
        if chosen_id in affected_vessels and t > t_start:
            f = min(1.0, (t - t_start) / (t_peak - t_start))
            current_shift = shift_val * f
            min_val += current_shift
            max_val += current_shift
            
        return random.uniform(min_val, max_val)

    def _generate_disease_timeline(self, max_t, progression_time, safe_start_time=14400):
        """
        Dynamically calculates the onset and peak based on the specific disease's progression velocity.
        """
        latest_possible_start = max(safe_start_time, max_t - progression_time - self.BUFFER_TIME)
        
        if latest_possible_start <= safe_start_time:
            t_start = safe_start_time
        else:
            t_start = random.uniform(safe_start_time, latest_possible_start)
            
        t_peak = t_start + progression_time
        return t_start, t_peak

    # ==========================================
    # SIMULATION EXECUTION
    # ==========================================

    def run_single_disease(self, alpha, beta, disease_key, target_confidence=70.0, safe_start_time=14400):
        if disease_key not in self.diseases:
            print(f"Error: Disease '{disease_key}' not found."); return None, None, None
            
        profile = self.diseases[disease_key]
        param = profile['param']
        progression_time = profile['progression_time'] # Extract specific velocity
        
        data_dict = self.dm.get_data_dict(param)
        precision = self.dm.get_decimal_precision(param)
        affected_vessels = self._get_affected_vessels(profile['zones'])
        shift_val = profile['shifts'][param]

        df = self._load_data()
        
        # Pass the dynamic progression time
        t_start, t_peak = self._generate_disease_timeline(df['Time_Seconds'].max(), progression_time, safe_start_time)
        print(f"[{disease_key.upper()}] Velocity: {progression_time/3600:.1f}h | Random Onset: Hour {t_start/3600:.1f} | Peak: Hour {t_peak/3600:.1f}")
        
        energy_model = EnergyModel(param_type=param)
        
        bot_states, buffers = {}, {}
        final_results, raw_results = [], []
        sample_id = 1
        conf_cache = {}

        for row in df.itertuples():
            nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
            if pd.isna(vessel): continue
                
            if nid not in bot_states:
                bot_states[nid] = {'energy': energy_model.E_max, 'is_awake': True, 'has_sampled_this_cycle': False, 'last_time': t, 'previous_vessel': None}
            
            state = self._update_bot_energy(bot_states[nid], t, energy_model)
            
            if vessel == 2 and state['previous_vessel'] != 2: state['has_sampled_this_cycle'] = False
                
            if state['is_awake'] and not state['has_sampled_this_cycle'] and random.random() < 0.05:
                val = self._get_diseased_value(data_dict, vessel, beta, t, affected_vessels, shift_val, t_start, t_peak)
                key = self.dm.assign_zone(vessel)
                
                raw_results.append({'Time_Seconds': t, 'Zone': key, 'Value': round(val, precision)})
                
                buffers.setdefault(key, []).append(val)
                state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
                state['has_sampled_this_cycle'] = True
                
                if len(buffers) == 20: 
                    if all(conf_cache.setdefault(len(v), StatsUtils.obtain_confidence_binomial(alpha, beta, len(v))) >= target_confidence for v in buffers.values()):
                        for k, vals in buffers.items():
                            final_results.append({
                                'Sample_ID': sample_id, 'Time_Seconds_End': t, 'Zone': k, 
                                'Median_Val': round(pd.Series(vals).median(), precision), 'Count': len(vals)
                            })
                        buffers, sample_id = {}, sample_id + 1
                        
            state['previous_vessel'] = vessel
                    
        return pd.DataFrame(raw_results), pd.DataFrame(final_results), t_start

    def run_combined_disease(self, alpha, beta, disease_key, target_confidence=70.0, safe_start_time=14400):
        if disease_key not in self.diseases:
            print(f"Error: Disease '{disease_key}' not found."); return None, None, None

        profile = self.diseases[disease_key]
        progression_time = profile['progression_time'] # Extract specific velocity
        affected_vessels = self._get_affected_vessels(profile['zones'])
        ph_shift, temp_shift = profile['shifts']['ph'], profile['shifts']['temp']
        
        ph_dict = self.dm.get_data_dict('ph')
        temp_dict = self.dm.get_data_dict('temp')

        df = self._load_data()
        
        # Pass the dynamic progression time
        t_start, t_peak = self._generate_disease_timeline(df['Time_Seconds'].max(), progression_time, safe_start_time)
        print(f"[{disease_key.upper()}] Velocity: {progression_time/3600:.1f}h | Random Onset: Hour {t_start/3600:.1f} | Peak: Hour {t_peak/3600:.1f}")
        
        energy_model = EnergyModel(param_type='combined')
        
        bot_states, buffers = {}, {}
        final_results, raw_results = [], []
        sample_id = 1
        conf_cache = {}

        for row in df.itertuples():
            nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
            if pd.isna(vessel): continue
                
            if nid not in bot_states:
                bot_states[nid] = {'energy': energy_model.E_max, 'is_awake': True, 'has_sampled_this_cycle': False, 'last_time': t, 'previous_vessel': None}
            
            state = self._update_bot_energy(bot_states[nid], t, energy_model)
            if vessel == 2 and state['previous_vessel'] != 2: state['has_sampled_this_cycle'] = False
                
            if state['is_awake'] and not state['has_sampled_this_cycle'] and random.random() < 0.05:
                val_ph = self._get_diseased_value(ph_dict, vessel, beta, t, affected_vessels, ph_shift, t_start, t_peak)
                val_temp = self._get_diseased_value(temp_dict, vessel, beta, t, affected_vessels, temp_shift, t_start, t_peak)
                
                key = self.dm.assign_zone(vessel)
                
                raw_results.append({
                    'Time_Seconds': t, 'Zone': key, 
                    'Value_pH': round(val_ph, 2), 'Value_Temp': round(val_temp, 3)
                })
                
                buf = buffers.setdefault(key, {'ph': [], 'temp': []})
                buf['ph'].append(val_ph)
                buf['temp'].append(val_temp)
                
                state['energy'] = max(0, state['energy'] - energy_model.action_consumption('combined'))
                state['has_sampled_this_cycle'] = True
                
                if len(buffers) == 20:
                    if all(conf_cache.setdefault(len(v['ph']), StatsUtils.obtain_confidence_binomial(alpha, beta, len(v['ph']))) >= target_confidence for v in buffers.values()):
                        for k, vals in buffers.items():
                            final_results.append({
                                'Sample_ID': sample_id, 'Time_Seconds_End': t, 'Zone': k, 
                                'Median_pH': round(pd.Series(vals['ph']).median(), 2),
                                'Median_Temp': round(pd.Series(vals['temp']).median(), 3), 
                                'Count': len(vals['ph'])
                            })
                        buffers, sample_id = {}, sample_id + 1
                        
            state['previous_vessel'] = vessel
                    
        return pd.DataFrame(raw_results), pd.DataFrame(final_results), t_start

if __name__ == "__main__":
    pass