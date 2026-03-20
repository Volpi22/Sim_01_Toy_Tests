import pandas as pd
import numpy as np
import random
import os
from utils import DataManager, StatsUtils, EnergyModel

class DiseaseSimulator:
    """
    Simulates physiological changes over time due to localized pathological conditions.
    Applies a time-scaled linear progression to simulate disease onset and peak severity.
    """
    
    # Time constants (in seconds)
    PROGRESSION_TIME = 28800  # 8 hours to reach peak disease severity
    BUFFER_TIME = 7200        # 2 hours minimum at peak severity before simulation ends

    def __init__(self, df_path='position_data.csv'):
        self.dm = DataManager()
        self.df_path = df_path
        self.zone_to_vessels = self.dm.ZONE_TO_VESSELS
        
        # --- DISEASE PROFILES ---
        # Defines the target zones and the absolute shift in physiological parameters at peak severity.
        self.diseases = {
            'stroke_ph': {
                'type': 'single', 'param': 'ph',
                'zones': [7],                     # Head Microcirculation
                'shifts': {'ph': -0.25}           # Severe localized acidosis
            },
            'dvt_temp': {
                'type': 'single', 'param': 'temp',
                'zones': [19],                    # Lower Microcirculation
                'shifts': {'temp': 1.8}           # Localized heat/inflammation
            },
            'pyelo_combined': {
                'type': 'combined', 
                'zones': [16],                    # Renal
                'shifts': {'ph': -0.2, 'temp': 2.0} # Acidosis + High Fever
            },
            'ischemia_combined': {
                'type': 'combined', 
                'zones': [19],                    # Lower Microcirculation
                'shifts': {'ph': -0.3, 'temp': -2.5} # Severe Hypoxia/Acidosis + Cold Tissue
            }
        }

    # ==========================================
    # INTERNAL HELPERS
    # ==========================================

    def _load_data(self):
        """Loads and sorts the dataset chronologically."""
        if not os.path.exists(self.df_path):
            raise FileNotFoundError(f"Missing input: {self.df_path}")
        cols = ['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID']
        df = pd.read_csv(self.df_path, header=None, names=cols)
        df.dropna(inplace=True)
        df.sort_values(by='Time_Seconds', inplace=True)
        return df

    def _update_bot_energy(self, state, t, energy_model):
        """Calculates energy harvesting, drain, and toggles awake/sleep states."""
        delta_t = t - state['last_time']
        state['last_time'] = t
        
        if delta_t > 0:
            state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
            state['energy'] = max(0, state['energy'] - (energy_model.idle_drain_per_sec * delta_t))
            
        if state['energy'] <= energy_model.E_off: state['is_awake'] = False
        elif state['energy'] >= energy_model.E_on: state['is_awake'] = True
        return state

    def _get_affected_vessels(self, zones):
        """Flattens a list of anatomical zones into their specific underlying vessel IDs."""
        vessels = []
        for z in zones:
            vessels.extend(self.zone_to_vessels.get(z, []))
        return vessels

    def _get_diseased_value(self, data_dict, vessel_id, beta, t, affected_vessels, shift_val, t_start, t_peak):
        """
        Generates a sensor reading with a time-scaled disease shift applied.
        """
        # 1. Apply Beta Localization Error (probability of reading the wrong vessel)
        if vessel_id not in data_dict:
            chosen_id = random.choice(list(data_dict.keys()))
        elif random.random() < beta:
            available_ids = list(data_dict.keys())
            available_ids.remove(vessel_id)
            chosen_id = random.choice(available_ids)
        else:
            chosen_id = vessel_id
            
        # 2. Get normal physiological baseline
        min_val, max_val = data_dict[chosen_id]
        
        # 3. Apply Disease Shift (Only if the *sampled* vessel is actually diseased)
        if chosen_id in affected_vessels and t > t_start:
            f = min(1.0, (t - t_start) / (t_peak - t_start))
            current_shift = shift_val * f
            
            min_val += current_shift
            max_val += current_shift
            
        return random.uniform(min_val, max_val)

    def _generate_disease_timeline(self, max_t):
        """Generates random onset and peak times for the disease progression."""
        t_start = random.uniform(0, max(0, max_t - self.PROGRESSION_TIME - self.BUFFER_TIME))
        t_peak = t_start + self.PROGRESSION_TIME
        return t_start, t_peak

    # ==========================================
    # SIMULATION EXECUTION
    # ==========================================

    def run_single_disease(self, alpha, beta, disease_key, target_confidence=70.0):
        """Simulates a localized disease affecting a single parameter (e.g., pH or Temp)."""
        if disease_key not in self.diseases:
            print(f"Error: Disease '{disease_key}' not found."); return None
            
        profile = self.diseases[disease_key]
        param = profile['param']
        
        data_dict = self.dm.get_data_dict(param)
        precision = self.dm.get_decimal_precision(param)
        affected_vessels = self._get_affected_vessels(profile['zones'])
        shift_val = profile['shifts'][param]

        df = self._load_data()
        t_start, t_peak = self._generate_disease_timeline(df['Time_Seconds'].max())
        print(f"[{disease_key.upper()}] Random Onset: Hour {t_start/3600:.1f} | Peak: Hour {t_peak/3600:.1f}")
        
        energy_model = EnergyModel(param_type=param)
        bot_states, buffers, final_results, sample_id = {}, {}, [], 1
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
                
                buffers.setdefault(key, []).append(val)
                state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
                state['has_sampled_this_cycle'] = True
                
                if len(buffers) == 20: 
                    # Cache confidence checks for performance
                    if all(conf_cache.setdefault(len(v), StatsUtils.obtain_confidence_binomial(alpha, beta, len(v))) >= target_confidence for v in buffers.values()):
                        for k, vals in buffers.items():
                            final_results.append({
                                'Sample_ID': sample_id, 'Time_Seconds_End': t, 'Zone': k, 
                                'Median_Val': round(pd.Series(vals).median(), precision), 'Count': len(vals)
                            })
                        buffers, sample_id = {}, sample_id + 1
                        
            state['previous_vessel'] = vessel
                    
        return pd.DataFrame(final_results)

    def run_combined_disease(self, alpha, beta, disease_key, target_confidence=70.0):
        """Simulates a complex disease affecting both pH and Temperature simultaneously."""
        if disease_key not in self.diseases:
            print(f"Error: Disease '{disease_key}' not found."); return None

        profile = self.diseases[disease_key]
        affected_vessels = self._get_affected_vessels(profile['zones'])
        ph_shift, temp_shift = profile['shifts']['ph'], profile['shifts']['temp']
        
        ph_dict = self.dm.get_data_dict('ph')
        temp_dict = self.dm.get_data_dict('temp')

        df = self._load_data()
        t_start, t_peak = self._generate_disease_timeline(df['Time_Seconds'].max())
        print(f"[{disease_key.upper()}] Random Onset: Hour {t_start/3600:.1f} | Peak: Hour {t_peak/3600:.1f}")
        
        energy_model = EnergyModel(param_type='combined')
        bot_states, buffers, final_results, sample_id = {}, {}, [], 1
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
                    
        return pd.DataFrame(final_results)

# ==========================================
# EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    simulator = DiseaseSimulator()
    
    print("\n--- Running Stroke Simulation (Zone 7, pH Drop) ---")
    stroke_df = simulator.run_single_disease(alpha=0.3, beta=0.1, disease_key='stroke_ph')
    
    print("\n--- Running Pyelonephritis Simulation (Zone 16, pH Drop + Temp Spike) ---")
    pyelo_df = simulator.run_combined_disease(alpha=0.3, beta=0.1, disease_key='pyelo_combined')
    
    if stroke_df is not None and not stroke_df.empty:
        print("\nStroke pH Results (Zone 7 should show marked drop in later samples):")
        print(stroke_df[stroke_df['Zone'] == 7].tail(5).to_string(index=False))
        
    if pyelo_df is not None and not pyelo_df.empty:
        print("\nPyelonephritis Results (Zone 16 should show Acidosis and Fever):")
        print(pyelo_df[pyelo_df['Zone'] == 16].tail(5).to_string(index=False))