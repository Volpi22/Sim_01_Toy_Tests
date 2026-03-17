import pandas as pd
import numpy as np
import random
import argparse
from utils import DataManager, StatsUtils, EnergyModel

class DiseaseSimulator:
    def __init__(self, df_path='position_data.csv'):
        self.dm = DataManager()
        self.df_path = df_path
        
        # Determine simulation length to set realistic disease timelines
        # Assuming a 24h simulation (86400 seconds)
        # Disease starts at Hour 8, peaks at Hour 16
        self.t_start = 28800 
        self.t_peak = 57600  
        
        # Map zones to their constituent vessels for easy lookup
        self.zone_to_vessels = self.dm.ZONE_TO_VESSELS
        
        # --- DISEASE PROFILES ---
        self.diseases = {
            'stroke_ph': {
                'type': 'single',
                'param': 'ph',
                'zones': [7],  # Head Microcirculation
                'shifts': {'ph': -0.25}  # Severe localized acidosis
            },
            'dvt_temp': {
                'type': 'single',
                'param': 'temp',
                'zones': [19], # Lower Microcirculation
                'shifts': {'temp': 1.8}  # Localized heat/inflammation
            },
            'pyelo_combined': {
                'type': 'combined',
                'zones': [16], # Renal
                'shifts': {'ph': -0.2, 'temp': 2.0} # Acidosis + High Fever
            },
            'ischemia_combined': {
                'type': 'combined',
                'zones': [19], # Lower Microcirculation
                'shifts': {'ph': -0.3, 'temp': -2.5} # Severe Hypoxia/Acidosis + Cold
            }
        }

    def _get_affected_vessels(self, zones):
        """Flattens a list of zones into a list of specific vessel IDs."""
        vessels = []
        for z in zones:
            vessels.extend(self.zone_to_vessels.get(z, []))
        return vessels

    def _get_diseased_value(self, data_dict, vessel_id, beta, t, affected_vessels, shift_val):
        """Generates a value, applying a time-scaled disease shift if applicable."""
        # 1. Beta Localization Error (Did the bot sample the wrong vessel?)
        if vessel_id not in data_dict:
            chosen_id = random.choice(list(data_dict.keys()))
        elif random.random() < beta:
            available_ids = list(data_dict.keys())
            available_ids.remove(vessel_id)
            chosen_id = random.choice(available_ids)
        else:
            chosen_id = vessel_id
            
        # 2. Base baseline range
        min_val, max_val = data_dict[chosen_id]
        
        # 3. Apply Disease Shift (Only if the *sampled* vessel is diseased)
        if chosen_id in affected_vessels and t > self.t_start:
            # Calculate interpolation factor (0.0 to 1.0)
            f = min(1.0, (t - self.t_start) / (self.t_peak - self.t_start))
            current_shift = shift_val * f
            
            min_val += current_shift
            max_val += current_shift
            
        return random.uniform(min_val, max_val)

    def run_single_disease(self, alpha, beta, disease_key, target_confidence=70.0):
        """Runs a time-series energy simulation for a single parameter disease."""
        if disease_key not in self.diseases:
            print(f"Error: Disease {disease_key} not found.")
            return None
            
        profile = self.diseases[disease_key]
        param = profile['param']
        
        data_dict = self.dm.get_data_dict(param)
        precision = self.dm.get_decimal_precision(param)
        affected_vessels = self._get_affected_vessels(profile['zones'])
        shift_val = profile['shifts'][param]

        df = pd.read_csv(self.df_path, header=None, names=['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID'])
        df.dropna(inplace=True)
        df.sort_values(by='Time_Seconds', inplace=True)
        
        energy_model = EnergyModel(param_type=param)
        bot_states, buffers, final_results, sample_id = {}, {}, [], 1
        conf_cache = {}
        
        def get_conf(n):
            if n not in conf_cache: conf_cache[n] = StatsUtils.obtain_confidence_binomial(alpha, beta, n)
            return conf_cache[n]

        for row in df.itertuples():
            nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
            if pd.isna(vessel): continue
                
            if nid not in bot_states:
                bot_states[nid] = {'energy': energy_model.E_max, 'is_awake': True, 'has_sampled_this_cycle': False, 'last_time': t, 'previous_vessel': None}
            state = bot_states[nid]
            
            delta_t = t - state['last_time']
            state['last_time'] = t
            if delta_t > 0:
                state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
                state['energy'] = max(0, state['energy'] - (energy_model.idle_drain_per_sec * delta_t))
                
            if state['energy'] <= energy_model.E_off: state['is_awake'] = False
            elif state['energy'] >= energy_model.E_on: state['is_awake'] = True
            
            if vessel == 2 and state['previous_vessel'] != 2: state['has_sampled_this_cycle'] = False
                
            if state['is_awake'] and not state['has_sampled_this_cycle']:
                if random.random() < 0.05: # Sampling chance
                    val = self._get_diseased_value(data_dict, vessel, beta, t, affected_vessels, shift_val)
                    key = self.dm.assign_zone(vessel)
                    
                    if key not in buffers: buffers[key] = []
                    buffers[key].append(val)
                    
                    state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
                    state['has_sampled_this_cycle'] = True
                    if state['energy'] <= energy_model.E_off: state['is_awake'] = False
                    
                    if len(buffers) == 20: # 20 zones
                        if all(get_conf(len(vals)) >= target_confidence for vals in buffers.values()):
                            for k, vals in buffers.items():
                                n = len(vals)
                                final_results.append({
                                    'Sample_ID': sample_id, 'Time_Seconds_End': t, 'Zone': k, 
                                    'Median_Val': round(pd.Series(vals).median(), precision), 'Count': n
                                })
                            buffers, sample_id = {}, sample_id + 1
                            
            state['previous_vessel'] = vessel
                    
        return pd.DataFrame(final_results)

    def run_combined_disease(self, alpha, beta, disease_key, target_confidence=70.0):
        """Runs a dual-parameter TS simulation applying different shifts to pH and Temp."""
        profile = self.diseases[disease_key]
        affected_vessels = self._get_affected_vessels(profile['zones'])
        ph_shift = profile['shifts']['ph']
        temp_shift = profile['shifts']['temp']
        
        ph_dict = self.dm.get_data_dict('ph')
        temp_dict = self.dm.get_data_dict('temp')

        df = pd.read_csv(self.df_path, header=None, names=['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID'])
        df.dropna(inplace=True)
        df.sort_values(by='Time_Seconds', inplace=True)
        
        energy_model = EnergyModel(param_type='combined')
        bot_states, buffers, final_results, sample_id = {}, {}, [], 1
        conf_cache = {}
        
        def get_conf(n):
            if n not in conf_cache: conf_cache[n] = StatsUtils.obtain_confidence_binomial(alpha, beta, n)
            return conf_cache[n]

        for row in df.itertuples():
            nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
            if pd.isna(vessel): continue
                
            if nid not in bot_states:
                bot_states[nid] = {'energy': energy_model.E_max, 'is_awake': True, 'has_sampled_this_cycle': False, 'last_time': t, 'previous_vessel': None}
            state = bot_states[nid]
            
            delta_t = t - state['last_time']
            state['last_time'] = t
            if delta_t > 0:
                state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
                state['energy'] = max(0, state['energy'] - (energy_model.idle_drain_per_sec * delta_t))
                
            if state['energy'] <= energy_model.E_off: state['is_awake'] = False
            elif state['energy'] >= energy_model.E_on: state['is_awake'] = True
            
            if vessel == 2 and state['previous_vessel'] != 2: state['has_sampled_this_cycle'] = False
                
            if state['is_awake'] and not state['has_sampled_this_cycle']:
                if random.random() < 0.05:
                    # Get dual values with their respective disease shifts
                    val_ph = self._get_diseased_value(ph_dict, vessel, beta, t, affected_vessels, ph_shift)
                    val_temp = self._get_diseased_value(temp_dict, vessel, beta, t, affected_vessels, temp_shift)
                    
                    key = self.dm.assign_zone(vessel)
                    if key not in buffers: buffers[key] = {'ph': [], 'temp': []}
                    buffers[key]['ph'].append(val_ph)
                    buffers[key]['temp'].append(val_temp)
                    
                    state['energy'] = max(0, state['energy'] - energy_model.action_consumption('combined'))
                    state['has_sampled_this_cycle'] = True
                    if state['energy'] <= energy_model.E_off: state['is_awake'] = False
                    
                    if len(buffers) == 20:
                        if all(get_conf(len(vals['ph'])) >= target_confidence for vals in buffers.values()):
                            for k, vals in buffers.items():
                                n = len(vals['ph'])
                                final_results.append({
                                    'Sample_ID': sample_id, 'Time_Seconds_End': t, 'Zone': k, 
                                    'Median_pH': round(pd.Series(vals['ph']).median(), 2),
                                    'Median_Temp': round(pd.Series(vals['temp']).median(), 3), 'Count': n
                                })
                            buffers, sample_id = {}, sample_id + 1
                            
            state['previous_vessel'] = vessel
                    
        return pd.DataFrame(final_results)

if __name__ == "__main__":
    simulator = DiseaseSimulator()
    
    # 1. Stroke (pH)
    print("Running Stroke Simulation (Zone 7, pH Drop)...")
    stroke_df = simulator.run_single_disease(alpha=0.3, beta=0.1, disease_key='stroke_ph')
    
    # 2. Pyelonephritis (Combined)
    print("Running Pyelonephritis Simulation (Zone 16, pH Drop + Temp Spike)...")
    pyelo_df = simulator.run_combined_disease(alpha=0.3, beta=0.1, disease_key='pyelo_combined')
    
    if stroke_df is not None:
        print("\nStroke pH Results (Zone 7 should show marked drop in later samples):")
        print(stroke_df[stroke_df['Zone'] == 7].tail(5))
        
    if pyelo_df is not None:
        print("\nPyelonephritis Results (Zone 16 should show Acidosis and Fever):")
        print(pyelo_df[pyelo_df['Zone'] == 16].tail(5))