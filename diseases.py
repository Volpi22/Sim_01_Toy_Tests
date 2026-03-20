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
    
    # Time constants (in seconds)
    PROGRESSION_TIME = 28800  # 8 hours to reach peak disease severity from onset
    BUFFER_TIME = 7200        # 2 hours minimum at peak severity before simulation ends

    def __init__(self, df_path='position_data.csv'):
        """Initializes the simulator with anatomical mappings and disease profiles."""
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
        """Loads the nanobot position dataset and sorts it chronologically."""
        if not os.path.exists(self.df_path):
            raise FileNotFoundError(f"Missing input: {self.df_path}")
        cols = ['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID']
        df = pd.read_csv(self.df_path, header=None, names=cols)
        df.dropna(inplace=True)
        df.sort_values(by='Time_Seconds', inplace=True)
        return df

    def _update_bot_energy(self, state, t, energy_model):
        """
        Calculates energy harvesting and idle drain over a time delta.
        Toggles the nanobot's awake/sleep state based on battery thresholds.
        """
        delta_t = t - state['last_time']
        state['last_time'] = t
        
        if delta_t > 0:
            state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
            state['energy'] = max(0, state['energy'] - (energy_model.idle_drain_per_sec * delta_t))
            
        # Power management
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
        Includes Beta measurement error logic.
        """
        # 1. Apply Beta Measurement Error (probability of reading the wrong vessel entirely)
        if vessel_id not in data_dict:
            chosen_id = random.choice(list(data_dict.keys()))
        elif random.random() < beta:
            available_ids = list(data_dict.keys())
            available_ids.remove(vessel_id)
            chosen_id = random.choice(available_ids)
        else:
            chosen_id = vessel_id
            
        # 2. Get normal physiological baseline bounds for the sampled vessel
        min_val, max_val = data_dict[chosen_id]
        
        # 3. Apply Disease Shift (Only if the sampled vessel is diseased AND the disease has started)
        if chosen_id in affected_vessels and t > t_start:
            # Calculate interpolation factor 'f' (0.0 to 1.0)
            # f = 0 at t_start, f = 1 at t_peak and beyond
            f = min(1.0, (t - t_start) / (t_peak - t_start))
            
            # Scale the severity of the disease shift based on time progression
            current_shift = shift_val * f
            
            min_val += current_shift
            max_val += current_shift
            
        return random.uniform(min_val, max_val)

    def _generate_disease_timeline(self, max_t, safe_start_time=14400):
        """
        Generates random onset and peak times for the disease.
        Ensures t_start happens AFTER the safe_start_time (e.g., 4 hours) 
        so the anomaly detector has time to build a healthy calibration baseline.
        """
        latest_possible_start = max(safe_start_time, max_t - self.PROGRESSION_TIME - self.BUFFER_TIME)
        
        # If the simulation is too short to allow a safe start, force it to start at the safe time
        if latest_possible_start <= safe_start_time:
            t_start = safe_start_time
        else:
            t_start = random.uniform(safe_start_time, latest_possible_start)
            
        t_peak = t_start + self.PROGRESSION_TIME
        return t_start, t_peak

    # ==========================================
    # SIMULATION EXECUTION
    # ==========================================

    def run_single_disease(self, alpha, beta, disease_key, target_confidence=70.0, safe_start_time=14400):
        """
        Simulates a localized disease affecting a single parameter (e.g., pH or Temp).
        Returns a tuple of: (raw_dataframe, aggregated_dataframe, start_time_seconds)
        """
        if disease_key not in self.diseases:
            print(f"Error: Disease '{disease_key}' not found."); return None, None, None
            
        profile = self.diseases[disease_key]
        param = profile['param']
        
        data_dict = self.dm.get_data_dict(param)
        precision = self.dm.get_decimal_precision(param)
        affected_vessels = self._get_affected_vessels(profile['zones'])
        shift_val = profile['shifts'][param]

        df = self._load_data()
        
        # Generate the safe disease timeline
        t_start, t_peak = self._generate_disease_timeline(df['Time_Seconds'].max(), safe_start_time)
        print(f"[{disease_key.upper()}] Safe Start Applied. Random Onset: Hour {t_start/3600:.1f} | Peak: Hour {t_peak/3600:.1f}")
        
        energy_model = EnergyModel(param_type=param)
        
        # Data trackers
        bot_states, buffers = {}, {}
        final_results, raw_results = [], []
        sample_id = 1
        conf_cache = {} # Caches binomial confidence calculations to save CPU cycles

        for row in df.itertuples():
            nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
            if pd.isna(vessel): continue
                
            # Initialize bot energy state if unseen
            if nid not in bot_states:
                bot_states[nid] = {'energy': energy_model.E_max, 'is_awake': True, 'has_sampled_this_cycle': False, 'last_time': t, 'previous_vessel': None}
            
            state = self._update_bot_energy(bot_states[nid], t, energy_model)
            
            # Reset sampling flag if the bot passes through the heart (Vessel 2 = new cycle)
            if vessel == 2 and state['previous_vessel'] != 2: state['has_sampled_this_cycle'] = False
                
            # 5% chance to sample if awake and hasn't sampled this cycle
            if state['is_awake'] and not state['has_sampled_this_cycle'] and random.random() < 0.05:
                val = self._get_diseased_value(data_dict, vessel, beta, t, affected_vessels, shift_val, t_start, t_peak)
                key = self.dm.assign_zone(vessel)
                
                # Save the raw ping for the unaggregated tracker
                raw_results.append({'Time_Seconds': t, 'Zone': key, 'Value': round(val, precision)})
                
                # Add to the aggregation buffer
                buffers.setdefault(key, []).append(val)
                state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
                state['has_sampled_this_cycle'] = True
                
                # Once ANY zone hits 20 samples across all bots, check if ALL zones meet confidence
                if len(buffers) == 20: 
                    if all(conf_cache.setdefault(len(v), StatsUtils.obtain_confidence_binomial(alpha, beta, len(v))) >= target_confidence for v in buffers.values()):
                        # Package the medians into the final output list
                        for k, vals in buffers.items():
                            final_results.append({
                                'Sample_ID': sample_id, 'Time_Seconds_End': t, 'Zone': k, 
                                'Median_Val': round(pd.Series(vals).median(), precision), 'Count': len(vals)
                            })
                        # Clear buffers to start a new aggregated packet
                        buffers, sample_id = {}, sample_id + 1
                        
            state['previous_vessel'] = vessel
                    
        return pd.DataFrame(raw_results), pd.DataFrame(final_results), t_start

    def run_combined_disease(self, alpha, beta, disease_key, target_confidence=70.0, safe_start_time=14400):
        """
        Simulates a complex disease affecting both pH and Temperature simultaneously.
        Returns a tuple of: (raw_dataframe, aggregated_dataframe, start_time_seconds)
        """
        if disease_key not in self.diseases:
            print(f"Error: Disease '{disease_key}' not found."); return None, None, None

        profile = self.diseases[disease_key]
        affected_vessels = self._get_affected_vessels(profile['zones'])
        ph_shift, temp_shift = profile['shifts']['ph'], profile['shifts']['temp']
        
        ph_dict = self.dm.get_data_dict('ph')
        temp_dict = self.dm.get_data_dict('temp')

        df = self._load_data()
        t_start, t_peak = self._generate_disease_timeline(df['Time_Seconds'].max(), safe_start_time)
        print(f"[{disease_key.upper()}] Safe Start Applied. Random Onset: Hour {t_start/3600:.1f} | Peak: Hour {t_peak/3600:.1f}")
        
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
                # Capture both metrics simultaneously
                val_ph = self._get_diseased_value(ph_dict, vessel, beta, t, affected_vessels, ph_shift, t_start, t_peak)
                val_temp = self._get_diseased_value(temp_dict, vessel, beta, t, affected_vessels, temp_shift, t_start, t_peak)
                
                key = self.dm.assign_zone(vessel)
                
                # Save the raw combined ping
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
                    # We check confidence based on the length of the pH list (temp list is identical length)
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