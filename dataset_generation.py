"""
Dataset Assembly Module (Predictive Multi-Rate Paradigm)
Processes the raw 1-week NS-3 trace into an unbroken, 1-minute step schedule
representing the continuous ground-truth homeostatic noise of the patient's body.
"""

import os
import pandas as pd
import numpy as np
import random
from utils import DataManager, ProcessorUtils, StatsUtils, EnergyModel

RAW_NS3_TRACE = os.path.expanduser("~/EXPERIMENTS/Sim_01_Toy_Tests/raw_ns3_traces/1_week_trace.csv")
OUTPUT_DIR = "datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "1_week_sampling_schedule.parquet")

class IoBNT_DataExtractor:
    def __init__(self, alpha=0.30, beta_ph=0.10, beta_temp=0.01):
        self.alpha = alpha
        self.beta_ph = beta_ph
        self.beta_temp = beta_temp
        self.dm = DataManager()
        
    def _load_data(self, input_csv):
        cols = ['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID']
        df = pd.read_csv(input_csv, header=None, names=cols)
        df.dropna(inplace=True)
        return df.sort_values(by='Time_Seconds')

    def generate_1_week_schedule(self, input_csv, max_minutes=10080):
        print("Initializing 1-week continuous baseline noise extraction...")
        df = self._load_data(input_csv)
        energy_model = EnergyModel(param_type='combined')
        
        ph_dict = self.dm.get_data_dict('ph')
        temp_dict = self.dm.get_data_dict('temp')
        
        bot_states = {}
        schedule_data = np.full((max_minutes, 40), np.nan)
        
        current_minute = 0
        minute_buffers = {z: {'ph': [], 'temp': []} for z in range(1, 21)}
        sampling_chance = 0.05

        for row in df.itertuples():
            nid, t_sec, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
            minute_idx = int(t_sec // 60)
            
            if minute_idx >= max_minutes:
                break
                
            if minute_idx > current_minute:
                self._aggregate_minute(current_minute, minute_buffers, schedule_data, ph_dict, temp_dict)
                current_minute = minute_idx
                minute_buffers = {z: {'ph': [], 'temp': []} for z in range(1, 21)}

            if nid not in bot_states:
                bot_states[nid] = {
                    'energy': energy_model.E_max, 'is_awake': True, 
                    'has_sampled_this_cycle': False, 'last_time': t_sec, 'prev_vessel': None
                }
                
            state = bot_states[nid]
            delta_t = t_sec - state['last_time']
            state['last_time'] = t_sec
            
            if delta_t > 0:
                state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
                state['energy'] = max(0, state['energy'] - (energy_model.idle_drain_per_sec * delta_t))
                
            if state['energy'] <= energy_model.E_off: state['is_awake'] = False
            elif state['energy'] >= energy_model.E_on: state['is_awake'] = True
            
            if vessel == 2 and state['prev_vessel'] != 2:
                state['has_sampled_this_cycle'] = False
                
            if state['is_awake'] and not state['has_sampled_this_cycle'] and random.random() < sampling_chance:
                zone = DataManager.assign_zone(vessel)
                if zone is not None:
                    v_ph = ProcessorUtils.get_random_value(ph_dict, vessel, self.beta_ph)
                    v_temp = ProcessorUtils.get_random_value(temp_dict, vessel, self.beta_temp)
                    minute_buffers[zone]['ph'].append(v_ph)
                    minute_buffers[zone]['temp'].append(v_temp)
                    
                    state['energy'] = max(0, state['energy'] - energy_model.action_consumption('combined'))
                    state['has_sampled_this_cycle'] = True
                    
            state['prev_vessel'] = vessel

        self._aggregate_minute(current_minute, minute_buffers, schedule_data, ph_dict, temp_dict)
        
        col_names = [f"pH_Zone_{z}" for z in range(1, 21)] + [f"Temp_Zone_{z}" for z in range(1, 21)]
        schedule_df = pd.DataFrame(schedule_data, columns=col_names)
        schedule_df.index.name = 'Minute'
        return schedule_df

    def _aggregate_minute(self, minute_idx, buffers, schedule_matrix, ph_dict, temp_dict):
        """Generates continuous homeostatic noise baseline mappings for every minute."""
        for z in range(1, 21):
            # If the specific minute buffer has records, use the median
            if len(buffers[z]['ph']) > 0:
                schedule_matrix[minute_idx, z - 1] = np.median(buffers[z]['ph'])
                schedule_matrix[minute_idx, (z - 1) + 20] = np.median(buffers[z]['temp'])
            else:
                # Map standard underlying homeostatic baseline bounds if pings were sparse
                # to guarantee an unbroken physical ground truth channel
                vessels_in_zone = DataManager.ZONE_TO_VESSELS[z]
                fallback_vessel = random.choice(vessels_in_zone)
                schedule_matrix[minute_idx, z - 1] = ProcessorUtils.get_random_value(ph_dict, fallback_vessel, 0.0)
                schedule_matrix[minute_idx, (z - 1) + 20] = ProcessorUtils.get_random_value(temp_dict, fallback_vessel, 0.0)

def build_rl_dataset():
    if not os.path.exists(RAW_NS3_TRACE):
        raise FileNotFoundError(f"Missing base trace: {RAW_NS3_TRACE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading and processing raw traces from {RAW_NS3_TRACE}...")
    extractor = IoBNT_DataExtractor(alpha=0.30, beta_ph=0.10, beta_temp=0.01)
    schedule_df = extractor.generate_1_week_schedule(RAW_NS3_TRACE, max_minutes=10080)
    
    float_cols = schedule_df.select_dtypes(include=['float64']).columns
    schedule_df[float_cols] = schedule_df[float_cols].astype('float32')
    
    schedule_df.to_parquet(OUTPUT_FILE)
    print(f"\nSUCCESS! Continuous Synthetic Baseline Noise Reservoir saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    build_rl_dataset()