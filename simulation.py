"""
Nanobot Simulation Module

This module provides the core simulation engine to evaluate the data collection 
efficiency of a nanonetwork. It models the behavior of individual nanobots 
navigating a vascular network, accounting for stochastic sensing errors, 
localization confidence, and dynamic energy constraints (harvesting vs. consumption).
"""

import argparse
import pandas as pd
import os
import random
from utils import DataManager, ProcessorUtils, StatsUtils, EnergyModel

class NanobotSimulator:
    """
    A simulator to evaluate nanobot data collection efficiency and statistical 
    confidence across different physical zones and energy constraints.
    """

    def __init__(self, alpha=0.05, beta=0.10, target_confidence=70.0):
        """
        Initializes the NanobotSimulator.
        
        @param alpha (float): The base probability of a localization error.
        @param beta (float): The base probability of a measurement/sensor error.
        @param target_confidence (float): The binomial confidence threshold required 
                                          to generate an aggregated statistical packet.
        """
        self.alpha = alpha
        self.beta = beta
        self.target_confidence = target_confidence
        self.dm = DataManager()
        self.input_file = 'position_data.csv'

    # ==========================================
    # DATA LOADING UTILITY
    # ==========================================

    def _load_data(self):
        """
        Loads and prepares the baseline position dataset with standard headers.
        
        @return pd.DataFrame: The cleaned and formatted spatial tracking data.
        """
        cols = ['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID']
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Missing input dataset: {self.input_file}")
        
        df = pd.read_csv(self.input_file, header=None, names=cols)
        df.dropna(inplace=True)
        return df

    # ==========================================
    # CORE ENERGY LOGIC
    # ==========================================

    def _update_bot_energy(self, state, t, energy_model):
        """
        Calculates energy harvesting and idle drain over a specific time delta.
        
        @param state (dict): The current state dictionary of the nanobot.
        @param t (float): The current simulation time in seconds.
        @param energy_model (EnergyModel): The mathematical energy constraints model.
        
        @return dict: The updated nanobot state dictionary.
        """
        delta_t = t - state['last_time']
        state['last_time'] = t
        
        if delta_t > 0:
            # 1. Harvest energy from the environment (e.g., piezoelectric from blood flow)
            state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
            # 2. Subtract continuous background idle consumption
            state['energy'] = max(0, state['energy'] - (energy_model.idle_drain_per_sec * delta_t))
        
        # Power management: Toggle awake/sleep state based on absolute hardware thresholds
        if state['energy'] <= energy_model.E_off:
            state['is_awake'] = False
        elif state['energy'] >= energy_model.E_on:
            state['is_awake'] = True
        
        return state

    # ==========================================
    # SIMULATION VARIANTS
    # ==========================================

    def run_random_static(self, param, mode='zone'):
        """
        Baseline simulation variant.
        Performs random sampling without continuous time tracking or energy constraints.
        Useful for establishing theoretical maximums or baseline statistical noise.
        
        @param param (str): The physiological parameter to simulate (e.g., 'ph', 'temp').
        @param mode (str): Grouping mode ('zone' for macroscopic, 'vessel' for microscopic).
        
        @return pd.DataFrame: The finalized static sampling results.
        """
        data_dict = self.dm.get_data_dict(param)
        precision = self.dm.get_decimal_precision(param)
        if not data_dict:
            return None

        df = self._load_data()
        
        # Group and split vessel IDs to simulate random distribution across navigated paths
        grouped = df.groupby('Nanobot_ID', sort=False)['Vessel_ID'].apply(list).reset_index()
        split_df = pd.concat(grouped.apply(ProcessorUtils.split_vessel_ids, axis=1).values, ignore_index=True)
        
        # Static sampling logic
        sim_df = split_df.copy()
        sim_df['Selected_Vessel'] = sim_df['Vessel_ID'].apply(random.choice)
        sim_df['Value'] = sim_df['Selected_Vessel'].apply(
            lambda x: ProcessorUtils.get_random_value(data_dict, x, self.beta)
        )
        
        # Aggregate results based on the chosen spatial resolution
        group_col = 'Zone' if mode == 'zone' else 'Selected_Vessel'
        if mode == 'zone':
            sim_df['Zone'] = sim_df['Selected_Vessel'].apply(DataManager.assign_zone)
        
        return self._finalize_results(sim_df, group_col, precision)

    def run_energy_simulation(self, param, mode='zone', is_timeseries=False):
        """
        Energy-aware simulation tracking longitudinal progression.
        Supports both single-snapshot aggregations and continuous time-series (TS) logging.
        
        @param param (str): The physiological parameter to simulate.
        @param mode (str): Grouping mode ('zone' or 'vessel').
        @param is_timeseries (bool): If True, generates continuous data packets over time.
        
        @return pd.DataFrame: The resulting dataset of sampled values.
        """
        data_dict = self.dm.get_data_dict(param)
        precision = self.dm.get_decimal_precision(param)
        energy_model = EnergyModel(param_type=param)
        df = self._load_data().sort_values(by='Time_Seconds')

        bot_states, buffers, final_results = {}, {}, []
        sample_id, sampling_chance = 1, 0.05
        target_count = 20 if mode == 'zone' else 94

        for row in df.itertuples():
            nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
            
            # Initialize nanobot state upon first appearance in the network
            if nid not in bot_states:
                bot_states[nid] = {
                    'energy': energy_model.E_max, 'is_awake': True, 
                    'has_sampled_this_cycle': False, 'last_time': t, 'previous_vessel': None
                }
            
            state = self._update_bot_energy(bot_states[nid], t, energy_model)
            
            # Reset the sampling flag if the nanobot cycles back through the heart (Vessel 2)
            if vessel == 2 and state['previous_vessel'] != 2:
                state['has_sampled_this_cycle'] = False
            
            # Attempt to sample if hardware constraints are met
            if state['is_awake'] and not state['has_sampled_this_cycle']:
                if random.random() < sampling_chance:
                    val = ProcessorUtils.get_random_value(data_dict, vessel, self.beta)
                    key = DataManager.assign_zone(vessel) if mode == 'zone' else vessel
                    
                    # Store sample in appropriate buffer structure
                    sample = {'Zone' if mode == 'zone' else 'Selected_Vessel': key, 'Value': val}
                    if not is_timeseries:
                        buffers.setdefault('all', []).append(sample)
                    else:
                        buffers.setdefault(key, []).append(val)
                    
                    # Deduct energy for the active measurement and transmission burst
                    state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
                    state['has_sampled_this_cycle'] = True
                    
                    # Time-Series Logic: Dispatch an aggregated packet once all zones reach confidence targets
                    if is_timeseries and len(buffers) == target_count:
                        if all(self._get_conf(len(v)) >= self.target_confidence for v in buffers.values()):
                            for k, vals in buffers.items():
                                final_results.append(self._format_ts_row(sample_id, t, k, vals, precision, mode))
                            buffers, sample_id = {}, sample_id + 1

            state['previous_vessel'] = vessel

        if is_timeseries:
            return pd.DataFrame(final_results)
        else:
            flat_buffer = pd.DataFrame(buffers.get('all', []))
            group_col = 'Zone' if mode == 'zone' else 'Selected_Vessel'
            return self._finalize_results(flat_buffer, group_col, precision)

    def run_combined_energy_ts(self, mode='zone'):
        """
        Specialized simulation for multi-parameter multimodal sensing (pH + Temp).
        Optimizes swarm energy efficiency by packing both independent readings 
        into a single telemetry payload.
        
        @param mode (str): Spatial grouping mode ('zone' or 'vessel').
        
        @return pd.DataFrame: Time-series dataset containing both pH and Temperature medians.
        """
        ph_dict = self.dm.get_data_dict('ph')
        temp_dict = self.dm.get_data_dict('temp')
        energy_model = EnergyModel(param_type='combined')
        df = self._load_data().sort_values(by='Time_Seconds')

        bot_states, buffers, final_results = {}, {}, []
        sample_id, sampling_chance, target_count = 1, 0.05, (20 if mode == 'zone' else 94)

        for row in df.itertuples():
            nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
            
            if nid not in bot_states:
                bot_states[nid] = {
                    'energy': energy_model.E_max, 'is_awake': True, 
                    'has_sampled_this_cycle': False, 'last_time': t, 'previous_vessel': None
                }
            
            state = self._update_bot_energy(bot_states[nid], t, energy_model)
            
            if vessel == 2 and state['previous_vessel'] != 2: 
                state['has_sampled_this_cycle'] = False

            if state['is_awake'] and not state['has_sampled_this_cycle'] and random.random() < sampling_chance:
                # Capture both metrics simultaneously to simulate a combined sensor array
                v_ph = ProcessorUtils.get_random_value(ph_dict, vessel, self.beta)
                v_temp = ProcessorUtils.get_random_value(temp_dict, vessel, self.beta)
                key = DataManager.assign_zone(vessel) if mode == 'zone' else vessel
                
                buf = buffers.setdefault(key, {'ph': [], 'temp': []})
                buf['ph'].append(v_ph)
                buf['temp'].append(v_temp)
                
                # Apply the higher energy penalty associated with a combined sensor reading
                state['energy'] = max(0, state['energy'] - energy_model.action_consumption('combined'))
                state['has_sampled_this_cycle'] = True

                # Evaluate swarm confidence utilizing the length of the buffers
                if len(buffers) == target_count:
                    if all(self._get_conf(len(v['ph'])) >= self.target_confidence for v in buffers.values()):
                        for k, v in buffers.items():
                            final_results.append({
                                'Sample_ID': sample_id, 
                                'Time_Seconds_End': t,
                                'Zone' if mode == 'zone' else 'Selected_Vessel': k,
                                'Median_pH': round(pd.Series(v['ph']).median(), 2),
                                'Median_Temp': round(pd.Series(v['temp']).median(), 3),
                                'Count': len(v['ph']),
                                'Binomial_%': self._get_conf(len(v['ph']))
                            })
                        buffers, sample_id = {}, sample_id + 1
                        
            state['previous_vessel'] = vessel
            
        return pd.DataFrame(final_results)

    # ==========================================
    # STATISTICS & FORMATTING HELPERS
    # ==========================================

    def _get_conf(self, n):
        """
        Helper method to calculate binomial confidence based on sample size.
        
        @param n (int): The number of samples collected.
        @return float: The calculated statistical confidence percentage.
        """
        return StatsUtils.obtain_confidence_binomial(self.alpha, self.beta, n)

    def _finalize_results(self, df, group_col, precision):
        """
        Groups raw simulation data by physical zone and attaches statistical bounds.
        
        @param df (pd.DataFrame): Raw sampled data.
        @param group_col (str): Column name to group by (e.g., 'Zone').
        @param precision (int): Number of decimal places to round the median to.
        
        @return pd.DataFrame: Statistically aggregated final results.
        """
        if df.empty:
            return None
            
        res = df.groupby(group_col).agg({'Value': 'median', group_col: 'count'})
        res.columns = ['Median_Val', 'Count']
        res = res.reset_index()
        
        res['Median_Val'] = res['Median_Val'].round(precision)
        res['Binomial_%'] = res['Count'].apply(self._get_conf)
        res['Hoeffding_%'] = res['Count'].apply(
            lambda n: StatsUtils.obtain_confidence_hoeffding(self.alpha, self.beta, n)
        )
        
        return res.sort_values(by='Count', ascending=False)

    def _format_ts_row(self, sid, t, key, vals, precision, mode):
        """
        Formats a single zone's aggregated data into a time-series dictionary row.
        
        @param sid (int): The sequential sample ID.
        @param t (float): The timestamp the packet was finalized.
        @param key (int/str): The Zone or Vessel identifier.
        @param vals (list): The raw values collected for this zone.
        @param precision (int): Decimal precision for the parameter.
        @param mode (str): Grouping mode ('zone' or 'vessel').
        
        @return dict: Formatted row ready for DataFrame appending.
        """
        n = len(vals)
        return {
            'Sample_ID': sid, 
            'Time_Seconds_End': t,
            'Zone' if mode == 'zone' else 'Selected_Vessel': key,
            'Median_Val': round(pd.Series(vals).median(), precision),
            'Count': n, 
            'Binomial_%': self._get_conf(n)
        }

if __name__ == "__main__":
    pass