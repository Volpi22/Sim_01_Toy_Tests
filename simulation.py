import argparse
import pandas as pd
import os
import random
from utils import DataManager, ProcessorUtils, StatsUtils, EnergyModel

def load_data(input_file):
    """Loads CSV and applies standard headers."""
    cols = ['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID']
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing input: {input_file}")
    return pd.read_csv(input_file, header=None, names=cols)

# ==========================================
# RANDOM SAMPLING SIMULATIONS (NON-ENERGY)
# ==========================================

def run_simulation_random(alpha, beta, param, mode='zone'):
    """
    Simulates a single random sample per nanobot cycle.
    Supports aggregation by 'zone' or 'vessel' using unified logic.
    """
    dm = DataManager()
    data_dict = dm.get_data_dict(param)
    precision = dm.get_decimal_precision(param)
    
    if not data_dict:
        print(f"Error: '{param}' data not found.")
        return None

    df = load_data('position_data.csv')
    df.dropna(inplace=True)
    
    # Segment data into discrete cycles (separated by heart/Vessel 2)
    grouped = df.groupby('Nanobot_ID', sort=False)['Vessel_ID'].apply(list).reset_index()
    split_df = pd.concat(grouped.apply(ProcessorUtils.split_vessel_ids, axis=1).values, ignore_index=True)
    
    # Selection and aggregation logic
    while True:
        sim_df = split_df.copy()
        
        # Randomly pick exactly ONE vessel per cycle to sample
        sim_df['Selected_Vessel'] = sim_df['Vessel_ID'].apply(lambda x: random.choice(x))
        sim_df['Value'] = sim_df['Selected_Vessel'].apply(lambda x: ProcessorUtils.get_random_value(data_dict, x, beta))
        
        if mode == 'zone':
            sim_df['Zone'] = sim_df['Selected_Vessel'].apply(DataManager.assign_zone)
            results = sim_df.groupby('Zone').agg({'Value': 'median', 'Selected_Vessel': 'count'}).reset_index()
            # Ensure spatial coverage for zone-based reports
            if len(results) == 20:
                break
        else:
            # Vessel aggregation logic
            results = sim_df.groupby('Selected_Vessel').agg({'Value': 'median', 'Nanobot_ID': 'count'}).reset_index()
            break

    results.rename(columns={'Value': 'Median_Val', 'Selected_Vessel': 'Count', 'Nanobot_ID': 'Count'}, inplace=True)
    results['Median_Val'] = results['Median_Val'].round(precision)
    
    # Statistical Confidence Calculations
    results['Binomial_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_binomial(alpha, beta, n))
    results['Hoeffding_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_hoeffding(alpha, beta, n))
    
    return results.sort_values(by='Count', ascending=False)

def run_simulation_ts_random(alpha, beta, param, target_confidence=70.0, mode='zone'):
    """
    Time-series simulation for random sampling.
    Dynamically routes to 20 zones or 94 vessels based on mode.
    """
    dm = DataManager()
    data_dict = dm.get_data_dict(param)
    precision = dm.get_decimal_precision(param)
    if not data_dict: return None

    df = load_data('position_data.csv')
    df.dropna(inplace=True)
    df.sort_values(by='Time_Seconds', inplace=True)
    
    target_count = 20 if mode == 'zone' else 94
    conf_cache = {}
    
    def get_conf(n):
        if n not in conf_cache:
            conf_cache[n] = StatsUtils.obtain_confidence_binomial(alpha, beta, n)
        return conf_cache[n]
    
    buffers, final_results, sample_id = {}, [], 1
    
    for row in df.itertuples():
        vessel, t = row.Vessel_ID, row.Time_Seconds
        if pd.isna(vessel): continue
        
        val = ProcessorUtils.get_random_value(data_dict, vessel, beta)
        key = DataManager.assign_zone(vessel) if mode == 'zone' else vessel
        
        if key not in buffers: buffers[key] = []
        buffers[key].append(val)
        
        if len(buffers) == target_count:
            if all(get_conf(len(vals)) >= target_confidence for vals in buffers.values()):
                for k, vals in buffers.items():
                    n = len(vals)
                    final_results.append({
                        'Sample_ID': sample_id, 'Time_Seconds_End': t,
                        ('Zone' if mode == 'zone' else 'Selected_Vessel'): k,
                        'Median_Val': round(pd.Series(vals).median(), precision),
                        'Count': n,
                        'Binomial_%': get_conf(n),
                        'Hoeffding_%': StatsUtils.obtain_confidence_hoeffding(alpha, beta, n)
                    })
                buffers, sample_id = {}, sample_id + 1
                
    return pd.DataFrame(final_results)

# ==========================================
# ENERGY-CONSTRAINED SIMULATIONS (ETWU)
# ==========================================

def run_simulation_energy(alpha, beta, param, mode='zone'):
    """Simulates sampling cycles driven by the physical Energy Model."""
    dm = DataManager()
    data_dict = dm.get_data_dict(param)
    precision = dm.get_decimal_precision(param)
    if not data_dict: return None

    df = load_data('position_data.csv')
    df.dropna(inplace=True)
    df.sort_values(by=['Time_Seconds'], inplace=True)
    
    energy_model = EnergyModel()
    bot_states, valid_samples = {}, []
    
    # Delay probability to avoid first-vessel clustering
    sampling_chance = 0.05 

    for row in df.itertuples():
        nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
        if pd.isna(vessel): continue
            
        if nid not in bot_states:
            bot_states[nid] = {
                'energy': energy_model.E_max, 'is_awake': True,
                'has_sampled_this_cycle': False, 'last_time': t,
                'previous_vessel': None  # Track entry/exit edges
            }
            
        state = bot_states[nid]
        delta_t = t - state['last_time']
        state['last_time'] = t
        
        if delta_t > 0:
            state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
            state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
            
        if state['energy'] <= energy_model.E_off: state['is_awake'] = False
        elif state['energy'] >= energy_model.E_on: state['is_awake'] = True
        
        # EDGE DETECTION: Only reset the cycle when ENTERING Vessel 2
        if vessel == 2 and state['previous_vessel'] != 2: 
            state['has_sampled_this_cycle'] = False
            
        # Sampling logic (no longer excludes Vessel 2)
        if state['is_awake'] and not state['has_sampled_this_cycle']:
            # The bot rolls to decide if it samples here, promoting deeper spatial dispersion
            if random.random() < sampling_chance:
                val = ProcessorUtils.get_random_value(data_dict, vessel, beta)
                entry = {'Nanobot_ID': nid, 'Selected_Vessel': vessel, 'Value': val}
                if mode == 'zone': entry['Zone'] = DataManager.assign_zone(vessel)
                valid_samples.append(entry)
                
                state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
                state['has_sampled_this_cycle'] = True
                if state['energy'] <= energy_model.E_off: state['is_awake'] = False
                
        # Update history for the next iteration
        state['previous_vessel'] = vessel

    sim_df = pd.DataFrame(valid_samples)
    if sim_df.empty: return None

    group_col = 'Zone' if mode == 'zone' else 'Selected_Vessel'
    count_col = 'Selected_Vessel' if mode == 'zone' else 'Nanobot_ID'
    
    results = sim_df.groupby(group_col).agg({'Value': 'median', count_col: 'count'}).reset_index()
    results.rename(columns={'Value': 'Median_Val', count_col: 'Count'}, inplace=True)
    results['Median_Val'] = results['Median_Val'].round(precision)
    results['Binomial_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_binomial(alpha, beta, n))
    results['Hoeffding_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_hoeffding(alpha, beta, n))
    
    return results.sort_values(by='Count', ascending=False)

def run_simulation_ts_energy(alpha, beta, param, target_confidence=70.0, mode='zone'):
    """Time-series aggregation driven by physical energy constraints."""
    dm = DataManager()
    data_dict = dm.get_data_dict(param)
    precision = dm.get_decimal_precision(param)
    if not data_dict: return None

    df = load_data('position_data.csv')
    df.dropna(inplace=True)
    df.sort_values(by='Time_Seconds', inplace=True)
    
    target_count = 20 if mode == 'zone' else 94
    conf_cache = {}
    def get_conf(n):
        if n not in conf_cache:
            conf_cache[n] = StatsUtils.obtain_confidence_binomial(alpha, beta, n)
        return conf_cache[n]
    
    energy_model = EnergyModel()
    bot_states, buffers, final_results, sample_id = {}, {}, [], 1
    
    # Delay probability to avoid first-vessel clustering
    sampling_chance = 0.05 
    
    for row in df.itertuples():
        nid, t, vessel = row.Nanobot_ID, row.Time_Seconds, row.Vessel_ID
        if pd.isna(vessel): continue
            
        if nid not in bot_states:
            bot_states[nid] = {
                'energy': energy_model.E_max, 'is_awake': True, 
                'has_sampled_this_cycle': False, 'last_time': t,
                'previous_vessel': None # Track entry/exit edges
            }
        state = bot_states[nid]
        
        delta_t = t - state['last_time']
        state['last_time'] = t
        if delta_t > 0:
            state['energy'] = energy_model.harvest_energy(state['energy'], delta_t)
            state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
            
        if state['energy'] <= energy_model.E_off: state['is_awake'] = False
        elif state['energy'] >= energy_model.E_on: state['is_awake'] = True
        
        # EDGE DETECTION: Only reset the cycle when entering Vessel 2
        if vessel == 2 and state['previous_vessel'] != 2: 
            state['has_sampled_this_cycle'] = False
            
        # Sampling logic (no longer excludes Vessel 2)
        if state['is_awake'] and not state['has_sampled_this_cycle']:
            # The bot rolls to decide if it samples here, promoting deeper spatial dispersion
            if random.random() < sampling_chance:
                val = ProcessorUtils.get_random_value(data_dict, vessel, beta)
                key = DataManager.assign_zone(vessel) if mode == 'zone' else vessel
                if key not in buffers: buffers[key] = []
                buffers[key].append(val)
                
                state['energy'] = max(0, state['energy'] - energy_model.action_consumption(param))
                state['has_sampled_this_cycle'] = True
                if state['energy'] <= energy_model.E_off: state['is_awake'] = False
                
                if len(buffers) == target_count:
                    if all(get_conf(len(vals)) >= target_confidence for vals in buffers.values()):
                        for k, vals in buffers.items():
                            n = len(vals)
                            final_results.append({
                                'Sample_ID': sample_id, 'Time_Seconds_End': t,
                                ('Zone' if mode == 'zone' else 'Selected_Vessel'): k, 
                                'Median_Val': round(pd.Series(vals).median(), precision),
                                'Count': n,
                                'Binomial_%': get_conf(n),
                                'Hoeffding_%': StatsUtils.obtain_confidence_hoeffding(alpha, beta, n)
                            })
                        buffers, sample_id = {}, sample_id + 1
                        
        # Update history for the next iteration
        state['previous_vessel'] = vessel
                
    return pd.DataFrame(final_results)

# ==========================================
# MAIN EXECUTION ROUTER
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Nanobot Simulator")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--param", type=str, default="press", choices=['press', 'ph', 'so2', 'temp'])
    parser.add_argument("--mode", type=str, default="long", choices=['zone', 'long', 'ts_zone', 'ts_long'])
    parser.add_argument("--energy", action="store_true", help="Enable energy constraints")
    
    args = parser.parse_args()
    report = None

    if args.energy:
        if args.mode == "zone": report = run_simulation_energy(args.alpha, args.beta, args.param, 'zone')
        elif args.mode == "long": report = run_simulation_energy(args.alpha, args.beta, args.param, 'long')
        elif args.mode == "ts_zone": report = run_simulation_ts_energy(args.alpha, args.beta, args.param, target_confidence=70.0, mode='zone')
        elif args.mode == "ts_long": report = run_simulation_ts_energy(args.alpha, args.beta, args.param, target_confidence=70.0, mode='long')
    else:
        if args.mode == "zone": report = run_simulation_random(args.alpha, args.beta, args.param, 'zone')
        elif args.mode == "long": report = run_simulation_random(args.alpha, args.beta, args.param, 'long')
        elif args.mode == "ts_zone": report = run_simulation_ts_random(args.alpha, args.beta, args.param, target_confidence=70.0, mode='zone')
        elif args.mode == "ts_long": report = run_simulation_ts_random(args.alpha, args.beta, args.param, target_confidence=70.0, mode='long')
        
    if report is not None:
        print(report.to_string(index=False))