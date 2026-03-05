import argparse
import pandas as pd
import os
import random
from utils import DataManager, ProcessorUtils, StatsUtils

def load_data(input_file):
    """Loads CSV and applies standard headers."""
    cols = ['Nanobot_ID', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Time_Seconds', 'Vessel_ID', 'Stream_ID']
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing input: {input_file}")
    return pd.read_csv(input_file, header=None, names=cols)

def run_simulation(alpha, beta, param):
    dm = DataManager()
    data_dict = dm.get_data_dict(param)
    
    if not data_dict:
        print(f"Error: '{param}' data not found.")
        return

    # Load and clean
    df = load_data('position_data.csv')
    df.dropna(inplace=True)
    
    # Segment data
    grouped = df.groupby('Nanobot_ID', sort=False)['Vessel_ID'].apply(list).reset_index()
    split_df = pd.concat(grouped.apply(ProcessorUtils.split_vessel_ids, axis=1).values, ignore_index=True)
    
    while True:
        sim_df = split_df.copy()
        # Extract random elements
        sim_df['Selected_Vessel'] = sim_df['Vessel_ID'].apply(lambda x: random.choice(x))
        sim_df['Value'] = sim_df['Selected_Vessel'].apply(lambda x: ProcessorUtils.get_random_value(data_dict, x, beta))
        sim_df['Zone'] = sim_df['Selected_Vessel'].apply(DataManager.assign_zone)
        
        # Group to check zone count
        results = sim_df.groupby('Zone').agg({'Value': 'median', 'Selected_Vessel': 'count'}).reset_index()
        
        if len(results) == 20:
            break
        
        print(f"Warning: Only {len(results)} zones found. Re-extracting random elements.")

    # Finalize formatting and stats
    results.rename(columns={'Value': 'Median_Val', 'Selected_Vessel': 'Count'}, inplace=True)
    results['Median_Val'] = results['Median_Val'].round(3)
    
    results['Binomial_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_binomial(alpha, beta, n))
    results['Hoeffding_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_hoeffding(alpha, beta, n))
    
    return results.sort_values(by='Count', ascending=False)

def run_simulation_long(alpha, beta, param):
    dm = DataManager()
    data_dict = dm.get_data_dict(param)
    
    if not data_dict:
        print(f"Error: '{param}' data not found.")
        return

    # Load and clean
    df = load_data('position_data.csv')
    df.dropna(inplace=True)
    
    # Segment data
    grouped = df.groupby('Nanobot_ID', sort=False)['Vessel_ID'].apply(list).reset_index()
    split_df = pd.concat(grouped.apply(ProcessorUtils.split_vessel_ids, axis=1).values, ignore_index=True)

    sim_df = split_df.copy()
    # Extract random elements
    sim_df['Selected_Vessel'] = sim_df['Vessel_ID'].apply(lambda x: random.choice(x))
    sim_df['Value'] = sim_df['Selected_Vessel'].apply(lambda x: ProcessorUtils.get_random_value(data_dict, x, beta))
    
    # Group by the vessel and count an existing column
    results = sim_df.groupby('Selected_Vessel').agg({'Value': 'median', 'Vessel_ID': 'count'}).reset_index()

    # Finalize formatting and stats
    results.rename(columns={'Value': 'Median_Val', 'Vessel_ID': 'Count'}, inplace=True)
    results['Median_Val'] = results['Median_Val'].round(3)
    
    # Statistical analysis
    results['Binomial_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_binomial(alpha, beta, n))
    results['Hoeffding_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_hoeffding(alpha, beta, n))
    
    return results.sort_values(by='Count', ascending=False)

def run_simulation_ts(alpha, beta, param, target_confidence=70.0, target_zones=20):
    """
    Simulates a time series where data is aggregated continuously. A new sample batch 
    is recorded only when all zones achieve the target confidence interval on the median.
    """
    dm = DataManager()
    data_dict = dm.get_data_dict(param)
    
    if not data_dict:
        print(f"Error: '{param}' data not found.")
        return None

    # Load and clean
    df = load_data('position_data.csv')
    df.dropna(inplace=True)
    
    # Sort chronologically to treat the dataframe as a true time series
    df.sort_values(by='Time_Seconds', inplace=True)
    
    # Cache confidence calculations to optimize the row-by-row loop
    conf_cache = {}
    def get_conf(n):
        if n not in conf_cache:
            conf_cache[n] = StatsUtils.obtain_confidence_binomial(alpha, beta, n)
        return conf_cache[n]
    
    zone_buffers = {}  # Stores values per zone for the current sample
    final_results = []
    sample_id = 1
    
    # Iterate through events sequentially
    for row in df.itertuples():
        vessel = row.Vessel_ID
        time_sec = row.Time_Seconds
        
        # Extract data for the current time step
        val = ProcessorUtils.get_random_value(data_dict, vessel, beta)
        zone = DataManager.assign_zone(vessel)
        
        # Buffer the value
        if zone not in zone_buffers:
            zone_buffers[zone] = []
        zone_buffers[zone].append(val)
        
        # Check if we've discovered all target zones
        if len(zone_buffers) == target_zones:
            # Check if ALL zones meet the target confidence threshold
            if all(get_conf(len(vals)) >= target_confidence for vals in zone_buffers.values()):
                
                # Condition met: Aggregate the sample metrics
                for z, vals in zone_buffers.items():
                    n = len(vals)
                    med_val = pd.Series(vals).median()
                    
                    final_results.append({
                        'Sample_ID': sample_id,
                        'Time_Seconds_End': time_sec,
                        'Zone': z,
                        'Median_Val': round(med_val, 3),
                        'Count': n,
                        'Binomial_%': get_conf(n),
                        'Hoeffding_%': StatsUtils.obtain_confidence_hoeffding(alpha, beta, n)
                    })
                
                # Reset buffers to start collecting the next sample
                zone_buffers = {}
                sample_id += 1
                
    result_df = pd.DataFrame(final_results)
    
    if result_df.empty:
        print("Warning: No samples reached the target confidence threshold before the data ended.")
        return result_df
        
    return result_df.sort_values(by=['Sample_ID', 'Zone'])

def run_simulation_ts_long(alpha, beta, param, target_confidence=70.0, target_vessels=94):
    """
    Simulates a time series aggregating data continuously per VESSEL instead of per zone. 
    A new sample batch is recorded only when all 94 vessels achieve the target confidence.
    """
    dm = DataManager()
    data_dict = dm.get_data_dict(param)
    
    if not data_dict:
        print(f"Error: '{param}' data not found.")
        return None

    # Load and clean
    df = load_data('position_data.csv')
    df.dropna(inplace=True)
    
    # Sort chronologically to treat the dataframe as a true time series
    df.sort_values(by='Time_Seconds', inplace=True)
    
    # Cache confidence calculations to optimize the row-by-row loop
    conf_cache = {}
    def get_conf(n):
        if n not in conf_cache:
            conf_cache[n] = StatsUtils.obtain_confidence_binomial(alpha, beta, n)
        return conf_cache[n]
    
    vessel_buffers = {}  # Stores values per vessel (instead of zone)
    final_results = []
    sample_id = 1
    
    # Iterate through events sequentially
    for row in df.itertuples():
        vessel = row.Vessel_ID
        time_sec = row.Time_Seconds
        
        # Skip invalid vessels to avoid infinite dictionary growth
        if pd.isna(vessel):
            continue
            
        # Extract data for the current time step
        val = ProcessorUtils.get_random_value(data_dict, vessel, beta)
        
        # Buffer the value for the specific vessel
        if vessel not in vessel_buffers:
            vessel_buffers[vessel] = []
        vessel_buffers[vessel].append(val)
        
        # Check if we've discovered all target vessels
        if len(vessel_buffers) == target_vessels:
            # Check if ALL vessels meet the target confidence threshold
            if all(get_conf(len(vals)) >= target_confidence for vals in vessel_buffers.values()):
                
                # Condition met: Aggregate the sample metrics
                for v, vals in vessel_buffers.items():
                    n = len(vals)
                    med_val = pd.Series(vals).median()
                    
                    final_results.append({
                        'Sample_ID': sample_id,
                        'Time_Seconds_End': time_sec,
                        'Selected_Vessel': v,
                        'Median_Val': round(med_val, 3),
                        'Count': n,
                        'Binomial_%': get_conf(n),
                        'Hoeffding_%': StatsUtils.obtain_confidence_hoeffding(alpha, beta, n)
                    })
                
                # Reset buffers to start collecting the next sample
                vessel_buffers = {}
                sample_id += 1
                
    result_df = pd.DataFrame(final_results)
    
    if result_df.empty:
        print("Warning: No samples reached the target confidence threshold before the data ended.")
        return result_df
        
    return result_df.sort_values(by=['Sample_ID', 'Selected_Vessel'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Nanobot Simulator")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--param", type=str, default="press", choices=['press', 'ph', 'so2', 'temp'])
    
    args = parser.parse_args()
    
    final_report = run_simulation_long(args.alpha, args.beta, args.param)
    if final_report is not None:
        print(final_report.to_string(index=False))