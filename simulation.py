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
    
    # Process readings
    sim_df = split_df.copy()
    sim_df['Selected_Vessel'] = sim_df['Vessel_ID'].apply(lambda x: random.choice(x))
    sim_df['Value'] = sim_df['Selected_Vessel'].apply(lambda x: ProcessorUtils.get_random_value(data_dict, x, beta))
    sim_df['Zone'] = sim_df['Selected_Vessel'].apply(DataManager.assign_zone)
    
    results = sim_df.groupby('Zone').agg({'Value': 'median', 'Selected_Vessel': 'count'}).reset_index()
    results.rename(columns={'Value': 'Median_Val', 'Selected_Vessel': 'Count'}, inplace=True)
    results['Median_Val'] = results['Median_Val'].round(3)
    
    # Statistical analysis
    results['Binomial_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_binomial(alpha, beta, n))
    results['Hoeffding_%'] = results['Count'].apply(lambda n: StatsUtils.obtain_confidence_hoeffding(alpha, beta, n))
    
    return results.sort_values(by='Count', ascending=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Nanobot Simulator")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--param", type=str, default="press", choices=['press', 'ph', 'so2', 'temp'])
    
    args = parser.parse_args()
    
    final_report = run_simulation(args.alpha, args.beta, args.param)
    if final_report is not None:
        print(final_report.to_string(index=False))