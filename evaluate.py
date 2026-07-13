"""
=============================================================================
Therapeutic Thermostat: Clinical Evaluation & Stress Testing
=============================================================================
Evaluates a trained DRQN specialist against 3 clinical scenarios for visual dashboards,
or runs a Frozen 100-Patient protocol for rigorous baseline comparison.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import PredictiveChemoEnv
from train import DRQN, CANCER_PROFILES

# Inside evaluate.py
def run_scenario(env, model, device, scenario_name):
    obs, _ = env.reset()
    
    # 1. Capture pure baselines
    baseline_k_elim = env.engine.drug['k_elim']
    baseline_threshold = env.critical_toxicity_threshold
    
    # 2. Reset environment to pure baselines BEFORE applying scenario modifiers
    env.engine.drug['k_elim'] = baseline_k_elim
    env.critical_toxicity_threshold = baseline_threshold
    env.engine.tumor_growth_rate = 0.025
    env.is_resistant = False 
    
    # 3. Apply scenario-specific mutations
    if scenario_name == "Low Growth (Late-Stage)":
        env.engine.tumor_growth_rate = 0.005 
    elif scenario_name == "Medium Growth (Typical)":
        env.engine.tumor_growth_rate = 0.025 
    elif scenario_name == "High Growth (Early-Stage)":
        env.engine.tumor_growth_rate = 0.050 
    elif scenario_name == "Resistance (Dead-End)":
        env.is_resistant = True 
    elif scenario_name == "Toxicity (Slippery Road)":
        env.engine.drug['k_elim'] *= 0.50                          
        env.critical_toxicity_threshold *= 0.75
        
    hidden = None
    terminated = False
    truncated = False
    total_reward = 0
    max_tox_observed = 0.0
    
    history = {
        'time': [], 'tumor_ph': [], 'tumor_temp': [],
        'clearance_temp': [], 'clearance_ph': [], 'dose': [], 'toxicity': []
    }
    action_counts = np.zeros(10)
    
    print(f"\n{'-'*155}")
    print(f" DEEP DIAGNOSTIC PROFILER: {scenario_name}")
    print(f"{'-'*155}")
    print(f"{'Time':<4} | {'Act':<3} | {'Dose':<4} | {'Ce':<5} | {'Hill':<5} | {'Kill':<6} | {'Grow':<6} | {'Mass':<5} | {'dMass':<7} | {'Dist':<5} | {'dDist':<7} | {'Rew':<6} | {'Q-Values (0->9)'}")
    print(f"{'-'*155}")

    while not (terminated or truncated):
        obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
        with torch.no_grad():
            q_vals, hidden = model(obs_tensor, hidden)
            action = q_vals[0, 0, :].argmax().item()
            
        action_counts[action] += 1
        next_obs, reward, terminated, truncated, info = env.step(action)

        current_tox = next_obs[60] * env.critical_toxicity_threshold
        max_tox_observed = max(max_tox_observed, current_tox)
        
        # [NEW] Step-by-step diagnostic log 
        ce = info['Ce']
        hill = info['hill']
        kill = info['kill']
        growth = info['growth']
        mass = info['mass']
        dmass = info['delta_mass']
        dist = info['curr_dist']
        ddist = info['distance_reduction']
        
        q_array = q_vals.squeeze().cpu().numpy()
        q_str = "[" + ", ".join([f"{q:5.2f}" for q in q_array]) + "]"
        
        print(f"{env.current_step:<4} | {action:<3} | {env.current_dose*100:>3.0f}% | {ce:>5.2f} | {hill:>5.2f} | {kill:>6.4f} | {growth:>6.4f} | {mass:>5.3f} | {dmass:>7.4f} | {dist:>5.2f} | {ddist:>7.4f} | {reward:>6.2f} | {q_str}")
        
        history['time'].append(env.current_step)
        history['tumor_ph'].append(info['real_ph'])
        history['tumor_temp'].append(info['real_temp'])
        history['clearance_temp'].append(info['clearance_temp'])
        history['clearance_ph'].append(info['clearance_ph']) 
        history['toxicity'].append(next_obs[60] * env.critical_toxicity_threshold) 
        history['dose'].append(next_obs[61] * 100)
        
        obs = next_obs
        total_reward += reward
        
    print(f"{'-'*155}")
    print(f"Final Action Distribution: {action_counts.astype(int).tolist()}")
    print(f"{'-'*155}\n")
        
    scenario_threshold = env.critical_toxicity_threshold
    actually_failed = max_tox_observed >= scenario_threshold
    
    return history, total_reward, actually_failed, scenario_threshold

# Inside evaluate.py -> Replace the existing run_frozen_ai_evaluation
def run_frozen_ai_evaluation(cancer_name, model_path, data_path, num_episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRQN(input_dim=64, num_actions=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path, 
        drug_name=CANCER_PROFILES[cancer_name]['drug'], 
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )

    scenarios = [
        "Low Growth (Late-Stage)", 
        "Medium Growth (Typical)", 
        "High Growth (Early-Stage)", 
        "Resistance (Dead-End)", 
        "Toxicity (Slippery Road)"
    ]

    scenario_results = []
    raw_episode_data = []

    for scenario in scenarios:
        print(f"Running Frozen 100-Patient Protocol for: {scenario}...")
        metrics = {'survivals': [], 'mhr': [], 'mat': [], 'returns': []}

        for ep_idx in range(num_episodes):
            # [LOCKED] Seeded reset to fetch deterministic variations
            obs, _ = env.reset(seed=42 + ep_idx)
            
            # --- 1. APPLY SCENARIO MODIFIERS ---
            baseline_k_elim = env.engine.drug['k_elim']
            baseline_threshold = env.critical_toxicity_threshold
            
            env.engine.drug['k_elim'] = baseline_k_elim
            env.critical_toxicity_threshold = baseline_threshold
            env.engine.tumor_growth_rate = 0.025
            env.is_resistant = False 
            
            if scenario == "Low Growth (Late-Stage)":
                env.engine.tumor_growth_rate = 0.005 
            elif scenario == "Medium Growth (Typical)":
                env.engine.tumor_growth_rate = 0.025 
            elif scenario == "High Growth (Early-Stage)":
                env.engine.tumor_growth_rate = 0.050 
            elif scenario == "Resistance (Dead-End)":
                env.is_resistant = True 
            elif scenario == "Toxicity (Slippery Road)":
                env.engine.drug['k_elim'] *= 0.50                          
                env.critical_toxicity_threshold *= 0.75
            # -----------------------------------

            # Capture initial Euclidean distance for MHR %
            initial_distance = env.prev_distance

            hidden = None
            total_reward = 0.0
            terminated, truncated = False, False
            max_tox_observed = 0.0
            final_distance = initial_distance
            
            while not (terminated or truncated):
                obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
                with torch.no_grad():
                    q_vals, hidden = model(obs_tensor, hidden)
                    action = q_vals[0, 0, :].argmax().item()
                    
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                max_tox_observed = max(max_tox_observed, env.total_toxicity)
                final_distance = info['curr_dist']
                
            # Calculate Mean Homeostatic Restoration (MHR %)
            # Clamped at 0 to prevent negative MHR if the tumor actually grew
            mhr_percent = max(0.0, ((initial_distance - final_distance) / initial_distance) * 100.0)
                
            metrics['returns'].append(total_reward)
            metrics['survivals'].append(0 if terminated else 1)
            metrics['mat'].append(max_tox_observed)
            metrics['mhr'].append(mhr_percent)
            
            raw_episode_data.append({
                "Policy": "Trained DRQN", "Scenario": scenario, "Episode": ep_idx, "Seed": 42 + ep_idx,
                "Return": total_reward, "Survival": 0 if terminated else 1,
                "MHR_Pct": mhr_percent, "MAT_Max_Tox": max_tox_observed
            })
            
        # Aggregate scenario stats
        scenario_results.append({
            "Scenario": scenario,
            "Accuracy (%)": np.mean(metrics['survivals']) * 100,
            "MHR (%)": np.mean(metrics['mhr']),
            "MAT": np.mean(metrics['mat']),
            "Mean_Return": np.mean(metrics['returns'])
        })

    # --- Print the Academic Table ---
    print(f"\n{'='*95}")
    print(f" FROZEN 100-PATIENT PROTOCOL RESULTS: {cancer_name}")
    print(f"{'='*95}")
    print(f"{'Clinical Scenario':<28} | {'Accuracy (Safety %)' :<20} | {'MHR (%)' :<15} | {'MAT (Max Tox)' :<15}")
    print(f"{'-'*95}")
    for res in scenario_results:
        print(f"{res['Scenario']:<28} | {res['Accuracy (%)']:>19.1f}% | {res['MHR (%)']:>14.1f}% | {res['MAT']:>15.2f}")
    print(f"{'='*95}\n")

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(scenario_results).to_csv(f"results/{cancer_name}_frozen_metrics_summary.csv", index=False)
    pd.DataFrame(raw_episode_data).to_csv(f"results/{cancer_name}_frozen_raw_episodes.csv", index=False)

def evaluate_specialist(cancer_name, model_path, data_path):
    print(f"\n{'='*80}")
    print(f" EVALUATING SPECIALIST DASHBOARD: {cancer_name}")
    print(f"{'='*80}\n")

    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path, 
        drug_name=CANCER_PROFILES[cancer_name]['drug'], 
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRQN(input_dim=64, num_actions=10).to(device)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}. Skipping...")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scenarios = [
        "Low Growth (Late-Stage)", 
        "Medium Growth (Typical)", 
        "High Growth (Early-Stage)", 
        "Resistance (Dead-End)", 
        "Toxicity (Slippery Road)"
    ]
    results = {}
    env.reset()
    
    tz_idx = env.tumor_zone - 1
    healthy_baseline_ph = env.mu_B_ph[tz_idx]
    healthy_baseline_temp = env.mu_B_temp[tz_idx]

    for scenario in scenarios:
        print(f"Running Scenario: {scenario}...")
        history, reward, failed, spec_threshold = run_scenario(env, model, device, scenario)
        results[scenario] = {
            'history': history, 'reward': reward,
            'failed': failed, 'threshold': spec_threshold,
            'final_tox': history['toxicity'][-1]
        }

    os.makedirs("results", exist_ok=True)
    report_path = f"results/{cancer_name}_stress_report.txt"
    with open(report_path, "w") as f:
        f.write(f"CLINICAL STRESS TEST REPORT: {cancer_name}\n")
        f.write("=" * 60 + "\n\n")
        for scenario, res in results.items():
            status = "CRITICAL FAILURE (Toxicity Breached)" if res['failed'] else "SUCCESS (Safely Managed)"
            f.write(f"--- {scenario} ---\n")
            f.write(f"Outcome       : {status}\n")
            f.write(f"Total Reward  : {res['reward']:.2f}\n")
            f.write(f"Final Toxicity: {res['final_tox']:.2f} / {res['threshold']:.1f}\n\n")
            
    plot_comparative_dashboard(cancer_name, results, scenarios, healthy_baseline_ph, healthy_baseline_temp)

def plot_comparative_dashboard(cancer_name, results, scenarios, healthy_ph, healthy_temp):
    plt.style.use('dark_background')
    num_cols = len(scenarios)
    fig, axs = plt.subplots(4, num_cols, figsize=(5 * num_cols, 14), sharex='col', sharey='row')
    fig.suptitle(f'AI Clinical Stress Tests: {cancer_name.replace("_", " ")}', fontsize=18, fontweight='bold', y=0.98)

    for col, scenario in enumerate(scenarios):
        history = results[scenario]['history']
        threshold = results[scenario]['threshold']
        smoothed_clearance = pd.Series(history['clearance_temp']).rolling(window=10, min_periods=1).mean()
        
        axs[0, col].set_title(scenario, fontsize=14, fontweight='bold', pad=10)
        
        axs[0, col].plot(history['time'], history['tumor_ph'], color='#00FFFF', linewidth=2) 
        axs[0, col].axhline(y=healthy_ph, color='#32CD32', linestyle='--', alpha=0.8) 
        axs[0, col].grid(True, alpha=0.15)
        if col == 0: axs[0, col].set_ylabel('Tumor pH')

        axs[1, col].plot(history['time'], history['tumor_temp'], color='#FF4500', linewidth=2, label='Tumor') 
        axs[1, col].plot(history['time'], smoothed_clearance, color='#FFD700', linewidth=2, alpha=0.8, label='Clearance Organ (Smoothed)') 
        axs[1, col].axhline(y=healthy_temp, color='#32CD32', linestyle='--', alpha=0.8)
        axs[1, col].grid(True, alpha=0.15)
        if col == 0: 
            axs[1, col].set_ylabel('Temp (°C)')
            axs[1, col].legend(loc='upper right', fontsize=10)

        axs[2, col].step(history['time'], history['dose'], color='#FF00FF', linewidth=2, where='post') 
        axs[2, col].set_ylim(-5, 115) 
        axs[2, col].grid(True, alpha=0.15)
        if col == 0: axs[2, col].set_ylabel('Infusion Dose (%)')

        tox_color = '#FF0000' if results[scenario]['failed'] else '#FFA500'
        axs[3, col].plot(history['time'], history['toxicity'], color=tox_color, linewidth=2)
        axs[3, col].axhline(y=threshold, color='#FF0000', linestyle='-', linewidth=2, alpha=0.7)
        axs[3, col].grid(True, alpha=0.15)
        axs[3, col].set_xlabel('Simulation Time (Minutes)')
        if col == 0: axs[3, col].set_ylabel('Systemic Toxicity')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = f"results/{cancer_name}_stress_dashboard.png"
    plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), help="Target clinical profile.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--frozen", action="store_true")
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    
    if args.frozen:
        if args.all:
            for cancer_name in CANCER_PROFILES.keys():
                model_path = f"models/{cancer_name}_drqn_best.pt"  
                run_frozen_ai_evaluation(cancer_name, model_path, dataset_path, num_episodes=100)
        elif args.cancer:
            model_path = f"models/{args.cancer}_drqn_best.pt"  
            run_frozen_ai_evaluation(args.cancer, model_path, dataset_path, num_episodes=100)
    else:
        if args.all:
            for cancer_name in CANCER_PROFILES.keys():
                model_path = f"models/{cancer_name}_drqn_best.pt" 
                evaluate_specialist(cancer_name, model_path, dataset_path)
        elif args.cancer:
            model_path = f"models/{args.cancer}_drqn_best.pt"  
            evaluate_specialist(args.cancer, model_path, dataset_path)