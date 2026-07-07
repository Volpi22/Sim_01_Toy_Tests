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

def run_scenario(env, model, device, scenario_name):
    """Runs a single simulation episode under a specific clinical scenario."""
    obs, _ = env.reset()
    
    # --- INJECT CLINICAL STRESS TESTS INTO THE PHYSICS ENGINE ---
    env.is_resistant = False # Safely reset flag
    
    if scenario_name == "Standard Session":
        # Force a perfectly average patient
        env.engine.drug['k_elim'] /= env.patient_clearance_factor
        
    elif scenario_name == "Resistance (Dead-End)":
        # Trigger the true biological resistance flag. The drug bounces off.
        env.engine.drug['k_elim'] /= env.patient_clearance_factor
        env.is_resistant = True 
        
    elif scenario_name == "Toxicity (Slippery Road)":
        # Severe organ impairment: Clearance rate drops by 90% (Highly Dangerous!)
        env.engine.drug['k_elim'] /= env.patient_clearance_factor  
        env.engine.drug['k_elim'] *= 0.10                          
        env.critical_toxicity_threshold *= 0.5 # Drop safe limits by 50%                        # Apply 70% impairment

    hidden = None
    terminated = False
    truncated = False
    total_reward = 0
    max_tox_observed = 0.0
    
    history = {
        'time': [],
        'tumor_ph': [],
        'tumor_temp': [],
        'clearance_temp': [],
        'clearance_ph': [],
        'dose': [],
        'toxicity': []
    }

    action_counts = np.zeros(10)
    
    print(f"\n{'-'*90}")
    print(f" REWARD PROFILER: {scenario_name}")
    print(f"{'-'*90}")
    print(f"{'Time':<6} | {'Efficacy':<12} | {'Sickness':<12} | {'Toxicity':<12} | {'Smooth':<10} | {'Total Reward'}")
    print(f"{'-'*90}")

    while not (terminated or truncated):
        obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
        
        with torch.no_grad():
            q_vals, hidden = model(obs_tensor, hidden)
            action = q_vals[0, 0, :].argmax().item()
            
        action_counts[action] += 1
            
        next_obs, reward, terminated, truncated, info = env.step(action)

        current_tox = next_obs[60] * env.critical_toxicity_threshold
        max_tox_observed = max(max_tox_observed, current_tox)
        
        # Profile the reward components every 10 minutes
        if env.current_step % 10 == 0:
            eff = info['efficacy_reward']
            sick = info['sickness_penalty']
            tox = info['toxicity_penalty']
            smooth = info['smoothing_penalty']
            
            # Note: reward here is pre-scaled by 100 in the env, so we log the raw components but scaled total
            print(f"t={env.current_step:<4} | {eff:>12.2f} | {sick:>12.2f} | {tox:>12.2f} | {smooth:>10.2f} | {reward:>12.2f}")
        
        history['time'].append(env.current_step)
        history['tumor_ph'].append(info['real_ph'])
        history['tumor_temp'].append(info['real_temp'])
        history['clearance_temp'].append(info['clearance_temp'])
        history['clearance_ph'].append(info['clearance_ph']) 
        
        history['toxicity'].append(next_obs[60] * env.critical_toxicity_threshold) 
        history['dose'].append(next_obs[61] * 100)
        
        obs = next_obs
        total_reward += reward
        
    print(f"{'-'*90}")
    print(f"Final Action Distribution: {action_counts.astype(int).tolist()}")
    print(f"{'-'*90}\n")
        
    # Calculate if they failed based on this scenario's specific threshold
    scenario_threshold = env.critical_toxicity_threshold
    actually_failed = max_tox_observed >= scenario_threshold
    
    return history, total_reward, actually_failed, scenario_threshold


def run_frozen_ai_evaluation(cancer_name, model_path, data_path, num_episodes=100):
    """Runs the exact same frozen 100-seed protocol used to evaluate the heuristic baselines."""
    print(f"\n{'='*110}")
    print(f" STAGE 5: FROZEN DRQN EVALUATION - {cancer_name}")
    print(f" Protocol: {num_episodes} Episodes | Locked Seeds (42-{41+num_episodes}) | Gamma=4.0 | Growth=0.006")
    print(f"{'='*110}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRQN(input_dim=63, num_actions=10).to(device)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}. Train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path, 
        drug_name=CANCER_PROFILES[cancer_name]['drug'], 
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )

    metrics = {
        'returns': [], 'masses': [], 'final_toxs': [], 
        'max_toxs': [], 'survivals': [], 'ep_lengths': [], 
        'distances': [], 'cum_doses': []
    }
    
    raw_episode_data = []

    print("Evaluating DRQN over 100 seeded patients...")
    
    for ep_idx in range(num_episodes):
        # FROZEN PROTOCOL: Exact same seeds as the baselines
        obs, _ = env.reset(seed=42 + ep_idx)
        
        # Lock in the representative biological profile
        env.engine.gamma = 4.0
        env.engine.tumor_growth_rate = 0.006
        env.engine.drug['k_elim'] /= env.patient_clearance_factor 
        
        hidden = None
        total_reward = 0.0
        terminated, truncated = False, False
        max_tox_observed = 0.0
        cumulative_dose = 0.0
        
        while not (terminated or truncated):
            obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
            with torch.no_grad():
                q_vals, hidden = model(obs_tensor, hidden)
                action = q_vals[0, 0, :].argmax().item()
                
            obs, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            max_tox_observed = max(max_tox_observed, env.total_toxicity)
            cumulative_dose += env.current_dose
            
        metrics['returns'].append(total_reward)
        metrics['masses'].append(env.engine.tumor_mass)
        metrics['final_toxs'].append(env.total_toxicity)
        metrics['max_toxs'].append(max_tox_observed)
        metrics['survivals'].append(0 if terminated else 1) # Note: Terminated should be False due to Critical Care rule
        metrics['cum_doses'].append(cumulative_dose)
        
        raw_episode_data.append({
            "Policy": "Trained DRQN", "Episode": ep_idx, "Seed": 42 + ep_idx,
            "Return": total_reward, "Survival": 0 if terminated else 1,
            "Final_Mass": env.engine.tumor_mass, "Max_Tox": max_tox_observed,
            "Cumul_Dose": cumulative_dose
        })
        
    # Compute Statistics
    mean_ret, std_ret = np.mean(metrics['returns']), np.std(metrics['returns'])
    mean_mass, std_mass = np.mean(metrics['masses']), np.std(metrics['masses'])
    mean_mtox, std_mtox = np.mean(metrics['max_toxs']), np.std(metrics['max_toxs'])
    mean_dose, std_dose = np.mean(metrics['cum_doses']), np.std(metrics['cum_doses'])
    surv_rate = np.mean(metrics['survivals']) * 100
    
    print(f"\n{'Policy':<25} | {'Return (Mean ± Std)':<20} | {'Surv%':<6} | {'Final Mass':<15} | {'Max Tox':<15} | {'Cumul. Dose'}")
    print(f"{'-'*110}")
    
    ret_str = f"{mean_ret:,.0f} ± {std_ret:,.0f}"
    mass_str = f"{mean_mass:.2f} ± {std_mass:.2f}"
    tox_str = f"{mean_mtox:.1f} ± {std_mtox:.1f}"
    dose_str = f"{mean_dose:,.0f} ± {std_dose:,.0f}"
    print(f"{'Trained DRQN':<25} | {ret_str:>20} | {surv_rate:>5.0f}% | {mass_str:>15} | {tox_str:>15} | {dose_str:>15}")
    print(f"{'='*110}\n")

    # Export to CSV
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{
        "Policy": "Trained DRQN", "Mean_Return": mean_ret, "Std_Return": std_ret,
        "Survival_Rate": surv_rate, "Mean_Final_Mass": mean_mass, "Std_Final_Mass": std_mass,
        "Mean_Max_Tox": mean_mtox, "Std_Max_Tox": std_mtox, "Mean_Cumul_Dose": mean_dose, "Std_Cumul_Dose": std_dose
    }]).to_csv(f"results/{cancer_name}_DRQN_summary.csv", index=False)
    pd.DataFrame(raw_episode_data).to_csv(f"results/{cancer_name}_DRQN_raw_episodes.csv", index=False)
    print(f"[INFO] DRQN summary and raw data safely saved to 'results/' directory.")

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
    model = DRQN(input_dim=63, num_actions=10).to(device)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}. Skipping...")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scenarios = ["Standard Session", "Resistance (Dead-End)", "Toxicity (Slippery Road)"]
    results = {}
    
    env.reset()
    
    tz_idx = env.tumor_zone - 1
    healthy_baseline_ph = env.mu_B_ph[tz_idx]
    healthy_baseline_temp = env.mu_B_temp[tz_idx]

    # 1. Run all scenarios
    for scenario in scenarios:
        print(f"Running Scenario: {scenario}...")
        history, reward, failed, spec_threshold = run_scenario(env, model, device, scenario)
        results[scenario] = {
            'history': history,
            'reward': reward,
            'failed': failed,
            'threshold': spec_threshold,
            'final_tox': history['toxicity'][-1]
        }

    # 2. Compile Text Report
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
            
    print(f"\n[INFO] Stress report saved to {report_path}")

    # 3. Generate Comparative Dashboard
    plot_comparative_dashboard(cancer_name, results, scenarios, healthy_baseline_ph, healthy_baseline_temp)

def plot_comparative_dashboard(cancer_name, results, scenarios, healthy_ph, healthy_temp):
    """Generates a 12-panel (4x3) comparative dashboard with smoothed telemetry."""
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 3, figsize=(18, 14), sharex='col', sharey='row')
    fig.suptitle(f'AI Clinical Stress Tests: {cancer_name.replace("_", " ")}', fontsize=18, fontweight='bold', y=0.98)

    for col, scenario in enumerate(scenarios):
        history = results[scenario]['history']
        threshold = results[scenario]['threshold']
        
        # Smooth the clearance organ data (10-minute rolling average)
        smoothed_clearance = pd.Series(history['clearance_temp']).rolling(window=10, min_periods=1).mean()
        
        # Column Titles
        axs[0, col].set_title(scenario, fontsize=14, fontweight='bold', pad=10)
        
        # Plot 1: pH (Row 0)
        axs[0, col].plot(history['time'], history['tumor_ph'], color='#00FFFF', linewidth=2) # Bright Cyan
        axs[0, col].axhline(y=healthy_ph, color='#32CD32', linestyle='--', alpha=0.8) # Lime Green baseline
        axs[0, col].grid(True, alpha=0.15)
        if col == 0: axs[0, col].set_ylabel('Tumor pH')

        # Plot 2: Temp (Row 1)
        axs[1, col].plot(history['time'], history['tumor_temp'], color='#FF4500', linewidth=2, label='Tumor') # OrangeRed
        axs[1, col].plot(history['time'], smoothed_clearance, color='#FFD700', linewidth=2, alpha=0.8, label='Clearance Organ (Smoothed)') # Gold
        axs[1, col].axhline(y=healthy_temp, color='#32CD32', linestyle='--', alpha=0.8)
        axs[1, col].grid(True, alpha=0.15)
        if col == 0: 
            axs[1, col].set_ylabel('Temp (°C)')
            axs[1, col].legend(loc='upper right', fontsize=10)

        # Plot 3: Dose (Row 2)
        axs[2, col].step(history['time'], history['dose'], color='#FF00FF', linewidth=2, where='post') # Magenta
        axs[2, col].set_ylim(-5, 115) 
        axs[2, col].grid(True, alpha=0.15)
        if col == 0: axs[2, col].set_ylabel('Infusion Dose (%)')

        # Plot 4: Toxicity (Row 3)
        # Color red if it failed, orange if safe
        tox_color = '#FF0000' if results[scenario]['failed'] else '#FFA500'
        axs[3, col].plot(history['time'], history['toxicity'], color=tox_color, linewidth=2)
        axs[3, col].axhline(y=threshold, color='#FF0000', linestyle='-', linewidth=2, alpha=0.7)
        axs[3, col].grid(True, alpha=0.15)
        axs[3, col].set_xlabel('Simulation Time (Minutes)')
        if col == 0: axs[3, col].set_ylabel('Systemic Toxicity')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = f"results/{cancer_name}_stress_dashboard.png"
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Dashboard saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Specialists against Clinical Stress Tests or Frozen Protocols")
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), help="Target clinical profile.")
    parser.add_argument("--all", action="store_true", help="Evaluate all trained specialists.")
    parser.add_argument("--frozen", action="store_true", help="Run the rigorous 100-patient frozen statistical protocol.")
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    
    if args.frozen:
        # Run the Rigorous Statistical Evaluation
        if args.all:
            for cancer_name in CANCER_PROFILES.keys():
                model_path = f"models/{cancer_name}_drqn_best.pt"  # <-- MODIFIED
                run_frozen_ai_evaluation(cancer_name, model_path, dataset_path, num_episodes=100)
        elif args.cancer:
            model_path = f"models/{args.cancer}_drqn_best.pt"  # <-- MODIFIED
            run_frozen_ai_evaluation(args.cancer, model_path, dataset_path, num_episodes=100)
        else:
            print("Please specify --cancer [name] or --all to run the frozen metrics.")
    else:
        # Run the Visual Dashboard Stress Tests
        if args.all:
            for cancer_name in CANCER_PROFILES.keys():
                model_path = f"models/{cancer_name}_drqn_best.pt"  # <-- MODIFIED
                evaluate_specialist(cancer_name, model_path, dataset_path)
        elif args.cancer:
            model_path = f"models/{args.cancer}_drqn_best.pt"  # <-- MODIFIED
            evaluate_specialist(args.cancer, model_path, dataset_path)
        else:
            print("Please specify --cancer [name] or --all to run the evaluation.")