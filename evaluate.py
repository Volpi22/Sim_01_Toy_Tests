"""
=============================================================================
Therapeutic Thermostat: Clinical Evaluation & Stress Testing
=============================================================================
Evaluates a trained DRQN specialist against 3 clinical scenarios:
1. Standard Session
2. Resistance Test (Tumor is immune to the drug)
3. Toxicity Prediction Test (Patient has severe organ impairment)

Outputs a comprehensive text report and a side-by-side comparative dashboard.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import PredictiveChemoEnv
from train import DRQN, CANCER_PROFILES

def run_scenario(env, model, device, scenario_name):
    """Runs a single simulation episode under a specific clinical scenario."""
    obs, _ = env.reset()
    
    # --- INJECT CLINICAL STRESS TESTS INTO THE PHYSICS ENGINE ---
    if scenario_name == "Standard Session":
        # Force a perfectly average patient by undoing the domain randomization
        env.engine.drug['k_elim'] /= env.patient_clearance_factor
        
    elif scenario_name == "Resistance (Dead-End)":
        # Tumor is totally immune. Drug has 0.0 effect on microenvironment.
        env.engine.pd['eta_ph'] = 0.0
        env.engine.pd['eta_temp'] = 0.0
        
    elif scenario_name == "Toxicity (Slippery Road)":
        # Severe organ impairment: Clearance rate drops by 70% from nominal baseline
        env.engine.drug['k_elim'] /= env.patient_clearance_factor  # Revert to baseline
        env.engine.drug['k_elim'] *= 0.30                          # Apply 70% impairment

    hidden = None
    terminated = False
    truncated = False
    total_reward = 0
    
    tz_idx = env.tumor_zone - 1
    
    history = {
        'time': [],
        'tumor_ph': [],
        'tumor_temp': [],
        'dose': [],
        'toxicity': []
    }

    while not (terminated or truncated):
        obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
        
        with torch.no_grad():
            q_vals, hidden = model(obs_tensor, hidden)
            action = q_vals[0, 0, :].argmax().item()
            
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        history['time'].append(env.current_step)
        history['tumor_ph'].append(next_obs[tz_idx])
        history['tumor_temp'].append(next_obs[tz_idx + 20])
        history['toxicity'].append(next_obs[40])
        history['dose'].append(next_obs[41] * 100)
        
        obs = next_obs
        total_reward += reward
        
    return history, total_reward, terminated

def run_metrics_evaluation(cancer_name, model_path, data_path, num_patients=100):
    print(f"\n{'='*80}")
    print(f" LAUNCHING LARGE-COHORT METRICS EVALUATION: {cancer_name}")
    print(f" Patients per scenario: {num_patients}")
    print(f"{'='*80}\n")

    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path, 
        drug_name=CANCER_PROFILES[cancer_name]['drug'], 
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRQN(input_dim=43, num_actions=10).to(device)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}. Skipping...")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scenarios = ["Standard Session", "Resistance (Dead-End)", "Toxicity (Slippery Road)"]
    metrics_results = {}

    # Boot up env to extract baseline anchors for the math
    env.reset()
    tz_idx = env.tumor_zone - 1
    healthy_baseline_ph = env.mu_B_ph[tz_idx]
    healthy_baseline_temp = env.mu_B_temp[tz_idx]

    def calc_normalized_distance(ph, temp):
        z_ph = (ph - healthy_baseline_ph) / 0.05
        z_temp = (temp - healthy_baseline_temp) / 0.2
        return np.sqrt(z_ph**2 + z_temp**2)

    for scenario in scenarios:
        print(f"Simulating Cohort: {scenario:<25} ", end="", flush=True)
        survivals = 0
        restoration_list = []
        toxicity_list = []
        
        for i in range(num_patients):
            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)
                
            history, _, terminated = run_scenario(env, model, device, scenario)
            
            if not terminated:
                survivals += 1
                
                # Secondary Metric 1: Homeostatic Restoration (%)
                init_ph = history['tumor_ph'][0]
                init_temp = history['tumor_temp'][0]
                init_dist = calc_normalized_distance(init_ph, init_temp)
                
                final_ph = history['tumor_ph'][-1]
                final_temp = history['tumor_temp'][-1]
                final_dist = calc_normalized_distance(final_ph, final_temp)
                
                if init_dist > 0:
                    restoration = max(0.0, ((init_dist - final_dist) / init_dist) * 100)
                else:
                    restoration = 0.0
                restoration_list.append(restoration)
                
                # Secondary Metric 2: Final Toxicity Margin
                toxicity_list.append(history['toxicity'][-1])
                
        safety_rate = (survivals / num_patients) * 100
        mean_restoration = np.mean(restoration_list) if survivals > 0 else 0.0
        mean_toxicity = np.mean(toxicity_list) if survivals > 0 else 0.0
        
        metrics_results[scenario] = {
            'safety': safety_rate,
            'restoration': mean_restoration,
            'toxicity': mean_toxicity
        }
        print(f" [{safety_rate:.0f}% Survival]")

    # Print Final Table
    print(f"\n{'='*75}")
    print(f" CLINICAL METRICS REPORT: {cancer_name}")
    print(f"{'='*75}")
    print(f"{'Scenario':<28} | {'Safety Rate':<12} | {'Restoration':<12} | {'Final Toxicity'}")
    print(f"{'-'*75}")
    
    for scenario in scenarios:
        res = metrics_results[scenario]
        print(f"{scenario:<28} | {res['safety']:>11.1f}% | {res['restoration']:>11.1f}% | {res['toxicity']:>10.2f} / {env.critical_toxicity_threshold}")
    print(f"{'='*75}\n")

def evaluate_specialist(cancer_name, model_path, data_path):
    print(f"\n{'='*80}")
    print(f" EVALUATING SPECIALIST: {cancer_name}")
    print(f"{'='*80}\n")

    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path, 
        drug_name=CANCER_PROFILES[cancer_name]['drug'], 
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRQN(input_dim=43, num_actions=10).to(device)
    
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
        history, reward, terminated = run_scenario(env, model, device, scenario)
        results[scenario] = {
            'history': history,
            'reward': reward,
            'failed': terminated,
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
            f.write(f"Final Toxicity: {res['final_tox']:.2f} / {env.critical_toxicity_threshold}\n\n")
            
    print(f"\n[INFO] Stress report saved to {report_path}")

    # 3. Generate Comparative Dashboard
    plot_comparative_dashboard(cancer_name, results, scenarios, healthy_baseline_ph, healthy_baseline_temp, env.critical_toxicity_threshold)

def plot_comparative_dashboard(cancer_name, results, scenarios, healthy_ph, healthy_temp, tox_threshold):
    """Generates a 12-panel (4x3) comparative dashboard."""
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 3, figsize=(18, 14), sharex='col', sharey='row')
    fig.suptitle(f'AI Clinical Stress Tests: {cancer_name.replace("_", " ")}', fontsize=18, fontweight='bold', y=0.98)

    for col, scenario in enumerate(scenarios):
        history = results[scenario]['history']
        
        # Column Titles
        axs[0, col].set_title(scenario, fontsize=14, fontweight='bold', pad=10)
        
        # Plot 1: pH (Row 0)
        axs[0, col].plot(history['time'], history['tumor_ph'], color='cyan', linewidth=2)
        axs[0, col].axhline(y=healthy_ph, color='lime', linestyle='--', alpha=0.7)
        axs[0, col].grid(True, alpha=0.2)
        if col == 0: axs[0, col].set_ylabel('Tumor pH')

        # Plot 2: Temp (Row 1)
        axs[1, col].plot(history['time'], history['tumor_temp'], color='tomato', linewidth=2)
        axs[1, col].axhline(y=healthy_temp, color='lime', linestyle='--', alpha=0.7)
        axs[1, col].grid(True, alpha=0.2)
        if col == 0: axs[1, col].set_ylabel('Tumor Temp (°C)')

        # Plot 3: Dose (Row 2)
        axs[2, col].step(history['time'], history['dose'], color='magenta', linewidth=2, where='post')
        axs[2, col].set_ylim(-5, 115) 
        axs[2, col].grid(True, alpha=0.2)
        if col == 0: axs[2, col].set_ylabel('Infusion Dose (%)')

        # Plot 4: Toxicity (Row 3)
        axs[3, col].plot(history['time'], history['toxicity'], color='orange', linewidth=2)
        axs[3, col].axhline(y=tox_threshold, color='red', linestyle='-', linewidth=2)
        axs[3, col].grid(True, alpha=0.2)
        axs[3, col].set_xlabel('Simulation Time (Minutes)')
        if col == 0: axs[3, col].set_ylabel('Systemic Toxicity')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = f"results/{cancer_name}_stress_dashboard.png"
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Dashboard saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Specialists against Clinical Stress Tests")
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), help="Target clinical profile.")
    parser.add_argument("--all", action="store_true", help="Evaluate all trained specialists.")
    parser.add_argument("--metrics", action="store_true", help="Run 100-patient statistical evaluation.")
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    
    if args.metrics:
        if args.all:
            for cancer_name in CANCER_PROFILES.keys():
                model_path = f"models/{cancer_name}_recurrent_drqn.pt"
                run_metrics_evaluation(cancer_name, model_path, dataset_path, num_patients=100)
        elif args.cancer:
            model_path = f"models/{args.cancer}_recurrent_drqn.pt"
            run_metrics_evaluation(args.cancer, model_path, dataset_path, num_patients=100)
        else:
            print("Please specify --cancer [name] or --all to run the metrics.")
    else:
        # Standard Plotting Evaluation
        if args.all:
            for cancer_name in CANCER_PROFILES.keys():
                model_path = f"models/{cancer_name}_recurrent_drqn.pt"
                evaluate_specialist(cancer_name, model_path, dataset_path)
        elif args.cancer:
            model_path = f"models/{args.cancer}_recurrent_drqn.pt"
            evaluate_specialist(args.cancer, model_path, dataset_path)
        else:
            print("Please specify --cancer [name] or --all to run the evaluation.")