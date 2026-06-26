"""
=============================================================================
Therapeutic Thermostat: Clinical Evaluation & Visualization
=============================================================================
Loads a trained DRQN specialist, runs a single 4-hour chemotherapy session,
and generates a multi-panel clinical dashboard plotting the agent's behavior.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import PredictiveChemoEnv
from train import DRQN, CANCER_PROFILES

def evaluate_specialist(cancer_name, model_path, data_path):
    print(f"\n{'='*60}")
    print(f" EVALUATING SPECIALIST: {cancer_name}")
    print(f"{'='*60}\n")

    # 1. Initialize Environment
    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path, 
        drug_name=CANCER_PROFILES[cancer_name]['drug'], 
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )
    
    # 2. Load Trained Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DRQN(input_dim=43, num_actions=10).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the agent first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # Set to evaluation mode (disables dropout, etc.)

    # 3. Trackers for Visualization
    history = {
        'time': [],
        'tumor_ph': [],
        'tumor_temp': [],
        'dose': [],
        'toxicity': []
    }

    obs, _ = env.reset()
    hidden = None
    terminated = False
    truncated = False
    total_reward = 0
    
    tz_idx = env.tumor_zone - 1
    healthy_baseline_ph = env.mu_B_ph[tz_idx]
    healthy_baseline_temp = env.mu_B_temp[tz_idx]

    print("Running 4-hour clinical simulation...")
    
    # 4. Run the Episode
    while not (terminated or truncated):
        # Convert observation to PyTorch tensor
        obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
        
        # Agent decides best action strictly based on learned policy (no exploration)
        with torch.no_grad():
            q_vals, hidden = model(obs_tensor, hidden)
            action = q_vals[0, 0, :].argmax().item()
            
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Log metrics for plotting
        history['time'].append(env.current_step)
        history['tumor_ph'].append(next_obs[tz_idx])           # Index for Tumor pH
        history['tumor_temp'].append(next_obs[tz_idx + 20])    # Index for Tumor Temperature (Offset by 20)
        history['toxicity'].append(next_obs[40])               # Index for Toxicity
        history['dose'].append(next_obs[41] * 100)             # Index for Dose (convert to %)
        
        obs = next_obs
        total_reward += reward

    # 5. Compile and Save Text Summary
    os.makedirs("results", exist_ok=True)
    
    result_status = "FAILURE. Toxicity threshold breached." if terminated else "SUCCESS. Patient completed session safely."
    
    summary_text = (
        f"Simulation Complete!\n"
        f"Total Accumulated Reward: {total_reward:.2f}\n"
        f"Final Toxicity Level: {history['toxicity'][-1]:.2f} / {env.critical_toxicity_threshold}\n"
        f"RESULT: {result_status}\n"
    )
    
    print(f"\n{summary_text}")
    
    txt_save_path = f"results/{cancer_name}_evaluation.txt"
    with open(txt_save_path, "w") as f:
        f.write(f"EVALUATION SUMMARY: {cancer_name}\n")
        f.write("-" * 40 + "\n")
        f.write(summary_text)
    print(f"Text output saved successfully to {txt_save_path}")

    # 6. Generate Clinical Dashboard
    plot_clinical_dashboard(cancer_name, history, healthy_baseline_ph, healthy_baseline_temp, env.critical_toxicity_threshold)

def plot_clinical_dashboard(cancer_name, history, healthy_ph, healthy_temp, tox_threshold):
    """Generates a 4-panel matplotlib dashboard of the episode."""
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f'Clinical Dashboard: {cancer_name.replace("_", " ")} Specialist', fontsize=16, fontweight='bold')

    # Plot 1: Tumor pH Efficacy
    axs[0].plot(history['time'], history['tumor_ph'], color='cyan', linewidth=2, label='Observed Tumor pH')
    axs[0].axhline(y=healthy_ph, color='lime', linestyle='--', alpha=0.7, label=f'Healthy Target ({healthy_ph:.2f})')
    axs[0].set_ylabel('Tumor pH')
    axs[0].set_title('Therapeutic Efficacy (Acidosis Reversal)')
    axs[0].legend(loc='lower right')
    axs[0].grid(True, alpha=0.2)

    # Plot 2: Tumor Temperature Efficacy
    axs[1].plot(history['time'], history['tumor_temp'], color='tomato', linewidth=2, label='Observed Tumor Temp')
    axs[1].axhline(y=healthy_temp, color='lime', linestyle='--', alpha=0.7, label=f'Healthy Target ({healthy_temp:.2f}°C)')
    axs[1].set_ylabel('Tumor Temp (°C)')
    axs[1].set_title('Therapeutic Efficacy (Hyperthermia Reversal)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.2)

    # Plot 3: Administered Dose
    axs[2].step(history['time'], history['dose'], color='magenta', linewidth=2, where='post')
    axs[2].set_ylabel('Dose (% of Nominal)')
    axs[2].set_title('AI-Controlled Chemotherapy Infusion Rate')
    
    # ---> UPDATED Y-AXIS TO SUPPORT 0% EMERGENCY STOP <---
    axs[2].set_ylim(-5, 115) 
    
    axs[2].grid(True, alpha=0.2)

    # Plot 4: Systemic Toxicity
    axs[3].plot(history['time'], history['toxicity'], color='orange', linewidth=2)
    axs[3].axhline(y=tox_threshold, color='red', linestyle='-', linewidth=2, label='Critical Failure Limit')
    axs[3].set_ylabel('Accumulated Toxicity')
    axs[3].set_title('Clearance Organ Safety Monitoring')
    axs[3].set_xlabel('Simulation Time (Minutes)')
    axs[3].legend(loc='upper left')
    axs[3].grid(True, alpha=0.2)

    plt.tight_layout()
    
    # Save the figure
    save_path = f"results/{cancer_name}_evaluation.png"
    plt.savefig(save_path, dpi=300)
    print(f"Dashboard saved successfully to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Trained AI Specialist")
    parser.add_argument("--cancer", type=str, required=True, choices=list(CANCER_PROFILES.keys()), 
                        help="The clinical profile to evaluate.")
    args = parser.parse_args()

    model_path = f"models/{args.cancer}_recurrent_drqn.pt"
    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    
    evaluate_specialist(args.cancer, model_path, dataset_path)