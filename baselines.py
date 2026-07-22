"""
=============================================================================
Therapeutic Thermostat: Baseline & Spatial Ablation Evaluation
=============================================================================
This script proves the necessity of a spatially-aware, closed-loop AI by 
evaluating three definitive clinical baselines:

1. Tumor-Only Vision: DRQN is blinded to clearance organ toxicity.
2. Clearance-Only Vision: DRQN is blinded to localized tumor growth.
3. Standard of Care (SOC): A static, clinically-optimized dose with a 
   simulated "Nurse Emergency Abort" if vitals crash.

Usage Example:
--------------
    python baselines.py --cancer Glioblastoma

Outputs:
--------------
    - results/<Cancer_Name>_baselines_summary.csv
=============================================================================
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from environment import PredictiveChemoEnv
from train import DRQN, CANCER_PROFILES

def find_optimal_static_dose(env, num_seeds=10):
    """
    Simulates decades of clinical trials by finding the absolute maximum 
    tolerable static dose for the 'average' (Medium Growth) patient.
    """
    best_action = 1
    print(f"\n{'-'*80}")
    print(f" [SOC GRID SEARCH] Finding Optimal Standard-of-Care Dose")
    print(f"{'-'*80}")
    
    # Test actions 1 through 9 (skipping 0)
    for action in range(1, 10):
        survivals = []
        for seed in range(num_seeds):
            env.reset(seed=42 + seed)
            
            # Lock to typical baseline patient
            env.engine.tumor_growth_rate = 0.025
            env.is_resistant = False
            
            terminated, truncated = False, False
            while not (terminated or truncated):
                # Run WITHOUT emergency abort to find inherently safe dose
                _, _, terminated, truncated, info = env.step(action)
                
            survivals.append(1 if info['survived'] else 0)
            
        surv_rate = np.mean(survivals)
        dose_pct = env.action_mapping[action] * 100
        
        if surv_rate == 1.0:
            best_action = action
            print(f"  -> Action {action} ({dose_pct:>3.0f}%): SURVIVED (100%)")
        else:
            print(f"  -> Action {action} ({dose_pct:>3.0f}%): FAILED   ({surv_rate*100:.0f}% Survival) - Threshold Breached")
            break # If this dose fails, higher doses will also fail.
            
    print(f"{'-'*80}")
    print(f" Selected Optimal SOC Action: {best_action} ({env.action_mapping[best_action]*100:.0f}%)")
    print(f"{'-'*80}\n")
    return best_action

def mask_observation(obs, mode, env):
    """
    Masks the 64-dimensional observation tensor by zeroing out telemetry
    outside the permitted spatial zones to simulate traditional sensor limits.
    """
    masked = np.zeros_like(obs)
    tz_idx = env.tumor_zone - 1
    cz_idx = env.engine.drug['clearance_zone'] - 1
    
    if mode == "Tumor-Only Vision":
        # Keep tumor spatial data (pH, Temp, local tox)
        masked[tz_idx] = obs[tz_idx]
        masked[20 + tz_idx] = obs[20 + tz_idx]
        masked[40 + tz_idx] = obs[40 + tz_idx]
        # Keep generic globals (dose, clearance factor)
        masked[61] = obs[61] 
        masked[63] = obs[63]
        # Purposely blinding global toxicity (60) and clearance conc (62)
        
    elif mode == "Clearance-Only Vision":
        # Keep clearance spatial data (pH, Temp, local tox)
        masked[cz_idx] = obs[cz_idx]
        masked[20 + cz_idx] = obs[20 + cz_idx]
        masked[40 + cz_idx] = obs[40 + cz_idx]
        # Keep global tox data
        masked[60] = obs[60]
        masked[61] = obs[61]
        masked[62] = obs[62]
        masked[63] = obs[63]
        
    return masked

def run_baselines(cancer_name, model_path, data_path, num_episodes=100):
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
    
    modes = ["Tumor-Only Vision", "Clearance-Only Vision", "Standard of Care"]
    scenario_results = []

    # Capture absolute baseline constants once
    obs, _ = env.reset(seed=42)
    ABSOLUTE_BASE_K_ELIM = env.engine.drug['k_elim']
    ABSOLUTE_BASE_THRESHOLD = env.critical_toxicity_threshold
    
    # 1. Find optimal static dose for Standard of Care
    optimal_soc_action = find_optimal_static_dose(env, num_seeds=15)
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f" EVALUATING BASELINE: {mode}")
        print(f"{'='*80}")
        
        for scenario in scenarios:
            metrics = {'survivals': [], 'mhr': [], 'mat': [], 'returns': []}

            for ep_idx in range(num_episodes):
                obs, _ = env.reset(seed=42 + ep_idx)
                
                # Reset environment to absolute baselines BEFORE applying scenario modifiers
                env.engine.drug['k_elim'] = ABSOLUTE_BASE_K_ELIM
                env.critical_toxicity_threshold = ABSOLUTE_BASE_THRESHOLD
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

                initial_distance = env.prev_distance
                hidden = None
                total_reward = 0.0
                terminated, truncated = False, False
                max_tox_observed = 0.0
                final_distance = initial_distance
                survived = True 
                
                # Emergency Abort Threshold (90% of Critical Limit) for SOC
                abort_threshold = env.critical_toxicity_threshold * 0.90
                emergency_aborted = False
                
                while not (terminated or truncated):
                    
                    if mode in ["Tumor-Only Vision", "Clearance-Only Vision"]:
                        # Mask the observation and use DRQN
                        obs_masked = mask_observation(obs, mode, env)
                        obs_tensor = torch.FloatTensor(obs_masked).view(1, 1, -1).to(device)
                        with torch.no_grad():
                            q_vals, hidden = model(obs_tensor, hidden)
                            action = q_vals[0, 0, :].argmax().item()
                    else:
                        # Standard of Care (Static Action + Abort)
                        if env.total_toxicity >= abort_threshold:
                            emergency_aborted = True
                        action = 0 if emergency_aborted else optimal_soc_action
                        
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    max_tox_observed = max(max_tox_observed, env.total_toxicity)
                    final_distance = info['curr_dist']
                    survived = info['survived']
                    
                mhr_percent = max(0.0, ((initial_distance - final_distance) / initial_distance) * 100.0)
                    
                metrics['returns'].append(total_reward)
                metrics['survivals'].append(1 if survived else 0)
                metrics['mat'].append(max_tox_observed)
                metrics['mhr'].append(mhr_percent)
                
            mean_acc = np.mean(metrics['survivals']) * 100
            mean_mhr = np.mean(metrics['mhr'])
            mean_mat = np.mean(metrics['mat'])
            mean_mer = np.mean(metrics['returns'])
            
            scenario_results.append({
                "Baseline Type": mode,
                "Scenario": scenario,
                "Accuracy (%)": mean_acc,
                "MHR (%)": mean_mhr,
                "MAT": mean_mat,
                "MER": mean_mer
            })
            
            print(f"{scenario:<28} | Acc: {mean_acc:>5.1f}% | MHR: {mean_mhr:>5.1f}% | MAT: {mean_mat:>5.1f}")

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(scenario_results)
    csv_path = f"results/{cancer_name}_baselines_summary.csv"
    df_results.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f" ALL BASELINES COMPLETE. Saved to {csv_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), required=True)
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    model_path = f"models/{args.cancer}_drqn_best.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}. Ensure the DRQN is trained first.")
    else:
        run_baselines(args.cancer, model_path, dataset_path, num_episodes=100)