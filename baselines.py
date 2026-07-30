"""
=============================================================================
Therapeutic Thermostat: Clinical Baselines Evaluation
=============================================================================
This script evaluates the clinical anchors against which the AI is compared:

1. Standard of Care (SOC): A static, clinically-optimized dose with a 
   simulated "Nurse Emergency Abort" if vitals crash.
2. Adaptive Therapy: A dynamic, rule-based clinical protocol that balances 
   tumor burden and Treatment-Related Adverse Events (TRAEs) by escalating, 
   de-escalating, and employing intermittent drug holidays.

Usage Example:
--------------
    python baselines.py --cancer Glioblastoma

Outputs:
--------------
    - results/<Cancer_Name>_clinical_baselines_summary.csv
=============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from environment import PredictiveChemoEnv
from train import CANCER_PROFILES

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


def run_baselines(cancer_name, data_path, num_episodes=100):
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
    
    modes = ["Standard of Care", "Adaptive Therapy"]
    scenario_results = []

    # Capture absolute baseline constants once
    obs, _ = env.reset(seed=42)
    ABSOLUTE_BASE_K_ELIM = env.engine.drug['k_elim']
    ABSOLUTE_BASE_THRESHOLD = env.critical_toxicity_threshold
    
    # 1. Find optimal static dose for Standard of Care
    optimal_soc_action = find_optimal_static_dose(env, num_seeds=15)
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f" EVALUATING CLINICAL BASELINE: {mode}")
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
                total_reward = 0.0
                terminated, truncated = False, False
                max_tox_observed = 0.0
                final_distance = initial_distance
                survived = True 
                
                # Baseline State Variables
                abort_threshold = env.critical_toxicity_threshold * 0.90
                emergency_aborted = False
                adaptive_action = optimal_soc_action
                
                while not (terminated or truncated):
                    
                    if mode == "Standard of Care":
                        # Standard of Care (Static Action + Emergency Abort)
                        if env.total_toxicity >= abort_threshold:
                            emergency_aborted = True
                        action = 0 if emergency_aborted else optimal_soc_action
                        
                    elif mode == "Adaptive Therapy":
                        # Adaptive Therapy (Balanced Triangle: Tumor Burden vs TRAEs)
                        holiday_threshold = env.critical_toxicity_threshold * 0.85
                        deescalate_threshold = env.critical_toxicity_threshold * 0.65
                        escalate_threshold = env.critical_toxicity_threshold * 0.40
                        
                        # Apply rules based on continuous physiological feedback
                        if env.total_toxicity >= holiday_threshold:
                            adaptive_action = 0  # Intermittent Strategy (Drug Holiday)
                        elif env.total_toxicity >= deescalate_threshold:
                            adaptive_action = max(1, adaptive_action - 1)  # De-escalation
                        elif env.total_toxicity < escalate_threshold and env.prev_distance > 0.5:
                            adaptive_action = min(9, adaptive_action + 1)  # Escalation (High Burden + Safe Vitals)
                            
                        action = adaptive_action
                        
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
    csv_path = f"results/{cancer_name}_clinical_baselines_summary.csv"
    df_results.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f" ALL BASELINES COMPLETE. Saved to {csv_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), required=True)
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    run_baselines(args.cancer, dataset_path, num_episodes=100)