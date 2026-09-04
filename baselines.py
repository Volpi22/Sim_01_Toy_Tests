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
=============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from environment import PredictiveChemoEnv
from train import CANCER_PROFILES

def find_optimal_static_dose(env, num_seeds=10):
    best_action = 1
    print(f"\n{'-'*80}")
    print(f" [SOC GRID SEARCH] Finding Optimal Standard-of-Care Dose")
    print(f"{'-'*80}")
    
    for action in range(1, 10):
        survivals = []
        for seed in range(num_seeds):
            env.reset(seed=42 + seed)
            env.engine.tumor_growth_rate = 0.025
            env.is_resistant = False
            
            terminated, truncated = False, False
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(action)
                
            survivals.append(1 if info['survived'] else 0)
            
        surv_rate = np.mean(survivals)
        dose_pct = env.action_mapping[action] * 100
        
        if surv_rate == 1.0:
            best_action = action
            print(f"  -> Action {action} ({dose_pct:>3.0f}%): SURVIVED (100%)")
        else:
            print(f"  -> Action {action} ({dose_pct:>3.0f}%): FAILED   ({surv_rate*100:.0f}% Survival) - Threshold Breached")
            break 
            
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

    obs, _ = env.reset(seed=42)
    ABSOLUTE_BASE_K_ELIM = env.engine.drug['k_elim']
    ABSOLUTE_BASE_THRESHOLD = env.critical_toxicity_threshold
    
    optimal_soc_action = find_optimal_static_dose(env, num_seeds=15)
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f" EVALUATING CLINICAL BASELINE: {mode}")
        print(f"{'='*80}")
        
        for scenario in scenarios:
            metrics = {'survivals': [], 'mhr': [], 'mat': [], 'returns': []}

            for ep_idx in range(num_episodes):
                obs, _ = env.reset(seed=42 + ep_idx)
                
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

                # --- CONCEPTUAL FIX: BYPASS EARLY TERMINATION ---
                # We save the true threshold to check survival, but set the env threshold 
                # to infinity so the simulation continues until the cancer is cured or time runs out.
                true_critical_tox = env.critical_toxicity_threshold
                env.critical_toxicity_threshold = 999999.0 

                initial_distance = env.prev_distance
                total_reward = 0.0
                terminated, truncated = False, False
                max_tox_observed = 0.0
                final_distance = initial_distance
                survived = True 
                
                abort_threshold = true_critical_tox * 0.90
                emergency_aborted = False
                adaptive_action = optimal_soc_action
                
                while not (terminated or truncated):
                    
                    if mode == "Standard of Care":
                        if env.total_toxicity >= abort_threshold:
                            emergency_aborted = True
                        action = 0 if emergency_aborted else optimal_soc_action
                        
                    elif mode == "Adaptive Therapy":
                        holiday_threshold = true_critical_tox * 0.85
                        deescalate_threshold = true_critical_tox * 0.65
                        escalate_threshold = true_critical_tox * 0.40
                        
                        if env.total_toxicity >= holiday_threshold:
                            adaptive_action = 0  
                        elif env.total_toxicity >= deescalate_threshold:
                            adaptive_action = max(1, adaptive_action - 1)  
                        elif env.total_toxicity < escalate_threshold and env.prev_distance > 0.5:
                            adaptive_action = min(9, adaptive_action + 1)  
                            
                        action = adaptive_action
                        
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Manually track if the patient crashed at any point
                    if env.total_toxicity >= true_critical_tox:
                        survived = False
                        
                    total_reward += reward
                    max_tox_observed = max(max_tox_observed, env.total_toxicity)
                    final_distance = info['curr_dist']
                    
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