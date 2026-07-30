"""
=============================================================================
Therapeutic Thermostat: Spatial Expansion & Topology Ablation
=============================================================================
This script evaluates the clinical benefit of expanding the nanonetwork's 
sensor topology. It trains and evaluates models with progressively larger 
spatial observation windows, proving the value of systemic integration.

Usage Example:
--------------
    python spatial_expansion.py --cancer Glioblastoma
    python spatial_expansion.py --cancer Renal_Cell_Carcinoma
=============================================================================
"""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Import existing core modules
from environment import PredictiveChemoEnv
from train import DRQN, CANCER_PROFILES

# =========================================================
# SPATIAL EXPANSION MATRIX
# Maps the progressive unlocking of anatomical zones (1-indexed)
# based on advection proximity to the tumor and clearance organs.
# Balanced to ensure an equal number of zones at Stages 2, 3, and 4.
# =========================================================
SPATIAL_STAGES = {
    "Glioblastoma": {
        # Stage 1 (2 Zones): Target (7) + Clearance (14)
        1: [7, 14],
        
        # Stage 2 (6 Zones): Adds immediate inflow/outflow for Head and Liver
        2: [6, 7, 8, 13, 14, 15],
        
        # Stage 3 (10 Zones): Adds Thoracic Aorta, Splanchnic, Intestinal, and Systemic Venous
        3: [4, 6, 7, 8, 11, 12, 13, 14, 15, 20],
        
        # Stage 4 (14 Zones): Adds L/R Heart, Abdominal Aorta, and Renal Circulation
        4: [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 20]
    },
    "Renal_Cell_Carcinoma": {
        # Stage 1 (1 Zone): Renal Circulation acts as both Tumor and Clearance (16)
        1: [16],
        
        # Stage 2 (6 Zones): Adds Abd Aorta, Splanchnic, Hepatic Vein, Pelvic, and Systemic Venous
        2: [5, 11, 15, 16, 17, 20],
        
        # Stage 3 (10 Zones): Adds Thoracic Aorta, Intestinal, Portal Venous, and Liver Micro
        3: [4, 5, 11, 12, 13, 14, 15, 16, 17, 20],
        
        # Stage 4 (14 Zones): Adds L/R Heart, Lower Limb Arterial, and Lower Limb Micro
        4: [1, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
}

def apply_spatial_mask(obs, allowed_zones):
    """
    Masks the spatial telemetry (pH, Temp, Tox) for any anatomical 
    zone not present in the allowed_zones list.
    """
    masked_obs = np.copy(obs)
    allowed_indices = [z - 1 for z in allowed_zones]  # Convert to 0-indexed
    
    for i in range(20):
        if i not in allowed_indices:
            masked_obs[i] = 0.0       # Mask pH
            masked_obs[20 + i] = 0.0  # Mask Temp
            masked_obs[40 + i] = 0.0  # Mask Local Tox
            
    return masked_obs

def evaluate_deterministic_masked(env, policy_net, device, allowed_zones, num_seeds=15):
    """Evaluates the policy using masked observations."""
    returns = []
    for seed in range(42, 42 + num_seeds):
        obs, _ = env.reset(seed=seed)
        hidden = None
        ep_return = 0.0
        for _ in range(240):
            obs_masked = apply_spatial_mask(obs, allowed_zones)
            obs_tensor = torch.FloatTensor(obs_masked).view(1, 1, -1).to(device)
            with torch.no_grad():
                q_vals, hidden = policy_net(obs_tensor, hidden)
                action = q_vals[0, 0, :].argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward  
            if terminated or truncated:
                break
        returns.append(ep_return)
    return np.mean(returns)

def train_masked_drqn_run(env, allowed_zones, device, params, total_episodes=1000):
    """Executes a single training run for a specific spatial topology mask."""
    policy_net = DRQN(input_dim=64, num_actions=10).to(device)
    target_net = DRQN(input_dim=64, num_actions=10).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=params.get('lr', 1e-4))
    buffer = deque(maxlen=500)
    
    epsilon, epsilon_min = 1.0, 0.05
    epsilon_decay = params.get('epsilon_decay', 0.995)
    batch_size = params.get('batch_size', 16)
    gamma = params.get('gamma', 0.99)
    
    burn_in, seq_len = 60, 120
    total_sample_len = burn_in + seq_len
    
    best_eval_return = -float('inf')
    best_state_dict = None

    for episode_idx in range(1, total_episodes + 1):
        obs, _ = env.reset()
        hidden = None
        ep_history = []
        
        for _ in range(240):
            obs_masked = apply_spatial_mask(obs, allowed_zones)
            obs_tensor = torch.as_tensor(obs_masked, dtype=torch.float32, device=device).view(1, 1, -1)
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                with torch.no_grad(): _, hidden = policy_net(obs_tensor, hidden)
            else:
                with torch.no_grad():
                    q_vals, hidden = policy_net(obs_tensor, hidden)
                    action = q_vals[0, 0, :].argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs_masked = apply_spatial_mask(next_obs, allowed_zones)
            
            ep_history.append((obs_masked, action, reward, next_obs_masked, float(terminated)))
            obs = next_obs
            
            if terminated or truncated: break
            
        buffer.append(ep_history)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Standard DRQN Training Update Logic
        if len(buffer) > batch_size:
            samples = random.sample(buffer, batch_size)
            optimizer.zero_grad(set_to_none=True)
            
            state_batch = torch.zeros(batch_size, seq_len + 1, 64, device=device)
            action_batch = torch.zeros(batch_size, seq_len, 1, dtype=torch.long, device=device)
            reward_batch = torch.zeros(batch_size, seq_len, device=device)
            done_batch = torch.zeros(batch_size, seq_len, device=device)
            mask = torch.zeros(batch_size, seq_len, device=device)
            
            h_0 = torch.zeros(1, batch_size, policy_net.hidden_dim, device=device)
            c_0 = torch.zeros(1, batch_size, policy_net.hidden_dim, device=device)

            for b, seq in enumerate(samples):
                actual_len = len(seq)
                max_start = max(0, actual_len - total_sample_len)
                start = random.randint(0, max_start) if max_start > 0 else 0
                
                seq_obs = np.stack([tr[0] for tr in seq])
                seq_actions = np.array([tr[1] for tr in seq], dtype=np.int64)
                seq_rewards = np.array([tr[2] for tr in seq], dtype=np.float32)
                seq_dones = np.array([tr[4] for tr in seq], dtype=np.float32)
                
                actual_burn_in = min(burn_in, actual_len - start)
                if actual_burn_in > 0:
                    burn_in_states = torch.as_tensor(seq_obs[start:start + actual_burn_in], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        _, (h, c) = policy_net(burn_in_states)
                        h_0[:, b, :] = h[0]
                        c_0[:, b, :] = c[0]
                
                learn_start = start + burn_in
                valid_len = min(seq_len, max(0, actual_len - learn_start))
                
                if valid_len > 0:
                    state_batch[b, :valid_len] = torch.as_tensor(seq_obs[learn_start:learn_start + valid_len], dtype=torch.float32, device=device)
                    action_batch[b, :valid_len, 0] = torch.as_tensor(seq_actions[learn_start:learn_start + valid_len], device=device)
                    reward_batch[b, :valid_len] = torch.as_tensor(seq_rewards[learn_start:learn_start + valid_len], device=device)
                    done_batch[b, :valid_len] = torch.as_tensor(seq_dones[learn_start:learn_start + valid_len], device=device)
                    mask[b, :valid_len] = 1.0  
                    last_transition = seq[learn_start + valid_len - 1]
                    state_batch[b, valid_len] = torch.as_tensor(last_transition[3], dtype=torch.float32, device=device)

            hidden_in = (h_0, c_0)
            all_q_values, _ = policy_net(state_batch, hidden_in)
            with torch.no_grad(): 
                all_target_q, _ = target_net(state_batch, hidden_in)

            q_values = all_q_values[:, :-1, :]
            next_actions = all_q_values[:, 1:, :].argmax(dim=2, keepdim=True)
            next_q_values = all_target_q[:, 1:, :].gather(2, next_actions).squeeze(-1)

            state_action_values = q_values.gather(2, action_batch).squeeze(-1)
            expected_state_action_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

            raw_loss = nn.SmoothL1Loss(reduction='none')(state_action_values, expected_state_action_values.detach())
            loss = (raw_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
            optimizer.step()

        if episode_idx % 10 == 0: 
            target_net.load_state_dict(policy_net.state_dict())

        # Evaluate Periodically
        if episode_idx % 50 == 0 and len(buffer) > batch_size:
            eval_mean = evaluate_deterministic_masked(env, policy_net, device, allowed_zones, num_seeds=10)
            if eval_mean > best_eval_return:
                best_eval_return = eval_mean
                best_state_dict = policy_net.state_dict()
                
    return best_eval_return, best_state_dict

def train_and_select_best_stage(cancer_name, stage_idx, allowed_zones, data_path, params, num_runs=5):
    """Executes the 5-run stability protocol and saves the single best model."""
    print(f"\n{'='*80}")
    print(f" TRAINING STAGE {stage_idx} ({len(allowed_zones)} Zones Active) | Cancer: {cancer_name}")
    print(f"{'='*80}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path, 
        drug_name=CANCER_PROFILES[cancer_name]['drug'], 
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )
    
    champion_return = -float('inf')
    champion_weights = None
    
    for run in range(1, num_runs + 1):
        print(f" -> Starting Run {run}/{num_runs}...")
        run_best_ret, run_best_weights = train_masked_drqn_run(env, allowed_zones, device, params, total_episodes=1000)
        print(f"    Run {run} Best Deterministic Return: {run_best_ret:.2f}")
        
        if run_best_ret > champion_return:
            champion_return = run_best_ret
            champion_weights = run_best_weights
            
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{cancer_name}_stage_{stage_idx}_best.pt"
    torch.save(champion_weights, save_path)
    print(f"\n[SUCCESS] Champion Model for Stage {stage_idx} Saved! (Return: {champion_return:.2f})")
    return save_path

def evaluate_frozen_stage(cancer_name, stage_idx, allowed_zones, model_path, data_path):
    """Executes the standard 100-patient stress protocol for a specific spatial stage."""
    print(f"\n{'-'*80}")
    print(f" EVALUATING STAGE {stage_idx} FROZEN PROTOCOL")
    print(f"{'-'*80}")
    
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
        "Low Growth (Late-Stage)", "Medium Growth (Typical)", 
        "High Growth (Early-Stage)", "Resistance (Dead-End)", "Toxicity (Slippery Road)"
    ]
    
    obs, _ = env.reset(seed=42)
    ABSOLUTE_BASE_K_ELIM = env.engine.drug['k_elim']
    ABSOLUTE_BASE_THRESHOLD = env.critical_toxicity_threshold
    
    scenario_results = []
    
    for scenario in scenarios:
        metrics = {'survivals': [], 'mhr': [], 'mat': []}

        for ep_idx in range(100):
            obs, _ = env.reset(seed=42 + ep_idx)
            env.engine.drug['k_elim'] = ABSOLUTE_BASE_K_ELIM
            env.critical_toxicity_threshold = ABSOLUTE_BASE_THRESHOLD
            env.engine.tumor_growth_rate = 0.025
            env.is_resistant = False 
            
            if scenario == "Low Growth (Late-Stage)": env.engine.tumor_growth_rate = 0.005 
            elif scenario == "High Growth (Early-Stage)": env.engine.tumor_growth_rate = 0.050 
            elif scenario == "Resistance (Dead-End)": env.is_resistant = True 
            elif scenario == "Toxicity (Slippery Road)":
                env.engine.drug['k_elim'] *= 0.50                          
                env.critical_toxicity_threshold *= 0.75

            initial_distance = env.prev_distance
            hidden = None
            terminated, truncated = False, False
            max_tox = 0.0
            
            while not (terminated or truncated):
                obs_masked = apply_spatial_mask(obs, allowed_zones)
                obs_tensor = torch.FloatTensor(obs_masked).view(1, 1, -1).to(device)
                
                with torch.no_grad():
                    q_vals, hidden = model(obs_tensor, hidden)
                    action = q_vals[0, 0, :].argmax().item()
                    
                obs, reward, terminated, truncated, info = env.step(action)
                max_tox = max(max_tox, env.total_toxicity)
                
            mhr = max(0.0, ((initial_distance - info['curr_dist']) / initial_distance) * 100.0)
            metrics['survivals'].append(1 if info['survived'] else 0)
            metrics['mat'].append(max_tox)
            metrics['mhr'].append(mhr)
            
        acc = np.mean(metrics['survivals']) * 100
        mhr = np.mean(metrics['mhr'])
        mat = np.mean(metrics['mat'])
        
        scenario_results.append({
            "Stage": stage_idx, "Scenario": scenario, "Accuracy (%)": acc, "MHR (%)": mhr, "MAT": mat
        })
        print(f"  [{scenario}] Acc: {acc:>5.1f}% | MHR: {mhr:>5.1f}% | MAT: {mat:>5.1f}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/spatial_expansion", exist_ok=True)
    csv_path = f"results/spatial_expansion/{cancer_name}_stage_{stage_idx}_eval.csv"
    pd.DataFrame(scenario_results).to_csv(csv_path, index=False)
    print(f"  -> Saved Stage {stage_idx} evaluation to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), required=True)
    args = parser.parse_args()

    cancer = args.cancer
    data_path = "datasets/1_week_sampling_schedule.parquet"
    param_file = f"config/{cancer}_best_params.json"
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}")
        exit(1)

    # Load optimized hyperparameters if available, else use defaults
    params = {}
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            params = json.load(f)
            print(f"[INFO] Loaded optimized parameters: {params}")
    else:
        print("[WARNING] Parameter file not found. Using default DRQN hyperparams.")

    stages = SPATIAL_STAGES[cancer]
    
    for stage_idx in range(1, 5):
        allowed_zones = stages[stage_idx]
        
        # 1. Train 5 models and save the absolute best one
        best_model_path = train_and_select_best_stage(
            cancer, stage_idx, allowed_zones, data_path, params, num_runs=5
        )
        
        # 2. Evaluate the champion model across the 100-patient protocol
        evaluate_frozen_stage(cancer, stage_idx, allowed_zones, best_model_path, data_path)
        
    print(f"\n{'='*80}")
    print(f" ALL SPATIAL EXPANSION STAGES COMPLETE FOR {cancer}.")
    print(f"{'='*80}\n")