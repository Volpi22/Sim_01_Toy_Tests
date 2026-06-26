"""
This script handles the training and automated hyperparameter optimization 
of Deep Recurrent Q-Network agents.

COMMAND-LINE ARGUMENTS:
    --cancer     : Name of the specific cancer profile to train. 
    --all        : Flag to train all 5 specialists sequentially.
    --episodes   : Number of episodes to train each agent. (Default: 1500)
    --optimize   : Flag to launch Optuna Hyperparameter Optimization.
    --n_trials   : Number of Optuna trials to run if optimizing. (Default: 30)

USAGE EXAMPLES:
1. Standard Training (Uses default or saved optimal parameters):
   $ python3 train.py --cancer Hepatocellular_Carcinoma --episodes 1500
   
2. Hyperparameter Optimization (Finds and saves the best parameters automatically):
   $ python3 train.py --cancer Glioblastoma --optimize --n_trials 30
=============================================================================
"""

import os
import random
import argparse
import json
import numpy as np
from collections import deque

import torch
torch.set_num_threads(torch.get_num_threads())
import torch.nn as nn
import torch.optim as optim

# Import Optuna for Bayesian Hyperparameter Optimization
import optuna

from environment import PredictiveChemoEnv

CANCER_PROFILES = {
    'Hepatocellular_Carcinoma': {'drug': 'Doxorubicin', 'zone': 14},
    'Colorectal_Cancer':        {'drug': '5-Fluorouracil', 'zone': 12},
    'Lung_Cancer':              {'drug': 'Paclitaxel', 'zone': 2},
    'Renal_Cell_Carcinoma':     {'drug': 'Cisplatin', 'zone': 16},
    'Glioblastoma':             {'drug': 'Carmustine', 'zone': 7}
}

class DRQN(nn.Module):
    """Deep Recurrent Q-Network Architecture for Multi-Rate Signal Fusion."""
    def __init__(self, input_dim=43, num_actions=10, hidden_dim=256):
        super(DRQN, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.q_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _ = x.size()
        
        flat_x = x.contiguous().view(batch_size * seq_len, -1)
        features = self.feature_network(flat_x)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, hidden_state = self.lstm(features, hidden_state)
        q_values = self.q_head(lstm_out)
        return q_values, hidden_state

class EpisodeReplayBuffer:
    """Sequential buffer to preserve chronological tracking data."""
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def push(self, episode):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_drqn_specialist(cancer_name, data_path, total_episodes=1500, 
                          batch_size=16, gamma=0.99, lr=1e-3, epsilon_decay=0.99, 
                          save_model=True, trial=None):
    """
    Core training loop. If 'trial' is provided, it acts as an Optuna evaluation 
    run and enables early pruning.
    """
    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path, 
        drug_name=CANCER_PROFILES[cancer_name]['drug'], 
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DRQN(input_dim=43, num_actions=10).to(device)
    target_net = DRQN(input_dim=43, num_actions=10).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = EpisodeReplayBuffer(capacity=500)
    
    epsilon = 1.0
    epsilon_min = 0.05
    
    recent_rewards = deque(maxlen=20)
    
    for episode_idx in range(total_episodes):
        obs, _ = env.reset()
        episode_history = []
        hidden = None
        
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals, next_hidden = policy_net(obs_tensor, hidden)
                    action = q_vals[0, 0, :].argmax().item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_history.append((obs, action, reward, next_obs, float(terminated)))
            
            obs = next_obs
            episode_reward += reward

        buffer.push(episode_history)
        recent_rewards.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # DRQN Sequence Optimization Step
        if len(buffer) > batch_size:
            samples = buffer.sample(batch_size)
            optimizer.zero_grad()
            
            loss = 0
            for seq in samples:
                states = torch.FloatTensor(np.array([t[0] for t in seq])).unsqueeze(0).to(device)
                actions = torch.LongTensor(np.array([t[1] for t in seq])).unsqueeze(0).unsqueeze(-1).to(device)
                rewards = torch.FloatTensor(np.array([t[2] for t in seq])).unsqueeze(0).to(device)
                next_states = torch.FloatTensor(np.array([t[3] for t in seq])).unsqueeze(0).to(device)
                dones = torch.FloatTensor(np.array([t[4] for t in seq])).unsqueeze(0).to(device)
                
                q_values, _ = policy_net(states)
                next_q_values, _ = target_net(next_states)
                
                state_action_values = q_values.gather(2, actions).squeeze(-1)
                expected_state_action_values = rewards + (gamma * next_q_values.max(dim=2)[0] * (1 - dones))
                
                loss += nn.MSELoss()(state_action_values, expected_state_action_values.detach())
            
            loss = loss / batch_size
            loss.backward()
            optimizer.step()
            
        # Unmuted Logging: Print progress every 10 episodes for BOTH optimization and final training
        if episode_idx % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
            phase_label = f"[Trial {trial.number}]" if trial is not None else "[Main Training]"
            print(f"{phase_label} Episode {episode_idx}/{total_episodes} | Return Score: {episode_reward:.2f} | Exploration Epsilon: {epsilon:.2f}")
        
        # Optuna Pruning Integration
        if trial is not None:
            trial.report(episode_reward, episode_idx)
            if trial.should_prune():
                print(f"[Trial {trial.number}] PRUNED: Agent failed to learn quickly enough. Moving to next trial.")
                raise optuna.exceptions.TrialPruned()

    # Save the model
    if save_model:
        os.makedirs("models", exist_ok=True)
        save_path = f"models/{cancer_name}_recurrent_drqn.pt"
        torch.save(policy_net.state_dict(), save_path)
        print(f"\n[SUCCESS] Recurrent Specialist saved to {save_path}\n")

    return np.mean(recent_rewards) if len(recent_rewards) > 0 else episode_reward


def optimize_specialist(cancer_name, data_path, n_trials=30, total_episodes=1000):
    """Optuna Study: intelligently searches for the best parameters and saves them."""
    print(f"\n{'='*60}")
    print(f" LAUNCHING OPTUNA HYPERPARAMETER OPTIMIZATION: {cancer_name}")
    print(f" Trials: {n_trials} | Episodes per Trial: {total_episodes}")
    print(f"{'='*60}\n")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.995)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        
        target_episode = int(0.8 * total_episodes)
        dynamic_decay = (0.05) ** (1.0 / target_episode)
        
        print(f"\n>>> STARTING TRIAL {trial.number} <<<")
        print(f"Testing Parameters -> LR: {lr:.5f} | Gamma: {gamma:.4f} | Batch Size: {batch_size} | Eps Decay: {dynamic_decay:.4f}")
        print("-" * 60)
        
        score = train_drqn_specialist(
            cancer_name=cancer_name, 
            data_path=data_path,
            total_episodes=total_episodes,
            batch_size=batch_size,
            gamma=gamma,
            lr=lr,
            epsilon_decay=dynamic_decay,
            save_model=False,
            trial=trial
        )
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n[OPTIMIZATION COMPLETE]")
    print(f"Best Trial: {study.best_trial.number}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    # --- SAVE BEST PARAMETERS TO JSON ---
    os.makedirs("config", exist_ok=True)
    config_path = f"config/{cancer_name}_best_params.json"
    with open(config_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\n[INFO] Saved optimal parameters to {config_path}")
        
    final_episodes = 1500
    final_decay = (0.05) ** (1.0 / int(0.8 * final_episodes))
        
    print("\n" + "="*60)
    print(f" RETRAINING FINAL AGENT WITH OPTIMAL PARAMETERS")
    print(f" Episodes: {final_episodes}")
    print("="*60 + "\n")
    
    train_drqn_specialist(
        cancer_name=cancer_name,
        data_path=data_path,
        total_episodes=final_episodes,
        batch_size=study.best_params["batch_size"],
        gamma=study.best_params["gamma"],
        lr=study.best_params["lr"],
        epsilon_decay=final_decay,
        save_model=True
    )

def run_standard_training(cancer_name, data_path, total_episodes):
    """Runs a standard training loop, injecting optimized parameters if available."""
    config_path = f"config/{cancer_name}_best_params.json"
    
    # 1. Safe default fallback parameters
    params = {"lr": 1e-3, "gamma": 0.99, "batch_size": 16}
    
    # 2. Check for saved Optuna parameters
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            saved_params = json.load(f)
            # Update the default dictionary with the saved one
            params.update(saved_params) 
        print(f"\n[INFO] Loaded optimal parameters from {config_path}:")
        print(f"       LR: {params['lr']:.5f} | Gamma: {params['gamma']:.4f} | Batch Size: {params['batch_size']}")
    else:
        print(f"\n[WARNING] No optimal parameters found in 'config/'. Using default values.")
        
    final_decay = (0.05) ** (1.0 / int(0.8 * total_episodes))
    
    print("\n" + "="*60)
    print(f" LAUNCHING STANDARD TRAINING: {cancer_name}")
    print(f" Episodes: {total_episodes}")
    print("="*60 + "\n")
    
    train_drqn_specialist(
        cancer_name=cancer_name,
        data_path=data_path,
        total_episodes=total_episodes,
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        lr=params["lr"],
        epsilon_decay=final_decay,
        save_model=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Optimize AI Recurrent Specialists")
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), help="Target clinical profile.")
    parser.add_argument("--all", action="store_true", help="Sequential batch training flag.")
    parser.add_argument("--episodes", type=int, default=1500, help="Episodes per training run.")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna Hyperparameter Optimization.")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials to test.")
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset at {dataset_path}. Run dataset_generation.py first.")

    # Execute Optuna Tuning
    if args.optimize:
        if args.cancer:
            optimize_specialist(args.cancer, dataset_path, n_trials=args.n_trials)
        elif args.all:
            for name in CANCER_PROFILES.keys():
                optimize_specialist(name, dataset_path, n_trials=args.n_trials)
        else:
            print("Please specify --cancer [name] or --all to run optimization.")
            
    # Execute Standard Training (with smart parameter loading)
    else:
        if args.all:
            for name in CANCER_PROFILES.keys():
                run_standard_training(name, dataset_path, args.episodes)
        elif args.cancer:
            run_standard_training(args.cancer, dataset_path, args.episodes)
        else:
            print("Please use --cancer [name] or --all to launch training.")