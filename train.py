"""
=============================================================================
Therapeutic Thermostat: DRQN Specialist Training Pipeline
=============================================================================
Trains targeted AI specialists for specific clinical pathologies using a 
Deep Recurrent Q-Network (DRQN) with Truncated Backpropagation Through Time.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt  # <-- Added for plotting
from collections import deque
from environment import PredictiveChemoEnv

# Clinical Targets mapping anatomical zones and primary chemotherapies
CANCER_PROFILES = {
    "Hepatocellular_Carcinoma": {"zone": 14, "drug": "Doxorubicin"},
    "Colorectal_Cancer": {"zone": 12, "drug": "5-Fluorouracil"},
    "Lung_Cancer": {"zone": 2, "drug": "Paclitaxel"},
    "Renal_Cell_Carcinoma": {"zone": 16, "drug": "Cisplatin"},
    "Glioblastoma": {"zone": 7, "drug": "Carmustine"}
}

class DRQN(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_actions=10):
        super(DRQN, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, hidden_state=None):
        x = torch.relu(self.fc1(x))
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        x = torch.relu(self.fc2(lstm_out))
        q_values = self.fc_out(x)
        return q_values, hidden_state

class EpisodeReplayBuffer:
    def __init__(self, capacity=500):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, episode):
        """Stores an entire episode trajectory."""
        self.buffer.append(episode)
        
    def sample(self, batch_size):
        """Samples a batch of full episodes."""
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

def evaluate_deterministic_policy(env, policy_net, device):
    """Runs a 0-epsilon evaluation on a fixed seed to measure true convergence."""
    obs, _ = env.reset(seed=42)
    hidden = None
    det_reward = 0.0
    
    for _ in range(240):
        obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
        with torch.no_grad():
            q_vals, hidden = policy_net(obs_tensor, hidden)
            action = q_vals[0, 0, :].argmax().item()
            
        obs, reward, _, _, _ = env.step(action)
        det_reward += reward  
        
    return det_reward

def train_drqn_specialist(cancer_name, data_path, total_episodes=1500, trial=None, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- FIXED ENVIRONMENT INSTANTIATION ---
    env = PredictiveChemoEnv(
        schedule_parquet_path=data_path,
        drug_name=CANCER_PROFILES[cancer_name]['drug'],
        tumor_zone=CANCER_PROFILES[cancer_name]['zone']
    )

    policy_net = DRQN(input_dim=63, num_actions=10).to(device)
    target_net = DRQN(input_dim=63, num_actions=10).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # --- VALIDATED HYPERPARAMETERS ---
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4) 
    buffer = EpisodeReplayBuffer(capacity=500) 
    
    batch_size = 16
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995 # Slower decay for 1500 episodes
    
    best_eval_return = -float('inf')
    training_history = []  # <-- ADDED FOR PLOTTING
    os.makedirs("models", exist_ok=True)

    for episode_idx in range(1, total_episodes + 1):
        obs, _ = env.reset()
        hidden = None
        
        ep_history = []
        ep_actions = []
        episode_reward = 0.0

        for _ in range(240):
            obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                with torch.no_grad():
                    _, hidden = policy_net(obs_tensor, hidden)
            else:
                with torch.no_grad():
                    q_vals, hidden = policy_net(obs_tensor, hidden)
                    action = q_vals[0, 0, :].argmax().item()

            next_obs, reward, _, _, _ = env.step(action)
            ep_history.append((obs, action, reward, next_obs, 0.0))
            ep_actions.append(action)
            episode_reward += reward
            obs = next_obs

        buffer.push(ep_history)
        training_history.append(episode_reward)  # <-- RECORD REWARD FOR PLOT
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # --- VALIDATED OPTIMIZATION BLOCK (DDQN + HUBER + CLIP + FULL EP) ---
        if len(buffer) > batch_size:
            samples = buffer.sample(batch_size)
            optimizer.zero_grad()

            state_batch = torch.zeros(batch_size, 241, 63).to(device)
            action_batch = torch.zeros(batch_size, 240, 1, dtype=torch.long).to(device)
            reward_batch = torch.zeros(batch_size, 240).to(device)
            done_batch = torch.zeros(batch_size, 240).to(device)

            for b, seq in enumerate(samples):
                for t, transition in enumerate(seq):
                    state_batch[b, t] = torch.FloatTensor(transition[0])
                    action_batch[b, t, 0] = transition[1]
                    reward_batch[b, t] = float(transition[2])
                    done_batch[b, t] = float(transition[4])
                state_batch[b, 240] = torch.FloatTensor(seq[-1][3])

            all_q_values, _ = policy_net(state_batch)
            with torch.no_grad():
                all_target_q, _ = target_net(state_batch)

            q_values = all_q_values[:, :-1, :]
            
            # Double DQN Evaluation
            next_actions = all_q_values[:, 1:, :].argmax(dim=2, keepdim=True)
            next_q_values = all_target_q[:, 1:, :].gather(2, next_actions).squeeze(-1)

            state_action_values = q_values.gather(2, action_batch).squeeze(-1)
            expected_state_action_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

            # Huber Loss
            loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.detach())
            loss.backward()
            
            lstm_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in policy_net.lstm.parameters() if p.grad is not None) ** 0.5
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            avg_q = state_action_values.mean().item()

        # Target Update (Validated Interval)
        if episode_idx % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Live Logging
        if trial is None and episode_idx % 10 == 0 and len(buffer) > batch_size:
            uniq_a = len(np.unique(ep_actions))
            
            # --- FIXED SURVIVAL CHECK ---
            survival = "ALIVE" if env.total_toxicity < env.critical_toxicity_threshold else "DEAD "
            
            log_str = (
                f"Ep {episode_idx:4d} | Rew: {episode_reward:7.1f} | Surv: {survival} | "
                f"Tox: {env.total_toxicity:5.1f} | Mass: {env.engine.tumor_mass:5.4f} | "
                f"UniqA: {uniq_a:2d} | Loss: {loss.item():5.2f} | AvgQ: {avg_q:5.1f} | "
                f"∇LSTM: {lstm_grad_norm:4.2f} | Eps: {epsilon:.2f}"
            )
            print(log_str)
            
        # --- BEST MODEL CHECKPOINTING GATE ---
        if episode_idx % 50 == 0 and len(buffer) > batch_size:
            eval_return = evaluate_deterministic_policy(env, policy_net, device)
            is_best = ""
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                model_path = f"models/{cancer_name}_drqn_best.pt"
                torch.save(policy_net.state_dict(), model_path)
                is_best = "  <-- NEW BEST MODEL SAVED"
                
            print(f"{'-'*90}")
            print(f">>> EVALUATION EP {episode_idx:4d} | Det. Return: {eval_return:7.2f}{is_best}")
            print(f"{'-'*90}")

    return best_eval_return, training_history  # <-- NOW RETURNS HISTORY TOO


def optuna_objective(trial, cancer_name, data_path):
    # Optuna logic remains in the file for future use, but won't interfere 
    # with our standard hardcoded variables due to **kwargs absorption.
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.90, 0.999)
    
    score, _ = train_drqn_specialist(
        cancer_name, 
        data_path, 
        total_episodes=300, 
        trial=trial,
        lr=lr, 
        gamma=gamma, 
        batch_size=batch_size, 
        epsilon_decay=epsilon_decay
    )
    return score


def run_standard_training(cancer_name, data_path, total_episodes):
    print(f"Starting Standard Training for {cancer_name}...")
    
    # Catch both returned variables
    best_score, history = train_drqn_specialist(cancer_name, data_path, total_episodes=total_episodes)
    
    # --- PLOTTING THE TRAINING CURVE ---
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    
    # Plot raw episode rewards
    plt.plot(history, color='cyan', alpha=0.3, label='Episode Reward')
    
    # Plot a 50-episode moving average for readability
    if len(history) >= 50:
        moving_avg = np.convolve(history, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(history)), moving_avg, color='lime', linewidth=2, label='50-Ep Moving Avg')
        
    plt.title(f"DRQN Training Curve: {cancer_name}")
    plt.xlabel("Episode")
    plt.ylabel("Total Scaled Reward")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    os.makedirs("results", exist_ok=True)
    plot_path = f"results/{cancer_name}_training_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Training plot saved to {plot_path}")
    print(f"Training Complete. Best Deterministic Return: {best_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), required=True)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}. Please generate it first.")
        exit(1)

    if args.optimize:
        print(f"Running Optuna Hyperparameter Optimization for {args.cancer}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: optuna_objective(trial, args.cancer, dataset_path), n_trials=20)
        print("\nBest Trial:")
        print(study.best_trial.params)
    else:
        run_standard_training(args.cancer, dataset_path, args.episodes)