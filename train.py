import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from collections import deque
from environment import PredictiveChemoEnv

torch.backends.mkldnn.enabled = False            

CANCER_PROFILES = {
    "Hepatocellular_Carcinoma": {"zone": 14, "drug": "Doxorubicin"},
    "Colorectal_Cancer": {"zone": 12, "drug": "5-Fluorouracil"},
    "Lung_Cancer": {"zone": 2, "drug": "Paclitaxel"},
    "Renal_Cell_Carcinoma": {"zone": 16, "drug": "Cisplatin"},
    "Glioblastoma": {"zone": 7, "drug": "Carmustine"}
}

class DRQN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_actions=10):
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
        return self.fc_out(x), hidden_state

class EpisodeReplayBuffer:
    def __init__(self, capacity=500):
        self.buffer = deque(maxlen=capacity)
    def push(self, episode):
        self.buffer.append(episode)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

def evaluate_deterministic_policy(env, policy_net, device, num_seeds=5):
    total_det_reward = 0.0
    for seed in range(42, 42 + num_seeds):
        obs, _ = env.reset(seed=seed)
        hidden = None
        for _ in range(240):
            obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
            with torch.no_grad():
                q_vals, hidden = policy_net(obs_tensor, hidden)
                action = q_vals[0, 0, :].argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_det_reward += reward  
            if terminated or truncated:
                break
    return total_det_reward / num_seeds

def train_drqn_specialist(cancer_name, data_path, total_episodes=1000, trial=None, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PredictiveChemoEnv(schedule_parquet_path=data_path, drug_name=CANCER_PROFILES[cancer_name]['drug'], tumor_zone=CANCER_PROFILES[cancer_name]['zone'])

    policy_net = DRQN(input_dim=64, num_actions=10).to(device)
    target_net = DRQN(input_dim=64, num_actions=10).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    lr = kwargs.get('lr', 1e-4)
    gamma = kwargs.get('gamma', 0.99)
    batch_size = kwargs.get('batch_size', 16)
    epsilon_decay = kwargs.get('epsilon_decay', 0.995)

    optimizer = optim.Adam(policy_net.parameters(), lr=lr) 
    buffer = EpisodeReplayBuffer(capacity=500) 
    epsilon, epsilon_min = 1.0, 0.05
    best_eval_return = -float('inf')
    training_history = [] 
    os.makedirs("models", exist_ok=True)

    # [LOCKED] 60-Minute Burn-in + 120-Minute Learning Window
    burn_in = 60
    seq_len = 120
    total_sample_len = burn_in + seq_len

    for episode_idx in range(1, total_episodes + 1):
        obs, _ = env.reset()
        hidden = None
        ep_history, ep_actions, episode_reward = [], [], 0.0

        for _ in range(240):
            obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                with torch.no_grad(): _, hidden = policy_net(obs_tensor, hidden)
            else:
                with torch.no_grad():
                    q_vals, hidden = policy_net(obs_tensor, hidden)
                    action = q_vals[0, 0, :].argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_history.append((obs, action, reward, next_obs, float(terminated)))
            ep_actions.append(action)
            episode_reward += reward
            obs = next_obs
            if terminated or truncated: break

        buffer.push(ep_history)
        training_history.append(episode_reward)  
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if len(buffer) > batch_size:
            samples = buffer.sample(batch_size)
            optimizer.zero_grad()
            
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
                
                rand_val = random.random()
                if rand_val < 0.30:
                    start = random.randint(0, max_start // 3)
                elif rand_val < 0.70:
                    start = random.randint(max_start // 3, 2 * (max_start // 3))
                else:
                    start = random.randint(2 * (max_start // 3), max_start)
                
                actual_burn_in = min(burn_in, actual_len - start)
                if actual_burn_in > 0:
                    burn_in_states = (
                        torch.as_tensor(
                            np.stack([tr[0] for tr in seq[start:start + actual_burn_in]]),
                            dtype=torch.float32,
                            device=device
                        )
                        .unsqueeze(0)
                    )
                        
                    with torch.no_grad():
                        _, (h, c) = policy_net(burn_in_states)
                        h_0[:, b, :] = h[0]
                        c_0[:, b, :] = c[0]
                
                learn_start = start + burn_in
                valid_len = min(seq_len, max(0, actual_len - learn_start))
                
                if valid_len > 0:
                    states = np.stack([tr[0] for tr in seq[learn_start:learn_start + valid_len]])
                    actions = np.array([tr[1] for tr in seq[learn_start:learn_start + valid_len]], dtype=np.int64)
                    rewards = np.array([tr[2] for tr in seq[learn_start:learn_start + valid_len]], dtype=np.float32)
                    dones = np.array([tr[4] for tr in seq[learn_start:learn_start + valid_len]], dtype=np.float32)

                    state_batch[b, :valid_len] = torch.as_tensor(states, dtype=torch.float32, device=device)
                    action_batch[b, :valid_len, 0] = torch.as_tensor(actions, device=device)
                    reward_batch[b, :valid_len] = torch.as_tensor(rewards, device=device)
                    done_batch[b, :valid_len] = torch.as_tensor(dones, device=device)
                    mask[b, :valid_len] = 1.0  

                if valid_len > 0:
                    last_transition = seq[learn_start + valid_len - 1]
                    state_batch[b, valid_len] = torch.as_tensor(
                        last_transition[3],
                        dtype=torch.float32,
                        device=device
                    )

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
            
            lstm_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in policy_net.lstm.parameters() if p.grad is not None) ** 0.5
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            avg_q = (state_action_values * mask).sum().item() / max(mask.sum().item(), 1.0)

        if episode_idx % 10 == 0: target_net.load_state_dict(policy_net.state_dict())

        if trial is None and episode_idx % 10 == 0 and len(buffer) > batch_size:
            uniq_a = len(np.unique(ep_actions))
            log_str = (
                f"Ep {episode_idx:4d} | Rew: {episode_reward:7.1f} | Surv: {'ALIVE' if not terminated else 'DEAD '} | "
                f"Tox: {env.total_toxicity:5.1f} | Mass: {env.engine.tumor_mass:5.4f} | "
                f"UniqA: {uniq_a:2d} | Loss: {loss.item():5.2f} | AvgQ: {avg_q:5.1f} | "
                f"∇LSTM: {lstm_grad_norm:4.2f} | Eps: {epsilon:.2f}"
            )
            print(log_str)
            
        if episode_idx % 50 == 0 and len(buffer) > batch_size:
            eval_return = evaluate_deterministic_policy(env, policy_net, device, num_seeds=5)
            is_best = ""
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(policy_net.state_dict(), f"models/{cancer_name}_drqn_best.pt")
                is_best = "  <-- NEW BEST MODEL SAVED"
            print(f">>> EVALUATION EP {episode_idx:4d} | Det. Return: {eval_return:7.2f}{is_best}")

    return best_eval_return, training_history 

def optuna_objective(trial, cancer_name, data_path):
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

def run_standard_training(cancer_name, data_path, total_episodes, **kwargs):
    print(f"Starting Standard Training for {cancer_name}...")
    if kwargs:
        print(f"Using Parameters: {kwargs}")
    else:
        print("Using Default Parameters.")
        
    best_score, history = train_drqn_specialist(
        cancer_name, 
        data_path, 
        total_episodes=total_episodes, 
        **kwargs
    )
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='cyan', alpha=0.3, label='Episode Reward')
    
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
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    os.makedirs("config", exist_ok=True)
    param_file = f"config/{args.cancer}_best_params.json"
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}. Please generate it first.")
        exit(1)

    if args.optimize:
        print(f"Running Optuna Hyperparameter Optimization for {args.cancer}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: optuna_objective(trial, args.cancer, dataset_path), n_trials=30)
        
        best_params = study.best_trial.params
        print(f"\nOptimization Complete! Best Parameters: {best_params}")
        
        with open(param_file, 'w') as f:
            json.dump(best_params, f)
            
        print(f"Saved optimal parameters to {param_file}. Launching full training...")
        run_standard_training(args.cancer, dataset_path, args.episodes, **best_params)

    else:
        if os.path.exists(param_file):
            print(f"[INFO] Loading optimized parameters from {param_file}...")
            with open(param_file, 'r') as f:
                loaded_params = json.load(f)
            run_standard_training(args.cancer, dataset_path, args.episodes, **loaded_params)
        else:
            print(f"[WARNING] Parameter file {param_file} not found. Running with hardcoded defaults.")
            run_standard_training(args.cancer, dataset_path, args.episodes)