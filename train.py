"""
=============================================================================
Therapeutic Thermostat: Native PyTorch DRQN Training Hub
=============================================================================
"""

import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
    def __init__(self, input_dim=43, num_actions=5, hidden_dim=256):
        super(DRQN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Dense feature encoder network
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Recurrent Core Core (LSTM)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Q-value output layer
        self.q_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, hidden_state=None):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.size()
        
        flat_x = x.contiguous().view(batch_size * seq_len, -1)
        features = self.feature_network(flat_x)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, hidden_state = self.lstm(features, hidden_state)
        q_values = self.q_head(lstm_out)
        return q_values, hidden_state

class EpisodeReplayBuffer:
    """Sequential buffer to preserve sequential chronological tracking data."""
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

def train_drqn_specialist(cancer_name, data_path, total_episodes=400, batch_size=16, gamma=0.99):
    print(f"\n{'='*60}")
    print(f" INITIALIZING RECURRENT TRAINING (DRQN): {cancer_name}")
    print(f" Drug: {CANCER_PROFILES[cancer_name]['drug']} | Target Zone: {CANCER_PROFILES[cancer_name]['zone']}")
    print(f"{'='*60}\n")

    env = PredictiveChemoEnv(schedule_parquet_path=data_path, drug_name=CANCER_PROFILES[cancer_name]['drug'], tumor_zone=CANCER_PROFILES[cancer_name]['zone'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DRQN(input_dim=43, num_actions=5).to(device)
    target_net = DRQN(input_dim=43, num_actions=5).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = EpisodeReplayBuffer(capacity=500)
    
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.95
    
    for episode_idx in range(total_episodes):
        obs, _ = env.reset()
        episode_history = []
        hidden = None
        
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
            
            # Action Selection via Epsilon-Greedy
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
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # DRQN Sequence Optimization Step
        if len(buffer) > batch_size:
            samples = buffer.sample(batch_size)
            optimizer.zero_grad()
            
            loss = 0
            for seq in samples:
                # Construct sequential batches chronologically
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
            
        if episode_idx % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode_idx} | Return Score: {episode_reward:.2f} | Exploration Epsilon: {epsilon:.2f}")

    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), f"models/{cancer_name}_recurrent_drqn.pt")
    print(f"\n[SUCCESS] Recurrent Specialist saved to models/{cancer_name}_recurrent_drqn.pt\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Recurrent Specialists for Chemotherapy Control")
    parser.add_argument("--cancer", type=str, choices=list(CANCER_PROFILES.keys()), help="Train a single specific recurrent profile.")
    parser.add_argument("--all", action="store_true", help="Sequential batch training flag.")
    args = parser.parse_args()

    dataset_path = "datasets/1_week_sampling_schedule.parquet"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset at {dataset_path}. Run dataset_generation.py first.")

    if args.all:
        for name in CANCER_PROFILES.keys():
            train_drqn_specialist(name, dataset_path)
    elif args.cancer:
        train_drqn_specialist(args.cancer, dataset_path)
    else:
        print("Please use --cancer [name] or --all to launch your recurrent training sequence.")