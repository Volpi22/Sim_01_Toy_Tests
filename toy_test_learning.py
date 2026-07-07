import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from train import DRQN, EpisodeReplayBuffer
from environment import PredictiveChemoEnv

class ToyChemoEnv(PredictiveChemoEnv):
    def __init__(self, schedule_parquet_path):
        super().__init__(schedule_parquet_path=schedule_parquet_path)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        toy_reward = info['efficacy_reward'] # Unscaled for mapping distribution
        return obs, toy_reward, False, truncated, info

def evaluate_deterministic_policy(env, policy_net, device, reward_scale):
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
        det_reward += (reward / reward_scale)
        
    return det_reward

def run_long_pilot_test():
    print(f"\n{'='*90}")
    print(f" TOY UNIT TEST: 1000-EPISODE CONVERGENCE RUN")
    print(f"{'='*90}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ToyChemoEnv(schedule_parquet_path="datasets/1_week_sampling_schedule.parquet")
    
    policy_net = DRQN(input_dim=63, num_actions=10).to(device)
    target_net = DRQN(input_dim=63, num_actions=10).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4) 
    buffer = EpisodeReplayBuffer(capacity=500) # Increased buffer size for longer run
    
    batch_size, gamma = 16, 0.99
    epsilon, epsilon_min = 1.0, 0.05
    epsilon_decay = 0.995  # SLOWED DOWN: Will reach 0.05 around Episode 600
    
    REWARD_SCALE = 100.0 
    best_eval_return = -float('inf')
    learning_curve = []

    os.makedirs("models", exist_ok=True)

    for episode_idx in range(1, 1001):
        obs, _ = env.reset()
        hidden = None
        ep_history, ep_actions = [], []
        ep_reward = 0.0
        
        # --- 1. LIVE ROLLOUT ---
        for _ in range(240):
            obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                with torch.no_grad(): _, hidden = policy_net(obs_tensor, hidden)
            else:
                with torch.no_grad():
                    q_vals, hidden = policy_net(obs_tensor, hidden)
                    action = q_vals[0, 0, :].argmax().item()
            
            next_obs, raw_reward, _, _, _ = env.step(action)
            scaled_reward = raw_reward / REWARD_SCALE
            
            ep_history.append((obs, action, scaled_reward, next_obs, 0.0))
            ep_actions.append(action)
            ep_reward += scaled_reward
            obs = next_obs
            
        buffer.push(ep_history)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # --- 2. FULL-EPISODE DDQN OPTIMIZATION ---
        if len(buffer) > batch_size:
            samples = buffer.sample(batch_size)
            optimizer.zero_grad()
            
            state_batch = torch.zeros(batch_size, 241, 63).to(device)
            action_batch = torch.zeros(batch_size, 240, 1, dtype=torch.long).to(device)
            reward_batch = torch.zeros(batch_size, 240).to(device)
            done_batch = torch.zeros(batch_size, 240).to(device)
            
            for b, seq in enumerate(samples):
                for t, trans in enumerate(seq):
                    state_batch[b, t] = torch.FloatTensor(trans[0])
                    action_batch[b, t, 0] = trans[1]
                    reward_batch[b, t] = float(trans[2])
                    done_batch[b, t] = float(trans[4])
                state_batch[b, 240] = torch.FloatTensor(seq[-1][3])
                
            all_q, _ = policy_net(state_batch)
            with torch.no_grad(): all_target_q, _ = target_net(state_batch)
                
            q_values = all_q[:, :-1, :]       
            next_actions = all_q[:, 1:, :].argmax(dim=2, keepdim=True)
            next_q_values = all_target_q[:, 1:, :].gather(2, next_actions).squeeze(-1)
            
            state_action_values = q_values.gather(2, action_batch).squeeze(-1)
            expected_state_action_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
            
            loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.detach())
            loss.backward()
            
            lstm_g = sum(p.grad.data.norm(2).item() ** 2 for p in policy_net.lstm.parameters() if p.grad is not None) ** 0.5
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            avg_q = state_action_values.mean().item()

        # Target Update
        if episode_idx % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Logging Live Training Metrics
        if episode_idx % 10 == 0 and len(buffer) > batch_size:
            uniq_a = len(np.unique(ep_actions))
            print(f"Train Ep {episode_idx:4d} | Live Rew: {ep_reward:6.1f} | UniqA: {uniq_a:2d} | Loss: {loss.item():5.2f} | AvgQ: {avg_q:5.1f} | ∇LSTM: {lstm_g:4.2f} | Eps: {epsilon:.2f}")

        # --- 3. DETERMINISTIC EVALUATION GATE ---
        if episode_idx % 50 == 0 and len(buffer) > batch_size:
            eval_return = evaluate_deterministic_policy(env, policy_net, device, REWARD_SCALE)
            learning_curve.append((episode_idx, eval_return))
            
            # Checkpoint Saver
            is_best = ""
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(policy_net.state_dict(), 'models/toy_drqn_best.pt')
                is_best = "  <-- NEW BEST MODEL SAVED"
                
            print(f"{'-'*90}")
            print(f">>> EVALUATION EP {episode_idx:4d} | Deterministic Return: {eval_return:7.2f}{is_best}")
            print(f"{'-'*90}")

    # Summary of the Learning Curve
    print(f"\n{'='*90}")
    print(" FINAL LEARNING CURVE (EVALUATION RETURNS)")
    print(f"{'='*90}")
    for ep, ret in learning_curve:
        print(f" Episode {ep:<4d} -> {ret:.2f}")
    print(f"{'='*90}\n")

if __name__ == "__main__":
    run_long_pilot_test()