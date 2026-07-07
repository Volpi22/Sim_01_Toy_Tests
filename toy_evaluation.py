import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from train import DRQN
from environment import PredictiveChemoEnv

class ToyChemoEnv(PredictiveChemoEnv):
    def __init__(self, schedule_parquet_path):
        super().__init__(schedule_parquet_path=schedule_parquet_path)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        toy_reward = info['efficacy_reward'] # Unscaled for mapping distribution
        return obs, toy_reward, False, truncated, info

def run_robust_evaluation(num_seeds=20):
    print(f"\n{'='*90}")
    print(f" ROBUST TOY EVALUATION: {num_seeds}-SEED BENCHMARK & Q-VALUE ANALYSIS")
    print(f"{'='*90}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ToyChemoEnv(schedule_parquet_path="datasets/1_week_sampling_schedule.parquet")
    
    policy_net = DRQN(input_dim=63, num_actions=10).to(device)
    try:
        policy_net.load_state_dict(torch.load('models/toy_drqn_best.pt', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("[WARNING] 'models/toy_drqn_best.pt' not found! Run the training pilot first.")
        return
    policy_net.eval()

    # The top baselines from our previous test
    policies = {
        "Constant 1 (25%)": lambda t: 1,
        "Alt 0 / 1": lambda t: 1 if (t // 10) % 2 == 0 else 0,
        "Trained DRQN": "DRQN_FLAG"
    }

    print(f"Evaluating policies across {num_seeds} distinct patient seeds...")
    results = []
    
    # Track data for plotting from the very first seed
    plot_actions = []
    plot_q_vals = []

    for name, policy_func in policies.items():
        seed_returns = []
        
        for seed_idx in range(num_seeds):
            obs, _ = env.reset(seed=100 + seed_idx) # Sweeping fresh seeds
            env.engine.gamma = 4.0
            env.engine.tumor_growth_rate = 0.006
            env.engine.drug['k_elim'] /= env.patient_clearance_factor
            
            hidden = None
            ep_return = 0.0
            
            for step in range(240):
                if policy_func == "DRQN_FLAG":
                    obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
                    with torch.no_grad():
                        q_vals, hidden = policy_net(obs_tensor, hidden)
                        act = q_vals[0, 0, :].argmax().item()
                        
                    # Save plotting data for the first seed of DRQN
                    if seed_idx == 0:
                        plot_actions.append(act)
                        plot_q_vals.append(q_vals[0, 0, :].cpu().numpy())
                else:
                    act = policy_func(step)
                    
                obs, raw_reward, _, _, _ = env.step(act)
                ep_return += raw_reward
                
            seed_returns.append(ep_return)
            
        mean_ret = np.mean(seed_returns)
        std_ret = np.std(seed_returns)
        
        results.append({
            "Policy": name,
            "Mean_Return": mean_ret,
            "Std_Return": std_ret,
            "Min_Return": np.min(seed_returns),
            "Max_Return": np.max(seed_returns)
        })

    # 1. Print the 20-Seed Benchmark Table
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False, float_format="%9.2f"))
    print(f"{'='*90}\n")

    # 2. Generate the Deep Diagnostic Plot (Action Timeline + Q-Value Trajectories)
    plot_q_vals = np.array(plot_q_vals) # Shape: (240, 10)
    
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]}, sharex=True)
    fig.suptitle(f'DRQN Deep Diagnostics (Representative Seed 100)', fontsize=16, fontweight='bold')

    # Top Plot: Action Sequence
    axs[0].step(range(240), plot_actions, where='post', color='cyan', linewidth=2)
    axs[0].set_ylabel('Action Selected', fontsize=12)
    axs[0].set_yticks(range(10))
    axs[0].grid(True, alpha=0.2)
    axs[0].set_title("Temporal Action Sequence", fontsize=12)

    # Bottom Plot: Q-Value Evolution (10 distinct lines)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for a in range(10):
        axs[1].plot(range(240), plot_q_vals[:, a], color=colors[a], alpha=0.8, linewidth=1.5, label=f'Action {a}')
        
    axs[1].set_xlabel('Simulation Minute', fontsize=12)
    axs[1].set_ylabel('Q-Value Magnitude', fontsize=12)
    axs[1].grid(True, alpha=0.2)
    axs[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Actions")
    axs[1].set_title("Q-Value Landscape Evolution over Time", fontsize=12)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = "results/robust_drqn_diagnostics.png"
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Diagnostics plot saved to {save_path}")

if __name__ == "__main__":
    run_robust_evaluation(num_seeds=20)