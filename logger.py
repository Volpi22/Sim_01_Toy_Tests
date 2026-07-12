import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

class EpisodeLogger:
    def __init__(self):
        self.transitions = []

    def record(self, step, action, prev_action, reward_scaled, info, terminated, truncated):
        # Fail loudly if expected keys are missing
        done_reason = "Toxicity Limit" if terminated else "Horizon Reached" if truncated else "Ongoing"

        self.transitions.append({
            "Step": step,
            "Action": action,
            "Prev_Action": prev_action,
            "Dose_Pct": info['current_dose'] * 100.0, 
            "Mass": info['mass'],
            "Distance": info['curr_dist'],
            "Distance_Reduction": info['distance_reduction'],
            "Toxicity": info['total_toxicity'],
            "Ce": info['Ce'],
            "Hill": info['hill'],
            "Reward_Raw": info['reward_raw'],
            "Reward_Scaled": reward_scaled,
            "Efficacy_Reward": info['efficacy_reward'],
            "Distance_Penalty": -info['distance_penalty'],
            "Toxicity_Penalty": -info['toxicity_penalty'],
            "Smoothing_Penalty": -info['smoothing_penalty'],
            "Terminal_Penalty": -info['terminal_penalty'],
            "Terminated": terminated,
            "Truncated": truncated,
            "Done_Reason": done_reason
        })

    def to_dataframe(self):
        return pd.DataFrame(self.transitions)

    def save_csv(self, df, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    def print_correlations(self, df):
        print("\n [CORRELATION MATRIX: NET REWARD VS STATES]")
        cols = ['Reward_Raw', 'Distance_Reduction', 'Toxicity', 'Distance', 'Hill', 'Mass']
        
        print(f" {'Feature':<20} | {'Pearson':<10} | {'Spearman':<10}")
        print("-" * 46)
        for col in cols:
            if col == 'Reward_Raw': continue
            pearson = df['Reward_Raw'].corr(df[col], method='pearson')
            spearman, _ = spearmanr(df['Reward_Raw'], df[col])
            print(f" {col:<20} | {pearson:>+10.3f} | {spearman:>+10.3f}")

    def plot_summary(self, df, save_path):
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("Episode Dynamics", fontsize=16)

        # 1. State Dynamics over Time
        axes[0, 0].plot(df['Step'], df['Mass'], label='Mass', color='blue')
        axes[0, 0].set_ylabel('Tumor Mass')
        ax2 = axes[0, 0].twinx()
        ax2.plot(df['Step'], df['Toxicity'], label='Toxicity', color='red', alpha=0.6)
        ax2.set_ylabel('Toxicity')
        axes[0, 0].set_title("Clinical State vs Time")
        
        # 2. Dose Schedule & Ce
        axes[0, 1].plot(df['Step'], df['Dose_Pct'], label='Dose %', drawstyle='steps-post', color='purple')
        axes[0, 1].set_ylabel('Dose %')
        ax3 = axes[0, 1].twinx()
        ax3.plot(df['Step'], df['Ce'], label='Ce (Compartment)', color='orange', alpha=0.7)
        ax3.set_ylabel('Ce')
        axes[0, 1].set_title("Dose Schedule & Exposure")

        # 3. Distance Reduction 
        axes[1, 0].plot(df['Step'], df['Distance_Reduction'], color='green')
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_title("Distance Reduction (Primary Efficacy Signal)")

        # 4. Raw Reward 
        axes[1, 1].plot(df['Step'], df['Reward_Raw'], color='black')
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1, 1].set_title("Net Reward per Step")

        # 5. Histograms: Action & Toxicity
        axes[2, 0].hist(df['Action'], bins=10, color='purple', alpha=0.7)
        axes[2, 0].set_title("Action Distribution")

        axes[2, 1].hist(df['Toxicity'], bins=20, color='red', alpha=0.7)
        axes[2, 1].set_title("Toxicity Distribution")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()