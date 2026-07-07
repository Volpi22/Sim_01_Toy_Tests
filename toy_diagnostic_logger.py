import torch
import pandas as pd
from train import DRQN
from environment import PredictiveChemoEnv

class ToyChemoEnv(PredictiveChemoEnv):
    def __init__(self, schedule_parquet_path):
        super().__init__(schedule_parquet_path=schedule_parquet_path)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        toy_reward = info['efficacy_reward'] # Unscaled for mapping distribution
        return obs, toy_reward, False, truncated, info

def run_diagnostic_logger():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ToyChemoEnv(schedule_parquet_path="datasets/1_week_sampling_schedule.parquet")
    
    policy_net = DRQN(input_dim=63, num_actions=10).to(device)
    policy_net.load_state_dict(torch.load('models/toy_drqn_best.pt', map_location=device, weights_only=True))
    policy_net.eval()

    obs, _ = env.reset(seed=42)
    env.engine.gamma = 4.0
    env.engine.tumor_growth_rate = 0.006
    env.engine.drug['k_elim'] /= env.patient_clearance_factor
    
    hidden = None
    log_data = []

    for step in range(240):
        obs_tensor = torch.FloatTensor(obs).view(1, 1, -1).to(device)
        with torch.no_grad():
            q_vals, hidden = policy_net(obs_tensor, hidden)
            q_array = q_vals[0, 0, :].cpu().numpy()
            act = q_array.argmax()

        obs, reward, _, _, info = env.step(act)
        
        # Build the row
        row = {
            "Minute": step,
            "Chosen_Action": act,
            "Tumor_Mass": env.engine.tumor_mass,
            "Drug_Conc": info.get('drug_concentration', info.get('Ce', 0.0)), # Safe fallback
            "Temperature": info.get('real_temp', 37.0),
            "pH": info.get('real_ph', 7.4)
        }
        
        # Add Q-values dynamically
        for i in range(10):
            row[f"Q{i}"] = q_array[i]
            
        log_data.append(row)

    df = pd.DataFrame(log_data)
    df.to_csv("results/toy_drqn_step_diagnostic.csv", index=False)
    print("\n[INFO] Diagnostic saved to results/toy_drqn_step_diagnostic.csv")
    print("Inspect this file to correlate action switches with physiological thresholds.\n")

if __name__ == "__main__":
    run_diagnostic_logger()