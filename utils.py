"""
Core Utilities Module

This module provides foundational support classes for the nanonetwork simulation:
1. DataManager: A repository for anatomical mapping, physiological ground-truth data,
   and PK/PD physics constants for the Therapeutic Thermostat.
2. ProcessorUtils: Helper functions for dataset manipulation and stochastic sampling.
3. StatsUtils: Statistical engines for calculating Binomial and Hoeffding confidence bounds.
4. EnergyModel: A physics-based model of a piezoelectric energy harvesting nanobot, 
   tracking capacitor charging cycles and hardware power expenditures.
"""

import pandas as pd
import numpy as np
import random
import math
from scipy.stats import binom

# ==========================================
# DATA MANAGEMENT & PHYSICS CONSTANTS
# ==========================================

class DataManager:
    """
    Centralized storage for anatomical mappings, physiological ground-truth ranges,
    compartmental volumes, advection topologies, and PK/PD constants.
    """
    
    # --- 1. ANATOMICAL MAPPING ---
    ZONE_TO_VESSELS = {
        1: [2], 2: [61, 94], 3: [57, 58], 4: [1, 3, 4, 8, 13, 18, 24],
        5: [19, 26, 31, 34], 6: [6, 7], 7: [9], 8: [81, 83],
        9: [5, 10, 11, 12, 14, 15, 16, 17, 20, 21, 22, 23, 27, 28, 29, 33], 
        10: [64, 65, 68, 69, 74, 75, 80, 82], 11: [25, 32, 35, 37], 12: [30, 39], 
        13: [76, 77, 84], 14: [36], 15: [90], 16: [40, 86], 
        17: [38, 41, 42, 43, 44, 45, 46], 18: [48, 50, 53, 55], 
        19: [47, 49, 51, 52, 54, 56, 59, 60],
        20: [62, 63, 66, 67, 70, 71, 72, 73, 78, 79, 85, 87, 88, 89, 91, 92, 93]
    }

    # --- 2. COMPARTMENTAL VOLUMES (Vi) in Liters ---
    VOLUMES = {
        **{i: 0.80 for i in range(1, 6)},    # Central
        **{i: 2.10 for i in range(6, 14)},   # Peripheral
        14: 0.60, 15: 0.60,                  # Hepatic
        16: 0.30,                            # Renal
        **{i: 2.10 for i in range(17, 21)}   # Peripheral
    }

    # --- 3. PHARMACOKINETICS (Relative Scale) ---
    DRUGS = {
        # C_safe = 1.20 means toxicity starts accumulating if organ hits 120% of nominal dose
        'Doxorubicin':   {'k_elim': 0.173 / 60, 'clearance_zone': 14, 'C_safe': 1.20},
        'Cisplatin':     {'k_elim': 0.029 / 60, 'clearance_zone': 16, 'C_safe': 1.15},
        'Paclitaxel':    {'k_elim': 0.046 / 60, 'clearance_zone': 14, 'C_safe': 1.25},
        '5-Fluorouracil':{'k_elim': 2.772 / 60, 'clearance_zone': 14, 'C_safe': 1.10},
        'Carmustine':    {'k_elim': 2.772 / 60, 'clearance_zone': 14, 'C_safe': 1.15}
    }

    # --- 4. PHARMACODYNAMICS (eta) ---
    # Shift at 1.0 (100%) concentration (e.g., Doxo drops pH by 0.15 at full dose)
    PD_SHIFTS = {
        'Doxorubicin':   {'eta_ph': 0.15, 'eta_temp': -0.40},
        'Cisplatin':     {'eta_ph': 0.20, 'eta_temp': -0.30},
        'Paclitaxel':    {'eta_ph': 0.10, 'eta_temp': -0.50},
        '5-Fluorouracil':{'eta_ph': 0.25, 'eta_temp': -0.20},
        'Carmustine':    {'eta_ph': 0.15, 'eta_temp': -0.35}
    }

    def __init__(self):
        """Initializes the baseline physiological dictionaries."""
        self.datasets = {
            'ph': {
                1: [7.35, 7.45], 2: [7.35, 7.45], 3: [7.35, 7.45], 4: [7.35, 7.45], 5: [7.35, 7.45], 
                6: [7.35, 7.45], 7: [7.35, 7.45], 8: [7.35, 7.45], 9: [7.35, 7.4], 10: [7.35, 7.45], 
                11: [7.35, 7.45], 12: [7.35, 7.45], 13: [7.35, 7.45], 14: [7.35, 7.45], 15: [7.35, 7.45], 
                16: [7.35, 7.45], 17: [7.35, 7.45], 18: [7.35, 7.45], 19: [7.35, 7.45], 20: [7.35, 7.45], 
                21: [7.35, 7.45], 22: [7.35, 7.45], 23: [7.35, 7.45], 24: [7.35, 7.45], 25: [7.35, 7.45], 
                26: [7.35, 7.45], 27: [7.35, 7.45], 28: [7.35, 7.45], 29: [7.35, 7.45], 30: [7.35, 7.4], 
                31: [7.35, 7.45], 32: [7.35, 7.45], 33: [7.35, 7.45], 34: [7.35, 7.45], 35: [7.35, 7.45], 
                36: [7.35, 7.4], 37: [7.35, 7.45], 38: [7.35, 7.45], 39: [7.35, 7.4], 40: [7.35, 7.4], 
                41: [7.35, 7.45], 42: [7.35, 7.45], 43: [7.35, 7.45], 44: [7.35, 7.45], 45: [7.35, 7.45], 
                46: [7.35, 7.45], 47: [7.35, 7.45], 48: [7.35, 7.45], 49: [7.35, 7.45], 50: [7.35, 7.45], 
                51: [7.35, 7.45], 52: [7.35, 7.45], 53: [7.35, 7.45], 54: [7.35, 7.45], 55: [7.35, 7.45], 
                56: [7.35, 7.45], 57: [7.35, 7.38], 58: [7.35, 7.38], 59: [7.35, 7.45], 60: [7.35, 7.45], 
                61: [7.35, 7.45], 62: [7.35, 7.38], 63: [7.35, 7.38], 64: [7.35, 7.38], 65: [7.35, 7.38], 
                66: [7.35, 7.38], 67: [7.35, 7.38], 68: [7.35, 7.38], 69: [7.35, 7.38], 70: [7.35, 7.38], 
                71: [7.35, 7.38], 72: [7.35, 7.38], 73: [7.35, 7.38], 74: [7.35, 7.38], 75: [7.35, 7.38], 
                76: [7.35, 7.38], 77: [7.35, 7.38], 78: [7.35, 7.38], 79: [7.35, 7.38], 80: [7.35, 7.38], 
                81: [7.35, 7.38], 82: [7.35, 7.38], 83: [7.35, 7.38], 84: [7.35, 7.38], 85: [7.35, 7.38], 
                86: [7.35, 7.38], 87: [7.35, 7.38], 88: [7.35, 7.38], 89: [7.35, 7.38], 90: [7.35, 7.38], 
                91: [7.35, 7.38], 92: [7.35, 7.38], 93: [7.35, 7.38], 94: [7.35, 7.45]
            },
            'temp': {
                1: [36.8, 37.2], 2: [36.8, 37.2], 3: [36.8, 37.2], 4: [36.8, 37.2], 5: [36.8, 37.2], 
                6: [36.8, 37.2], 7: [36.8, 37.2], 8: [36.8, 37.2], 9: [36.8, 37.2], 10: [36.8, 37.2], 
                11: [36.8, 37.2], 12: [36.8, 37.2], 13: [36.8, 37.2], 14: [36.8, 37.2], 15: [36.8, 37.2], 
                16: [36.8, 37.2], 17: [36.8, 37.2], 18: [36.8, 37.2], 19: [36.8, 37.2], 20: [36.8, 37.2], 
                21: [36.8, 37.2], 22: [36.8, 37.2], 23: [36.8, 37.2], 24: [36.8, 37.2], 25: [36.8, 37.2], 
                26: [36.8, 37.2], 27: [36.8, 37.2], 28: [36.8, 37.2], 29: [36.8, 37.2], 30: [36.8, 37.2], 
                31: [36.8, 37.2], 32: [36.8, 37.2], 33: [36.8, 37.2], 34: [36.8, 37.2], 35: [36.8, 37.2], 
                36: [37.2, 37.8], 37: [36.8, 37.2], 38: [36.8, 37.2], 39: [36.8, 37.2], 40: [36.8, 37.0], 
                41: [36.8, 37.2], 42: [36.8, 37.2], 43: [36.8, 37.2], 44: [36.8, 37.2], 45: [36.8, 37.2], 
                46: [36.8, 37.2], 47: [36.0, 36.8], 48: [36.8, 37.2], 49: [36.0, 36.8], 50: [36.8, 37.2], 
                51: [36.0, 36.8], 52: [36.0, 36.8], 53: [36.5, 37.0], 54: [36.0, 36.8], 55: [36.5, 37.0], 
                56: [36.0, 36.8], 57: [36.8, 37.0], 58: [36.8, 37.0], 59: [36.0, 36.8], 60: [36.0, 36.8], 
                61: [36.8, 37.2], 62: [35.0, 36.5], 63: [35.0, 36.5], 64: [36.8, 37.0], 65: [36.8, 37.0], 
                66: [35.0, 36.5], 67: [35.0, 36.5], 68: [36.8, 37.0], 69: [36.8, 37.0], 70: [36.8, 37.0], 
                71: [35.0, 36.5], 72: [36.8, 37.0], 73: [35.0, 36.5], 74: [36.8, 37.0], 75: [36.8, 37.0], 
                76: [37.0, 37.5], 77: [37.0, 37.5], 78: [36.8, 37.0], 79: [36.8, 37.0], 80: [36.8, 37.0], 
                81: [36.5, 37.0], 82: [36.8, 37.0], 83: [36.5, 37.0], 84: [37.0, 37.5], 85: [36.8, 37.0], 
                86: [36.8, 37.0], 87: [36.8, 37.0], 88: [36.8, 37.0], 89: [36.8, 37.0], 90: [37.5, 38.0], 
                91: [36.8, 37.0], 92: [36.8, 37.0], 93: [36.8, 37.0], 94: [36.8, 37.2]
            }
        }

    def get_data_dict(self, param_key: str) -> dict:
        return self.datasets.get(param_key.lower())

    def get_decimal_precision(self, param_type: str) -> int:
        precisions = {'ph': 2, 'temp': 3}
        return precisions.get(param_type, 3)

    @staticmethod
    def assign_zone(vessel_id: int):
        for zone, vessels in DataManager.ZONE_TO_VESSELS.items():
            if vessel_id in vessels:
                return zone
        return None

    @staticmethod
    def get_advection_matrix():
        """
        Builds the 20x20 sparse flow matrix (Q_ij) in L/min based on Cardiac Output.
        Uses a 21x21 array to allow 1-based indexing matching the anatomical zones.
        """
        Q = np.zeros((21, 21)) 
        
        Q[2, 1] = 5.0; Q[3, 2] = 5.0; Q[1, 4] = 5.0 
        Q[4, 6] = 0.70; Q[4, 9] = 0.50; Q[4, 5] = 3.80
        Q[5, 11] = 1.25; Q[5, 16] = 1.10; Q[5, 17] = 0.10; Q[5, 18] = 1.35
        Q[6, 7] = 0.70; Q[7, 8] = 0.70; Q[8, 20] = 0.70
        Q[9, 10] = 0.50; Q[10, 20] = 0.50
        Q[11, 12] = 1.25; Q[12, 13] = 1.25; Q[13, 14] = 1.25
        Q[14, 15] = 1.35; Q[15, 20] = 1.35                   
        Q[16, 20] = 1.10
        Q[17, 20] = 0.10
        Q[18, 19] = 1.35; Q[19, 20] = 1.35
        Q[20, 3] = 5.0 
        
        return Q

# ==========================================
# PROCESSING UTILS
# ==========================================

class ProcessorUtils:
    @staticmethod
    def split_vessel_ids(row: pd.Series) -> pd.DataFrame:
        vessel_ids = row['Vessel_ID']
        segments = []
        current_segment = []
        
        for i, v_id in enumerate(vessel_ids):
            if v_id == 2:
                if i == 0 or vessel_ids[i-1] != 2:
                    if current_segment: 
                        segments.append(current_segment)
                    current_segment = [2]
                else:
                    current_segment.append(2)
            else:
                current_segment.append(v_id)
                
        if current_segment: 
            segments.append(current_segment)
            
        return pd.DataFrame({
            'Nanobot_ID': [row['Nanobot_ID']] * len(segments),
            'Vessel_ID': segments
        })

    @staticmethod
    def get_random_value(data_dict: dict, vessel_id: int, beta: float) -> float:
        if vessel_id not in data_dict:
            chosen_id = random.choice(list(data_dict.keys()))
        elif random.random() < beta:
            available_ids = list(data_dict.keys())
            available_ids.remove(vessel_id)
            chosen_id = random.choice(available_ids)
        else:
            chosen_id = vessel_id
            
        min_val, max_val = data_dict[chosen_id]
        return random.uniform(min_val, max_val)

# ==========================================
# STATISTICAL UTILS
# ==========================================

class StatsUtils:
    @staticmethod
    def obtain_confidence_binomial(alpha: float, beta: float, n: int) -> float:
        if n <= 0: return 0.0
        n_adj = n if n % 2 != 0 else n - 1
        p_valid = 1 - (alpha + beta - (alpha * beta))
        failure_threshold = math.floor(n_adj / 2)
        prob_failure = binom.cdf(failure_threshold, n_adj, p_valid)
        return (1 - prob_failure) * 100

    @staticmethod
    def obtain_confidence_hoeffding(alpha: float, beta: float, n: int) -> float:
        if n <= 0: return 0.0
        n_adj = n if n % 2 != 0 else n - 1
        p_valid = 1 - (alpha + beta - (alpha * beta))
        if p_valid <= 0.5: return 0.0
        t = p_valid - 0.5
        prob_failure_bound = math.exp(-2 * n_adj * (t ** 2))
        return max(0.0, 1 - prob_failure_bound) * 100

# ==========================================
# ENERGY MODELING
# ==========================================

class EnergyModel:
    def __init__(self, param_type='press', e_max=800e-12, e_off=40e-12):
        self.C_cap = 9e-9             
        self.V_g = 0.42               
        self.delta_Q = 6e-12          
        self.vibration_freq = 50      
        
        self.E_tx_pulse = 1e-12       
        self.E_rx_pulse = 0.1e-12     
        self.idle_drain_per_sec = 1e-12 
        self.header_bits = 32         
        
        self.E_max = e_max            
        self.E_off = e_off            
        
        task_cost = self.action_consumption(param_type)
        self.E_on = self.E_off + task_cost + 0.1e-12

    def get_current_cycles(self, energy: float) -> float:
        if energy >= self.E_max: return float('inf')
        if energy <= 0: return 0
        try:
            inner = 1 - math.sqrt((2 * energy) / (self.C_cap * (self.V_g**2)))
            if inner <= 0: return float('inf')
            cycles = math.ceil((-self.V_g * self.C_cap / self.delta_Q) * math.log(inner))
            return max(0, cycles)
        except ValueError:
            return 0

    def harvest_energy(self, current_energy: float, delta_t: float) -> float:
        if current_energy >= self.E_max: return self.E_max
        cycles_passed = int(delta_t * self.vibration_freq)
        if cycles_passed == 0: return current_energy
        current_cycles = self.get_current_cycles(current_energy)
        if current_cycles == float('inf'): return self.E_max
        new_cycles = current_cycles + cycles_passed
        voltage_factor = 1 - math.exp(- (new_cycles * self.delta_Q) / (self.V_g * self.C_cap))
        new_energy = 0.5 * self.C_cap * (self.V_g * voltage_factor)**2
        return min(new_energy, self.E_max)
        
    def get_payload_size(self, param_type: str) -> int:
        sizes = {'ph': 8, 'temp': 16, 'combined': 24, 'triple': 40}
        return sizes.get(param_type, 32)

    def get_sensing_cost(self, param_type: str) -> float: 
        costs = {
            'ph': 1.5e-15,
            'temp': 97.64e-15,
            'toxicity': 97.64e-15,
            'combined': 1.5e-15 + 97.64e-15,
            'triple': 1.5e-15 + 97.64e-15 + 97.64e-15
        }
        return costs.get(param_type, 0.5e-15)

    def action_consumption(self, param_type: str) -> float:
        sensing_cost = self.get_sensing_cost(param_type)
        N_data = self.get_payload_size(param_type)
        N_bits = self.header_bits + N_data
        tx_cost = N_bits * 0.5 * self.E_tx_pulse 
        return sensing_cost + tx_cost