import pandas as pd
import numpy as np
import random
import math
from scipy.stats import binom

class DataManager:
    """Storage for anatomical mappings and physiological ranges."""
    
    # Accurate zone mapping
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

    def __init__(self):
        # Full physiological data
        self.datasets = {
            'press': {
                1: [80.0, 120.0], 2: [6.0, 12.0], 3: [80.0, 120.0], 4: [80.0, 120.0], 5: [80.0, 120.0], 
                6: [70.0, 110.0], 7: [70.0, 110.0], 8: [80.0, 120.0], 9: [20.0, 35.0], 10: [80.0, 120.0], 
                11: [80.0, 120.0], 12: [80.0, 120.0], 13: [80.0, 120.0], 14: [80.0, 120.0], 15: [80.0, 120.0], 
                16: [80.0, 120.0], 17: [80.0, 120.0], 18: [80.0, 120.0], 19: [80.0, 120.0], 20: [80.0, 120.0], 
                21: [80.0, 120.0], 22: [80.0, 120.0], 23: [80.0, 120.0], 24: [80.0, 120.0], 25: [80.0, 120.0], 
                26: [80.0, 120.0], 27: [80.0, 120.0], 28: [80.0, 120.0], 29: [80.0, 120.0], 30: [20.0, 35.0], 
                31: [80.0, 120.0], 32: [80.0, 120.0], 33: [80.0, 120.0], 34: [80.0, 120.0], 35: [80.0, 120.0], 
                36: [20.0, 35.0], 37: [80.0, 120.0], 38: [80.0, 120.0], 39: [20.0, 35.0], 40: [20.0, 35.0], 
                41: [80.0, 120.0], 42: [80.0, 120.0], 43: [80.0, 120.0], 44: [70.0, 110.0], 45: [80.0, 120.0], 
                46: [70.0, 110.0], 47: [35.0, 50.0], 48: [70.0, 110.0], 49: [35.0, 50.0], 50: [70.0, 110.0], 
                51: [35.0, 50.0], 52: [35.0, 50.0], 53: [70.0, 110.0], 54: [35.0, 50.0], 55: [70.0, 110.0], 
                56: [35.0, 50.0], 57: [0.0, 5.0], 58: [0.0, 5.0], 59: [35.0, 50.0], 60: [35.0, 50.0], 61: [6.0, 12.0], 
                62: [10.0, 15.0], 63: [10.0, 15.0], 64: [2.0, 6.0], 65: [2.0, 6.0], 66: [2.0, 6.0], 67: [2.0, 6.0], 
                68: [2.0, 6.0], 69: [2.0, 6.0], 70: [2.0, 6.0], 71: [2.0, 6.0], 72: [2.0, 6.0], 73: [2.0, 6.0], 
                74: [2.0, 6.0], 75: [2.0, 6.0], 76: [10.0, 15.0], 77: [10.0, 15.0], 78: [2.0, 6.0], 79: [2.0, 6.0], 
                80: [2.0, 6.0], 81: [2.0, 6.0], 82: [2.0, 6.0], 83: [2.0, 6.0], 84: [10.0, 15.0], 85: [2.0, 6.0], 
                86: [2.0, 6.0], 87: [2.0, 6.0], 88: [2.0, 6.0], 89: [2.0, 6.0], 90: [2.0, 6.0], 91: [2.0, 6.0], 
                92: [2.0, 6.0], 93: [2.0, 6.0], 94: [6.0, 12.0]
            },
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
            'so2': {
                1: [97.0, 99.0], 2: [97.0, 99.0], 3: [97.0, 99.0], 4: [97.0, 99.0], 5: [97.0, 99.0], 
                6: [97.0, 99.0], 7: [97.0, 99.0], 8: [97.0, 99.0], 9: [80.0, 85.0], 10: [97.0, 99.0], 
                11: [97.0, 99.0], 12: [97.0, 99.0], 13: [97.0, 99.0], 14: [97.0, 99.0], 15: [97.0, 99.0], 
                16: [97.0, 99.0], 17: [97.0, 99.0], 18: [97.0, 99.0], 19: [97.0, 99.0], 20: [97.0, 99.0], 
                21: [97.0, 99.0], 22: [97.0, 99.0], 23: [97.0, 99.0], 24: [97.0, 99.0], 25: [97.0, 99.0], 
                26: [97.0, 99.0], 27: [97.0, 99.0], 28: [97.0, 99.0], 29: [97.0, 99.0], 30: [80.0, 85.0], 
                31: [97.0, 99.0], 32: [97.0, 99.0], 33: [97.0, 99.0], 34: [97.0, 99.0], 35: [97.0, 99.0], 
                36: [75.0, 80.0], 37: [97.0, 99.0], 38: [97.0, 99.0], 39: [80.0, 85.0], 40: [85.0, 90.0], 
                41: [97.0, 99.0], 42: [97.0, 99.0], 43: [97.0, 99.0], 44: [97.0, 99.0], 45: [97.0, 99.0], 
                46: [97.0, 99.0], 47: [95.0, 97.0], 48: [97.0, 99.0], 49: [95.0, 97.0], 50: [97.0, 99.0], 
                51: [95.0, 97.0], 52: [95.0, 97.0], 53: [97.0, 99.0], 54: [95.0, 97.0], 55: [97.0, 99.0], 
                56: [95.0, 97.0], 57: [70.0, 75.0], 58: [70.0, 75.0], 59: [95.0, 97.0], 60: [95.0, 97.0], 
                61: [97.0, 99.0], 62: [70.0, 75.0], 63: [70.0, 75.0], 64: [70.0, 75.0], 65: [70.0, 75.0], 
                66: [70.0, 75.0], 67: [70.0, 75.0], 68: [70.0, 75.0], 69: [70.0, 75.0], 70: [70.0, 75.0], 
                71: [70.0, 75.0], 72: [70.0, 75.0], 73: [70.0, 75.0], 74: [70.0, 75.0], 75: [70.0, 75.0], 
                76: [75.0, 80.0], 77: [75.0, 80.0], 78: [70.0, 75.0], 79: [70.0, 75.0], 80: [70.0, 75.0], 
                81: [60.0, 70.0], 82: [70.0, 75.0], 83: [60.0, 70.0], 84: [75.0, 80.0], 85: [70.0, 75.0], 
                86: [70.0, 75.0], 87: [70.0, 75.0], 88: [70.0, 75.0], 89: [70.0, 75.0], 90: [65.0, 75.0], 
                91: [70.0, 75.0], 92: [70.0, 75.0], 93: [70.0, 75.0], 94: [97.0, 99.0]
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

    def get_data_dict(self, param_key):
        """Returns the specific parameter dictionary."""
        return self.datasets.get(param_key.lower())

    def get_decimal_precision(self, param_type):
        """Maps the parameter to its decimal rounding precision."""
        if param_type == 'so2': return 0  
        elif param_type == 'ph': return 2  
        elif param_type == 'temp': return 3  
        elif param_type == 'press': return 4  
        return 3

    @staticmethod
    def assign_zone(vessel_id):
        for zone, vessels in DataManager.ZONE_TO_VESSELS.items():
            if vessel_id in vessels:
                return zone
        return None

class ProcessorUtils:
    @staticmethod
    def split_vessel_ids(row):
        vessel_ids = row['Vessel_ID']
        segments = []
        current_segment = []
        for i, v_id in enumerate(vessel_ids):
            if v_id == 2:
                if i == 0 or vessel_ids[i-1] != 2:
                    if current_segment: segments.append(current_segment)
                    current_segment = [2]
                else:
                    current_segment.append(2)
            else:
                current_segment.append(v_id)
        if current_segment: segments.append(current_segment)
        return pd.DataFrame({
            'Nanobot_ID': [row['Nanobot_ID']] * len(segments),
            'Vessel_ID': segments
        })

    @staticmethod
    def get_random_value(data_dict, vessel_id, beta):
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

class StatsUtils:
    @staticmethod
    def obtain_confidence_binomial(alpha, beta, n):
        if n <= 0: return 0.0
        n_adj = n if n % 2 != 0 else n - 1
        epsilon = alpha + beta - (alpha * beta)
        p_valid = 1 - epsilon
        failure_threshold = math.floor(n_adj / 2)
        prob_failure = binom.cdf(failure_threshold, n_adj, p_valid)
        return (1 - prob_failure) * 100

    @staticmethod
    def obtain_confidence_hoeffding(alpha, beta, n):
        if n <= 0: return 0.0
        n_adj = n if n % 2 != 0 else n - 1
        epsilon = alpha + beta - (alpha * beta)
        p_valid = 1 - epsilon
        if p_valid <= 0.5: return 0.0
        t = p_valid - 0.5
        prob_failure_bound = math.exp(-2 * n_adj * (t ** 2))
        return max(0.0, 1 - prob_failure_bound) * 100
    
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

    def get_current_cycles(self, energy):
        if energy >= self.E_max: return float('inf')
        if energy <= 0: return 0
        try:
            inner = 1 - math.sqrt((2 * energy) / (self.C_cap * (self.V_g**2)))
            if inner <= 0: return float('inf')
            cycles = math.ceil((-self.V_g * self.C_cap / self.delta_Q) * math.log(inner))
            return max(0, cycles)
        except ValueError:
            return 0

    def harvest_energy(self, current_energy, delta_t):
        if current_energy >= self.E_max: return self.E_max
        cycles_passed = int(delta_t * self.vibration_freq)
        if cycles_passed == 0: return current_energy
        current_cycles = self.get_current_cycles(current_energy)
        if current_cycles == float('inf'): return self.E_max
        new_cycles = current_cycles + cycles_passed
        new_energy = 0.5 * self.C_cap * (self.V_g * (1 - math.exp(- (new_cycles * self.delta_Q) / (self.V_g * self.C_cap))))**2
        return min(new_energy, self.E_max)
        
    def get_payload_size(self, param_type):
        if param_type == 'ph': return 8
        elif param_type == 'so2': return 8
        elif param_type == 'temp': return 16
        elif param_type == 'press': return 32
        elif param_type == 'combined': return 24 # 8 bits (pH) + 16 bits (Temp)
        else: return 32

    def get_sensing_cost(self, param_type): 
        if param_type == 'ph': return 1.5e-15
        elif param_type == 'temp': return 97.64e-15
        elif param_type == 'press': return 836e-15
        elif param_type == 'so2': return 48e-9
        elif param_type == 'combined': return 1.5e-15 + 97.64e-15 # 99.14 fJ
        else: return 0.5e-15

    def action_consumption(self, param_type):
        sensing_cost = self.get_sensing_cost(param_type)
        N_data = self.get_payload_size(param_type)
        N_bits = self.header_bits + N_data
        tx_cost = N_bits * 0.5 * self.E_tx_pulse 
        return sensing_cost + tx_cost