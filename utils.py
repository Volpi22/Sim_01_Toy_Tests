"""
Core Utilities Module

This module provides foundational support classes for the nanonetwork simulation:
1. DataManager: A repository for anatomical mapping and physiological ground-truth data.
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
# DATA MANAGEMENT
# ==========================================

class DataManager:
    """
    Centralized storage for anatomical mappings and physiological ground-truth ranges.
    Maps localized 'Zones' to specific micro-vascular 'Vessel_IDs' and provides reference 
    min/max boundary values for physical parameters across the entire body.
    """
    
    # Anatomical zone mapping (Zone ID -> List of Vessel IDs)
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
        """Initializes the baseline physiological dictionaries."""
        # Physiological baseline ranges [Min, Max] per specific Vessel ID
        self.datasets = {
            'press': {
                # Pressure in mmHg
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
                # pH Levels (Slightly alkaline to neutral)
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
                # Oxygen Saturation in %
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
                # Temperature in Celsius
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
        """
        Retrieves the expected bounds dictionary for a given physical parameter.
        
        @param param_key (str): The string identifier for the parameter (e.g., 'ph').
        @return dict: The dictionary mapping vessel IDs to [min, max] ranges.
        """
        return self.datasets.get(param_key.lower())

    def get_decimal_precision(self, param_type: str) -> int:
        """
        Returns the appropriate number of decimal places for data rounding based on sensor type.
        
        @param param_type (str): The string identifier for the parameter.
        @return int: The number of decimal places.
        """
        precisions = {'so2': 0, 'ph': 2, 'temp': 3, 'press': 4}
        return precisions.get(param_type, 3)

    @staticmethod
    def assign_zone(vessel_id: int):
        """
        Iterates over the anatomical mapping to find the zone containing a specific vessel.
        
        @param vessel_id (int): The unique integer ID of the vessel.
        @return int or None: The Zone ID, or None if the vessel is unmapped.
        """
        for zone, vessels in DataManager.ZONE_TO_VESSELS.items():
            if vessel_id in vessels:
                return zone
        return None

# ==========================================
# PROCESSING UTILS
# ==========================================

class ProcessorUtils:
    """Helper methods for handling data structures and generating stochastic data."""

    @staticmethod
    def split_vessel_ids(row: pd.Series) -> pd.DataFrame:
        """
        Splits lists of sequential vessel IDs when a nanobot passes through the heart 
        (Vessel ID = 2). This operation allows the simulator to discretize a continuous 
        trajectory into separate circulatory 'laps' or cycles.
        
        @param row (pd.Series): A single row containing a list of navigated 'Vessel_ID's.
        @return pd.DataFrame: An expanded DataFrame with discrete circulatory segments.
        """
        vessel_ids = row['Vessel_ID']
        segments = []
        current_segment = []
        
        for i, v_id in enumerate(vessel_ids):
            if v_id == 2:
                # If the heart (vessel 2) is encountered, close the current segment 
                # (unless it is the very first vessel of the simulation)
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
        """
        Generates a stochastic physical reading within the expected bounds of the vessel.
        Utilizes `beta` to simulate a probability of measurement error (e.g., reading 
        data bounds from a randomly selected, incorrect vessel).
        
        @param data_dict (dict): The dictionary mapping vessel bounds.
        @param vessel_id (int): The actual vessel the nanobot is traversing.
        @param beta (float): The probability (0.0 to 1.0) of a sensing error.
        
        @return float: The generated stochastic reading.
        """
        if vessel_id not in data_dict:
            chosen_id = random.choice(list(data_dict.keys()))
        elif random.random() < beta:
            # Inject error: Sample from an incorrect vessel's bounds
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
    """Methods for calculating data confidence using statistical bounding theorems."""

    @staticmethod
    def obtain_confidence_binomial(alpha: float, beta: float, n: int) -> float:
        """
        Calculates statistical confidence using a Binomial Cumulative Distribution Function.
        
        The baseline probability of a valid sample is derived as:
        p_{valid} = 1 - (\alpha + \beta - \alpha\beta)
        
        @param alpha (float): Localization error probability.
        @param beta (float): Sensing error probability.
        @param n (int): The number of samples collected in the packet.
        
        @return float: The confidence percentage (0.0 to 100.0).
        """
        if n <= 0: 
            return 0.0
            
        # Adjust sample size to force a strict majority threshold
        n_adj = n if n % 2 != 0 else n - 1
        
        p_valid = 1 - (alpha + beta - (alpha * beta))
        failure_threshold = math.floor(n_adj / 2)
        
        prob_failure = binom.cdf(failure_threshold, n_adj, p_valid)
        return (1 - prob_failure) * 100

    @staticmethod
    def obtain_confidence_hoeffding(alpha: float, beta: float, n: int) -> float:
        """
        Calculates a statistical confidence bound utilizing Hoeffding's Inequality.
        
        @param alpha (float): Localization error probability.
        @param beta (float): Sensing error probability.
        @param n (int): The number of samples collected in the packet.
        
        @return float: The confidence percentage (0.0 to 100.0).
        """
        if n <= 0: 
            return 0.0
            
        n_adj = n if n % 2 != 0 else n - 1
        
        p_valid = 1 - (alpha + beta - (alpha * beta))
        
        # Hoeffding is unreliable/invalid if the underlying valid probability is worse than random
        if p_valid <= 0.5: 
            return 0.0
        
        t = p_valid - 0.5
        prob_failure_bound = math.exp(-2 * n_adj * (t ** 2))
        return max(0.0, 1 - prob_failure_bound) * 100

# ==========================================
# ENERGY MODELING
# ==========================================

class EnergyModel:
    """
    Simulates a piezoelectric/vibrational energy harvesting system for nanobots.
    Calculates non-linear capacitor charging over time and discrete energy expenditures 
    per sensor activation and packet transmission.
    """
    def __init__(self, param_type='press', e_max=800e-12, e_off=40e-12):
        """
        Initializes the hardware constants and energy limits of the nanobot.
        
        @param param_type (str): The primary sensor modality determining power drain.
        @param e_max (float): The maximum capacity of the storage capacitor (Joules).
        @param e_off (float): The minimum operational threshold before brown-out (Joules).
        """
        # --- Hardware Specifications ---
        self.C_cap = 9e-9             # Capacitor rating (Farads)
        self.V_g = 0.42               # Generation voltage (Volts)
        self.delta_Q = 6e-12          # Charge increment per vibration cycle (Coulombs)
        self.vibration_freq = 50      # Harvesting vibrations per second (Hz)
        
        # --- Energy Expenditures (Joules) ---
        self.E_tx_pulse = 1e-12       # Transmission energy per bit 
        self.E_rx_pulse = 0.1e-12     # Reception energy per bit
        self.idle_drain_per_sec = 1e-12 
        self.header_bits = 32         
        
        # --- Power Thresholds ---
        self.E_max = e_max            
        self.E_off = e_off            
        
        # Determine activation threshold dynamically based on the cost of the next action
        task_cost = self.action_consumption(param_type)
        self.E_on = self.E_off + task_cost + 0.1e-12

    def get_current_cycles(self, energy: float) -> float:
        """
        Reverse-calculates the requisite number of charging cycles to reach a given energy state.
        Derived by solving the capacitor energy equation for 'n' cycles.
        
        @param energy (float): The current energy state in Joules.
        @return float: The equivalent number of vibration cycles. Returns infinity if fully charged.
        """
        if energy >= self.E_max: 
            return float('inf')
        if energy <= 0: 
            return 0
            
        try:
            inner = 1 - math.sqrt((2 * energy) / (self.C_cap * (self.V_g**2)))
            if inner <= 0: 
                return float('inf')
            
            cycles = math.ceil((-self.V_g * self.C_cap / self.delta_Q) * math.log(inner))
            return max(0, cycles)
        except ValueError:
            return 0

    def harvest_energy(self, current_energy: float, delta_t: float) -> float:
        """
        Calculates the cumulative energy harvested into the capacitor over a given elapsed time.
        Models the non-linear charging curve of a physical capacitor.
        
        @param current_energy (float): The existing stored energy.
        @param delta_t (float): Time elapsed since the last update (in seconds).
        
        @return float: The newly calculated energy state, capped at E_max.
        """
        if current_energy >= self.E_max: 
            return self.E_max
        
        cycles_passed = int(delta_t * self.vibration_freq)
        if cycles_passed == 0: 
            return current_energy
        
        current_cycles = self.get_current_cycles(current_energy)
        if current_cycles == float('inf'): 
            return self.E_max
        
        new_cycles = current_cycles + cycles_passed
        
        # Calculate new stored energy via the capacitor charging formula
        voltage_factor = 1 - math.exp(- (new_cycles * self.delta_Q) / (self.V_g * self.C_cap))
        new_energy = 0.5 * self.C_cap * (self.V_g * voltage_factor)**2
        
        return min(new_energy, self.E_max)
        
    def get_payload_size(self, param_type: str) -> int:
        """
        Determines the data payload bit-size for the specific sensor type.
        
        @param param_type (str): The sensor identifier.
        @return int: Data size in bits.
        """
        sizes = {'ph': 8, 'so2': 8, 'temp': 16, 'press': 32, 'combined': 24}
        return sizes.get(param_type, 32)

    def get_sensing_cost(self, param_type: str) -> float: 
        """
        Retrieves the hardware energy cost required to physically actuate the sensor.
        
        @param param_type (str): The sensor identifier.
        @return float: The sensing cost in Joules.
        """
        costs = {
            'ph': 1.5e-15,
            'temp': 97.64e-15,
            'press': 836e-15,
            'so2': 48e-9,
            'combined': 1.5e-15 + 97.64e-15  # 99.14 fJ for dual-actuation
        }
        return costs.get(param_type, 0.5e-15)

    def action_consumption(self, param_type: str) -> float:
        """
        Calculates the aggregate energy consumed by sensing the environment 
        AND transmitting the subsequent data packet.
        
        @param param_type (str): The sensor identifier.
        @return float: Total action energy expenditure in Joules.
        """
        sensing_cost = self.get_sensing_cost(param_type)
        N_data = self.get_payload_size(param_type)
        N_bits = self.header_bits + N_data
        
        tx_cost = N_bits * 0.5 * self.E_tx_pulse 
        return sensing_cost + tx_cost