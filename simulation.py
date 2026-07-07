"""
Digital Patient Physics Engine
Solves the continuous mass-balance ODEs, models Pharmacodynamic delay (Hysteresis),
and handles Hemodynamic stress responses.
"""
import numpy as np
import copy
from utils import DataManager

class DigitalPatientEngine:
    def __init__(self, drug_name='Doxorubicin'):
        self.Q_base = DataManager.get_advection_matrix()
        self.V = np.array([DataManager.VOLUMES.get(i, 1.0) for i in range(21)])
        
        self.C = np.zeros(21)  
        self.Ce = np.zeros(21) 
        
        # DEEP COPY to prevent mutating the global DataManager dictionary
        self.drug = copy.deepcopy(DataManager.DRUGS[drug_name])
        self.pd = copy.deepcopy(DataManager.PD_SHIFTS[drug_name])
        
        self.k_e0 = 0.1 
        self.tumor_mass = 1.0  
        
        self.gamma = 4.0
        self.tumor_growth_rate = 0.006

    def step_pharmacokinetics(self, infusion_rate_pct, dt_minutes=1.0, injection_zone=20, cardiac_multiplier=1.0):
        """
        Euler integration step for the mass-balance equation with dynamic heart rate.
        Uses micro-stepping to prevent numerical instability (exploding gradients) in small organs.
        """
        Q_dynamic = self.Q_base * cardiac_multiplier
        
        # Break the 1-minute AI step into 60 smaller physics micro-steps (seconds)
        micro_steps = 60
        dt = dt_minutes / micro_steps
        
        for _ in range(micro_steps):
            dC_dt = np.zeros(21)
            
            for i in range(1, 21):
                # Advection uses the dynamically scaled Q matrix
                inflow = sum(Q_dynamic[j, i] * self.C[j] for j in range(1, 21))
                outflow = sum(Q_dynamic[i, j] * self.C[i] for j in range(1, 21))
                
                # Elimination only in the specific clearance zone
                elimination = 0
                if i == self.drug['clearance_zone']:
                    elimination = self.drug['k_elim'] * self.C[i] * self.V[i]
                    
                source = infusion_rate_pct if i == injection_zone else 0
                
                # Mass balance equation
                dC_dt[i] = (inflow - outflow - elimination + source) / self.V[i]
                
            # Update Plasma Concentration safely via micro-step
            self.C += dC_dt * dt
            self.C = np.clip(self.C, 0, None)
            
            # Update Effect-Site Concentration (Pharmacodynamic Delay) safely
            dCe_dt = self.k_e0 * (self.C - self.Ce)
            self.Ce += dCe_dt * dt
            self.Ce = np.clip(self.Ce, 0, None)
            
        return self.C

    def apply_pharmacodynamics(self, base_ph_array, base_temp_array, tz_idx, max_ph_offset, max_temp_offset):
        """
        1. Translates Effect Compartment (Ce) into Tumor Mass reduction (Hill equation).
        2. Simulates natural tumor regrowth.
        3. Maps the latent Tumor Mass into observable physiological pH/Temp offsets.
        4. Applies secondary toxicity side effects.
        """
        c_safe = self.drug['C_safe']
        
        # --- LATENT TUMOR DYNAMICS ---
        ce_tumor = self.Ce[tz_idx + 1] 
        ec50 = 0.70 * c_safe
        
        # Use the dynamic attributes
        hill_fraction = (ce_tumor ** self.gamma) / ((ec50 ** self.gamma) + (ce_tumor ** self.gamma) + 1e-8)
        
        # Use the dynamic attributes
        growth = self.tumor_growth_rate * self.tumor_mass * (1.0 - self.tumor_mass) 
        kill = 0.05 * hill_fraction * self.tumor_mass              
        
        self.tumor_mass = np.clip(self.tumor_mass + growth - kill, 0.0, 1.0)
        
        # --- MAP TO PHYSIOLOGY ---
        current_ph = np.copy(base_ph_array)
        current_temp = np.copy(base_temp_array)
        
        current_ph[tz_idx] += (self.tumor_mass * max_ph_offset)
        current_temp[tz_idx] += (self.tumor_mass * max_temp_offset)
        
        # --- SECONDARY TOXICITY EFFECTS ---
        for i in range(20):
            plasma_conc = self.C[i + 1] 
            if plasma_conc > c_safe:
                tox_overage = plasma_conc - c_safe
                current_temp[i] += (0.50 * tox_overage)
                current_ph[i] -= (0.20 * tox_overage)
        
        return current_ph, current_temp