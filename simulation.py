"""
Digital Patient Physics Engine
Solves the continuous mass-balance ODEs, models Pharmacodynamic delay (Hysteresis),
and handles Hemodynamic stress responses.
"""
import numpy as np
from utils import DataManager

class DigitalPatientEngine:
    def __init__(self, drug_name='Doxorubicin'):
        self.Q_base = DataManager.get_advection_matrix()
        self.V = np.array([DataManager.VOLUMES.get(i, 1.0) for i in range(21)])
        
        self.C = np.zeros(21)  # Blood Plasma Concentration
        self.Ce = np.zeros(21) # Tissue Effect-Site Concentration (Hysteresis)
        
        self.drug = DataManager.DRUGS[drug_name]
        self.pd = DataManager.PD_SHIFTS[drug_name]
        
        # Biological delay constant for drug diffusing into tissues (approx 10 min half-life)
        self.k_e0 = 0.1 

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

    def apply_pharmacodynamics(self, base_ph_array, base_temp_array):
        """
        1. Applies the primary therapeutic shift driven by the EFFECT COMPARTMENT (Ce).
        2. Applies secondary physiological side effects (Fever/Acidosis) if local plasma 
           toxicity exceeds safe thresholds.
        """
        # 1. Primary Therapeutic Effect
        current_ph = base_ph_array + (self.pd['eta_ph'] * self.Ce[1:])
        current_temp = base_temp_array + (self.pd['eta_temp'] * self.Ce[1:])
        
        # 2. Secondary Systemic Side Effects (Toxicity-Induced)
        c_safe = self.drug['C_safe']
        
        for i in range(20):
            plasma_conc = self.C[i + 1] # 1-based indexing in the C array
            
            # If the local organ's concentration exceeds the safe limit, apply side effects
            if plasma_conc > c_safe:
                tox_overage = plasma_conc - c_safe
                
                # Side Effect 1: Drug-induced localized fever (+0.5 °C per unit of severe overage)
                current_temp[i] += (0.50 * tox_overage)
                
                # Side Effect 2: Drug-induced metabolic acidosis (-0.2 pH per unit of severe overage)
                current_ph[i] -= (0.20 * tox_overage)
        
        return current_ph, current_temp