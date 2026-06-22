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
        """
        # 1. Apply Toxicity-Induced Hemodynamic Shift
        Q_dynamic = self.Q_base * cardiac_multiplier
        
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
            
            dC_dt[i] = (inflow - outflow - elimination + source) / self.V[i]
            
        # 2. Update Plasma Concentration
        self.C += dC_dt * dt_minutes
        self.C = np.clip(self.C, 0, None)
        
        # 3. Update Effect-Site Concentration (Pharmacodynamic Delay)
        dCe_dt = self.k_e0 * (self.C - self.Ce)
        self.Ce += dCe_dt * dt_minutes
        self.Ce = np.clip(self.Ce, 0, None)
        
        return self.C

    def apply_pharmacodynamics(self, base_ph_array, base_temp_array):
        """
        Applies the shift driven by the EFFECT COMPARTMENT (Ce), not the blood plasma.
        """
        current_ph = base_ph_array + (self.pd['eta_ph'] * self.Ce[1:])
        current_temp = base_temp_array + (self.pd['eta_temp'] * self.Ce[1:])
        
        return current_ph, current_temp