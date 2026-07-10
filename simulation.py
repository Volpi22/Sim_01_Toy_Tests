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
        
        self.drug = copy.deepcopy(DataManager.DRUGS[drug_name])
        self.pd = copy.deepcopy(DataManager.PD_SHIFTS[drug_name])
        
        self.k_e0 = 0.1 
        self.tumor_mass = 0.85  # Prevents Gompertz ln(1) trap
        
        self.gamma = 4.0
        self.tumor_growth_rate = 0.025
        
        self.real_ph = None
        self.real_temp = None

    def step_pharmacokinetics(self, infusion_rate_pct, dt_minutes=1.0, injection_zone=20, cardiac_multiplier=1.0):
        Q_dynamic = self.Q_base * cardiac_multiplier
        micro_steps = 60
        dt = dt_minutes / micro_steps
        
        for _ in range(micro_steps):
            dC_dt = np.zeros(21)
            for i in range(1, 21):
                inflow = sum(Q_dynamic[j, i] * self.C[j] for j in range(1, 21))
                outflow = sum(Q_dynamic[i, j] * self.C[i] for j in range(1, 21))
                
                elimination = 0
                if i == self.drug['clearance_zone']:
                    elimination = self.drug['k_elim'] * self.C[i] * self.V[i]
                    
                source = infusion_rate_pct if i == injection_zone else 0
                dC_dt[i] = (inflow - outflow - elimination + source) / self.V[i]
                
            self.C += dC_dt * dt
            self.C = np.clip(self.C, 0, None)
            
            dCe_dt = self.k_e0 * (self.C - self.Ce)
            self.Ce += dCe_dt * dt
            self.Ce = np.clip(self.Ce, 0, None)
            
        return self.C

    def apply_pharmacodynamics(self, base_ph_array, base_temp_array, mu_B_ph, mu_B_temp, tz_idx, max_ph_offset, max_temp_offset):
        c_safe = self.drug['C_safe']
        
        # [LOCKED] Moderately increased Emax to counteract physiological delays
        e_max = self.drug.get('Emax', 0.08) 
        
        # [LOCKED] Pathological initialization restored
        if self.real_ph is None:
            self.real_ph = np.copy(mu_B_ph)
            self.real_temp = np.copy(mu_B_temp)
            self.real_ph[tz_idx] += (self.tumor_mass * max_ph_offset)
            self.real_temp[tz_idx] += (self.tumor_mass * max_temp_offset)
            
        ce_tumor = self.Ce[tz_idx + 1] 
        
        growth_ratio = (self.tumor_growth_rate - 0.005) / 0.045
        ec50_factor = 0.60 - (0.30 * growth_ratio)
        ec50 = ec50_factor * c_safe
        
        hill_fraction = (ce_tumor ** self.gamma) / ((ec50 ** self.gamma) + (ce_tumor ** self.gamma) + 1e-8)
        k_eff = 1.0 + 0.2 * np.tanh(self.tumor_mass)
        
        gompertz_growth = self.tumor_growth_rate * self.tumor_mass * np.log(k_eff / max(1e-8, self.tumor_mass))
        stage_multiplier = (1.1 - self.tumor_mass)
        kill = e_max * hill_fraction * max(0.01, self.tumor_mass) * stage_multiplier              
        
        self.tumor_mass = np.clip(self.tumor_mass + gompertz_growth - kill, 0.05, 1.0)
        
        target_ph = np.copy(base_ph_array)
        target_temp = np.copy(base_temp_array)
        target_ph[tz_idx] += (self.tumor_mass * max_ph_offset)
        target_temp[tz_idx] += (self.tumor_mass * max_temp_offset)
        
        for i in range(20):
            plasma_conc = self.C[i + 1] 
            if plasma_conc > c_safe:
                tox_overage = plasma_conc - c_safe
                target_temp[i] += (0.50 * tox_overage)
                target_ph[i] -= (0.20 * tox_overage)
                
        k_ph = 0.03    
        k_temp = 0.12  
        
        self.real_ph += k_ph * (target_ph - self.real_ph)
        self.real_temp += k_temp * (target_temp - self.real_temp)
        
        return self.real_ph, self.real_temp