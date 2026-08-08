"""
Boric-free interfacial buffering and competitive iron-ammine complexation in ammonium baths.

Physics and Chemistry
---------------------
To replace toxic, heavily regulated boric acid (H₃BO₃) buffers, industrial iron
electrowinning electrolytes often employ ammonium salts (e.g., (NH₄)₂SO₄ or NH₄Cl).
This module models the dual-action thermodynamic behavior of the ammonium system:

1. **High-Capacity Interfacial Buffering**:
   The ammonium-ammonia conjugate pair buffers the cathode boundary layer against
   extreme alkaline excursions driven by the parasitic HER:
     NH₄⁺  ⇌  NH₃ + H⁺   (pKa = 9.25 at 25 °C)
   Using van 't Hoff equations, the pKa drops significantly with temperature:
     pKa(60 °C) ≈ 8.2
   This brings the peak buffering capacity directly into the optimal local pH
   envelope of the cathode.

2. **Competitive Ferrous Ammine Complexation**:
   Dissolved free ammonia (NH₃) generated at higher interfacial pH coordinates with
   ferrous iron to form soluble mononuclear complexes:
     Fe²⁺ + n NH₃  ⇌  Fe(NH₃)ₙ²⁺   (for n = 1 to 6)
   The overall stability constants βₙ dictate the complex distribution:
     βₙ = [Fe(NH₃)ₙ²⁺] / ([Fe²⁺] * [NH₃]ⁿ)

3. **Ferrous Hydroxide Suppression**:
   Precipitation of Fe(OH)₂ (which ruins Faradaic efficiency and deposit quality)
   occurs when the ion product exceeds the solubility product:
     Q = [Fe²⁺]_free * [OH⁻]²  >  Ksp_FeOH2  (Ksp ≈ 10⁻¹⁵.¹ at 25 °C)
   By coordinating free Fe²⁺ into soluble ammine complexes, the free Fe²⁺ activity
   is suppressed:
     [Fe²⁺]_free = [Fe]_total / (1 + Sum( βₙ * [NH₃]ⁿ ))
   This pushes the thermodynamic onset of hydroxide precipitation to much higher
   local pH levels, widening the stable operating current density envelope.

References
----------
* Bjerrum, J. (1941). "Metal Ammine Formation in Aqueous Solution." P. Haase and Son, Copenhagen.
* Baes, C. F., & Mesmer, R. E. (1976). "The Hydrolysis of Cations." John Wiley & Sons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AmmoniumSpeciationResult:
    """Solved concentrations of all species in the ammonium-iron electrolyte."""

    pH: float
    free_fe2_M: float
    free_nh3_M: float
    nh4_M: float
    fe_ammine_M: List[float]            # List of Fe(NH3)n^2+ concentrations for n = 1..6
    total_Fe_M: float
    total_ammonia_M: float
    is_hydroxide_precipitated: bool
    saturation_ratio_FeOH2: float       # Q / Ksp (values > 1 indicate oversaturation)


class AmmoniumBufferModel:
    """Thermodynamic solver for the mixed ammonium-iron-water system."""

    def __init__(self, temperature_C: float = 60.0):
        self.temperature_C = max(float(temperature_C), 0.0)
        self.temp_K = self.temperature_C + 273.15
        self.r_gas = 8.314

        # ── Temperature-dependent constants via van 't Hoff or empirical fits ──

        # 1. NH4+ -> NH3 + H+ dissociation
        # pKa at 25 °C is 9.245; dH0 of dissociation ≈ 52.2 kJ/mol
        pka_25 = 9.245
        ka_25 = 10.0 ** (-pka_25)
        dh_diss_j_mol = 52.2e3
        self.ka = ka_25 * math.exp(-(dh_diss_j_mol / self.r_gas) * (1.0 / self.temp_K - 1.0 / 298.15))
        self.pka = -math.log10(self.ka)

        # 2. Fe(OH)2 Solubility product Ksp vs T
        # Ksp at 25 °C is ~8.0e-16 (pKsp ≈ 15.1); dissolution is endothermic
        pksp_25 = 15.1
        ksp_25 = 10.0 ** (-pksp_25)
        dh_ksp_j_mol = 30.0e3
        self.ksp_feoh2 = ksp_25 * math.exp(-(dh_ksp_j_mol / self.r_gas) * (1.0 / self.temp_K - 1.0 / 298.15))

        # 3. Water autoprotolysis constant Kw vs T
        pkw_25 = 14.0
        kw_25 = 10.0 ** (-pkw_25)
        dh_kw_j_mol = 55.8e3
        self.kw = kw_25 * math.exp(-(dh_kw_j_mol / self.r_gas) * (1.0 / self.temp_K - 1.0 / 298.15))

        # 4. Stepwise stability constants of Fe(NH3)n^2+ at 25 °C (overall beta_n)
        # Log beta values for n = 1 to 6
        self.log_beta_25 = [1.4, 2.2, 2.5, 2.6, 2.4, 1.4]
        # Coordination is mildly exothermic, stability decreases slightly with T (dH0 ~ -15 kJ/mol per ligand)
        dh_complex_j_mol_per_step = -15.0e3
        self.beta = []
        for n, log_b in enumerate(self.log_beta_25, 1):
            b_25 = 10.0 ** log_b
            b_t = b_25 * math.exp(-(n * dh_complex_j_mol_per_step / self.r_gas) * (1.0 / self.temp_K - 1.0 / 298.15))
            self.beta.append(b_t)

    def solve_speciation(
        self,
        pH: float,
        total_Fe_M: float,
        total_ammonia_M: float,
    ) -> AmmoniumSpeciationResult:
        """
        Solve the mass-balance equations for the iron-ammine system using a robust Newton-Raphson
        iteration to find the free [NH3] concentration.

        Parameters
        ----------
        pH : float
            Electrolyte pH.
        total_Fe_M : float
            Total dissolved ferrous concentration (mol/L).
        total_ammonia_M : float
            Total dissolved ammonium/ammonia supporting salt concentration (mol/L).

        Returns
        -------
        AmmoniumSpeciationResult
            Complete speciation details.
        """
        h_act = 10.0 ** (-max(float(pH), 0.0))
        tot_fe = max(float(total_Fe_M), 0.0)
        tot_nh = max(float(total_ammonia_M), 0.0)

        # Buffering coefficient for free NH4+ to free NH3: [NH4+] = [NH3] * h_act / Ka
        f_nh4_to_nh3 = h_act / self.ka

        # Solve for free_nh3 (x) using Newton-Raphson on the total ammonia mass balance residue:
        # Total NH3 = x * (1 + h/Ka) + [Fe2+]_free * Sum( n * beta_n * x^n )
        # where [Fe2+]_free = total_Fe / (1 + Sum( beta_n * x^n ))
        
        # Initial guess: x0 is based on the assumption of no complexation
        x = tot_nh / (1.0 + f_nh4_to_nh3)
        
        tol = 1e-12
        max_iter = 100
        for _ in range(max_iter):
            # Evaluate sums
            sum_b_xn = 0.0
            sum_n_b_xn = 0.0
            sum_n2_b_xn = 0.0
            
            for i, b in enumerate(self.beta, 1):
                term = b * (x ** i)
                sum_b_xn += term
                sum_n_b_xn += i * term
                sum_n2_b_xn += (i ** 2) * term
                
            denom_fe = 1.0 + sum_b_xn
            free_fe = tot_fe / denom_fe if denom_fe > 0 else 0.0
            
            # Residual function for ammonia mass balance:
            # f(x) = x * (1 + f_nh4_to_nh3) + free_fe * sum_n_b_xn - tot_nh = 0
            f = x * (1.0 + f_nh4_to_nh3) + free_fe * sum_n_b_xn - tot_nh
            
            # Derivative df/dx:
            # df/dx = (1 + f_nh4_to_nh3) + free_fe * sum_n2_b_xn/x - (sum_n_b_xn * d[Fe2+]/dx)
            # d[Fe2+]/dx = -tot_fe / denom_fe² * d(denom_fe)/dx = -free_fe / denom_fe * sum_n_b_xn/x
            d_fe_dx = - (free_fe * sum_n_b_xn) / (x * denom_fe) if x > 1e-15 else 0.0
            df_dx = (1.0 + f_nh4_to_nh3) + free_fe * (sum_n2_b_xn / x) + d_fe_dx * sum_n_b_xn if x > 1e-15 else (1.0 + f_nh4_to_nh3)
            
            dx = f / df_dx
            x_new = x - dx
            
            # Bound x to keep it physical
            if x_new <= 0:
                x_new = x / 2.0
            x = x_new
            
            if abs(dx) < tol:
                break

        # Calculate final species concentrations
        sum_b_xn = 0.0
        fe_ammine = []
        for b in self.beta:
            sum_b_xn += b * (x ** len(fe_ammine) + 1) # wait, len of fe_ammine + 1 is the power n
            
        # Re-compute with final solved x
        final_beta_terms = [b * (x ** i) for i, b in enumerate(self.beta, 1)]
        denom_fe = 1.0 + sum(final_beta_terms)
        free_fe = tot_fe / denom_fe if denom_fe > 0 else 0.0
        
        fe_ammine = [free_fe * term for term in final_beta_terms]
        nh4 = x * f_nh4_to_nh3

        # ── Precipitation Check ──
        oh_act = self.kw / h_act
        q = free_fe * (oh_act ** 2)
        sat_ratio = q / self.ksp_feoh2
        is_precip = sat_ratio > 1.0

        return AmmoniumSpeciationResult(
            pH=pH,
            free_fe2_M=free_fe,
            free_nh3_M=x,
            nh4_M=nh4,
            fe_ammine_M=fe_ammine,
            total_Fe_M=tot_fe,
            total_ammonia_M=tot_nh,
            is_hydroxide_precipitated=is_precip,
            saturation_ratio_FeOH2=sat_ratio,
        )

    def get_buffer_capacity(self, pH: float, total_ammonia_M: float) -> float:
        """
        Calculate the analytical buffer capacity beta_buf = dB/dpH (mol/L) of the ammonium pair.
        Equation: beta_buf = 2.303 * C_tot * (Ka * [H+] / (Ka + [H+])²)
        """
        h_act = 10.0 ** (-max(float(pH), 0.0))
        denom = self.ka + h_act
        return 2.303 * total_ammonia_M * (self.ka * h_act) / (denom ** 2)


def main() -> None:
    """CLI entrypoint for ammonium buffer and complexation analysis."""
    print("=================================================================")
    print(" Ammonium Boric-Free Buffering and Ferrous Ammine Complexation")
    print("=================================================================")
    
    # Compare 25°C vs 60°C operating conditions
    for t in [25.0, 60.0]:
        model = AmmoniumBufferModel(t)
        print(f"\nAt T = {t:.1f} °C:")
        print(f"  Ammonium pKa                  : {model.pka:.2f}")
        print(f"  Fe(OH)₂ Solubility Product Ksp: {model.ksp_feoh2:.2e}")
        
        # Speciation at different pH values
        print("  Speciation for 1.5 M Fe²⁺ + 1.0 M Total Ammonium:")
        for ph in [3.0, 5.0, 7.0, 8.0, 9.0]:
            res = model.solve_speciation(ph, 1.5, 1.0)
            complexed_fe = sum(res.fe_ammine_M)
            print(f"    pH {ph:.1f} | Free Fe²⁺: {res.free_fe2_M:5.3f} M | Complexed Fe: {complexed_fe:5.3f} M | Sat Ratio: {res.saturation_ratio_FeOH2:.2e} | Precip? {res.is_hydroxide_precipitated}")


if __name__ == "__main__":
    main()
