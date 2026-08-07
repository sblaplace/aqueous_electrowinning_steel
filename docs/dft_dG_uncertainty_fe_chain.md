# DFT dG_H* -> FE-at-the-gate uncertainty chain
# Task t_88ba62d1 — CHEM_PHYS_IMPROVEMENTS_V2.md 3.1

## FE-at-the-gate probability distribution (Monte Carlo, N=2000, seed=42)

| stat | value |
|------|-------|
| mean | 91.43 % |
| std  | 7.13 pt |
| p5   | 77.82 % |
| p50  | 92.93 % |
| p95  | 99.22 % |
| min  | 50.00 % |
| max  | 99.50 % |

theta_H (Volmer coverage) distribution: mean 0.810, std 0.079.

## Sensitivity at the gate (top-3, N=2000)
- current_efficiency_percent: {her_i0: 0.63, fe_i0: 0.52, dG_Hstar_eV: 0.37}
- theta_H: {dG_Hstar_eV: 0.98, ...}  (dG_H* is the dominant driver of the chain's direct surface-state output)

## Mechanism check (±0.15 eV band, single-point at 60 C, eta_HER=0.20 V)
| dG_H* (eV) | i0,H_eff swing | theta_H | FE                |
|------------|----------------|---------|-------------------|
| -0.25 weak | 1.61x          | 0.617   | 88.09 % (lower)   |
| -0.40 nom  | 1.00x          | 0.822   | 93.00 % (base)    |
| -0.55 strong| 0.50x         | 0.920   | 97.42 % (higher)  |

FE swings ~9.3 points across the +/-0.15 eV band with the correct sign
(weaker H binding -> more H evolution -> lower FE).

## Chain wiring
dG_Hstar_eV (registry, normal mean -0.40 std 0.15 eV) ->
  surface_state.volmer_coverage theta_H(eta) ->
  i0,H_eff = her_i0 * swing(dG_H*) ->
  FE = 93 * (fe_i0/(fe_i0 + i0,H_eff)) normalized at nominal ->
  specific_energy_kWh_per_kg -> energy_cost (technoeconomic LCOFe proxy).

## Files
- models/uncertainty/parameter_registry.py  — added dG_Hstar_eV entry (surface_state module)
- models/uncertainty/monte_carlo.py         — rewired FE block to surface-state HER chain
- tests/test_uncertainty_dft_lcofe.py       — 5 tests locking the mechanism
