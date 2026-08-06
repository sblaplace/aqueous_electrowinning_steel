# Pitzer Activity Model + Arrhenius Kinetics — Physics Upgrade Note

**Date:** 2026-08-05
**Status:** Level-0 *screening* — model change, **not gate evidence**
**Modules:** `models/pitzer.py` (new), `models/speciation.py` (rework),
`models/kinetics.py`, `models/diffusion_layer_1d.py`, `models/transport.py`,
`models/cell_physics.py`, `models/operating_window.py`
**Tests:** `tests/test_pitzer.py` (new), `tests/test_speciation.py` (rewritten),
updates to `tests/test_kinetics.py`, `tests/test_diffusion_layer_1d.py`

## What changed and why

### 1. Bath activities: Pitzer replaces an out-of-range Davies model (the big one)

The speciation model previously applied the **Davies equation** (valid to
I ≈ 0.5 mol/kg) inside a self-consistent FeSO₄⁰ pair equilibrium at the
bath's actual ionic strength (I ≈ 1.5–5.4 mol/kg).  Outside its calibrated
range, Davies badly overestimates the divalent-ion activity coefficient
(γ₂ ≈ 0.68 at I=1.6), which — because the neutral pair was assigned γ = 1 —
forced **≈97 % of the dissolved iron into phantom FeSO₄⁰ pairs**.  Verified
on the old default for the reference bath (1 M FeSO₄, 0.5 M Na₂SO₄, 50 °C):

| Quantity | Davies (superseded) | Pitzer (new default) | Plausibility check |
|---|---|---|---|
| Ion-paired Fe(II) | 0.97 M (97.4 %) | 0.27 M (27 %, secondary estimate) | spectroscopic contact-pair estimates for divalent sulfates: ~10–25 % |
| Free [Fe²⁺] | 0.026 M | 1.00 M (fully dissociated convention) | — |
| γ(Fe²⁺) | 0.68 | 0.053 | γ±(FeSO₄, 1 m) ≈ 0.05 (Reardon & Beckie 1987) |
| a(Fe²⁺) | 0.018 | 0.052 | — |
| E_rev(Fe²⁺/Fe) | −0.503 V | −0.488 V | +15 mV |
| κ (50 °C) | 13.5 S/m | 11.4 S/m | pure 1 M FeSO₄ at 25 °C ≈ 5–6 S/m measured |
| pH_precip(Fe(OH)₂) | 6.13 | 5.90 | — |
| water activity | (not computed) | 0.957 | — |

The new engine (`models/pitzer.py`) implements the full multicomponent
Pitzer equations for Fe²⁺–Na⁺–H⁺ ∥ SO₄²⁻–HSO₄⁻ on the molal scale,

including: the 2–2-electrolyte α₁ = 1.4 / α₂ = 12 β² convention (which
absorbs ferrous-sulfate association), Harvie-unsymmetric same-sign mixing
(electrostatic ᵈθ terms — matters here because Na⁺/Fe²⁺ mix unequal
charges), a molar→molal conversion via a documented apparent-molar-volume
density estimate, the osmotic coefficient, and water activity.  Parameters
are the Harvie–Møller–Weare (1984) / PHREEQC-pitzer.dat set with Fe–SO₄
from the Pitzer (1991) tabulation; provenance is recorded in the module.

**Validation** (`tests/test_pitzer.py`):
- γ±(NaCl, 0.001–1 m) reproduces the tabulated curve to <1 % (this pins the
  whole multicomponent machinery),
- γ±(Na₂SO₄, 1 m) = 0.204 vs published 0.204; φ(1 m) = 0.657 vs ≈0.66,
- γ±(FeSO₄, 0.1 m) = 0.159 vs the Kobylin et al. (2011) anchor 0.164
  (R&B 0.161; assessment spread 0.150–0.164),
- γ±(MgSO₄, 0.1 m) = 0.165 vs ≈0.163,
- characteristic FeSO₄ γ± minimum near 2 m with upturn toward the copperas
  solubility limit (3.58 m).

The legacy path is preserved under `solve_speciation(..., model="davies")`
with its failure mode pinned as regression archaeology; the run-specific
warning is in the module docstring.  `cell_physics` and `operating_window`
now also feed the Nernst term with the speciation **activity** (previously
a nominal concentration was passed, with γ ≡ 1 semantics silently
overwriting the speciation-derived E_rev inside `CellVoltageModel`).

#### Net effect at the RC-1 reference point (j = 100 mA/cm², 50 °C)

| | V_cell | Specific energy | E_cathode | IR_electrolyte |
|---|---|---|---|---|
| Davies | 4.509 V | 4348 kWh/t | −0.496 V | 1.03 V |
| Pitzer | 4.702 V | 4534 kWh/t | −0.481 V | 1.22 V |
| Δ | **+0.19 V (+4 %)** | **+186 kWh/t** | +15 mV | +0.19 V |

The two corrections partially cancel: conductivity gets more honest
(the Davies κ was elevated *and* depressed for two different wrong
reasons), the Nernst term gets less negative by 15 mV.  FE at this point
is unchanged (it is set by kinetics/transport, not bulk activities).

### 2. Arrhenius temperature dependence for exchange currents

Previously `fe_i0`/`her_i0` were temperature-independent constants;
temperature moved only Nernst terms and, in `diffusion_layer_1d`,
diffusivities.  Now, in `kinetics.py` (and threaded through
`transport.py` and `diffusion_layer_1d.py`):

- i0 values are anchored at **50 °C** (`kinetics_ref_K = 323.15 K`), so
  results at the bath reference condition are unchanged by construction;
- apparent activation energies (screening ±50 %, literature-family):
  Fe deposition **Ea = 50 kJ/mol**, HER on Fe **Ea = 60 kJ/mol**;
- Fe²⁺ diffusivity in `DepositionKinetics` is Arrhenius-scaled
  (18 kJ/mol, 25 °C anchor) as `diffusion_layer_1d` already did.

**Physical consequence worth reading:** because HER is the more
temperature-activated branch, galvanostatic **FE now falls modestly with
temperature** at HER-active conditions (reference engine: 0.918 at 25 °C
→ 0.879 at 80 °C) — the well-known CE-peaks-at-moderate-T behaviour of
ferrous/zinc electrowinning — instead of rising monotonically as the old
model claimed.  Tests that pinned the old direction
(`test_fe_increases_with_temperature`) were rewritten to pin the
mechanistic decomposition; plan temperature sweeps accordingly.

## Limitations (unchanged honesty)

- Pitzer binary parameters are 25 °C values; only Aφ responds to T.
  Reardon & Beckie (1987) publish fitted T-dependence over 10–60 °C —
  flagged in code as a follow-up.  Treat >60 °C activity numbers as
  extrapolated.
- Density estimate is apparent-volume screening level (±2 %); a measured
  bath density should be supplied when available.
- Conductivity retains an empirical DHO-style attenuation calibrated to
  pure FeSO₄/Na₂SO₄ near 1 M (±15–20 %); measured bath κ supersedes.
- Chloride-route (AWARE) chemistry is not yet in the Pitzer parameter set
  (Fe–Cl, Cl pairing, θ/ψ with chloride) — a listed next step.
- θ(SO₄²⁻,HSO₄⁻) and ψ triplets for the exact Fe²⁺–Na⁺–H⁺ ∥ SO₄²⁻–HSO₄⁻
  mixture are set to zero/secondary values; at bath pH ≥ 2 their effect is
  second-order but measurable against data when it exists.

## Next physics steps (from the same audit)

1. Wiring `scripts/dft_h_adsorption_fe.py` ΔG(H_ads) into a
   Volmer–Heyrovsky–Tafel microkinetic HER model — makes HER a predicted
   quantity rather than a suppressed-HER assumption.
2. Full Butler–Volmer reverse branches (mixed-potential corrosion,
   honest PRE dissolution) — replaces the Tafel-only clipping and the
   pulse-module heuristic split.
3. Fe³⁺ redox shuttle consumption at the cathode + O₂-based bath aging
   (a `bath_dynamics`/`closed_loop` source term).
4. Temperature-dependent Pitzer parameter fits (R&B 1987 functions).
