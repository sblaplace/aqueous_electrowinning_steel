# Chemistry & Physics Improvements — Round 2

> *What would actually improve the chemistry and physics in this repo?*
>
> Companion to `CHEM_PHYS_REVIEW.md`. This document identifies gaps I found
> **beyond** what that review already calls out. Where the first review
> focused on the screening → decision-grade leap in existing modules,
> this one looks at missing physics, unclosed coupling loops, and
> cross-module seams that no single module owns.

---

## 1. Unclosed coupling loops (the physics exists in two modules but they don't talk)

### 1.1 Stress ↔ hydrogen absorption are decoupled

**Modules:** `internal_stress.py`, `hydrogen_embrittlement.py`

Both modules are 1000+ lines each and physically self-consistent, but they
are not coupled. The IPZ hydrogen-entry model in
`hydrogen_embrittlement.py` takes a *fixed* stress state; the Stoney /
bent-strip model in `internal_stress.py` returns a *fixed* stress. In
reality, these form a feedback loop:

- Hydrogen-enhanced localized plasticity (HELP): absorbed H lowers the
  critical stress for dislocation glide, so higher C_H → faster stress
  relaxation → lower driving force for peel.
- Hydrogen-induced decohesion (HID): H at grain boundaries lowers
  cohesive strength, so the stress needed to open a crack is a function
  of C_H.
- The stress field itself modifies trap density: work-hardening during
  deposition creates dislocations that are H traps, so σ → N_t → C_H,eff.

**Concrete addition:** a `stress_hydrogen_coupling.py` that wraps both
modules in a fixed-point iteration (or a small ODE system) at each time
step of `closed_loop.py`. The minimum viable version: at each time step,
compute σ(t) from the current deposit thickness, feed σ into the IPZ
entry model as a trap-density modifier (`N_t = N_t,0 + k_σ · ρ_disl(σ)`),
and feed the resulting C_H back into the stress model as a plasticization
factor. This would let the program predict whether a 100-hour run ends
with the foil curled, cracked, or intact — the peel question in a
single number.

### 1.2 Bulk speciation exists but is not inside the Nernst-Planck film

**Modules:** `speciation.py`, `diffusion_layer_1d.py`

`speciation.py` correctly computes FeSO₄⁰ contact-pair fraction (~15–25%
of total Fe²⁺ at 1.5 M FeSO₄) from thermodynamic K. But
`diffusion_layer_1d.py` treats all Fe²⁺ as free and migrating. The
neutral pair FeSO₄⁰ does **not** migrate (z = 0), so ignoring it
overestimates the transport-limited current by roughly the pair
fraction — i.e. the film model's j_lim is ~20% too high in the sulfate
bath.

**Concrete addition:** add FeSO₄⁰ as a seventh transported species in
`diffusion_layer_1d.py` with z = 0 and a local fast-equilibrium closure
with Fe²⁺ + SO₄²⁻. The ODE gains one variable and one algebraic
constraint; the pair fraction follows from the local SO₄²⁻
concentration, which the film model already computes. This is a ~50-line
change that shifts j_lim and FE predictions in the right direction for
the reference-cell operating point.

### 1.3 Membrane transport has no water electro-osmotic drag

**Module:** `membrane_transport.py`

The docstring explicitly calls this out ("no water osmotic drag"). At
300 mA/cm² through Nafion, the electro-osmotic drag coefficient
(n_w ≈ 2–3 H₂O per H⁺) delivers ~0.5–1 mL/(m²·hr) of water from
catholyte to anolyte. Over a 100-hour run, this shifts catholyte
concentration by several percent — enough to move the Pitzer activity
correction and the Fe²⁺ feed policy. For the program's "closed
charge/mass/electrolyte balance" goal, this is a real leak.

**Concrete addition:** a `water_drag.py` with drag coefficient
n_w(j, T, membrane_age) and the resulting volume flux. Wire it into
`membrane_transport.py:simulate()` as a catholyte volume loss per
timestep. The `bath_dynamics.py` CSTR already tracks volume; this just
adds the trans-membrane term.

---

## 2. Missing physics that no module currently addresses

### 2.1 No quantitative H₂ safety envelope

`gas_holdup.py` computes void fraction and current redistribution. The
deployment package mentions H₂ monitors and LEL alarms. But there is no
module that answers: *given this cell's H₂ generation rate, enclosure
volume, and ventilation, how long until the enclosure reaches 25% LEL
(1% v/v H₂)?*

This matters because the program is "redeployable" — different sites,
different enclosures. A safety model is the kind of thing a site HSE
review will ask for on day one.

**Concrete addition:** `h2_safety.py` with:
- H₂ generation rate from HER partial current (Faraday + ideal gas)
- Enclosure mass balance: dC/dt = G/V − Q_vent·C/V (generation minus ventilation)
- Time to LEL fraction (25%, 50%, 100%) for a given enclosure
- Minimum ventilation rate for steady-state below 25% LEL
- Worst-case scenario: fan failure → accumulation curve

This is ~100 lines and connects `gas_holdup.py` → the deployment
package → a real safety document. It should be a required output of
`reference_cell_pipeline.py`.

### 2.2 No proper TTT/CCT for arbitrary cooling paths

`carburization.py` estimates martensite fraction from a single quench
rate (exponential f_mart ≈ 1 − exp(−q/30)). `tempering.py` has Ms
(Andrews) and Hollomon-Jaffe. But neither provides a
time-temperature-transformation (TTT) or continuous-cooling-transformation
(CCT) diagram.

The deposit-to-sheet route goes through cold rolling + recrystallization
(JMAK in `thermomechanical.py`), but the post-carburization quench is a
separate thermal history. The program cannot predict:

- Whether a 0.5 mm carburized case forms martensite, bainite, or pearlite
  at the surface for a given quench severity (H-factor)
- Whether the core transforms during the same quench
- The intercritical annealing window for dual-phase steel routes

**Concrete addition:** a `ttt_cct.py` module with:
- Skeletal TTT: nose position from Bhadeshia-type incubation model
  (τ_nose = f(grain_size, composition))
- Avrami-type isothermal transformation fractions for ferrite, pearlite,
  bainite
- Additivity rule (Scheil) for continuous cooling
- Quench severity parameter (Grossmann H) → cooling curve → microstructure

This would make the carburization and thermomechanical pipelines
predict the *phase* of the final product, not just its hardness.

### 2.3 No solute drag / grain-boundary segregation during recrystallization

`thermomechanical.py` does JMAK recrystallization kinetics with n and k₀
as screening parameters. But it doesn't track where the alloying
elements (Ni, C, S, P, Mn) go during annealing. In reality:

- Solute drag (Cahn / Lücke–Stüwe): Ni and Mn slow grain boundary
  migration, refining the recrystallized grain
- Grain-boundary segregation (McLean): S and P partition to boundaries
  during annealing, which is the root cause of temper embrittlement
- Carbon in supersaturated solid solution precipitates as carbides at
  boundaries, pinning them (Zener drag)

**Concrete addition:** extend `thermomechanical.py` with a solute-drag
correction to the JMAK growth rate: `G_eff = G₀ / (1 + α·c_solute)`.
This is one line per solute, with α values from the literature (Cahn
1962; Hutchinson et al. 1984). It makes the recrystallized grain size a
function of composition, not just temperature and prior strain.

### 2.4 No ascorbic-acid / antioxidant kinetic model

The bath spec mentions ascorbic acid as a sacrificial antioxidant
(Fe³⁺ + ascorbate → Fe²⁺ + dehydroascorbate). The `fe3_shuttle.py`
references `ascorbic_acid_M` as a parameter but never uses it kinetically.

**Concrete addition:** add to `fe3_shuttle.py` or `bath_startup.py`:
- Ascorbic acid oxidation rate: r = k_asc · [Fe³⁺] · [AscH⁻]
- Consumption rate → feed rate for steady-state bath
- pH dependence (ascorbate pKa₁ = 4.10, pKa₂ = 11.6; active form is AscH⁻)

This closes the loop on the "how long does the bath last before I need to
add more reductant?" question, which is an operating-cost input.

### 2.5 No anion-specific HER exchange current as the default

`surface_state.py` has the full machinery (Temkin coverage, facet mix,
anion adsorption). `diffusion_layer_1d.py` does NOT use it by default —
it falls back to the constant-i₀ kinetics from `kinetics.py`. This is
the #1 item in the first review and is *almost* wired but not actually
connected.

**Concrete addition:** wire `SurfaceStateKinetics` into
`diffusion_layer_1d.py` as the default HER branch. This is a one-class
swap in the constructor (add `surface_state: bool = False` flag, default
to True after validation against the existing constant-i₀ results).

---

## 3. Cross-scale uncertainty propagation

### 3.1 DFT ΔG_H* → FE → LCOFe chain is not connected

The uncertainty module (`uncertainty/`) does Monte Carlo and Sobol GSA on
the screening parameters. But the DFT anchoring chain is not in the
uncertainty budget:

```
ΔG_H* = −0.40 ± 0.15 eV    (DFT uncertainty)
   → θ_H(η)                 (surface_state.py)
      → i₀,H_eff            (kinetics)
         → FE(j)            (diffusion_layer_1d)
            → V_cell, kWh/t (voltage_decomposition)
               → LCOFe      (technoeconomic)
```

The ±0.15 eV DFT uncertainty propagates to a factor of ~2–3 in i₀,H
(through the exponential in the Volmer equilibrium), which propagates
to ~10–15% in FE at the gate condition, which propagates to ~$50–100/t
Fe in LCOFe. That's the *dominant* uncertainty in the economics and
it's not tracked.

**Concrete addition:** add `ΔG_H*` to the parameter registry
(`uncertainty/parameter_registry.py`) with its DFT uncertainty, and
include it in the Monte Carlo sweep. The output is a probability
distribution for FE at the gate, which is what the kill criterion needs
to be honest about.

### 3.2 No validation boundary tracking

`theory_confidence.py` reports confidence per module. But the codebase
doesn't track which operating points have been *validated against
data*. As the experimental program starts producing measurements, the
validation status of each module needs to be queryable:

```python
validation_status("diffusion_layer_1d", j=300, pH=2.5, T=60)
# Returns: {"phase_I_voltammetry": "validated",
#           "FE_at_300mA_cm2": "pending",
#           "pulse_waveform": "not_started"}
```

**Concrete addition:** a `validation_ledger.py` that maps (module,
operating-region) → validation status, with timestamps and links to
data files. This is the infrastructure that makes the transition from
screening to design evidence-based rather than aspirational.

---

## 4. Electrochemical and transport gaps

### 4.1 No 2D current distribution on the rotating cylinder

`cell_architecture.py` evaluates productivity for five reactor types.
`hull_cell.py` does a 1D primary current distribution. But the rotating
cylinder electrode (the highest-productivity option) has an azimuthally
uniform but *axially varying* current distribution — the top and bottom
of the cylinder see different δ because of the flow field.

**Concrete addition:** a `rotating_cylinder_distribution.py` that
computes δ(y) from the Von Kármán–Pohlhausen or the empirical
Corcos-like correlation (δ ~ r^0.7 · ω^-0.5 · y^0.1), then maps j(y) =
f(δ(y)) through the FE engine. This gives the *average* FE over the
cylinder height, not just the local FE at one point. It's what connects
the bench-scale Hull cell to the pilot-scale drum.

### 4.2 No explicit treatment of the passive film breakdown at the Fe/drum interface

`adhesion_peel.py` treats peel stress from deposit internal stress. But
the actual peel mechanism on a Ti drum is:

1. Deposit grows with compressive stress (peens against drum)
2. On cooling (if T > ambient) or during reverse pulse, stress goes
   tensile
3. At the Fe/Ti interface, a thin TiO₂/Fe oxide forms during idle
4. The interface oxide sets the *thermodynamic* work of adhesion W_ad
5. When σ_residual × t_deposit > G_c(interface), the foil peels

The model has step 1 and 2, but not steps 3–5. The "will it peel?"
question reduces to W_ad(interface), which depends on the interfacial
oxide chemistry.

**Concrete addition:** an `interface_chemistry.py` that computes
W_ad(Fe/TiO₂, Fe/Fe-oxide, bare-Fe/Ti) from the thermodynamic work of
adhesion (surface energies + interfacial energy), and returns a
critical peel thickness t_crit = G_c / σ_residual. Wire into
`adhesion_peel.py` as the interface boundary condition.

---

## 5. Process and control improvements

### 5.1 No feed-forward pH control model

`process_control.py` implements PID control of cell voltage and
temperature. But pH control is the most important slow loop — local pH
at the cathode determines whether Fe(OH)₂ precipitates (which kills FE
and contaminates the deposit). The bath has buffer capacity from
bisulfate and boric acid, but neither the buffer depletion nor the
base-feed policy is modeled.

**Concrete addition:** a `ph_control.py` that:
- Tracks catholyte pH from the H⁺ balance (HER consumption, buffer
  release, makeup acid)
- Computes the base/acid feed rate to hold pH within band
- Reports buffer capacity depletion (when does the boric acid run out?)
- Couples to `bath_dynamics.py` as a controlled input

### 5.2 No anode dissolution uniformity / passivation model

`anode.py` models OER and CER kinetics on a DSA. But for the soluble
iron anode route (Fe → Fe²⁺ + 2e⁻), the anode *dissolves*, and the
dissolution is not uniform — it forms channels and pits. This affects:
- Anode lifetime (replacement interval)
- Local Fe²⁺ concentration in the anolyte (channeling → non-uniform
  feed)
- The Fe²⁺/Fe³⁺ ratio in the anolyte (which sets crossover composition)

**Concrete addition:** an `anode_dissolution.py` with a simple
current-distribution-on-the-anode model (primary current distribution
on a dissolving surface) that predicts the anode shape evolution and
lifetime.

---

## 6. Data and calibration improvements

### 6.1 No standardized raw-data schema for the first experiments

`experimental_data.py` exists but the `experiments/` directory is empty
(no wet-lab data yet). When the first measurements arrive, the format
they arrive in will determine how quickly the calibration pipeline can
consume them.

**Concrete addition:** define an `experiments/data_schema.json` (or
equivalent Pydantic model) that specifies:
- Required columns for each Phase (I: LSV/CV; II: Hull cell weights;
  III: deposit composition; IV: long-run logs)
- Units, uncertainty columns, metadata (bath composition, T, pH,
  reference electrode)
- A validator that rejects non-conformant CSVs before they enter
  `calibration_pipeline.py`

This is not physics, but it's the single thing that determines whether
the transition from "screening" to "data-driven" takes weeks or months.

### 6.2 No cross-validation between modules

`tests/test_cross_model_consistency.py` exists, but it checks
*structural* consistency (imports work, APIs match). It does not check
*numerical* consistency: if `diffusion_layer_1d` predicts FE = 82% at
a reference point, does `pulse.py` at the DC limit agree? Does
`coupled_cell_physics.py` reproduce `diffusion_layer_1d` at zero
gas-holdup? Does `technoeconomic.py`'s energy input match
`voltage_decomposition.py`'s output at the same operating point?

**Concrete addition:** add numerical cross-checks to
`test_cross_model_consistency.py`:
- DC-limit pulse FE vs. diffusion-layer FE (tolerance: 5%)
- Coupled-cell V_cell vs. voltage-decomposition sum (tolerance: 10 mV)
- Techno-economic kWh/t vs. physics kWh/t (tolerance: 2%)

These would catch the drift that accumulates when modules are updated
independently.

---

## Priority summary

| # | Improvement | Lines of code | Impact on kill criteria |
|---|---|---|---|
| 1.1 | Stress–H coupling | ~200 | Peel window prediction |
| 2.5 | Wire surface_state into FE engine | ~50 | FE at gate (Tier 1.1 from review) |
| 1.2 | FeSO₄⁰ in Nernst-Planck film | ~50 | j_lim accuracy |
| 2.1 | H₂ safety envelope | ~100 | Site deployment permit |
| 3.1 | DFT uncertainty → LCOFe | ~100 | Honest kill criterion |
| 2.2 | TTT/CCT diagrams | ~300 | Phase prediction |
| 1.3 | Electro-osmotic water drag | ~50 | Mass balance closure |
| 2.4 | Ascorbic acid kinetics | ~50 | Operating cost |
| 4.1 | 2D current on rotating cylinder | ~200 | Scale-up prediction |
| 4.2 | Interface chemistry (peel) | ~150 | Peel window prediction |
| 5.1 | Feed-forward pH control | ~150 | Long-run stability |
| 2.3 | Solute drag on recrystallization | ~50 | Final grain size |
| 6.1 | Raw data schema | ~100 | Calibration speed |
| 6.2 | Cross-module numerical checks | ~100 | Model integrity |

The top five are the minimum viable physics upgrade: they close the
coupling loops that exist in two modules, wire the mechanism layer
(surface_state) into the default prediction path, and add the safety
and uncertainty bookkeeping that the program's stated "decision-grade"
ambition requires.

— Round 2 review, August 2026.
