# Chemistry & Physics Improvement Suggestions

A focused review of *this* repository — `sblaplace/aqueous_electrowinning_steel`.
These are not generic "improve your model" suggestions; they are gaps I
actually found while reading the model code (`models/*.py`) and tests. I've
tried to rank by impact on the program’s decision-grade questions
(Phase II–IV gates: FE ≥ 70 %, V_cell ≤ 4,000 kWh/t Fe, peelable foil,
structural grade).  High-impact items first.

---

## Tier 1 — directly moves the kill-criterion numbers

### 1.1 HER mechanism is single-channel — no anion-specific or facet-specific chemistry
**Where:** `models/kinetics.py`, `models/her_microkinetics.py`,
`models/diffusion_layer_1d.py`.

The HER partial current is one Butler–Volmer branch with a single `i₀,H`
that is *the* design lever (the README's "95.8 % FE with inhibitor" case
is just `i₀,H` going from 1e-2 → 1e-5 A/m²). That is fine for screening,
but the program's whole case rests on this number, and the code currently
cannot represent the *real* chemistry of how bath chemistry suppresses HER:

- No specific adsorption of anions. The HER Tafel slope and i₀ on Fe
  depend strongly on whether the surface is H-covered (acid) or
  OH-covered (alkaline) and on what is in the inner Helmholtz plane
  (Cl⁻, SO₄²⁻, HSO₄⁻, borate). The AWARE result — 99 % FE in
  concentrated LiCl — is a Cl⁻/H₂O adsorption story, not a
  concentration story. Today the README's "AWARE Acidic" scenario uses
  a single `FE = 0.85` *input* rather than a chemistry-derived one.
- No coverage-dependent i₀. `i₀,H` is a constant; Volmer–Heyrovský
  predicts it should vary with θ_H (Temkin isotherm). You already have
  `her_microkinetics.py` (DFT-anchored Volmer–Heyrovský, ΔG_H* ≈ −0.40 eV)
  — wire it back into `kinetics.py` as the surface-state-dependent i₀.
- No facet dependence. As-deposited iron is nanocrystalline and
  exposes many orientations; the relative area of (110) vs (100) vs (211)
  changes with overpotential and pulse waveform. Each has its own
  ΔG_H* and Tafel slope. The 1 µm grain size that drives
  `mechanical_properties.HallPetch` is also the grain size that sets
  the HER site distribution.

**Concrete addition:** a `surface_state.py` module that returns
`i₀(η, θ_H, θ_OH, Γ_Cl, facet_mix)` from a coupled Volmer–Heyrovsky /
Temkin microkinetic model, parameterised by the bath anion, pH and
overpotential. Even a 1-equation correction (Temkin-averaged i₀ vs
coverage) would let the FE engine predict why chloride helps instead of
having it as an exogenous scenario knob.

### 1.2 Solid-phase Fe chemistry in the deposit is treated as pure Fe
**Where:** `models/mechanical_properties.py`, `models/steel_grade.py`,
`models/co_deposition.py`, `models/fe3_shuttle.py`, `models/carburization.py`.

Real electrodeposited iron always carries dissolved O, S, P, N, H, C and
inclusions of Fe(OH)₂, FeOOH, Fe₃O₄, basic sulfates. The current
mechanical model uses Ni and C as the only solid-solution strengtheners
and ignores these because they enter as separate "defects" rather than
as alloying elements:

- **Oxygen in deposit.** O in electrodeposited Fe comes from co-deposited
  Fe(OH)₂/FeOOH particles; it sets the upper-bound yield strength
  (a 1000 ppm O deposit is hard, brittle, and bad for rolling) and
  controls whether the as-deposited foil can be cold-rolled at all.
  There is no `oxygen_in_iron.py` analogue to the existing
  `hydrogen_embrittlement.py`. Adding it would let the program
  predict the deposit-quality ceiling for the chosen pulse waveform,
  not just the chemistry ceiling.
- **Inclusion population** (FeOOH, Fe₃O₄, basic Fe-sulfate). The
  `deposit_morphology.py` has a *morphology* classification but
  nothing tracks *inclusion volume fraction* through the bath. Yet
  the inclusion content is what determines whether the foil passes
  a Charpy or bend test. `co_deposition.py` has a `ParticleIncorporation`
  class for carbon; reuse the Langmuir framework for endogenous
  Fe-hydroxide particles.
- **S, P, Mn, Si, B.** These define whether the deposit routes to
  AISI 1005 vs 1018 vs 10xx with sulfides. The grade router treats them
  as absent. If the program later wants "low-sulfur deep-drawing
  steel," the model needs at least a placeholder
  `bath_impurity_codeposition.py` driven by the
  `models/impurity_codeposition.py` BathKinetics machinery (which
  already has Cu/Ni/Zn — extending to Mn²⁺, S²⁻/SO₃²⁻, SiO₃²⁻ is a
  small step).

### 1.3 Passivation / oxide film on Fe is not in the Nernst–Planck solve
**Where:** `models/transport.py`, `models/diffusion_layer_1d.py`,
`models/anode.py`, `models/fe3_shuttle.py`.

The Pourbaix diagram (`models/pourbaix.py`) correctly shows that below
about pH 9 and above the Fe²⁺/Fe line, Fe is in the passive Fe(OH)₂/FeOOH
region — and at lower potentials the surface can carry a Fe(OH)₂ film
even in acid. In the Nernst–Planck film, however, the cathode surface
is implicitly bare Fe. That is fine for a screening engine but it
misses two things that matter at decision-grade:

- **Fe(OH)₂ film resistance.** At pH > 4 or with low dissolved O₂, the
  precipitate predicted by the `precipitation_sink` diagnostic can
  form a real film, raising the surface overpotential by 10s of mV
  and feeding the peel/stress story. A coupled film-thickness ODE
  (precipitation flux from `diffusion_layer_1d` minus dissolution
  by Fe²⁺-promoted reductive dissolution) is the right next step.
- **Fe³⁺ boundary layer at the anode.** The anode module models OER
  and CER but not the local Fe³⁺ accumulation when anolyte Fe²⁺
  is oxidised. Fe³⁺ lowers the local pH, raises the OER overpotential,
  and (via the shuttle) is the entire Fe(OH)₃ sludge story at the
  cathode. The anode film is currently thin.

### 1.4 The AWARE / concentrated-chloride physics is not modelled — only stipulated
**Where:** `models/scenarios.py` (line 285: `AWARE_ACIDIC`), `models/anode.py`.

The README's entire acidic-route story rests on AWARE. Today the
anion-rich (10–12 M LiCl) bath enters the model as a *scenario with
preset numbers* rather than as a derivation from physics:

- Conductivity at 10 M LiCl is set by input, not computed.
- The reported "near-unity FE" is not derived from chloride-induced
  HER suppression — it is a scenario parameter.
- Cl₂/OER selectivity at the DSA is in `anode.py` (E0_CER = 1.360 V,
  kinetics) but there is no `anolyte_chloride_chemistry.py` linking
  it to the membrane and to the catholyte Fe inventory.

**Concrete addition:** a chloride-aware `bath.py` that lets the
existing `transport.py` and `kinetics.py` handle Cl⁻ as the supporting
anion with the right mobility and pairing (FeCl⁺, FeCl₂(aq) are
real ion pairs with K ≈ 10⁰·⁵–10¹ in concentrated chloride — currently
the Pitzer model has NaCl/Na₂SO₄/MgSO₄/FeSO₄ anchors but not
Fe–Cl–water).

---

## Tier 2 — fills a known explicit gap

### 2.1 No Fe(OH)₃ / FeOOH phase speciation
**Where:** `models/fe3_shuttle.py`, `models/pourbaix.py`,
`models/bath_dynamics.py`.

`fe3_solubility_cap_M` treats Fe(OH)₃ as the only Fe(III) solid and
ignores that aging/hydrolysis at near-neutral pH goes through
ferrihydrite → goethite (α-FeOOH) → hematite (α-Fe₂O₃) on very
different timescales, with the *initial* phase being the more soluble
2-line ferrihydrite. This matters because:

- The sludge bleed in the closed-loop CSTR is sized against the
  Fe(OH)₃ Ksp; if the actual phase is ferrihydrite (Ksp ~ 10⁻³⁹·⁴)
  or akaganeite in chloride (β-FeOOH, formed from FeCl₃ hydrolysis),
  the bleed is wrong by 1–2 decades.
- The Fe(OH)₃ → FeOOH aging releases H⁺ on a different schedule
  than instant precipitation, and that schedule controls pH drift
  in the bath.

A `ferric_hydroxide_phases.py` with the three relevant Ksp values
and Ostwald-stepping rules would close this.

### 2.2 Nernst–Planck uses infinite-dilution diffusivities with one Ea
**Where:** `models/thermodynamic_constants.py:DIFFUSION_EA_J_MOL = 18e3`,
`models/transport.py`, `models/diffusion_layer_1d.py`.

`D_FE2 = 7.2e-10 m²/s` at 25 °C is the *infinite-dilution* value. In a
1.5 M FeSO₄ + 0.5 M Na₂SO₄ bath, the real D is 30–50 % lower
(see Lobo & Quaresma data cited in the Pitzer README). One Arrhenius
Ea for all species hides the fact that D_Fe²⁺ in concentrated
sulfate is a *strong* function of composition. `pitzer.py` has the
activity coefficients; coupling the Stokes–Einstein-style D(γ, c)
to the Nernst–Planck solve is a 50-line change and would make the
high-current FE predictions (≥ 300 mA/cm²) materially more accurate.

### 2.3 Charge-transfer kinetics are assumed temperature-independent except for one Ea
**Where:** `models/kinetics.py:EA_FE_DEPOSITION_J_MOL`, `EA_HER_ON_FE_J_MOL`.

The Tafel slopes and symmetry factors α are held constant. In reality
α (and hence the BV slope RT/αF) is mildly potential-dependent
(Marcus-like, especially at high |η|), and the pre-exponential
includes an entropic term that varies with double-layer structure
(Frumkin correction). A `frumkin.py` that returns an
`α_eff(η, ψ₁)` and a heat-of-activation correction would:
(a) make the high-η (> 200 mV) predictions more honest and
(b) let the model represent leveler/additive adsorption as
"ψ₁ is shifted by Γ_organic × μ_dipole" — which is *the* way
saccharin, thiourea, and chloride work in practice.

### 2.4 No dendritic-growth ODE coupled to the Nernst–Planck film
**Where:** `models/deposit_morphology.py:dendrite_critical_current`,
`models/pulse.py`, `models/diffusion_layer_1d.py`.

`dendrite_critical_current` returns a *static* threshold; the pulse
model has a 1-D diffusion film but the surface is flat. Real dendrite
initiation needs a Mullins–Sekerka / Barton–Bockris instability
analysis: at the screening length λ = (D·γ·Ω / (j·∂c/∂x|_surf))^(1/2)
a perturbation of that wavelength grows. Adding a stability criterion
and a growth-rate ODE would let `pulse.py` *predict* the morphology
of the deposit instead of warning after the fact. This is also
where the AWARE "coherent foil" claim and the rotating-cylinder
"powder" claim actually disagree.

> **Resolved (task t_7b23bd93).** `deposit_morphology.MullinsSekerkaGrowthModel`
> implements the screening length λ_c = (D·γ·Ω/(j·(∂c/∂x)|_surf))^(1/2) (Fick
> closure ∂c/∂x|_surf = j/(zFD) by default, Nernst when surface/bulk concs are
> given) and the growth-rate ODE da/dt = σa with dispersion
> σ(k) = v·k·(1 − (k/k_c)²), v = j·Ω/(zF). Opt-in `predict_morphology` on
> `PulseDepositionModel` (default off) drives the ODE with the film's own Fe²⁺
> gradient and reports λ_c, σ, amplitude gain, and a dendrites/coherent label on
> `PulseResult.morphology`. `predict_morphology(..., growth_model=..., ...)` and
> `predict_dendrite_growth(...)` expose it in the current sweep. At Fe
> electrowinning currents λ_c is nm-scale, so real (µm+) roughness is always
> unstable — the model sides with the dendritic/powder claim absent agitation
> or additives.

### 2.5 Pulse-reverse anodic dissolution is hard-coded as Fe²⁺ release
**Where:** `models/pulse.py`.

The reverse pulse in PRE is supposed to dissolve the high-aspect-ratio
tips of incipient dendrites and any Fe(OH)₂ film. The model returns
Fe²⁺ to the film at 100 % anodic CE. Real anodic dissolution at
the potentials typical of a reverse pulse is *not* all Fe²⁺ — there's
a non-trivial Fe³⁺ fraction (the same E0_Fe3/Fe2 = 0.771 V line from
`pourbaix.py`) that becomes significant at low pH and high anodic
overpotential. That Fe³⁺ produced during the reverse pulse then
seeds the Fe(OH)₃ sludge and the H₂-evolution problem at restart.
A Fe²⁺/Fe³⁺ split on the anodic branch would close the loop.

### 2.6 Additive / leveler / brightener adsorption is a single parameter
**Where:** `models/internal_stress.py:saccharin_g_L`,
`models/deposit_morphology.py`, `models/co_deposition.py`.

Saccharin, thiourea, PEG, chloride, coumarin all have literature-
documented mechanisms: site blocking, grain refinement via increased
nucleation rate, and stress relief by hydrogen recombination catalysis
at the surface. Today the model has one `saccharin_g_L` input that
nudges σ_res. A `leveler_kinetics.py` with Langmuir adsorption
isotherms, a Γ-dependent nucleation rate, and a Γ-dependent H
recombination overpotential would turn the existing morphology
screening from "is the deposit coarse or fine?" to "what additive
package gets us to structural grade?". This is the single biggest
gap between the program as modelled and what the lab will actually
do on day 1.

### 2.7 Interfacial (double-layer) capacitance model is constant
**Where:** `models/bath_dynamics.py:C_dl_F_m2 = 0.02`,
`models/eis.py`, `models/calibration_pipeline.py`.

`C_dl = 20 µF/cm²` is a screening value; real values on Fe in
FeSO₄/H₂SO₄ vary from 15 to 60 µF/cm² with potential and from
baths with adsorbed anions. EIS is one of the four main
characterisation outputs the program expects
(`models/calibration.py`, `models/eis.py`); a Gouy–Chapman–Stern
model that returns C_dl(φ_M, Γ_Cl, I) would let the EIS fit return
something other than a screening-level number. The Frumkin
correction above (2.3) is the same change.

---

## Tier 3 — structural and scale-up fidelity

### 3.1 JMAK recrystallization uses screening Avrami parameters
**Where:** `models/thermomechanical.py` (the entire `JMAK` flow).

The README is honest that n and k₀ are screening. The cleanest
single addition: fit a small lookup of measured Fe (and Fe-Ni)
recrystallization data from the open literature (Leslie, Humphreys)
and replace the screening values with a strain-and-T-dependent
table. That makes the deposit-to-sheet gate honest.

### 3.2 No texture / preferred-orientation evolution
**Where:** `models/deposit_morphology.py`, `models/thermomechanical.py`.

Electrodeposited iron has a strong (110) fibre texture at low η and
shifts toward (211) at high η; this carries into the recrystallized
grain texture after rolling. The mechanical model is isotropic
Hall–Petch. Adding even a one-line texture factor
(`f(110)_fraction → Δσ_texture`) — well documented for Fe — would
make the grain-size-only YS estimate a *lower* bound and tighten the
grade router.

### 3.3 Gas hold-up uses water-electrolysis drift-flux with no Fe-specific tuning
**Where:** `models/gas_holdup.py` (uses Zuber–Findlay, Stephan–Vogt).

The README is explicit that the dominant uncertainty is the bubble
departure diameter (2× in d_b → 4× in ε_g). The risk: the electrolyte
in an Fe cell is much more surfactant-laden (Fe-hydroxide, ascorbate,
organic additives) than pure water, which shifts d_b from ~1 mm
toward ~0.3 mm. A `surfactant_correction.py` with a Bancroft-style
σ(Γ) closure and the resulting d_b would let the hold-up module
report a defensible range instead of an inherited one.

### 3.4 No thermal diffusion / Soret effect in the film
**Where:** `models/transport.py`, `models/thermal_balance.py`.

At 300 mA/cm² and the program's 2.6 V cell, ohmic + reaction heat
raise the cathode film by a few K above the bulk (`thermal_balance.py`
catches the bulk-averaged version). A temperature profile across
the film changes D, k_T, and the local E₀; in concentrated baths
the Soret coefficient is large enough that the temperature-driven
Fe²⁺ flux is non-negligible. A single `Soret_transport.py` coupled
to the 1-D solver would let the high-j model carry its own
temperature field instead of borrowing from a lumped balance.

### 3.5 No creep / stress relaxation during long runs
**Where:** `models/internal_stress.py`, `models/hydrogen_embrittlement.py`,
`models/closed_loop.py`.

The program runs at 100–300 mA/cm² for hours. The internal-stress
model returns a snapshot; in reality stress relaxes by:
dislocation glide, GB diffusion (Coble), and H-enhanced localised
plasticity. The same stress relaxation is what determines whether
the deposit can survive being wound on a drum at thickness. A
simple `stress_relaxation.py` with a log-linear σ(t) = σ₀(1 −
A·ln(1 + t/τ)) closure, coupled to the temperature and H fields,
would let `closed_loop.py` report a *defect rate* with a stress
mechanism attached.

### 3.6 CSTR / closed-loop model is ideal-mixed
**Where:** `models/closed_loop.py`, `models/bath_dynamics.py`,
`models/crate.py`.

A bench cell at bench scale is well-mixed. A crate of 50 cells in
series with the recycle loop it implies is *not*: residence-time
distribution matters for the Fe(OH)₃ bleed policy and for
catholyte pH drift. A tanks-in-series upgrade (N = 2–4) would
be a small change and would make the program ready to discuss
the crate stage honestly.

### 3.7 Pourbaix diagram is the 25 °C form; high-T diagram is sketched, not computed
**Where:** `models/pourbaix.py`, `docs/SIM_POURBAIX*`.

The reader sees a 25 °C Pourbaix in the README. The program runs at
60 °C and is allowed up to 90 °C. The 60 °C Fe–H₂O Pourbaix
shifts the Fe²⁺/Fe line, the Fe(OH)₂/Fe line, and the HER line in
*different* directions, and the *gap* between Fe²⁺ deposition and
HER is the program’s central lever. A `pourbaix_at_T.py` that
recomputes all five boundary lines with the standard ΔG_f(T) and
ΔH_f data already implicit in the van't Hoff framework would make
this central trade-off visible at the operating temperature.

---

## Tier 4 — process and integration improvements

### 4.1 Batch-to-continuous interface (start-up / shut-down) is sparse
**Where:** `models/bath_startup.py`, `models/process_control.py`.

The bath_startup module handles the first 48 h. There is no
*symmetric* shut-down / restart model — and the program is designed
to be redeployable ("a reconfigurable, redeployable production
platform (cell → crate → site)"). Shutting down a divided sulfate
cell with an Fe cathode means H₂O back-diffusion, Fe(OH)₃ ageing
during idle, and re-passivation of the Fe deposit. A `shutdown.py`
that mirrors `bath_startup.py` would round out the deployability
story.

### 4.2 No Fe mass-balance closure across the cell
**Where:** `models/closed_loop.py`, `models/fe3_shuttle.py`,
`models/run_record.py`, `models/campaign.py`.

A weighed, characterised iron deposit with a closed charge/mass/
electrolyte balance is the program’s stated primary artifact
(README line: "The primary artifact is a weighed, characterized
iron deposit with a closed charge/mass/electrolyte balance — not
a photograph."). The Fe-shuttle and closed-loop modules track
Fe²⁺ and Fe³⁺ in the bath, but I do not see a single
`iron_mass_balance.py` that, given a run record, returns
`m_Fe_deposited + m_Fe_OH3_sludge + m_Fe3_crossover =
m_Fe_leed_in - m_Fe_in_bath_change` with units and a closure
fraction. This is one of the cheaper additions in the list and
one of the most valuable for the experimental program.

### 4.3 Validation hooks for the Fe–Ni Pourbaix / activity data
**Where:** `models/pitzer.py` (test_anchors at line 400+).

`pitzer.py` validates against NaCl, Na₂SO₄, MgSO₄, FeSO₄ anchors.
There is no anchor against published FeCl₂ or Fe–Cl–water data.
Adding two or three (γ±(FeCl₂, 0.1 m), γ±(FeCl₂, 1 m), φ(FeCl₂ +
LiCl mixed)) would both harden the AWARE-path code and de-risk the
chloride recommendation.

### 4.4 No 'nucleation + early-growth' transient in the LSV/CV synthetic
**Where:** `models/voltammetry.py`, `models/run_voltammetry.py`.

Phase I reads LSV/CV for the *steady-state* Tafel region. The
*rising* part of the first scan after a clean is where the
nucleation overpotential and the monolayer-formation kinetics
live. A `nucleation_transient.py` that adds an A·t^(−1/2) or
Scharifker–Hills 3D progressive/instantaneous nucleation branch
to the synthetic CV would give the calibration pipeline a target
for the parameter that is most diagnostic of deposit quality
and most missing from the model.

---

## Cross-cutting suggestions

### A. Add an `assumptions.md` that lists each screening assumption with a known-validity range

The codebase is admirably self-aware — almost every module header
states its scope. A single `docs/ASSUMPTIONS.md` that enumerates
every *screening* assumption with the parameter range over which
it is valid would make the leap from "screening" to "design"
mechanical, not motivational. Candidates:

| Assumption | Where | Valid range / known break |
|---|---|---|
| Single Tafel slope, single α | `kinetics.py` | |η| < 100 mV |
| Infinite-dilution D | `thermodynamic_constants.py` | I < 0.2 m |
| Constant surface activity | `kinetics.py:197` | bulk γ within 2× of surface γ |
| Single Fe(OH)₃ solid | `fe3_shuttle.py` | pH < 3; breaks at 4–6 |
| Bare metal surface in Nernst–Planck | `transport.py` | pH < 4, η < −0.7 V |
| Ideal-mixed CSTR | `closed_loop.py` | single cell |
| 1-D planar film | `transport.py` | electrode ≥ 10×δ |

### B. Tie the screening models to the literature anchors they cite

The headers cite Huang & Zhang (2004), Harvie et al. (1984), PHREEQC
pitzer.dat, Yuan et al. (2009), AWARE (2024–2025), etc. A short
`references/anchors.md` per claim — author, year, data point, value
the model uses, value the paper reports — would let the screening
numbers be checked in 30 seconds instead of 30 minutes, and is the
right first step before any of the Tier 1 additions.

### C. Add a `model_limitations.py` to `models/` that *self-reports*

The most honest way to keep "we are screening" from being a
synonym for "we don't know" is to add a module that the test
suite queries: every screening assumption in a single dict,
with the (j, T, pH, I) range over which it is valid and a flag
when an operating point falls outside that range. This is the
self-checking version of `theory_confidence.py` and is the
single cheapest piece of code in this list for the program’s
"decision-grade" rhetoric.

---

## Summary

What the repo gets right (and shouldn't be changed without
calibration): Pitzer multicomponent activity, full BV kinetics,
Nernst–Planck with migration, Volmer–Heyrovský DFT anchor, Fe³⁺
shuttle, gas hold-up with self-stirring, hydrogen embrittlement
with IPZ, adhesion/peel with coupon protocol, four-phase
experimental matrix, JMAK rolling.

What would most move the kill-criterion numbers (FE ≥ 70 %,
V_cell ≤ 4,000 kWh/t Fe, peelable foil, structural grade):

1. **Surface-state–dependent HER kinetics** — chloride and
   additives are not knobs, they are mechanisms.
2. **Solid-phase Fe chemistry** — O, S, P, H, inclusions, not
   just Ni and C.
3. **Fe(OH)₃ / FeOOH phase speciation** with aging kinetics.
4. **Composition-dependent D(γ, c)** in the Nernst–Planck solve.
5. **Addtive / leveler Langmuir kinetics** instead of one
   `saccharin_g_L` parameter.

The first two unlock the others; the others tighten them. None
of them require new code architecture — only new module
boundaries and the willingness to call a few literature anchors
by name.

— review of the public state of the repo, August 2026.
