# Chemistry & Physics Improvements — Round 5 (From Iron Electrowinning to Steel)

> **Scope**: physical & chemical mechanisms that (a) are **not** covered by the
> four prior review rounds (`CHEM_PHYS_REVIEW.md`, `docs/CHEM_PHYS_IMPROVEMENTS_V2.md`,
> `docs/CHEM_PHYS_IMPROVEMENTS_V3.md`, `docs/CHEM_PHYS_IMPROVEMENTS_V4.md`) and
> (b) specifically bridge the program's chosen near-term product — **melt-shop
> iron feedstock** (`docs/RESEARCH_PROGRAM.md`, Option A) — and the eventual
> **steel** product path (Option B).
>
> The four rounds covered single-channel HER, solid-phase O/S/P chemistry, ferric
> hydrolysis, stress–H, dissolved oxygen, membrane heating, coalescence stress,
> Mullins–Sekerka, thermogalvanics, retrograde solubility, osmotic water drag,
> BDD kinetics, shunt currents, H trapping, ore leaching, etc. — most now built in
> `models/`. What is *still missing* is the set of mechanisms below: the chemistry
> that decides **what hydrogen does to the finished product**, the electrocrystallization
> physics that turns **operating parameters into grain size and stress**, and the
> feedstock/tramp-element chemistry that governs **whether recycled iron can become
> steel at all**.

---

## Theme A — Carbon: how "steel" actually gets into the deposit

### A1. Electrochemical carbon co-deposition (dissolved carbon species → interstitial C)

**The gap.** The repo puts carbon into iron *after* deposition (`carburization.py`,
`carbon_potential.py`) or as *particles* (`co_deposition.py`, Guglielmi solid-particle
composite plating). There is no model of co-reducing **dissolved carbon species**
(CO₂, carbonate, formate, CO, or urea) at the strongly-negative Fe deposition
potential (−0.44 V vs. SHE) so that carbon enters the growing layer *as interstitial C*
— the one-step "electrowin steel, not just iron" route.

**The chemistry.** At the potentials where Fe deposits, the reduction of dissolved
carbon species to elemental carbon is thermodynamically accessible, e.g.

```
CO₂ + 4H⁺ + 4e⁻ → C + 2H₂O
HCOO⁻ (formate) + 3H⁺ + 2e⁻ → C + 2H₂O   (CO₂ is barely soluble in acid sulfate;
                                          formate / CO / carbonate are the usable
                                          dissolved carbon vectors in this bath)
```

The useful physics to model:
1. **Activity control → deposit C wt%.** The carbon activity at the interface is set by
   the dissolved-carbon concentration, pH, and local H₂O/CO₂ equilibrium, not by a
   carburizing gas. Deposit C tracks `a_C,interfacial` through the same
   carbon-activity machinery `carbon_potential.py` already uses, just driven
   electrochemically instead of by a gas atmosphere.
2. **Competition with Fe and HER.** Carbon incorporation steals current from Fe and
   co-evolves H₂ (formate/CO₂ reduction is H-consuming), so FE and C wt% are coupled.
3. **Carbon also enters *unwanted* from organic additives.** Saccharin, thiourea, and
   coumarin (see `additive_aging.py`) fragment and carry C (and S) into the deposit.
   This is the *same* interstitial-carbon channel and should be in one model so the
   optimizer can trade "additive that improves stress/morphology" against "additive
   that adds too much C/S for the target grade."

**Concrete addition.** `models/carbon_electrodeposition.py`:
- `carbon_activity(j_Fe, C_dissolved, pH, T, theta_organic)` → interfacial a_C,
- `co_deposit_carbon_wt_pct(...)` → deposit C wt% (electrochemical + additive-derived),
- wire into `diffusion_layer_1d.py` as an additional partial-current branch and into
  `bath_dynamics.py`/`closed_loop.py` for dissolved-carbon mass balance.

**Gate impact.** Turns "carbon added later in a melt furnace" (Option A) into a
first-class *electrowon deposit property*; predicts the C wt% window that lands a
deposit in AISI 1005/1018/1045 without any post-processing. This is the single
biggest missing item for the literal "electrowinning for **steel**" headline.

### A2. As-deposited Fe–C layer → steel-grade router (integration, not new constants)

**The gap.** The product path has three carbon-bearing pipelines that do not talk:
`co_deposition.py` (C in the layer), `carbon_potential.py`/`carburization.py`
(post-deposition carburizing), and `tempering.py`/`thermomechanical.py` (final
microstructure/mechanical). None maps an as-deposited carbon-bearing iron layer
through the Fe–C phase field to an actual steel grade.

**The physics.** If a deposit is carbon-bearing as-plated (A1) or carburized (existing),
its final microstructure — ferrite, pearlite, martensite, retained austenite — is set by
C wt% × cooling history. The repo already has the components (Ms via Andrews,
Hollomon-Jaffe in `tempering.py`, JMAK in `thermomechanical.py`), but no single
`steel_grade` entry takes an *as-deposited* C content and predicts the grade and
mechanicals end-to-end.

**Concrete addition.** `models/as_deposited_grade.py` (or extend `steel_grade.py`):
- input = deposit C (from A1 or carburization), Mn/S/P (from `bath_impurity_codeposition.py`),
  thermal path,
- output = Fe–C phase fractions + AISI grade + YS/UTS/HV + cold-roll verdict,
- reuse the existing TTT/tempering/JMAK kernels so it is a *coupling seam*, not new
  materials physics.

**Gate impact.** Gives Option B a single "will this run make structural-grade steel?"
answer and, for Option A, a "does this carbon content disqualify the flake as a clean
melt-shop charge or make it a pre-alloyed steel feed?"

---

## Theme B — Hydrogen: from the cathode all the way to the finished steel

### B1. Recombination poisons (S, As, Sb, Se, Te, P, CN⁻) as the *master* switch on absorbed H

**The gap.** `hydrogen_embrittlement.py` uses a fixed screening
`absorption_fraction` (~0.05, scaled by pH/T/j). But the *dominant* real-world control
on how much of the evolved H₂ actually enters the iron lattice is the surface coverage
of **recombination poisons** — the classic cathodic poisons (S, As, Sb, Se, Te, P,
CN⁻) that block the Tafel/Heyrovský H₂-recombination step, forcing a far larger
fraction of adsorbed H into the deposit. Electrowon iron from a sulfate/chloride bath
made from pickle liquor is precisely the case where trace S in the feedstock makes
the deposit H-rich.

**The chemistry.**

```
   Volmer      H⁺ + e⁻ → H_ads
   Recombination (poisoned by θ_S, θ_As, ...):
   Heyrovský/Tafel  H_ads + H⁺ + e⁻ → H₂   ← blocked ⇒ H_ads enters lattice instead
```

The link is not currently closed: `bath_impurity_codeposition.py` routes S into the
*deposit* (grade effects), and `additive_aging.py` releases S from saccharin/thiourea,
but neither feeds an **H-absorption-promotion factor**. This is the missing arrow:

```
feedstock S → bath S²⁻/HS⁻ → θ_poison(η) → absorption_fraction(θ_poison) → C_H
                                                              → bakeout time (B3)
```

**Concrete addition.** `models/recombination_poison.py`:
- `theta_poison(C_sulfide, C_arsenic, eta, T)` (Langmuir/Temkin, competitive),
- `absorption_promotion_factor(theta_poison)` — replaces the fixed
  `absorption_fraction` in `hydrogen_embrittlement.py` (permeation flux can rise by
  orders of magnitude in the presence of As/S; screening, needs anchor),
- wire S from `additive_aging.py` and `bath_impurity_codeposition.py` into it.

**Gate impact.** Converts "hydrogen embrittlement" from a per-run screening number into
a *feedstock- and additive-controlled* prediction; tells you which pickle-liquor
impurities and which brighteners make the deposit H-laden and need longer bakeout —
or force a purification step (`purification.py`) before the cell.

### B2. Hydrogen carried into the melt → white-spot / flake risk in the steel product

**The gap.** The near-term product is **melt-shop feedstock** (Option A). None of the
repo's H models answers the question the melt-shop buyer will ask: *"your flake is
H-rich — what does that do to my ingot?"*

**The physics.** Liquid Fe dissolves far more H than solid Fe: at 1600 °C and 1 atm
H₂, liquid iron holds on the order of **25 ppm H**, while the δ/γ solid retains only a
few ppm. When an H-rich electrowon charge melts and resolidifies, the excess H
supersaturates and exsolves as molecular H₂ in voids and microporosity, producing
**internal flake cracks ("white spots", "fish-eyes")** — the classic hydrogen-induced
defect in large steel sections.

**Concrete addition.** `models/melt_hydrogen.py`:
- `H_in_liquid_Fe(C_H_flake, charge_ratio, T)` from the Sieverts square-root law and the
  liquid/solid solubility gap,
- `flake_risk_index(...)` → white-spot propensity (function of section size/cooling),
- recommend a de-embrittlement bake (reuse `hydrogen_trapping.py` bakeout) or
  melt-side degassing to bring the charge below the threshold.

**Gate impact.** Closes the Option-A loop: it makes "low-H flake" a *product spec* for
the feedstock business, not just a deposit-quality nicety. Directly relevant to the
program's stated "the primary artifact is a weighed, characterized iron deposit with a
closed charge/mass/electrolyte balance" — add *hydrogen* to that ledger.

### B3. H₂ bubble engulfment → deposit porosity, pinholes, blistering

**The gap.** `gas_holdup.py` tracks void fraction and current redistribution in the
*channel*; `deposit_morphology.py` classifies morphology; but nothing models whether
an H₂ bubble is **engulfed by the advancing deposit front** (→ porosity, pinholes,
blisters) or detaches first. Porosity is a top deposit-quality / steel-quality metric
and is currently not predicted by any mechanism.

**The physics.** A growing H₂ bubble has a detachment radius set by the balance of
capillary adhesion vs. buoyancy + convective drag (hence surface tension, contact
angle, flow). If the deposit front advances (thickness rate `v_dep ∝ j·FE`) past the
bubble before it detaches, the bubble is captured. The relevant criterion:

```
detach if  bubble growth/rise time  <  deposit advance time to cover the bubble
porosity fraction  ≈  Σ(captured bubbles per area) × bubble volume
```

Bubble size is exactly the `d_b` uncertainty `gas_holdup` flags as dominant; this
closes it into the deposit.

**Concrete addition.** `models/bubble_engulfment.py`:
- `bubble_detachment_radius(gamma, theta_contact, rho, g, flow)` ,
- `deposit_porosity(j, FE, d_b, v_dep, coverage)` → porosity vol% + pinhole/blister flag,
- feed porosity back into `oxygen_in_iron.py` / `mechanical_properties.py` (density and
  cold-roll ceiling) and `deposit_metrology.py`.

**Gate impact.** Makes "dense, blister-free foil/flake" a *predicted* output of the
pulse waveform and flow setpoint rather than a post-mortem, and couples the 
gas-holdup `d_b` uncertainty to the deposit-quality gate where it actually bites.

---

## Theme C — Electrocrystallization: operating parameters → deposit quality (predicted, not assumed)

### C1. Nucleation density → grain size → Hall–Petch (make grain size an output)

**The gap.** `mechanical_properties.py` takes **grain size as an input** to the
Hall–Petch relation. But grain size is set *in the cell* by nucleation: high
overpotential and additive coverage → high nucleation-site density → fine grains →
high YS. Currently `deposit_morphology.py` describes nucleation only *qualitatively*.

**The physics.** Classical 3-D nucleation (atomistic / Scharifker–Hills style) gives a
nucleation rate that rises steeply with overpotential:

```
J_nuc = A · exp( −B / η² )      (3-D progressive nucleation)
N₀ = nucleation-site density (suppressed by adsorbing additives/site blocking)
d_grain ≈ (N₀)^(−1/3)          (fully covered substrate → grain size)
σ_y = σ₀ + k_y · d_grain^(−1/2)   (Hall–Petch, already in mechanical_properties.py)
```

Coupling these turns the *entire* YS/UTS/grade chain into a function of `(j, η, T,
additive coverage)` — which is exactly what the "what additive package gets us to
structural grade?" question needs.

**Concrete addition.** `models/nucleation_grain.py`:
- `nucleation_density(eta, T, gamma, theta_additive)` ,
- `grain_size(...)` , fed into `mechanical_properties.HallPetch` as the default
  (grain-size input becomes an optional override),
- link to `mullins_sekerka.py` (grain size sets the surface-diffusion length that sets
  the smoothing/stability crossover).

**Gate impact.** Converts the mechanical/grade screens from "assuming a grain size" to
"grain size predicted from the plating recipe," making the deposit-to-structural-grade
gate chemistry-derived.

### C2. Non-equilibrium point-defect (vacancy/interstitial) intrinsic stress

**The gap.** The review rounds cover intrinsic stress from **crystallite coalescence**
(V4 `coalescence_stress.py`) and from **hydrogen** (`internal_stress.py`). A third,
distinct electrocrystallization source of intrinsic stress is missing: the
**non-equilibrium supersaturation of point defects (excess vacancies, trapped
interstitials)** injected at high overpotential, which coalesce or get trapped in the
growing deposit and contribute to its residual stress — often in the opposite sense to
coalescence stress and strongly potential-dependent.

**The physics.** At high deposition overpotential, the Faradaic flux deposits atoms
faster than they can reach equilibrium sites, injecting a vacancy/interstitial
supersaturation. As the deposit thickens these defects anneal/coalesce, generating a
thickness- and overpotential-dependent stress component:

```
σ_pt(η, t) ≈ g(η) · [1 − exp(−t/τ_anneal)]
```

This is the mechanism behind the well-known rise in tensile stress of Fe/Ni deposits at
high current density, and it feeds directly into the peel/harvest and crack-initiation
questions `internal_stress.py` and `adhesion_peel.py` already answer.

**Concrete addition.** `models/point_defect_stress.py`:
- `point_defect_stress(eta, T, t_deposition, theta_additive)` → intrinsic stress term,
- add as a third source term in `internal_stress.py` alongside coalescence + H,
- expose the anneal time constant as a calibration target for bent-strip coupons
  (`internal_stress.py` protocol).

**Gate impact.** Gives the peel/harvest and crack gates a physically-motivated
current-density dependence instead of a single screening σ; explains why higher j (which
helps FE and productivity) simultaneously raises peel risk.

### C3. Adatom surface-diffusion / kink-step kinetic incorporation barrier

**The gap.** The charge-transfer kinetics in `kinetics.py`/`bdd_kinetics.py` end at the
surface; `pulse.py`'s off-time "healing" and `mullins_sekerka.py`'s smoothing both
implicitly rely on **surface diffusion**, but there is no explicit adatom/step model.
This leaves an unallocated part of the cathodic overpotential (the "crystallization
overpotential") and makes the off-time smoothing in pulse plating unquantified.

**The physics.** Crystal growth is limited by (1) charge transfer, (2) transport, and
(3) **surface diffusion of adatoms to step/kink sites + kink incorporation**. The
third adds a crystallization overpotential

```
η_cryst ≈ (RT/F) · ln(1 + j / (j_0,surf))   with  j_0,surf ∝ D_s · c_adatom · ρ_kink
```

where `D_s` is the surface diffusivity (temperature- and additive-suppressed) and `ρ_kink`
is the kink/step density. This is the physical knob that additive levelers turn, and it
sets the off-time surface-diffusion length that `pulse.py` and `mullins_sekerka.py`
consume.

**Concrete addition.** `models/adatom_kinetics.py`:
- `surface_diffusivity(T, theta_additive)` ,
- `crystallization_overpotential(j, D_s, c_adatom, rho_kink)` ,
- provide `off_time_healing_length(t_off, D_s)` to `pulse.py` and `mullins_sekerka.py`
  so the smoothing term becomes quantitative instead of assumed.

**Gate impact.** Fills the "third overpotential" in the voltage decomposition
(`voltage_decomposition.py`) and turns pulse-reverse healing from a narrative into a
computable quantity — directly relevant to the energy (kWh/t) kill criterion.

---

## Theme D — Fluid & interface physics in the cell

### D1. Marangoni & electrocapillary surface flows

**The gap.** `solutal_convection.py` (V3) and `mhd_convection.py` capture buoyancy and
Lorentz stirring. Missing are **surface-tension-driven flows**: thermocapillary
(temperature gradient along the interface), solutocapillary (surfactant/additive
concentration gradient), and **electrocapillary** (potential-dependent surface tension
via the Lippmann equation). These stir the boundary layer, thin `δ`, and alter local
current distribution — especially in the additive-laden, non-isothermal industrial cell
that the reviews' temperature-gradient items (V4 §5, V4 §6) set up.

**The chemistry.** Surface tension of the metal/electrolyte interface depends on
potential (electrocapillary maximum ~ pzc), temperature, and additive coverage. A
gradient in any of these drives a Marangoni shear stress at the interface:

```
τ_Marangoni = ∇γ(φ_M, T, Γ_org, Γ_cl)
```

**Concrete addition.** `models/marangoni.py`:
- `surface_tension_gradient(eta, T_gradient, theta_additive, theta_cl, I)` via the
  Lippmann + Gibbs-adsorption relations,
- `marangoni_shear(∇γ)` → effective velocity → corrected `δ_eff` into
  `diffusion_layer_1d.py` / `boundary_layer.py`,
- flag when Marangoni competes with buoyancy (Richardson-type ratio).

**Gate impact.** Adds a mechanism that can either *help* (stirring → higher j_lim → FE)
or *hurt* (non-uniform current → edge/dendrite risk) and is a natural outcome of the
additive + temperature-gradient operating point the program targets.

### D2. Terminal / edge-effect current crowding → thickness non-uniformity

**The gap.** `hull_cell.py` handles a 1-D primary current distribution; `gas_holdup.py`
handles axial redistribution; the rotating-cylinder distribution was raised (V2 §4.1).
Missing is the **terminal (edge) effect**: the primary current crowds at deposit
boundaries, so the foil/flake grows thicker at its edges. That edge thickening is what
drives **edge cracking on cold rolling** — a central rollability gate in
`oxygen_in_iron.py` — and non-uniform composition.

**The physics.** Near a deposit edge, the current density rises toward
`j_edge/j_center` from a few % to ~×2 depending on geometry, electrode/deposit
conductivity, and shield placement. Edge-thick deposits crack on rolling and carry more
co-deposited O/H at the edges (higher local j). A simple secondary-current edge
correction turns "is the foil rollable?" into a statement about the *edges*, not just
the center.

**Concrete addition.** `models/edge_effect.py`:
- `edge_current_ratio(geometry, kappa_elec/kappa_sol, thickness)` (closed-form /
  secondary-distribution approximation),
- `thickness_profile(x)` → feed max/min thickness and edge-O/H into the rollability gate,
- recommend edge masking/shield sizing.

**Gate impact.** Directly tightens the cold-roll ceiling in `oxygen_in_iron.py` and the
thickness-uniformity spec in `deposit_metrology.py` — cheap, high-value for the
foil-to-sheet path.

---

## Theme E — Feedstock & recycled-material steel chemistry

### E1. Tramp-element surface hot-shortness (Cu, Sn) on rolling — recycled-feed ceiling

**The gap.** `impurity_codeposition.py`/`purification.py` handle Cu/Sn as *bath trace
impurities*, but nothing models their **effect in the finished steel**, which is the
real ceiling for recycled/waste feedstocks. The program is feedstock-first and lists
waste streams (pickle liquor, steel-mill dust, scrap-adjacent feeds) as the beachhead —
exactly the feeds that carry Cu/Sn/Ni.

**The physics.** On heating for hot rolling, tramp Cu (with Sn) segregates to the
steel/scale interface; above the Cu–Fe eutectic (~1094 °C) the Cu-rich phase is *liquid*
and wets/penetrates austenite grain boundaries → **surface hot-shortness**, i.e.
alligatoring/edge cracking in the rolled sheet. Sn lowers the effective melting
temperature and worsens it. This is a hard quality ceiling for scrap-derived steel that
the Option-B product path must respect.

**Concrete addition.** `models/hot_shortness.py`:
- `cu_segregation_index(C_cu, C_sn, T_roll)` (surface enrichment during scale formation),
- `hot_shortness_risk(...)` → rolling-temperature ceiling and allowable residual Cu+Sn
  for a given grade,
- wire into `thermomechanical.py`'s rolling gate and into `feedstock_logistics.py` as a
  feed-quality constraint.

**Gate impact.** Makes "which recycled feedstock can actually make which steel grade?"
a quantitative filter, closing the loop between the supply-chain model
(`supply_chain.py`) and the physical product gates.

### E2. Ligand/chelant complexation to widen the deposition pH window

**The gap.** The research report lists chelating ligands (citrate, glycine, gluconate,
EDTA) as the strategy for neutral/alkaline operation, but there is **no module** for
Fe(II)-ligand solution speciation and its effect on the deposition potential and
Fe(OH)₂ window. This is the chemistry that would let the program operate outside the
acidic HER-dominated regime.

**The chemistry.** A ligand L raises the pH at which Fe²⁺ precipitates by lowering free
`a_Fe²⁺` via complexes `FeL²⁺`, `FeL₂`, protonated `FeHL`, etc.:

```
Fe²⁺ + nL ⇌ FeL_n^(2−)          (log β: glycine ~3–4, citrate ~4–5, EDTA ~14)
pH_ppt(ligand)  >  pH_ppt(no ligand)   (keeps Fe soluble to pH 7–9)
```

But complexation also shifts `E_eq(Fe²⁺/Fe)` negative (raises cell voltage) and changes
the interfacial pH/HER balance. A speciation model (reusing `speciation.py`/`pitzer.py`
machinery) turns the "chelant or not" choice into a computed trade of FE, voltage, and
morphology rather than a recipe input.

**Concrete addition.** `models/fe_ligand_speciation.py`:
- `ligand_speciation(pH, T, C_Fe, C_ligand)` → free a_Fe²⁺ and pH_ppt (from stability
  constants + protonation, temperature-corrected),
- `shifted_deposition_potential(...)` → ΔE_eq and required voltage,
- feed free a_Fe²⁺ into `diffusion_layer_1d.py`/`kinetics.py` and the new pH-ppt into
  the precipitation sink.

**Gate impact.** Opens a model-based path to non-acidic deposition, which — per the
repo's own thermodynamics (§5.1) — is where the Fe/HER margin narrows and where
morphology control becomes the binding constraint; also de-risks the "boric-free,
ligand-complexed" bath variants.

---

## Priority summary

| # | New module | Added physical/chemical reality | Primary decision-metric impact | Beyond rounds 1–4 | Status |
|---|---|---|---|---|---|
| B1 | `recombination_poison.py` | S/As/Sb/Se/Te/P control absorbed-H (replaces fixed `absorption_fraction`) | H content, bakeout time, feedstock purity spec | new | **implemented** |
| A1 | `carbon_electrodeposition.py` | Co-reduction of CO₂/formate/CO → interstitial C; one-step steel | Deposit C wt%, steel grade in-cell | new | **implemented** |
| E1 | `hot_shortness.py` | Tramp Cu/Sn surface hot-shortness ceiling for recycled feed | Recycled-feed → grade filter | new | **implemented** |
| B2 | `melt_hydrogen.py` | H → white-spot/flake risk in the Option-A melt-shop product | Feedstock H product spec | new | **implemented** |
| B3 | `bubble_engulfment.py` | H₂ bubble capture → porosity/pinholes/blisters | Deposit density & foil quality | new | **implemented** |
| C1 | `nucleation_grain.py` | Nucleation density → grain size → Hall–Petch (grain as output) | YS/grade chemistry-derived | new | **implemented** |
| C2 | `point_defect_stress.py` | Non-equilibrium vacancy/interstitial intrinsic stress | Peel/harvest & crack gates | new | **implemented** |
| D1 | `marangoni.py` | Thermocapillary/solutocapillary/electrocapillary stirring | δ_eff, current distribution, FE | new | **implemented** |
| E2 | `fe_ligand_speciation.py` | Fe-ligand complexation widens pH window | Non-acidic operation, FE/morphology | new | **implemented** |
| C3 | `adatom_kinetics.py` | Adatom surface-diffusion / kink incorporation (crystallization η) | Voltage decomposition, pulse healing | new | **implemented** |
| D2 | `edge_effect.py` | Terminal/edge current crowding → edge thickness/O/H | Cold-roll ceiling, metrology | new | **implemented** |
| A2 | `as_deposited_grade.py` | As-deposited Fe–C → phase fractions + grade (coupling seam) | Option-B structural-grade verdict | integration | **implemented** |

## Implementation status

All twelve Round-5 modules are **implemented on this branch**, each as a
standalone `models/*.py` module with a `main()` CLI, a `tests/test_*.py` suite,
a `pyproject.toml` entry point, and a `SCREENING_FLAG = "unvalidated (L1)"`
header:

* `models/recombination_poison.py` (+`aq-steel-recombination-poison`)
* `models/carbon_electrodeposition.py` (+`aq-steel-carbon-electrodeposition`)
* `models/melt_hydrogen.py` (+`aq-steel-melt-hydrogen`)
* `models/nucleation_grain.py` (+`aq-steel-nucleation-grain`)
* `models/bubble_engulfment.py` (+`aq-steel-bubble-engulfment`)
* `models/hot_shortness.py` (+`aq-steel-hot-shortness`)
* `models/point_defect_stress.py` (+`aq-steel-point-defect-stress`)
* `models/marangoni.py` (+`aq-steel-marangoni`)
* `models/fe_ligand_speciation.py` (+`aq-steel-ligand-speciation`)
* `models/adatom_kinetics.py` (+`aq-steel-adatom-kinetics`)
* `models/edge_effect.py` (+`aq-steel-edge-effect`)
* `models/as_deposited_grade.py` (+`aq-steel-as-deposited-grade`)

### Wiring into the existing pipeline

Several modules are wired into the existing (previously hard-coded) model path,
all opt-in and backwards compatible:

* **B1 → `hydrogen_embrittlement.py`**: `hydrogen_uptake_from_electrolysis` now
  accepts `poison_concentrations_M` (and `cathodic_overpotential_V`); when
  supplied, absorbed-H and `C_H_diffusible_ppm` are scaled by the
  recombination-poison promotion factor and a `recombination_poison` sub-dict is
  added. Unchanged when no poisons are passed.
* **C1 → `mechanical_properties.py`**: `MechanicalPropertiesModel.predict` gains
  `use_nucleation_grain_model` / `cathodic_overpotential_V` /
  `additive_coverage_fraction` to source grain size from the nucleation model
  instead of the empirical current-density correlation (default off).
* **C2 → `internal_stress.py`**: `deposit_stress_from_conditions` gains
  `include_point_defect_stress` to add the point-defect intrinsic term to the
  stress decomposition (opt-in).
* **D2 → `oxygen_in_iron.py`**: `OxygenInIronModel.predict` gains
  `include_edge_effect` to evaluate the cold-roll gate against the edge O
  loading (the binding edge constraint) instead of the center value (opt-in).

The remaining modules (E1, D1, E2, C3, A2, and the A1 carbon channel) are
standalone and ready for analogous wiring into `thermomechanical.py` (rolling
gate), `boundary_layer.py` (δ_eff), `speciation.py` (pH window),
`voltage_decomposition.py` (crystallization η) and `steel_grade.py`
respectively — kept as separate follow-up to avoid over-editing large existing
modules in one pass.

## What this buys for the program's stated decisions

- **Option A (melt-shop feedstock)** is tightened by B2 (H spec for the buyer), B1
  (feedstock S → H), and E1 (tramp-element ceiling on what recycled feed is even
  eligible).
- **Option B (structural steel)** is the direct beneficiary of A1/A2 (carbon in-cell),
  C1/C2/C3 (grain size, stress, overpotential from the recipe), B3 (porosity), and
  E1/E2 (feed + pH chemistry).
- The unifying theme is that the four review rounds largely completed the *cell-level*
  physics; round 5 is about the **deposit-to-product** physics and the **feedstock-to-
  steel** chemistry that the program's own roadmap (`docs/NEXT_STEPS.md`,
  `docs/RESEARCH_PROGRAM.md`) names as the next gates.

Like every module in this repo, each item above should ship with a `SCREENING_FLAG`
("unvalidated (L1)") and a named literature anchor (a short `references/anchors.md`
row per claim) so the numbers are checkable in seconds, per the repo's established
convention.

— *Round 5 chemistry & physics review, August 2026.*
