# Chemistry & Physics Improvements — Round 6 (The Cell-to-Steel Chain)

> **Scope**: physical & chemical mechanisms that (a) are **not covered** by the
> five prior review rounds (`CHEM_PHYS_REVIEW.md` and
> `docs/CHEM_PHYS_IMPROVEMENTS_V2..V5.md`) and (b) were **audit-checked against
> the current `models/` tree** before writing (see the overlap audit at the end).
>
> Rounds 1–5 progressively closed the **cell** (transport, kinetics, surface state,
> membranes, stacks, pulsing, gas, thermal) and the **deposit** (nucleation, grain,
> stress, hydrogen, morphology, grade routing). What is still essentially
> unmodelled is (1) **everything that happens to the iron after it leaves the
> cathode and before it becomes steel** — corrosion at open circuit, oxidation,
> rinsing, densification, and the melt-shop remelt balance that decides whether
> the program's stated primary product ("a qualified iron feedstock for a melt
> shop") is actually buyable; (2) two **whole chemistries** the platform already
> implies but never computes — the homogeneous chlorine chain on the chloride
> route and the aqueous Fenton radical chain in *any* aerated Fe²⁺ bath; (3) the
> **reductive side of the titanium drum** (hydriding), which is the foil branch's
> unpriced service-life mechanism; and (4) the **measurement-chain physics**
> (junction potentials, deposit aging) that the Q3/RDE calibration campaign will
> silently inherit if they are not bounded first.
>
> Items **§1.1** (`models/deposit_corrosion.py`), **§1.2**
> (`models/product_oxidation.py`), **§1.3** (`models/rinse_carryover.py`) and
> **§1.5** (`models/melt_balance.py`) have since been implemented at screening
> (L1) scope with named anchors, per repo convention; everything else below
> remains a proposal.

---

## 1. The post-harvest gap: between the cell and the steel

Round 5 took the deposit *from the bath* up to the melt-shop *question* (H white
spots, tramp hot-shortness). But the physical chain **harvest → dry → store →
ship → charge → melt** is where electrowon iron actually succeeds or fails, and
it is the largest contiguous unmodelled region in the repository. The melt-shop
buyer does not buy FE or V_cell; they buy a certificate: **Fe units in, yield
out, no surprises**. Today the repo cannot even in principle generate one,
because five of its seven line items do not exist as physics.

### 1.1 Deposit corrosion at open circuit and ferric etch — the silent leak in the iron ledger

> **Status: implemented (L1)** — `models/deposit_corrosion.py` (`python -m models.deposit_corrosion`, CLI `aq-steel-deposit-corrosion`), wired into `run_record.py` (predicted idle ledger terms) and `closed_loop.py` (`campaign_idle_accounting`).

**The gap.** `run_record.py` demands a closed charge/mass/electrolyte balance;
NEXT_STEPS §1 makes the iron ledger gate evidence. But there is no model for
the deposit losing mass **without any current through the cell**:

1. **Open-circuit corrosion / redissolution.** Freshly deposited, nanocrystalline,
   hydrogen-charged iron is at −0.4 V or below vs SHE sitting in a pH 2–4 bath.
   At open circuit it runs a mixed-potential corrosion cell on itself,
   `Fe + 2H⁺ → Fe²⁺ + H₂`, at corrosion current densities `j_corr` that for
   sulfuric-pickle-grade iron in aerated warm acid are of order
   1–50 µA/cm² (screening). Over an 8-hour idle or a weekend shutdown this is
   0.1–2 µm of deposit and an unbooked Fe term — small per night, systematic
   with the number of idle cycles, and exactly the kind of term that shows up
   as "the ledger never quite closes." `INDEPENDENT_SHUTDOWN.md` covers the
   *safety* of shutdown; `kinetics.py`'s BV dissolution branch covers the
   deposit only when it is polarised anodic of `E_eq`. Neither covers the
   *unpowered* state.
2. **Ferric etch.** Membrane crossover Fe³⁺ (and PRE-produced Fe³⁺ from the
   Round-1 item 2.5, on the kanban) chemically attacks the metal:
   `2Fe³⁺ + Fe → 3Fe²⁺`. This is the well-documented current-efficiency killer
   in the classic USBM chloride electrowinning flowsheets, and it leaks
   *cathode iron back into the bath without electrons*. `fe3_shuttle.py`
   models the *electrochemical* shuttle (`Fe³⁺ + e⁻ → Fe²⁺` at the cathode);
   the homogeneous/mixed-potential etch of the deposit surface is a distinct,
   second channel with a different rate law (half-order in a_Fe³⁺, strong T
   dependence, and acid-catalysed).

**Why it matters for steel.** Both channels turn *product* into *recirculating
solute*: they degrade apparent FE (under-estimated by gravimetry), distort the
iron ledger in the direction of false pessimism, and — during PRE — set how
much reverse-pulse "smoothing" really costs in metal.

**Concrete addition.** `models/deposit_corrosion.py`:
```
corrosion_current(a_H, a_O2, a_Fe3, T, theta_additive)   → j_corr (mixed potential)
ferric_etch_flux(a_Fe3_surf, a_H, T)                     → mol Fe/(m²·s)
mass_loss_over_idle(t_idle, ...)                         → µm, g, ledger entry
```
Wire into `run_record.py` as a *predicted idle-loss term* in the iron ledger
(so residuals are tested against it, not absorbed by "uncertainty"), and into
`closed_loop.py` so campaign-level Fe inventory sees nights/weekends. Anchor:
classic iron-in-acid mixed-potential data (Stern 1955; Kelly 1965) and the
chloride-EW ferric-etch literature (USBM RI series).

### 1.2 Product oxidation, drying, and pyrophoricity — the post-harvest oxygen budget

> **Status: implemented (L1)** — `models/product_oxidation.py`
> (`python -m models.product_oxidation`, CLI `aq-steel-product-oxidation`),
> feeding `melt_balance.py` post-harvest O pickup live.

**The gap.** `oxygen_in_iron.py` and `bubble_engulfment.py` bound the oxygen the
deposit is *born* with. Nothing bounds the oxygen it *gains* between the
scraper and the furnace — and for fine electrowon powder this is not a slow
aging term, it is a **safety and yield** term. Column powder from the rotating
cylinder (the architecture screen's only 5×-clearing option is *powder only*)
is high-area, hydrogen-bearing, freshly washed iron. On air it passivates
(Mott–Cabrera logarithmic film, ~2–4 nm at RT), and if dried hot and fast it
self-heats: the rate of exothermic oxidation scales with surface area, the
heat loss only with bulk — the classical **Semenov thermal-runaway criterion**.
Pyrophoric iron powder events are routine in the PM industry.

**The physics.**
```
dO/dt = oxidation_flux(S_A, T, p_O2, film)               (log → parabolic crossover)
self-heating:  C_p ρ dT/dt = S_A · Q_ox · w_ox(T) − UA(T − T_amb)
runaway if  ∂(generation)/∂T ≥ ∂(loss)/∂T at T_crit        (Semenov)
passivation window:  controlled low-p_O2 / low-humidity / T ramp → fixed film
```

**Why it matters for steel.** (a) Harvested-product O lands directly in the
melt balance (§1.5): every 1 wt% O in the charge is ~4.5 wt% FeO that must be
boiled out with carbon. (b) The *passivation recipe* becomes a product spec:
"flake, passivated, ≤1.5 wt% O, non-pyrophoric" is a buyable article; wet
reactive powder is not. (c) It sets the dryer/inert-bin line items that
`dark_mill.py` currently costs only as CAPEX lines with no physics.

**Concrete addition.** `models/product_oxidation.py` (oxide-film growth +
thermal runaway + passivation protocol → product O(wt%) and T_crit vs surface
area), with its feed from `cell_architecture.py` (product form/area) and its
output into §1.5. Anchors: Mott–Cabrera oxidation theory; Semenov (1928);
NFPA/industrial PM handling practice for iron powder.

### 1.3 Rinse chemistry — electrolyte carryover becomes melt-shop sulfur

> **Status: implemented (L1)** — `models/rinse_carryover.py`
> (`python -m models.rinse_carryover`, CLI `aq-steel-rinse-carryover`),
> feeding `melt_balance.py` charge S live.  *Implementation erratum:* the
> cascade formula printed below, `c_n = c_0·(r/(1+r))^n`, uses
> drag-out:water as "r"; with the standard tankhouse definition (r =
> water:drag-out) the exact counter-current result is
> `c_n/c_0 = 1/(1 + r + r² + … + r^n)` — implemented as the latter.

**The gap.** `hot_shortness.py` (Round 5, E1) caps S (via Cu/Sn synergy) in the
*finished steel*, and `bath_impurity_codeposition.py` puts bath S into the
deposit. The third sulfur channel is never modelled at all: **adherent bath
liquor**. A flake web pulled from a sulfate bath carries a Landau–Levich
liquid film; a powder column carries interstitial liquor. Dried, that film is
Na₂SO₄ + FeSO₄ + boronate; charged into a melt, the sulfate decomposes and the
sulfur enters the metal. The same residual film poisons briquetting (§1.4,
galvanic staining) and is the reason tankhouse copper/nickel products are
rinsed in cascades.

**The physics.** Withdrawal film thickness `h_f ~ 0.94 · l_c · Ca^(2/3)`
(Landau–Levich, capillary number from web speed and rheology — `bath_rheology.py`
exists); then a counter-current rinse train with
```
c_n = c_0 · (r / (1 + r))^n          r = rinse ratio, n = stages
carryover_S(kg S per t Fe) = m_liquor · c_SO4 · (M_S/M_SO4)  →  ppm in charge
```

**Why it matters for steel.** It turns "how much rinsing?" from a plumbing
decision into a **steel-grade decision**: for LOW_SULFUR_S_MAX = 0.020 (already
a constant in `bath_impurity_codeposition.py`) the model tells you the stage
count and rinse ratio that keep charge S under the split — *including* the
Round-5 E1 hot-shortness interaction — and it prices the water and effluent
and the drag-OUT return loop (rinse #1 returns to the tank, preserving the
water balance that `chemical_osmosis.py` already tracks).

**Concrete addition.** `models/rinse_carryover.py` (film + rinse cascade →
charge-borne S/Na/B/P budget per tonne), feeding §1.5 and `deposit_metrology.py`
(conductivity endpoint acceptance on the final rinse). Anchor: Landau–Levich
(1942); standard tankhouse rinse-ratio practice.

### 1.4 Densification — briquetting is the product-form gate

**The gap.** The architecture screen reports product *form* (powder / flake /
foil) but never the last unit operation of Option A: **getting it shippable**.
Melt shops will not pneumatically fine-powder an EAF roof; the product must be
briquetted or pelletised to a size and strength spec (low fines generation in
transit — fines are the pyrophoric and fume fractions, §1.2/§1.5). Iron-flake
compaction is mechanical metallurgy that the repo already owns the inputs for
(as-deposited σ_y, HV, elongation from `mechanical_properties.py`) but never
runs.

**The physics.** Heckel compaction:
```
ln(1 / (1 − D_rel)) = K · P + A        1/K ≈ 3·σ_y  (ranking yield pressure)
green strength  σ_g = σ_g,0 · exp(b · D_rel)
springback + ejection force, die wall friction
O/H passivation fixes before hot dense-state sintering (optional 600–800 °C)
```
Surface-area-dependent S/A also drives the ferromagnetic agglomeration of
powder in hoppers (ratholing, bridging) — a handling reliability term, cheaply
bounded with a Jenike-style flow-factor screening.

**Why it matters for steel.** It defines the product *grade for shipment*
(density, crush strength, fines), the briquetting press kWh/t (OPEX), and —
via passivation-first vs sinter-first — the residual O that §1.2 and §1.5
exchange. It is also the point where a deliberately *friable* deposit (an
inverted design goal per RESEARCH_PROGRAM Option A) becomes a virtue, not a
defect: friable flake compacts at lower K·P.

**Concrete addition.** `models/briquetting.py`: Heckel constants from deposit
σ_y (screening, literature-anchored for electrolytic iron powder), springback,
green-strength vs relative density, fines fraction, press energy per t; output
a shippable-product spec block into `feedstock_logistics.py`/`dark_mill.py`.
Anchors: Heckel (1961); PM iron-powder compaction data (e.g., Höganäs
electrolytic-powder grade literature).

### 1.5 The melt-shop verdict — a remelt mass/energy balance for electrowon charge

> **Status: implemented (L1)** — `models/melt_balance.py` (`python -m models.melt_balance`, CLI `aq-steel-melt-balance`).

**The gap.** `melt_hydrogen.py` (Round 5, B2) answers the buyer's hydrogen
question. The buyer's other three questions have **no model anywhere**: (a)
process physics *your flake gets dropped into an EAF/induction charge along
with scrap/DRI* — what is the iron yield, (b) how much CO boil does its O
content drive, (c) how much slag and fume, (d) can it be late-added for trim
(which is the premium use for clean fine feed)? This is the keystone of the
feedstock business case — "what does a tonne of your product do in my furnace?"
— and the repository cannot compute it.

**The physics (a standard charge balance, specialised to this product):**
```
Charge O (from §1.1–1.3 + as-deposited)  +  charge C  →  CO boil (mol, kJ exotherm)
Fe in oxides (FeO/FeOOH/film) → recovered by Boil C vs lost to slag FeO (basicity-dependent)
residual S (§1.3) → lime requirement to fix as CaS at given basicity; constrains slag volume
fines fraction (§1.4) + charging practice → off-gas dust loss (1–5% screening)
H (already modelled) → rimming action/degas requirement
late-addition window: density & melt-in rate of briquette vs flake vs powder
Energy credit/debit: endothermic melt-in, exothermic C–O, vs DRI/scrap baseline
```

**Why it matters for steel.** It converts every upstream unit into the buyer's
three numbers — **Fe yield %, boil behaviour, slag/fume make** — and therefore
prices the product against HBI/DRI/pig iron in `dark_mill.py`/`feedstock_logistics.py`
with physics instead of a flat $/t. It is the **completion of Option A**: where
the program currently says "qualified feedstock," this is the qualification
ledger. It should consume `oxygen_in_iron`, `rinse_carryover`,
`product_oxidation`, `melt_hydrogen`, `briquetting` and `hot_shortness`
outputs and emit one verdict table per run.

**Concrete addition.** `models/melt_balance.py` + `aq-steel-melt-balance` CLI:
charge-to-tap mass/energy balance for EAF and induction routes; sensitivity of
yield to charge O/S/fines; comparison vs DRI and scrap baselines. Anchors:
standard IISI/Turkdogan steelmaking balances; EAF dust-generation literature.

---

## 2. Chlorine chemistry of the chloride route (beyond CER selectivity)

### 2.1 Homogeneous chlorine speciation: Cl₂ → HOCl → ClO₃⁻, and the oxidative back-attack

**The gap.** `anode.py` models the CER *electrode* reaction and its Nernst
potential; `fe_chloride_speciation.py` covers *catholyte* Fe–Cl pairing; the
Round-1 AWARE item (on the kanban) covers chloride-side transport/activity.
**Nobody owns the fate of the chlorine after it is made** — which, per a
century of chlor-alkali engineering, is the chemistry that decides whether a
concentrated-chloride divided cell ages in hours or years:

```
Cl₂ + H₂O ⇌ HOCl + H⁺ + Cl⁻          K ≈ 4×10⁻⁴ M² (25 °C) — partitions the Cl₂
HOCl ⇌ H⁺ + OCl⁻                      pK_a = 7.5
at 60–90 °C near pH > 3:  HOCl/OCl⁻ → ClO₃⁻ (chlorate, strongly T-accelerated)
at very high anode / on PbO₂:      ClO₃⁻ → ClO₄⁻ (perchlorate — slower)
headspace:  Cl₂ gas (Henry) — inventory against ccw dilution & scrubber load
```

Consequences, in order of how the program will meet them:

1. **Anolyte aging.** The anolyte is not "acidic chloride"; it is a chlorine/
   hypochlorite/chlorate chemical cocktail whose redox capacity drifts on a
   10–100 h scale and whose disposal is regulated.
2. **Oxidative back-attack.** Any HOCl/ClO₃⁻/dissolved Cl₂ that crosses the
   divider is a powerful oxidant arriving at a cathode pH 2: it chemically
   re-oxidises Fe²⁺ (feeding Round-1's ferric problem), **etches the deposit
   (§1.1, a second oxidant)**, and is a classical destroyer of organic
   brighteners (feeding `additive_aging.py` by a channel it does not have).
3. **Gas duty.** Off-gas chlorine from the anolyte (Henry) sizes scrubber,
   caustic quench and monitoring — a site/field-approval document the
   RESEARCH_PROGRAM redeployment story needs irrespective of FE.

**Why it matters for steel.** The AWARE route's entire promise is high FE in
concentrated chloride. Whether that FE survives 500 hours is decided mostly by
this chain: oxidant crossover kills CE on iron in chloride practice (the
USBM experience), and chlorate generation rates at 70–90 °C are *fast*.

**Concrete addition.** `models/chlorine_speciation.py`: equilibrium partition
(hydrolysis, dissociation, Henry) → steady-state oxidative-inventory; T-scaled
chlorate formation kinetics (Eigen–Kustin/Hine-type rates); divider cross-flux
coupling into `membrane_transport.py`; oxidative consumption terms exported to
`bath_startup.py`(Fe²⁺), `additive_aging.py`(organics), §1.1 (deposit etch);
off-gas chlorine duty for `h2_safety.py`-style documents. Anchors: chlor-alkali
standard texts; Eigen & Kustin (1962) hydrolysis kinetics; Adams/Tilak chlorate
reviews.

---

## 3. Radical chemistry: the Fenton bridge

### 3.1 The 2e⁻ ORR → H₂O₂ → OH• channel: membrane unzipping, radical additive aging

**The gap.** `dissolved_oxygen.py` (Round 4) carries ORR as a 4-electron water
reaction and models bulk Fe²⁺ autoxidation; `bath_startup.py` handles the
ascorbate side. Both *skip the intermediate* — and the intermediate is the
dangerous species. ORR on iron in acid proceeds in part through the 2-electron
pathway, `O₂ + 2H⁺ + 2e⁻ → H₂O₂`, and the homogeneous autoxidation chain also
transits peroxide. In an Fe²⁺ bath, peroxide does not accumulate; it decomposes:

```
Fe²⁺ + H₂O₂ → Fe³⁺ + OH⁻ + OH•     (Fenton, k ≈ 40–75 M⁻¹s⁻¹, 25 °C)
OH• + (anything organic) → minutes-timescale unselective oxidation (k ~ 10⁹ M⁻¹s⁻¹)
OH• + −CF₂− (ionomer)      → radical unzip, F⁻ release — the classic PFSA failure
```

Three concrete consequences, none representable today:

1. **Membrane/ionomer lifetime.** Iron-hydrolysis products plus peroxide are
   the worst-case PFSA aging cocktail (the fuel-cell community's canonical
   degradation mode). Radical flux at the cathode-facing membrane surface —
   not bulk averaged temperature — may set membrane life, complementing
   `membrane_thermal.py` (T-driven) and `membrane_fouling.py` (flux-driven).
2. **A second additive-aging path.** `additive_aging.py` has anodic, cathodic
   and hydrolysis channels. Fenton's OH• oxidation is often *faster than all
   three by orders of magnitude* for sulfur-bearing brighteners (saccharin,
   thiourea), and it has a diagnostic signature: consumption tracks dissolved
   O₂, not current. Misattributing it to electrochemical consumption would
   corrupt the dosing policy.
3. **Off-time passivation.** Peroxide arriving at a freshly deposited surface
   during idle/startup passivates it (chemical, not potential-driven), feeding
   the surface state that the next-onset nucleation sees — a subtle coupling to
   `nucleation_grain.py` and the first-seconds CV behaviour.

**Why it matters for steel.** Membrane lifetime and additive dosing are OPEX
terms in the 4,000 kWh/t / $-per-tonne economics, and radical attack is the
difference between "replace the cassette yearly" and "quarterly."

**Concrete addition.** `models/fenton_chemistry.py`: 2e-ORR partial current
(fraction η_2e × j_lim,ORR), peroxide steady state (generation vs Fenton
sink), OH• production flux, partitioned sinks (fraction to ionomer, fraction
to additives, fraction to Fe²⁺); exports aging-multipliers to
`additive_aging.py` and a first-order membrane-unzip rate (F⁻ release proxy)
to `closed_loop.py`. Anchors: Walling (1975), Buxton et al. (1988) radical
rate constants; PFSA radical-degradation literature (e.g., Collier et al.
2009; Kundu et al.).

---

## 4. The reductive side of the substrate

### 4.1 Titanium hydriding of the drum — the foil branch's unpriced life mechanism

**The gap.** `substrate_passivation.py` models the **oxidative** degradation of
the drum (TiO₂ parabolic growth over the campaign — "peel fails at hour 800,
not hour 8"). But a drum in service is a *cathode*. At the hydrogen-adjacent
potentials of an iron cathode in acid, titanium takes up hydrogen, and the
Ti–H system is unforgiving: solid-solution α-Ti (H in hcp) crosses a terminal
solubility (≲ 30–100 ppm at 60–90 °C, screening) into the δ-TiH₍₂₋ₓ₎ hydride,
with ~20–25 % atomic-volume expansion. The classic copper-foil-drum service
experience is that hydriding sets drum re-skinning intervals — and it is the
*reductive* counterpart that `substrate_passivation.py` explicitly anticipates
("anodic vs cathodic exposure") but does not compute.

**The physics.**
```
entry:   H_ads (θ_H from surface_state.py) → absorbed flux into Ti (Sieverts, T, E > E_hyd)
growth:  H diffusion in α-Ti (D_H^Ti, Arrhenius) → terminal solubility → δ-hydride front
stress:  volume expansion → compressive layer stress → TiO₂ scale cracking/spall
couple:  spalled/patchy oxide changes W_ad (interface) over campaign hours
         → adhesion_peel peel window drifts with time-on-drum, not just cycles
```

**Why it matters for steel.** The drum-and-strip route is the *only* continuous
coherent-foil path in the architecture screen, and its gating unknown is
already stated as "iron peels from a titanium drum" (coupon test pending).
Hydriding is the mechanism that makes that answer **time-dependent**: the
coupon can pass on day one and fail in month two. A drum-reskinning interval
and a re-passivation recipe are an OPEX line and a campaign-planning input,
and the model turns the adhesion/peel question from a constant into a curve
`G_c(interface, hours)`.

**Concrete addition.** `models/ti_hydriding.py` (Sieverts entry vs potential,
α→δ front kinetics, expansion-stress → scale-spall criterion, W_ad drift);
wire into `adhesion_peel.py` (time-dependent interface state) and
`closed_loop.py` (campaign drum-life ledger). Anchors: Ti–H Pourbaix/Patton
reviews; McQuillan & classic Ti hydriding kinetics; foil-machine drum-service
practice.

---

## 5. Measurement-chain physics — bound it *before* Q3/RDE calibration

The NEXT_STEPS work item #2 is a calibration campaign (Hull + divided-cell
FE + RDE volumetric H₂ — see docs/Q3_RDE_VOLUMETRIC_H2.md) whose entire output
is fitted parameters for `kinetics.py`/`diffusion_layer_1d.py`. Fits are only
as good as their systematic-error bounds, and two systematic error physics are
currently owned by nobody.

### 5.1 Liquid-junction and glass-electrode bias in concentrated iron brine

**The gap.** The Pitzer engine computes the thermodynamic quantity
`a_H = m_H · γ_H`. The bench will read **operational pH** — a glass electrode
referenced through a KCl bridge against a standard-buffer scale. Between those
two, in a 1.5 M FeSO₄ + supporting-salt bath of ionic strength 3–6 m, sit:

1. a **liquid-junction potential** at the bridge of 10–40 mV (Henderson/Planck;
   transport numbers of H⁺, Fe²⁺, Na⁺, HSO₄⁻/SO₄²⁻ all computable from the same
   mobilities `transport.py` already carries — which computes the *film*
   junction potential but not the *measurement* one);
2. **bridge clogging** by Fe(OH)₃ colloid (drift on a timescale of days — a
   recognisable, timestampable signature);
3. the Nernst slope itself moving with T (66.1 mV/pH at 60 °C) — pH meters
   correct this only against *buffer* pH(T) tables, which do not exist for
   this matrix;
4. the single-ion-activity ambiguity: an operational pH is not
   −log m_H·γ_H(Pitzer convention) — at I ~ 4 the offset is ~0.2–0.5 units,
   systematic, and feeds *directly* into the surface-pH boundary condition of
   the FE engine via the fitted exchange currents.

**Why it matters for steel.** The FE kill criterion is a pH-sensitive number at
the program's operating point (surface pH decides Fe(OH)₂ vs plating), and the
Q3/RDE campaign is explicitly "measure HER Tafel first" against a fitted
surface state. A 0.3-unit systematic pH offset propagates to the HER branch
fit as a biased i₀ and a biased pH-dependence exponent — invisible in R²,
fatal at extrapolation to 300 mA/cm². This is the cheapest $0-hardware module
in this review: it is software and a 2-point liquid-junction check
(HCl/LiCl concentration-cell protocol).

**Concrete addition.** `models/ph_metrology.py`: Henderson/Planck junction
potential vs bath composition (reusing ionic mobilities), bridge-aging model
(drift), operational-Pitzer conversion function `pH_op ↔ a_H(I, T)`, a
`MeasurementBias` record consumed by `calibration.py`/`kinetics_fit_pipeline.py`
so fitted kinetic parameters carry their metrology covariance. Anchors:
Henderson (1907), Bates (glass-electrode determination of pH), IUPAC pH
convention.

### 5.2 Room-temperature aging ("self-annealing") of the deposit between harvest and metrology

**The gap.** The QA artifact is "a weighed, characterised deposit" — weighed
*when*? Electrodeposits carry extreme stored energy (grain ~1 µm or finer,
H-charged, point-defect supersaturated per Round-5 C2), and electrodeposited
Cu/Ni are *demonstrated* to age at room temperature: hardness, resistivity,
residual stress and even grain size drift over hours–days (log-time kinetics,
activation-energy spectrum). For H-charged electrowon iron — where the
repository already models H egress through `hydrogen_trapping.py`, and where
`internal_stress.py` returns a *snapshot* — the time-stamp between harvest and
measurement is currently an uncontrolled variable common to every
characterisation record (`characterization.py`), the bent-strip protocol, and
the peel coupon.

**The physics.**
```
σ(t)/σ(0) = 1 − A·ln(1 + t/τ(σ, T, C_H))      (log-time recovery)
H diffusible: C_H(t) follows McNabb–Foster egress (already in hydrogen_trapping)
consequence: stress relaxation couples to H content *as it leaves* —
             the V2 stress-H coupling item is itself time-stamped
```

**Why it matters for steel.** Two runs measured at +4 h and +48 h disagree by
an amount that can exceed the inter-variable comparison the program cares
about (does saccharin raise or drop σ?). The metrology standard must either fix
the delay or correct for it; the model enables the correction and *forces* the
delay field into `run_record.py` (harvest→weigh and harvest→HV times are
charge-ledger-class metadata).

**Concrete addition.** `models/deposit_aging.py`: log-time recovery with T/H
dependence for σ, HV, resistivity; a recommended metrology-time standard; and
a mandatory `aging_hours` field check in `run_record` QA (fail-soft warning).
Anchors: self-annealing electrodeposited-Cu literature (Lingk/Gross; Stangl
et al.); classic recovery kinetics.

---

## 6. Platform-loop separations and site logistics

### 6.1 Bipolar-membrane salt splitting — turn the sulfate purge into the acid and base the plant already buys

**The gap.** The closed loop bleeds a sulfate-bearing purge (supporting salt +
impurity rejection, per `closed_loop.py`/`purification.py`), while the plant
simultaneously *imports* acid (pH/pickle duty) and base (neutralisation,
ascorbate handling, chlor-alkali-type scrubber, §2.1). **Bipolar-membrane
electrodialysis (BPMED)** splits Na₂SO₄ (or NaCl on the chloride route) back
into dilute H₂SO₄ and NaOH on site — the standard chlor-alkali-adjacent trick,
and a natural extension of the reconfigurable-platform thesis the
RESEARCH_PROGRAM argues for (process change at the cheapest layer; the purge
becomes a reagent generator). There is no membrane-separation physics in the
repo beyond the cell divider and the recirculation fouling model.

**The physics.** BPMED runs on **water dissociation kinetics at the
bipolar junction** — catalysed hydrolysis under the junction's ~10⁸–10⁹ V/m
field (Second Wien effect) :
```
H₂O → H⁺ + OH⁻   at the BPM junction  (i-limited, ~0.1–0.5 A/cm² practice)
Na₂SO₄ → H₂SO₄ (≈1 M) + NaOH (≈1–2 M)   at κ·ΔE ≈ 2.0–2.5 V per cell-unit
```
product purity limited by SO₄²⁻/Cl⁻ co-transport through the anion layers
and H⁺/OH⁻ leakage — i.e., by permselectivity, which is measurable.

**Why it matters for steel.** It deletes two reagent import lines, one waste
export line, and (with §3.1's membrane chemistry) is the only route that makes
multi-year closed-loop operation stoichiometrically clean — the water-chemistry
argument that a site permit will ask for. It is also *the* option-value move
consonant with the platform thesis: sulfate and chloride baths then differ by
a cartridge, not a plant.

**Concrete addition.** `models/salt_splitting.py`: junction water-dissociation
current density (Wien-effect screening law), stack sizing per purge flow,
co-ion leakage and product strengths, kWh per kg acid/base and per tonne Fe,
coupled to `closed_loop.py` purge flow and `technoeconomic.py` OPEX with an
economic-vs-export comparison. Anchors: BPMED engineering literature
(Simons; Mani/Chlanda reviews).

### 6.2 Freeze and crystallization windows of a deployable brine

**The gap.** The platform is meant to be redeployed — "drain and preserve the
chemistry." `site_layer.py` tracks *civil* frost depth; nothing tracks whether
the **process liquid** survives the site. A 1.5 M FeSO₄ brine starts laying
melanterite/ice near −2 °C (binary eutectic; mixed Na₂SO₄–FeSO₄–H₂SO₄ brines
similar to slightly lower), and freeze–thaw cycles fractionate the bath
(excluded brine concentrates, then re-dilutes non-uniformly): a chemistry
state the twin cannot currently represent. `fe_sulfate_solubility.py` covers
the **high-T** retrograde side (heat-exchanger scaling); the cold side is open
for exactly the same Pitzer machinery.

**Why it matters for steel.** Winter-idle and transport sit inside the
site-redeployment story; the preservation recipe ("drain to 1.2 M, add acid,
hold above −5 °C") is computable instead of folk knowledge, with the liquidus
from the same activity model the repo validates against anchors today.

**Concrete addition.** `models/brine_freezing.py`: Pitzer-based liquidus/solidus
for the working brine family; freeze-fraction vs T and composition; excluded-brine
composition during partial freeze; export freeze-risk windows to `site_layer.py`/
`dark_mill.py` and a preservation-recipe generator. Anchors: Pitzer liquidus
calculations; FeSO₄–H₂O phase diagram data.

### 6.3 Acid-mist aerosol from bubble bursting — the hygiene document nobody can write yet

**The gap.** RESEARCH_PROGRAM's practitioner quote names "acid mist and health
limits" as one of the ten failure modes that only a tankhouse veteran flags;
`gas_holdup.py` computes the *bubbles*; `h2_safety.py` the *gas-phase hazard*;
nobody models the **aerosol**. Every H₂ (and anode O₂/Cl₂) bubble that bursts
at the free surface ejects film drops (µm-class, numerous) and jet drops
(~0.1·d_b, a few per bubble). They carry bath — acid, Fe salts, borates —
airborne above the cell, where the regulated quantity is the inhaled
concentration (order-1 mg/m³ H₂SO₄ occupational limits), the inventory loss
(Fe is in the mist — the iron ledger again), and the equipment corrosion tax.

**The physics.**
```
jet-drop production per unit area ∝ bubble flux × f(d_b)     (Spiel/Blanchard distributions)
film-drop spectrum ~ d_b^(−3/2)-ish — the Round-1 surfactant/d_b uncertainty lands here too
airborne flux (g/(m²·h)) → enclosure concentration via room ACH (h2_safety machinery)
mitigation levers: hoods, demister pads, mesh suppressors, foam blankets, tank covers
```

**Why it matters for steel.** Participatory siting/field approval (the
RESEARCH_PROGRAM evidence-lifecycle) will request this number; it is also a
*real* design term for the deployable unit (enclosed + extraction or hooded
bench). Couples the repo's dominant flagged d_b uncertainty into a compliance
quantity.

**Concrete addition.** `models/acid_mist.py`: bubble-burst drop statistics from
`gas_holdup` d_b → airborne mass flux → enclosure concentration under ACH →
vs limits verdict + capture-sizing; Fe-mass mist term exported to the iron
ledger. Anchors: electrolyte-mist industrial hygiene literature (Cu/Zn
tankhouses); Spiel (1998) jet-drop distributions.

---

## 7. Steel metallurgy that starts in the bath

### 7.1 Nitrogen pickup from ammoniacal baths and interstitial strain aging — the downstream forming gate

**The gap.** Round 4's `ammonium_buffer.py` brings NH₄⁺/NH₃ into the bath
(buffer + ammine complexation, a boric-free path); `carbon_electrodeposition.py`
(Round 5, A1) pilots interstitial **carbon** into the deposit. The ammonium
route's free **interstitial nitrogen** channel is nobody's: ammoniacal Fe
plating can co-insert N from ammonia/amine decomposition at the cathode, and
C+N interstitials are the classical agents of **strain aging** (Cottrell
atmospheres → yield-point return → Lüders bands on the re-annealed sheet).
For Option B sheet this is the forming-surface killer: the JMAK/thermomechanical
chain ends at "recrystallised, AISI grade," but a sheet that Lüder-bands at
the customer's press or at a temper mill is not shippable deep-drawing stock.

**The physics.**
```
bath side:   NH3(aq) ↔ cathode-amine adsorbates → N_ads → [N] in deposit   (screening)
metal side:  Cottrell atmosphere t_form ∝ D_C,N(T)·(ρ_disl/C^2)  — return time at 20–150 °C
consequence: σ_y return Δσ ~ 20–60 MPa; Lüders strain % vs storage time and pre-strain
levers:      degas/bake (hydrogen_trapping machinery, T>120 °C — N diffuses slower than H),
             Scavenging: transform interstitials into TiN/-BN/AlN (requires co-deposited
             scavengers — links to E1/E2 feed impurities and the A1 carbon model), 
             temper-rolling (skin-pass) as the standard industrial fix
```

**Why it matters for steel.** This is the single ligand of the **Option B
sheet-quality chain** that no existing module can even flag today: if the
boric-free ammonium bath quietly raises [N], the "AISI 1018-like (weldable)"
verdict from `steel_grade.py`/`as_deposited_grade.py` survives strength
screening and fails the customer's press. Modelling it also closes the loop in
the useful direction: it converts "ammonium vs boric buffer" (Round 4) into a
**composition-dependent, not regulatory-only, trade-off.**

**Concrete addition.** `models/strain_aging.py`: [N] uptake screening from
ammoniacal chemistry, Cottrell-atmosphere return-time/YPA model, Lüders-strain
risk flag + temper-roll prescription; wire [N] into `as_deposited_grade.py`/
`thermomechanical.py` anneal verdicts, and the uptake side into
`ammonium_buffer.py`'s bath diagnostics. Anchors: Cottrell & Bilby (1949);
Baird's strain-aging reviews; ammoniacal iron plating literature.

---

## Shorter items (one-liners worth a card each)

- **Boron metallurgy routing.** `bath_impurity_codeposition.py` already inserts
  B into the deposit; the *grade* side ignores it. Trace B (5–30 ppm) is the
  cheapest hardenability multiplier in steel (austenite-GB segregation);
  >~100 ppm precipitates brittle Fe₂B eutectic films at GBs. A two-branch
  `steel_grade.py` rule turns a bath-contamination term into a potential
  feature. (Grossmann hardenability; B-segregation literature.)
- **Soft-magnetic foil/powder product option.** The drum's 25–50 µm foil form
  factor plus electrolytic purity is the eddy-current-optimal laminate for
  sub-kHz magnetic cores and electrolytic-iron PM parts — a priced product
  family, not a research curiosity. Core-loss ∝ (hysteresis + eddy) vs thickness,
  impurity, grain is a self-contained model; relevant if Option A economics
  stalls. (Bozorth; Steinmetz loss decomposition.)
- **Pickling-inhibitor carryover in pickle-liquor feeds.** Thiourea/amine
  inhibitors from the feedstock are recombination poisons (Round 5, B1) the
  program gets for free, plus a foam/emulsion term. One feed-fingerprint field
  in `feedstock_logistics.py` with a routing rule to `recombination_poison.py`.
- **Foam stability.** Surfactant + gas ⇒ stable foam on the tank (safety:
  carries mist §6.3; process: carryover). One closure on surface tension vs
  additive suite inside `acid_mist.py`-family physics.
- **Magnetic agglomeration of ferromagnetic powder in handling** (hopper
  flow-factor screening) — folded into §1.4 above but separable.

---

## Priority summary

| # | New module | Added reality | Primary decision-metric impact | Novelty check |
|---|---|---|---|---|
| 1.5 | `melt_balance.py` | Melt-shop verdict: yield / CO-boil / slag / fume from electrowon charge | **Completes Option A**; prices product vs DRI/scrap | no yield/slag/CO-boil model exists (`melt_hydrogen` is H-only) |
| 1.1 | `deposit_corrosion.py` | OC corrosion + ferric etch of the deposit, unpowered | Iron-ledger closure; honest FE; PRE cost | `kinetics` BV is polarised-only; `fe3_shuttle` is electrochemical |
| 3.1 | `fenton_chemistry.py` | H₂O₂/OH• chain: membrane unzip + radical additive aging | Membrane/reagent OPEX; campaign life | `dissolved_oxygen` is 4e-only; `additive_aging` lacks radicals |
| 2.1 | `chlorine_speciation.py` | Cl₂→HOCl→ClO₃⁻ chain + oxidative back-attack | **AWARE-route viability vs time**; scrubber duty | `anode.py` ends at CER; no homogeneous chlorine chain anywhere |
| 5.1 | `ph_metrology.py` | Junction/glass-electrode bias, operational↔Pitzer pH | Calibration integrity (Q3/RDE + Bayes fits) | only the *film* junction potential exists (`transport.py`) |
| 4.1 | `ti_hydriding.py` | Reductive side of the drum: Ti–H hydride, scale spall, G_c(t) | Foil-branch life; drum OPEX | `substrate_passivation` is oxidative-only |
| 1.2 | `product_oxidation.py` | Post-harvest oxidation, pyrophoricity, passivation | Product spec + safety + melt O | nothing post-harvest exists |
| 1.3 | `rinse_carryover.py` | Film + rinse cascade → charge S/Na/B | Sulfur grade gate; water/effluent | absent |
| 1.4 | `briquetting.py` | Heckel compaction, green strength, fines | Product form & logistics gate | only agglomeration failure-*mentions* exist |
| 6.1 | `salt_splitting.py` | BPMED purge → acid + base regeneration | Closed-loop stoichiometry & OPEX | no membrane separation beyond cell divider |
| 5.2 | `deposit_aging.py` | RT self-annealing σ/HV drift + metrology-time standard | QA reproducibility | absent |
| 6.3 | `acid_mist.py` | Bubble-burst aerosol, enclosure ACH → hygiene verdict | Site approval; iron-ledger mist term | named in RESEARCH_PROGRAM, never modelled |
| 6.2 | `brine_freezing.py` | Brine liquidus, freeze windows, preservation recipes | Redeployability in cold sites | `fe_sulfate_solubility` is high-T only |
| 7.1 | `strain_aging.py` | N uptake + Cottrell aging, Lüders-band gate | Option-B forming quality | N chemistry absent entirely |

---

## What this buys for the program's stated decisions

- **Option A (melt-shop feedstock)** stops being an assertion and becomes a
  computed certificate: §1.1–1.5 chain *deposit properties → product form →
  melt-shop yield*, and §6.1/§6.3 close the loop and the site dossier around
  it. The pair §1.5 + §1.2 is the difference between "we make iron units" and
  "we make a briquetted, passivated, yield-guaranteed feedstock."
- **Option B (foil/sheet)** gets its two unpriced life mechanisms (§4.1 drum,
  §7.1 forming surface) and the metrology backbone (§5.1–5.2) that the Q3/RDE
  + Bayes calibration is about to lean on.
- **The chloride route** gets its first *longevity* model (§2.1), without which
  AWARE-style FE numbers are a day-one property; **every aqueous route** gets
  the radical-aging channel (§3.1) that decides membrane and additive life.
- Common theme of Rounds 1–5: the physics *inside the cell*. Round 6's theme:
  **the iron has to survive the journey to the furnace** — and most of what can
  silently kill a melt-shop-feedstock business (or the chloride route's
  longevity) lives between harvest and tap.

## Overlap audit (checked against `models/` and prior rounds before writing)

- Post-harvest chain (oxidation/pyrophoricity/rinse/briquetting/melt balance):
  no hits for `pyrophor`, `briquet`/`Heckel`/`compaction` (only failure-mode
  *mentions* of agglomeration), `rins`, `CO boil`/`slag` in product models;
  `melt_hydrogen.py` is explicitly H-only.
- Chlorine chain: `anode.py` covers CER electrochemistry only; no
  `HOCl`/`chlorate`/`NaClO` strings in models; Round-1 AWARE item is
  catholyte-side transport/activity (distinct).
- Fenton/peroxide: zero hits for `fenton`/`peroxide`/`H2O2` in models/docs.
- Ti hydriding: zero hits for `TiH`/titanium hydride; `substrate_passivation.py`
  is the oxidative counterpart only.
- pH/junction metrology: `transport.py` computes the film diffusion potential;
  TIER0_ARCHAEOLOGY references a surface-pH electrode *experiment*; no
  measurement-bias model exists for the calibration chain.
- Deposit aging: no `self-anneal`/aging-of-deposit models.
- BPM/salt splitting: zero hits; freezing: `site_layer.py` frost depth is civil
  only; `fe_sulfate_solubility.py` is 56.7–90 °C side only.
- Acid mist: appears once as a practitioner-quote in RESEARCH_PROGRAM.md; no
  model.
- Nitrogen/strain aging: `ammonium_buffer.py` (Round 4) is bath chemistry only;
  no N-in-metal or Cottrell/Lüders physics anywhere.

All items above are **proposals (status: proposed) unless marked "Status:
implemented" in their section headers** (currently §1.1, §1.2, §1.3,
§1.5). Each
should ship with the repo's conventions: a standalone `models/*.py` with CLI,
`tests/test_*.py`, `pyproject.toml` entry point, `SCREENING_FLAG = "unvalidated
(L1)"` header, and one row per numeric claim in `references/anchors.md`.

— *Round 6 chemistry & physics review, August 2026.*
