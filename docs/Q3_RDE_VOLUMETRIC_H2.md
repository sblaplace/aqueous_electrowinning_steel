# Q3 — RDE + volumetric H₂: separate Fe from HER kinetics (measure HER Tafel FIRST)

**Status:** Level-0 synthetic method validation + bench protocol. **NOT gate evidence.**
**Board:** aqueous-steel · **Task:** t_bd314a95 · **Code:** `models/rde_volumetric_h2.py`,
`models/run_rde_volumetric_h2.py`, `tests/test_rde_volumetric_h2.py`
**Author/date:** 2026-08-07

---

## 1. Why this experiment, in one paragraph

`docs/SCREENING_SENSITIVITY_BUDGET.md` (#34) ranked **`her_tafel_V` — the HER
Tafel slope — as the single dominant remaining L0 uncertainty**: at the low
screening factor (0.105 V/decade) the reference-cell faradaic efficiency
collapses to 0.7599 against the 0.80 floor. So the parameter that most
threatens the go/no-go is the *HER* branch, not Fe. Q3 is therefore
deliberately **"measure HER Tafel first"**: pin the HER branch independently on
the Fe-free supporting electrolyte, then fit Fe against it, and confirm the
resulting split with volumetric hydrogen — a *measurement* closing the charge
ledger, not a second fit. This is the physical campaign named in NEXT_STEPS
#2.2/#2.3 ("RDE/Levich — separate Fe kinetics from HER on the actual bath and
cathode surface; the first three calibrate the cathode model").

A polarization curve of the Fe bath alone is underdetermined for the Fe/HER
split (two unknown Tafel slopes + two exchange currents vs a single total
current — the limitation `models/calibration.py` documents). The two-step +
independent-closure sequence removes the degeneracy by construction.

---

## 2. The three-part method

| Step | What | Why | Output |
|---|---|---|---|
| **0** | Reference/beaker + **B0 bath** and its **Fe-free baseline** | The task's prerequisite; the actual cathode surface & electrolyte | working electrochemical setup |
| **1** | **HER branch, Fe-free baseline** (same surface/pH/T/reference) | With no Fe²⁺ the total cathodic current **is** the HER branch → unambiguous `b_her`, `i0_her` (the #34 unknown) | `b_her`, `i0_her` |
| **2** | **Levich transport separation** on the Fe bath (RDE 400–2500 rpm) | Plateau `i_lim` vs `ω^½` → **D** and the Nernst film thickness `δ` that calibrates `diffusion_layer_1d` | `D`, `δ`, `i_lim(ω)` |
| **3** | **Fe branch fit with HER HELD FIXED** (from step 1) | Only 2 free params (`i0_fe`, `b_fe`) vs a full polarization curve → non-degenerate Fe/HER separation | `i0_fe`, `b_fe` |
| **4** | **Volumetric H₂ + gravimetric Fe** at 3–5 current densities | Independent charge-ledger closure `FE_HER(gas) + FE_Fe(mass) ≈ 1`; catches a wrong slope-fit split with a measurement | `FE_HER`, `FE_Fe`, closure residual |

### Step 1 is the entire point
Mark it clearly in the run sheets and metadata: this run is **Fe-free**, so the
only cathodic Faradaic process is hydrogen. Record it **before** the Fe bath so
the cathode surface state is fresh and HER is not entangled with Fe.

---

## 3. Apparatus & the Fe-free baseline

All equipment sourcing in `docs/EQUIPMENT_LIST.md` / `docs/SHOPPING_LIST.md`,
build in `docs/REFERENCE_CELL_PIPELINE.md`, bath in `docs/BATH_SPEC.md` &
`docs/FIRST_LAB_DAY.md`. The Q3 cell can be the **beaker / reference cell** from
the D1 reference-cell spec; it becomes an RDE cell by adding the rotating-disk
working electrode.

- **Working electrode:** the **actual cathode material** (per the reference
  cell), polished to a defined roughness, exposed area measured, in an RDE tip
  (or a static coupon for step 1 if HER transport is negligible).
- **Counter:** Pt or DSA anode, separated if the reference cell is divided.
- **Reference:** the cell's reference (Ag/AgCl or SHE-equivalent); convert to
  SHE with the documented `Ref_V` per run.
- **Fe bath (B0):** 1 M FeSO₄·7H₂O, boric acid, pH 2.0 ± 0.05, 50 °C
  (per FIRST_LAB_DAY; use the actual B0 recipe).
- **Fe-free baseline (B0⁻):** identical anion/T/pH/**surface prep** but **no
  FeSO₄** — sulfate supplied to match the sulphate activity (e.g. Na₂SO₄ or
  H₂SO₄) + boric acid, pH 2.0. The anion/pH/temperature must match so the HER
  branch measured here is the one that operates in the Fe bath.
- **Gas burette / manometric cell** for volumetric H₂ (inverted burette over
  the divided cathode compartment, or a calibrated manometer; record T, P and
  correct for water-vapour saturation at L1).
- **Potentiostat** for LSV/EIS and a **coulometer (Ah)** in series (per
  FIRST_LAB_DAY pre-flight).
- **Fe³⁺ discipline & H₂ safety:** inherit every rule from FIRST_LAB_DAY (§2
  Fe³⁺, §0 H₂ ventilation). H₂ is generated in every cathodic run; no sealed
  vessels near the powered cell.

---

## 4. Step-by-step protocol

### Pre-flight
- pH meter calibrated (4.00/7.00) same day; reference validated against a fresh
  electrode; Ah coulometer zeroed; scale tared.
- `experiments/data/` templates copied with run dates; every run manifest
  validates against `models/run_manifest.py`.

### Step 1 — HER first (Fe-free)
1. Fill the cell with the **Fe-free baseline (B0⁻)**, same T as the Fe run.
2. Insert the actual cathode RDE surface; inert-gas purge/de-air.
3. **EIS** near `E_eq(HER)` → `Rct` → implied exchange current (cross-check;
   `models/eis.exchange_current_from_rct`).
4. **LSV/RDE:** sweep cathodic from `E_eq(HER)` to ~`−0.9` V vs SHE at the
   rotation matrix (400–2500 rpm) and at least one slow scan rate; n≥3
   replicate sweeps.
5. Fit `b_her`, `i0_her` with
   `models.rde_volumetric_h2.fit_her_from_free_bath`.
6. **Gate:** HER Tafel fit `R² ≥ 0.98`, slope in 0.06–0.20 V/decade (iron-group
   HER), EIS-Rct consistent within stated tolerance.

### Step 2 — Levich transport (Fe bath)
1. Replace with **B0** (1 M Fe²⁺), same T/cathode.
2. RDE rotation matrix 400–2500 rpm; LSV cathodic to where Fe is clearly
   transport-limited.
3. Extract `D` and `δ` from the plateau Levich plot (reuse
   `models.rde_levich.analyze_rde_polarization` / `run_rde_levich`).
4. **Gate:** Levich `R² ≥ 0.995` on the plateau; recovered D within 25% of the
   Fe²⁺ literature anchor (7.2e-10 m²/s at 25 °C, Arrhenius-scaled).

### Step 3 — Fe with HER fixed
1. Feed the Fe-bath LSV across the rotation matrix and the **fixed** step-1 HER
   branch into `models.rde_volumetric_h2.fit_fe_given_her_on_rde`
   (or `fit_fe_kinetics_given_her` with per-row Levich limits).
2. Fit `i0_fe`, `b_fe`.
3. **Gate:** Fe Tafel `R² ≥ 0.98`, slope in 0.06–0.18 V/decade (single 2e⁻
   step), positive `i0_fe`.

### Step 4 — Volumetric H₂ + gravimetric closure
1. Run **galvanostatic** at 3–5 current densities spanning HER-lean → HER-rich
   (e.g. 30, 60, 100, 150, 250 mA/cm²), fixed time each.
2. Per run: record **applied charge (Ah)**, **H₂ volume (mL, T, P)**, **deposit
   Fe dry mass**.
3. Close the ledger with
   `models.rde_volumetric_h2.volumetric_h2_closure`:
   `FE_HER = 2F·n_H₂/Q`, `FE_Fe = (m·2F/M_Fe)/Q`, `closure = FE_HER + FE_Fe`.
4. Cross-check the fitted HER branch at the operating potential against the
   gas-derived HER charge (`her_branch_residual`).
5. **Gate:** `|closure − 1| ≤ 0.05` at every j (tighten after metrology is
   qualified); no systematic residual vs j.

---

## 5. Data contract

Match the ingest adapters in `models/kinetics_fit_pipeline.py` and reuse the
`models/experimental_data` / `run_manifest` pipeline.

**RDE / LSV export** (one CSV per sweep):

| Column | Meaning | Maps to |
|---|---|---|
| `Voltage_V` | potential vs reference | `potential_V_vs_ref` |
| `Current_A` | current | `current_A` → `current_density_A_m2` |
| `Area_cm2` | working electrode area | `working_electrode_area_cm2` |
| `pH` | bulk pH | `pH` |
| `Temp_C` | temperature | `temperature_C` |
| `Fe_M` | Fe²⁺ conc (0 for B0⁻) | `fe2_concentration_M` |
| `Ref_V` | reference→SHE | `reference_to_she_V` |
| `Omega_rpm` | RDE rotation | RDE Levich separation |
| `Bath_FeFree` | `True`/`False` | flags step 1 vs step 2 data |

**Volumetric closure** (one row/run in the campaign manifest):

| Column | Meaning |
|---|---|
| `Charge_Ah` | applied charge over the run |
| `H2_Volume_mL`, `T_K`, `P_Pa` | gas volume and state |
| `Deposit_Fe_kg` | dry deposit Fe mass |
| `Run_time_s`, `Electrode_area_m2` | run duration & area |
| `Current_density_A_m2` | galvanostatic setpoint |

---

## 6. How the outputs feed the cathode model

The fitted `(i0_her, b_her)` and `(i0_fe, b_fe, δ)` replace the screening
defaults in `models/kinetics.py` / `models/cell_physics.py` (the #34
"calibrate this first" action), via the calibration pipeline
(`models/kinetics_fit_pipeline.py` / `calibration_pipeline.py`). `δ` goes to
`diffusion_layer_1d`/`transport.py` as a *measured* boundary layer instead of a
free parameter. The volumetric ledger is the independent charge-closure check
the model-ladder Level 1 requires (NEXT_STEPS §Model credibility ladder,
acceptance criteria).

Once `b_her` is measured, the #34 verdict-flip scenario (FE 0.7599 at the low
end) is resolved with data: if the measured slope is comfortably above the
flip threshold the reference verdict is robust to the formerly-dominant
unknown; if it is near it, the next measurement (gap, conductivity, contact
resistance) becomes the binding risk.

---

## 7. Scope & honesty

- **This module is L0.** The numbers in `run_rde_volumetric_h2` are synthetic
  demonstrations that the procedure *recovers* known kinetics (all five
  self-test verdicts PASS). Real wet-lab accuracy, instrument artefacts, and
  the water-vapour / non-ideality gas correction are L1.
- **Not gate evidence** — gates are measurement-only in `models/process_gates.py`.
- The fitted branches are for the **one bath/cathode/temperature** measured;
  temperature/pH/surface-state dependence is extrapolation (see
  `models/surface_state.py`), deferred.

See `models/rde_volumetric_h2.model_scope()` and `measurement_spec()` for the
machine-readable version of this contract.
