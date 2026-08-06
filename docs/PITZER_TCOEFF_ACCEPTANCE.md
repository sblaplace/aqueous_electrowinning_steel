# Pitzer T-Coefficient Tables — Sourcing Outcome & Acceptance Gate

**Date:** 2026-08-06
**Status:** Level-0 *screening* — parameter/provenance work, **not gate
evidence**
**Modules:** `models/pitzer.py` (registry/gate + the shipped table)
**Tests:** `tests/test_pitzer_tcoeffs.py`, `tests/test_pitzer.py`
**Related:** `docs/SIM_PITZER_ACTIVITY.md` (the activity-model note)

## Outcome (one paragraph)

The Fe²⁺–SO₄²⁻ pair now ships **verbatim published temperature functions**:
the Kobylin, Sippola & Taskinen (2011, CALPHAD 35(4):499–511, Table 6)
MTDATA-form coefficients, valid 10–90 °C, applied through the acceptance
gate described below.  The Reardon & Beckie (1987) functions the framework
was originally waiting for remain **not wireable from this environment** —
the paper is paywalled and no transcription with publication-grade
provenance could be verified; the shopping list in §4 stays open.  This
note records what was verified, how, and exactly what a future table (R&B,
Kobylin-2012 ternary, Fe–Cl, …) must do to be installed.

## 1. What was sourced and why Kobylin, not Reardon & Beckie

The literature search found:

* **R&B 1987** (*Modelling chemical equilibria of acid mine-drainage: The
  FeSO₄–H₂SO₄–H₂O system*, GCA 51:2355–2368, doi 10.1016/0016-7037(87)90290-0):
  abstract-verifiable facts only — model domain 10–90 °C (binary) / 10–60 °C
  (ternary, ≤6 m H₂SO₄), second-dissociation constant from Pitzer et al.
  (1977), and their Ksp(T) relations (typographically mangled in the
  publisher's abstract: unusable).  Their β⁰/β¹/β²/Cφ(T) tables are behind
  the paywall; mirrors expose only the 25 °C projection (Sandia/WIPP AP-176
  prints β⁰=0.2568, β¹=3.063, β²=−42, Cφ=0.0209 — note the repo's shipped
  anchor set (Pitzer 1991 tabulation) is 0.2568/3.063/**−42.42**/**0.0213**,
  a small tabulation discrepancy recorded here for the record).
* **Kobylin et al. 2011** — coefficients for the full MTDATA temperature
  function exposed in the paper's searchable full text, and the functional
  form itself confirmed from the author's doctoral dissertation (Aalto
  2013, eq. 26):
  `p(T) = A + B·T + C·T·ln T + D·T² + E·T³ + F/T` with, for FeSO4,
  C = E = 0.  Same 2–2 electrolyte convention (α₁=1.4, α₂=12) as this
  repo's pair.  Kobylin's set is a **superseding** CALPHAD re-assessment
  that resolved the documented internal enthalpy inconsistency of the R&B
  fits (ΔHs ≈ 16.1 vs 21.2 kJ/mol at 20 °C; see the thesis's §1.5.1).
* **Kobylin et al. 2012** (ternary FeSO₄–H₂SO₄–H₂O, A+F/T forms) — Table 3
  paywalled; not sourced.
* **Marion/FREZCHEM** — adopted R&B with β¹ refit; validated −2…25 °C only;
  not usable over the bath's 25–80 °C window.

**Decision (2026-08-06, user-approved):** wire the Kobylin 2011 set
verbatim — the only candidate whose coefficients *and* functional form are
publicly verifiable and whose anchor reproduction could be demonstrated on
this repo's machinery.  R&B stays a pending shopping-list item; wiring its
25 °C projection *as if it were a T-function* is exactly the kind of
"sourced" the gate was built to refuse.

## 2. The installed table and its verification

`KOBYLIN2011_FESO4_TTABLE` (registered name `kobylin-2011-feso4`), verbatim:

| parameter | A | B | D·10⁻⁵ (printed) | F·10³ (printed) |
|---|---|---|---|---|
| β⁰ | 5.1934 | −0.0161 | 1.8349 | −0.5083 |
| β¹ | 15.8514 | 0.0085 | −6.0442 | −3.2053 |
| β² | −16.2142 | 0 | 0 | 0 |
| Cφ | −0.0588 | 0 | 0 | 0.0128 |

with `t_range_C = (10, 90)` — the window jointly certified by the R&B
binary validity (10–90 °C) and the Kobylin assessment; outside it,
`PitzerPair.at_T` raises its extrapolation warning.

Verification performed (and pinned in `tests/test_pitzer_tcoeffs.py`):

* **Transcription is exact**: the 298.15 K projection evaluates to
  Kobylin's printed 25 °C values (0.3194 / 2.2621 / −16.2142 / −0.0159).
* **Anchor reproduction**: γ±(FeSO4, 0.1 m, 25 °C) = **0.1628** on this
  machinery vs Kobylin's published anchor **0.164** (0.74 %) — inside the
  0.150–0.164 assessment spread that also brackets R&B's **0.161**
  (1.11 %).  Both anchors are `FESO4_GAMMA_ANCHORS`; the verifier must
  pass them.
* **Shape**: γ±(0.1 m) is smooth and monotone-decreasing over the window
  (0.168 at 10 °C → 0.126 at 80 °C); the characteristic ~2 m γ± minimum
  with an upturn toward the copperas solubility molality (3.58 m) is
  preserved.
* The frozen-set numbers this replaces (Aφ(T)-only response): γ±(25 °C)
  0.1587 → 0.1628 (+2.6 %), γ±(50 °C) 0.1430 → 0.1486 (+4.0 %), and the
  RC-1 mixture γ(Fe²⁺) 0.0692 → 0.0685 (−0.9 %) at 50 °C — all inside the
  mutual spread of the two published assessments.  Downstream effects
  (Nernst E_rev, κ, V_cell) are sub-mV at bath temperature.

## 3. The acceptance gate (unchanged machinery, now with one passed table)

`models/pitzer.py` exports:

* `TCoeffTable` — pair key, 4×4 coefficient rows, validity window, a
  **required provenance string with table/equation numbers**, and
  `t_form ∈ {"eq36", "mtd"}`;
* `register_t_coeff_table` / `apply_t_coeff_library` /
  `revert_t_coeff_library` — install/remove candidates; revert restores
  `PITZER_BINARY_SHIPPED` (for Fe–SO4 the shipped state **includes** the
  accepted Kobylin table);
* `verify_t_coeff_table(table, anchors)` — temporarily installs the table
  and checks every `GammaAnchor` through the full machinery; returns a
  pass report with per-anchor errors;
* `FESO4_GAMMA_ANCHORS` — the γ± anchors the current table passed;
* `RB1987_ACCEPTANCE_ANCHORS` — **empty, pinned**: populated only when
  R&B values with table/equation-cited provenance exist.

## 4. Standing shopping list (still open)

1. **R&B 1987 β(T)/Cφ(T) tables** for Fe²⁺–SO₄²⁻ (and Fe²⁺–HSO₄⁻, θ/ψ
   acid terms) — needs the paper's actual tables with table numbers cited
   (GCA 51:2355–2368; not available in this environment).  If sourced:
   `register_t_coeff_table` with table/equation-cited provenance, then
   `verify_t_coeff_table` against γ± anchors at {10, 25, 40, 60} °C plus
   copperas solubility vs T, and only then `apply` + repin tests.
2. **Kobylin et al. 2012 ternary set** (Fe²⁺–HSO₄⁻, ψ terms, A+F/T forms,
   Table 3) — paywalled; same procedure when sourced.  Matters at bath
   pH ≲ 2.
3. **Fe–Cl pair** for the AWARE chloride route (unchanged from the SIM
   note).

## 5. Limitations (unchanged honesty)

* The Kobylin set is a **different parameter correlation** than the R&B
  anchor set; both reproduce γ± within their mutual spread, but absolute
  γ values at high molality (≥1 m) carry the assessment-level uncertainty
  (~5–10 %), and the Cφ sign flip (0.0213 → −0.0159) is a correlation
  trade-off, not new physics.  Anything calibrated to a *measured* bath
  γ/κ should supersede both.
* The **osmotic coefficient** is far more sensitive to the β⁰/β¹
  partition than γ± is (φ and ln γ± weight β¹ differently at high I:
  e^−α√I vs g(α√I)).  Swapping the parameter partition moves φ much more
  than any pinned γ: RC-1 mixture at 50 °C φ = 0.706 → 0.547, water
  activity 0.957 → 0.966 (pure 1 m FeSO4 at 25 °C: φ = 0.604 → 0.469).
  This is inherent to the two *published* correlations, not a machinery
  bug — both are verbatim — but osmotic/water-activity numbers should be
  treated as assessment-uncertain (~±0.1 in φ) until a measured bath
  water activity exists.
* Mixture terms (θ, ψ, ᵈθ) are still applied against these binaries per
  the module's documented defaults; Kobylin's binary assessment says
  nothing new about them.
* >90 °C is extrapolated; <10 °C was not checked against the freezing-
  point/depression data the Kobylin model also fits (window starts at 10
  °C deliberately, matching R&B/Galvanic bath operations).
* None of this changes the L0 screening status: activity numbers are
  inputs to a screening engine, not gate evidence.
