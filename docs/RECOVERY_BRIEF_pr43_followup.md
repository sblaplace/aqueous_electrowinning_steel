# RECOVERY BRIEF — PR #43 follow-up physics (interrupted turn)

Status: recovery + completion record. The lost uncommitted working-tree edits
from the interrupted Arena turn were recovered and the BV-core edits they gated
implemented. All four follow-up items shipped on PR #43 (head `f59480b`). The
remaining items below (Pitzer T-form, registry entries, docs) were planned in
the lost turn but not written — they are still open.

Restored / shipped scope (all on `recover/fe3-herm`, base `6d361c6`, PR #43):

  - models/fe3_shuttle.py            (VERBATIM)            Fe3+ shuttle / O2 bath aging, L0
  - models/her_microkinetics.py      (RECONSTRUCTED)       DFT HER Volmer-Heyrovsky check, L0
  - tests/test_fe3_shuttle.py        (VERBATIM)            11 tests, green
  - tests/test_her_microkinetics.py  (RECONSTRUCTED)       8 tests, green
  - models/kinetics.py / transport.py / diffusion_layer_1d.py  (BV core edits) — IMPLEMENTED
  - tests/test_butler_volmer.py      (VERBATIM)            restored + green
  - tests/test_kinetics.py                                  polarization CE mask updated

Verification on base 6d361c6: 19/19 pass, ruff-clean. `python -m models.fe3_shuttle`
reproduces the recorded output (sealed 0.003%, open 0.006% CE loss, sludge active
for open/crossover). `her_microkinetics.consistency_report` reproduces the recorded
numbers exactly: slope ratio 0.916, i_ratio_25C=1.351, i_ratio_70C=0.811, theta=1.0.

Design intent (from the lost turn, preserved here so nothing is re-derived):
  - fe3_shuttle: reuse bath_startup.fe2_oxidation_rate + dissolved_o2_saturation.
    Sources = homogeneous autoxidation (pinned O2 fraction) + optional anolyte O2
    crossover fault; sink = mass-transfer-limited cathodic reduction. Steady-state
    identity i_shuttle,ss = F·(V/A)·r_prod (independent of k_m). Fe(OH)3 cap
    (logKsp=-38.7) makes open-air sit AT the cap: CE penalty stays <0.2 pp even
    open; the real damage is iron-inventory sludge bleed, not CE. That IS the
    honest L0 finding.
  - her_microkinetics: empirical HER branch stays default at operation. This module
    is a screening consistency check (does DFT-anchored Heyrovsky-RDS @ DG_H*≈-0.40 eV
    reproduce empirical 140 mV/dec? yes, 128 vs 140 = within ~20%). k_Hey inherits the
    empirical apparent Ea (60 kJ/mol) so off-anchor T-ratios isolate the slope form.

================================================================================
DONE — Butler-Volmer core edits (implemented in f59480b; kept here as the record)
================================================================================

These were in the lost uncommitted working tree and only partially captured
(diffs present, final files truncated). They gate tests/test_butler_volmer.py
(see below). **Implemented in commit f59480b on PR #43; spec retained below so
the physics is auditable.** The test file is now restored and green.

Intent (from the turn): the repo kinetics were Tafel-only, which has two
artifacts the full Butler-Volmer form fixes: i(E_eq) = i0 != 0 (thermodynamically
wrong — BV gives i(0)=0 exactly) and no dissolution branch anodic of E_eq.
Numeric footprint at operating points is ~3e-8 (reverse term negligible), so all
existing FE/V results must NOT move at screening precision. BV is default-ON.

### models/kinetics.py
- Add anodic-branch slope constants, derived once at 25 °C via alpha_a·n = n - alpha_c·n
  with alpha_c·n = b(25 °C)-scale:
    FE:  b_c=0.120 -> alpha_c·n = 0.05916/0.120 = 0.493 -> alpha_a·n = 2-0.493 = 1.507
         -> FE_ANODIC_SLOPE_V = 0.05916/1.507 ≈ 0.0393 V/dec
    HER: alpha_c = 0.05916/0.140 = 0.423 -> alpha_a = 1-0.423 = 0.577
         -> HER_ANODIC_SLOPE_V = 0.05916/0.577 ≈ 0.1025 V/dec
  (recorded values: FE_ANODIC_SLOPE_V=0.0393, HER_ANODIC_SLOPE_V=0.1025)
- Add dataclass ButlerVolmerBranch mirroring TafelBranch API with an anodic_slope_V
  field. current(E) is SIGNED, cathodic positive:
        i = i0·(10^((Eeq-E)/b_c) - 10^((E-Eeq)/b_a))   -> 0 at E=Eeq, negative anodic
  Koutecky-Levich blend (i_lim) caps ONLY the cathodic arm; negative (dissolution)
  branch returns i_kin unchanged (no KL).
- DepositionKinetics: add use_butler_volmer: bool = True default, fe_anodic_slope_V /
  her_anodic_slope_V fields. Keep fe_branch/her_branch properties returning TafelBranch
  (E_eq consumers); use BV branches in partial_currents when flag on.
- Guard: in solvers, negative BV net current -> treat branch as off (max(net, floor))
  so log-space bisection stays non-negative. Document: dissolution flux outside the
  screening (cathodic-only) envelope.
- CE defined only where both partial currents are cathodic (galvanostatic concept).
  polarization_curve docstring: "CE = i_Fe/i_total is a galvanostatic concept,
  defined where both partial currents are cathodic; outside that regime it is a
  signed ratio, not a current efficiency."

### models/transport.py
- Replace _tafel_current with _bv_current(E, i0, slope_c, slope_a, E_eq).
- _kinetic_currents: fe_kin = max(bv(...), 0.0); her likewise. KL blend unchanged.
- Add fe_anodic_slope_V/her_anodic_slope_V fields defaulting to the new constants.
- Import FE_ANODIC_SLOPE_V, HER_ANODIC_SLOPE_V from .kinetics.

### models/diffusion_layer_1d.py
- Same: _tafel_current (line ~550) -> _bv_current; use at the two Picard-loop sites
  (lines ~594-595 seed, ~631-632 in-loop) with max(...,0.0) guards.
- Add anodic-slope fields; import the constants.

### tests/test_kinetics.py
- test_polarization_curve_shapes: sweep tail runs anodic of E_eq(Fe) where Fe
  dissolves (i_fe<0) and CE is undefined. Change the assertion to mask by
  `cathodic = i_tot > 0.0` (or both partials cathodic) with a dated 2026-08 comment.
  Recorded edit comment: "with full Butler-Volmer branches the sweep tail runs anodic
  of E_eq(Fe), where Fe dissolves (i_fe < 0) and CE as a ratio is undefined; the [0,1]
  bound holds wherever the net current is actually cathodic."

### tests/test_butler_volmer.py  (RESTORED — landed in f59480b; content below for the record)
This test file was fully captured and is verbatim-recoverable, but could not import
until kinetics.py gained ButlerVolmerBranch / FE_ANODIC_SLOPE_V / HER_ANODIC_SLOPE_V
and transport.py/DepositionKinetics gained the anodic-slope wiring. **It is now
restored and green on PR #43 (f59480b).** Full content retained below for audit.

--------------------------------------------------------------------------------
BEGIN tests/test_butler_volmer.py
--------------------------------------------------------------------------------
"""Assertions for the full Butler–Volmer branches (2026-08 addition).
The repo's kinetics used to be Tafel-only, which has two artefacts:
i(E_eq) = i0 ≠ 0 and no representation of dissolution anodic of E_eq.
These tests pin the BV corrections without expecting any change at
operating overpotentials (the reverse term is 10^-3..10^-8 there).
"""
from math import isfinite
import numpy as np
import pytest
from models.kinetics import (
    FE_ANODIC_SLOPE_V,
    HER_ANODIC_SLOPE_V,
    ButlerVolmerBranch,
    DepositionKinetics,
    TafelBranch,
)
from models.transport import NernstPlanckFilm

FE_I0 = 1.0e-2
FE_E_EQ = -0.440

@pytest.fixture()
def fe_bv():
    return ButlerVolmerBranch(
        FE_I0, 0.120, FE_E_EQ, i_lim=None, anodic_slope_V=FE_ANODIC_SLOPE_V
    )

def test_current_is_zero_at_equilibrium(fe_bv):
    """The defining BV property absent in Tafel-only form."""
    assert fe_bv.current(FE_E_EQ) == pytest.approx(0.0, abs=1e-12)
    her = ButlerVolmerBranch(1e-6, 0.140, -0.12, None, HER_ANODIC_SLOPE_V)
    assert her.current(-0.12) == pytest.approx(0.0, abs=1e-12)

def test_cathodic_side_recovers_tafel_limit(fe_bv):
    """At |η| ≥ 150 mV the reverse term is ≲1e-4 of the forward one."""
    tafel = TafelBranch(FE_I0, 0.120, FE_E_EQ)
    for eta in (0.15, 0.25, 0.40, 0.70):
        E = FE_E_EQ - eta
        i_bv = fe_bv.current(E)
        i_tf = tafel.current(E)
        assert abs(i_bv - i_tf) / i_tf < 1e-4, f"η={eta}: {i_bv} vs {i_tf}"

def test_anodic_side_is_signed_and_grows(fe_bv):
    """Anodic of E_eq the branch is net-oxidation (Fe dissolves)."""
    i1 = fe_bv.current(FE_E_EQ + 0.10)
    i2 = fe_bv.current(FE_E_EQ + 0.20)
    assert i1 < 0.0 and i2 < 0.0
    # One decade per anodic slope unit: ~0.0392 V/dec.
    ratio = abs(i2) / abs(i1)
    assert ratio == pytest.approx(10.0 ** (0.10 / FE_ANODIC_SLOPE_V), rel=0.05)

def test_koutecky_levich_only_caps_the_cathodic_arm():
    """i_lim must blend the cathodic arm and leave dissolution alone."""
    b = ButlerVolmerBranch(1.0, 0.120, FE_E_EQ, i_lim=100.0,
                           anodic_slope_V=FE_ANODIC_SLOPE_V)
    i_cat = b.current(FE_E_EQ - 0.5)
    assert i_cat < 100.0  # capped by transport
    i_an = b.current(FE_E_EQ + 0.15)
    i_an_no_lim = ButlerVolmerBranch(
        1.0, 0.120, FE_E_EQ, None, FE_ANODIC_SLOPE_V
    ).current(FE_E_EQ + 0.15)
    assert i_an == pytest.approx(i_an_no_lim)

def test_anodic_slope_bookkeeping_values():
    """α_a·n = n − α_c·n with α_c·n read from the 25 °C cathodic slope."""
    from models.electrochemistry import FARADAY as F_REPO, R_GAS as R_REPO
    b25 = 2.303 * R_REPO * 298.15 / F_REPO
    assert FE_ANODIC_SLOPE_V == pytest.approx(b25 / (2.0 - b25 / 0.120))
    assert HER_ANODIC_SLOPE_V == pytest.approx(b25 / (1.0 - b25 / 0.140))

def test_deposition_kinetics_matches_tafel_only_at_operating_points():
    """Galvanostatic answers must not move at screening precision."""
    k_bv = DepositionKinetics(temperature_C=50.0)
    k_tf = DepositionKinetics(temperature_C=50.0, use_butler_volmer=False)
    for j in (10.0, 50.0, 100.0, 200.0, 400.0):
        assert k_bv.efficiency_at_current(j) == pytest.approx(
            k_tf.efficiency_at_current(j), abs=1e-4
        )
        assert k_bv.potential_at_current(j) == pytest.approx(
            k_tf.potential_at_current(j), abs=1e-4
        )

def test_deposition_kinetics_fe_branch_zero_at_equilibrium():
    k = DepositionKinetics(temperature_C=50.0)
    i_fe, i_h, i_tot = k.partial_currents(k.fe_E_eq)
    assert float(i_fe) == pytest.approx(0.0, abs=1e-9)
    assert float(i_h) > 0.0  # HER still runs cathodically at E_eq(Fe)
    assert isfinite(float(i_tot))

def test_polarization_curve_shows_dissolution_anodic_of_Eeq():
    k = DepositionKinetics(temperature_C=50.0)
    E = np.linspace(-0.50, -0.30, 41)
    _, i_fe, _, i_tot, _ = k.polarization_curve(E)
    assert float(i_fe[0]) > 0.0
    assert float(i_fe[-1]) < 0.0  # Fe dissolving at the anodic end

@pytest.mark.parametrize("j", [25.0, 100.0, 250.0])
def test_nernst_planck_solve_is_tafel_compatible(j):
    """Film-model galvanostatic solve: BV must reproduce Tafel answers."""
    film = NernstPlanckFilm(temperature_C=50.0)
    state = film.solve(j)
    assert state.converged
    assert 0.9 < state.current_efficiency <= 1.0
    assert state.applied_current_A_m2 == pytest.approx(j * 10.0, rel=1e-4)
--------------------------------------------------------------------------------
END tests/test_butler_volmer.py
--------------------------------------------------------------------------------

================================================================================
STILL OPEN (planned in the lost turn, not written; carry into a follow-up)

*** UPDATE 2026-08-06 (Arena follow-up turn): all three items below are now ***
*** DONE on PR #43 — T-form framework in commit 001fa7e, registry entries in  ***
*** commit 20f2db2, docs/README in the docs commit closing this brief.        ***
*** Original plan text retained underneath for the record.                    ***
================================================================================

### Pitzer T-form framework (models/pitzer.py) — planned, not implemented
- Extend PitzerPair with t_coeffs: Tuple[Tuple[float,float,float,float], ...] =
  ((0,)*4, (0,)*4, (0,)*4, (0,)*4) — order (beta0, beta1, beta2, Cphi), each a
  4-coefficient EQ3/6-Sandia polynomial. Simple 3-coefficient form:
      p(T) = a + c1·(1/T - 1/Tr) + c2·ln(T/Tr) + c3·(T - Tr),  Tr = 298.15
- Add PitzerPair.at_T(T_C) -> constructing a T-adjusted copy (frozen dataclass).
- In solve_pitzer, _pair(c,a) returns p.at_T(T_C). Only Aφ currently responds to T
  (binary params frozen / Zomaitis simplification) — all-zero t_coeffs must keep
  current numbers byte-identical.
- No verified Na2SO4/H2SO4 T-coefficient table was sourced in the turn; ship frozen
  zeros + documented window flag, extending when a verified table lands.

### Registry entries (models/uncertainty/parameter_registry.py) — planned
- anolyte_conductivity_S_m: divided-cell anode-side conductivity assumed/unmeasured,
  screening. mean 60, std 20, uniform (20,100), module="cell". Negligible on planner
  variance (std²=400 vs 4.28e8 total).
- fe2_autoxidation_k_ref: mean 1e-4, lognormal, std 5e-5, bounds (1e-5,1e-3),
  source "Singer&Stumm screening (bath_startup.py)", module="bath_startup".

### Docs / README — planned
- New docs/SIM_BUTLER_VOLMER.md (BV + HER microkinetics check) and
  docs/SIM_BATH_REDOX.md (fe3_shuttle) per repo SIM_*.md convention.
- Mark shipped items in SIM_PITZER_ACTIVITY.md next-steps; add README rows for
  fe3_shuttle (+ kinetics BV note, pitzer T-framework note).

================================================================================
VERIFICATION (from the turn — now run and green on PR #43)
================================================================================
Full fast tier is green: `tests/test_transport.py tests/test_kinetics.py
tests/test_diffusion_layer_1d.py tests/test_fe3_shuttle.py
tests/test_her_microkinetics.py tests/test_butler_volmer.py`
`tests/test_operating_window.py tests/test_theory_confidence.py
tests/test_cell_physics.py` + ruff. BV parity holds: operating-point
FE/V unchanged to printed precision (theory_confidence values stay put).
Recorded parity reference reproduced: j=300 -> FE=0.985112, V=5.770117; j=100 ->
FE=0.995400, V=3.664786; j=200 -> FE=0.993171, V=4.712985. 177 tests pass.

Commit small and incremental (the loss happened because a single big turn's
uncommitted edits were wiped) — each module/test lands as its own commit so a
future hang can't lose everything again.
