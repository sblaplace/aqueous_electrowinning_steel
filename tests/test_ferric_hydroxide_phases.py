"""Fe(III) hydroxide/oxyhydroxide phase speciation + Ostwald aging kinetics.

Two layers:

1. Module-level (fast) — the phase ladder closes CHEM_PHYS_REVIEW.md Tier 2.1:
   the Ksp set, Ostwald ordering, initial-phase-by-anion, the blended-cap
   trajectory, inventory conservation, the H+-release mechanism (an aged sludge
   pulls Fe3+ out of solution — releasing 3 H+/Fe — at a fixed dissolved
   condition, where a fresh one does not), and the phase-aware [Fe3+]/shuttle/
   sludge sizing in ``fe3_shuttle``.

2. Integration (slow) — the H+ *schedule* end to end through ``bath_dynamics``:
   with ``ferric_phase_aging_enabled`` the controlling phase matures, so the
   dissolved Fe3+ pins ~2+ decades below the legacy Fe(OH)3 assumption and a
   real sludge bleed (with its 3 H+/Fe release) appears where the legacy model
   predicts none.  The flag is inert unless the Fe3+ shuttle itself is on.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.ferric_hydroxide_phases import (
    CHLORIDE_OSTWALD,
    SULFATE_OSTWALD,
    PHASES,
    age_inventory,
    blended_cap_M,
    initial_phase_for_bath,
    ostwald_ladder,
    solubility_cap_M,
)

# ---------------------------------------------------------------------------
# Module level: ordering, path choice, cap trajectory, conservation
# ---------------------------------------------------------------------------

ORDER_BY_INSOLUBILITY = [
    "feoh3_amorphous", "ferrihydrite_2line", "akaganeite",
    "goethite", "hematite",
]


def test_phase_ordering_by_ksp():
    """Ksp (and the [Fe3+] cap at fixed pH) falls monotonically with maturity."""
    log_ksps = [PHASES[p].log_ksp for p in ORDER_BY_INSOLUBILITY]
    for a, b in zip(log_ksps, log_ksps[1:]):
        assert a > b, (a, b)
    # [Fe3+] cap at a fixed pH follows the same order (Ksp/[OH-]^3 scales Ksp).
    caps = [solubility_cap_M(p, 2.0) for p in ORDER_BY_INSOLUBILITY]
    for a, b in zip(caps, caps[1:]):
        assert a > b
    # 2-line ferrihydrite (the task's initial phase) sits ~1 decade below the
    # legacy Fe(OH)3; goethite/hematite a further 1-2 decades below that.
    feoh3 = PHASES["feoh3_amorphous"].ksp
    assert PHASES["ferrihydrite_2line"].ksp / feoh3 == pytest.approx(10.0 ** -0.7, rel=1e-3)
    assert PHASES["goethite"].ksp / feoh3 == pytest.approx(10.0 ** -2.0, rel=1e-3)
    assert PHASES["hematite"].ksp / feoh3 == pytest.approx(10.0 ** -4.0, rel=1e-3)


def test_default_cap_matches_feoh3_baseline():
    """The legacy Fe(OH)3 phase cap reproduces the pinned [Fe3+] cap scale."""
    # Legacy Fe(OH)3 at pH 2 = 2e-3 M (pinned in test_fe3_shuttle).
    assert solubility_cap_M("feoh3_amorphous", 2.0) == pytest.approx(2.00e-3, rel=0.02)


def test_initial_phase_by_bath_anion():
    """Sulfate -> ferrihydrite; chloride (FeCl3 hydrolysis) -> akaganeite."""
    assert initial_phase_for_bath("sulfate") == "ferrihydrite_2line"
    assert initial_phase_for_bath("SulfAtE") == "ferrihydrite_2line"
    assert initial_phase_for_bath("chloride") == "akaganeite"
    assert initial_phase_for_bath("fecl3") == "akaganeite"
    assert ostwald_ladder("sulfate") == SULFATE_OSTWALD
    assert ostwald_ladder("chloride") == CHLORIDE_OSTWALD
    assert CHLORIDE_OSTWALD[0] == "akaganeite"
    assert CHLORIDE_OSTWALD[1:] == ["goethite", "hematite"]


def test_blended_cap_trajectory_monotonic_with_aging():
    """As a fresh iron sludge matures, the effective cap falls monotonically."""
    inventory = {"ferrihydrite_2line": 1.0, "goethite": 0.0, "hematite": 0.0}
    cap0 = blended_cap_M(2.0, inventory)
    assert cap0 == pytest.approx(solubility_cap_M("ferrihydrite_2line", 2.0), rel=1e-9)
    prev = cap0
    for _ in range(5):
        inventory = age_inventory(
            inventory, dt_hr=200.0, temperature_C=60.0,
            t12_fh_gh_hr=400.0, t12_gh_hem_hr=400.0)
        cap = blended_cap_M(2.0, inventory)
        assert cap <= prev * (1.0 + 1e-9)
        prev = cap
    # Fully aged toward the sulfate tail, the cap has dropped materially.
    assert cap0 / prev > 5.0


def test_age_inventory_conserves_mass_and_reaches_terminal():
    """Ostwald stepping conserves Fe and drives the inventory to hematite."""
    inv = {"ferrihydrite_2line": 1.0, "goethite": 0.0, "hematite": 0.0}
    total = sum(inv.values())
    inv = age_inventory(inv, dt_hr=1.0, temperature_C=25.0,
                        t12_fh_gh_hr=0.1, t12_gh_hem_hr=0.1)
    assert sum(inv.values()) == pytest.approx(total, rel=1e-9)
    assert inv["hematite"] == pytest.approx(total, rel=0.01)
    assert inv["ferrihydrite_2line"] < 1e-3


def test_age_inventory_leaves_hematite_terminal_sink():
    inv = {"hematite": 5.0}
    out = age_inventory(inv, dt_hr=1.0, temperature_C=60.0,
                        t12_fh_gh_hr=0.01, t12_gh_hem_hr=0.01)
    assert out["hematite"] == pytest.approx(5.0, rel=1e-9)
    assert sum(out.values()) == pytest.approx(5.0, rel=1e-9)


def test_aged_inventory_releases_more_h_at_same_dissolved_state():
    """An aged sludge pulls more Fe3+ out (=> 3 H+/Fe) than a fresh one.

    The bath's operator split precipitates ``max(0, [Fe3+] - blended_cap)`` and
    releases 3 H+ per Fe precipitated.  With the dissolved Fe3+ fixed, a sludge
    that has been aged to the less-soluble tail sits at a lower cap, so more
    Fe3+ precipitates (and hence more H+ is released) than the same bath holding
    a fresh, still-soluble ferrihydrite sludge.  This is the mechanism by which
    the H+ release is gated by (follows) the aging schedule rather than being
    fixed by an instantaneous single-solid assumption.
    """
    fresh = {"ferrihydrite_2line": 1.0, "goethite": 0.0, "hematite": 0.0}
    aged = {"ferrihydrite_2line": 0.01, "goethite": 0.99, "hematite": 0.0}
    fe3 = 3.0e-5  # between the ferrihydrite cap (~3.5e-5) and goethite (~2.1e-5)
    precip_fresh = max(0.0, fe3 - blended_cap_M(2.0, fresh))
    precip_aged = max(0.0, fe3 - blended_cap_M(2.0, aged))
    assert precip_fresh == 0.0          # fresh phase holds the Fe3+ in solution
    assert precip_aged > 0.0            # ...aged phase forces it out -> 3 H+/Fe
    assert precip_aged > 50.0 * precip_fresh


# ---------------------------------------------------------------------------
# fe3_shuttle: phase-aware sizing (the cap/shuttle span 1-2 decades)
# ---------------------------------------------------------------------------

def test_phase_aware_sizing_ordering():
    """The phase moves the controlling [Fe3+] and shuttle by 1-2 decades.

    ``steady_state`` sizes the Fe3+ / sludge picture against the phase's Ksp
    ([Fe3+]_ss pins at the cap).  Switching from the legacy Fe(OH)3 to a mature
    phase drops the dissolved Fe3+ and the shuttle current by ~2 decades
    (goethite) to ~4 decades (hematite) — the anchor the closed-loop bleed is
    sized against.  The physical bleed *flux* is bounded by production, so the
    ordering is monotonic but bounded; the mis-sizing lives in the [Fe3+]/shuttle
    term, which is what must be bled/recycled.
    """
    from models.fe3_shuttle import ShuttleParams, open_headspace, steady_state

    def ss(phase):
        return steady_state(ShuttleParams(pH=2.35, temperature_C=50.0,
                                          ferric_phase=phase), open_headspace())

    base = ss(None)
    for phase in ["ferrihydrite_2line", "akaganeite", "goethite", "hematite"]:
        assert ss(phase)["feoh3_precipitation_active"]

    order = [None, "ferrihydrite_2line", "akaganeite", "goethite", "hematite"]
    caps = [ss(p)["fe3_solubility_cap_M"] for p in order]
    for a, b in zip(caps, caps[1:]):
        assert a > b
    fe3ss = [ss(p)["fe3_ss_M"] for p in order]
    for a, b in zip(fe3ss, fe3ss[1:]):
        assert a > b
    i_sh = [ss(p)["i_shuttle_A_m2"] for p in order]
    for a, b in zip(i_sh, i_sh[1:]):
        assert a > b
    # ~2 decades on goethite, ~4 on hematite, vs the Fe(OH)3 baseline.
    assert ss("goethite")["fe3_ss_M"] == pytest.approx(base["fe3_ss_M"] / 1e2, rel=0.2)
    assert ss("hematite")["i_shuttle_A_m2"] <= base["i_shuttle_A_m2"] / 1e3
    # bleed flux is monotonic with phase maturity (production-bounded)
    bleeds = [ss(p)["iron_sludge_loss_g_L_day"] for p in order]
    for a, b in zip(bleeds, bleeds[1:]):
        assert b >= a
    assert bleeds[-1] > bleeds[0]


# ---------------------------------------------------------------------------
# Integration: H+ release schedule through bath_dynamics (slow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    from models.twin_physics import CellProcessModel
    return CellProcessModel()


def _base_dp(**over):
    dp = {
        "temperature_C": 50.0,
        "pH": 2.0,
        "cell_voltage_V": 5.0,
        "j_avg_mA_cm2": 0.0,          # no plating drive in redox-focused tests
        "electrode_area_m2": 1.0,
        "electrolyte_volume_L": 800.0,
        "fe2_M": 1.0,
        "recirculation_flow_L_hr": 0.0,   # isolate the redox block from mixing
        "reservoir_volume_L": 2000.0,
        "catholyte_volume_L": 800.0,
        "anolyte_volume_L": 2000.0,
        "buffer_capacity_beta": 1e9,  # pin pH unless a test wants it free
        "acid_dose_rate_M_hr": 0.0,
        "pH_control_gain_M_hr_ph": 0.0,
        "fe2_makeup_rate_M_hr": 0.0,
    }
    dp.update(over)
    return dp


def _x0():
    return np.array([50.0, 50.0, 1.0, 2.0, 1e-3, 0.0, 5.0])


def _aux0():
    from models.bath_dynamics import BathAux
    return BathAux(T_reservoir_C=50.0, fe2_reservoir_M=1.0, pH_reservoir=2.0)


def _enabled_dp(fast=True, aging=True, **over):
    """Design point with the Fe3+ shuttle on (and optionally phase aging on)."""
    from models.bath_dynamics import apply_fe3_scenario
    from models.fe3_shuttle import sealed_divided_cell
    dp = apply_fe3_scenario(_base_dp(fe3_k_ox_ref=1.0), sealed_divided_cell())
    dp["ferric_phase_aging_enabled"] = bool(aging)
    dp["bath_anion"] = "sulfate"
    t12 = 0.05 if fast else 1e6
    dp["ferric_aging_t12_fh_gh_hr"] = t12
    dp["ferric_aging_t12_gh_hem_hr"] = t12
    dp.update(over)
    return dp


def _immobilized_mol(aux):
    return float(sum((aux.ferric_phase_inventory or {}).values()))


def test_aging_flag_inert_without_fe3_shuttle(model):
    """``ferric_phase_aging_enabled`` alone (no Fe3+ shuttle) changes nothing."""
    from models.bath_dynamics import step
    dp_plain = _base_dp()
    dp_flagged = _base_dp(fe3_shuttle_enabled=False, ferric_phase_aging_enabled=True)
    xa, xb = _x0(), _x0()
    aa, ab = _aux0(), _aux0()
    for _ in range(15):
        xa, aa = step(xa, aa, 0.05, dp_plain, model)
        xb, ab = step(xb, ab, 0.05, dp_flagged, model)
    np.testing.assert_array_equal(xa, xb)
    assert aa.to_dict() == ab.to_dict()


@pytest.mark.slow
def test_aging_releases_h_and_bleeds_where_legacy_predicts_none(model):
    """The H+ schedule follows the controlling phase, not instant Fe(OH)3.

    At this operating point the Fe3+ production sits above the aged-phase cap
    but *below* the legacy Fe(OH)3 cap.  The legacy single-solid model therefore
    predicts no sludge and no precipitation-H+ (Fe3+ stays dissolved and
    shuttles), while phase-aware aging pins dissolved Fe3+ ~2+ decades lower and
    drives a real bleed — releasing 3 H+/Fe on the aging schedule.
    """
    from models.bath_dynamics import step
    dp_legacy = _enabled_dp(aging=False)
    dp_aging = _enabled_dp(aging=True, fast=True)

    xl, al = _x0(), _aux0()
    xa, aa = _x0(), _aux0()
    n_steps, dt = 400, 0.02     # 8 hr
    for _ in range(n_steps):
        xl, al = step(xl, al, dt, dp_legacy, model)
        xa, aa = step(xa, aa, dt, dp_aging, model)

    # Legacy: dissolved Fe3+ pinned near the (high) Fe(OH)3 cap, no sludge.
    assert al.fe3_catholyte_M == pytest.approx(2.0e-3, rel=0.2)
    assert _immobilized_mol(al) == pytest.approx(0.0, abs=1e-9)

    # Phase-aware aging: dissolved Fe3+ ~2+ decades lower, real sludge bleed
    # (=> the 3 H+/Fe release the legacy model misses entirely).
    assert aa.fe3_catholyte_M < al.fe3_catholyte_M / 1e2
    assert _immobilized_mol(aa) > 0.0
    # solid >> dissolved: the bleed is real, not an accounting artifact
    assert _immobilized_mol(aa) > aa.fe3_catholyte_M * 800.0


@pytest.mark.slow
def test_phase_inventory_matures_to_terminal_over_time(model):
    """Over a long run the inventory ages ferrihydrite -> goethite -> hematite."""
    from models.bath_dynamics import step
    dp = _enabled_dp(aging=True, fast=True)
    x, aux = _x0(), _aux0()
    for _ in range(500):
        x, aux = step(x, aux, 0.1, dp, model)
    inv = aux.ferric_phase_inventory
    assert inv["hematite"] > 0.0
    assert inv["hematite"] > inv["ferrihydrite_2line"]
    assert sum(inv.values()) > 0.0
