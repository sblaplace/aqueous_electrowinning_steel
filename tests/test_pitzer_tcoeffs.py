"""Acceptance gate for verified Pitzer T-coefficient tables (2026-08).

Status (2026-08-06): the gate has VETTED ITS FIRST TABLE.  The shipped
Fe²⁺–SO₄²⁻ pair carries the Kobylin, Sippola & Taskinen (2011, CALPHAD
35:499–511, Table 6) MTDATA-form temperature functions verbatim
(``t_form="mtd"``), validated by reproducing both published γ±(FeSO4,
0.1 m, 25 °C) anchors (0.164 Kobylin / 0.161 Reardon & Beckie) on this
repo's machinery to ≈0.7 %/1.1 %.  The R&B (1987) β(T)/Cφ(T) functions
remain paywalled — ``RB1987_ACCEPTANCE_ANCHORS`` stays in its documented
pending state.  See docs/PITZER_TCOEFF_ACCEPTANCE.md.

What this file pins:

1.  Shipped state: the Kobylin table is in the library, applied verbatim
    to the Fe–SO4 pair; every other pair stays frozen; the acceptance
    anchors pass the verifier with their recorded errors.
2.  The plumbing works for candidates: register → apply → the pair
    evolves with T (and warns outside its certified window); revert
    restores the SHIPPED state (which for Fe–SO4 includes the accepted
    table — PITZER_BINARY_SHIPPED is the source of truth).
3.  The verifier discriminates: synthetic anchors generated FROM a
    candidate pass it and reject a wrong candidate — so a future R&B
    transcription cannot wave a bad table through.
4.  The R&B acceptance anchors stay in their documented *pending* state.
"""

from __future__ import annotations

import warnings

import pytest

from models.pitzer import (
    FESO4_GAMMA_ANCHORS,
    KOBYLIN2011_FESO4_TTABLE,
    PITZER_BINARY,
    PITZER_BINARY_SHIPPED,
    RB1987_ACCEPTANCE_ANCHORS,
    T_COEFF_LIBRARY,
    GammaAnchor,
    PitzerPair,
    TCoeffTable,
    apply_t_coeff_library,
    mean_activity_coefficient_pure,
    register_t_coeff_table,
    revert_t_coeff_library,
    solve_pitzer,
    verify_t_coeff_table,
)

_FE_SO4 = ("Fe2+", "SO4-2")


@pytest.fixture(autouse=True)
def _clean_library():
    """Each test sees the shipped state (Kobylin table applied) and leaves
    the module exactly that way."""
    PITZER_BINARY.update(PITZER_BINARY_SHIPPED)
    for name in [n for n in T_COEFF_LIBRARY if n != "kobylin-2011-feso4"]:
        del T_COEFF_LIBRARY[name]
    T_COEFF_LIBRARY["kobylin-2011-feso4"] = KOBYLIN2011_FESO4_TTABLE
    yield
    PITZER_BINARY.update(PITZER_BINARY_SHIPPED)
    for name in [n for n in T_COEFF_LIBRARY if n != "kobylin-2011-feso4"]:
        del T_COEFF_LIBRARY[name]
    T_COEFF_LIBRARY["kobylin-2011-feso4"] = KOBYLIN2011_FESO4_TTABLE


def _synthetic_table() -> TCoeffTable:
    """A made-up (NOT literature) eq36 table exercising the at_T machinery."""
    return TCoeffTable(
        cation="Fe2+", anion="SO4-2",
        t_coeffs=(
            (0.2568, -30.0, 0.002, 0.0),   # β⁰ drifts with T
            (3.063, -120.0, 0.0, 0.0),     # β¹
            (-42.42, 0.0, 0.0, 0.0),       # β² anchored only
            (0.0213, 0.0, 0.0, 0.0),       # Cφ anchored only
        ),
        t_range_C=(10.0, 60.0),
        provenance="synthetic test fixture (tests/test_pitzer_tcoeffs.py) — NOT literature data",
    )


class TestShippedState:
    def test_library_shipped_table_and_pending_rb_anchors(self):
        """The Kobylin table is registered; R&B anchors remain pending."""
        assert T_COEFF_LIBRARY == {"kobylin-2011-feso4": KOBYLIN2011_FESO4_TTABLE}
        assert RB1987_ACCEPTANCE_ANCHORS == ()
        assert [a.gamma_expected for a in FESO4_GAMMA_ANCHORS] == [0.164, 0.161]

    def test_shipped_pair_carries_the_table_verbatim(self):
        pair = PITZER_BINARY[_FE_SO4]
        table = T_COEFF_LIBRARY["kobylin-2011-feso4"]
        # the applied pair and the library copy cannot drift apart
        assert pair.t_coeffs == table.t_coeffs
        assert pair.t_range_C == table.t_range_C
        assert pair.t_form == table.t_form == "mtd"
        # verbatim transcription of Kobylin et al. (2011) Table 6
        assert pair.t_coeffs == (
            (5.1934, -0.0161, 1.8349e-5, -508.3),    # β⁰
            (15.8514, 0.0085, -6.0442e-5, -3205.3),  # β¹
            (-16.2142, 0.0, 0.0, 0.0),               # β²
            (-0.0588, 0.0, 0.0, 12.8),               # Cφ
        )

    def test_kobylin_25C_projection(self):
        """The verbatim table evaluates to Kobylin's printed 25 °C values."""
        p = PITZER_BINARY[_FE_SO4].at_T(25.0)
        assert p.beta0 == pytest.approx(0.3194, abs=1e-4)
        assert p.beta1 == pytest.approx(2.2621, abs=1e-4)
        assert p.beta2 == pytest.approx(-16.2142, abs=1e-6)
        assert p.Cphi == pytest.approx(-0.01587, abs=1e-4)

    def test_other_pairs_stay_frozen(self):
        for key, pair in PITZER_BINARY.items():
            if key == _FE_SO4:
                continue
            assert pair.at_T(10.0) is pair, key
            assert pair.at_T(80.0) is pair, key

    def test_gamma_temperature_track(self):
        """γ±(FeSO4, 0.1 m) across the certified window — Aφ(T) + Kobylin
        T-functions together (2026-08-06 machinery; monotone decreasing,
        25 °C value the published anchor)."""
        gammas = {t: mean_activity_coefficient_pure("Fe2+", "SO4-2", 0.1, T_C=t)
                  for t in (10.0, 25.0, 40.0, 50.0, 60.0, 80.0)}
        assert gammas[10.0] == pytest.approx(0.16814, abs=1e-4)
        assert gammas[25.0] == pytest.approx(0.16279, abs=1e-4)
        assert gammas[40.0] == pytest.approx(0.15497, abs=1e-4)
        assert gammas[50.0] == pytest.approx(0.14862, abs=1e-4)
        assert gammas[60.0] == pytest.approx(0.14155, abs=1e-4)
        assert gammas[80.0] == pytest.approx(0.12579, abs=1e-4)
        ts = sorted(gammas)
        assert all(gammas[a] > gammas[b] for a, b in zip(ts, ts[1:]))

    def test_verifier_passes_the_shipped_table(self):
        """The acceptance demonstration: shipped table vs published anchors
        reproduces both to within their 2 % tolerances (recorded errors
        ≈ 0.7 % / 1.1 %)."""
        rep = verify_t_coeff_table(T_COEFF_LIBRARY["kobylin-2011-feso4"],
                                   FESO4_GAMMA_ANCHORS)
        assert rep["passed"] is True
        assert [r["passed"] for r in rep["rows"]] == [True, True]
        assert rep["rows"][0]["rel_err"] == pytest.approx(0.0074, abs=2e-3)
        assert rep["rows"][1]["rel_err"] == pytest.approx(0.0111, abs=2e-3)

    def test_out_of_window_warns(self):
        pair = PITZER_BINARY[_FE_SO4]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pair.at_T(50.0)
            assert not any("outside" in str(x.message) for x in w)
            pair.at_T(95.0)
            assert any("outside" in str(x.message) for x in w)


class TestPlumbing:
    def test_register_validate_and_reject_bad_tables(self):
        t = _synthetic_table()
        register_t_coeff_table("synth", t)
        assert T_COEFF_LIBRARY["synth"] is t
        with pytest.raises(ValueError):  # duplicate
            register_t_coeff_table("synth", t)
        with pytest.raises(ValueError):  # provenance required
            register_t_coeff_table("noprov", TCoeffTable(
                "Fe2+", "SO4-2", t.t_coeffs, (10.0, 60.0), ""))
        with pytest.raises(ValueError):  # unknown t_form
            register_t_coeff_table("badform", TCoeffTable(
                "Fe2+", "SO4-2", t.t_coeffs, (10.0, 60.0), "x", t_form="bogus"))
        with pytest.raises(KeyError):  # unknown pair
            register_t_coeff_table("nopair", TCoeffTable(
                "Fe2+", "Cl-", t.t_coeffs, (10.0, 60.0), "x"))  # Fe-Cl not in set
        with pytest.raises(ValueError):  # malformed rows
            register_t_coeff_table("badrows", TCoeffTable(
                "Fe2+", "SO4-2", ((0.0, 0.0, 0.0, 0.0),), (10.0, 60.0), "x"))

    def test_apply_evolves_pair_and_warns_outside_window(self):
        baseline = solve_pitzer({"Fe2+": 0.1, "SO4-2": 0.1}, T_C=50.0)
        register_t_coeff_table("synth", _synthetic_table())
        apply_t_coeff_library("synth")
        pair = PITZER_BINARY[_FE_SO4]
        assert pair.t_form == "eq36"
        assert pair.at_T(50.0) is not pair
        evolved = solve_pitzer({"Fe2+": 0.1, "SO4-2": 0.1}, T_C=50.0)
        assert evolved.gamma["Fe2+"] != baseline.gamma["Fe2+"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pair.at_T(80.0)
            assert any("outside" in str(x.message) for x in w)
        revert_t_coeff_library("synth")
        # revert restores the SHIPPED pair (which keeps its Kobylin table)
        assert PITZER_BINARY[_FE_SO4] is PITZER_BINARY_SHIPPED[_FE_SO4]
        restored = solve_pitzer({"Fe2+": 0.1, "SO4-2": 0.1}, T_C=50.0)
        assert restored.gamma["Fe2+"] == baseline.gamma["Fe2+"]

    def test_verify_discriminates_candidates(self):
        """Self-consistency: anchors from a candidate pass it, reject another."""
        t = _synthetic_table()
        anchors = tuple(
            GammaAnchor(
                t_C=temp, molality=0.1,
                gamma_expected=self._gamma_with(t, temp),
                rel_tol=1e-9, source="synthetic self-consistency fixture")
            for temp in (10.0, 25.0, 50.0))
        assert verify_t_coeff_table(t, anchors)["passed"] is True

        wrong = TCoeffTable("Fe2+", "SO4-2",
                            ((0.2568, +85.0, 0.0, 0.0),
                             (3.063, 0.0, 0.0, 0.0),
                             (-42.42, 0.0, 0.0, 0.0),
                             (0.0213, 0.0, 0.0, 0.0)),
                            (10.0, 60.0), "wrong synthetic fixture")
        report = verify_t_coeff_table(wrong, anchors)
        assert report["passed"] is False
        # anchors are (10, 25, 50 °C): the wrong table matches at the Tr
        # anchor (25 °C, eq36 c1 vanishes there) and is off elsewhere.
        assert report["rows"][1]["rel_err"] == 0.0
        assert report["rows"][2]["rel_err"] > 1e-6
        assert report["rows"][0]["rel_err"] > 1e-6

    def test_verify_installs_table_t_form(self):
        """An mtd candidate must be verified AS mtd (not read as eq36)."""
        anchors = tuple(
            GammaAnchor(t, 0.1,
                        self._gamma_with(T_COEFF_LIBRARY["kobylin-2011-feso4"], t),
                        1e-9, "self-consistency")
            for t in (25.0, 50.0))
        assert verify_t_coeff_table(T_COEFF_LIBRARY["kobylin-2011-feso4"],
                                    anchors)["passed"] is True

    @staticmethod
    def _gamma_with(table: TCoeffTable, t_C: float) -> float:
        saved = PITZER_BINARY[(table.cation, table.anion)]
        try:
            PITZER_BINARY[(table.cation, table.anion)] = PitzerPair(
                saved.beta0, saved.beta1, saved.beta2, saved.Cphi,
                alpha1=saved.alpha1, alpha2=saved.alpha2,
                t_coeffs=table.t_coeffs, t_range_C=table.t_range_C,
                t_form=table.t_form)
            return mean_activity_coefficient_pure(
                table.cation, table.anion, 0.1, T_C=t_C)
        finally:
            PITZER_BINARY[(table.cation, table.anion)] = saved
