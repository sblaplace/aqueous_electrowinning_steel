"""Cross-modal bottom-up theory — one shared parameter set, many observable domains.

The repository's physics is bottom-up: a single set of transport/kinetic/thermal
properties should reproduce the cell chemistry, the electrolyte transport, the
thermal transient *and* the envelope/environmental behaviour at once.  That is
the strongest form of the "one theory, many independent observable domains"
principle: if the individual cells disagree when forced through the *same*
parameterization, then either a constant is drifting or (more interestingly) the
parameterization is *internally inconsistent* — no single set of numbers can
simultaneously satisfy every modality.

This module is the consistency harness that forces the four existing seams
(``cell_physics``, ``thermal_balance``, ``crate``, ``env_coupling``) through
one :class:`SharedScenario` and reports per-modality pass/fail with the
controlling parameters.

The headline conflict it surfaces
--------------------------------
Electrochem/transport assume a fixed operating temperature ``T_operating_C``
at which kinetics + diffusivities are evaluated.  But the *same* parameter set,
run through the thermal balance, computes the *actual* bath steady-state
temperature ``T_ss`` from the very ``V_cell`` and ``I`` the electrochem model
produces (heat generation ``Q_gen = I·(V_cell − E_therm)``) against ambient and
jacket losses.  If that closes on a temperature far from ``T_operating_C``,
then no single parameterization holds — e.g. the chemistry is valid at 60 °C but
the thermally-equilibrated cell sits at 118 °C.  The harness flags this as a
THERMAL failure and names the controlling parameters (``V_cell``, ``I``,
``UA_amb_W_K``, ``volume_L``, ``UA_jacket_W_K``).  ``find_consistent_cooling``
shows the one degree of freedom (heat removal) that re-closes the loop.

The environmental link: the same wind gust drives the crate stability verdict
and (via ``env_coupling``) a forced-convection term that augments the thermal
model's ambient loss, so the thermal transient and the envelope verdict read the
same site.  All numbers remain L0 screening like the source seams.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from .cell_physics import BathRecipe, CellGeometry, CellPhysics, ProcessConditions, OperatingPoint
from .thermal_balance import CellThermalParams, simulate_thermal_transient
from .crate import Crate, CrateConfig, CrateSpec, WindLoad, GroundSpec, EnvironmentalLoads
from .env_coupling import disturbance_from_environment, DisturbanceInputs
from .uncertainty import REGISTRY
from .thermal_balance import evaporative_heat_loss_W

# Number of mA/cm^2 per A/m^2 (j in mA/cm2 x10 = A/m2), and E_therm source.
_MA_CM2_TO_A_M2 = 10.0


# ---------------------------------------------------------------------------
# The one shared parameterization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedScenario:
    """A single parameterization passed through every modality.

    Fields are grouped by the seam that consumes them, but the harness feeds
    the *same* object everywhere — that is the point.
    """

    # ── electrochem / transport (cell_physics + diffusion_layer / Nernst-Planck)
    T_operating_C: float = 60.0          # assumed operating T for kinetics/transport
    j_mA_cm2: float = 150.0              # applied current density
    fe_conc_M: float = 1.0
    support_M: float = 0.5
    pH: float = 2.0
    boundary_layer_m: float = 50e-6
    fe_i0: float = 1.0e-2
    her_i0: float = 1.0e-6
    fe_tafel_V: float = 0.120
    her_tafel_V: float = 0.140
    interelectrode_gap_m: float = 0.02
    membrane_R_ohm_m2: float = 3.0e-4
    contact_R_ohm_m2: float = 5.0e-4

    # ── thermal (thermal_balance.py)
    electrode_area_m2: float = 0.05      # geometric cathode area -> total current
    volume_L: float = 40.0
    hardware_C_J_K: float = 500.0
    UA_amb_W_K: float = 3.0
    A_surface_m2: float = 0.04           # open-top evap surface
    relative_humidity: float = 0.5
    thermoneutral_V: float = 1.28
    cooling_active: bool = False
    T_cool_in_C: float = 20.0
    UA_jacket_W_K: float = 25.0
    crate_surface_area_m2: float = 8.0   # heat-exchange surface for wind convection

    # ── crate / envelope (crate.py) + environment (env_coupling.py)
    gust_m_s: float = 0.0
    terrain: str = "open"
    rain_mm_hr: float = 0.0
    flood_m: float = 0.0
    ingress: bool = False
    T_ambient_C: float = 25.0
    crate_mass_kg: float = 4500.0
    crate_length_m: float = 12.19
    crate_width_m: float = 2.44
    crate_height_m: float = 2.59
    drag_coefficient: float = 1.2
    soil_bearing_kPa: float = 100.0

    # ── consistency thresholds
    FE_floor: float = 0.50
    transport_margin: float = 1.0        # j must be <= transport_limit * margin
    T_consistent_tol_C: float = 15.0    # |T_ss - T_operating| beyond this -> conflict
    T_safe_max_C: float = 90.0
    T_safe_min_C: float = 0.0

    @classmethod
    def from_registry(cls, **overrides: float) -> "SharedScenario":
        """Build the default scenario from the central parameter registry.

        This is what makes the registry the single source of truth: electrochem
        exchange currents, transport diffusivities (via operating T), thermal
        props, and crate/site props all read their *nominals* from REGISTRY.
        """
        def nom(name: str) -> float:
            return float(REGISTRY[name].mean)

        return cls(
            # electrochem / transport
            T_operating_C=nom("T_operating_C"),
            fe_i0=nom("fe_i0"),
            her_i0=nom("her_i0"),
            fe_tafel_V=nom("fe_tafel_V"),
            her_tafel_V=nom("her_tafel_V"),
            interelectrode_gap_m=nom("interelectrode_gap_m"),
            membrane_R_ohm_m2=nom("membrane_R_ohm_m2"),
            contact_R_ohm_m2=nom("contact_resistance_ohm_m2"),
            # thermal
            electrode_area_m2=nom("electrode_area_m2"),
            volume_L=nom("volume_L"),
            hardware_C_J_K=nom("hardware_C_J_K"),
            UA_amb_W_K=nom("UA_amb_W_K"),
            A_surface_m2=nom("A_surface_m2"),
            relative_humidity=nom("relative_humidity"),
            thermoneutral_V=nom("thermoneutral_V"),
            T_ambient_C=nom("T_ambient_C"),
            UA_jacket_W_K=nom("UA_jacket_W_K"),
            # crate / envelope
            crate_mass_kg=nom("crate_mass_kg"),
            crate_length_m=nom("crate_length_m"),
            crate_width_m=nom("crate_width_m"),
            crate_height_m=nom("crate_height_m"),
            drag_coefficient=nom("drag_coefficient"),
            soil_bearing_kPa=nom("soil_bearing_kPa"),
            **overrides,
        )

    def with_(self, **changes: float) -> "SharedScenario":
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Per-modality result
# ---------------------------------------------------------------------------


@dataclass
class ModalityResult:
    name: str
    passed: bool
    detail: str
    controlling: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.controlling = dict(self.controlling or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.name,
            "pass": bool(self.passed),
            "detail": self.detail,
            "controlling_parameters": {
                k: float(v) for k, v in self.controlling.items()
            },
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class CrossModalReport:
    scenario: SharedScenario
    electrochem: ModalityResult            # cell chemistry (FE, precipitation)
    transport: ModalityResult              # charge/mass transport discipline
    thermal: ModalityResult                # transient closes on T_operating (the conflict)
    crate: ModalityResult                  # envelope structural verdict
    environment: ModalityResult            # env -> thermal wiring consistency
    T_ss_C: float
    V_cell_V: float
    current_A: float
    consistent: bool

    def modalities(self) -> List[ModalityResult]:
        return [self.electrochem, self.transport, self.thermal, self.crate, self.environment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consistent": bool(self.consistent),
            "T_ss_C": float(self.T_ss_C),
            "V_cell_V": float(self.V_cell_V),
            "current_A": float(self.current_A),
            "modalities": [m.to_dict() for m in self.modalities()],
            "scenario": self.scenario.__dict__,
        }

    def summary(self) -> str:
        lines = [
            "=" * 76,
            "CROSS-MODAL BOTTOM-UP THEORY — one shared parameter set",
            "=" * 76,
            f"j={self.scenario.j_mA_cm2:.0f} mA/cm2  T_op={self.scenario.T_operating_C:.0f} C  "
            f"I={self.current_A:.1f} A  V_cell={self.V_cell_V:.2f} V  T_ss={self.T_ss_C:.1f} C",
            "",
        ]
        for m in self.modalities():
            tag = "PASS" if m.passed else "FAIL"
            ctrl = ", ".join(f"{k}={v:.3g}" for k, v in m.controlling.items())
            lines.append(f"  [{tag}] {m.name:<11} {m.detail}")
            lines.append(f"          controlling: {ctrl}")
        lines.append("")
        verdict = (
            "MUTUALLY CONSISTENT — one parameter set closes every modality"
            if self.consistent
            else "CONFLICT — no single parameter set satisfies every modality"
        )
        lines.append(f"  VERDICT: {verdict}")
        lines.append("=" * 76)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _to_cell_inputs(scenario: SharedScenario):
    bath = BathRecipe(
        c_FeSO4_M=scenario.fe_conc_M,
        c_Na2SO4_M=scenario.support_M,
        pH=scenario.pH,
    )
    geometry = CellGeometry(
        interelectrode_gap_m=scenario.interelectrode_gap_m,
        membrane=True,
        membrane_area_resistance_ohm_m2=scenario.membrane_R_ohm_m2,
        contact_resistance_ohm_m2=scenario.contact_R_ohm_m2,
    )
    conditions = ProcessConditions(
        temperature_C=scenario.T_operating_C,
        boundary_layer_m=scenario.boundary_layer_m,
        fe_i0=scenario.fe_i0,
        her_i0=scenario.her_i0,
        fe_tafel_V=scenario.fe_tafel_V,
        her_tafel_V=scenario.her_tafel_V,
    )
    return bath, geometry, conditions


def _environment_disturbance(scenario: SharedScenario) -> DisturbanceInputs:
    env_state = {
        "wind_gust_m_s": scenario.gust_m_s,
        "T_ambient_C": scenario.T_ambient_C,
        "rain_intensity_mm_hr": scenario.rain_mm_hr,
        "flood_depth_m": scenario.flood_m,
        "ingress_detected": scenario.ingress,
    }
    return disturbance_from_environment(env_state, None)


def _crate_config(scenario: SharedScenario) -> CrateConfig:
    return CrateConfig(
        crate=CrateSpec(
            length_m=scenario.crate_length_m,
            width_m=scenario.crate_width_m,
            height_m=scenario.crate_height_m,
            mass_kg=scenario.crate_mass_kg,
            drag_coefficient=scenario.drag_coefficient,
        ),
        wind=WindLoad(
            gust_m_s=scenario.gust_m_s,
            direction="broadside",
            terrain=scenario.terrain,
            temperature_C=scenario.T_ambient_C,
        ),
        ground=GroundSpec(p_allow_kPa=scenario.soil_bearing_kPa, flood_depth_m=scenario.flood_m),
        env=EnvironmentalLoads(rain_intensity_mm_hr=scenario.rain_mm_hr),
    )


def _thermal_params(
    scenario: SharedScenario,
    point: OperatingPoint,
    disturbance: DisturbanceInputs,
) -> CellThermalParams:
    I_A = point.j_mA_cm2 * _MA_CM2_TO_A_M2 * scenario.electrode_area_m2
    T_amb_use = disturbance.T_ambient_C if disturbance.enabled else scenario.T_ambient_C
    # Wind-driven forced convection (env_coupling) augments ambient heat loss.
    h_conv = disturbance.h_conv_W_m2_K
    UA_eff = scenario.UA_amb_W_K + h_conv * scenario.crate_surface_area_m2
    return CellThermalParams(
        V_cell=point.V_cell,
        current_A=I_A,
        volume_L=scenario.volume_L,
        hardware_C_J_K=scenario.hardware_C_J_K,
        T_init_C=T_amb_use,
        T_amb_C=T_amb_use,
        UA_amb_W_K=UA_eff,
        A_surface_m2=scenario.A_surface_m2,
        relative_humidity=scenario.relative_humidity,
        cooling_active=scenario.cooling_active,
        T_cool_in_C=scenario.T_cool_in_C,
        UA_jacket_W_K=scenario.UA_jacket_W_K,
    )


def _thermal_steady_state(
    scenario: SharedScenario,
    point: OperatingPoint,
    disturbance: DisturbanceInputs,
    search_max_C: float = 300.0,
    n_points: int = 600,
) -> Dict[str, float]:
    """Solve the true steady-state cell temperature from the exact heat balance.

    Using the *same* loss terms as thermal_balance.py (ambient UA, evaporation,
    optional jacket), find T* where generation equals losses:

        Q_gen = UA_amb·(T − T_amb) + Q_evap(T) + [UA_jacket·(T − T_cool)]

    A fixed 4 h transient never reaches equilibrium for a large bath (massive
    thermal inertia), so the verdict must use the converged T*, not the t=4 h
    cut.  ``runaway=True`` if generation exceeds the maximum removable power
    within the search band.
    """
    T_amb_use = disturbance.T_ambient_C if disturbance.enabled else scenario.T_ambient_C
    h_conv = disturbance.h_conv_W_m2_K
    UA_eff = scenario.UA_amb_W_K + h_conv * scenario.crate_surface_area_m2

    I_A = point.j_mA_cm2 * _MA_CM2_TO_A_M2 * scenario.electrode_area_m2
    q_gen = max(0.0, I_A * (point.V_cell - scenario.thermoneutral_V))

    def loss(T: float) -> float:
        q_amb = UA_eff * (T - T_amb_use)
        q_evap = evaporative_heat_loss_W(
            T, T_amb_use, scenario.A_surface_m2, scenario.relative_humidity
        )
        q_jacket = (
            scenario.UA_jacket_W_K * (T - scenario.T_cool_in_C)
            if scenario.cooling_active
            else 0.0
        )
        return q_amb + q_evap + q_jacket

    def f(T: float) -> float:
        return (q_gen - loss(T)) / max(scenario.UA_amb_W_K + scenario.UA_jacket_W_K, 1e-3)

    lo = T_amb_use - 5.0
    hi = max(search_max_C, lo + 10.0)
    # If generation still exceeds losses at the top of the band -> runaway.
    if f(hi) > 0.0:
        return {"T_ss_C": float("inf"), "q_gen_W": q_gen, "runaway": True, "converged": False}

    # Bisection for the root in [lo, hi].
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    T_ss = 0.5 * (lo + hi)
    return {"T_ss_C": float(T_ss), "q_gen_W": q_gen, "runaway": False, "converged": True}


def run_all_modalities(scenario: SharedScenario) -> CrossModalReport:
    """Run every modality with the one shared parameter set, report pass/fail."""
    # ── Electrochem + transport (cell_physics, one solve) ──────────────────
    bath, geometry, conditions = _to_cell_inputs(scenario)
    cp = CellPhysics(bath, geometry, conditions)
    point = cp.solve_at_j(scenario.j_mA_cm2)

    # ── Environment + crate (same site data) ───────────────────────────────
    disturbance = _environment_disturbance(scenario)
    crate_verdict = Crate().evaluate(_crate_config(scenario))

    # ── Thermal: transient trajectory + converged steady state ─────────────
    thermal = _thermal_params(scenario, point, disturbance)
    therm = simulate_thermal_transient(thermal, t_end_hr=4.0)
    ss = _thermal_steady_state(scenario, point, disturbance)
    T_ss = ss["T_ss_C"]
    I_A = point.j_mA_cm2 * _MA_CM2_TO_A_M2 * scenario.electrode_area_m2

    # ── 1. Electrochem chemistry ───────────────────────────────────────────
    fe_ok = point.current_efficiency >= scenario.FE_floor and not point.precipitation_active
    electrochem = ModalityResult(
        "electrochem",
        fe_ok,
        f"FE={point.current_efficiency*100:.1f}% (floor {scenario.FE_floor*100:.0f}%), "
        f"precip={'YES' if point.precipitation_active else 'no'}, surf_pH={point.surface_pH:.2f}",
        {"fe_i0": scenario.fe_i0, "her_i0": scenario.her_i0,
         "j_mA_cm2": scenario.j_mA_cm2, "T_operating_C": scenario.T_operating_C,
         "pH": scenario.pH},
    )

    # ── 2. Transport discipline ────────────────────────────────────────────
    j_lim = point.transport_limit_mA_cm2 * scenario.transport_margin
    transport_ok = point.transport_limit_mA_cm2 > 0 and scenario.j_mA_cm2 <= j_lim + 1e-9
    transport = ModalityResult(
        "transport",
        transport_ok,
        f"j={scenario.j_mA_cm2:.0f} <= transport limit {point.transport_limit_mA_cm2:.0f} "
        f"(diff {point.diffusion_limit_mA_cm2:.0f}) mA/cm2",
        {"j_mA_cm2": scenario.j_mA_cm2, "transport_limit_mA_cm2": point.transport_limit_mA_cm2,
         "boundary_layer_m": scenario.boundary_layer_m, "T_operating_C": scenario.T_operating_C},
    )

    # ── 3. Thermal closure (the headline cross-modal conflict) ─────────────
    safe_band = scenario.T_safe_min_C <= T_ss <= scenario.T_safe_max_C
    t_consistent = abs(T_ss - scenario.T_operating_C) <= scenario.T_consistent_tol_C
    thermal_ok = safe_band and t_consistent
    thermal = ModalityResult(
        "thermal",
        thermal_ok,
        f"T_ss={T_ss:.1f} C vs assumed T_op={scenario.T_operating_C:.0f} C "
        f"(tol {scenario.T_consistent_tol_C:.0f} C), safe band [{scenario.T_safe_min_C:.0f},{scenario.T_safe_max_C:.0f}] C, "
        f"Q_gen={therm['heat_gen_power_W']:.0f} W",
        {"V_cell_V": point.V_cell, "current_A": I_A, "UA_amb_W_K": scenario.UA_amb_W_K,
         "volume_L": scenario.volume_L, "UA_jacket_W_K": scenario.UA_jacket_W_K,
         "cooling_active": float(scenario.cooling_active),
         "thermoneutral_V": scenario.thermoneutral_V, "T_ambient_C": disturbance.T_ambient_C},
    )

    # ── 4. Crate / envelope ────────────────────────────────────────────────
    crate = ModalityResult(
        "crate",
        crate_verdict.stable,
        f"FS_overturn={crate_verdict.fs_overturn:.2f}, FS_bearing={crate_verdict.fs_bearing:.2f}, "
        f"FS_slide={crate_verdict.fs_slide:.2f}, ingress={crate_verdict.ingress_risk}",
        {"gust_m_s": scenario.gust_m_s, "crate_mass_kg": scenario.crate_mass_kg,
         "crate_height_m": scenario.crate_height_m, "soil_bearing_kPa": scenario.soil_bearing_kPa},
    )

    # ── 5. Environment -> thermal wiring consistency ───────────────────────
    # The thermal transient must have used the same ambient temperature and
    # wind-driven convection the crate/env layer derived from the same site.
    env_same_T = abs(disturbance.T_ambient_C - scenario.T_ambient_C) < 1e-6
    # If there is wind, the disturbance must be enabled and should have
    # augmented the thermal ambient loss (h_conv applied -> UA_eff > base).
    if scenario.gust_m_s > 0:
        env_ok = disturbance.enabled and disturbance.h_conv_W_m2_K > 0 and env_same_T
        detail = (f"env enabled, h_conv={disturbance.h_conv_W_m2_K:.1f} W/m2K augments thermal loss, "
                  f"T_amb={disturbance.T_ambient_C:.1f} C == {scenario.T_ambient_C:.1f} C")
    else:
        env_ok = not disturbance.enabled and env_same_T
        detail = (f"no wind/rain/ingress -> env disabled (no-op), "
                  f"T_amb={disturbance.T_ambient_C:.1f} C == {scenario.T_ambient_C:.1f} C")
    environment = ModalityResult(
        "environment",
        env_ok,
        detail,
        {"gust_m_s": scenario.gust_m_s, "rain_mm_hr": scenario.rain_mm_hr,
         "T_ambient_C": scenario.T_ambient_C, "h_conv_W_m2_K": disturbance.h_conv_W_m2_K},
    )

    consistent = all(m.passed for m in (electrochem, transport, thermal, crate, environment))

    return CrossModalReport(
        scenario=scenario,
        electrochem=electrochem,
        transport=transport,
        thermal=thermal,
        crate=crate,
        environment=environment,
        T_ss_C=T_ss,
        V_cell_V=point.V_cell,
        current_A=I_A,
        consistent=consistent,
    )


# ---------------------------------------------------------------------------
# Design closure: the one degree of freedom that re-closes the loop
# ---------------------------------------------------------------------------


def find_consistent_cooling(
    scenario: SharedScenario,
    ua_grid: Optional[List[float]] = None,
) -> Optional[float]:
    """Find the jacket ``UA`` (W/K) that brings T_ss back onto T_operating.

    This is the single free design parameter that re-closes the thermal loop:
    the heat the electrochem parameterization dumps into the bath (V_cell·I)
    must be removed by a matched jacket for the *same* parameter set to hold
    both the 60 °C chemistry and the thermal steady state.  Returns ``None`` if
    no jacket size in the grid closes within tolerance.
    """
    grid = ua_grid or [0.0, 2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1000.0]
    best = None
    best_dev = float("inf")
    for ua in grid:
        s = scenario.with_(cooling_active=True, UA_jacket_W_K=float(ua))
        rep = run_all_modalities(s)
        dev = abs(rep.T_ss_C - scenario.T_operating_C)
        if dev < best_dev:
            best_dev = dev
            best = ua
        if dev <= scenario.T_consistent_tol_C:
            return ua
    return best if best_dev <= scenario.T_consistent_tol_C else None


def consistent_scenario(scenario: Optional[SharedScenario] = None) -> SharedScenario:
    """Return a copy of ``scenario`` with jacket cooling sized to close the loop.

    The resulting parameter set is *internally consistent*: electrochem,
    transport, thermal and crate/envelope all pass.
    """
    base = scenario or SharedScenario.from_registry()
    ua = find_consistent_cooling(base)
    if ua is None:
        raise RuntimeError(
            "No jacket size closes the thermal loop for this parameterization; "
            "the parameter set is thermally unsustainable at any screened cooling size."
        )
    return base.with_(cooling_active=True, UA_jacket_W_K=ua)


def cross_modal_summary(scenario: Optional[SharedScenario] = None) -> str:
    """Convenience: run the default shared parameterization and print the report."""
    return run_all_modalities(scenario or SharedScenario.from_registry()).summary()


def main() -> None:
    print(cross_modal_summary())


if __name__ == "__main__":
    main()
