"""Gas-hold-up ↔ cell-physics coupling: the reachability verdict with bubbles in.

Why this module exists
----------------------

``models/operating_point_optimizer.py`` (#40) reports the headline

    "Energy gate IS reachable: minimum energy 3,306 kWh/t at j = 150 mA/cm²,
     gap = 1.5 mm, contact = 1.0e-4 Ω·m²."

That verdict is computed from the **uncoupled** ``cell_physics`` voltage solve,
which feeds a fixed ``anode_bubble_fraction = 0.10`` into ``CellVoltageModel``
and otherwise treats the electrolyte as gas-free and uniform.  It therefore
structurally ignores the three things ``models/gas_holdup.py`` was written to
capture on a vertical channel:

1. the **axial cathodic H₂ void-fraction profile** ε(y) fed by ``1 − FE``,
2. the **Bruggeman conductivity penalty** κ_eff = κ(1 − ε)^1.5, and
3. the **current redistribution** that follows from the equipotential
   electrodes seeing an axially varying resistance.

This module is the bridge.  It composes the two existing 1-D models —
``CellPhysics`` for the voltage/FE/transport solve and ``gas_holdup`` for the
two-phase channel — and re-derives the #40 reachability verdict with the gas
ohmic penalty added to the cell voltage.  It **modifies neither upstream
module**; every two-phase quantity comes from ``gas_holdup``'s own functions
(``holdup_profile``, ``solve_current_distribution``, ``bruggeman_conductivity``,
``solve_coupled`` / ``CoupledGasResult``), not from a reimplementation.

Model tier and honesty framing
------------------------------

**Level-0 → Level-1 boundary. Every number below is a transparent, bottom-up
prediction, ``unvalidated (L0)``.**  It is **NOT gate evidence** — the FE and
energy gates are measurement-only and live in ``models/process_gates.py``.  No
gas hold-up, bubble-size or void-fraction measurement exists in this
repository; the two-phase correlations are transferred from water electrolysis
and chlor-alkali practice (see ``gas_holdup.measurement_protocol()``).

Coupling this pair of reduced-order models is the disciplined fidelity step the
repository's own docs ask for: ``docs/SIM_THEORY_CONFIDENCE.md`` states that "a
calibrated compartment/strip model is preferable to an unvalidated full CFD
model", and ``docs/NEXT_STEPS.md`` explicitly defers unvalidated full CFD.  No
CFD, FEM, phase-field or DFT is used or implied here.

How the coupling is applied
---------------------------

Two coupled energies are reported, and both are printed, because the model's
two gas effects pull in opposite directions:

* ``coupled_specific_energy_kWh_t`` — the **headline**, conservative number:
  the gas *ohmic* penalty ``CoupledGasResult.ohmic_penalty_V`` is added to
  ``CellPhysics``'s ``V_cell`` and the energy identity ``E = 959.9 · V / FE`` is
  re-evaluated at the **uncoupled** FE.  The gas correction can only ever raise
  the voltage and the energy, so this number can never flatter the gate.
* ``coupled_energy_with_FE_shift_kWh_t`` — the full two-way coupling: same
  voltage, but the FE the coupled solve computes (``area_average_FE``, which
  includes the model's bubble-microconvection ``FE_shift``).  In this model at
  RC-1 scale that shift is *favourable*, so this number sits slightly **below**
  the headline.  It is reported, never used to declare the gate reached on its
  own; the headline verdict is the conservative one.

FE is always a **derived** output of ``CellPhysics`` (injected into
``gas_holdup.solve_coupled`` as its ``fe_model`` hook), never an injected
tuning parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from .cell_physics import CellPhysics
from .economics_from_physics import (
    FE_TARGET_MIN,
    SPECIFIC_ENERGY_TARGET_MAX_KWH_T,
    TRANSPORT_LIMIT_MARGIN_MIN,
    ReferenceCell,
    reference_cell as _economics_reference_cell,
)
from .electrochemistry import specific_energy_kWh_per_t
from .gas_holdup import (  # noqa: F401  (re-exported on purpose: reuse, not reimplementation)
    ChannelGeometry,
    CoupledGasResult,
    bruggeman_conductivity,
    drift_flux_void_fraction,
    holdup_profile,
    solve_coupled,
    solve_current_distribution,
)
from .operating_point_optimizer import (
    energy_gate_reachable as _uncoupled_energy_gate_reachable,
)

SCREENING_FLAG = "unvalidated (L0)"
BOUNDARY_NOTE = (
    "Level-0 → Level-1 boundary: coupled two reduced-order models "
    "(cell_physics + gas_holdup). Prediction only, NOT gate evidence; "
    "gates are measurement-only (models/process_gates.py)."
)

#: RC-1 channel dimensions (docs/REFERENCE_CELL_DESIGN_BASIS.md,
#: processes/reference_cell_rc1.yaml): 50 mm tall × 20 mm wide electrode
#: (= 10 cm²) in a 3 mm deep channel.
RC1_CHANNEL_HEIGHT_M = 0.050
RC1_CHANNEL_WIDTH_M = 0.020
RC1_CHANNEL_DEPTH_M = 0.003
RC1_LIQUID_FLOW_L_MIN = 0.25

#: Axial discretisation and fixed-point settings for the coupled solve.
#: These mirror ``run_gas_holdup.main()``'s call pattern (n_segments=4,
#: max_iterations=6) so the coupling inherits the same numerics as the
#: standalone gas model; ``solve_coupled``'s own default tolerance (2e-3 on
#: both the FE and current vectors) is used unchanged.
N_SEGMENTS = 4
MAX_ITERATIONS = 6

#: Joint lever space re-scanned under coupling — identical to #40's
#: ``operating_point_optimizer.solve_window``.
J_VALUES_MA_CM2: Tuple[float, ...] = (150.0, 300.0)
GAP_VALUES_M: Tuple[float, ...] = (1.5e-3, 3.0e-3)
MEMBRANE_VALUES_OHM_M2: Tuple[float, ...] = (3.0e-4,)
CONTACT_VALUES_OHM_M2: Tuple[float, ...] = (1.0e-4, 5.0e-4)


# ─── The coupled cell ─────────────────────────────────────────────────

@dataclass(frozen=True)
class CoupledCell:
    """RC-1 seen by both physics models at once.

    ``electro`` is the #38 ``ReferenceCell`` (bath, ``CellGeometry``,
    ``ProcessConditions``); ``channel`` is the ``gas_holdup.ChannelGeometry``
    built from the *same* RC-1 channel dimensions.  Holding both in one object
    is the coupling point: one cell, two physics.
    """

    electro: ReferenceCell
    channel: ChannelGeometry

    @property
    def name(self) -> str:
        return self.electro.name


def coupled_reference_cell(
    *,
    gap_m: Optional[float] = None,
    contact_resistance_ohm_m2: Optional[float] = None,
    membrane_area_resistance_ohm_m2: Optional[float] = None,
    channel_height_m: float = RC1_CHANNEL_HEIGHT_M,
) -> CoupledCell:
    """Build the RC-1 cell with both physics attached.

    Reuses ``economics_from_physics.reference_cell()`` verbatim for the
    electrochemistry side, and builds the ``gas_holdup.ChannelGeometry`` from
    the RC-1 channel dimensions (10 cm² electrode = 50 mm × 20 mm, 3 mm deep
    channel).  The interelectrode gap is shared by both models, so overriding
    it moves the ohmic path in the voltage solve *and* the gassy resistance in
    the channel solve together.
    """
    cell = _economics_reference_cell()
    geometry = cell.geometry

    gap = float(gap_m) if gap_m is not None else geometry.interelectrode_gap_m
    if gap <= 0.0:
        raise ValueError("gap_m must be positive")
    contact = (
        float(contact_resistance_ohm_m2)
        if contact_resistance_ohm_m2 is not None
        else geometry.contact_resistance_ohm_m2
    )
    if contact < 0.0:
        raise ValueError("contact_resistance_ohm_m2 must be non-negative")
    membrane_r = (
        float(membrane_area_resistance_ohm_m2)
        if membrane_area_resistance_ohm_m2 is not None
        else geometry.membrane_area_resistance_ohm_m2
    )

    electro = replace(
        cell,
        interelectrode_gap_m=gap,
        geometry=replace(
            geometry,
            interelectrode_gap_m=gap,
            contact_resistance_ohm_m2=contact,
            membrane_area_resistance_ohm_m2=membrane_r,
        ),
    )
    channel = ChannelGeometry(
        height_m=channel_height_m,
        width_m=RC1_CHANNEL_WIDTH_M,
        depth_m=RC1_CHANNEL_DEPTH_M,
        interelectrode_gap_m=gap,
        liquid_flow_L_min=RC1_LIQUID_FLOW_L_MIN,
    )
    return CoupledCell(electro=electro, channel=channel)


# ─── FE hook: FE stays a CellPhysics output ───────────────────────────

_PHYSICS_CACHE: Dict[Tuple[Any, ...], CellPhysics] = {}
_FE_CACHE: Dict[Tuple[Any, ...], float] = {}
_GAS_CACHE: Dict[Tuple[Any, ...], CoupledGasResult] = {}


def _cell_key(cell: CoupledCell) -> Tuple[Any, ...]:
    b, g, c = cell.electro.bath, cell.electro.geometry, cell.electro.conditions
    return (
        cell.electro.name,
        b.c_FeSO4_M, b.c_Na2SO4_M, b.c_H2SO4_M, b.c_H3BO3_M, b.pH,
        g.interelectrode_gap_m, g.membrane, g.membrane_area_resistance_ohm_m2,
        g.contact_resistance_ohm_m2, g.anode_bubble_fraction,
        g.anode_chemistry, g.anode_fe2_conc_M, g.anode_fe_dissolution_i0_A_m2,
        c.temperature_C, c.boundary_layer_m, c.flow_regime, c.transport_model,
        cell.channel.height_m, cell.channel.width_m, cell.channel.depth_m,
        cell.channel.interelectrode_gap_m, cell.channel.liquid_flow_L_min,
    )


def _physics_for(cell: CoupledCell, boundary_layer_m: Optional[float] = None) -> CellPhysics:
    """``CellPhysics`` instance, cached by (cell, δ) — speciation is expensive."""
    e = cell.electro
    conditions = e.conditions
    if boundary_layer_m is not None:
        conditions = replace(conditions, boundary_layer_m=float(boundary_layer_m))
    key = _cell_key(cell) + (conditions.boundary_layer_m,)
    physics = _PHYSICS_CACHE.get(key)
    if physics is None:
        physics = CellPhysics(e.bath, e.geometry, conditions)
        _PHYSICS_CACHE[key] = physics
    return physics


def _cell_physics_fe_model(cell: CoupledCell):
    """Return a ``gas_holdup.solve_coupled``-compatible ``fe_model`` hook.

    The hook signature ``(j_mA_cm2, delta_m, temperature_C, fe_conc_M,
    pH_bulk)`` is fixed by ``gas_holdup.solve_coupled``.  Routing it into
    ``CellPhysics`` keeps FE a *derived* quantity of the same engine that
    produces the uncoupled baseline — the coupled and uncoupled FE therefore
    differ only because of the gas terms, never because of a different FE
    model.
    """

    def fe_model(
        j_mA_cm2: float,
        delta_m: float,
        temperature_C: float,
        fe_conc_M: float,
        pH_bulk: float,
    ) -> float:
        key = _cell_key(cell) + (round(float(j_mA_cm2), 6), round(float(delta_m), 12))
        cached = _FE_CACHE.get(key)
        if cached is not None:
            return cached
        point = _physics_for(cell, boundary_layer_m=float(delta_m)).solve_at_j(float(j_mA_cm2))
        fe = float(point.current_efficiency)
        _FE_CACHE[key] = fe
        return fe

    return fe_model


# ─── Coupled gas solve ────────────────────────────────────────────────

def coupled_gas_state(
    cell: CoupledCell,
    j_mA_cm2: float,
    *,
    kappa_S_m: float,
    gas_off: bool = False,
) -> CoupledGasResult:
    """Run ``gas_holdup``'s own coupled fixed point for this cell.

    ``gas_off=True`` builds the **null case**: the FE hook is replaced by a
    perfect-FE stub so the Faradaic gas fraction ``1 − FE`` is exactly zero.
    With no gas there is no void, no Bruggeman penalty and no redistribution,
    so ``ohmic_penalty_V`` must collapse to ~0 and the coupled solve must
    reproduce the uncoupled ``cell_physics`` answer.  That is the closure test.
    """
    key = _cell_key(cell) + (round(float(j_mA_cm2), 6), round(float(kappa_S_m), 9), gas_off)
    cached = _GAS_CACHE.get(key)
    if cached is not None:
        return cached

    if gas_off:
        def fe_model(j, delta_m, temperature_C, fe_conc_M, pH_bulk):  # noqa: ANN001
            return 1.0
    else:
        fe_model = _cell_physics_fe_model(cell)

    e = cell.electro
    result = solve_coupled(
        j_mean_mA_cm2=float(j_mA_cm2),
        geometry=cell.channel,
        temperature_C=e.conditions.temperature_C,
        kappa_S_m=float(kappa_S_m),
        delta_forced_m=e.conditions.boundary_layer_m,
        fe_conc_M=e.bath.c_FeSO4_M,
        pH_bulk=e.bath.pH,
        n_segments=N_SEGMENTS,
        max_iterations=MAX_ITERATIONS,
        fe_model=fe_model,
    )
    _GAS_CACHE[key] = result
    return result


# ─── The coupled operating point ──────────────────────────────────────

def coupled_operating_point(
    cell: Optional[CoupledCell] = None,
    j_mA_cm2: float = 150.0,
    *,
    contact_resistance_ohm_m2: Optional[float] = None,
    gap_m: Optional[float] = None,
    gas_off: bool = False,
) -> Dict[str, Any]:
    """Solve one operating point with the axial gas physics coupled in.

    Steps, in order:

    1. ``CellPhysics.solve_at_j`` → uncoupled ``V_cell``, FE, transport limit.
    2. ``gas_holdup.solve_coupled`` (fed the FE from step 1 through its
       ``fe_model`` hook) → axial void profile ε(y), Bruggeman κ_eff(y),
       redistributed j(y), and the ``CoupledGasResult``.
    3. ``V_coupled = V_uncoupled + CoupledGasResult.ohmic_penalty_V``.
    4. Energy from the program identity ``E = 959.9 · V / FE``
       (``electrochemistry.specific_energy_kWh_per_t``), at the uncoupled FE
       for the headline number and at the coupled FE for the contrast.

    The #38/#40 transport-limit rule is honoured: a point at or beyond the
    migration-enhanced transport limit is returned with ``valid = False`` and
    is never priced or counted as reachable.
    """
    if j_mA_cm2 <= 0.0:
        raise ValueError("j_mA_cm2 must be positive")
    if cell is None or gap_m is not None or contact_resistance_ohm_m2 is not None:
        base_gap = gap_m if gap_m is not None else (cell.electro.interelectrode_gap_m if cell else None)
        base_contact = contact_resistance_ohm_m2
        if base_contact is None and cell is not None:
            base_contact = cell.electro.geometry.contact_resistance_ohm_m2
        cell = coupled_reference_cell(gap_m=base_gap, contact_resistance_ohm_m2=base_contact)

    point = _physics_for(cell).solve_at_j(float(j_mA_cm2))
    transport_limit = float(point.transport_limit_mA_cm2)
    transport_margin = transport_limit / float(j_mA_cm2)
    valid = bool(transport_margin > TRANSPORT_LIMIT_MARGIN_MIN)

    kappa = float(point.conductivity_S_m)
    gas = coupled_gas_state(cell, float(j_mA_cm2), kappa_S_m=kappa, gas_off=gas_off)
    profile = gas.profile

    fe_uncoupled = float(point.current_efficiency)
    fe_coupled = float(gas.area_average_FE)
    v_uncoupled = float(point.V_cell)
    ohmic_penalty_V = float(gas.ohmic_penalty_V)
    v_coupled = v_uncoupled + ohmic_penalty_V

    energy_uncoupled = float(point.specific_energy_kWh_t)
    energy_coupled = specific_energy_kWh_per_t(v_coupled, fe_uncoupled)
    energy_coupled_fe = specific_energy_kWh_per_t(v_coupled, fe_coupled)

    gate = SPECIFIC_ENERGY_TARGET_MAX_KWH_T
    return {
        "j_mA_cm2": float(j_mA_cm2),
        "interelectrode_gap_m": cell.electro.geometry.interelectrode_gap_m,
        "contact_resistance_ohm_m2": cell.electro.geometry.contact_resistance_ohm_m2,
        "membrane_area_resistance_ohm_m2": cell.electro.geometry.membrane_area_resistance_ohm_m2,
        # transport gating, same rule as #38/#40
        "transport_limit_mA_cm2": transport_limit,
        "transport_margin": transport_margin,
        "valid": valid,
        # voltage
        "V_cell_uncoupled": v_uncoupled,
        "V_cell_coupled": v_coupled,
        "gas_ohmic_penalty_V": ohmic_penalty_V,
        "gas_ohmic_gas_free_V": float(gas.ohmic_gas_free_V),
        "delta_V": v_coupled - v_uncoupled,
        # efficiency
        "FE_uncoupled": fe_uncoupled,
        "FE_coupled": fe_coupled,
        "FE_shift_pp": float(gas.FE_shift),
        # energy
        "uncoupled_specific_energy_kWh_t": energy_uncoupled,
        "coupled_specific_energy_kWh_t": energy_coupled,
        "coupled_energy_with_FE_shift_kWh_t": energy_coupled_fe,
        "energy_delta_kWh_t": energy_coupled - energy_uncoupled,
        "energy_delta_pct": 100.0 * (energy_coupled - energy_uncoupled) / energy_uncoupled,
        # verdicts (conservative headline: ohmic-only coupling)
        "energy_gate_kWh_t": gate,
        "energy_gate_pass_uncoupled": bool(valid and energy_uncoupled <= gate),
        "energy_gate_pass_coupled": bool(valid and energy_coupled <= gate),
        "fe_gate_pass_coupled": bool(fe_coupled >= FE_TARGET_MIN),
        # two-phase state
        "mean_void_fraction": float(profile.mean_void_fraction),
        "outlet_void_fraction": float(profile.outlet_void_fraction),
        "conductivity_penalty": float(profile.conductivity_penalty),
        "current_uniformity": float(profile.current_uniformity),
        "bubble_diameter_um": float(profile.bubble_diameter_m) * 1e6,
        "hydrogen_flow_L_h_wet": float(gas.hydrogen_flow_L_h),
        "gas_solve_converged": bool(gas.converged),
        "gas_solve_iterations": int(gas.iterations),
        "gas_off": bool(gas_off),
        "flag": SCREENING_FLAG,
        "boundary_note": BOUNDARY_NOTE,
    }


# ─── Isolated gas impact ──────────────────────────────────────────────

def gas_impact_summary(
    cell: Optional[CoupledCell] = None,
    j_mA_cm2: float = 150.0,
) -> Dict[str, Any]:
    """Isolate what the gas terms alone do at one operating point.

    Reports the axial two-phase state (mean/outlet void fraction, Bruggeman
    conductivity penalty, current uniformity), the voltage the gas costs
    (``ohmic_penalty_V``), the FE the bubbles move (``FE_shift``), and the share
    of the coupled ``V_cell`` and of the coupled energy that the gas accounts
    for.  The Bruggeman check re-evaluates ``gas_holdup.bruggeman_conductivity``
    at the profile's mean void fraction, so the number quoted here is the
    upstream function's, not a local copy.
    """
    if cell is None:
        cell = coupled_reference_cell()
    row = coupled_operating_point(cell, j_mA_cm2)

    kappa_bulk = float(_physics_for(cell).solve_at_j(float(j_mA_cm2)).conductivity_S_m)
    kappa_eff_at_mean = bruggeman_conductivity(kappa_bulk, row["mean_void_fraction"])

    v_coupled = row["V_cell_coupled"]
    e_coupled = row["coupled_specific_energy_kWh_t"]
    return {
        "j_mA_cm2": row["j_mA_cm2"],
        "mean_void_fraction": row["mean_void_fraction"],
        "outlet_void_fraction": row["outlet_void_fraction"],
        "conductivity_penalty": row["conductivity_penalty"],
        "kappa_bulk_S_m": kappa_bulk,
        "kappa_eff_at_mean_void_S_m": float(kappa_eff_at_mean),
        "current_uniformity": row["current_uniformity"],
        "bubble_diameter_um": row["bubble_diameter_um"],
        "ohmic_penalty_V": row["gas_ohmic_penalty_V"],
        "ohmic_penalty_mV": row["gas_ohmic_penalty_V"] * 1000.0,
        "FE_shift_pp": row["FE_shift_pp"],
        "gas_share_of_V_cell_pct": 100.0 * row["gas_ohmic_penalty_V"] / v_coupled,
        "energy_contribution_kWh_t": row["energy_delta_kWh_t"],
        "gas_share_of_energy_pct": 100.0 * row["energy_delta_kWh_t"] / e_coupled,
        "hydrogen_flow_L_h_wet": row["hydrogen_flow_L_h_wet"],
        "gas_solve_converged": row["gas_solve_converged"],
        "measurement_priority": (
            "Bubble departure diameter and channel void fraction "
            "(gas_holdup.measurement_protocol()) — void enters the rise velocity "
            "quadratically, so this is the cheapest term that could move the coupled "
            "verdict."
        ),
        "flag": SCREENING_FLAG,
        "boundary_note": BOUNDARY_NOTE,
    }


# ─── The joint space, re-scanned under coupling ───────────────────────

def coupled_solve_window(
    *,
    contact_resistance_ohm_m2: Optional[float] = None,
) -> list[Dict[str, Any]]:
    """Re-scan #40's joint (j × gap × membrane × contact) space, coupled."""
    contacts = (
        (float(contact_resistance_ohm_m2),)
        if contact_resistance_ohm_m2 is not None
        else CONTACT_VALUES_OHM_M2
    )
    rows: list[Dict[str, Any]] = []
    for j in J_VALUES_MA_CM2:
        for gap in GAP_VALUES_M:
            for mem in MEMBRANE_VALUES_OHM_M2:
                for contact in contacts:
                    cell = coupled_reference_cell(
                        gap_m=gap,
                        contact_resistance_ohm_m2=contact,
                        membrane_area_resistance_ohm_m2=mem,
                    )
                    try:
                        rows.append(coupled_operating_point(cell, j))
                    except (ValueError, RuntimeError):
                        rows.append({
                            "j_mA_cm2": j,
                            "interelectrode_gap_m": gap,
                            "contact_resistance_ohm_m2": contact,
                            "membrane_area_resistance_ohm_m2": mem,
                            "valid": False,
                            "FE_uncoupled": 0.0,
                            "FE_coupled": 0.0,
                            "uncoupled_specific_energy_kWh_t": float("inf"),
                            "coupled_specific_energy_kWh_t": float("inf"),
                            "flag": SCREENING_FLAG,
                        })
    return rows


def coupled_energy_gate_reachable(
    cell: Optional[CoupledCell] = None,  # noqa: ARG001 - signature parity with #40
    *,
    contact_resistance_ohm_m2: Optional[float] = None,
) -> Dict[str, Any]:
    """Re-derive #40's reachability verdict with the gas coupling switched on.

    Scans the same joint space as ``operating_point_optimizer.solve_window``
    (j ∈ {150, 300} mA/cm², gap ∈ {1.5, 3.0} mm, membrane 3.0e-4 Ω·m²,
    contact ∈ {1.0e-4, 5.0e-4} Ω·m²), applying the same FE ≥ 70 % and
    transport-margin filters, and reports the uncoupled and coupled minima side
    by side so the degradation is explicit.

    ``cell`` is accepted for call-signature parity with #40; the scan builds
    its own RC-1 cells across the lever grid.
    """
    window = coupled_solve_window(contact_resistance_ohm_m2=contact_resistance_ohm_m2)
    uncoupled = _uncoupled_energy_gate_reachable(
        contact_resistance_ohm_m2=contact_resistance_ohm_m2
    )

    feasible = [
        r for r in window
        if r.get("valid") and r.get("FE_uncoupled", 0.0) >= FE_TARGET_MIN
    ]
    gate = SPECIFIC_ENERGY_TARGET_MAX_KWH_T

    if not feasible:
        return {
            "uncoupled_min_energy": uncoupled["min_energy_kWh_t"],
            "coupled_min_energy": float("inf"),
            "reachable_uncoupled": bool(uncoupled["reachable"]),
            "reachable_coupled": False,
            "best_combination_coupled": None,
            "best_combination_uncoupled": uncoupled["best_combination"],
            "energy_delta": float("inf"),
            "energy_gate_kWh_t": gate,
            "verdict": (
                "No valid coupled operating point meets FE >= 70 % inside the "
                "scanned joint space."
            ),
            "window": window,
            "flag": SCREENING_FLAG,
            "boundary_note": BOUNDARY_NOTE,
        }

    best_coupled = min(feasible, key=lambda r: r["coupled_specific_energy_kWh_t"])
    best_uncoupled_row = min(feasible, key=lambda r: r["uncoupled_specific_energy_kWh_t"])
    coupled_min = float(best_coupled["coupled_specific_energy_kWh_t"])
    uncoupled_min = float(best_uncoupled_row["uncoupled_specific_energy_kWh_t"])
    reachable_coupled = bool(coupled_min <= gate)
    reachable_uncoupled = bool(uncoupled_min <= gate)

    if reachable_coupled and reachable_uncoupled:
        verdict = (
            f"The #40 reachable operating point SURVIVES the coupled gas correction: "
            f"minimum energy moves {uncoupled_min:.1f} → {coupled_min:.1f} kWh/t Fe "
            f"({coupled_min - uncoupled_min:+.1f} kWh/t), still <= {gate:.0f} kWh/t, at "
            f"j={best_coupled['j_mA_cm2']:.0f} mA/cm², "
            f"gap={best_coupled['interelectrode_gap_m'] * 1e3:.1f} mm, "
            f"contact={best_coupled['contact_resistance_ohm_m2']:.1e} Ω·m²."
        )
    elif reachable_uncoupled and not reachable_coupled:
        verdict = (
            f"The #40 reachable verdict DOES NOT SURVIVE coupling: minimum energy moves "
            f"{uncoupled_min:.1f} → {coupled_min:.1f} kWh/t Fe "
            f"({coupled_min - uncoupled_min:+.1f} kWh/t), which exceeds the "
            f"{gate:.0f} kWh/t gate. The thesis is not yet backed; the gas/flow terms "
            f"become a measurement priority (gas_holdup.measurement_protocol())."
        )
    else:
        verdict = (
            f"Energy gate is NOT reachable either way in the scanned joint space: "
            f"uncoupled minimum {uncoupled_min:.1f} kWh/t, coupled minimum "
            f"{coupled_min:.1f} kWh/t, both against a {gate:.0f} kWh/t gate."
        )

    return {
        "uncoupled_min_energy": uncoupled_min,
        "coupled_min_energy": coupled_min,
        "reachable_uncoupled": reachable_uncoupled,
        "reachable_coupled": reachable_coupled,
        "best_combination_coupled": best_coupled,
        "best_combination_uncoupled": best_uncoupled_row,
        "energy_delta": coupled_min - uncoupled_min,
        "energy_delta_pct": 100.0 * (coupled_min - uncoupled_min) / uncoupled_min,
        "optimizer_min_energy_reference": uncoupled["min_energy_kWh_t"],
        "optimizer_reachable_reference": bool(uncoupled["reachable"]),
        "energy_gate_kWh_t": gate,
        "verdict": verdict,
        "window": window,
        "flag": SCREENING_FLAG,
        "boundary_note": BOUNDARY_NOTE,
    }


# ─── Closure diagnostic ───────────────────────────────────────────────

def coupling_closure(
    cell: Optional[CoupledCell] = None,
    j_mA_cm2: float = 150.0,
) -> Dict[str, Any]:
    """Ledger check: V_uncoupled + ohmic_penalty − V_coupled ≈ 0, and the
    null-gas case reproduces ``cell_physics`` exactly.

    This is the anti-double-counting guard: the coupling layer must add the gas
    ohmic term once and nothing else.
    """
    if cell is None:
        cell = coupled_reference_cell()
    on = coupled_operating_point(cell, j_mA_cm2)
    off = coupled_operating_point(cell, j_mA_cm2, gas_off=True)
    return {
        "j_mA_cm2": float(j_mA_cm2),
        "voltage_residual_V": abs(
            on["V_cell_uncoupled"] + on["gas_ohmic_penalty_V"] - on["V_cell_coupled"]
        ),
        "null_gas_ohmic_penalty_V": off["gas_ohmic_penalty_V"],
        "null_gas_energy_residual_kWh_t": abs(
            off["coupled_specific_energy_kWh_t"] - off["uncoupled_specific_energy_kWh_t"]
        ),
        "null_gas_void_fraction": off["mean_void_fraction"],
        "flag": SCREENING_FLAG,
    }


# ─── Report ───────────────────────────────────────────────────────────

def main() -> Dict[str, Any]:
    """Print the coupled-vs-uncoupled contrast, verdict and gas impact."""
    print("COUPLED CELL PHYSICS × GAS HOLD-UP — Level-0 → Level-1 boundary")
    print(f"Status: {SCREENING_FLAG}. NOT gate evidence; gates are measurement-only")
    print("        (models/process_gates.py). Coupled reduced-order models, no CFD.")
    print("-" * 78)

    reach = coupled_energy_gate_reachable()
    window = reach["window"]

    print("\nJoint space (same levers as #40), uncoupled vs coupled:")
    print("   j    gap   contact   | V_unc  V_cpl   ΔV(mV) | E_unc   E_cpl    ΔE   | valid")
    for r in sorted(window, key=lambda x: (x["j_mA_cm2"], x["interelectrode_gap_m"],
                                           x["contact_resistance_ohm_m2"])):
        if not r.get("valid"):
            print(f"  {r['j_mA_cm2']:4.0f} {r['interelectrode_gap_m'] * 1e3:4.1f}mm "
                  f"{r['contact_resistance_ohm_m2']:.0e} | "
                  f"{'invalid (transport limit)':>52} | False")
            continue
        print(
            f"  {r['j_mA_cm2']:4.0f} {r['interelectrode_gap_m'] * 1e3:4.1f}mm "
            f"{r['contact_resistance_ohm_m2']:.0e} | "
            f"{r['V_cell_uncoupled']:5.3f}  {r['V_cell_coupled']:5.3f}  "
            f"{r['gas_ohmic_penalty_V'] * 1e3:6.3f} | "
            f"{r['uncoupled_specific_energy_kWh_t']:7.1f} {r['coupled_specific_energy_kWh_t']:7.1f} "
            f"{r['energy_delta_kWh_t']:+6.2f} | True   [{SCREENING_FLAG}]"
        )

    print("\nReachability under coupling:")
    print(f"  uncoupled minimum energy : {reach['uncoupled_min_energy']:.1f} kWh/t Fe "
          f"(reachable: {reach['reachable_uncoupled']}) [{SCREENING_FLAG}]")
    print(f"  coupled   minimum energy : {reach['coupled_min_energy']:.1f} kWh/t Fe "
          f"(reachable: {reach['reachable_coupled']}) [{SCREENING_FLAG}]")
    print(f"  degradation (coupled − uncoupled): {reach['energy_delta']:+.2f} kWh/t "
          f"({reach['energy_delta_pct']:+.3f} %) [{SCREENING_FLAG}]")
    print(f"  {reach['verdict']} [{SCREENING_FLAG}]")

    best = reach["best_combination_coupled"]
    j_best = float(best["j_mA_cm2"])
    cell_best = coupled_reference_cell(
        gap_m=best["interelectrode_gap_m"],
        contact_resistance_ohm_m2=best["contact_resistance_ohm_m2"],
    )

    print("\nGas impact at the coupled best point:")
    impact = gas_impact_summary(cell_best, j_best)
    print(f"  mean void fraction        : {impact['mean_void_fraction'] * 100:.4f} % "
          f"(outlet {impact['outlet_void_fraction'] * 100:.4f} %) [{SCREENING_FLAG}]")
    print(f"  Bruggeman κ penalty       : {impact['conductivity_penalty']:.5f}× "
          f"(κ {impact['kappa_bulk_S_m']:.2f} → {impact['kappa_eff_at_mean_void_S_m']:.2f} S/m)")
    print(f"  current uniformity min/max: {impact['current_uniformity']:.5f}")
    print(f"  gas ohmic penalty         : {impact['ohmic_penalty_mV']:.4f} mV "
          f"({impact['gas_share_of_V_cell_pct']:.4f} % of V_cell) [{SCREENING_FLAG}]")
    print(f"  FE shift (bubble coupling): {impact['FE_shift_pp']:+.4f} pp [{SCREENING_FLAG}]")
    print(f"  energy contribution       : {impact['energy_contribution_kWh_t']:+.3f} kWh/t "
          f"({impact['gas_share_of_energy_pct']:+.4f} %) [{SCREENING_FLAG}]")
    print(f"  wet H₂ at the vent        : {impact['hydrogen_flow_L_h_wet']:.3f} L/h")

    closure = coupling_closure(cell_best, j_best)
    print("\nClosure (anti-double-counting):")
    print(f"  |V_uncoupled + ohmic_penalty − V_coupled| = "
          f"{closure['voltage_residual_V']:.2e} V")
    print(f"  null-gas case: void {closure['null_gas_void_fraction']:.2e}, "
          f"penalty {closure['null_gas_ohmic_penalty_V']:.2e} V, "
          f"energy residual {closure['null_gas_energy_residual_kWh_t']:.2e} kWh/t")

    print("\nAlso reported (full two-way coupling, for contrast only):")
    row_best = coupled_operating_point(cell_best, j_best)
    print(f"  coupled energy at the model's own coupled FE "
          f"({row_best['FE_coupled'] * 100:.3f} %): "
          f"{row_best['coupled_energy_with_FE_shift_kWh_t']:.1f} kWh/t Fe [{SCREENING_FLAG}]")
    print("  The headline coupled number above uses the uncoupled FE, so the gas "
          "correction\n  can only ever raise energy — it is never allowed to flatter the gate.")

    print(f"\n{BOUNDARY_NOTE}")
    print("Measurement that would close this: " + impact["measurement_priority"])
    return {"reachability": reach, "gas_impact": impact, "closure": closure}


if __name__ == "__main__":
    main()
