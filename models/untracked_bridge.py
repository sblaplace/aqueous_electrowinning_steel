"""
Bridge connections for untracked #2, #4, #5 → existing pipeline.

Not on board A/B.  This file is the minimal wiring layer: it shows exactly
how to call the new modules from ``transport.py`` / ``kinetics.py`` /
``closed_loop.py`` / ``adhesion_peel.py`` without restructuring them.

Usage pattern: import bridge, call at timestep / calibration step.
"""
from __future__ import annotations

from typing import Dict, Optional

# Untracked modules (drafted this session)
from . import microph_buffer, additive_aging, substrate_passivation

# Existing modules (import guarded — may fail without full env)
try:
    from . import diffusion_layer_1d, bath_dynamics, adhesion_peel, closed_loop
except Exception:
    diffusion_layer_1d = None  # type: ignore
    bath_dynamics = None
    adhesion_peel = None
    closed_loop = None


def bridge_microph_to_transport(
    pH_bulk: float,
    C_fe2_bulk_M: float,
    C_so4_bulk_M: float,
    j_local_A_m2: float,
    T_C: float = 60.0,
    film_thick_m: float = 50e-6,
) -> Dict[str, float]:
    """
    Call site in ``models/diffusion_layer_1d.py`` or ``transport.py``:
    At each film-step / timestep, compute local surface conditions
    and inject the correction dict into the Nernst–Planck solve.
    """
    p_surf, beta, risk, precip = microph_buffer.compute_surface_pH(
        pH_bulk, C_fe2_bulk_M, C_so4_bulk_M, j_local_A_m2, T_C, film_thickness_m=film_thick_m
    )
    feed = microph_buffer.feed_to_diffusion_layer(p_surf, beta, j_local_A_m2, C_fe2_bulk_M, C_so4_bulk_M)
    # Example insertion into diffusion_layer_1d (conceptual):
    # diffusion_layer_1d.update_precipitation_sink(feed["precipitation_sink_M_s"])
    # diffusion_layer_1d.scale_diffusivity(feed["effective_diffusivity_scale"])
    return feed


def bridge_additive_to_closed_loop(
    C_initial_M: float,
    t_hours: float,
    additive_id: str = "saccharin",
    j_local_A_cm2: float = 0.2,
    pH_surf: float = 2.5,
    T_C: float = 60.0,
) -> Dict[str, float]:
    """
    Call site in ``models/closed_loop.py`` or ``bath_dynamics.py``:
    At each CSTR timestep, decay the leveler concentration and
    pass the effective surface coverage to the morphology/feed policy.
    """
    decay = additive_aging.decay_rate_per_hour(T_C, j_local_A_cm2, pH_surf, additive_id)
    C_eff = additive_aging.effective_leveler_coverage(
        C_initial_M, t_hours, decay_dict=decay,
        replenishment_M_per_hour=additive_aging.DEFAULT_REPL_RATE_MOL_L_DAY / 24.0
    )
    feed = additive_aging.feed_to_bath_dynamics(C_eff, additive_id, C_initial_M, t_hours)
    # Conceptual insertion into closed_loop / bath_dynamics:
    # bath_dynamics.update_additive_concentration(additive_id, C_eff)
    # closed_loop.update_bleed_policy(feed["half_life_h"])
    return feed


def bridge_substrate_to_adhesion(
    t_hours: float,
    T_C: float = 60.0,
    E_V_vs_SHE: float = 1.2,
    pH: float = 1.5,
    pO2_bar: float = 0.21,
    exposure_mode: str = "anodic",
) -> Dict[str, float]:
    """
    Call site in ``models/adhesion_peel.py`` / ``internal_stress.py``:
    At campaign-time milestones (start, 100 h, 500 h, 1000 h),
    update critical peel thickness and fracture energy.
    """
    delta_ox, k_p, rate = substrate_passivation.oxide_growth_parabolic(
        t_hours, T_C, E_V_vs_SHE, pH, pO2_bar, exposure_mode
    )
    feed = substrate_passivation.feed_to_adhesion_peel(delta_ox)
    # Conceptual insertion:
    # adhesion_peel.update_critical_thickness(feed["critical_peel_thickness_nm"])
    # adhesion_peel.update_fracture_energy(feed["interfacial_fracture_energy_J_m2"])
    # internal_stress.update_peel_viability(feed["peel_viable"])
    return feed
