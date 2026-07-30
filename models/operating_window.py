"""Pre-lab operating window classifier and parameter boundary mapper.

Evaluates multi-dimensional operational space (current density, pH, temperature,
iron concentration, boundary layer thickness) against coupled kinetic,
transport, thermodynamic precipitation, voltage, and energy constraints to map
safe experimental windows before wet-lab trials.

References:
- SIDERWIN Project Deliverables (2021). Operating window for alkaline iron electrowinning.
- AWARE Process (2024). High-efficiency acidic electrowinning operating parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np

from .kinetics import DepositionKinetics
from .boundary_layer import CathodeBoundaryLayer
from .speciation import SolutionComposition, solve_speciation
from .electrochemistry import CellVoltageModel


# Status classification codes
STATUS_PASS = 0
FAIL_LOW_FE = 1
FAIL_HYDROXIDE_PRECIPITATION = 2
FAIL_HIGH_VOLTAGE = 3
FAIL_MASS_TRANSFER = 4
FAIL_THERMAL_MEMBRANE = 5


@dataclass
class OperatingWindowConstraints:
    """Pre-lab threshold criteria for viable electrowinning."""
    min_FE: float = 0.70              # Minimum acceptable Faradaic efficiency (70%)
    max_V_cell: float = 3.50          # V, maximum cell voltage (corresponds to ~4000 kWh/t Fe)
    max_specific_energy_kWh_t: float = 4000.0 # kWh/t Fe maximum energy
    max_j_ratio_lim: float = 0.90     # j / j_lim maximum ratio to avoid dendritic powder
    max_membrane_T_C: float = 75.0    # °C, maximum safe membrane temperature


def evaluate_operating_point(
    j_mA_cm2: float,
    pH_bulk: float,
    T_C: float,
    c_Fe2_M: float = 1.0,
    delta_um: float = 30.0,
    fe_i0: float = 0.05,
    her_i0: float = 1.0e-5,
    divided_cell: bool = False,
    gap_cm: float = 0.5,
    constraints: OperatingWindowConstraints = OperatingWindowConstraints()
) -> Dict[str, Any]:
    """Evaluate a single candidate operating point against all physical constraints.
    
    Returns status code, pass boolean, failure reasons, and detailed metrics.
    """
    reasons = []
    status = STATUS_PASS
    
    # 1. Thermal constraint check
    if T_C > constraints.max_membrane_T_C:
        reasons.append(f"Temperature {T_C:.1f} °C exceeds membrane limit {constraints.max_membrane_T_C} °C")
        status = FAIL_THERMAL_MEMBRANE

    # 2. Speciation & Thermodynamic precipitation limit
    spec_comp = SolutionComposition(c_FeSO4=c_Fe2_M, c_Na2SO4=0.5, c_H2SO4=10.0**(-pH_bulk), T_C=T_C)
    spec_res = solve_speciation(spec_comp)
    pH_precip = spec_res["pH_precip_Fe_OH2"]
    
    # 3. Boundary layer cathode surface pH & Fe(OH)2 supersaturation calculation
    bl_model = CathodeBoundaryLayer(
        bulk_pH=pH_bulk,
        fe_conc_M=c_Fe2_M,
        boundary_layer_m=delta_um * 1e-6,
        temperature_C=T_C,
        fe_i0=fe_i0,
        her_i0=her_i0
    )
    bl_res = bl_model.solve(j_mA_cm2)
    pH_surface = bl_res.surface_pH
    
    # Hydroxide precipitation occurs when Fe(OH)2 supersaturation >= 1.0 or active precipitation flag
    if bl_res.precipitation_active or bl_res.feoh2_supersaturation >= 1.0:
        reasons.append(f"Fe(OH)2 precipitation active (supersaturation = {bl_res.feoh2_supersaturation:.2f} >= 1.0)")
        if status == STATUS_PASS:
            status = FAIL_HYDROXIDE_PRECIPITATION
            
    # 4. Kinetics & Faradaic Efficiency
    kin_model = DepositionKinetics(
        pH=pH_bulk,
        temperature_C=T_C,
        fe_conc_M=c_Fe2_M,
        boundary_layer_m=delta_um * 1e-6,
        fe_i0=fe_i0,
        her_i0=her_i0
    )
    FE = kin_model.efficiency_at_current(j_mA_cm2)
    j_lim = kin_model.i_lim / 10.0 # Convert A/m^2 to mA/cm^2
    E_cath = kin_model.potential_at_current(j_mA_cm2)
    eta_c = abs(E_cath - kin_model.fe_E_eq)
    
    # Check Mass transfer limit
    if j_lim > 0 and (j_mA_cm2 / j_lim) > constraints.max_j_ratio_lim:
        reasons.append(f"Current density {j_mA_cm2:.1f} mA/cm2 exceeds {constraints.max_j_ratio_lim*100:.0f}% of j_lim ({j_lim:.1f} mA/cm2)")
        if status == STATUS_PASS:
            status = FAIL_MASS_TRANSFER
            
    # Check Faradaic Efficiency
    if FE < constraints.min_FE:
        reasons.append(f"FE ({FE*100:.1f}%) below minimum requirement ({constraints.min_FE*100:.1f}%)")
        if status == STATUS_PASS:
            status = FAIL_LOW_FE

    # 5. Cell voltage & Specific energy
    v_model = CellVoltageModel(
        temperature_C=T_C,
        fe2_conc_M=c_Fe2_M,
        eta_cathode=eta_c,
        interelectrode_gap_m=gap_cm / 100.0,
        j_operating_mA_cm2=j_mA_cm2,
        divided_cell=divided_cell
    )
    V_cell = v_model.V_cell
    
    # Specific energy E = 959.9 * V_cell / FE (kWh/t Fe)
    FE_eff = max(0.01, FE)
    specific_energy = 959.9 * V_cell / FE_eff
    
    if V_cell > constraints.max_V_cell or specific_energy > constraints.max_specific_energy_kWh_t:
        reasons.append(f"V_cell ({V_cell:.2f} V) or energy ({specific_energy:.0f} kWh/t) exceeds threshold")
        if status == STATUS_PASS:
            status = FAIL_HIGH_VOLTAGE

    is_pass = (status == STATUS_PASS)
    
    return {
        "j_mA_cm2": j_mA_cm2,
        "pH_bulk": pH_bulk,
        "T_C": T_C,
        "c_Fe2_M": c_Fe2_M,
        "status_code": status,
        "is_pass": is_pass,
        "reasons": reasons,
        "FE": float(FE),
        "pH_surface": float(pH_surface),
        "pH_precip": float(pH_precip),
        "feoh2_supersaturation": float(bl_res.feoh2_supersaturation),
        "V_cell": float(V_cell),
        "specific_energy_kWh_t": float(specific_energy),
        "j_lim_mA_cm2": float(j_lim),
        "j_ratio_lim": float(j_mA_cm2 / max(1e-3, j_lim)),
    }


def map_2d_operating_window(
    param_x_name: str = "j_mA_cm2",
    x_vals: np.ndarray = np.linspace(50, 500, 10),
    param_y_name: str = "pH_bulk",
    y_vals: np.ndarray = np.linspace(1.5, 4.0, 10),
    fixed_params: Dict[str, float] = None,
    constraints: OperatingWindowConstraints = OperatingWindowConstraints()
) -> Dict[str, Any]:
    """Map 2D grid of operating conditions and classify each point."""
    if fixed_params is None:
        fixed_params = {"T_C": 50.0, "c_Fe2_M": 1.0, "delta_um": 30.0, "j_mA_cm2": 200.0, "pH_bulk": 2.5}
        
    nx = len(x_vals)
    ny = len(y_vals)
    
    pass_mask = np.zeros((ny, nx), dtype=bool)
    status_grid = np.zeros((ny, nx), dtype=int)
    FE_grid = np.zeros((ny, nx), dtype=float)
    V_cell_grid = np.zeros((ny, nx), dtype=float)
    pH_surf_grid = np.zeros((ny, nx), dtype=float)
    energy_grid = np.zeros((ny, nx), dtype=float)
    
    for i, y in enumerate(y_vals):
        for j_idx, x in enumerate(x_vals):
            p = fixed_params.copy()
            p[param_x_name] = float(x)
            p[param_y_name] = float(y)
            
            res = evaluate_operating_point(
                j_mA_cm2=p["j_mA_cm2"],
                pH_bulk=p["pH_bulk"],
                T_C=p["T_C"],
                c_Fe2_M=p["c_Fe2_M"],
                delta_um=p.get("delta_um", 30.0),
                constraints=constraints
            )
            
            pass_mask[i, j_idx] = res["is_pass"]
            status_grid[i, j_idx] = res["status_code"]
            FE_grid[i, j_idx] = res["FE"]
            V_cell_grid[i, j_idx] = res["V_cell"]
            pH_surf_grid[i, j_idx] = res["pH_surface"]
            energy_grid[i, j_idx] = res["specific_energy_kWh_t"]
            
    pass_fraction = float(np.mean(pass_mask))
    
    return {
        "x_name": param_x_name,
        "x_vals": x_vals,
        "y_name": param_y_name,
        "y_vals": y_vals,
        "pass_mask": pass_mask,
        "status_grid": status_grid,
        "FE_grid": FE_grid,
        "V_cell_grid": V_cell_grid,
        "pH_surf_grid": pH_surf_grid,
        "energy_grid": energy_grid,
        "pass_fraction": pass_fraction,
    }
