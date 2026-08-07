"""Pulsed electrodeposition Pareto optimization — parameter sweep + front analysis.

Sweeps the (j_peak × duty_cycle × frequency × waveform × mechanism) space using
existing screening models (PhaseIIICoDeposition, MechanicalPropertiesModel,
technoeconomic cost functions) and builds Pareto fronts for competing objectives.

No physics is reimplemented; frequency effects not captured by the base screening
models are handled via physically-motivated correction factors (see
``_frequency_grain_factor``).

Usage
-----
::

    from models.pulse_optimization import PulseOptimizationSweep
    sweep = PulseOptimizationSweep()
    results = sweep.run_full_sweep()         # returns DataFrame
    pareto = sweep.compute_pareto_fronts(results)
    recs   = sweep.recommend_operating_points(pareto)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .co_deposition import PhaseIIICoDeposition
from .electrochemistry import specific_energy_kWh_per_kg
from .mechanical_properties import (
    MechanicalPropertiesModel,
    estimate_grain_size_um,
)


# ─── Default parameter grid ─────────────────────────────────────────────

J_PEAK_VALUES = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0]      # mA/cm²
DUTY_CYCLE_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
FREQUENCY_VALUES = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]      # Hz
WAVEFORM_VALUES = ["pe", "pre"]
MECHANISM_VALUES = [
    "hydroxide_suppression",
    "intermediate_adsorption",
    "mixed_metal_intermediate",
]

# Representative cell voltage for energy cost calculation (V)
CELL_VOLTAGE_V = 2.5

# Electricity price for energy cost ($/kWh)
ELECTRICITY_PRICE = 0.04


def _frequency_grain_factor(frequency_Hz: float, ref_Hz: float = 10.0) -> float:
    """Screening correction for grain size vs pulse frequency.

    Higher frequency → shorter off-time per cycle but more frequent
    nucleation events per second.  Net effect at constant duty and j_peak:
    grain refines weakly with frequency (saturates above ~100 Hz).

    d_corrected = d_base · f(f)

    where f(ref) = 1.0 and f decreases weakly for f > ref.
    """
    if frequency_Hz <= 0:
        return 1.0
    ratio = frequency_Hz / ref_Hz
    # Sub-linear correction: finer grains at higher frequency,
    # saturating above ~100 Hz.  Calibrated so that:
    #   f(1 Hz)   ≈ 1.08   (slightly coarser)
    #   f(10 Hz)  = 1.00   (reference)
    #   f(100 Hz) ≈ 0.93   (moderately finer)
    #   f(1000 Hz)≈ 0.88   (approaches limit)
    return float(1.0 / (1.0 + 0.08 * np.log10(np.maximum(ratio, 0.01))))


def _frequency_ce_factor(frequency_Hz: float, ref_Hz: float = 10.0) -> float:
    """Screening correction for current efficiency vs pulse frequency.

    Higher frequency → more uniform surface concentration → slightly
    better mass-transport utilization → marginally higher CE.
    Effect is weak (±2-3%).
    """
    if frequency_Hz <= 0:
        return 1.0
    ratio = frequency_Hz / ref_Hz
    # Very weak enhancement: +1% per decade above ref, -1% per decade below
    return float(1.0 + 0.01 * np.log10(np.maximum(ratio, 0.01)))


def _frequency_carbon_factor(frequency_Hz: float, ref_Hz: float = 10.0) -> float:
    """Screening correction for carbon incorporation vs pulse frequency.

    Higher frequency → more uniform current distribution during on-time →
    slightly higher particle incorporation.  Very weak effect.
    """
    if frequency_Hz <= 0:
        return 1.0
    ratio = frequency_Hz / ref_Hz
    return float(1.0 + 0.005 * np.log10(np.maximum(ratio, 0.01)))


# ─── Pareto front utilities ─────────────────────────────────────────────


def is_non_dominated(
    objectives: np.ndarray,
    maximize: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Identify non-dominated points in a multi-objective set.

    Parameters
    ----------
    objectives : (N, M) array — each row is a point, each column an objective.
    maximize : (M,) boolean array — True for maximization, False for minimization.
               If None, all objectives are minimized.

    Returns
    -------
    mask : (N,) boolean array — True for non-dominated points.
    """
    n, m = objectives.shape
    if maximize is None:
        maximize = np.zeros(m, dtype=bool)

    # Transform so that all objectives are to be minimized
    obj = objectives.copy()
    for j in range(m):
        if maximize[j]:
            obj[:, j] = -obj[:, j]

    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for k in range(n):
            if k == i or not mask[k]:
                continue
            # k dominates i if k <= i in all objectives and k < i in at least one
            if np.all(obj[k] <= obj[i]) and np.any(obj[k] < obj[i]):
                mask[i] = False
                break
    return mask


def compute_pareto_front(
    df: pd.DataFrame,
    obj_min: List[str],
    obj_max: List[str],
) -> pd.DataFrame:
    """Extract Pareto front from a DataFrame.

    Parameters
    ----------
    df : DataFrame with objective columns.
    obj_min : Column names to minimize.
    obj_max : Column names to maximize.

    Returns
    -------
    Subset of df containing only non-dominated rows.
    """
    cols = obj_min + obj_max
    obj = df[cols].values.astype(float)
    maximize = np.array([False] * len(obj_min) + [True] * len(obj_max))
    mask = is_non_dominated(obj, maximize=maximize)
    return df.loc[mask].copy()


# ─── Main sweep engine ──────────────────────────────────────────────────


@dataclass
class PulseOptimizationSweep:
    """Full parameter sweep over pulsed electrodeposition operating space."""

    j_peak_values: List[float] = field(default_factory=lambda: list(J_PEAK_VALUES))
    duty_cycle_values: List[float] = field(default_factory=lambda: list(DUTY_CYCLE_VALUES))
    frequency_values: List[float] = field(default_factory=lambda: list(FREQUENCY_VALUES))
    waveform_values: List[str] = field(default_factory=lambda: list(WAVEFORM_VALUES))
    mechanism_values: List[str] = field(default_factory=lambda: list(MECHANISM_VALUES))

    cell_voltage_V: float = CELL_VOLTAGE_V
    electricity_price: float = ELECTRICITY_PRICE

    def grid_size(self) -> int:
        """Total number of parameter combinations."""
        return (
            len(self.j_peak_values)
            * len(self.duty_cycle_values)
            * len(self.frequency_values)
            * len(self.waveform_values)
            * len(self.mechanism_values)
        )

    def _build_point(
        self,
        j_peak: float,
        duty_cycle: float,
        frequency: float,
        waveform: str,
        mechanism: str,
    ) -> Dict[str, Any]:
        """Evaluate a single operating point using existing models."""
        # j_avg for PE: j_peak * duty; for PRE: slightly lower due to reverse
        if waveform == "pe":
            j_avg = j_peak * duty_cycle
        else:  # pre — reverse takes 20% of off-time at -0.2 * j_peak
            t_on_frac = duty_cycle
            t_off_frac = 1.0 - t_on_frac
            t_rev_frac = t_off_frac * 0.2
            j_avg = j_peak * t_on_frac + (-0.2 * j_peak) * t_rev_frac
            j_avg = max(j_avg, 1.0)  # floor

        # --- Co-deposition (alloy + carbon) ---
        codep = PhaseIIICoDeposition(
            mechanism_fe_ni=mechanism,  # type: ignore[arg-type]
            temperature_C=60.0,
        )
        codep_result = codep.run_at_current_pulsed(
            j_avg_mA_cm2=j_avg,
            j_peak_mA_cm2=j_peak,
            duty_cycle=duty_cycle,
            waveform=waveform,  # type: ignore[arg-type]
        )

        carbon_raw = codep_result["carbon_incorporation"]["predicted_carbon_wt_percent"]
        ni_wt = codep_result["alloy_kinetics"]["ni_wt_percent"]
        adjusted_ce = codep_result["integrated_metrics"]["adjusted_overall_current_efficiency_percent"]

        # Frequency corrections (screening)
        ce_pct = float(np.clip(adjusted_ce * _frequency_ce_factor(frequency), 0.0, 100.0))
        carbon_wt = carbon_raw * _frequency_carbon_factor(frequency)

        # --- Grain size ---
        wf_map = {"pe": "pe", "pre": "pre"}
        grain_base = estimate_grain_size_um(
            j_avg_mA_cm2=j_avg,
            j_peak_mA_cm2=j_peak,
            duty_cycle=duty_cycle,
            waveform=wf_map.get(waveform, "dc"),
            temperature_C=60.0,
        )
        grain_um = grain_base * _frequency_grain_factor(frequency)

        # --- Yield strength (Hall-Petch + SS + carbon dispersion) ---
        mech_model = MechanicalPropertiesModel()
        mech_result = mech_model.predict(
            j_avg_mA_cm2=j_avg,
            j_peak_mA_cm2=j_peak,
            duty_cycle=duty_cycle,
            waveform=wf_map.get(waveform, "dc"),
            temperature_C=60.0,
            ni_wt_percent=ni_wt,
            carbon_wt_percent=carbon_wt,
            current_efficiency_percent=ce_pct,
        )
        yield_MPa = mech_result.sigma_y_MPa
        uts_MPa = mech_result.uts_MPa

        # --- Energy cost ($/kg Fe) ---
        ce_frac = ce_pct / 100.0
        if ce_frac > 0.01:
            energy_kWh_per_kg = specific_energy_kWh_per_kg(
                self.cell_voltage_V, ce_frac
            )
            energy_cost_per_kg = energy_kWh_per_kg * self.electricity_price
        else:
            energy_kWh_per_kg = float("inf")
            energy_cost_per_kg = float("inf")

        return {
            "j_peak_mA_cm2": j_peak,
            "duty_cycle": duty_cycle,
            "frequency_Hz": frequency,
            "waveform": waveform,
            "mechanism": mechanism,
            "j_avg_mA_cm2": round(j_avg, 2),
            "grain_size_um": round(grain_um, 4),
            "yield_strength_MPa": round(yield_MPa, 1),
            "uts_MPa": round(uts_MPa, 1),
            "current_efficiency_pct": round(ce_pct, 2),
            "carbon_wt_pct": round(carbon_wt, 3),
            "energy_cost_USD_per_kg": round(energy_cost_per_kg, 4),
            "ni_wt_pct": round(ni_wt, 2),
            "elongation_pct": round(mech_result.elongation_pct, 1),
        }

    def run_full_sweep(self, progress: bool = False) -> pd.DataFrame:
        """Run the complete parameter grid sweep.

        Returns DataFrame with one row per combination and all objective columns.
        """
        combos = list(product(
            self.j_peak_values,
            self.duty_cycle_values,
            self.frequency_values,
            self.waveform_values,
            self.mechanism_values,
        ))
        records: List[Dict[str, Any]] = []
        for idx, (jp, dc, f, wf, mech) in enumerate(combos):
            if progress and idx % 100 == 0:
                print(f"  sweep: {idx}/{len(combos)}")
            try:
                rec = self._build_point(jp, dc, f, wf, mech)
                records.append(rec)
            except Exception as e:
                # Skip points that fail convergence (rare edge cases)
                if progress:
                    print(f"  skip ({jp}, {dc}, {f}, {wf}, {mech}): {e}")
                continue
        return pd.DataFrame(records)

    def compute_pareto_fronts(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Compute three Pareto fronts from sweep results.

        Front 1: Min grain size vs max current efficiency
        Front 2: Max yield strength vs min energy cost
        Front 3: Max carbon wt% vs min grain size

        Returns dict mapping front name → DataFrame of non-dominated rows.
        """
        # Filter out infinite energy costs
        valid = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["grain_size_um", "current_efficiency_pct",
                     "yield_strength_MPa", "energy_cost_USD_per_kg",
                     "carbon_wt_pct"]
        )

        fronts: Dict[str, pd.DataFrame] = {}

        # Front 1: min grain vs max FE
        fronts["grain_vs_efficiency"] = compute_pareto_front(
            valid,
            obj_min=["grain_size_um"],
            obj_max=["current_efficiency_pct"],
        )

        # Front 2: max strength vs min energy cost
        fronts["strength_vs_energy"] = compute_pareto_front(
            valid,
            obj_min=["energy_cost_USD_per_kg"],
            obj_max=["yield_strength_MPa"],
        )

        # Front 3: max carbon vs min grain
        fronts["carbon_vs_grain"] = compute_pareto_front(
            valid,
            obj_min=["grain_size_um"],
            obj_max=["carbon_wt_pct"],
        )

        return fronts

    def recommend_operating_points(
        self,
        fronts: Dict[str, pd.DataFrame],
        n_points: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend operating points from Pareto analysis.

        Strategy: pick points that appear on multiple fronts (breadth),
        then fill with extreme points from individual fronts for coverage.
        """
        # Score each point by how many fronts it appears on
        all_fronts_df = pd.concat(fronts.values()).drop_duplicates()
        # Count front membership by checking which rows from the full sweep appear
        # on each front
        front_ids = {}
        for name, fdf in fronts.items():
            for idx in fdf.index:
                front_ids.setdefault(idx, set()).add(name)

        # Build recommendations
        recs: List[Dict[str, Any]] = []

        # 1. Points on 2+ fronts (compromise solutions)
        multi = [idx for idx, names in front_ids.items() if len(names) >= 2]
        for idx in multi[:2]:
            row = all_fronts_df.loc[idx] if idx in all_fronts_df.index else None
            if row is not None:
                recs.append({
                    "type": "compromise",
                    "fronts": sorted(front_ids[idx]),
                    **{c: row[c] for c in all_fronts_df.columns},
                })

        # 2. Best grain-size point from grain_vs_efficiency
        gf = fronts["grain_vs_efficiency"]
        if len(gf) > 0:
            best_grain = gf.loc[gf["grain_size_um"].idxmin()]
            recs.append({
                "type": "finest_grain",
                "front": "grain_vs_efficiency",
                **{c: best_grain[c] for c in gf.columns},
            })

        # 3. Best strength/energy from strength_vs_energy
        sf = fronts["strength_vs_energy"]
        if len(sf) > 0:
            best_se = sf.loc[(sf["yield_strength_MPa"] / sf["energy_cost_USD_per_kg"]).idxmax()]
            recs.append({
                "type": "best_strength_per_energy",
                "front": "strength_vs_energy",
                **{c: best_se[c] for c in sf.columns},
            })

        # 4. Best carbon from carbon_vs_grain
        cf = fronts["carbon_vs_grain"]
        if len(cf) > 0:
            best_c = cf.loc[cf["carbon_wt_pct"].idxmax()]
            recs.append({
                "type": "highest_carbon",
                "front": "carbon_vs_grain",
                **{c: best_c[c] for c in cf.columns},
            })

        # 5. Fill to n_points with highest yield strength from strength_vs_energy
        if len(recs) < n_points and len(sf) > 0:
            already = {r.get("j_peak_mA_cm2") for r in recs}
            for _, row in sf.sort_values("yield_strength_MPa", ascending=False).iterrows():
                if row["j_peak_mA_cm2"] not in already:
                    recs.append({
                        "type": "high_strength",
                        "front": "strength_vs_energy",
                        **{c: row[c] for c in sf.columns},
                    })
                    already.add(row["j_peak_mA_cm2"])
                    if len(recs) >= n_points:
                        break

        # Deduplicate by operating parameters
        seen = set()
        unique_recs = []
        for r in recs:
            key = (r["j_peak_mA_cm2"], r["duty_cycle"], r["frequency_Hz"],
                   r["waveform"], r["mechanism"])
            if key not in seen:
                seen.add(key)
                unique_recs.append(r)

        return unique_recs[:n_points]
