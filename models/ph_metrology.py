"""Operational pH is not Pitzer pH: a measurement-bias screen (V6 §5.1).

Gap / physics / impact / implementation
----------------------------------------
The thermodynamic models use ``pH_Pitzer = -log10(a_H+)``.  The laboratory
will instead report a glass-electrode reading through a KCl bridge.  In the
concentrated FeSO4/Na2SO4 brine this introduces a liquid-junction potential,
a single-ion convention offset, temperature-dependent Nernst slope, and a
bridge-clogging drift.  A lovely R² in a HER fit cannot identify those errors:
they alias into the fitted pH exponent and exchange current.

This deliberately small L1 module uses transport's *live ionic diffusivities*
to form Nernst--Einstein conductance weights and a Planck--Henderson-style
log-concentration junction screen.  It converts between an operational meter
reading and the Pitzer convention, reports the correction and its uncertainty,
and makes the required two-point concentration-cell check explicit.

It is not a Pitzer calculation or a certified pH method.  Ion pairing,
activity gradients in the bridge, electrode asymmetry, colloid transport and
individual glass-membrane calibration are shelved physics.  The comparative
result -- whether a proposed RDE pH trend survives a 0.2--0.5 pH-unit
systematic -- is decision-grade; the reported decimal is not.

References: Henderson (1907); Bates, *Determination of pH* (1973); IUPAC
Recommendations on pH measurement (2002); CHEM_PHYS_IMPROVEMENTS_V6 §5.1.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .anchors import get_anchor
from .electrochemistry import FARADAY, R_GAS
from .thermodynamic_constants import D_CL_25, D_FE2_25, D_H_25, D_NA_25, D_SO4_25

SCREENING_FLAG = "unvalidated (L1)"

# Species tuple: name, charge, live 25-C infinite-dilution diffusivity.
# K+ is intentionally given the Na+ proxy: the bridge is a screening boundary,
# not a claim that K+ and Na+ have identical concentrated-brine mobilities.
_SPECIES = (
    ("H+", 1, D_H_25),
    ("Fe2+", 2, D_FE2_25),
    ("Na+", 1, D_NA_25),
    ("K+", 1, D_NA_25),
    ("Cl-", -1, D_CL_25),
    ("SO4--", -2, D_SO4_25),
)


def _a(key: str) -> float:
    return float(get_anchor(key).value)


@dataclass(frozen=True)
class BathComposition:
    """Concentrations in mol/L; defaults match the divided-cell sulfate bath."""

    fe2_M: float = 1.5
    na2so4_M: float = 0.5
    h_M: float = 0.01

    @property
    def ionic_strength_M(self) -> float:
        # SO4 balances FeSO4 + Na2SO4; the acid counterion is omitted from this
        # compact screen, so this is a stated lower-bound matrix proxy.
        so4 = self.fe2_M + self.na2so4_M
        return 0.5 * (4 * self.fe2_M + self.na2so4_M + self.h_M + 4 * so4)

    def concentrations(self) -> Dict[str, float]:
        return {"H+": self.h_M, "Fe2+": self.fe2_M, "Na+": 2 * self.na2so4_M,
                "K+": 0.0, "Cl-": 0.0, "SO4--": self.fe2_M + self.na2so4_M}


@dataclass(frozen=True)
class Bridge:
    """KCl bridge; ``age_days`` invokes the anchored clogging-drift proxy."""

    kcl_M: float = 3.0
    age_days: float = 0.0

    def concentrations(self) -> Dict[str, float]:
        return {"H+": 0.0, "Fe2+": 0.0, "Na+": 0.0, "K+": self.kcl_M,
                "Cl-": self.kcl_M, "SO4--": 0.0}


@dataclass(frozen=True)
class MeasurementBias:
    """Bias ledger consumed by a calibration fit or run-record sidecar."""

    temperature_C: float
    ionic_strength_M: float
    junction_mV: float
    bridge_drift_mV: float
    single_ion_offset_pH: float
    nernst_slope_mV_pH: float
    total_offset_pH: float
    uncertainty_pH: float
    two_point_check_required: bool
    warning: str


def nernst_slope_mV_pH(temperature_C: float = 60.0) -> float:
    """Ideal glass-electrode magnitude, 2.303 RT/F, at the bath temperature."""
    return 1000.0 * math.log(10.0) * R_GAS * (temperature_C + 273.15) / FARADAY


def _conductance_weights(concentrations: Dict[str, float]) -> Dict[str, float]:
    """Nernst--Einstein weights proportional to z² D c, normalized."""
    raw = {name: z * z * d * max(concentrations.get(name, 0.0), 0.0)
           for name, z, d in _SPECIES}
    total = sum(raw.values())
    return {name: value / total if total else 0.0 for name, value in raw.items()}


def junction_potential_mV(bath: BathComposition = BathComposition(),
                          bridge: Bridge = Bridge(), temperature_C: float = 60.0) -> float:
    """Planck--Henderson screening potential, bridge minus bath (mV).

    Conductance fractions are averaged across the two solutions.  A small
    concentration floor represents the diffuse overlap zone; it prevents an
    infinite ideal-solution potential when a bridge ion is absent from the
    bath.  This is exactly where a concentration-cell measurement replaces
    the proxy.
    """
    left, right = bridge.concentrations(), bath.concentrations()
    wl, wr = _conductance_weights(left), _conductance_weights(right)
    floor = _a("PH_METROLOGY_OVERLAP_M")
    dimensionless = 0.0
    for name, z, _d in _SPECIES:
        weight = 0.5 * (wl[name] + wr[name])
        dimensionless += weight / z * math.log(
            max(left[name], floor) / max(right[name], floor))
    return 1000.0 * R_GAS * (temperature_C + 273.15) / FARADAY * dimensionless


def bridge_drift_mV(age_days: float) -> float:
    """Positive log-time Fe(OH)3-clogging drift magnitude; sign is empirical."""
    if age_days < 0:
        raise ValueError("age_days must be non-negative")
    return _a("PH_METROLOGY_DRIFT_MV_DAY") * math.log1p(age_days)


def evaluate_bias(bath: BathComposition = BathComposition(), bridge: Bridge = Bridge(),
                  temperature_C: float = 60.0,
                  single_ion_offset_pH: Optional[float] = None) -> MeasurementBias:
    """Return the pH-convention ledger at the stated bath/meter condition."""
    if temperature_C <= -273.15:
        raise ValueError("temperature_C must exceed absolute zero")
    offset = (_a("PH_METROLOGY_SINGLE_ION_OFFSET_PH") if single_ion_offset_pH is None
              else single_ion_offset_pH)
    junction = junction_potential_mV(bath, bridge, temperature_C)
    slope = nernst_slope_mV_pH(temperature_C)
    drift = bridge_drift_mV(bridge.age_days)
    # E_j and clogging are voltage errors expressed on the local Nernst slope.
    correction = offset + (junction + drift) / slope
    uncertainty = math.sqrt(_a("PH_METROLOGY_JUNCTION_SIGMA_MV") ** 2 +
                            _a("PH_METROLOGY_SINGLE_ION_SIGMA_PH") ** 2 * slope ** 2) / slope
    return MeasurementBias(
        temperature_C=temperature_C, ionic_strength_M=bath.ionic_strength_M,
        junction_mV=junction, bridge_drift_mV=drift, single_ion_offset_pH=offset,
        nernst_slope_mV_pH=slope, total_offset_pH=correction,
        uncertainty_pH=uncertainty, two_point_check_required=True,
        warning="Run HCl/LiCl concentration-cell checks before fitting a pH exponent.",
    )


def operational_to_pitzer(pH_operational: float, bias: MeasurementBias) -> float:
    """Convert a meter reading to the module's Pitzer-convention pH estimate."""
    return pH_operational - bias.total_offset_pH


def pitzer_to_operational(pH_pitzer: float, bias: MeasurementBias) -> float:
    """Predict what the operational meter would report under this bias ledger."""
    return pH_pitzer + bias.total_offset_pH


def two_point_protocol() -> List[Dict[str, Any]]:
    """Minimal $0-hardware concentration-cell check recorded with RDE runs."""
    return [
        {"step": 1, "action": "Measure bridge potential in 0.01 M HCl / LiCl check solution."},
        {"step": 2, "action": "Repeat in 0.10 M HCl / LiCl; record temperature and bridge age."},
        {"step": 3, "action": "Compare observed delta-mV with junction_potential_mV; carry residual into fit covariance."},
    ]


def model_scope() -> Dict[str, Any]:
    return {"screening_flag": SCREENING_FLAG,
            "live_derivations": ["thermodynamic_constants D_H/D_Fe2/D_Na/D_SO4 used as Nernst--Einstein conductance weights", "electrochemistry R_GAS and FARADAY set temperature-dependent glass slope"],
            "screening_proxies_anchored": ["bridge overlap concentration floor", "log-time colloid-clogging drift", "single-ion convention offset and junction uncertainty"],
            "out_of_scope": ["Pitzer activity calculation and ion pairing", "actual K+/Cl- concentrated-brine mobilities", "glass membrane asymmetry, buffer calibration and colloid transport", "automatic wiring into calibration.py / kinetics_fit_pipeline.py (no common fit API yet)"]}


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Operational-pH / Pitzer measurement-bias screen (V6 §5.1).")
    parser.add_argument("--pH", type=float, default=2.0, help="operational meter reading")
    parser.add_argument("--temperature", type=float, default=60.0)
    parser.add_argument("--bridge-age-days", type=float, default=0.0)
    args = parser.parse_args(argv)
    bias = evaluate_bias(bridge=Bridge(age_days=args.bridge_age_days), temperature_C=args.temperature)
    print(f"ph_metrology [{SCREENING_FLAG}]")
    print(f"  I proxy {bias.ionic_strength_M:.2f} M; slope {bias.nernst_slope_mV_pH:.2f} mV/pH")
    print(f"  junction {bias.junction_mV:+.1f} mV; drift {bias.bridge_drift_mV:+.1f} mV")
    print(f"  operational → Pitzer: {args.pH:.3f} → {operational_to_pitzer(args.pH, bias):.3f}")
    print(f"  correction {bias.total_offset_pH:+.3f} ± {bias.uncertainty_pH:.3f} pH; {bias.warning}")


if __name__ == "__main__":  # pragma: no cover
    main()
