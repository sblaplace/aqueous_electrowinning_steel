"""Environmental-disturbance -> process-operating-point adapter (L0).

The crate layer (`models/crate.py`) turns weather/site into a *stability
verdict* and the operating twin (`models/operating_twin.py`) turns the same
observations into *trip/shutdown decisions*.  This module is the missing link:
it maps those measured/modelled environmental observations into **process
disturbance inputs** consumed by the cell thermal / Fe2+ / pH balances in
`models/bath_dynamics.step()`.

Consumption contract
--------------------
``disturbance_from_environment(env_state, crate_state)`` is a pure,
deterministic adapter: same inputs always produce the same
:class:`DisturbanceInputs`.  With no environmental data (or data for a
coupling-unaware environment) it returns ``enabled=False`` and all-zero
disturbances, so the EKF is byte-identical to the uncoupled case (the brief's
"coupling off by default" guarantee).  When real env/crate observations are
present, it returns ``enabled=True`` and physically-directional terms:

* **Ambient temperature** — the thermal balance's ``T_ambient`` (replaces the
  fixed ``T_ambient_C`` design-point default), which drives ambient heat loss.
* **Wind-driven convection** — a forced-convection heat-transfer coefficient
  that increases with wind gust, driving convective heat loss from the cell /
  reservoir surface.
* **Rain cooling** — an additional cooling term proportional to rainfall.
* **Ingress dilution** — flooding / ingress adds a dilution term that lowers
  bulk Fe2+ concentration and drags pH toward neutral rainwater.

No state vector or measurement model is touched; disturbances enter only as
control/auxiliary inputs.  This is L0 screening — the correlations are
first-principles-shape but unvalidated pending a real site survey.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

# ---------------------------------------------------------------------------
# Forced-convection correlation coefficients
# ---------------------------------------------------------------------------
# h_conv = H_CONV_BASE + H_CONV_WIND_K * gust_m_s ** H_CONV_WIND_EXP  [W/m2K]
# Natural-convection floor ~5 W/m2K; a 40 m/s gust gives ~5 + 3*40^0.7 ~= 45.
H_CONV_BASE = 5.0                 # W/m2K — natural convection floor
H_CONV_WIND_K = 3.0               # W/m2K per (m/s)^0.7
H_CONV_WIND_EXP = 0.7
# Rain cooling: W/m2 per mm/hr of rainfall.
RAIN_COOLING_W_M2_PER_MMHR = 0.5
# Ingress dilution characteristic rate (1/hr) when flooding.
INGRESS_DILUTION_PER_M_FLOOD = 0.10   # 1/hr per metre of flood depth
INGRESS_DILUTION_BASE = 0.05          # 1/hr when ingress detected, no flood


@dataclass
class DisturbanceInputs:
    """Process disturbance terms derived from environmental observations.

    ``enabled=False`` (the default) means "applies nothing" — the EKF step
    behaves exactly as if the coupling were absent.
    """

    T_ambient_C: float = 25.0
    h_conv_W_m2_K: float = 0.0
    rain_cooling_W_m2: float = 0.0
    ingress_dilution_rate_1_hr: float = 0.0
    enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "T_ambient_C": self.T_ambient_C,
            "h_conv_W_m2_K": self.h_conv_W_m2_K,
            "rain_cooling_W_m2": self.rain_cooling_W_m2,
            "ingress_dilution_rate_1_hr": self.ingress_dilution_rate_1_hr,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DisturbanceInputs":
        return cls(
            T_ambient_C=float(d.get("T_ambient_C", 25.0)),
            h_conv_W_m2_K=float(d.get("h_conv_W_m2_K", 0.0)),
            rain_cooling_W_m2=float(d.get("rain_cooling_W_m2", 0.0)),
            ingress_dilution_rate_1_hr=float(d.get("ingress_dilution_rate_1_hr", 0.0)),
            enabled=bool(d.get("enabled", False)),
        )


def _number(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if f == f else default  # NaN -> default


def _wind_gust(env_state: Mapping[str, Any], crate_state: Optional[Mapping[str, Any]]) -> float:
    """Resolve the 3-s gust (m/s) from env or crate observations."""
    v = env_state.get("wind_gust_m_s")
    if v is not None:
        return max(0.0, _number(v))
    if crate_state is not None:
        # Crate WindLoad nests under "wind" for a CrateConfig; gust_m_s directly.
        v = crate_state.get("gust_m_s")
        if v is None:
            wind = crate_state.get("wind")
            if isinstance(wind, Mapping):
                v = wind.get("gust_m_s")
        if v is not None:
            return max(0.0, _number(v))
    return 0.0


def _ambient_T(env_state: Mapping[str, Any], crate_state: Optional[Mapping[str, Any]]) -> float:
    """Resolve ambient temperature (C) from env, crate wind, or a default."""
    for key in ("T_ambient_C", "ambient_temperature_C", "temperature_C"):
        v = env_state.get(key)
        if v is not None:
            return float(v)
    if crate_state is not None:
        v = crate_state.get("temperature_C")
        if v is None and isinstance(crate_state.get("wind"), Mapping):
            v = crate_state["wind"].get("temperature_C")
        if v is not None:
            return float(v)
    return 25.0


def disturbance_from_environment(
    env_state: Optional[Mapping[str, Any]] = None,
    crate_state: Optional[Mapping[str, Any]] = None,
) -> DisturbanceInputs:
    """Map environmental / crate observations to process disturbance inputs.

    Pure and deterministic.  With no observations it returns the zero /
    ``enabled=False`` default so the coupling is a no-op.
    """
    env_state = dict(env_state or {})
    crate_state = dict(crate_state or {}) if crate_state else None

    wind = _wind_gust(env_state, crate_state)
    T_amb = _ambient_T(env_state, crate_state)
    rain = env_state.get("rain_intensity_mm_hr")
    rain = max(0.0, _number(rain)) if rain is not None else 0.0
    flood = env_state.get("flood_depth_m")
    flood = max(0.0, _number(flood)) if flood is not None else 0.0
    ingress = bool(env_state.get("ingress_detected", False))

    if wind <= 0 and rain <= 0 and flood <= 0 and not ingress:
        return DisturbanceInputs()

    h_conv = H_CONV_BASE + H_CONV_WIND_K * (wind ** H_CONV_WIND_EXP)
    rain_cooling = RAIN_COOLING_W_M2_PER_MMHR * rain
    dilution = INGRESS_DILUTION_PER_M_FLOOD * flood + (
        INGRESS_DILUTION_BASE if ingress else 0.0
    )

    return DisturbanceInputs(
        T_ambient_C=T_amb,
        h_conv_W_m2_K=h_conv,
        rain_cooling_W_m2=rain_cooling,
        ingress_dilution_rate_1_hr=dilution,
        enabled=True,
    )
