"""
Capacitive double-layer charging dynamics and high-frequency wave distortion in pulse plating.

Physics and Chemistry
---------------------
Pulse and pulse-reverse electrodeposition models (e.g. :mod:`models.pulse`)
generally assume an ideal rectangular current waveform where current steps
instantaneously between cathodic, reverse, and rest levels.  In a real
electrochemical cell, however, the electrode interface consists of an
interfacial double-layer capacitance C_dl (20–60 µF/cm²) in parallel with the
Faradaic charge-transfer resistance R_ct, in series with the uncompensated
solution resistance R_ohm.

When a potential or galvanostatic pulse is applied:
1. The total current splits into capacitive displacement and Faradaic current:
     i_total(t) = i_C(t) + i_F(t) = C_dl (dη/dt) + i_F(t)

2. The interfacial charging time constant is:
     τ_cell = R_ohm · C_dl   (or τ_eff = (R_ohm · R_ct / (R_ohm + R_ct)) · C_dl)
   For typical iron electrowinning baths (R_ohm ≈ 0.5–3.0 Ω·cm², C_dl ≈ 30 µF/cm²),
   τ_cell ranges from **0.05 to 1.0 ms**.

3. **High-frequency pulse attenuation**: If the pulse on-time t_on is comparable to
   or shorter than τ_cell (e.g. f > 1 kHz, t_on < 0.5 ms), the double layer does
   not fully charge during the ON period.  The Faradaic current never reaches the
   commanded peak current j_peak, and Faradaic current continues to flow during the
   nominal OFF pause.  The waveform smooths into an attenuated ripple-DC,
   destroying the grain-refining high-instantaneous-overpotential benefit.

4. **Cutoff frequency**:
     f_cutoff = 1 / (2π · R_ohm · C_dl)
   Sets the rigorous upper limit on pulse frequency for any given cell geometry
   and electrolyte conductivity.

References
----------
* Ibl, N. (1980). "Some theoretical aspects of pulse electrolysis." Surf.
  Technol., 10(2), 81–104.
* Puippe, J. C., & Leaman, F. (1986). "Theory and Practice of Pulse Plating."
  American Electroplaters and Surface Finishers Society (AESF).
* Landolt, D., & Marlot, A. (2003). "Electrochemical methods in microelectronics."
  Electrochim. Acta, 48(20-22), 3185–3204.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional



@dataclass(frozen=True)
class PulseCircuitParams:
    """Equivalent circuit parameters of the electrochemical cell interface."""

    c_dl_uF_cm2: float = 30.0       # Double-layer capacitance (µF/cm²)
    r_ohm_ohm_cm2: float = 1.2      # Uncompensated solution resistance (Ω·cm²)
    r_ct_ohm_cm2: float = 4.0       # Charge-transfer resistance (Ω·cm²)

    @property
    def c_dl_F_m2(self) -> float:
        """Double layer capacitance in SI units (F/m²)."""
        return self.c_dl_uF_cm2 * 1e-2  # 1 µF/cm² = 0.01 F/m²

    @property
    def r_ohm_ohm_m2(self) -> float:
        """Ohmic resistance in SI units (Ω·m²)."""
        return self.r_ohm_ohm_cm2 * 1e-4

    @property
    def r_ct_ohm_m2(self) -> float:
        """Charge-transfer resistance in SI units (Ω·m²)."""
        return self.r_ct_ohm_cm2 * 1e-4

    @property
    def tau_charging_s(self) -> float:
        """Series ohmic-capacitive charging time constant (seconds)."""
        return self.r_ohm_ohm_m2 * self.c_dl_F_m2

    @property
    def tau_effective_s(self) -> float:
        """Effective parallel relaxation time constant (seconds)."""
        r_parallel = (self.r_ohm_ohm_m2 * self.r_ct_ohm_m2) / (self.r_ohm_ohm_m2 + self.r_ct_ohm_m2)
        return r_parallel * self.c_dl_F_m2

    @property
    def cutoff_frequency_Hz(self) -> float:
        """3dB cutoff frequency (Hz) above which pulse shape collapses."""
        tau = max(self.tau_charging_s, 1e-9)
        return 1.0 / (2.0 * math.pi * tau)


@dataclass
class FilteredPulseResult:
    """Faradaic waveform distortion and efficiency under RC double-layer filtering."""

    frequency_Hz: float
    nominal_duty_cycle: float
    nominal_t_on_ms: float
    nominal_t_off_ms: float
    nominal_peak_current_mA_cm2: float
    actual_peak_faradaic_mA_cm2: float
    actual_trough_faradaic_mA_cm2: float
    peak_attenuation_ratio: float      # actual_peak / nominal_peak (1.0 = ideal square)
    waveform_fidelity: str            # "high fidelity", "moderate distortion", "severe RC filtering"
    effective_duty_cycle: float        # Faradaic duty cycle (broadened by discharge)
    capacitive_energy_loss_percent: float # Energy wasted charging/discharging C_dl
    is_frequency_feasible: bool        # True if f <= 0.5 * f_cutoff


def simulate_pulse_rc_response(
    frequency_Hz: float,
    duty_cycle: float,
    peak_current_mA_cm2: float,
    circuit: Optional[PulseCircuitParams] = None,
) -> FilteredPulseResult:
    """
    Simulate the steady-state periodic Faradaic current response to a square current pulse.

    Parameters
    ----------
    frequency_Hz : float
        Pulse frequency (Hz).
    duty_cycle : float
        Nominal ON fraction (0 < duty_cycle < 1).
    peak_current_mA_cm2 : float
        Nominal peak cathodic current (mA/cm²).
    circuit : PulseCircuitParams, optional
        Electrochemical interface parameters.

    Returns
    -------
    FilteredPulseResult
        Distortion metrics and effective Faradaic current characteristics.
    """
    if circuit is None:
        circuit = PulseCircuitParams()

    f = max(float(frequency_Hz), 0.1)
    duty = min(max(float(duty_cycle), 0.001), 0.999)
    j_peak = max(float(peak_current_mA_cm2), 0.0)

    period_s = 1.0 / f
    t_on_s = duty * period_s
    t_off_s = (1.0 - duty) * period_s
    tau = circuit.tau_charging_s

    # Analytical periodic steady state:
    # During ON:  j(t) = j_peak * (1 - exp(-t/tau)) + j_min * exp(-t/tau)
    # At end of ON: j_max = j_peak * (1 - exp(-t_on/tau)) + j_min * exp(-t_on/tau)
    # During OFF: j(t) = j_max * exp(-(t - t_on)/tau)
    # At end of OFF: j_min = j_max * exp(-t_off/tau)

    exp_on = math.exp(-t_on_s / tau)
    exp_off = math.exp(-t_off_s / tau)

    # Solve 2x2 linear system for j_max and j_min:
    # j_max - exp_on * j_min = j_peak * (1 - exp_on)
    # -exp_off * j_max + j_min = 0  => j_min = j_max * exp_off
    denom = 1.0 - exp_on * exp_off
    if denom > 1e-12:
        j_max = (j_peak * (1.0 - exp_on)) / denom
        j_min = j_max * exp_off
    else:
        # Very high frequency limit: average current
        j_max = duty * j_peak
        j_min = j_max

    j_max = min(j_max, j_peak)
    j_min = max(j_min, 0.0)

    attenuation = j_max / max(j_peak, 1e-9)

    # Capacitive dissipation: E_cap = 0.5 * C_dl * Delta_V^2 * 2 * f
    delta_v = (j_peak * 10.0) * circuit.r_ohm_ohm_m2 * (1.0 - exp_on)  # Approximate IR ripple
    p_cap = circuit.c_dl_F_m2 * (delta_v ** 2) * f                      # W/m²
    p_total = (duty * j_peak * 10.0) * ((duty * j_peak * 10.0) * circuit.r_ohm_ohm_m2 + 1.5)
    loss_pct = min((p_cap / max(p_total, 1e-6)) * 100.0, 50.0)

    if attenuation >= 0.90 and j_min <= 0.10 * j_peak:
        fidelity = "high fidelity (clean rectangular pulses)"
    elif attenuation >= 0.60:
        fidelity = "moderate distortion (RC rounding of pulse edges)"
    else:
        fidelity = "severe RC filtering (waveform collapsed toward ripple-DC)"

    is_feasible = f <= (0.5 * circuit.cutoff_frequency_Hz)

    # Effective Faradaic duty cycle (time spent above 50% peak)
    # Estimate based on integral area ratio
    eff_duty = duty * (1.0 + (1.0 - attenuation) * 0.5)
    eff_duty = min(max(eff_duty, duty), 1.0)

    return FilteredPulseResult(
        frequency_Hz=f,
        nominal_duty_cycle=duty,
        nominal_t_on_ms=t_on_s * 1e3,
        nominal_t_off_ms=t_off_s * 1e3,
        nominal_peak_current_mA_cm2=j_peak,
        actual_peak_faradaic_mA_cm2=j_max,
        actual_trough_faradaic_mA_cm2=j_min,
        peak_attenuation_ratio=attenuation,
        waveform_fidelity=fidelity,
        effective_duty_cycle=eff_duty,
        capacitive_energy_loss_percent=loss_pct,
        is_frequency_feasible=is_feasible,
    )


def max_practical_frequency_Hz(
    duty_cycle: float = 0.20,
    min_fidelity_ratio: float = 0.85,
    circuit: Optional[PulseCircuitParams] = None,
) -> float:
    """
    Compute the maximum pulse frequency where Faradaic current reaches min_fidelity_ratio of peak.
    """
    if circuit is None:
        circuit = PulseCircuitParams()

    tau = circuit.tau_charging_s
    # In one pulse ON period, we need 1 - exp(-t_on/tau) >= min_fidelity_ratio
    # -t_on/tau = ln(1 - min_fidelity_ratio) => t_on = -tau * ln(1 - min_fidelity_ratio)
    fidelity = min(max(float(min_fidelity_ratio), 0.5), 0.99)
    t_on_min = -tau * math.log(1.0 - fidelity)
    period_min = t_on_min / max(duty_cycle, 0.01)
    return 1.0 / max(period_min, 1e-6)


def main() -> None:
    """CLI entrypoint for pulse RC filter analysis."""
    print("=================================================================")
    print(" Pulse RC Double-Layer Filtering & Cutoff Frequency Analysis")
    print("=================================================================")
    circuit = PulseCircuitParams()
    print(f"Cell interfacial time constant: {circuit.tau_charging_s*1e6:.1f} µs")
    print(f"3dB Cutoff frequency          : {circuit.cutoff_frequency_Hz:.1f} Hz\n")
    print("Frequency sweep (20% duty, 200 mA/cm² peak):")
    for f in [10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]:
        res = simulate_pulse_rc_response(f, 0.20, 200.0, circuit)
        print(f"  f = {f:6.0f} Hz | Peak Faradaic: {res.actual_peak_faradaic_mA_cm2:5.1f} mA/cm² ({res.peak_attenuation_ratio*100:4.1f}%) | {res.waveform_fidelity}")


if __name__ == "__main__":
    main()
