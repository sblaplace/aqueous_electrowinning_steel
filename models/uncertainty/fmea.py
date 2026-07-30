"""
Failure Mode and Effects Analysis (FMEA) for the aqueous electrowinning
steel production chain.

Systematically identifies failure modes across all subsystems, computes
Risk Priority Numbers (RPN = Severity × Occurrence × Detection), identifies
critical failure paths, and proposes mitigations.

Subsystem coverage:
  1. Electrochemistry — HER competition, bath contamination, anode degradation, pH runaway
  2. Co-deposition — anomalous suppression, carbon agglomeration, inclusion incorporation
  3. Microstructure — grain growth, preferred texture, hydrogen embrittlement
  4. Heat treatment — decarburization, oxidation, quench cracking, retained austenite, distortion
  5. Process — membrane fouling, electrolyte decomposition, temp excursion, current interruption
  6. Quality control — measurement error, specification drift, sampling bias

References:
  - IEC 60812:2018 — Failure modes and effects analysis (FMEA and FMEA)
  - ASTM E2020 — Standard guide for FMEA
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureMode:
    """Single failure mode in the FMEA table.

    RPN is computed automatically: severity × occurrence × detection.
    residual_RPN is set after mitigation is applied.
    """

    id: str
    component: str           # subsystem / component name
    mode: str                # failure mode description
    effect: str              # effect on product / process
    severity: int            # 1–10
    cause: str               # root cause
    occurrence: int          # 1–10
    detection: int           # 1–10 (higher = harder to detect)
    mitigation: str = ""
    residual_severity: int = 0
    residual_occurrence: int = 0
    residual_detection: int = 0

    def __post_init__(self):
        if not (1 <= self.severity <= 10):
            raise ValueError(f"severity must be 1-10, got {self.severity}")
        if not (1 <= self.occurrence <= 10):
            raise ValueError(f"occurrence must be 1-10, got {self.occurrence}")
        if not (1 <= self.detection <= 10):
            raise ValueError(f"detection must be 1-10, got {self.detection}")

    @property
    def rpn(self) -> int:
        """Risk Priority Number = S × O × D."""
        return self.severity * self.occurrence * self.detection

    @property
    def residual_rpn(self) -> int:
        """Residual RPN after mitigation.  Returns 0 if no mitigation."""
        if self.mitigation and self.residual_severity > 0:
            return (self.residual_severity
                    * self.residual_occurrence
                    * self.residual_detection)
        return 0

    @property
    def rpn_reduction(self) -> int:
        """RPN reduction from mitigation."""
        return self.rpn - self.residual_rpn


@dataclass
class FMEAReport:
    """Aggregated FMEA report for the full production chain."""

    failure_modes: List[FailureMode] = field(default_factory=list)
    design_point: Dict[str, Any] = field(default_factory=dict)
    mc_summary: Dict[str, Any] = field(default_factory=dict)

    # ── Aggregate metrics ──────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.failure_modes)

    @property
    def mean_rpn(self) -> float:
        if not self.failure_modes:
            return 0.0
        return float(np.mean([fm.rpn for fm in self.failure_modes]))

    @property
    def max_rpn(self) -> int:
        if not self.failure_modes:
            return 0
        return max(fm.rpn for fm in self.failure_modes)

    @property
    def critical_count(self) -> int:
        """Number of failure modes with RPN > 100."""
        return sum(1 for fm in self.failure_modes if fm.rpn > 100)

    def by_component(self) -> Dict[str, List[FailureMode]]:
        """Group failure modes by component/subsystem."""
        groups: Dict[str, List[FailureMode]] = {}
        for fm in self.failure_modes:
            groups.setdefault(fm.component, []).append(fm)
        return groups

    def ranked(self, descending: bool = True) -> List[FailureMode]:
        """Return failure modes sorted by RPN."""
        return sorted(self.failure_modes, key=lambda fm: fm.rpn, reverse=descending)

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable report."""
        return {
            "total_modes": self.total,
            "mean_rpn": round(self.mean_rpn, 1),
            "max_rpn": self.max_rpn,
            "critical_count": self.critical_count,
            "modes": [
                {
                    "id": fm.id,
                    "component": fm.component,
                    "mode": fm.mode,
                    "effect": fm.effect,
                    "severity": fm.severity,
                    "cause": fm.cause,
                    "occurrence": fm.occurrence,
                    "detection": fm.detection,
                    "rpn": fm.rpn,
                    "mitigation": fm.mitigation,
                    "residual_rpn": fm.residual_rpn,
                }
                for fm in self.ranked()
            ],
        }


# ---------------------------------------------------------------------------
# Default failure modes library
# ---------------------------------------------------------------------------

def _default_failure_modes() -> List[FailureMode]:
    """Return the full default FMEA table (>=20 modes across 6 subsystems)."""

    return [
        # ── 1. Electrochemistry ────────────────────────────────────────
        FailureMode(
            id="EC-001",
            component="Electrochemistry",
            mode="HER competition",
            effect="Reduced faradaic efficiency, hydrogen porosity in deposit",
            severity=7,
            cause="High overpotential, low pH, catalytic cathode surface",
            occurrence=7,
            detection=4,
            mitigation="pH control (3.0–3.5), pulse plating to limit peak overpotential, HER-inhibiting surfactants",
            residual_severity=7, residual_occurrence=3, residual_detection=3,
        ),
        FailureMode(
            id="EC-002",
            component="Electrochemistry",
            mode="Bath contamination",
            effect="Inclusions in deposit, poor surface finish, reduced ductility",
            severity=8,
            cause="Dissolved Cr, Cu, Zn from auxiliary equipment or anode corrosion",
            occurrence=5,
            detection=5,
            mitigation="Continuous electrolyte purification (ion exchange), anode material selection, regular ICP-OES monitoring",
            residual_severity=8, residual_occurrence=2, residual_detection=2,
        ),
        FailureMode(
            id="EC-003",
            component="Electrochemistry",
            mode="Anode degradation",
            effect="Increased cell voltage, contamination, process instability",
            severity=6,
            cause="Oxidative dissolution of DSA or Pb-based anode at high current density",
            occurrence=4,
            detection=6,
            mitigation="DSA coating with IrO₂/TiO₂, voltage monitoring with automated anode replacement alerts",
            residual_severity=6, residual_occurrence=2, residual_detection=3,
        ),
        FailureMode(
            id="EC-004",
            component="Electrochemistry",
            mode="pH runaway",
            effect="Precipitation of Fe(OH)₃, bath destabilization, deposit defects",
            severity=8,
            cause="Buffer exhaustion at high throughput, acid consumption exceeding replenishment",
            occurrence=4,
            detection=4,
            mitigation="Automated pH dosing system with redundant sensors, buffer capacity monitoring",
            residual_severity=8, residual_occurrence=2, residual_detection=2,
        ),

        # ── 2. Co-deposition ──────────────────────────────────────────
        FailureMode(
            id="CD-001",
            component="Co-deposition",
            mode="Anomalous suppression",
            effect="Unexpected Ni/Fe ratio shift, inconsistent alloy composition",
            severity=7,
            cause="Hydroxide suppression mechanism instability at boundary layer",
            occurrence=6,
            detection=5,
            mitigation="Real-time ICP-OES at cathode boundary, controlled mass transport (rotating cylinder electrode)",
            residual_severity=7, residual_occurrence=3, residual_detection=3,
        ),
        FailureMode(
            id="CD-002",
            component="Co-deposition",
            mode="Carbon agglomeration",
            effect="Non-uniform carbon distribution, weak spots, poor wear resistance",
            severity=8,
            cause="Inadequate particle suspension, settling in quiescent zones",
            occurrence=5,
            detection=6,
            mitigation="Ultrasonic agitation, surfactant dispersants, pulsed flow to prevent settling",
            residual_severity=8, residual_occurrence=2, residual_detection=3,
        ),
        FailureMode(
            id="CD-003",
            component="Co-deposition",
            mode="Inclusion incorporation",
            effect="Brittle deposits, crack initiation sites, fatigue life reduction",
            severity=9,
            cause="Organic breakdown products, colloidal impurities co-depositing with metal",
            occurrence=3,
            detection=7,
            mitigation="Electrolyte filtration (<1 µm), activated carbon treatment, periodic bath replacement",
            residual_severity=9, residual_occurrence=2, residual_detection=3,
        ),

        # ── 3. Microstructure ─────────────────────────────────────────
        FailureMode(
            id="MS-001",
            component="Microstructure",
            mode="Grain growth during processing",
            effect="Reduced yield strength (Hall-Petch), loss of nanostructure benefits",
            severity=7,
            cause="Thermal excursion during post-deposition handling or slow cooling",
            occurrence=4,
            detection=5,
            mitigation="EBSD monitoring, controlled cooling rate specification, thermal budget tracking",
            residual_severity=7, residual_occurrence=2, residual_detection=3,
        ),
        FailureMode(
            id="MS-002",
            component="Microstructure",
            mode="Preferred texture (columnar grains)",
            effect="Anisotropic mechanical properties, weak transverse direction",
            severity=6,
            cause="High current density without pulse reversal, limited nucleation sites",
            occurrence=5,
            detection=6,
            mitigation="Pulse plating with reverse pulse, grain refiner addition, current density optimization",
            residual_severity=6, residual_occurrence=3, residual_detection=3,
        ),
        FailureMode(
            id="MS-003",
            component="Microstructure",
            mode="Hydrogen embrittlement",
            effect="Delayed cracking, catastrophic brittle fracture under load",
            severity=10,
            cause="Co-deposited hydrogen during high-HER plating conditions",
            occurrence=5,
            detection=7,
            mitigation="Post-deposition bake (200°C/4h), low-CE plating window avoidance, hydrogen permeation monitoring",
            residual_severity=10, residual_occurrence=2, residual_detection=4,
        ),

        # ── 4. Heat treatment ─────────────────────────────────────────
        FailureMode(
            id="HT-001",
            component="Heat treatment",
            mode="Decarburization",
            effect="Soft surface layer, reduced wear/case hardening effectiveness",
            severity=7,
            cause="Oxygen ingress in furnace, insufficient carburizing atmosphere control",
            occurrence=5,
            detection=4,
            mitigation="Endo-gas atmosphere control, oxygen probe monitoring, anti-scale coating for non-carburized surfaces",
            residual_severity=7, residual_occurrence=2, residual_detection=2,
        ),
        FailureMode(
            id="HT-002",
            component="Heat treatment",
            mode="Oxidation",
            effect="Surface scale, dimensional change, post-treatment grinding required",
            severity=5,
            cause="Furnace atmosphere breach, quench medium contamination",
            occurrence=4,
            detection=3,
            mitigation="Vacuum or inert gas quenching, sealed furnace design, atmosphere dew point monitoring",
            residual_severity=5, residual_occurrence=2, residual_detection=2,
        ),
        FailureMode(
            id="HT-003",
            component="Heat treatment",
            mode="Quench cracking",
            effect="Part rejection, catastrophic failure in service",
            severity=10,
            cause="Excessive thermal stress during rapid quench, sharp geometry features",
            occurrence=3,
            detection=8,
            mitigation="Interrupted quench (martempering), generous fillet radii in design, UT inspection post-quench",
            residual_severity=10, residual_occurrence=1, residual_detection=4,
        ),
        FailureMode(
            id="HT-004",
            component="Heat treatment",
            mode="Excessive retained austenite",
            effect="Dimensional instability, soft spots, transformation-induced cracking",
            severity=7,
            cause="High carbon/nickel content shifting Ms below room temperature",
            occurrence=5,
            detection=5,
            mitigation="Sub-zero treatment (-80°C), double temper, XRD retained austenite monitoring",
            residual_severity=7, residual_occurrence=2, residual_detection=3,
        ),
        FailureMode(
            id="HT-005",
            component="Heat treatment",
            mode="Distortion / warpage",
            effect="Out-of-tolerance dimensions, machining allowance exceeded",
            severity=6,
            cause="Non-uniform heating/cooling, gravity sag in furnace, asymmetric geometry",
            occurrence=6,
            detection=4,
            mitigation="Fixture-based quenching, symmetric racking, CFD-optimized furnace flow",
            residual_severity=6, residual_occurrence=3, residual_detection=2,
        ),

        # ── 5. Process ────────────────────────────────────────────────
        FailureMode(
            id="PR-001",
            component="Process",
            mode="Membrane fouling",
            effect="Increased cell voltage, reduced throughput, electrolyte crossover",
            severity=7,
            cause="Fe(OH)₃ precipitation, CaSO₄ scaling, organic/biofilm growth on cation-exchange membrane",
            occurrence=7,
            detection=4,
            mitigation="Periodic acid wash cycle, electrolyte pre-filtration, anti-fouling membrane coating",
            residual_severity=7, residual_occurrence=3, residual_detection=3,
        ),
        FailureMode(
            id="PR-002",
            component="Process",
            mode="Electrolyte decomposition",
            effect="Gas evolution, additive depletion, deposit quality degradation",
            severity=6,
            cause="Electrolytic breakdown of organic additives at high overpotential",
            occurrence=4,
            detection=5,
            mitigation="Additive concentration monitoring (CVSA), replenishment dosing system, current density limits",
            residual_severity=6, residual_occurrence=2, residual_detection=3,
        ),
        FailureMode(
            id="PR-003",
            component="Process",
            mode="Temperature excursion",
            effect="Accelerated side reactions, changed kinetics, bath instability",
            severity=7,
            cause="Joule heating at high current, cooling system failure, ambient variation",
            occurrence=5,
            detection=3,
            mitigation="PID temperature control with redundant TC, heat exchanger sizing, alarm interlocks",
            residual_severity=7, residual_occurrence=2, residual_detection=1,
        ),
        FailureMode(
            id="PR-004",
            component="Process",
            mode="Current interruption",
            effect="Layered deposit with weak interlayer bonds, delamination risk",
            severity=8,
            cause="Power supply failure, contact resistance drift, anode short circuit",
            occurrence=3,
            detection=5,
            mitigation="UPS backup, contact resistance monitoring with auto-shutdown, redundant rectifiers",
            residual_severity=8, residual_occurrence=1, residual_detection=2,
        ),

        # ── 6. Quality control ────────────────────────────────────────
        FailureMode(
            id="QC-001",
            component="Quality control",
            mode="Measurement error in hardness testing",
            effect="False accept/reject, incorrect tempering parameter adjustment",
            severity=6,
            cause="Indenter wear, surface roughness, insufficient test load for thin layers",
            occurrence=4,
            detection=4,
            mitigation="Indenter calibration schedule, surface preparation SOP, micro-hardness for thin layers",
            residual_severity=6, residual_occurrence=2, residual_detection=2,
        ),
        FailureMode(
            id="QC-002",
            component="Quality control",
            mode="Sampling bias in composition analysis",
            effect="Systematic error in Ni/C content, incorrect mechanical property prediction",
            severity=7,
            cause="Non-uniform deposit composition, single-point ICP-OES sampling",
            occurrence=5,
            detection=5,
            mitigation="Multi-point sampling grid, cross-section line scan, statistical process control",
            residual_severity=7, residual_occurrence=2, residual_detection=3,
        ),
        FailureMode(
            id="QC-003",
            component="Quality control",
            mode="Specification drift over campaign",
            effect="Gradual quality degradation not caught by pass/fail checks",
            severity=8,
            cause="Slow process parameter drift, bath aging, equipment wear",
            occurrence=5,
            detection=6,
            mitigation="SPC trend analysis with Western Electric rules, campaign-level Cpk tracking, periodic re-validation",
            residual_severity=8, residual_occurrence=2, residual_detection=3,
        ),
    ]


# ---------------------------------------------------------------------------
# FMEA generation
# ---------------------------------------------------------------------------

def generate_fmea(
    design_point: Optional[Dict[str, Any]] = None,
    mc_result: Optional[Any] = None,
) -> FMEAReport:
    """Generate an FMEA report for the full electrowinning chain.

    Parameters
    ----------
    design_point : dict, optional
        Operating conditions.  Used for context but does not change the
        default failure mode library (modes are physics-based, not
        scenario-specific).
    mc_result : MonteCarloResult, optional
        If provided, occurrence/detection ratings are adjusted based on
        Monte Carlo pass rates — lower pass rates increase occurrence.

    Returns
    -------
    FMEAReport
    """
    modes = _default_failure_modes()

    # Adjust occurrence based on MC results if available
    if mc_result is not None:
        modes = _adjust_from_mc(modes, mc_result)

    report = FMEAReport(
        failure_modes=modes,
        design_point=design_point or {},
        mc_summary=_mc_summary(mc_result) if mc_result is not None else {},
    )

    return report


def _adjust_from_mc(
    modes: List[FailureMode],
    mc_result: Any,
) -> List[FailureMode]:
    """Adjust failure mode ratings based on Monte Carlo results.

    Maps low pass rates to higher occurrence for related failure modes.
    """
    pass_rates = getattr(mc_result, "pass_rates", {})
    if not pass_rates:
        return modes

    # Map MC outputs to failure mode IDs
    mc_output_map = {
        "current_efficiency_percent": ["EC-001"],
        "elongation_pct": ["MS-003", "CD-003"],
        "porosity": ["EC-001", "CD-002"],
        "grain_size_um": ["MS-001"],
        "surface_hv": ["HT-001", "HT-004"],
        "specific_energy_kWh_per_kg": ["EC-003", "PR-001"],
    }

    # Build set of affected mode IDs with low pass rates
    affected: Dict[str, float] = {}
    for spec_name, related_ids in mc_output_map.items():
        for key, rate in pass_rates.items():
            if spec_name in key or key in spec_name:
                for mid in related_ids:
                    # Worst case: keep the lowest pass rate for each mode
                    if mid not in affected or rate < affected[mid]:
                        affected[mid] = rate
                break

    adjusted: List[FailureMode] = []
    for fm in modes:
        if fm.id in affected:
            rate = affected[fm.id]
            # Map pass rate to occurrence bump: rate=1.0 → no change,
            # rate=0.5 → +2, rate=0.0 → +4 (capped at 10)
            bump = int(round(4 * (1.0 - rate)))
            new_occ = min(fm.occurrence + bump, 10)
            if new_occ != fm.occurrence:
                fm = FailureMode(
                    id=fm.id,
                    component=fm.component,
                    mode=fm.mode,
                    effect=fm.effect,
                    severity=fm.severity,
                    cause=fm.cause,
                    occurrence=new_occ,
                    detection=fm.detection,
                    mitigation=fm.mitigation,
                    residual_severity=fm.residual_severity,
                    residual_occurrence=fm.residual_occurrence,
                    residual_detection=fm.residual_detection,
                )
        adjusted.append(fm)

    return adjusted


def _mc_summary(mc_result: Any) -> Dict[str, Any]:
    """Extract key metrics from MC result for the report."""
    return {
        "n_samples": getattr(mc_result, "n_samples", 0),
        "overall_confidence": getattr(mc_result, "overall_confidence", 0.0),
        "pass_rates": {
            k: round(v, 4)
            for k, v in getattr(mc_result, "pass_rates", {}).items()
        },
    }


# ---------------------------------------------------------------------------
# Critical failure paths
# ---------------------------------------------------------------------------

def critical_failure_paths(
    fmea: FMEAReport,
    rpn_threshold: int = 100,
) -> List[FailureMode]:
    """Identify failure modes with RPN above the threshold.

    Parameters
    ----------
    fmea : FMEAReport
        The FMEA report to analyze.
    rpn_threshold : int
        RPN cutoff for critical classification (default 100).

    Returns
    -------
    list of FailureMode
        Critical modes sorted by RPN descending.
    """
    critical = [fm for fm in fmea.failure_modes if fm.rpn > rpn_threshold]
    return sorted(critical, key=lambda fm: fm.rpn, reverse=True)


# ---------------------------------------------------------------------------
# Mitigation roadmap
# ---------------------------------------------------------------------------

def mitigation_roadmap(fmea: FMEAReport) -> List[Dict[str, Any]]:
    """Generate a prioritized mitigation roadmap.

    Returns a list of mitigation actions sorted by RPN reduction (highest
    impact first), each with implementation priority and estimated effort.

    Parameters
    ----------
    fmea : FMEAReport
        The FMEA report.

    Returns
    -------
    list of dict
        Each dict has keys: id, mode, component, rpn, residual_rpn,
        rpn_reduction, mitigation, priority (1-3), estimated_effort.
    """
    roadmap: List[Dict[str, Any]] = []

    for fm in fmea.ranked():
        if not fm.mitigation:
            continue

        reduction = fm.rpn_reduction
        residual = fm.residual_rpn

        # Priority: 1 = must-do (RPN>100 or residual>50), 2 = should-do, 3 = nice-to-have
        if fm.rpn > 100 or residual > 50:
            priority = 1
            effort = "high" if fm.rpn > 200 else "medium"
        elif fm.rpn > 50:
            priority = 2
            effort = "medium"
        else:
            priority = 3
            effort = "low"

        roadmap.append({
            "id": fm.id,
            "mode": fm.mode,
            "component": fm.component,
            "rpn": fm.rpn,
            "residual_rpn": residual,
            "rpn_reduction": reduction,
            "mitigation": fm.mitigation,
            "priority": priority,
            "estimated_effort": effort,
        })

    # Sort: priority first (ascending), then RPN reduction (descending)
    roadmap.sort(key=lambda x: (x["priority"], -x["rpn_reduction"]))
    return roadmap
