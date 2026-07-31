"""
Global (Sobol) sensitivity analysis of the 1D diffusion-layer FE engine.

This is the answer to a self-identified weakness in the program's prior
uncertainty work.  ``docs/RESEARCH_PROGRAM.md`` (What to Freeze) noted:

    "Sobol indices over 76 invented priors are sensitivity analysis of a
    fiction.  Run them on the 10-parameter transport model instead, where
    they'll actually tell you which experiment to do next."

That is exactly what this module does.  It applies a proper *Saltelli* Sobol
decomposition to :class:`~models.diffusion_layer_1d.DiffusionLayer1D` -- the
FE prediction engine -- over **10 physically-grounded experimental levers**
(none invented; all directly controllable or measurable in the lab):

===============================  ===========================================
Parameter                        Experimental lever / uncertainty source
===============================  ===========================================
``j_mA_cm2``                     Applied current density (operating point)
``fe_conc_M``                    Bath Fe²⁺ concentration (recipe, titration)
``pH_bulk``                      Bulk electrolyte pH (recipe, control)
``temperature_C``                Electrolyte temperature (thermostat)
``delta_m``                      Diffusion-layer thickness (agitation / flow)
``buffer_conc_M``                Borate buffer concentration (recipe)
``her_i0``                       HER exchange current (cathode surface state)
``fe_i0``                        Fe exchange current (intrinsic Fe kinetics)
``her_tafel_V``                  HER Tafel slope (reaction mechanism)
``fe_tafel_V``                   Fe Tafel slope (reaction mechanism)
===============================  ===========================================

For each output (Faradaic efficiency, cell voltage, surface pH) we report
**first-order** ``S1`` (direct effect) and **total-order** ``ST`` (direct +
all interactions) Sobol indices, then rank parameters by ``ST`` on FE to
answer *"which experiment / measurement to do next."*

Design
------
* Saltelli A/B/AB sampling over ``scipy.stats.qmc.Sobol`` (scrambled,
  independent streams for A and B).
* Model evaluations use ``DiffusionLayer1D(fast_mode=True)`` -- the relaxed
  solver whose FE/V_cell agree with the tight solver to a few hundredths of
  a percent at 10-20x lower cost (validated in ``test_transport_sensitivity``).
* Optional process-level parallelism (``n_workers``) since each solve is
  ODE-bound; ``n_workers=1`` for deterministic, in-process tests.

The bounds below are the *design/uncertainty ranges* of the experimental
space, not model-coefficient priors.  ``ST`` over this space tells you which
lever you must control/measure most tightly to pin FE and cell voltage.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

from .diffusion_layer_1d import DiffusionLayer1D

# ---------------------------------------------------------------------------
# Parameter space (10 experimental levers)
# ---------------------------------------------------------------------------

#: name -> (bounds, transform, unit, lever label, experiment guidance)
_PARAM_DEFS: List[Dict[str, Any]] = [
    dict(
        name="j_mA_cm2",
        bounds=(20.0, 300.0),
        transform="log",
        unit="mA/cm²",
        lever="Applied current density",
        advice=(
            "Hold and record current density as the master operating point. "
            "If it dominates, screen j early in the DOE before locking any "
            "bath recipe."
        ),
    ),
    dict(
        name="fe_conc_M",
        bounds=(0.2, 2.0),
        transform="linear",
        unit="mol/L",
        lever="Bath Fe²⁺ concentration",
        advice=(
            "Verify Fe²⁺ by titration/ICP at start and end of every run "
            "(electrowinning depletes it). If it dominates, the bath "
            "chemistry spec is the gating experiment."
        ),
    ),
    dict(
        name="pH_bulk",
        bounds=(0.5, 4.0),
        transform="linear",
        unit="",
        lever="Bulk electrolyte pH",
        advice=(
            "Control and measure pH to a few hundredths. If it dominates, "
            "the allowable pH window is the single most important thing to "
            "map first."
        ),
    ),
    dict(
        name="temperature_C",
        bounds=(25.0, 90.0),
        transform="linear",
        unit="°C",
        lever="Electrolyte temperature",
        advice=(
            "Thermostat the cell and log temperature with an in-cell probe. "
            "If it dominates, temperature control is a first-order "
            "requirement, not a nicety."
        ),
    ),
    dict(
        name="delta_m",
        bounds=(1.0e-5, 1.0e-4),
        transform="log",
        unit="m",
        lever="Diffusion-layer thickness (agitation / flow)",
        advice=(
            "δ is set by flow, electrode geometry and bubbles. If it "
            "dominates, hydrodynamics is the highest-value experiment: "
            "measure δ (or its surrogate, the limiting current) and verify "
            "agitation control."
        ),
    ),
    dict(
        name="buffer_conc_M",
        bounds=(0.0, 0.8),
        transform="linear",
        unit="mol/L",
        lever="Borate buffer concentration",
        advice=(
            "If buffer dominates, its concentration is a primary bath "
            "variable to tune; screen it deliberately rather than ad hoc."
        ),
    ),
    dict(
        name="her_i0",
        bounds=(1.0e-5, 1.0),
        transform="log",
        unit="A/m²",
        lever="HER exchange current (cathode surface state)",
        advice=(
            "HER kinetics depend on the electrode surface condition, "
            "poisoning and any coating. If it dominates, cathode-surface "
            "reproducibility is the gating experiment: clean, well-defined "
            "surfaces and HER-suppression trials come first."
        ),
    ),
    dict(
        name="fe_i0",
        bounds=(1.0, 100.0),
        transform="log",
        unit="A/m²",
        lever="Fe exchange current (intrinsic Fe kinetics)",
        advice=(
            "If it dominates, prioritize measuring Fe deposition kinetics "
            "(polarization curves) on the target substrate before tuning "
            "anything else."
        ),
    ),
    dict(
        name="her_tafel_V",
        bounds=(0.08, 0.20),
        transform="linear",
        unit="V/dec",
        lever="HER Tafel slope (mechanism)",
        advice=(
            "If it dominates, pin the HER mechanism by Tafel analysis on "
            "the real cathode before scaling up; a wrong mechanism guess "
            "rewards the wrong mitigation."
        ),
    ),
    dict(
        name="fe_tafel_V",
        bounds=(0.08, 0.20),
        transform="linear",
        unit="V/dec",
        lever="Fe Tafel slope (mechanism)",
        advice=(
            "If it dominates, pin the Fe deposition mechanism by Tafel "
            "analysis first, as it sets how FE and V_cell respond to "
            "current density."
        ),
    ),
]

#: Output quantities computed for each evaluated operating point.
OUTPUT_KEYS: Tuple[str, ...] = ("FE_pct", "V_cell_V", "surface_pH")

#: Fixed design value for the supporting electrolyte (unsupported-bath regime,
#: the migration-dominated case highlighted in the README).
SUPPORT_CONC_M_FIXED = 0.0


@dataclass(frozen=True)
class ParameterSpec:
    """One experimental lever: name, range, transform, units, guidance."""

    name: str
    bounds: Tuple[float, float]
    transform: str
    unit: str
    lever: str
    advice: str


def parameter_space() -> List[ParameterSpec]:
    """Return the 10-lever experimental space as :class:`ParameterSpec`s."""
    return [ParameterSpec(**d) for d in _PARAM_DEFS]


def _names() -> List[str]:
    return [p.name for p in parameter_space()]


# ---------------------------------------------------------------------------
# Sampling (Saltelli) and transform
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(n))) if n > 1 else 1


def _transform(u: np.ndarray, spec: ParameterSpec) -> np.ndarray:
    """Map unit-interval samples ``u`` to physical values for one parameter."""
    lo, hi = spec.bounds
    if spec.transform == "log":
        lo = np.log10(max(lo, 1e-300))
        hi = np.log10(max(hi, 1e-300))
        return 10.0 ** (lo + u * (hi - lo))
    return lo + u * (hi - lo)


def saltelli_matrices(
    n: int, d: int | None = None, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Build Saltelli A, B, and AB_i matrices (each ``n x d`` in [0,1)).

    A and B are taken from *different dimensions* of a single (scrambled)
    Sobol sequence of dimension ``2*d + 2`` -- the standard Saltelli scheme,
    which keeps A and B independent while retaining low-discrepancy coverage.
    ``AB_i`` is B with column ``i`` replaced by A's column ``i``.

    Returns ``(A, B, AB)``.
    """
    d = d if d is not None else len(_names())
    # A = first d dims, B = next d dims (2 spare dims, conventional).
    x = qmc.Sobol(d=2 * d + 2, scramble=True, seed=seed).random(
        _next_pow2(n)
    )[:n]
    a, b = x[:, :d], x[:, d : 2 * d]
    ab = []
    for i in range(d):
        abi = b.copy()
        abi[:, i] = a[:, i]
        ab.append(abi)
    return a, b, ab


def physical_points(unit: np.ndarray):
    """Map unit-interval samples to physical parameter values.

    Accepts a 2-D matrix (returns ``{name: 1-D array}``) or a single 1-D row
    (returns ``{name: float}``) for one operating point.
    """
    specs = parameter_space()
    arr = np.atleast_2d(np.asarray(unit, dtype=float))
    out = {s.name: _transform(arr[:, i], s) for i, s in enumerate(specs)}
    if np.asarray(unit).ndim == 1:
        return {k: float(v[0]) for k, v in out.items()}
    return out


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def _eval_point(point: Dict[str, float]) -> Dict[str, float]:
    """Evaluate FE/V_cell/surface_pH for one physical parameter set."""
    j = point["j_mA_cm2"]
    kwargs = {k: v for k, v in point.items() if k != "j_mA_cm2"}
    model = DiffusionLayer1D(
        fast_mode=True, support_conc_M=SUPPORT_CONC_M_FIXED, **kwargs
    )
    result = model.solve(float(j))
    return {
        "FE_pct": result.fe_percent,
        "V_cell_V": result.V_cell,
        "surface_pH": result.surface_pH,
    }


def _safe_eval(point: Dict[str, float]) -> Dict[str, float]:
    try:
        return _eval_point(point)
    except Exception:
        return {k: float("nan") for k in OUTPUT_KEYS}


def _evaluate_points(
    points: Sequence[Dict[str, float]], n_workers: int = 1
) -> Dict[str, np.ndarray]:
    """Evaluate many physical points; returns per-output arrays.

    Failed solves return NaN and are dropped by the caller.
    """
    if n_workers and n_workers > 1 and len(points) > 1:
        if os.name == "nt":  # pragma: no cover - fork-only envs on Linux
            n_workers = 1
        else:
            with mp.Pool(processes=n_workers) as pool:
                raw = pool.map(_safe_eval, points, chunksize=8)
    else:
        raw = [_safe_eval(p) for p in points]
    out: Dict[str, np.ndarray] = {k: [] for k in OUTPUT_KEYS}
    for r in raw:
        for k in OUTPUT_KEYS:
            out[k].append(r[k])
    return {k: np.asarray(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Sobol index computation
# ---------------------------------------------------------------------------

def _sobol_from_arrays(
    y_a: np.ndarray, y_b: np.ndarray, y_ab: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order (S1) and total-order (ST) Sobol indices, vectorized.

    ``y_a``: (n,) outputs at A; ``y_b``: (n,) at B; ``y_ab``: list of (n,)
    outputs at each AB_i.  Returns ``(S1, ST)`` each of length d.
    """
    f0 = float(np.mean(y_a))
    var = float(np.mean(y_a ** 2) - f0 ** 2)
    d = len(y_ab)
    s1 = np.full(d, np.nan)
    st = np.full(d, np.nan)
    if var <= 0.0:
        return s1, st
    for i in range(d):
        s1[i] = (float(np.mean(y_a * y_ab[i])) - f0 ** 2) / var
        st[i] = 1.0 - (float(np.mean(y_b * y_ab[i])) - f0 ** 2) / var
    return s1, st


# ---------------------------------------------------------------------------
# Orchestration and results
# ---------------------------------------------------------------------------

@dataclass
class SobolOutput:
    """Sobol decomposition of a single model output."""

    output: str
    mean: float
    var: float
    s1: np.ndarray
    st: np.ndarray
    param_names: List[str] = field(default_factory=list)
    #: parameters sorted by total-order index, descending
    rank_by_st: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.param_names:
            self.param_names = _names()
        self.rank_by_st = [
            name for _, name in sorted(
                zip(self.st, self.param_names), key=lambda t: -t[0]
            )
        ]


@dataclass
class SensitivityResult:
    """Full GSA result for the transport / FE engine."""

    n_samples: int
    n_evaluated: int
    n_failed: int
    outputs: List[SobolOutput]
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def fe_output(self) -> SobolOutput:
        return next(o for o in self.outputs if o.output == "FE_pct")

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_evaluated": self.n_evaluated,
            "n_failed": self.n_failed,
            "parameters": _names(),
            "outputs": {
                o.output: {
                    "mean": o.mean,
                    "var": o.var,
                    "S1": [float(x) for x in o.s1],
                    "ST": [float(x) for x in o.st],
                    "rank_by_ST": o.rank_by_st,
                }
                for o in self.outputs
            },
            "recommendations": self.recommendations,
        }


def _build_recommendations(
    fe: SobolOutput, specs: List[ParameterSpec], n_top: int = 5
) -> List[Dict[str, Any]]:
    """Compose 'which experiment to do next' guidance from FE total indices."""
    recs = []
    for rank, name in enumerate(fe.rank_by_st[:n_top], start=1):
        spec = next(s for s in specs if s.name == name)
        idx = fe.param_names.index(name)
        recs.append(
            dict(
                rank=rank,
                parameter=name,
                lever=spec.lever,
                unit=spec.unit,
                S1=float(fe.s1[idx]),
                ST=float(fe.st[idx]),
                advice=spec.advice,
            )
        )
    return recs


def run_analysis(
    n_samples: int = 128,
    seed: int = 0,
    n_workers: int = 1,
    outputs: Sequence[str] = OUTPUT_KEYS,
) -> SensitivityResult:
    """Run the full Saltelli-Sobol GSA on the FE prediction engine.

    Parameters
    ----------
    n_samples : int
        Saltelli base sample count (A/B each of size ``n_samples``); the
        total solve count is ``(2*d + 2) * n_samples``.
    seed : int
        RNG seed for the Sobol samplers (deterministic).
    n_workers : int
        Process pool size for parallel evaluation (1 = in-process).
    outputs : sequence of str
        Which outputs to decompose (subset of ``OUTPUT_KEYS``).
    """
    specs = parameter_space()
    d = len(specs)
    a, b, ab = saltelli_matrices(n_samples, d, seed)

    # Flatten into physical points and evaluate once.
    unit_all = np.concatenate([a, b] + ab, axis=0)  # ( (d+2)n, d )
    all_out = _evaluate_points(
        [physical_points(row) for row in unit_all], n_workers=n_workers
    )

    # Re-slice outputs back into A/B/AB_i for each requested output.
    n = n_samples
    sobol_outputs = []
    # Drop evaluation rows that are non-finite for ANY requested output, so
    # every index set uses the same rows.
    combined_good = np.ones(n, dtype=bool)
    for key in outputs:
        y_all = all_out[key]
        y_a, y_b = y_all[:n], y_all[n : 2 * n]
        y_ab = [y_all[(2 + i) * n : (3 + i) * n] for i in range(d)]
        combined_good &= (
            np.isfinite(y_a) & np.isfinite(y_b)
            & np.all(np.isfinite(np.vstack(y_ab)), axis=0)
        )
    n_failed = int(np.sum(~combined_good))
    good = combined_good

    for key in outputs:
        y_all = all_out[key]
        y_a, y_b = y_all[:n], y_all[n : 2 * n]
        y_ab = [y_all[(2 + i) * n : (3 + i) * n] for i in range(d)]
        s1, st = _sobol_from_arrays(
            y_a[good], y_b[good], [ya[good] for ya in y_ab]
        )
        sobol_outputs.append(
            SobolOutput(
                output=key,
                mean=float(np.mean(y_a[good])),
                var=float(np.var(y_a[good])),
                s1=s1,
                st=st,
                param_names=[s.name for s in specs],
            )
        )

    fe = next(o for o in sobol_outputs if o.output == "FE_pct")
    recs = _build_recommendations(fe, specs)

    return SensitivityResult(
        n_samples=n_samples,
        n_evaluated=int(np.sum(good)),
        n_failed=n_failed,
        outputs=sobol_outputs,
        recommendations=recs,
    )
