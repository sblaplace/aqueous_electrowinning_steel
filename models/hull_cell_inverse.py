"""Inverse Hull-cell analysis: measured thickness profile → local FE(j) calibration.

The forward map (:func:`hull_cell.hull_current_distribution`) assigns a
primary current density ``j_primary(s)`` to every strip of an angled Hull
panel, normalized to the applied current.  A real gate-2 run measures the
deposit thickness profile ``h(s)`` across that same panel (masked strips,
point micrometer, or profilometry trace).  Combining the two inverts the
Faraday arithmetic strip-by-strip:

.. math::

    m''(s)   &= \\rho_\\mathrm{Fe}\\, h(s)                     & \\text{(local Fe mass per area)}\\\\
    Q''(s)   &= m''(s)\\, nF / M                              & \\text{(local charge consumed)}\\\\
    j_\\mathrm{act}(s) &= Q''(s) / t_\\mathrm{run}            & \\text{(local actual current)}\\\\
    \\mathrm{FE}(s) &= j_\\mathrm{act}(s) / j_\\mathrm{primary}(s) & \\text{(local apparent Fe FE)}

The result is a **FE-vs-j curve from a single panel**: the angled cell
exposes a range of current densities at once, and the thickness profile says
how much of the current at each density actually made iron.

Two identities make the analysis checkable:

1. **Gravimetric consistency.**  The current-weighted mean of the local FE
   profile must equal the whole-panel gravimetric FE of the same panel:

   .. math::

       \\sum_i \\mathrm{FE}_i I_i \\Big/ \\sum_i I_i =
       \\frac{m_\\mathrm{grav}\\, nF}{M\\, I_\\mathrm{total}\\, t_\\mathrm{run}}

   The module reports this as a mass-closure ratio between the
   profile-integrated iron mass and the weighed mass gain.

2. **Tafel-consistent shape.**  With Fe and HER both in the Tafel region,
   ``logit(FE) = ln(FE/(1-FE))`` is linear in ``ln(j)`` with slope
   ``b = 1 - α_H/α_Fe < 0``.  A two-parameter fit
   ``FE(j) = sigmoid(a + b·ln j)`` of the recovered profile is therefore a
   compact calibration that the FE engine (``diffusion_layer_1d.py``) and the
   Bayesian calibration pipeline can consume.  The fit is performed by
   nonlinear least squares **in FE space** (not on the logit transform):
   logit-space least squares is biased under measurement noise because logit
   is convex for FE > 0.5 — noisy high-FE strips dominate ``E[logit(FE)]`` —
   while FE-space residuals are unbiased.  Monte-Carlo round trips (see
   :func:`fit_fe_vs_j`) quantify the protocol consequence: a point-micrometer
   profile on one panel pins FE at a reference current density to ~±2 %, while
   the slope needs lower thickness noise, more strips, or repeated panels.

As with ``hull_cell.py``, this is **apparent Fe FE**: it is only an iron
efficiency when the deposit is verified as iron (composition, no retained
salts/oxide/moisture).  FE above 100% is retained as a QA flag, not clipped.
The primary map's assumptions (no edge/shield effects, no kinetics, no
bubbles, no conductivity gradients) are inherited unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .electrochemistry import FARADAY as FARADAY_CONSTANT_C_MOL
from .electrochemistry import M_FE_G as MOLAR_MASS_FE_G_MOL
from .electrochemistry import Z_FE as ELECTRONS_PER_FE

IRON_DENSITY_G_CM3 = 7.874  # g/cm³, bulk α-Fe at 25 °C

DISTRIBUTION_REQUIRED_COLUMNS = {
    "position_cm_from_near_edge",
    "current_density_mA_cm2",
    "segment_area_cm2",
    "segment_current_A",
}

#: Relative departure from mass closure accepted before flagging the profile
#: as inconsistent with the weighing (loose on purpose: profilometry on a
#: rough deposit easily reads 10 % high or low).
DEFAULT_MASS_CLOSURE_TOLERANCE = 0.15

LOGIT_FIT_EPS = 1e-6  # clip FE into (eps, 1-eps) before logit transform


def _validate_distribution(distribution: pd.DataFrame) -> None:
    missing = DISTRIBUTION_REQUIRED_COLUMNS - set(distribution.columns)
    if missing:
        raise ValueError(
            f"Distribution is missing columns: {', '.join(sorted(missing))}"
        )
    if distribution.empty:
        raise ValueError("Distribution must not be empty")


def thickness_from_faraday(
    current_density_mA_cm2: float | np.ndarray,
    faradaic_efficiency: float | np.ndarray,
    duration_s: float,
    *,
    density_g_cm3: float = IRON_DENSITY_G_CM3,
    molar_mass_g_mol: float = MOLAR_MASS_FE_G_MOL,
    n_electrons: int = ELECTRONS_PER_FE,
) -> np.ndarray:
    """Deposit thickness (µm) implied by current density, FE, and time.

    Forward Faraday law: ``h = j·FE·t·M / (nF·ρ)``.  This is the quantity a
    (perfect) thickness measurement would report for a known local
    ``j`` and ``FE``.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    j_A_m2 = np.atleast_1d(np.asarray(current_density_mA_cm2, dtype=float)) * 10.0
    fe = np.atleast_1d(np.asarray(faradaic_efficiency, dtype=float))
    if np.any(~np.isfinite(j_A_m2)) or np.any(j_A_m2 <= 0):
        raise ValueError("current_density_mA_cm2 must be finite and positive")
    if np.any(~np.isfinite(fe)) or np.any(fe < 0):
        raise ValueError("faradaic_efficiency must be finite and non-negative")
    if not (np.isfinite(density_g_cm3) and density_g_cm3 > 0):
        raise ValueError("density_g_cm3 must be finite and positive")
    if not (np.isfinite(molar_mass_g_mol) and molar_mass_g_mol > 0):
        raise ValueError("molar_mass_g_mol must be finite and positive")
    if isinstance(n_electrons, bool) or int(n_electrons) != n_electrons or n_electrons <= 0:
        raise ValueError("n_electrons must be a positive integer")

    rho_kg_m3 = density_g_cm3 * 1000.0
    M_kg_mol = molar_mass_g_mol * 1e-3
    h_m = (
        j_A_m2
        * fe
        * duration_s
        * M_kg_mol
        / (int(n_electrons) * FARADAY_CONSTANT_C_MOL * rho_kg_m3)
    )
    return h_m * 1e6  # µm


def faradaic_efficiency_from_thickness(
    current_density_mA_cm2: float | np.ndarray,
    thickness_um: float | np.ndarray,
    duration_s: float,
    *,
    density_g_cm3: float = IRON_DENSITY_G_CM3,
    molar_mass_g_mol: float = MOLAR_MASS_FE_G_MOL,
    n_electrons: int = ELECTRONS_PER_FE,
) -> np.ndarray:
    """Apparent Fe FE implied by a measured thickness at a known local j.

    Inverse of :func:`thickness_from_faraday`: the measured deposit mass per
    area is converted to the charge that must have passed through that strip
    (at 100 % Fe), and divided by the charge the primary map says the strip
    carried.
    """
    j_A_m2 = np.atleast_1d(np.asarray(current_density_mA_cm2, dtype=float)) * 10.0
    h_um = np.atleast_1d(np.asarray(thickness_um, dtype=float))
    if np.any(~np.isfinite(j_A_m2)) or np.any(j_A_m2 <= 0):
        raise ValueError("current_density_mA_cm2 must be finite and positive")
    if np.any(~np.isfinite(h_um)) or np.any(h_um < 0):
        raise ValueError("thickness_um must be finite and non-negative")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not (np.isfinite(density_g_cm3) and density_g_cm3 > 0):
        raise ValueError("density_g_cm3 must be finite and positive")
    if not (np.isfinite(molar_mass_g_mol) and molar_mass_g_mol > 0):
        raise ValueError("molar_mass_g_mol must be finite and positive")
    if isinstance(n_electrons, bool) or int(n_electrons) != n_electrons or n_electrons <= 0:
        raise ValueError("n_electrons must be a positive integer")

    # Q'' [C/m²] = h_um·ρ_g_cm3·nF/M_g_mol  (the unit conversion cancels).
    charge_density_C_m2 = (
        h_um
        * density_g_cm3
        * int(n_electrons)
        * FARADAY_CONSTANT_C_MOL
        / molar_mass_g_mol
    )
    j_actual_A_m2 = charge_density_C_m2 / duration_s
    return j_actual_A_m2 / j_A_m2


def thickness_to_local_faradaic_efficiency(
    distribution: pd.DataFrame,
    thickness_um: np.ndarray | pd.Series,
    duration_s: float,
    *,
    thickness_uncertainty_um: float | np.ndarray | None = None,
    density_g_cm3: float = IRON_DENSITY_G_CM3,
    molar_mass_g_mol: float = MOLAR_MASS_FE_G_MOL,
    n_electrons: int = ELECTRONS_PER_FE,
) -> pd.DataFrame:
    """Convert a measured thickness profile into strip-by-strip apparent FE.

    ``distribution`` must be the forward map
    (:func:`hull_cell.hull_current_distribution`) whose rows correspond
    one-to-one to the thickness measurements — i.e. build the map with the
    same ``n_segments`` as the number of thickness readings taken across the
    panel (the Day-1 run sheet's strip count is the natural choice).

    Returns a DataFrame with one row per strip:

    - ``position_cm_from_near_edge``, ``current_density_mA_cm2`` (primary),
      ``segment_area_cm2``, ``segment_current_A`` — copied from the map;
    - ``deposit_thickness_um`` — as measured;
    - ``deposit_mass_mg_cm2`` — thickness × ρ_Fe;
    - ``deposition_current_density_mA_cm2`` — the current density that would
      produce the measured deposit at 100 % Fe (i.e. charge-equivalent of the
      deposit divided by run time; this is *not* the total local current);
    - ``apparent_faradaic_efficiency`` — deposition / primary (may exceed 1);
    - ``fe_uncertainty`` — propagated from ``thickness_uncertainty_um``
      (None if no thickness uncertainty was supplied);
    - ``fe_qa_flag`` — ``"above_100"``, ``"zero_deposit"``, or ``"ok"``.
    """
    _validate_distribution(distribution)
    if len(thickness_um) != len(distribution):
        raise ValueError(
            "thickness_um must have one entry per distribution row "
            f"(got {len(thickness_um)} for {len(distribution)} rows)"
        )
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")

    h = np.asarray(thickness_um, dtype=float)
    if np.any(~np.isfinite(h)) or np.any(h < 0):
        raise ValueError("thickness_um must be finite and non-negative")

    j_primary = distribution["current_density_mA_cm2"].to_numpy(float)
    fe = faradaic_efficiency_from_thickness(
        j_primary,
        h,
        duration_s,
        density_g_cm3=density_g_cm3,
        molar_mass_g_mol=molar_mass_g_mol,
        n_electrons=n_electrons,
    )

    local = pd.DataFrame({
        "position_cm_from_near_edge": distribution[
            "position_cm_from_near_edge"
        ].to_numpy(float),
        "current_density_mA_cm2": j_primary,
        "segment_area_cm2": distribution["segment_area_cm2"].to_numpy(float),
        "segment_current_A": distribution["segment_current_A"].to_numpy(float),
        "deposit_thickness_um": h,
        "deposit_mass_mg_cm2": h * density_g_cm3 * 1e-1,  # µm × g/cm³ → mg/cm²
        "deposition_current_density_mA_cm2": fe * j_primary,
        "apparent_faradaic_efficiency": fe,
    })

    if thickness_uncertainty_um is not None:
        sigma_h = np.asarray(thickness_uncertainty_um, dtype=float)
        if sigma_h.ndim == 0:
            sigma_h = np.full_like(h, float(sigma_h))
        if len(sigma_h) != len(h):
            raise ValueError("thickness_uncertainty_um must match thickness_um length")
        if np.any(~np.isfinite(sigma_h)) or np.any(sigma_h < 0):
            raise ValueError("thickness_uncertainty_um must be finite and non-negative")
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(h > 0, sigma_h / h, np.nan)
        local["fe_uncertainty"] = np.where(np.isfinite(relative), fe * relative, np.nan)
    else:
        local["fe_uncertainty"] = np.nan

    qa = np.full(len(local), "ok", dtype=object)
    qa[fe > 1.0] = "above_100"
    qa[h <= 0] = "zero_deposit"
    local["fe_qa_flag"] = qa
    return local


def implied_panel_faradaic_efficiency(local: pd.DataFrame) -> float:
    """Current-weighted mean of the local FE profile.

    Identical to the whole-panel gravimetric FE if the profile and the
    weighing describe the same panel; this is the profile-side check of the
    mass-closure identity.
    """
    _validate_local(local)
    numerator = float((local["apparent_faradaic_efficiency"] * local["segment_current_A"]).sum())
    denominator = float(local["segment_current_A"].sum())
    if denominator <= 0:
        raise ValueError("Distribution carries no current; cannot form a weighted mean")
    return numerator / denominator


def _validate_local(local: pd.DataFrame) -> None:
    required = {
        "segment_current_A",
        "apparent_faradaic_efficiency",
        "current_density_mA_cm2",
        "segment_area_cm2",
    }
    missing = required - set(local.columns)
    if missing:
        raise ValueError(f"Local FE table is missing columns: {', '.join(sorted(missing))}")


def mass_closure(
    distribution: pd.DataFrame,
    thickness_um: np.ndarray | pd.Series,
    gravimetric_mass_gain_g: float,
    *,
    tolerance: float = DEFAULT_MASS_CLOSURE_TOLERANCE,
    density_g_cm3: float = IRON_DENSITY_G_CM3,
) -> dict:
    """Compare profile-integrated iron mass with the weighed mass gain.

    The closure ratio is ``integrated / gravimetric``; values inside
    ``[1 - tolerance, 1 + tolerance]`` pass.  A ratio far from 1 flags
    profilometry bias (rough/porous deposit), retained salts, missed or
    double-counted area, or a weighing error — before the profile is trusted
    as a FE map.
    """
    _validate_distribution(distribution)
    if len(thickness_um) != len(distribution):
        raise ValueError("thickness_um must have one entry per distribution row")
    h = np.asarray(thickness_um, dtype=float)
    if np.any(~np.isfinite(h)) or np.any(h < 0):
        raise ValueError("thickness_um must be finite and non-negative")
    if not np.isfinite(gravimetric_mass_gain_g) or gravimetric_mass_gain_g <= 0:
        raise ValueError("gravimetric_mass_gain_g must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not (np.isfinite(density_g_cm3) and density_g_cm3 > 0):
        raise ValueError("density_g_cm3 must be finite and positive")

    # µm × cm² × g/cm³ × 1e-4 → g
    integrated_g = float((h * distribution["segment_area_cm2"].to_numpy(float)).sum())
    integrated_g *= density_g_cm3 * 1e-4

    ratio = integrated_g / gravimetric_mass_gain_g
    balanced = (1.0 - tolerance) <= ratio <= (1.0 + tolerance)
    note = (
        "profile-integrated mass agrees with the gravimetric mass gain"
        if balanced
        else (
            "profile-integrated mass disagrees with the gravimetric mass gain "
            "(profilometry bias, retained salts, or weighing error) — resolve "
            "before trusting the FE map"
        )
    )
    return {
        "integrated_mass_g": integrated_g,
        "gravimetric_mass_g": float(gravimetric_mass_gain_g),
        "closure_ratio": ratio,
        "tolerance": float(tolerance),
        "mass_balanced": balanced,
        "note": note,
    }


def aggregate_fe_curve(
    local: pd.DataFrame,
    *,
    n_bins: int = 6,
    j_edges_mA_cm2: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bin the strip-wise FE profile into a FE(j) calibration table.

    Bins are log-spaced in the primary current density by default (the Hull
    map is roughly log-linear in position).  Each bin reports the
    current-weighted mean FE — the quantity consistent with gravimetry — plus
    the spread of the strip values, the strip count, and the fraction of
    panel area/current the bin covers.
    """
    _validate_local(local)
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or int(n_bins) < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    n_bins = int(n_bins)

    j = local["current_density_mA_cm2"].to_numpy(float)
    if np.any(j <= 0):
        raise ValueError("Local FE table must carry strictly positive current densities")

    if j_edges_mA_cm2 is None:
        edges = np.geomspace(j.min(), j.max(), n_bins + 1)
    else:
        edges = np.asarray(j_edges_mA_cm2, dtype=float)
        if edges.ndim != 1 or len(edges) < 3 or np.any(np.diff(edges) <= 0):
            raise ValueError("j_edges_mA_cm2 must be a strictly increasing 1-D array")
        edges = np.unique(edges)

    bin_index = np.clip(np.searchsorted(edges, j, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        mask = bin_index == i
        if not np.any(mask):
            continue
        sub = local.loc[mask]
        i_current = sub["segment_current_A"].to_numpy(float)
        i_fe = sub["apparent_faradaic_efficiency"].to_numpy(float)
        j_bin = j[mask]
        weighted_mean = float((i_fe * i_current).sum() / i_current.sum())
        rows.append({
            "j_min_mA_cm2": float(edges[i]),
            "j_max_mA_cm2": float(edges[i + 1]),
            "j_geometric_mean_mA_cm2": float(np.exp(np.mean(np.log(j_bin)))),
            "fe_current_weighted_mean": weighted_mean,
            "fe_strip_std": float(i_fe.std(ddof=1)) if len(i_fe) > 1 else float("nan"),
            "n_strips": int(len(sub)),
            "area_fraction": float(sub["segment_area_cm2"].sum() / local["segment_area_cm2"].sum()),
            "current_fraction": float(i_current.sum() / local["segment_current_A"].sum()),
        })
    if not rows:
        raise ValueError("No current-density bins contained any strip")
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class LogitFeFit:
    """Two-parameter Tafel-consistent FE(j) calibration: logit(FE) = a + b·ln(j).

    Parameters are estimated by nonlinear least squares **in FE space** (see
    :func:`fit_fe_vs_j`); ``predict`` evaluates the equivalent sigmoid
    ``FE(j) = 1/(1 + exp(-(a + b·ln j)))``.
    """

    a: float
    b: float
    r_squared: Optional[float]
    n_points: int
    reference_j_mA_cm2: float
    fe_at_reference: float
    note: str = ""

    def predict(self, j_mA_cm2: float | np.ndarray) -> np.ndarray:
        """Predicted FE at current density(ies), clipped to (0, 1)."""
        return logit_fe(j_mA_cm2, self.a, self.b)

    def to_dict(self) -> dict:
        return asdict(self)


def _sigmoid_fe(lnj: np.ndarray, a: float, b: float) -> np.ndarray:
    """FE = sigmoid(a + b·ln j), the model function for the FE-space fit."""
    return 1.0 / (1.0 + np.exp(-np.clip(a + b * lnj, -700.0, 700.0)))


def logit_fe(j_mA_cm2: float | np.ndarray, a: float, b: float) -> np.ndarray:
    """FE(j) = sigmoid(a + b·ln j), the model form used by the fit."""
    j = np.atleast_1d(np.asarray(j_mA_cm2, dtype=float))
    if np.any(~np.isfinite(j)) or np.any(j <= 0):
        raise ValueError("j_mA_cm2 must be finite and positive")
    logit = a + b * np.log(j)
    fe = 1.0 / (1.0 + np.exp(-np.clip(logit, -700.0, 700.0)))
    return np.clip(fe, LOGIT_FIT_EPS, 1.0 - LOGIT_FIT_EPS)


def fit_fe_vs_j(
    local: pd.DataFrame,
    *,
    reference_j_mA_cm2: float = 100.0,
) -> LogitFeFit:
    """Fit FE(j) = sigmoid(a + b·ln j) to the strip-wise FE profile.

    Under Tafel kinetics for both Fe deposition and HER,
    ``b = 1 - α_H/α_Fe < 0`` — FE falls as current density rises because HER
    takes an increasing share.

    The fit is nonlinear least squares **in FE space**, weighted by the
    propagated ``fe_uncertainty`` when present.  This matters: least squares
    on the logit transform is biased under measurement noise, because logit is
    convex for FE > 0.5 and noisy high-FE strips (the thin, far-edge end of a
    Hull panel) drag ``E[logit(FE)]`` far above ``logit(E[FE])``.  Fitting the
    sigmoid directly in FE space keeps the estimate unbiased (verified by
    Monte-Carlo round trip: with ~10 strips, thickness noise σ_h = 2 µm gives
    slope error ≈ ±0.2 and FE@100 mA/cm² error ≈ ±1.7 %; at σ_h = 0.5 µm the
    slope error drops to ≈ ±0.05).  In protocol terms: a point-micrometer
    profile on one panel pins **FE at a reference current density**; pinning
    the slope needs profilometry, more strips, or repeated panels.

    Strips with FE outside (0, 1) (QA flags ``above_100`` / ``zero_deposit``)
    are excluded from the fit but retained in the returned table; the fit
    requires at least three usable strips.
    """
    _validate_local(local)
    if not (np.isfinite(reference_j_mA_cm2) and reference_j_mA_cm2 > 0):
        raise ValueError("reference_j_mA_cm2 must be finite and positive")

    usable = local.loc[
        (local["apparent_faradaic_efficiency"] > LOGIT_FIT_EPS)
        & (local["apparent_faradaic_efficiency"] < 1.0 - LOGIT_FIT_EPS)
    ]
    if len(usable) < 3:
        return LogitFeFit(
            a=float("nan"), b=float("nan"), r_squared=None, n_points=len(usable),
            reference_j_mA_cm2=float(reference_j_mA_cm2), fe_at_reference=float("nan"),
            note="Fewer than 3 usable strips (0 < FE < 1); no calibration fit returned.",
        )

    x = np.log(usable["current_density_mA_cm2"].to_numpy(float))
    fe = usable["apparent_faradaic_efficiency"].to_numpy(float)
    fe_clipped = np.clip(fe, LOGIT_FIT_EPS, 1.0 - LOGIT_FIT_EPS)

    # Starting guess from logit OLS: biased under noise, but a robust start.
    p0 = np.polyfit(x, np.log(fe_clipped / (1.0 - fe_clipped)), 1)[::-1]

    sigma = None
    if "fe_uncertainty" in usable.columns:
        sig = usable["fe_uncertainty"].to_numpy(float)
        finite = np.isfinite(sig) & (sig > 0)
        if finite.sum() >= 3:
            # curve_fit treats sigma as absolute errors; rows without a
            # measured uncertainty get a neutral weight (≈ negligible vs the
            # 1/σ² ≈ 10²–10⁴ weights of the measured rows).
            sigma = np.where(finite, sig, 1.0)

    try:
        popt, _ = curve_fit(
            _sigmoid_fe, x, fe, p0=p0, sigma=sigma, maxfev=20000
        )
    except RuntimeError:
        return LogitFeFit(
            a=float("nan"), b=float("nan"), r_squared=None, n_points=len(usable),
            reference_j_mA_cm2=float(reference_j_mA_cm2), fe_at_reference=float("nan"),
            note="FE-space NLS did not converge; check the profile for QA flags.",
        )

    a, b = popt
    prediction = _sigmoid_fe(x, a, b)
    ss_res = float(np.sum((fe - prediction) ** 2))
    ss_tot = float(np.sum((fe - fe.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return LogitFeFit(
        a=float(a),
        b=float(b),
        r_squared=r_squared,
        n_points=int(len(usable)),
        reference_j_mA_cm2=float(reference_j_mA_cm2),
        fe_at_reference=float(logit_fe(reference_j_mA_cm2, a, b)[0]),
        note=(
            "FE-space NLS fit of FE(j) = sigmoid(a + b·ln j); b = 1 - α_H/α_Fe < 0 "
            "expected. Apparent Fe FE pending deposit composition verification."
        ),
    )


def synthesize_thickness_profile(
    distribution: pd.DataFrame,
    duration_s: float,
    fe_logit_a: float,
    fe_logit_b: float,
    *,
    noise_sigma_um: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate a synthetic 'measured' thickness profile from a known FE(j).

    Uses the forward map's local current densities and a logit FE model, then
    (optionally) adds Gaussian measurement noise.  Intended for tests,
    demonstrations, and Monte-Carlo-style uncertainty studies — the recovery
    of ``fe_logit_a``/``fe_logit_b`` by :func:`fit_fe_vs_j` is the round-trip
    check the test suite relies on.
    """
    _validate_distribution(distribution)
    j = distribution["current_density_mA_cm2"].to_numpy(float)
    fe = logit_fe(j, fe_logit_a, fe_logit_b)
    h_true = thickness_from_faraday(j, fe, duration_s)
    if noise_sigma_um > 0:
        if rng is None:
            rng = np.random.default_rng()
        h_measured = h_true + rng.normal(0.0, noise_sigma_um, size=h_true.shape)
        h_measured = np.clip(h_measured, 0.0, None)
    else:
        h_measured = h_true.copy()
    return pd.DataFrame({
        "position_cm_from_near_edge": distribution[
            "position_cm_from_near_edge"
        ].to_numpy(float),
        "current_density_mA_cm2": j,
        "true_faradaic_efficiency": fe,
        "true_thickness_um": h_true,
        "measured_thickness_um": h_measured,
    })


def analyze_hull_panel(
    distribution: pd.DataFrame,
    thickness_um: np.ndarray | pd.Series,
    duration_s: float,
    gravimetric_mass_gain_g: float | None = None,
    *,
    thickness_uncertainty_um: float | np.ndarray | None = None,
    n_bins: int = 6,
    mass_closure_tolerance: float = DEFAULT_MASS_CLOSURE_TOLERANCE,
    reference_j_mA_cm2: float = 100.0,
) -> dict:
    """Run the full inverse pipeline for one panel and bundle the results.

    Returns a dict with the local FE table, the binned FE(j) calibration, the
    logit fit, the mass-closure check (when a gravimetric mass is supplied),
    and the profile-implied panel FE — everything the driver serializes to a
    JSON report.
    """
    local = thickness_to_local_faradaic_efficiency(
        distribution,
        thickness_um,
        duration_s,
        thickness_uncertainty_um=thickness_uncertainty_um,
    )
    binned = aggregate_fe_curve(local, n_bins=n_bins)
    fit = fit_fe_vs_j(local, reference_j_mA_cm2=reference_j_mA_cm2)
    result: dict = {
        "local_faradaic_efficiency": local,
        "binned_fe_curve": binned,
        "logit_fit": fit,
        "implied_panel_faradaic_efficiency": implied_panel_faradaic_efficiency(local),
    }
    if gravimetric_mass_gain_g is not None:
        result["mass_closure"] = mass_closure(
            distribution,
            thickness_um,
            gravimetric_mass_gain_g,
            tolerance=mass_closure_tolerance,
        )
    return result
