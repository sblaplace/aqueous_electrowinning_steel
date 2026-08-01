"""
Numba-accelerated Nernst-Planck film integration.

Provides JIT-compiled versions of the ODE right-hand side and an adaptive
RK45 (Dormand-Prince) integrator that replaces scipy.integrate.solve_ivp
for the cathode film transport problem.

The entire integration loop — RHS evaluation, step-size control, and
interpolation to the output grid — runs inside numba-compiled code,
eliminating Python callback overhead and the Fortran bridge cost that
dominate the scipy.solve_ivp path.

Fallback: if numba is unavailable, transport.py falls back to solve_ivp.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):
        """Identity decorator when numba is unavailable."""
        def wrapper(fn):
            return fn
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return wrapper

# Water autoprotolysis in (mol/m^3)^2
KW_SI = 1.0e-8


# ─── JIT-compiled adaptive RK45 integrator ────────────────────────

def _make_integrate():
    """Build the numba-jitted film integrator.

    Returns a function with signature:
        integrate_film(y0, n_fe, n_s, x_start, x_end, x_eval,
                       d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe,
                       rtol, atol, max_steps)
    Returns a (5, n_eval) array of solution values at x_eval points.
    """

    # Dormand-Prince coefficients
    A21 = 0.2
    A31 = 3.0 / 40.0
    A32 = 9.0 / 40.0
    A41 = 44.0 / 45.0
    A42 = -56.0 / 15.0
    A43 = 32.0 / 9.0
    A51 = 19372.0 / 6561.0
    A52 = -25360.0 / 2187.0
    A53 = 64448.0 / 6561.0
    A54 = -212.0 / 729.0
    A61 = 9017.0 / 3168.0
    A62 = -355.0 / 33.0
    A63 = 46732.0 / 5247.0
    A64 = 49.0 / 176.0
    A65 = -5103.0 / 18656.0
    A71 = 35.0 / 384.0
    A73 = 500.0 / 1113.0
    A74 = 125.0 / 192.0
    A75 = -2187.0 / 6784.0
    A76 = 11.0 / 84.0

    # Error weights (5th - 4th order difference)
    E1 = 71.0 / 57600.0
    E3 = -71.0 / 16695.0
    E4 = 71.0 / 1920.0
    E5 = -17253.0 / 339200.0
    E6 = 22.0 / 525.0
    E7 = -1.0 / 40.0

    @_njit(cache=True)
    def _rhs(x, y, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe):
        c_fe = y[0] if y[0] > floor_fe else floor_fe
        c_h = y[1] if y[1] > 1e-20 else 1e-20
        c_na = y[2] if y[2] > 0.0 else 0.0
        c_so4 = y[3] if y[3] > 0.0 else 0.0

        c_oh = KW_SI / c_h
        k = KW_SI / (c_h * c_h)
        a = d_h + d_oh * k
        b = (d_h * c_h + d_oh * c_oh) / a

        numerator = -2.0 * n_fe / d_fe - (1.0 + k) * n_s / a
        denominator = 4.0 * c_fe + (1.0 + k) * b + c_na + 4.0 * c_so4
        f_dphi = numerator / denominator

        dc_fe = -n_fe / d_fe - 2.0 * c_fe * f_dphi
        dc_h = -n_s / a - b * f_dphi
        dc_na = -c_na * f_dphi
        dc_so4 = 2.0 * c_so4 * f_dphi
        dphi = f_dphi / f_RT

        return np.array([dc_fe, dc_h, dc_na, dc_so4, dphi])

    @_njit(cache=True)
    def _rk45_step(x, y, h, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe):
        """One Dormand-Prince step. Returns (y_new, error_norm)."""
        k1 = _rhs(x, y, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe)

        y2 = y + h * A21 * k1
        k2 = _rhs(x + h * A21, y2, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe)

        y3 = y + h * (A31 * k1 + A32 * k2)
        k3 = _rhs(x + h * (A31 + A32), y3, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe)

        y4 = y + h * (A41 * k1 + A42 * k2 + A43 * k3)
        k4 = _rhs(x + h * 0.4444444444444444, y4, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe)

        y5 = y + h * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4)
        k5 = _rhs(x + h * 0.9333333333333333, y5, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe)

        y6 = y + h * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5)
        k6 = _rhs(x + h, y6, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe)

        y_new = y + h * (A71 * k1 + A73 * k3 + A74 * k4 + A75 * k5 + A76 * k6)
        k7 = _rhs(x + h, y_new, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe)

        err_vec = h * (E1 * k1 + E3 * k3 + E4 * k4 + E5 * k5 + E6 * k6 + E7 * k7)
        return y_new, err_vec

    @_njit(cache=True)
    def integrate_film(
        y0, n_fe, n_s, x_start, x_end, x_eval,
        d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe,
        rtol=1e-8, atol=1e-10, max_steps=500000,
    ):
        """Adaptive RK45 integration of the Nernst-Planck film ODE.

        Integrates from x_start to x_end (bulk -> electrode), recording
        the solution at each x_eval point by stepping exactly to it.
        Returns (5, n_eval) array.
        """
        n_eval = len(x_eval)
        result = np.empty((5, n_eval), dtype=np.float64)

        forward = x_end > x_start
        sign = 1.0 if forward else -1.0

        span = abs(x_end - x_start)
        h = sign * span / 500.0
        h_min = sign * span * 1e-14
        h_max = sign * span * 0.5

        x = x_start
        y = y0.copy()

        eval_idx = 0

        for step in range(max_steps):
            # Record eval points at or before current x
            while eval_idx < n_eval:
                x_target = x_eval[eval_idx]
                if forward:
                    if x + h * 0.01 < x_target:
                        break
                else:
                    if x + h * 0.01 > x_target:
                        break
                # If we're very close to the target, just use current y
                for i in range(5):
                    result[i, eval_idx] = y[i]
                eval_idx += 1

            if eval_idx >= n_eval:
                break

            # Don't overshoot the next eval point
            x_next_eval = x_eval[eval_idx]
            h_to_eval = x_next_eval - x
            if abs(h_to_eval) < abs(h):
                h = h_to_eval

            # Don't overshoot the end
            h_to_end = x_end - x
            if abs(h_to_end) < abs(h):
                h = h_to_end

            if abs(h) < abs(h_min):
                h = h_min * sign

            # Try a step
            y_new, err_vec = _rk45_step(
                x, y, h, n_fe, n_s, d_fe, d_h, d_oh, d_na, d_so4, f_RT, floor_fe,
            )

            # Error norm
            err = 0.0
            for i in range(5):
                scale = atol + rtol * max(abs(y[i]), abs(y_new[i]))
                err += (err_vec[i] / scale) ** 2
            err = np.sqrt(err / 5.0)

            if err <= 1.0 or abs(h) <= abs(h_min):
                # Accept step
                x = x + h
                y = y_new

            # Step size adjustment
            if err < 1e-30:
                err = 1e-30

            safety = 0.9
            h_new = h * safety * err ** (-0.2)

            # Limit step size changes
            if abs(h_new) < abs(h) * 0.2:
                h_new = h * 0.2
            elif abs(h_new) > abs(h) * 5.0:
                h_new = h * 5.0

            if abs(h_new) < abs(h_min):
                h_new = h_min * sign
            if abs(h_new) > abs(h_max):
                h_new = h_max * sign

            h = h_new

        # Fill remaining eval points
        while eval_idx < n_eval:
            for i in range(5):
                result[i, eval_idx] = y[i]
            eval_idx += 1

        return result

    return integrate_film


# ─── Module-level compiled functions (lazy init) ──────────────────

_integrate_film_jit = None


def get_integrate_film_jit():
    global _integrate_film_jit
    if _integrate_film_jit is None and _HAS_NUMBA:
        _integrate_film_jit = _make_integrate()
    return _integrate_film_jit


def has_numba() -> bool:
    """Check if numba JIT is available."""
    return _HAS_NUMBA
