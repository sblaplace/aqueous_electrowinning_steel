import numpy as np
import pandas as pd
import pytest
from models.tafel import fit_tafel


def test_recovers_synthetic_tafel_slope_and_exchange_current():
    eta = np.linspace(0.10, 0.30, 8)
    slope = 0.12
    i0 = 2e-4
    current = i0 * 10 ** (eta / slope)
    data = pd.DataFrame({"potential_V_vs_ref": -eta, "current_A": -current})
    result = fit_tafel(data, potential_min_V=-0.30, potential_max_V=-0.10)
    assert result.slope_V_decade == pytest.approx(slope, rel=1e-10)
    assert result.exchange_current_A == pytest.approx(i0, rel=1e-10)
    assert result.r_squared == pytest.approx(1.0)


def test_rejects_insufficient_points():
    data = pd.DataFrame({"potential_V_vs_ref": [-.1, -.2], "current_A": [-1, -2]})
    with pytest.raises(ValueError, match="three"):
        fit_tafel(data, potential_min_V=-.2, potential_max_V=-.1)
