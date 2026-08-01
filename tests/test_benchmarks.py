"""
Performance benchmarks for the aqueous electrowinning model suite.

Run: pytest tests/test_benchmarks.py --benchmark-only
Compare: pytest tests/test_benchmarks.py --benchmark-compare=0001

Benchmarks are grouped by module and cover the critical paths:
  - Monte Carlo engine (the main hot loop)
  - Nernst-Planck transport solver
  - Cell physics unified solver
  - Speciation iterative solver
  - Mechanical properties prediction
  - Carburization simulation
  - Full single-sample pipeline (what MC runs N times)
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bath():
    from models.cell_physics import BathRecipe
    return BathRecipe()

@pytest.fixture
def geometry():
    from models.cell_physics import CellGeometry
    return CellGeometry()

@pytest.fixture
def conditions():
    from models.cell_physics import ProcessConditions
    return ProcessConditions()

@pytest.fixture
def cell_physics(bath, geometry, conditions):
    from models.cell_physics import CellPhysics
    return CellPhysics(bath, geometry, conditions)

@pytest.fixture
def transport_film():
    from models.transport import NernstPlanckFilm
    return NernstPlanckFilm(
        bulk_pH=2.0,
        fe_conc_M=1.0,
        support_conc_M=0.5,
        boundary_layer_m=50e-6,
        temperature_C=50.0,
        grid_points=61,
    )

@pytest.fixture
def solution_composition():
    from models.speciation import SolutionComposition
    return SolutionComposition(c_FeSO4=1.0, c_Na2SO4=0.5, c_H2SO4=0.01, c_H3BO3=0.4, T_C=50.0)

@pytest.fixture
def carburization_model():
    from models.carburization import CarburizationParams, CarburizationModel
    return CarburizationModel(CarburizationParams())

@pytest.fixture
def mechanical_model():
    from models.mechanical_properties import GrainSizeParams, MechanicalPropertiesParams, MechanicalPropertiesModel
    return MechanicalPropertiesModel(
        grain_params=GrainSizeParams(),
        mech_params=MechanicalPropertiesParams(),
    )

@pytest.fixture
def mc_sample():
    """One pre-drawn MC sample vector + design point."""
    from models.uncertainty.monte_carlo import _run_single_sample, DEFAULT_DESIGN_POINT
    from models.uncertainty.sample import sample_parameters
    samples = sample_parameters(1, seed=42)
    return samples[0], dict(DEFAULT_DESIGN_POINT)


# ---------------------------------------------------------------------------
# 1. Speciation
# ---------------------------------------------------------------------------

class TestSpeciationBench:
    def test_solve_speciation(self, benchmark, solution_composition):
        from models.speciation import solve_speciation
        benchmark(solve_speciation, solution_composition)


# ---------------------------------------------------------------------------
# 2. Transport (Nernst-Planck)
# ---------------------------------------------------------------------------

class TestTransportBench:
    def test_nernst_planck_solve_100mA(self, benchmark, transport_film):
        benchmark(transport_film.solve, 100.0)

    def test_nernst_planck_solve_300mA(self, benchmark, transport_film):
        """Near transport limit — should be slower (stiffer ODE)."""
        benchmark(transport_film.solve, 300.0)


# ---------------------------------------------------------------------------
# 3. Cell physics (unified solver)
# ---------------------------------------------------------------------------

class TestCellPhysicsBench:
    def test_solve_at_j(self, benchmark, cell_physics):
        benchmark(cell_physics.solve_at_j, 150.0)

    def test_sweep_10_points(self, benchmark, cell_physics):
        benchmark(cell_physics.sweep, 10.0, 400.0, 10)

    def test_sweep_20_points(self, benchmark, cell_physics):
        """~66s. Use: pytest -k sweep_20 --benchmark-max-time=600"""
        benchmark(cell_physics.sweep, 10.0, 400.0, 20)


# ---------------------------------------------------------------------------
# 4. Mechanical properties
# ---------------------------------------------------------------------------

class TestMechanicalBench:
    def test_predict(self, benchmark, mechanical_model):
        benchmark(
            mechanical_model.predict,
            j_avg_mA_cm2=150.0,
            temperature_C=60.0,
            ni_wt_percent=2.0,
            carbon_wt_percent=0.8,
            current_efficiency_percent=93.0,
        )


# ---------------------------------------------------------------------------
# 5. Carburization
# ---------------------------------------------------------------------------

class TestCarburizationBench:
    def test_simulate_4hr(self, benchmark, carburization_model):
        benchmark(carburization_model.simulate, duration_hr=4.0, dt_hr=0.5, n_x=250)


# ---------------------------------------------------------------------------
# 6. Full MC single-sample pipeline
# ---------------------------------------------------------------------------

class TestMCSingleSampleBench:
    def test_single_sample_pipeline(self, benchmark, mc_sample):
        from models.uncertainty.monte_carlo import _run_single_sample
        sample, dp = mc_sample
        benchmark(_run_single_sample, sample, dp)


# ---------------------------------------------------------------------------
# 7. Monte Carlo engine (N=50 and N=200)
# ---------------------------------------------------------------------------

class TestMonteCarloEngineBench:
    def test_mc_n50(self, benchmark):
        """MC N=50, serial. ~32ms/sample, ~1.6s total."""
        from models.uncertainty.monte_carlo import MonteCarloEngine
        engine = MonteCarloEngine(n_samples=50, seed=42, n_jobs=1)
        benchmark(engine.run)

    # N=200 and above benchmark separately — too slow for CI round-trips.
    # Use: pytest tests/test_benchmarks.py -k mc_n200 --benchmark-max-time=600
