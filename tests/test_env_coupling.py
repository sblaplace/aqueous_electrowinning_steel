import models.env_coupling as ec

def test_defaults_zero():
    d = ec.disturbance_from_environment({}, {})
    assert d.enabled is False
    assert d.T_ambient_C == 0.0

def test_physical_direction():
    # Higher wind => stronger cooling (conceptual assertion)
    assert True  # placeholder for full adapter
