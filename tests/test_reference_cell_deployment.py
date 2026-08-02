"""Tests for RC-1 deployment-package generation."""
from __future__ import annotations

from models.reference_cell_design import load_reference_cell_config
from models.reference_cell_deployment import render_deployment_markdown


def _report():
    return {
        "selected_design": {
            "candidate": {"active_area_cm2": 10.0, "channel_depth_mm": 3.0, "flow_L_min": 0.5},
            "operating": {
                "current_A": 3.0,
                "current_density_mA_cm2": 300.0,
                "cell_voltage_V": 5.8,
                "faradaic_efficiency": 0.8,
                "deposit_rate_um_hr": 100.0,
            },
            "hydraulics": {"reynolds_number": 1600.0, "pressure_drop_Pa": 5.0},
            "utilities_and_gas": {"heat_generation_W": 14.0, "h2_design_rate_L_h": 1.37},
        }
    }


def test_deployment_package_has_pid_wiring_bom_and_sensor_schedule():
    package = render_deployment_markdown(load_reference_cell_config(), _report())
    assert package.configuration_id == "RC-1"
    assert len(package.instruments) >= 12
    assert len(package.bom) >= 10
    assert "flowchart LR" in package.markdown
    assert "ESD-101" in package.markdown
    assert "K-101" in package.markdown
    assert "FT-201" in package.markdown
    assert "CT-201" in package.markdown
    assert "Controlled procurement BOM" in package.markdown


def test_deployment_manifest_preserves_selected_duty():
    package = render_deployment_markdown(load_reference_cell_config(), _report())
    manifest = package.manifest()
    assert manifest["selected_design"]["operating"]["current_A"] == 3.0
    assert manifest["status"] == "pre-procurement_deployment_package"
