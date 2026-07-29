import pytest
from pathlib import Path
from models.pid import generate_pid_overview, generate_pid_detailed

def test_pid_overview(tmp_path):
    out = tmp_path / "pid_overview.png"
    p = generate_pid_overview(out)
    assert p.exists()
    assert p.stat().st_size > 10000

def test_pid_detailed(tmp_path):
    out = tmp_path / "pid_detailed.png"
    p = generate_pid_detailed(out)
    assert p.exists()
    assert p.stat().st_size > 10000
