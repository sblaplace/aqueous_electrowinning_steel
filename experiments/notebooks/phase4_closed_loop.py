#!/usr/bin/env python3
"""Validate and summarize a Phase IV durability/closed-loop time series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "time_hr", "current_A", "anode_area_m2", "cell_voltage_V",
    "fe_M", "ligand_M", "impurity_M", "anode_mass_loss_mg_m2",
}


def analyze(path: str | Path) -> dict:
    data = pd.read_csv(path)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if len(data) < 2 or data[list(REQUIRED)].isna().any().any():
        raise ValueError("at least two complete numeric rows are required")
    if (np.diff(data["time_hr"]) <= 0).any():
        raise ValueError("time_hr must be strictly increasing")
    if (data[["current_A", "anode_area_m2"]] <= 0).any().any():
        raise ValueError("current and anode area must be positive")

    dt = np.diff(data["time_hr"].to_numpy())
    current_mid = 0.5 * (data["current_A"].to_numpy()[1:] + data["current_A"].to_numpy()[:-1])
    charge_kAh = float(np.sum(current_mid * dt) / 1000.0)
    area = float(data["anode_area_m2"].iloc[0])
    mass_loss = float(data["anode_mass_loss_mg_m2"].iloc[-1] - data["anode_mass_loss_mg_m2"].iloc[0])
    wear = mass_loss * area / max(charge_kAh, 1e-30)
    voltage_drift = float(data["cell_voltage_V"].iloc[-1] - data["cell_voltage_V"].iloc[0])
    return {
        "provenance": "Analysis of user-supplied measurements; inspect source metadata before interpretation.",
        "duration_hr": float(data["time_hr"].iloc[-1] - data["time_hr"].iloc[0]),
        "charge_kAh": charge_kAh,
        "anode_mass_loss_mg_m2": mass_loss,
        "wear_mg_per_kAh": wear,
        "cell_voltage_drift_V": voltage_drift,
        "final_fe_M": float(data["fe_M"].iloc[-1]),
        "final_ligand_M": float(data["ligand_M"].iloc[-1]),
        "final_impurity_M": float(data["impurity_M"].iloc[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Phase IV canonical CSV")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    result = analyze(args.csv)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
