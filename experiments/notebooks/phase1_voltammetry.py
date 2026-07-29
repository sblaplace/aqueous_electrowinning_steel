"""Executable Phase I CV/LSV analysis example.

Run from the repository root with:
    python experiments/notebooks/phase1_voltammetry.py path/to/run.csv
"""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from models.experimental_data import load_measurements, summarize_run
from models.voltammetry import baseline_correct, extrema, plot_polarization, scan_rate_V_s

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default="experiments/data/voltammetry_template.csv")
parser.add_argument("--output", default="docs/figures/phase1_polarization.png")
args = parser.parse_args()

data = baseline_correct(load_measurements(args.csv))
print(summarize_run(data))
try:
    print({"scan_rate_V_s": scan_rate_V_s(data), **extrema(data)})
except ValueError as exc:
    print(f"scan rate unavailable: {exc}")
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
plot_polarization(data)
plt.tight_layout()
plt.savefig(args.output, dpi=180)
print(f"wrote {args.output}")
