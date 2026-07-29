"""Experimental campaign manifest validation and traceability reporting.

The manifest deliberately stores file links rather than copying instrument data.
Raw exports stay immutable; processed data, run metadata, and characterization
records remain tied to one ``run_id`` for an auditable experimental record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "run_id", "phase", "technique", "status", "raw_file", "processed_file", "metadata_file",
}
VALID_PHASES = {"I", "II", "III", "IV"}
VALID_STATUSES = {"planned", "in_progress", "complete", "excluded"}
REQUIRED_METADATA = {
    "sample_id", "operator", "instrument", "calibration_date", "electrolyte_id",
    "working_electrode", "counter_electrode", "reference_electrode", "temperature_C",
    "agitation", "preparation",
}


def _path(value: str, manifest: Path) -> Path:
    """Resolve a nonempty manifest path relative to the manifest itself."""
    return (manifest.parent / value).resolve()


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate the tabular campaign manifest's structural fields."""
    path = Path(path)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required manifest columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Campaign manifest must contain at least one run")
    if frame["run_id"].str.strip().eq("").any() or frame["run_id"].duplicated().any():
        raise ValueError("run_id values must be nonempty and unique")
    invalid_phase = set(frame["phase"]) - VALID_PHASES
    invalid_status = set(frame["status"]) - VALID_STATUSES
    if invalid_phase:
        raise ValueError(f"Invalid phase values: {', '.join(sorted(invalid_phase))}")
    if invalid_status:
        raise ValueError(f"Invalid status values: {', '.join(sorted(invalid_status))}")
    return frame


def validate_manifest(path: str | Path) -> dict:
    """Return a per-run QA report; only complete runs require all linked files."""
    manifest = Path(path)
    frame = load_manifest(manifest)
    runs: list[dict] = []
    for row in frame.to_dict("records"):
        flags: list[str] = []
        complete = row["status"] == "complete"
        for column in ("raw_file", "processed_file", "metadata_file"):
            value = row[column].strip()
            if not value:
                if complete:
                    flags.append(f"missing_{column}")
            elif not _path(value, manifest).is_file():
                flags.append(f"file_not_found:{column}")
        metadata_path = row["metadata_file"].strip()
        if metadata_path and _path(metadata_path, manifest).is_file():
            try:
                metadata = json.loads(_path(metadata_path, manifest).read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                flags.append("metadata_not_valid_json")
            else:
                missing = REQUIRED_METADATA - set(metadata)
                if missing:
                    flags.append("metadata_missing:" + ",".join(sorted(missing)))
        characterization = row.get("characterization_file", "").strip()
        if characterization and not _path(characterization, manifest).is_file():
            flags.append("file_not_found:characterization_file")
        if complete and not characterization:
            flags.append("missing_characterization_file")
        runs.append({"run_id": row["run_id"], "status": row["status"], "flags": flags,
                     "ready_for_analysis": not flags and complete})
    return {
        "manifest": str(manifest), "n_runs": len(runs),
        "n_complete": sum(run["status"] == "complete" for run in runs),
        "n_ready_for_analysis": sum(run["ready_for_analysis"] for run in runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an experimental campaign manifest.")
    parser.add_argument("manifest", help="Path to campaign_manifest.csv")
    parser.add_argument("--output", help="Optional JSON QA-report destination")
    args = parser.parse_args()
    report = validate_manifest(args.manifest)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
