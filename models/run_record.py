"""Versioned experiment-record contract and end-to-end QA pipeline.

This module is the boundary between instrument exports and the modeling suite.
It deliberately keeps three contracts distinct:

* **voltammetry** uses ``potential_V_vs_ref`` and ``current_A``;
* **plating** uses a normalized ``current_actual_A`` and ``voltage_V`` trace;
  the Hull-cell spelling (``current_A``/``cell_voltage_V``) is accepted only
  through an explicit, lossless adapter; and
* **campaign** is the cross-run CSV index validated by :mod:`models.campaign`.

A plating run is a directory containing a manifest and linked sidecars.  The
loader never changes raw files.  It maps a copy of the trace into the plating
contract, computes measured quantities, and emits a JSON-serializable QA
report.  Missing measurements remain missing: they are not replaced with model
predictions or silently treated as zero.

The module intentionally uses small, dependency-light validators rather than
requiring ``jsonschema`` at runtime.  The JSON templates in ``templates/``
and ``experiments/data/`` remain the human-facing schema documents.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .characterization import load_characterization, summarize_characterization
from .electrochemistry import FARADAY, M_FE_G, Z_FE
from .experimental_data import REQUIRED_COLUMNS as VOLTAMMETRY_REQUIRED_COLUMNS
from .plating_data import (
    AnomalyFlag,
    PlatingDerived,
    compute_derived,
    detect_anomalies,
)
from .reference_cell_spec import load_spec as _load_refcell_spec
from .reference_cell_spec import verify_spec as _verify_refcell_spec
from .run_manifest import (
    load_bath_batch,
    load_experiment_manifest,
    validate_experiment_manifest,
)


CONTRACT_NAME = "aqueous-electrowinning.run-record"
SCHEMA_VERSION = "1.0"

PLATING_REQUIRED_COLUMNS = frozenset({"timestamp_s", "current_actual_A", "voltage_V"})
MASS_LOG_REQUIRED_COLUMNS = frozenset({"mass_before_g", "mass_after_g"})
ENERGY_LOG_REQUIRED_COLUMNS = frozenset({"component", "energy_Wh"})
CAMPAIGN_REQUIRED_COLUMNS = frozenset({
    "run_id", "phase", "technique", "status", "raw_file", "processed_file", "metadata_file",
})
CAMPAIGN_PHASES = frozenset({"I", "II", "III", "IV"})
CAMPAIGN_STATUSES = frozenset({"planned", "in_progress", "complete", "excluded"})

# These are the minimum fields in the sidecar documented by
# templates/metadata_template.json and enforced by campaign.py.
METADATA_REQUIRED_FIELDS = frozenset({
    "sample_id",
    "operator",
    "instrument",
    "calibration_date",
    "electrolyte_id",
    "working_electrode",
    "counter_electrode",
    "reference_electrode",
    "temperature_C",
    "agitation",
    "preparation",
})

RECORD_STATUSES = frozenset({"planned", "in_progress", "complete", "excluded"})
DEFAULT_FILES = {
    "timeseries_csv": "timeseries.csv",
    "bath_batch_json": "bath_batch.json",
    "metadata_json": "metadata.json",
    "reference_cell_json": "reference_cell.json",
    "mass_log_csv": "mass_log.csv",
    "characterization_csv": "characterization.csv",
    "video_index_csv": "video_index.csv",
    "energy_log_csv": "energy_log.csv",
}
# Older manifests used these names.  They are accepted as aliases so a
# migration does not require hand-editing every existing record.
FILE_ALIASES = {
    "bath_batch_json": ("bath_batch_json", "bath_batch_file"),
    "metadata_json": ("metadata_json", "metadata_file"),
    "reference_cell_json": ("reference_cell_json", "reference_cell_file"),
    "timeseries_csv": ("timeseries_csv", "timeseries_file"),
    "mass_log_csv": ("mass_log_csv", "mass_file"),
    "characterization_csv": (
        "characterization_csv",
        "characterization_file",
        "characterization",
    ),
    "video_index_csv": ("video_index_csv", "video_index_file"),
    "energy_log_csv": ("energy_log_csv", "energy_file"),
}
KNOWN_ENERGY_COMPONENTS = frozenset({
    "pumps", "heating", "cooling", "gas_handling", "drying", "other_auxiliary",
})


class DataContractError(ValueError):
    """Raised when a file cannot be loaded under its declared contract."""


@dataclass(frozen=True)
class ContractIssue:
    """One path-addressable data-contract or QA issue."""

    path: str
    message: str
    severity: str = "error"  # error | warning

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message, "severity": self.severity}


@dataclass
class ContractReport:
    """Validation result used by the loaders and the JSON QA report."""

    contract: str
    schema_version: str = SCHEMA_VERSION
    issues: list[ContractIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ContractIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ContractIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def add(self, path: str, message: str, severity: str = "error") -> None:
        if severity not in {"error", "warning"}:
            raise ValueError("severity must be 'error' or 'warning'")
        self.issues.append(ContractIssue(path, message, severity))

    def extend(self, other: "ContractReport", prefix: str = "") -> None:
        for issue in other.issues:
            path = f"{prefix}.{issue.path}" if prefix and issue.path else prefix or issue.path
            self.issues.append(ContractIssue(path, issue.message, issue.severity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "schema_version": self.schema_version,
            "valid": self.valid,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _raise_report(report: ContractReport, label: str) -> None:
    if report.valid:
        return
    text = "; ".join(f"{issue.path}: {issue.message}" for issue in report.errors)
    raise DataContractError(f"{label} does not satisfy {report.contract}: {text}")


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        frame = pd.read_csv(path, keep_default_na=True)
    except Exception as exc:  # pandas exposes several parser exception types
        raise DataContractError(f"Could not read {label} CSV {path}: {exc}") from exc
    if frame.empty:
        raise DataContractError(f"{label} CSV must contain at least one row: {path}")
    return frame


def _numeric_columns(frame: pd.DataFrame, columns: Iterable[str], report: ContractReport) -> None:
    """Check numeric columns without mutating the caller's frame."""
    for column in columns:
        if column not in frame:
            continue
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            report.add(column, f"must contain numeric values: {exc}")
            continue
        if values.isna().any():
            report.add(column, "contains missing numeric values")


def _finite_columns(frame: pd.DataFrame, columns: Iterable[str], report: ContractReport) -> None:
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            report.add(column, "must contain finite values")


def normalize_plating_timeseries(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize supported plating trace spellings to the run-record schema.

    The canonical columns are ``current_actual_A`` and ``voltage_V``.  The
    Hull-cell schema uses ``current_A`` and ``cell_voltage_V``; those columns
    are renamed only when the canonical equivalent is absent.  If both forms
    are present, the record is rejected rather than guessing which instrument
    channel is authoritative.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    normalized = frame.copy()
    mapping: dict[str, str] = {}
    pairs = (("current_actual_A", "current_A"), ("voltage_V", "cell_voltage_V"))
    for canonical, alias in pairs:
        if canonical in normalized.columns and alias in normalized.columns:
            raise DataContractError(
                f"ambiguous plating trace: both {canonical!r} and {alias!r} are present"
            )
        if canonical not in normalized.columns and alias in normalized.columns:
            normalized = normalized.rename(columns={alias: canonical})
            mapping[alias] = canonical
    return normalized, mapping


def validate_plating_timeseries(frame: pd.DataFrame) -> ContractReport:
    """Validate a normalized plating timeseries without deriving results."""
    report = ContractReport("plating-timeseries")
    if not isinstance(frame, pd.DataFrame):
        report.add("", "must be a pandas DataFrame")
        return report
    missing = PLATING_REQUIRED_COLUMNS - set(frame.columns)
    for column in sorted(missing):
        report.add(column, "required column is missing")
    if frame.empty:
        report.add("", "must contain at least one row")
        return report
    _numeric_columns(frame, PLATING_REQUIRED_COLUMNS, report)
    _finite_columns(frame, PLATING_REQUIRED_COLUMNS, report)
    if "timestamp_s" in frame:
        timestamps = pd.to_numeric(frame["timestamp_s"], errors="coerce").to_numpy(float)
        if np.isfinite(timestamps).all() and (np.diff(timestamps) < 0).any():
            report.add("timestamp_s", "must be monotonically non-decreasing")
        if len(timestamps) < 2:
            report.add("timestamp_s", "at least two points are required for a plating run")
        elif timestamps[-1] <= timestamps[0]:
            report.add("timestamp_s", "run duration must be positive")
    if "current_setpoint_A" in frame:
        _numeric_columns(frame, ("current_setpoint_A",), report)
    return report


def load_plating_timeseries(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load and normalize a plating trace, returning ``(frame, column_map)``."""
    path = Path(path)
    frame = _read_csv(path, "plating timeseries")
    try:
        normalized, mapping = normalize_plating_timeseries(frame)
    except DataContractError:
        raise
    report = validate_plating_timeseries(normalized)
    _raise_report(report, str(path))
    for column in PLATING_REQUIRED_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
    return normalized, mapping


def validate_voltammetry(frame: pd.DataFrame) -> ContractReport:
    """Validate the separate Phase-I voltammetry contract."""
    report = ContractReport("voltammetry-timeseries")
    if not isinstance(frame, pd.DataFrame):
        report.add("", "must be a pandas DataFrame")
        return report
    missing = set(VOLTAMMETRY_REQUIRED_COLUMNS) - set(frame.columns)
    for column in sorted(missing):
        report.add(column, "required column is missing")
    if frame.empty:
        report.add("", "must contain at least one row")
        return report
    numeric = set(VOLTAMMETRY_REQUIRED_COLUMNS) | {
        "cycle", "temperature_C", "pH", "fe2_concentration_M",
    }
    _numeric_columns(frame, numeric, report)
    _finite_columns(frame, VOLTAMMETRY_REQUIRED_COLUMNS, report)
    if "working_electrode_area_cm2" in frame:
        area = pd.to_numeric(frame["working_electrode_area_cm2"], errors="coerce")
        if np.isfinite(area.to_numpy(float)).all() and (area <= 0).any():
            report.add("working_electrode_area_cm2", "must be positive")
    if "timestamp_s" in frame:
        timestamps = pd.to_numeric(frame["timestamp_s"], errors="coerce").to_numpy(float)
        if np.isfinite(timestamps).all() and (np.diff(timestamps) < 0).any():
            report.add("timestamp_s", "must be monotonically non-decreasing")
    return report


def load_voltammetry(path: str | Path) -> pd.DataFrame:
    """Load the canonical voltammetry contract and derive current density."""
    path = Path(path)
    frame = _read_csv(path, "voltammetry")
    report = validate_voltammetry(frame)
    _raise_report(report, str(path))
    for column in set(VOLTAMMETRY_REQUIRED_COLUMNS) & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["current_density_A_m2"] = (
        frame["current_A"] / (frame["working_electrode_area_cm2"] * 1e-4)
    )
    frame["current_density_mA_cm2"] = (
        frame["current_A"] / frame["working_electrode_area_cm2"] * 1000.0
    )
    return frame


def validate_campaign_manifest(frame: pd.DataFrame) -> ContractReport:  # noqa: C901
    """Validate the cross-run campaign-index contract in memory."""
    report = ContractReport("campaign-manifest")
    if not isinstance(frame, pd.DataFrame):
        report.add("", "must be a pandas DataFrame")
        return report
    missing = CAMPAIGN_REQUIRED_COLUMNS - set(frame.columns)
    for column in sorted(missing):
        report.add(column, "required column is missing")
    if frame.empty:
        report.add("", "must contain at least one row")
        return report
    for column in CAMPAIGN_REQUIRED_COLUMNS:
        if column in frame and frame[column].astype(str).str.strip().eq("").any():
            report.add(column, "must not contain blank values")
    if "run_id" in frame:
        if frame["run_id"].duplicated().any():
            report.add("run_id", "values must be unique")
    if "phase" in frame:
        invalid = set(frame["phase"]) - CAMPAIGN_PHASES
        if invalid:
            report.add("phase", "invalid value(s): " + ", ".join(sorted(map(str, invalid))))
    if "status" in frame:
        invalid = set(frame["status"]) - CAMPAIGN_STATUSES
        if invalid:
            report.add("status", "invalid value(s): " + ", ".join(sorted(map(str, invalid))))
    if "schema_version" in frame:
        versions = set(frame["schema_version"].astype(str).str.strip())
        if versions - {SCHEMA_VERSION}:
            report.add(
                "schema_version",
                "unsupported version(s): " + ", ".join(sorted(versions - {SCHEMA_VERSION})),
            )
    return report


def load_campaign_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate the campaign index without checking linked files.

    Use :func:`models.campaign.validate_manifest` afterward when file links and
    metadata readiness must also be checked.
    """
    path = Path(path)
    frame = _read_csv(path, "campaign manifest")
    report = validate_campaign_manifest(frame)
    _raise_report(report, str(path))
    return frame


def validate_mass_log(frame: pd.DataFrame) -> ContractReport:  # noqa: C901
    """Validate the run-level dry-mass record."""
    report = ContractReport("plating-mass-log")
    if not isinstance(frame, pd.DataFrame):
        report.add("", "must be a pandas DataFrame")
        return report
    missing = MASS_LOG_REQUIRED_COLUMNS - set(frame.columns)
    for column in sorted(missing):
        report.add(column, "required column is missing")
    if frame.empty:
        report.add("", "must contain at least one row")
        return report
    numeric = MASS_LOG_REQUIRED_COLUMNS | {
        "blank_mass_change_g", "mass_uncertainty_g", "blank_mass_uncertainty_g",
        "electrode_area_cm2",
    }
    _numeric_columns(frame, numeric, report)
    _finite_columns(frame, numeric, report)
    for column in ("mass_before_g", "mass_after_g"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            if (values < 0).any():
                report.add(column, "must be non-negative")
    for column in ("mass_uncertainty_g", "blank_mass_uncertainty_g", "electrode_area_cm2"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            if (values < 0).any():
                report.add(column, "must be non-negative")
    if len(frame) != 1:
        report.add(
            "",
            "run-level mass log must contain exactly one row; use the Hull-cell module for panel coupons",
        )
    return report


def load_mass_log_record(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = _read_csv(path, "mass log")
    report = validate_mass_log(frame)
    _raise_report(report, str(path))
    for column in MASS_LOG_REQUIRED_COLUMNS | {
        "blank_mass_change_g", "mass_uncertainty_g", "blank_mass_uncertainty_g",
        "electrode_area_cm2",
    }:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def validate_energy_log(frame: pd.DataFrame) -> ContractReport:
    """Validate auxiliary energy measurements without double-counting stack DC."""
    report = ContractReport("auxiliary-energy-log")
    if not isinstance(frame, pd.DataFrame):
        report.add("", "must be a pandas DataFrame")
        return report
    missing = ENERGY_LOG_REQUIRED_COLUMNS - set(frame.columns)
    for column in sorted(missing):
        report.add(column, "required column is missing")
    if frame.empty:
        report.add("", "must contain at least one row")
        return report
    if "component" in frame:
        components = frame["component"].astype(str).str.strip()
        if components.eq("").any():
            report.add("component", "must not contain blank component names")
        unknown = set(components) - KNOWN_ENERGY_COMPONENTS
        if unknown:
            report.add(
                "component",
                "unknown component(s): " + ", ".join(sorted(unknown)),
            )
    _numeric_columns(frame, ("energy_Wh", "uncertainty_Wh"), report)
    _finite_columns(frame, ("energy_Wh", "uncertainty_Wh"), report)
    for column in ("energy_Wh", "uncertainty_Wh"):
        if column in frame and (pd.to_numeric(frame[column], errors="coerce") < 0).any():
            report.add(column, "must be non-negative")
    return report


def load_energy_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = _read_csv(path, "energy log")
    report = validate_energy_log(frame)
    _raise_report(report, str(path))
    frame["energy_Wh"] = pd.to_numeric(frame["energy_Wh"], errors="raise")
    if "uncertainty_Wh" in frame:
        frame["uncertainty_Wh"] = pd.to_numeric(frame["uncertainty_Wh"], errors="raise")
    return frame


def validate_metadata(data: Any) -> ContractReport:  # noqa: C901
    """Validate the run metadata sidecar documented in ``metadata_template``."""
    report = ContractReport("run-metadata")
    if not isinstance(data, dict):
        report.add("", "must be a JSON object")
        return report
    for key in sorted(METADATA_REQUIRED_FIELDS):
        if key not in data:
            report.add(key, "required metadata field is missing")
        elif key != "temperature_C" and not isinstance(data[key], str):
            report.add(key, "must be a string")
        elif isinstance(data[key], str) and not data[key].strip():
            report.add(key, "must not be blank")
    if "temperature_C" in data:
        try:
            temp = float(data["temperature_C"])
            if not np.isfinite(temp):
                raise ValueError
        except (TypeError, ValueError):
            report.add("temperature_C", "must be a finite number")
    if "calibration_date" in data:
        try:
            date.fromisoformat(str(data["calibration_date"]))
        except ValueError:
            report.add("calibration_date", "must be an ISO date (YYYY-MM-DD)")
    if "raw_export_sha256" in data and data["raw_export_sha256"]:
        digest = str(data["raw_export_sha256"]).strip()
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            report.add("raw_export_sha256", "must be a 64-character hexadecimal SHA-256 digest")
    return report


def _validate_batch_dict(data: Any) -> ContractReport:  # noqa: C901
    report = ContractReport("bath-batch")
    if not isinstance(data, dict):
        report.add("", "must be a JSON object")
        return report
    schema_version = data.get("schema_version")
    if schema_version is None:
        report.add(
            "schema_version",
            f"missing; assuming contract version {SCHEMA_VERSION}",
            severity="warning",
        )
    elif schema_version != SCHEMA_VERSION:
        report.add(
            "schema_version",
            f"unsupported version {schema_version!r}; expected {SCHEMA_VERSION!r}",
        )
    for key in (
        "batch_id", "date_mixed", "operator", "composition", "source_chemicals", "storage"
    ):
        if key not in data:
            report.add(key, "required bath-batch field is missing")
    for key in ("source_chemicals", "storage"):
        if key in data and not isinstance(data[key], dict):
            report.add(key, "must be an object")
    composition = data.get("composition")
    if not isinstance(composition, dict):
        report.add("composition", "must be an object")
    else:
        for key in ("fe2_g_L", "h3bo3_g_L", "pH", "volume_mL"):
            if key not in composition:
                report.add(f"composition.{key}", "required composition field is missing")
        for key in ("fe2_g_L", "h3bo3_g_L", "pH", "volume_mL"):
            if key in composition:
                try:
                    value = float(composition[key])
                    if not np.isfinite(value):
                        raise ValueError
                except (TypeError, ValueError):
                    report.add(f"composition.{key}", "must be a finite number")
        if "pH" in composition:
            try:
                pH = float(composition["pH"])
                if not 0.0 <= pH <= 14.0:
                    report.add("composition.pH", "must lie between 0 and 14")
            except (TypeError, ValueError):
                pass
        for key in ("fe2_g_L", "h3bo3_g_L", "volume_mL"):
            if key in composition:
                try:
                    if float(composition[key]) < 0:
                        report.add(f"composition.{key}", "must be non-negative")
                except (TypeError, ValueError):
                    pass
    return report


# Reference-cell metrology groups per docs/REFERENCE_CELL_SPEC.md §7/§9.
# ``mass``, ``thickness_map`` and ``composition`` are required to declare a
# reference deposit record; the remaining four close a *complete* one.
REFCELL_METROLOGY_REQUIRED = frozenset({"mass", "thickness_map", "composition"})
REFCELL_METROLOGY_COMPLETE = frozenset({
    "mass", "thickness_map", "composition",
    "morphology", "porosity", "adhesion", "hydrogen_content",
})
# Reference-cell runs declare experiment_type "divided_cell" (see the D1 spec).
REFCELL_EXPERIMENT_TYPES = frozenset({"divided_cell", "reference_cell"})


def _canonical_refcell_spec_path() -> Path:
    """Return the repo's canonical frozen reference-cell spec path.

    ``models/run_record.py`` resolves to ``models/``; its parent is the repo
    root, so the canonical spec is ``<repo>/processes/reference_cell_spec.v1.json``.
    """
    return Path(__file__).resolve().parent.parent / "processes" / "reference_cell_spec.v1.json"


def _check_sha256_digest(sha: Any, label: str, report: ContractReport) -> bool:
    """Report an invalid 64-hex SHA-256 digest; return True when it is valid."""
    if isinstance(sha, str) and len(sha) == 64 and all(
        c in "0123456789abcdefABCDEF" for c in sha
    ):
        return True
    report.add(label, "must be a 64-hex SHA-256 digest")
    return False


def _validate_sample_list(values: Any, phase: str, report: ContractReport) -> None:
    if not isinstance(values, list):
        report.add(f"samples.{phase}", "must be a list of sample entries")
        return
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            report.add(f"samples.{phase}[{index}]", "must be an object")
            continue
        for req_field in ("sample_id", "timestamp_s", "loop"):
            if req_field not in item:
                report.add(f"samples.{phase}[{index}].{req_field}", "required field is missing")


def validate_reference_cell_record(  # noqa: C901
    data: Any,
    *,
    spec_file: str | Path | None = None,
    record_status: str = "complete",
) -> ContractReport:
    """Validate a ``reference_cell.json`` raw-linked sidecar (D1 §9).

    This is the consumer that ``models.run_record`` applies to every reference
    run. It enforces the invariants documented in ``docs/REFERENCE_CELL_SPEC.md``
    §9.2:

    - ``spec_sha256`` is a 64-hex digest and, when the canonical spec file is
      reachable, matches that file's content hash (tamper-evident pin);
    - ``samples.before`` and ``samples.after`` are required in a complete
      record; ``samples.during`` is required when a run claims drift/crossover
      (represented by a non-empty ``samples.during`` key presence is optional);
    - ``deposit_metrology.mass/thickness_map/composition`` are required; the
      remaining groups are warnings on a complete record;
    - ``as_built_deviations`` items must carry a ``path``, an ``as_built`` value
      and an ``authorized_by``.

    ``spec_file`` is the canonical frozen spec (defaults to the repo's
    ``processes/reference_cell_spec.v1.json``). When it is unreachable the
    content pin is downgraded to a warning (format is still enforced).
    """
    report = ContractReport("reference-cell-record")
    if not isinstance(data, dict):
        report.add("", "must be a JSON object")
        return report

    schema_version = data.get("schema_version")
    if schema_version is None:
        report.add("schema_version", "required")
    elif schema_version != SCHEMA_VERSION:
        report.add(
            "schema_version",
            f"unsupported version {schema_version!r}; expected {SCHEMA_VERSION!r}",
        )
    if not data.get("run_id"):
        report.add("run_id", "required")

    pin = data.get("reference_cell_spec")
    if not isinstance(pin, dict):
        report.add("reference_cell_spec", "required object pinning spec version + sha256")
    else:
        if not isinstance(pin.get("spec_version"), str) or not pin.get("spec_version"):
            report.add("reference_cell_spec.spec_version", "required")
        if not isinstance(pin.get("configuration_id"), str) or not pin.get("configuration_id"):
            report.add("reference_cell_spec.configuration_id", "required")
        sha = pin.get("spec_sha256")
        if isinstance(sha, str) and _check_sha256_digest(
            sha, "reference_cell_spec.spec_sha256", report
        ):
            if spec_file is not None and Path(spec_file).is_file():
                try:
                    spec_data = _load_refcell_spec(spec_file)
                except (ValueError, FileNotFoundError) as exc:
                    report.add(
                        "reference_cell_spec.spec_sha256",
                        f"cannot load canonical spec for content check: {exc}",
                    )
                else:
                    declared = str(spec_data.get("sha256", "")).lower()
                    if declared != sha.lower():
                        report.add(
                            "reference_cell_spec.spec_sha256",
                            "does not match the canonical spec file content hash "
                            f"(declared {declared or '(unfrozen)'})",
                        )
                    ok, _ = _verify_refcell_spec(spec_file)
                    if not ok:
                        report.add(
                            "reference_cell_spec.spec_sha256",
                            "canonical spec file fails its own content hash; it is not frozen",
                        )
            else:
                report.add(
                    "reference_cell_spec.spec_sha256",
                    "canonical spec file not reachable; content check unavailable",
                    severity="warning",
                )
        elif sha is not None:
            _check_sha256_digest(sha, "reference_cell_spec.spec_sha256", report)

    deviations = data.get("as_built_deviations")
    if deviations is not None:
        if not isinstance(deviations, list):
            report.add("as_built_deviations", "must be a list")
        else:
            for index, item in enumerate(deviations):
                if not isinstance(item, dict):
                    report.add(f"as_built_deviations[{index}]", "must be an object")
                    continue
                for field in ("path", "as_built", "authorized_by"):
                    if field not in item:
                        report.add(f"as_built_deviations[{index}].{field}", "required field is missing")

    rectifier = data.get("rectifier")
    if rectifier is not None and not isinstance(rectifier, dict):
        report.add("rectifier", "must be an object")

    samples = data.get("samples")
    if not isinstance(samples, dict):
        report.add("samples", "required object with before/after sample lists")
    else:
        for phase in ("before", "after"):
            values = samples.get(phase)
            if record_status == "complete" and not values:
                report.add(f"samples.{phase}", "required at least one entry in a complete record")
            elif values is not None:
                _validate_sample_list(values, phase, report)
        if "during" in samples:
            _validate_sample_list(samples.get("during"), "during", report)

    metrology = data.get("deposit_metrology")
    if not isinstance(metrology, dict):
        report.add("deposit_metrology", "required object of measurement groups")
    else:
        for key in sorted(REFCELL_METROLOGY_REQUIRED):
            entry = metrology.get(key)
            if entry is None:
                if record_status == "complete":
                    report.add(f"deposit_metrology.{key}", "required in a complete reference record")
            elif not isinstance(entry, dict):
                report.add(f"deposit_metrology.{key}", "must be an object")
            elif not entry.get("file"):
                report.add(f"deposit_metrology.{key}.file", "required raw-linked file")
        for key in sorted(REFCELL_METROLOGY_COMPLETE - REFCELL_METROLOGY_REQUIRED):
            if metrology.get(key) is None and record_status == "complete":
                report.add(
                    f"deposit_metrology.{key}",
                    "required for a complete reference deposit record",
                    severity="warning",
                )
    return report


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataContractError(f"{label} is not valid JSON: {exc}") from exc


def _resolve_file_map(root: Path, manifest: dict[str, Any], report: ContractReport) -> dict[str, Path]:
    raw_files = manifest.get("files", {})
    if raw_files is None:
        raw_files = {}
    if not isinstance(raw_files, dict):
        report.add("manifest.files", "must be an object when present")
        raw_files = {}
    paths: dict[str, Path] = {}
    for key, default in DEFAULT_FILES.items():
        value: Any = None
        for alias in FILE_ALIASES[key]:
            if alias in raw_files:
                value = raw_files[alias]
                break
        if value is None:
            value = default
        if not isinstance(value, str) or not value.strip():
            report.add(f"manifest.files.{key}", "must be a nonempty relative path")
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            report.add(f"manifest.files.{key}", "must be relative to the run directory")
            continue
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            report.add(
                f"manifest.files.{key}",
                "must resolve inside the run directory",
            )
            continue
        paths[key] = resolved
    return paths


def _record_status(manifest: dict[str, Any], report: ContractReport) -> str:
    status = manifest.get("record_status", "complete")
    if status not in RECORD_STATUSES:
        report.add(
            "manifest.record_status",
            f"invalid status {status!r}; valid values: {', '.join(sorted(RECORD_STATUSES))}",
        )
        return "complete"
    return str(status)


def _required_file_issue(report: ContractReport, path: Path, label: str, status: str) -> None:
    if path.is_file():
        return
    severity = "warning" if status in {"planned", "in_progress", "excluded"} else "error"
    report.add(label, f"file not found: {path}", severity=severity)


def _load_manifest_for_report(root: Path, report: ContractReport) -> dict[str, Any] | None:
    path = root / "manifest.json"
    if not path.is_file():
        report.add("manifest.json", f"file not found: {path}")
        return None
    try:
        data = load_experiment_manifest(path)
    except (DataContractError, ValueError, FileNotFoundError) as exc:
        # load_experiment_manifest raises a compact ValueError.  Re-run the
        # report-producing validator when the JSON itself is usable.
        try:
            raw = _load_json(path, "experiment manifest")
        except (DataContractError, FileNotFoundError) as json_exc:
            report.add("manifest.json", str(json_exc))
            return None
        if isinstance(raw, dict):
            manifest_report = validate_experiment_manifest(raw)
            report.extend(manifest_report, prefix="manifest")
            if manifest_report.valid:
                data = raw
            else:
                return raw
        else:
            report.add("manifest.json", str(exc))
            return None
    manifest_report = validate_experiment_manifest(data)
    report.extend(manifest_report, prefix="manifest")
    schema_version = data.get("schema_version")
    if schema_version is None:
        report.add(
            "manifest.schema_version",
            f"missing; assuming contract version {SCHEMA_VERSION}",
            severity="warning",
        )
    elif schema_version != SCHEMA_VERSION:
        report.add(
            "manifest.schema_version",
            f"unsupported version {schema_version!r}; expected {SCHEMA_VERSION!r}",
        )
    return data


def _extract_cathodic_sign(manifest: dict[str, Any], report: ContractReport) -> str:
    conventions = manifest.get("measurement_conventions", {})
    value = manifest.get("current_sign_convention")
    if value is None and isinstance(conventions, dict):
        value = conventions.get("cathodic_sign") or conventions.get("current_sign")
    if value is None:
        return "negative"
    text = str(value).strip().lower()
    if text in {"negative", "cathodic_negative", "cathodic-negative"}:
        return "negative"
    if text in {"positive", "cathodic_positive", "cathodic-positive"}:
        return "positive"
    report.add(
        "manifest.current_sign_convention",
        "must be one of negative, positive, cathodic_negative, cathodic_positive",
    )
    return "negative"


def _cathode_area(manifest: dict[str, Any]) -> float | None:
    setup = manifest.get("setup", {})
    if not isinstance(setup, dict):
        return None
    cathode = setup.get("cathode", {})
    if not isinstance(cathode, dict):
        return None
    value = cathode.get("area_cm2")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def _fe_mass_fraction(characterization: pd.DataFrame | None) -> tuple[float | None, str | None]:
    """Return an independently measured Fe fraction when one is available."""
    if characterization is None or characterization.empty:
        return None, None
    analyte = characterization["analyte"].astype(str).str.strip().str.casefold()
    unit = characterization["unit"].astype(str).str.strip().str.casefold()
    rows = characterization.loc[
        analyte.eq("fe") & unit.isin({"wt%", "mass%"})
    ]
    if rows.empty:
        return None, None
    techniques = rows["technique"].astype(str).str.strip().unique().tolist()
    if len(techniques) > 1:
        return None, (
            "Fe composition is reported by multiple techniques; select one measurement basis "
            "before closing the iron ledger"
        )
    values = pd.to_numeric(rows["value"], errors="coerce")
    if values.isna().any() or (values < 0).any() or (values > 100).any():
        return None, "Fe composition record is outside 0–100 wt%; iron ledger withheld"
    return (
        float(values.mean() / 100.0),
        f"mean of {len(values)} {techniques[0]} Fe composition observation(s)",
    )


def compute_ledgers(  # noqa: C901
    derived: PlatingDerived,
    *,
    bath_batch: dict[str, Any] | None = None,
    characterization: pd.DataFrame | None = None,
    energy_log: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute charge, iron, and energy ledgers from measured inputs.

    The result distinguishes *measured*, *derived*, and *missing* entries.
    A mass-only apparent FE is never relabeled as an Fe-specific balance.  An
    iron ledger becomes quantitative only when deposit Fe composition is
    independently recorded; an energy ledger remains partial until auxiliary
    energy loads are measured.
    """
    fe_fraction, fe_note = _fe_mass_fraction(characterization)
    deposit_mass = derived.net_deposit_mass_g
    deposit_fe_mass = None
    deposit_fe_mol = None
    deposit_fe_charge = None
    if fe_fraction is not None and deposit_mass is not None:
        deposit_fe_mass = float(deposit_mass * fe_fraction)
        deposit_fe_mol = deposit_fe_mass / M_FE_G
        deposit_fe_charge = deposit_fe_mol * Z_FE * FARADAY

    composition = (bath_batch or {}).get("composition", {})
    initial_fe_mol = None
    if isinstance(composition, dict):
        try:
            fe2_g_L = float(composition["fe2_g_L"])
            volume_mL = float(composition["volume_mL"])
            if fe2_g_L >= 0 and volume_mL > 0:
                initial_fe_mol = fe2_g_L * volume_mL / 1000.0 / M_FE_G
        except (KeyError, TypeError, ValueError):
            pass

    analysis = (bath_batch or {}).get("analysis", {})
    post_fe_mol = None
    if isinstance(analysis, dict) and initial_fe_mol is not None:
        try:
            post_fe2_g_L = float(analysis["fe2_measured_g_L"])
            volume_mL = float(composition["volume_mL"])
            if post_fe2_g_L >= 0 and volume_mL > 0:
                post_fe_mol = post_fe2_g_L * volume_mL / 1000.0 / M_FE_G
        except (KeyError, TypeError, ValueError):
            pass

    iron_missing: list[str] = []
    if initial_fe_mol is None:
        iron_missing.append("initial Fe inventory (bath composition/volume)")
    if deposit_fe_mol is None:
        iron_missing.append("deposit Fe mass (independent composition)")
    if post_fe_mol is None:
        iron_missing.append("post-run Fe inventory (bath analysis)")
    # A residual from initial minus deposit minus post-run bath is useful, but
    # it is not a closed balance until precipitate/solids and other measured
    # Fe-bearing streams are recorded explicitly.
    solids_fe_mol = None
    other_fe_mol = None
    if isinstance(analysis, dict):
        try:
            solids_fe_mol = float(analysis["solids_fe_mol"])
            other_fe_mol = float(analysis["other_fe_mol"])
            if solids_fe_mol < 0 or other_fe_mol < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            solids_fe_mol = None
            other_fe_mol = None
    if solids_fe_mol is None:
        iron_missing.append("precipitate/solids Fe inventory")
    if other_fe_mol is None:
        iron_missing.append("other measured Fe-bearing streams")
    iron_closure_mol = None
    if (
        initial_fe_mol is not None
        and deposit_fe_mol is not None
        and post_fe_mol is not None
        and solids_fe_mol is not None
        and other_fe_mol is not None
    ):
        iron_closure_mol = initial_fe_mol - deposit_fe_mol - post_fe_mol - solids_fe_mol - other_fe_mol

    auxiliary_by_component: dict[str, float] = {}
    auxiliary_uncertainty_Wh = None
    if energy_log is not None and not energy_log.empty:
        for component, group in energy_log.groupby(energy_log["component"].astype(str).str.strip()):
            auxiliary_by_component[component] = float(pd.to_numeric(group["energy_Wh"]).sum())
        if "uncertainty_Wh" in energy_log:
            uncertainty = pd.to_numeric(energy_log["uncertainty_Wh"], errors="coerce")
            if not uncertainty.isna().any():
                auxiliary_uncertainty_Wh = float(np.sqrt(np.square(uncertainty).sum()))
    missing_energy = sorted(KNOWN_ENERGY_COMPONENTS - set(auxiliary_by_component))
    stack_energy = float(derived.energy_Wh)
    total_energy = stack_energy + sum(auxiliary_by_component.values())

    return {
        "charge": {
            "status": "partial" if deposit_fe_charge is None else "partial_with_fe_deposit",
            "applied_cathodic_charge_C": float(derived.charge_C),
            "apparent_fe_from_total_mass": derived.faradaic_efficiency,
            "apparent_fe_percent_from_total_mass": derived.faradaic_efficiency_percent,
            "measured_fe_deposition_charge_C": deposit_fe_charge,
            "unresolved_charge_C": (
                float(derived.charge_C - deposit_fe_charge)
                if deposit_fe_charge is not None else None
            ),
            "missing": ([] if deposit_fe_charge is not None else
                        ["independent deposit Fe composition and/or hydrogen/other-product measurement"]),
            "note": "Apparent FE uses total dry mass; it is not an Fe-specific charge closure without composition.",
        },
        "iron": {
            "status": "closed" if iron_closure_mol is not None else "partial",
            "initial_fe_inventory_mol": initial_fe_mol,
            "deposit_fe_mass_g": deposit_fe_mass,
            "deposit_fe_mol": deposit_fe_mol,
            "post_run_fe_inventory_mol": post_fe_mol,
            "solids_fe_mol": solids_fe_mol,
            "other_fe_mol": other_fe_mol,
            "unaccounted_fe_mol": iron_closure_mol,
            "missing": iron_missing,
            "composition_note": fe_note,
            "note": "A partial ledger is reported rather than inferring precipitate or crossover losses.",
        },
        "energy": {
            "status": "closed" if not missing_energy else "partial",
            "stack_electrical_Wh": stack_energy,
            "auxiliary_energy_Wh": auxiliary_by_component,
            "auxiliary_energy_uncertainty_Wh": auxiliary_uncertainty_Wh,
            "total_measured_energy_Wh": total_energy,
            "missing_components": missing_energy,
            "note": "Stack energy is integrated from measured V×I; auxiliary loads are included only when logged.",
        },
    }


def _metric_observations(
    derived: PlatingDerived,
    *,
    ledgers: dict[str, Any],
) -> dict[str, tuple[float | None, str, str]]:
    """Return report metrics as (value, unit, provenance) tuples."""
    return {
        "charge_C": (derived.charge_C, "C", "measured current integrated over time"),
        "duration_s": (derived.duration_s, "s", "measured timestamps"),
        "mean_voltage_V": (derived.mean_voltage_V, "V", "measured cell voltage"),
        "energy_Wh": (derived.energy_Wh, "Wh", "measured stack V×I integration"),
        "current_density_mA_cm2": (
            derived.current_density_mA_cm2,
            "mA/cm2",
            "measured current and manifest cathode area",
        ),
        "apparent_faradaic_efficiency": (
            derived.faradaic_efficiency,
            "fraction",
            "dry mass gain and measured charge; apparent until composition verified",
        ),
        "apparent_faradaic_efficiency_percent": (
            derived.faradaic_efficiency_percent,
            "%",
            "dry mass gain and measured charge; apparent until composition verified",
        ),
        "iron_deposition_charge_C": (
            ledgers["charge"]["measured_fe_deposition_charge_C"],
            "C",
            "independent deposit Fe composition and dry mass",
        ),
    }


def _build_gate_evidence(  # noqa: C901
    manifest: dict[str, Any],
    metrics: dict[str, tuple[float | None, str, str]],
    report: ContractReport,
) -> dict[str, Any]:
    """Materialize optional, explicitly declared measured gate observations."""
    declarations = manifest.get("gate_evidence", [])
    if declarations is None:
        declarations = []
    result: dict[str, Any] = {
        "status": "not_declared",
        "source": "experimental",
        "records": [],
        "note": "QA readiness is not a process-gate pass; declare candidate/gate mappings for evaluation.",
    }
    if not isinstance(declarations, list):
        report.add("manifest.gate_evidence", "must be a list when present")
        return result
    records: list[dict[str, Any]] = []
    for index, declaration in enumerate(declarations):
        path = f"manifest.gate_evidence[{index}]"
        if not isinstance(declaration, dict):
            report.add(path, "must be an object")
            continue
        for key in ("candidate_id", "gate_id", "metric", "value_from", "unit"):
            if not declaration.get(key):
                report.add(f"{path}.{key}", "is required")
        metric = declaration.get("metric")
        value_from = declaration.get("value_from")
        if value_from not in metrics:
            if metrics:
                report.add(
                    f"{path}.value_from",
                    f"unknown measured metric {value_from!r}; valid: {', '.join(sorted(metrics))}",
                )
            else:
                report.add(
                    f"{path}.value_from",
                    f"metric {value_from!r} is unavailable because this run has no derived measurements",
                    severity="warning",
                )
            continue
        value, derived_unit, provenance = metrics[value_from]
        if value is None:
            report.add(
                f"{path}.value_from",
                f"metric {value_from!r} is unavailable from this run",
            )
            continue
        declared_unit = str(declaration.get("unit", ""))
        if declared_unit != derived_unit:
            report.add(
                f"{path}.unit",
                f"does not match measured metric unit {derived_unit!r}",
            )
        records.append({
            "run_id": manifest.get("run_id"),
            "candidate_id": str(declaration["candidate_id"]),
            "gate_id": str(declaration["gate_id"]),
            "metric": str(metric),
            "value": float(value),
            "unit": declared_unit,
            "source": "experimental",
            "value_from": value_from,
            "provenance": provenance,
            "notes": str(declaration.get("notes", "")),
        })
    result["records"] = records
    result["status"] = "ready_for_gate_evaluation" if records and not report.errors else "pending"
    return result


@dataclass
class RunRecord:
    """Loaded, normalized plating run plus its QA artifacts."""

    directory: Path
    manifest: dict[str, Any]
    bath_batch: dict[str, Any]
    metadata: dict[str, Any]
    timeseries: pd.DataFrame
    mass_log: pd.DataFrame | None
    characterization: pd.DataFrame | None
    video_index: pd.DataFrame | None
    energy_log: pd.DataFrame | None
    derived: PlatingDerived
    anomalies: list[AnomalyFlag]
    qa_report: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return the machine-readable QA report."""
        return self.qa_report


def build_qa_report(  # noqa: C901
    run_dir: str | Path,
    *,
    reference_cell_spec_file: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect a run directory and return a non-throwing JSON-ready report.

    This is the preferred entry point for planned/incomplete runs.  It reports
    every missing or malformed component in one pass.  Use
    :func:`load_run_record` when analysis code should fail fast instead.

    ``reference_cell_spec_file`` overrides the canonical frozen spec used to
    content-check a ``reference_cell.json`` pin (defaults to the repo's
    ``processes/reference_cell_spec.v1.json``).
    """
    root = Path(run_dir).resolve()
    if reference_cell_spec_file is None:
        reference_cell_spec_file = _canonical_refcell_spec_path()
    report = ContractReport(CONTRACT_NAME)
    manifest = _load_manifest_for_report(root, report)
    if manifest is None:
        return {
            "contract": CONTRACT_NAME,
            "schema_version": SCHEMA_VERSION,
            "run_dir": str(root),
            "run_id": None,
            "record_status": None,
            "valid": False,
            "ready_for_analysis": False,
            "files": {},
            "metrics": {},
            "ledgers": {},
            "gate_evidence": {"status": "pending", "records": []},
            "issues": [issue.to_dict() for issue in report.issues],
        }

    status = _record_status(manifest, report)
    paths = _resolve_file_map(root, manifest, report)
    files: dict[str, Any] = {}
    required_keys = {"timeseries_csv", "bath_batch_json", "metadata_json"}
    for key, path in paths.items():
        exists = path.is_file()
        files[key] = {
            "path": str(path),
            "exists": exists,
            "required": key in required_keys,
        }
        if key in required_keys:
            _required_file_issue(report, path, key, status)

    bath_batch: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    timeseries: pd.DataFrame | None = None
    mass_log: pd.DataFrame | None = None
    characterization: pd.DataFrame | None = None
    video_index: pd.DataFrame | None = None
    energy_log: pd.DataFrame | None = None
    column_map: dict[str, str] = {}
    component_reports: dict[str, Any] = {}

    # Bath batch
    if paths.get("bath_batch_json", Path()).is_file():
        try:
            bath_batch = load_bath_batch(paths["bath_batch_json"])
            batch_report = _validate_batch_dict(bath_batch)
            component_reports["bath_batch"] = batch_report.to_dict()
            report.extend(batch_report, prefix="bath_batch")
            if bath_batch.get("batch_id") != manifest.get("bath_batch"):
                report.add(
                    "bath_batch.batch_id",
                    f"does not match manifest.bath_batch {manifest.get('bath_batch')!r}",
                )
        except (ValueError, DataContractError, FileNotFoundError) as exc:
            report.add("bath_batch", str(exc))

    # Metadata sidecar
    if paths.get("metadata_json", Path()).is_file():
        try:
            metadata = _load_json(paths["metadata_json"], "run metadata")
            metadata_report = validate_metadata(metadata)
            component_reports["metadata"] = metadata_report.to_dict()
            report.extend(metadata_report, prefix="metadata")
            if isinstance(metadata, dict) and metadata.get("operator") != manifest.get("operator"):
                report.add(
                    "metadata.operator",
                    "does not match manifest.operator",
                    severity="warning",
                )
        except (ValueError, DataContractError, FileNotFoundError) as exc:
            report.add("metadata", str(exc))

    # Reference-cell raw-linked record (D1 §9). Optional for non-reference
    # runs; required for a run that declares a reference-cell experiment type.
    reference_cell_data: dict[str, Any] | None = None
    refcell_path = paths.get("reference_cell_json", Path())
    experiment_type = str(manifest.get("experiment_type", "")).strip()
    is_refcell_experiment = experiment_type.casefold() in REFCELL_EXPERIMENT_TYPES
    if is_refcell_experiment and not refcell_path.is_file():
        report.add(
            "reference_cell",
            "divided-cell/reference-cell run requires a reference_cell.json sidecar",
        )
    if refcell_path.is_file():
        try:
            reference_cell_data = _load_json(refcell_path, "reference-cell record")
            refcell_report = validate_reference_cell_record(
                reference_cell_data,
                spec_file=reference_cell_spec_file,
                record_status=status,
            )
            component_reports["reference_cell"] = refcell_report.to_dict()
            report.extend(refcell_report, prefix="reference_cell")
        except (ValueError, DataContractError, FileNotFoundError) as exc:
            report.add("reference_cell", str(exc))

    # Normalized plating trace
    if paths.get("timeseries_csv", Path()).is_file():
        try:
            timeseries, column_map = load_plating_timeseries(paths["timeseries_csv"])
            ts_report = validate_plating_timeseries(timeseries)
            component_reports["timeseries"] = ts_report.to_dict()
            report.extend(ts_report, prefix="timeseries")
        except (ValueError, DataContractError, FileNotFoundError) as exc:
            report.add("timeseries", str(exc))

    if paths.get("mass_log_csv", Path()).is_file():
        try:
            mass_log = load_mass_log_record(paths["mass_log_csv"])
            mass_report = validate_mass_log(mass_log)
            component_reports["mass_log"] = mass_report.to_dict()
            report.extend(mass_report, prefix="mass_log")
        except (ValueError, DataContractError, FileNotFoundError) as exc:
            report.add("mass_log", str(exc))

    if paths.get("characterization_csv", Path()).is_file():
        try:
            characterization = load_characterization(paths["characterization_csv"])
            component_reports["characterization"] = {
                "valid": True,
                "summary": summarize_characterization(characterization),
            }
        except (ValueError, FileNotFoundError) as exc:
            report.add("characterization", str(exc))

    if paths.get("video_index_csv", Path()).is_file():
        try:
            video_index = pd.read_csv(paths["video_index_csv"], keep_default_na=True)
            missing = {"timestamp_s", "camera", "filename"} - set(video_index.columns)
            for column in sorted(missing):
                report.add(f"video_index.{column}", "required column is missing")
            if "timestamp_s" in video_index:
                video_index["timestamp_s"] = pd.to_numeric(video_index["timestamp_s"], errors="raise")
        except (ValueError, DataContractError, FileNotFoundError) as exc:
            report.add("video_index", str(exc))

    if paths.get("energy_log_csv", Path()).is_file():
        try:
            energy_log = load_energy_log(paths["energy_log_csv"])
            energy_report = validate_energy_log(energy_log)
            component_reports["energy_log"] = energy_report.to_dict()
            report.extend(energy_report, prefix="energy_log")
        except (ValueError, DataContractError, FileNotFoundError) as exc:
            report.add("energy_log", str(exc))

    # A complete record without optional sidecars is valid but not gate-ready.
    optional_missing = [
        key
        for key in ("mass_log_csv", "characterization_csv", "video_index_csv", "energy_log_csv")
        if not files.get(key, {}).get("exists", False)
    ]
    if status == "complete" and optional_missing:
        report.add(
            "optional_sidecars",
            "missing: " + ", ".join(optional_missing),
            severity="warning",
        )

    derived: PlatingDerived | None = None
    anomalies: list[AnomalyFlag] = []
    ledgers: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if timeseries is not None and not validate_plating_timeseries(timeseries).errors:
        sign = _extract_cathodic_sign(manifest, report)
        area = _cathode_area(manifest)
        if area is None:
            report.add(
                "manifest.setup.cathode.area_cm2",
                "missing or invalid; current density will be unavailable",
                severity="warning",
            )
        try:
            derived = compute_derived(
                timeseries,
                mass_log,
                cathode_area_cm2=area,
                cathodic_sign=sign,
            )
            anomalies = detect_anomalies(timeseries)
            ledgers = compute_ledgers(
                derived,
                bath_batch=bath_batch,
                characterization=characterization,
                energy_log=energy_log,
            )
            metrics = {
                key: {
                    "value": value,
                    "unit": unit,
                    "provenance": provenance,
                }
                for key, (value, unit, provenance) in _metric_observations(
                    derived, ledgers=ledgers
                ).items()
                if value is not None
            }
        except (ValueError, TypeError, KeyError) as exc:
            report.add("derived", f"could not compute derived quantities: {exc}")

    metric_tuples = (
        _metric_observations(derived, ledgers=ledgers)
        if derived is not None and ledgers
        else {}
    )
    gate_evidence = _build_gate_evidence(manifest, metric_tuples, report)

    required_present = all(files.get(key, {}).get("exists", False) for key in required_keys)
    ready = (
        status == "complete"
        and required_present
        and report.valid
        and timeseries is not None
        and derived is not None
    )
    # A run can be analysis-ready with an apparent FE, but process-gate
    # evidence still requires an explicit manifest mapping and QA validity.
    if gate_evidence["status"] == "ready_for_gate_evaluation" and not ready:
        gate_evidence["status"] = "pending"

    return {
        "contract": CONTRACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(root),
        "run_id": manifest.get("run_id"),
        "record_status": status,
        "valid": report.valid,
        "ready_for_analysis": ready,
        "files": files,
        "column_mapping": column_map,
        "components": component_reports,
        "metrics": metrics,
        "ledgers": ledgers,
        "anomalies": [
            {
                "kind": anomaly.kind,
                "message": anomaly.message,
                "severity": anomaly.severity,
                "timestamp_s": anomaly.timestamp_s,
            }
            for anomaly in anomalies
        ],
        "gate_evidence": gate_evidence,
        "issues": [issue.to_dict() for issue in report.issues],
    }


def load_run_record(run_dir: str | Path, *, strict: bool = True) -> RunRecord:
    """Load a complete run directory for analysis.

    Parameters
    ----------
    strict:
        When true (the default), any validation error or an incomplete record
        raises :class:`DataContractError`.  Warnings are retained in the QA
        report and do not prevent loading.
    """
    root = Path(run_dir).resolve()
    qa = build_qa_report(root)
    if strict and (not qa["valid"] or not qa["ready_for_analysis"]):
        issues = "; ".join(
            f"{item['path']}: {item['message']}"
            for item in qa["issues"]
            if item["severity"] == "error"
        )
        if not issues:
            issues = "required files or derived measurements are incomplete"
        raise DataContractError(f"Run record is not ready for analysis: {issues}")

    manifest = load_experiment_manifest(root / "manifest.json")
    paths_report = ContractReport(CONTRACT_NAME)
    paths = _resolve_file_map(root, manifest, paths_report)
    timeseries, _ = load_plating_timeseries(paths["timeseries_csv"])
    mass_log = (
        load_mass_log_record(paths["mass_log_csv"])
        if paths["mass_log_csv"].is_file() else None
    )
    characterization = (
        load_characterization(paths["characterization_csv"])
        if paths["characterization_csv"].is_file() else None
    )
    video_index = (
        pd.read_csv(paths["video_index_csv"], keep_default_na=True)
        if paths["video_index_csv"].is_file() else None
    )
    energy_log = (
        load_energy_log(paths["energy_log_csv"])
        if paths["energy_log_csv"].is_file() else None
    )
    bath_batch = load_bath_batch(paths["bath_batch_json"])
    metadata = _load_json(paths["metadata_json"], "run metadata")
    sign = _extract_cathodic_sign(manifest, paths_report)
    derived = compute_derived(
        timeseries,
        mass_log,
        cathode_area_cm2=_cathode_area(manifest),
        cathodic_sign=sign,
    )
    anomalies = detect_anomalies(timeseries)
    return RunRecord(
        directory=root,
        manifest=manifest,
        bath_batch=bath_batch,
        metadata=metadata,
        timeseries=timeseries,
        mass_log=mass_log,
        characterization=characterization,
        video_index=video_index,
        energy_log=energy_log,
        derived=derived,
        anomalies=anomalies,
        qa_report=qa,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize one plating run directory.")
    parser.add_argument("run_dir", help="Directory containing manifest.json and linked run files")
    parser.add_argument("--output", help="Write the JSON QA report to this path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when the run is not complete/analysis-ready",
    )
    args = parser.parse_args()
    report = build_qa_report(args.run_dir)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.strict and not report["ready_for_analysis"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
