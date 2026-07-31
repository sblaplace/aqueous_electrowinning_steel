"""Run manifest — experiment type registry and manifest validation.

Extends the existing campaign manifest system (``campaign.py``) with
plating-specific experiment types and video-recording validation.

Plating experiment types
------------------------
- ``hull_cell``         – angled-panel screening (Phase II)
- ``beaker_galvanostatic`` – simple beaker cell, constant current
- ``divided_cell``      – membrane-separated anolyte/catholyte

Each manifest links to a ``bath_batch_id`` that references a
``bath_batch.json`` record for full traceability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_EXPERIMENT_TYPES = frozenset({
    "hull_cell",
    "beaker_galvanostatic",
    "divided_cell",
})

VALID_VIDEO_STATUSES = frozenset({
    "complete",
    "partial",
    "none",
    "not_applicable",
})

# Configuration lifecycle states (proving-ground architecture)
VALID_CONFIGURATION_STATES = frozenset({
    "experimental",      # boundary-crossing permitted only in proving ground
    "qualified",         # measured envelope, known failure modes, recovery procedure
    "field_approved",    # conservative subset signed for named hardware/feed/site
})

MANIFEST_REQUIRED_KEYS = frozenset({
    "run_id",
    "date",
    "operator",
    "experiment_type",
    "bath_batch",
    "equipment",
    "setup",
    "video",
})


@dataclass(frozen=True)
class ManifestValidationIssue:
    """One structural or semantic issue found in a manifest."""
    path: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass(frozen=True)
class ManifestValidationReport:
    """Result of validating one experiment manifest."""
    valid: bool
    issues: list[ManifestValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ManifestValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ManifestValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [{"path": i.path, "message": i.message, "severity": i.severity}
                       for i in self.issues],
        }


def validate_experiment_manifest(data: dict[str, Any]) -> ManifestValidationReport:
    """Validate a plating experiment manifest dict against the schema rules.

    Checks structural presence, experiment_type membership, video status,
    and bath_batch linkage.  Returns a report even when the manifest is
    valid (issues list will be empty).
    """
    issues: list[ManifestValidationIssue] = []

    # ── Top-level required keys ────────────────────────────────────────
    for key in sorted(MANIFEST_REQUIRED_KEYS):
        if key not in data:
            issues.append(ManifestValidationIssue(
                path=key, message=f"Missing required key '{key}'"))

    experiment_type = data.get("experiment_type")
    if experiment_type is not None and experiment_type not in VALID_EXPERIMENT_TYPES:
        issues.append(ManifestValidationIssue(
            path="experiment_type",
            message=(f"Unknown experiment_type '{experiment_type}'; "
                     f"valid types: {', '.join(sorted(VALID_EXPERIMENT_TYPES))}")))

    # ── bath_batch: must be nonempty string ────────────────────────────
    bath_batch = data.get("bath_batch")
    if bath_batch is not None:
        if not isinstance(bath_batch, str) or not bath_batch.strip():
            issues.append(ManifestValidationIssue(
                path="bath_batch", message="bath_batch must be a nonempty string"))

    # ── video block ────────────────────────────────────────────────────
    video = data.get("video")
    if isinstance(video, dict):
        status = video.get("recording_status")
        if status not in VALID_VIDEO_STATUSES:
            issues.append(ManifestValidationIssue(
                path="video.recording_status",
                message=(f"Invalid recording_status '{status}'; "
                         f"valid: {', '.join(sorted(VALID_VIDEO_STATUSES))}")))
        if status == "none" and not video.get("notes", "").strip():
            issues.append(ManifestValidationIssue(
                path="video.notes",
                message="recording_status='none' requires a justification in video.notes"))
    elif "video" in data:
        issues.append(ManifestValidationIssue(
            path="video", message="video must be an object"))

    # ── Configuration block (optional; lifecycle state for proving-ground runs) ──
    configuration = data.get("configuration")
    if configuration is not None:
        if not isinstance(configuration, dict):
            issues.append(ManifestValidationIssue(
                path="configuration", message="configuration must be an object"))
        else:
            state = configuration.get("state")
            if state is not None and state not in VALID_CONFIGURATION_STATES:
                issues.append(ManifestValidationIssue(
                    path="configuration.state",
                    message=(f"Invalid configuration state '{state}'; "
                             f"valid: {', '.join(sorted(VALID_CONFIGURATION_STATES))}")))
            # Boundary-crossing runs require proving-ground containment
            boundary_crossing = configuration.get("boundary_crossing", False)
            if boundary_crossing:
                location = configuration.get("location", "")
                if location != "proving_ground":
                    issues.append(ManifestValidationIssue(
                        path="configuration.location",
                        message="boundary_crossing=true requires configuration.location='proving_ground'"))
                if not configuration.get("abort_conditions"):
                    issues.append(ManifestValidationIssue(
                        path="configuration.abort_conditions",
                        message="boundary_crossing=true requires explicit abort_conditions"))
                if not configuration.get("containment_plan"):
                    issues.append(ManifestValidationIssue(
                        path="configuration.containment_plan",
                        message="boundary_crossing=true requires a containment_plan"))

    # ── equipment block ────────────────────────────────────────────────
    equipment = data.get("equipment")
    if isinstance(equipment, dict):
        if "power_supply" not in equipment:
            issues.append(ManifestValidationIssue(
                path="equipment.power_supply",
                message="Missing required equipment.power_supply"))
        if "cell" not in equipment:
            issues.append(ManifestValidationIssue(
                path="equipment.cell",
                message="Missing required equipment.cell"))
    elif "equipment" in data:
        issues.append(ManifestValidationIssue(
            path="equipment", message="equipment must be an object"))

    # ── setup block ────────────────────────────────────────────────────
    setup = data.get("setup")
    if isinstance(setup, dict):
        for electrode in ("anode", "cathode"):
            block = setup.get(electrode)
            if not isinstance(block, dict):
                issues.append(ManifestValidationIssue(
                    path=f"setup.{electrode}",
                    message=f"setup.{electrode} must be an object"))
            elif "material" not in block:
                issues.append(ManifestValidationIssue(
                    path=f"setup.{electrode}.material",
                    message=f"setup.{electrode} requires a 'material' field"))
    elif "setup" in data:
        issues.append(ManifestValidationIssue(
            path="setup", message="setup must be an object"))

    return ManifestValidationReport(
        valid=not any(i.severity == "error" for i in issues),
        issues=issues,
    )


def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate an experiment manifest JSON file.

    Raises ``ValueError`` if the JSON is malformed or required keys are
    missing.  Returns the parsed dict on success (validation warnings
    may still be present — check with ``validate_experiment_manifest``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object")
    report = validate_experiment_manifest(data)
    if not report.valid:
        errors = "; ".join(f"{i.path}: {i.message}" for i in report.errors)
        raise ValueError(f"Manifest validation failed: {errors}")
    return data


def load_bath_batch(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a bath_batch.json file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bath batch file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Bath batch is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Bath batch must be a JSON object")
    for key in ("batch_id", "date_mixed", "operator", "composition", "storage"):
        if key not in data:
            raise ValueError(f"Bath batch missing required key '{key}'")
    composition = data.get("composition", {})
    for key in ("fe2_g_L", "h3bo3_g_L", "pH", "volume_mL"):
        if key not in composition:
            raise ValueError(f"Bath batch composition missing required key '{key}'")
    return data
