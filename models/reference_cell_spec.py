"""Immutable reference-cell configuration spec: load, freeze, and verify.

This module is the machine side of the D1 "immutable reference configuration".
The canonical human document is ``docs/REFERENCE_CELL_SPEC.md`` and the machine
spec is ``processes/reference_cell_spec.v1.json``.

Immutability
------------
The spec's envelope carries a ``sha256`` that is a content hash of the spec
object *with the ``sha256`` field cleared*, serialized canonically (sorted keys,
two-space indent, trailing newline). Because the hash is computed from a
canonical re-serialization, it is independent of the on-disk file's
whitespace/formatting and is tamper-evident: any change to a value changes the
hash. Changing a value therefore requires a *new* spec version, never an edit.

``models.run_record`` consumes the pinned version: every reference run's
``reference_cell.json`` records ``spec_version`` + ``spec_sha256`` and
``models.run_record.validate_reference_cell_record`` verifies that the digest
matches the canonical file when it is reachable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_NAME = "aqueous-electrowinning.reference-cell-spec"
SPEC_VERSION = "1.0"
CONFIGURATION_ID = "RC-1"

# Canonical location of the frozen machine spec, relative to the repo root.
CANONICAL_SPEC_RELATIVE = Path("processes/reference_cell_spec.v1.json")


def canonical_json(data: Any) -> str:
    """Serialize a spec object to the canonical byte span for hashing.

    The ``sha256`` field (if present) is excluded so the digest of a spec never
    depends on the digest itself. Sorted keys + two-space indent + trailing
    newline keep the digest stable across cosmetic whitespace changes.
    """
    payload = dict(data) if isinstance(data, dict) else data
    if isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if k != "sha256"}
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def spec_hash(data: dict[str, Any]) -> str:
    """SHA-256 (hex) of a spec dict over its canonical serialization."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Raw SHA-256 of a file's bytes (used for raw exports / sidecars)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a reference-cell spec JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reference-cell spec not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"reference-cell spec is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("reference-cell spec must be a JSON object")
    if data.get("schema") != SCHEMA_NAME:
        raise ValueError(
            f"spec schema {data.get('schema')!r} != expected {SCHEMA_NAME!r}"
        )
    return data


def verify_spec(path: str | Path) -> tuple[bool, str]:
    """Verify that the on-disk ``sha256`` matches the content hash.

    Returns ``(ok, message)``. ``ok`` is True when the embedded ``sha256`` is a
    valid 64-hex digest and equals ``spec_hash(data)``.
    """
    data = load_spec(path)
    declared = data.get("sha256", "")
    computed = spec_hash(data)
    if not declared:
        return False, "spec has no sha256; freeze it first"
    if not (isinstance(declared, str) and len(declared) == 64
            and all(c in "0123456789abcdef" for c in declared.lower())):
        return False, f"declared sha256 is not a 64-hex digest: {declared!r}"
    if declared.lower() != computed:
        return False, "sha256 mismatch; spec content differs from the frozen hash"
    return True, f"spec {data.get('spec_version')} sha256 OK ({computed})"


def freeze_spec(path: str | Path) -> str:
    """Stamp a spec file's ``sha256`` from its own content.

    Idempotent: after freezing, ``verify_spec`` passes. The file is rewritten
    with the computed hash in place.
    """
    path = Path(path)
    data = load_spec(path)
    if "sha256" in data and data["sha256"]:
        # already frozen; confirm it is self-consistent rather than overwriting
        ok, _ = verify_spec(path)
        if ok:
            return data["sha256"]
    digest = spec_hash(data)
    data["sha256"] = digest
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return digest


def canonical_spec_path(repo_root: str | Path) -> Path:
    """Resolve the canonical frozen spec relative to a repo root."""
    return Path(repo_root) / CANONICAL_SPEC_RELATIVE


def verify_run_pin(
    spec_sha256: str,
    *,
    spec_file: str | Path | None = None,
) -> tuple[bool, str]:
    """Verify that a run's pinned digest matches the canonical spec content.

    Used by :func:`models.run_record.validate_reference_cell_record`. When
    ``spec_file`` is given and reachable, the declared ``spec_sha256`` must
    equal the canonical content hash of that file. When no file is supplied,
    only the digest *format* is checked (content verification is a warning; the
    run is valid but not content-linked).
    """
    if not (isinstance(spec_sha256, str) and len(spec_sha256) == 64
            and all(c in "0123456789abcdef" for c in spec_sha256.lower())):
        return False, "spec_sha256 is not a 64-hex digest"
    if spec_file is None:
        return True, "spec sha256 format OK (spec file not supplied for content check)"
    path = Path(spec_file)
    if not path.is_file():
        return False, f"spec file not reachable for content check: {path}"
    try:
        data = load_spec(path)
    except (ValueError, FileNotFoundError) as exc:
        return False, f"cannot load spec for content check: {exc}"
    declared = data.get("sha256", "")
    if declared.lower() != spec_sha256.lower():
        return False, (
            f"spec file sha256 {declared.lower()!r} does not match run pin "
            f"{spec_sha256.lower()!r}"
        )
    ok, _ = verify_spec(path)
    if not ok:
        return False, "spec file fails its own content hash"
    return True, "run pin matches canonical spec content hash"


if __name__ == "__main__":
    # python -m models.reference_cell_spec <spec_path>  -> freeze
    # python -m models.reference_cell_spec --verify <spec_path> -> verify
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--verify":
        ok, msg = verify_spec(args[1] if len(args) > 1 else CANONICAL_SPEC_RELATIVE)
        print(msg)
        raise SystemExit(0 if ok else 1)
    if args:
        digest = freeze_spec(args[0])
        print(f"frozen sha256: {digest}")
    else:
        digest = freeze_spec(CANONICAL_SPEC_RELATIVE)
        print(f"frozen sha256: {digest}")
