"""Process candidate registry — schema, loader, and validation.

Reads ``processes/candidates.yaml`` and returns typed candidate records.
The registry is the single source of truth for flowsheet hypotheses;
candidate evaluations and process gates read from here, never from
hard-coded route assumptions in individual models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "processes" / "candidates.yaml"

VALID_CANDIDATE_STATUSES = frozenset({
    "active",
    "competitor_watch",
    "hypothesis",
    "archived",
})

VALID_PRIORITIES = frozenset({
    "primary",
    "comparator",
    "exploratory",
})

VALID_PRODUCT_FORMS = frozenset({
    "flake",
    "powder_or_particle",
    "plate_or_foil",
    "near_net_shape",
})

VALID_DEPLOYMENTS = frozenset({
    "modular_proving_ground",
    "centralized_or_modular",
    "fixed_pilot",
})


@dataclass(frozen=True)
class UnitOperation:
    """One stage in a candidate flowsheet."""

    id: str
    type: str
    chemistry: str = ""
    assumed: bool = False
    evidence_refs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Gate:
    """A measured gate a candidate must pass."""

    id: str
    description: str
    metric: str
    threshold: Optional[float] = None
    qualitative_threshold: Optional[str] = None
    evidence: Optional[str] = None
    status: str = "pending"  # pending | passed | failed | bypassed


@dataclass(frozen=True)
class Candidate:
    """A complete flowsheet hypothesis."""

    id: str
    name: str
    status: str
    priority: str
    product_form: str
    deployment: str
    summary: str
    feedstocks: List[str]
    unit_operations: List[UnitOperation]
    gates: List[Gate]
    cathode: Dict[str, Any] = field(default_factory=dict)
    separator: Dict[str, Any] = field(default_factory=dict)
    anode: Dict[str, Any] = field(default_factory=dict)
    harvesting: Dict[str, Any] = field(default_factory=dict)
    recycle: Dict[str, Any] = field(default_factory=dict)
    ip_notes: str = ""

    def assumed_operations(self) -> List[UnitOperation]:
        """Unit operations still based on assumption rather than evidence."""
        return [op for op in self.unit_operations if op.assumed]

    def pending_gates(self) -> List[Gate]:
        """Gates not yet passed."""
        return [g for g in self.gates if g.status == "pending"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "priority": self.priority,
            "product_form": self.product_form,
            "deployment": self.deployment,
            "summary": self.summary,
            "feedstocks": self.feedstocks,
            "unit_operations": [
                {
                    "id": op.id,
                    "type": op.type,
                    "chemistry": op.chemistry,
                    "assumed": op.assumed,
                    "evidence_refs": op.evidence_refs,
                }
                for op in self.unit_operations
            ],
            "gates": [
                {
                    "id": g.id,
                    "description": g.description,
                    "metric": g.metric,
                    "threshold": g.threshold,
                    "qualitative_threshold": g.qualitative_threshold,
                    "evidence": g.evidence,
                    "status": g.status,
                }
                for g in self.gates
            ],
            "cathode": self.cathode,
            "separator": self.separator,
            "anode": self.anode,
            "harvesting": self.harvesting,
            "recycle": self.recycle,
            "ip_notes": self.ip_notes,
        }


def _parse_unit_operation(raw: Dict[str, Any]) -> UnitOperation:
    return UnitOperation(
        id=str(raw["id"]),
        type=str(raw["type"]),
        chemistry=str(raw.get("chemistry", "")),
        assumed=bool(raw.get("assumed", False)),
        evidence_refs=[str(r) for r in raw.get("evidence_refs", [])],
    )


def _parse_gate(raw: Dict[str, Any]) -> Gate:
    threshold = raw.get("threshold")
    qualitative = raw.get("qualitative_threshold")
    # YAML may give a string for qualitative gates; keep both fields distinct.
    if isinstance(threshold, str) and qualitative is None:
        qualitative = threshold
        threshold = None
    return Gate(
        id=str(raw["id"]),
        description=str(raw["description"]),
        metric=str(raw["metric"]),
        threshold=float(threshold) if threshold is not None else None,
        qualitative_threshold=str(qualitative) if qualitative is not None else None,
        evidence=raw.get("evidence"),
        status=str(raw.get("status", "pending")),
    )


def _parse_candidate(raw: Dict[str, Any]) -> Candidate:
    return Candidate(
        id=str(raw["id"]),
        name=str(raw["name"]),
        status=str(raw["status"]),
        priority=str(raw["priority"]),
        product_form=str(raw["product_form"]),
        deployment=str(raw["deployment"]),
        summary=str(raw["summary"]).strip(),
        feedstocks=[str(f) for f in raw.get("feedstocks", [])],
        unit_operations=[_parse_unit_operation(op) for op in raw.get("unit_operations", [])],
        gates=[_parse_gate(g) for g in raw.get("gates", [])],
        cathode=dict(raw.get("cathode", {})),
        separator=dict(raw.get("separator", {})),
        anode=dict(raw.get("anode", {})),
        harvesting=dict(raw.get("harvesting", {})),
        recycle=dict(raw.get("recycle", {})),
        ip_notes=str(raw.get("ip_notes", "")).strip(),
    )


def load_registry(path: Optional[str | Path] = None) -> List[Candidate]:
    """Load and validate the candidate registry YAML.

    Raises
    ------
    ValueError
        If the registry is malformed, has duplicate candidate IDs, or
        contains invalid status/priority/product/deployment values.
    """
    path = Path(path) if path is not None else REGISTRY_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Candidate registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "candidates" not in data:
        raise ValueError("Registry must be a mapping with a 'candidates' key")
    raw_candidates = data["candidates"]
    if not isinstance(raw_candidates, list):
        raise ValueError("'candidates' must be a list")

    candidates: List[Candidate] = []
    seen_ids: set[str] = set()
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("Each candidate must be a mapping")
        candidate = _parse_candidate(raw)
        if candidate.id in seen_ids:
            raise ValueError(f"Duplicate candidate id: {candidate.id}")
        seen_ids.add(candidate.id)
        if candidate.status not in VALID_CANDIDATE_STATUSES:
            raise ValueError(
                f"Candidate {candidate.id}: invalid status '{candidate.status}'"
            )
        if candidate.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Candidate {candidate.id}: invalid priority '{candidate.priority}'"
            )
        if candidate.product_form not in VALID_PRODUCT_FORMS:
            raise ValueError(
                f"Candidate {candidate.id}: invalid product_form '{candidate.product_form}'"
            )
        if candidate.deployment not in VALID_DEPLOYMENTS:
            raise ValueError(
                f"Candidate {candidate.id}: invalid deployment '{candidate.deployment}'"
            )
        candidates.append(candidate)
    return candidates


def get_candidate(candidate_id: str, path: Optional[str | Path] = None) -> Candidate:
    """Return one candidate by id."""
    for candidate in load_registry(path):
        if candidate.id == candidate_id:
            return candidate
    raise KeyError(f"Candidate not found: {candidate_id}")


def registry_summary(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Compact summary for CLI / decision reports."""
    candidates = load_registry(path)
    return {
        "n_candidates": len(candidates),
        "candidates": [
            {
                "id": c.id,
                "name": c.name,
                "status": c.status,
                "priority": c.priority,
                "n_assumed_operations": len(c.assumed_operations()),
                "n_pending_gates": len(c.pending_gates()),
            }
            for c in candidates
        ],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the process candidate registry.")
    parser.add_argument("--registry", default=None, help="Path to candidates.yaml")
    parser.add_argument("--candidate", default=None, help="Print one candidate as JSON")
    args = parser.parse_args()

    if args.candidate:
        print(json.dumps(get_candidate(args.candidate, args.registry).to_dict(), indent=2))
    else:
        print(json.dumps(registry_summary(args.registry), indent=2))


if __name__ == "__main__":
    main()
