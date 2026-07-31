"""Process gates — measurement-only evaluation of candidate hypotheses.

Gates read only from experimental records (campaign manifests, processed
runs, characterization) or literature anchors. Model predictions are never
gate evidence. A candidate passes when every required gate has measured
evidence meeting its threshold; it fails when any gate has measured
evidence below threshold; otherwise it is pending.

The gate engine is deliberately small: it consumes a compact evidence
table (run_id → metrics) and a candidate registry, and returns a
pass/fail/pending verdict per candidate. It does not know how to fit
models, load instrument files, or simulate anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .process_registry import Candidate, load_registry

VALID_GATE_STATUSES = frozenset({"pending", "passed", "failed", "bypassed"})


@dataclass(frozen=True)
class EvidenceRecord:
    """One measured observation tied to a run and candidate."""

    run_id: str
    candidate_id: str
    gate_id: str
    metric: str
    value: float
    unit: str = ""
    source: str = "experimental"  # experimental | literature
    passed: Optional[bool] = None  # None = threshold not evaluated here
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "gate_id": self.gate_id,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "passed": self.passed,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class GateVerdict:
    """Verdict for one gate on one candidate."""

    gate_id: str
    description: str
    metric: str
    threshold: Optional[float]
    status: str  # pending | passed | failed
    qualitative_threshold: Optional[str] = None
    evidence: List[EvidenceRecord] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "description": self.description,
            "metric": self.metric,
            "threshold": self.threshold,
            "qualitative_threshold": self.qualitative_threshold,
            "status": self.status,
            "evidence": [e.to_dict() for e in self.evidence],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CandidateVerdict:
    """Overall verdict for one candidate."""

    candidate_id: str
    name: str
    status: str  # pending | passed | failed
    gates: List[GateVerdict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "status": self.status,
            "gates": [g.to_dict() for g in self.gates],
            "reason": self.reason,
        }


def evaluate_gate(
    gate,
    evidence: List[EvidenceRecord],
) -> GateVerdict:
    """Evaluate one gate against measured evidence.

    Rules:
    - No evidence → pending.
    - Any experimental evidence below threshold → failed.
    - All evidence meets threshold → passed.
    - Literature evidence alone does not pass a gate; it informs only.
    - Gates with threshold=None are qualitative: any experimental evidence
      passes, none leaves pending.
    """
    gate_evidence = [e for e in evidence if e.gate_id == gate.id]
    experimental = [e for e in gate_evidence if e.source == "experimental"]

    if not gate_evidence:
        return GateVerdict(
            gate_id=gate.id,
            description=gate.description,
            metric=gate.metric,
            threshold=gate.threshold,
            status="pending",
            qualitative_threshold=gate.qualitative_threshold,
            evidence=[],
            reason="no evidence recorded",
        )

    if not experimental:
        return GateVerdict(
            gate_id=gate.id,
            description=gate.description,
            metric=gate.metric,
            threshold=gate.threshold,
            status="pending",
            qualitative_threshold=gate.qualitative_threshold,
            evidence=gate_evidence,
            reason="literature evidence only; experimental measurement required",
        )

    if gate.threshold is None:
        return GateVerdict(
            gate_id=gate.id,
            description=gate.description,
            metric=gate.metric,
            threshold=None,
            status="passed",
            qualitative_threshold=gate.qualitative_threshold,
            evidence=gate_evidence,
            reason="qualitative gate: experimental evidence present",
        )

    failures = [e for e in experimental if e.value < gate.threshold]
    if failures:
        return GateVerdict(
            gate_id=gate.id,
            description=gate.description,
            metric=gate.metric,
            threshold=gate.threshold,
            status="failed",
            qualitative_threshold=gate.qualitative_threshold,
            evidence=gate_evidence,
            reason=f"{len(failures)} experimental observation(s) below threshold {gate.threshold}",
        )

    return GateVerdict(
        gate_id=gate.id,
        description=gate.description,
        metric=gate.metric,
        threshold=gate.threshold,
        status="passed",
        qualitative_threshold=gate.qualitative_threshold,
        evidence=gate_evidence,
        reason=f"all {len(experimental)} experimental observation(s) meet threshold {gate.threshold}",
    )


def evaluate_candidate(
    candidate: Candidate,
    evidence: List[EvidenceRecord],
) -> CandidateVerdict:
    """Evaluate all gates for one candidate."""
    candidate_evidence = [e for e in evidence if e.candidate_id == candidate.id]
    gate_verdicts = [evaluate_gate(g, candidate_evidence) for g in candidate.gates]

    if any(v.status == "failed" for v in gate_verdicts):
        status = "failed"
        failed = [v.gate_id for v in gate_verdicts if v.status == "failed"]
        reason = f"failed gates: {', '.join(failed)}"
    elif all(v.status == "passed" for v in gate_verdicts) and gate_verdicts:
        status = "passed"
        reason = "all gates passed"
    else:
        status = "pending"
        pending = [v.gate_id for v in gate_verdicts if v.status == "pending"]
        reason = f"pending gates: {', '.join(pending)}"

    return CandidateVerdict(
        candidate_id=candidate.id,
        name=candidate.name,
        status=status,
        gates=gate_verdicts,
        reason=reason,
    )


def evaluate_all(
    evidence: List[EvidenceRecord],
    registry_path: Optional[str | Path] = None,
) -> List[CandidateVerdict]:
    """Evaluate every candidate in the registry against the evidence table."""
    candidates = load_registry(registry_path)
    return [evaluate_candidate(c, evidence) for c in candidates]


def load_evidence_json(path: str | Path) -> List[EvidenceRecord]:
    """Load an evidence table from JSON.

    Expected shape: a list of objects with keys run_id, candidate_id,
    gate_id, metric, value, and optional unit/source/notes.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evidence JSON must be a list of records")
    records = []
    for item in data:
        records.append(
            EvidenceRecord(
                run_id=str(item["run_id"]),
                candidate_id=str(item["candidate_id"]),
                gate_id=str(item["gate_id"]),
                metric=str(item["metric"]),
                value=float(item["value"]),
                unit=str(item.get("unit", "")),
                source=str(item.get("source", "experimental")),
                notes=str(item.get("notes", "")),
            )
        )
    return records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate process gates against measured evidence.")
    parser.add_argument("--registry", default=None, help="Path to candidates.yaml")
    parser.add_argument("--evidence", default=None, help="Path to evidence JSON")
    args = parser.parse_args()

    evidence = load_evidence_json(args.evidence) if args.evidence else []
    verdicts = evaluate_all(evidence, args.registry)
    print(json.dumps([v.to_dict() for v in verdicts], indent=2))


if __name__ == "__main__":
    main()
