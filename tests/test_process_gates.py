"""Tests for the measurement-only process gate engine."""


from models.process_gates import (
    EvidenceRecord,
    evaluate_gate,
    evaluate_candidate,
    evaluate_all,
    load_evidence_json,
)
from models.process_registry import load_registry


class TestEvaluateGate:
    def _gate(self, threshold=None):
        from models.process_registry import Gate
        return Gate(
            id="g1",
            description="test gate",
            metric="fe",
            threshold=threshold,
        )

    def test_no_evidence_pending(self):
        verdict = evaluate_gate(self._gate(0.7), [])
        assert verdict.status == "pending"
        assert "no evidence" in verdict.reason

    def test_literature_only_pending(self):
        evidence = [
            EvidenceRecord(
                run_id="lit-1",
                candidate_id="c1",
                gate_id="g1",
                metric="fe",
                value=0.95,
                source="literature",
            )
        ]
        verdict = evaluate_gate(self._gate(0.7), evidence)
        assert verdict.status == "pending"
        assert "literature evidence only" in verdict.reason

    def test_experimental_passes(self):
        evidence = [
            EvidenceRecord(
                run_id="r1",
                candidate_id="c1",
                gate_id="g1",
                metric="fe",
                value=0.75,
            )
        ]
        verdict = evaluate_gate(self._gate(0.7), evidence)
        assert verdict.status == "passed"

    def test_experimental_fails(self):
        evidence = [
            EvidenceRecord(
                run_id="r1",
                candidate_id="c1",
                gate_id="g1",
                metric="fe",
                value=0.60,
            )
        ]
        verdict = evaluate_gate(self._gate(0.7), evidence)
        assert verdict.status == "failed"

    def test_qualitative_gate_passes_with_any_experimental(self):
        evidence = [
            EvidenceRecord(
                run_id="r1",
                candidate_id="c1",
                gate_id="g1",
                metric="morphology",
                value=1.0,
            )
        ]
        verdict = evaluate_gate(self._gate(None), evidence)
        assert verdict.status == "passed"


class TestEvaluateCandidate:
    def test_all_gates_pass(self):
        candidates = load_registry()
        candidate = candidates[0]
        evidence = []
        for gate in candidate.gates:
            evidence.append(
                EvidenceRecord(
                    run_id=f"r-{gate.id}",
                    candidate_id=candidate.id,
                    gate_id=gate.id,
                    metric=gate.metric,
                    value=1.0,
                )
            )
        verdict = evaluate_candidate(candidate, evidence)
        assert verdict.status == "passed"

    def test_any_fail_fails_candidate(self):
        candidates = load_registry()
        candidate = candidates[0]
        numeric_gate = next(g for g in candidate.gates if g.threshold is not None)
        evidence = [
            EvidenceRecord(
                run_id="r1",
                candidate_id=candidate.id,
                gate_id=numeric_gate.id,
                metric=numeric_gate.metric,
                value=0.0,
            )
        ]
        verdict = evaluate_candidate(candidate, evidence)
        assert verdict.status == "failed"

    def test_no_evidence_pending(self):
        candidates = load_registry()
        candidate = candidates[0]
        verdict = evaluate_candidate(candidate, [])
        assert verdict.status == "pending"


class TestEvaluateAll:
    def test_returns_verdict_per_candidate(self):
        verdicts = evaluate_all([])
        assert len(verdicts) >= 3
        assert all(v.status == "pending" for v in verdicts)


class TestLoadEvidence:
    def test_loads_json(self, tmp_path):
        import json
        path = tmp_path / "evidence.json"
        path.write_text(json.dumps([
            {
                "run_id": "r1",
                "candidate_id": "c1",
                "gate_id": "g1",
                "metric": "fe",
                "value": 0.75,
            }
        ]))
        records = load_evidence_json(path)
        assert len(records) == 1
        assert records[0].value == 0.75
