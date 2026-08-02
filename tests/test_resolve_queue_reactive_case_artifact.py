#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Contract tests for the final queue-reactive case-study handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "queue_case_artifact",
    ROOT / "scripts" / "resolve_queue_reactive_case_artifact.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


class QueueReactiveCaseArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve()

        self.files: dict[str, pathlib.Path] = {}
        for name in (
            "deployment_config", "value_policy", "background_policy_mapping",
            "cluster_map", "candidate_config", "executable",
        ):
            path = self.root / f"{name}.dat"
            path.write_text(f"{name}\n", encoding="utf-8")
            self.files[name] = path

        self.training_report = self.root / "strict_training.json"
        write_json(self.training_report, {
            "evaluation_role": "training_fit", "passed": True,
        })
        self.validation_report = self.root / "strict_validation.json"
        write_json(self.validation_report, {
            "evaluation_role": "development_validation", "passed": True,
        })
        self.heldout_config = self.root / "heldout_config.csv"
        self.heldout_config.write_text("symbol\nQQQ\n", encoding="utf-8")

        entries = []
        for path in sorted(self.files.values(), key=str):
            entries.append({
                "path": str(path),
                "sha256": digest(path),
                "roles": ["test_runtime_artifact"],
            })
        transitive = {
            "schema_version": 1,
            "entry_count": len(entries),
            "entries": entries,
            "manifest_sha256": MODULE.sha256_json(entries),
        }
        self.freeze = self.root / "training_freeze.json"
        write_json(self.freeze, {
            "schema_version": 1,
            "status": "expanded_training_adequacy_frozen",
            "training_only": True,
            "heldout_inputs_read": False,
            "frozen_before_any_heldout_run": True,
            "strict_training_gate_passed": True,
            "full_universe_training_adequacy_passed": True,
            "heldout_execution_authorized": True,
            "allowed_heldout_role": "development_validation",
            "ordinary_market_shared_mm_disabled": True,
            "one_rank_execution": True,
            "training_dates": [
                "2019-01-30", "2019-03-27", "2019-07-30",
                "2019-10-30", "2019-12-30",
            ],
            "selection": {"local_candidate": {
                "identifier": "local_0",
                "enabled": True,
                "interval_ms": 1000,
                "quantity_multiplier": 1.0,
                "improvement_probability": 0.2,
                "spread_elasticity": 0.5,
                "max_improvement_probability": 0.75,
            }},
            "frozen_artifacts": {
                name: {"path": str(path), "sha256": digest(path)}
                for name, path in self.files.items()
            },
            "transitive_runtime_artifacts": transitive,
            "strict_training_report": {
                "path": str(self.training_report),
                "sha256": digest(self.training_report),
            },
        })
        self.manifest = self.root / "heldout_run_manifest.json"
        write_json(self.manifest, {
            "schema_version": 1,
            "status": "heldout_adequacy_passed",
            "evaluation_role": "development_validation",
            "validation_claimed": True,
            "certification_claimed": False,
            "training_freeze": {
                "path": str(self.freeze), "sha256": digest(self.freeze),
            },
            "simulation_config": {
                "path": str(self.heldout_config),
                "sha256": digest(self.heldout_config),
            },
            "all_other_simulation_fields_frozen": True,
            "strict_report": {
                "path": str(self.validation_report),
                "sha256": digest(self.validation_report),
                "passed": True,
            },
        })

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_passed_manifest_resolves_hash_bound_queue_model(self) -> None:
        result = MODULE.resolve_artifact(self.manifest)
        self.assertEqual(result["artifact_role"], "queue_reactive_validation_handoff")
        self.assertEqual(result["background_model"], "queue-reactive-v1")
        self.assertEqual(
            result["background_policy_mapping"],
            str(self.files["background_policy_mapping"]),
        )
        self.assertEqual(result["case_config"], str(self.heldout_config))
        self.assertEqual(result["local_candidate"]["interval_ms"], 1000.0)
        self.assertEqual(
            result["local_candidate"]["spread_elasticity"], 0.5,
        )
        self.assertEqual(
            result["local_candidate"]["max_improvement_probability"],
            0.75,
        )
        self.assertEqual(result["transitive_runtime_artifact_count"], 6)

    def test_legacy_local_candidate_gets_nonbinding_adaptive_defaults(self) -> None:
        payload = json.loads(self.freeze.read_text(encoding="utf-8"))
        local = payload["selection"]["local_candidate"]
        del local["spread_elasticity"]
        del local["max_improvement_probability"]
        write_json(self.freeze, payload)
        self._refresh_freeze_references()
        result = MODULE.resolve_artifact(self.manifest)
        self.assertEqual(result["local_candidate"]["spread_elasticity"], 0.0)
        self.assertEqual(
            result["local_candidate"]["max_improvement_probability"], 1.0,
        )

    def test_invalid_adaptive_local_policy_fails_closed(self) -> None:
        payload = json.loads(self.freeze.read_text(encoding="utf-8"))
        payload["selection"]["local_candidate"][
            "max_improvement_probability"
        ] = 0.1
        write_json(self.freeze, payload)
        self._refresh_freeze_references()
        with self.assertRaisesRegex(MODULE.ArtifactError, "exceeds its cap"):
            MODULE.resolve_artifact(self.manifest)

    def _refresh_freeze_references(self) -> None:
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["training_freeze"]["sha256"] = digest(self.freeze)
        write_json(self.manifest, manifest)

    def _enable_six_component_protocol(self) -> pathlib.Path:
        certificate = self.root / "six_component_protocol_certificate.json"
        write_json(certificate, {
            "schema_version": 1,
            "status": "six_component_training_adequacy_passed",
            "passed": True,
            "protocol": {
                "classification": "retrospective_development_reanalysis",
                "marketwide_metrics_are_authoritative": True,
            },
        })
        write_json(self.training_report, {
            "evaluation_role": "training_fit",
            "gate_protocol": "marketwide-six-v2",
            "passed": True,
        })
        write_json(self.validation_report, {
            "evaluation_role": "development_validation",
            "gate_protocol": "marketwide-six-v2",
            "passed": True,
        })
        freeze = json.loads(self.freeze.read_text(encoding="utf-8"))
        freeze["training_adequacy_protocol"] = "marketwide-six-v2"
        freeze["training_adequacy_gate_passed"] = True
        freeze["strict_training_gate_passed"] = False
        freeze["marketwide_six_component_training_gate_passed"] = True
        freeze["strict_training_report"]["sha256"] = digest(
            self.training_report
        )
        freeze["protocol_revision_certificate"] = {
            "path": str(certificate), "sha256": digest(certificate),
        }
        transitive = freeze["transitive_runtime_artifacts"]
        transitive["entries"].append({
            "path": str(certificate),
            "sha256": digest(certificate),
            "roles": ["protocol_revision_certificate"],
        })
        transitive["entries"].sort(key=lambda row: row["path"])
        transitive["entry_count"] = len(transitive["entries"])
        transitive["manifest_sha256"] = MODULE.sha256_json(
            transitive["entries"]
        )
        write_json(self.freeze, freeze)

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["gate_protocol"] = "marketwide-six-v2"
        manifest["strict_report"]["sha256"] = digest(self.validation_report)
        manifest["training_freeze"]["sha256"] = digest(self.freeze)
        write_json(self.manifest, manifest)
        return certificate

    def _enable_verified_multi_rank_freeze(self, ranks: int = 32) -> pathlib.Path:
        evidence = self.root / "rank_equivalence.json"
        write_json(evidence, {
            "status": "rank_equivalence_passed",
            "reference_ranks": 1,
            "production_ranks": ranks,
            "summary_bytes_equal": True,
            "terminal_state_hash_equal": True,
            "executable": {
                "path": str(self.files["executable"]),
                "sha256": digest(self.files["executable"]),
            },
        })
        payload = json.loads(self.freeze.read_text(encoding="utf-8"))
        payload["one_rank_execution"] = False
        payload["execution"] = {
            "mpi_ranks_per_run": ranks,
            "parallelism": "whole_book_mpi",
            "rank_equivalence_required": True,
        }
        payload["rank_equivalence"] = {
            "path": str(evidence), "sha256": digest(evidence),
        }
        transitive = payload["transitive_runtime_artifacts"]
        transitive["entries"].append({
            "path": str(evidence),
            "sha256": digest(evidence),
            "roles": ["rank_equivalence_evidence"],
        })
        transitive["entries"].sort(key=lambda row: row["path"])
        transitive["entry_count"] = len(transitive["entries"])
        transitive["manifest_sha256"] = MODULE.sha256_json(transitive["entries"])
        write_json(self.freeze, payload)
        self._refresh_freeze_references()
        return evidence

    def test_verified_multi_rank_freeze_authorizes_handoff(self) -> None:
        self._enable_verified_multi_rank_freeze(32)
        result = MODULE.resolve_artifact(self.manifest)
        self.assertEqual(result["artifact_role"], "queue_reactive_validation_handoff")
        self.assertEqual(result["transitive_runtime_artifact_count"], 7)

    def test_six_component_training_and_validation_protocol_resolves(self) -> None:
        certificate = self._enable_six_component_protocol()
        result = MODULE.resolve_artifact(self.manifest)
        self.assertEqual(
            result["training_adequacy_protocol"], "marketwide-six-v2",
        )
        self.assertEqual(
            result["protocol_revision_certificate"], str(certificate),
        )
        self.assertEqual(result["transitive_runtime_artifact_count"], 7)

    def test_six_component_manifest_cannot_claim_strict_nine_protocol(self) -> None:
        self._enable_six_component_protocol()
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["gate_protocol"] = "strict-nine-v1"
        write_json(self.manifest, manifest)
        with self.assertRaisesRegex(
            MODULE.ArtifactError, "disagrees|differs",
        ):
            MODULE.resolve_artifact(self.manifest)

    def test_tampered_multi_rank_evidence_fails_closed(self) -> None:
        evidence = self._enable_verified_multi_rank_freeze(32)
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        payload["production_ranks"] = 16
        write_json(evidence, payload)
        with self.assertRaisesRegex(MODULE.ArtifactError, "hash mismatch"):
            MODULE.resolve_artifact(self.manifest)

    def test_training_freeze_requires_explicit_diagnostic_override(self) -> None:
        with self.assertRaisesRegex(MODULE.ArtifactError, "diagnostic-only"):
            MODULE.resolve_artifact(self.freeze)
        result = MODULE.resolve_artifact(
            self.freeze, allow_training_freeze=True,
        )
        self.assertEqual(
            result["artifact_role"],
            "queue_reactive_expanded_training_freeze",
        )
        self.assertEqual(
            result["case_config"], str(self.files["deployment_config"]),
        )

    def test_tampered_transitive_artifact_fails_closed(self) -> None:
        self.files["background_policy_mapping"].write_text(
            "tampered\n", encoding="utf-8",
        )
        with self.assertRaisesRegex(MODULE.ArtifactError, "hash mismatch"):
            MODULE.resolve_artifact(self.manifest)

    def test_failed_validation_cannot_authorize_case_study(self) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["status"] = "heldout_adequacy_failed"
        payload["validation_claimed"] = False
        write_json(self.manifest, payload)
        with self.assertRaisesRegex(MODULE.ArtifactError, "must be a passed"):
            MODULE.resolve_artifact(self.manifest)

    def test_stratified_selection_cannot_back_heldout_manifest(self) -> None:
        payload = json.loads(self.freeze.read_text(encoding="utf-8"))
        payload["status"] = (
            "stratified_training_selection_frozen_pending_full_universe"
        )
        payload["heldout_execution_authorized"] = False
        payload["strict_training_gate_passed"] = False
        payload["small_panel_strict_training_report"] = payload.pop(
            "strict_training_report"
        )
        payload.pop("full_universe_training_adequacy_passed")
        payload.pop("allowed_heldout_role")
        write_json(self.freeze, payload)
        self._refresh_freeze_references()
        with self.assertRaisesRegex(
            MODULE.ArtifactError, "full-universe 2019",
        ):
            MODULE.resolve_artifact(self.manifest)

    def test_symlinked_handoff_is_rejected(self) -> None:
        link = self.root / "manifest_link.json"
        link.symlink_to(self.manifest)
        with self.assertRaisesRegex(MODULE.ArtifactError, "direct regular file"):
            MODULE.resolve_artifact(link)

    def test_strict_report_payload_must_match_manifest(self) -> None:
        write_json(self.validation_report, {
            "evaluation_role": "development_validation", "passed": False,
        })
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["strict_report"]["sha256"] = digest(self.validation_report)
        write_json(self.manifest, payload)
        with self.assertRaisesRegex(MODULE.ArtifactError, "payload is not a PASS"):
            MODULE.resolve_artifact(self.manifest)


class QueueReactiveSubmissionSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "submit_real_universe_case_study.sh").read_text(
            encoding="utf-8",
        )

    def test_final_runner_forwards_frozen_background(self) -> None:
        self.assertIn('--background-model "${BACKGROUND_MODEL}"', self.source)
        self.assertIn(
            'RUNNER+=(--background-policy-csv '
            '"${CALIBRATED_BACKGROUND_POLICY_PATH}")',
            self.source,
        )

    def test_final_science_requires_validated_queue_handoff(self) -> None:
        self.assertIn(
            '"${CALIBRATION_ARTIFACT_ROLE}" != '
            '"queue_reactive_validation_handoff"',
            self.source,
        )
        self.assertIn(
            'QUEUE_REACTIVE_MODEL_ARTIFACT pointing to a passed '
            'heldout_run_manifest.json',
            self.source,
        )

    def test_transitive_freeze_provenance_is_written_to_campaign(self) -> None:
        self.assertIn("transitive_runtime_manifest_sha256", self.source)
        self.assertIn("CALIBRATION_TRANSITIVE_ARTIFACT_COUNT", self.source)

    def test_final_runner_forwards_frozen_adaptive_local_controls(self) -> None:
        self.assertIn(
            '--local-mm-spread-elasticities '
            '"${CALIBRATED_LOCAL_MM_SPREAD_ELASTICITY}"',
            self.source,
        )
        self.assertIn(
            '--local-mm-max-improvement-probabilities '
            '"${CALIBRATED_LOCAL_MM_MAX_IMPROVEMENT_PROBABILITY}"',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
