#!/usr/bin/env python3
"""Resolve and verify the frozen queue-reactive model used by the financial case study.

The normal input is ``heldout_run_manifest.json`` from the separate held-out
evaluation.  The resolver verifies that evaluation, the training freeze it
references, every direct frozen artifact and the complete transitive runtime
manifest before returning any path to the launcher.  A training freeze may be
used only with the explicit ``--allow-training-freeze`` diagnostic option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import stat
import sys
from collections.abc import Mapping, Sequence


STRICT_NINE_PROTOCOL = "strict-nine-v1"
MARKETWIDE_SIX_PROTOCOL = "marketwide-six-v2"
SUPPORTED_GATE_PROTOCOLS = (
    STRICT_NINE_PROTOCOL,
    MARKETWIDE_SIX_PROTOCOL,
)


class ArtifactError(RuntimeError):
    """A malformed, missing or hash-mismatched freeze artifact."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def direct_regular_file(value: object, *, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{label} must be a non-empty path string")
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ArtifactError(f"{label} must be a direct regular file: {path}")
    return path.resolve()


def digest_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ArtifactError(f"{label} must be a SHA-256 hexadecimal string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ArtifactError(f"{label} is not hexadecimal") from error
    return value.lower()


def verified_record(value: object, *, label: str) -> tuple[pathlib.Path, str]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must be a path/hash object")
    path = direct_regular_file(value.get("path"), label=f"{label}.path")
    expected = digest_text(value.get("sha256"), label=f"{label}.sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise ArtifactError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return path, expected


def read_json(path: pathlib.Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot parse {label}: {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must contain a JSON object")
    return value


def finite_number(
    value: object, *, label: str, minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ArtifactError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ArtifactError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ArtifactError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ArtifactError(f"{label} must not exceed {maximum}")
    return result


def verify_transitive_manifest(
    value: object,
    *,
    required_paths: Mapping[pathlib.Path, str],
) -> tuple[str, int]:
    if not isinstance(value, Mapping):
        raise ArtifactError("training freeze lacks transitive_runtime_artifacts")
    if value.get("schema_version") != 1:
        raise ArtifactError("unsupported transitive runtime manifest schema")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ArtifactError("transitive runtime artifact list must be non-empty")
    if value.get("entry_count") != len(entries):
        raise ArtifactError("transitive runtime artifact count is inconsistent")
    expected_manifest = digest_text(
        value.get("manifest_sha256"), label="transitive manifest sha256",
    )
    observed_manifest = sha256_json(entries)
    if observed_manifest != expected_manifest:
        raise ArtifactError("transitive runtime artifact manifest digest is invalid")

    observed: dict[pathlib.Path, str] = {}
    previous_path = ""
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ArtifactError(f"transitive entry {index} is not an object")
        path, expected = verified_record(raw, label=f"transitive entry {index}")
        path_text = str(path)
        if previous_path and path_text <= previous_path:
            raise ArtifactError(
                "transitive runtime entries must be unique and path-sorted"
            )
        previous_path = path_text
        roles = raw.get("roles")
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role for role in roles)
            or roles != sorted(set(roles))
        ):
            raise ArtifactError(f"transitive entry {index} has invalid roles")
        observed[path] = expected

    for required_path, required_hash in required_paths.items():
        if observed.get(required_path) != required_hash:
            raise ArtifactError(
                "direct frozen artifact is absent or hash-inconsistent in the "
                f"transitive manifest: {required_path}"
            )
    return expected_manifest, len(entries)


def resolve_artifact(
    artifact_path: pathlib.Path,
    *,
    allow_training_freeze: bool = False,
) -> dict[str, object]:
    artifact_path = direct_regular_file(
        str(artifact_path), label="queue-reactive case artifact",
    )
    artifact = read_json(artifact_path, label="queue-reactive case artifact")
    status = artifact.get("status")

    validation_manifest_path: pathlib.Path | None = None
    validation_manifest_hash = ""
    validation_report_path: pathlib.Path | None = None
    validation_report_hash = ""
    heldout_simulation_path: pathlib.Path | None = None
    heldout_simulation_hash = ""
    evaluation_role = ""
    validation_gate_protocol = ""

    if status == "heldout_adequacy_passed":
        if artifact.get("schema_version") != 1:
            raise ArtifactError("unsupported held-out manifest schema")
        if artifact.get("validation_claimed") is not True:
            raise ArtifactError("held-out manifest does not claim passed validation")
        strict_record = artifact.get("strict_report")
        if not isinstance(strict_record, Mapping) or strict_record.get("passed") is not True:
            raise ArtifactError("held-out strict report is not marked passed")
        validation_report_path, validation_report_hash = verified_record(
            strict_record, label="held-out strict report",
        )
        strict_report = read_json(
            validation_report_path, label="held-out strict report",
        )
        if strict_report.get("passed") is not True:
            raise ArtifactError("held-out strict report payload is not a PASS")
        evaluation_role = str(artifact.get("evaluation_role", ""))
        if evaluation_role != "development_validation":
            raise ArtifactError(
                "the 2020 evaluation must be labelled development_validation"
            )
        if strict_report.get("evaluation_role") != evaluation_role:
            raise ArtifactError(
                "held-out strict report evaluation role disagrees with its manifest"
            )
        validation_gate_protocol = str(
            artifact.get("gate_protocol", STRICT_NINE_PROTOCOL)
        )
        if validation_gate_protocol not in SUPPORTED_GATE_PROTOCOLS:
            raise ArtifactError("held-out manifest names an unknown gate protocol")
        report_protocol = str(
            strict_report.get("gate_protocol", STRICT_NINE_PROTOCOL)
        )
        if report_protocol != validation_gate_protocol:
            raise ArtifactError(
                "held-out strict report gate protocol disagrees with its manifest"
            )
        freeze_path, freeze_hash = verified_record(
            artifact.get("training_freeze"), label="training freeze",
        )
        heldout_simulation_path, heldout_simulation_hash = verified_record(
            artifact.get("simulation_config"), label="held-out simulation config",
        )
        if artifact.get("all_other_simulation_fields_frozen") is not True:
            raise ArtifactError(
                "held-out manifest does not freeze all non-opening simulation fields"
            )
        validation_manifest_path = artifact_path
        validation_manifest_hash = sha256_file(artifact_path)
        artifact_role = "queue_reactive_validation_handoff"
    elif status in {
        "stratified_training_selection_frozen_pending_full_universe",
        "expanded_training_adequacy_frozen",
    }:
        if not allow_training_freeze:
            raise ArtifactError(
                "a training freeze is diagnostic-only; provide a passed "
                "heldout_run_manifest.json for the final case study"
            )
        freeze_path = artifact_path
        freeze_hash = sha256_file(artifact_path)
        artifact_role = (
            "queue_reactive_expanded_training_freeze"
            if status == "expanded_training_adequacy_frozen"
            else "queue_reactive_stratified_selection_freeze"
        )
    else:
        raise ArtifactError(
            "artifact must be a passed held-out manifest or a training freeze"
        )

    freeze = read_json(freeze_path, label="training freeze")
    freeze_status = freeze.get("status")
    one_rank_execution = freeze.get("one_rank_execution") is True
    execution = freeze.get("execution")
    multi_rank_execution = (
        freeze.get("one_rank_execution") is False
        and isinstance(execution, Mapping)
        and isinstance(execution.get("mpi_ranks_per_run"), int)
        and not isinstance(execution.get("mpi_ranks_per_run"), bool)
        and int(execution["mpi_ranks_per_run"]) > 1
        and execution.get("parallelism") == "whole_book_mpi"
        and execution.get("rank_equivalence_required") is True
    )
    if (
        freeze.get("schema_version") != 1
        or freeze_status not in {
            "stratified_training_selection_frozen_pending_full_universe",
            "expanded_training_adequacy_frozen",
        }
        or freeze.get("training_only") is not True
        or freeze.get("heldout_inputs_read") is not False
        or freeze.get("frozen_before_any_heldout_run") is not True
        or freeze.get("ordinary_market_shared_mm_disabled") is not True
        or not (one_rank_execution or multi_rank_execution)
    ):
        raise ArtifactError("training freeze does not satisfy the queue-reactive freeze contract")
    training_gate_protocol = str(
        freeze.get("training_adequacy_protocol", STRICT_NINE_PROTOCOL)
    )
    if training_gate_protocol not in SUPPORTED_GATE_PROTOCOLS:
        raise ArtifactError("training freeze names an unknown gate protocol")
    if validation_manifest_path is not None \
            and validation_gate_protocol != training_gate_protocol:
        raise ArtifactError(
            "held-out gate protocol differs from the frozen training protocol"
        )
    strict_nine_passed = (
        training_gate_protocol == STRICT_NINE_PROTOCOL
        and freeze.get("strict_training_gate_passed") is True
    )
    marketwide_six_passed = (
        training_gate_protocol == MARKETWIDE_SIX_PROTOCOL
        and freeze.get("marketwide_six_component_training_gate_passed") is True
        and freeze.get("strict_training_gate_passed") is False
    )
    expanded_training_passed = (
        freeze_status == "expanded_training_adequacy_frozen"
        and freeze.get("full_universe_training_adequacy_passed") is True
        and freeze.get("training_adequacy_gate_passed", True) is True
        and (strict_nine_passed or marketwide_six_passed)
        and freeze.get("heldout_execution_authorized") is True
        and freeze.get("allowed_heldout_role") == "development_validation"
    )
    if validation_manifest_path is not None and not expanded_training_passed:
        raise ArtifactError(
            "held-out validation is not backed by a passed full-universe "
            "2019 training-adequacy freeze"
        )
    if (
        freeze_status
        == "stratified_training_selection_frozen_pending_full_universe"
        and freeze.get("heldout_execution_authorized") is not False
    ):
        raise ArtifactError(
            "stratified selection freeze improperly authorizes held-out use"
        )
    training_dates = freeze.get("training_dates")
    if (
        not isinstance(training_dates, list)
        or len(training_dates) != 5
        or len(set(training_dates)) != 5
        or any(not isinstance(day, str) or not day for day in training_dates)
    ):
        raise ArtifactError("training freeze must contain five distinct dates")

    frozen = freeze.get("frozen_artifacts")
    if not isinstance(frozen, Mapping):
        raise ArtifactError("training freeze lacks frozen_artifacts")
    direct: dict[str, tuple[pathlib.Path, str]] = {}
    for name in (
        "deployment_config", "value_policy", "background_policy_mapping",
        "cluster_map", "candidate_config", "executable",
    ):
        direct[name] = verified_record(
            frozen.get(name), label=f"frozen artifact {name}",
        )
    required_paths = {path: digest for path, digest in direct.values()}
    protocol_certificate_path: pathlib.Path | None = None
    protocol_certificate_hash = ""
    if training_gate_protocol == MARKETWIDE_SIX_PROTOCOL:
        protocol_certificate_path, protocol_certificate_hash = verified_record(
            freeze.get("protocol_revision_certificate"),
            label="six-component protocol revision certificate",
        )
        protocol_certificate = read_json(
            protocol_certificate_path,
            label="six-component protocol revision certificate",
        )
        protocol_definition = protocol_certificate.get("protocol")
        if (
            protocol_certificate.get("schema_version") != 1
            or protocol_certificate.get("status")
                != "six_component_training_adequacy_passed"
            or protocol_certificate.get("passed") is not True
            or not isinstance(protocol_definition, Mapping)
            or protocol_definition.get("classification")
                != "retrospective_development_reanalysis"
            or protocol_definition.get("marketwide_metrics_are_authoritative")
                is not True
        ):
            raise ArtifactError(
                "six-component protocol revision certificate is malformed"
            )
        required_paths[protocol_certificate_path] = protocol_certificate_hash
    if multi_rank_execution:
        rank_path, rank_digest = verified_record(
            freeze.get("rank_equivalence"), label="rank-equivalence evidence",
        )
        rank_evidence = read_json(
            rank_path, label="rank-equivalence evidence",
        )
        expected_ranks = int(execution["mpi_ranks_per_run"])  # type: ignore[index]
        executable_digest = direct["executable"][1]
        evidence_executable = rank_evidence.get("executable")
        if (
            rank_evidence.get("status") != "rank_equivalence_passed"
            or rank_evidence.get("reference_ranks") != 1
            or rank_evidence.get("production_ranks") != expected_ranks
            or rank_evidence.get("summary_bytes_equal") is not True
            or rank_evidence.get("terminal_state_hash_equal") is not True
            or not isinstance(evidence_executable, Mapping)
            or evidence_executable.get("sha256") != executable_digest
        ):
            raise ArtifactError(
                "rank-equivalence evidence does not authorize the frozen MPI execution"
            )
        required_paths[rank_path] = rank_digest
    transitive_hash, transitive_count = verify_transitive_manifest(
        freeze.get("transitive_runtime_artifacts"),
        required_paths=required_paths,
    )

    training_report_key = (
        "strict_training_report"
        if expanded_training_passed else "small_panel_strict_training_report"
    )
    training_report_path, training_report_hash = verified_record(
        freeze.get(training_report_key), label="training adequacy report",
    )
    training_report = read_json(
        training_report_path, label="training adequacy report",
    )
    if training_report.get("evaluation_role") != "training_fit":
        raise ArtifactError("training adequacy report has the wrong evaluation role")
    if expanded_training_passed and training_report.get("passed") is not True:
        raise ArtifactError("full-universe training adequacy report is not a PASS")
    training_report_protocol = str(
        training_report.get("gate_protocol", STRICT_NINE_PROTOCOL)
    )
    if training_report_protocol != training_gate_protocol:
        raise ArtifactError(
            "training adequacy report gate protocol disagrees with its freeze"
        )

    selection = freeze.get("selection")
    if not isinstance(selection, Mapping):
        raise ArtifactError("training freeze lacks selection")
    local = selection.get("local_candidate")
    if not isinstance(local, Mapping):
        raise ArtifactError("training freeze lacks its local candidate")
    enabled = local.get("enabled")
    if not isinstance(enabled, bool):
        raise ArtifactError("selected local-MM enablement must be Boolean")
    interval = finite_number(
        local.get("interval_ms"), label="selected local-MM interval", minimum=1.0,
    )
    quantity = finite_number(
        local.get("quantity_multiplier"),
        label="selected local-MM quantity multiplier", minimum=0.0,
    )
    if quantity <= 0.0:
        raise ArtifactError(
            "selected local-MM quantity multiplier must be strictly positive"
        )
    improvement = finite_number(
        local.get("improvement_probability"),
        label="selected local-MM improvement probability", minimum=0.0,
        maximum=1.0,
    )
    spread_elasticity = finite_number(
        local.get("spread_elasticity", 0.0),
        label="selected local-MM spread elasticity", minimum=0.0,
    )
    max_improvement_probability = finite_number(
        local.get("max_improvement_probability", 1.0),
        label="selected local-MM maximum improvement probability",
        minimum=0.0, maximum=1.0,
    )
    if improvement > max_improvement_probability:
        raise ArtifactError(
            "selected local-MM base improvement probability exceeds its cap"
        )
    if not enabled and spread_elasticity != 0.0:
        raise ArtifactError(
            "disabled selected local-MM policy has nonzero spread elasticity"
        )

    deployment_path, deployment_hash = direct["deployment_config"]
    if heldout_simulation_path is None:
        case_config_path, case_config_hash = deployment_path, deployment_hash
    else:
        case_config_path, case_config_hash = (
            heldout_simulation_path, heldout_simulation_hash,
        )
    background_path, background_hash = direct["background_policy_mapping"]
    value_path, value_hash = direct["value_policy"]
    cluster_path, cluster_hash = direct["cluster_map"]
    executable_path, executable_hash = direct["executable"]

    return {
        "schema_version": 1,
        "artifact_role": artifact_role,
        "input_artifact": str(artifact_path),
        "input_artifact_sha256": sha256_file(artifact_path),
        "validation_manifest": (
            str(validation_manifest_path) if validation_manifest_path else ""
        ),
        "validation_manifest_sha256": validation_manifest_hash,
        "evaluation_role": evaluation_role,
        "training_adequacy_protocol": training_gate_protocol,
        "training_freeze": str(freeze_path),
        "training_freeze_sha256": freeze_hash,
        "training_dates": list(training_dates),
        "case_config": str(case_config_path),
        "case_config_sha256": case_config_hash,
        "deployment_config": str(deployment_path),
        "deployment_config_sha256": deployment_hash,
        "value_policy": str(value_path),
        "value_policy_sha256": value_hash,
        "background_model": "queue-reactive-v1",
        "background_policy_mapping": str(background_path),
        "background_policy_mapping_sha256": background_hash,
        "cluster_map": str(cluster_path),
        "cluster_map_sha256": cluster_hash,
        "executable": str(executable_path),
        "executable_sha256": executable_hash,
        "transitive_runtime_manifest_sha256": transitive_hash,
        "transitive_runtime_artifact_count": transitive_count,
        "strict_training_report": str(training_report_path),
        "strict_training_report_sha256": training_report_hash,
        "protocol_revision_certificate": (
            str(protocol_certificate_path)
            if protocol_certificate_path is not None else ""
        ),
        "protocol_revision_certificate_sha256": protocol_certificate_hash,
        "strict_validation_report": (
            str(validation_report_path) if validation_report_path else ""
        ),
        "strict_validation_report_sha256": validation_report_hash,
        "local_candidate": {
            "identifier": str(local.get("identifier", "")),
            "enabled": enabled,
            "interval_ms": interval,
            "quantity_multiplier": quantity,
            "improvement_probability": improvement,
            "spread_elasticity": spread_elasticity,
            "max_improvement_probability": max_improvement_probability,
        },
        "decision_window_ms": 1000.0,
        "hawkes_activity_scale": 0.3,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact", type=pathlib.Path, required=True)
    result.add_argument(
        "--allow-training-freeze", action="store_true",
        help="allow a training-only diagnostic artifact without held-out validation",
    )
    result.add_argument("--output", type=pathlib.Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = resolve_artifact(
            args.artifact, allow_training_freeze=args.allow_training_freeze,
        )
    except (ArtifactError, OSError, json.JSONDecodeError) as error:
        print(f"queue-reactive case artifact validation failed: {error}", file=sys.stderr)
        return 1
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
