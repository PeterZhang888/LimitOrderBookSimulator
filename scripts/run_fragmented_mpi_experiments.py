#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Run a validated fragmented-market MPI matrix and write raw/summary CSVs.

Every runtime agent control is part of each scenario identity, so timing or
hash results from distinct behavioural settings cannot be pooled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence


RESULT_PREFIX = "fragmented_mpi_lob "


class MpiRunTimeout(RuntimeError):
    """Raised after a timed-out launcher and its local process group are reaped."""

    def __init__(self, command: Sequence[str], timeout_seconds: float,
                 stdout: str, stderr: str) -> None:
        super().__init__(
            f"MPI launcher exceeded the {timeout_seconds:g}-second per-run timeout"
        )
        self.command = list(command)
        self.timeout_seconds = timeout_seconds
        self.stdout = stdout
        self.stderr = stderr


def stable_sha256(value: object) -> str:
    """Hash a canonical JSON value for campaign and per-run identities."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_argument(value: object) -> object:
    """Convert argparse values to a stable, JSON-serializable representation."""
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (list, tuple)):
        return [canonical_argument(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): canonical_argument(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float = 5.0,
) -> tuple[str, str]:
    """Terminate a launcher session, escalating to SIGKILL after a short grace."""
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        stdout, stderr = process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    return stdout or "", stderr or ""


def run_mpi_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    """Run one MPI launcher in a new session so timeout cleanup reaches children."""
    process = subprocess.Popen(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(environment),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = terminate_process_group(process)
        # ``communicate`` after termination normally returns the complete
        # captured streams. Retain TimeoutExpired's partial output for unusual
        # Popen implementations that return an empty final value.
        if not stdout and error.stdout:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout
        if not stderr and error.stderr:
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr
        raise MpiRunTimeout(command, float(timeout_seconds), stdout, stderr) from error
    except BaseException:
        terminate_process_group(process)
        raise
    completed = subprocess.CompletedProcess(
        list(command), process.returncode, stdout or "", stderr or "",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def comma_ints(text: str, option: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} requires comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{option} values must be positive")
    return values


def comma_floats(text: str, option: str) -> list[float]:
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} requires comma-separated numbers") from error
    if (not values or any(not math.isfinite(value) or value <= 0.0
                          for value in values)):
        raise argparse.ArgumentTypeError(
            f"{option} values must be finite and positive"
        )
    return values


def comma_probabilities(text: str, option: str) -> list[float]:
    """Parse a nonempty comma list of finite probabilities in [0, 1]."""
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{option} requires comma-separated numbers"
        ) from error
    if (not values or any(not math.isfinite(value) or not 0.0 <= value <= 1.0
                          for value in values)):
        raise argparse.ArgumentTypeError(
            f"{option} values must be finite probabilities in [0, 1]"
        )
    return values


def comma_nonnegative_floats(text: str, option: str) -> list[float]:
    """Parse a nonempty comma list of finite non-negative numbers."""
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{option} requires comma-separated numbers"
        ) from error
    if (not values or any(not math.isfinite(value) or value < 0.0
                          for value in values)):
        raise argparse.ArgumentTypeError(
            f"{option} values must be finite and non-negative"
        )
    return values


def comma_choices(text: str, option: str) -> list[str]:
    values = text.split(",")
    if not values or any(value not in {"on", "off"} for value in values):
        raise argparse.ArgumentTypeError(f"{option} values must be on or off")
    return values


def shared_mode_choices(text: str) -> list[str]:
    aliases = {"on": "global", "global": "global",
               "uncoupled": "uncoupled", "off": "off"}
    raw = text.split(",")
    if not raw or any(value not in aliases for value in raw):
        raise argparse.ArgumentTypeError(
            "--shared-mm-modes values must be global, uncoupled, or off"
        )
    return [aliases[value] for value in raw]


def parse_result(stdout: str) -> dict[str, str]:
    line = next((row for row in stdout.splitlines() if row.startswith(RESULT_PREFIX)), None)
    if line is None:
        raise RuntimeError(f"simulator output has no result line:\n{stdout}")
    fields: dict[str, str] = {}
    for match in re.finditer(r"([a-z_]+)=([^ ]+)", line):
        fields[match.group(1)] = match.group(2)
    required = {
        "ranks",
        "assets",
        "lobs",
        "wall_seconds",
        "processed_orders",
        "risk_limit_per_asset",
        "state_hash",
    }
    missing = required.difference(fields)
    if missing:
        raise RuntimeError(f"simulator result is missing {sorted(missing)}")
    return fields


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically replace a CSV so an interrupted checkpoint stays readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_checkpoint(path: Path) -> list[dict[str, str]]:
    """Load a nonempty, internally keyed raw checkpoint for safe resumption."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"resume checkpoint has no CSV header: {path}")
            required = {"campaign_sha256", "run_key"}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise RuntimeError(
                    f"resume checkpoint {path} lacks fields {sorted(missing)}; "
                    "use a new output path or --overwrite"
                )
            rows = [dict(row) for row in reader]
    except csv.Error as error:
        raise RuntimeError(f"cannot parse resume checkpoint {path}: {error}") from error
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        key = row.get("run_key", "")
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise RuntimeError(f"invalid run_key in {path}:{line_number}")
        if key in seen:
            raise RuntimeError(f"duplicate run_key in {path}:{line_number}: {key}")
        seen.add(key)
    return rows


def verified_artifact(path_text: str, digest_text: str, label: str) -> None:
    """Require a resumed side artifact to exist and match its recorded digest."""
    if not path_text:
        if digest_text:
            raise RuntimeError(f"resume row has {label} digest but no path")
        return
    path = Path(path_text)
    if not path.is_file():
        raise RuntimeError(f"resume row references missing {label}: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", digest_text):
        raise RuntimeError(f"resume row has invalid {label} SHA-256")
    observed = sha256_file(path)
    if observed != digest_text:
        raise RuntimeError(
            f"resume row {label} SHA-256 mismatch: expected {digest_text}, "
            f"observed {observed} for {path}"
        )


def validate_resumed_row(
    row: Mapping[str, str],
    *,
    campaign_sha256: str,
    run_key: str,
    expected: Mapping[str, str],
) -> None:
    """Reject stale, malformed, or partially materialized checkpoint rows."""
    if row.get("campaign_sha256") != campaign_sha256:
        raise RuntimeError(
            "resume checkpoint belongs to a different command/configuration campaign"
        )
    if row.get("run_key") != run_key:
        raise RuntimeError("resume checkpoint run key does not match the planned case")
    for field, value in expected.items():
        if row.get(field) != value:
            raise RuntimeError(
                f"resume row {run_key} has {field}={row.get(field)!r}; "
                f"expected {value!r}"
            )
    try:
        if not math.isfinite(float(row["wall_seconds"])) \
                or float(row["wall_seconds"]) < 0.0:
            raise ValueError("invalid wall time")
        if int(row["processed_orders"]) < 0:
            raise ValueError("invalid processed order count")
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"resume row {run_key} has invalid result fields") from error
    if not row.get("state_hash"):
        raise RuntimeError(f"resume row {run_key} has no state hash")
    for field, label in (
        ("metrics_csv", "metrics CSV"),
        ("shock_targets_csv", "shock-target CSV"),
        ("asset_summary_csv", "asset-summary CSV"),
    ):
        verified_artifact(
            row.get(field, ""), row.get(f"{field}_sha256", ""), label,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def background_artifact_manifest(mapping_path: Path) -> tuple[str, int]:
    """Hash the queue-policy mapping and every file it references.

    The mapping is only an indirection table.  Hashing that CSV alone would
    allow a cluster policy or improvement-mark distribution to change without
    changing the campaign identity.  Resolve relative paths exactly as the
    C++ loader does and bind the complete runtime input set instead.
    """
    required = {
        "symbol",
        "cluster_id",
        "policy_file",
        "limit_buy_improvement_file",
        "limit_sell_improvement_file",
    }
    try:
        with mapping_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or [])
            missing = required - fields
            if missing:
                raise SystemExit(
                    "background-policy-csv lacks required columns: "
                    + ", ".join(sorted(missing))
                )
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise SystemExit(
            f"cannot read background-policy-csv {mapping_path}: {error}"
        ) from error
    if not rows:
        raise SystemExit("background-policy-csv contains no symbol rows")

    artifacts: set[Path] = {mapping_path.resolve()}
    observed_symbols: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        symbol = (row.get("symbol") or "").strip()
        if not symbol or symbol in observed_symbols:
            raise SystemExit(
                f"background-policy-csv has an empty or duplicate symbol at line {line_number}"
            )
        observed_symbols.add(symbol)
        for field in (
            "policy_file",
            "limit_buy_improvement_file",
            "limit_sell_improvement_file",
        ):
            value = (row.get(field) or "").strip()
            if not value:
                raise SystemExit(
                    f"background-policy-csv has empty {field} at line {line_number}"
                )
            path = Path(value)
            resolved = (
                path.resolve()
                if path.is_absolute()
                else (mapping_path.parent / path).resolve()
            )
            if not resolved.is_file():
                raise SystemExit(
                    f"background-policy-csv references missing {field}: {resolved}"
                )
            artifacts.add(resolved)
    manifest = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(artifacts, key=str)
    ]
    return stable_sha256(manifest), len(manifest)


def float_label(value: float) -> str:
    """Return a compact, filesystem-safe label for a positive finite float."""
    return repr(value).replace("-", "m").replace("+", "p").replace(".", "p")


def control_values(
    hawkes_activity_scale: float,
    local_mm_interval_ms: float,
    local_mm_quantity_multiplier: float,
    local_mm_improvement_probability: float,
    local_mm_spread_elasticity: float,
    local_mm_max_improvement_probability: float,
    shared_quote_quantity: int,
) -> dict[str, str]:
    """Canonical strings used in CSV provenance, grouping, and hash scopes."""
    return {
        "hawkes_activity_scale": repr(hawkes_activity_scale),
        "local_mm_interval_ms": repr(local_mm_interval_ms),
        "local_mm_quantity_multiplier": repr(local_mm_quantity_multiplier),
        "local_mm_improvement_probability": repr(
            local_mm_improvement_probability
        ),
        "local_mm_spread_elasticity": repr(local_mm_spread_elasticity),
        "local_mm_max_improvement_probability": repr(
            local_mm_max_improvement_probability
        ),
        "shared_quote_quantity": str(shared_quote_quantity),
    }


def make_run_key(
    campaign_sha256: str,
    *,
    asset_count: int,
    rank_count: int,
    risk_limit: float,
    shared_mode: str,
    shock_mode: str,
    capacity_threshold: float,
    controls: Mapping[str, str],
    repetition: int,
    seed: int,
) -> str:
    """Return the stable identity of one exact matrix invocation."""
    return stable_sha256(
        {
            "campaign_sha256": campaign_sha256,
            "run": {
                "asset_count": asset_count,
                "rank_count": rank_count,
                "risk_limit": repr(risk_limit),
                "shared_mode": shared_mode,
                "shock_mode": shock_mode,
                "capacity_threshold": repr(capacity_threshold),
                "controls": dict(controls),
                "repetition": repetition,
                "seed": seed,
            },
        }
    )


def verify_or_record_control(
    result: dict[str, str],
    field: str,
    expected: str,
) -> None:
    """Keep runner provenance and executable-reported resolved values consistent.

    The runner always records the requested values.  Simulator versions that
    echo resolved controls are checked as an additional guard against a
    mismatched executable or an ignored command-line option.
    """
    observed = result.get(field)
    if observed is not None:
        try:
            matches = math.isclose(
                float(observed), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12
            )
        except ValueError as error:
            raise RuntimeError(
                f"simulator returned non-numeric {field}={observed!r}"
            ) from error
        if not matches:
            raise RuntimeError(
                f"simulator resolved {field}={observed}, expected {expected}"
            )
    result[field] = expected
    result[f"requested_{field}"] = expected


def verify_or_record_exact(
    result: dict[str, str], field: str, expected: str,
) -> None:
    """Verify an executable-reported categorical/path value when available."""
    observed = result.get(field)
    if observed is not None and observed != expected:
        raise RuntimeError(
            f"simulator resolved {field}={observed!r}, expected {expected!r}"
        )
    result[field] = expected
    result[f"requested_{field}"] = expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--base-config", type=Path)
    source_group.add_argument(
        "--universe-config",
        type=Path,
        help="run every real asset row in this exact empirical-universe CSV",
    )
    parser.add_argument(
        "--background-model",
        choices=("legacy", "queue-reactive-v1"),
        default="legacy",
        help=(
            "background-flow implementation; queue-reactive-v1 requires the "
            "frozen --background-policy-csv"
        ),
    )
    parser.add_argument(
        "--background-policy-csv",
        type=Path,
        help="frozen queue-reactive symbol-to-policy mapping",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--shock-targets-dir", type=Path)
    parser.add_argument(
        "--campaign-manifest", type=Path,
        help=("immutable case-study/universe provenance JSON; its path and SHA-256 "
              "are attached to every raw row"),
    )
    parser.add_argument(
        "--asset-summary-dir",
        type=Path,
        help="write one per-asset fixed-clock summary CSV for each matrix run",
    )
    parser.add_argument("--assets", default="101")
    parser.add_argument("--ranks", default="1,2,4")
    parser.add_argument("--risk-limits", default="100")
    parser.add_argument(
        "--local-inventory-limit", type=float, default=100.0,
        help="fixed asset-specific shared-MM inventory-skew scale",
    )
    parser.add_argument(
        "--capacity-thresholds", default="0.5",
        help="comma-separated global capacity activation thresholds u_0",
    )
    parser.add_argument("--shared-mm-modes", default="on")
    parser.add_argument("--shock-modes", default="on")
    parser.add_argument(
        "--hawkes-activity-scales",
        default="0.3",
        help=(
            "comma-separated runtime Hawkes activity scales; each value is a "
            "separate deterministic scenario (default: 0.3)"
        ),
    )
    parser.add_argument(
        "--local-mm-intervals-ms",
        default="1000.0",
        help=(
            "comma-separated local market-maker refresh intervals in ms; "
            "each value is a separate scenario (default: 1000.0)"
        ),
    )
    parser.add_argument(
        "--local-mm-quantity-multipliers",
        default="1.0",
        help=(
            "comma-separated multipliers of empirical local-MM quote size; "
            "each value is a separate scenario (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--local-mm-improvement-probabilities",
        default="0.0",
        help=(
            "comma-separated probabilities that a local-MM refresh improves "
            "the current same-side BBO by one tick (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--local-mm-spread-elasticities",
        default="0.0",
        help=(
            "comma-separated non-negative spread elasticities for the local "
            "market maker (default: 0.0, the legacy constant policy)"
        ),
    )
    parser.add_argument(
        "--local-mm-max-improvement-probabilities",
        default="1.0",
        help=(
            "comma-separated caps on the spread-responsive local-MM "
            "improvement probability (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--shared-quote-quantities",
        default="200",
        help=(
            "comma-separated fixed shared-MM quote quantities; each value is "
            "a separate scenario (default: 200)"
        ),
    )
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--window-ms", type=float, default=1000.0)
    parser.add_argument(
        "--metrics-interval-ms",
        type=float,
        default=0.0,
        help=(
            "global market-monitoring cadence; zero means the decision-window "
            "cadence. This changes diagnostics only, never market evolution"
        ),
    )
    parser.add_argument("--shock-time-seconds", type=float, default=30.0)
    parser.add_argument("--shock-fraction", type=float, default=0.01)
    parser.add_argument("--shock-target-count", type=int, default=0)
    parser.add_argument("--shock-target-seed", type=int, default=20200130)
    parser.add_argument(
        "--shock-cluster-csv", type=Path,
        help="symbol,cluster_id mapping used for a stratified target mask",
    )
    parser.add_argument("--shock-quantity", type=int, default=5000)
    parser.add_argument(
        "--shock-top-depth-multiple",
        type=float,
        default=0.0,
        help=(
            "when positive, submit each sell shock as this multiple of that "
            "book's bid-side top-of-book depth immediately before the stress"
        ),
    )
    parser.add_argument(
        "--shared-quote-relative",
        action="store_true",
        help="scale shared-MM quotes by each book's empirical local quote size",
    )
    parser.add_argument(
        "--shared-quote-multiplier",
        type=float,
        default=1.0,
        help="multiplier used with --shared-quote-relative",
    )
    parser.add_argument(
        "--shared-quote-levels",
        type=int,
        default=1,
        help="number of shared-MM quote levels per side",
    )
    parser.add_argument(
        "--disable-local-mm",
        action="store_true",
        help="disable the local book-maintenance market maker",
    )
    parser.add_argument(
        "--disable-value-agent",
        action="store_true",
        help="disable the uncalibrated fundamental/value stabiliser",
    )
    parser.add_argument(
        "--value-agent-policy-csv",
        type=Path,
        help=(
            "per-symbol policy CSV generated by cluster calibration; it must "
            "cover the supplied universe exactly"
        ),
    )
    parser.add_argument(
        "--asset-summary-interval-ms",
        type=float,
        default=0.0,
        help=(
            "cadence for --asset-summary-dir; zero means the decision-window "
            "cadence"
        ),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20200130)
    parser.add_argument(
        "--seed-step",
        type=int,
        default=0,
        help=(
            "add this value to the seed after every repetition; use zero for "
            "repeat-timing tests and one for common-random-number science runs"
        ),
    )
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--bind-to", choices=["none", "core"], default="core")
    parser.add_argument("--map-by", choices=["slot", "core"], default="slot")
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "wall-clock limit for each MPI invocation; zero disables the "
            "limit (default: 0)"
        ),
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "verify and reuse completed cases in --output, then run only "
            "missing cases"
        ),
    )
    checkpoint_group.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing --output checkpoint with a fresh campaign",
    )
    args = parser.parse_args()

    assets = [0] if args.universe_config is not None else comma_ints(
        args.assets, "--assets"
    )
    ranks = comma_ints(args.ranks, "--ranks")
    risk_limits = comma_floats(args.risk_limits, "--risk-limits")
    shared_modes = shared_mode_choices(args.shared_mm_modes)
    shock_modes = comma_choices(args.shock_modes, "--shock-modes")
    capacity_thresholds = comma_floats(
        args.capacity_thresholds, "--capacity-thresholds"
    )
    hawkes_activity_scales = comma_floats(
        args.hawkes_activity_scales, "--hawkes-activity-scales"
    )
    local_mm_intervals_ms = comma_floats(
        args.local_mm_intervals_ms, "--local-mm-intervals-ms"
    )
    local_mm_quantity_multipliers = comma_floats(
        args.local_mm_quantity_multipliers,
        "--local-mm-quantity-multipliers",
    )
    local_mm_improvement_probabilities = comma_probabilities(
        args.local_mm_improvement_probabilities,
        "--local-mm-improvement-probabilities",
    )
    local_mm_spread_elasticities = comma_nonnegative_floats(
        args.local_mm_spread_elasticities,
        "--local-mm-spread-elasticities",
    )
    local_mm_max_improvement_probabilities = comma_probabilities(
        args.local_mm_max_improvement_probabilities,
        "--local-mm-max-improvement-probabilities",
    )
    if max(local_mm_improvement_probabilities) > min(
            local_mm_max_improvement_probabilities):
        raise SystemExit(
            "every local-MM base improvement probability must be no greater "
            "than every requested maximum improvement probability"
        )
    shared_quote_quantities = comma_ints(
        args.shared_quote_quantities, "--shared-quote-quantities"
    )
    # In relative mode C++ intentionally ignores the fixed share quantity and
    # scales every book from its empirical local quote size instead.  Running
    # more than one fixed quantity there would create duplicate effective
    # scenarios that look like a shared-liquidity sensitivity in the CSV.
    if args.shared_quote_relative and len(shared_quote_quantities) != 1:
        raise SystemExit(
            "shared-quote-relative cannot be combined with multiple "
            "shared-quote-quantities; use one quantity or run fixed mode"
        )
    if "off" in shared_modes and len(shared_quote_quantities) != 1:
        raise SystemExit(
            "a shared-quote-quantity sweep cannot include shared-mm=off; "
            "shared-mm=off ignores quote quantity and would duplicate cases"
        )
    if args.disable_local_mm and (
        len(local_mm_intervals_ms) != 1
        or len(local_mm_quantity_multipliers) != 1
        or len(local_mm_improvement_probabilities) != 1
        or local_mm_spread_elasticities != [0.0]
        or local_mm_max_improvement_probabilities != [1.0]
    ):
        raise SystemExit(
            "a local-MM control sweep or adaptive policy requires local MM "
            "enabled; --disable-local-mm accepts only adaptive defaults 0/1"
        )
    if args.repetitions <= 0 or args.duration_seconds <= 0:
        raise SystemExit("repetitions and duration must be positive")
    if (args.background_model == "queue-reactive-v1") \
            != (args.background_policy_csv is not None):
        raise SystemExit(
            "queue-reactive-v1 requires --background-policy-csv, while legacy "
            "mode forbids it"
        )
    if args.background_model == "queue-reactive-v1" and (
        len(hawkes_activity_scales) != 1
        or not math.isclose(
            hawkes_activity_scales[0], 0.30, rel_tol=0.0, abs_tol=1.0e-12,
        )
    ):
        raise SystemExit(
            "queue-reactive-v1 freezes activity scale inside its fitted policy; "
            "--hawkes-activity-scales must be the single compatibility value 0.30"
        )
    if not math.isfinite(args.window_ms) or args.window_ms <= 0.0:
        raise SystemExit("window-ms must be finite and positive")
    if args.seed_step < 0:
        raise SystemExit("seed-step must be non-negative")
    if (not math.isfinite(args.run_timeout_seconds)
            or args.run_timeout_seconds < 0.0):
        raise SystemExit("run-timeout-seconds must be finite and non-negative")
    if (not math.isfinite(args.local_inventory_limit)
            or args.local_inventory_limit <= 0.0):
        raise SystemExit("local-inventory-limit must be finite and positive")
    if any(value >= 1.0 for value in capacity_thresholds):
        raise SystemExit("capacity-thresholds must be below one")
    if args.shock_target_count < 0 or args.shock_target_seed < 0:
        raise SystemExit("shock target count and seed must be non-negative")
    if (not math.isfinite(args.shock_top_depth_multiple)
            or args.shock_top_depth_multiple < 0.0):
        raise SystemExit("shock-top-depth-multiple must be finite and non-negative")
    if (not math.isfinite(args.shared_quote_multiplier)
            or args.shared_quote_multiplier <= 0.0):
        raise SystemExit("shared-quote-multiplier must be finite and positive")
    if args.shared_quote_levels <= 0:
        raise SystemExit("shared-quote-levels must be positive")
    if (not math.isfinite(args.asset_summary_interval_ms)
            or args.asset_summary_interval_ms < 0.0):
        raise SystemExit("asset-summary-interval-ms must be finite and non-negative")
    if (not math.isfinite(args.metrics_interval_ms)
            or args.metrics_interval_ms < 0.0):
        raise SystemExit("metrics-interval-ms must be finite and non-negative")
    if args.metrics_interval_ms > 0.0:
        window_ns = round(args.window_ms * 1_000_000.0)
        metrics_ns = round(args.metrics_interval_ms * 1_000_000.0)
        if metrics_ns < window_ns or metrics_ns % window_ns != 0:
            raise SystemExit(
                "metrics-interval-ms must be an exact integer multiple of "
                "window-ms"
            )
    if args.shock_time_seconds >= args.duration_seconds and "on" in shock_modes:
        raise SystemExit("shock time must be less than duration")

    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    executable_path = args.executable.resolve()
    if not executable_path.is_file():
        raise SystemExit(f"executable is not a regular file: {executable_path}")
    executable_sha256 = sha256_file(executable_path)
    config_path = (args.universe_config or args.base_config).resolve()
    if not config_path.is_file():
        raise SystemExit(f"input config is not a regular file: {config_path}")
    config_sha256 = sha256_file(config_path)
    background_policy_path: Path | None = None
    background_policy_sha256 = ""
    background_artifacts_sha256 = ""
    background_artifact_count = 0
    if args.background_policy_csv is not None:
        background_policy_path = args.background_policy_csv.resolve()
        if not background_policy_path.is_file():
            raise SystemExit(
                "background-policy-csv is not a regular file: "
                f"{background_policy_path}"
            )
        background_policy_sha256 = sha256_file(background_policy_path)
        (
            background_artifacts_sha256,
            background_artifact_count,
        ) = background_artifact_manifest(background_policy_path)
    policy_path: Path | None = None
    policy_sha256 = ""
    if args.value_agent_policy_csv is not None:
        policy_path = args.value_agent_policy_csv.resolve()
        if not policy_path.is_file():
            raise SystemExit(
                f"value-agent-policy-csv is not a regular file: {policy_path}"
            )
        policy_sha256 = sha256_file(policy_path)
    cluster_path: Path | None = None
    cluster_sha256 = ""
    if args.shock_cluster_csv is not None:
        cluster_path = args.shock_cluster_csv.resolve()
        if not cluster_path.is_file():
            raise SystemExit(f"shock-cluster-csv is not a regular file: {cluster_path}")
        cluster_sha256 = sha256_file(cluster_path)
    campaign_manifest_path: Path | None = None
    campaign_manifest_sha256 = ""
    if args.campaign_manifest is not None:
        campaign_manifest_path = args.campaign_manifest.resolve()
        if not campaign_manifest_path.is_file():
            raise SystemExit(
                f"campaign-manifest is not a regular file: {campaign_manifest_path}"
            )
        campaign_manifest_sha256 = sha256_file(campaign_manifest_path)

    # The campaign digest deliberately excludes operational choices that may
    # change on a retry (timeout, --resume, output/summary filenames). Every
    # option that can alter a simulation or a side-artifact path remains in it.
    excluded_campaign_arguments = {
        "output", "summary", "resume", "overwrite", "run_timeout_seconds",
    }
    campaign_arguments = {
        name: canonical_argument(value)
        for name, value in vars(args).items()
        if name not in excluded_campaign_arguments
    }
    campaign_sha256 = stable_sha256(
        {
            "schema": 1,
            "arguments": campaign_arguments,
            "executable_sha256": executable_sha256,
            "input_config_sha256": config_sha256,
            "background_model": args.background_model,
            "background_policy_sha256": background_policy_sha256,
            "background_artifacts_sha256": background_artifacts_sha256,
            "value_agent_policy_sha256": policy_sha256,
            "shock_cluster_sha256": cluster_sha256,
            "campaign_manifest_sha256": campaign_manifest_sha256,
        }
    )

    summary_path = args.summary or args.output.with_name(
        args.output.stem + "_summary.csv"
    )
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif args.output.exists() and not args.resume:
        raise SystemExit(
            f"output checkpoint already exists: {args.output}; "
            "use --resume or --overwrite"
        )

    rows = read_checkpoint(args.output) if args.resume and args.output.exists() else []
    for row in rows:
        if row.get("campaign_sha256") != campaign_sha256:
            raise RuntimeError(
                "resume checkpoint belongs to a different command, input, "
                "executable, or side-artifact campaign"
            )
    resume_by_key = {row["run_key"]: row for row in rows}
    planned_run_keys: set[str] = set()
    for (
        planned_asset_count,
        planned_risk_limit,
        planned_shared_mode,
        planned_shock_mode,
        planned_capacity_threshold,
        planned_hawkes_scale,
        planned_local_interval,
        planned_local_multiplier,
        planned_local_improvement_probability,
        planned_local_spread_elasticity,
        planned_local_max_improvement_probability,
        planned_shared_quantity,
        planned_repetition,
        planned_rank_count,
    ) in product(
        assets,
        risk_limits,
        shared_modes,
        shock_modes,
        capacity_thresholds,
        hawkes_activity_scales,
        local_mm_intervals_ms,
        local_mm_quantity_multipliers,
        local_mm_improvement_probabilities,
        local_mm_spread_elasticities,
        local_mm_max_improvement_probabilities,
        shared_quote_quantities,
        range(1, args.repetitions + 1),
        ranks,
    ):
        planned_controls = control_values(
            planned_hawkes_scale,
            planned_local_interval,
            planned_local_multiplier,
            planned_local_improvement_probability,
            planned_local_spread_elasticity,
            planned_local_max_improvement_probability,
            planned_shared_quantity,
        )
        planned_seed = args.seed + (planned_repetition - 1) * args.seed_step
        planned_key = make_run_key(
            campaign_sha256,
            asset_count=planned_asset_count,
            rank_count=planned_rank_count,
            risk_limit=planned_risk_limit,
            shared_mode=planned_shared_mode,
            shock_mode=planned_shock_mode,
            capacity_threshold=planned_capacity_threshold,
            controls=planned_controls,
            repetition=planned_repetition,
            seed=planned_seed,
        )
        if planned_key in planned_run_keys:
            raise SystemExit(
                "experiment matrix contains duplicate effective cases; remove "
                "duplicate values from comma-separated arguments"
            )
        planned_run_keys.add(planned_key)
    unplanned_checkpoint_keys = set(resume_by_key).difference(planned_run_keys)
    if unplanned_checkpoint_keys:
        raise RuntimeError(
            "resume checkpoint contains run keys outside the planned campaign"
        )
    if set(resume_by_key) != planned_run_keys:
        # A summary is valid only for a complete matrix. Remove a stale file
        # before a missing run can fail or time out again.
        summary_path.unlink(missing_ok=True)
    # Validate generic result fields and every side artifact before spending
    # time on any missing case. Exact scenario fields are checked in-loop.
    for run_key, row in resume_by_key.items():
        validate_resumed_row(
            row,
            campaign_sha256=campaign_sha256,
            run_key=run_key,
            expected={},
        )
    unused_resume_keys = set(resume_by_key)
    ordered_rows: list[dict[str, str]] = []
    expected_hash: dict[tuple[object, ...], str] = {}
    timeout_seconds = (
        args.run_timeout_seconds if args.run_timeout_seconds > 0.0 else None
    )

    for artifact_dir in (
        args.metrics_dir, args.shock_targets_dir, args.asset_summary_dir,
    ):
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)

    for asset_count in assets:
        for risk_limit in risk_limits:
            for shared_mode in shared_modes:
                for (
                    shock_mode,
                    capacity_threshold,
                    hawkes_activity_scale,
                    local_mm_interval_ms,
                    local_mm_quantity_multiplier,
                    local_mm_improvement_probability,
                    local_mm_spread_elasticity,
                    local_mm_max_improvement_probability,
                    shared_quote_quantity,
                ) in product(
                    shock_modes,
                    capacity_thresholds,
                    hawkes_activity_scales,
                    local_mm_intervals_ms,
                    local_mm_quantity_multipliers,
                    local_mm_improvement_probabilities,
                    local_mm_spread_elasticities,
                    local_mm_max_improvement_probabilities,
                    shared_quote_quantities,
                ):
                    controls = control_values(
                        hawkes_activity_scale,
                        local_mm_interval_ms,
                        local_mm_quantity_multiplier,
                        local_mm_improvement_probability,
                        local_mm_spread_elasticity,
                        local_mm_max_improvement_probability,
                        shared_quote_quantity,
                    )
                    control_label = (
                        f"hawkes{float_label(hawkes_activity_scale)}_"
                        f"localms{float_label(local_mm_interval_ms)}_"
                        f"localq{float_label(local_mm_quantity_multiplier)}_"
                        f"localp{float_label(local_mm_improvement_probability)}_"
                        f"locale{float_label(local_mm_spread_elasticity)}_"
                        f"localcap{float_label(local_mm_max_improvement_probability)}_"
                        f"sharedq{shared_quote_quantity}"
                    )
                    case = (
                        asset_count,
                        risk_limit,
                        shared_mode,
                        shock_mode,
                        capacity_threshold,
                        controls["hawkes_activity_scale"],
                        controls["local_mm_interval_ms"],
                        controls["local_mm_quantity_multiplier"],
                        controls["local_mm_improvement_probability"],
                        controls["local_mm_spread_elasticity"],
                        controls["local_mm_max_improvement_probability"],
                        controls["shared_quote_quantity"],
                    )
                    for repetition in range(1, args.repetitions + 1):
                        run_seed = args.seed + (repetition - 1) * args.seed_step
                        for rank_count in ranks:
                            command = [args.mpirun]
                            if args.oversubscribe:
                                command.append("--oversubscribe")
                            command.extend(
                                [
                                    "--bind-to",
                                    args.bind_to,
                                    "--map-by",
                                    args.map_by,
                                    "-x",
                                    "PATH",
                                    "-x",
                                    "LD_LIBRARY_PATH",
                                    "-np",
                                    str(rank_count),
                                ]
                            )
                            command.append(str(executable_path))
                            command.extend([
                                "--background-model", args.background_model,
                            ])
                            if background_policy_path is not None:
                                command.extend([
                                    "--background-policy-csv",
                                    str(background_policy_path),
                                ])
                            if args.universe_config is not None:
                                command.extend([
                                    "--universe-config",
                                    str(config_path),
                                ])
                            else:
                                command.extend([
                                    "--base-config",
                                    str(config_path),
                                    "--assets",
                                    str(asset_count),
                                ])
                            command.extend(
                                [
                                    "--duration-seconds",
                                    str(args.duration_seconds),
                                    "--window-ms",
                                    str(args.window_ms),
                                    "--metrics-interval-ms",
                                    str(args.metrics_interval_ms),
                                    "--seed",
                                    str(run_seed),
                                    "--risk-limit-per-asset",
                                    str(risk_limit),
                                    "--local-inventory-limit",
                                    str(args.local_inventory_limit),
                                    "--capacity-threshold",
                                    str(capacity_threshold),
                                    "--hawkes-activity-scale",
                                    controls["hawkes_activity_scale"],
                                    "--local-mm-interval-ms",
                                    controls["local_mm_interval_ms"],
                                    "--local-mm-quantity-multiplier",
                                    controls["local_mm_quantity_multiplier"],
                                    "--local-mm-improvement-probability",
                                    controls["local_mm_improvement_probability"],
                                    "--local-mm-spread-elasticity",
                                    controls["local_mm_spread_elasticity"],
                                    "--local-mm-max-improvement-probability",
                                    controls[
                                        "local_mm_max_improvement_probability"
                                    ],
                                    "--shared-quote-quantity",
                                    controls["shared_quote_quantity"],
                                    "--shared-quote-levels",
                                    str(args.shared_quote_levels),
                                    "--shock-time-seconds",
                                    str(args.shock_time_seconds),
                                    "--shock-fraction",
                                    str(args.shock_fraction),
                                    "--shock-target-count",
                                    str(args.shock_target_count),
                                    "--shock-target-seed",
                                    str(args.shock_target_seed),
                                    "--shock-quantity",
                                    str(args.shock_quantity),
                                    "--shock-top-depth-multiple",
                                    str(args.shock_top_depth_multiple),
                                ]
                            )
                            if args.shared_quote_relative:
                                command.append("--shared-quote-relative")
                                command.extend([
                                    "--shared-quote-multiplier",
                                    str(args.shared_quote_multiplier),
                                ])
                            if args.disable_local_mm:
                                command.append("--disable-local-mm")
                            if args.disable_value_agent:
                                command.append("--disable-value-agent")
                            if policy_path is not None:
                                command.extend([
                                    "--value-agent-policy-csv",
                                    str(policy_path),
                                ])
                            if cluster_path is not None:
                                command.extend([
                                    "--shock-cluster-csv", str(cluster_path),
                                ])
                            if shared_mode == "off":
                                command.append("--disable-shared-mm")
                            elif shared_mode == "uncoupled":
                                command.append("--uncoupled-shared-mm")
                            if shock_mode == "on":
                                command.append("--shock")
                            metrics_path: Path | None = None
                            if args.metrics_dir is not None:
                                asset_label = (
                                    "universe" if args.universe_config is not None
                                    else f"a{asset_count}"
                                )
                                metrics_name = (
                                    f"{asset_label}_r{rank_count}_"
                                    f"risk{float_label(risk_limit)}_"
                                    f"u{float_label(capacity_threshold)}_"
                                    f"mm{shared_mode}_shock{shock_mode}_"
                                    f"{control_label}_"
                                    f"rep{repetition}.csv"
                                )
                                metrics_path = (args.metrics_dir / metrics_name).resolve()
                                command.extend(["--metrics-csv", str(metrics_path)])
                            targets_path: Path | None = None
                            if args.shock_targets_dir is not None:
                                asset_label = (
                                    "universe" if args.universe_config is not None
                                    else f"a{asset_count}"
                                )
                                targets_name = (
                                    f"{asset_label}_r{rank_count}_"
                                    f"risk{float_label(risk_limit)}_"
                                    f"u{float_label(capacity_threshold)}_"
                                    f"mm{shared_mode}_shock{shock_mode}_"
                                    f"{control_label}_"
                                    f"rep{repetition}.csv"
                                )
                                targets_path = (
                                    args.shock_targets_dir / targets_name
                                ).resolve()
                                command.extend([
                                    "--shock-targets-csv", str(targets_path),
                                ])
                            asset_summary_path: Path | None = None
                            if args.asset_summary_dir is not None:
                                asset_label = (
                                    "universe" if args.universe_config is not None
                                    else f"a{asset_count}"
                                )
                                summary_name = (
                                    f"{asset_label}_r{rank_count}_"
                                    f"risk{float_label(risk_limit)}_"
                                    f"u{float_label(capacity_threshold)}_"
                                    f"mm{shared_mode}_shock{shock_mode}_"
                                    f"{control_label}_"
                                    f"rep{repetition}.csv"
                                )
                                asset_summary_path = (
                                    args.asset_summary_dir / summary_name
                                ).resolve()
                                command.extend([
                                    "--asset-summary-csv",
                                    str(asset_summary_path),
                                ])
                                if args.asset_summary_interval_ms > 0.0:
                                    command.extend([
                                        "--asset-summary-interval-ms",
                                        str(args.asset_summary_interval_ms),
                                    ])

                            input_mode = (
                                "empirical_universe"
                                if args.universe_config is not None
                                else "synthetic_templates"
                            )
                            run_key = make_run_key(
                                campaign_sha256,
                                asset_count=asset_count,
                                rank_count=rank_count,
                                risk_limit=risk_limit,
                                shared_mode=shared_mode,
                                shock_mode=shock_mode,
                                capacity_threshold=capacity_threshold,
                                controls=controls,
                                repetition=repetition,
                                seed=run_seed,
                            )
                            expected_resume_fields = {
                                "ranks": str(rank_count),
                                "repetition": str(repetition),
                                "seed": str(run_seed),
                                "shared_mm_mode": shared_mode,
                                "shock_mode": shock_mode,
                                "input_mode": input_mode,
                                "input_config_sha256": config_sha256,
                                "campaign_manifest_sha256": campaign_manifest_sha256,
                                "requested_risk_limit_per_asset": repr(risk_limit),
                                "requested_capacity_threshold": str(
                                    capacity_threshold
                                ),
                                "requested_metrics_interval_ms": str(
                                    args.metrics_interval_ms
                                ),
                                "control_scenario": control_label,
                                "metrics_csv": (
                                    str(metrics_path) if metrics_path is not None else ""
                                ),
                                "shock_targets_csv": (
                                    str(targets_path) if targets_path is not None else ""
                                ),
                                "asset_summary_csv": (
                                    str(asset_summary_path)
                                    if asset_summary_path is not None else ""
                                ),
                                "background_model": args.background_model,
                                "background_policy_csv": (
                                    str(background_policy_path)
                                    if background_policy_path is not None else ""
                                ),
                                "background_policy_sha256": background_policy_sha256,
                                "background_artifacts_sha256": background_artifacts_sha256,
                                "background_artifact_count": str(
                                    background_artifact_count
                                ),
                            }
                            expected_resume_fields.update(controls)
                            if args.universe_config is None:
                                expected_resume_fields["assets"] = str(asset_count)

                            resumed = run_key in resume_by_key
                            if resumed:
                                result = dict(resume_by_key[run_key])
                                validate_resumed_row(
                                    result,
                                    campaign_sha256=campaign_sha256,
                                    run_key=run_key,
                                    expected=expected_resume_fields,
                                )
                                unused_resume_keys.discard(run_key)
                            else:
                                try:
                                    completed = run_mpi_command(
                                        command,
                                        environment=environment,
                                        timeout_seconds=timeout_seconds,
                                    )
                                except MpiRunTimeout as error:
                                    print(
                                        "MPI RUN TIMED OUT\n"
                                        f"command: {shlex.join(command)}\n"
                                        f"timeout seconds: {error.timeout_seconds:g}\n"
                                        "--- stdout ---\n"
                                        f"{error.stdout or '<empty>'}\n"
                                        "--- stderr ---\n"
                                        f"{error.stderr or '<empty>'}\n"
                                        f"checkpoint: {args.output}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    raise RuntimeError(
                                        "MPI run timed out; the launcher process group "
                                        "was terminated and completed cases remain "
                                        "checkpointed"
                                    ) from error
                                except subprocess.CalledProcessError as error:
                                    print(
                                        "MPI LAUNCH FAILED\n"
                                        f"command: {shlex.join(command)}\n"
                                        f"return code: {error.returncode}\n"
                                        "--- stdout ---\n"
                                        f"{error.stdout or '<empty>'}\n"
                                        "--- stderr ---\n"
                                        f"{error.stderr or '<empty>'}\n"
                                        f"checkpoint: {args.output}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    raise RuntimeError(
                                        "MPI launcher failed; full diagnostics are above"
                                    ) from error

                                result = parse_result(completed.stdout)
                                try:
                                    reported_risk_limit = float(
                                        result["risk_limit_per_asset"]
                                    )
                                except ValueError as error:
                                    raise RuntimeError(
                                        "simulator returned a non-numeric "
                                        "risk_limit_per_asset"
                                    ) from error
                                if not math.isclose(
                                    reported_risk_limit,
                                    risk_limit,
                                    rel_tol=1.0e-12,
                                    abs_tol=1.0e-12,
                                ):
                                    raise RuntimeError(
                                        "simulator risk limit does not match the "
                                        f"requested value: {reported_risk_limit} != "
                                        f"{risk_limit}"
                                    )
                                if "capacity_threshold" in result:
                                    try:
                                        reported_capacity = float(
                                            result["capacity_threshold"]
                                        )
                                    except ValueError as error:
                                        raise RuntimeError(
                                            "simulator returned a non-numeric "
                                            "capacity_threshold"
                                        ) from error
                                    if not math.isclose(
                                        reported_capacity,
                                        capacity_threshold,
                                        rel_tol=1.0e-12,
                                        abs_tol=1.0e-12,
                                    ):
                                        raise RuntimeError(
                                            "simulator capacity threshold does not "
                                            "match the requested value"
                                        )
                                for field, expected in controls.items():
                                    verify_or_record_control(result, field, expected)
                                verify_or_record_exact(
                                    result, "background_model", args.background_model
                                )
                                expected_background_echo = (
                                    str(background_policy_path)
                                    if background_policy_path is not None else "none"
                                )
                                verify_or_record_exact(
                                    result,
                                    "background_policy",
                                    expected_background_echo,
                                )
                                result["campaign_sha256"] = campaign_sha256
                                result["run_key"] = run_key
                                result["executable"] = str(executable_path)
                                result["executable_sha256"] = executable_sha256
                                result["runner_hostname"] = os.uname().nodename
                                result["slurm_job_id"] = environment.get(
                                    "SLURM_JOB_ID", ""
                                )
                                result["slurm_job_nodelist"] = environment.get(
                                    "SLURM_JOB_NODELIST", ""
                                )
                                result["slurm_job_partition"] = environment.get(
                                    "SLURM_JOB_PARTITION", ""
                                )
                                result["control_scenario"] = control_label
                                result["repetition"] = str(repetition)
                                result["seed"] = str(run_seed)
                                result["shared_mm_mode"] = shared_mode
                                result["shock_mode"] = shock_mode
                                result["input_config"] = str(config_path)
                                result["input_config_sha256"] = config_sha256
                                result["input_mode"] = input_mode
                                result["background_policy_csv"] = (
                                    str(background_policy_path)
                                    if background_policy_path is not None else ""
                                )
                                result["background_policy_sha256"] = (
                                    background_policy_sha256
                                )
                                result["background_artifacts_sha256"] = (
                                    background_artifacts_sha256
                                )
                                result["background_artifact_count"] = str(
                                    background_artifact_count
                                )
                                result["campaign_manifest"] = (
                                    str(campaign_manifest_path)
                                    if campaign_manifest_path is not None else ""
                                )
                                result["campaign_manifest_sha256"] = (
                                    campaign_manifest_sha256
                                )
                                result["requested_risk_limit_per_asset"] = repr(
                                    risk_limit
                                )
                                result["requested_window_ms"] = str(args.window_ms)
                                result["requested_metrics_interval_ms"] = str(
                                    args.metrics_interval_ms
                                )
                                result["requested_shock_time_seconds"] = str(
                                    args.shock_time_seconds
                                )
                                result["requested_shock_fraction"] = str(
                                    args.shock_fraction
                                )
                                result["requested_shock_target_count"] = str(
                                    args.shock_target_count
                                )
                                result["requested_shock_target_seed"] = str(
                                    args.shock_target_seed
                                )
                                result["requested_local_inventory_limit"] = str(
                                    args.local_inventory_limit
                                )
                                result["requested_capacity_threshold"] = str(
                                    capacity_threshold
                                )
                                result["shock_cluster_csv"] = (
                                    str(cluster_path)
                                    if cluster_path is not None else ""
                                )
                                result["shock_cluster_sha256"] = cluster_sha256
                                result["requested_shock_top_depth_multiple"] = str(
                                    args.shock_top_depth_multiple
                                )
                                result["requested_shared_quote_relative"] = str(
                                    int(args.shared_quote_relative)
                                )
                                result["requested_shared_quote_multiplier"] = str(
                                    args.shared_quote_multiplier
                                )
                                result["requested_shared_quote_levels"] = str(
                                    args.shared_quote_levels
                                )
                                result["requested_local_mm_enabled"] = str(
                                    int(not args.disable_local_mm)
                                )
                                result["requested_value_agent_enabled"] = str(
                                    int(not args.disable_value_agent)
                                )
                                result["value_agent_policy_csv"] = (
                                    str(policy_path)
                                    if policy_path is not None else ""
                                )
                                result["value_agent_policy_sha256"] = policy_sha256
                                result["requested_asset_summary_interval_ms"] = str(
                                    args.asset_summary_interval_ms
                                )
                                for field, path in (
                                    ("metrics_csv", metrics_path),
                                    ("shock_targets_csv", targets_path),
                                    ("asset_summary_csv", asset_summary_path),
                                ):
                                    if path is None:
                                        result[field] = ""
                                        result[f"{field}_sha256"] = ""
                                    else:
                                        if not path.is_file():
                                            raise RuntimeError(
                                                "simulator completed successfully but "
                                                f"did not create {field}: {path}"
                                            )
                                        result[field] = str(path)
                                        result[f"{field}_sha256"] = sha256_file(path)
                                validate_resumed_row(
                                    result,
                                    campaign_sha256=campaign_sha256,
                                    run_key=run_key,
                                    expected=expected_resume_fields,
                                )

                            hash_case = (*case, run_seed)
                            if hash_case not in expected_hash:
                                expected_hash[hash_case] = result["state_hash"]
                            elif result["state_hash"] != expected_hash[hash_case]:
                                raise RuntimeError(
                                    "rank/repetition invariance failed for "
                                    f"case={hash_case}: expected "
                                    f"{expected_hash[hash_case]}, "
                                    f"observed {result['state_hash']} at "
                                    f"ranks={rank_count}, repetition={repetition}"
                                )
                            ordered_rows.append(result)
                            if not resumed:
                                rows.append(result)
                                # Persist every completed run before launching the
                                # next potentially long or failure-prone MPI case.
                                write_csv(args.output, rows)
                            print(
                                f"{'resumed ' if resumed else ''}"
                                f"assets={result['assets']} ranks={rank_count} "
                                f"risk={risk_limit:g} "
                                f"mm={shared_mode} shock={shock_mode} "
                                f"{control_label} "
                                f"rep={repetition} wall={result['wall_seconds']} "
                                f"hash={result['state_hash']}",
                                flush=True,
                            )

    if unused_resume_keys:
        raise RuntimeError(
            "resume checkpoint contains completed cases that were not consumed"
        )
    if len(ordered_rows) != len(planned_run_keys):
        raise RuntimeError(
            "internal error: completed row count does not match planned matrix"
        )
    rows = ordered_rows
    write_csv(args.output, rows)

    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    grouped_rows: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("background_model", "legacy"),
            row.get("background_policy_sha256", ""),
            row.get("background_artifacts_sha256", ""),
            row["assets"],
            row["ranks"],
            row["risk_limit_per_asset"],
            row.get("capacity_threshold", row["requested_capacity_threshold"]),
            row["shared_mm_mode"],
            row["shock_mode"],
            row["hawkes_activity_scale"],
            row["local_mm_interval_ms"],
            row["local_mm_quantity_multiplier"],
            row["local_mm_improvement_probability"],
            row["local_mm_spread_elasticity"],
            row["local_mm_max_improvement_probability"],
            row["shared_quote_quantity"],
        )
        grouped[key].append(float(row["wall_seconds"]))
        grouped_rows[key].append(row)
    medians = {key: statistics.median(values) for key, values in grouped.items()}
    summary_rows: list[dict[str, str]] = []
    for key, median_wall in sorted(medians.items()):
        (
            background_model,
            background_policy_sha256,
            background_artifacts_sha256,
            asset,
            rank,
            risk,
            capacity_threshold,
            shared_mode,
            shock_mode,
            hawkes_activity_scale,
            local_mm_interval_ms,
            local_mm_quantity_multiplier,
            local_mm_improvement_probability,
            local_mm_spread_elasticity,
            local_mm_max_improvement_probability,
            shared_quote_quantity,
        ) = key
        baseline_key = (
            background_model,
            background_policy_sha256,
            background_artifacts_sha256,
            asset,
            "1",
            risk,
            capacity_threshold,
            shared_mode,
            shock_mode,
            hawkes_activity_scale,
            local_mm_interval_ms,
            local_mm_quantity_multiplier,
            local_mm_improvement_probability,
            local_mm_spread_elasticity,
            local_mm_max_improvement_probability,
            shared_quote_quantity,
        )
        baseline = medians.get(baseline_key)
        speedup = baseline / median_wall if baseline is not None else float("nan")
        efficiency = speedup / int(rank) if baseline is not None else float("nan")
        case_rows = grouped_rows[key]
        wall_values = grouped[key]
        def median_field(field: str) -> float:
            values = [float(row[field]) for row in case_rows if row.get(field, "")]
            return statistics.median(values) if values else float("nan")
        processed_orders = median_field("processed_orders")
        summary_rows.append(
            {
                "background_model": background_model,
                "background_policy_sha256": background_policy_sha256,
                "background_artifacts_sha256": background_artifacts_sha256,
                "background_artifact_count": case_rows[0].get(
                    "background_artifact_count", "0"
                ),
                "assets": asset,
                "ranks": rank,
                "risk_limit_per_asset": risk,
                "capacity_threshold": capacity_threshold,
                "shared_mm_mode": shared_mode,
                "shock_mode": shock_mode,
                "hawkes_activity_scale": hawkes_activity_scale,
                "local_mm_interval_ms": local_mm_interval_ms,
                "local_mm_quantity_multiplier": local_mm_quantity_multiplier,
                "local_mm_improvement_probability": (
                    local_mm_improvement_probability
                ),
                "local_mm_spread_elasticity": local_mm_spread_elasticity,
                "local_mm_max_improvement_probability": (
                    local_mm_max_improvement_probability
                ),
                "shared_quote_quantity": shared_quote_quantity,
                "median_wall_seconds": f"{median_wall:.9f}",
                "minimum_wall_seconds": f"{min(wall_values):.9f}",
                "maximum_wall_seconds": f"{max(wall_values):.9f}",
                "speedup_vs_rank1": f"{speedup:.9f}",
                "parallel_efficiency": f"{efficiency:.9f}",
                "median_orders_per_wall_second": (
                    f"{processed_orders / median_wall:.9f}"
                    if math.isfinite(processed_orders) else "nan"
                ),
                "median_communication_fraction": f"{median_field('communication_fraction'):.9f}",
                "median_min_compute_seconds": f"{median_field('min_compute_seconds'):.9f}",
                "median_mean_compute_seconds": f"{median_field('mean_compute_seconds'):.9f}",
                "median_max_compute_seconds": f"{median_field('max_compute_seconds'):.9f}",
                "median_compute_imbalance": f"{median_field('compute_imbalance'):.9f}",
                "median_min_orders_per_rank": f"{median_field('min_orders_per_rank'):.9f}",
                "median_mean_orders_per_rank": f"{median_field('mean_orders_per_rank'):.9f}",
                "median_max_orders_per_rank": f"{median_field('max_orders_per_rank'):.9f}",
                "median_min_books_per_rank": f"{median_field('min_books_per_rank'):.9f}",
                "median_mean_books_per_rank": f"{median_field('mean_books_per_rank'):.9f}",
                "median_max_books_per_rank": f"{median_field('max_books_per_rank'):.9f}",
            }
        )
    write_csv(summary_path, summary_rows)
    print(f"raw_output={args.output}")
    print(f"summary_output={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
