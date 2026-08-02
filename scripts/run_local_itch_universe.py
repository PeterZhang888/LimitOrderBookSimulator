#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Build one resumable, all-symbol empirical ITCH universe on a workstation.

This is the local counterpart of ``submit_itch50_universe_calibration.sh``.
It deliberately runs no MPI and never copies the multi-gigabyte raw archive.
Successful extraction batches receive content-bound completion records, so an
interrupted workstation run can be restarted with the same command.  The
optional fixed-cohort mode additionally binds each result to the raw cohort
declaration and to its canonical ordered-symbol hash.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from datetime import date
from typing import Any, Sequence


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(value)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_checked(command: Sequence[str], *, stdout_path: pathlib.Path | None = None,
                stderr_path: pathlib.Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        list(command), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if stdout_path is not None:
        atomic_text(stdout_path, completed.stdout)
    if stderr_path is not None:
        atomic_text(stderr_path, completed.stderr)
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"command returned {completed.returncode}: {' '.join(command)}\n{diagnostic}"
        )
    return completed


def read_symbols(path: pathlib.Path) -> list[str]:
    symbols = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError(f"invalid or duplicate symbol list: {path}")
    if "QQQ" not in symbols:
        raise ValueError(f"QQQ is absent from selected universe: {path}")
    return symbols


def requested_selection(
    fixed_symbols: pathlib.Path | None, max_symbols: int | None,
) -> dict[str, Any]:
    """Describe the caller's selection request before any result is reused."""

    if fixed_symbols is not None and max_symbols is not None:
        raise ValueError("--fixed-symbols and --max-symbols are mutually exclusive")
    if fixed_symbols is not None:
        if not fixed_symbols.is_file():
            raise ValueError(
                f"--fixed-symbols is not a readable regular file: {fixed_symbols}"
            )
        return {
            "mode": "fixed_symbols",
            "fixed_symbols_input": {
                "path": str(fixed_symbols),
                "raw_sha256": sha256_file(fixed_symbols),
            },
            "max_symbols": None,
        }
    if max_symbols is not None:
        return {
            "mode": "eligible_capped",
            "fixed_symbols_input": None,
            "max_symbols": max_symbols,
        }
    return {
        "mode": "all_eligible",
        "fixed_symbols_input": None,
        "max_symbols": None,
    }


def validate_selection_manifest(
    manifest_path: pathlib.Path,
    symbols_path: pathlib.Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Validate and return the immutable selection record for batching."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid selection provenance {manifest_path}: {error}") from error
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported selection provenance schema: {manifest_path}")
    for field in ("mode", "max_symbols"):
        if manifest.get(field) != request[field]:
            raise ValueError(
                f"selection provenance {field} differs from this request: "
                f"{manifest.get(field)!r} != {request[field]!r}; use a fresh --result-root"
            )

    observed_fixed = manifest.get("fixed_symbols_input")
    requested_fixed = request["fixed_symbols_input"]
    if requested_fixed is None:
        if observed_fixed is not None:
            raise ValueError(
                f"selection provenance unexpectedly binds a fixed cohort: {manifest_path}"
            )
    else:
        if not isinstance(observed_fixed, dict):
            raise ValueError(f"selection provenance omits fixed cohort: {manifest_path}")
        for field in ("path", "raw_sha256"):
            if observed_fixed.get(field) != requested_fixed[field]:
                raise ValueError(
                    f"fixed-symbol {field} differs from existing selection provenance; "
                    "use a fresh --result-root"
                )

    selected = manifest.get("selected_symbols")
    if not isinstance(selected, dict):
        raise ValueError(f"selection provenance omits selected_symbols: {manifest_path}")
    symbols = read_symbols(symbols_path)
    if selected.get("ordered_symbols") != symbols:
        raise ValueError(
            f"selected symbol file differs from provenance: {symbols_path}"
        )
    if selected.get("count") != len(symbols):
        raise ValueError(f"selected symbol count differs from provenance: {symbols_path}")
    rendered_sha256 = hashlib.sha256(
        "".join(f"{symbol}\n" for symbol in symbols).encode("utf-8")
    ).hexdigest()
    if selected.get("canonical_sha256") != rendered_sha256:
        raise ValueError(
            f"selected symbol canonical hash differs from provenance: {symbols_path}"
        )
    if sha256_file(symbols_path) != rendered_sha256:
        raise ValueError(
            f"selected symbol file is not in canonical LF-terminated form: {symbols_path}"
        )
    if request["mode"] == "fixed_symbols":
        expected_order = ["QQQ", *sorted(symbol for symbol in symbols if symbol != "QQQ")]
        if symbols != expected_order:
            raise ValueError(
                "fixed cohort is not ordered as QQQ first followed by lexicographic order"
            )
        if observed_fixed.get("normalized_count") != len(symbols):
            raise ValueError("fixed cohort normalized count differs from selected count")

    return {
        "mode": manifest["mode"],
        "max_symbols": manifest["max_symbols"],
        "fixed_symbols_input": manifest["fixed_symbols_input"],
        "selected_symbols": selected,
        "provenance": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
    }


def valid_batch_marker(
    path: pathlib.Path,
    *,
    archive_sha256: str,
    symbols_sha256: str,
    batch_name: str,
    extractor_sha256: str,
    state_targets_sha256: str | None,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "complete"
        and value.get("batch") == batch_name
        and value.get("archive_sha256") == archive_sha256
        and value.get("symbols_sha256") == symbols_sha256
        and value.get("extractor_sha256") == extractor_sha256
        and value.get("state_targets_sha256") == state_targets_sha256
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--itch", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--symbols-per-batch", type=int, default=128)
    parser.add_argument("--concurrent-batches", type=int, default=1)
    parser.add_argument("--snapshot-ms", type=int, default=1000)
    parser.add_argument(
        "--state-targets-csv",
        help=(
            "optional symbol-level depth targets forwarded unchanged to every "
            "extraction batch; its content hash is bound into the batch plan, "
            "completion markers and final metadata"
        ),
    )
    parser.add_argument("--target-window-seconds", type=int, nargs="+", default=[300, 3600])
    parser.add_argument("--progress-seconds", type=float, default=60.0)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--max-symbols", type=int)
    selection_group.add_argument(
        "--fixed-symbols",
        help=(
            "newline-delimited predeclared cohort used unchanged on every date; "
            "QQQ and startup eligibility are enforced by the selector"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        parsed_date = date.fromisoformat(args.date)
    except ValueError as error:
        raise SystemExit(f"--date must be ISO YYYY-MM-DD: {error}")
    for name in ("symbols_per_batch", "concurrent_batches", "snapshot_ms"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not args.target_window_seconds or any(value <= 0 for value in args.target_window_seconds):
        raise SystemExit("--target-window-seconds must contain positive values")
    if args.max_symbols is not None and args.max_symbols <= 0:
        raise SystemExit("--max-symbols must be positive")

    project = pathlib.Path(args.project_dir).expanduser().resolve()
    archive = pathlib.Path(args.itch).expanduser().resolve()
    root = pathlib.Path(args.result_root).expanduser().resolve()
    fixed_symbols_path = (
        None if args.fixed_symbols is None
        else pathlib.Path(args.fixed_symbols).expanduser().resolve()
    )
    state_targets_path = (
        None if args.state_targets_csv is None
        else pathlib.Path(args.state_targets_csv).expanduser().resolve()
    )
    try:
        selection_request = requested_selection(
            fixed_symbols_path, args.max_symbols
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not archive.is_file():
        raise SystemExit(f"ITCH archive is not a regular file: {archive}")
    if state_targets_path is not None and not state_targets_path.is_file():
        raise SystemExit(
            f"state-target CSV is not a readable regular file: {state_targets_path}"
        )
    scripts = project / "scripts"
    required_scripts = (
        "select_itch50_universe.py", "extract_itch50_symbols.py",
        "assemble_itch50_universe_batches.py", "build_itch_universe_config.py",
    )
    for filename in required_scripts:
        if not (scripts / filename).is_file():
            raise SystemExit(f"missing project script: {scripts / filename}")
    extractor_sha256 = sha256_file(scripts / "extract_itch50_symbols.py")
    state_targets_record = (
        None
        if state_targets_path is None
        else {
            "path": str(state_targets_path),
            "sha256": sha256_file(state_targets_path),
        }
    )

    compact = parsed_date.strftime("%Y%m%d")
    selection = root / "selection"
    batch_inputs = root / "batch_inputs"
    batch_outputs = root / "batch_outputs"
    batch_logs = root / "batch_logs"
    data_root = root / "empirical_data"
    full_catalog = selection / f"startup_catalog_{compact}.csv"
    selected_symbols = selection / f"selected_symbols_{compact}.txt"
    selection_provenance = selection / f"selection_provenance_{compact}.json"
    opening_bbo = root / f"opening_bbo_{compact}.csv"
    candidates = root / f"selected_candidates_{compact}.csv"
    exclusions = root / f"extractor_exclusions_{compact}.csv"
    assembly_manifest = root / f"batch_assembly_{compact}.json"
    universe_config = root / f"nasdaq_common_plus_qqq_{compact}.csv"
    universe_provenance = root / f"nasdaq_common_plus_qqq_{compact}.provenance.json"
    metadata_path = root / "calibration_job_metadata.json"
    root.mkdir(parents=True, exist_ok=True)

    if universe_config.is_file() and universe_provenance.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            observed_selection = validate_selection_manifest(
                selection_provenance, selected_symbols, selection_request
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "completed result cannot be reused under the requested selection: "
                f"{error}"
            ) from error
        if metadata.get("selection") != observed_selection:
            raise RuntimeError(
                f"completed metadata does not bind the current selection: {metadata_path}"
            )
        if metadata.get("queue_reactive_state_targets") != state_targets_record:
            raise RuntimeError(
                "completed metadata does not bind the requested state targets; "
                "use a fresh --result-root"
            )
        if metadata.get("extractor_sha256") != extractor_sha256:
            raise RuntimeError(
                "completed metadata was produced by a different extractor source; "
                "use a fresh --result-root"
            )
        print(json.dumps({
            "status": "already_complete", "date": args.date,
            "result_root": str(root), "universe_config": str(universe_config),
        }, sort_keys=True))
        return 0

    digest_path = root / "input.sha256"
    if digest_path.is_file():
        pieces = digest_path.read_text(encoding="utf-8").strip().split()
        if len(pieces) < 1 or len(pieces[0]) != 64:
            raise RuntimeError(f"invalid saved archive digest: {digest_path}")
        archive_sha256 = pieces[0]
    else:
        print(f"hashing {archive}", flush=True)
        archive_sha256 = sha256_file(archive)
        atomic_text(digest_path, f"{archive_sha256}  {archive}\n")

    if not (
        full_catalog.is_file()
        and selected_symbols.is_file()
        and selection_provenance.is_file()
    ):
        selection.mkdir(parents=True, exist_ok=True)
        selector = [
            sys.executable, str(scripts / "select_itch50_universe.py"),
            "--itch", str(archive), "--catalog-out", str(full_catalog),
            "--symbols-out", str(selected_symbols),
            "--provenance-out", str(selection_provenance),
        ]
        if args.max_symbols is not None:
            selector.extend(["--max-symbols", str(args.max_symbols)])
        if fixed_symbols_path is not None:
            selector.extend(["--fixed-symbols", str(fixed_symbols_path)])
        completed = run_checked(selector)
        atomic_text(root / "selection_summary.txt", completed.stdout)
    try:
        selection_record = validate_selection_manifest(
            selection_provenance, selected_symbols, selection_request
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    symbols = list(selection_record["selected_symbols"]["ordered_symbols"])

    batches = [symbols[start:start + args.symbols_per_batch]
               for start in range(0, len(symbols), args.symbols_per_batch)]
    plan = {
        "schema_version": 2,
        "date": args.date,
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "selected_symbols": len(symbols),
        "symbols_per_batch": args.symbols_per_batch,
        "batch_count": len(batches),
        "snapshot_ms": args.snapshot_ms,
        "target_window_seconds": args.target_window_seconds,
        "selection": selection_record,
        "extractor_sha256": extractor_sha256,
        "queue_reactive_state_targets": state_targets_record,
    }
    plan_path = root / "batch_plan.json"
    if plan_path.is_file():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous != plan:
            raise RuntimeError(
                f"existing batch plan differs from requested plan: {plan_path}; "
                "use a fresh --result-root"
            )
    else:
        atomic_json(plan_path, plan)
    batch_inputs.mkdir(parents=True, exist_ok=True)
    batch_outputs.mkdir(parents=True, exist_ok=True)
    batch_logs.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[int, pathlib.Path, str, pathlib.Path]] = []
    for index, batch_symbols in enumerate(batches):
        batch_name = f"batch_{index:05d}"
        symbol_file = batch_inputs / f"symbols_{index:05d}.txt"
        rendered = "\n".join(batch_symbols) + "\n"
        if symbol_file.exists() and symbol_file.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"existing batch symbol file differs: {symbol_file}")
        if not symbol_file.exists():
            atomic_text(symbol_file, rendered)
        symbols_digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        marker = batch_logs / f"{batch_name}.complete.json"
        if not valid_batch_marker(
            marker, archive_sha256=archive_sha256,
            symbols_sha256=symbols_digest, batch_name=batch_name,
            extractor_sha256=extractor_sha256,
            state_targets_sha256=(
                None
                if state_targets_record is None
                else str(state_targets_record["sha256"])
            ),
        ):
            pending.append((index, symbol_file, symbols_digest, marker))

    print(
        f"date={args.date} symbols={len(symbols)} batches={len(batches)} "
        f"complete={len(batches) - len(pending)} pending={len(pending)} "
        f"concurrency={args.concurrent_batches}",
        flush=True,
    )

    def extract_batch(item: tuple[int, pathlib.Path, str, pathlib.Path]) -> str:
        index, symbol_file, symbols_digest, marker = item
        batch_name = f"batch_{index:05d}"
        output_root = batch_outputs / batch_name
        output_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(scripts / "extract_itch50_symbols.py"),
            "--input", str(archive), "--input-sha256", archive_sha256,
            "--symbols-file", str(symbol_file), "--date", args.date,
            "--start", "09:30:00", "--end", "16:00:00",
            "--snapshot-ms", str(args.snapshot_ms),
            "--target-window-seconds",
            *[str(value) for value in args.target_window_seconds],
            "--output-root", str(output_root),
            "--progress-seconds", str(args.progress_seconds),
            "--skip-invalid-openings",
        ]
        if state_targets_path is not None:
            command.extend(["--state-targets-csv", str(state_targets_path)])
        completed = run_checked(
            command,
            stdout_path=batch_logs / f"{batch_name}.json",
            stderr_path=batch_logs / f"{batch_name}.err",
        )
        try:
            summary = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid extractor JSON for {batch_name}") from error
        atomic_json(marker, {
            "status": "complete", "batch": batch_name,
            "archive_sha256": archive_sha256,
            "symbols_sha256": symbols_digest,
            "extractor_sha256": extractor_sha256,
            "state_targets_sha256": (
                None
                if state_targets_record is None
                else state_targets_record["sha256"]
            ),
            "extractor_summary": summary,
        })
        return batch_name

    if pending:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrent_batches,
        ) as executor:
            futures = {executor.submit(extract_batch, item): item[0] for item in pending}
            for completed_count, future in enumerate(
                concurrent.futures.as_completed(futures), start=1,
            ):
                batch_name = future.result()
                print(
                    f"completed {batch_name} ({completed_count}/{len(pending)} pending batches)",
                    flush=True,
                )

    if not (assembly_manifest.is_file() and data_root.is_dir()):
        run_checked([
            sys.executable, str(scripts / "assemble_itch50_universe_batches.py"),
            "--symbols-file", str(selected_symbols), "--source-catalog", str(full_catalog),
            "--batch-root", str(batch_outputs), "--trading-date", args.date,
            "--data-root", str(data_root), "--opening-bbo-out", str(opening_bbo),
            "--candidate-catalog-out", str(candidates), "--exclusions-out", str(exclusions),
            "--manifest-out", str(assembly_manifest),
        ], stdout_path=root / "assembly_summary.json")

    if not (universe_config.is_file() and universe_provenance.is_file()):
        run_checked([
            sys.executable, str(scripts / "build_itch_universe_config.py"),
            "--data-root", str(data_root), "--trading-date", args.date,
            "--catalog", str(candidates), "--opening-bbo", str(opening_bbo),
            "--output", str(universe_config), "--provenance", str(universe_provenance),
            "--absolute-paths", "--activity-scale", "0.30", "--hawkes-beta", "10.0",
            "--balance-strength", "1.0", "--quote-quantity-fraction", "0.5",
            "--minimum-quote-quantity", "10", "--maximum-quote-quantity", "1000",
        ], stdout_path=root / "universe_build_summary.json")

    with universe_config.open(encoding="utf-8") as source:
        book_count = max(0, sum(1 for _ in source) - 1)
    provenance_value = json.loads(universe_provenance.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 2,
        "execution_environment": "local_workstation",
        "calibration_scope": (
            "Per-symbol marginal ITCH inputs for policy-eligible NASDAQ common stocks "
            "plus QQQ; no fitted cross-asset dependence is asserted."
        ),
        "raw_itch": {"path": str(archive), "sha256": archive_sha256},
        "trading_date": args.date,
        "selection": selection_record,
        "extractor_sha256": extractor_sha256,
        "queue_reactive_state_targets": state_targets_record,
        "batching": {
            "symbols_per_batch": args.symbols_per_batch,
            "maximum_concurrent_batches": args.concurrent_batches,
            "snapshot_ms": args.snapshot_ms,
            "target_window_seconds": args.target_window_seconds,
        },
        "configuration": {
            "path": str(universe_config),
            "sha256": sha256_file(universe_config),
            "book_count": book_count,
            "qqq_book_id": 0,
            "absolute_data_paths": True,
        },
        "builder_provenance": str(universe_provenance),
        "accepted_count": int(provenance_value["accepted_count"]),
        "rejected_count": int(provenance_value["rejected_count"]),
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous != metadata:
            raise RuntimeError(f"existing metadata differs: {metadata_path}")
    else:
        atomic_json(metadata_path, metadata)
    print(json.dumps({
        "status": "complete", "date": args.date, "books": book_count,
        "result_root": str(root), "universe_config": str(universe_config),
        "metadata": str(metadata_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
