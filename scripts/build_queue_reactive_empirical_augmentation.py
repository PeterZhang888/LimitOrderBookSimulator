#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Re-extract queue-reactive sufficient statistics from fixed raw ITCH files.

The legacy compact empirical bundle predates the queue-reactive model.  This
driver reads each raw archive in bounded symbol batches and retains only the
new training sufficient statistics plus a signed audit block.  It never
modifies the legacy bundle.  ``apply_queue_reactive_empirical_augmentation.py``
merges the resulting sidecars into a hard-linked copy on the cluster.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from typing import Any, Mapping, Sequence


EXPECTED_DATES = (
    "2019-01-30",
    "2019-03-27",
    "2019-07-30",
    "2019-10-30",
    "2019-12-30",
    "2020-01-30",
)
REQUIRED_QUEUE_FILES = (
    "limit_buy_improvement_distribution.txt",
    "limit_sell_improvement_distribution.txt",
    "intraday_event_counts.csv",
    "queue_state_counts.csv",
    "queue_state_exposure.csv",
    "event_count_lag_moments.csv",
)


class AugmentationError(RuntimeError):
    """The queue-reactive augmentation cannot be certified."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_csv(
    path: pathlib.Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_dated_path(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ISO-DATE=PATH")
    raw_date, raw_path = value.split("=", 1)
    try:
        normalized = date.fromisoformat(raw_date).isoformat()
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid date: {raw_date}") from error
    path = pathlib.Path(raw_path).expanduser().resolve()
    return normalized, path


def read_symbols(path: pathlib.Path, expected: int) -> list[str]:
    symbols = [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(symbols) != expected or len(symbols) != len(set(symbols)):
        raise AugmentationError(
            f"expected {expected} unique symbols in {path}, observed {len(symbols)}"
        )
    if symbols != ["QQQ", *sorted(symbol for symbol in symbols if symbol != "QQQ")]:
        raise AugmentationError(
            "fixed cohort must be QQQ followed by lexicographically ordered symbols"
        )
    return symbols


def build_state_targets(
    pooled_config: pathlib.Path,
    symbols: Sequence[str],
    output: pathlib.Path,
) -> dict[str, object]:
    try:
        with pooled_config.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or [])
            required = {
                "symbol", "target_mean_bid_depth", "target_mean_ask_depth",
            }
            if not required.issubset(fields):
                raise AugmentationError(
                    f"pooled config lacks state-target columns: {pooled_config}"
                )
            rows = list(reader)
    except OSError as error:
        raise AugmentationError(f"cannot read pooled config: {error}") from error
    by_symbol: dict[str, tuple[float, float]] = {}
    for row in rows:
        symbol = str(row["symbol"]).strip().upper()
        if not symbol or symbol in by_symbol:
            raise AugmentationError(f"blank or duplicate pooled symbol: {symbol!r}")
        try:
            bid = float(row["target_mean_bid_depth"])
            ask = float(row["target_mean_ask_depth"])
        except (TypeError, ValueError) as error:
            raise AugmentationError(f"nonnumeric state target for {symbol}") from error
        if not all(math.isfinite(value) and value > 0.0 for value in (bid, ask)):
            raise AugmentationError(f"invalid state target for {symbol}")
        by_symbol[symbol] = (bid, ask)
    if set(by_symbol) != set(symbols):
        raise AugmentationError(
            "pooled-config symbols do not equal the fixed certification cohort"
        )
    rendered = [
        {
            "symbol": symbol,
            "target_mean_bid_depth": f"{by_symbol[symbol][0]:.17g}",
            "target_mean_ask_depth": f"{by_symbol[symbol][1]:.17g}",
        }
        for symbol in symbols
    ]
    if output.exists():
        temporary = output.with_name(output.name + ".candidate")
        atomic_csv(
            temporary,
            ("symbol", "target_mean_bid_depth", "target_mean_ask_depth"),
            rendered,
        )
        if temporary.read_bytes() != output.read_bytes():
            temporary.unlink(missing_ok=True)
            raise AugmentationError(f"existing state targets differ: {output}")
        temporary.unlink()
    else:
        atomic_csv(
            output,
            ("symbol", "target_mean_bid_depth", "target_mean_ask_depth"),
            rendered,
        )
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "source_pooled_config": {
            "path": str(pooled_config),
            "sha256": sha256_file(pooled_config),
        },
        "symbol_count": len(symbols),
    }


def validate_queue_block(
    block: object, *, symbol: str, source_manifest: pathlib.Path,
) -> dict[str, object]:
    if not isinstance(block, dict):
        raise AugmentationError(
            f"extractor manifest lacks queue-reactive audit: {source_manifest}"
        )
    conservation = block.get("event_count_conservation")
    exposure = block.get("exposure")
    artifacts = block.get("artifacts")
    if (
        block.get("schema_version") != 2
        or block.get("training_only") is not True
        or block.get("queue_policy_estimation_ready") is not True
        or not isinstance(conservation, dict)
        or conservation.get("totals_equal") is not True
        or conservation.get("equals_legacy_quantity_observation_counts") is not True
        or not isinstance(exposure, dict)
        or exposure.get("exact_nanosecond_conservation") is not True
        or not isinstance(artifacts, dict)
    ):
        raise AugmentationError(
            f"extractor did not certify queue-reactive artifacts for {symbol}"
        )
    observed_names = {str(value) for value in artifacts.values()}
    if not set(REQUIRED_QUEUE_FILES).issubset(observed_names):
        raise AugmentationError(
            f"extractor queue-artifact list is incomplete for {symbol}"
        )
    return block


def marker_valid(
    marker: pathlib.Path,
    *,
    raw_sha256: str,
    symbols_sha256: str,
    extractor_sha256: str,
    state_targets_sha256: str,
) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        payload.get("status") == "complete"
        and payload.get("raw_sha256") == raw_sha256
        and payload.get("symbols_sha256") == symbols_sha256
        and payload.get("extractor_sha256") == extractor_sha256
        and payload.get("state_targets_sha256") == state_targets_sha256
    ):
        return False
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return False
    for record in records:
        if not isinstance(record, dict):
            return False
        for item in record.get("files", []):
            if not isinstance(item, dict):
                return False
            path = pathlib.Path(str(item.get("path", "")))
            if not path.is_file() or sha256_file(path) != item.get("sha256"):
                return False
    return True


def verify_completed_manifest(
    root: pathlib.Path,
    payload: object,
    *,
    symbols: Sequence[str],
    cohort_sha256: str,
    extractor_sha256: str,
    state_targets_sha256: str,
) -> dict[str, object]:
    """Revalidate every retained artifact before treating a run as complete."""
    if not isinstance(payload, dict):
        raise AugmentationError("completed augmentation manifest is not an object")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("role")
        != "queue_reactive_sufficient_statistics_augmentation"
        or payload.get("legacy_empirical_bundle_modified") is not False
    ):
        raise AugmentationError("existing augmentation manifest is not complete")
    cohort = payload.get("cohort")
    state_targets = payload.get("state_targets")
    extractor = payload.get("extractor")
    if (
        not isinstance(cohort, dict)
        or cohort.get("symbol_count") != len(symbols)
        or cohort.get("canonical_sha256") != cohort_sha256
        or not isinstance(state_targets, dict)
        or state_targets.get("sha256") != state_targets_sha256
        or not isinstance(extractor, dict)
        or extractor.get("sha256") != extractor_sha256
    ):
        raise AugmentationError(
            "completed augmentation is bound to different inputs or source code"
        )
    records = payload.get("records")
    expected_count = len(EXPECTED_DATES) * len(symbols)
    if (
        not isinstance(records, list)
        or len(records) != expected_count
        or payload.get("record_count") != expected_count
        or payload.get("records_sha256") != sha256_json(records)
    ):
        raise AugmentationError("completed augmentation record set is invalid")
    expected_identities = {
        (trading_date, symbol)
        for trading_date in EXPECTED_DATES
        for symbol in symbols
    }
    observed_identities: set[tuple[str, str]] = set()
    required_names = {
        *REQUIRED_QUEUE_FILES,
        "source_extractor_manifest.json",
        "queue_reactive_training_artifacts.json",
    }
    for record in records:
        if not isinstance(record, dict):
            raise AugmentationError("completed augmentation contains an invalid record")
        identity = (
            str(record.get("trading_date", "")),
            str(record.get("symbol", "")),
        )
        if identity in observed_identities or identity not in expected_identities:
            raise AugmentationError(
                f"completed augmentation contains an invalid identity: {identity}"
            )
        observed_identities.add(identity)
        trading_date, symbol = identity
        expected_relative = pathlib.PurePosixPath(
            f"itch_{trading_date.replace('-', '')}/empirical_data/"
            f"itch_{trading_date.replace('-', '')}_{symbol.lower()}"
        )
        relative = pathlib.PurePosixPath(
            str(record.get("relative_directory", ""))
        )
        if relative != expected_relative:
            raise AugmentationError(
                f"completed augmentation path differs for {trading_date}/{symbol}"
            )
        files = record.get("files")
        if not isinstance(files, list):
            raise AugmentationError(
                f"completed augmentation files are absent for {trading_date}/{symbol}"
            )
        observed_names: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise AugmentationError("completed augmentation file record is invalid")
            name = str(item.get("relative_name", ""))
            if pathlib.PurePath(name).name != name or name in observed_names:
                raise AugmentationError(f"unsafe or duplicate retained file: {name}")
            observed_names.add(name)
            path = root / pathlib.Path(*relative.parts) / name
            if not path.is_file() or sha256_file(path) != item.get("sha256"):
                raise AugmentationError(f"retained artifact is missing or changed: {path}")
        if observed_names != required_names:
            raise AugmentationError(
                f"retained artifact set differs for {trading_date}/{symbol}"
            )
    if observed_identities != expected_identities:
        raise AugmentationError("completed augmentation cohort is incomplete")
    return payload


def run_extractor(
    *,
    extractor: pathlib.Path,
    archive: pathlib.Path,
    archive_sha256: str,
    trading_date: str,
    symbol_file: pathlib.Path,
    state_targets: pathlib.Path,
    output_root: pathlib.Path,
    progress_seconds: float,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(extractor),
        "--input", str(archive),
        "--input-sha256", archive_sha256,
        "--symbols-file", str(symbol_file),
        "--date", trading_date,
        "--start", "09:30:00",
        "--end", "16:00:00",
        "--snapshot-ms", "1000",
        "--target-window-seconds", "300", "3600",
        "--output-root", str(output_root),
        "--state-targets-csv", str(state_targets),
        "--progress-seconds", str(progress_seconds),
        "--skip-invalid-openings",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise AugmentationError(
            f"extractor failed ({completed.returncode}): {diagnostic}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AugmentationError("extractor returned invalid JSON") from error


def retain_batch(
    *,
    batch_symbols: Sequence[str],
    trading_date: str,
    temporary_root: pathlib.Path,
    date_output: pathlib.Path,
) -> list[dict[str, object]]:
    compact = trading_date.replace("-", "")
    records: list[dict[str, object]] = []
    for symbol in batch_symbols:
        dirname = f"itch_{compact}_{symbol.lower()}"
        source = temporary_root / dirname
        manifest_path = source / f"itch_manifest_{symbol.lower()}_{compact}.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AugmentationError(
                f"cannot read extractor manifest for {trading_date}/{symbol}: {error}"
            ) from error
        if manifest.get("symbol") != symbol or manifest.get("trading_date") != trading_date:
            raise AugmentationError(f"extractor identity mismatch for {trading_date}/{symbol}")
        block = validate_queue_block(
            manifest.get("queue_reactive_training_artifacts"),
            symbol=symbol,
            source_manifest=manifest_path,
        )
        destination = date_output / "empirical_data" / dirname
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        files: list[dict[str, str]] = []
        for filename in REQUIRED_QUEUE_FILES:
            source_file = source / filename
            if not source_file.is_file():
                raise AugmentationError(
                    f"required queue artifact is missing: {source_file}"
                )
            destination_file = destination / filename
            shutil.copy2(source_file, destination_file)
            files.append({
                "path": str(destination_file),
                "relative_name": filename,
                "sha256": sha256_file(destination_file),
            })
        source_manifest_copy = destination / "source_extractor_manifest.json"
        shutil.copy2(manifest_path, source_manifest_copy)
        sidecar = destination / "queue_reactive_training_artifacts.json"
        sidecar_payload = {
            "schema_version": 1,
            "trading_date": trading_date,
            "symbol": symbol,
            "queue_reactive_training_artifacts": block,
            "source_extractor_manifest": {
                "path": str(source_manifest_copy),
                "sha256": sha256_file(source_manifest_copy),
            },
            "legacy_distribution_observation_counts": manifest.get(
                "distribution_observation_counts"
            ),
            "legacy_placement_counts": manifest.get("placement_counts"),
        }
        atomic_json(sidecar, sidecar_payload)
        files.extend([
            {
                "path": str(source_manifest_copy),
                "relative_name": source_manifest_copy.name,
                "sha256": sha256_file(source_manifest_copy),
            },
            {
                "path": str(sidecar),
                "relative_name": sidecar.name,
                "sha256": sha256_file(sidecar),
            },
        ])
        records.append({
            "trading_date": trading_date,
            "symbol": symbol,
            "relative_directory": str(
                pathlib.Path(f"itch_{compact}") / "empirical_data" / dirname
            ),
            "files": files,
        })
    return records


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--project-dir", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    result.add_argument("--pooled-config", type=pathlib.Path, required=True)
    result.add_argument("--symbols-file", type=pathlib.Path, required=True)
    result.add_argument(
        "--itch", action="append", type=parse_dated_path, required=True,
        metavar="DATE=PATH",
    )
    result.add_argument("--output-root", type=pathlib.Path, required=True)
    result.add_argument("--expected-symbols", type=int, default=1480)
    result.add_argument("--symbols-per-batch", type=int, default=256)
    result.add_argument("--concurrent-batches", type=int, default=2)
    result.add_argument("--progress-seconds", type=float, default=60.0)
    return result


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.expected_symbols <= 0 or args.symbols_per_batch <= 0:
        raise AugmentationError("symbol counts and batch size must be positive")
    if args.concurrent_batches <= 0 or args.progress_seconds <= 0.0:
        raise AugmentationError("concurrency and progress interval must be positive")
    project = args.project_dir.expanduser().resolve()
    pooled = args.pooled_config.expanduser().resolve()
    symbol_path = args.symbols_file.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    extractor = project / "scripts" / "extract_itch50_symbols.py"
    if not extractor.is_file() or not pooled.is_file() or not symbol_path.is_file():
        raise AugmentationError("project, pooled config or symbol cohort is missing")
    dated_archives = dict(args.itch)
    if set(dated_archives) != set(EXPECTED_DATES) or len(args.itch) != len(EXPECTED_DATES):
        raise AugmentationError(
            "--itch must supply each predeclared 2019 training date and 2020-01-30 exactly once"
        )
    if any(not path.is_file() for path in dated_archives.values()):
        raise AugmentationError("one or more raw ITCH archives are missing")
    output_root.mkdir(parents=True, exist_ok=True)
    symbols = read_symbols(symbol_path, args.expected_symbols)
    state_targets_path = output_root / "queue_reactive_state_targets.csv"
    target_record = build_state_targets(pooled, symbols, state_targets_path)
    extractor_sha256 = sha256_file(extractor)
    cohort_sha256 = hashlib.sha256(
        "".join(f"{symbol}\n" for symbol in symbols).encode("utf-8")
    ).hexdigest()

    complete_manifest = output_root / "queue_reactive_augmentation_manifest.json"
    if complete_manifest.is_file():
        payload = json.loads(complete_manifest.read_text(encoding="utf-8"))
        verified = verify_completed_manifest(
            output_root,
            payload,
            symbols=symbols,
            cohort_sha256=cohort_sha256,
            extractor_sha256=extractor_sha256,
            state_targets_sha256=str(target_record["sha256"]),
        )
        print(json.dumps({
            "status": "already_complete",
            "manifest": str(complete_manifest),
            "manifest_sha256": sha256_file(complete_manifest),
        }, sort_keys=True))
        return verified

    date_records: list[dict[str, object]] = []
    all_symbol_records: list[dict[str, object]] = []
    for trading_date in EXPECTED_DATES:
        archive = dated_archives[trading_date]
        date_root = output_root / f"itch_{trading_date.replace('-', '')}"
        date_root.mkdir(parents=True, exist_ok=True)
        raw_digest_file = date_root / "raw_itch_sha256.json"
        if raw_digest_file.is_file():
            raw_record = json.loads(raw_digest_file.read_text(encoding="utf-8"))
            if (
                raw_record.get("size_bytes") != archive.stat().st_size
                or raw_record.get("path") != str(archive)
            ):
                raise AugmentationError(
                    f"raw archive identity changed for {trading_date}"
                )
        else:
            raw_record = {
                "path": str(archive),
                "basename": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            }
            atomic_json(raw_digest_file, raw_record)
        raw_sha256 = str(raw_record["sha256"])
        batches = [
            symbols[index:index + args.symbols_per_batch]
            for index in range(0, len(symbols), args.symbols_per_batch)
        ]
        input_root = date_root / "batch_inputs"
        log_root = date_root / "batch_logs"
        work_root = date_root / ".batch_work"
        input_root.mkdir(exist_ok=True)
        log_root.mkdir(exist_ok=True)
        work_root.mkdir(exist_ok=True)
        pending: list[tuple[int, list[str], pathlib.Path, pathlib.Path]] = []
        for index, batch_symbols in enumerate(batches):
            rendered = "".join(f"{symbol}\n" for symbol in batch_symbols)
            symbol_file = input_root / f"symbols_{index:05d}.txt"
            if symbol_file.exists() and symbol_file.read_text(encoding="utf-8") != rendered:
                raise AugmentationError(f"batch symbol file changed: {symbol_file}")
            if not symbol_file.exists():
                symbol_file.write_text(rendered, encoding="utf-8")
            marker = log_root / f"batch_{index:05d}.complete.json"
            symbols_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            if not marker_valid(
                marker,
                raw_sha256=raw_sha256,
                symbols_sha256=symbols_sha256,
                extractor_sha256=extractor_sha256,
                state_targets_sha256=str(target_record["sha256"]),
            ):
                pending.append((index, batch_symbols, symbol_file, marker))
        print(
            f"{trading_date}: batches={len(batches)} complete={len(batches)-len(pending)} "
            f"pending={len(pending)}",
            flush=True,
        )

        def process(item: tuple[int, list[str], pathlib.Path, pathlib.Path]) -> int:
            index, batch_symbols, symbol_file, marker = item
            temporary = work_root / f"batch_{index:05d}"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            summary = run_extractor(
                extractor=extractor,
                archive=archive,
                archive_sha256=raw_sha256,
                trading_date=trading_date,
                symbol_file=symbol_file,
                state_targets=state_targets_path,
                output_root=temporary,
                progress_seconds=args.progress_seconds,
            )
            if summary.get("symbols") != list(batch_symbols):
                raise AugmentationError(
                    f"extractor did not return the complete ordered batch {index}"
                )
            records = retain_batch(
                batch_symbols=batch_symbols,
                trading_date=trading_date,
                temporary_root=temporary,
                date_output=date_root,
            )
            rendered = symbol_file.read_bytes()
            marker_payload = {
                "schema_version": 1,
                "status": "complete",
                "batch_index": index,
                "raw_sha256": raw_sha256,
                "symbols_sha256": hashlib.sha256(rendered).hexdigest(),
                "extractor_sha256": extractor_sha256,
                "state_targets_sha256": target_record["sha256"],
                "extractor_summary": summary,
                "records": records,
                "records_sha256": sha256_json(records),
            }
            atomic_json(marker, marker_payload)
            shutil.rmtree(temporary)
            return index

        if pending:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrent_batches
            ) as executor:
                futures = [executor.submit(process, item) for item in pending]
                for completed_index, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1,
                ):
                    index = future.result()
                    print(
                        f"{trading_date}: completed batch {index:05d} "
                        f"({completed_index}/{len(pending)})",
                        flush=True,
                    )
        observed: list[dict[str, object]] = []
        for index in range(len(batches)):
            marker = log_root / f"batch_{index:05d}.complete.json"
            payload = json.loads(marker.read_text(encoding="utf-8"))
            records = payload.get("records")
            if not isinstance(records, list):
                raise AugmentationError(f"invalid batch marker: {marker}")
            observed.extend(records)
        if [record.get("symbol") for record in observed] != symbols:
            raise AugmentationError(
                f"assembled augmentation cohort differs on {trading_date}"
            )
        all_symbol_records.extend(observed)
        date_records.append({
            "trading_date": trading_date,
            "raw_itch": raw_record,
            "symbol_count": len(observed),
            "batch_count": len(batches),
            "records_sha256": sha256_json(observed),
        })

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "role": "queue_reactive_sufficient_statistics_augmentation",
        "legacy_empirical_bundle_modified": False,
        "training_and_development_validation_dates": list(EXPECTED_DATES),
        "cohort": {
            "symbol_count": len(symbols),
            "canonical_sha256": cohort_sha256,
            "declaration": {
                "path": str(symbol_path),
                "sha256": sha256_file(symbol_path),
            },
        },
        "state_targets": target_record,
        "extractor": {
            "path": str(extractor),
            "sha256": extractor_sha256,
        },
        "dates": date_records,
        "record_count": len(all_symbol_records),
        "records": all_symbol_records,
        "records_sha256": sha256_json(all_symbol_records),
    }
    if len(all_symbol_records) != len(EXPECTED_DATES) * len(symbols):
        raise AugmentationError("augmentation record count is incomplete")
    atomic_json(complete_manifest, manifest)
    print(json.dumps({
        "status": "complete",
        "manifest": str(complete_manifest),
        "manifest_sha256": sha256_file(complete_manifest),
        "record_count": len(all_symbol_records),
    }, sort_keys=True))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    try:
        build(parser().parse_args(argv))
    except (AugmentationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"queue-reactive augmentation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
