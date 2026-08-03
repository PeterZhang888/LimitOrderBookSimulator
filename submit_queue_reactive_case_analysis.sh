#!/usr/bin/env bash
# Analyse an already completed financial path matrix without rerunning MPI.
# Submit from the source-release directory with SOURCE_RESULT_DIR exported.
#SBATCH --job-name=lob-final-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -Eeuo pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"
: "${SOURCE_RESULT_DIR:?export SOURCE_RESULT_DIR as the completed financial result directory}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
SOURCE_RESULT_DIR="$(cd "${SOURCE_RESULT_DIR}" && pwd -P)"
POSTPROCESS_DIR="${POSTPROCESS_DIR:-${SOURCE_RESULT_DIR}/postprocessing}"

if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh
fi
if ! type module >/dev/null 2>&1; then
    echo "ERROR: the module command is unavailable." >&2
    exit 2
fi
module purge
module load python/3.11.6-gcc-8.5.0-l2ohhiv

for command_name in python3 sha256sum; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: ${command_name} is unavailable after loading modules." >&2
        exit 2
    fi
done

mkdir -p "${PROJECT_DIR}/slurm"
cd "${PROJECT_DIR}"
sha256sum -c SOURCE_MANIFEST.sha256

GLOBAL_RAW="${SOURCE_RESULT_DIR}/financial_global_raw.csv"
MECHANISM_RAW="${SOURCE_RESULT_DIR}/mechanism_preflight_raw.csv"
UNCOUPLED_RAW="${SOURCE_RESULT_DIR}/financial_uncoupled_raw.csv"
SHARED_OFF_RAW="${SOURCE_RESULT_DIR}/financial_shared_off_raw.csv"
CASE_ARTIFACT="${SOURCE_RESULT_DIR}/portable_case/portable_queue_reactive_case.json"
for required in \
    "${GLOBAL_RAW}" \
    "${MECHANISM_RAW}" \
    "${UNCOUPLED_RAW}" \
    "${SHARED_OFF_RAW}" \
    "${CASE_ARTIFACT}"
do
    if [[ ! -s "${required}" ]]; then
        echo "ERROR: required completed-path artifact is missing: ${required}" >&2
        exit 2
    fi
done

if [[ -e "${POSTPROCESS_DIR}" ]] && [[ -n "$(find "${POSTPROCESS_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: refusing to mix analysis into non-empty directory: ${POSTPROCESS_DIR}" >&2
    exit 2
fi
mkdir -p "${POSTPROCESS_DIR}"

mapfile -t CASE_PATHS < <(python3 - "${CASE_ARTIFACT}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads(manifest.read_text(encoding="utf-8"))
runtime = payload.get("runtime_artifacts")
if not isinstance(runtime, dict):
    raise SystemExit("portable case artifact lacks runtime_artifacts")
for key in ("case_config", "cluster_map"):
    record = runtime.get(key)
    if not isinstance(record, dict):
        raise SystemExit(f"portable case artifact lacks {key}")
    path = pathlib.Path(str(record.get("path", ""))).resolve()
    if not path.is_file():
        raise SystemExit(f"portable runtime artifact is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.get("sha256"):
        raise SystemExit(f"portable runtime artifact hash mismatch: {path}")
    print(path)
PY
)
if (( ${#CASE_PATHS[@]} != 2 )); then
    echo "ERROR: could not resolve the portable universe and cluster map." >&2
    exit 2
fi
UNIVERSE_CONFIG="${CASE_PATHS[0]}"
CLUSTER_MAP="${CASE_PATHS[1]}"

python3 "${PROJECT_DIR}/scripts/validate_truncated_full_prefix.py" \
    --short-raw "${MECHANISM_RAW}" \
    --full-raw "${GLOBAL_RAW}" \
    --output "${POSTPROCESS_DIR}/truncated_full_prefix_certificate.json"

python3 "${PROJECT_DIR}/scripts/analyze_fragmented_shared_liquidity_case.py" \
    --global-raw "${GLOBAL_RAW}" \
    --uncoupled-raw "${UNCOUPLED_RAW}" \
    --shared-off-raw "${SHARED_OFF_RAW}" \
    --universe-input "${CASE_ARTIFACT}" \
    --shock-time-seconds 11700 \
    --horizon-seconds 1800 \
    --rank 16 \
    --output-dir "${POSTPROCESS_DIR}/financial_analysis"

python3 "${PROJECT_DIR}/scripts/analyze_cluster_liquidity_heterogeneity.py" \
    --global-raw "${GLOBAL_RAW}" \
    --uncoupled-raw "${UNCOUPLED_RAW}" \
    --shared-off-raw "${SHARED_OFF_RAW}" \
    --universe-config "${UNIVERSE_CONFIG}" \
    --cluster-assignments "${CLUSTER_MAP}" \
    --shock-time-seconds 11700 \
    --horizon-seconds 1800 \
    --rank 16 \
    --output-dir "${POSTPROCESS_DIR}/financial_cluster_analysis"

python3 - "${POSTPROCESS_DIR}" "${SOURCE_RESULT_DIR}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(sys.argv[2]).resolve()
artifacts = []
for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
    artifacts.append({
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    })
payload = {
    "schema_version": 1,
    "status": "completed_financial_paths_postprocessed",
    "source_result_dir": str(source),
    "artifact_count": len(artifacts),
    "hash_bound_artifacts": artifacts,
}
(root / "postprocessing_completion.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "postprocessing_dir=${POSTPROCESS_DIR}"
echo "completion_manifest=${POSTPROCESS_DIR}/postprocessing_completion.json"
date --iso-8601=seconds
