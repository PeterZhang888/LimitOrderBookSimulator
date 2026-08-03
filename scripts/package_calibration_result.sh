#!/usr/bin/env bash
# Package one completed or failed calibration result inside a Slurm allocation.

set -Eeuo pipefail

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

(( $# == 3 )) || fail \
    "usage: package_calibration_result.sh PROJECT CAL_JOB OUTPUT_DIRECTORY"

PROJECT_INPUT="$1"
CAL_JOB="$2"
OUTPUT_INPUT="$3"

[[ "${CAL_JOB}" =~ ^[0-9]+$ ]] || fail "CAL_JOB must contain only digits"
[[ -d "${PROJECT_INPUT}" ]] || fail "project directory is missing: ${PROJECT_INPUT}"
[[ -d "${OUTPUT_INPUT}" ]] || fail "output directory is missing: ${OUTPUT_INPUT}"

PROJECT_DIR="$(cd "${PROJECT_INPUT}" && pwd -P)"
OUTPUT_DIR="$(cd "${OUTPUT_INPUT}" && pwd -P)"
RESULT_REL="results/seagull/cluster_value_calibration_${CAL_JOB}"
OUT_REL="slurm/lob-cluster-cal-${CAL_JOB}.out"
ERR_REL="slurm/lob-cluster-cal-${CAL_JOB}.err"
[[ -d "${PROJECT_DIR}/${RESULT_REL}" ]] \
    || fail "calibration result directory is missing: ${PROJECT_DIR}/${RESULT_REL}"

PACKAGE_LABEL="${PACKAGE_LABEL:-complete}"
[[ "${PACKAGE_LABEL}" =~ ^[a-z][a-z0-9_-]*$ ]] \
    || fail "PACKAGE_LABEL must start with a lowercase letter and contain only lowercase letters, digits, underscores or hyphens"
PACKAGE_NAME="calibration_${CAL_JOB}_${PACKAGE_LABEL}.tar.gz"
PACKAGE_PATH="${OUTPUT_DIR}/${PACKAGE_NAME}"
CHECKSUM_PATH="${PACKAGE_PATH}.sha256"
[[ ! -e "${PACKAGE_PATH}" && ! -e "${CHECKSUM_PATH}" ]] \
    || fail "refusing to overwrite an existing package or checksum"

TEMP_TAG="${SLURM_JOB_ID:-$$}"
PACKAGE_TEMP="${OUTPUT_DIR}/.${PACKAGE_NAME}.${TEMP_TAG}.tmp"
CHECKSUM_TEMP="${OUTPUT_DIR}/.${PACKAGE_NAME}.sha256.${TEMP_TAG}.tmp"
cleanup() {
    rm -f -- "${PACKAGE_TEMP}" "${CHECKSUM_TEMP}"
}
trap cleanup EXIT

PACKAGE_INPUTS=("${RESULT_REL}")
[[ -e "${PROJECT_DIR}/${OUT_REL}" ]] && PACKAGE_INPUTS+=("${OUT_REL}")
[[ -e "${PROJECT_DIR}/${ERR_REL}" ]] && PACKAGE_INPUTS+=("${ERR_REL}")

cd "${PROJECT_DIR}"
tar -czf "${PACKAGE_TEMP}" "${PACKAGE_INPUTS[@]}"
mv -- "${PACKAGE_TEMP}" "${PACKAGE_PATH}"

cd "${OUTPUT_DIR}"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${PACKAGE_NAME}" > "${CHECKSUM_TEMP}"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${PACKAGE_NAME}" > "${CHECKSUM_TEMP}"
else
    fail "neither sha256sum nor shasum is available"
fi
mv -- "${CHECKSUM_TEMP}" "${CHECKSUM_PATH}"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c "${PACKAGE_NAME}.sha256"
else
    shasum -a 256 -c "${PACKAGE_NAME}.sha256"
fi

echo "package=${PACKAGE_PATH}"
echo "checksum=${CHECKSUM_PATH}"
