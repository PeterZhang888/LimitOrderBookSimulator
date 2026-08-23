#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$EXPERIMENT_DIR/../.." && pwd)
exec bash "$PROJECT_DIR/scripts/submit_experiment.sh" "$EXPERIMENT_DIR"
