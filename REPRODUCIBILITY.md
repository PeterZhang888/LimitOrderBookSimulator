# Reproducibility

## Source integrity

Verify the release before configuration:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
```

The manifest excludes build directories, Slurm output, runtime results and raw
or expanded empirical data.

## Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=ON \
  -DLOB_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Cluster runs should use a single compiler/MPI toolchain for configuration,
linking and execution. `scripts/seagull_deterministic_build.sh` implements the
recorded Seagull build contract.

## External empirical inputs

The complete workflow requires:

- five 2019 training-session extraction roots;
- the 30 January 2020 development-validation extraction root;
- pooled per-symbol empirical distributions and Hawkes-rate files;
- the frozen cluster assignments, background policy and value policy;
- the passed training-freeze and development-validation manifests.

These inputs are intentionally not committed. Paths are supplied through the
environment variables documented by the corresponding `submit_*.sh` launcher.

## Calibration and validation

```bash
sbatch submit_five_day_pooled_training.sh
sbatch submit_cluster_value_agent_calibration.sh
sbatch submit_queue_reactive_full_validation_hpc.sh
```

Each stage writes a manifest consumed by the next stage. A financial treatment
must not run unless the validation handoff and its referenced hashes pass.

## Case study

After providing `POOL_ROOT`, `SELECTION_ROOT`, `DATA_ROOT` and `EVIDENCE_ROOT`:

```bash
sbatch --export=ALL,EXPERIMENT=preflight submit_queue_reactive_case_study.sh
sbatch --export=ALL,EXPERIMENT=science submit_queue_reactive_case_study.sh
sbatch --export=ALL,EXPERIMENT=scaling submit_queue_reactive_case_study.sh
```

The preflight compares canonical terminal-state hashes at one and 32 ranks.
The science matrix uses paired shock/control seeds. Compact retained outputs are
under `results/final-case-study/`.

## Interpretation boundary

The retained case-study run confirms deterministic rank equivalence and direct
shock execution. The shared market maker reached zero quote scale before the
shock, so that run does not identify cross-asset contagion through an active
common dealer. The detailed audit is in
`docs/case-study/case_study_analysis_report.md`.
