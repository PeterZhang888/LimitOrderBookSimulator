# R33 six-component continuation

## Purpose and evidential status

This is an **in-place continuation patch** for
`coupled_lob_r32_activity_regime_20260801`. It does not create a new model,
rerun parameter selection, or erase the failed nine-component report. It
implements a versioned retrospective development protocol after the R32
results showed that return kurtosis, cluster extrema and exact ACF
distribution moments were being used as hard gates despite their high
one-session sampling variability.

The `marketwide-six-v2` hard gate contains:

1. background-event activity;
2. mean spread;
3. combined bid-plus-ask top depth, combined per symbol before scoring;
4. mid-price move rate;
5. return variance; and
6. lag-one absolute-return autocorrelation.

Return kurtosis, separate bid/ask depth, cluster-level scores and exact ACF
mean/median/p90 errors remain in the output as diagnostics. Complete
fixed-clock sampling, zero invalid/one-sided observations, equal symbol
universes and valid numerical inputs remain mandatory structural checks.

The original `strict-nine-v1` report and its residual table are hash-bound by
`six_component_training_certificate.json`. The continuation independently
recomputes the six-component decision from those residuals; it does not trust
the certificate's displayed aggregate values. On the supplied R32 iteration-1
evidence, the authoritative reanalysis is:

| Training date | Six-component robust score | Gross-failure fraction |
|---|---:|---:|
| 2019-01-30 | 1.416896359 | 0.025676 |
| 2019-03-27 | 1.292462634 | 0.012838 |
| 2019-07-30 | 1.433554623 | 0.035135 |
| 2019-10-30 | 1.284515419 | 0.018243 |
| 2019-12-30 | 1.317425420 | 0.020270 |

All five dates satisfy the frozen limits: robust score at most 1.5,
component score at most 2.5, per-component coverage at least 0.95, and gross
failure fraction at most 0.10.

The subsequent 30 January 2020 run remains **development validation**. Its
outcome is not known from the 2019 evidence and must not be described as passed
until `heldout_run_manifest.json` exists and records a pass.

## Why the patch must be applied in place

The completed R32 checkpoints hash the exact simulator command, including the
absolute executable path. Extracting this patch as a new project would change
that path and prevent verified checkpoint reuse. Apply the patch over this
existing directory on Seagull:

```text
/home/users/mschpc/2025/czhang4/coupled_lob_r32_activity_regime_20260801
```

The continuation requires `ACTIVITY_SCALE=1.25`, because the selected result
is the R32 `scale_1p25` branch. A different value is a different simulation
command and must not be used with these checkpoints.

## Changed files

- `scripts/evaluate_strict_model_validation.py`
- `scripts/calibrate_queue_reactive_model.py`
- `scripts/resolve_queue_reactive_case_artifact.py`
- `tests/test_strict_model_validation.py`
- `tests/test_queue_reactive_calibration_driver.py`
- `tests/test_resolve_queue_reactive_case_artifact.py`
- `submit_queue_reactive_full_validation_hpc.sh`

Regenerate `SOURCE_MANIFEST.sha256` only after applying the patch to the full
R32 tree; the small overlay archive deliberately does not replace that
full-tree manifest.

The continuation must retain R32's exact scale-1.25 executable path,
`build-seagull-r32-scale_1p25/fragmented_mpi_lob`. Resume checkpoints hash the
complete command, including that path. The launcher therefore reconfigures
that existing build directory; do not override it with a new build path when
continuing the R32 scale-1.25 result.

The pool's `heldout_common.csv` supplies only the frozen runtime inputs and
2020 opening state. The development-validation target configuration is
resolved separately from `pooling_provenance.json:heldout.source_config` and
verified against `source_config_sha256`. Because the hash-bound source rows
retain extraction-host paths, a separate evaluator-only configuration rebases
`target_data_dir` onto `heldout.target_root`; all 1,480 target directories,
manifest symbols and 2020-01-30 dates are checked. The source config is not
modified. This prevents pooled 2019 manifests or stale local paths from being
mistaken for 2020 empirical validation targets.

## Local verification performed before release

- Python compilation of all scripts;
- Bash syntax validation of the Slurm launcher;
- 54 focused unit tests covering both gate versions, fail-closed inputs,
  continuation/freeze semantics, held-out propagation and artifact resolution;
- an independent row-level reanalysis of all 1,480 symbols on all five R32
  training dates; and
- a negative test proving that altered residual evidence is rejected.

The supplied input archive omitted several unrelated root launchers, so a
repository-wide discovery run could not be meaningfully completed in that
truncated local tree. The Slurm launcher runs the focused regression suite
inside the complete patched R32 tree before touching the continuation.
