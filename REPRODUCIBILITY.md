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
The empirical adequacy claim applies to the ordinary-market baseline.  The
shared-dealer correction is a post-validation counterfactual-treatment and
observation amendment: the portable-case manifest records both executable
hashes, records that no calibrated parameter changed, and does not extend the
ordinary-market validation claim to the dealer mechanism.  Rank-equivalence
and the mechanism certificate test that amendment separately before any
financial path is accepted.

## Liquidity-shock experiment

After providing `POOL_ROOT`, `SELECTION_ROOT`, `DATA_ROOT` and `EVIDENCE_ROOT`:

```bash
sbatch --export=ALL,EXPERIMENT=preflight submit_queue_reactive_case_study.sh
sbatch --export=ALL,EXPERIMENT=mechanism submit_queue_reactive_case_study.sh
sbatch --export=ALL,EXPERIMENT=financial submit_queue_reactive_case_study.sh
sbatch --nodes=2 --ntasks=32 --export=ALL,EXPERIMENT=performance submit_queue_reactive_case_study.sh
```

If all 200 financial paths completed but post-processing stopped, reuse those
hash-bound paths without rerunning MPI:

```bash
sbatch --export=ALL,SOURCE_RESULT_DIR=/absolute/path/to/queue_case_JOBID \
  submit_queue_reactive_case_analysis.sh
```

The analysis-only job verifies the source manifest, raw path records, portable
runtime hashes, executable hashes and campaign-manifest hashes before writing
new outputs below `SOURCE_RESULT_DIR/postprocessing`.  Portable manifest
schema 5 is also compatible with its original producer's omission of the
cohort identity's `schema_version` tag: the analyzer supplies only that tag in
memory after independently verifying the exact 1,480-symbol sequence and both
immutable artifact hashes.  Newly produced identities contain the tag
directly.

The rank preflight compares canonical terminal-state hashes at 1 and 16 ranks;
the performance series verifies the same realization at 1, 2, 4, 8, 16 and
32 ranks.  A fixed full-session stochastic-normalization horizon is used in
both shortened and full paths, and their common prefix must match exactly.
The mechanism phase then certifies the shared dealer before the financial
matrix is authorized. Its policy must request a bid and an ask in every one of
the 1,480 books throughout the 60-second pre-shock window. At least 95% of
books must retain both orders at every observation after immediate executions
are accounted for. The executed protocol separately requires a median quote
scale of at least 0.25, utilization no greater than 0.90, at least 5% dealer
participation in aggregate BBO depth, nonzero realized dealer inventory in at
least 25% of books, and absorption of at least 2.5% of the executed stress
quantity. Target-book bid participation is not used as a gate because the
inventory-adverse intervention contains both buys and sells; realized
absorption is the side-neutral materiality test. These conditions distinguish
universal quoting from material trading participation. Inventory is generated
by fills and is not assigned artificially.

The frozen stress selects 10% of the books by a cluster-stratified rule.  Each
selected book receives an aggressive order equal to three times its held-out
opening best-bid depth.  Its side is chosen to worsen the shared dealer's
pre-shock inventory: a buy when the dealer is short and a sell otherwise.
This reference quantity and direction are fixed before the counterfactual
treatments, so the shock cannot change with the dealer's contemporaneous
liquidity.  The analysis additionally compares every asset-level shock-dose
manifest across global, uncoupled and shared-dealer-absent paths.  The shared
dealer displays one pooled empirical median limit-order-size equivalent on
each side of every book and retains price--time priority when its desired
price and size are unchanged.  Both quote size and local capacity use the
symbol's empirical quote-size proxy.  The global and uncoupled treatments have
the same nominal aggregate capacity; the latter applies it independently by
asset.  The two capacity treatments are 800 and 1,600 units per asset, with an
800-unit mean local inventory-skew scale.  The performance phase times the
same full-session, shock-enabled realization at 1--32 ranks.  The financial
phase uses 20 paired shock/control seeds at 16 ranks (80 global, 80
capacity-matched uncoupled and 40 shared-dealer-absent paths) and records
market-wide and per-cluster fixed-clock series.

## Interpretation boundary

The reported campaign passed the rank-equivalence, horizon-prefix and
mechanism gates before executing the financial matrix. Compact result evidence
is stored in `results/final-case-study/`, and the academic interpretation is
in `docs/case-study/final_results_report.md`. The full external archive is
identified by its SHA-256 digest; it is not required for source compilation,
but it is required to reproduce the independent time-resolved analysis.
