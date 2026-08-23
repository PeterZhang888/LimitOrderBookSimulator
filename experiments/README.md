# Experiment submission

Every experiment directory provides the same interface:

```bash
bash experiments/NN_experiment/compile.sh
bash experiments/NN_experiment/submit.sh
```

`compile.sh` builds the shared simulator executables. One successful build is
enough for every experiment. `submit.sh` launches the selected formal Seagull
campaign.

Empirical runs use the repository inputs:

```text
data/empirical/universe.csv
data/empirical/background_policy.csv
data/empirical/value_policy.csv
```

`data/empirical/clusters.csv` is available when optional cluster-level output
is requested. No external input paths are required.

For a shorter full-session check covering all experiments on no more than two
nodes per job, use:

```bash
bash scripts/submit_seagull_validation.sh
```

## Synthetic campaigns

Submit one complete 10,000-book session with:

```bash
bash experiments/00_full_synthetic/submit.sh
```

The strong-scaling driver submits five workloads crossed with nine rank counts
from 1 to 256, giving 45 independently sized jobs:

```bash
bash experiments/01_strong_scaling/submit.sh
```

Job numbers and result paths are recorded in
`results/runs/strong_scaling_<timestamp>/submitted_jobs.csv`.

## Empirical campaigns

Unless a treatment changes it, empirical runs use cyclic MPI ownership,
synchronous observations, blocking risk reduction and one thread per rank.

Experiment 06 compares five MPI--OpenMP layouts at each total core count:

```text
16 cores: 16x1, 8x2, 4x4, 2x8, 1x16
32 cores: 32x1, 16x2, 8x4, 4x8, 2x16
```

A preparation run estimates the work associated with each book. Threaded
layouts then assign every book permanently to one thread. The preparation run
is not included in the timing comparison.

```bash
bash experiments/06_mpi_openmp/submit.sh
```

Each layout is repeated seven times in rotating order. CPU placement and
scientific outputs are checked before timings are summarised. A launch must
also pass the declared small-collective latency check; rejected preflights are
logged, while every completed repetition is retained. The additional pair,
profiler and collective-stall jobs are diagnostics rather than formal matrix
cells.

Experiment 07 crosses blocking and non-blocking risk reductions with bounded
lookahead disabled or enabled. All four cells use buffered observations. A
lookahead certificate may skip only the next risk boundary.

Experiment 08 writes the full simulated return panels used for temporal
stylised-fact analysis. The empirical comparison panel is not included; see
[`DATA.md`](../DATA.md).

The main performance campaigns use seven full-session repetitions unless a
submission file states otherwise. Weak scaling uses three repetitions. The
standalone synthetic and stylised-fact runs use one complete session.
