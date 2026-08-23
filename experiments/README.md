# Experiment submission

Each experiment directory provides the same two-file interface:

```bash
bash experiments/NN_experiment/compile.sh
bash experiments/NN_experiment/submit.sh
```

`compile.sh` calls the central verified build; it does not create a separate
copy of the simulator. `submit.sh` launches that experiment's existing formal
Seagull submission. The empirical submission files automatically use:

```text
data/empirical/universe.csv
data/empirical/background_policy.csv
data/empirical/value_policy.csv
data/empirical/clusters.csv
```

No external input paths are required. The main formal performance jobs use
seven full-session repetitions unless their submission file states otherwise;
weak scaling uses three. The stylised-fact job and the standalone 10,000-book
run use one complete session by design.

For a fresh-clone validation using no more than two nodes and one completed
run per configuration, use `bash scripts/submit_seagull_validation.sh` from
the repository root. This validation keeps the full simulated session but is
not a replacement for the repeated performance campaigns.

The included artificial input needs no external data. Submit one complete
10,000-book session with:

```bash
mkdir -p slurm results/seagull
bash experiments/00_full_synthetic/submit.sh
```

For synthetic strong scaling, use:

```bash
bash experiments/01_strong_scaling/submit.sh
```

The strong-scaling submission file dispatches one independently sized Slurm
job for each rank count from 1 to 256. This avoids reserving 16 nodes while a
small-rank case is running and prevents the complete seven-repetition sweep
from exceeding the time limit of one allocation. The submitted job numbers
and result directories are written to
`results/seagull/strong_scaling_<date>/submitted_jobs.csv`.

The control in experiments 01, 03--05, 07, 09 and 10 is cyclic ownership,
synchronous observations, blocking `MPI_Allreduce`, one thread per rank and no
other optimisation. Experiment 08 uses a cyclic, buffered blocking control
because bounded lookahead requires buffered observational output. It contains
the four factorial cells: blocking, nonblocking, lookahead, and nonblocking
plus lookahead. The current certificate skips at most the next risk reduction;
the following boundary synchronises normally.

Create `slurm/` before submission because Slurm opens the output file before
the script starts:

```bash
mkdir -p slurm results/seagull
bash experiments/04_rank_ownership/submit.sh
```

The formal OpenMP experiment first performs one full cyclic preparation run
at each total core count to measure the work associated with every complete
book. It converts that output into the common scheduling-cost input used by
all permanent-owner cells. The preparation output is kept under
`cost_preparation/` and is not a timed treatment. Submit the complete 16- and
32-core decomposition matrices with:

```bash
mkdir -p slurm results/seagull
bash experiments/07_mpi_openmp/submit.sh
```

MPI ownership remains cyclic. In every threaded layout, measured book costs
are used to assign each rank's books to threads once, and the assignment is
retained for the complete session. Each total-core job uses seven rotating
blocks, validates CPU placement and directly checks scientific outputs before
reporting timing. Each timed launch first has to pass the declared small-
`MPI_Allreduce` latency gate. Every successfully completed repetition is
retained regardless of its execution time. Preflight rejections and completed
runs are recorded in `attempts.csv`; seven completed repetitions are required
for every layout.
Runtime ratios above 1.15 are reported as variability warnings without
discarding repetitions or failing an otherwise valid job. Per-asset files,
counts, accounting values and all
non-spread metric columns must agree exactly. The three derived mean-spread
columns use the explicitly reported floating-point tolerance documented in
the main README because different MPI rank layouts change only the association
order of their double-precision sums.

The additional pair and window-profile files in `07_mpi_openmp/` are focused
diagnostics retained to explain the phase-based control and the origin of its
overhead. They are not required to reproduce the final decomposition matrix.

Experiment 09 writes the full rank-local simulated return panels used for the
temporal diagnostics. The derived empirical comparison panel is not present
in the supplied runtime-data archives; see `DATA.md`.
