# Experiment submission

Run `scripts/build_seagull.sh` once, then submit an experiment from the
repository root. The empirical submission files automatically use:

```text
data/empirical/universe.csv
data/empirical/background_policy.csv
data/empirical/value_policy.csv
data/empirical/clusters.csv
```

No external input paths are required. All formal performance jobs use seven
full-session repetitions by default. The stylised-fact job and the standalone
10,000-book run use one complete session by design.

The included artificial input needs no external data. Submit one complete
10,000-book session with:

```bash
mkdir -p slurm results/seagull
sbatch experiments/00_full_synthetic/submit_seagull.sh
```

For synthetic strong scaling, use:

```bash
bash experiments/01_strong_scaling/submit_seagull.sh
```

The strong-scaling submission file dispatches one independently sized Slurm
job for each rank count from 1 to 256. This avoids reserving 16 nodes while a
small-rank case is running and prevents the complete seven-repetition sweep
from exceeding the time limit of one allocation. The submitted job numbers
and result directories are written to
`results/seagull/strong_scaling_<date>/submitted_jobs.csv`.

The control in experiments 01, 03--07, 09 and 10 is cyclic ownership,
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
sbatch experiments/04_rank_ownership/submit_seagull.sh
```

The formal OpenMP experiment first performs one full cyclic preparation run
at each total core count to measure the work associated with every complete
book. It converts that output into the common scheduling-cost input used by
all permanent-owner cells. The preparation output is kept under
`cost_preparation/` and is not a timed treatment. Submit the complete 16-,
32- and 64-core decomposition matrix with:

```bash
mkdir -p slurm results/seagull
bash experiments/07_mpi_openmp/submit_seagull.sh
```

MPI ownership remains cyclic. In every threaded layout, measured book costs
are used to assign each rank's books to threads once, and the assignment is
retained for the complete session. Each total-core job uses seven rotating
blocks, validates CPU placement and directly checks scientific outputs before
reporting timing. Each timed launch first has to pass the declared small-
`MPI_Allreduce` latency gate, and any completed repetition of at least 120
seconds is rejected. All rejected and accepted attempts are recorded in
`attempts.csv`; seven accepted repetitions are required for every layout.
Per-asset files, counts, accounting values and all
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
