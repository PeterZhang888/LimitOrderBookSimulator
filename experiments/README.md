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

The OpenMP experiment first performs one full cyclic preparation run to
measure the work associated with each complete book. It converts that output
into the scheduling-cost input used by all weighted-static cells. The
preparation output is kept under `cost_preparation/` and is not a timed
treatment.

Two focused launchers provide the matched process--thread comparisons used to
interpret the OpenMP results:

```bash
mkdir -p slurm results/seagull
sbatch experiments/07_mpi_openmp/submit_64core_mpi_hybrid_pair.sh
sbatch experiments/07_mpi_openmp/submit_16core_mpi_openmp_pair.sh
```

The first compares 64 MPI ranks with 32 MPI ranks times two OpenMP threads on
the same four nodes. The second compares 16 MPI ranks with one MPI-free
process times 16 OpenMP threads on the same node and physical cores. Each job
uses seven alternating-order pairs, validates CPU placement, and directly
compares the complete scientific CSV outputs before reporting timing.

Experiment 09 writes the full rank-local simulated return panels used for the
temporal diagnostics. The derived empirical comparison panel is not present
in the supplied runtime-data archives; see `DATA.md`.
