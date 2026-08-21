# Experiment submission

Run `scripts/build_seagull.sh` once, then submit an experiment from the
repository root. The common empirical variables are:

```bash
export PROJECT_DIR="$PWD"
export UNIVERSE_CONFIG=/path/to/frozen_universe.csv
export BACKGROUND_MODEL=queue-reactive-v1
export BACKGROUND_POLICY_CSV=/path/to/background_policy.csv
export VALUE_POLICY_CSV=/path/to/value_agent_policy.csv
export REPETITIONS=7
```

The included artificial input needs no external data. Submit one complete
10,000-book session with:

```bash
mkdir -p slurm results/seagull
sbatch experiments/00_full_synthetic/submit_seagull.sh
```

For synthetic strong scaling, use:

```bash
export BASE_CONFIG="$PWD/examples/synthetic/templates.csv"
export ASSET_COUNT=10000
export BACKGROUND_MODEL=legacy
```

The control in experiments 01, 03--07, 09 and 10 is cyclic ownership,
synchronous observations, blocking `MPI_Allreduce`, one thread per rank and no
other optimisation. Experiment 08 uses a cyclic, buffered blocking control
because bounded lookahead requires buffered observational output. It contains
the four factorial cells: blocking, nonblocking, lookahead, and nonblocking
plus lookahead.

Create `slurm/` before submission because Slurm opens the output file before
the script starts:

```bash
mkdir -p slurm results/seagull
sbatch experiments/04_rank_ownership/submit_seagull.sh
```
