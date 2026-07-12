# Integrated Closed-Loop LOB Strategy Framework

This package is not a toy wrapper. The experimental market maker is inserted directly into the existing Hawkes-background limit order book simulator:

1. background Hawkes events and background agents generate market activity;
2. the experimental strategy observes the current top-of-book state and Hawkes intensity;
3. the strategy submits bid/ask limit orders into the real `LimitOrderBook` with an owner id;
4. the real matching engine processes market orders, cancellations and fills;
5. fills are returned through `get_and_clear_execution_reports(owner_id)` and update strategy inventory, cash and PnL.

The background configuration is the calibrated full-day configuration from the empirical-moment search. The simulation horizon is 23,400 seconds per day.

## Compile on Callan

```bash
cd /home/users/mschpc/2025/czhang4/thesis_LOB/integrated_lob_strategy_framework
bash compile.sh
```

## Run train/validation/test for one strategy

Strategy ids:

- `1`: Inventory Linear-Skew Market Maker
- `2`: Queue-Aware Fill-Hazard Market Maker
- `3`: Hawkes Signal-Hazard Market Maker

```bash
sbatch slurm/submit_tuner.sh 1
sbatch slurm/submit_validator.sh 1
bash slurm/submit_test.sh 1 100
```

## Run all strategies

```bash
for sid in 1 2 3; do
    sbatch slurm/submit_tuner.sh $sid
done

# after tuner jobs finish
for sid in 1 2 3; do
    sbatch slurm/submit_validator.sh $sid
done

# after validator jobs finish
for sid in 1 2 3; do
    bash slurm/submit_test.sh $sid 100
done
```

## Outputs

- `results/training_metrics_strategy_<ID>.csv`
- `configs/top3_strategy_<ID>.txt`
- `results/validation_metrics_strategy_<ID>.csv`
- `configs/best_config_strategy_<ID>.txt`
- `results/final_test_<ID>_<N>ranks.csv`
- `logs/quote_log_<ID>_1.csv` for rank 1 on the first test day

## SLURM allocation

The test wrapper uses:

```bash
sbatch --ntasks=$NTASKS --cpus-per-task=1 ...
```

It does not hard-code nodes or tasks per node.
