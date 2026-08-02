# R30 persistent-liquidity-regime validation release

## Evidence for the repair

R29 job 45483 behaved correctly as a no-waste pilot: every MPI realization
completed, rank equivalence passed, and the 25-run matrix was never started.
The model was rejected scientifically. At the identical base seed, its
market-order-only regime changed mean absolute-return ACF by only 0.00018 to
0.00070 relative to R28, against remaining errors of 0.027 to 0.050. Increasing
that same loading would therefore be an unsupported parameter adjustment.

## R30 mechanism

R30 retains one bounded coupling per liquidity cluster, estimated from the
five 2019 training sessions. The standardized persistent log-variance state
now changes all liquidity-removing background types: market buys, market
sells, bid cancellations and ask cancellations. Limit additions dominate
quiet regimes. The complete six-type vector is renormalized, preserving the
accepted-event clock and the empirically calibrated total event rate. A zero
coupling remains the exact legacy path.

This is the smallest structural change consistent with the R29 evidence:
cancellations are a major route to queue depletion, whereas market orders
alone did not transmit the latent regime to realized LOB volatility.

The R28 residual-derived cluster vector remains the training-only starting
estimate. `COUPLING_SCALE` supplies one global regularized sensitivity value;
it does not open ten independent cluster searches. Its default is 1.0, and
the accepted range is [0.25, 4]. Any alternative must be selected using only
the 2019 pilot, never the 2020 development-validation date.

## Execution contract

Before the 25-run strict matrix, the Slurm job performs:

1. a SHA-256 audit of every packaged source/configuration file;
2. a deterministic R30 build and mechanism/protocol tests;
3. one-rank versus 32-rank state-hash equivalence;
4. a three-date, one-seed, 1,480-symbol directional pilot.

If the pilot rejects the mechanism, the job stops in about the pilot runtime.
If it passes, one fixed parameter set is assessed on five 2019 dates and five
seeds. Automatic full-session refinement is disabled. Only a passed 2019
freeze can open the frozen 2020 development-validation run.

## Scientific status

Compilation and local mechanism/protocol tests are necessary but not empirical
validation. R30 is adequate only if the unchanged 2019 strict gate and the
subsequent 2020 development-validation gate write the final handoff. No
threshold is weakened and a rejected model never creates that handoff.

## Seagull submission

```bash
BASE=/home/users/mschpc/2025/czhang4
PROJECT="$BASE/coupled_lob_r30_liquidity_regime_20260801"
SELECTION_ROOT="$BASE/coupled_lob_r27_target_protocol_20260801/results/seagull/queue_selection_45480"
POOL_ROOT="$BASE/coupled_lob_r26_grid_projection_20260801/results/seagull/five_day_pool_45477"

cd "$PROJECT"
mkdir -p slurm results/seagull

VALIDATION_JOB=$(sbatch --parsable \
  --export="ALL,SELECTION_ROOT=$SELECTION_ROOT,POOL_ROOT=$POOL_ROOT,COUPLING_SCALE=1.0" \
  submit_queue_reactive_full_validation_hpc.sh)
VALIDATION_JOB="${VALIDATION_JOB%%;*}"
echo "VALIDATION_JOB=$VALIDATION_JOB"
```

Monitor:

```bash
squeue -j "$VALIDATION_JOB" -o "%.18i %.24j %.10T %.12M %.50R"
sacct -j "$VALIDATION_JOB" \
  --format=JobID,JobName,State,ExitCode,Elapsed,ReqCPUS,MaxRSS,NodeList
```

Interpret:

```bash
RESULT="$PROJECT/results/seagull/queue_full_validation_${VALIDATION_JOB}"

test -s "$RESULT/full_training_adequacy/directional_pilot/pilot_decision.json" \
  && python3 -m json.tool \
     "$RESULT/full_training_adequacy/directional_pilot/pilot_decision.json"

if test -s "$RESULT/development_validation/heldout_run_manifest.json"; then
  echo "2019 STRICT TRAINING AND 2020 DEVELOPMENT VALIDATION: PASS"
elif test -s "$RESULT/full_training_adequacy/expanded_training_freeze.json"; then
  echo "2019 PASSED; 2020 DID NOT PASS OR COMPLETE"
else
  echo "NO HANDOFF: inspect the pilot or strict-training diagnostics"
fi
```

Do not change `COUPLING_SCALE` after examining 2020 output. If scale 1.0 is
rejected by the 2019 pilot, preserve that result before considering one
predeclared training-only sensitivity value.
