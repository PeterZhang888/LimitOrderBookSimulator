# R31 stochastic-baseline repair

## Why this change exists

The R28 full-universe residuals localized the main error in lag-one absolute
return autocorrelation to liquidity clusters 2--5 and 9.  R29 and R30 attempted
to transmit the persistent state by changing the mixture of accepted event
types.  The Hawkes stream deliberately renormalized that mixture, so total
hazard and every accepted timestamp were conserved.  R29 therefore barely
moved the target, while R30 increased cancellations and made the October pilot
substantially slower without supplying the missing active/quiet event clock.

R31 removes that production coupling.  For clusters with a positive frozen
loading, Hawkes immigration in second `k` is instead multiplied by

```
exp(H_k - 0.5 * sigma_H^2),
H_k = rho H_(k-1) + sigma_H sqrt(1-rho^2) epsilon_k.
```

Here `rho` is the frozen persistence and
`sigma_H = fundamental_log_variance_std * fundamental_order_flow_coupling`.
The lognormal correction gives an unconditional multiplier mean of one.  No
new fitted coefficient is introduced, excitation stays additive and
subcritical, and a zero loading remains bit-compatible with the earlier clock.

This is a Cox--Hawkes activity modulation, not a claim that one extra agent
solves every book-level discrepancy.  It follows the literature's distinction
between state-dependent event composition and stochastic/event-history-driven
intensity.  Relevant precedents are Huang, Lehalle and Rosenbaum's
queue-reactive model; Wu, Rambaldi, Muzy and Bacry's queue-reactive Hawkes
model; Morariu-Patrichi and Pakkanen's state-dependent Hawkes framework; and
Yu and Potiron's stochastic-baseline Hawkes model.

## Evidence completed locally

- The release builds with AppleClang 17 and Open MPI 5.0.9.
- The Hawkes clock and fragmented-model semantic tests pass.
- All 29 calibration-driver and R31 protocol tests pass.
- In a fixed-seed clock diagnostic with the same nominal 15 events/second,
  the constant baseline had lag-one one-second count autocorrelation 0.0256;
  the deliberately strong stochastic-baseline test had 0.8718 and mean 13.12.
  This verifies that R31 changes the missing mechanism, not that empirical
  validation has already passed.

Some unrelated legacy CTest targets require data files omitted from this
source-only release or multi-process sockets disallowed in the local sandbox.
They are not counted as R31 evidence.

## Fail-closed Seagull workflow

The submission script defaults to `PILOT_ONLY=on`.  It builds and tests the
release, verifies rank equivalence, and runs one seed on the three predeclared
difficult 2019 dates.  It then stops.  No 25-run matrix and no 2020 development
validation are launched unless this pilot produces
`directional_pilot_handoff.json`.

Only after inspecting that handoff should the same result directory be resumed
with `PILOT_ONLY=off,RESUME=on`.  This second job applies the unchanged strict
2019 adequacy gate; it opens the 2020 inputs only if the training gate passes.

R31 is therefore a tested structural hypothesis, not a promised validation
success.  If the pilot rejects it, preserve the diagnostics and do not weaken
the frozen thresholds.
