# R32 session-recentred activity-regime pilot

R32 is a bounded training-only repair of the absolute-return persistence
shortfall observed in R28--R31. It does not relax any empirical adequacy gate
and it does not open the 2020 development-validation inputs during parameter
selection.

For cluster loading `theta`, the Hawkes immigration vector in one-second bin
`k` is multiplied by

```text
exp(clamp(H_k, -3 theta, 3 theta) - log Z)
```

where `H_k` is a deterministic stationary AR(1) log-activity state and `Z` is
the arithmetic mean of the bounded exponential multipliers over the exact
simulated session. Consequently, the configured session-average immigration
multiplier is exactly one. This preserves the activity level in expectation;
it does **not** claim that a realized Hawkes event count is fixed pathwise.

R31 used `theta * latent_volatility_std`, which attenuated the intended
activity loading by factors of roughly 3--50 across active clusters and did
not materially change the directional-pilot ACF. R32 uses `theta` directly.

The permitted training grid is one global scale in `{0.50, 0.75, 1.00, 1.25}`
applied to the ten frozen cluster loadings. Each candidate first runs only the
three-date, one-seed directional pilot. A rejected pilot writes its complete
decision and exits the Slurm job successfully as a completed candidate. A
build error, missing artifact, hash mismatch, MPI error or malformed result
still fails the job.

The full 25-run 2019 matrix may be resumed only from a candidate containing a
passed `directional_pilot_handoff.json`. The 2020 development validation may
run only after the unchanged full-training gate writes
`expanded_training_freeze.json`.

Use `scripts/summarize_r32_activity_grid.py` after the four pilot jobs. It
audits the recorded scale and result contract, excludes rejected candidates,
and selects among passed candidates using the predeclared minimax normalized
ACF-error rule. If no pilot passes, the script exits nonzero and the model must
not proceed to the expensive matrix.
