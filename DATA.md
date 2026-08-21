# Data inputs

The simulator accepts either a repeated artificial template through
`--base-config` or a complete empirical universe through `--universe-config`.

`examples/synthetic/` is artificial and redistributable. It contains the
quantity, distance and background-rate files needed for a complete run.

The empirical thesis experiments require separately supplied Nasdaq-derived
inputs:

- the frozen universe CSV;
- per-asset empirical distribution files referenced by that CSV;
- the frozen queue-reactive background policy CSV;
- the frozen Value Agent policy CSV;
- the liquidity-cluster CSV when cluster output is requested.

These paths are supplied through environment variables before submission and
are not embedded in the source repository.
