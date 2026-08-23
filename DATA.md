# Data inputs

> **Original data source.** The raw order-message files used to prepare these
> inputs are available from the
> [official Nasdaq TotalView--ITCH archive](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/).
> Raw messages are not redistributed here. Use or redistribution of the
> derived inputs may also be subject to Nasdaq's applicable terms.

The simulator accepts either a repeated artificial template through
`--base-config` or a complete empirical universe through `--universe-config`.

`examples/synthetic/` contains the artificial template used for the synthetic
full-session, strong-scaling and weak-scaling experiments.

`data/empirical/` contains the frozen runtime inputs used by the 1,480-book
experiments:

- `universe.csv`: one row per complete LOB and repository-relative paths to
  its empirical distributions and pooled Hawkes rates;
- `background_policy.csv`: the frozen liquidity-cluster queue-response policy
  assigned to each asset;
- `value_policy.csv`: the frozen Value Agent controls;
- `clusters.csv`: the liquidity-cluster assignment used for cluster output;
- `pooled/`: ten order-message quantity/distance distributions and one Hawkes
  rate file for every asset;
- `policies/`: the ten frozen liquidity-cluster policy groups.

The pooled inputs were estimated from the five 2019 training sessions used in
the thesis. The universe and agent controls are those used for the 30 January
2020 development-validation market. They are already in the format consumed
by the executable; reproducing the supplied experiments does not require
repeating ITCH extraction or calibration.

Raw Nasdaq ITCH messages are not included and are not needed for the reported
simulation campaigns. Run `bash scripts/validate_empirical_data.sh` to check
that every file referenced by the frozen universe and policies is present.

The repository does not contain the one-second empirical return panel required
for the final empirical-versus-simulated temporal comparison. Experiment 08
regenerates and validates only the simulated return panels.
