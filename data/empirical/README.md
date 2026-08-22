# Frozen empirical runtime inputs

This directory is the self-contained input package for the thesis's
1,480-book simulation campaigns. It contains derived distributions, pooled
Hawkes rates and frozen agent policies in the exact file layout read by the
simulator.

The five 2019 training sessions are 30 January, 27 March, 30 July, 30 October
and 30 December. The frozen universe and agent controls are those used for the
30 January 2020 development-validation market. No fitting is performed when an
experiment is submitted.

The files are organised as follows:

- `universe.csv` defines all 1,480 complete LOBs;
- `pooled/` contains each asset's order-message distributions and Hawkes rates;
- `background_policy.csv` maps every asset to one frozen cluster policy;
- `policies/` contains the ten cluster policies and their improvement
  distributions;
- `value_policy.csv` contains the Value Agent controls;
- `clusters.csv` contains the liquidity-cluster labels.

Raw Nasdaq ITCH message files are not included. They are not read by the
simulation executables and are not needed to repeat the reported simulation
experiments.
