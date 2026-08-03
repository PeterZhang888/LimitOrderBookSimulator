# Verification

## Source-only checks

The repository supports the following checks without proprietary data:

```bash
cmake -S . -B build -DLOB_REQUIRE_MPI=OFF -DLOB_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build -LE 'empirical|mpi' --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
sha256sum -c SOURCE_MANIFEST.sha256
```

The release audit also applies `bash -n` to every shell launcher and parses all
Python sources before packaging.

## MPI acceptance checks

Production acceptance requires a real MPI build on the target cluster.  The
liquidity-shock preflight compares 1- and 16-rank state hashes; the performance
job additionally checks 32 ranks using the same cohort, model parameters and
random streams.  A second, full-horizon
mechanism preflight verifies that the shared dealer remains active immediately
before the shock and absorbs a material quantity when it arrives. Acceptance
requires two-sided requested coverage in all 1,480 books, at least 95%
two-sided resting coverage after executions, economic capacity headroom,
market-wide BBO participation, realized inventory dispersion and at least
2.5% realized shock absorption. Neither preflight result is inferred from the
superseded diagnostic campaign.

The `fragmented_horizon_prefix` regression fixes the full-session
normalization horizon and proves exact equality between a shortened path and
the corresponding full-run prefix.  It also verifies that enabling dormant
shared-dealer treatment controls leaves the shared-dealer-off baseline
unchanged.  Financial post-processing rejects unequal asset-level shock-dose
manifests, unmatched capacity controls or incomplete fill ownership.

## Data-dependent checks

Tests that open empirical mark distributions, Hawkes-rate files or reconstructed
ITCH targets require the external data described in `DATA.md`. Absence of those
fixtures is reported as a skip by Python tests. CTest registers the corresponding
C++ tests with the `empirical` label; run them with `ctest -L empirical` after
installing the data. Multi-process launcher tests carry the `mpi` label and must
run in an environment where MPI may open its control sockets.

## Release criteria

A release is accepted only when:

1. source-manifest verification passes;
2. configuration and compilation succeed;
3. source-only C++ and Python tests pass;
4. shell and Python syntax checks pass;
5. the cluster rank-equivalence preflight passes before production execution;
6. the shared-dealer mechanism certificate passes before financial execution.
