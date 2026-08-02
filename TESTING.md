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

Production acceptance requires a real MPI build on the target cluster. The
case-study preflight compares one-rank and production-rank state hashes using
the same cohort, model parameters and random streams. The retained preflight
records equal hashes at one and 32 ranks.

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
5. the cluster rank-equivalence preflight passes before production execution.
