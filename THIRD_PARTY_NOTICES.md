# Third-party notices

## SplitMix64

The following files contain C++ adaptations of the SplitMix64 transition or
mixing steps written by Sebastiano Vigna:

- `include/common/AgentUtilities.hpp`;
- `include/simulation/MultiAssetTypes.hpp`;
- `src/exchange/BackgroundHawkesAgent.cpp`.

Source and licence notice:
<https://prng.di.unimi.it/splitmix64.c>

The reference implementation is dedicated to the public domain and expressly
permits use, copying, modification, and distribution. Local adaptations use
C++ integer types and project-specific state, naming, and domain separation.

No other third-party source code was identified in this repository by the
available provenance audit.

The project builds against software supplied by the user's system:

- a C++20 standard library and compiler;
- CMake;
- an MPI implementation such as Open MPI;
- an OpenMP implementation supplied by the compiler toolchain;
- Python and its standard library.

Those components remain governed by their own licences and are not redistributed
here. Nasdaq TotalView--ITCH data and specifications remain attributable to
Nasdaq. Raw ITCH archives are not included.
