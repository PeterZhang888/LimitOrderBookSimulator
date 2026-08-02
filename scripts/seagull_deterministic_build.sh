#!/usr/bin/env bash
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
# Shared, source-only build contract for the certified Seagull workflow.
#
# Calibration and the final case study intentionally build in separate,
# job-specific directories.  They must nevertheless produce the same bytes:
# the case-study handoff rejects an executable whose SHA-256 differs from the
# binary used for fitting and validation.  Both submission scripts therefore
# source this file and call the same function with no additional CMake flags.

LOB_DETERMINISTIC_BUILD_CONTRACT_VERSION="seagull_release_mpi_v1"
LOB_DETERMINISTIC_SOURCE_DATE_EPOCH="1577836800"

lob_deterministic_configure_and_build() {
    if (( $# != 4 )); then
        echo "ERROR: lob_deterministic_configure_and_build needs PROJECT_DIR BUILD_DIR MPI_LIB_DIR BUILD_JOBS" >&2
        return 2
    fi
    local project_dir="$1"
    local build_dir="$2"
    local mpi_lib_dir="$3"
    local build_jobs="$4"
    local compiler_path ninja_path

    compiler_path="$(command -v mpicxx)"
    ninja_path="$(command -v ninja)"
    [[ -n "${compiler_path}" && -x "${compiler_path}" ]] || {
        echo "ERROR: mpicxx is unavailable for deterministic build" >&2
        return 2
    }
    [[ -n "${ninja_path}" && -x "${ninja_path}" ]] || {
        echo "ERROR: ninja is unavailable for deterministic build" >&2
        return 2
    }
    [[ -d "${project_dir}" && -d "${mpi_lib_dir}" ]] || {
        echo "ERROR: deterministic build received an invalid project/MPI directory" >&2
        return 2
    }
    [[ "${build_jobs}" =~ ^[0-9]+$ ]] && (( build_jobs > 0 )) || {
        echo "ERROR: deterministic build jobs must be a positive integer" >&2
        return 2
    }

    # Do not let a login-shell compiler/linker override silently change one of
    # the two binaries. SOURCE_DATE_EPOCH also gives any timestamp-aware build
    # step the same reference time.
    unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
    export LC_ALL=C
    export TZ=UTC
    export SOURCE_DATE_EPOCH="${LOB_DETERMINISTIC_SOURCE_DATE_EPOCH}"
    export ZERO_AR_DATE=1

    echo "deterministic_build_contract=${LOB_DETERMINISTIC_BUILD_CONTRACT_VERSION}"
    echo "deterministic_compiler=${compiler_path}"
    echo "deterministic_ninja=${ninja_path}"
    echo "deterministic_mpi_lib_dir=${mpi_lib_dir}"

    cmake -S "${project_dir}" -B "${build_dir}" -G Ninja \
        -DCMAKE_MAKE_PROGRAM="${ninja_path}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${compiler_path}" \
        -DCMAKE_CXX_COMPILER_LAUNCHER= \
        -DCMAKE_BUILD_RPATH="${mpi_lib_dir}" \
        -DCMAKE_INSTALL_RPATH="${mpi_lib_dir}" \
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=OFF \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DLOB_REQUIRE_MPI=ON \
        -DLOB_BUILD_TESTS=ON
    cmake --build "${build_dir}" --parallel "${build_jobs}"
}
