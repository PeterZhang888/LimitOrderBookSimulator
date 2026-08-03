#!/usr/bin/env python3
"""Guards for the common Seagull build used by all retained MPI workflows."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "scripts" / "seagull_deterministic_build.sh"
BUILDING_LAUNCHERS = (
    ROOT / "submit_cluster_value_agent_calibration.sh",
    ROOT / "submit_queue_reactive_full_validation_hpc.sh",
    ROOT / "submit_queue_reactive_case_study.sh",
)


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DeterministicBuildContractTest(unittest.TestCase):
    def test_every_mpi_launcher_uses_the_shared_build_contract(self) -> None:
        for script in BUILDING_LAUNCHERS:
            source = script.read_text(encoding="utf-8")
            self.assertIn("scripts/seagull_deterministic_build.sh", source)
            self.assertTrue(
                'source "${PROJECT_DIR}/scripts/seagull_deterministic_build.sh"'
                in source
                or 'source "${DETERMINISTIC_BUILD_CONTRACT}"' in source
            )
            self.assertEqual(
                source.count("lob_deterministic_configure_and_build"), 1
            )
            self.assertNotIn('cmake -S "${PROJECT_DIR}"', source)

    def test_launchers_pin_the_same_compiler_and_mpi_modules(self) -> None:
        required = (
            "gcc/15.2.0-gcc-8.5.0-r7c4jsu",
            "openmpi/5.0.9-gcc-15.2.0-2irqibq",
            "cmake/3.31.9-gcc-15.2.0-ylutpfi",
            "ninja/1.13.0-gcc-15.2.0-nukwcsd",
        )
        for script in BUILDING_LAUNCHERS:
            source = script.read_text(encoding="utf-8")
            for module in required:
                self.assertIn(module, source)

    def test_contract_contains_binary_affecting_cmake_controls(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        for token in (
            'LOB_DETERMINISTIC_BUILD_CONTRACT_VERSION="seagull_release_mpi_v1"',
            "-DCMAKE_BUILD_TYPE=Release",
            '-DCMAKE_CXX_COMPILER="${compiler_path}"',
            "-DCMAKE_CXX_COMPILER_LAUNCHER=",
            '-DCMAKE_BUILD_RPATH="${mpi_lib_dir}"',
            '-DCMAKE_INSTALL_RPATH="${mpi_lib_dir}"',
            "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
            "-DLOB_REQUIRE_MPI=ON",
            "-DLOB_BUILD_TESTS=ON",
        ):
            self.assertIn(token, source)
        for variable in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"):
            self.assertIn(variable, source)

    def test_build_and_cohort_contracts_are_hashed(self) -> None:
        pool = load_module(
            "pool_build_contract_test",
            ROOT / "scripts" / "pool_multiday_empirical_universe.py",
        )
        calibration = load_module(
            "calibration_build_contract_test",
            ROOT / "scripts" / "calibrate_cluster_value_agents.py",
        )
        for relative in (
            "scripts/seagull_deterministic_build.sh",
            "scripts/certification_cohort.py",
            "config/certification_symbols_1480.txt",
            "config/certification_symbols_1480_origin.json",
            "submit_queue_reactive_full_validation_hpc.sh",
            "submit_queue_reactive_case_study.sh",
        ):
            self.assertIn(relative, pool.WORKFLOW_SEMANTICS_FILES)
            self.assertIn(relative, calibration.WORKFLOW_SEMANTICS_FILES)


if __name__ == "__main__":
    unittest.main()
