cmake_minimum_required(VERSION 3.20)
project(whole_book_mpi_lob LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LOB_REQUIRE_MPI
       "Fail configuration when a real MPI implementation is unavailable" OFF)
option(LOB_ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LOB_BUILD_TESTS "Build correctness tests" ON)

if(LOB_REQUIRE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
else()
    find_package(MPI QUIET COMPONENTS CXX)
endif()

# This is the dependency closure of the simulator used by the empirical
# calibration, validation, scaling and inventory-stress workflows.  Historical
# alternative simulators and agents are intentionally not part of this release.
add_library(dlob_core
    src/common/EmpiricalDistribution.cpp
    src/exchange/BackgroundHawkesAgent.cpp
    src/exchange/DistributedLimitOrderBook.cpp
    src/simulation/FragmentedMpiSimulator.cpp
    src/simulation/MultiAssetConfiguration.cpp
    src/simulation/QueueReactiveBackgroundPolicy.cpp
)

target_include_directories(dlob_core PUBLIC include)
target_compile_options(dlob_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(MPI_CXX_FOUND)
    target_link_libraries(dlob_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(dlob_core PUBLIC LOB_REAL_MPI_BUILD=1)
else()
    if(LOB_REQUIRE_MPI)
        message(FATAL_ERROR "LOB_REQUIRE_MPI=ON but MPI C++ was not found")
    endif()
    message(WARNING
        "MPI was not found. Building single-process correctness mode only.")
    target_compile_definitions(dlob_core PUBLIC LOB_FORCE_MPI_STUB=1)
endif()

if(LOB_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dlob_core PUBLIC
        -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(dlob_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(fragmented_mpi_lob src/fragmented_mpi_main.cpp)
target_link_libraries(fragmented_mpi_lob PRIVATE dlob_core)
target_compile_options(fragmented_mpi_lob PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(LOB_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        function(add_lob_test target source test_name)
            add_executable(${target} ${source})
            target_link_libraries(${target} PRIVATE dlob_core)
            add_test(NAME ${test_name} COMMAND ${target})
        endfunction()

        add_lob_test(test_lob_reports
            tests/test_lob_reports.cpp lob_reports)
        add_lob_test(test_trade_execution
            tests/test_trade_execution.cpp trade_execution)
        add_lob_test(test_cancel_nearest_level
            tests/test_cancel_nearest_level.cpp cancel_nearest_level)
        add_lob_test(test_background_quote_improvement
            tests/test_background_quote_improvement.cpp
            background_quote_improvement)
        add_lob_test(test_background_hawkes_stream
            tests/test_background_hawkes_stream.cpp background_hawkes_stream)
        add_lob_test(test_queue_reactive_background_policy
            tests/test_queue_reactive_background_policy.cpp
            queue_reactive_background_policy)
        add_lob_test(test_trade_tape_hash
            tests/test_trade_tape_hash.cpp trade_tape_hash)
        add_lob_test(test_fragmented_calibration_output
            tests/test_fragmented_calibration_output.cpp
            fragmented_calibration_output)
        add_lob_test(test_fragmented_shared_dealer_coverage
            tests/test_fragmented_shared_dealer_coverage.cpp
            fragmented_shared_dealer_coverage)
        add_lob_test(test_fragmented_horizon_prefix
            tests/test_fragmented_horizon_prefix.cpp
            fragmented_horizon_prefix)
        add_lob_test(test_fragmented_model_semantics
            tests/test_fragmented_model_semantics.cpp
            fragmented_model_semantics)

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            foreach(test_target IN ITEMS
                    test_lob_reports
                    test_trade_execution
                    test_cancel_nearest_level
                    test_background_quote_improvement
                    test_background_hawkes_stream
                    test_queue_reactive_background_policy
                    test_trade_tape_hash
                    test_fragmented_calibration_output
                    test_fragmented_shared_dealer_coverage
                    test_fragmented_horizon_prefix
                    test_fragmented_model_semantics)
                target_compile_options(${test_target} PRIVATE -UNDEBUG)
            endforeach()
        endif()

        add_test(
            NAME fragmented_mpi_smoke_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 2
                --assets 8
                --window-ms 1000
                --base-config
                    ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --shock
                --shock-time-seconds 1
                --metrics-csv
                    ${CMAKE_BINARY_DIR}/fragmented_mpi_smoke_metrics.csv
        )
        set_tests_properties(fragmented_mpi_smoke_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            LABELS empirical)

        add_test(
            NAME fragmented_local_mm_spread_response_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --assets 4
                --window-ms 1000
                --base-config
                    ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --disable-shared-mm
                --disable-value-agent
                --local-mm-improvement-probability 0.25
                --local-mm-spread-elasticity 1.0
                --local-mm-max-improvement-probability 1.0
        )
        set_tests_properties(fragmented_local_mm_spread_response_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            LABELS empirical)

        add_test(
            NAME fragmented_local_mm_invalid_spread_elasticity_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --assets 4
                --base-config
                    ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --local-mm-spread-elasticity -0.1
        )
        set_tests_properties(
            fragmented_local_mm_invalid_spread_elasticity_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            WILL_FAIL TRUE)

        add_test(
            NAME fragmented_local_mm_invalid_probability_cap_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --assets 4
                --base-config
                    ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --local-mm-improvement-probability 0.5
                --local-mm-max-improvement-probability 0.25
        )
        set_tests_properties(
            fragmented_local_mm_invalid_probability_cap_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            WILL_FAIL TRUE)

        foreach(policy_variant IN ITEMS legacy periodic news recheck gap)
            if(policy_variant STREQUAL "legacy")
                set(policy_file fragmented_value_policy.csv)
            else()
                set(policy_file fragmented_value_policy_${policy_variant}.csv)
            endif()
            add_test(
                NAME fragmented_value_policy_${policy_variant}_local
                COMMAND fragmented_mpi_lob
                    --duration-seconds 1
                    --universe-config
                        ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                    --window-ms 1000
                    --disable-shared-mm
                    --disable-local-mm
                    --value-agent-policy-csv
                        ${CMAKE_SOURCE_DIR}/tests/data/${policy_file}
            )
            set_tests_properties(
                fragmented_value_policy_${policy_variant}_local PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                TIMEOUT 120
                LABELS empirical)
        endforeach()

        add_test(
            NAME fragmented_value_policy_invalid_trigger_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --universe-config
                    ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --window-ms 1000
                --disable-shared-mm
                --disable-local-mm
                --value-agent-policy-csv
                    ${CMAKE_SOURCE_DIR}/tests/data/fragmented_value_policy_invalid_trigger.csv
        )
        set_tests_properties(fragmented_value_policy_invalid_trigger_local
            PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            WILL_FAIL TRUE)

        add_test(
            NAME fragmented_value_policy_invalid_rechecks_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --universe-config
                    ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --window-ms 1000
                --disable-shared-mm
                --disable-local-mm
                --value-agent-policy-csv
                    ${CMAKE_SOURCE_DIR}/tests/data/fragmented_value_policy_invalid_rechecks.csv
        )
        set_tests_properties(fragmented_value_policy_invalid_rechecks_local
            PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            WILL_FAIL TRUE)

        foreach(invalid_gap_variant IN ITEMS
                gap_elasticity gap_cap zero_gap_threshold)
            add_test(
                NAME fragmented_value_policy_invalid_${invalid_gap_variant}_local
                COMMAND fragmented_mpi_lob
                    --duration-seconds 1
                    --universe-config
                        ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                    --window-ms 1000
                    --disable-shared-mm
                    --disable-local-mm
                    --value-agent-policy-csv
                        ${CMAKE_SOURCE_DIR}/tests/data/fragmented_value_policy_invalid_${invalid_gap_variant}.csv
            )
            set_tests_properties(
                fragmented_value_policy_invalid_${invalid_gap_variant}_local
                PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                TIMEOUT 120
                WILL_FAIL TRUE)
        endforeach()

        set_tests_properties(fragmented_calibration_output PROPERTIES
            LABELS empirical)

        find_package(Python3 QUIET COMPONENTS Interpreter)
        if(Python3_Interpreter_FOUND)
            add_test(
                NAME workflow_contracts_python
                COMMAND ${Python3_EXECUTABLE} -m unittest discover
                    -s tests -p test_*.py)
            set_tests_properties(workflow_contracts_python PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endif()
    endif()
endif()
