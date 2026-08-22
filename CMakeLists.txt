cmake_minimum_required(VERSION 3.20)
project(whole_book_market_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LOB_REQUIRE_MPI "Require a real MPI implementation" ON)
option(LOB_FORCE_MPI_STUB "Build the one-process OpenMP executable" OFF)
option(LOB_ENABLE_OPENMP "Enable rank-local OpenMP worksharing" ON)
option(LOB_OPENMP_WINDOW_ONLY
    "Use OpenMP only for expensive event-window processing" OFF)
option(LOB_BUILD_TESTS "Build correctness tests" ON)

if(LOB_REQUIRE_MPI AND LOB_FORCE_MPI_STUB)
    message(FATAL_ERROR
        "LOB_REQUIRE_MPI and LOB_FORCE_MPI_STUB cannot both be enabled")
elseif(LOB_FORCE_MPI_STUB)
    set(MPI_CXX_FOUND FALSE)
elseif(LOB_REQUIRE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
else()
    find_package(MPI QUIET COMPONENTS CXX)
endif()

if(LOB_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

add_library(lob_core
    src/common/EmpiricalDistribution.cpp
    src/exchange/BackgroundHawkesAgent.cpp
    src/exchange/LimitOrderBook.cpp
    src/simulation/DistributedMarketSimulator.cpp
    src/simulation/MultiAssetConfiguration.cpp
    src/simulation/QueueReactiveBackgroundPolicy.cpp
)
target_include_directories(lob_core PUBLIC include)
target_compile_options(lob_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(MPI_CXX_FOUND)
    target_link_libraries(lob_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(lob_core PUBLIC LOB_REAL_MPI_BUILD=1)
else()
    if(LOB_REQUIRE_MPI)
        message(FATAL_ERROR "MPI C++ was not found")
    endif()
    target_compile_definitions(lob_core PUBLIC LOB_FORCE_MPI_STUB=1)
endif()

if(LOB_ENABLE_OPENMP)
    target_link_libraries(lob_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lob_core PUBLIC LOB_HAS_OPENMP=1)
else()
    target_compile_definitions(lob_core PUBLIC LOB_HAS_OPENMP=0)
endif()
if(LOB_OPENMP_WINDOW_ONLY)
    target_compile_definitions(lob_core PUBLIC LOB_OPENMP_WINDOW_ONLY=1)
else()
    target_compile_definitions(lob_core PUBLIC LOB_OPENMP_WINDOW_ONLY=0)
endif()

add_executable(lob_mpi src/main.cpp)
target_link_libraries(lob_mpi PRIVATE lob_core)
target_compile_options(lob_mpi PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(LOB_FORCE_MPI_STUB)
    set_target_properties(lob_mpi PROPERTIES OUTPUT_NAME lob_openmp)
endif()

if(LOB_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        function(add_lob_test target source)
            add_executable(${target} ${source})
            target_link_libraries(${target} PRIVATE lob_core)
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(${target} PRIVATE -UNDEBUG)
            endif()
            add_test(NAME ${target} COMMAND ${target})
            set_tests_properties(${target} PROPERTIES TIMEOUT 120)
        endfunction()

        add_lob_test(test_lob_reports tests/test_lob_reports.cpp)
        add_lob_test(test_trade_execution tests/test_trade_execution.cpp)
        add_lob_test(test_background_hawkes_stream
            tests/test_background_hawkes_stream.cpp)
        add_lob_test(test_background_quote_improvement
            tests/test_background_quote_improvement.cpp)
        add_lob_test(test_cancel_nearest_level
            tests/test_cancel_nearest_level.cpp)
        add_lob_test(test_queue_reactive_background_policy
            tests/test_queue_reactive_background_policy.cpp)
        add_lob_test(test_distributed_model_semantics
            tests/test_distributed_model_semantics.cpp)
        if(LOB_ENABLE_OPENMP)
            add_lob_test(test_distributed_optimization_equivalence
                tests/test_distributed_optimization_equivalence.cpp)
        endif()
        add_lob_test(test_distributed_shared_dealer_coverage
            tests/test_distributed_shared_dealer_coverage.cpp)
    endif()
endif()
