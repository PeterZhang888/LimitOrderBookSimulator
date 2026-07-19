#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

namespace dlob {

struct RunConfig {
    int duration_seconds = 5;
    int sync_window_us = 10'000;
    int population_scale = 1;
    int market_makers = 3;
    int momentum_traders = 600;
    int informed_traders = 290;
    int institutional_traders = 10;
    int expected_ranks = 0;
    std::uint64_t seed = 12345;
    std::string profile = "debug";
    std::string output_dir = "results/debug";
};

RunConfig parse_run_config(int argc, char** argv);
void print_usage(std::ostream& output, const char* program_name);

} // namespace dlob
