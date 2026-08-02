// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
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
    std::string data_dir = "data";

    bool event_driven = true;
    bool use_shared_snapshot = true;
    int market_maker_batch_horizon_us = 100'000;
    int momentum_batch_horizon_us = 250'000;
    int informed_batch_horizon_us = 250'000;
    int institutional_batch_horizon_us = 1'000'000;

    double market_maker_interval_ms = 20.0;
    int market_maker_min_spread_ticks = 2;
    double momentum_rate_per_second = 0.20;
    double momentum_threshold_ticks = 0.25;
    double informed_rate_per_second = 0.05;
    double informed_signal_precision = 1.0;
    double institutional_rate_per_second = 0.01;
    double institutional_participation_cap = 0.10;
    double max_wall_seconds = 0.0;
};

RunConfig parse_run_config(int argc, char** argv);
void print_usage(std::ostream& output, const char* program_name);

} // namespace dlob
