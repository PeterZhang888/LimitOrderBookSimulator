#include "common/RunConfig.hpp"

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace dlob {
namespace {

template <typename Integer>
Integer parse_integer(std::string_view text, const char* option) {
    Integer value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("Invalid value for ") + option + ": " + std::string(text));
    }
    return value;
}

double parse_double(std::string_view text, const char* option) {
    std::string value(text);
    std::size_t used = 0;
    const double parsed = std::stod(value, &used);
    if (used != value.size()) {
        throw std::invalid_argument(std::string("Invalid value for ") + option + ": " + value);
    }
    return parsed;
}

void apply_profile(RunConfig& config, const std::string& profile) {
    config.profile = profile;
    if (profile == "debug") {
        config.duration_seconds = 5;
        config.market_makers = 3;
        config.momentum_traders = 600;
        config.informed_traders = 290;
        config.institutional_traders = 10;
        config.output_dir = "results/debug";
    } else if (profile == "baseline") {
        config.duration_seconds = 600;
        config.market_makers = 3;
        config.momentum_traders = 6'000;
        config.informed_traders = 2'900;
        config.institutional_traders = 100;
        config.output_dir = "results/baseline";
    } else if (profile == "scale10") {
        config.duration_seconds = 600;
        config.population_scale = 10;
        config.market_makers = 3;
        config.momentum_traders = 6'000;
        config.informed_traders = 2'900;
        config.institutional_traders = 100;
        config.output_dir = "results/scale10";
    } else {
        throw std::invalid_argument("Unknown profile: " + profile);
    }
}

std::string require_value(int& index, int argc, char** argv, const char* option) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("Missing value after ") + option);
    return argv[++index];
}

bool is_flag(const std::string& arg) {
    return arg == "--event-driven" || arg == "--legacy-fixed-window"
        || arg == "--shared-snapshot" || arg == "--no-shared-snapshot";
}

} // namespace

RunConfig parse_run_config(int argc, char** argv) {
    RunConfig config;
    apply_profile(config, "debug");

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            apply_profile(config, require_value(i, argc, argv, "--profile"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(std::cout, argv[0]);
            std::exit(0);
        } else if (arg.rfind("--", 0) == 0 && !is_flag(arg)) {
            ++i;
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            ++i;
        } else if (arg == "--duration-seconds") {
            config.duration_seconds = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--sync-window-us") {
            config.sync_window_us = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--population-scale") {
            config.population_scale = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--market-makers") {
            config.market_makers = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--momentum") {
            config.momentum_traders = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--informed") {
            config.informed_traders = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--institutional") {
            config.institutional_traders = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--expected-ranks") {
            config.expected_ranks = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--seed") {
            config.seed = parse_integer<std::uint64_t>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--output-dir") {
            config.output_dir = require_value(i, argc, argv, arg.c_str());
        } else if (arg == "--data-dir") {
            config.data_dir = require_value(i, argc, argv, arg.c_str());
        } else if (arg == "--event-driven") {
            config.event_driven = true;
        } else if (arg == "--legacy-fixed-window") {
            config.event_driven = false;
        } else if (arg == "--shared-snapshot") {
            config.use_shared_snapshot = true;
        } else if (arg == "--no-shared-snapshot") {
            config.use_shared_snapshot = false;
        } else if (arg == "--mm-batch-horizon-us") {
            config.market_maker_batch_horizon_us = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--momentum-batch-horizon-us") {
            config.momentum_batch_horizon_us = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--informed-batch-horizon-us") {
            config.informed_batch_horizon_us = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--institutional-batch-horizon-us") {
            config.institutional_batch_horizon_us = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--mm-interval-ms") {
            config.market_maker_interval_ms = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--mm-min-spread-ticks") {
            config.market_maker_min_spread_ticks = parse_integer<int>(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--momentum-rate") {
            config.momentum_rate_per_second = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--momentum-threshold") {
            config.momentum_threshold_ticks = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--informed-rate") {
            config.informed_rate_per_second = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--informed-precision") {
            config.informed_signal_precision = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--institutional-rate") {
            config.institutional_rate_per_second = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--institutional-cap") {
            config.institutional_participation_cap = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--max-wall-seconds") {
            config.max_wall_seconds = parse_double(require_value(i, argc, argv, arg.c_str()), arg.c_str());
        } else if (arg == "--help" || arg == "-h") {
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (config.duration_seconds <= 0) throw std::invalid_argument("--duration-seconds must be positive");
    if (config.sync_window_us < 100) throw std::invalid_argument("--sync-window-us must be at least 100");
    if (config.population_scale <= 0) throw std::invalid_argument("--population-scale must be positive");
    if (config.expected_ranks < 0) throw std::invalid_argument("--expected-ranks cannot be negative");
    if (config.market_maker_interval_ms <= 0.0 || config.market_maker_min_spread_ticks <= 0
        || config.momentum_rate_per_second < 0.0 || config.momentum_threshold_ticks < 0.0
        || config.informed_rate_per_second < 0.0 || config.informed_signal_precision <= 0.0
        || config.institutional_rate_per_second < 0.0
        || config.institutional_participation_cap <= 0.0
        || config.institutional_participation_cap > 1.0
        || config.max_wall_seconds < 0.0) {
        throw std::invalid_argument("Invalid behavioural parameter value");
    }
    return config;
}

void print_usage(std::ostream& output, const char* program_name) {
    output << "Usage: " << program_name << " [options]\n\n"
           << "  --profile debug|baseline|scale10\n"
           << "  --duration-seconds N\n"
           << "  --event-driven | --legacy-fixed-window\n"
           << "  --shared-snapshot | --no-shared-snapshot\n"
           << "  --mm-batch-horizon-us N\n"
           << "  --momentum-batch-horizon-us N\n"
           << "  --informed-batch-horizon-us N\n"
           << "  --institutional-batch-horizon-us N\n"
           << "  --mm-interval-ms X\n"
           << "  --mm-min-spread-ticks N\n"
           << "  --momentum-rate X --momentum-threshold X\n"
           << "  --informed-rate X --informed-precision X\n"
           << "  --institutional-rate X --institutional-cap X\n"
           << "  --max-wall-seconds X (0 disables the event-driven guard)\n"
           << "  --market-makers N --momentum N --informed N --institutional N\n"
           << "  --expected-ranks N --seed N --data-dir PATH --output-dir PATH\n";
}

} // namespace dlob
