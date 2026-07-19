#include "common/RunConfig.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace dlob {
namespace {

template <typename Integer>
Integer parse_integer(std::string_view text, const char* option) {
    Integer value{};
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(std::string("Invalid value for ") + option + ": " + std::string(text));
    }
    return value;
}

void apply_profile(RunConfig& config, const std::string& profile) {
    config.profile = profile;
    if (profile == "debug") {
        config.duration_seconds = 5;
        config.sync_window_us = 10'000;
        config.population_scale = 1;
        config.market_makers = 3;
        config.momentum_traders = 600;
        config.informed_traders = 290;
        config.institutional_traders = 10;
        config.output_dir = "results/debug";
    } else if (profile == "baseline") {
        config.duration_seconds = 600;
        config.sync_window_us = 10'000;
        config.population_scale = 1;
        config.market_makers = 3;
        config.momentum_traders = 6'000;
        config.informed_traders = 2'900;
        config.institutional_traders = 100;
        config.output_dir = "results/baseline";
    } else if (profile == "scale10") {
        config.duration_seconds = 600;
        config.sync_window_us = 10'000;
        config.population_scale = 10;
        config.market_makers = 3;
        config.momentum_traders = 6'000;
        config.informed_traders = 2'900;
        config.institutional_traders = 100;
        config.output_dir = "results/scale10";
    } else {
        throw std::invalid_argument("Unknown profile: " + profile + " (use debug, baseline, or scale10)");
    }
}

std::string require_value(int& index, int argc, char** argv, const char* option) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("Missing value after ") + option);
    return argv[++index];
}

} // namespace

RunConfig parse_run_config(int argc, char** argv) {
    RunConfig config;
    apply_profile(config, "debug");

    // First pass applies the selected profile so later explicit options override it,
    // regardless of where --profile appears on the command line.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            apply_profile(config, require_value(i, argc, argv, "--profile"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(std::cout, argv[0]);
            std::exit(0);
        } else if (arg.rfind("--", 0) == 0) {
            // Skip the value in this discovery pass for options that take one.
            if (arg != "--profile") ++i;
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            ++i; // already applied above
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
        } else if (arg == "--help" || arg == "-h") {
            // handled in first pass
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (config.duration_seconds <= 0) throw std::invalid_argument("--duration-seconds must be positive");
    if (config.sync_window_us < 100) throw std::invalid_argument("--sync-window-us must be at least 100");
    if (config.population_scale <= 0) throw std::invalid_argument("--population-scale must be positive");
    if (config.market_makers < 0 || config.momentum_traders < 0
        || config.informed_traders < 0 || config.institutional_traders < 0) {
        throw std::invalid_argument("Agent counts cannot be negative");
    }
    if (config.expected_ranks < 0) throw std::invalid_argument("--expected-ranks cannot be negative");
    if (config.output_dir.empty()) throw std::invalid_argument("--output-dir cannot be empty");
    return config;
}

void print_usage(std::ostream& output, const char* program_name) {
    output << "Usage: " << program_name << " [options]\n\n"
           << "Profiles:\n"
           << "  --profile debug|baseline|scale10\n\n"
           << "Overrides:\n"
           << "  --duration-seconds N\n"
           << "  --sync-window-us N\n"
           << "  --population-scale N\n"
           << "  --market-makers N\n"
           << "  --momentum N\n"
           << "  --informed N\n"
           << "  --institutional N\n"
           << "  --expected-ranks N       Fail if MPI world size differs (0 disables)\n"
           << "  --seed N\n"
           << "  --output-dir PATH\n";
}

} // namespace dlob
