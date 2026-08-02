#include "simulation/FragmentedMpiSimulator.hpp"

#include "simulation/MultiAssetConfiguration.hpp"
#include "simulation/QueueReactiveBackgroundPolicy.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

template <typename Integer>
Integer parse_integer(std::string_view text, const char* option) {
    Integer value{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return value;
}

double parse_double(const std::string& text, const char* option) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return value;
}

std::string require_value(int& index, int argc, char** argv, const char* option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + option);
    }
    return argv[++index];
}

std::vector<std::string> split_csv_row(const std::string& row) {
    std::vector<std::string> fields;
    std::string current;
    for (const char character : row) {
        if (character == ',') {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(character);
        }
    }
    fields.push_back(current);
    return fields;
}

std::string trim_copy(std::string text) {
    const auto is_space = [](unsigned char character) {
        return std::isspace(character) != 0;
    };
    const auto begin = std::find_if_not(text.begin(), text.end(), is_space);
    const auto end = std::find_if_not(text.rbegin(), text.rend(), is_space).base();
    return begin < end ? std::string(begin, end) : std::string();
}

std::size_t require_csv_column(const std::vector<std::string>& header,
                               const char* name) {
    for (std::size_t index = 0; index < header.size(); ++index) {
        if (trim_copy(header[index]) == name) {
            return index;
        }
    }
    throw std::runtime_error(
        std::string("input CSV is missing required column ") + name);
}

bool parse_policy_bool(std::string value, std::size_t line_number) {
    value = trim_copy(std::move(value));
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    if (value == "1" || value == "true" || value == "on") return true;
    if (value == "0" || value == "false" || value == "off") return false;
    throw std::runtime_error(
        "invalid enabled value at value-agent policy CSV line "
        + std::to_string(line_number));
}

dlob::FragmentedValueTriggerMode parse_value_trigger_mode(
    std::string value, std::size_t line_number) {
    value = trim_copy(std::move(value));
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    if (value == "periodic_gap") {
        return dlob::FragmentedValueTriggerMode::PeriodicGap;
    }
    if (value == "news_impulse") {
        return dlob::FragmentedValueTriggerMode::NewsImpulse;
    }
    throw std::runtime_error(
        "invalid value_trigger_mode at value-agent policy CSV line "
        + std::to_string(line_number)
        + "; expected periodic_gap or news_impulse");
}

std::vector<dlob::FragmentedValueAgentPolicy> load_value_agent_policies(
    const std::string& path,
    const std::vector<dlob::MultiAssetBookConfig>& assets) {
    if (path.empty()) return {};
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open value-agent policy CSV: " + path);
    }
    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("empty value-agent policy CSV: " + path);
    }
    const std::vector<std::string> header = split_csv_row(header_line);
    const std::size_t symbol_column = require_csv_column(header, "symbol");
    const std::size_t enabled_column = require_csv_column(header, "enabled");
    const std::size_t threshold_column = require_csv_column(
        header, "value_threshold_bps");
    const auto participation_found = std::find(
        header.begin(), header.end(), "value_depth_participation");
    const auto quantity_found = std::find(
        header.begin(), header.end(), "value_order_quantity");
    const auto trigger_found = std::find(
        header.begin(), header.end(), "value_trigger_mode");
    const auto rechecks_found = std::find(
        header.begin(), header.end(), "value_maximum_news_rechecks");
    const auto gap_elasticity_found = std::find(
        header.begin(), header.end(), "value_gap_elasticity");
    const auto maximum_participation_found = std::find(
        header.begin(), header.end(), "value_max_depth_participation");
    if (participation_found == header.end() && quantity_found == header.end()) {
        throw std::runtime_error(
            "value-agent policy CSV requires value_depth_participation "
            "(certification model) or legacy value_order_quantity");
    }
    const std::size_t participation_column = participation_found == header.end()
        ? 0U : static_cast<std::size_t>(
            std::distance(header.begin(), participation_found));
    const std::size_t quantity_column = quantity_found == header.end()
        ? 0U : static_cast<std::size_t>(
            std::distance(header.begin(), quantity_found));
    const std::size_t trigger_column = trigger_found == header.end()
        ? 0U : static_cast<std::size_t>(
            std::distance(header.begin(), trigger_found));
    const std::size_t rechecks_column = rechecks_found == header.end()
        ? 0U : static_cast<std::size_t>(
            std::distance(header.begin(), rechecks_found));
    const std::size_t gap_elasticity_column =
        gap_elasticity_found == header.end()
        ? 0U : static_cast<std::size_t>(
            std::distance(header.begin(), gap_elasticity_found));
    const std::size_t maximum_participation_column =
        maximum_participation_found == header.end()
        ? 0U : static_cast<std::size_t>(
            std::distance(header.begin(), maximum_participation_found));
    std::size_t maximum_column = std::max({
        symbol_column, enabled_column, threshold_column});
    if (participation_found != header.end()) {
        maximum_column = std::max(maximum_column, participation_column);
    }
    if (quantity_found != header.end()) {
        maximum_column = std::max(maximum_column, quantity_column);
    }
    if (trigger_found != header.end()) {
        maximum_column = std::max(maximum_column, trigger_column);
    }
    if (rechecks_found != header.end()) {
        maximum_column = std::max(maximum_column, rechecks_column);
    }
    if (gap_elasticity_found != header.end()) {
        maximum_column = std::max(maximum_column, gap_elasticity_column);
    }
    if (maximum_participation_found != header.end()) {
        maximum_column = std::max(
            maximum_column, maximum_participation_column);
    }
    const std::size_t minimum_fields = 1U + maximum_column;

    std::unordered_map<std::string, dlob::FragmentedValueAgentPolicy> by_symbol;
    std::string row;
    std::size_t line_number = 1;
    while (std::getline(input, row)) {
        ++line_number;
        if (row.empty()) continue;
        const std::vector<std::string> fields = split_csv_row(row);
        if (fields.size() < minimum_fields) {
            throw std::runtime_error(
                "short value-agent policy CSV row at line "
                + std::to_string(line_number));
        }
        const std::string symbol = trim_copy(fields[symbol_column]);
        if (symbol.empty()) {
            throw std::runtime_error(
                "empty symbol at value-agent policy CSV line "
                + std::to_string(line_number));
        }
        dlob::FragmentedValueAgentPolicy policy;
        policy.enabled = parse_policy_bool(fields[enabled_column], line_number);
        policy.threshold_bps = parse_double(
            trim_copy(fields[threshold_column]), "value_threshold_bps");
        if (participation_found != header.end()) {
            policy.depth_participation = parse_double(
                trim_copy(fields[participation_column]),
                "value_depth_participation");
        }
        if (quantity_found != header.end()) {
            policy.order_quantity = parse_integer<int>(
                trim_copy(fields[quantity_column]), "value_order_quantity");
        }
        if (trigger_found != header.end()) {
            policy.trigger_mode = parse_value_trigger_mode(
                fields[trigger_column], line_number);
        }
        if (rechecks_found != header.end()) {
            policy.maximum_news_rechecks = parse_integer<int>(
                trim_copy(fields[rechecks_column]),
                "value_maximum_news_rechecks");
        }
        if (gap_elasticity_found != header.end()) {
            policy.gap_elasticity = parse_double(
                trim_copy(fields[gap_elasticity_column]),
                "value_gap_elasticity");
        }
        if (maximum_participation_found != header.end()) {
            policy.maximum_depth_participation = parse_double(
                trim_copy(fields[maximum_participation_column]),
                "value_max_depth_participation");
        }
        if (!std::isfinite(policy.threshold_bps) || policy.threshold_bps < 0.0
            || !std::isfinite(policy.depth_participation)
            || policy.depth_participation < 0.0
            || policy.depth_participation > 1.0
            || (policy.depth_participation == 0.0
                && policy.order_quantity <= 0)
            || policy.maximum_news_rechecks < 0
            || policy.maximum_news_rechecks > 16
            || (policy.trigger_mode
                    == dlob::FragmentedValueTriggerMode::PeriodicGap
                && policy.maximum_news_rechecks != 0)
            || !std::isfinite(policy.gap_elasticity)
            || policy.gap_elasticity < 0.0
            || !std::isfinite(policy.maximum_depth_participation)
            || policy.maximum_depth_participation <= 0.0
            || policy.maximum_depth_participation > 1.0
            || policy.maximum_depth_participation
                < policy.depth_participation
            || (policy.gap_elasticity > 0.0
                && (policy.depth_participation <= 0.0
                    || policy.threshold_bps <= 0.0))) {
            throw std::runtime_error(
                "invalid policy values at value-agent policy CSV line "
                + std::to_string(line_number));
        }
        if (!by_symbol.emplace(symbol, policy).second) {
            throw std::runtime_error(
                "duplicate symbol in value-agent policy CSV: " + symbol);
        }
    }

    if (by_symbol.size() != assets.size()) {
        throw std::runtime_error(
            "value-agent policy CSV must contain exactly one row for each asset");
    }
    std::vector<dlob::FragmentedValueAgentPolicy> policies;
    policies.reserve(assets.size());
    for (const dlob::MultiAssetBookConfig& asset : assets) {
        const auto found = by_symbol.find(asset.symbol);
        if (found == by_symbol.end()) {
            throw std::runtime_error(
                "value-agent policy CSV is missing asset symbol " + asset.symbol);
        }
        policies.push_back(found->second);
    }
    return policies;
}

std::vector<int> load_shock_clusters(
    const std::string& path,
    const std::vector<dlob::MultiAssetBookConfig>& assets) {
    if (path.empty()) return {};
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open shock-cluster CSV: " + path);
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty shock-cluster CSV: " + path);
    }
    const std::vector<std::string> header = split_csv_row(line);
    const std::size_t symbol_column = require_csv_column(header, "symbol");
    const std::size_t cluster_column = require_csv_column(header, "cluster_id");
    const std::size_t minimum_fields = 1U + std::max(symbol_column, cluster_column);
    std::unordered_map<std::string, int> by_symbol;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const std::vector<std::string> fields = split_csv_row(line);
        if (fields.size() < minimum_fields) {
            throw std::runtime_error("short shock-cluster CSV row at line "
                                     + std::to_string(line_number));
        }
        const std::string symbol = trim_copy(fields[symbol_column]);
        const int cluster = parse_integer<int>(
            trim_copy(fields[cluster_column]), "cluster_id");
        if (symbol.empty() || cluster < 0 || !by_symbol.emplace(symbol, cluster).second) {
            throw std::runtime_error("invalid or duplicate shock-cluster row at line "
                                     + std::to_string(line_number));
        }
    }
    std::vector<int> clusters;
    clusters.reserve(assets.size());
    for (const dlob::MultiAssetBookConfig& asset : assets) {
        const auto found = by_symbol.find(asset.symbol);
        if (found == by_symbol.end()) {
            throw std::runtime_error(
                "shock-cluster CSV is missing asset symbol " + asset.symbol);
        }
        clusters.push_back(found->second);
    }
    return clusters;
}

struct Options {
    int duration_seconds = 60;
    int assets = 101;
    double window_ms = 1000.0;
    // Zero means use the causal shared-risk decision-window cadence.
    double metrics_interval_ms = 0.0;
    double hawkes_activity_scale = 0.30;
    std::string background_model = "legacy";
    std::string background_policy_csv;
    // Zero means use the global decision-window cadence.
    double local_mm_interval_ms = 0.0;
    double local_mm_quantity_multiplier = 1.0;
    double local_mm_improvement_probability = 0.0;
    double local_mm_spread_elasticity = 0.0;
    double local_mm_max_improvement_probability = 1.0;
    // Zero means use the decision-window cadence.
    double asset_summary_interval_ms = 0.0;
    std::uint64_t seed = 20200130;
    std::string base_config = "config/qqq_aapl_msft_amzn_20200130.csv";
    std::string universe_config;
    std::string metrics_csv;
    std::string asset_summary_csv;
    std::string shock_targets_csv;
    std::string shock_cluster_csv;
    bool local_market_makers = true;
    bool value_agents = true;
    double value_agent_interval_ms = 1000.0;
    double value_threshold_bps = 8.0;
    int value_quantity = 50;
    std::string value_agent_policy_csv;
    bool shared_market_maker = true;
    bool global_shared_capacity = true;
    int shared_quote_quantity = 200;
    int shared_quote_levels = 1;
    bool shared_quote_relative_to_asset = false;
    double shared_quote_multiplier = 1.0;
    double local_inventory_limit = 100.0;
    double global_risk_limit_per_asset = 100.0;
    double capacity_threshold = 0.5;
    bool shock = false;
    double shock_time_seconds = 30.0;
    double shock_fraction = 0.01;
    int shock_target_count = 0;
    std::uint64_t shock_target_seed = 20200130;
    int shock_quantity = 5'000;
    double shock_top_depth_multiple = 0.0;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "MPI simulation of an ITCH-calibrated limit-order-book market.\n"
        << "Each asset has one local book; ranks exchange only shared-firm risk "
           "at decision boundaries. Aggregate diagnostics have an independent "
           "sampling cadence.\n\n"
        << "  --duration-seconds N       simulated seconds (default 60)\n"
        << "  --assets N                synthetic logical assets (default 101)\n"
        << "  --window-ms X             global decision window (default 1000)\n"
        << "  --metrics-interval-ms X   global diagnostic sampling cadence "
           "(default: decision window; no effect on market state)\n"
        << "  --hawkes-activity-scale X global background Hawkes activity multiplier "
           "for legacy mode (default 0.30)\n"
        << "  --background-model NAME  legacy or queue-reactive-v1\n"
        << "  --background-policy-csv PATH\n"
        << "                           frozen training-only policy mapping; required "
           "by queue-reactive-v1\n"
        << "  --local-mm-interval-ms X local-MM quote refresh cadence "
           "(default: global decision window; no extra MPI communication)\n"
        << "  --local-mm-quantity-multiplier X multiplier for empirical local-MM quote size\n"
        << "  --local-mm-improvement-probability X local-maker probability of repairing a wide spread\n"
        << "  --local-mm-spread-elasticity X spread-response exponent for local repair (default 0)\n"
        << "  --local-mm-max-improvement-probability X cap for spread-responsive repair (default 1)\n"
        << "  --base-config PATH        four-template synthetic-market CSV\n"
        << "  --universe-config PATH    exact empirical universe CSV; uses every row\n"
        << "  --seed N                  rank-independent random seed\n"
        << "  --metrics-csv PATH        write boundary liquidity series on rank 0\n"
        << "  --asset-summary-csv PATH  write per-asset fixed-clock calibration moments\n"
        << "  --asset-summary-interval-ms X\n"
        << "                           sampling cadence for --asset-summary-csv "
           "(default: decision window)\n"
        << "  --shock-targets-csv PATH  write deterministic shock target list on rank 0\n"
        << "  --global-risk-limit-per-asset X shared-MM global capacity / assets\n"
        << "  --risk-limit-per-asset X backward-compatible alias for the preceding option\n"
        << "  --local-inventory-limit X fixed local inventory-skew scale\n"
        << "  --capacity-threshold X global withdrawal activation u_0\n"
        << "  --shared-quote-quantity N shared-MM quantity per asset book\n"
        << "  --shared-quote-levels N   shared-MM levels per side\n"
        << "  --shared-quote-relative   scale shared quotes by each asset's calibrated size\n"
        << "  --shared-quote-multiplier X multiplier used with --shared-quote-relative\n"
        << "  --disable-shared-mm       remove cross-asset shared firm\n"
        << "  --uncoupled-shared-mm     keep shared quotes/skew but force global phi=1\n"
        << "  --disable-local-mm        remove local market makers\n"
        << "  --disable-value-agent     remove stabilising value agents\n"
        << "  --value-agent-interval-ms X local value-decision cadence "
           "(default 1000; independent of --window-ms)\n"
        << "  --value-threshold-bps X   value-agent intervention threshold\n"
        << "  --value-quantity N        legacy fixed-share value quantity\n"
        << "  --value-agent-policy-csv PATH\n"
        << "                           per-symbol cluster policy: symbol,enabled,"
           "value_threshold_bps,value_depth_participation"
           "[,value_trigger_mode,value_maximum_news_rechecks,"
           "value_gap_elasticity,value_max_depth_participation]\n"
        << "  --shock                   enable common-random-number sell shock\n"
        << "  --shock-time-seconds X    shock time (default 30)\n"
        << "  --shock-fraction X        fraction of assets shocked (default 0.01)\n"
        << "  --shock-target-count N    exact target count; overrides fraction\n"
        << "  --shock-target-seed N     fixed mask seed, independent of path seed\n"
        << "  --shock-cluster-csv PATH  symbol,cluster_id mapping for stratified mask\n"
        << "  --shock-quantity N        fixed sell quantity per shocked asset\n"
        << "  --shock-top-depth-multiple X override quantity with X times bid depth at t_s-\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--duration-seconds") {
            options.duration_seconds = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--assets") {
            options.assets = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--window-ms") {
            options.window_ms = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--metrics-interval-ms") {
            options.metrics_interval_ms = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--hawkes-activity-scale") {
            options.hawkes_activity_scale = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--background-model") {
            options.background_model = require_value(
                index, argc, argv, argument.c_str());
        } else if (argument == "--background-policy-csv") {
            options.background_policy_csv = require_value(
                index, argc, argv, argument.c_str());
        } else if (argument == "--local-mm-interval-ms") {
            options.local_mm_interval_ms = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--local-mm-quantity-multiplier") {
            options.local_mm_quantity_multiplier = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--local-mm-improvement-probability") {
            options.local_mm_improvement_probability = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--local-mm-spread-elasticity") {
            options.local_mm_spread_elasticity = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--local-mm-max-improvement-probability") {
            options.local_mm_max_improvement_probability = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--asset-summary-interval-ms") {
            options.asset_summary_interval_ms = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--base-config") {
            options.base_config = require_value(index, argc, argv, argument.c_str());
        } else if (argument == "--universe-config") {
            options.universe_config = require_value(index, argc, argv, argument.c_str());
        } else if (argument == "--seed") {
            options.seed = parse_integer<std::uint64_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--metrics-csv") {
            options.metrics_csv = require_value(index, argc, argv, argument.c_str());
        } else if (argument == "--asset-summary-csv") {
            options.asset_summary_csv = require_value(
                index, argc, argv, argument.c_str());
        } else if (argument == "--shock-targets-csv") {
            options.shock_targets_csv = require_value(index, argc, argv, argument.c_str());
        } else if (argument == "--shock-cluster-csv") {
            options.shock_cluster_csv = require_value(index, argc, argv, argument.c_str());
        } else if (argument == "--risk-limit-per-asset") {
            options.global_risk_limit_per_asset = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--global-risk-limit-per-asset") {
            options.global_risk_limit_per_asset = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--local-inventory-limit") {
            options.local_inventory_limit = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--capacity-threshold") {
            options.capacity_threshold = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shared-quote-quantity") {
            options.shared_quote_quantity = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shared-quote-levels") {
            options.shared_quote_levels = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shared-quote-relative") {
            options.shared_quote_relative_to_asset = true;
        } else if (argument == "--shared-quote-multiplier") {
            options.shared_quote_multiplier = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--disable-shared-mm") {
            options.shared_market_maker = false;
        } else if (argument == "--uncoupled-shared-mm") {
            options.global_shared_capacity = false;
        } else if (argument == "--disable-local-mm") {
            options.local_market_makers = false;
        } else if (argument == "--disable-value-agent") {
            options.value_agents = false;
        } else if (argument == "--value-agent-interval-ms") {
            options.value_agent_interval_ms = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--value-threshold-bps") {
            options.value_threshold_bps = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--value-quantity") {
            options.value_quantity = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--value-agent-policy-csv") {
            options.value_agent_policy_csv = require_value(
                index, argc, argv, argument.c_str());
        } else if (argument == "--shock") {
            options.shock = true;
        } else if (argument == "--shock-time-seconds") {
            options.shock_time_seconds = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shock-fraction") {
            options.shock_fraction = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shock-target-count") {
            options.shock_target_count = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shock-target-seed") {
            options.shock_target_seed = parse_integer<std::uint64_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shock-quantity") {
            options.shock_quantity = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shock-top-depth-multiple") {
            options.shock_top_depth_multiple = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.duration_seconds <= 0 || options.assets <= 0
        || !std::isfinite(options.window_ms) || options.window_ms <= 0.0
        || !std::isfinite(options.metrics_interval_ms)
        || options.metrics_interval_ms < 0.0
        || !std::isfinite(options.hawkes_activity_scale)
        || options.hawkes_activity_scale <= 0.0
        || !std::isfinite(options.local_mm_interval_ms)
        || options.local_mm_interval_ms < 0.0
        || !std::isfinite(options.local_mm_quantity_multiplier)
        || options.local_mm_quantity_multiplier <= 0.0
        || !std::isfinite(options.local_mm_improvement_probability)
        || options.local_mm_improvement_probability < 0.0
        || options.local_mm_improvement_probability > 1.0
        || !std::isfinite(options.local_mm_spread_elasticity)
        || options.local_mm_spread_elasticity < 0.0
        || !std::isfinite(options.local_mm_max_improvement_probability)
        || options.local_mm_max_improvement_probability
            < options.local_mm_improvement_probability
        || options.local_mm_max_improvement_probability > 1.0
        || !std::isfinite(options.asset_summary_interval_ms)
        || options.asset_summary_interval_ms < 0.0
        || !std::isfinite(options.shock_time_seconds)
        || options.shock_time_seconds < 0.0
        || !std::isfinite(options.shock_fraction)
        || options.shock_fraction <= 0.0 || options.shock_fraction > 1.0
        || options.shock_quantity <= 0
        || !std::isfinite(options.shock_top_depth_multiple)
        || options.shock_top_depth_multiple < 0.0
        || options.shock_target_count < 0
        || !std::isfinite(options.local_inventory_limit)
        || options.local_inventory_limit <= 0.0
        || !std::isfinite(options.global_risk_limit_per_asset)
        || options.global_risk_limit_per_asset <= 0.0
        || !std::isfinite(options.capacity_threshold)
        || options.capacity_threshold < 0.0 || options.capacity_threshold >= 1.0
        || options.shared_quote_quantity <= 0
        || options.shared_quote_levels <= 0
        || !std::isfinite(options.shared_quote_multiplier)
        || options.shared_quote_multiplier <= 0.0
        || !std::isfinite(options.value_agent_interval_ms)
        || options.value_agent_interval_ms <= 0.0
        || !std::isfinite(options.value_threshold_bps)
        || options.value_threshold_bps < 0.0
        || options.value_quantity <= 0) {
        throw std::invalid_argument("invalid fragmented-market options");
    }
    if (options.background_model != "legacy"
        && options.background_model != "queue-reactive-v1") {
        throw std::invalid_argument(
            "--background-model must be legacy or queue-reactive-v1");
    }
    if ((options.background_model == "queue-reactive-v1")
            != !options.background_policy_csv.empty()) {
        throw std::invalid_argument(
            "queue-reactive-v1 requires --background-policy-csv, and legacy "
            "mode forbids it");
    }
    return options;
}

std::vector<dlob::MultiAssetBookConfig> expand_templates(
    const std::vector<dlob::MultiAssetBookConfig>& base,
    int asset_count) {
    if (base.size() != 4U) {
        throw std::invalid_argument(
            "base configuration must contain QQQ, AAPL, MSFT and AMZN");
    }
    std::vector<dlob::MultiAssetBookConfig> assets;
    assets.reserve(static_cast<std::size_t>(asset_count));
    for (int index = 0; index < asset_count; ++index) {
        const std::size_t template_index = index == 0
            ? 0U : static_cast<std::size_t>((index - 1) % 3 + 1);
        dlob::MultiAssetBookConfig asset = base[template_index];
        asset.symbol = index == 0
            ? "QQQ" : asset.symbol + "_SYNTH_" + std::to_string(index);
        asset.basket_weight = 0.0;
        assets.push_back(std::move(asset));
    }
    return assets;
}

std::vector<dlob::MultiAssetBookConfig> resolve_asset_configs(
    const Options& options) {
    if (!options.universe_config.empty()) {
        const std::vector<dlob::MultiAssetBookConfig> universe =
            dlob::load_multi_asset_book_configs(options.universe_config);
        if (universe.empty()) {
            throw std::invalid_argument("empirical universe configuration is empty");
        }
        return universe;
    }
    return expand_templates(
        dlob::load_multi_asset_book_configs(options.base_config), options.assets);
}

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed\n";
        return 1;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<dlob::MultiAssetBookConfig> asset_configs =
            resolve_asset_configs(options);
        dlob::QueueReactiveBackgroundBundle background_bundle;
        if (options.background_model == "queue-reactive-v1") {
            background_bundle = dlob::load_queue_reactive_background_bundle(
                options.background_policy_csv, asset_configs,
                options.seed, 100);
        }
        const std::vector<dlob::FragmentedValueAgentPolicy> value_policies =
            load_value_agent_policies(options.value_agent_policy_csv, asset_configs);
        const std::vector<int> shock_clusters =
            load_shock_clusters(options.shock_cluster_csv, asset_configs);

        const double window_ns_double = options.window_ms * 1'000'000.0;
        const double metrics_interval_ms = options.metrics_interval_ms > 0.0
            ? options.metrics_interval_ms : options.window_ms;
        const double metrics_interval_ns_double =
            metrics_interval_ms * 1'000'000.0;
        const double local_mm_interval_ms = options.local_mm_interval_ms > 0.0
            ? options.local_mm_interval_ms : options.window_ms;
        const double local_mm_interval_ns_double = local_mm_interval_ms * 1'000'000.0;
        const double value_agent_interval_ns_double =
            options.value_agent_interval_ms * 1'000'000.0;
        const double shock_ns_double = options.shock_time_seconds * 1e9;
        const double asset_summary_ms = options.asset_summary_interval_ms > 0.0
            ? options.asset_summary_interval_ms : options.window_ms;
        const double asset_summary_ns_double = asset_summary_ms * 1'000'000.0;
        if (window_ns_double > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())
            || local_mm_interval_ns_double > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())
            || value_agent_interval_ns_double > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())
            || metrics_interval_ns_double > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())
            || shock_ns_double > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())
            || asset_summary_ns_double > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument(
                "window, metrics, local-MM, value-agent, shock, or "
                "asset-summary timestamp is too large");
        }
        const auto local_mm_interval_ns = static_cast<std::int64_t>(
            std::llround(local_mm_interval_ns_double));
        const auto value_agent_interval_ns = static_cast<std::int64_t>(
            std::llround(value_agent_interval_ns_double));
        if (local_mm_interval_ns <= 0) {
            throw std::invalid_argument("local-MM interval must round to at least one ns");
        }
        if (value_agent_interval_ns <= 0) {
            throw std::invalid_argument(
                "value-agent interval must round to at least one ns");
        }

        dlob::FragmentedMpiConfig config;
        config.duration_seconds = options.duration_seconds;
        config.asset_count = static_cast<int>(asset_configs.size());
        config.decision_window_ns = static_cast<std::int64_t>(
            std::llround(window_ns_double));
        config.global_metrics_interval_ns = static_cast<std::int64_t>(
            std::llround(metrics_interval_ns_double));
        config.hawkes_activity_scale = options.hawkes_activity_scale;
        config.local_mm_interval_ns = local_mm_interval_ns;
        config.local_mm_quantity_multiplier = options.local_mm_quantity_multiplier;
        config.local_mm_improvement_probability =
            options.local_mm_improvement_probability;
        config.local_mm_spread_elasticity =
            options.local_mm_spread_elasticity;
        config.local_mm_max_improvement_probability =
            options.local_mm_max_improvement_probability;
        config.asset_summary_interval_ns = static_cast<std::int64_t>(
            std::llround(asset_summary_ns_double));
        config.seed = options.seed;
        config.asset_configs = asset_configs;
        config.background_configs = std::move(background_bundle.configs);
        config.background_model = options.background_model;
        config.enable_local_market_makers = options.local_market_makers;
        config.enable_value_agents = options.value_agents;
        config.value_agent_interval_ns = value_agent_interval_ns;
        config.value_threshold_bps = options.value_threshold_bps;
        config.value_depth_participation = 0.0;
        config.value_order_quantity = options.value_quantity;
        config.value_agent_policies = value_policies;
        config.enable_shared_market_maker = options.shared_market_maker;
        config.enable_global_shared_capacity = options.global_shared_capacity;
        config.shared_quote_quantity = options.shared_quote_quantity;
        config.shared_quote_levels = options.shared_quote_levels;
        config.shared_quote_relative_to_asset = options.shared_quote_relative_to_asset;
        config.shared_quote_multiplier = options.shared_quote_multiplier;
        config.shared_local_inventory_scale = options.local_inventory_limit;
        config.shared_global_risk_limit_per_asset =
            options.global_risk_limit_per_asset;
        config.shared_capacity_threshold = options.capacity_threshold;
        config.enable_shock = options.shock;
        config.shock_time_ns = static_cast<std::int64_t>(
            std::llround(shock_ns_double));
        config.shock_asset_fraction = options.shock_fraction;
        config.shock_target_count = options.shock_target_count;
        config.shock_target_seed = options.shock_target_seed;
        config.shock_cluster_ids = shock_clusters;
        config.shock_quantity_per_asset = options.shock_quantity;
        config.shock_top_depth_multiple = options.shock_top_depth_multiple;
        config.metrics_csv = options.metrics_csv;
        config.asset_summary_csv = options.asset_summary_csv;
        config.shock_targets_csv = options.shock_targets_csv;

        dlob::FragmentedMpiSimulator simulator(MPI_COMM_WORLD, std::move(config));
        const dlob::FragmentedMpiResult result = simulator.run();
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(9)
                << "fragmented_mpi_lob"
                << " ranks=" << result.world_size
                << " assets=" << result.asset_count
                << " lobs=" << result.lob_count
                << " simulated_seconds=" << options.duration_seconds
                << " window_ms=" << options.window_ms
                << " metrics_interval_ms=" << metrics_interval_ms
                << " fundamental_news_interval_ms=1000.000000000"
                << " windows=" << result.windows
                << " hawkes_activity_scale=" << options.hawkes_activity_scale
                << " background_model=" << options.background_model
                << " background_policy="
                << (options.background_policy_csv.empty()
                    ? "none" : options.background_policy_csv)
                << " local_mm_interval_ms=" << local_mm_interval_ms
                << " local_mm_quantity_multiplier="
                << options.local_mm_quantity_multiplier
                << " local_mm_improvement_probability="
                << options.local_mm_improvement_probability
                << " local_mm_spread_elasticity="
                << options.local_mm_spread_elasticity
                << " local_mm_max_improvement_probability="
                << options.local_mm_max_improvement_probability
                << " local_mm_refresh_boundaries="
                << result.local_mm_refresh_boundaries
                << " wall_seconds=" << result.wall_seconds
                << " max_initialization_seconds="
                << result.max_initialization_seconds
                << " max_compute_seconds=" << result.max_compute_seconds
                << " max_communication_seconds="
                << result.max_communication_seconds
                << " communication_fraction=" << result.communication_fraction
                << " processed_orders=" << result.processed_orders
                << " trades=" << result.trades
                << " collective_calls=" << result.collective_calls
                << " local_mm=" << (options.local_market_makers ? 1 : 0)
                << " value_agent=" << (options.value_agents ? 1 : 0)
                << " value_agent_interval_ms="
                << (static_cast<double>(value_agent_interval_ns) / 1'000'000.0)
                << " value_agent_policy="
                << (options.value_agent_policy_csv.empty()
                    ? "global" : options.value_agent_policy_csv)
                << " shared_mm=" << (options.shared_market_maker ? 1 : 0)
                << " shared_mm_mode="
                << (!options.shared_market_maker ? "off"
                    : (options.global_shared_capacity ? "global" : "uncoupled"))
                << " shared_quote_relative="
                << (options.shared_quote_relative_to_asset ? 1 : 0)
                << " shared_quote_quantity=" << options.shared_quote_quantity
                << " shared_quote_multiplier=" << options.shared_quote_multiplier
                << " shared_quote_levels=" << options.shared_quote_levels
                << " local_inventory_limit=" << options.local_inventory_limit
                << " global_risk_limit_per_asset="
                << options.global_risk_limit_per_asset
                << " risk_limit_per_asset="
                << options.global_risk_limit_per_asset
                << " capacity_threshold=" << options.capacity_threshold
                << " shock=" << (options.shock ? 1 : 0)
                << " shock_time_seconds=" << options.shock_time_seconds
                << " shock_fraction=" << options.shock_fraction
                << " shock_target_count_requested=" << options.shock_target_count
                << " shock_target_seed=" << options.shock_target_seed
                << " shock_top_depth_multiple=" << options.shock_top_depth_multiple
                << " shock_target_assets=" << result.shock_target_assets
                << " shock_assets=" << result.shock_assets
                << " shock_requested_quantity=" << result.shock_requested_quantity
                << " shock_executed_quantity=" << result.shock_executed_quantity
                << " shock_shared_mm_quantity=" << result.shock_shared_mm_quantity
                << " withdrawal_windows=" << result.withdrawal_windows
                << " final_shared_gross_exposure="
                << result.final_shared_gross_exposure
                << " final_shared_utilization="
                << result.final_shared_utilization
                << " minimum_shared_quote_scale="
                << result.minimum_shared_quote_scale
                << " peak_affected_fraction=" << result.peak_affected_fraction
                << " peak_affected_unshocked_fraction="
                << result.peak_affected_unshocked_fraction
                << " peak_mean_spread_bps=" << result.peak_mean_spread_bps
                << " final_mean_spread_bps=" << result.final_mean_spread_bps
                << " final_mean_top_depth=" << result.final_mean_top_depth
                << " final_affected_shocked_fraction="
                << result.final_affected_shocked_fraction
                << " final_affected_unshocked_fraction="
                << result.final_affected_unshocked_fraction
                << " minimum_two_sided_book_fraction="
                << result.minimum_two_sided_book_fraction
                << " min_compute_seconds=" << result.min_compute_seconds
                << " mean_compute_seconds=" << result.mean_compute_seconds
                << " compute_imbalance=" << result.compute_imbalance
                << " min_orders_per_rank=" << result.min_orders_per_rank
                << " mean_orders_per_rank=" << result.mean_orders_per_rank
                << " max_orders_per_rank=" << result.max_orders_per_rank
                << " min_books_per_rank=" << result.min_books_per_rank
                << " mean_books_per_rank=" << result.mean_books_per_rank
                << " max_books_per_rank=" << result.max_books_per_rank
                << " state_hash=0x" << std::hex << result.state_hash << std::dec
                << '\n';
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fragmented_mpi_lob rank " << rank
                  << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
}
