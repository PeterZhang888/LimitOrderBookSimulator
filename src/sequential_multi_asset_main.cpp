#include "simulation/SequentialMultiAssetSimulator.hpp"
#include "simulation/MultiAssetConfiguration.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

template <typename Integer>
Integer parse_integer(std::string_view text, const char* option) {
    Integer value{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid value for ") + option
                                    + ": " + std::string(text));
    }
    return value;
}

double parse_positive_double(const std::string& text, const char* option) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string("invalid value for ") + option
                                    + ": " + text);
    }
    return value;
}

double parse_nonnegative_double(const std::string& text, const char* option) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string("invalid value for ") + option
                                    + ": " + text);
    }
    return value;
}

std::string require_value(int& index, int argc, char** argv, const char* option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + option);
    }
    return argv[++index];
}

dlob::LiquidityShockConfig& shock_config(dlob::SequentialMultiAssetConfig& config) {
    if (!config.liquidity_shock.has_value()) {
        config.liquidity_shock.emplace();
    }
    return *config.liquidity_shock;
}

dlob::Side parse_side(std::string_view text, const char* option) {
    if (text == "buy") return dlob::Side::Buy;
    if (text == "sell") return dlob::Side::Sell;
    throw std::invalid_argument(std::string("invalid value for ") + option
                                + ": " + std::string(text));
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Exact one-process reference for the interacting multi-asset LOB model.\n\n"
        << "  --duration-seconds N    simulated seconds (23400 for a full day)\n"
        << "  --sample-interval-ms X state-trace interval (default 1000 ms)\n"
        << "  --seed N                model seed, independent of MPI ranks\n"
        << "  --books N               books (1=QQQ baseline, 2+=cross-asset)\n"
        << "  --data-dir PATH         empirical distribution directory\n"
        << "  --hawkes-rates-file CSV optional six-event configured_mu calibration\n"
        << "  --book-config-file CSV  per-book symbols, ITCH inputs, BBOs, and basket\n"
        << "  --quote-interval-ms X   shared/local market-maker refresh interval\n"
        << "  --quote-quantity N      quantity on each market-maker quote\n"
        << "  --quote-levels N        adjacent price levels quoted on each side\n"
        << "  --quote-growth N        integer size multiplier for deeper levels\n"
        << "  --shock-time-ns N       schedule one deterministic liquidity shock\n"
        << "  --shock-book N          target book for the optional shock\n"
        << "  --shock-side buy|sell   market-order side (default sell)\n"
        << "  --shock-quantity N      market-order quantity; must be positive\n"
        << "  --enable-etf-arbitrage  enable QQQ/reduced-basket arbitrage agent\n"
        << "  --arbitrage-etf-book N  ETF book id (default 0)\n"
        << "  --arbitrage-trigger-bps X entry threshold (default 5 bps)\n"
        << "  --arbitrage-release-bps X hysteresis release threshold\n"
        << "  --arbitrage-interval-ms X decision interval (default 100 ms)\n"
        << "  --arbitrage-quantity N  ETF-leg quantity (default 100)\n"
        << "  --arbitrage-max-component-quantity N cap each component leg\n"
        << "  --arbitrage-order-latency-ns N positive order latency\n"
        << "  --enable-value-agent  enable one stabilising value agent per book\n"
        << "  --value-threshold-bps X minimum absolute mispricing\n"
        << "  --value-response-bps X size step above the threshold\n"
        << "  --value-base-quantity N minimum value order quantity\n"
        << "  --value-max-quantity N maximum value order quantity\n"
        << "  --value-max-inventory N absolute inventory limit per book\n"
        << "  --value-fundamental-volatility-bps X latent volatility per sqrt(second)\n"
        << "  --value-interval-ms X value decision interval\n"
        << "  --value-order-latency-ns N positive value-order latency\n"
        << "  --exposure-threshold X shared-maker cross-book hedge threshold\n"
        << "  --enable-shared-mm-hedging activate shared-maker cross-book hedges\n"
        << "  --max-hedge-quantity N cap each shared-maker hedge leg\n"
        << "  --output-dir PATH       summary output directory\n"
        << "  -h, --help              show this help\n";
}

dlob::SequentialMultiAssetConfig parse_arguments(int argc, char** argv) {
    dlob::SequentialMultiAssetConfig config;
    std::string book_config_file;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--duration-seconds") {
            config.duration_seconds = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--sample-interval-ms") {
            const double milliseconds = parse_positive_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
            const double maximum_milliseconds = static_cast<double>(
                std::numeric_limits<std::int64_t>::max()) / 1'000'000.0;
            if (milliseconds > maximum_milliseconds) {
                throw std::invalid_argument("--sample-interval-ms is too large");
            }
            config.sample_interval_ns = static_cast<std::int64_t>(
                std::llround(milliseconds * 1'000'000.0));
            if (config.sample_interval_ns <= 0) {
                throw std::invalid_argument("--sample-interval-ms rounds below 1 ns");
            }
        } else if (argument == "--seed") {
            config.seed = parse_integer<std::uint64_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--books") {
            config.book_count = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--data-dir") {
            config.data_dir = require_value(index, argc, argv, argument.c_str());
        } else if (argument == "--hawkes-rates-file") {
            config.hawkes_rates_file = require_value(
                index, argc, argv, argument.c_str());
        } else if (argument == "--book-config-file") {
            book_config_file = require_value(
                index, argc, argv, argument.c_str());
        } else if (argument == "--quote-interval-ms") {
            const double milliseconds = parse_positive_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
            const double maximum_milliseconds = static_cast<double>(
                std::numeric_limits<std::int64_t>::max()) / 1'000'000.0;
            if (milliseconds > maximum_milliseconds) {
                throw std::invalid_argument("--quote-interval-ms is too large");
            }
            config.market_maker_quote_interval_ns = static_cast<std::int64_t>(
                std::llround(milliseconds * 1'000'000.0));
        } else if (argument == "--quote-quantity") {
            config.market_maker_order_quantity = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--quote-levels") {
            config.market_maker_quote_levels = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--quote-growth") {
            config.market_maker_quote_quantity_growth = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shock-time-ns") {
            shock_config(config).time_ns = parse_integer<std::int64_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shock-book") {
            shock_config(config).book_id = parse_integer<dlob::BookId>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--shock-side") {
            const std::string value = require_value(
                index, argc, argv, argument.c_str());
            shock_config(config).side = parse_side(value, argument.c_str());
        } else if (argument == "--shock-quantity") {
            shock_config(config).quantity = parse_integer<std::int32_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--enable-etf-arbitrage") {
            config.etf_arbitrage.enabled = true;
        } else if (argument == "--arbitrage-etf-book") {
            config.etf_arbitrage.etf_book_id = parse_integer<dlob::BookId>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--arbitrage-trigger-bps") {
            config.etf_arbitrage.trigger_bps = parse_positive_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--arbitrage-release-bps") {
            config.etf_arbitrage.release_bps = parse_nonnegative_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--arbitrage-interval-ms") {
            const double milliseconds = parse_positive_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
            const double maximum_milliseconds = static_cast<double>(
                std::numeric_limits<std::int64_t>::max()) / 1'000'000.0;
            if (milliseconds > maximum_milliseconds) {
                throw std::invalid_argument("--arbitrage-interval-ms is too large");
            }
            config.etf_arbitrage.decision_interval_ns = static_cast<std::int64_t>(
                std::llround(milliseconds * 1'000'000.0));
        } else if (argument == "--arbitrage-quantity") {
            config.etf_arbitrage.etf_order_quantity = parse_integer<std::int32_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--arbitrage-max-component-quantity") {
            config.etf_arbitrage.max_component_quantity = parse_integer<std::int32_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--arbitrage-order-latency-ns") {
            config.etf_arbitrage.order_latency_ns = parse_integer<std::int64_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--enable-value-agent") {
            config.fundamental_value.enabled = true;
        } else if (argument == "--value-threshold-bps") {
            config.fundamental_value.threshold_bps = parse_positive_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--value-response-bps") {
            config.fundamental_value.response_step_bps = parse_positive_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--value-base-quantity") {
            config.fundamental_value.base_order_quantity = parse_integer<std::int32_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--value-max-quantity") {
            config.fundamental_value.max_order_quantity = parse_integer<std::int32_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--value-max-inventory") {
            config.fundamental_value.max_abs_inventory = parse_integer<std::int64_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--value-fundamental-volatility-bps") {
            config.fundamental_value.fundamental_volatility_bps_sqrt_second =
                parse_nonnegative_double(
                    require_value(index, argc, argv, argument.c_str()),
                    argument.c_str());
        } else if (argument == "--value-interval-ms") {
            const double milliseconds = parse_positive_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
            const double maximum_milliseconds = static_cast<double>(
                std::numeric_limits<std::int64_t>::max()) / 1'000'000.0;
            if (milliseconds > maximum_milliseconds) {
                throw std::invalid_argument("--value-interval-ms is too large");
            }
            config.fundamental_value.decision_interval_ns =
                static_cast<std::int64_t>(
                    std::llround(milliseconds * 1'000'000.0));
        } else if (argument == "--value-order-latency-ns") {
            config.fundamental_value.order_latency_ns = parse_integer<std::int64_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--exposure-threshold") {
            config.market_maker_exposure_threshold = parse_nonnegative_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--enable-shared-mm-hedging") {
            config.enable_shared_market_maker_hedging = true;
        } else if (argument == "--max-hedge-quantity") {
            config.max_hedge_quantity = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--output-dir") {
            config.output_dir = require_value(index, argc, argv, argument.c_str());
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (!book_config_file.empty()) {
        config.book_configs = dlob::load_multi_asset_book_configs(book_config_file);
        config.book_count = static_cast<int>(config.book_configs.size());
    }
    return config;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const dlob::SequentialMultiAssetConfig config = parse_arguments(argc, argv);
        dlob::SequentialMultiAssetSimulator simulator(config);
        const dlob::SequentialMultiAssetResult result = simulator.run();

        std::cout << std::setprecision(6) << std::fixed
                  << "sequential_multi_asset"
                  << " books=" << result.books.size()
                  << " simulated_seconds=" << config.duration_seconds
                  << " wall_seconds=" << result.wall_seconds
                  << " events=" << result.processed_events
                  << " trades=" << result.combined_trade_count
                  << " trade_hash=0x" << std::hex << result.combined_trade_hash << std::dec
                  << " cross_reactions=" << result.cross_book_reaction_events
                  << " hedge_orders=" << result.hedge_order_events
                  << " liquidity_shocks=" << result.liquidity_shock_events
                  << " arbitrage_decisions=" << result.arbitrage_decision_events
                  << " arbitrage_orders=" << result.arbitrage_order_events
                  << " value_decisions=" << result.value_decision_events
                  << " value_orders=" << result.value_order_events
                  << " structurally_valid=" << (result.structurally_valid ? 1 : 0)
                  << '\n';
        for (const dlob::MultiAssetBookSummary& book : result.books) {
            std::cout << "book=" << book.book_id
                      << " symbol=" << book.symbol
                      << " bid=" << book.final_state.best_bid_ticks
                      << " ask=" << book.final_state.best_ask_ticks
                      << " inventory=" << book.market_maker_inventory
                          << " cash_ticks=" << book.market_maker_cash_ticks
                          << " arbitrage_inventory=" << book.arbitrage_inventory
                          << " arbitrage_cash_ticks=" << book.arbitrage_cash_ticks
                          << " value_inventory=" << book.value_agent_inventory
                      << " value_cash_ticks=" << book.value_agent_cash_ticks
                      << " fundamental_ticks="
                      << book.final_fundamental_value_ticks
                      << " trades=" << book.trade_count
                      << " hash=0x" << std::hex << book.trade_hash << std::dec
                      << '\n';
        }
        std::cout << "summary_csv=" << result.summary_csv << '\n';
        return result.structurally_valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "sequential_multi_asset: " << error.what() << '\n';
        return 1;
    }
}
