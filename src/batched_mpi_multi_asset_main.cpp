#include "simulation/BatchedMpiMultiAssetSimulator.hpp"

#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
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

struct Options {
    int duration_seconds = 60;
    int books = 101;
    double window_ms = 1000.0;
    std::uint64_t seed = 20200130;
    std::string base_config = "config/qqq_aapl_msft_amzn_20200130.csv";
    int quote_levels = 3;
    double trigger_bps = 0.5;
    double release_bps = 0.25;
    bool arbitrage = true;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "One-second batched MPI benchmark using QQQ plus replicated "
           "AAPL/MSFT/AMZN calibrated templates.\n\n"
        << "  --duration-seconds N  simulated duration (default 60)\n"
        << "  --books N             total books including QQQ (default 101)\n"
        << "  --window-ms X         shared-agent update interval (default 1000)\n"
        << "  --base-config PATH    four-asset calibrated CSV\n"
        << "  --seed N              rank-independent model seed\n"
        << "  --quote-levels N      shared-MM levels per side (default 3)\n"
        << "  --trigger-bps X       ETF-arbitrage entry threshold\n"
        << "  --release-bps X       ETF-arbitrage hysteresis threshold\n"
        << "  --disable-arbitrage   retain shared MM but omit ETF arbitrage\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--duration-seconds") {
            options.duration_seconds = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--books") {
            options.books = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--window-ms") {
            options.window_ms = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--base-config") {
            options.base_config = require_value(index, argc, argv, argument.c_str());
        } else if (argument == "--seed") {
            options.seed = parse_integer<std::uint64_t>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--quote-levels") {
            options.quote_levels = parse_integer<int>(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--trigger-bps") {
            options.trigger_bps = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--release-bps") {
            options.release_bps = parse_double(
                require_value(index, argc, argv, argument.c_str()), argument.c_str());
        } else if (argument == "--disable-arbitrage") {
            options.arbitrage = false;
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.duration_seconds <= 0 || options.books < 4
        || !std::isfinite(options.window_ms) || options.window_ms <= 0.0
        || options.quote_levels <= 0 || options.quote_levels > 100
        || !std::isfinite(options.trigger_bps) || options.trigger_bps <= 0.0
        || !std::isfinite(options.release_bps) || options.release_bps < 0.0
        || options.release_bps >= options.trigger_bps) {
        throw std::invalid_argument("invalid batched benchmark configuration");
    }
    return options;
}

std::vector<dlob::MultiAssetBookConfig> expand_templates(
    const std::vector<dlob::MultiAssetBookConfig>& base,
    int book_count) {
    if (base.size() != 4U) {
        throw std::invalid_argument(
            "base configuration must contain QQQ, AAPL, MSFT and AMZN");
    }
    std::vector<std::size_t> counts(3U, 0U);
    for (int index = 1; index < book_count; ++index) {
        ++counts[static_cast<std::size_t>((index - 1) % 3)];
    }

    std::vector<dlob::MultiAssetBookConfig> books;
    books.reserve(static_cast<std::size_t>(book_count));
    books.push_back(base.front());
    books.front().symbol = "QQQ";
    books.front().basket_weight = 0.0;
    for (int index = 1; index < book_count; ++index) {
        const std::size_t template_index =
            static_cast<std::size_t>((index - 1) % 3) + 1U;
        dlob::MultiAssetBookConfig book = base[template_index];
        book.symbol += "_R" + std::to_string(index);
        const std::size_t category = template_index - 1U;
        book.basket_weight /= static_cast<double>(counts[category]);
        books.push_back(std::move(book));
    }
    return books;
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
        const std::vector<dlob::MultiAssetBookConfig> base =
            dlob::load_multi_asset_book_configs(options.base_config);

        dlob::SequentialMultiAssetConfig config;
        config.duration_seconds = options.duration_seconds;
        config.book_count = options.books;
        config.seed = options.seed;
        config.book_configs = expand_templates(base, options.books);
        config.market_maker_quote_levels = options.quote_levels;
        config.market_maker_quote_quantity_growth = 2;
        config.enable_shared_market_maker_hedging = false;
        config.etf_arbitrage.enabled = options.arbitrage;
        config.etf_arbitrage.etf_book_id = 0;
        config.etf_arbitrage.trigger_bps = options.trigger_bps;
        config.etf_arbitrage.release_bps = options.release_bps;
        config.etf_arbitrage.etf_order_quantity = 100;
        config.etf_arbitrage.max_component_quantity = 10'000;

        const double window_ns_double = options.window_ms * 1'000'000.0;
        if (window_ns_double > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("window is too large");
        }
        const auto window_ns = static_cast<std::int64_t>(
            std::llround(window_ns_double));
        config.market_maker_quote_interval_ns = window_ns;
        config.etf_arbitrage.decision_interval_ns = window_ns;

        dlob::BatchedMpiMultiAssetSimulator simulator(
            MPI_COMM_WORLD, std::move(config), window_ns);
        const dlob::BatchedMpiMultiAssetResult result = simulator.run();
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "batched_mpi_multi_asset"
                      << " ranks=" << result.world_size
                      << " books=" << result.book_count
                      << " simulated_seconds=" << options.duration_seconds
                      << " window_ms=" << options.window_ms
                      << " windows=" << result.windows
                      << " wall_seconds=" << result.wall_seconds
                      << " initialization_seconds=" << result.initialization_seconds
                      << " max_compute_seconds=" << result.max_compute_seconds
                      << " max_communication_seconds="
                      << result.max_communication_seconds
                      << " controller_seconds=" << result.controller_seconds
                      << " load_imbalance=" << result.load_imbalance
                      << " processed_orders=" << result.processed_orders
                      << " trades=" << result.trades
                      << " mm_orders=" << result.market_maker_orders
                      << " arbitrage_orders=" << result.arbitrage_orders
                      << " stale_snapshot_uses=" << result.stale_snapshot_uses
                      << " state_hash=0x" << std::hex << result.state_hash << std::dec
                      << '\n';
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "batched_mpi_multi_asset rank " << rank
                  << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
}
