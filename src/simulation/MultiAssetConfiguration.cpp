#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace dlob {
namespace {

std::vector<std::string> split_csv_row(const std::string& row) {
    std::vector<std::string> fields;
    std::istringstream input(row);
    std::string field;
    while (std::getline(input, field, ',')) fields.push_back(field);
    if (!row.empty() && row.back() == ',') fields.emplace_back();
    return fields;
}

std::size_t column_index(const std::vector<std::string>& header,
                         const char* name) {
    const auto found = std::find(header.begin(), header.end(), name);
    if (found == header.end()) {
        throw std::runtime_error(std::string("book configuration is missing column ")
                                 + name);
    }
    return static_cast<std::size_t>(std::distance(header.begin(), found));
}

std::optional<std::size_t> optional_column_index(
    const std::vector<std::string>& header,
    const char* name) {
    const auto found = std::find(header.begin(), header.end(), name);
    if (found == header.end()) return std::nullopt;
    return static_cast<std::size_t>(std::distance(header.begin(), found));
}

template <typename Integer>
Integer parse_integer(const std::string& text,
                      const char* label,
                      std::size_t line_number) {
    std::size_t used = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(text, &used);
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + label
                                 + " at book-config line "
                                 + std::to_string(line_number));
    }
    if (used != text.size()
        || parsed < static_cast<long long>(std::numeric_limits<Integer>::min())
        || parsed > static_cast<long long>(std::numeric_limits<Integer>::max())) {
        throw std::runtime_error(std::string("invalid ") + label
                                 + " at book-config line "
                                 + std::to_string(line_number));
    }
    return static_cast<Integer>(parsed);
}

double parse_double(const std::string& text,
                    const char* label,
                    std::size_t line_number) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + label
                                 + " at book-config line "
                                 + std::to_string(line_number));
    }
    if (used != text.size() || !std::isfinite(value)) {
        throw std::runtime_error(std::string("invalid ") + label
                                 + " at book-config line "
                                 + std::to_string(line_number));
    }
    return value;
}

std::optional<std::array<double, 6>> load_calibrated_mu(
    const std::string& rates_file) {
    if (rates_file.empty()) return std::nullopt;
    std::ifstream input(rates_file);
    if (!input) {
        throw std::runtime_error("cannot open Hawkes rate calibration: "
                                 + rates_file);
    }
    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("empty Hawkes rate calibration: " + rates_file);
    }
    const std::vector<std::string> header = split_csv_row(header_line);
    const std::size_t event_index = column_index(header, "event_type");
    const std::size_t mu_index = column_index(header, "configured_mu");
    const std::size_t required_fields = std::max(event_index, mu_index) + 1U;
    constexpr std::array<const char*, 6> names{{
        "limit_buy", "limit_sell", "market_buy", "market_sell",
        "cancel_bid", "cancel_ask"}};
    std::array<double, 6> values{};
    std::array<bool, 6> observed{};
    std::string row;
    std::size_t line_number = 1;
    while (std::getline(input, row)) {
        ++line_number;
        if (row.empty()) continue;
        const std::vector<std::string> fields = split_csv_row(row);
        if (fields.size() < required_fields) {
            throw std::runtime_error("short Hawkes-rate row at line "
                                     + std::to_string(line_number));
        }
        const auto event = std::find_if(
            names.begin(), names.end(),
            [&](const char* name) { return fields[event_index] == name; });
        if (event == names.end()) continue;
        const std::size_t index = static_cast<std::size_t>(
            std::distance(names.begin(), event));
        if (observed[index]) {
            throw std::runtime_error("duplicate Hawkes event type: "
                                     + fields[event_index]);
        }
        values[index] = parse_double(fields[mu_index], "configured_mu", line_number);
        if (values[index] < 0.0) {
            throw std::runtime_error("configured_mu must be non-negative");
        }
        observed[index] = true;
    }
    if (std::find(observed.begin(), observed.end(), false) != observed.end()) {
        throw std::runtime_error("Hawkes rate CSV does not contain all six event types");
    }
    return values;
}

std::string data_file(const std::string& directory, const char* filename) {
    return (std::filesystem::path(directory) / filename).string();
}

void validate_book_configs(const std::vector<MultiAssetBookConfig>& books) {
    if (books.empty()) throw std::invalid_argument("at least one book is required");
    std::set<std::string> symbols;
    for (const MultiAssetBookConfig& book : books) {
        if (book.symbol.empty() || book.data_dir.empty()
            || !std::isfinite(book.fundamental_price_ticks)
            || book.fundamental_price_ticks <= 0.0
            || !std::isfinite(book.beta) || book.beta == 0.0
            || !std::isfinite(book.basket_weight) || book.basket_weight < 0.0
            || book.market_maker_quote_quantity < 0
            || book.target_spread_ticks <= 0
            || !std::isfinite(book.quote_improvement_probability)
            || book.quote_improvement_probability < 0.0
            || book.quote_improvement_probability > 1.0) {
            throw std::invalid_argument("invalid per-book multi-asset configuration");
        }
        const bool any_opening = book.initial_best_bid_ticks != 0
            || book.initial_best_ask_ticks != 0
            || book.initial_best_bid_depth != 0
            || book.initial_best_ask_depth != 0;
        if (any_opening
            && (book.initial_best_bid_ticks <= 0
                || book.initial_best_ask_ticks <= book.initial_best_bid_ticks
                || book.initial_best_bid_depth <= 0
                || book.initial_best_ask_depth <= 0)) {
            throw std::invalid_argument("invalid calibrated opening BBO");
        }
        if (!symbols.insert(book.symbol).second) {
            throw std::invalid_argument("duplicate symbol in multi-asset configuration");
        }
    }
}

} // namespace

std::vector<MultiAssetBookConfig> load_multi_asset_book_configs(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open book configuration: " + path.string());
    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("empty book configuration: " + path.string());
    }
    const std::vector<std::string> header = split_csv_row(header_line);
    const std::size_t id_col = column_index(header, "book_id");
    const std::size_t symbol_col = column_index(header, "symbol");
    const std::size_t data_col = column_index(header, "data_dir");
    const std::size_t rates_col = column_index(header, "hawkes_rates_file");
    const std::size_t fundamental_col = column_index(header, "fundamental_price_ticks");
    const std::size_t bid_col = column_index(header, "initial_best_bid_ticks");
    const std::size_t ask_col = column_index(header, "initial_best_ask_ticks");
    const std::size_t bid_depth_col = column_index(header, "initial_best_bid_depth");
    const std::size_t ask_depth_col = column_index(header, "initial_best_ask_depth");
    const std::size_t beta_col = column_index(header, "beta");
    const std::size_t weight_col = column_index(header, "basket_weight");
    const std::size_t quote_col = column_index(header, "market_maker_quote_quantity");
    const std::size_t spread_col = column_index(header, "target_spread_ticks");
    const std::optional<std::size_t> improvement_col = optional_column_index(
        header, "quote_improvement_probability");
    const std::size_t required = 1U + std::max({
        id_col, symbol_col, data_col, rates_col, fundamental_col, bid_col, ask_col,
        bid_depth_col, ask_depth_col, beta_col, weight_col, quote_col, spread_col});

    std::vector<MultiAssetBookConfig> books;
    std::string row;
    std::size_t line_number = 1;
    while (std::getline(input, row)) {
        ++line_number;
        if (row.empty()) continue;
        const std::vector<std::string> fields = split_csv_row(row);
        if (fields.size() < required) {
            throw std::runtime_error("short book-config row at line "
                                     + std::to_string(line_number));
        }
        const BookId id = parse_integer<BookId>(fields[id_col], "book_id", line_number);
        if (id != static_cast<BookId>(books.size())) {
            throw std::runtime_error("book_id values must be contiguous from zero");
        }
        MultiAssetBookConfig book;
        book.symbol = fields[symbol_col];
        book.data_dir = fields[data_col];
        book.hawkes_rates_file = fields[rates_col];
        book.fundamental_price_ticks = parse_double(
            fields[fundamental_col], "fundamental_price_ticks", line_number);
        book.initial_best_bid_ticks = parse_integer<std::int32_t>(
            fields[bid_col], "initial_best_bid_ticks", line_number);
        book.initial_best_ask_ticks = parse_integer<std::int32_t>(
            fields[ask_col], "initial_best_ask_ticks", line_number);
        book.initial_best_bid_depth = parse_integer<std::int32_t>(
            fields[bid_depth_col], "initial_best_bid_depth", line_number);
        book.initial_best_ask_depth = parse_integer<std::int32_t>(
            fields[ask_depth_col], "initial_best_ask_depth", line_number);
        book.beta = parse_double(fields[beta_col], "beta", line_number);
        book.basket_weight = parse_double(
            fields[weight_col], "basket_weight", line_number);
        book.market_maker_quote_quantity = parse_integer<std::int32_t>(
            fields[quote_col], "market_maker_quote_quantity", line_number);
        book.target_spread_ticks = parse_integer<std::int32_t>(
            fields[spread_col], "target_spread_ticks", line_number);
        if (improvement_col.has_value()) {
            if (*improvement_col >= fields.size()) {
                throw std::runtime_error(
                    "short quote-improvement field at book-config line "
                    + std::to_string(line_number));
            }
            book.quote_improvement_probability = parse_double(
                fields[*improvement_col], "quote_improvement_probability",
                line_number);
        }
        books.push_back(std::move(book));
    }
    validate_book_configs(books);
    return books;
}

std::vector<MultiAssetBookConfig> resolve_multi_asset_book_configs(
    const SequentialMultiAssetConfig& config) {
    if (!config.book_configs.empty()) {
        if (config.book_configs.size() != static_cast<std::size_t>(config.book_count)) {
            throw std::invalid_argument("book_count differs from per-book configuration");
        }
        validate_book_configs(config.book_configs);
        return config.book_configs;
    }
    std::vector<MultiAssetBookConfig> books;
    books.reserve(static_cast<std::size_t>(config.book_count));
    for (int index = 0; index < config.book_count; ++index) {
        MultiAssetBookConfig book;
        book.symbol = config.book_count == 1
            ? "QQQ" : "BOOK_" + std::to_string(index);
        book.data_dir = config.data_dir;
        book.hawkes_rates_file = config.hawkes_rates_file;
        if (book.hawkes_rates_file.empty()) {
            const std::filesystem::path fallback =
                std::filesystem::path(config.data_dir) / "hawkes_rates_qqq_20200130.csv";
            if (std::filesystem::exists(fallback)) {
                book.hawkes_rates_file = fallback.string();
            }
        }
        book.fundamental_price_ticks = config.fundamental_price_ticks;
        books.push_back(std::move(book));
    }
    validate_book_configs(books);
    return books;
}

BackgroundHawkesConfig make_multi_asset_background_config(
    const SequentialMultiAssetConfig& config,
    const MultiAssetBookConfig& book,
    BookId book_id) {
    BackgroundHawkesConfig background;
    background.seed = stable_sequence(background_entity(book_id), config.seed);
    background.tick_size = config.tick_size;
    background.quote_improvement_probability =
        book.quote_improvement_probability;
    const auto calibrated = load_calibrated_mu(book.hawkes_rates_file);
    if (calibrated.has_value()) background.mu = *calibrated;
    background.limit_buy_quantity_file = data_file(book.data_dir, "limit_buy_quantity_distribution.txt");
    background.limit_sell_quantity_file = data_file(book.data_dir, "limit_sell_quantity_distribution.txt");
    background.market_buy_quantity_file = data_file(book.data_dir, "market_buy_quantity_distribution.txt");
    background.market_sell_quantity_file = data_file(book.data_dir, "market_sell_quantity_distribution.txt");
    background.cancel_bid_quantity_file = data_file(book.data_dir, "cancel_bid_quantity_distribution.txt");
    background.cancel_ask_quantity_file = data_file(book.data_dir, "cancel_ask_quantity_distribution.txt");
    background.limit_buy_distance_file = data_file(book.data_dir, "limit_buy_distance_distribution.txt");
    background.limit_sell_distance_file = data_file(book.data_dir, "limit_sell_distance_distribution.txt");
    background.cancel_bid_distance_file = data_file(book.data_dir, "cancel_bid_distance_distribution.txt");
    background.cancel_ask_distance_file = data_file(book.data_dir, "cancel_ask_distance_distribution.txt");
    return background;
}

SharedMarketMakerConfig make_multi_asset_market_maker_config(
    const SequentialMultiAssetConfig& config,
    const std::vector<MultiAssetBookConfig>& books) {
    SharedMarketMakerConfig maker;
    maker.logical_owner_id = 900'001;
    maker.message_source_rank = 0;
    maker.quote_quantity = config.market_maker_order_quantity;
    maker.quote_levels = config.market_maker_quote_levels;
    maker.quote_quantity_growth = config.market_maker_quote_quantity_growth;
    maker.quote_half_spread_ticks = config.tick_size;
    maker.price_tick_size = config.tick_size;
    maker.order_latency_ns = config.market_maker_order_latency_ns;
    maker.exposure_threshold = config.market_maker_exposure_threshold;
    maker.enable_cross_book_hedging =
        config.enable_shared_market_maker_hedging;
    maker.hedge_lot_size = config.hedge_lot_size;
    maker.max_hedge_quantity = config.max_hedge_quantity;
    maker.report_latency_ns = config.report_latency_ns;
    maker.reaction_latency_ns = config.cross_book_reaction_latency_ns;
    maker.network_latency_ns = config.hedge_order_latency_ns;
    maker.books.reserve(books.size());
    double component_weight_sum = 0.0;
    for (std::size_t index = 1; index < books.size(); ++index) {
        component_weight_sum += books[index].basket_weight;
    }
    for (std::size_t index = 0; index < books.size(); ++index) {
        const BookId id = static_cast<BookId>(index);
        const BookId hedge = books.size() == 1U
            ? id : (index == 0U ? BookId{1} : BookId{0});
        SharedMarketMakerBookConfig book_config{
            id, books[index].beta, hedge,
            books[index].market_maker_quote_quantity,
            books[index].target_spread_ticks, {}};
        if (books.size() > 1U) {
            if (index == 0U) {
                for (std::size_t component = 1;
                     component < books.size(); ++component) {
                    const double weight = component_weight_sum > 0.0
                        ? books[component].basket_weight : 1.0;
                    book_config.hedge_routes.push_back(
                        SharedMarketMakerHedgeRoute{
                            static_cast<BookId>(component), weight});
                }
            } else {
                book_config.hedge_routes.push_back(
                    SharedMarketMakerHedgeRoute{BookId{0}, 1.0});
            }
        }
        maker.books.push_back(std::move(book_config));
    }
    return maker;
}

} // namespace dlob
