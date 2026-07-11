#include "FullSimulator.hpp"

#include "BackgroundHawkesAgent.hpp"
#include "EmpiricalDistribution.hpp"
#include "LargeInstitutionalAgent.hpp"
#include "LimitOrderBook.hpp"
#include "MomentumAgent.hpp"
#include "Order.hpp"
#include "PassiveMarketMakerAgent.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {
int scaled_quantity(int q, double scale) {
    return std::max(1, static_cast<int>(std::llround(static_cast<double>(q) * scale)));
}

void create_initial_book(LimitOrderBook& book, const SimulationConfig& config) {
    const double s = config.initial_depth_scale;
    std::uint64_t id = 1;
    const int bid_prices[] = {2203400,2203300,2203200,2203100,2203000,2202900,2202800,2202600,2202500,2202400};
    const int bid_qty[]    = {1623,1723,2100,1100,1200,200,564,500,700,200};
    const int ask_prices[] = {2203700,2203800,2203900,2204000,2204100,2204200,2204300,2204400,2204500,2204600};
    const int ask_qty[]    = {823,823,1823,1923,1923,1223,823,200,823,823};
    for (int i = 0; i < 10; ++i) {
        book.add_limit_order(Order{id++, 0, OrderType::Limit, Side::Buy, scaled_quantity(bid_qty[i], s), bid_prices[i], 0});
    }
    for (int i = 0; i < 10; ++i) {
        book.add_limit_order(Order{id++, 0, OrderType::Limit, Side::Sell, scaled_quantity(ask_qty[i], s), ask_prices[i], 0});
    }
}

void record_snapshot(
    const LimitOrderBook& book,
    const std::uint64_t cumulative_aggressive_before,
    SimulationMetrics& metrics
) {
    if (book.has_bid() && book.has_ask() && book.spread_ticks() >= 0) {
        metrics.mid_prices.push_back(book.mid_price());
        metrics.spreads.push_back(static_cast<double>(book.spread_ticks()));
        metrics.best_bid_depths.push_back(static_cast<double>(book.quantity_at_best_bid()));
        metrics.best_ask_depths.push_back(static_cast<double>(book.quantity_at_best_ask()));
        const std::uint64_t cumulative_now = book.cumulative_aggressive_buy_quantity() + book.cumulative_aggressive_sell_quantity();
        const double volume = cumulative_now >= cumulative_aggressive_before
            ? static_cast<double>(cumulative_now - cumulative_aggressive_before)
            : 0.0;
        metrics.aggressive_volume_by_sample.push_back(volume);
    }
}

int sample_scaled(EmpiricalDistribution& distribution, std::mt19937_64& rng, double scale) {
    return std::max(1, static_cast<int>(std::llround(static_cast<double>(distribution.sample(rng)) * scale)));
}

int distance_sample(EmpiricalDistribution& distribution, std::mt19937_64& rng, int shift) {
    return std::max(0, distribution.sample(rng) + shift);
}

std::uint64_t cumulative_aggressive_quantity(const LimitOrderBook& book) {
    return book.cumulative_aggressive_buy_quantity() + book.cumulative_aggressive_sell_quantity();
}
}

SimulationMetrics run_full_simulation(const SimulationConfig& config) {
    SimulationMetrics metrics;
    metrics.calibration_id = config.calibration_id;
    metrics.seed = config.seed;
    metrics.duration_seconds = static_cast<double>(config.end_time_ns - config.start_time_ns) / 1e9;
    metrics.sample_interval_seconds = static_cast<double>(config.sample_interval_ns) / 1e9;

    const auto whole_start = std::chrono::steady_clock::now();

    LimitOrderBook book(config.tick_size);
    create_initial_book(book, config);
    std::uint64_t next_order_id = 21;

    const auto hawkes_start = std::chrono::steady_clock::now();
    BackgroundHawkesAgent background_agent(config);
    const std::vector<HawkesEvent> events = background_agent.simulate(config.start_time_ns, config.end_time_ns);
    const auto hawkes_end = std::chrono::steady_clock::now();
    metrics.hawkes_generation_wall_seconds = std::chrono::duration<double>(hawkes_end - hawkes_start).count();

    for (const HawkesEvent& e : events) {
        const int type_index = static_cast<int>(e.type);
        if (type_index >= 0 && type_index < NUM_EVENT_TYPES) metrics.raw_hawkes_counts[static_cast<std::size_t>(type_index)] += 1.0;
    }
    metrics.raw_hawkes_event_count = static_cast<double>(events.size());

    std::mt19937_64 rng(config.seed + 1ULL);
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);

    EmpiricalDistribution limit_buy_quantity;   limit_buy_quantity.set_fallback(25, 200);   limit_buy_quantity.load_from_csv(config.limit_buy_quantity_file, config.quantity_column_name);
    EmpiricalDistribution limit_sell_quantity;  limit_sell_quantity.set_fallback(25, 200);  limit_sell_quantity.load_from_csv(config.limit_sell_quantity_file, config.quantity_column_name);
    EmpiricalDistribution market_buy_quantity;  market_buy_quantity.set_fallback(25, 700);  market_buy_quantity.load_from_csv(config.market_buy_quantity_file, config.quantity_column_name);
    EmpiricalDistribution market_sell_quantity; market_sell_quantity.set_fallback(25, 700); market_sell_quantity.load_from_csv(config.market_sell_quantity_file, config.quantity_column_name);
    EmpiricalDistribution cancel_bid_quantity;  cancel_bid_quantity.set_fallback(25, 500);  cancel_bid_quantity.load_from_csv(config.cancel_bid_quantity_file, config.quantity_column_name);
    EmpiricalDistribution cancel_ask_quantity;  cancel_ask_quantity.set_fallback(25, 500);  cancel_ask_quantity.load_from_csv(config.cancel_ask_quantity_file, config.quantity_column_name);

    EmpiricalDistribution limit_buy_distance;   limit_buy_distance.set_fallback(0, 5);   limit_buy_distance.load_from_csv(config.limit_buy_distance_file, config.distance_column_name);
    EmpiricalDistribution limit_sell_distance;  limit_sell_distance.set_fallback(0, 5);  limit_sell_distance.load_from_csv(config.limit_sell_distance_file, config.distance_column_name);
    EmpiricalDistribution cancel_bid_distance;  cancel_bid_distance.set_fallback(0, 5);  cancel_bid_distance.load_from_csv(config.cancel_bid_distance_file, config.distance_column_name);
    EmpiricalDistribution cancel_ask_distance;  cancel_ask_distance.set_fallback(0, 5);  cancel_ask_distance.load_from_csv(config.cancel_ask_distance_file, config.distance_column_name);

    std::vector<PassiveMarketMakerAgent> market_makers;
    market_makers.reserve(config.num_market_maker_agents);
    for (int i = 0; i < config.num_market_maker_agents; ++i) {
        market_makers.emplace_back(
            config.tick_size,
            config.market_maker_num_levels,
            config.market_maker_order_quantity,
            config.market_maker_level_spacing_ticks,
            config.market_maker_min_spread_ticks * config.tick_size,
            config.market_maker_quote_skip_probability,
            config.market_maker_recovery_quote_skip_probability,
            config.seed + static_cast<std::uint64_t>(i) * 1000ULL
        );
    }

    std::vector<MomentumAgent> momentum_agents;
    momentum_agents.reserve(config.num_momentum_agents);
    for (int i = 0; i < config.num_momentum_agents; ++i) {
        momentum_agents.emplace_back(
            config.momentum_base_lookback_ns,
            config.momentum_order_quantity,
            config.momentum_threshold_ticks,
            config.tick_size,
            config.momentum_order_flow_imbalance_threshold,
            config.momentum_depth_imbalance_threshold,
            config.momentum_strong_depth_imbalance_threshold
        );
    }

    std::vector<LargeInstitutionalAgent> institutional_agents;
    const int institutional_count = config.num_buy_institutional_agents + config.num_sell_institutional_agents;
    institutional_agents.reserve(institutional_count);
    int buy_created = 0;
    int sell_created = 0;
    for (int i = 0; i < institutional_count; ++i) {
        Side side;
        if (buy_created < config.num_buy_institutional_agents && (i % 2 == 0 || sell_created >= config.num_sell_institutional_agents)) {
            side = Side::Buy;
            ++buy_created;
        } else {
            side = Side::Sell;
            ++sell_created;
        }
        const int parent = side == Side::Buy ? config.buy_institutional_parent_quantity : config.sell_institutional_parent_quantity;
        const std::int64_t start = config.institutional_base_start_time_ns + static_cast<std::int64_t>(i) * config.institutional_start_spacing_ns;
        const std::int64_t end = std::min(config.end_time_ns, start + config.institutional_duration_ns);
        institutional_agents.emplace_back(side, parent, config.institutional_child_quantity, config.institutional_participation_cap, start, end, config.seed + static_cast<std::uint64_t>(i) * 99991ULL);
    }

    std::int64_t next_sample_time_ns = config.start_time_ns;
    std::int64_t next_market_maker_time_ns = config.start_time_ns;
    std::int64_t next_momentum_time_ns = config.start_time_ns;
    std::int64_t next_institutional_time_ns = config.start_time_ns;
    std::uint64_t last_snapshot_aggressive_quantity = cumulative_aggressive_quantity(book);

    const auto app_start = std::chrono::steady_clock::now();

    for (const HawkesEvent& event : events) {
        while (next_sample_time_ns <= event.time_ns && next_sample_time_ns <= config.end_time_ns) {
            record_snapshot(book, last_snapshot_aggressive_quantity, metrics);
            last_snapshot_aggressive_quantity = cumulative_aggressive_quantity(book);
            next_sample_time_ns += config.sample_interval_ns;
        }

        while (next_market_maker_time_ns <= event.time_ns && next_market_maker_time_ns <= config.end_time_ns) {
            const std::uint64_t before_id = next_order_id;
            for (auto& agent : market_makers) {
                metrics.market_maker_cancel_count += static_cast<double>(agent.wake_up(book, next_order_id, next_market_maker_time_ns));
            }
            metrics.market_maker_order_count += static_cast<double>(next_order_id - before_id);
            next_market_maker_time_ns += config.market_maker_interval_ns;
        }

        while (next_momentum_time_ns <= event.time_ns && next_momentum_time_ns <= config.end_time_ns) {
            for (auto& agent : momentum_agents) {
                agent.record_mid_price(book, next_momentum_time_ns);
                const std::uint64_t before = cumulative_aggressive_quantity(book);
                const int executed = agent.wake_up(book, next_momentum_time_ns);
                const std::uint64_t after = cumulative_aggressive_quantity(book);
                if (after > before || executed > 0) {
                    metrics.momentum_market_order_count += 1.0;
                    metrics.strategic_market_order_executed_quantity += static_cast<double>(std::max(0, executed));
                }
            }
            next_momentum_time_ns += config.momentum_interval_ns;
        }

        while (next_institutional_time_ns <= event.time_ns && next_institutional_time_ns <= config.end_time_ns) {
            for (auto& agent : institutional_agents) {
                const int executed = agent.wake_up(book, next_institutional_time_ns);
                if (executed > 0) {
                    metrics.institutional_market_order_count += 1.0;
                    metrics.strategic_market_order_executed_quantity += static_cast<double>(executed);
                }
            }
            next_institutional_time_ns += config.institutional_interval_ns;
        }

        switch (event.type) {
            case EventType::LimitBuy: {
                if (!book.has_bid()) break;
                int quantity = sample_scaled(limit_buy_quantity, rng, 1.0);
                const int distance = distance_sample(limit_buy_distance, rng, config.limit_distance_shift_ticks);
                int price = book.best_bid() - distance * config.tick_size;
                if (book.has_ask() && book.best_ask() - book.best_bid() >= 3 * config.tick_size && uniform_01(rng) < config.quote_improvement_probability) {
                    price = std::min(book.best_bid() + config.tick_size, book.best_ask() - config.tick_size);
                }
                if (price > 0 && (!book.has_ask() || price < book.best_ask())) {
                    book.add_limit_order(Order{next_order_id++, event.time_ns, OrderType::Limit, Side::Buy, quantity, price, 0});
                    metrics.limit_order_count += 1.0;
                    metrics.limit_buy_count += 1.0;
                }
                break;
            }
            case EventType::LimitSell: {
                if (!book.has_ask()) break;
                int quantity = sample_scaled(limit_sell_quantity, rng, 1.0);
                const int distance = distance_sample(limit_sell_distance, rng, config.limit_distance_shift_ticks);
                int price = book.best_ask() + distance * config.tick_size;
                if (book.has_bid() && book.best_ask() - book.best_bid() >= 3 * config.tick_size && uniform_01(rng) < config.quote_improvement_probability) {
                    price = std::max(book.best_ask() - config.tick_size, book.best_bid() + config.tick_size);
                }
                if (price > 0 && (!book.has_bid() || price > book.best_bid())) {
                    book.add_limit_order(Order{next_order_id++, event.time_ns, OrderType::Limit, Side::Sell, quantity, price, 0});
                    metrics.limit_order_count += 1.0;
                    metrics.limit_sell_count += 1.0;
                }
                break;
            }
            case EventType::MarketBuy: {
                const bool had_ask = book.has_ask();
                const int best_ask_before = had_ask ? book.best_ask() : 0;
                const int quantity = sample_scaled(market_buy_quantity, rng, config.market_order_quantity_scale);
                metrics.market_order_count += 1.0;
                metrics.market_buy_count += 1.0;
                const int executed = book.submit_market_order(Side::Buy, quantity, event.time_ns);
                metrics.market_order_executed_quantity += static_cast<double>(std::max(0, executed));
                if (had_ask && executed > 0 && (!book.has_ask() || book.best_ask() != best_ask_before)) metrics.market_order_best_removal_count += 1.0;
                break;
            }
            case EventType::MarketSell: {
                const bool had_bid = book.has_bid();
                const int best_bid_before = had_bid ? book.best_bid() : 0;
                const int quantity = sample_scaled(market_sell_quantity, rng, config.market_order_quantity_scale);
                metrics.market_order_count += 1.0;
                metrics.market_sell_count += 1.0;
                const int executed = book.submit_market_order(Side::Sell, quantity, event.time_ns);
                metrics.market_order_executed_quantity += static_cast<double>(std::max(0, executed));
                if (had_bid && executed > 0 && (!book.has_bid() || book.best_bid() != best_bid_before)) metrics.market_order_best_removal_count += 1.0;
                break;
            }
            case EventType::CancelBid: {
                const bool had_bid = book.has_bid();
                const int best_bid_before = had_bid ? book.best_bid() : 0;
                const int quantity = sample_scaled(cancel_bid_quantity, rng, config.cancel_quantity_scale);
                const int distance = distance_sample(cancel_bid_distance, rng, 0);
                metrics.cancel_count += 1.0;
                book.cancel_at_distance(Side::Buy, distance, quantity, config.tick_size);
                if (had_bid && (!book.has_bid() || book.best_bid() != best_bid_before)) metrics.cancel_best_removal_count += 1.0;
                break;
            }
            case EventType::CancelAsk: {
                const bool had_ask = book.has_ask();
                const int best_ask_before = had_ask ? book.best_ask() : 0;
                const int quantity = sample_scaled(cancel_ask_quantity, rng, config.cancel_quantity_scale);
                const int distance = distance_sample(cancel_ask_distance, rng, 0);
                metrics.cancel_count += 1.0;
                metrics.cancel_ask_count += 1.0;
                book.cancel_at_distance(Side::Sell, distance, quantity, config.tick_size);
                if (had_ask && (!book.has_ask() || book.best_ask() != best_ask_before)) metrics.cancel_best_removal_count += 1.0;
                break;
            }
        }
    }

    while (next_sample_time_ns <= config.end_time_ns) {
        record_snapshot(book, last_snapshot_aggressive_quantity, metrics);
        last_snapshot_aggressive_quantity = cumulative_aggressive_quantity(book);
        next_sample_time_ns += config.sample_interval_ns;
    }

    const auto app_end = std::chrono::steady_clock::now();
    metrics.order_book_application_wall_seconds = std::chrono::duration<double>(app_end - app_start).count();

    metrics.market_order_best_removal_rate = metrics.market_order_count > 0.0
        ? metrics.market_order_best_removal_count / metrics.market_order_count
        : 0.0;
    metrics.cancel_best_removal_rate = metrics.cancel_count > 0.0
        ? metrics.cancel_best_removal_count / metrics.cancel_count
        : 0.0;

    metrics.total_message_count = metrics.raw_hawkes_event_count
        + metrics.market_maker_order_count
        + metrics.market_maker_cancel_count
        + metrics.momentum_market_order_count
        + metrics.institutional_market_order_count;

    const auto whole_end = std::chrono::steady_clock::now();
    metrics.total_wall_seconds = std::chrono::duration<double>(whole_end - whole_start).count();
    return metrics;
}
