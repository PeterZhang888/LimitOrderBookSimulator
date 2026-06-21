#include "SimulationRunner.hpp"

#include "LimitOrderBook.hpp"
#include "HawkesProcess.hpp"
#include "EmpiricalDistribution.hpp"
#include "MarketMakerAgent.hpp"
#include "MomentumAgent.hpp"
#include "LargeInstitutionalAgent.hpp"
#include "AutonomousMarketMakerAgent.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

double mean_value(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    return std::accumulate(values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double probability) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const double position =
        probability * static_cast<double>(values.size() - 1);

    const std::size_t lower =
        static_cast<std::size_t>(std::floor(position));

    const std::size_t upper =
        static_cast<std::size_t>(std::ceil(position));

    if (lower == upper) {
        return values[lower];
    }

    const double weight =
        position - static_cast<double>(lower);

    return values[lower] * (1.0 - weight)
        + values[upper] * weight;
}

double standard_deviation(
    const std::vector<double>& values
) {
    if (values.size() < 2) {
        return 0.0;
    }

    const double average = mean_value(values);
    double squared_sum = 0.0;

    for (double value : values) {
        const double difference = value - average;
        squared_sum += difference * difference;
    }

    return std::sqrt(
        squared_sum
        / static_cast<double>(values.size() - 1)
    );
}

double squared_relative_loss(
    double simulated,
    double target
) {
    if (std::abs(target) < 1e-12) {
        const double difference = simulated - target;
        return difference * difference;
    }

    const double relative_difference =
        (simulated - target) / target;

    return relative_difference * relative_difference;
}

void record_snapshot(
    const LimitOrderBook& book,
    std::int64_t sample_time_ns,
    std::vector<double>& spreads,
    std::vector<double>& mid_prices,
    std::ofstream* output,
    bool write_output
) {
    const double sample_time_seconds =
        static_cast<double>(sample_time_ns)
        / 1'000'000'000.0;

    if (book.has_bid() && book.has_ask()) {
        const int best_bid = book.best_bid();
        const int best_ask = book.best_ask();
        const int spread = best_ask - best_bid;

        if (spread < 0) {
            return;
        }

        const double mid_price =
            0.5
            * static_cast<double>(
                best_bid + best_ask
            );

        spreads.push_back(
            static_cast<double>(spread)
        );

        mid_prices.push_back(mid_price);

        if (
            write_output
            && output != nullptr
            && output->is_open()
        ) {
            (*output)
                << sample_time_ns << ","
                << sample_time_seconds << ","
                << best_bid << ","
                << best_ask << ","
                << spread << ","
                << mid_price << "\n";
        }

        return;
    }

    if (
        write_output
        && output != nullptr
        && output->is_open()
    ) {
        (*output)
            << sample_time_ns << ","
            << sample_time_seconds
            << ",,,,\n";
    }
}

int scaled_quantity(
    int quantity,
    double scale
) {
    const int result = static_cast<int>(
        std::round(
            static_cast<double>(quantity) * scale
        )
    );

    return std::max(1, result);
}

void create_initial_book(
    LimitOrderBook& book,
    const SimulationConfig& config
) {
    const double scale = config.initial_depth_scale;

    book.add_limit_order(Order{
        1,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(1623, scale),
        2203400
    });

    book.add_limit_order(Order{
        2,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(1723, scale),
        2203300
    });

    book.add_limit_order(Order{
        3,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(2100, scale),
        2203200
    });

    book.add_limit_order(Order{
        4,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(1100, scale),
        2203100
    });

    book.add_limit_order(Order{
        5,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(1200, scale),
        2203000
    });

    book.add_limit_order(Order{
        6,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(200, scale),
        2202900
    });

    book.add_limit_order(Order{
        7,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(564, scale),
        2202800
    });

    book.add_limit_order(Order{
        8,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(500, scale),
        2202600
    });

    book.add_limit_order(Order{
        9,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(700, scale),
        2202500
    });

    book.add_limit_order(Order{
        10,
        0,
        OrderType::Limit,
        Side::Buy,
        scaled_quantity(200, scale),
        2202400
    });

    book.add_limit_order(Order{
        11,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(823, scale),
        2203700
    });

    book.add_limit_order(Order{
        12,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(823, scale),
        2203800
    });

    book.add_limit_order(Order{
        13,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(1823, scale),
        2203900
    });

    book.add_limit_order(Order{
        14,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(1923, scale),
        2204000
    });

    book.add_limit_order(Order{
        15,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(1923, scale),
        2204100
    });

    book.add_limit_order(Order{
        16,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(1223, scale),
        2204200
    });

    book.add_limit_order(Order{
        17,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(823, scale),
        2204300
    });

    book.add_limit_order(Order{
        18,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(200, scale),
        2204400
    });

    book.add_limit_order(Order{
        19,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(823, scale),
        2204500
    });

    book.add_limit_order(Order{
        20,
        0,
        OrderType::Limit,
        Side::Sell,
        scaled_quantity(823, scale),
        2204600
    });
}

bool valid_price(long long price) {
    return (
        price > 0
        && price
            <= static_cast<long long>(
                std::numeric_limits<int>::max()
            )
    );
}

// ============================================================
// Adaptive autonomous market-maker configuration
// ============================================================

// Six tunable strategy parameters.
constexpr int
    AUTONOMOUS_MM_BASE_HALF_SPREAD_TICKS = 2;

constexpr int
    AUTONOMOUS_MM_BASE_ORDER_QUANTITY = 10;

constexpr double
    AUTONOMOUS_MM_INVENTORY_RISK_STRENGTH = 2.0;

constexpr double
    AUTONOMOUS_MM_VOLATILITY_STRENGTH = 0.50;

constexpr double
    AUTONOMOUS_MM_SIGNAL_STRENGTH = 0.50;

constexpr double
    AUTONOMOUS_MM_AGGRESSIVE_INVENTORY_RATIO = 0.80;

// Fixed risk and engineering settings.
constexpr int
    AUTONOMOUS_MM_MAX_INVENTORY = 500;

constexpr std::int64_t
    AUTONOMOUS_MM_REFRESH_INTERVAL_NS =
        1000LL * 1000000LL;

void write_autonomous_mm_path_result(
    const SimulationConfig& config,
    const AutonomousMarketMakerAgent& autonomous_mm,
    double autonomous_mm_final_pnl,
    const CalibrationResult& result
) {
    const std::string filename =
        "mc_full_day_results/autonomous_mm_path_"
        + std::to_string(config.calibration_id)
        + ".csv";

    std::ofstream output(filename);

    if (!output.is_open()) {
        return;
    }

    output
        << "path_id,seed,pnl,inventory,cash,"
        << "max_drawdown,max_abs_inventory,"
        << "fills,traded_quantity,"
        << "final_norm_mid,total_loss,"
        << "mean_spread,median_spread,"
        << "p90_spread,p95_spread,mid_move_rate,"
        << "base_half_spread_ticks,"
        << "base_order_quantity,"
        << "max_inventory,"
        << "inventory_risk_strength,"
        << "volatility_strength,"
        << "signal_strength,"
        << "aggressive_inventory_ratio,"
        << "refresh_interval_ms,"
        << "final_volatility_ticks,"
        << "final_signal,"
        << "defensive_actions,"
        << "aggressive_actions\n";

    output
        << config.calibration_id << ","
        << config.seed << ","
        << autonomous_mm_final_pnl << ","
        << autonomous_mm.inventory() << ","
        << autonomous_mm.cash() << ","
        << autonomous_mm.max_drawdown() << ","
        << autonomous_mm.max_abs_inventory() << ","
        << autonomous_mm.number_of_fills() << ","
        << autonomous_mm.total_traded_quantity() << ","
        << result.final_norm_mid << ","
        << result.total_loss << ","
        << result.mean_spread << ","
        << result.median_spread << ","
        << result.p90_spread << ","
        << result.p95_spread << ","
        << result.mid_move_rate << ","
        << AUTONOMOUS_MM_BASE_HALF_SPREAD_TICKS << ","
        << AUTONOMOUS_MM_BASE_ORDER_QUANTITY << ","
        << AUTONOMOUS_MM_MAX_INVENTORY << ","
        << AUTONOMOUS_MM_INVENTORY_RISK_STRENGTH << ","
        << AUTONOMOUS_MM_VOLATILITY_STRENGTH << ","
        << AUTONOMOUS_MM_SIGNAL_STRENGTH << ","
        << AUTONOMOUS_MM_AGGRESSIVE_INVENTORY_RATIO << ","
        << AUTONOMOUS_MM_REFRESH_INTERVAL_NS
               / 1000000LL
        << ","
        << autonomous_mm.current_volatility_ticks()
        << ","
        << autonomous_mm.current_signal()
        << ","
        << autonomous_mm.defensive_actions()
        << ","
        << autonomous_mm.aggressive_actions()
        << "\n";
}

} // namespace

CalibrationResult run_single_simulation(
    const SimulationConfig& config
) {
    CalibrationResult result;

    result.calibration_id =
        config.calibration_id;

    result.hawkes_beta =
        config.hawkes_beta;

    result.limit_branching =
        config.limit_branching;

    result.market_branching =
        config.market_branching;

    result.cancel_branching =
        config.cancel_branching;

    result.num_market_maker_agents =
        config.num_market_maker_agents;

    result.market_maker_order_quantity =
        config.market_maker_order_quantity;

    result.market_maker_min_spread_ticks =
        config.market_maker_min_spread_ticks;

    result.num_momentum_agents =
        config.num_momentum_agents;

    result.momentum_order_quantity =
        config.momentum_order_quantity;

    result.momentum_threshold_ticks =
        config.momentum_threshold_ticks;

    if (config.use_mixed_institutional_sides) {
        result.num_institutional_agents =
            config.num_buy_institutional_agents
            + config.num_sell_institutional_agents;
    } else {
        result.num_institutional_agents =
            config.num_institutional_agents;
    }

    result.institutional_parent_quantity =
        config.institutional_parent_quantity;

    result.institutional_child_quantity =
        config.institutional_child_quantity;

    result.institutional_interval_seconds =
        static_cast<int>(
            config.institutional_interval_ns
            / 1'000'000'000LL
        );

    LimitOrderBook book;
    create_initial_book(book, config);

    std::uint64_t next_order_id = 21;

    using Vector6 = HawkesProcess::Vector6;
    using Matrix6 = HawkesProcess::Matrix6;

    Vector6 empirical_rates = {
        config.empirical_rate_limit_buy,
        config.empirical_rate_limit_sell,
        config.empirical_rate_market_buy,
        config.empirical_rate_market_sell,
        config.empirical_rate_cancel_bid,
        config.empirical_rate_cancel_ask
    };

    Vector6 mu{};
    Matrix6 alpha{};
    Matrix6 branching{};

    branching[0][0] =
        config.limit_branching;

    branching[1][1] =
        config.limit_branching;

    branching[2][2] =
        config.market_branching;

    branching[3][3] =
        config.market_branching;

    branching[4][4] =
        config.cancel_branching;

    branching[5][5] =
        config.cancel_branching;

    for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
        for (int j = 0; j < NUM_EVENT_TYPES; ++j) {
            alpha[i][j] =
                config.hawkes_beta
                * branching[i][j];
        }
    }

    for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
        double endogenous_rate = 0.0;

        for (int j = 0; j < NUM_EVENT_TYPES; ++j) {
            endogenous_rate +=
                branching[i][j]
                * empirical_rates[j];
        }

        mu[i] =
            empirical_rates[i]
            - endogenous_rate;

        if (mu[i] < 1e-9) {
            mu[i] = 1e-9;
        }
    }

    HawkesProcess hawkes(
        mu,
        alpha,
        config.hawkes_beta,
        config.seed
    );

    std::mt19937_64 rng(
        config.seed + 1
    );

    std::uniform_real_distribution<double>
        uniform_01(0.0, 1.0);

    const double quote_improvement_probability =
        std::clamp(
            config.quote_improvement_probability,
            0.0,
            1.0
        );

    EmpiricalDistribution limit_buy_quantity;
    EmpiricalDistribution limit_sell_quantity;
    EmpiricalDistribution market_buy_quantity;
    EmpiricalDistribution market_sell_quantity;
    EmpiricalDistribution cancel_bid_quantity;
    EmpiricalDistribution cancel_ask_quantity;

    limit_buy_quantity.load_from_csv(
        config.limit_buy_quantity_file,
        config.quantity_column_name
    );

    limit_sell_quantity.load_from_csv(
        config.limit_sell_quantity_file,
        config.quantity_column_name
    );

    market_buy_quantity.load_from_csv(
        config.market_buy_quantity_file,
        config.quantity_column_name
    );

    market_sell_quantity.load_from_csv(
        config.market_sell_quantity_file,
        config.quantity_column_name
    );

    cancel_bid_quantity.load_from_csv(
        config.cancel_bid_quantity_file,
        config.quantity_column_name
    );

    cancel_ask_quantity.load_from_csv(
        config.cancel_ask_quantity_file,
        config.quantity_column_name
    );

    EmpiricalDistribution limit_buy_distance;
    EmpiricalDistribution limit_sell_distance;
    EmpiricalDistribution cancel_bid_distance;
    EmpiricalDistribution cancel_ask_distance;

    limit_buy_distance.load_from_csv(
        config.limit_buy_distance_file,
        config.distance_column_name
    );

    limit_sell_distance.load_from_csv(
        config.limit_sell_distance_file,
        config.distance_column_name
    );

    cancel_bid_distance.load_from_csv(
        config.cancel_bid_distance_file,
        config.distance_column_name
    );

    cancel_ask_distance.load_from_csv(
        config.cancel_ask_distance_file,
        config.distance_column_name
    );

    std::vector<MarketMakerAgent> market_makers;

    market_makers.reserve(
        config.num_market_maker_agents
    );

    for (
        int i = 0;
        i < config.num_market_maker_agents;
        ++i
    ) {
        int levels =
            config.market_maker_num_levels;

        int quantity =
            config.market_maker_order_quantity;

        int minimum_spread_ticks =
            config.market_maker_min_spread_ticks;

        if (config.heterogeneous_market_makers) {
            levels +=
                i
                * config.market_maker_num_levels_step;

            quantity +=
                i
                * config.market_maker_order_quantity_step;

            minimum_spread_ticks +=
                i
                * config.market_maker_min_spread_ticks_step;
        }

        levels = std::max(
            config.market_maker_min_num_levels,
            levels
        );

        quantity = std::max(
            config.market_maker_min_order_quantity,
            quantity
        );

        market_makers.emplace_back(
            config.tick_size,
            levels,
            quantity,
            config.market_maker_level_spacing_ticks,
            minimum_spread_ticks * config.tick_size
        );
    }

    std::vector<MomentumAgent> momentum_agents;

    momentum_agents.reserve(
        config.num_momentum_agents
    );

    for (
        int i = 0;
        i < config.num_momentum_agents;
        ++i
    ) {
        std::int64_t lookback_ns =
            config.momentum_base_lookback_ns;

        int quantity =
            config.momentum_order_quantity;

        double threshold_ticks =
            config.momentum_threshold_ticks;

        if (config.heterogeneous_momentum_agents) {
            lookback_ns +=
                static_cast<std::int64_t>(i)
                * config.momentum_lookback_step_ns;

            quantity +=
                i
                * config.momentum_order_quantity_step;

            threshold_ticks +=
                static_cast<double>(i)
                * config.momentum_threshold_step_ticks;
        }

        momentum_agents.emplace_back(
            lookback_ns,
            quantity,
            threshold_ticks,
            config.tick_size
        );
    }

    std::vector<LargeInstitutionalAgent>
        institutional_agents;

    int institutional_count =
        config.num_institutional_agents;

    if (config.use_mixed_institutional_sides) {
        institutional_count =
            config.num_buy_institutional_agents
            + config.num_sell_institutional_agents;
    }

    institutional_agents.reserve(
        institutional_count
    );

    int buy_agents_created = 0;
    int sell_agents_created = 0;

    for (int i = 0; i < institutional_count; ++i) {
        Side side;

        if (config.use_mixed_institutional_sides) {
            const bool preferred_sell_position =
                (i == 1 || i == 4);

            if (
                preferred_sell_position
                && sell_agents_created
                    < config.num_sell_institutional_agents
            ) {
                side = Side::Sell;
                ++sell_agents_created;
            } else if (
                buy_agents_created
                < config.num_buy_institutional_agents
            ) {
                side = Side::Buy;
                ++buy_agents_created;
            } else {
                side = Side::Sell;
                ++sell_agents_created;
            }
        } else {
            side =
                config.institutional_side >= 0
                ? Side::Buy
                : Side::Sell;
        }

        int parent_quantity;

        if (config.use_mixed_institutional_sides) {
            parent_quantity =
                side == Side::Buy
                ? config.buy_institutional_parent_quantity
                : config.sell_institutional_parent_quantity;
        } else {
            parent_quantity =
                config.institutional_parent_quantity;
        }

        int child_quantity =
            config.institutional_child_quantity;

        if (config.heterogeneous_institutional_agents) {
            parent_quantity +=
                i
                * config.institutional_parent_quantity_step;

            child_quantity +=
                i
                * config.institutional_child_quantity_step;
        }

        parent_quantity =
            std::max(1, parent_quantity);

        child_quantity =
            std::max(1, child_quantity);

        const std::int64_t start_time_ns =
            config.institutional_base_start_time_ns
            + static_cast<std::int64_t>(i)
                * config.institutional_start_spacing_ns;

        const std::int64_t end_time_ns =
            std::min<std::int64_t>(
                config.end_time_ns,
                start_time_ns
                    + config.institutional_duration_ns
            );

        institutional_agents.emplace_back(
            side,
            parent_quantity,
            child_quantity,
            start_time_ns,
            end_time_ns
        );
    }

    const std::int64_t
        autonomous_mm_refresh_interval_ns =
            AUTONOMOUS_MM_REFRESH_INTERVAL_NS;

    AutonomousMarketMakerAgent autonomous_mm(
        config.tick_size,
        AUTONOMOUS_MM_BASE_HALF_SPREAD_TICKS,
        AUTONOMOUS_MM_BASE_ORDER_QUANTITY,
        AUTONOMOUS_MM_MAX_INVENTORY,
        AUTONOMOUS_MM_INVENTORY_RISK_STRENGTH,
        AUTONOMOUS_MM_VOLATILITY_STRENGTH,
        AUTONOMOUS_MM_SIGNAL_STRENGTH,
        AUTONOMOUS_MM_AGGRESSIVE_INVENTORY_RATIO,
        autonomous_mm_refresh_interval_ns
    );

    std::ofstream timeseries_output;
    std::ofstream events_output;

    if (config.write_detailed_outputs) {
        timeseries_output.open(
            config.simulated_timeseries_file
        );

        timeseries_output
            << "time_ns,time_seconds,best_bid,"
            << "best_ask,spread,mid_price\n";

        events_output.open(
            config.simulated_events_file
        );

        events_output
            << "time_ns,time_seconds,event_type,"
            << "quantity,distance\n";
    }

    const std::vector<HawkesEvent> events =
        hawkes.simulate(
            config.start_time_ns,
            config.end_time_ns
        );

    std::int64_t next_sample_time_ns =
        config.start_time_ns;

    std::int64_t next_market_maker_time_ns =
        config.start_time_ns;

    std::int64_t next_momentum_time_ns =
        config.start_time_ns;

    std::int64_t next_institutional_time_ns =
        config.start_time_ns;

    std::int64_t next_autonomous_mm_time_ns =
        config.start_time_ns;

    std::vector<double> spreads;
    std::vector<double> mid_prices;

    for (const HawkesEvent& event : events) {
        while (
            next_sample_time_ns <= event.time_ns
            && next_sample_time_ns
                <= config.end_time_ns
        ) {
            record_snapshot(
                book,
                next_sample_time_ns,
                spreads,
                mid_prices,
                &timeseries_output,
                config.write_detailed_outputs
            );

            next_sample_time_ns +=
                config.sample_interval_ns;
        }

        while (
            next_market_maker_time_ns <= event.time_ns
            && next_market_maker_time_ns
                <= config.end_time_ns
        ) {
            for (auto& agent : market_makers) {
                agent.wake_up(
                    book,
                    next_order_id,
                    next_market_maker_time_ns
                );
            }

            next_market_maker_time_ns +=
                config.market_maker_interval_ns;
        }

        while (
            next_momentum_time_ns <= event.time_ns
            && next_momentum_time_ns
                <= config.end_time_ns
        ) {
            for (auto& agent : momentum_agents) {
                agent.record_mid_price(
                    book,
                    next_momentum_time_ns
                );

                agent.wake_up(
                    book,
                    next_momentum_time_ns
                );
            }

            next_momentum_time_ns +=
                config.momentum_interval_ns;
        }

        while (
            next_institutional_time_ns <= event.time_ns
            && next_institutional_time_ns
                <= config.end_time_ns
        ) {
            for (
                auto& agent :
                institutional_agents
            ) {
                agent.wake_up(
                    book,
                    next_institutional_time_ns
                );
            }

            next_institutional_time_ns +=
                config.institutional_interval_ns;
        }

        while (
            next_autonomous_mm_time_ns <= event.time_ns
            && next_autonomous_mm_time_ns
                <= config.end_time_ns
        ) {
            autonomous_mm.wake_up(
                book,
                next_order_id,
                next_autonomous_mm_time_ns
            );

            next_autonomous_mm_time_ns +=
                autonomous_mm_refresh_interval_ns;
        }

        if (event.type == EventType::LimitBuy) {
            if (!book.has_bid()) {
                continue;
            }

            const int quantity = std::max(
                1,
                limit_buy_quantity.sample(rng)
            );

            const int adjusted_distance =
                limit_buy_distance.sample(rng)
                + config.limit_distance_shift_ticks;

            long long price_value =
                static_cast<long long>(
                    book.best_bid()
                )
                - static_cast<long long>(
                    adjusted_distance
                )
                * static_cast<long long>(
                    config.tick_size
                );

            if (
                book.has_ask()
                && book.best_ask() > book.best_bid()
                && book.best_ask() - book.best_bid()
                    >= 3 * config.tick_size
                && uniform_01(rng)
                    < quote_improvement_probability
            ) {
                const long long improved_bid =
                    static_cast<long long>(
                        book.best_bid()
                    )
                    + static_cast<long long>(
                        config.tick_size
                    );

                const long long highest_passive_buy =
                    static_cast<long long>(
                        book.best_ask()
                    )
                    - static_cast<long long>(
                        config.tick_size
                    );

                if (
                    improved_bid
                    <= highest_passive_buy
                ) {
                    price_value = improved_bid;
                }
            }

            if (!valid_price(price_value)) {
                continue;
            }

            const int price =
                static_cast<int>(price_value);

            if (
                config.write_detailed_outputs
                && events_output.is_open()
            ) {
                events_output
                    << event.time_ns << ","
                    << static_cast<double>(
                           event.time_ns
                       )
                           / 1'000'000'000.0
                    << ",limit_buy,"
                    << quantity << ","
                    << adjusted_distance << "\n";
            }

            book.add_limit_order(Order{
                next_order_id++,
                event.time_ns,
                OrderType::Limit,
                Side::Buy,
                quantity,
                price
            });
        }

        else if (
            event.type == EventType::LimitSell
        ) {
            if (!book.has_ask()) {
                continue;
            }

            const int quantity = std::max(
                1,
                limit_sell_quantity.sample(rng)
            );

            const int adjusted_distance =
                limit_sell_distance.sample(rng)
                + config.limit_distance_shift_ticks;

            long long price_value =
                static_cast<long long>(
                    book.best_ask()
                )
                + static_cast<long long>(
                    adjusted_distance
                )
                * static_cast<long long>(
                    config.tick_size
                );

            if (
                book.has_bid()
                && book.best_ask() > book.best_bid()
                && book.best_ask() - book.best_bid()
                    >= 3 * config.tick_size
                && uniform_01(rng)
                    < quote_improvement_probability
            ) {
                const long long improved_ask =
                    static_cast<long long>(
                        book.best_ask()
                    )
                    - static_cast<long long>(
                        config.tick_size
                    );

                const long long lowest_passive_sell =
                    static_cast<long long>(
                        book.best_bid()
                    )
                    + static_cast<long long>(
                        config.tick_size
                    );

                if (
                    improved_ask
                    >= lowest_passive_sell
                ) {
                    price_value = improved_ask;
                }
            }

            if (!valid_price(price_value)) {
                continue;
            }

            const int price =
                static_cast<int>(price_value);

            if (
                config.write_detailed_outputs
                && events_output.is_open()
            ) {
                events_output
                    << event.time_ns << ","
                    << static_cast<double>(
                           event.time_ns
                       )
                           / 1'000'000'000.0
                    << ",limit_sell,"
                    << quantity << ","
                    << adjusted_distance << "\n";
            }

            book.add_limit_order(Order{
                next_order_id++,
                event.time_ns,
                OrderType::Limit,
                Side::Sell,
                quantity,
                price
            });
        }

        else if (
            event.type == EventType::MarketBuy
        ) {
            int quantity = static_cast<int>(
                std::round(
                    static_cast<double>(
                        market_buy_quantity.sample(rng)
                    )
                    * config.market_order_quantity_scale
                )
            );

            quantity = std::max(1, quantity);

            if (
                config.write_detailed_outputs
                && events_output.is_open()
            ) {
                events_output
                    << event.time_ns << ","
                    << static_cast<double>(
                           event.time_ns
                       )
                           / 1'000'000'000.0
                    << ",market_buy,"
                    << quantity << ",\n";
            }

            book.submit_market_order(
                Side::Buy,
                quantity
            );
        }

        else if (
            event.type == EventType::MarketSell
        ) {
            int quantity = static_cast<int>(
                std::round(
                    static_cast<double>(
                        market_sell_quantity.sample(rng)
                    )
                    * config.market_order_quantity_scale
                )
            );

            quantity = std::max(1, quantity);

            if (
                config.write_detailed_outputs
                && events_output.is_open()
            ) {
                events_output
                    << event.time_ns << ","
                    << static_cast<double>(
                           event.time_ns
                       )
                           / 1'000'000'000.0
                    << ",market_sell,"
                    << quantity << ",\n";
            }

            book.submit_market_order(
                Side::Sell,
                quantity
            );
        }

        else if (
            event.type == EventType::CancelBid
        ) {
            const int quantity = std::max(
                1,
                cancel_bid_quantity.sample(rng)
            );

            const int distance =
                cancel_bid_distance.sample(rng);

            if (
                config.write_detailed_outputs
                && events_output.is_open()
            ) {
                events_output
                    << event.time_ns << ","
                    << static_cast<double>(
                           event.time_ns
                       )
                           / 1'000'000'000.0
                    << ",cancel_bid,"
                    << quantity << ","
                    << distance << "\n";
            }

            book.cancel_at_distance(
                Side::Buy,
                distance,
                quantity,
                config.tick_size
            );
        }

        else if (
            event.type == EventType::CancelAsk
        ) {
            const int quantity = std::max(
                1,
                cancel_ask_quantity.sample(rng)
            );

            const int distance =
                cancel_ask_distance.sample(rng);

            if (
                config.write_detailed_outputs
                && events_output.is_open()
            ) {
                events_output
                    << event.time_ns << ","
                    << static_cast<double>(
                           event.time_ns
                       )
                           / 1'000'000'000.0
                    << ",cancel_ask,"
                    << quantity << ","
                    << distance << "\n";
            }

            book.cancel_at_distance(
                Side::Sell,
                distance,
                quantity,
                config.tick_size
            );
        }
    }

    while (
        next_autonomous_mm_time_ns
        <= config.end_time_ns
    ) {
        autonomous_mm.wake_up(
            book,
            next_order_id,
            next_autonomous_mm_time_ns
        );

        next_autonomous_mm_time_ns +=
            autonomous_mm_refresh_interval_ns;
    }

    autonomous_mm.process_fills(
        book,
        config.end_time_ns
    );

    const double autonomous_mm_final_pnl =
        autonomous_mm.final_pnl(book);

    std::cout
        << "AutonomousMM path "
        << config.calibration_id
        << " seed "
        << config.seed
        << " pnl "
        << autonomous_mm_final_pnl
        << " inventory "
        << autonomous_mm.inventory()
        << " cash "
        << autonomous_mm.cash()
        << " max_drawdown "
        << autonomous_mm.max_drawdown()
        << " max_abs_inventory "
        << autonomous_mm.max_abs_inventory()
        << " fills "
        << autonomous_mm.number_of_fills()
        << " traded_quantity "
        << autonomous_mm.total_traded_quantity()
        << " volatility_ticks "
        << autonomous_mm.current_volatility_ticks()
        << " signal "
        << autonomous_mm.current_signal()
        << " defensive_actions "
        << autonomous_mm.defensive_actions()
        << " aggressive_actions "
        << autonomous_mm.aggressive_actions()
        << "\n";

    while (
        next_sample_time_ns
        <= config.end_time_ns
    ) {
        record_snapshot(
            book,
            next_sample_time_ns,
            spreads,
            mid_prices,
            &timeseries_output,
            config.write_detailed_outputs
        );

        next_sample_time_ns +=
            config.sample_interval_ns;
    }

    if (timeseries_output.is_open()) {
        timeseries_output.close();
    }

    if (events_output.is_open()) {
        events_output.close();
    }

    std::vector<double> mid_changes;

    if (mid_prices.size() >= 2) {
        mid_changes.reserve(
            mid_prices.size() - 1
        );
    }

    for (
        std::size_t i = 1;
        i < mid_prices.size();
        ++i
    ) {
        mid_changes.push_back(
            mid_prices[i] - mid_prices[i - 1]
        );
    }

    result.mean_spread =
        mean_value(spreads);

    result.median_spread =
        percentile(spreads, 0.50);

    result.p90_spread =
        percentile(spreads, 0.90);

    result.p95_spread =
        percentile(spreads, 0.95);

    result.mean_mid_change =
        mean_value(mid_changes);

    result.std_mid_change =
        standard_deviation(mid_changes);

    if (!mid_changes.empty()) {
        std::size_t zero_count = 0;

        for (double change : mid_changes) {
            if (std::abs(change) < 1e-12) {
                ++zero_count;
            }
        }

        result.zero_mid_change_ratio =
            static_cast<double>(zero_count)
            / static_cast<double>(
                mid_changes.size()
            );

        result.mid_move_rate =
            1.0
            - result.zero_mid_change_ratio;
    }

    if (mid_prices.size() >= 2) {
        result.final_norm_mid =
            mid_prices.back()
            - mid_prices.front();
    }

    double weighted_loss = 0.0;
    double total_weight = 0.0;

    auto add_loss =
        [&weighted_loss, &total_weight](
            double weight,
            double simulated,
            double target
        ) {
            weighted_loss +=
                weight
                * squared_relative_loss(
                    simulated,
                    target
                );

            total_weight += weight;
        };

    add_loss(
        config.weight_mean_spread,
        result.mean_spread,
        config.target_mean_spread
    );

    add_loss(
        config.weight_median_spread,
        result.median_spread,
        config.target_median_spread
    );

    add_loss(
        config.weight_p90_spread,
        result.p90_spread,
        config.target_p90_spread
    );

    add_loss(
        config.weight_p95_spread,
        result.p95_spread,
        config.target_p95_spread
    );

    add_loss(
        config.weight_mean_mid_change,
        result.mean_mid_change,
        config.target_mean_mid_change
    );

    add_loss(
        config.weight_std_mid_change,
        result.std_mid_change,
        config.target_std_mid_change
    );

    add_loss(
        config.weight_zero_mid_change_ratio,
        result.zero_mid_change_ratio,
        config.target_zero_mid_change_ratio
    );

    add_loss(
        config.weight_mid_move_rate,
        result.mid_move_rate,
        config.target_mid_move_rate
    );

    add_loss(
        config.weight_final_norm_mid,
        result.final_norm_mid,
        config.target_final_norm_mid
    );

    result.total_loss =
        total_weight > 0.0
        ? weighted_loss / total_weight
        : weighted_loss;

    write_autonomous_mm_path_result(
        config,
        autonomous_mm,
        autonomous_mm_final_pnl,
        result
    );

    if (config.write_detailed_outputs) {
        std::ofstream summary_output(
            config.simulated_summary_file
        );

        summary_output
            << "metric,value\n";

        summary_output
            << "calibration_id,"
            << config.calibration_id
            << "\n";

        summary_output
            << "total_loss,"
            << result.total_loss
            << "\n";

        summary_output
            << "hawkes_beta,"
            << config.hawkes_beta
            << "\n";

        summary_output
            << "limit_branching,"
            << config.limit_branching
            << "\n";

        summary_output
            << "market_branching,"
            << config.market_branching
            << "\n";

        summary_output
            << "cancel_branching,"
            << config.cancel_branching
            << "\n";

        summary_output
            << "num_market_maker_agents,"
            << config.num_market_maker_agents
            << "\n";

        summary_output
            << "market_maker_order_quantity,"
            << config.market_maker_order_quantity
            << "\n";

        summary_output
            << "num_momentum_agents,"
            << config.num_momentum_agents
            << "\n";

        summary_output
            << "num_buy_institutional_agents,"
            << config.num_buy_institutional_agents
            << "\n";

        summary_output
            << "num_sell_institutional_agents,"
            << config.num_sell_institutional_agents
            << "\n";

        summary_output
            << "buy_institutional_parent_quantity,"
            << config.buy_institutional_parent_quantity
            << "\n";

        summary_output
            << "sell_institutional_parent_quantity,"
            << config.sell_institutional_parent_quantity
            << "\n";

        summary_output
            << "institutional_child_quantity,"
            << config.institutional_child_quantity
            << "\n";

        summary_output
            << "mean_spread,"
            << result.mean_spread
            << "\n";

        summary_output
            << "median_spread,"
            << result.median_spread
            << "\n";

        summary_output
            << "p90_spread,"
            << result.p90_spread
            << "\n";

        summary_output
            << "p95_spread,"
            << result.p95_spread
            << "\n";

        summary_output
            << "mean_mid_change,"
            << result.mean_mid_change
            << "\n";

        summary_output
            << "std_mid_change,"
            << result.std_mid_change
            << "\n";

        summary_output
            << "zero_mid_change_ratio,"
            << result.zero_mid_change_ratio
            << "\n";

        summary_output
            << "mid_move_rate,"
            << result.mid_move_rate
            << "\n";

        summary_output
            << "final_norm_mid,"
            << result.final_norm_mid
            << "\n";

        summary_output
            << "autonomous_mm_pnl,"
            << autonomous_mm_final_pnl
            << "\n";

        summary_output
            << "autonomous_mm_inventory,"
            << autonomous_mm.inventory()
            << "\n";

        summary_output
            << "autonomous_mm_cash,"
            << autonomous_mm.cash()
            << "\n";

        summary_output
            << "autonomous_mm_max_drawdown,"
            << autonomous_mm.max_drawdown()
            << "\n";

        summary_output
            << "autonomous_mm_max_abs_inventory,"
            << autonomous_mm.max_abs_inventory()
            << "\n";

        summary_output
            << "autonomous_mm_fills,"
            << autonomous_mm.number_of_fills()
            << "\n";

        summary_output
            << "autonomous_mm_traded_quantity,"
            << autonomous_mm.total_traded_quantity()
            << "\n";

        summary_output
            << "autonomous_mm_final_volatility_ticks,"
            << autonomous_mm.current_volatility_ticks()
            << "\n";

        summary_output
            << "autonomous_mm_final_signal,"
            << autonomous_mm.current_signal()
            << "\n";

        summary_output
            << "autonomous_mm_defensive_actions,"
            << autonomous_mm.defensive_actions()
            << "\n";

        summary_output
            << "autonomous_mm_aggressive_actions,"
            << autonomous_mm.aggressive_actions()
            << "\n";
    }

    return result;
}
