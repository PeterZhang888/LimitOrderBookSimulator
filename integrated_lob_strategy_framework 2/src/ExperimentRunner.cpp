#include "ExperimentRunner.hpp"

#include "BackgroundHawkesAgent.hpp"
#include "EmpiricalDistribution.hpp"
#include "LargeInstitutionalAgent.hpp"
#include "LimitOrderBook.hpp"
#include "MomentumAgent.hpp"
#include "Order.hpp"
#include "PassiveMarketMakerAgent.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int STRATEGY_OWNER_BASE = 9000;
constexpr std::int64_t STRATEGY_INTERVAL_NS = 1000LL * 1000LL * 1000LL;
constexpr int STRATEGY_BASE_ORDER_SIZE = 100;
constexpr int STRATEGY_BASE_SPREAD_TICKS = 4;

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
    for (int i = 0; i < 10; ++i) book.add_limit_order(Order{id++, 0, OrderType::Limit, Side::Buy, scaled_quantity(bid_qty[i], s), bid_prices[i], 0});
    for (int i = 0; i < 10; ++i) book.add_limit_order(Order{id++, 0, OrderType::Limit, Side::Sell, scaled_quantity(ask_qty[i], s), ask_prices[i], 0});
}

int sample_scaled(EmpiricalDistribution& distribution, std::mt19937_64& rng, double scale) {
    return std::max(1, static_cast<int>(std::llround(static_cast<double>(distribution.sample(rng)) * scale)));
}

int distance_sample(EmpiricalDistribution& distribution, std::mt19937_64& rng, int shift) {
    return std::max(0, distribution.sample(rng) + shift);
}

class HawkesIntensityTracker {
public:
    explicit HawkesIntensityTracker(const SimulationConfig& config)
        : mu_(config.hawkes_mu), alpha_(config.hawkes_alpha), beta_(std::max(1e-9, config.hawkes_beta)) {
        excitation_.fill(0.0);
        last_time_ns_ = config.start_time_ns;
        average_intensity_ = std::accumulate(mu_.begin(), mu_.end(), 0.0);
    }

    void advance_to(std::int64_t time_ns) {
        if (time_ns <= last_time_ns_) return;
        const double dt = static_cast<double>(time_ns - last_time_ns_) / 1e9;
        const double decay = std::exp(-beta_ * dt);
        for (double& x : excitation_) x *= decay;
        last_time_ns_ = time_ns;
    }

    void observe_event(EventType type) {
        const int j = static_cast<int>(type);
        if (j < 0 || j >= NUM_EVENT_TYPES) return;
        for (int i = 0; i < NUM_EVENT_TYPES; ++i) excitation_[i] += std::max(0.0, alpha_[i][j]);
        const double total = total_intensity();
        average_intensity_ = 0.999 * average_intensity_ + 0.001 * total;
    }

    double total_intensity() const {
        double s = 0.0;
        for (int i = 0; i < NUM_EVENT_TYPES; ++i) s += std::max(0.0, mu_[i] + excitation_[i]);
        return s;
    }

    double average_intensity() const { return std::max(1e-9, average_intensity_); }

private:
    std::array<double, 6> mu_{};
    std::array<std::array<double, 6>, 6> alpha_{};
    std::array<double, 6> excitation_{};
    double beta_ = 8.0;
    double average_intensity_ = 1.0;
    std::int64_t last_time_ns_ = 0;
};

MarketState make_market_state(const LimitOrderBook& book, const SimulationConfig& config, const HawkesIntensityTracker& tracker, std::int64_t time_ns) {
    MarketState state;
    state.timestamp_ns = time_ns;
    state.tick_size = config.tick_size;
    state.best_bid_ticks = book.has_bid() ? book.best_bid() : 0;
    state.best_ask_ticks = book.has_ask() ? book.best_ask() : 0;
    state.mid_price_ticks = book.mid_price();
    state.spread_ticks = book.spread_ticks();
    state.bid_queue_depth = book.quantity_at_best_bid();
    state.ask_queue_depth = book.quantity_at_best_ask();
    state.hawkes_total_intensity = tracker.total_intensity();
    state.hawkes_average_intensity = tracker.average_intensity();
    return state;
}

void record_pnl_sample(
    const ExperimentalMarketMaker& strategy,
    const LimitOrderBook& book,
    std::vector<double>& pnl_path,
    std::vector<double>& abs_inventory_path,
    std::vector<double>& spread_path
) {
    if (!book.has_bid() || !book.has_ask()) return;
    pnl_path.push_back(strategy.mark_to_market(book.mid_price()));
    abs_inventory_path.push_back(std::abs(static_cast<double>(strategy.inventory())));
    spread_path.push_back(static_cast<double>(book.spread_ticks()));
}

double intraday_sharpe_from_pnl_path(const std::vector<double>& pnl_path) {
    if (pnl_path.size() < 3) return 0.0;
    std::vector<double> increments;
    increments.reserve(pnl_path.size() - 1);
    for (std::size_t i = 1; i < pnl_path.size(); ++i) increments.push_back(pnl_path[i] - pnl_path[i - 1]);
    const double m = mean(increments);
    const double s = sample_stddev(increments);
    return s > 1e-12 ? m / s * std::sqrt(static_cast<double>(increments.size())) : 0.0;
}

void maybe_log_quote(
    std::ofstream& log,
    bool enabled,
    const MarketState& state,
    const StrategyQuote& quote,
    const ExperimentalMarketMaker& strategy
) {
    if (!enabled || !log.is_open()) return;
    const std::int64_t one_minute_ns = 60LL * 1000LL * 1000LL * 1000LL;
    if (state.timestamp_ns % one_minute_ns != 0) return;
    log << state.timestamp_ns << ','
        << state.best_bid_ticks << ',' << state.best_ask_ticks << ',' << state.mid_price_ticks << ','
        << state.bid_queue_depth << ',' << state.ask_queue_depth << ','
        << state.hawkes_total_intensity << ',' << state.hawkes_average_intensity << ','
        << strategy.inventory() << ',' << strategy.cash_ticks() << ','
        << quote.bid_price_ticks << ',' << quote.ask_price_ticks << ','
        << quote.bid_volume << ',' << quote.ask_volume << '\n';
}
}

SimulationConfig calibrated_background_config(std::uint64_t seed, int day_id, std::int64_t duration_seconds) {
    SimulationConfig c;
    c.seed = seed + static_cast<std::uint64_t>(day_id) * 1000003ULL;
    c.calibration_id = 4080;
    c.start_time_ns = 0;
    c.end_time_ns = duration_seconds * 1000LL * 1000LL * 1000LL;
    c.sample_interval_ns = 1LL * 1000LL * 1000LL * 1000LL;

    // Calibrated full-day background configuration selected from the empirical-moment loss search.
    const double mu_scale = 1.09426745516;
    const double market_mu_scale = 1.0318492531;
    const double cancel_mu_scale = 2.39917115235;
    const double beta = 22.0837737116;
    const double self_branch = 0.125945108713;
    const double cross_branch = 0.105285697369;
    const double directional_branch = 0.0138432140667;

    c.hawkes_beta = beta;
    c.hawkes_mu = {{12.0 * mu_scale, 12.0 * mu_scale, 2.2 * market_mu_scale, 2.2 * market_mu_scale, 18.0 * cancel_mu_scale, 18.0 * cancel_mu_scale}};
    for (int i = 0; i < NUM_EVENT_TYPES; ++i) for (int j = 0; j < NUM_EVENT_TYPES; ++j) c.hawkes_alpha[i][j] = 0.0;
    for (int i = 0; i < NUM_EVENT_TYPES; ++i) c.hawkes_alpha[i][i] = beta * self_branch;
    c.hawkes_alpha[0][1] = beta * cross_branch; c.hawkes_alpha[1][0] = beta * cross_branch;
    c.hawkes_alpha[2][3] = beta * cross_branch; c.hawkes_alpha[3][2] = beta * cross_branch;
    c.hawkes_alpha[4][5] = beta * cross_branch; c.hawkes_alpha[5][4] = beta * cross_branch;
    c.hawkes_alpha[2][0] = beta * directional_branch;
    c.hawkes_alpha[5][2] = beta * directional_branch;
    c.hawkes_alpha[3][1] = beta * directional_branch;
    c.hawkes_alpha[4][3] = beta * directional_branch;

    c.initial_depth_scale = 0.852470801374;
    c.quote_improvement_probability = 0.145553227046;
    c.market_order_quantity_scale = 2.51008050802;
    c.cancel_quantity_scale = 1.23571269155;
    c.limit_distance_shift_ticks = 2;
    c.num_market_maker_agents = 3;
    c.market_maker_order_quantity = 111;
    c.market_maker_min_spread_ticks = 6;
    c.market_maker_quote_skip_probability = 0.249354202848;
    c.market_maker_interval_ns = 1484LL * 1000LL * 1000LL;
    c.num_momentum_agents = 5;
    c.momentum_order_quantity = 616;
    c.momentum_threshold_ticks = 0.492900318362;
    c.num_buy_institutional_agents = 2;
    c.num_sell_institutional_agents = 1;
    c.institutional_child_quantity = 953;
    c.institutional_participation_cap = 0.0885782264727;
    c.institutional_interval_ns = 11LL * 1000LL * 1000LL * 1000LL;
    return c;
}

double mean(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

double sample_stddev(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    const double m = mean(x);
    double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return std::sqrt(s / static_cast<double>(x.size() - 1));
}

double annualized_sharpe_from_daily_pnl(const std::vector<double>& daily_pnl) {
    const double m = mean(daily_pnl);
    const double s = sample_stddev(daily_pnl);
    return s > 1e-12 ? std::sqrt(252.0) * m / s : 0.0;
}

double max_drawdown_from_cumulative_pnl(const std::vector<double>& daily_pnl) {
    double cumulative = 0.0;
    double peak = 0.0;
    double max_dd = 0.0;
    for (double x : daily_pnl) {
        cumulative += x;
        peak = std::max(peak, cumulative);
        max_dd = std::max(max_dd, peak - cumulative);
    }
    return max_dd;
}

StrategyDayResult run_strategy_day(
    int strategy_id,
    double hyperparameter,
    int day_id,
    int replica_id,
    std::uint64_t base_seed,
    bool log_quotes,
    const std::string& quote_log_path
) {
    const std::uint64_t seed = base_seed + static_cast<std::uint64_t>(replica_id) + static_cast<std::uint64_t>(day_id);
    SimulationConfig config = calibrated_background_config(seed, day_id, 23400);

    LimitOrderBook book(config.tick_size);
    create_initial_book(book, config);
    std::uint64_t next_order_id = 1000000ULL + static_cast<std::uint64_t>(day_id) * 100000ULL + static_cast<std::uint64_t>(replica_id) * 1000ULL;

    BackgroundHawkesAgent background_agent(config);
    const std::vector<HawkesEvent> events = background_agent.simulate(config.start_time_ns, config.end_time_ns);
    HawkesIntensityTracker intensity_tracker(config);

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
    for (int i = 0; i < config.num_market_maker_agents; ++i) {
        market_makers.emplace_back(config.tick_size, config.market_maker_num_levels, config.market_maker_order_quantity,
            config.market_maker_level_spacing_ticks, config.market_maker_min_spread_ticks * config.tick_size,
            config.market_maker_quote_skip_probability, config.market_maker_recovery_quote_skip_probability,
            config.seed + static_cast<std::uint64_t>(i) * 1000ULL);
    }

    std::vector<MomentumAgent> momentum_agents;
    for (int i = 0; i < config.num_momentum_agents; ++i) {
        momentum_agents.emplace_back(config.momentum_base_lookback_ns, config.momentum_order_quantity,
            config.momentum_threshold_ticks, config.tick_size, config.momentum_order_flow_imbalance_threshold,
            config.momentum_depth_imbalance_threshold, config.momentum_strong_depth_imbalance_threshold);
    }

    std::vector<LargeInstitutionalAgent> institutional_agents;
    const int institutional_count = config.num_buy_institutional_agents + config.num_sell_institutional_agents;
    int buy_created = 0, sell_created = 0;
    for (int i = 0; i < institutional_count; ++i) {
        Side side;
        if (buy_created < config.num_buy_institutional_agents && (i % 2 == 0 || sell_created >= config.num_sell_institutional_agents)) { side = Side::Buy; ++buy_created; }
        else { side = Side::Sell; ++sell_created; }
        const int parent = side == Side::Buy ? config.buy_institutional_parent_quantity : config.sell_institutional_parent_quantity;
        const std::int64_t start = config.institutional_base_start_time_ns + static_cast<std::int64_t>(i) * config.institutional_start_spacing_ns;
        const std::int64_t end = std::min(config.end_time_ns, start + config.institutional_duration_ns);
        institutional_agents.emplace_back(side, parent, config.institutional_child_quantity, config.institutional_participation_cap, start, end, config.seed + static_cast<std::uint64_t>(i) * 99991ULL);
    }

    const int owner_id = STRATEGY_OWNER_BASE + strategy_id;
    auto strategy = make_strategy(strategy_id, hyperparameter, owner_id, STRATEGY_BASE_ORDER_SIZE, STRATEGY_BASE_SPREAD_TICKS);

    std::ofstream quote_log;
    if (log_quotes) {
        std::filesystem::create_directories("logs");
        quote_log.open(quote_log_path);
        quote_log << "timestamp_ns,best_bid,best_ask,mid,bid_depth,ask_depth,hawkes_intensity,avg_hawkes_intensity,inventory,cash_ticks,bid_quote,ask_quote,bid_volume,ask_volume\n";
    }

    std::int64_t next_strategy_time_ns = config.start_time_ns;
    std::int64_t next_market_maker_time_ns = config.start_time_ns;
    std::int64_t next_momentum_time_ns = config.start_time_ns;
    std::int64_t next_institutional_time_ns = config.start_time_ns;

    std::vector<double> pnl_path;
    std::vector<double> abs_inventory_path;
    std::vector<double> spread_path;

    auto wake_strategy_until = [&](std::int64_t until_ns) {
        while (next_strategy_time_ns <= until_ns && next_strategy_time_ns <= config.end_time_ns) {
            intensity_tracker.advance_to(next_strategy_time_ns);
            strategy->process_execution_reports(book.get_and_clear_execution_reports(owner_id));
            if (book.has_bid() && book.has_ask()) {
                const MarketState state = make_market_state(book, config, intensity_tracker, next_strategy_time_ns);
                StrategyQuote quote = strategy->get_quote(state);
                book.cancel_orders_by_owner(owner_id);
                if (quote.should_quote) {
                    if (quote.bid_volume > 0) book.add_limit_order(Order{next_order_id++, next_strategy_time_ns, OrderType::Limit, Side::Buy, quote.bid_volume, quote.bid_price_ticks, owner_id});
                    if (quote.ask_volume > 0) book.add_limit_order(Order{next_order_id++, next_strategy_time_ns, OrderType::Limit, Side::Sell, quote.ask_volume, quote.ask_price_ticks, owner_id});
                }
                maybe_log_quote(quote_log, log_quotes, state, quote, *strategy);
                record_pnl_sample(*strategy, book, pnl_path, abs_inventory_path, spread_path);
            }
            next_strategy_time_ns += STRATEGY_INTERVAL_NS;
        }
    };

    for (const HawkesEvent& event : events) {
        wake_strategy_until(event.time_ns);
        intensity_tracker.advance_to(event.time_ns);

        while (next_market_maker_time_ns <= event.time_ns && next_market_maker_time_ns <= config.end_time_ns) {
            for (auto& agent : market_makers) agent.wake_up(book, next_order_id, next_market_maker_time_ns);
            next_market_maker_time_ns += config.market_maker_interval_ns;
        }
        while (next_momentum_time_ns <= event.time_ns && next_momentum_time_ns <= config.end_time_ns) {
            for (auto& agent : momentum_agents) { agent.record_mid_price(book, next_momentum_time_ns); agent.wake_up(book, next_momentum_time_ns); }
            next_momentum_time_ns += config.momentum_interval_ns;
        }
        while (next_institutional_time_ns <= event.time_ns && next_institutional_time_ns <= config.end_time_ns) {
            for (auto& agent : institutional_agents) agent.wake_up(book, next_institutional_time_ns);
            next_institutional_time_ns += config.institutional_interval_ns;
        }

        switch (event.type) {
            case EventType::LimitBuy: {
                if (!book.has_bid()) break;
                const int quantity = sample_scaled(limit_buy_quantity, rng, 1.0);
                const int distance = distance_sample(limit_buy_distance, rng, config.limit_distance_shift_ticks);
                int price = book.best_bid() - distance * config.tick_size;
                if (book.has_ask() && book.best_ask() - book.best_bid() >= 3 * config.tick_size && uniform_01(rng) < config.quote_improvement_probability) price = std::min(book.best_bid() + config.tick_size, book.best_ask() - config.tick_size);
                if (price > 0 && (!book.has_ask() || price < book.best_ask())) book.add_limit_order(Order{next_order_id++, event.time_ns, OrderType::Limit, Side::Buy, quantity, price, 0});
                break;
            }
            case EventType::LimitSell: {
                if (!book.has_ask()) break;
                const int quantity = sample_scaled(limit_sell_quantity, rng, 1.0);
                const int distance = distance_sample(limit_sell_distance, rng, config.limit_distance_shift_ticks);
                int price = book.best_ask() + distance * config.tick_size;
                if (book.has_bid() && book.best_ask() - book.best_bid() >= 3 * config.tick_size && uniform_01(rng) < config.quote_improvement_probability) price = std::max(book.best_ask() - config.tick_size, book.best_bid() + config.tick_size);
                if (price > 0 && (!book.has_bid() || price > book.best_bid())) book.add_limit_order(Order{next_order_id++, event.time_ns, OrderType::Limit, Side::Sell, quantity, price, 0});
                break;
            }
            case EventType::MarketBuy: {
                const int quantity = sample_scaled(market_buy_quantity, rng, config.market_order_quantity_scale);
                book.submit_market_order(Side::Buy, quantity, event.time_ns);
                break;
            }
            case EventType::MarketSell: {
                const int quantity = sample_scaled(market_sell_quantity, rng, config.market_order_quantity_scale);
                book.submit_market_order(Side::Sell, quantity, event.time_ns);
                break;
            }
            case EventType::CancelBid: {
                const int quantity = sample_scaled(cancel_bid_quantity, rng, config.cancel_quantity_scale);
                const int distance = distance_sample(cancel_bid_distance, rng, 0);
                book.cancel_at_distance(Side::Buy, distance, quantity, config.tick_size);
                break;
            }
            case EventType::CancelAsk: {
                const int quantity = sample_scaled(cancel_ask_quantity, rng, config.cancel_quantity_scale);
                const int distance = distance_sample(cancel_ask_distance, rng, 0);
                book.cancel_at_distance(Side::Sell, distance, quantity, config.tick_size);
                break;
            }
        }
        intensity_tracker.observe_event(event.type);
        strategy->process_execution_reports(book.get_and_clear_execution_reports(owner_id));
    }

    wake_strategy_until(config.end_time_ns);
    strategy->process_execution_reports(book.get_and_clear_execution_reports(owner_id));
    book.cancel_orders_by_owner(owner_id);

    const double final_mid = book.has_bid() && book.has_ask() ? book.mid_price() : 0.0;
    const double final_pnl = strategy->mark_to_market(final_mid);
    if (pnl_path.empty()) pnl_path.push_back(final_pnl);

    StrategyDayResult result;
    result.strategy_id = strategy_id;
    result.replica_id = replica_id;
    result.day_id = day_id;
    result.hyperparameter = hyperparameter;
    result.final_pnl = final_pnl;
    result.final_inventory = strategy->inventory();
    result.avg_spread = mean(spread_path);
    result.total_traded_volume = strategy->total_traded_volume();
    result.sharpe_ratio = intraday_sharpe_from_pnl_path(pnl_path);
    result.max_drawdown = max_drawdown_from_cumulative_pnl(std::vector<double>{final_pnl});
    result.mean_abs_position = mean(abs_inventory_path);
    return result;
}

ReplicaSummary run_test_replica(
    int strategy_id,
    double hyperparameter,
    int replica_id,
    std::uint64_t base_seed,
    int first_day,
    int last_day,
    bool enable_rank1_quote_log
) {
    std::vector<double> daily_pnl;
    std::vector<double> daily_spread;
    std::vector<double> daily_abs_position;
    ReplicaSummary summary;
    summary.replica_id = replica_id;
    summary.strategy_id = strategy_id;
    summary.hyperparameter = hyperparameter;

    for (int day = first_day; day <= last_day; ++day) {
        const bool log_quotes = enable_rank1_quote_log && replica_id == 1 && day == first_day;
        const std::string log_path = "logs/quote_log_" + std::to_string(strategy_id) + "_" + std::to_string(replica_id) + ".csv";
        StrategyDayResult r = run_strategy_day(strategy_id, hyperparameter, day, replica_id, base_seed, log_quotes, log_path);
        daily_pnl.push_back(r.final_pnl);
        daily_spread.push_back(r.avg_spread);
        daily_abs_position.push_back(r.mean_abs_position);
        summary.final_inventory = r.final_inventory;
        summary.total_traded_volume += r.total_traded_volume;
    }

    summary.final_pnl = std::accumulate(daily_pnl.begin(), daily_pnl.end(), 0.0);
    summary.avg_spread = mean(daily_spread);
    summary.sharpe_ratio = annualized_sharpe_from_daily_pnl(daily_pnl);
    summary.max_drawdown = max_drawdown_from_cumulative_pnl(daily_pnl);
    summary.mean_abs_position = mean(daily_abs_position);
    return summary;
}
