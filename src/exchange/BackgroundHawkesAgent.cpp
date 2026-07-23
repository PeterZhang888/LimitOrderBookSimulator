#include "exchange/BackgroundHawkesAgent.hpp"

#include "common/DataPaths.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dlob {

BackgroundHawkesConfig::BackgroundHawkesConfig() {
    for (auto& row : alpha) row.fill(0.0);
    for (std::size_t i = 0; i < alpha.size(); ++i) alpha[i][i] = 0.20;
    alpha[2][0] = 0.04;
    alpha[3][1] = 0.04;
    alpha[4][0] = 0.10;
    alpha[5][1] = 0.10;
}

BackgroundHawkesAgent::BackgroundHawkesAgent(const BackgroundHawkesConfig& config)
    : config_(config), rng_(config.seed + 911382323ULL) {
    if (!std::isfinite(config_.activity_scale) || config_.activity_scale <= 0.0) {
        throw std::invalid_argument("background Hawkes activity scale must be positive");
    }
    limit_buy_quantity_.set_fallback(25, 200);
    limit_sell_quantity_.set_fallback(25, 200);
    market_buy_quantity_.set_fallback(25, 700);
    market_sell_quantity_.set_fallback(25, 700);
    cancel_bid_quantity_.set_fallback(25, 500);
    cancel_ask_quantity_.set_fallback(25, 500);
    limit_buy_distance_.set_fallback(0, 5);
    limit_sell_distance_.set_fallback(0, 5);
    cancel_bid_distance_.set_fallback(0, 5);
    cancel_ask_distance_.set_fallback(0, 5);

    limit_buy_quantity_.load_from_csv(resolve_data_file(config_.limit_buy_quantity_file), "quantity");
    limit_sell_quantity_.load_from_csv(resolve_data_file(config_.limit_sell_quantity_file), "quantity");
    market_buy_quantity_.load_from_csv(resolve_data_file(config_.market_buy_quantity_file), "quantity");
    market_sell_quantity_.load_from_csv(resolve_data_file(config_.market_sell_quantity_file), "quantity");
    cancel_bid_quantity_.load_from_csv(resolve_data_file(config_.cancel_bid_quantity_file), "quantity");
    cancel_ask_quantity_.load_from_csv(resolve_data_file(config_.cancel_ask_quantity_file), "quantity");
    limit_buy_distance_.load_from_csv(resolve_data_file(config_.limit_buy_distance_file), "distance_ticks");
    limit_sell_distance_.load_from_csv(resolve_data_file(config_.limit_sell_distance_file), "distance_ticks");
    cancel_bid_distance_.load_from_csv(resolve_data_file(config_.cancel_bid_distance_file), "distance_ticks");
    cancel_ask_distance_.load_from_csv(resolve_data_file(config_.cancel_ask_distance_file), "distance_ticks");
}

std::vector<HawkesEvent> BackgroundHawkesAgent::simulate(std::int64_t start_time_ns,
                                                         std::int64_t end_time_ns) {
    if (end_time_ns <= start_time_ns) return {};

    std::vector<HawkesEvent> events;
    const double duration_seconds = static_cast<double>(end_time_ns - start_time_ns) / 1e9;
    events.reserve(static_cast<std::size_t>(std::max(0.0, duration_seconds * 100.0)));

    double time_seconds = static_cast<double>(start_time_ns) / 1e9;
    const double end_seconds = static_cast<double>(end_time_ns) / 1e9;
    std::array<double, 6> excitation{};

    while (time_seconds < end_seconds) {
        std::array<double, 6> upper{};
        double upper_sum = 0.0;
        for (std::size_t i = 0; i < upper.size(); ++i) {
            upper[i] = std::max(0.0, config_.activity_scale * config_.mu[i] + excitation[i]);
            upper_sum += upper[i];
        }
        if (upper_sum <= 1e-12) break;

        const double dt = -std::log(rng_.uniform01()) / upper_sum;
        time_seconds += dt;
        if (time_seconds >= end_seconds) break;

        const double decay = std::exp(-std::max(1e-6, config_.beta) * dt);
        for (double& value : excitation) value *= decay;

        std::array<double, 6> candidate{};
        double candidate_sum = 0.0;
        for (std::size_t i = 0; i < candidate.size(); ++i) {
            candidate[i] = std::max(0.0, config_.activity_scale * config_.mu[i] + excitation[i]);
            candidate_sum += candidate[i];
        }
        if (rng_.uniform01() * upper_sum > candidate_sum) continue;

        double draw = rng_.uniform01() * candidate_sum;
        std::size_t event_index = 0;
        for (; event_index + 1 < candidate.size(); ++event_index) {
            draw -= candidate[event_index];
            if (draw <= 0.0) break;
        }

        const auto time_ns = static_cast<std::int64_t>(std::llround(time_seconds * 1e9));
        events.push_back(HawkesEvent{time_ns, static_cast<HawkesEventType>(event_index)});
        for (std::size_t i = 0; i < excitation.size(); ++i) {
            excitation[i] += std::max(0.0, config_.alpha[i][event_index]);
        }
    }
    return events;
}

OrderMessage BackgroundHawkesAgent::make_order(const HawkesEvent& event,
                                                const MarketState& state,
                                                std::uint64_t sequence) {
    OrderMessage message;
    message.generated_time_ns = event.time_ns;
    message.arrival_time_ns = event.time_ns;
    message.sequence = sequence;
    message.tie_breaker = rng_.next_u64();
    message.source_rank = 0;
    message.owner_id = 0;
    message.agent_kind = AgentKind::Background;

    const int tick = std::max(1, config_.tick_size);
    switch (event.type) {
        case HawkesEventType::LimitBuy: {
            message.action = OrderAction::Limit;
            message.side = Side::Buy;
            message.quantity = limit_buy_quantity_.sample(rng_);
            message.distance_ticks = std::max(0, limit_buy_distance_.sample(rng_));
            int price = state.best_bid_ticks > 0
                ? state.best_bid_ticks - message.distance_ticks * tick
                : static_cast<int>(std::llround(state.mid_price_ticks)) - tick;
            if (state.best_bid_ticks > 0 && state.best_ask_ticks - state.best_bid_ticks >= 2 * tick
                && rng_.uniform01() < config_.quote_improvement_probability) {
                price = std::min(state.best_bid_ticks + tick, state.best_ask_ticks - tick);
            }
            message.price_ticks = price;
            break;
        }
        case HawkesEventType::LimitSell: {
            message.action = OrderAction::Limit;
            message.side = Side::Sell;
            message.quantity = limit_sell_quantity_.sample(rng_);
            message.distance_ticks = std::max(0, limit_sell_distance_.sample(rng_));
            int price = state.best_ask_ticks > 0
                ? state.best_ask_ticks + message.distance_ticks * tick
                : static_cast<int>(std::llround(state.mid_price_ticks)) + tick;
            if (state.best_bid_ticks > 0 && state.best_ask_ticks - state.best_bid_ticks >= 2 * tick
                && rng_.uniform01() < config_.quote_improvement_probability) {
                price = std::max(state.best_ask_ticks - tick, state.best_bid_ticks + tick);
            }
            message.price_ticks = price;
            break;
        }
        case HawkesEventType::MarketBuy:
            message.action = OrderAction::Market;
            message.side = Side::Buy;
            message.quantity = market_buy_quantity_.sample(rng_);
            break;
        case HawkesEventType::MarketSell:
            message.action = OrderAction::Market;
            message.side = Side::Sell;
            message.quantity = market_sell_quantity_.sample(rng_);
            break;
        case HawkesEventType::CancelBid:
            message.action = OrderAction::CancelAtDistance;
            message.side = Side::Buy;
            message.quantity = cancel_bid_quantity_.sample(rng_);
            message.distance_ticks = std::max(0, cancel_bid_distance_.sample(rng_));
            break;
        case HawkesEventType::CancelAsk:
            message.action = OrderAction::CancelAtDistance;
            message.side = Side::Sell;
            message.quantity = cancel_ask_quantity_.sample(rng_);
            message.distance_ticks = std::max(0, cancel_ask_distance_.sample(rng_));
            break;
    }
    return message;
}

} // namespace dlob
