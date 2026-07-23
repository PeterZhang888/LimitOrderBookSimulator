#include "agents/FundamentalValueAgent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace dlob {
namespace {

constexpr double nanoseconds_per_second = 1'000'000'000.0;

double uniform_open(std::uint64_t bits) {
    // The upper 53 bits produce a value strictly inside (0, 1), avoiding the
    // singularity of log(0) in the deterministic Box-Muller transform.
    constexpr double denominator = 9'007'199'254'740'992.0; // 2^53
    const std::uint64_t mantissa = bits >> 11U;
    return (static_cast<double>(mantissa) + 0.5) / denominator;
}

double deterministic_normal(std::uint64_t seed,
                            BookId book_id,
                            std::uint64_t sequence) {
    const StableEntityId entity = fundamental_value_entity(book_id);
    const double first = uniform_open(stable_sequence(entity ^ seed, sequence, 0));
    const double second = uniform_open(stable_sequence(entity ^ seed, sequence, 1));
    return std::sqrt(-2.0 * std::log(first))
        * std::cos(2.0 * std::numbers::pi * second);
}

std::int64_t checked_notional(std::int32_t price_ticks,
                              std::int32_t quantity) {
    if (price_ticks <= 0 || quantity <= 0
        || static_cast<std::int64_t>(price_ticks)
            > std::numeric_limits<std::int64_t>::max() / quantity) {
        throw std::overflow_error("value-agent trade notional overflows");
    }
    return static_cast<std::int64_t>(price_ticks) * quantity;
}

} // namespace

FundamentalValueAgent::FundamentalValueAgent(
    FundamentalValueConfig config,
    BookId book_id,
    double initial_fundamental_ticks,
    std::uint64_t model_seed)
    : config_(config),
      book_id_(book_id),
      owner_id_(fundamental_value_owner_id(book_id)),
      model_seed_(model_seed),
      fundamental_value_ticks_(initial_fundamental_ticks) {
    if (!std::isfinite(initial_fundamental_ticks)
        || initial_fundamental_ticks <= 0.0) {
        throw std::invalid_argument("value agent needs a positive initial fundamental");
    }
    if (!std::isfinite(config_.threshold_bps) || config_.threshold_bps <= 0.0
        || !std::isfinite(config_.response_step_bps)
        || config_.response_step_bps <= 0.0
        || config_.base_order_quantity <= 0
        || config_.max_order_quantity < config_.base_order_quantity
        || config_.max_abs_inventory <= 0
        || !std::isfinite(config_.fundamental_volatility_bps_sqrt_second)
        || config_.fundamental_volatility_bps_sqrt_second < 0.0
        || config_.decision_interval_ns <= 0
        || config_.order_latency_ns <= 0
        || config_.order_latency_ns >= config_.decision_interval_ns) {
        throw std::invalid_argument("invalid fundamental-value-agent configuration");
    }
}

void FundamentalValueAgent::advance_fundamental(
    std::uint64_t decision_sequence) {
    if (config_.fundamental_volatility_bps_sqrt_second == 0.0) return;
    const double interval_seconds = static_cast<double>(
        config_.decision_interval_ns) / nanoseconds_per_second;
    const double sigma = config_.fundamental_volatility_bps_sqrt_second
        * std::sqrt(interval_seconds) / 10'000.0;
    const double innovation = deterministic_normal(
        model_seed_, book_id_, decision_sequence);
    const double multiplier = std::exp(sigma * innovation - 0.5 * sigma * sigma);
    const double updated = fundamental_value_ticks_ * multiplier;
    if (!std::isfinite(updated) || updated < 1.0
        || updated > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("fundamental-value process left the valid price range");
    }
    fundamental_value_ticks_ = updated;
}

std::optional<OrderMessage> FundamentalValueAgent::make_order(
    const MarketState& state,
    std::int64_t decision_time_ns,
    std::uint64_t decision_sequence) {
    if (decision_sequence == 0 || decision_time_ns < 0
        || state.book_id != book_id_) {
        throw std::invalid_argument("invalid value-agent decision input");
    }
    advance_fundamental(decision_sequence);
    if (state.best_bid_ticks <= 0 || state.best_ask_ticks <= state.best_bid_ticks
        || !std::isfinite(state.mid_price_ticks) || state.mid_price_ticks <= 0.0) {
        last_mispricing_bps_ = 0.0;
        return std::nullopt;
    }

    last_mispricing_bps_ = 10'000.0
        * (state.mid_price_ticks / fundamental_value_ticks_ - 1.0);
    if (std::abs(last_mispricing_bps_) <= config_.threshold_bps) {
        return std::nullopt;
    }

    const Side side = last_mispricing_bps_ < 0.0 ? Side::Buy : Side::Sell;
    const double excess = std::abs(last_mispricing_bps_) - config_.threshold_bps;
    const auto multiplier = static_cast<std::int64_t>(
        1.0 + std::floor(excess / config_.response_step_bps));
    const std::int64_t requested = std::min<std::int64_t>(
        static_cast<std::int64_t>(config_.base_order_quantity) * multiplier,
        config_.max_order_quantity);
    const std::int64_t capacity = side == Side::Buy
        ? config_.max_abs_inventory - inventory_
        : config_.max_abs_inventory + inventory_;
    const std::int64_t bounded = std::min(requested, capacity);
    if (bounded <= 0) return std::nullopt;

    if (decision_time_ns > std::numeric_limits<std::int64_t>::max()
            - config_.order_latency_ns) {
        throw std::overflow_error("value-agent order latency overflows event time");
    }
    OrderMessage order;
    order.generated_time_ns = decision_time_ns;
    order.arrival_time_ns = decision_time_ns + config_.order_latency_ns;
    order.sequence = stable_sequence(
        fundamental_value_entity(book_id_), decision_sequence, 0);
    order.tie_breaker = stable_sequence(
        fundamental_value_entity(book_id_), decision_sequence, 1);
    order.source_rank = 0;
    order.owner_id = owner_id_;
    order.agent_kind = AgentKind::Value;
    order.action = OrderAction::Market;
    order.side = side;
    order.quantity = static_cast<std::int32_t>(bounded);
    order.book_id = book_id_;
    return order;
}

void FundamentalValueAgent::on_trade(const TradeExecution& trade) {
    if (trade.book_id != book_id_) return;
    const std::int64_t notional = checked_notional(
        trade.price_ticks, trade.quantity);
    if (trade.buyer_owner_id == owner_id_) {
        inventory_ += trade.quantity;
        cash_ticks_ -= notional;
    }
    if (trade.seller_owner_id == owner_id_) {
        inventory_ -= trade.quantity;
        cash_ticks_ += notional;
    }
    if (std::abs(inventory_) > config_.max_abs_inventory) {
        throw std::logic_error("value-agent inventory limit was violated");
    }
}

} // namespace dlob
