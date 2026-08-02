#include "calibration/SimulationRecorder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dlob::calibration {

const char* event_bucket_name(EmpiricalEventBucket bucket) {
    switch (bucket) {
        case EmpiricalEventBucket::LimitBuy: return "limit_buy";
        case EmpiricalEventBucket::LimitSell: return "limit_sell";
        case EmpiricalEventBucket::MarketBuy: return "market_buy";
        case EmpiricalEventBucket::MarketSell: return "market_sell";
        case EmpiricalEventBucket::CancelBid: return "cancel_bid";
        case EmpiricalEventBucket::CancelAsk: return "cancel_ask";
        case EmpiricalEventBucket::Count: break;
    }
    return "unknown";
}

SimulationRecorder::SimulationRecorder(std::uint64_t seed,
                                       std::size_t reservoir_capacity,
                                       int tick_size)
    : tick_size_(std::max(1, tick_size)) {
    for (std::size_t i = 0; i < reservoirs_.size(); ++i) {
        reservoirs_[i].capacity = reservoir_capacity;
        reservoirs_[i].values.reserve(reservoir_capacity);
        reservoirs_[i].state = seed
            ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(i + 1));
    }
}

std::uint64_t SimulationRecorder::Reservoir::next_u64() {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

void SimulationRecorder::Reservoir::add(int value) {
    ++seen;
    if (capacity == 0) return;
    if (values.size() < capacity) {
        values.push_back(value);
        return;
    }
    const std::uint64_t index = next_u64() % seen;
    if (index < capacity) values[static_cast<std::size_t>(index)] = value;
}

bool SimulationRecorder::bucket_for(const OrderMessage& message,
                                    EmpiricalEventBucket& bucket) {
    if (message.action == OrderAction::Limit) {
        bucket = message.side == Side::Buy
            ? EmpiricalEventBucket::LimitBuy
            : EmpiricalEventBucket::LimitSell;
        return true;
    }
    if (message.action == OrderAction::Market) {
        bucket = message.side == Side::Buy
            ? EmpiricalEventBucket::MarketBuy
            : EmpiricalEventBucket::MarketSell;
        return true;
    }
    if (message.action == OrderAction::CancelAtDistance) {
        bucket = message.side == Side::Buy
            ? EmpiricalEventBucket::CancelBid
            : EmpiricalEventBucket::CancelAsk;
        return true;
    }
    return false;
}

void SimulationRecorder::observe_order(const OrderMessage& message) {
    if (message.action == OrderAction::CancelOwner) {
        ++owner_cancel_messages_;
        return;
    }
    EmpiricalEventBucket bucket{};
    if (!bucket_for(message, bucket)) return;
    const std::size_t index = static_cast<std::size_t>(bucket);
    ++event_counts_[index];
    if (message.quantity > 0) reservoirs_[index].add(message.quantity);
}

void SimulationRecorder::observe_state(const MarketState& state) {
    if (state.best_bid_ticks <= 0 || state.best_ask_ticks <= state.best_bid_ticks) return;
    const double mid = state.mid_price_ticks > 0.0
        ? state.mid_price_ticks
        : 0.5 * static_cast<double>(state.best_bid_ticks + state.best_ask_ticks);
    if (!(mid > 0.0) || !std::isfinite(mid)) return;

    state_trace_.push_back(state);
    ++snapshots_;
    spread_sum_ += static_cast<double>(state.best_ask_ticks - state.best_bid_ticks)
        / static_cast<double>(tick_size_);
    bid_depth_sum_ += static_cast<double>(std::max(0, state.best_bid_depth));
    ask_depth_sum_ += static_cast<double>(std::max(0, state.best_ask_depth));

    if (previous_mid_ > 0.0) {
        if (mid != previous_mid_) ++mid_moves_;
        const double value = std::log(mid / previous_mid_);
        if (std::isfinite(value)) {
            ++return_count_;
            return_sum_ += value;
            const double value2 = value * value;
            return_sum2_ += value2;
            return_sum4_ += value2 * value2;
            const double absolute = std::abs(value);
            abs_return_sum_ += absolute;
            abs_return_sum2_ += absolute * absolute;
            if (have_previous_abs_return_) {
                abs_pair_product_sum_ += absolute * previous_abs_return_;
                ++abs_pair_count_;
            }
            previous_abs_return_ = absolute;
            have_previous_abs_return_ = true;
        }
    }
    previous_mid_ = mid;
}

SimulationRecord SimulationRecorder::finalize() const {
    SimulationRecord record;
    record.event_counts = event_counts_;
    record.owner_cancel_messages = owner_cancel_messages_;
    record.state_trace = state_trace_;
    for (std::size_t i = 0; i < reservoirs_.size(); ++i) {
        record.quantity_samples[i] = reservoirs_[i].values;
    }

    record.market.snapshots = snapshots_;
    if (snapshots_ > 0) {
        const double n = static_cast<double>(snapshots_);
        record.market.mean_spread_ticks = spread_sum_ / n;
        record.market.mean_bid_depth = bid_depth_sum_ / n;
        record.market.mean_ask_depth = ask_depth_sum_ / n;
        if (snapshots_ > 1) {
            record.market.mid_move_rate = static_cast<double>(mid_moves_)
                / static_cast<double>(snapshots_ - 1);
        }
    }

    if (return_count_ > 0) {
        const double n = static_cast<double>(return_count_);
        const double mean = return_sum_ / n;
        const double second = return_sum2_ / n;
        const double variance = std::max(0.0, second - mean * mean);
        record.market.return_variance = variance;
        if (variance > std::numeric_limits<double>::epsilon()) {
            const double fourth_raw = return_sum4_ / n;
            record.market.return_kurtosis = fourth_raw / (variance * variance);
        }
    }

    if (abs_pair_count_ > 0 && return_count_ > 0) {
        const double n = static_cast<double>(return_count_);
        const double mean = abs_return_sum_ / n;
        const double variance = std::max(0.0, abs_return_sum2_ / n - mean * mean);
        if (variance > std::numeric_limits<double>::epsilon()) {
            const double cross = abs_pair_product_sum_ / static_cast<double>(abs_pair_count_);
            record.market.absolute_return_acf1 = (cross - mean * mean) / variance;
        }
    }
    return record;
}

} // namespace dlob::calibration
