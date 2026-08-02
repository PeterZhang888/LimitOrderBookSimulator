#pragma once

#include "common/DistributedTypes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace dlob::detail {

// Fixed-memory fixed-clock moments used by the fragmented simulator.  A
// one-sided observation breaks return adjacency, exactly as it does in the
// ITCH extractor: the first later two-sided observation starts a new block.
struct AssetMomentAccumulator {
    std::uint64_t snapshots = 0;
    std::uint64_t invalid_snapshots = 0;
    double spread_sum = 0.0;
    double bid_depth_sum = 0.0;
    double ask_depth_sum = 0.0;
    double previous_mid = 0.0;
    std::uint64_t adjacent_pairs = 0;
    std::uint64_t mid_moves = 0;
    std::uint64_t return_count = 0;
    double return_sum = 0.0;
    double return_sum2 = 0.0;
    double return_sum4 = 0.0;
    double abs_return_sum = 0.0;
    double abs_return_sum2 = 0.0;
    double abs_pair_product_sum = 0.0;
    std::uint64_t abs_pair_count = 0;
    double previous_abs_return = 0.0;
    bool have_previous_abs_return = false;

    void observe(const MarketState& state, int tick_size) {
        if (state.best_bid_ticks <= 0
            || state.best_ask_ticks <= state.best_bid_ticks
            || tick_size <= 0) {
            ++invalid_snapshots;
            previous_mid = 0.0;
            return;
        }
        const double mid = state.mid_price_ticks > 0.0
            ? state.mid_price_ticks
            : 0.5 * static_cast<double>(
                state.best_bid_ticks + state.best_ask_ticks);
        if (!(mid > 0.0) || !std::isfinite(mid)) {
            ++invalid_snapshots;
            previous_mid = 0.0;
            return;
        }
        ++snapshots;
        spread_sum += static_cast<double>(
            state.best_ask_ticks - state.best_bid_ticks)
            / static_cast<double>(tick_size);
        bid_depth_sum += static_cast<double>(std::max(0, state.best_bid_depth));
        ask_depth_sum += static_cast<double>(std::max(0, state.best_ask_depth));
        if (previous_mid > 0.0) {
            ++adjacent_pairs;
            if (mid != previous_mid) ++mid_moves;
            const double value = std::log(mid / previous_mid);
            if (std::isfinite(value)) {
                ++return_count;
                return_sum += value;
                const double value2 = value * value;
                return_sum2 += value2;
                return_sum4 += value2 * value2;
                const double absolute = std::abs(value);
                abs_return_sum += absolute;
                abs_return_sum2 += absolute * absolute;
                // The empirical extractor concatenates the valid return
                // blocks before calculating this ACF, so only midpoint
                // adjacency resets at a one-sided clock observation.
                if (have_previous_abs_return) {
                    abs_pair_product_sum += absolute * previous_abs_return;
                    ++abs_pair_count;
                }
                previous_abs_return = absolute;
                have_previous_abs_return = true;
            }
        }
        previous_mid = mid;
    }

    [[nodiscard]] std::array<double, 7> finalize() const {
        std::array<double, 7> values{};
        if (snapshots > 0) {
            const double count = static_cast<double>(snapshots);
            values[0] = spread_sum / count;
            values[1] = bid_depth_sum / count;
            values[2] = ask_depth_sum / count;
        }
        if (adjacent_pairs > 0) {
            values[3] = static_cast<double>(mid_moves)
                / static_cast<double>(adjacent_pairs);
        }
        if (return_count > 0) {
            const double count = static_cast<double>(return_count);
            const double mean = return_sum / count;
            const double variance = std::max(
                0.0, return_sum2 / count - mean * mean);
            values[4] = variance;
            if (variance > std::numeric_limits<double>::epsilon()) {
                values[5] = (return_sum4 / count) / (variance * variance);
            }
        }
        if (abs_pair_count > 0 && return_count > 0) {
            const double count = static_cast<double>(return_count);
            const double mean = abs_return_sum / count;
            const double variance = std::max(
                0.0, abs_return_sum2 / count - mean * mean);
            if (variance > std::numeric_limits<double>::epsilon()) {
                const double cross = abs_pair_product_sum
                    / static_cast<double>(abs_pair_count);
                values[6] = (cross - mean * mean) / variance;
            }
        }
        return values;
    }
};

} // namespace dlob::detail
