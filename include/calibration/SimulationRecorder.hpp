// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "common/DistributedTypes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dlob::calibration {

enum class EmpiricalEventBucket : std::size_t {
    LimitBuy = 0,
    LimitSell,
    MarketBuy,
    MarketSell,
    CancelBid,
    CancelAsk,
    Count
};

inline constexpr std::size_t empirical_event_bucket_count =
    static_cast<std::size_t>(EmpiricalEventBucket::Count);

const char* event_bucket_name(EmpiricalEventBucket bucket);

struct MarketFeatureSummary {
    double mean_spread_ticks = 0.0;
    double mean_bid_depth = 0.0;
    double mean_ask_depth = 0.0;
    double mid_move_rate = 0.0;
    double return_variance = 0.0;
    double return_kurtosis = 0.0;
    double absolute_return_acf1 = 0.0;
    std::uint64_t snapshots = 0;
};

struct SimulationRecord {
    std::array<std::vector<int>, empirical_event_bucket_count> quantity_samples;
    std::array<std::uint64_t, empirical_event_bucket_count> event_counts{};
    std::uint64_t owner_cancel_messages = 0;
    MarketFeatureSummary market{};
    std::vector<MarketState> state_trace;
};

class SimulationRecorder {
public:
    explicit SimulationRecorder(std::uint64_t seed,
                                std::size_t reservoir_capacity = 8192,
                                int tick_size = 100);

    void observe_order(const OrderMessage& message);
    void observe_state(const MarketState& state);
    SimulationRecord finalize() const;

private:
    struct Reservoir {
        std::vector<int> values;
        std::uint64_t seen = 0;
        std::uint64_t state = 0;
        std::size_t capacity = 0;

        void add(int value);
        std::uint64_t next_u64();
    };

    static bool bucket_for(const OrderMessage& message, EmpiricalEventBucket& bucket);

    std::array<Reservoir, empirical_event_bucket_count> reservoirs_{};
    std::array<std::uint64_t, empirical_event_bucket_count> event_counts_{};
    std::uint64_t owner_cancel_messages_ = 0;
    int tick_size_ = 100;

    std::uint64_t snapshots_ = 0;
    double spread_sum_ = 0.0;
    double bid_depth_sum_ = 0.0;
    double ask_depth_sum_ = 0.0;
    double previous_mid_ = 0.0;
    std::uint64_t mid_moves_ = 0;

    std::uint64_t return_count_ = 0;
    double return_sum_ = 0.0;
    double return_sum2_ = 0.0;
    double return_sum4_ = 0.0;
    double abs_return_sum_ = 0.0;
    double abs_return_sum2_ = 0.0;
    double abs_pair_product_sum_ = 0.0;
    std::uint64_t abs_pair_count_ = 0;
    double previous_abs_return_ = 0.0;
    bool have_previous_abs_return_ = false;
    std::vector<MarketState> state_trace_;
};

} // namespace dlob::calibration
