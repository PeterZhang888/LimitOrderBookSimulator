#pragma once

#include "common/DistributedTypes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace dlob {

class FastRng {
public:
    explicit FastRng(std::uint64_t seed = 1) : state_(seed == 0 ? 1 : seed) {}

    std::uint64_t next_u64() {
        state_ += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }

    double uniform01() {
        return std::max(1e-12, static_cast<double>(next_u64() >> 11U) * (1.0 / 9007199254740992.0));
    }

    int uniform_int(int lower, int upper) {
        if (upper <= lower) return lower;
        const std::uint64_t span = static_cast<std::uint64_t>(upper - lower + 1);
        return lower + static_cast<int>(next_u64() % span);
    }

    double normal01() {
        constexpr double two_pi = 6.283185307179586476925286766559;
        return std::sqrt(-2.0 * std::log(uniform01())) * std::cos(two_pi * uniform01());
    }

    std::int64_t exponential_wait_ns(double rate_per_second) {
        if (rate_per_second <= 0.0) return std::numeric_limits<std::int64_t>::max();
        const double wait_seconds = -std::log(uniform01()) / rate_per_second;
        return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::llround(wait_seconds * 1e9)));
    }

private:
    std::uint64_t state_;
};

struct LatencyConfig {
    std::int64_t market_maker_base_ns = 5'000;
    std::int64_t market_maker_jitter_ns = 5'000;
    std::int64_t informed_base_ns = 20'000;
    std::int64_t informed_jitter_ns = 20'000;
    std::int64_t momentum_base_ns = 50'000;
    std::int64_t momentum_jitter_ns = 50'000;
    std::int64_t institutional_base_ns = 100'000;
    std::int64_t institutional_jitter_ns = 100'000;
};

class OrderMessageBuilder {
public:
    OrderMessageBuilder(int source_rank, LatencyConfig latency = {})
        : source_rank_(source_rank), latency_(latency) {}

    std::int64_t sample_latency_ns(AgentKind kind, FastRng& rng) const {
        std::int64_t base = 0;
        std::int64_t jitter = 0;
        switch (kind) {
            case AgentKind::MarketMaker:
                base = latency_.market_maker_base_ns;
                jitter = latency_.market_maker_jitter_ns;
                break;
            case AgentKind::Informed:
                base = latency_.informed_base_ns;
                jitter = latency_.informed_jitter_ns;
                break;
            case AgentKind::Momentum:
                base = latency_.momentum_base_ns;
                jitter = latency_.momentum_jitter_ns;
                break;
            case AgentKind::Institutional:
                base = latency_.institutional_base_ns;
                jitter = latency_.institutional_jitter_ns;
                break;
            case AgentKind::Background:
                break;
        }
        return base + (jitter > 0 ? static_cast<std::int64_t>(rng.uniform01() * static_cast<double>(jitter)) : 0);
    }

    void emit(std::vector<OrderMessage>& out,
              AgentKind kind,
              int owner_id,
              OrderAction action,
              Side side,
              int quantity,
              int price_ticks,
              int distance_ticks,
              std::int64_t decision_time_ns,
              FastRng& rng,
              std::int64_t forced_latency_ns = -1) {
        if (quantity <= 0 && action != OrderAction::CancelOwner) return;
        const std::int64_t latency = forced_latency_ns >= 0 ? forced_latency_ns : sample_latency_ns(kind, rng);
        OrderMessage message;
        message.generated_time_ns = decision_time_ns;
        message.arrival_time_ns = decision_time_ns + latency;
        message.sequence = (static_cast<std::uint64_t>(source_rank_) << 56U)
            | (next_sequence_++ & 0x00FFFFFFFFFFFFFFULL);
        message.tie_breaker = mix64(message.sequence ^ 0xd1b54a32d192ed03ULL);
        message.source_rank = source_rank_;
        message.owner_id = owner_id;
        message.agent_kind = kind;
        message.action = action;
        message.side = side;
        message.quantity = quantity;
        message.price_ticks = price_ticks;
        message.distance_ticks = distance_ticks;
        out.push_back(message);
    }

private:
    static std::uint64_t mix64(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    int source_rank_ = 0;
    LatencyConfig latency_{};
    std::uint64_t next_sequence_ = 1;
};

inline void update_cash_inventory(const AgentReport& report, int& inventory, double& cash_ticks) {
    if (report.kind != ReportKind::Fill || report.fill_quantity <= 0) return;
    const double notional = static_cast<double>(report.fill_quantity) * report.fill_price_ticks;
    if (report.side == Side::Buy) {
        inventory += report.fill_quantity;
        cash_ticks -= notional;
    } else {
        inventory -= report.fill_quantity;
        cash_ticks += notional;
    }
}

inline std::int64_t safe_add_time(std::int64_t time_ns, std::int64_t wait_ns) {
    if (wait_ns == std::numeric_limits<std::int64_t>::max()
        || time_ns > std::numeric_limits<std::int64_t>::max() - wait_ns) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return time_ns + wait_ns;
}

} // namespace dlob
