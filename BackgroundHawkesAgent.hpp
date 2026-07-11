#pragma once

#include "Order.hpp"
#include "SimulationConfig.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

class BackgroundHawkesAgent {
public:
    explicit BackgroundHawkesAgent(const SimulationConfig& config);
    std::vector<HawkesEvent> simulate(std::int64_t start_time_ns, std::int64_t end_time_ns);

private:
    std::array<double, 6> mu_{};
    std::array<std::array<double, 6>, 6> alpha_{};
    double beta_ = 8.0;
    std::mt19937_64 rng_;
};
