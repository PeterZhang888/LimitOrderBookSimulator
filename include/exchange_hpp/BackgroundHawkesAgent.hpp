#pragma once

#include "common/DistributedTypes.hpp"
#include "common/EmpiricalDistribution.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace dlob {

struct BackgroundHawkesConfig {
    std::array<double, 6> mu{18.0, 18.0, 3.5, 3.5, 28.0, 28.0};
    std::array<std::array<double, 6>, 6> alpha{};
    double beta = 10.0;
    int tick_size = 100;
    double quote_improvement_probability = 0.05;
    std::uint64_t seed = 12345;

    std::string limit_buy_quantity_file = "data/limit_buy_quantity_distribution.txt";
    std::string limit_sell_quantity_file = "data/limit_sell_quantity_distribution.txt";
    std::string market_buy_quantity_file = "data/market_buy_quantity_distribution.txt";
    std::string market_sell_quantity_file = "data/market_sell_quantity_distribution.txt";
    std::string cancel_bid_quantity_file = "data/cancel_bid_quantity_distribution.txt";
    std::string cancel_ask_quantity_file = "data/cancel_ask_quantity_distribution.txt";
    std::string limit_buy_distance_file = "data/limit_buy_distance_distribution.txt";
    std::string limit_sell_distance_file = "data/limit_sell_distance_distribution.txt";
    std::string cancel_bid_distance_file = "data/cancel_bid_distance_distribution.txt";
    std::string cancel_ask_distance_file = "data/cancel_ask_distance_distribution.txt";

    BackgroundHawkesConfig();
};

class BackgroundHawkesAgent {
public:
    explicit BackgroundHawkesAgent(const BackgroundHawkesConfig& config);

    std::vector<HawkesEvent> simulate(std::int64_t start_time_ns, std::int64_t end_time_ns);
    OrderMessage make_order(const HawkesEvent& event,
                            const MarketState& state,
                            std::uint64_t sequence);

private:
    BackgroundHawkesConfig config_;
    FastRng rng_;
    EmpiricalDistribution limit_buy_quantity_;
    EmpiricalDistribution limit_sell_quantity_;
    EmpiricalDistribution market_buy_quantity_;
    EmpiricalDistribution market_sell_quantity_;
    EmpiricalDistribution cancel_bid_quantity_;
    EmpiricalDistribution cancel_ask_quantity_;
    EmpiricalDistribution limit_buy_distance_;
    EmpiricalDistribution limit_sell_distance_;
    EmpiricalDistribution cancel_bid_distance_;
    EmpiricalDistribution cancel_ask_distance_;
};

} // namespace dlob
