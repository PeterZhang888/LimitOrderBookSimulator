#pragma once

#include "SimulationConfig.hpp"
#include "Order.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct ParameterVector {
    std::vector<std::string> names;
    std::vector<double> values;
};

class CalibrationSearchSpace {
public:
    explicit CalibrationSearchSpace(int n_candidates)
        : n_candidates_(n_candidates > 0 ? n_candidates : 1) {}

    int total_candidates() const { return n_candidates_; }

    SimulationConfig make_config(int candidate_id, int repeat_id, std::uint64_t base_seed, std::int64_t end_time_ns) const {
        SimulationConfig c;
        c.calibration_id = candidate_id;
        c.seed = base_seed + static_cast<std::uint64_t>(candidate_id) * 1009ULL + static_cast<std::uint64_t>(repeat_id) * 9176ULL;
        if (end_time_ns > 0) c.end_time_ns = end_time_ns;

        const double u0 = unit(candidate_id, 0);
        const double u1 = unit(candidate_id, 1);
        const double u2 = unit(candidate_id, 2);
        const double u3 = unit(candidate_id, 3);
        const double u4 = unit(candidate_id, 4);
        const double u5 = unit(candidate_id, 5);
        const double u6 = unit(candidate_id, 6);
        const double u7 = unit(candidate_id, 7);
        const double u8 = unit(candidate_id, 8);
        const double u9 = unit(candidate_id, 9);
        const double u10 = unit(candidate_id, 10);
        const double u11 = unit(candidate_id, 11);
        const double u12 = unit(candidate_id, 12);
        const double u13 = unit(candidate_id, 13);
        const double u14 = unit(candidate_id, 14);
        const double u15 = unit(candidate_id, 15);
        const double u16 = unit(candidate_id, 16);
        const double u17 = unit(candidate_id, 17);

        const double mu_scale = range(u0, 0.70, 1.40);
        const double market_mu_scale = range(u1, 0.60, 1.80);
        const double cancel_mu_scale = range(u2, 0.60, 1.80);
        const double beta = range(u3, 5.0, 18.0);
        const double self_branch = range(u4, 0.05, 0.35);
        const double cross_branch = range(u5, 0.00, 0.08);
        const double directional_branch = range(u6, 0.00, 0.12);

        c.hawkes_beta = beta;
        c.hawkes_mu = {{12.0 * mu_scale, 12.0 * mu_scale, 2.2 * market_mu_scale, 2.2 * market_mu_scale, 18.0 * cancel_mu_scale, 18.0 * cancel_mu_scale}};
        for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
            for (int j = 0; j < NUM_EVENT_TYPES; ++j) c.hawkes_alpha[i][j] = 0.0;
        }
        for (int i = 0; i < NUM_EVENT_TYPES; ++i) c.hawkes_alpha[i][i] = beta * self_branch;
        // Symmetric cross-excitation within passive/aggressive/cancel pairs.
        c.hawkes_alpha[0][1] = beta * cross_branch; c.hawkes_alpha[1][0] = beta * cross_branch;
        c.hawkes_alpha[2][3] = beta * cross_branch; c.hawkes_alpha[3][2] = beta * cross_branch;
        c.hawkes_alpha[4][5] = beta * cross_branch; c.hawkes_alpha[5][4] = beta * cross_branch;
        // Directional pressure channels: buys excite market buys and ask cancellations; sells analogously.
        c.hawkes_alpha[2][0] = beta * directional_branch;
        c.hawkes_alpha[5][2] = beta * directional_branch;
        c.hawkes_alpha[3][1] = beta * directional_branch;
        c.hawkes_alpha[4][3] = beta * directional_branch;

        c.initial_depth_scale = range(u7, 0.60, 1.40);
        c.quote_improvement_probability = range(u8, 0.00, 0.12);
        c.market_order_quantity_scale = range(u9, 0.60, 1.80);
        c.cancel_quantity_scale = range(u10, 0.60, 1.80);
        c.limit_distance_shift_ticks = static_cast<int>(std::llround(range(u11, -1.0, 3.0)));

        c.num_market_maker_agents = static_cast<int>(std::llround(range(u12, 1.0, 4.0)));
        c.market_maker_order_quantity = static_cast<int>(std::llround(range(u13, 40.0, 150.0)));
        c.market_maker_min_spread_ticks = static_cast<int>(std::llround(range(u14, 2.0, 8.0)));
        c.market_maker_quote_skip_probability = range(u15, 0.00, 0.35);
        c.market_maker_interval_ns = static_cast<std::int64_t>(std::llround(range(u16, 250.0, 1500.0))) * 1000LL * 1000LL;

        c.num_momentum_agents = static_cast<int>(std::llround(range(u17, 1.0, 5.0)));
        c.momentum_order_quantity = 300;
        c.momentum_threshold_ticks = 0.25;

        // Keep institutional pressure moderate in the baseline search. These can be widened later.
        c.num_buy_institutional_agents = 2;
        c.num_sell_institutional_agents = 2;
        c.institutional_child_quantity = 1000;
        c.institutional_participation_cap = 0.20;
        return c;
    }

    ParameterVector parameter_vector(int candidate_id) const {
        ParameterVector p;
        p.names = {
            "mu_scale", "market_mu_scale", "cancel_mu_scale", "hawkes_beta",
            "self_branch", "cross_branch", "directional_branch", "initial_depth_scale",
            "quote_improvement_probability", "market_order_quantity_scale", "cancel_quantity_scale",
            "limit_distance_shift_ticks", "num_market_maker_agents", "market_maker_order_quantity",
            "market_maker_min_spread_ticks", "market_maker_quote_skip_probability", "market_maker_interval_ms",
            "num_momentum_agents"
        };
        SimulationConfig c = make_config(candidate_id, 0, 1, 0);
        p.values = {
            range(unit(candidate_id, 0), 0.70, 1.40),
            range(unit(candidate_id, 1), 0.60, 1.80),
            range(unit(candidate_id, 2), 0.60, 1.80),
            c.hawkes_beta,
            range(unit(candidate_id, 4), 0.05, 0.35),
            range(unit(candidate_id, 5), 0.00, 0.08),
            range(unit(candidate_id, 6), 0.00, 0.12),
            c.initial_depth_scale,
            c.quote_improvement_probability,
            c.market_order_quantity_scale,
            c.cancel_quantity_scale,
            static_cast<double>(c.limit_distance_shift_ticks),
            static_cast<double>(c.num_market_maker_agents),
            static_cast<double>(c.market_maker_order_quantity),
            static_cast<double>(c.market_maker_min_spread_ticks),
            c.market_maker_quote_skip_probability,
            static_cast<double>(c.market_maker_interval_ns / 1000000LL),
            static_cast<double>(c.num_momentum_agents)
        };
        return p;
    }

private:
    int n_candidates_ = 1;

    static std::uint64_t splitmix64(std::uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    double unit(int candidate_id, int dimension) const {
        const std::uint64_t h = splitmix64(static_cast<std::uint64_t>(candidate_id) * 1315423911ULL + static_cast<std::uint64_t>(dimension) * 2654435761ULL);
        return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
    }

    static double range(double u, double lo, double hi) { return lo + (hi - lo) * u; }
};
