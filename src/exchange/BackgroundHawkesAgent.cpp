#include "BackgroundHawkesAgent.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

BackgroundHawkesAgent::BackgroundHawkesAgent(const SimulationConfig& config)
    : mu_(config.hawkes_mu), alpha_(config.hawkes_alpha), beta_(std::max(1e-6, config.hawkes_beta)), rng_(config.seed + 911382323ULL) {}

std::vector<HawkesEvent> BackgroundHawkesAgent::simulate(std::int64_t start_time_ns, std::int64_t end_time_ns) {
    if (end_time_ns <= start_time_ns) return {};

    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
    std::vector<HawkesEvent> events;
    events.reserve(static_cast<std::size_t>((end_time_ns - start_time_ns) / 1000000000LL * 70));

    double t = static_cast<double>(start_time_ns) / 1e9;
    const double end_t = static_cast<double>(end_time_ns) / 1e9;
    std::array<double, 6> excitation{};
    excitation.fill(0.0);

    while (t < end_t) {
        std::array<double, 6> lambda{};
        double lambda_sum = 0.0;
        for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
            lambda[i] = std::max(0.0, mu_[i] + excitation[i]);
            lambda_sum += lambda[i];
        }
        if (lambda_sum <= 1e-12) break;

        const double u = std::max(1e-12, uniform_01(rng_));
        const double dt = -std::log(u) / lambda_sum;
        t += dt;
        if (t >= end_t) break;

        const double decay = std::exp(-beta_ * dt);
        for (double& x : excitation) x *= decay;

        std::array<double, 6> lambda_candidate{};
        double candidate_sum = 0.0;
        for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
            lambda_candidate[i] = std::max(0.0, mu_[i] + excitation[i]);
            candidate_sum += lambda_candidate[i];
        }

        if (uniform_01(rng_) * lambda_sum > candidate_sum) continue;

        double draw = uniform_01(rng_) * candidate_sum;
        int event_index = 0;
        for (; event_index < NUM_EVENT_TYPES - 1; ++event_index) {
            draw -= lambda_candidate[event_index];
            if (draw <= 0.0) break;
        }

        const std::int64_t time_ns = static_cast<std::int64_t>(std::llround(t * 1e9));
        events.push_back(HawkesEvent{time_ns, static_cast<EventType>(event_index)});

        // Event j excites future intensity i by alpha[i][j].
        for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
            excitation[i] += std::max(0.0, alpha_[i][event_index]);
        }
    }

    return events;
}
