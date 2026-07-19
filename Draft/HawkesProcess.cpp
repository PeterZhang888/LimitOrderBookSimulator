#include "HawkesProcess.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

// Convert EventType to readable text
std::string event_type_to_string(EventType type) {
    switch (type) {
        case EventType::LimitBuy:
            return "limit_buy";
        case EventType::LimitSell:
            return "limit_sell";
        case EventType::MarketBuy:
            return "market_buy";
        case EventType::MarketSell:
            return "market_sell";
        case EventType::CancelBid:
            return "cancel_bid";
        case EventType::CancelAsk:
            return "cancel_ask";
        default:
            return "unknown";
    }
}

// Constructor
HawkesProcess::HawkesProcess(
    const Vector6& mu,
    const Matrix6& alpha,
    double beta,
    std::uint64_t seed
)
    : mu_(mu),
      alpha_(alpha),
      beta_(beta),
      rng_(seed),
      uniform_(0.0, 1.0)
{
    if (beta_ <= 0.0) {
        throw std::invalid_argument("beta must be positive");
    }
}

// Compute the 6 intensities at time t
HawkesProcess::Vector6 HawkesProcess::compute_intensity(double t) const {
    Vector6 lambda = mu_;

    const double decay_cutoff = 1e-6;

    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
        const auto& event = *it;

        double event_time_seconds =
            static_cast<double>(event.time_ns) / 1'000'000'000.0;

        if (event_time_seconds >= t) {
            continue;
        }

        double age = t - event_time_seconds;
        double decay = std::exp(-beta_ * age);

        if (decay < decay_cutoff) {
            break;
        }

        int past_type = static_cast<int>(event.type);

        for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
            lambda[i] += alpha_[i][past_type] * decay;
        }
    }

    return lambda;
}

// Sum all six intensities
double HawkesProcess::total_intensity(const Vector6& lambda) const {
    double total = 0.0;

    for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
        total += lambda[i];
    }

    return total;
}

// Sample event type proportional to lambda_i
EventType HawkesProcess::sample_event_type(
    const Vector6& lambda,
    double total_lambda
) {
    if (total_lambda <= 0.0) {
        throw std::runtime_error("total_lambda must be positive");
    }

    double u = uniform_(rng_) * total_lambda;
    double cumulative = 0.0;

    for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
        cumulative += lambda[i];

        if (u <= cumulative) {
            return static_cast<EventType>(i);
        }
    }

    return static_cast<EventType>(NUM_EVENT_TYPES - 1);
}

std::optional<HawkesEvent> HawkesProcess::simulate_next(
    std::int64_t current_time_ns,
    std::int64_t end_time_ns,
    const Vector6& state_multiplier
) {
    std::int64_t t_ns = current_time_ns;

    while (t_ns < end_time_ns) {
        double t_seconds =
            static_cast<double>(t_ns) / 1'000'000'000.0;

        Vector6 lambda = compute_intensity(t_seconds);

        for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
            lambda[i] *= state_multiplier[i];

            if (lambda[i] < 0.0) {
                lambda[i] = 0.0;
            }
        }

        double lambda_total = 0.0;
        for (double x : lambda) {
            lambda_total += x;
        }

        if (lambda_total <= 0.0) {
            return std::nullopt;
        }

        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        double u1 = uniform(rng_);
        if (u1 <= 0.0) {
            continue;
        }

        double dt_seconds = -std::log(u1) / lambda_total;

        std::int64_t dt_ns =
            std::max<std::int64_t>(
                1,
                static_cast<std::int64_t>(
                    std::llround(dt_seconds * 1'000'000'000.0)
                )
            );

        std::int64_t candidate_time_ns = t_ns + dt_ns;

        if (candidate_time_ns > end_time_ns) {
            return std::nullopt;
        }

        double candidate_time_seconds =
            static_cast<double>(candidate_time_ns) / 1'000'000'000.0;

        Vector6 candidate_lambda =
            compute_intensity(candidate_time_seconds);

        for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
            candidate_lambda[i] *= state_multiplier[i];

            if (candidate_lambda[i] < 0.0) {
                candidate_lambda[i] = 0.0;
            }
        }

        double candidate_total = 0.0;
        for (double x : candidate_lambda) {
            candidate_total += x;
        }

        double u2 = uniform(rng_);

        if (u2 * lambda_total <= candidate_total) {
            double u3 = uniform(rng_) * candidate_total;

            double cumulative = 0.0;
            int selected_type = 0;

            for (int i = 0; i < NUM_EVENT_TYPES; ++i) {
                cumulative += candidate_lambda[i];

                if (u3 <= cumulative) {
                    selected_type = i;
                    break;
                }
            }

            HawkesEvent event{
                candidate_time_ns,
                static_cast<EventType>(selected_type)
            };

            history_.push_back(event);

            return event;
        }

        t_ns = candidate_time_ns;
    }

    return std::nullopt;
}

// Simulate Hawkes events from start_time to end_time
std::vector<HawkesEvent> HawkesProcess::simulate(
    std::int64_t start_time_ns,
    std::int64_t end_time_ns
){
    history_.clear();

    std::int64_t t_ns = start_time_ns;

    while (t_ns < end_time_ns) {
        double t_seconds = static_cast<double>(t_ns) / 1'000'000'000.0;
        Vector6 lambda = compute_intensity(t_seconds);
        double lambda_total = total_intensity(lambda);

        if (lambda_total <= 0.0) {
            break;
        }

        double u1 = uniform_(rng_);

        if (u1 <= 0.0) {
            continue;
        }

        double dt_seconds = -std::log(u1) / lambda_total;

        std::int64_t dt_ns =
            std::max<std::int64_t>(
                1,
                static_cast<std::int64_t>(
                    std::llround(dt_seconds * 1'000'000'000.0)
                )
            );

        std::int64_t candidate_time_ns = t_ns + dt_ns;

        if (candidate_time_ns > end_time_ns) {
            break;
        }

        double candidate_time_seconds = static_cast<double>(candidate_time_ns) / 1'000'000'000.0;
        Vector6 candidate_lambda = compute_intensity(candidate_time_seconds);
        double candidate_total = total_intensity(candidate_lambda);

        double u2 = uniform_(rng_);

        if (u2 <= candidate_total / lambda_total) {
            EventType event_type =
                sample_event_type(candidate_lambda, candidate_total);

            HawkesEvent event{
                candidate_time_ns,
                event_type
            };

            history_.push_back(event);
        }

        t_ns = candidate_time_ns;
    }

    return history_;
}

// Return generated history
const std::vector<HawkesEvent>& HawkesProcess::history() const {
    return history_;
}
