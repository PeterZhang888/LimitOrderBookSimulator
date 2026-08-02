// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "exchange/BackgroundHawkesAgent.hpp"

#include "common/DataPaths.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace dlob {
namespace {

constexpr double nanoseconds_per_second = 1.0e9;
constexpr double minimum_intensity = 1.0e-12;
constexpr std::uint64_t stochastic_baseline_domain =
    0x510e'527f'ade6'82d1ULL;

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9e37'79b9'7f4a'7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58'476d'1ce4'e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d0'49bb'1331'11ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] double stochastic_baseline_normal(
    std::uint64_t seed,
    std::uint64_t bin_index) noexcept {
    constexpr double denominator = 9'007'199'254'740'992.0; // 2^53
    const auto open_uniform = [](std::uint64_t bits) noexcept {
        return (static_cast<double>(bits >> 11U) + 0.5) / denominator;
    };
    const std::uint64_t key = seed ^ stochastic_baseline_domain
        ^ (bin_index * 0xd1b5'4a32'd192'ed03ULL);
    const double first = open_uniform(splitmix64(key));
    const double second = open_uniform(splitmix64(
        key ^ 0xa409'3822'299f'31d0ULL));
    return std::sqrt(-2.0 * std::log(first))
        * std::cos(2.0 * std::numbers::pi * second);
}

[[nodiscard]] BackgroundHawkesMatrix integrated_branching_matrix(
    const BackgroundHawkesConfig& config) {
    BackgroundHawkesMatrix integrated{};
    for (std::size_t row = 0; row < integrated.size(); ++row) {
        for (std::size_t column = 0; column < integrated[row].size(); ++column) {
            integrated[row][column] = config.alpha[row][column] / config.beta
                + config.slow_alpha[row][column] / config.slow_beta;
        }
    }
    return integrated;
}

[[nodiscard]] double maximum_integrated_column_sum(
    const BackgroundHawkesMatrix& matrix) {
    double maximum = 0.0;
    for (std::size_t column = 0; column < matrix.size(); ++column) {
        double sum = 0.0;
        for (std::size_t row = 0; row < matrix.size(); ++row) {
            sum += matrix[row][column];
        }
        maximum = std::max(maximum, sum);
    }
    return maximum;
}

[[nodiscard]] bool has_state_response(const BackgroundHawkesConfig& config) {
    for (const auto& row : config.state_log_multiplier_coefficients) {
        for (const double value : row) {
            if (value != 0.0) return true;
        }
    }
    return false;
}

// Perron iteration on I + matrix avoids the period-two oscillation of an
// otherwise valid sparse nonnegative matrix.  For a nonnegative matrix the
// Perron root shifts by exactly one.  The fixed six-dimensional problem and
// generous iteration cap make this both deterministic and inexpensive.
[[nodiscard]] double nonnegative_spectral_radius(
    const BackgroundHawkesMatrix& matrix) {
    BackgroundHawkesVector vector{};
    vector.fill(1.0);
    double previous = -1.0;
    double estimate = 0.0;
    for (int iteration = 0; iteration < 20'000; ++iteration) {
        BackgroundHawkesVector product{};
        for (std::size_t row = 0; row < matrix.size(); ++row) {
            product[row] = vector[row];
            for (std::size_t column = 0; column < matrix[row].size(); ++column) {
                product[row] += matrix[row][column] * vector[column];
            }
        }
        double norm = 0.0;
        for (const double value : product) norm = std::max(norm, value);
        if (!(norm > 0.0) || !std::isfinite(norm)) {
            throw std::invalid_argument(
                "background Hawkes branching matrix has invalid spectral radius");
        }
        for (double& value : product) value /= norm;
        vector = product;

        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t row = 0; row < matrix.size(); ++row) {
            double shifted_product = vector[row];
            for (std::size_t column = 0; column < matrix[row].size(); ++column) {
                shifted_product += matrix[row][column] * vector[column];
            }
            numerator += vector[row] * shifted_product;
            denominator += vector[row] * vector[row];
        }
        estimate = numerator / denominator - 1.0;
        if (iteration > 64
            && std::abs(estimate - previous)
                <= 1.0e-14 * std::max(1.0, std::abs(estimate))) {
            break;
        }
        previous = estimate;
    }
    return std::max(0.0, estimate);
}

void validate_hawkes_clock_config(const BackgroundHawkesConfig& config) {
    if (!std::isfinite(config.activity_scale) || config.activity_scale <= 0.0) {
        throw std::invalid_argument("background Hawkes activity scale must be positive");
    }
    if (!std::isfinite(config.beta) || config.beta <= 0.0
        || !std::isfinite(config.slow_beta) || config.slow_beta <= 0.0) {
        throw std::invalid_argument(
            "background Hawkes fast and slow decay rates must be positive");
    }
    if (!std::isfinite(config.stochastic_baseline_persistence)
        || config.stochastic_baseline_persistence < 0.0
        || config.stochastic_baseline_persistence >= 1.0
        || !std::isfinite(config.stochastic_baseline_std)
        || config.stochastic_baseline_std < 0.0
        || config.stochastic_baseline_std > 2.5
        || config.stochastic_baseline_bin_width_ns <= 0
        || !std::isfinite(config.stochastic_baseline_standardized_bound)
        || config.stochastic_baseline_standardized_bound < 1.0
        || config.stochastic_baseline_standardized_bound > 8.0) {
        throw std::invalid_argument(
            "background Hawkes stochastic-baseline configuration is invalid");
    }
    if (config.stochastic_baseline_std > 0.0
        && config.stochastic_baseline_persistence <= 0.0) {
        throw std::invalid_argument(
            "stochastic Hawkes baseline requires positive persistence");
    }
    if (config.stochastic_baseline_std > 0.0
        && config.stochastic_baseline_normalization_bins == 0U) {
        throw std::invalid_argument(
            "stochastic Hawkes baseline requires a positive normalization horizon");
    }
    if (config.tick_size <= 0 || config.target_spread_ticks <= 0) {
        throw std::invalid_argument(
            "background Hawkes tick size and target spread must be positive");
    }
    for (std::size_t row = 0; row < config.alpha.size(); ++row) {
        if (!std::isfinite(config.mu[row]) || config.mu[row] < 0.0) {
            throw std::invalid_argument(
                "background Hawkes immigration rates must be finite and nonnegative");
        }
        for (std::size_t column = 0; column < config.alpha[row].size(); ++column) {
            if (!std::isfinite(config.alpha[row][column])
                || config.alpha[row][column] < 0.0
                || !std::isfinite(config.slow_alpha[row][column])
                || config.slow_alpha[row][column] < 0.0) {
                throw std::invalid_argument(
                    "background Hawkes excitation amplitudes must be finite and nonnegative");
            }
        }
        for (const double coefficient :
             config.state_log_multiplier_coefficients[row]) {
            if (!std::isfinite(coefficient)) {
                throw std::invalid_argument(
                    "background Hawkes state coefficients must be finite");
            }
        }
    }
    if (!std::isfinite(config.state_reference_bid_depth)
        || config.state_reference_bid_depth <= 0.0
        || !std::isfinite(config.state_reference_ask_depth)
        || config.state_reference_ask_depth <= 0.0
        || !std::isfinite(config.state_log_multiplier_bound)
        || config.state_log_multiplier_bound <= 0.0
        || config.state_log_multiplier_bound > 50.0) {
        throw std::invalid_argument(
            "background Hawkes state references and log bound are invalid");
    }

    const BackgroundHawkesMatrix integrated =
        integrated_branching_matrix(config);
    const double spectral_radius = nonnegative_spectral_radius(integrated);
    if (!std::isfinite(spectral_radius) || spectral_radius >= 0.95) {
        throw std::invalid_argument(
            "background Hawkes integrated branching spectral radius must be below 0.95");
    }
    // With state-dependent type selection, an event's type can change while
    // its already sampled timestamp remains fixed.  Unequal excitation
    // columns then change the total hazard after that event.  A strict
    // induced one-norm bound gives a transparent subcriticality certificate
    // for every possible type relabelling, not merely for the latent linear
    // Hawkes process used to derive immigration rates.
    if (has_state_response(config)
        && maximum_integrated_column_sum(integrated) >= 0.75) {
        throw std::invalid_argument(
            "state-responsive background Hawkes maximum integrated column sum must be below 0.75");
    }

    if (!config.intraday_factors.empty()) {
        if (config.intraday_bin_width_ns <= 0) {
            throw std::invalid_argument(
                "background Hawkes intraday bin width must be positive");
        }
        BackgroundHawkesVector sums{};
        for (const BackgroundHawkesVector& bin : config.intraday_factors) {
            for (std::size_t index = 0; index < bin.size(); ++index) {
                if (!std::isfinite(bin[index]) || bin[index] < 0.0) {
                    throw std::invalid_argument(
                        "background Hawkes intraday factors must be finite and nonnegative");
                }
                sums[index] += bin[index];
            }
        }
        const double bins = static_cast<double>(config.intraday_factors.size());
        for (const double sum : sums) {
            const double mean = sum / bins;
            if (std::abs(mean - 1.0) > 1.0e-10) {
                throw std::invalid_argument(
                    "background Hawkes intraday factors must have type-wise mean one");
            }
        }
    }

    if (config.validate_stationary_target) {
        for (std::size_t row = 0;
             row < config.stationary_target_rates.size(); ++row) {
            const double target = config.stationary_target_rates[row];
            if (!std::isfinite(target) || target < 0.0) {
                throw std::invalid_argument(
                    "background Hawkes stationary targets must be finite and nonnegative");
            }
            double reconstructed = config.activity_scale * config.mu[row];
            for (std::size_t column = 0;
                 column < config.stationary_target_rates.size(); ++column) {
                reconstructed += integrated[row][column]
                    * config.stationary_target_rates[column];
            }
            const double tolerance = 1.0e-10
                * std::max({1.0, std::abs(target), std::abs(reconstructed)});
            if (std::abs(reconstructed - target) > tolerance) {
                throw std::invalid_argument(
                    "background Hawkes immigration rates do not reconstruct stationary target");
            }
        }
    }
}

} // namespace

BackgroundHawkesConfig::BackgroundHawkesConfig() {
    for (auto& row : alpha) row.fill(0.0);
    for (auto& row : slow_alpha) row.fill(0.0);
    for (auto& row : state_log_multiplier_coefficients) row.fill(0.0);
    // The compact empirical artifacts identify marginal event-type rates but
    // not lagged cross-type kernels.  Retain within-type clustering through a
    // diagonal self-exciting structure; do not impose unestimated cross-flow
    // excitation that can make sparse empirical rate vectors infeasible with
    // nonnegative Hawkes baselines.
    for (std::size_t i = 0; i < alpha.size(); ++i) alpha[i][i] = 0.20;
}

BackgroundHawkesStream::BackgroundHawkesStream(
    const BackgroundHawkesConfig& config,
    std::int64_t start_time_ns)
    : config_(config),
      clock_rng_(config.seed),
      state_response_enabled_(has_state_response(config)),
      stochastic_baseline_enabled_(config.stochastic_baseline_std > 0.0),
      time_seconds_(static_cast<double>(start_time_ns)
                    / nanoseconds_per_second) {
    if (start_time_ns < 0) {
        throw std::invalid_argument(
            "background Hawkes stream start time must be nonnegative");
    }
    validate_hawkes_clock_config(config_);
    if (stochastic_baseline_enabled_) {
        stochastic_baseline_log_normalizer_ =
            stochastic_baseline_log_normalizer();
        const double origin = static_cast<double>(
            config_.stochastic_baseline_origin_ns) / nanoseconds_per_second;
        const double width = static_cast<double>(
            config_.stochastic_baseline_bin_width_ns) / nanoseconds_per_second;
        if (time_seconds_ >= origin) {
            const double raw = std::floor((time_seconds_ - origin) / width);
            if (raw > static_cast<double>(
                    std::numeric_limits<std::uint64_t>::max())) {
                throw std::overflow_error(
                    "stochastic Hawkes baseline bin index overflow");
            }
            stochastic_baseline_bin_index_ = static_cast<std::uint64_t>(raw);
        }
        stochastic_baseline_log_state_ =
            stochastic_baseline_log_state_at_bin(
                stochastic_baseline_bin_index_);
    }
}

BackgroundHawkesVector BackgroundHawkesStream::intraday_factor_at(
    double time_seconds) const {
    BackgroundHawkesVector factors{};
    factors.fill(1.0);
    if (config_.intraday_factors.empty()) return factors;

    const double origin = static_cast<double>(config_.intraday_origin_ns)
        / nanoseconds_per_second;
    const double width = static_cast<double>(config_.intraday_bin_width_ns)
        / nanoseconds_per_second;
    std::size_t index = 0;
    if (time_seconds > origin) {
        const double raw = std::floor((time_seconds - origin) / width);
        if (raw > 0.0) {
            index = static_cast<std::size_t>(std::min(
                raw,
                static_cast<double>(config_.intraday_factors.size() - 1U)));
        }
    }
    return config_.intraday_factors[index];
}

double BackgroundHawkesStream::next_intraday_boundary_seconds(
    double time_seconds) const {
    if (config_.intraday_factors.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const double origin = static_cast<double>(config_.intraday_origin_ns)
        / nanoseconds_per_second;
    const double width = static_cast<double>(config_.intraday_bin_width_ns)
        / nanoseconds_per_second;
    if (time_seconds < origin) return origin;
    const double raw_index = std::floor((time_seconds - origin) / width);
    const auto index = static_cast<std::size_t>(std::max(0.0, raw_index));
    if (index + 1U >= config_.intraday_factors.size()) {
        return std::numeric_limits<double>::infinity();
    }
    return origin + static_cast<double>(index + 1U) * width;
}

double BackgroundHawkesStream::stochastic_baseline_log_state_at_bin(
    std::uint64_t bin_index) const {
    if (!stochastic_baseline_enabled_) return 0.0;
    const double persistence = config_.stochastic_baseline_persistence;
    const double stationary_std = config_.stochastic_baseline_std;
    double state = stationary_std * stochastic_baseline_normal(
        config_.seed, 0U);
    const double innovation_std = stationary_std
        * std::sqrt((1.0 - persistence) * (1.0 + persistence));
    for (std::uint64_t index = 1U; index <= bin_index; ++index) {
        state = persistence * state + innovation_std
            * stochastic_baseline_normal(config_.seed, index);
    }
    if (!std::isfinite(state)) {
        throw std::overflow_error(
            "stochastic Hawkes baseline state is not finite");
    }
    return state;
}

double BackgroundHawkesStream::bounded_stochastic_baseline_log_state(
    double state) const {
    const double bound = config_.stochastic_baseline_standardized_bound
        * config_.stochastic_baseline_std;
    return std::clamp(state, -bound, bound);
}

double BackgroundHawkesStream::stochastic_baseline_log_normalizer() const {
    if (!stochastic_baseline_enabled_) return 0.0;
    const std::uint64_t bins =
        config_.stochastic_baseline_normalization_bins;
    const double persistence = config_.stochastic_baseline_persistence;
    const double stationary_std = config_.stochastic_baseline_std;
    const double innovation_std = stationary_std
        * std::sqrt((1.0 - persistence) * (1.0 + persistence));
    double state = stationary_std * stochastic_baseline_normal(
        config_.seed, 0U);
    double multiplier_sum = 0.0;
    for (std::uint64_t index = 0U; index < bins; ++index) {
        if (index > 0U) {
            state = persistence * state + innovation_std
                * stochastic_baseline_normal(config_.seed, index);
        }
        multiplier_sum += std::exp(
            bounded_stochastic_baseline_log_state(state));
    }
    const double mean = multiplier_sum / static_cast<double>(bins);
    if (!(mean > 0.0) || !std::isfinite(mean)) {
        throw std::overflow_error(
            "stochastic Hawkes baseline session normalizer is invalid");
    }
    return std::log(mean);
}

double BackgroundHawkesStream::stochastic_baseline_multiplier_at(
    double time_seconds) const {
    if (!stochastic_baseline_enabled_) return 1.0;
    const double origin = static_cast<double>(
        config_.stochastic_baseline_origin_ns) / nanoseconds_per_second;
    if (time_seconds < origin) return 1.0;
    const double width = static_cast<double>(
        config_.stochastic_baseline_bin_width_ns) / nanoseconds_per_second;
    const double raw = std::floor((time_seconds - origin) / width);
    if (raw < 0.0 || raw > static_cast<double>(
            std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error(
            "stochastic Hawkes baseline diagnostic bin overflow");
    }
    const double state = stochastic_baseline_log_state_at_bin(
        static_cast<std::uint64_t>(raw));
    const double multiplier = std::exp(
        bounded_stochastic_baseline_log_state(state)
        - stochastic_baseline_log_normalizer_);
    if (!(multiplier > 0.0) || !std::isfinite(multiplier)) {
        throw std::overflow_error(
            "stochastic Hawkes baseline multiplier is invalid");
    }
    return multiplier;
}

double BackgroundHawkesStream::current_stochastic_baseline_multiplier() const {
    if (!stochastic_baseline_enabled_) return 1.0;
    const double origin = static_cast<double>(
        config_.stochastic_baseline_origin_ns) / nanoseconds_per_second;
    if (time_seconds_ < origin) return 1.0;
    const double multiplier = std::exp(
        bounded_stochastic_baseline_log_state(
            stochastic_baseline_log_state_)
        - stochastic_baseline_log_normalizer_);
    if (!(multiplier > 0.0) || !std::isfinite(multiplier)) {
        throw std::overflow_error(
            "current stochastic Hawkes baseline multiplier is invalid");
    }
    return multiplier;
}

double BackgroundHawkesStream::next_stochastic_baseline_boundary_seconds(
    double time_seconds) const {
    if (!stochastic_baseline_enabled_) {
        return std::numeric_limits<double>::infinity();
    }
    const double origin = static_cast<double>(
        config_.stochastic_baseline_origin_ns) / nanoseconds_per_second;
    const double width = static_cast<double>(
        config_.stochastic_baseline_bin_width_ns) / nanoseconds_per_second;
    if (time_seconds < origin) return origin;
    const double raw = std::floor((time_seconds - origin) / width);
    return origin + (raw + 1.0) * width;
}

void BackgroundHawkesStream::advance_stochastic_baseline_to(
    double time_seconds) {
    if (!stochastic_baseline_enabled_) return;
    const double origin = static_cast<double>(
        config_.stochastic_baseline_origin_ns) / nanoseconds_per_second;
    if (time_seconds < origin) return;
    const double width = static_cast<double>(
        config_.stochastic_baseline_bin_width_ns) / nanoseconds_per_second;
    const double raw = std::floor((time_seconds - origin) / width);
    if (raw < 0.0 || raw > static_cast<double>(
            std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error(
            "stochastic Hawkes baseline runtime bin overflow");
    }
    const auto target = static_cast<std::uint64_t>(raw);
    const double persistence = config_.stochastic_baseline_persistence;
    const double stationary_std = config_.stochastic_baseline_std;
    const double innovation_std = stationary_std
        * std::sqrt((1.0 - persistence) * (1.0 + persistence));
    while (stochastic_baseline_bin_index_ < target) {
        ++stochastic_baseline_bin_index_;
        stochastic_baseline_log_state_ =
            persistence * stochastic_baseline_log_state_
            + innovation_std * stochastic_baseline_normal(
                config_.seed, stochastic_baseline_bin_index_);
    }
    if (!std::isfinite(stochastic_baseline_log_state_)) {
        throw std::overflow_error(
            "stochastic Hawkes baseline runtime state is not finite");
    }
}

BackgroundHawkesVector BackgroundHawkesStream::baseline_intensities_at(
    double time_seconds,
    double stochastic_baseline_multiplier) const {
    const BackgroundHawkesVector intraday = intraday_factor_at(time_seconds);
    BackgroundHawkesVector baseline{};
    for (std::size_t index = 0; index < baseline.size(); ++index) {
        baseline[index] = config_.activity_scale * config_.mu[index]
            * intraday[index];
        if (stochastic_baseline_enabled_) {
            baseline[index] *= stochastic_baseline_multiplier;
        }
    }
    return baseline;
}

BackgroundHawkesVector BackgroundHawkesStream::apply_state_type_response(
    BackgroundHawkesVector intensities,
    const MarketState& state,
    double liquidity_removal_log_score) const {
    if (!std::isfinite(liquidity_removal_log_score)) {
        throw std::invalid_argument(
            "background Hawkes liquidity-removal log score must be finite");
    }
    // Avoid even harmless additional arithmetic in compatibility mode.  The
    // training fit excludes unavailable/one-sided states, so do not
    // extrapolate its coefficients when either displayed top queue is absent.
    const bool valid_fitted_state = state.best_bid_ticks > 0
        && state.best_ask_ticks > state.best_bid_ticks
        && state.best_bid_depth > 0
        && state.best_ask_depth > 0;
    // Do not increase depletion while the reduced book is one-sided. Its
    // reflecting boundary must restore a fitted two-sided state before either
    // empirical queue response or persistent removal regime is applied.
    if (!valid_fitted_state) {
        return intensities;
    }
    const bool removal_regime_enabled = liquidity_removal_log_score != 0.0;
    if (!state_response_enabled_ && !removal_regime_enabled) return intensities;

    double total_hazard = 0.0;
    for (const double value : intensities) total_hazard += value;
    if (total_hazard <= minimum_intensity) {
        return intensities;
    }

    double spread_ticks = static_cast<double>(config_.target_spread_ticks);
    if (state.best_bid_ticks > 0 && state.best_ask_ticks > state.best_bid_ticks) {
        spread_ticks = static_cast<double>(
            static_cast<std::int64_t>(state.best_ask_ticks)
            - static_cast<std::int64_t>(state.best_bid_ticks))
            / static_cast<double>(std::max(1, config_.tick_size));
    }
    const double bid_depth = static_cast<double>(
        std::max(0, state.best_bid_depth));
    const double ask_depth = static_cast<double>(
        std::max(0, state.best_ask_depth));
    const double depth_sum = bid_depth + ask_depth;
    const auto depth_feature = [](double depth, double reference) {
        if (!(depth > 0.0)) return std::log(0.35);
        const double ratio = depth / reference;
        if (ratio < 0.5) return std::log(0.35);
        if (ratio < 1.5) return 0.0;
        return std::log(2.5);
    };
    const double raw_imbalance = depth_sum > 0.0
        ? (bid_depth - ask_depth) / depth_sum : 0.0;
    const double imbalance_feature = raw_imbalance < -0.6 ? -0.8
        : raw_imbalance < -0.2 ? -0.4
        : raw_imbalance < 0.2 ? 0.0
        : raw_imbalance < 0.6 ? 0.4 : 0.8;
    const std::array<double, background_hawkes_state_feature_count> features{
        spread_ticks <= 1.0 ? 0.0 : std::log(2.0),
        depth_feature(bid_depth, config_.state_reference_bid_depth),
        depth_feature(ask_depth, config_.state_reference_ask_depth),
        imbalance_feature,
    };

    BackgroundHawkesVector raw_multiplier{};
    double weighted_sum = 0.0;
    for (std::size_t type = 0; type < intensities.size(); ++type) {
        double score = 0.0;
        if (state_response_enabled_) {
            for (std::size_t feature = 0; feature < features.size(); ++feature) {
                score += config_.state_log_multiplier_coefficients[type][feature]
                    * features[feature];
            }
        }
        if (type == static_cast<std::size_t>(HawkesEventType::MarketBuy)
            || type == static_cast<std::size_t>(HawkesEventType::MarketSell)
            || type == static_cast<std::size_t>(HawkesEventType::CancelBid)
            || type == static_cast<std::size_t>(HawkesEventType::CancelAsk)) {
            score += liquidity_removal_log_score;
        }
        score = std::clamp(
            score,
            -config_.state_log_multiplier_bound,
            config_.state_log_multiplier_bound);
        raw_multiplier[type] = std::exp(score);
        weighted_sum += intensities[type] * raw_multiplier[type];
    }
    if (!(weighted_sum > 0.0) || !std::isfinite(weighted_sum)) {
        throw std::runtime_error(
            "background Hawkes state normalization has invalid weighted hazard");
    }
    const double normalization = total_hazard / weighted_sum;
    for (std::size_t type = 0; type < intensities.size(); ++type) {
        intensities[type] *= raw_multiplier[type] * normalization;
    }
    return intensities;
}

BackgroundHawkesVector BackgroundHawkesStream::baseline_intensities(
    std::int64_t time_ns,
    const MarketState& state,
    double liquidity_removal_log_score) const {
    const double time_seconds = static_cast<double>(time_ns)
        / nanoseconds_per_second;
    return apply_state_type_response(
        baseline_intensities_at(
            time_seconds,
            stochastic_baseline_multiplier_at(time_seconds)),
        state,
        liquidity_removal_log_score);
}

BackgroundHawkesVector BackgroundHawkesStream::current_type_intensities(
    const MarketState& state,
    double liquidity_removal_log_score) const {
    BackgroundHawkesVector intensities = baseline_intensities_at(
        time_seconds_, current_stochastic_baseline_multiplier());
    for (std::size_t index = 0; index < intensities.size(); ++index) {
        intensities[index] = std::max(
            0.0,
            intensities[index] + fast_excitation_[index]
                + slow_excitation_[index]);
    }
    return apply_state_type_response(
        intensities, state, liquidity_removal_log_score);
}

void BackgroundHawkesStream::decay_to(double next_time_seconds) {
    if (next_time_seconds < time_seconds_) {
        throw std::logic_error("background Hawkes clock cannot move backwards");
    }
    const double elapsed = next_time_seconds - time_seconds_;
    const double fast_decay = std::exp(-config_.beta * elapsed);
    const double slow_decay = std::exp(-config_.slow_beta * elapsed);
    for (std::size_t index = 0; index < fast_excitation_.size(); ++index) {
        fast_excitation_[index] *= fast_decay;
        slow_excitation_[index] *= slow_decay;
    }
    time_seconds_ = next_time_seconds;
}

void BackgroundHawkesStream::cache_next_time() {
    while (!pending_time_ns_.has_value()) {
        const BackgroundHawkesVector intraday = intraday_factor_at(time_seconds_);
        const double activity_multiplier =
            current_stochastic_baseline_multiplier();
        BackgroundHawkesVector upper{};
        double upper_sum = 0.0;
        for (std::size_t index = 0; index < upper.size(); ++index) {
            double immigration = config_.activity_scale * config_.mu[index]
                * intraday[index];
            if (stochastic_baseline_enabled_) {
                immigration *= activity_multiplier;
            }
            upper[index] = std::max(
                0.0,
                immigration + fast_excitation_[index]
                    + slow_excitation_[index]);
            upper_sum += upper[index];
        }
        const double intraday_boundary =
            next_intraday_boundary_seconds(time_seconds_);
        const double activity_boundary =
            next_stochastic_baseline_boundary_seconds(time_seconds_);
        const double boundary = std::min(
            intraday_boundary, activity_boundary);
        if (upper_sum <= minimum_intensity) {
            // A new activity bin cannot turn a structurally zero immigration
            // vector positive.  Only an intraday-factor boundary can do so.
            if (std::isfinite(intraday_boundary)) {
                decay_to(intraday_boundary);
                advance_stochastic_baseline_to(intraday_boundary);
                continue;
            }
            pending_time_ns_ = std::numeric_limits<std::int64_t>::max();
            return;
        }

        const double wait_seconds = -std::log(clock_rng_.uniform01()) / upper_sum;
        const double candidate_time = time_seconds_ + wait_seconds;
        if (candidate_time >= boundary) {
            decay_to(boundary);
            advance_stochastic_baseline_to(boundary);
            continue;
        }
        decay_to(candidate_time);

        const BackgroundHawkesVector candidate_intraday =
            intraday_factor_at(time_seconds_);
        double candidate_sum = 0.0;
        for (std::size_t index = 0; index < upper.size(); ++index) {
            double immigration = config_.activity_scale * config_.mu[index]
                * candidate_intraday[index];
            if (stochastic_baseline_enabled_) {
                // No activity boundary was crossed, so the same constant
                // multiplier used by the dominating rate remains valid.
                immigration *= activity_multiplier;
            }
            candidate_sum += std::max(
                0.0,
                immigration + fast_excitation_[index]
                    + slow_excitation_[index]);
        }
        if (clock_rng_.uniform01() * upper_sum > candidate_sum) continue;

        const double time_ns = time_seconds_ * nanoseconds_per_second;
        pending_time_ns_ = time_ns >= static_cast<double>(
                std::numeric_limits<std::int64_t>::max())
            ? std::numeric_limits<std::int64_t>::max()
            : static_cast<std::int64_t>(std::llround(time_ns));
    }
}

std::int64_t BackgroundHawkesStream::peek_time_ns() {
    cache_next_time();
    return *pending_time_ns_;
}

HawkesEvent BackgroundHawkesStream::pop(
    const MarketState& state,
    double liquidity_removal_log_score) {
    const std::int64_t timestamp = peek_time_ns();
    if (timestamp == std::numeric_limits<std::int64_t>::max()) {
        return HawkesEvent{timestamp, HawkesEventType::LimitBuy};
    }

    // The fitted queue state explains the conditional event-type mixture, so
    // apply it to the complete pre-type vector rather than immigration alone.
    // Its normalization preserves the accepted-event hazard already used by
    // cache_next_time(), leaving the event-time clock state-independent.
    BackgroundHawkesVector intensity = current_type_intensities(
        state, liquidity_removal_log_score);
    double intensity_sum = 0.0;
    for (const double value : intensity) intensity_sum += value;
    if (!(intensity_sum > minimum_intensity) || !std::isfinite(intensity_sum)) {
        throw std::runtime_error(
            "background Hawkes accepted time has no valid type intensity");
    }

    double draw = clock_rng_.uniform01() * intensity_sum;
    std::size_t event_index = 0;
    for (; event_index + 1U < intensity.size(); ++event_index) {
        draw -= intensity[event_index];
        if (draw <= 0.0) break;
    }
    for (std::size_t index = 0; index < fast_excitation_.size(); ++index) {
        fast_excitation_[index] += config_.alpha[index][event_index];
        slow_excitation_[index] += config_.slow_alpha[index][event_index];
    }
    pending_time_ns_.reset();
    ++accepted_events_;
    return HawkesEvent{
        timestamp, static_cast<HawkesEventType>(event_index)};
}

BackgroundHawkesAgent::BackgroundHawkesAgent(const BackgroundHawkesConfig& config)
    : config_(config), rng_(config.seed + 911382323ULL) {
    validate_hawkes_clock_config(config_);
    if (config_.target_spread_ticks <= 0) {
        throw std::invalid_argument("background target spread must be positive");
    }
    if (!std::isfinite(config_.quote_improvement_probability)
        || config_.quote_improvement_probability < 0.0
        || config_.quote_improvement_probability > 1.0) {
        throw std::invalid_argument(
            "background quote-improvement probability must lie in [0,1]");
    }
    const bool queue_targets_disabled = config_.target_mean_bid_depth == 0.0
        && config_.target_mean_ask_depth == 0.0;
    const bool queue_targets_valid = std::isfinite(config_.target_mean_bid_depth)
        && std::isfinite(config_.target_mean_ask_depth)
        && config_.target_mean_bid_depth > 0.0
        && config_.target_mean_ask_depth > 0.0;
    if (!queue_targets_disabled && !queue_targets_valid) {
        throw std::invalid_argument(
            "queue-reactive cancellation requires two positive mean depths");
    }
    // Odd lots are part of the empirical ITCH mark distributions.  The
    // fallback lower bound also acts as the CSV loader's acceptance floor, so
    // using the legacy value 25 silently discarded valid 1--24 share marks.
    limit_buy_quantity_.set_fallback(1, 200);
    limit_sell_quantity_.set_fallback(1, 200);
    market_buy_quantity_.set_fallback(1, 700);
    market_sell_quantity_.set_fallback(1, 700);
    cancel_bid_quantity_.set_fallback(1, 500);
    cancel_ask_quantity_.set_fallback(1, 500);
    limit_buy_distance_.set_fallback(0, 5);
    limit_sell_distance_.set_fallback(0, 5);
    cancel_bid_distance_.set_fallback(0, 5);
    cancel_ask_distance_.set_fallback(0, 5);
    limit_buy_improvement_.set_fallback(1, 1);
    limit_sell_improvement_.set_fallback(1, 1);

    auto load_required = [](EmpiricalDistribution& distribution,
                            const std::string& path,
                            const char* column) {
        const std::string resolved = resolve_data_file(path);
        if (!distribution.load_from_csv(resolved, column)) {
            throw std::runtime_error(
                "cannot load required empirical mark distribution: " + resolved);
        }
    };
    load_required(limit_buy_quantity_, config_.limit_buy_quantity_file, "quantity");
    load_required(limit_sell_quantity_, config_.limit_sell_quantity_file, "quantity");
    load_required(market_buy_quantity_, config_.market_buy_quantity_file, "quantity");
    load_required(market_sell_quantity_, config_.market_sell_quantity_file, "quantity");
    load_required(cancel_bid_quantity_, config_.cancel_bid_quantity_file, "quantity");
    load_required(cancel_ask_quantity_, config_.cancel_ask_quantity_file, "quantity");
    load_required(limit_buy_distance_, config_.limit_buy_distance_file, "distance_ticks");
    load_required(limit_sell_distance_, config_.limit_sell_distance_file, "distance_ticks");
    load_required(cancel_bid_distance_, config_.cancel_bid_distance_file, "distance_ticks");
    load_required(cancel_ask_distance_, config_.cancel_ask_distance_file, "distance_ticks");
    const bool has_buy_improvement = !config_.limit_buy_improvement_file.empty();
    const bool has_sell_improvement = !config_.limit_sell_improvement_file.empty();
    if (has_buy_improvement != has_sell_improvement) {
        throw std::invalid_argument(
            "background inside-improvement files must be supplied together");
    }
    if (has_buy_improvement) {
        load_required(
            limit_buy_improvement_, config_.limit_buy_improvement_file,
            "improvement_ticks");
        load_required(
            limit_sell_improvement_, config_.limit_sell_improvement_file,
            "improvement_ticks");
    }
}

std::vector<HawkesEvent> BackgroundHawkesAgent::simulate(std::int64_t start_time_ns,
                                                         std::int64_t end_time_ns) {
    if (end_time_ns <= start_time_ns) return {};

    const bool has_slow_excitation = std::any_of(
        config_.slow_alpha.begin(), config_.slow_alpha.end(),
        [](const BackgroundHawkesVector& row) {
            return std::any_of(row.begin(), row.end(),
                               [](double value) { return value != 0.0; });
        });
    if (has_slow_excitation || !config_.intraday_factors.empty()
        || has_state_response(config_)) {
        throw std::logic_error(
            "BackgroundHawkesAgent::simulate does not accept queue-reactive "
            "clock configurations; use BackgroundHawkesStream so book state "
            "is supplied at each event");
    }

    std::vector<HawkesEvent> events;
    const double duration_seconds = static_cast<double>(end_time_ns - start_time_ns) / 1e9;
    events.reserve(static_cast<std::size_t>(std::max(0.0, duration_seconds * 100.0)));

    double time_seconds = static_cast<double>(start_time_ns) / 1e9;
    const double end_seconds = static_cast<double>(end_time_ns) / 1e9;
    std::array<double, 6> excitation{};

    while (time_seconds < end_seconds) {
        std::array<double, 6> upper{};
        double upper_sum = 0.0;
        for (std::size_t i = 0; i < upper.size(); ++i) {
            upper[i] = std::max(0.0, config_.activity_scale * config_.mu[i] + excitation[i]);
            upper_sum += upper[i];
        }
        if (upper_sum <= 1e-12) break;

        const double dt = -std::log(rng_.uniform01()) / upper_sum;
        time_seconds += dt;
        if (time_seconds >= end_seconds) break;

        const double decay = std::exp(-std::max(1e-6, config_.beta) * dt);
        for (double& value : excitation) value *= decay;

        std::array<double, 6> candidate{};
        double candidate_sum = 0.0;
        for (std::size_t i = 0; i < candidate.size(); ++i) {
            candidate[i] = std::max(0.0, config_.activity_scale * config_.mu[i] + excitation[i]);
            candidate_sum += candidate[i];
        }
        if (rng_.uniform01() * upper_sum > candidate_sum) continue;

        double draw = rng_.uniform01() * candidate_sum;
        std::size_t event_index = 0;
        for (; event_index + 1 < candidate.size(); ++event_index) {
            draw -= candidate[event_index];
            if (draw <= 0.0) break;
        }

        const auto time_ns = static_cast<std::int64_t>(std::llround(time_seconds * 1e9));
        events.push_back(HawkesEvent{time_ns, static_cast<HawkesEventType>(event_index)});
        for (std::size_t i = 0; i < excitation.size(); ++i) {
            excitation[i] += std::max(0.0, config_.alpha[i][event_index]);
        }
    }
    return events;
}

OrderMessage BackgroundHawkesAgent::make_order(const HawkesEvent& event,
                                                const MarketState& state,
                                                std::uint64_t sequence) {
    OrderMessage message;
    message.generated_time_ns = event.time_ns;
    message.arrival_time_ns = event.time_ns;
    message.sequence = sequence;
    message.tie_breaker = rng_.next_u64();
    message.source_rank = 0;
    message.owner_id = 0;
    message.agent_kind = AgentKind::Background;

    const int tick = std::max(1, config_.tick_size);
    auto queue_reactive_quantity = [](int sampled,
                                      int current_depth,
                                      double target_depth) {
        if (sampled <= 0 || !(target_depth > 0.0)) return sampled;
        constexpr double maximum_multiplier = 4.0;
        const double multiplier = std::clamp(
            static_cast<double>(std::max(0, current_depth)) / target_depth,
            0.0, maximum_multiplier);
        const double scaled = std::round(static_cast<double>(sampled) * multiplier);
        return static_cast<int>(std::clamp(
            scaled, 0.0, static_cast<double>(std::numeric_limits<int>::max())));
    };
    switch (event.type) {
        case HawkesEventType::LimitBuy: {
            message.action = OrderAction::Limit;
            message.side = Side::Buy;
            message.quantity = limit_buy_quantity_.sample(rng_);
            message.distance_ticks = std::max(
                0, limit_buy_distance_.sample(rng_));
            // The compact artifacts collapse at-best and inside additions
            // into the same distance-zero mark.  Their identifiable reduced
            // form is therefore a shared label probability conditional on a
            // sampled zero mark.  Eligibility is the extractor's fixed
            // geometric condition spread >= two ticks; the rounded mean
            // spread target is deliberately not part of this definition.
            const bool improvement_eligible = state.best_bid_ticks > 0
                && state.best_ask_ticks > state.best_bid_ticks
                && static_cast<std::int64_t>(state.best_ask_ticks)
                    - state.best_bid_ticks
                    >= 2LL * tick;
            const bool improve = message.distance_ticks == 0
                && improvement_eligible
                && (config_.quote_improvement_probability >= 1.0
                    || (config_.quote_improvement_probability > 0.0
                        && rng_.uniform01()
                            < config_.quote_improvement_probability));
            int price = 0;
            if (improve) {
                const std::int64_t gap_ticks = (
                    static_cast<std::int64_t>(state.best_ask_ticks)
                    - state.best_bid_ticks) / tick;
                const int improvement_ticks = limit_buy_improvement_.loaded()
                    ? std::clamp(
                        limit_buy_improvement_.sample(rng_), 1,
                        static_cast<int>(std::max<std::int64_t>(1, gap_ticks - 1)))
                    : 1;
                price = state.best_bid_ticks + improvement_ticks * tick;
            } else if (state.best_bid_ticks > 0) {
                const std::int64_t raw = static_cast<std::int64_t>(
                    state.best_bid_ticks)
                    - static_cast<std::int64_t>(message.distance_ticks) * tick;
                price = static_cast<int>(std::clamp<std::int64_t>(
                    raw, std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::max()));
            } else if (state.best_ask_ticks > tick) {
                // Reconstruct only the missing side.  A one-sided book has a
                // zero midpoint, so anchoring to mid_price_ticks would create
                // a negative, permanently rejected bid.
                price = state.best_ask_ticks - tick;
            } else if (state.best_ask_ticks <= 0) {
                const double reference = state.mid_price_ticks > 0.0
                    ? state.mid_price_ticks : state.fundamental_value_ticks;
                const double bounded = std::clamp(
                    reference - static_cast<double>(tick),
                    1.0,
                    static_cast<double>(std::numeric_limits<int>::max()));
                price = static_cast<int>(std::llround(bounded));
            }
            message.price_ticks = price;
            break;
        }
        case HawkesEventType::LimitSell: {
            message.action = OrderAction::Limit;
            message.side = Side::Sell;
            message.quantity = limit_sell_quantity_.sample(rng_);
            message.distance_ticks = std::max(
                0, limit_sell_distance_.sample(rng_));
            const bool improvement_eligible = state.best_bid_ticks > 0
                && state.best_ask_ticks > state.best_bid_ticks
                && static_cast<std::int64_t>(state.best_ask_ticks)
                    - state.best_bid_ticks
                    >= 2LL * tick;
            const bool improve = message.distance_ticks == 0
                && improvement_eligible
                && (config_.quote_improvement_probability >= 1.0
                    || (config_.quote_improvement_probability > 0.0
                        && rng_.uniform01()
                            < config_.quote_improvement_probability));
            int price = 0;
            if (improve) {
                const std::int64_t gap_ticks = (
                    static_cast<std::int64_t>(state.best_ask_ticks)
                    - state.best_bid_ticks) / tick;
                const int improvement_ticks = limit_sell_improvement_.loaded()
                    ? std::clamp(
                        limit_sell_improvement_.sample(rng_), 1,
                        static_cast<int>(std::max<std::int64_t>(1, gap_ticks - 1)))
                    : 1;
                price = state.best_ask_ticks - improvement_ticks * tick;
            } else if (state.best_ask_ticks > 0) {
                const std::int64_t raw = static_cast<std::int64_t>(
                    state.best_ask_ticks)
                    + static_cast<std::int64_t>(message.distance_ticks) * tick;
                price = static_cast<int>(std::min<std::int64_t>(
                    raw, std::numeric_limits<int>::max()));
            } else if (state.best_bid_ticks > 0) {
                // Preserve the extant bid and place the missing ask one tick
                // above it; never reset a one-sided book to a zero midpoint.
                const std::int64_t raw = static_cast<std::int64_t>(
                    state.best_bid_ticks) + tick;
                price = static_cast<int>(std::min<std::int64_t>(
                    raw, std::numeric_limits<int>::max()));
            } else {
                const double reference = state.mid_price_ticks > 0.0
                    ? state.mid_price_ticks : state.fundamental_value_ticks;
                const double bounded = std::clamp(
                    reference + static_cast<double>(tick),
                    1.0,
                    static_cast<double>(std::numeric_limits<int>::max()));
                price = static_cast<int>(std::llround(bounded));
            }
            message.price_ticks = price;
            break;
        }
        case HawkesEventType::MarketBuy:
            message.action = OrderAction::Market;
            message.side = Side::Buy;
            message.quantity = market_buy_quantity_.sample(rng_);
            break;
        case HawkesEventType::MarketSell:
            message.action = OrderAction::Market;
            message.side = Side::Sell;
            message.quantity = market_sell_quantity_.sample(rng_);
            break;
        case HawkesEventType::CancelBid: {
            message.action = OrderAction::CancelAtDistance;
            message.side = Side::Buy;
            message.quantity = cancel_bid_quantity_.sample(rng_);
            message.distance_ticks = std::max(0, cancel_bid_distance_.sample(rng_));
            // Cancellation flow represents turnover of anonymous displayed
            // liquidity throughout the retained reduced band.  Apply the same
            // depth response to every retained mark so depletion at one level
            // cannot leave positive-distance flow removing the final side
            // reserve at full size.  Strategic-maker depth remains excluded
            // from this background-only response even when its quote improves
            // ahead of the leading anonymous level.
            if (config_.cancellation_quantity_depth_scaling
                && message.distance_ticks < reduced_background_depth_levels) {
                message.quantity = queue_reactive_quantity(
                    message.quantity, state.background_best_bid_depth,
                    config_.target_mean_bid_depth);
            }
            break;
        }
        case HawkesEventType::CancelAsk: {
            message.action = OrderAction::CancelAtDistance;
            message.side = Side::Sell;
            message.quantity = cancel_ask_quantity_.sample(rng_);
            message.distance_ticks = std::max(0, cancel_ask_distance_.sample(rng_));
            if (config_.cancellation_quantity_depth_scaling
                && message.distance_ticks < reduced_background_depth_levels) {
                message.quantity = queue_reactive_quantity(
                    message.quantity, state.background_best_ask_depth,
                    config_.target_mean_ask_depth);
            }
            break;
        }
    }
    return message;
}

} // namespace dlob
