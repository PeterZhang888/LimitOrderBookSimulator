#pragma once

#include "simulation/MultiAssetTypes.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace dlob::detail {

// Stateless normal innovation keyed by the logical symbol stream, model seed,
// and decision index.  It is independent of MPI ownership and traversal order.
[[nodiscard]] inline double deterministic_fundamental_normal(
    StableEntityId symbol_stream,
    std::uint64_t model_seed,
    std::uint64_t decision_index) noexcept {
    constexpr double denominator = 9'007'199'254'740'992.0; // 2^53
    auto open_uniform = [](std::uint64_t bits) noexcept {
        const std::uint64_t mantissa = bits >> 11U;
        return (static_cast<double>(mantissa) + 0.5) / denominator;
    };
    const StableEntityId entity = symbol_stream ^ model_seed ^ 0x6a09e667f3bcc909ULL;
    const double first = open_uniform(stable_sequence(entity, decision_index, 0U));
    const double second = open_uniform(stable_sequence(entity, decision_index, 1U));
    return std::sqrt(-2.0 * std::log(first))
        * std::cos(2.0 * std::numbers::pi * second);
}

[[nodiscard]] inline double deterministic_fundamental_uniform(
    StableEntityId symbol_stream,
    std::uint64_t model_seed,
    std::uint64_t decision_index,
    std::uint32_t draw_index = 2U) noexcept {
    constexpr double denominator = 9'007'199'254'740'992.0; // 2^53
    const StableEntityId entity = symbol_stream ^ model_seed ^ 0xbb67ae8584caa73bULL;
    const std::uint64_t mantissa = stable_sequence(
        entity, decision_index, draw_index) >> 11U;
    return (static_cast<double>(mantissa) + 0.5) / denominator;
}

// Keep volatility-state innovations on a domain separate from the price-news
// uniforms above.  Stateless draws remain functions only of the persistent
// symbol identity, model seed and logical decision boundary.
inline constexpr StableEntityId fundamental_log_variance_domain =
    0x3c6e'f372'fe94'f82bULL;

[[nodiscard]] inline double initial_fundamental_log_variance(
    double persistence,
    double stationary_std,
    StableEntityId symbol_stream,
    std::uint64_t model_seed) {
    if (!std::isfinite(persistence) || persistence < 0.0
        || persistence >= 1.0 || !std::isfinite(stationary_std)
        || stationary_std < 0.0) {
        throw std::invalid_argument(
            "invalid deterministic fundamental log-variance configuration");
    }
    if (stationary_std == 0.0) return 0.0;
    const double initial = stationary_std * deterministic_fundamental_normal(
        symbol_stream ^ fundamental_log_variance_domain,
        model_seed, 0U);
    if (!std::isfinite(initial)) {
        throw std::overflow_error(
            "initial fundamental log variance is not finite");
    }
    return initial;
}

// One stationary AR(1) step for log variance.  stationary_std denotes the
// standard deviation of the state rather than of its innovation, hence the
// sqrt(1-phi^2) factor.  Starting from initial_fundamental_log_variance keeps
// every boundary marginally N(0, stationary_std^2).
[[nodiscard]] inline double advance_fundamental_log_variance(
    double current_log_variance,
    double persistence,
    double stationary_std,
    StableEntityId symbol_stream,
    std::uint64_t model_seed,
    std::uint64_t decision_index) {
    if (!std::isfinite(current_log_variance)
        || !std::isfinite(persistence) || persistence < 0.0
        || persistence >= 1.0 || !std::isfinite(stationary_std)
        || stationary_std < 0.0 || decision_index == 0U) {
        throw std::invalid_argument(
            "invalid deterministic fundamental log-variance step");
    }
    if (stationary_std == 0.0) return 0.0;
    const double innovation_scale = stationary_std
        * std::sqrt((1.0 - persistence) * (1.0 + persistence));
    const double innovation = deterministic_fundamental_normal(
        symbol_stream ^ fundamental_log_variance_domain,
        model_seed, decision_index);
    const double updated = persistence * current_log_variance
        + innovation_scale * innovation;
    if (!std::isfinite(updated)) {
        throw std::overflow_error(
            "fundamental log-variance process is not finite");
    }
    return updated;
}

// If H ~ N(0, s^2), M = exp(H/2 - s^2/4) satisfies E[M^2] = 1.
// Therefore stochastic volatility changes persistence and higher moments
// without changing the configured unconditional return-variance target.
[[nodiscard]] inline double fundamental_volatility_multiplier(
    double log_variance,
    double stationary_std) {
    if (!std::isfinite(log_variance) || !std::isfinite(stationary_std)
        || stationary_std < 0.0) {
        throw std::invalid_argument(
            "invalid deterministic fundamental volatility multiplier");
    }
    if (stationary_std == 0.0) return 1.0;
    const double multiplier = std::exp(
        0.5 * log_variance - 0.25 * stationary_std * stationary_std);
    if (!(multiplier > 0.0) || !std::isfinite(multiplier)) {
        throw std::overflow_error(
            "fundamental volatility multiplier is not finite and positive");
    }
    return multiplier;
}

struct FundamentalMomentMatchedLaw {
    double rare_probability = 0.0;
    double common_magnitude = 0.0;
    double rare_magnitude = 0.0;
};

// Parameters of a symmetric two-magnitude law with exactly zero mean, unit
// variance and the requested kurtosis.  For k=1 both magnitudes are one.  For
// k>1, a rare large magnitude supplies the fourth moment while the common
// magnitude stays strictly positive; therefore news timing remains governed
// only by the separately calibrated move probability.
[[nodiscard]] inline FundamentalMomentMatchedLaw
fundamental_moment_matched_law(double conditional_kurtosis) {
    if (!std::isfinite(conditional_kurtosis)
        || conditional_kurtosis < 1.0) {
        throw std::invalid_argument(
            "conditional fundamental kurtosis must be at least one");
    }
    const double rare_probability = 0.5 / conditional_kurtosis;
    if (!(rare_probability > 0.0) || rare_probability > 0.5
        || !std::isfinite(rare_probability)) {
        throw std::overflow_error(
            "conditional fundamental kurtosis is too large for the "
            "moment-matched innovation law");
    }
    const double spread = std::sqrt(
        rare_probability / (1.0 - rare_probability)
        * (conditional_kurtosis - 1.0));
    const double common_squared = std::max(0.0, 1.0 - spread);
    const double rare_squared = 1.0 + std::sqrt(
        (1.0 - rare_probability) / rare_probability
        * (conditional_kurtosis - 1.0));
    const FundamentalMomentMatchedLaw law{
        rare_probability,
        std::sqrt(common_squared),
        std::sqrt(rare_squared),
    };
    if (!std::isfinite(law.common_magnitude)
        || !std::isfinite(law.rare_magnitude)) {
        throw std::overflow_error(
            "moment-matched fundamental innovation magnitudes are not finite");
    }
    return law;
}

// Exact log moment-generating function of the symmetric two-magnitude law.
// This replaces the Gaussian sigma^2/2 correction when innovations are
// heavy-tailed.  The log-cosh/log-sum-exp form remains finite for arguments
// whose moment-generating function would overflow in ordinary arithmetic.
[[nodiscard]] inline double fundamental_moment_matched_log_mgf(
    double conditional_kurtosis,
    double sigma) {
    if (!std::isfinite(sigma)) {
        throw std::invalid_argument(
            "fundamental innovation scale must be finite");
    }
    if (sigma == 0.0) return 0.0;
    const FundamentalMomentMatchedLaw law =
        fundamental_moment_matched_law(conditional_kurtosis);
    const auto log_cosh = [](double value) {
        const double magnitude = std::abs(value);
        if (!std::isfinite(magnitude)) {
            throw std::overflow_error(
                "fundamental innovation log-MGF argument is not finite");
        }
        return magnitude + std::log1p(std::exp(-2.0 * magnitude))
            - std::log(2.0);
    };
    const double common_term = std::log1p(-law.rare_probability)
        + log_cosh(sigma * law.common_magnitude);
    const double rare_term = std::log(law.rare_probability)
        + log_cosh(sigma * law.rare_magnitude);
    const double maximum = std::max(common_term, rare_term);
    const double result = maximum + std::log(
        std::exp(common_term - maximum) + std::exp(rare_term - maximum));
    if (!std::isfinite(result)) {
        throw std::overflow_error(
            "fundamental innovation log-MGF is not finite");
    }
    return result;
}

// Draw from the moment-matched law above with a stateless sign and magnitude.
[[nodiscard]] inline double deterministic_moment_matched_innovation(
    double conditional_kurtosis,
    StableEntityId symbol_stream,
    std::uint64_t model_seed,
    std::uint64_t decision_index) {
    const FundamentalMomentMatchedLaw law =
        fundamental_moment_matched_law(conditional_kurtosis);
    const bool rare = deterministic_fundamental_uniform(
        symbol_stream, model_seed, decision_index, 3U)
        < law.rare_probability;
    const double magnitude = rare
        ? law.rare_magnitude : law.common_magnitude;
    const double sign = deterministic_fundamental_uniform(
        symbol_stream, model_seed, decision_index, 4U) < 0.5 ? -1.0 : 1.0;
    return sign * magnitude;
}

// One step of a multiplicative martingale reference-value process.  Volatility
// is expressed in bps/sqrt(second), matching sqrt of the one-second empirical
// log-return variance after multiplication by 10,000.  The exact log-MGF of
// the configured innovation law, rather than a Gaussian approximation, is the
// conditional martingale correction.
[[nodiscard]] inline double advance_fundamental_value(
    double current_value_ticks,
    double volatility_bps_sqrt_second,
    double move_probability_per_second,
    double conditional_kurtosis,
    double elapsed_seconds,
    StableEntityId symbol_stream,
    std::uint64_t model_seed,
    std::uint64_t decision_index) {
    if (!(current_value_ticks > 0.0) || !std::isfinite(current_value_ticks)
        || !(volatility_bps_sqrt_second >= 0.0)
        || !std::isfinite(volatility_bps_sqrt_second)
        || !(move_probability_per_second >= 0.0)
        || move_probability_per_second > 1.0
        || !std::isfinite(move_probability_per_second)
        || conditional_kurtosis < 1.0
        || !std::isfinite(conditional_kurtosis)
        || !(elapsed_seconds > 0.0) || !std::isfinite(elapsed_seconds)
        || decision_index == 0U) {
        throw std::invalid_argument("invalid deterministic fundamental step");
    }
    if (volatility_bps_sqrt_second == 0.0
        || move_probability_per_second == 0.0) {
        return current_value_ticks;
    }
    const double move_probability = 1.0 - std::pow(
        1.0 - move_probability_per_second, elapsed_seconds);
    if (deterministic_fundamental_uniform(
            symbol_stream, model_seed, decision_index) >= move_probability) {
        return current_value_ticks;
    }
    // Conditional jump variance is enlarged by 1/p so the unconditional
    // variance remains the pooled one-second training variance.
    const double sigma = volatility_bps_sqrt_second
        * std::sqrt(elapsed_seconds / move_probability) / 10'000.0;
    const double innovation = deterministic_moment_matched_innovation(
        conditional_kurtosis, symbol_stream, model_seed, decision_index);
    const double martingale_correction = fundamental_moment_matched_log_mgf(
        conditional_kurtosis, sigma);
    const double updated = current_value_ticks
        * std::exp(sigma * innovation - martingale_correction);
    if (!std::isfinite(updated) || updated < 1.0
        || updated > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(
            "deterministic fundamental process left the valid price range");
    }
    return updated;
}

} // namespace dlob::detail
