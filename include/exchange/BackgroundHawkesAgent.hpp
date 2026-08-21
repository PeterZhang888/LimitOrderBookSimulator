#pragma once

#include "common/DistributedTypes.hpp"
#include "common/EmpiricalDistribution.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dlob {

inline constexpr double fixed_background_hawkes_activity_scale = 0.30;
inline constexpr std::size_t background_hawkes_event_type_count = 6U;
inline constexpr std::size_t background_hawkes_state_feature_count = 4U;

using BackgroundHawkesVector =
    std::array<double, background_hawkes_event_type_count>;
using BackgroundHawkesMatrix = std::array<
    BackgroundHawkesVector, background_hawkes_event_type_count>;
using BackgroundStateCoefficientMatrix = std::array<
    std::array<double, background_hawkes_state_feature_count>,
    background_hawkes_event_type_count>;

struct BackgroundHawkesConfig {
    // The rate files are derived with the conventional 0.30 baseline.  This
    // remains an explicit runtime multiplier so the distributed calibration
    // can select a single market-wide activity adjustment without rewriting
    // the empirically derived mu values for every candidate.
    double activity_scale = fixed_background_hawkes_activity_scale;
    BackgroundHawkesVector mu{18.0, 18.0, 3.5, 3.5, 28.0, 28.0};
    // alpha/beta is the legacy fast exponential kernel.  The optional slow
    // kernel is additive, so the integrated branching matrix is
    // alpha / beta + slow_alpha / slow_beta.  A zero slow matrix is exactly
    // the legacy one-timescale process.
    BackgroundHawkesMatrix alpha{};
    double beta = 10.0;
    BackgroundHawkesMatrix slow_alpha{};
    double slow_beta = 1.0;

    // When enabled, construction verifies the latent linear-Hawkes stationary
    // equation
    //
    // target = activity_scale * mu
    //        + (alpha / beta + slow_alpha / slow_beta) * target.
    //
    // This fail-closed audit prevents a fitted excitation matrix from being
    // combined with stale immigration rates.  It does not prove that the
    // state-responsive realized marginal type rates equal target; those are
    // assessed by simulation.  It is disabled for legacy CSV configurations,
    // which do not carry the target vector at runtime.
    bool validate_stationary_target = false;
    BackgroundHawkesVector stationary_target_rates{};

    // Optional piecewise-constant event-type seasonality.  Each row is one
    // time bin and each column follows HawkesEventType order.  Every event
    // type must have arithmetic mean one across bins, preserving the
    // stationary rate represented by mu.  Empty means constant factor one.
    std::vector<BackgroundHawkesVector> intraday_factors;
    std::int64_t intraday_origin_ns = 0;
    std::int64_t intraday_bin_width_ns = 300'000'000'000LL;

    // Optional Cox--Hawkes baseline modulation. Within each fixed-width bin,
    // immigration is multiplied by exp(clamp(H_k) - log_normalizer), where
    // H_k is a stationary Gaussian AR(1). When normalization_bins is positive,
    // log_normalizer is the realized log-mean-exp over that fixed session.
    // This preserves the discrete session-average immigration multiplier; it
    // does not claim to fix the realized Hawkes event count pathwise.
    // Excitation remains additive and subcritical. A zero standard deviation
    // is the exact compatibility mode.
    double stochastic_baseline_persistence = 0.0;
    double stochastic_baseline_std = 0.0;
    std::int64_t stochastic_baseline_origin_ns = 0;
    std::int64_t stochastic_baseline_bin_width_ns = 1'000'000'000LL;
    std::uint64_t stochastic_baseline_normalization_bins = 0;
    double stochastic_baseline_standardized_bound = 3.0;

    // Log multipliers for four contemporaneous state features:
    // 0 fitted spread-bin representative (0 for one tick, log(2) wider);
    // 1/2 fitted displayed-depth-bin representatives (log(.35), 0,
    // log(2.5)); 3 fitted imbalance-bin representative in
    // {-0.8,-0.4,0,0.4,0.8}.  These encodings exactly match the
    // training extractor rather than applying a continuous function that was
    // never estimated.
    //
    // At pop(), raw type multipliers are normalized by the complete current
    // pre-type intensities: seasonal immigration plus fast and slow Hawkes
    // excitation.  State can therefore change the event-type mix but never
    // the total accepted-event hazard, which makes a time cached by
    // peek_time_ns() safe across intervening book-state changes.
    BackgroundStateCoefficientMatrix state_log_multiplier_coefficients{};
    double state_reference_bid_depth = 1.0;
    double state_reference_ask_depth = 1.0;
    double state_log_multiplier_bound = 4.0;
    int tick_size = 100;
    int target_spread_ticks = 1;
    // Compact ITCH artifacts identify one aggregate fraction of zero-distance
    // limit marks that represent inside-spread orders.  Apply this shared
    // maximum-symmetry label probability to buy and sell zero marks when the
    // pre-add spread geometrically admits an inside price.  It is not the
    // side/state-conditional probability P(inside | spread eligible), which
    // would require richer joint marks from a new extraction.
    double quote_improvement_probability = 0.05;
    // When positive, cancellation marks inside the retained ten-level band
    // receive a fixed exponent-one queue response: quantity is multiplied by
    // anonymous owner-zero BBO depth / the frozen five-day pooled training
    // mean, capped at four.  These are direct empirical targets, not free
    // search parameters.  Zero preserves the legacy exogenous-flow model.
    double target_mean_bid_depth = 0.0;
    double target_mean_ask_depth = 0.0;
    // Cancellation mark size may be scaled by current anonymous depth.  This
    // is separate from the queue policy's conditional event-type mixture:
    // one controls the arrival type, the other the depletion size.  Keeping
    // the bounded mark response supplies mean reversion in reduced books.
    bool cancellation_quantity_depth_scaling = true;
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
    // Optional side-specific distances of genuine inside-spread additions,
    // measured from the pre-event same-side best quote.  Both files must be
    // supplied together.  Empty preserves the historical one-tick mapping.
    std::string limit_buy_improvement_file;
    std::string limit_sell_improvement_file;

    BackgroundHawkesConfig();
};

// Incremental Hawkes event clock used by an event-driven simulator.  Waiting
// times depend on time-of-day and the two decaying excitation vectors, but not
// on book state.  Event type is deliberately selected only by pop(state), so
// a caller may safely cache the next timestamp while earlier external orders
// change the book.  The stream owns only clock/type RNG state; empirical mark
// sampling remains private to BackgroundHawkesAgent.
class BackgroundHawkesStream {
public:
    explicit BackgroundHawkesStream(
        const BackgroundHawkesConfig& config,
        std::int64_t start_time_ns = 0);

    [[nodiscard]] std::int64_t peek_time_ns();
    HawkesEvent pop(
        const MarketState& state,
        double liquidity_removal_log_score = 0.0);

    [[nodiscard]] std::uint64_t accepted_events() const noexcept {
        return accepted_events_;
    }

    // Exposed for deterministic diagnostics and calibration tests.  This
    // applies the fitted state response to seasonal immigration alone; runtime
    // pop() applies the same transformation after adding both excitation
    // vectors.  The sum is independent of state by construction.
    [[nodiscard]] BackgroundHawkesVector baseline_intensities(
        std::int64_t time_ns,
        const MarketState& state,
        double liquidity_removal_log_score = 0.0) const;

    // Complete type intensities at the stream's current Hawkes-clock state.
    // This diagnostic includes seasonal immigration and both decayed
    // excitation vectors, then applies the same hazard-preserving state
    // response used by pop().
    [[nodiscard]] BackgroundHawkesVector current_type_intensities(
        const MarketState& state,
        double liquidity_removal_log_score = 0.0) const;

private:
    [[nodiscard]] BackgroundHawkesVector baseline_intensities_at(
        double time_seconds,
        double stochastic_baseline_multiplier) const;
    [[nodiscard]] BackgroundHawkesVector apply_state_type_response(
        BackgroundHawkesVector intensities,
        const MarketState& state,
        double liquidity_removal_log_score) const;
    [[nodiscard]] BackgroundHawkesVector intraday_factor_at(
        double time_seconds) const;
    [[nodiscard]] double next_intraday_boundary_seconds(
        double time_seconds) const;
    [[nodiscard]] double stochastic_baseline_log_state_at_bin(
        std::uint64_t bin_index) const;
    [[nodiscard]] double stochastic_baseline_log_normalizer() const;
    [[nodiscard]] double bounded_stochastic_baseline_log_state(
        double state) const;
    [[nodiscard]] double stochastic_baseline_multiplier_at(
        double time_seconds) const;
    [[nodiscard]] double current_stochastic_baseline_multiplier() const;
    [[nodiscard]] double next_stochastic_baseline_boundary_seconds(
        double time_seconds) const;
    void advance_stochastic_baseline_to(double time_seconds);
    void decay_to(double next_time_seconds);
    void cache_next_time();

    BackgroundHawkesConfig config_;
    FastRng clock_rng_;
    bool state_response_enabled_ = false;
    bool stochastic_baseline_enabled_ = false;
    double time_seconds_ = 0.0;
    std::uint64_t stochastic_baseline_bin_index_ = 0;
    double stochastic_baseline_log_state_ = 0.0;
    double stochastic_baseline_log_normalizer_ = 0.0;
    BackgroundHawkesVector fast_excitation_{};
    BackgroundHawkesVector slow_excitation_{};
    std::optional<std::int64_t> pending_time_ns_;
    std::uint64_t accepted_events_ = 0;
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
    EmpiricalDistribution limit_buy_improvement_;
    EmpiricalDistribution limit_sell_improvement_;
};

} // namespace dlob
