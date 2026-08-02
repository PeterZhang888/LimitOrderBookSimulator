// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "exchange/BackgroundHawkesAgent.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using dlob::BackgroundHawkesConfig;
using dlob::FastRng;
using dlob::HawkesEvent;
using dlob::HawkesEventType;

class LegacyStreamingHawkesReference {
public:
    explicit LegacyStreamingHawkesReference(BackgroundHawkesConfig config)
        : config_(std::move(config)), rng_(config_.seed) {}

    HawkesEvent next() {
        while (true) {
            std::array<double, 6> upper{};
            double upper_sum = 0.0;
            for (std::size_t index = 0; index < upper.size(); ++index) {
                upper[index] = std::max(
                    0.0,
                    config_.activity_scale * config_.mu[index]
                        + excitation_[index]);
                upper_sum += upper[index];
            }
            assert(upper_sum > 1.0e-12);

            const double wait_seconds = -std::log(rng_.uniform01()) / upper_sum;
            time_seconds_ += wait_seconds;
            const double decay = std::exp(
                -std::max(1.0e-6, config_.beta) * wait_seconds);
            for (double& value : excitation_) value *= decay;

            std::array<double, 6> candidate{};
            double candidate_sum = 0.0;
            for (std::size_t index = 0; index < candidate.size(); ++index) {
                candidate[index] = std::max(
                    0.0,
                    config_.activity_scale * config_.mu[index]
                        + excitation_[index]);
                candidate_sum += candidate[index];
            }
            if (rng_.uniform01() * upper_sum > candidate_sum) continue;

            double draw = rng_.uniform01() * candidate_sum;
            std::size_t event_index = 0;
            for (; event_index + 1U < candidate.size(); ++event_index) {
                draw -= candidate[event_index];
                if (draw <= 0.0) break;
            }
            for (std::size_t index = 0; index < excitation_.size(); ++index) {
                excitation_[index] += std::max(
                    0.0, config_.alpha[index][event_index]);
            }
            const double time_ns = time_seconds_ * 1.0e9;
            const auto timestamp = time_ns >= static_cast<double>(
                    std::numeric_limits<std::int64_t>::max())
                ? std::numeric_limits<std::int64_t>::max()
                : static_cast<std::int64_t>(std::llround(time_ns));
            return HawkesEvent{
                timestamp, static_cast<HawkesEventType>(event_index)};
        }
    }

private:
    BackgroundHawkesConfig config_;
    FastRng rng_;
    double time_seconds_ = 0.0;
    std::array<double, 6> excitation_{};
};

bool throws_invalid_argument(const BackgroundHawkesConfig& config) {
    try {
        const dlob::BackgroundHawkesStream stream(config);
        (void)stream;
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

double sum(const dlob::BackgroundHawkesVector& values) {
    double total = 0.0;
    for (const double value : values) total += value;
    return total;
}

double lag_one_autocorrelation(const std::vector<double>& values) {
    assert(values.size() > 2U);
    double mean = 0.0;
    for (const double value : values) mean += value;
    mean /= static_cast<double>(values.size());
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double centered = values[index] - mean;
        denominator += centered * centered;
        if (index > 0U) {
            numerator += centered * (values[index - 1U] - mean);
        }
    }
    assert(denominator > 0.0);
    return numerator / denominator;
}

std::vector<double> one_second_event_counts(
    const BackgroundHawkesConfig& config,
    std::size_t seconds) {
    dlob::BackgroundHawkesStream stream(config);
    dlob::MarketState state;
    state.best_bid_ticks = 10'000;
    state.best_ask_ticks = 10'100;
    state.best_bid_depth = 100;
    state.best_ask_depth = 100;
    std::vector<double> counts(seconds, 0.0);
    const auto end_ns = static_cast<std::int64_t>(seconds)
        * 1'000'000'000LL;
    while (true) {
        const std::int64_t time_ns = stream.peek_time_ns();
        if (time_ns >= end_ns) break;
        const HawkesEvent event = stream.pop(state);
        const auto bin = static_cast<std::size_t>(
            event.time_ns / 1'000'000'000LL);
        assert(bin < counts.size());
        counts[bin] += 1.0;
    }
    return counts;
}

void set_zero_excitation(BackgroundHawkesConfig& config) {
    for (auto& row : config.alpha) row.fill(0.0);
    for (auto& row : config.slow_alpha) row.fill(0.0);
}

void set_mark_files(BackgroundHawkesConfig& config,
                    const std::filesystem::path& quantity,
                    const std::filesystem::path& distance) {
    config.limit_buy_quantity_file = quantity.string();
    config.limit_sell_quantity_file = quantity.string();
    config.market_buy_quantity_file = quantity.string();
    config.market_sell_quantity_file = quantity.string();
    config.cancel_bid_quantity_file = quantity.string();
    config.cancel_ask_quantity_file = quantity.string();
    config.limit_buy_distance_file = distance.string();
    config.limit_sell_distance_file = distance.string();
    config.cancel_bid_distance_file = distance.string();
    config.cancel_ask_distance_file = distance.string();
}

void assert_order_equal(const dlob::OrderMessage& left,
                        const dlob::OrderMessage& right) {
    assert(left.generated_time_ns == right.generated_time_ns);
    assert(left.arrival_time_ns == right.arrival_time_ns);
    assert(left.sequence == right.sequence);
    assert(left.tie_breaker == right.tie_breaker);
    assert(left.source_rank == right.source_rank);
    assert(left.owner_id == right.owner_id);
    assert(left.agent_kind == right.agent_kind);
    assert(left.action == right.action);
    assert(left.side == right.side);
    assert(left.quantity == right.quantity);
    assert(left.price_ticks == right.price_ticks);
    assert(left.distance_ticks == right.distance_ticks);
    assert(left.book_id == right.book_id);
}

} // namespace

int main() {
    using namespace dlob;

    // The new incremental clock is bit-exact with the former fragmented
    // one-timescale stream when every new option retains its legacy default.
    BackgroundHawkesConfig legacy;
    legacy.seed = 0x5a17ULL;
    LegacyStreamingHawkesReference reference(legacy);
    BackgroundHawkesStream stream(legacy);
    MarketState neutral_state;
    for (int index = 0; index < 1'000; ++index) {
        const HawkesEvent expected = reference.next();
        const std::int64_t first_peek = stream.peek_time_ns();
        assert(stream.peek_time_ns() == first_peek);
        const HawkesEvent actual = stream.pop(neutral_state);
        assert(actual.time_ns == expected.time_ns);
        assert(actual.type == expected.type);
    }
    assert(stream.accepted_events() == 1'000U);

    // A state response redistributes type-specific intensity but leaves its
    // total invariant.  Therefore an already cached time remains valid if an
    // external order changes the state before pop().
    BackgroundHawkesConfig reactive;
    set_zero_excitation(reactive);
    reactive.seed = 0x715ULL;
    reactive.tick_size = 100;
    reactive.target_spread_ticks = 1;
    reactive.state_reference_bid_depth = 100.0;
    reactive.state_reference_ask_depth = 100.0;
    reactive.state_log_multiplier_coefficients[0][0] = 1.0;
    reactive.state_log_multiplier_coefficients[1][0] = -0.5;
    reactive.state_log_multiplier_coefficients[4][1] = 0.75;
    reactive.state_log_multiplier_coefficients[5][2] = 0.75;

    MarketState narrow;
    narrow.best_bid_ticks = 10'000;
    narrow.best_ask_ticks = 10'100;
    narrow.background_best_bid_depth = 100;
    narrow.background_best_ask_depth = 100;
    narrow.best_bid_depth = 100;
    narrow.best_ask_depth = 100;
    MarketState wide = narrow;
    wide.best_ask_ticks = 11'000;
    wide.background_best_bid_depth = 400;
    wide.background_best_ask_depth = 25;
    wide.best_bid_depth = 400;
    wide.best_ask_depth = 25;

    BackgroundHawkesStream reactive_stream(reactive);
    const BackgroundHawkesVector narrow_rates =
        reactive_stream.baseline_intensities(0, narrow);
    const BackgroundHawkesVector wide_rates =
        reactive_stream.baseline_intensities(0, wide);
    assert(std::abs(sum(narrow_rates) - sum(wide_rates)) <= 1.0e-12);
    assert(wide_rates[0] > narrow_rates[0]);
    assert(wide_rates[4] / narrow_rates[4]
           > wide_rates[5] / narrow_rates[5]);

    // Retain the former hazard-conserving type-response API as a diagnostic.
    // It is deliberately no longer driven by persistent volatility in the
    // production simulator: conserving total intensity cannot create the
    // missing persistent active and quiet intervals.
    BackgroundHawkesConfig flow_regime;
    set_zero_excitation(flow_regime);
    BackgroundHawkesStream flow_stream(flow_regime);
    const BackgroundHawkesVector quiet_flow =
        flow_stream.baseline_intensities(0, narrow, -0.7);
    const BackgroundHawkesVector neutral_flow =
        flow_stream.baseline_intensities(0, narrow, 0.0);
    const BackgroundHawkesVector active_flow =
        flow_stream.baseline_intensities(0, narrow, 0.7);
    assert(std::abs(sum(quiet_flow) - sum(neutral_flow)) <= 1.0e-12);
    assert(std::abs(sum(active_flow) - sum(neutral_flow)) <= 1.0e-12);
    const auto removal_sum = [](const BackgroundHawkesVector& values) {
        return values[static_cast<std::size_t>(HawkesEventType::MarketBuy)]
            + values[static_cast<std::size_t>(HawkesEventType::MarketSell)]
            + values[static_cast<std::size_t>(HawkesEventType::CancelBid)]
            + values[static_cast<std::size_t>(HawkesEventType::CancelAsk)];
    };
    const auto provision_sum = [](const BackgroundHawkesVector& values) {
        return values[static_cast<std::size_t>(HawkesEventType::LimitBuy)]
            + values[static_cast<std::size_t>(HawkesEventType::LimitSell)];
    };
    const auto cancellation_sum = [](const BackgroundHawkesVector& values) {
        return values[static_cast<std::size_t>(HawkesEventType::CancelBid)]
            + values[static_cast<std::size_t>(HawkesEventType::CancelAsk)];
    };
    assert(removal_sum(quiet_flow) < removal_sum(neutral_flow));
    assert(removal_sum(active_flow) > removal_sum(neutral_flow));
    assert(cancellation_sum(quiet_flow) < cancellation_sum(neutral_flow));
    assert(cancellation_sum(active_flow) > cancellation_sum(neutral_flow));
    assert(provision_sum(quiet_flow) > provision_sum(neutral_flow));
    assert(provision_sum(active_flow) < provision_sum(neutral_flow));
    MarketState one_sided = narrow;
    one_sided.best_ask_ticks = 0;
    one_sided.best_ask_depth = 0;
    assert(flow_stream.baseline_intensities(0, one_sided, 0.7)
           == flow_stream.baseline_intensities(0, one_sided, 0.0));
    bool nonfinite_regime_rejected = false;
    try {
        static_cast<void>(flow_stream.baseline_intensities(
            0, narrow, std::numeric_limits<double>::quiet_NaN()));
    } catch (const std::invalid_argument&) {
        nonfinite_regime_rejected = true;
    }
    assert(nonfinite_regime_rejected);

    // A mean-normalized stochastic immigration baseline must change the event
    // clock, preserve its long-run mean approximately in a finite sample, and
    // produce clearly positive lag-one count persistence.  The zero-standard-
    // deviation control exercises exact compatibility mode.
    BackgroundHawkesConfig constant_clock;
    set_zero_excitation(constant_clock);
    constant_clock.seed = 0x31c0ffeeULL;
    constant_clock.mu.fill(0.0);
    constant_clock.mu[0] = 50.0; // 15 expected arrivals/second after scale .3
    const std::vector<double> constant_counts =
        one_second_event_counts(constant_clock, 2'000U);

    BackgroundHawkesConfig stochastic_clock = constant_clock;
    stochastic_clock.stochastic_baseline_persistence = 0.95;
    stochastic_clock.stochastic_baseline_std = 0.60;
    stochastic_clock.stochastic_baseline_bin_width_ns = 1'000'000'000LL;
    stochastic_clock.stochastic_baseline_normalization_bins = 2'000U;

    // The configured session normalizer must preserve the arithmetic mean of
    // the immigration multiplier itself, rather than claiming that a random
    // realized Hawkes event count is pathwise fixed.
    dlob::BackgroundHawkesStream multiplier_clock(stochastic_clock);
    dlob::MarketState multiplier_state;
    multiplier_state.best_bid_ticks = 10'000;
    multiplier_state.best_ask_ticks = 10'100;
    multiplier_state.best_bid_depth = 100;
    multiplier_state.best_ask_depth = 100;
    double multiplier_mean = 0.0;
    for (std::uint64_t index = 0; index < 2'000U; ++index) {
        const auto intensities = multiplier_clock.baseline_intensities(
            static_cast<std::int64_t>(index) * 1'000'000'000LL,
            multiplier_state);
        multiplier_mean += intensities[0] / 15.0;
    }
    multiplier_mean /= 2'000.0;
    assert(std::abs(multiplier_mean - 1.0) < 1.0e-12);
    const std::vector<double> stochastic_counts =
        one_second_event_counts(stochastic_clock, 2'000U);
    double stochastic_mean = 0.0;
    for (const double value : stochastic_counts) stochastic_mean += value;
    stochastic_mean /= static_cast<double>(stochastic_counts.size());
    const double constant_count_acf =
        lag_one_autocorrelation(constant_counts);
    const double stochastic_count_acf =
        lag_one_autocorrelation(stochastic_counts);
    std::cout << "stochastic_baseline_diagnostic mean=" << stochastic_mean
              << " constant_count_acf=" << constant_count_acf
              << " stochastic_count_acf=" << stochastic_count_acf << '\n';
    assert(std::abs(stochastic_mean - 15.0) / 15.0 < 0.20);
    assert(stochastic_count_acf > constant_count_acf + 0.15);

    BackgroundHawkesConfig invalid_stochastic = stochastic_clock;
    invalid_stochastic.stochastic_baseline_std = 2.51;
    assert(throws_invalid_argument(invalid_stochastic));

    BackgroundHawkesStream cached_then_changed(reactive);
    BackgroundHawkesStream changed_before_peek(reactive);
    const std::int64_t cached_time = cached_then_changed.peek_time_ns();
    assert(changed_before_peek.peek_time_ns() == cached_time);
    const HawkesEvent changed_after_cache = cached_then_changed.pop(wide);
    const HawkesEvent changed_before_cache = changed_before_peek.pop(wide);
    assert(changed_after_cache.time_ns == changed_before_cache.time_ns);
    assert(changed_after_cache.type == changed_before_cache.type);

    // The fitted state response must act on Hawkes excitation as well as on
    // immigration.  Force the first event to be type zero, let it excite only
    // type one, then check the exact hazard-preserving transformation of the
    // complete pre-type vector (immigration 3 plus excitation 5).
    BackgroundHawkesConfig reactive_excitation;
    set_zero_excitation(reactive_excitation);
    reactive_excitation.seed = 0x8e51ULL;
    reactive_excitation.mu.fill(0.0);
    reactive_excitation.mu[0] = 10.0;
    reactive_excitation.alpha[1][0] = 5.0;
    reactive_excitation.tick_size = 100;
    reactive_excitation.target_spread_ticks = 1;
    reactive_excitation.state_reference_bid_depth = 100.0;
    reactive_excitation.state_reference_ask_depth = 100.0;
    reactive_excitation.state_log_multiplier_coefficients[0][0] = -1.0;
    reactive_excitation.state_log_multiplier_coefficients[1][0] = 1.0;

    BackgroundHawkesStream excitation_stream(reactive_excitation);
    assert(excitation_stream.pop(narrow).type == HawkesEventType::LimitBuy);
    const BackgroundHawkesVector unadjusted_excitation_rates =
        excitation_stream.current_type_intensities(narrow);
    const BackgroundHawkesVector adjusted_excitation_rates =
        excitation_stream.current_type_intensities(wide);
    assert(std::abs(sum(unadjusted_excitation_rates) - 8.0) <= 1.0e-12);
    assert(std::abs(sum(adjusted_excitation_rates) - 8.0) <= 1.0e-12);
    const double expected_normalization = 8.0 / (3.0 * 0.5 + 5.0 * 2.0);
    assert(std::abs(
        adjusted_excitation_rates[0]
            - 3.0 * 0.5 * expected_normalization) <= 1.0e-12);
    assert(std::abs(
        adjusted_excitation_rates[1]
            - 5.0 * 2.0 * expected_normalization) <= 1.0e-12);
    assert(adjusted_excitation_rates[1]
           > unadjusted_excitation_rates[1]);

    // Event-type intraday profiles are optional and must be normalized so the
    // fitted stationary target retains its interpretation.
    BackgroundHawkesConfig seasonal;
    set_zero_excitation(seasonal);
    seasonal.intraday_bin_width_ns = 1'000'000'000LL;
    seasonal.intraday_factors = {
        BackgroundHawkesVector{0.5, 1.5, 0.25, 1.75, 0.75, 1.25},
        BackgroundHawkesVector{1.5, 0.5, 1.75, 0.25, 1.25, 0.75},
    };
    BackgroundHawkesStream seasonal_stream(seasonal);
    const BackgroundHawkesVector first_bin =
        seasonal_stream.baseline_intensities(500'000'000LL, narrow);
    const BackgroundHawkesVector second_bin =
        seasonal_stream.baseline_intensities(1'500'000'000LL, narrow);
    assert(first_bin[0] == seasonal.activity_scale * seasonal.mu[0] * 0.5);
    assert(second_bin[0] == seasonal.activity_scale * seasonal.mu[0] * 1.5);
    BackgroundHawkesConfig unnormalized = seasonal;
    unnormalized.intraday_factors[1][0] = 1.0;
    assert(throws_invalid_argument(unnormalized));

    BackgroundHawkesConfig closed_then_open;
    set_zero_excitation(closed_then_open);
    closed_then_open.seed = 17;
    closed_then_open.mu.fill(0.0);
    closed_then_open.mu[0] = 100.0;
    closed_then_open.intraday_bin_width_ns = 1'000'000'000LL;
    BackgroundHawkesVector closed{};
    BackgroundHawkesVector open{};
    open.fill(2.0);
    closed_then_open.intraday_factors = {closed, open};
    BackgroundHawkesStream opening_stream(closed_then_open);
    assert(opening_stream.peek_time_ns() >= 1'000'000'000LL);

    // The complete two-timescale branching matrix must reproduce the frozen
    // stationary target from the configured immigration vector.
    BackgroundHawkesConfig fitted;
    set_zero_excitation(fitted);
    fitted.beta = 8.0;
    fitted.slow_beta = 0.5;
    fitted.alpha[0][0] = 0.4;       // integrated fast weight 0.05
    fitted.alpha[2][3] = 0.08;      // integrated fast cross weight 0.01
    fitted.slow_alpha[0][1] = 0.05; // integrated slow cross weight 0.10
    fitted.slow_alpha[4][4] = 0.10; // integrated slow weight 0.20
    fitted.stationary_target_rates = {10.0, 8.0, 3.0, 4.0, 12.0, 11.0};
    for (std::size_t row = 0; row < fitted.mu.size(); ++row) {
        double endogenous = 0.0;
        for (std::size_t column = 0; column < fitted.mu.size(); ++column) {
            endogenous += (
                fitted.alpha[row][column] / fitted.beta
                + fitted.slow_alpha[row][column] / fitted.slow_beta)
                * fitted.stationary_target_rates[column];
        }
        fitted.mu[row] =
            (fitted.stationary_target_rates[row] - endogenous)
            / fitted.activity_scale;
        assert(fitted.mu[row] >= 0.0);
    }
    fitted.validate_stationary_target = true;
    BackgroundHawkesStream fitted_stream(fitted);
    assert(fitted_stream.peek_time_ns() > 0);

    BackgroundHawkesConfig stale_immigration = fitted;
    stale_immigration.mu[0] += 0.1;
    assert(throws_invalid_argument(stale_immigration));

    BackgroundHawkesConfig unstable;
    set_zero_excitation(unstable);
    unstable.beta = 1.0;
    unstable.alpha[0][1] = 0.95;
    unstable.alpha[1][0] = 0.95;
    assert(throws_invalid_argument(unstable));
    unstable.alpha[0][1] = 0.949;
    unstable.alpha[1][0] = 0.949;
    assert(!throws_invalid_argument(unstable));

    // Spectral radius alone does not protect the queue-reactive process from
    // state-driven event-type relabelling.  This matrix has spectral radius
    // zero but one event type would inject more than unit total future
    // excitation.  Reject it only when state response is active; the zero-
    // response legacy path retains its historical validation semantics.
    BackgroundHawkesConfig unsafe_column;
    set_zero_excitation(unsafe_column);
    unsafe_column.beta = 1.0;
    unsafe_column.alpha[0][1] = 0.4;
    unsafe_column.alpha[2][1] = 0.4;
    unsafe_column.alpha[4][1] = 0.4;
    assert(!throws_invalid_argument(unsafe_column));
    unsafe_column.state_log_multiplier_coefficients[0][0] = 0.1;
    assert(throws_invalid_argument(unsafe_column));

    // Clock/type draws and empirical mark draws live in different objects and
    // streams.  Advancing the clock cannot perturb an agent's next mark.
    const std::filesystem::path quantity_file =
        std::filesystem::temp_directory_path()
        / "dlob_hawkes_stream_quantity.csv";
    const std::filesystem::path distance_file =
        std::filesystem::temp_directory_path()
        / "dlob_hawkes_stream_distance.csv";
    {
        std::ofstream output(quantity_file);
        assert(output);
        output << "quantity,count\n37,1\n91,1\n";
    }
    {
        std::ofstream output(distance_file);
        assert(output);
        output << "distance_ticks,count\n0,1\n2,1\n";
    }
    BackgroundHawkesConfig marks;
    marks.seed = 0x99ULL;
    marks.quote_improvement_probability = 0.0;
    set_mark_files(marks, quantity_file, distance_file);
    BackgroundHawkesAgent untouched_agent(marks);
    BackgroundHawkesAgent clock_isolated_agent(marks);
    BackgroundHawkesStream independent_clock(marks);
    for (int index = 0; index < 100; ++index) {
        (void)independent_clock.pop(narrow);
    }
    const HawkesEvent mark_event{123, HawkesEventType::LimitBuy};
    const OrderMessage untouched =
        untouched_agent.make_order(mark_event, narrow, 7);
    const OrderMessage isolated =
        clock_isolated_agent.make_order(mark_event, narrow, 7);
    assert_order_equal(untouched, isolated);

    return 0;
}
