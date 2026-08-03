#include "simulation/AssetMomentAccumulator.hpp"
#include "simulation/DeterministicFundamentalProcess.hpp"
#include "simulation/FragmentedQuotePlacement.hpp"
#include "simulation/MultiAssetConfiguration.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    using namespace dlob;

    MarketState wide;
    wide.best_bid_ticks = 10'000;
    wide.best_ask_ticks = 10'500;
    wide.mid_price_ticks = 10'250.0;

    const detail::FragmentedQuotePrices passive =
        detail::fragmented_quote_prices(
            wide, 10'250.0, 100, 2, false);
    assert(passive.bid == 10'000);
    assert(passive.ask == 10'500);

    const detail::FragmentedQuotePrices improved =
        detail::fragmented_quote_prices(
            wide, 10'250.0, 100, 2, true);
    assert(improved.bid == 10'100);
    assert(improved.ask == 10'400);
    assert(improved.ask - improved.bid == 300);

    MarketState one_tick_wide;
    one_tick_wide.best_bid_ticks = 10'000;
    one_tick_wide.best_ask_ticks = 10'300;
    const detail::FragmentedQuotePrices one_tick_tighter =
        detail::fragmented_quote_prices(
            one_tick_wide, 10'150.0, 100, 2, true);
    assert(one_tick_tighter.bid == 10'100);
    assert(one_tick_tighter.ask == 10'300);
    assert(one_tick_tighter.ask - one_tick_tighter.bid == 200);

    MarketState missing_ask;
    missing_ask.best_bid_ticks = 10'000;
    const detail::FragmentedQuotePrices repaired_ask =
        detail::fragmented_quote_prices(
            missing_ask, 50'000.0, 100, 2, false);
    assert(repaired_ask.bid == 10'000);
    assert(repaired_ask.ask == 10'200);

    MarketState missing_bid;
    missing_bid.best_ask_ticks = 10'200;
    const detail::FragmentedQuotePrices repaired_bid =
        detail::fragmented_quote_prices(
            missing_bid, 50'000.0, 100, 2, false);
    assert(repaired_bid.bid == 10'000);
    assert(repaired_bid.ask == 10'200);

    assert(!detail::deterministic_quote_improvement(
        7, 0x80000, 3, 9, 0.0));
    assert(detail::deterministic_quote_improvement(
        7, 0x80000, 3, 9, 1.0));
    const bool repeated = detail::deterministic_quote_improvement(
        7, 0x80000, 3, 9, 0.5);
    assert(repeated == detail::deterministic_quote_improvement(
        7, 0x80000, 3, 9, 0.5));
    bool saw_improvement = false;
    bool saw_passive = false;
    for (std::uint64_t refresh = 0; refresh < 128; ++refresh) {
        const bool decision = detail::deterministic_quote_improvement(
            7, 0x80000, 3, refresh, 0.5);
        saw_improvement = saw_improvement || decision;
        saw_passive = saw_passive || !decision;
    }
    assert(saw_improvement && saw_passive);

    // eta=0 is the exact constant-probability legacy branch.  A one-target
    // spread cannot amplify p0, while progressively wider spreads increase it
    // monotonically until the declared cap is reached.
    assert(detail::local_mm_effective_improvement_probability(
        0.25, 1.0, 800, 200, 0.0) == 0.25);
    assert(detail::local_mm_effective_improvement_probability(
        0.25, 1.0, 200, 200, 1.0) == 0.25);
    assert(detail::local_mm_effective_improvement_probability(
        0.25, 1.0, 400, 200, 1.0) == 0.5);
    assert(detail::local_mm_effective_improvement_probability(
        0.25, 1.0, 800, 200, 1.0) == 1.0);
    assert(detail::local_mm_effective_improvement_probability(
        0.25, 0.6, 800, 200, 1.0) == 0.6);

    // Repair-only local quoting must not recreate a continuous synthetic
    // depth floor.
    assert(!detail::fragmented_quote_required(
        true, false, false, false, false));
    assert(!detail::fragmented_quote_required(
        true, false, false, true, false));
    assert(detail::fragmented_quote_required(
        true, false, false, true, true));
    assert(detail::fragmented_quote_required(
        true, false, true, false, false));
    assert(detail::fragmented_quote_required(
        true, true, false, false, false));
    assert(detail::fragmented_quote_required(
        false, false, false, false, false));

    assert(std::abs(detail::shared_capacity_quote_scale(
        0.25, 0.5, 0.05, true) - 1.0) < 1.0e-12);
    assert(std::abs(detail::shared_capacity_quote_scale(
        0.75, 0.5, 0.05, true) - 0.5) < 1.0e-12);
    assert(std::abs(detail::shared_capacity_quote_scale(
        2.0, 0.5, 0.05, true) - 0.05) < 1.0e-12);
    assert(std::abs(detail::shared_capacity_quote_scale(
        2.0, 0.5, 0.05, false) - 1.0) < 1.0e-12);
    assert(detail::inventory_adverse_shock_side(-1) == Side::Buy);
    assert(detail::inventory_adverse_shock_side(0) == Side::Sell);
    assert(detail::inventory_adverse_shock_side(1) == Side::Sell);

    // Global capacity constrains only risk-increasing quotes.  A long dealer
    // may still sell at zero global scale and a short dealer may still buy;
    // the reducing side is capped at current inventory to prevent overshoot.
    const detail::SharedQuotePlan flat =
        detail::risk_managed_shared_quote_plan(0, 0.0, 100.0);
    assert(flat.bid_scale == 0.0 && flat.ask_scale == 0.0);
    const detail::SharedQuotePlan long_position =
        detail::risk_managed_shared_quote_plan(40, 0.0, 100.0);
    assert(long_position.bid_scale == 0.0);
    assert(long_position.ask_scale == 1.4);
    assert(long_position.ask_reduces_inventory);
    assert(long_position.ask_total_limit == 40);
    const detail::SharedQuotePlan short_position =
        detail::risk_managed_shared_quote_plan(-25, 0.0, 100.0);
    assert(short_position.bid_scale == 1.25);
    assert(short_position.ask_scale == 0.0);
    assert(short_position.bid_reduces_inventory);
    assert(short_position.bid_total_limit == 25);
    const detail::SharedQuotePlan partially_constrained =
        detail::risk_managed_shared_quote_plan(20, 0.5, 100.0);
    assert(partially_constrained.bid_scale == 0.4);
    assert(partially_constrained.ask_scale == 1.2);

    // A wide spread that straddles value is not an executable mispricing,
    // even when its midpoint lies away from value.
    assert(!detail::fundamental_value_side(
        9'000, 12'000, 10'000.0, 5.0).has_value());
    assert(detail::fundamental_value_side(
        9'000, 9'900, 10'000.0, 5.0) == Side::Buy);
    assert(detail::fundamental_value_side(
        10'100, 12'000, 10'000.0, 5.0) == Side::Sell);
    assert(detail::fundamental_value_passive_price(
        Side::Buy, 9'000, 9'900, 10'000.0, 100) == 9'100);
    assert(detail::fundamental_value_passive_price(
        Side::Sell, 10'100, 12'000, 10'000.0, 100) == 11'900);
    assert(detail::fundamental_value_passive_price(
        Side::Buy, 9'900, 10'000, 10'100.0, 100) == 9'900);
    assert(detail::fundamental_value_passive_price(
        Side::Sell, 10'000, 10'100, 9'900.0, 100) == 10'100);

    // Ceil participation is preserved except when it would request the entire
    // finite side: the matching engine would already reserve one share, so
    // pre-capping that request to D-1 preserves fills while eliminating a
    // repeated artificial request.  A deliberate 100% policy still reaches
    // the independent boundary guard.
    assert(detail::fundamental_value_participation_quantity(0, 0.5) == 0);
    assert(detail::fundamental_value_participation_quantity(1, 0.5) == 0);
    assert(detail::fundamental_value_participation_quantity(2, 0.5) == 1);
    assert(detail::fundamental_value_participation_quantity(3, 0.5) == 2);
    assert(detail::fundamental_value_participation_quantity(5, 0.5) == 3);
    assert(detail::fundamental_value_participation_quantity(10, 0.25) == 3);
    assert(detail::fundamental_value_participation_quantity(10, 0.5) == 5);
    assert(detail::fundamental_value_participation_quantity(10, 1.0) == 10);

    const double buy_gap = detail::fundamental_value_executable_gap_bps(
        Side::Buy, 10'000, 10'200, 11'200.0);
    const double sell_gap = detail::fundamental_value_executable_gap_bps(
        Side::Sell, 11'000, 11'200, 10'000.0);
    assert(std::abs(buy_gap - 892.8571428571429) < 1.0e-12);
    assert(sell_gap == 1'000.0);
    assert(detail::fundamental_value_executable_gap_bps(
        Side::Buy, 10'000, 10'200, 10'100.0) == 0.0);

    // eta=0 is a deliberate exact legacy branch: it does not divide by a
    // zero threshold or perturb the original participation through pow().
    assert(detail::fundamental_value_effective_participation(
        0.10, 1.0, buy_gap, 0.0, 0.0) == 0.10);
    assert(std::abs(detail::fundamental_value_effective_participation(
        0.10, 0.50, 20.0, 10.0, 1.0) - 0.20) < 1.0e-15);
    assert(std::abs(detail::fundamental_value_effective_participation(
        0.10, 0.50, 20.0, 10.0, 2.0) - 0.40) < 1.0e-15);
    assert(detail::fundamental_value_effective_participation(
        0.10, 0.50, 100.0, 10.0, 2.0) == 0.50);
    assert(detail::fundamental_value_effective_participation(
        0.10, 0.50, 20.0, 0.0, 1.0) == 0.0);
    assert(detail::fundamental_value_effective_participation(
        0.50, 0.25, 20.0, 10.0, 1.0) == 0.0);

    detail::AssetMomentAccumulator moments;
    MarketState first;
    first.best_bid_ticks = 9'900;
    first.best_ask_ticks = 10'100;
    first.mid_price_ticks = 10'000.0;
    first.best_bid_depth = 10;
    first.best_ask_depth = 20;
    moments.observe(first, 100);

    MarketState invalid;
    invalid.best_bid_ticks = 9'900;
    moments.observe(invalid, 100);

    MarketState after_gap = first;
    after_gap.best_bid_ticks = 19'900;
    after_gap.best_ask_ticks = 20'100;
    after_gap.mid_price_ticks = 20'000.0;
    moments.observe(after_gap, 100);

    MarketState adjacent = after_gap;
    adjacent.best_bid_ticks = 20'000;
    adjacent.best_ask_ticks = 20'200;
    adjacent.mid_price_ticks = 20'100.0;
    moments.observe(adjacent, 100);

    assert(moments.snapshots == 3);
    assert(moments.invalid_snapshots == 1);
    assert(moments.adjacent_pairs == 1);
    assert(moments.return_count == 1);
    assert(moments.mid_moves == 1);
    const auto values = moments.finalize();
    assert(values[3] == 1.0);

    // The latent-value process is stateless with respect to execution order:
    // identical logical identifiers reproduce exactly, while a different
    // symbol stream does not alias the path. Zero volatility retains the
    // legacy static reference.
    constexpr StableEntityId symbol_stream = 0x12345678ULL;
    const double first_fundamental = detail::advance_fundamental_value(
        100'000.0, 2.0, 1.0, 3.0, 1.0, symbol_stream, 20200130, 1);
    assert(first_fundamental == detail::advance_fundamental_value(
        100'000.0, 2.0, 1.0, 3.0, 1.0, symbol_stream, 20200130, 1));
    assert(first_fundamental != detail::advance_fundamental_value(
        100'000.0, 2.0, 1.0, 3.0, 1.0, symbol_stream + 1, 20200130, 1));
    assert(detail::advance_fundamental_value(
        100'000.0, 0.0, 1.0, 3.0, 1.0, symbol_stream, 20200130, 1) == 100'000.0);

    // Across the deterministic innovation sequence, a 2 bps/sqrt(second)
    // input reproduces its declared one-second log-return variance.
    constexpr std::uint64_t draws = 20'000;
    double sum = 0.0;
    double sum2 = 0.0;
    for (std::uint64_t index = 1; index <= draws; ++index) {
        const double next = detail::advance_fundamental_value(
            100'000.0, 2.0, 1.0, 3.0, 1.0, symbol_stream, 20200130, index);
        const double value = std::log(next / 100'000.0);
        sum += value;
        sum2 += value * value;
    }
    const double mean = sum / static_cast<double>(draws);
    const double variance = sum2 / static_cast<double>(draws) - mean * mean;
    const double declared_variance = std::pow(2.0 / 10'000.0, 2.0);
    assert(std::abs(variance / declared_variance - 1.0) < 0.05);

    // Sparse news preserves the same unconditional variance while producing
    // approximately the declared nonzero-move frequency.
    constexpr double sparse_probability = 0.05;
    sum = 0.0;
    sum2 = 0.0;
    std::uint64_t nonzero = 0;
    for (std::uint64_t index = 1; index <= draws; ++index) {
        const double next = detail::advance_fundamental_value(
            100'000.0, 2.0, sparse_probability, 3.0, 1.0,
            symbol_stream, 20200130, index);
        const double value = std::log(next / 100'000.0);
        nonzero += value != 0.0 ? 1U : 0U;
        sum += value;
        sum2 += value * value;
    }
    const double sparse_mean = sum / static_cast<double>(draws);
    const double sparse_variance = sum2 / static_cast<double>(draws)
        - sparse_mean * sparse_mean;
    const double observed_probability = static_cast<double>(nonzero)
        / static_cast<double>(draws);
    assert(std::abs(observed_probability - sparse_probability) < 0.01);
    assert(std::abs(sparse_variance / declared_variance - 1.0) < 0.12);

    // The conditional innovation generator reproduces both second and fourth
    // moments without changing the probability that a news event occurs.
    constexpr double requested_kurtosis = 12.0;
    double innovation_sum2 = 0.0;
    double innovation_sum4 = 0.0;
    for (std::uint64_t index = 1; index <= draws; ++index) {
        const double innovation = detail::deterministic_moment_matched_innovation(
            requested_kurtosis, symbol_stream, 20200130, index);
        innovation_sum2 += innovation * innovation;
        innovation_sum4 += innovation * innovation * innovation * innovation;
    }
    const double innovation_second = innovation_sum2 / static_cast<double>(draws);
    const double innovation_fourth = innovation_sum4 / static_cast<double>(draws);
    assert(std::abs(innovation_second - 1.0) < 0.05);
    assert(std::abs(innovation_fourth
                    / (innovation_second * innovation_second)
                    - requested_kurtosis) < 0.8);

    // The largest tail candidates used by the frozen calibration protocol are
    // around 2,800.  Verify that the law remains finite there and that the
    // log-MGF correction makes the multiplicative price factor an exact
    // martingale under the implemented (non-Gaussian) innovation law.
    constexpr double high_combined_kurtosis = 2'800.0;
    constexpr double high_log_variance_std = 0.15;
    const double high_innovation_kurtosis = high_combined_kurtosis
        * std::exp(-high_log_variance_std * high_log_variance_std);
    const detail::FundamentalMomentMatchedLaw high_law =
        detail::fundamental_moment_matched_law(high_innovation_kurtosis);
    const double high_common_second = high_law.common_magnitude
        * high_law.common_magnitude;
    const double high_rare_second = high_law.rare_magnitude
        * high_law.rare_magnitude;
    const double high_second =
        (1.0 - high_law.rare_probability) * high_common_second
        + high_law.rare_probability * high_rare_second;
    const double high_fourth =
        (1.0 - high_law.rare_probability)
            * high_common_second * high_common_second
        + high_law.rare_probability * high_rare_second * high_rare_second;
    assert(std::isfinite(high_law.common_magnitude));
    assert(std::isfinite(high_law.rare_magnitude));
    assert(std::abs(high_second - 1.0) < 1.0e-12);
    assert(std::abs(
        high_fourth / (high_second * high_second)
            * std::exp(high_log_variance_std * high_log_variance_std)
            / high_combined_kurtosis
        - 1.0) < 1.0e-12);

    constexpr double high_sigma = 0.02;
    const double high_log_mgf = detail::fundamental_moment_matched_log_mgf(
        high_innovation_kurtosis, high_sigma);
    const double high_expected_factor = std::exp(-high_log_mgf) * (
        (1.0 - high_law.rare_probability)
            * std::cosh(high_sigma * high_law.common_magnitude)
        + high_law.rare_probability
            * std::cosh(high_sigma * high_law.rare_magnitude));
    assert(std::isfinite(high_log_mgf));
    assert(std::abs(high_expected_factor - 1.0) < 1.0e-13);

    // Exercise the combined deterministic volatility multiplier and innovation
    // streams, not only their separate analytical moments.  The innovation
    // kurtosis removes exp(s^2), so their product recovers the requested
    // marginal fourth/second-squared moment without changing variance.
    constexpr double combined_kurtosis = 100.0;
    constexpr double combined_std = 0.15;
    const double combined_innovation_kurtosis = combined_kurtosis
        * std::exp(-combined_std * combined_std);
    constexpr std::uint64_t combined_draws = 200'000;
    double combined_sum2 = 0.0;
    double combined_sum4 = 0.0;
    for (std::uint64_t index = 1; index <= combined_draws; ++index) {
        const double log_variance = combined_std
            * detail::deterministic_fundamental_normal(
                symbol_stream ^ detail::fundamental_log_variance_domain,
                20200130, index);
        const double multiplier = detail::fundamental_volatility_multiplier(
            log_variance, combined_std);
        const double innovation = detail::deterministic_moment_matched_innovation(
            combined_innovation_kurtosis, symbol_stream, 20200130, index);
        const double value = multiplier * innovation;
        combined_sum2 += value * value;
        combined_sum4 += value * value * value * value;
    }
    const double combined_count = static_cast<double>(combined_draws);
    const double combined_second = combined_sum2 / combined_count;
    const double combined_fourth = combined_sum4 / combined_count;
    const double observed_combined_kurtosis = combined_fourth
        / (combined_second * combined_second);
    assert(std::abs(combined_second - 1.0) < 0.05);
    assert(std::abs(observed_combined_kurtosis / combined_kurtosis - 1.0)
        < 0.15);

    // A zero log-variance standard deviation is exactly the legacy path: the
    // hidden state is zero and the volatility multiplier is exactly one.
    constexpr double log_variance_persistence = 0.95;
    assert(detail::initial_fundamental_log_variance(
        log_variance_persistence, 0.0, symbol_stream, 20200130) == 0.0);
    assert(detail::advance_fundamental_log_variance(
        0.0, log_variance_persistence, 0.0,
        symbol_stream, 20200130, 1) == 0.0);
    assert(detail::fundamental_volatility_multiplier(0.0, 0.0) == 1.0);
    const double legacy_path = detail::advance_fundamental_value(
        100'000.0, 2.0, 1.0, 3.0, 1.0,
        symbol_stream, 20200130, 17);
    const double disabled_persistent_path = detail::advance_fundamental_value(
        100'000.0,
        2.0 * detail::fundamental_volatility_multiplier(0.0, 0.0),
        1.0, 3.0, 1.0, symbol_stream, 20200130, 17);
    assert(disabled_persistent_path == legacy_path);

    // The active state is deterministic by logical identity and starts in
    // its stationary distribution.  It must not alias another symbol stream.
    constexpr double log_variance_std = 0.6;
    const double initial_log_variance =
        detail::initial_fundamental_log_variance(
            log_variance_persistence, log_variance_std,
            symbol_stream, 20200130);
    assert(initial_log_variance == detail::initial_fundamental_log_variance(
        log_variance_persistence, log_variance_std,
        symbol_stream, 20200130));
    assert(initial_log_variance != detail::initial_fundamental_log_variance(
        log_variance_persistence, log_variance_std,
        symbol_stream + 1U, 20200130));

    constexpr std::uint64_t variance_draws = 100'000;
    double log_variance = initial_log_variance;
    double log_variance_sum = 0.0;
    double log_variance_sum2 = 0.0;
    double lagged_sum = 0.0;
    double lagged_left_sum = 0.0;
    double lagged_right_sum = 0.0;
    double lagged_left_sum2 = 0.0;
    double lagged_right_sum2 = 0.0;
    double multiplier_second_moment = 0.0;
    for (std::uint64_t index = 1; index <= variance_draws; ++index) {
        const double previous = log_variance;
        log_variance = detail::advance_fundamental_log_variance(
            log_variance, log_variance_persistence, log_variance_std,
            symbol_stream, 20200130, index);
        const double multiplier = detail::fundamental_volatility_multiplier(
            log_variance, log_variance_std);
        log_variance_sum += log_variance;
        log_variance_sum2 += log_variance * log_variance;
        lagged_sum += previous * log_variance;
        lagged_left_sum += previous;
        lagged_right_sum += log_variance;
        lagged_left_sum2 += previous * previous;
        lagged_right_sum2 += log_variance * log_variance;
        multiplier_second_moment += multiplier * multiplier;
    }
    const double variance_count = static_cast<double>(variance_draws);
    const double log_variance_mean = log_variance_sum / variance_count;
    const double observed_log_variance = log_variance_sum2 / variance_count
        - log_variance_mean * log_variance_mean;
    assert(std::abs(observed_log_variance
                    / (log_variance_std * log_variance_std) - 1.0) < 0.05);
    const double lagged_left_mean = lagged_left_sum / variance_count;
    const double lagged_right_mean = lagged_right_sum / variance_count;
    const double lagged_covariance = lagged_sum / variance_count
        - lagged_left_mean * lagged_right_mean;
    const double lagged_left_variance = lagged_left_sum2 / variance_count
        - lagged_left_mean * lagged_left_mean;
    const double lagged_right_variance = lagged_right_sum2 / variance_count
        - lagged_right_mean * lagged_right_mean;
    const double observed_persistence = lagged_covariance
        / std::sqrt(lagged_left_variance * lagged_right_variance);
    assert(std::abs(observed_persistence - log_variance_persistence) < 0.02);
    assert(std::abs(multiplier_second_moment / variance_count - 1.0) < 0.05);

    // Both optional CSV fields are parsed and rejected outside their declared
    // domains before a simulation begins.
    const std::filesystem::path config_path =
        std::filesystem::temp_directory_path()
        / ("dlob_log_variance_config_"
           + std::to_string(stable_sequence(symbol_stream, 20200130))
           + ".csv");
    auto write_config = [&](double persistence, double state_std,
                            double flow_coupling) {
        std::ofstream output(config_path, std::ios::trunc);
        assert(output);
        output
            << "book_id,symbol,data_dir,hawkes_rates_file,"
               "fundamental_price_ticks,"
               "fundamental_volatility_bps_sqrt_second,"
               "fundamental_move_probability_per_second,"
               "fundamental_conditional_kurtosis,"
               "fundamental_log_variance_persistence,"
               "fundamental_log_variance_std,"
               "fundamental_order_flow_coupling,"
               "initial_best_bid_ticks,initial_best_ask_ticks,"
               "initial_best_bid_depth,initial_best_ask_depth,beta,"
               "basket_weight,market_maker_quote_quantity,"
               "target_spread_ticks\n"
            << "0,TEST,.,,100000,2,0.5,4,"
            << persistence << ',' << state_std << ',' << flow_coupling
            << ",99000,101000,10,12,1,0,100,2\n";
    };
    write_config(0.85, 0.6, 0.4);
    const std::vector<MultiAssetBookConfig> parsed =
        load_multi_asset_book_configs(config_path);
    assert(parsed.size() == 1U);
    assert(parsed[0].fundamental_log_variance_persistence == 0.85);
    assert(parsed[0].fundamental_log_variance_std == 0.6);
    assert(parsed[0].fundamental_order_flow_coupling == 0.4);
    auto invalid_config_rejected = [&](double persistence, double state_std,
                                       double flow_coupling) {
        write_config(persistence, state_std, flow_coupling);
        try {
            static_cast<void>(load_multi_asset_book_configs(config_path));
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    assert(invalid_config_rejected(1.0, 0.6, 0.0));
    assert(invalid_config_rejected(0.5, -0.1, 0.0));
    assert(invalid_config_rejected(0.5, 0.0, 0.1));
    assert(invalid_config_rejected(0.5, 0.6, 2.51));
    std::error_code remove_error;
    std::filesystem::remove(config_path, remove_error);

    std::cout << "fragmented quote and fixed-clock semantics tests passed\n";
    return 0;
}
