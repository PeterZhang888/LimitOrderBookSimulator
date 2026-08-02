// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "simulation/FragmentedMpiSimulator.hpp"
#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::filesystem::path source_root() {
    std::filesystem::path root = std::filesystem::path(__FILE__).parent_path()
        .parent_path();
    if (!std::filesystem::exists(root / "config")) {
        root = std::filesystem::current_path();
    }
    return std::filesystem::absolute(root);
}

std::vector<std::string> split(const std::string& row) {
    std::vector<std::string> values;
    std::istringstream input(row);
    std::string value;
    while (std::getline(input, value, ',')) values.push_back(value);
    return values;
}

dlob::FragmentedMpiResult run_with_summary(
    const std::filesystem::path& output,
    bool clustered_policy,
    std::int64_t local_mm_interval_ns = 0,
    double hawkes_activity_scale = 0.30,
    double local_mm_quantity_multiplier = 1.0,
    bool misprice_qqq = false,
    std::int64_t global_metrics_interval_ns = 0,
    std::int64_t decision_window_ns = 1'000'000'000LL,
    std::int64_t value_agent_interval_ns = 1'000'000'000LL,
    double local_mm_improvement_probability = 0.0,
    double local_mm_spread_elasticity = 0.0,
    double local_mm_max_improvement_probability = 1.0) {
    dlob::FragmentedMpiConfig config;
    config.duration_seconds = 2;
    config.decision_window_ns = decision_window_ns;
    config.asset_summary_interval_ns = 1'000'000'000LL;
    config.seed = 20200130;
    config.hawkes_activity_scale = hawkes_activity_scale;
    config.local_mm_interval_ns = local_mm_interval_ns;
    config.value_agent_interval_ns = value_agent_interval_ns;
    config.local_mm_quantity_multiplier = local_mm_quantity_multiplier;
    config.local_mm_improvement_probability =
        local_mm_improvement_probability;
    config.local_mm_spread_elasticity = local_mm_spread_elasticity;
    config.local_mm_max_improvement_probability =
        local_mm_max_improvement_probability;
    config.global_metrics_interval_ns = global_metrics_interval_ns;
    config.asset_configs = dlob::load_multi_asset_book_configs(
        source_root() / "config" / "qqq_aapl_msft_amzn_20200130.csv");
    for (dlob::MultiAssetBookConfig& asset : config.asset_configs) {
        asset.data_dir = (source_root() / asset.data_dir).string();
        asset.hawkes_rates_file = (
            source_root() / asset.hawkes_rates_file).string();
    }
    config.asset_count = static_cast<int>(config.asset_configs.size());
    if (misprice_qqq) {
        config.asset_configs[0].fundamental_price_ticks =
            static_cast<double>(config.asset_configs[0].initial_best_ask_ticks + 1'000);
    }
    config.enable_shared_market_maker = false;
    config.enable_value_agents = true;
    config.asset_summary_csv = output.string();
    if (clustered_policy) {
        config.value_agent_policies = {
            {true, 0.0, 0.10, 10},
            {false, 8.0, 0.10, 50},
            {false, 8.0, 0.10, 50},
            {false, 8.0, 0.10, 50},
        };
    } else {
        config.enable_value_agents = false;
    }
    return dlob::FragmentedMpiSimulator(MPI_COMM_WORLD, std::move(config)).run();
}

using SummaryRow = std::unordered_map<std::string, std::string>;

SummaryRow read_single_summary_row(const std::filesystem::path& output) {
    std::ifstream input(output);
    assert(input);
    std::string header;
    std::string row;
    assert(std::getline(input, header));
    assert(std::getline(input, row));
    std::string unexpected_row;
    assert(!std::getline(input, unexpected_row));

    const std::vector<std::string> columns = split(header);
    const std::vector<std::string> values = split(row);
    assert(values.size() == columns.size());
    SummaryRow result;
    for (std::size_t index = 0; index < columns.size(); ++index) {
        const bool inserted = result.emplace(columns[index], values[index]).second;
        assert(inserted);
    }
    return result;
}

std::uint64_t summary_count(const SummaryRow& row, const std::string& name) {
    const auto found = row.find(name);
    assert(found != row.end());
    return std::stoull(found->second);
}

SummaryRow run_value_case(const std::filesystem::path& output,
                          double depth_participation,
                          double fundamental_price_ticks = 11'200.0,
                          int duration_seconds = 1,
                          int opening_depth = 16,
                          std::uint64_t* processed_orders = nullptr,
                          dlob::FragmentedValueTriggerMode trigger_mode =
                              dlob::FragmentedValueTriggerMode::PeriodicGap,
                          double move_probability = 0.0,
                          double volatility_bps_sqrt_second = 0.0,
                          std::int64_t value_agent_interval_ns =
                              1'000'000'000LL,
                          int maximum_news_rechecks = 0,
                          double threshold_bps = 0.0,
                          double gap_elasticity = 0.0,
                          double maximum_depth_participation = 1.0) {
    dlob::FragmentedMpiConfig config;
    config.duration_seconds = duration_seconds;
    config.decision_window_ns = 1'000'000'000LL;
    config.asset_summary_interval_ns = 1'000'000'000LL;
    config.seed = 20200130;
    // Keep the stochastic background valid but outside this one-second test,
    // so the requested value quantity depends only on the known opening book.
    config.hawkes_activity_scale = 1.0e-12;
    config.enable_local_market_makers = false;
    config.enable_shared_market_maker = false;
    config.enable_value_agents = true;
    config.value_agent_interval_ns = value_agent_interval_ns;
    config.asset_summary_csv = output.string();

    std::vector<dlob::MultiAssetBookConfig> templates =
        dlob::load_multi_asset_book_configs(
            source_root() / "config" / "qqq_aapl_msft_amzn_20200130.csv");
    assert(!templates.empty());
    dlob::MultiAssetBookConfig asset = templates.front();
    asset.data_dir = (source_root() / asset.data_dir).string();
    asset.hawkes_rates_file = (
        source_root() / asset.hawkes_rates_file).string();
    asset.initial_best_bid_ticks = 10'000;
    asset.initial_best_ask_ticks = 10'200;
    asset.initial_best_bid_depth = opening_depth;
    asset.initial_best_ask_depth = opening_depth;
    // The ten deterministic opening levels end at 11,100.  This reference
    // therefore makes the entire ask side reachable by the protected order.
    asset.fundamental_price_ticks = fundamental_price_ticks;
    asset.fundamental_volatility_bps_sqrt_second =
        volatility_bps_sqrt_second;
    asset.fundamental_move_probability_per_second = move_probability;
    config.asset_configs = {std::move(asset)};
    config.asset_count = 1;
    config.value_agent_policies = {
        {true, threshold_bps, depth_participation, 50},
    };
    config.value_agent_policies.front().trigger_mode = trigger_mode;
    config.value_agent_policies.front().maximum_news_rechecks =
        maximum_news_rechecks;
    config.value_agent_policies.front().gap_elasticity = gap_elasticity;
    config.value_agent_policies.front().maximum_depth_participation =
        maximum_depth_participation;

    const dlob::FragmentedMpiResult result =
        dlob::FragmentedMpiSimulator(MPI_COMM_WORLD, std::move(config)).run();
    assert(result.asset_count == 1);
    assert(result.lob_count == 1U);
    if (processed_orders != nullptr) {
        *processed_orders = result.processed_orders;
    }
    return read_single_summary_row(output);
}

void test_news_impulse_value_trigger() {
    const std::filesystem::path temporary =
        std::filesystem::temp_directory_path();
    std::uint64_t no_news_processed = 0;
    const SummaryRow no_news = run_value_case(
        temporary / "dlob_fragmented_value_news_no_move.csv",
        0.10, 11'200.0, 2, 16, &no_news_processed,
        dlob::FragmentedValueTriggerMode::NewsImpulse,
        0.0, 1.0);
    // An opening valuation gap is not fresh news.  With no fundamental move,
    // news mode must neither trade at t=0 nor revive the stale gap at t=1.
    assert(summary_count(no_news, "value_order_count") == 0U);
    assert(no_news_processed == 0U);

    std::uint64_t one_news_processed = 0;
    const SummaryRow one_news = run_value_case(
        temporary / "dlob_fragmented_value_news_one_move.csv",
        0.10, 11'200.0, 2, 16, &one_news_processed,
        dlob::FragmentedValueTriggerMode::NewsImpulse,
        1.0, 1.0);
    // The forced t=1 fundamental move creates exactly one opportunity.  The
    // coincident periodic clock is filtered by mode, so it cannot double
    // submit, and the order arrives before the t=2 terminal boundary.
    assert(summary_count(one_news, "value_order_count") == 1U);
    assert(one_news_processed == 1U);
}

void test_news_impulse_bounded_rechecks() {
    const std::filesystem::path temporary =
        std::filesystem::temp_directory_path();
    std::uint64_t processed = 0;
    const SummaryRow repeated_news = run_value_case(
        temporary / "dlob_fragmented_value_news_recheck.csv",
        0.10, 11'200.0, 3, 16, &processed,
        dlob::FragmentedValueTriggerMode::NewsImpulse,
        1.0, 1.0, 1'000'000'000LL, 1);
    // Forced innovations occur at t=1 and t=2.  Each causes one immediate
    // decision.  The t=2 innovation postpones the t=1 recheck, so coincident
    // clocks cannot manufacture a duplicate third order.
    assert(summary_count(repeated_news, "value_order_count") == 2U);
    assert(processed == 2U);

    std::uint64_t recheck_processed = 0;
    const SummaryRow one_news_then_recheck = run_value_case(
        temporary / "dlob_fragmented_value_one_news_then_recheck.csv",
        0.10, 11'200.0, 2, 16, &recheck_processed,
        dlob::FragmentedValueTriggerMode::NewsImpulse,
        1.0, 1.0, 500'000'000LL, 1);
    // Fundamental news wakes once per second, while this test's value clock
    // wakes every half second.  The t=1 innovation causes an immediate order
    // and exactly one causally later recheck at t=1.5, both before termination.
    assert(summary_count(one_news_then_recheck, "value_order_count") == 2U);
    assert(recheck_processed == 2U);
}

void test_value_depth_participation_monotonic() {
    const std::filesystem::path temporary =
        std::filesystem::temp_directory_path();
    std::uint64_t quarter_processed = 0;
    std::uint64_t half_processed = 0;
    const SummaryRow quarter = run_value_case(
        temporary / "dlob_fragmented_value_participation_025.csv", 0.25,
        11'200.0, 1, 16, &quarter_processed);
    const SummaryRow half = run_value_case(
        temporary / "dlob_fragmented_value_participation_050.csv", 0.50,
        11'200.0, 1, 16, &half_processed);

    // A depth-16 calibrated opening book has 148 displayed shares on each
    // side across its ten deterministic levels.  Ceil(pD) is retained; both
    // values are exact integers here.
    assert(summary_count(quarter, "background_event_count") == 0U);
    assert(summary_count(half, "background_event_count") == 0U);
    assert(summary_count(quarter, "value_order_count") == 1U);
    assert(summary_count(half, "value_order_count") == 1U);
    // Protected market policies are non-resting.  They must not manufacture
    // a preceding CancelOwner event at every decision boundary.
    assert(quarter_processed == 1U);
    assert(half_processed == 1U);
    assert(summary_count(quarter, "value_requested_quantity") == 37U);
    assert(summary_count(half, "value_requested_quantity") == 74U);
    assert(summary_count(half, "value_requested_quantity")
           > summary_count(quarter, "value_requested_quantity"));
    assert(summary_count(quarter, "value_boundary_truncation_events") == 0U);
    assert(summary_count(half, "value_boundary_truncation_events") == 0U);
}

void test_gap_sensitive_value_participation() {
    const std::filesystem::path temporary =
        std::filesystem::temp_directory_path();
    const SummaryRow legacy = run_value_case(
        temporary / "dlob_fragmented_value_gap_legacy.csv",
        0.10, 11'200.0, 1, 16, nullptr,
        dlob::FragmentedValueTriggerMode::PeriodicGap,
        0.0, 0.0, 1'000'000'000LL, 0,
        100.0, 0.0, 1.0);
    const SummaryRow responsive = run_value_case(
        temporary / "dlob_fragmented_value_gap_responsive.csv",
        0.10, 11'200.0, 1, 16, nullptr,
        dlob::FragmentedValueTriggerMode::PeriodicGap,
        0.0, 0.0, 1'000'000'000LL, 0,
        100.0, 1.0, 0.25);

    // The opening executable gap is about 893 bps.  eta=0 retains the old
    // ceil(0.10 * 148)=15 request exactly, whereas eta=1 reaches the declared
    // 25% ceiling and requests ceil(0.25 * 148)=37 shares.
    assert(summary_count(legacy, "value_order_count") == 1U);
    assert(summary_count(responsive, "value_order_count") == 1U);
    assert(summary_count(legacy, "value_requested_quantity") == 15U);
    assert(summary_count(responsive, "value_requested_quantity") == 37U);
}

void test_subunit_value_participation_stops_before_reflection() {
    const SummaryRow depleted = run_value_case(
        std::filesystem::temp_directory_path()
            / "dlob_fragmented_value_stops_before_reflection.csv",
        0.50,
        11'200.0,
        5,
        1);

    // Unit opening depth produces twelve represented ask shares after the
    // deterministic ten-level depth shape.  Successive 50% decisions request
    // 6, 3 and 2 shares, leaving the protected final share.  The next signal
    // submits no order instead of requesting that share on every boundary.
    assert(summary_count(depleted, "value_order_count") == 3U);
    assert(summary_count(depleted, "value_requested_quantity") == 11U);
    assert(summary_count(depleted, "value_boundary_truncation_events") == 0U);
}

void test_value_boundary_source_attribution() {
    const SummaryRow full = run_value_case(
        std::filesystem::temp_directory_path()
            / "dlob_fragmented_value_boundary.csv",
        1.0);

    assert(summary_count(full, "background_event_count") == 0U);
    assert(summary_count(full, "value_order_count") == 1U);
    assert(summary_count(full, "value_requested_quantity") == 148U);

    assert(summary_count(full, "removal_boundary_truncation_events") == 1U);
    assert(summary_count(full, "removal_boundary_truncated_quantity") == 1U);
    assert(summary_count(full, "market_boundary_truncation_events") == 1U);
    assert(summary_count(full, "market_boundary_truncated_quantity") == 1U);
    assert(summary_count(full, "value_boundary_truncation_events") == 1U);
    assert(summary_count(full, "value_boundary_truncated_quantity") == 1U);

    assert(summary_count(full, "cancel_boundary_truncation_events") == 0U);
    assert(summary_count(full, "cancel_boundary_truncated_quantity") == 0U);
    assert(summary_count(full, "background_boundary_truncation_events") == 0U);
    assert(summary_count(full, "background_boundary_truncated_quantity") == 0U);
    assert(summary_count(full, "other_boundary_truncation_events") == 0U);
    assert(summary_count(full, "other_boundary_truncated_quantity") == 0U);
}

} // namespace

int main(int argc, char** argv) {
    assert(MPI_Init(&argc, &argv) == MPI_SUCCESS);
    // The checked-in empirical config is CRLF.  Its final optional column
    // must be parsed rather than silently falling back to 0.05.
    const std::vector<dlob::MultiAssetBookConfig> parsed_config =
        dlob::load_multi_asset_book_configs(
            source_root() / "config" / "qqq_aapl_msft_amzn_20200130.csv");
    assert(parsed_config.size() == 4U);
    assert(std::abs(parsed_config[0].quote_improvement_probability
                    - 0.20942932818264257) < 1.0e-12);
    assert(std::abs(parsed_config[3].quote_improvement_probability
                    - 0.1020482990298574) < 1.0e-12);

    // A symbol must retain the same stochastic stream when calibration writes
    // a smaller subset and consequently assigns a different local book_id.
    dlob::SequentialMultiAssetConfig stream_config;
    stream_config.seed = 424242;
    dlob::MultiAssetBookConfig qqq_stream_book = parsed_config[0];
    dlob::MultiAssetBookConfig aapl_stream_book = parsed_config[1];
    for (dlob::MultiAssetBookConfig* book : {
             &qqq_stream_book, &aapl_stream_book}) {
        book->data_dir = (source_root() / book->data_dir).string();
        book->hawkes_rates_file = (
            source_root() / book->hawkes_rates_file).string();
    }
    const dlob::BackgroundHawkesConfig qqq_as_book_zero =
        dlob::make_multi_asset_background_config(
            stream_config, qqq_stream_book, 0);
    const dlob::BackgroundHawkesConfig qqq_as_book_seventeen =
        dlob::make_multi_asset_background_config(
            stream_config, qqq_stream_book, 17);
    const dlob::BackgroundHawkesConfig aapl_stream =
        dlob::make_multi_asset_background_config(
            stream_config, aapl_stream_book, 0);
    assert(qqq_as_book_zero.seed == qqq_as_book_seventeen.seed);
    assert(qqq_as_book_zero.seed != aapl_stream.seed);

    const std::filesystem::path output = std::filesystem::temp_directory_path()
        / "dlob_fragmented_calibration_summary.csv";
    const dlob::FragmentedMpiResult baseline = run_with_summary(output, false);
    assert(baseline.asset_count == 4);
    assert(baseline.lob_count == 4U);

    std::ifstream input(output);
    assert(input);
    std::string header;
    assert(std::getline(input, header));
    assert(header.find("mean_spread_ticks") != std::string::npos);
    const std::vector<std::string> columns = split(header);
    const auto column_index = [&](const std::string& name) {
        const auto found = std::find(columns.begin(), columns.end(), name);
        assert(found != columns.end());
        return static_cast<std::size_t>(std::distance(columns.begin(), found));
    };
    const std::size_t sample_count_column = column_index("sample_count");
    const std::size_t expected_count_column = column_index("expected_sample_count");
    const std::size_t two_sided_column = column_index("two_sided_sample_fraction");
    const std::size_t structurally_valid_column = column_index("structurally_valid");
    const std::size_t total_boundary_column =
        column_index("removal_boundary_truncation_events");
    const std::size_t background_boundary_column =
        column_index("background_boundary_truncation_events");
    const std::size_t value_order_column = column_index("value_order_count");
    const std::size_t value_boundary_column =
        column_index("value_boundary_truncation_events");
    const std::size_t other_boundary_column =
        column_index("other_boundary_truncation_events");
    (void)column_index("removal_boundary_truncated_quantity");
    (void)column_index("background_boundary_truncated_quantity");
    (void)column_index("value_requested_quantity");
    (void)column_index("value_boundary_truncated_quantity");
    (void)column_index("other_boundary_truncated_quantity");
    (void)column_index("market_boundary_truncation_events");
    (void)column_index("market_boundary_truncated_quantity");
    (void)column_index("cancel_boundary_truncation_events");
    (void)column_index("cancel_boundary_truncated_quantity");
    int rows = 0;
    std::string row;
    while (std::getline(input, row)) {
        const std::vector<std::string> values = split(row);
        assert(values.size() == columns.size());
        assert(values[sample_count_column] == "2");
        assert(values[expected_count_column] == "2");
        assert(values[two_sided_column] == "1");
        assert(values[structurally_valid_column] == "1");
        assert(values[value_order_column] == "0");
        assert(std::stoull(values[total_boundary_column])
               == std::stoull(values[background_boundary_column])
                  + std::stoull(values[value_boundary_column])
                  + std::stoull(values[other_boundary_column]));
        ++rows;
    }
    assert(rows == 4);

    // The resolved explicit defaults must reproduce the historical one-second
    // local-MM schedule exactly.  This guards the new calibration controls
    // against silently changing the reference model.
    const dlob::FragmentedMpiResult explicit_defaults = run_with_summary(
        output, false, 1'000'000'000LL, 0.30, 1.0);
    assert(explicit_defaults.state_hash == baseline.state_hash);
    assert(explicit_defaults.collective_calls == baseline.collective_calls);
    assert(explicit_defaults.local_mm_refresh_boundaries
           == baseline.local_mm_refresh_boundaries);

    // eta=0 must bypass spread scaling exactly.  Changing only the otherwise
    // unused cap therefore leaves every order, fill, and canonical state bit
    // unchanged for a non-zero historical repair probability.
    const dlob::FragmentedMpiResult legacy_capped = run_with_summary(
        output, false, 0, 0.30, 1.0, false, 0,
        1'000'000'000LL, 1'000'000'000LL,
        0.50, 0.0, 0.50);
    const dlob::FragmentedMpiResult legacy_uncapped = run_with_summary(
        output, false, 0, 0.30, 1.0, false, 0,
        1'000'000'000LL, 1'000'000'000LL,
        0.50, 0.0, 1.00);
    assert(legacy_capped.state_hash == legacy_uncapped.state_hash);
    assert(legacy_capped.processed_orders
           == legacy_uncapped.processed_orders);
    assert(legacy_capped.trades == legacy_uncapped.trades);

    // A half-second local quote cadence creates more local refreshes but does
    // not add a shared-risk reduction between the one-second global windows.
    const dlob::FragmentedMpiResult half_second_local = run_with_summary(
        output, false, 500'000'000LL, 0.30, 1.0);
    assert(half_second_local.local_mm_refresh_boundaries
           > baseline.local_mm_refresh_boundaries);
    assert(half_second_local.collective_calls == baseline.collective_calls);

    // Market-wide monitoring is not a causal input.  Sampling it only at the
    // end of this two-second run must preserve all simulated state while
    // removing the otherwise redundant one-second diagnostic reduction.
    // Per-asset calibration moments remain on their independent one-second
    // local clock.
    const dlob::FragmentedMpiResult sparse_global_metrics = run_with_summary(
        output, false, 0, 0.30, 1.0, false, 2'000'000'000LL);
    assert(sparse_global_metrics.state_hash == baseline.state_hash);
    assert(sparse_global_metrics.processed_orders == baseline.processed_orders);
    assert(sparse_global_metrics.trades == baseline.trades);
    assert(sparse_global_metrics.local_mm_refresh_boundaries
           == baseline.local_mm_refresh_boundaries);
    assert(sparse_global_metrics.collective_calls < baseline.collective_calls);

    const dlob::FragmentedMpiResult mispriced_without_policy = run_with_summary(
        output, false, 0, 0.30, 1.0, true);
    const dlob::FragmentedMpiResult clustered = run_with_summary(
        output, true, 0, 0.30, 1.0, true);
    // With QQQ value above the ask, a zero-threshold policy submits a
    // deterministic buy at the first decision boundary.  Compare against the
    // same mispriced configuration without the policy.
    assert(clustered.state_hash != mispriced_without_policy.state_hash);

    // The shared-risk window is an MPI execution/model cadence, not the
    // fundamental-news or local value-decision clock.  With the shared maker
    // disabled and all local/diagnostic clocks held at one second, halving
    // only the global window must not change financial state.
    const dlob::FragmentedMpiResult clustered_half_second_window =
        run_with_summary(
            output, true, 1'000'000'000LL, 0.30, 1.0, true,
            1'000'000'000LL,
            500'000'000LL, 1'000'000'000LL);
    assert(clustered_half_second_window.state_hash == clustered.state_hash);
    assert(clustered_half_second_window.processed_orders
           == clustered.processed_orders);
    assert(clustered_half_second_window.trades == clustered.trades);
    assert(clustered_half_second_window.local_mm_refresh_boundaries
           == clustered.local_mm_refresh_boundaries);
    assert(clustered_half_second_window.windows > clustered.windows);
    assert(clustered_half_second_window.collective_calls
           == clustered.collective_calls);

    test_value_depth_participation_monotonic();
    test_gap_sensitive_value_participation();
    test_subunit_value_participation_stops_before_reflection();
    test_value_boundary_source_attribution();
    test_news_impulse_value_trigger();
    test_news_impulse_bounded_rechecks();
    assert(MPI_Finalize() == MPI_SUCCESS);
    return 0;
}
