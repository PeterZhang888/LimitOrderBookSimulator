#pragma once

#include "mpi/MpiCompat.hpp"
#include "simulation/MultiAssetTypes.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dlob {

// PeriodicGap preserves the historical once-per-clock value decision.
// NewsImpulse reacts only after the asset's latent fundamental actually
// changes.  The policy's bounded recheck count may add finitely many causally
// delayed decisions at later value clocks; a newer price innovation replaces,
// rather than duplicates, the pending sequence.  This represents staged
// execution without an unbounded train of IOC orders against a stale gap.
enum class FragmentedValueTriggerMode : std::uint8_t {
    PeriodicGap = 0,
    NewsImpulse = 1,
};

// A compact behavioural policy shared by one or more empirical liquidity
// clusters.  It deliberately contains only the parameters that the
// coarse-grained fragmented simulator actually uses for its value agent.
// Per-symbol ITCH background and opening-book inputs remain in
// MultiAssetBookConfig; this vector is not a second per-symbol calibration.
struct FragmentedValueAgentPolicy {
    bool enabled = true;
    double threshold_bps = 8.0;
    // Fraction of total contemporaneous displayed depth on the opposite side
    // submitted as a market order protected at the perceived fundamental.
    // A value in (0, 1] gives the policy the same economic meaning across thin
    // and deep books.  Zero retains the legacy fixed-share fallback
    // for old synthetic benchmark inputs only.
    double depth_participation = 0.0;
    int order_quantity = 50;
    // Appended to preserve existing four-field aggregate initializers.
    FragmentedValueTriggerMode trigger_mode =
        FragmentedValueTriggerMode::PeriodicGap;
    // Used only with NewsImpulse.  Zero is the one-shot model; a positive
    // value permits exactly this many later rechecks while a valuation gap
    // remains executable.  It is bounded at input validation.
    int maximum_news_rechecks = 0;
    // Log--log slope that increases displayed-depth participation as the
    // executable valuation gap moves beyond threshold_bps.  Zero is an
    // explicit legacy mode: it bypasses the scaling calculation and retains
    // the historical depth_participation bit-for-bit.
    double gap_elasticity = 0.0;
    // Hard participation ceiling for gap-sensitive sizing.  Keeping the
    // default at one preserves every existing policy and aggregate
    // initializer; calibrated policies may select a lower, still-positive
    // ceiling no smaller than depth_participation.
    double maximum_depth_participation = 1.0;
};

// A coarse-grained distributed market model.  Each logical asset owns exactly
// one displayed limit-order book and is assigned to one MPI rank.  Price-time
// matching remains local; only the shared firm's global risk and aggregate
// diagnostics are reduced at decision boundaries.
struct FragmentedMpiConfig {
    int duration_seconds = 60;
    int asset_count = 101;
    std::int64_t decision_window_ns = 1'000'000'000LL;
    std::uint64_t seed = 20200130;
    int tick_size = 100;
    double initial_depth_scale = 1.0;
    std::vector<MultiAssetBookConfig> asset_configs;
    // Empty selects the exact legacy background construction.  The
    // queue-reactive CLI supplies one fully audited configuration per asset;
    // keeping it separate from empirical book inputs prevents a partially
    // specified new model from silently falling back to legacy defaults.
    std::vector<BackgroundHawkesConfig> background_configs;
    std::string background_model = "legacy";

    // A global activity adjustment applied to the baseline intensity in each
    // asset's ITCH-derived Hawkes-rate file.  It is deliberately separate
    // from mu so a calibration candidate does not silently regenerate rates
    // and cancel its own effect.
    double hawkes_activity_scale = 0.30;
    bool enable_local_market_makers = true;
    // The legacy local-MM controls now govern a separately owned, sparse
    // repair policy.  It may add a quote only when a side is missing, a top
    // queue is shallow, or an eligible wide spread is improved; it never
    // relocates or relabels owner-zero ITCH-derived liquidity.
    // They may refresh more or less frequently than the global shared-risk
    // boundary.  Extra local refreshes do not communicate.
    // Zero means "match decision_window_ns", preserving the historical
    // one-local-refresh-per-global-boundary default for direct API callers.
    std::int64_t local_mm_interval_ns = 0;
    double local_mm_quantity_multiplier = 1.0;
    // Dedicated local-revision inside-spread policy.  It must not reuse the
    // empirical background-addition probability from each asset config.
    double local_mm_improvement_probability = 0.0;
    // When positive, make the local repair probability increase with the
    // contemporaneous spread relative to that asset's empirical target:
    // p_eff=min(p_max,p_0 max(s/s_target,1)^eta).  The explicit eta=0 branch
    // is the exact historical constant-probability policy.
    double local_mm_spread_elasticity = 0.0;
    double local_mm_max_improvement_probability = 1.0;
    bool enable_value_agents = true;
    // Value decisions are local and therefore use their own deterministic
    // clock rather than the MPI shared-risk decision window.  Keeping the
    // default at one second preserves the historical model when the global
    // window also has its legacy one-second value.
    std::int64_t value_agent_interval_ns = 1'000'000'000LL;
    double value_threshold_bps = 8.0;
    double value_depth_participation = 0.0;
    int value_order_quantity = 50;
    // Empty preserves the one-global-policy legacy behaviour.  Otherwise the
    // vector is aligned with asset_configs and permits a common calibrated
    // policy for every member of a liquidity cluster.
    std::vector<FragmentedValueAgentPolicy> value_agent_policies;

    bool enable_shared_market_maker = true;
    // When false, the shared supplier remains present and retains its local
    // inventory skew, but the market-wide capacity multiplier is fixed at
    // one.  This is the clean mechanism control for the stress experiment.
    bool enable_global_shared_capacity = true;
    int shared_quote_quantity = 200;
    int shared_quote_levels = 1;
    // In empirical-universe mode a common fixed share size is inappropriate
    // across very different stocks.  This option scales a symbol's own
    // calibrated local quote size instead.
    bool shared_quote_relative_to_asset = false;
    double shared_quote_multiplier = 1.0;
    // Asset-specific inventory skew and market-wide capacity are deliberately
    // separate.  Changing the latter must not silently change local quoting.
    double shared_local_inventory_scale = 100.0;
    // Limit is in beta-weighted inventory units per logical asset.  The firm's
    // global limit is this value times asset_count.
    double shared_global_risk_limit_per_asset = 100.0;
    // Capacity activation point u_0 in phi(u;u_0).  It is a scenario
    // assumption, not a calibrated value-agent threshold.
    double shared_capacity_threshold = 0.5;

    bool enable_shock = false;
    std::int64_t shock_time_ns = 30'000'000'000LL;
    double shock_asset_fraction = 0.01;
    // A positive count overrides shock_asset_fraction (one-book propagation
    // uses one; the distributed stress normally leaves this at zero).
    int shock_target_count = 0;
    // Independent from seed so path variation does not change treatment-set
    // composition.  Cluster IDs, when supplied, make the mask stratified.
    std::uint64_t shock_target_seed = 20200130;
    std::vector<int> shock_cluster_ids;
    int shock_quantity_per_asset = 5'000;
    // When positive, overrides the fixed quantity with this multiple of the
    // contemporaneous pre-intervention bid-side top-of-book depth.  The
    // intervention is a sell order, so this is its immediately executable
    // liquidity rather than an opening or bid-plus-ask proxy.
    double shock_top_depth_multiple = 0.0;

    std::string metrics_csv;
    // Global market-wide monitoring is an observation, not a causal input.
    // It may therefore be sampled less frequently than shared-risk updates.
    // Zero means "match decision_window_ns", preserving historical output.
    // A coarser value removes avoidable diagnostic collectives without
    // changing any order, fill, inventory, random stream, or state hash.
    std::int64_t global_metrics_interval_ns = 0;
    // Optional per-asset fixed-clock moment summaries.  They have the same
    // one-book interpretation as the corresponding ITCH targets and may be
    // sampled less often than the decision window for a market-wide
    // distributional check.
    std::string asset_summary_csv;
    std::int64_t asset_summary_interval_ns = 1'000'000'000LL;
    // Optional provenance artifact.  The deterministic target mask is written
    // for both shock and matched-control runs, making the treated asset set
    // explicit rather than implicit in a seed/hash calculation.
    std::string shock_targets_csv;
};

struct FragmentedMpiResult {
    int world_size = 1;
    int asset_count = 0;
    std::uint64_t lob_count = 0;
    std::uint64_t windows = 0;
    // Local-MM schedule boundaries, including t=0 but excluding the terminal
    // boundary where no new quote can execute.  This makes cadence auditable
    // without conflating it with MPI collective count.
    std::uint64_t local_mm_refresh_boundaries = 0;
    std::uint64_t processed_orders = 0;
    std::uint64_t trades = 0;
    std::uint64_t collective_calls = 0;
    // Deterministic treatment-set size, retained in both stress and matched
    // control paths.  It is not the count of orders actually injected.
    std::uint64_t shock_target_assets = 0;
    // Backward-compatible alias for older result consumers.  New analyses
    // should use shock_target_assets together with the shock-enabled flag.
    std::uint64_t shock_assets = 0;
    std::uint64_t withdrawal_windows = 0;
    double wall_seconds = 0.0;
    double max_initialization_seconds = 0.0;
    double min_compute_seconds = 0.0;
    double mean_compute_seconds = 0.0;
    double max_compute_seconds = 0.0;
    double compute_imbalance = 1.0;
    std::uint64_t min_orders_per_rank = 0;
    double mean_orders_per_rank = 0.0;
    std::uint64_t max_orders_per_rank = 0;
    std::uint64_t min_books_per_rank = 0;
    double mean_books_per_rank = 0.0;
    std::uint64_t max_books_per_rank = 0;
    double max_communication_seconds = 0.0;
    double communication_fraction = 0.0;
    double final_shared_gross_exposure = 0.0;
    double final_shared_utilization = 0.0;
    double minimum_shared_quote_scale = 1.0;
    double peak_affected_fraction = 0.0;
    double peak_mean_spread_bps = 0.0;
    double final_mean_spread_bps = 0.0;
    double final_mean_top_depth = 0.0;
    double final_affected_shocked_fraction = 0.0;
    double final_affected_unshocked_fraction = 0.0;
    double peak_affected_unshocked_fraction = 0.0;
    double minimum_two_sided_book_fraction = 1.0;
    std::uint64_t shock_executed_quantity = 0;
    std::uint64_t shock_shared_mm_quantity = 0;
    std::uint64_t shock_requested_quantity = 0;
    std::uint64_t state_hash = 0;
};

class FragmentedMpiSimulator {
public:
    FragmentedMpiSimulator(MPI_Comm communicator, FragmentedMpiConfig config);
    ~FragmentedMpiSimulator();

    FragmentedMpiSimulator(const FragmentedMpiSimulator&) = delete;
    FragmentedMpiSimulator& operator=(const FragmentedMpiSimulator&) = delete;
    FragmentedMpiSimulator(FragmentedMpiSimulator&&) noexcept;
    FragmentedMpiSimulator& operator=(FragmentedMpiSimulator&&) noexcept;

    [[nodiscard]] FragmentedMpiResult run();

private:
    class Impl;
    Impl* impl_ = nullptr;
};

} // namespace dlob
