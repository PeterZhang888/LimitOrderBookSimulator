#include "simulation/DistributedMarketSimulator.hpp"

#include "common/AgentUtilities.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/LimitOrderBook.hpp"
#include "simulation/AssetMomentAccumulator.hpp"
#include "simulation/DeterministicFundamentalProcess.hpp"
#include "simulation/QuotePlacement.hpp"
#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef LOB_HAS_OPENMP
#define LOB_HAS_OPENMP 0
#endif

#if LOB_HAS_OPENMP
#include <omp.h>
#endif

namespace dlob {
namespace {

constexpr std::int64_t nanoseconds_per_second = 1'000'000'000LL;
constexpr std::int64_t fundamental_news_interval_ns = nanoseconds_per_second;
constexpr std::int64_t agent_latency_ns = 5'000;
constexpr std::int32_t shared_market_maker_owner = 900'001;
constexpr StableEntityId local_market_maker_entity_base = 0x0008'0000ULL;
constexpr StableEntityId shared_maker_entity = 0x0009'0000ULL;
constexpr StableEntityId value_entity_base = 0x000a'0000ULL;
constexpr double risk_fixed_point_scale = 1'000'000.0;
constexpr std::size_t global_metric_field_count = 38U;
constexpr std::size_t cluster_metric_field_count = 7U;

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}

int checked_bytes(std::size_t count, std::size_t width, const char* label) {
    if (width != 0U
        && count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / width) {
        throw std::overflow_error(std::string(label) + " exceeds MPI int byte count");
    }
    return static_cast<int>(count * width);
}

std::int64_t checked_duration_ns(int seconds) {
    if (seconds <= 0
        || static_cast<std::int64_t>(seconds)
            > std::numeric_limits<std::int64_t>::max() / nanoseconds_per_second) {
        throw std::invalid_argument("duration must be positive and representable");
    }
    return static_cast<std::int64_t>(seconds) * nanoseconds_per_second;
}

std::int64_t checked_add_time(std::int64_t time_ns, std::int64_t delta_ns) {
    if (delta_ns < 0
        || time_ns > std::numeric_limits<std::int64_t>::max() - delta_ns) {
        throw std::overflow_error("simulation timestamp overflow");
    }
    return time_ns + delta_ns;
}

int action_priority(OrderAction action) {
    switch (action) {
        case OrderAction::CancelOwner: return 0;
        case OrderAction::ConservedLimit: return 1;
        case OrderAction::Limit: return 2;
        case OrderAction::Market: return 3;
        case OrderAction::CancelAtDistance: return 4;
    }
    return 4;
}

bool order_less(const OrderMessage& left, const OrderMessage& right) {
    if (left.arrival_time_ns != right.arrival_time_ns) {
        return left.arrival_time_ns < right.arrival_time_ns;
    }
    const int left_priority = action_priority(left.action);
    const int right_priority = action_priority(right.action);
    if (left_priority != right_priority) return left_priority < right_priority;
    if (left.book_id != right.book_id) return left.book_id < right.book_id;
    if (left.agent_kind != right.agent_kind) return left.agent_kind < right.agent_kind;
    if (left.sequence != right.sequence) return left.sequence < right.sequence;
    return left.tie_breaker < right.tie_breaker;
}

std::int64_t checked_int64(__int128 value, const char* label) {
    if (value < static_cast<__int128>(std::numeric_limits<std::int64_t>::min())
        || value > static_cast<__int128>(
            std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(std::string(label) + " overflow");
    }
    return static_cast<std::int64_t>(value);
}

std::uint64_t checked_uint64(unsigned __int128 value, const char* label) {
    if (value > static_cast<unsigned __int128>(
            std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error(std::string(label) + " overflow");
    }
    return static_cast<std::uint64_t>(value);
}

struct LocalBook {
    BookId book_id = 0;
    LimitOrderBook lob;
    std::uint64_t trade_count = 0;
    std::uint64_t processed_orders = 0;

    LocalBook(BookId id, int tick_size) : book_id(id), lob(tick_size, id) {}
};

struct LocalAsset {
    BookId asset_id = 0;
    StableEntityId stochastic_stream_id = 0;
    MultiAssetBookConfig config;
    double fundamental_value_ticks = 0.0;
    double fundamental_log_variance = 0.0;
    // Boundary-local signal consumed immediately by the news-impulse value
    // scheduler.  It records an actual price change, not merely a clock tick.
    bool fresh_fundamental_news = false;
    // The bounded staged-response mode may revisit a surviving valuation gap
    // once, and only once, at a later value clock.  A new price innovation
    // moves this due time forward instead of creating a second pending task.
    int remaining_value_rechecks = 0;
    std::int64_t value_recheck_due_ns = 0;
    BackgroundHawkesAgent background;
    BackgroundHawkesStream hawkes;
    LocalBook book;
    std::vector<OrderMessage> pending_orders;
    std::int64_t shared_inventory = 0;
    std::int64_t shared_cash_ticks = 0;
    std::uint64_t shared_buy_quantity = 0;
    std::uint64_t shared_sell_quantity = 0;
    std::uint64_t shared_fill_count = 0;
    std::int64_t value_inventory = 0;
    std::uint64_t shock_executed_quantity = 0;
    std::uint64_t shock_shared_mm_quantity = 0;
    std::uint64_t shock_local_mm_quantity = 0;
    std::uint64_t shock_value_agent_quantity = 0;
    std::uint64_t shock_background_quantity = 0;
    std::uint64_t shock_other_quantity = 0;
    std::uint64_t shock_requested_quantity = 0;
    // 0=not injected, 1=aggressive buy, 2=aggressive sell.
    std::int32_t shock_side_code = 0;
    std::int64_t shock_pre_shared_inventory = 0;
    bool shock_pre_inventory_recorded = false;
    bool shock_injected = false;
    double shared_requested_bid_depth = 0.0;
    double shared_requested_ask_depth = 0.0;
    double shared_requested_quote_depth = 0.0;
    double shared_risk_reducing_requested_quote_depth = 0.0;
    double shared_risk_increasing_requested_quote_depth = 0.0;
    std::uint64_t value_order_count = 0;
    std::uint64_t value_requested_quantity = 0;
    std::uint64_t background_limit_requested_quantity = 0;
    std::uint64_t background_limit_resting_quantity = 0;
    std::uint64_t background_market_requested_quantity = 0;
    std::uint64_t background_market_executed_quantity = 0;
    std::uint64_t background_cancel_requested_quantity = 0;
    std::uint64_t background_cancel_effective_quantity = 0;
    std::uint64_t removal_boundary_truncation_events = 0;
    std::uint64_t removal_boundary_truncated_quantity = 0;
    std::uint64_t market_boundary_truncation_events = 0;
    std::uint64_t market_boundary_truncated_quantity = 0;
    std::uint64_t cancel_boundary_truncation_events = 0;
    std::uint64_t cancel_boundary_truncated_quantity = 0;
    std::uint64_t background_boundary_truncation_events = 0;
    std::uint64_t background_boundary_truncated_quantity = 0;
    std::uint64_t value_boundary_truncation_events = 0;
    std::uint64_t value_boundary_truncated_quantity = 0;
    std::uint64_t other_boundary_truncation_events = 0;
    std::uint64_t other_boundary_truncated_quantity = 0;
    std::uint64_t local_quote_revision_requested_quantity = 0;
    std::uint64_t local_quote_revision_moved_quantity = 0;
    std::uint64_t local_quote_revision_no_donor_events = 0;
    double baseline_mean_spread_bps = 0.0;
    double baseline_top_depth = 0.0;
    detail::AssetMomentAccumulator calibration_moments;
    // Diagnostic-only counters used to construct a measured partition for a
    // later run.  They do not enter any market decision or random stream.
    std::uint64_t measured_processed_orders = 0;
    std::uint64_t measured_processing_nanoseconds = 0;

    LocalAsset(BookId id,
               MultiAssetBookConfig asset_config,
               const BackgroundHawkesConfig& background_config,
               int tick_size,
               std::uint64_t model_seed)
        : asset_id(id),
          stochastic_stream_id(stable_symbol_stream_id(asset_config.symbol)),
          config(std::move(asset_config)),
          fundamental_value_ticks(config.fundamental_price_ticks),
          fundamental_log_variance(
              detail::initial_fundamental_log_variance(
                  config.fundamental_log_variance_persistence,
                  config.fundamental_log_variance_std,
                  stochastic_stream_id,
                  model_seed)),
          background(background_config),
          hawkes(background_config),
          book(id, tick_size) {}
};

struct BookResultWire {
    MarketState state{};
    std::uint64_t processed_orders = 0;
    std::uint64_t trade_count = 0;
};

struct AssetResultWire {
    BookId asset_id = 0;
    std::int64_t shared_inventory = 0;
    std::int64_t shared_cash_ticks = 0;
    std::uint64_t shared_buy_quantity = 0;
    std::uint64_t shared_sell_quantity = 0;
    std::uint64_t shared_fill_count = 0;
    double shared_mark_mid_ticks = 0.0;
    double shared_liquidation_cash_change_ticks = 0.0;
    std::uint64_t shared_liquidation_unliquidated_quantity = 0;
    std::int64_t value_inventory = 0;
    std::uint64_t shock_requested_quantity = 0;
    std::int32_t shock_side_code = 0;
    std::int64_t shock_pre_shared_inventory = 0;
    double fundamental_value_ticks = 0.0;
    double fundamental_log_variance = 0.0;
    std::int64_t value_recheck_due_ns = 0;
    std::int32_t remaining_value_rechecks = 0;
};

struct AssetMomentWire {
    BookId asset_id = 0;
    std::uint64_t sample_count = 0;
    std::uint64_t invalid_sample_count = 0;
    std::uint64_t background_event_count = 0;
    std::uint64_t background_limit_requested_quantity = 0;
    std::uint64_t background_limit_resting_quantity = 0;
    std::uint64_t background_market_requested_quantity = 0;
    std::uint64_t background_market_executed_quantity = 0;
    std::uint64_t background_cancel_requested_quantity = 0;
    std::uint64_t background_cancel_effective_quantity = 0;
    std::uint64_t removal_boundary_truncation_events = 0;
    std::uint64_t removal_boundary_truncated_quantity = 0;
    std::uint64_t market_boundary_truncation_events = 0;
    std::uint64_t market_boundary_truncated_quantity = 0;
    std::uint64_t cancel_boundary_truncation_events = 0;
    std::uint64_t cancel_boundary_truncated_quantity = 0;
    std::uint64_t background_boundary_truncation_events = 0;
    std::uint64_t background_boundary_truncated_quantity = 0;
    std::uint64_t value_order_count = 0;
    std::uint64_t value_requested_quantity = 0;
    std::uint64_t value_boundary_truncation_events = 0;
    std::uint64_t value_boundary_truncated_quantity = 0;
    std::uint64_t other_boundary_truncation_events = 0;
    std::uint64_t other_boundary_truncated_quantity = 0;
    std::uint64_t local_quote_revision_requested_quantity = 0;
    std::uint64_t local_quote_revision_moved_quantity = 0;
    std::uint64_t local_quote_revision_no_donor_events = 0;
    std::array<double, 7> values{};
};

struct RankResultWire {
    std::array<unsigned long long, 9> counts{};
    unsigned long long books = 0;
    double wall_seconds = 0.0;
    double initialization_seconds = 0.0;
    double compute_seconds = 0.0;
    double communication_seconds = 0.0;
    double risk_collective_seconds = 0.0;
    double observation_collective_seconds = 0.0;
    double terminal_collective_seconds = 0.0;
    double boundary_wait_seconds = 0.0;
    double risk_overlap_work_seconds = 0.0;
    double risk_wait_after_overlap_seconds = 0.0;
};

struct AssetWorkWire {
    BookId asset_id = 0;
    std::int32_t owner_rank = 0;
    std::uint64_t processed_orders = 0;
    std::uint64_t background_events = 0;
    std::uint64_t processing_nanoseconds = 0;
};

struct BoundaryArrivalWire {
    std::int64_t time_ns = 0;
    std::uint64_t boundary_index = 0;
    std::int32_t rank = 0;
    double arrival_seconds = 0.0;
    double work_interval_seconds = 0.0;
    double collective_seconds = 0.0;
};

// One diagnostic row for one rank and one simulated interval.  The phase
// fields are exclusive: their sum, together with other_seconds, equals the
// measured interval total.  Rows stay rank-local during the session and are
// gathered only after the final simulated boundary.
struct WindowPhaseWire {
    std::int64_t start_time_ns = 0;
    std::int64_t end_time_ns = 0;
    std::uint64_t window_index = 0;
    std::int32_t rank = 0;
    double event_processing_seconds = 0.0;
    double risk_local_seconds = 0.0;
    double risk_collective_seconds = 0.0;
    double asset_moments_seconds = 0.0;
    double return_panel_seconds = 0.0;
    double local_market_maker_seconds = 0.0;
    double risk_finalize_seconds = 0.0;
    double global_metrics_local_seconds = 0.0;
    double global_metrics_collective_seconds = 0.0;
    double global_metrics_write_seconds = 0.0;
    double fundamental_seconds = 0.0;
    double shared_market_maker_seconds = 0.0;
    double news_value_agent_seconds = 0.0;
    double periodic_value_agent_seconds = 0.0;
    double other_seconds = 0.0;
    double total_window_seconds = 0.0;
};

static_assert(std::is_trivially_copyable_v<BookResultWire>);
static_assert(std::is_trivially_copyable_v<AssetResultWire>);
static_assert(std::is_trivially_copyable_v<AssetMomentWire>);
static_assert(std::is_trivially_copyable_v<RankResultWire>);
static_assert(std::is_trivially_copyable_v<AssetWorkWire>);
static_assert(std::is_trivially_copyable_v<BoundaryArrivalWire>);
static_assert(std::is_trivially_copyable_v<WindowPhaseWire>);

struct AggregateMetricSums {
    double spread_sum_bps = 0.0;
    double top_depth_sum = 0.0;
    double affected_asset_count = 0.0;
    double two_sided_book_count = 0.0;
    double shocked_asset_count = 0.0;
    double unshocked_asset_count = 0.0;
    double affected_shocked_asset_count = 0.0;
    double affected_unshocked_asset_count = 0.0;
    double shocked_spread_sum_bps = 0.0;
    double unshocked_spread_sum_bps = 0.0;
    double shocked_two_sided_asset_count = 0.0;
    double unshocked_two_sided_asset_count = 0.0;
    double shocked_top_depth_sum = 0.0;
    double unshocked_top_depth_sum = 0.0;
    double unshocked_shared_quote_depth_sum = 0.0;
    double shocked_shared_inventory_sum = 0.0;
    double value_order_count = 0.0;
    double value_requested_quantity = 0.0;
    double shared_requested_quote_depth_sum = 0.0;
    double shared_risk_reducing_quote_depth_sum = 0.0;
    double shared_risk_increasing_quote_depth_sum = 0.0;
    double shared_resting_quote_depth_sum = 0.0;
    double shared_risk_reducing_resting_depth_sum = 0.0;
    double shared_risk_increasing_resting_depth_sum = 0.0;
    double unshocked_shared_resting_depth_sum = 0.0;
    double shared_active_asset_count = 0.0;
    double shared_two_sided_active_asset_count = 0.0;
    double shocked_bid_top_depth_sum = 0.0;
    double shocked_shared_bid_resting_depth_sum = 0.0;
    double shared_best_bid_depth_sum = 0.0;
    double shared_best_ask_depth_sum = 0.0;
    double shared_at_best_bid_asset_count = 0.0;
    double shared_at_best_ask_asset_count = 0.0;
    double shared_nonzero_inventory_asset_count = 0.0;
    double shared_absolute_inventory_sum = 0.0;
    double shocked_shared_absolute_inventory_sum = 0.0;
    double shared_requested_active_asset_count = 0.0;
    double shared_requested_two_sided_asset_count = 0.0;
};

struct ScheduledQuoteDepth {
    std::uint64_t bid = 0;
    std::uint64_t ask = 0;
};

struct AggregateMetrics {
    double mean_spread_bps = 0.0;
    double mean_top_depth = 0.0;
    double affected_fraction = 0.0;
    double two_sided_book_fraction = 1.0;
    double affected_shocked_fraction = 0.0;
    double affected_unshocked_fraction = 0.0;
    double shocked_mean_spread_bps = 0.0;
    double unshocked_mean_spread_bps = 0.0;
    double shocked_mean_top_depth = 0.0;
    double unshocked_mean_top_depth = 0.0;
};

struct GlobalObservationFrame {
    std::int64_t time_ns = 0;
    std::array<double, global_metric_field_count> local{};
    double shared_gross_exposure = 0.0;
    double shared_utilization = 0.0;
    double shared_quote_scale = 1.0;
    std::size_t risk_frame_index = std::numeric_limits<std::size_t>::max();
};

struct ClusterObservationFrame {
    std::int64_t time_ns = 0;
    std::vector<double> local;
};

struct RiskObservationFrame {
    std::int64_t time_ns = 0;
    long long local_fixed_exposure = 0;
    double local_uncoupled_scale_sum = 0.0;
};

int bounded_positive_quantity(double raw_quantity, const char* label) {
    if (!std::isfinite(raw_quantity) || raw_quantity <= 0.0) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    // Bound before rounding so an unusually liquid symbol or an accidental
    // multiplier cannot overflow the OrderMessage integer quantity.
    const double bounded = std::min(
        raw_quantity, static_cast<double>(std::numeric_limits<int>::max()));
    return std::max(1, static_cast<int>(std::llround(bounded)));
}

} // namespace

class DistributedMarketSimulator::Impl {
public:
    Impl(MPI_Comm communicator, SimulationConfig config)
        : communicator_(communicator),
          config_(std::move(config)),
          end_time_ns_(checked_duration_ns(config_.duration_seconds)) {
        check_mpi(MPI_Comm_rank(communicator_, &rank_), "MPI_Comm_rank");
        check_mpi(MPI_Comm_size(communicator_, &world_size_), "MPI_Comm_size");
        if (config_.local_mm_interval_ns == 0) {
            config_.local_mm_interval_ns = config_.decision_window_ns;
        }
        if (config_.global_metrics_interval_ns == 0) {
            config_.global_metrics_interval_ns = config_.decision_window_ns;
        }
        validate_config();
#if LOB_HAS_OPENMP
        omp_set_dynamic(0);
#endif
        default_value_agent_policy_.enabled = true;
        default_value_agent_policy_.threshold_bps = config_.value_threshold_bps;
        default_value_agent_policy_.depth_participation =
            config_.value_depth_participation;
        default_value_agent_policy_.order_quantity = config_.value_order_quantity;
        // Select the deterministic intervention set before partitioning so a
        // measured-cost file may explicitly weight those assets.
        select_shock_assets();
        initialize_asset_owners();
        initialize_shared_capacity_weights();
        if (!config_.shock_cluster_ids.empty()) {
            cluster_count_ = 1 + *std::max_element(
                config_.shock_cluster_ids.begin(),
                config_.shock_cluster_ids.end());
        }
    }

    ~Impl() {
#if LOB_HAS_MPI_PERSISTENT_COLLECTIVES
        if (risk_request_ != MPI_REQUEST_NULL) {
            (void)MPI_Request_free(&risk_request_);
        }
#endif
    }

    SimulationResult run() {
#if LOB_HAS_OPENMP
        if (config_.persistent_openmp_team && config_.worker_threads > 1) {
            SimulationResult result;
            std::exception_ptr failure;
#pragma omp parallel num_threads(config_.worker_threads) shared(result, failure)
            {
                // MPI_THREAD_FUNNELED requires all MPI calls on the process's
                // initial thread.  The master owns the session while the
                // persistent workers execute the asset tasks it creates.
#pragma omp master
                {
                    try {
                        result = run_session();
                    } catch (...) {
                        failure = std::current_exception();
                    }
                }
            }
            if (failure) std::rethrow_exception(failure);
            return result;
        }
#endif
        return run_session();
    }

private:
    template <typename Function>
    void profile_window_phase(
        double WindowPhaseWire::*field,
        Function&& function) {
        if (active_window_phase_ == nullptr) {
            function();
            return;
        }
        const double start = MPI_Wtime();
        function();
        (active_window_phase_->*field) += MPI_Wtime() - start;
    }

    template <typename Function>
    void profile_risk_phase(Function&& function) {
        if (active_window_phase_ == nullptr) {
            function();
            return;
        }
        const double start = MPI_Wtime();
        function();
        active_risk_total_seconds_ += MPI_Wtime() - start;
    }

    template <typename Function>
    void profile_global_metrics_phase(Function&& function) {
        if (active_window_phase_ == nullptr) {
            function();
            return;
        }
        const double start = MPI_Wtime();
        function();
        active_global_metrics_total_seconds_ += MPI_Wtime() - start;
    }

    void finish_window_phase_profile(double interval_start_seconds) {
        if (active_window_phase_ == nullptr) return;
        WindowPhaseWire& row = *active_window_phase_;
        row.risk_finalize_seconds = std::max(
            0.0,
            active_risk_total_seconds_
                - row.risk_local_seconds
                - row.risk_collective_seconds
                - row.asset_moments_seconds
                - row.return_panel_seconds
                - row.local_market_maker_seconds);
        row.global_metrics_write_seconds = std::max(
            0.0,
            active_global_metrics_total_seconds_
                - row.global_metrics_local_seconds
                - row.global_metrics_collective_seconds);
        row.total_window_seconds = MPI_Wtime() - interval_start_seconds;
        const double accounted =
            row.event_processing_seconds
            + row.risk_local_seconds
            + row.risk_collective_seconds
            + row.asset_moments_seconds
            + row.return_panel_seconds
            + row.local_market_maker_seconds
            + row.risk_finalize_seconds
            + row.global_metrics_local_seconds
            + row.global_metrics_collective_seconds
            + row.global_metrics_write_seconds
            + row.fundamental_seconds
            + row.shared_market_maker_seconds
            + row.news_value_agent_seconds
            + row.periodic_value_agent_seconds;
        row.other_seconds = std::max(0.0, row.total_window_seconds - accounted);
        window_phase_profiles_.push_back(row);
        active_window_phase_ = nullptr;
        active_risk_total_seconds_ = 0.0;
        active_global_metrics_total_seconds_ = 0.0;
    }

    SimulationResult run_session() {
        check_mpi(MPI_Barrier(communicator_), "MPI_Barrier(start)");
        ++collective_calls_;
        wall_start_seconds_ = MPI_Wtime();
        last_risk_completion_seconds_ = wall_start_seconds_;

        const double initialization_start = MPI_Wtime();
        initialize_local_assets();
        initialization_seconds_ = MPI_Wtime() - initialization_start;
        open_metrics_output();
        open_return_panel_output();
        update_shared_risk(0, [&]() {
            record_asset_moments(0);
            record_return_panel(0);
            schedule_local_market_makers(0, 0);
            if (config_.enable_local_market_makers) {
                ++local_mm_refresh_boundaries_;
            }
        });
        observe_global_metrics(0);
        schedule_shared_market_makers(0, 0);
        schedule_value_agents(0, 0);

        // Global boundaries are the only points at which ranks must agree on
        // the shared firm's inventory.  Local market-makers, fundamental
        // news, and value agents each wake on their own deterministic clocks
        // without an MPI collective.  Process the books over the union of all
        // clocks so no local policy inherits the communication cadence and no
        // agent observes a future state.
        std::int64_t current_ns = 0;
        std::int64_t next_global_boundary_ns = config_.decision_window_ns;
        std::int64_t next_local_refresh_ns = config_.enable_local_market_makers
            ? config_.local_mm_interval_ns
            : std::numeric_limits<std::int64_t>::max();
        std::int64_t next_fundamental_news_ns = fundamental_news_interval_ns;
        std::int64_t next_value_decision_ns = config_.enable_value_agents
            ? config_.value_agent_interval_ns
            : std::numeric_limits<std::int64_t>::max();
        std::uint64_t global_boundary_index = 1;
        std::uint64_t local_refresh_index = 1;
        std::uint64_t fundamental_news_index = 1;
        std::uint64_t value_decision_index = 1;
        std::uint64_t phase_window_index = 0;
        if (!config_.window_phase_profile_csv.empty()) {
            const std::uint64_t expected_global_windows =
                static_cast<std::uint64_t>(
                    end_time_ns_ / config_.decision_window_ns) + 1U;
            window_phase_profiles_.reserve(
                static_cast<std::size_t>(expected_global_windows));
        }
        while (current_ns < end_time_ns_) {
            const std::int64_t end_ns = std::min({
                end_time_ns_, next_global_boundary_ns,
                next_local_refresh_ns, next_fundamental_news_ns,
                next_value_decision_ns});
            std::optional<WindowPhaseWire> phase_row;
            double phase_interval_start = 0.0;
            if (!config_.window_phase_profile_csv.empty()) {
                phase_row.emplace();
                phase_row->start_time_ns = current_ns;
                phase_row->end_time_ns = end_ns;
                phase_row->window_index = ++phase_window_index;
                phase_row->rank = static_cast<std::int32_t>(rank_);
                active_window_phase_ = &*phase_row;
                active_risk_total_seconds_ = 0.0;
                active_global_metrics_total_seconds_ = 0.0;
                phase_interval_start = MPI_Wtime();
            }
            const double compute_start = MPI_Wtime();
            for_each_local_asset([&](LocalAsset& asset) {
                if (config_.asset_work_csv.empty()) {
                    process_window(asset, current_ns, end_ns);
                    return;
                }
                const std::uint64_t before = asset.book.processed_orders;
                const auto asset_start = std::chrono::steady_clock::now();
                process_window(asset, current_ns, end_ns);
                const auto asset_end = std::chrono::steady_clock::now();
                asset.measured_processed_orders +=
                    asset.book.processed_orders - before;
                const auto elapsed = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(asset_end - asset_start).count();
                if (elapsed > 0) {
                    const auto measured = static_cast<std::uint64_t>(elapsed);
                    if (measured > std::numeric_limits<std::uint64_t>::max()
                            - asset.measured_processing_nanoseconds) {
                        throw std::overflow_error(
                            "per-asset processing timer overflow");
                    }
                    asset.measured_processing_nanoseconds += measured;
                }
            });
            const double compute_elapsed = MPI_Wtime() - compute_start;
            compute_seconds_ += compute_elapsed;
            if (active_window_phase_ != nullptr) {
                active_window_phase_->event_processing_seconds +=
                    compute_elapsed;
            }

            const bool terminal_boundary = end_ns == end_time_ns_;
            // Preserve the final partial global window from the original
            // implementation: it produces a final risk/metric observation
            // even when duration is not an exact decision-window multiple.
            const bool global_boundary = terminal_boundary
                || end_ns == next_global_boundary_ns;
            const bool local_refresh = !terminal_boundary
                && end_ns == next_local_refresh_ns;
            const bool fundamental_news = !terminal_boundary
                && end_ns == next_fundamental_news_ns;
            const bool value_decision = !terminal_boundary
                && end_ns == next_value_decision_ns;
            if (global_boundary) {
                profile_risk_phase([&]() {
                    update_shared_risk(end_ns, [&]() {
                        profile_window_phase(
                            &WindowPhaseWire::asset_moments_seconds,
                            [&]() { record_asset_moments(end_ns); });
                        profile_window_phase(
                            &WindowPhaseWire::return_panel_seconds,
                            [&]() { record_return_panel(end_ns); });
                        if (local_refresh) {
                            profile_window_phase(
                                &WindowPhaseWire::local_market_maker_seconds,
                                [&]() {
                                    schedule_local_market_makers(
                                        end_ns, local_refresh_index);
                                });
                            ++local_mm_refresh_boundaries_;
                            next_local_refresh_ns = checked_add_time(
                                next_local_refresh_ns,
                                config_.local_mm_interval_ns);
                            ++local_refresh_index;
                        }
                    });
                });
                if (terminal_boundary
                    || end_ns % config_.global_metrics_interval_ns == 0) {
                    profile_global_metrics_phase(
                        [&]() { observe_global_metrics(end_ns); });
                }
            }
            if (!terminal_boundary) {
                // At a coincident one-second wake time this is exactly the
                // historical order: local maker, fundamental news, shared
                // maker, then value agent.  Separate indices keep every
                // entity's stochastic stream stable when only the MPI window
                // is changed.
                if (local_refresh && !global_boundary) {
                    profile_window_phase(
                        &WindowPhaseWire::local_market_maker_seconds,
                        [&]() {
                            schedule_local_market_makers(
                                end_ns, local_refresh_index);
                        });
                    ++local_mm_refresh_boundaries_;
                    next_local_refresh_ns = checked_add_time(
                        next_local_refresh_ns, config_.local_mm_interval_ns);
                    ++local_refresh_index;
                }
                if (fundamental_news) {
                    profile_window_phase(
                        &WindowPhaseWire::fundamental_seconds,
                        [&]() {
                            advance_fundamental_news(
                                fundamental_news_index);
                        });
                    next_fundamental_news_ns = checked_add_time(
                        next_fundamental_news_ns,
                        fundamental_news_interval_ns);
                    ++fundamental_news_index;
                }
                if (global_boundary) {
                    profile_window_phase(
                        &WindowPhaseWire::shared_market_maker_seconds,
                        [&]() {
                            schedule_shared_market_makers(
                                end_ns, global_boundary_index);
                        });
                }
                if (fundamental_news) {
                    profile_window_phase(
                        &WindowPhaseWire::news_value_agent_seconds,
                        [&]() {
                            schedule_news_impulse_value_agents(
                                end_ns, fundamental_news_index - 1U);
                        });
                }
                if (value_decision) {
                    profile_window_phase(
                        &WindowPhaseWire::periodic_value_agent_seconds,
                        [&]() {
                            schedule_value_agents(
                                end_ns, value_decision_index);
                        });
                    next_value_decision_ns = checked_add_time(
                        next_value_decision_ns,
                        config_.value_agent_interval_ns);
                    ++value_decision_index;
                }
            }
            if (global_boundary) {
                ++window_count_;
                if (!terminal_boundary && end_ns == next_global_boundary_ns) {
                    next_global_boundary_ns = checked_add_time(
                        next_global_boundary_ns, config_.decision_window_ns);
                    ++global_boundary_index;
                }
            }
            finish_window_phase_profile(phase_interval_start);
            current_ns = end_ns;
        }

        flush_buffered_observations();
        release_persistent_risk_collective();
        const std::vector<BookResultWire> books = gather_book_results();
        const std::vector<AssetResultWire> assets = gather_asset_results();
        const std::vector<AssetMomentWire> moments = gather_asset_moments();
        write_asset_summary(moments);
        write_shock_targets(assets);
        write_asset_work(gather_asset_work());
        write_boundary_arrivals(gather_boundary_arrivals());
        write_window_phase_profiles(gather_window_phase_profiles());
        compute_shared_financial_results(books, assets);

        check_mpi(MPI_Barrier(communicator_), "MPI_Barrier(finish)");
        ++collective_calls_;
        const double local_wall = MPI_Wtime() - wall_start_seconds_;
        return reduce_result(local_wall);
    }

    template <typename Function>
    void for_each_local_index(std::size_t count, Function&& function) {
#if LOB_HAS_OPENMP
        if (config_.worker_threads > 1 && count > 1U) {
            std::exception_ptr failure;
            if (omp_in_parallel() != 0) {
#pragma omp taskgroup
                {
                    for (std::size_t index = 0; index < count; ++index) {
#pragma omp task firstprivate(index) shared(failure)
                        {
                            try {
                                function(index);
                            } catch (...) {
#pragma omp critical(dlob_parallel_asset_failure)
                                {
                                    if (!failure) {
                                        failure = std::current_exception();
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                const int thread_count = std::min(
                    config_.worker_threads, static_cast<int>(count));
                if (config_.openmp_schedule
                        == OpenMpSchedule::WeightedStatic) {
                    if (local_thread_buckets_.size()
                            != static_cast<std::size_t>(thread_count)) {
                        throw std::logic_error(
                            "weighted-static OpenMP buckets are not initialized");
                    }
#pragma omp parallel num_threads(thread_count)
                    {
                        const auto& bucket = local_thread_buckets_.at(
                            static_cast<std::size_t>(omp_get_thread_num()));
                        for (const std::size_t index : bucket) {
                            try {
                                function(index);
                            } catch (...) {
#pragma omp critical(dlob_parallel_asset_failure)
                                {
                                    if (!failure) {
                                        failure = std::current_exception();
                                    }
                                }
                            }
                        }
                    }
                } else if (config_.openmp_schedule
                        == OpenMpSchedule::Guided) {
#pragma omp parallel for schedule(guided) num_threads(thread_count)
                    for (std::ptrdiff_t index = 0;
                         index < static_cast<std::ptrdiff_t>(count); ++index) {
                        try {
                            function(static_cast<std::size_t>(index));
                        } catch (...) {
#pragma omp critical(dlob_parallel_asset_failure)
                            {
                                if (!failure) {
                                    failure = std::current_exception();
                                }
                            }
                        }
                    }
                } else if (config_.openmp_schedule
                               == OpenMpSchedule::Static) {
#pragma omp parallel for schedule(static) num_threads(thread_count)
                    for (std::ptrdiff_t index = 0;
                         index < static_cast<std::ptrdiff_t>(count); ++index) {
                        try {
                            function(static_cast<std::size_t>(index));
                        } catch (...) {
#pragma omp critical(dlob_parallel_asset_failure)
                            {
                                if (!failure) {
                                    failure = std::current_exception();
                                }
                            }
                        }
                    }
                } else {
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
                    for (std::ptrdiff_t index = 0;
                         index < static_cast<std::ptrdiff_t>(count); ++index) {
                        try {
                            function(static_cast<std::size_t>(index));
                        } catch (...) {
#pragma omp critical(dlob_parallel_asset_failure)
                            {
                                if (!failure) {
                                    failure = std::current_exception();
                                }
                            }
                        }
                    }
                }
            }
            if (failure) std::rethrow_exception(failure);
            return;
        }
#endif
        for (std::size_t index = 0; index < count; ++index) function(index);
    }

    template <typename Function>
    void for_each_local_asset(Function&& function) {
        for_each_local_index(local_assets_.size(), [&](std::size_t index) {
            function(*local_assets_[index]);
        });
    }

    template <typename Function>
    void for_each_short_phase_asset(Function&& function) {
        if (config_.openmp_window_only) {
            for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
                function(*asset);
            }
            return;
        }
        for_each_local_asset(std::forward<Function>(function));
    }

    void validate_config() const {
        if (world_size_ <= 0 || config_.asset_count <= 0
            || config_.decision_window_ns <= 0
            || config_.worker_threads <= 0
            || config_.decision_window_ns > end_time_ns_
            || (config_.stochastic_baseline_normalization_horizon_ns > 0
                && config_.stochastic_baseline_normalization_horizon_ns
                    < end_time_ns_)
            || config_.local_mm_interval_ns <= 0
            || config_.value_agent_interval_ns <= 0
            || config_.global_metrics_interval_ns < config_.decision_window_ns
            || config_.global_metrics_interval_ns
                % config_.decision_window_ns != 0
            || config_.tick_size <= 0
            || !std::isfinite(config_.initial_depth_scale)
            || config_.initial_depth_scale <= 0.0
            || config_.asset_configs.size()
                != static_cast<std::size_t>(config_.asset_count)
            || !std::isfinite(config_.value_threshold_bps)
            || config_.value_threshold_bps < 0.0
            || !std::isfinite(config_.value_depth_participation)
            || config_.value_depth_participation < 0.0
            || config_.value_depth_participation > 1.0
            || config_.value_order_quantity <= 0
            || !std::isfinite(config_.hawkes_activity_scale)
            || config_.hawkes_activity_scale <= 0.0
            || !std::isfinite(config_.local_mm_quantity_multiplier)
            || config_.local_mm_quantity_multiplier <= 0.0
            || !std::isfinite(config_.local_mm_improvement_probability)
            || config_.local_mm_improvement_probability < 0.0
            || config_.local_mm_improvement_probability > 1.0
            || !std::isfinite(config_.local_mm_spread_elasticity)
            || config_.local_mm_spread_elasticity < 0.0
            || !std::isfinite(
                config_.local_mm_max_improvement_probability)
            || config_.local_mm_max_improvement_probability
                < config_.local_mm_improvement_probability
            || config_.local_mm_max_improvement_probability > 1.0
            || config_.shared_quote_quantity <= 0
            || config_.shared_quote_levels <= 0
            || config_.shared_quote_levels > 16
            || !std::isfinite(config_.shared_quote_multiplier)
            || config_.shared_quote_multiplier <= 0.0
            || !std::isfinite(config_.shared_local_inventory_scale)
            || config_.shared_local_inventory_scale <= 0.0
            || !std::isfinite(config_.shared_global_risk_limit_per_asset)
            || config_.shared_global_risk_limit_per_asset <= 0.0
            || !std::isfinite(config_.shared_capacity_threshold)
            || config_.shared_capacity_threshold < 0.0
            || config_.shared_capacity_threshold >= 1.0
            || !std::isfinite(config_.shared_minimum_quote_scale)
            || config_.shared_minimum_quote_scale < 0.0
            || config_.shared_minimum_quote_scale > 1.0
            || !std::isfinite(config_.shared_price_unit_usd)
            || config_.shared_price_unit_usd <= 0.0
            || config_.shared_terminal_fallback_distance_ticks < 0
            || !std::isfinite(config_.shock_asset_fraction)
            || config_.shock_asset_fraction <= 0.0
            || config_.shock_asset_fraction > 1.0
            || config_.shock_target_count < 0
            || config_.shock_target_count > config_.asset_count
            || config_.shock_quantity_per_asset <= 0
            || !std::isfinite(config_.shock_top_depth_multiple)
            || config_.shock_top_depth_multiple < 0.0
            || !std::isfinite(config_.shock_reference_bid_depth_multiple)
            || config_.shock_reference_bid_depth_multiple < 0.0
            || (config_.shock_top_depth_multiple > 0.0
                && config_.shock_reference_bid_depth_multiple > 0.0)) {
            throw std::invalid_argument("invalid distributed MPI configuration");
        }
        if (config_.risk_lookahead_max_windows > 0U
            && (!config_.enable_shared_market_maker
                || !config_.enable_global_shared_capacity
                || config_.shared_inventory_policy
                    != SharedMarketMakerInventoryPolicy::GrossPooled
                || !config_.buffer_global_observations)) {
            throw std::invalid_argument(
                "risk lookahead requires a globally capacity-constrained "
                "shared market maker and buffered observations");
        }
        if (!config_.window_phase_profile_csv.empty()
            && config_.use_nonblocking_risk_collective) {
            throw std::invalid_argument(
                "window phase profiling requires the blocking risk path so "
                "reported phases remain exclusive");
        }
        if (!config_.window_phase_profile_csv.empty()
            && config_.profile_boundary_wait) {
            throw std::invalid_argument(
                "window phase profiling cannot be combined with the "
                "extra boundary-wait barrier");
        }
        if (config_.persistent_openmp_team
            && config_.openmp_schedule
                != OpenMpSchedule::DynamicOne) {
            throw std::invalid_argument(
                "persistent OpenMP tasks currently require dynamic,1 mode; "
                "compare guided and static as separate non-persistent ablations");
        }
        if (config_.persistent_openmp_team
            && config_.openmp_window_only) {
            throw std::invalid_argument(
                "persistent OpenMP and window-only OpenMP are separate "
                "treatments and cannot be enabled together");
        }
#if !LOB_HAS_OPENMP
        if (config_.worker_threads != 1) {
            throw std::invalid_argument(
                "worker_threads > 1 requires an OpenMP-enabled build");
        }
#endif
        if (config_.enable_shock
            && (config_.shock_time_ns < 0 || config_.shock_time_ns >= end_time_ns_)) {
            throw std::invalid_argument("shock time must be inside the simulation");
        }
        if (!config_.asset_summary_csv.empty()
            && (config_.asset_summary_interval_ns <= 0
                || config_.asset_summary_interval_ns < config_.decision_window_ns
                || config_.asset_summary_interval_ns % config_.decision_window_ns != 0
                || end_time_ns_ % config_.asset_summary_interval_ns != 0)) {
            throw std::invalid_argument(
                "per-asset calibration summary requires an interval "
                "that is an exact multiple of the decision window and session");
        }
        if (!config_.return_panel_prefix.empty()
            && (config_.return_panel_interval_ns < config_.decision_window_ns
                || config_.return_panel_interval_ns
                    % config_.decision_window_ns != 0)) {
            throw std::invalid_argument(
                "return-panel interval must be an exact multiple of the "
                "decision window");
        }
        if (!config_.value_agent_policies.empty()
            && config_.value_agent_policies.size()
                != static_cast<std::size_t>(config_.asset_count)) {
            throw std::invalid_argument(
                "value-agent policies must be empty or aligned with asset configs");
        }
        if (!config_.background_configs.empty()
            && config_.background_configs.size()
                != static_cast<std::size_t>(config_.asset_count)) {
            throw std::invalid_argument(
                "background configs must be empty or aligned with asset configs");
        }
        if ((config_.partition_mode == PartitionMode::RealizedCostLpt
             || config_.openmp_schedule
                    == OpenMpSchedule::WeightedStatic)
            && config_.realized_partition_costs.size()
                != static_cast<std::size_t>(config_.asset_count)) {
            throw std::invalid_argument(
                "measured-cost partitioning/scheduling requires one cost per asset");
        }
        if (std::any_of(
                config_.realized_partition_costs.begin(),
                config_.realized_partition_costs.end(),
                [](double cost) {
                    return !(cost > 0.0) || !std::isfinite(cost);
                })) {
            throw std::invalid_argument(
                "realized partition costs must be finite and positive");
        }
        if ((config_.background_model == "legacy")
                != config_.background_configs.empty()
            || (config_.background_model != "legacy"
                && config_.background_model != "queue-reactive-v1")) {
            throw std::invalid_argument(
                "background model and configuration bundle are inconsistent");
        }
        if (!config_.shock_cluster_ids.empty()
            && config_.shock_cluster_ids.size()
                != static_cast<std::size_t>(config_.asset_count)) {
            throw std::invalid_argument(
                "shock cluster IDs must be empty or aligned with asset configs");
        }
        if (!config_.cluster_metrics_csv.empty()
            && config_.shock_cluster_ids.empty()) {
            throw std::invalid_argument(
                "cluster metrics require aligned shock cluster IDs");
        }
        if (std::any_of(
                config_.shock_cluster_ids.begin(),
                config_.shock_cluster_ids.end(),
                [](int cluster) { return cluster < 0; })) {
            throw std::invalid_argument("shock cluster IDs must be non-negative");
        }
        for (const ValueAgentPolicy& policy : config_.value_agent_policies) {
            if (!std::isfinite(policy.threshold_bps) || policy.threshold_bps < 0.0
                || !std::isfinite(policy.depth_participation)
                || policy.depth_participation < 0.0
                || policy.depth_participation > 1.0
                || (policy.depth_participation == 0.0
                    && policy.order_quantity <= 0)
                || policy.maximum_news_rechecks < 0
                || policy.maximum_news_rechecks > 16
                || (policy.trigger_mode
                        == ValueTriggerMode::PeriodicGap
                    && policy.maximum_news_rechecks != 0)
                || !std::isfinite(policy.gap_elasticity)
                || policy.gap_elasticity < 0.0
                || !std::isfinite(policy.maximum_depth_participation)
                || policy.maximum_depth_participation <= 0.0
                || policy.maximum_depth_participation > 1.0
                || policy.maximum_depth_participation
                    < policy.depth_participation
                || (policy.gap_elasticity > 0.0
                    && (policy.depth_participation <= 0.0
                        || policy.threshold_bps <= 0.0))) {
                throw std::invalid_argument("invalid value-agent policy");
            }
        }
        const std::uint64_t lob_count = static_cast<std::uint64_t>(config_.asset_count);
        if (lob_count > static_cast<std::uint64_t>(
                std::numeric_limits<BookId>::max())) {
            throw std::invalid_argument("asset count exceeds BookId range");
        }
        for (const MultiAssetBookConfig& book : config_.asset_configs) {
            if (book.initial_best_bid_ticks <= 0
                || book.initial_best_ask_ticks <= book.initial_best_bid_ticks
                || book.initial_best_bid_depth <= 0
                || book.initial_best_ask_depth <= 0
                || !std::isfinite(book.fundamental_price_ticks)
                || book.fundamental_price_ticks <= 0.0
                || !std::isfinite(
                    book.fundamental_volatility_bps_sqrt_second)
                || book.fundamental_volatility_bps_sqrt_second < 0.0
                || !std::isfinite(
                    book.fundamental_move_probability_per_second)
                || book.fundamental_move_probability_per_second < 0.0
                || book.fundamental_move_probability_per_second > 1.0
                || !std::isfinite(book.fundamental_conditional_kurtosis)
                || book.fundamental_conditional_kurtosis < 1.0
                || !std::isfinite(
                    book.fundamental_log_variance_persistence)
                || book.fundamental_log_variance_persistence < 0.0
                || book.fundamental_log_variance_persistence >= 1.0
                || !std::isfinite(book.fundamental_log_variance_std)
                || book.fundamental_log_variance_std < 0.0
                || !std::isfinite(book.fundamental_order_flow_coupling)
                || book.fundamental_order_flow_coupling < 0.0
                || book.fundamental_order_flow_coupling > 2.5
                || (book.fundamental_order_flow_coupling > 0.0
                    && book.fundamental_log_variance_std <= 0.0)
                || !std::isfinite(book.beta)
                || ((config_.shared_quote_relative_to_asset
                     || config_.shared_capacity_relative_to_asset)
                    && book.market_maker_quote_quantity <= 0)) {
                throw std::invalid_argument("invalid distributed asset template");
            }
        }
    }

    void select_shock_assets() {
        shock_mask_.assign(static_cast<std::size_t>(config_.asset_count), false);
        // The same deterministic target set is retained in a matched control
        // run.  Only the treatment inserts shock orders; the control still
        // reports target/non-target metrics for a like-for-like comparison.
        const auto count = static_cast<std::size_t>(config_.shock_target_count > 0
            ? config_.shock_target_count
            : std::max<long long>(
                1LL, std::llround(config_.shock_asset_fraction
                                  * static_cast<double>(config_.asset_count))));
        std::vector<std::pair<std::uint64_t, std::size_t>> ranked;
        ranked.reserve(static_cast<std::size_t>(config_.asset_count));
        for (std::size_t asset = 0;
             asset < static_cast<std::size_t>(config_.asset_count); ++asset) {
            ranked.emplace_back(
                stable_sequence(liquidity_shock_entity(static_cast<BookId>(asset)),
                                config_.shock_target_seed),
                asset);
        }
        std::sort(ranked.begin(), ranked.end());
        if (config_.shock_cluster_ids.empty()) {
            for (std::size_t index = 0; index < count; ++index) {
                shock_mask_[ranked[index].second] = true;
            }
        } else {
            // Largest-remainder proportional allocation followed by a stable
            // within-cluster deterministic ranking. This keeps the principal mask
            // composition fixed while covering the empirical liquidity strata.
            std::map<int, std::vector<std::pair<std::uint64_t, std::size_t>>> groups;
            for (const auto& item : ranked) {
                groups[config_.shock_cluster_ids[item.second]].push_back(item);
            }
            struct Quota { int cluster = 0; std::size_t count = 0; double remainder = 0.0; };
            std::vector<Quota> quotas;
            std::size_t assigned = 0;
            for (const auto& [cluster, members] : groups) {
                const double exact = static_cast<double>(count) *
                    static_cast<double>(members.size()) /
                    static_cast<double>(config_.asset_count);
                const auto base = static_cast<std::size_t>(std::floor(exact));
                quotas.push_back(Quota{cluster, base, exact - static_cast<double>(base)});
                assigned += base;
            }
            std::sort(quotas.begin(), quotas.end(), [](const Quota& left,
                                                       const Quota& right) {
                if (left.remainder != right.remainder) {
                    return left.remainder > right.remainder;
                }
                return left.cluster < right.cluster;
            });
            for (std::size_t index = 0; assigned < count; ++index, ++assigned) {
                ++quotas[index % quotas.size()].count;
            }
            for (const Quota& quota : quotas) {
                const auto& members = groups.at(quota.cluster);
                for (std::size_t index = 0; index < quota.count; ++index) {
                    shock_mask_[members[index].second] = true;
                }
            }
        }
        shock_asset_count_ = static_cast<std::uint64_t>(count);
    }

    void initialize_shared_capacity_weights() {
        shared_capacity_weights_.assign(
            static_cast<std::size_t>(config_.asset_count), 1.0);
        if (!config_.shared_capacity_relative_to_asset) return;
        double total_proxy = 0.0;
        for (const MultiAssetBookConfig& asset : config_.asset_configs) {
            total_proxy += static_cast<double>(
                asset.market_maker_quote_quantity);
        }
        const double mean_proxy = total_proxy
            / static_cast<double>(config_.asset_count);
        if (!(mean_proxy > 0.0) || !std::isfinite(mean_proxy)) {
            throw std::invalid_argument(
                "invalid empirical shared-capacity normalization");
        }
        for (std::size_t index = 0;
             index < shared_capacity_weights_.size(); ++index) {
            shared_capacity_weights_[index] = static_cast<double>(
                config_.asset_configs[index].market_maker_quote_quantity)
                / mean_proxy;
        }
    }

    double shared_capacity_weight(BookId asset_id) const {
        return shared_capacity_weights_.at(
            static_cast<std::size_t>(asset_id));
    }

    double uncoupled_quote_scale(const LocalAsset& asset) const {
        const double capacity = config_.shared_global_risk_limit_per_asset
            * shared_capacity_weight(asset.asset_id);
        const double utilization = std::abs(
            asset.config.beta * static_cast<double>(asset.shared_inventory))
            / capacity;
        return detail::shared_capacity_quote_scale(
            utilization,
            config_.shared_capacity_threshold,
            config_.shared_minimum_quote_scale,
            true);
    }

    double local_inventory_scale(const LocalAsset& asset) const {
        const double beta = std::max(1.0e-12, std::abs(asset.config.beta));
        return config_.shared_local_inventory_scale
            * shared_capacity_weight(asset.asset_id) / beta;
    }

    void write_shock_targets(const std::vector<AssetResultWire>& assets) const {
        if (rank_ != 0 || config_.shock_targets_csv.empty()) return;
        std::vector<std::uint64_t> requested(
            static_cast<std::size_t>(config_.asset_count), 0U);
        std::vector<std::int32_t> side_codes(
            static_cast<std::size_t>(config_.asset_count), 0);
        std::vector<std::int64_t> pre_shock_inventories(
            static_cast<std::size_t>(config_.asset_count), 0);
        for (const AssetResultWire& asset : assets) {
            if (asset.asset_id >= static_cast<BookId>(config_.asset_count)) {
                throw std::logic_error("invalid asset in shock target gather");
            }
            requested[static_cast<std::size_t>(asset.asset_id)] =
                asset.shock_requested_quantity;
            side_codes[static_cast<std::size_t>(asset.asset_id)] =
                asset.shock_side_code;
            pre_shock_inventories[static_cast<std::size_t>(asset.asset_id)] =
                asset.shock_pre_shared_inventory;
        }
        const std::filesystem::path path(config_.shock_targets_csv);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("cannot open shock target CSV: "
                                     + path.string());
        }
        output << "asset_id,symbol,cluster_id,is_shock_target,shock_enabled,"
                  "requested_quantity,requested_sell_quantity,"
                  "requested_buy_quantity,shock_side,"
                  "pre_shock_shared_inventory,direction_rule,mask_seed\n";
        for (std::size_t index = 0; index < config_.asset_configs.size(); ++index) {
            const bool enabled_target = shock_mask_[index]
                && config_.enable_shock;
            const std::uint64_t quantity = enabled_target
                ? requested[index] : 0U;
            const std::int32_t side_code = enabled_target
                ? side_codes[index] : 0;
            output << index << ','
                   << config_.asset_configs[index].symbol << ','
                   << (config_.shock_cluster_ids.empty()
                       ? -1 : config_.shock_cluster_ids[index]) << ','
                   << (shock_mask_[index] ? 1 : 0) << ','
                   << (config_.enable_shock ? 1 : 0) << ','
                   << quantity << ','
                   << (side_code == 2 ? quantity : 0U) << ','
                   << (side_code == 1 ? quantity : 0U) << ','
                   << (side_code == 1 ? "buy"
                       : (side_code == 2 ? "sell" : "none")) << ','
                   << (enabled_target
                       ? pre_shock_inventories[index] : 0) << ','
                   << (config_.shock_inventory_adverse
                       ? "inventory_adverse" : "fixed_sell")
                   << ',' << config_.shock_target_seed
                   << '\n';
        }
    }

    void initialize_asset_owners() {
        asset_owner_ranks_.resize(static_cast<std::size_t>(config_.asset_count));
        const bool complete_background = config_.background_configs.size()
            == static_cast<std::size_t>(config_.asset_count);
        const bool realized = config_.partition_mode
            == PartitionMode::RealizedCostLpt;
        const bool weighted = realized || config_.partition_mode
                == PartitionMode::GreedyBackgroundRate
            || (config_.partition_mode == PartitionMode::Auto
                && complete_background);
        if (weighted && !complete_background && !realized) {
            throw std::invalid_argument(
                "weighted partition requires one background configuration "
                "per asset");
        }
        weighted_partition_active_ = weighted;
        realized_cost_partition_active_ = realized;

        struct WeightedAsset {
            double weight = 0.0;
            int asset = 0;
        };
        std::vector<WeightedAsset> assets;
        assets.reserve(static_cast<std::size_t>(config_.asset_count));
        for (int asset = 0; asset < config_.asset_count; ++asset) {
            double weight = 1.0;
            if (realized) {
                weight = config_.realized_partition_costs[
                    static_cast<std::size_t>(asset)];
            } else if (complete_background) {
                const BackgroundHawkesConfig& background =
                    config_.background_configs[static_cast<std::size_t>(asset)];
                const BackgroundHawkesVector& rates =
                    background.validate_stationary_target
                    ? background.stationary_target_rates : background.mu;
                weight = std::accumulate(rates.begin(), rates.end(), 0.0);
                if (!background.validate_stationary_target) {
                    weight *= background.activity_scale;
                }
            }
            if (!(weight > 0.0) || !std::isfinite(weight)) {
                throw std::invalid_argument(
                    "partition encountered invalid background rate");
            }
            assets.push_back(WeightedAsset{weight, asset});
        }
        std::vector<double> rank_loads(
            static_cast<std::size_t>(world_size_), 0.0);
        if (!weighted) {
            for (const WeightedAsset& asset : assets) {
                const int rank = asset.asset % world_size_;
                asset_owner_ranks_[static_cast<std::size_t>(asset.asset)] = rank;
                rank_loads[static_cast<std::size_t>(rank)] += asset.weight;
            }
        } else {
            std::sort(
                assets.begin(), assets.end(),
                [](const WeightedAsset& left, const WeightedAsset& right) {
                    if (left.weight != right.weight) {
                        return left.weight > right.weight;
                    }
                    return left.asset < right.asset;
                });
            for (const WeightedAsset& asset : assets) {
                const auto owner = std::min_element(
                    rank_loads.begin(), rank_loads.end());
                const int rank = static_cast<int>(
                    std::distance(rank_loads.begin(), owner));
                asset_owner_ranks_[static_cast<std::size_t>(asset.asset)] = rank;
                *owner += asset.weight;
            }
        }
        const double mean = std::accumulate(
            rank_loads.begin(), rank_loads.end(), 0.0)
            / static_cast<double>(world_size_);
        predicted_partition_imbalance_ = mean > 0.0
            ? *std::max_element(rank_loads.begin(), rank_loads.end()) / mean
            : 1.0;
    }

    int owner_rank_for_asset(BookId asset_id) const {
        return asset_owner_ranks_.at(static_cast<std::size_t>(asset_id));
    }

    void initialize_local_assets() {
        local_assets_.clear();
        std::vector<int> owned_asset_indices;
        owned_asset_indices.reserve(static_cast<std::size_t>(
            (config_.asset_count + world_size_ - 1) / world_size_));
        for (int asset_index = 0; asset_index < config_.asset_count; ++asset_index) {
            const BookId asset_id = static_cast<BookId>(asset_index);
            if (owner_rank_for_asset(asset_id) != rank_) continue;
            owned_asset_indices.push_back(asset_index);
        }
        local_assets_.resize(owned_asset_indices.size());
        const auto initialize_one = [&](std::size_t local_index) {
            const int asset_index = owned_asset_indices[local_index];
            const BookId asset_id = static_cast<BookId>(asset_index);
            const MultiAssetBookConfig& source =
                config_.asset_configs[static_cast<std::size_t>(asset_index)];

            BackgroundHawkesConfig background;
            if (config_.background_configs.empty()) {
                background = make_multi_asset_background_config(
                    source, asset_id, config_.seed, config_.tick_size);
                background.activity_scale = config_.hawkes_activity_scale;
            } else {
                background = config_.background_configs[
                    static_cast<std::size_t>(asset_index)];
            }
            // Interpret the selected order-flow coupling directly as the
            // standard deviation theta of a persistent log-activity regime.
            // A previous formulation multiplied it by the latent-volatility standard deviation,
            // reducing several cluster loadings by factors of 3--50 and
            // producing no measurable ACF response. The discrete session
            // normalizer preserves the mean immigration multiplier, while
            // the realized Hawkes event count remains stochastic.
            if (source.fundamental_order_flow_coupling > 0.0) {
                background.stochastic_baseline_persistence =
                    source.fundamental_log_variance_persistence;
                background.stochastic_baseline_std =
                    source.fundamental_order_flow_coupling;
                background.stochastic_baseline_origin_ns = 0;
                background.stochastic_baseline_bin_width_ns =
                    fundamental_news_interval_ns;
                const std::int64_t normalization_horizon =
                    config_.stochastic_baseline_normalization_horizon_ns > 0
                    ? config_.stochastic_baseline_normalization_horizon_ns
                    : end_time_ns_;
                background.stochastic_baseline_normalization_bins =
                    static_cast<std::uint64_t>(
                        (normalization_horizon
                            + fundamental_news_interval_ns - 1)
                        / fundamental_news_interval_ns);
            }
            auto asset = std::make_unique<LocalAsset>(
                asset_id, source, background, config_.tick_size, config_.seed);
            asset->book.lob.seed_calibrated_book(
                source.initial_best_bid_ticks,
                source.initial_best_ask_ticks,
                source.initial_best_bid_depth,
                source.initial_best_ask_depth,
                config_.initial_depth_scale);
            compute_baseline(*asset);
            if (config_.enable_shock
                && shock_mask_[static_cast<std::size_t>(asset_index)]
                && config_.shock_top_depth_multiple <= 0.0
                && !config_.shock_inventory_adverse) {
                const int shock_quantity =
                    config_.shock_reference_bid_depth_multiple > 0.0
                    ? bounded_positive_quantity(
                        config_.shock_reference_bid_depth_multiple
                            * static_cast<double>(
                                source.initial_best_bid_depth),
                        "reference-depth sell-side stress quantity")
                    : config_.shock_quantity_per_asset;
                asset->shock_requested_quantity =
                    static_cast<std::uint64_t>(shock_quantity);
                asset->shock_injected = true;
                asset->pending_orders.push_back(make_market_order(
                    asset_id,
                    config_.shock_time_ns,
                    liquidity_shock_owner_id,
                    AgentKind::Institutional,
                    Side::Sell,
                    shock_quantity,
                    stable_sequence(liquidity_shock_entity(asset_id), 1)));
            }
            local_assets_[local_index] = std::move(asset);
        };
        if (config_.parallel_asset_initialization) {
            initialize_thread_buckets(owned_asset_indices);
            for_each_local_index(owned_asset_indices.size(), initialize_one);
        } else {
            for (std::size_t index = 0; index < owned_asset_indices.size(); ++index) {
                initialize_one(index);
            }
            initialize_thread_buckets(owned_asset_indices);
        }
    }

    void initialize_thread_buckets(const std::vector<int>& owned_assets) {
        local_thread_buckets_.clear();
        if (config_.openmp_schedule
                != OpenMpSchedule::WeightedStatic
            || config_.worker_threads <= 1 || owned_assets.size() <= 1U) {
            return;
        }
        const std::size_t thread_count = std::min(
            static_cast<std::size_t>(config_.worker_threads),
            owned_assets.size());
        local_thread_buckets_.resize(thread_count);
        std::vector<double> thread_loads(thread_count, 0.0);
        std::vector<std::size_t> order(owned_assets.size());
        std::iota(order.begin(), order.end(), 0U);
        std::sort(order.begin(), order.end(), [&](std::size_t left,
                                                  std::size_t right) {
            const double left_cost = config_.realized_partition_costs.at(
                static_cast<std::size_t>(owned_assets[left]));
            const double right_cost = config_.realized_partition_costs.at(
                static_cast<std::size_t>(owned_assets[right]));
            if (left_cost != right_cost) return left_cost > right_cost;
            return owned_assets[left] < owned_assets[right];
        });
        for (const std::size_t local_index : order) {
            const auto owner = std::min_element(
                thread_loads.begin(), thread_loads.end());
            const std::size_t thread = static_cast<std::size_t>(
                std::distance(thread_loads.begin(), owner));
            local_thread_buckets_[thread].push_back(local_index);
            thread_loads[thread] += config_.realized_partition_costs.at(
                static_cast<std::size_t>(owned_assets[local_index]));
        }
        for (std::vector<std::size_t>& bucket : local_thread_buckets_) {
            std::sort(bucket.begin(), bucket.end());
        }
    }

    void compute_baseline(LocalAsset& asset) const {
        const MarketState state = asset.book.lob.state(
            0, asset.config.fundamental_price_ticks);
        if (state.best_bid_ticks > 0
            && state.best_ask_ticks > state.best_bid_ticks) {
            asset.baseline_mean_spread_bps = static_cast<double>(
                state.best_ask_ticks - state.best_bid_ticks)
                / asset.config.fundamental_price_ticks * 10'000.0;
        }
        asset.baseline_top_depth = static_cast<double>(state.best_bid_depth)
            + static_cast<double>(state.best_ask_depth);
    }

    OrderMessage make_market_order(BookId asset_id,
                                   std::int64_t timestamp_ns,
                                   std::int32_t owner_id,
                                   AgentKind kind,
                                   Side side,
                                   int quantity,
                                   std::uint64_t sequence) const {
        OrderMessage order;
        order.book_id = asset_id;
        order.generated_time_ns = timestamp_ns;
        order.arrival_time_ns = timestamp_ns;
        order.sequence = sequence;
        order.tie_breaker = stable_sequence(sequence, asset_id, 1);
        order.source_rank = 0;
        order.owner_id = owner_id;
        order.agent_kind = kind;
        order.action = OrderAction::Market;
        order.side = side;
        order.quantity = quantity;
        return order;
    }

    void account_trades(LocalAsset& asset, LocalBook& book) {
        std::vector<TradeExecution> trades = book.lob.take_trades();
        for (const TradeExecution& trade : trades) {
            ++book.trade_count;
            if (trade.buyer_owner_id == shared_market_maker_owner
                && trade.seller_owner_id == shared_market_maker_owner) {
                throw std::logic_error(
                    "Shared Market Maker cannot trade with itself");
            }
            const __int128 notional = static_cast<__int128>(trade.price_ticks)
                * static_cast<__int128>(trade.quantity);
            if (trade.buyer_owner_id == shared_market_maker_owner) {
                asset.shared_inventory = checked_int64(
                    static_cast<__int128>(asset.shared_inventory)
                        + static_cast<__int128>(trade.quantity),
                    "shared inventory");
                asset.shared_cash_ticks = checked_int64(
                    static_cast<__int128>(asset.shared_cash_ticks) - notional,
                    "shared cash");
                asset.shared_buy_quantity = checked_uint64(
                    static_cast<unsigned __int128>(asset.shared_buy_quantity)
                        + static_cast<unsigned __int128>(trade.quantity),
                    "shared buy quantity");
                asset.shared_fill_count = checked_uint64(
                    static_cast<unsigned __int128>(asset.shared_fill_count) + 1,
                    "shared fill count");
            }
            if (trade.seller_owner_id == shared_market_maker_owner) {
                asset.shared_inventory = checked_int64(
                    static_cast<__int128>(asset.shared_inventory)
                        - static_cast<__int128>(trade.quantity),
                    "shared inventory");
                asset.shared_cash_ticks = checked_int64(
                    static_cast<__int128>(asset.shared_cash_ticks) + notional,
                    "shared cash");
                asset.shared_sell_quantity = checked_uint64(
                    static_cast<unsigned __int128>(asset.shared_sell_quantity)
                        + static_cast<unsigned __int128>(trade.quantity),
                    "shared sell quantity");
                asset.shared_fill_count = checked_uint64(
                    static_cast<unsigned __int128>(asset.shared_fill_count) + 1,
                    "shared fill count");
            }
            const bool shock_seller =
                trade.seller_owner_id == liquidity_shock_owner_id;
            const bool shock_buyer =
                trade.buyer_owner_id == liquidity_shock_owner_id;
            if (shock_seller || shock_buyer) {
                asset.shock_executed_quantity +=
                    static_cast<std::uint64_t>(trade.quantity);
                const std::int32_t counterparty = shock_seller
                    ? trade.buyer_owner_id : trade.seller_owner_id;
                if (counterparty == shared_market_maker_owner) {
                    asset.shock_shared_mm_quantity +=
                        static_cast<std::uint64_t>(trade.quantity);
                } else if (counterparty
                           == local_market_maker_owner_id(asset.asset_id)) {
                    asset.shock_local_mm_quantity +=
                        static_cast<std::uint64_t>(trade.quantity);
                } else if (counterparty
                           == fundamental_value_owner_id(asset.asset_id)) {
                    asset.shock_value_agent_quantity +=
                        static_cast<std::uint64_t>(trade.quantity);
                } else if (counterparty == 0) {
                    asset.shock_background_quantity +=
                        static_cast<std::uint64_t>(trade.quantity);
                } else {
                    asset.shock_other_quantity +=
                        static_cast<std::uint64_t>(trade.quantity);
                }
            }
            const std::int32_t value_owner = fundamental_value_owner_id(asset.asset_id);
            if (trade.buyer_owner_id == value_owner) {
                asset.value_inventory += trade.quantity;
            }
            if (trade.seller_owner_id == value_owner) {
                asset.value_inventory -= trade.quantity;
            }
        }
        (void)book.lob.take_reports();
    }

    ApplyResult apply_to_book(LocalAsset& asset,
                              const OrderMessage& source,
                              int quantity) {
        LocalBook& book = asset.book;
        OrderMessage order = source;
        order.book_id = book.book_id;
        order.quantity = quantity;
        const ApplyResult result = book.lob.apply(order);
        if (result.boundary_truncated_quantity > 0) {
            ++asset.removal_boundary_truncation_events;
            asset.removal_boundary_truncated_quantity +=
                static_cast<std::uint64_t>(result.boundary_truncated_quantity);
            if (source.action == OrderAction::Market) {
                ++asset.market_boundary_truncation_events;
                asset.market_boundary_truncated_quantity +=
                    static_cast<std::uint64_t>(
                        result.boundary_truncated_quantity);
            } else if (source.action == OrderAction::CancelAtDistance) {
                ++asset.cancel_boundary_truncation_events;
                asset.cancel_boundary_truncated_quantity +=
                    static_cast<std::uint64_t>(
                        result.boundary_truncated_quantity);
            } else {
                throw std::logic_error(
                    "unexpected order action reached the removal boundary");
            }
            if (source.agent_kind == AgentKind::Background) {
                ++asset.background_boundary_truncation_events;
                asset.background_boundary_truncated_quantity +=
                    static_cast<std::uint64_t>(
                        result.boundary_truncated_quantity);
            } else if (source.agent_kind == AgentKind::Value) {
                ++asset.value_boundary_truncation_events;
                asset.value_boundary_truncated_quantity +=
                    static_cast<std::uint64_t>(
                        result.boundary_truncated_quantity);
            } else {
                ++asset.other_boundary_truncation_events;
                asset.other_boundary_truncated_quantity +=
                    static_cast<std::uint64_t>(
                        result.boundary_truncated_quantity);
            }
        }
        ++book.processed_orders;
        account_trades(asset, book);
        return result;
    }

    ApplyResult apply_order(LocalAsset& asset, const OrderMessage& order) {
        if (order.book_id != asset.book.book_id) {
            throw std::logic_error("order targets a book outside its asset shard");
        }
        if (order.owner_id == liquidity_shock_owner_id
            && !asset.shock_pre_inventory_recorded) {
            asset.shock_pre_shared_inventory = asset.shared_inventory;
            asset.shock_side_code = order.side == Side::Buy ? 1 : 2;
            asset.shock_pre_inventory_recorded = true;
        }
        const ApplyResult result = apply_to_book(
            asset, order, std::max(0, order.quantity));
        if (order.action == OrderAction::ConservedLimit) {
            asset.local_quote_revision_requested_quantity +=
                static_cast<std::uint64_t>(result.requested_quantity);
            asset.local_quote_revision_moved_quantity +=
                static_cast<std::uint64_t>(result.resting_quantity);
            if (result.requested_quantity > 0 && result.resting_quantity == 0) {
                ++asset.local_quote_revision_no_donor_events;
            }
        }
        return result;
    }

    void process_window(LocalAsset& asset,
                        std::int64_t start_ns,
                        std::int64_t end_ns) {
        std::sort(asset.pending_orders.begin(), asset.pending_orders.end(), order_less);
        std::size_t pending_index = 0;
        while (true) {
            const bool has_pending = pending_index < asset.pending_orders.size()
                && asset.pending_orders[pending_index].arrival_time_ns < end_ns;
            const std::int64_t next_background_time =
                asset.hawkes.peek_time_ns();
            const bool has_background = next_background_time < end_ns;
            const bool has_dynamic_shock = config_.enable_shock
                && (config_.shock_top_depth_multiple > 0.0
                    || config_.shock_inventory_adverse)
                && shock_mask_[static_cast<std::size_t>(asset.asset_id)]
                && !asset.shock_injected
                && config_.shock_time_ns >= start_ns
                && config_.shock_time_ns < end_ns;
            if (!has_pending && !has_background && !has_dynamic_shock) break;

            const std::int64_t pending_time = has_pending
                ? asset.pending_orders[pending_index].arrival_time_ns
                : std::numeric_limits<std::int64_t>::max();
            const std::int64_t background_time = has_background
                ? next_background_time : std::numeric_limits<std::int64_t>::max();
            const std::int64_t shock_time = has_dynamic_shock
                ? config_.shock_time_ns : std::numeric_limits<std::int64_t>::max();
            if (shock_time <= pending_time && shock_time <= background_time) {
                int quantity = config_.shock_quantity_per_asset;
                if (config_.shock_top_depth_multiple > 0.0) {
                    const Side depth_side = config_.shock_inventory_adverse
                        ? detail::inventory_adverse_shock_side(
                            asset.shared_inventory)
                        : Side::Sell;
                    const int contemporaneous_depth = std::max(
                        1, depth_side == Side::Buy
                            ? asset.book.lob.best_ask_depth()
                            : asset.book.lob.best_bid_depth());
                    quantity = bounded_positive_quantity(
                        config_.shock_top_depth_multiple
                            * static_cast<double>(contemporaneous_depth),
                        "contemporaneous top-depth stress quantity");
                } else if (config_.shock_reference_bid_depth_multiple > 0.0) {
                    // Opening bid depth is a fixed dose unit even when the
                    // state-contingent intervention is an aggressive buy.
                    // This preserves an identical quantity vector across
                    // global, uncoupled, and shared-dealer-absent mechanisms.
                    quantity = bounded_positive_quantity(
                        config_.shock_reference_bid_depth_multiple
                            * static_cast<double>(
                                asset.config.initial_best_bid_depth),
                        "reference-depth inventory stress quantity");
                }
                const Side shock_side = config_.shock_inventory_adverse
                    ? detail::inventory_adverse_shock_side(
                        asset.shared_inventory)
                    : Side::Sell;
                asset.shock_requested_quantity =
                    static_cast<std::uint64_t>(quantity);
                asset.shock_injected = true;
                apply_order(asset, make_market_order(
                    asset.asset_id, config_.shock_time_ns,
                    liquidity_shock_owner_id, AgentKind::Institutional,
                    shock_side, quantity,
                    stable_sequence(liquidity_shock_entity(asset.asset_id), 1)));
                continue;
            }
            if (pending_time <= background_time) {
                OrderMessage order = asset.pending_orders[pending_index++];
                order.arrival_time_ns = std::max(start_ns, order.arrival_time_ns);
                apply_order(asset, order);
                continue;
            }

            LocalBook& book = asset.book;
            const MarketState pre_event_state = book.lob.state(
                next_background_time, asset.fundamental_value_ticks);
            const HawkesEvent event = asset.hawkes.pop(pre_event_state);
            const std::uint64_t event_sequence = asset.hawkes.accepted_events();
            OrderMessage order = asset.background.make_order(
                event, pre_event_state,
                stable_sequence(background_entity(asset.asset_id), event_sequence));
            order.book_id = book.book_id;
            const ApplyResult result = apply_order(asset, order);
            if (order.action == OrderAction::Limit) {
                asset.background_limit_requested_quantity +=
                    static_cast<std::uint64_t>(result.requested_quantity);
                asset.background_limit_resting_quantity +=
                    static_cast<std::uint64_t>(result.resting_quantity);
            } else if (order.action == OrderAction::Market) {
                asset.background_market_requested_quantity +=
                    static_cast<std::uint64_t>(result.requested_quantity);
                asset.background_market_executed_quantity +=
                    static_cast<std::uint64_t>(result.executed_quantity);
            } else if (order.action == OrderAction::CancelAtDistance) {
                asset.background_cancel_requested_quantity +=
                    static_cast<std::uint64_t>(result.requested_quantity);
                asset.background_cancel_effective_quantity +=
                    static_cast<std::uint64_t>(result.cancelled_quantity);
            }
        }

        if (pending_index > 0) {
            asset.pending_orders.erase(
                asset.pending_orders.begin(),
                asset.pending_orders.begin()
                    + static_cast<std::ptrdiff_t>(pending_index));
        }
    }

    ScheduledQuoteDepth append_quotes(LocalAsset& asset,
                       LocalBook& book,
                       std::int64_t decision_time_ns,
                       std::uint64_t boundary_index,
                       std::int32_t owner_id,
                       StableEntityId entity,
                       int base_quantity,
                       int levels,
                       double improvement_probability,
                       double improvement_spread_elasticity,
                       double maximum_improvement_probability,
                       bool quote_only_when_repairing,
                       bool conserve_empirical_liquidity,
                       bool preserve_unchanged_priority,
                       double bid_scale,
                       double ask_scale,
                       std::uint64_t bid_total_limit =
                           std::numeric_limits<std::uint64_t>::max(),
                       std::uint64_t ask_total_limit =
                           std::numeric_limits<std::uint64_t>::max()) {
        const std::int64_t arrival_time_ns = checked_add_time(
            decision_time_ns, agent_latency_ns);
        const MarketState state = book.lob.state(
            decision_time_ns, asset.fundamental_value_ticks);
        std::uint32_t child = 0;
        auto make_order = [&](OrderAction action, Side side, int quantity, int price) {
            OrderMessage order;
            order.book_id = book.book_id;
            order.generated_time_ns = decision_time_ns;
            order.arrival_time_ns = arrival_time_ns;
            order.sequence = stable_sequence(
                entity ^ static_cast<StableEntityId>(book.book_id),
                boundary_index + 1U, child++);
            order.tie_breaker = stable_sequence(order.sequence, book.book_id, child);
            order.source_rank = 0;
            order.owner_id = action == OrderAction::ConservedLimit ? 0 : owner_id;
            order.agent_kind = action == OrderAction::ConservedLimit
                ? AgentKind::Background : AgentKind::MarketMaker;
            order.action = action;
            order.side = side;
            order.quantity = quantity;
            order.price_ticks = price;
            asset.pending_orders.push_back(order);
        };

        if (base_quantity <= 0
            || (bid_scale <= 0.0 && ask_scale <= 0.0)) {
            // A continuously refreshed shared quote must be withdrawn when
            // its scale reaches zero.  A repair-only local quote remains a
            // normal resting order until it fills or another repair replaces
            // it; unconditional cancellation would create on/off oscillation.
            if (!conserve_empirical_liquidity
                && !quote_only_when_repairing
                && (!preserve_unchanged_priority
                    || !book.lob.owner_resting_quotes(owner_id).empty())) {
                make_order(OrderAction::CancelOwner, Side::Buy, 0, 0);
            }
            return {};
        }

        const std::int64_t tick = config_.tick_size;
        const std::int64_t target_spread = std::max<std::int64_t>(
            config_.tick_size,
            static_cast<std::int64_t>(config_.tick_size)
                * asset.config.target_spread_ticks);
        const std::int64_t current_spread =
            state.best_bid_ticks > 0
                && state.best_ask_ticks > state.best_bid_ticks
            ? static_cast<std::int64_t>(state.best_ask_ticks)
                - state.best_bid_ticks
            : target_spread;
        const double effective_improvement_probability =
            detail::local_mm_effective_improvement_probability(
                improvement_probability,
                maximum_improvement_probability,
                current_spread,
                target_spread,
                improvement_spread_elasticity);
        const bool improve_wide_spread = detail::deterministic_quote_improvement(
            config_.seed, entity ^ asset.stochastic_stream_id, 0,
            boundary_index,
            effective_improvement_probability);
        const bool one_sided = state.best_bid_ticks <= 0
            || state.best_ask_ticks <= state.best_bid_ticks;
        const bool shallow_top = !one_sided
            && (state.best_bid_depth < base_quantity
                || state.best_ask_depth < base_quantity);
        const bool wide_spread = !one_sided
            && static_cast<std::int64_t>(state.best_ask_ticks)
                - state.best_bid_ticks > target_spread;
        if (!detail::quote_required(
                quote_only_when_repairing,
                one_sided,
                shallow_top,
                wide_spread,
                improve_wide_spread)) {
            return {};
        }
        const detail::QuotePrices prices =
            detail::quote_prices(
                state, asset.fundamental_value_ticks,
                config_.tick_size, asset.config.target_spread_ticks,
                improve_wide_spread);
        const std::int64_t bid = prices.bid;
        const std::int64_t ask = prices.ask;

        struct PlannedQuote {
            Side side = Side::Buy;
            int quantity = 0;
            int price_ticks = 0;
        };
        std::vector<PlannedQuote> planned;
        ScheduledQuoteDepth scheduled;
        std::uint64_t bid_remaining = bid_total_limit;
        std::uint64_t ask_remaining = ask_total_limit;
        for (int level = 0; level < levels; ++level) {
            const std::int64_t level_bid = bid - static_cast<std::int64_t>(level) * tick;
            const std::int64_t level_ask = ask + static_cast<std::int64_t>(level) * tick;
            // Inventory skew can enlarge one side of a shared quote.  Route
            // it through the same bounded conversion used for empirical quote
            // multipliers rather than narrowing an out-of-range double to int.
            const int desired_bid_quantity = bid_scale > 0.0
                ? bounded_positive_quantity(
                    static_cast<double>(base_quantity) * bid_scale,
                    "bid quote quantity")
                : 0;
            const int desired_ask_quantity = ask_scale > 0.0
                ? bounded_positive_quantity(
                    static_cast<double>(base_quantity) * ask_scale,
                    "ask quote quantity")
                : 0;
            const int bid_quantity = static_cast<int>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(desired_bid_quantity), bid_remaining));
            const int ask_quantity = static_cast<int>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(desired_ask_quantity), ask_remaining));
            if (level_bid > 0 && bid_quantity > 0
                && level_bid <= std::numeric_limits<std::int32_t>::max()) {
                planned.push_back(PlannedQuote{
                    Side::Buy, bid_quantity, static_cast<int>(level_bid)});
                scheduled.bid += static_cast<std::uint64_t>(bid_quantity);
                bid_remaining -= static_cast<std::uint64_t>(bid_quantity);
            }
            if (level_ask > level_bid && ask_quantity > 0
                && level_ask <= std::numeric_limits<std::int32_t>::max()) {
                planned.push_back(PlannedQuote{
                    Side::Sell, ask_quantity, static_cast<int>(level_ask)});
                scheduled.ask += static_cast<std::uint64_t>(ask_quantity);
                ask_remaining -= static_cast<std::uint64_t>(ask_quantity);
            }
        }

        // A shared dealer that blindly cancels and recreates an unchanged BBO
        // quote every decision window loses price--time priority every second.
        // Preserve existing quantities at the desired prices and append only
        // a top-up.  A price change or a required size reduction still uses a
        // cancel/replace, matching normal exchange priority semantics.
        std::map<std::pair<int, int>, std::int64_t> current;
        bool reset_owner = !conserve_empirical_liquidity
            && !quote_only_when_repairing;
        if (reset_owner && preserve_unchanged_priority) {
            reset_owner = false;
            std::map<std::pair<int, int>, std::int64_t> desired;
            for (const PlannedQuote& quote : planned) {
                desired[{static_cast<int>(quote.side), quote.price_ticks}]
                    += quote.quantity;
            }
            for (const OwnerRestingQuote& quote
                    : book.lob.owner_resting_quotes(owner_id)) {
                const auto key = std::make_pair(
                    static_cast<int>(quote.side), quote.price_ticks);
                current[key] += quote.quantity;
            }
            for (const auto& [key, quantity] : current) {
                const auto found = desired.find(key);
                if (found == desired.end() || quantity > found->second) {
                    reset_owner = true;
                    break;
                }
            }
        }
        if (reset_owner) {
            make_order(OrderAction::CancelOwner, Side::Buy, 0, 0);
            current.clear();
        }
        const OrderAction quote_action = conserve_empirical_liquidity
            ? OrderAction::ConservedLimit : OrderAction::Limit;
        for (const PlannedQuote& quote : planned) {
            const auto key = std::make_pair(
                static_cast<int>(quote.side), quote.price_ticks);
            const std::int64_t existing = preserve_unchanged_priority
                ? current[key] : 0;
            const std::int64_t delta =
                static_cast<std::int64_t>(quote.quantity) - existing;
            if (delta > 0) {
                make_order(quote_action, quote.side,
                           static_cast<int>(delta), quote.price_ticks);
            }
        }
        return scheduled;
    }

    void schedule_local_market_makers(std::int64_t decision_time_ns,
                                      std::uint64_t refresh_index) {
        if (!config_.enable_local_market_makers) return;
        const double compute_start = MPI_Wtime();
        for_each_short_phase_asset([&](LocalAsset& asset) {
            LocalBook& book = asset.book;
            const int local_quantity = bounded_positive_quantity(
                config_.local_mm_quantity_multiplier * static_cast<double>(
                    asset.config.market_maker_quote_quantity),
                "local market-maker quote quantity");
            (void)append_quotes(
                asset, book, decision_time_ns, refresh_index,
                local_market_maker_owner_id(asset.asset_id),
                local_market_maker_entity_base, local_quantity, 1,
                config_.local_mm_improvement_probability,
                config_.local_mm_spread_elasticity,
                config_.local_mm_max_improvement_probability,
                true,
                false,
                false,
                1.0, 1.0);
        });
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    void advance_fundamental_news(std::uint64_t news_index) {
        const double compute_start = MPI_Wtime();
        for_each_short_phase_asset([&](LocalAsset& asset) {
            const double previous_fundamental = asset.fundamental_value_ticks;
            asset.fresh_fundamental_news = false;
            double effective_fundamental_volatility =
                asset.config.fundamental_volatility_bps_sqrt_second;
            // Keep the disabled path bit-for-bit identical to the legacy
            // process: no extra arithmetic is performed when std is zero.
            if (asset.config.fundamental_log_variance_std > 0.0) {
                asset.fundamental_log_variance =
                    detail::advance_fundamental_log_variance(
                        asset.fundamental_log_variance,
                        asset.config.fundamental_log_variance_persistence,
                        asset.config.fundamental_log_variance_std,
                        asset.stochastic_stream_id,
                        config_.seed,
                        news_index);
                effective_fundamental_volatility *=
                    detail::fundamental_volatility_multiplier(
                        asset.fundamental_log_variance,
                        asset.config.fundamental_log_variance_std);
            }
            asset.fundamental_value_ticks = detail::advance_fundamental_value(
                asset.fundamental_value_ticks,
                effective_fundamental_volatility,
                asset.config.fundamental_move_probability_per_second,
                asset.config.fundamental_conditional_kurtosis,
                static_cast<double>(fundamental_news_interval_ns)
                    / static_cast<double>(nanoseconds_per_second),
                asset.stochastic_stream_id,
                config_.seed,
                news_index);
            asset.fresh_fundamental_news =
                asset.fundamental_value_ticks != previous_fundamental;
        });
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    void schedule_shared_market_makers(std::int64_t decision_time_ns,
                                       std::uint64_t boundary_index) {
        if (!config_.enable_shared_market_maker) return;
        const double compute_start = MPI_Wtime();
        for_each_short_phase_asset([&](LocalAsset& asset) {
            LocalBook& book = asset.book;
            const double shared_base_quantity =
                config_.shared_quote_relative_to_asset
                ? config_.shared_quote_multiplier * static_cast<double>(
                    asset.config.market_maker_quote_quantity)
                : static_cast<double>(config_.shared_quote_quantity);
            const int shared_book_quantity = bounded_positive_quantity(
                shared_base_quantity, "shared market-maker quote quantity");
            const bool asset_local = config_.shared_inventory_policy
                == SharedMarketMakerInventoryPolicy::AssetLocal;
            const detail::SharedQuotePlan plan =
                detail::risk_managed_shared_quote_plan(
                    asset.shared_inventory,
                    !asset_local && config_.enable_global_shared_capacity
                        ? shared_quote_scale_
                        : uncoupled_quote_scale(asset),
                    local_inventory_scale(asset));
            const ScheduledQuoteDepth scheduled = append_quotes(
                asset, book, decision_time_ns, boundary_index,
                shared_market_maker_owner,
                shared_maker_entity,
                shared_book_quantity,
                config_.shared_quote_levels,
                0.0,
                0.0,
                1.0,
                false,
                false,
                true,
                plan.bid_scale, plan.ask_scale,
                plan.bid_total_limit, plan.ask_total_limit);
            asset.shared_requested_quote_depth = static_cast<double>(
                scheduled.bid + scheduled.ask);
            asset.shared_requested_bid_depth = static_cast<double>(scheduled.bid);
            asset.shared_requested_ask_depth = static_cast<double>(scheduled.ask);
            asset.shared_risk_reducing_requested_quote_depth = 0.0;
            if (plan.bid_reduces_inventory) {
                asset.shared_risk_reducing_requested_quote_depth +=
                    static_cast<double>(scheduled.bid);
            }
            if (plan.ask_reduces_inventory) {
                asset.shared_risk_reducing_requested_quote_depth +=
                    static_cast<double>(scheduled.ask);
            }
            asset.shared_risk_increasing_requested_quote_depth =
                asset.shared_requested_quote_depth
                - asset.shared_risk_reducing_requested_quote_depth;
        });
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    [[nodiscard]] ScheduledQuoteDepth candidate_shared_quote_depth_for_lookahead(
        const LocalAsset& asset) const {
        const double raw_base = config_.shared_quote_relative_to_asset
            ? config_.shared_quote_multiplier * static_cast<double>(
                asset.config.market_maker_quote_quantity)
            : static_cast<double>(config_.shared_quote_quantity);
        const int base = bounded_positive_quantity(
            raw_base, "lookahead candidate shared quote quantity");
        // A skipped boundary is permitted only when the current reduction has
        // established phi=1.  Construct the exact inventory-aware quote plan
        // that will therefore be requested immediately after this boundary.
        // Prices and cancel/top-up behaviour can only reduce the executable
        // quantity, so counting the complete desired depth is conservative.
        const detail::SharedQuotePlan plan =
            detail::risk_managed_shared_quote_plan(
                asset.shared_inventory, 1.0, local_inventory_scale(asset));
        const auto total_for_side = [&](double scale,
                                        std::uint64_t total_limit,
                                        const char* label) {
            if (!(scale > 0.0)) return std::uint64_t{0};
            const int per_level = bounded_positive_quantity(
                static_cast<double>(base) * scale, label);
            const std::uint64_t desired =
                static_cast<std::uint64_t>(per_level)
                * static_cast<std::uint64_t>(config_.shared_quote_levels);
            return std::min(desired, total_limit);
        };
        return ScheduledQuoteDepth{
            total_for_side(
                plan.bid_scale, plan.bid_total_limit,
                "lookahead candidate bid quote quantity"),
            total_for_side(
                plan.ask_scale, plan.ask_total_limit,
                "lookahead candidate ask quote quantity")};
    }

    [[nodiscard]] bool shared_quote_may_fill_before_refresh(
        LocalAsset& asset, std::int64_t decision_time_ns) const {
        const std::int64_t refresh_arrival_ns = checked_add_time(
            decision_time_ns, agent_latency_ns);

        // The Hawkes stream exposes its next accepted-event time without
        // consuming the stream.  If no background event can arrive before
        // the dealer's cancel/replace message, the resting dealer quote
        // cannot be reached through background flow during the latency gap.
        if (asset.hawkes.peek_time_ns() < refresh_arrival_ns) return true;

        // Orders already queued at the boundary are also known exactly.  A
        // removal cannot execute against the dealer, whereas a marketable
        // limit or market order can.  Equality is safe: CancelOwner has the
        // highest action priority at a common arrival timestamp.
        for (const OrderMessage& order : asset.pending_orders) {
            if (order.arrival_time_ns >= refresh_arrival_ns) continue;
            if (order.action == OrderAction::Limit
                || order.action == OrderAction::ConservedLimit
                || order.action == OrderAction::Market) {
                return true;
            }
        }

        // Dynamic shocks are generated inside process_window rather than
        // stored in pending_orders.  At an equal timestamp the shock is
        // deliberately processed before pending cancellation messages.
        const bool pending_dynamic_shock = config_.enable_shock
            && (config_.shock_top_depth_multiple > 0.0
                || config_.shock_inventory_adverse)
            && shock_mask_[static_cast<std::size_t>(asset.asset_id)]
            && !asset.shock_injected
            && config_.shock_time_ns >= decision_time_ns
            && config_.shock_time_ns <= refresh_arrival_ns;
        return pending_dynamic_shock;
    }

    void schedule_value_agents(std::int64_t decision_time_ns,
                               std::uint64_t decision_index) {
        if (!config_.enable_value_agents) return;
        const double compute_start = MPI_Wtime();
        for_each_short_phase_asset([&](LocalAsset& asset) {
            const ValueAgentPolicy& policy =
                value_agent_policy(asset.asset_id);
            if (policy.trigger_mode
                == ValueTriggerMode::PeriodicGap) {
                schedule_value_agent(
                    asset, decision_time_ns, decision_index);
            } else if (policy.trigger_mode
                           == ValueTriggerMode::NewsImpulse
                       && asset.remaining_value_rechecks > 0
                       && decision_time_ns >= asset.value_recheck_due_ns) {
                schedule_value_agent(
                    asset, decision_time_ns, decision_index);
                --asset.remaining_value_rechecks;
                if (asset.remaining_value_rechecks > 0) {
                    asset.value_recheck_due_ns = checked_add_time(
                        asset.value_recheck_due_ns,
                        config_.value_agent_interval_ns);
                } else {
                    asset.value_recheck_due_ns = 0;
                }
            }
        });
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    void schedule_news_impulse_value_agents(
        std::int64_t decision_time_ns, std::uint64_t news_index) {
        const double compute_start = MPI_Wtime();
        for_each_short_phase_asset([&](LocalAsset& asset) {
            const ValueAgentPolicy& policy =
                value_agent_policy(asset.asset_id);
            if (config_.enable_value_agents && asset.fresh_fundamental_news) {
                if (policy.trigger_mode
                    == ValueTriggerMode::NewsImpulse) {
                    schedule_value_agent(asset, decision_time_ns, news_index);
                    asset.remaining_value_rechecks =
                        policy.maximum_news_rechecks;
                    asset.value_recheck_due_ns =
                        asset.remaining_value_rechecks > 0
                        ? checked_add_time(
                            decision_time_ns,
                            config_.value_agent_interval_ns)
                        : 0;
                }
            }
            // News-mode decisions are immediate.  Consume the signal even if
            // no executable gap exists, so later book movement cannot revive
            // stale information.
            asset.fresh_fundamental_news = false;
        });
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    [[nodiscard]] const ValueAgentPolicy& value_agent_policy(
        BookId asset_id) const {
        if (config_.value_agent_policies.empty()) {
            return default_value_agent_policy_;
        }
        return config_.value_agent_policies.at(
            static_cast<std::size_t>(asset_id));
    }

    void schedule_value_agent(LocalAsset& asset,
                              std::int64_t decision_time_ns,
                              std::uint64_t boundary_index) {
        if (!config_.enable_value_agents) return;
        const ValueAgentPolicy& policy =
            value_agent_policy(asset.asset_id);
        if (!policy.enabled) return;
        const LocalBook& book = asset.book;
        if (!book.lob.has_ask() || !book.lob.has_bid()) return;
        const double fundamental = asset.fundamental_value_ticks;

        // A value trader responds only to an executable mispricing.  The
        // certification policy sizes the order as a fraction of total
        // displayed opposite-side depth and protects it at the perceived
        // fundamental.
        // It may therefore consume several levels when the signal is large,
        // but cannot trade beyond its valuation.  The LOB's value-order
        // reserve keeps the reduced finite book two-sided and attributes any
        // dependence on that boundary to the explicit adequacy diagnostic.
        const std::optional<Side> side = detail::fundamental_value_side(
            book.lob.best_bid(), book.lob.best_ask(), fundamental,
            policy.threshold_bps);
        const std::int64_t arrival_time = checked_add_time(
            decision_time_ns, agent_latency_ns);
        const std::int32_t owner = fundamental_value_owner_id(asset.asset_id);
        // Positive depth participation always creates an IOC-like protected
        // market order, so it can never leave a resting value quote to cancel.
        // Retain cancel/replace only for the legacy fixed-share limit fallback;
        // otherwise every calibrated policy would manufacture one no-op event
        // per asset and decision boundary, distorting computational-work counts.
        if (!(policy.depth_participation > 0.0)) {
            OrderMessage cancel;
            cancel.book_id = asset.asset_id;
            cancel.generated_time_ns = decision_time_ns;
            cancel.arrival_time_ns = arrival_time;
            cancel.sequence = stable_sequence(
                value_entity_base
                    + static_cast<StableEntityId>(asset.asset_id),
                boundary_index + 1U, 0U);
            cancel.tie_breaker = stable_sequence(
                cancel.sequence, asset.asset_id, 0U);
            cancel.source_rank = 0;
            cancel.owner_id = owner;
            cancel.agent_kind = AgentKind::Value;
            cancel.action = OrderAction::CancelOwner;
            cancel.side = Side::Buy;
            asset.pending_orders.push_back(cancel);
        }
        if (!side.has_value()) return;

        const double grid = static_cast<double>(config_.tick_size);
        const int fundamental_limit = *side == Side::Buy
            ? static_cast<int>(std::floor(fundamental / grid) * grid)
            : static_cast<int>(std::ceil(fundamental / grid) * grid);
        const int protected_price = *side == Side::Buy
            ? std::max(book.lob.best_ask(), fundamental_limit)
            : std::min(book.lob.best_bid(), fundamental_limit);
        const std::int64_t protected_depth = *side == Side::Buy
            ? book.lob.total_ask_depth()
            : book.lob.total_bid_depth();
        const double executable_gap_bps =
            detail::fundamental_value_executable_gap_bps(
                *side, book.lob.best_bid(), book.lob.best_ask(), fundamental);
        const double effective_participation =
            detail::fundamental_value_effective_participation(
                policy.depth_participation,
                policy.maximum_depth_participation,
                executable_gap_bps,
                policy.threshold_bps,
                policy.gap_elasticity);
        const int protected_quantity = policy.depth_participation > 0.0
            ? detail::fundamental_value_participation_quantity(
                protected_depth, effective_participation)
            : static_cast<int>(std::min<std::int64_t>(
                policy.order_quantity, protected_depth));
        if (protected_price <= 0 || protected_quantity <= 0) return;
        OrderMessage order;
        order.book_id = asset.asset_id;
        order.generated_time_ns = decision_time_ns;
        order.arrival_time_ns = arrival_time;
        order.sequence = stable_sequence(
            value_entity_base
                + static_cast<StableEntityId>(asset.asset_id),
            boundary_index + 1U, 1U);
        order.tie_breaker = stable_sequence(
            order.sequence, asset.asset_id, 1U);
        order.source_rank = 0;
        order.owner_id = owner;
        order.agent_kind = AgentKind::Value;
        order.action = policy.depth_participation > 0.0
            ? OrderAction::Market : OrderAction::Limit;
        order.side = *side;
        order.quantity = protected_quantity;
        order.price_ticks = protected_price;
        asset.pending_orders.push_back(order);
        ++asset.value_order_count;
        asset.value_requested_quantity +=
            static_cast<std::uint64_t>(protected_quantity);
    }

    void release_persistent_risk_collective() noexcept {
#if LOB_HAS_MPI_PERSISTENT_COLLECTIVES
        if (risk_request_ != MPI_REQUEST_NULL) {
            (void)MPI_Request_free(&risk_request_);
            risk_request_ = MPI_REQUEST_NULL;
        }
#endif
        persistent_risk_width_ = 0;
    }

    long long local_fixed_exposure() {
        const auto contribution = [](const LocalAsset& asset) {
            const double exposure = std::abs(
                asset.config.beta * static_cast<double>(asset.shared_inventory));
            return std::llround(exposure * risk_fixed_point_scale);
        };
        if (config_.parallel_boundary_reductions) {
            if (fixed_exposure_contributions_.size() != local_assets_.size()) {
                fixed_exposure_contributions_.resize(local_assets_.size());
            }
            for_each_local_index(local_assets_.size(), [&](std::size_t index) {
                fixed_exposure_contributions_[index] =
                    contribution(*local_assets_[index]);
            });
        }
        long long total = 0;
        for (std::size_t index = 0; index < local_assets_.size(); ++index) {
            const long long fixed = config_.parallel_boundary_reductions
                ? fixed_exposure_contributions_[index]
                : contribution(*local_assets_[index]);
            if (fixed > std::numeric_limits<long long>::max() - total) {
                throw std::overflow_error(
                    "shared market-maker exposure overflow");
            }
            total += fixed;
        }
        return total;
    }

    template <typename CapacityIndependentWork>
    void update_shared_risk(
        std::int64_t time_ns,
        CapacityIndependentWork&& capacity_independent_work) {
        const double profiled_local_start = active_window_phase_ != nullptr
            ? MPI_Wtime() : 0.0;
        bool profiled_local_finished = false;
        const auto finish_profiled_local = [&]() {
            if (active_window_phase_ != nullptr && !profiled_local_finished) {
                active_window_phase_->risk_local_seconds +=
                    MPI_Wtime() - profiled_local_start;
                profiled_local_finished = true;
            }
        };
        if (!config_.enable_shared_market_maker) {
            shared_gross_exposure_ = 0.0;
            shared_utilization_ = 0.0;
            shared_quote_scale_ = 1.0;
            finish_profiled_local();
            capacity_independent_work();
            return;
        }
        if (config_.shared_inventory_policy
                == SharedMarketMakerInventoryPolicy::AssetLocal) {
            shared_gross_exposure_ = 0.0;
            shared_utilization_ = 0.0;
            shared_quote_scale_ = 1.0;
            finish_profiled_local();
            capacity_independent_work();
            return;
        }
        ++risk_boundaries_;
        last_risk_boundary_was_skipped_ = false;
        const long long local_fixed = local_fixed_exposure();
        const long long maximum_safe_local_sum =
            std::numeric_limits<long long>::max()
            / static_cast<long long>(world_size_);
        if (local_fixed > maximum_safe_local_sum) {
            throw std::overflow_error(
                "shared market-maker global exposure may overflow "
                "MPI_LONG_LONG");
        }
        const bool terminal_boundary = time_ns == end_time_ns_;
        if (config_.risk_lookahead_max_windows > 0U
            && risk_lookahead_remaining_ > 0U
            && !terminal_boundary) {
            --risk_lookahead_remaining_;
            ++risk_lookahead_skipped_boundaries_;
            last_risk_boundary_was_skipped_ = true;
            // The integer proof guarantees that the synchronous capacity
            // function is exactly one.  Exact exposure is observational only
            // on this boundary and is reconstructed by one terminal reduction.
            shared_quote_scale_ = 1.0;
            risk_observations_.push_back(RiskObservationFrame{
                time_ns, local_fixed, 0.0});
            finish_profiled_local();
            capacity_independent_work();
            return;
        }
        if (!config_.enable_global_shared_capacity
            && config_.buffer_global_observations) {
            double local_scale_sum = 0.0;
            for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
                local_scale_sum += uncoupled_quote_scale(*asset);
            }
            risk_observations_.push_back(RiskObservationFrame{
                time_ns, local_fixed, local_scale_sum});
            finish_profiled_local();
            capacity_independent_work();
            return;
        }
        risk_local_fixed_[0] = local_fixed;
        risk_local_fixed_[1] = 0;
        risk_global_fixed_[0] = 0;
        risk_global_fixed_[1] = 0;
        // Construct the second reduction value only while the preceding
        // globally agreed capacity response is exactly one.  Once capacity
        // is active, this proof cannot skip a boundary and its O(assets)
        // construction is pure overhead.  Repeated activation disables the
        // optimisation for the rest of the run without changing semantics.
        const bool evaluate_lookahead_bound =
            config_.risk_lookahead_max_windows > 0U
            && !risk_lookahead_permanently_disabled_
            && shared_quote_scale_ == 1.0
            && !terminal_boundary;
        const int risk_width = evaluate_lookahead_bound ? 2 : 1;
        if (risk_width == 2) {
            ++risk_lookahead_bound_evaluations_;
            for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
                const ScheduledQuoteDepth candidate =
                    candidate_shared_quote_depth_for_lookahead(*asset);
                std::uint64_t outstanding_bid = candidate.bid;
                std::uint64_t outstanding_ask = candidate.ask;
                if (shared_quote_may_fill_before_refresh(*asset, time_ns)) {
                    const auto add_outstanding = [&](Side side,
                                                     std::uint64_t quantity) {
                        std::uint64_t& total = side == Side::Buy
                            ? outstanding_bid : outstanding_ask;
                        if (quantity
                                > std::numeric_limits<std::uint64_t>::max()
                                    - total) {
                            throw std::overflow_error(
                                "lookahead outstanding quote quantity overflow");
                        }
                        total += quantity;
                    };
                    add_outstanding(
                        Side::Buy,
                        static_cast<std::uint64_t>(
                            asset->book.lob.owner_resting_depth(
                                shared_market_maker_owner, Side::Buy)));
                    add_outstanding(
                        Side::Sell,
                        static_cast<std::uint64_t>(
                            asset->book.lob.owner_resting_depth(
                                shared_market_maker_owner, Side::Sell)));
                    for (const OrderMessage& order : asset->pending_orders) {
                        if (order.owner_id == shared_market_maker_owner
                            && (order.action == OrderAction::Limit
                                || order.action
                                    == OrderAction::ConservedLimit)
                            && order.quantity > 0) {
                            add_outstanding(
                                order.side,
                                static_cast<std::uint64_t>(order.quantity));
                        }
                    }
                }
                const long double inventory = static_cast<long double>(
                    asset->shared_inventory);
                const long double largest_future_magnitude = std::max(
                    std::abs(inventory
                        + static_cast<long double>(outstanding_bid)),
                    std::abs(inventory
                        - static_cast<long double>(outstanding_ask)));
                const long double increase = std::max(
                    0.0L, largest_future_magnitude - std::abs(inventory));
                const long double raw_fixed =
                    std::abs(static_cast<long double>(asset->config.beta))
                    * increase
                    * static_cast<long double>(risk_fixed_point_scale);
                if (!std::isfinite(raw_fixed)
                    || raw_fixed > static_cast<long double>(
                        std::numeric_limits<long long>::max())) {
                    throw std::overflow_error(
                        "lookahead outstanding-quote bound overflow");
                }
                const long long fixed = static_cast<long long>(
                    std::ceil(raw_fixed));
                if (fixed > std::numeric_limits<long long>::max()
                        - risk_local_fixed_[1]) {
                    throw std::overflow_error(
                        "lookahead local outstanding-quote bound overflow");
                }
                risk_local_fixed_[1] += fixed;
            }
            if (risk_local_fixed_[1] > maximum_safe_local_sum) {
                throw std::overflow_error(
                    "lookahead global exposure bound may overflow "
                    "MPI_LONG_LONG");
            }
        }
        const double risk_arrival_seconds = MPI_Wtime();
        finish_profiled_local();
        if (!config_.boundary_arrival_csv.empty()) {
            boundary_arrivals_.push_back(BoundaryArrivalWire{
                time_ns,
                risk_boundaries_,
                static_cast<std::int32_t>(rank_),
                risk_arrival_seconds - wall_start_seconds_,
                risk_arrival_seconds - last_risk_completion_seconds_,
                0.0});
        }
        if (config_.profile_boundary_wait) {
            const double wait_start = MPI_Wtime();
            check_mpi(MPI_Barrier(communicator_),
                      "MPI_Barrier(shared-risk arrival profile)");
            boundary_wait_seconds_ += MPI_Wtime() - wait_start;
            ++boundary_wait_calls_;
            ++collective_calls_;
        }
        double collective_elapsed = 0.0;
        bool work_completed = false;
        const auto overlap_or_wait = [&](MPI_Request& request,
                                         const char* wait_label) {
            std::exception_ptr work_failure;
            if (config_.use_nonblocking_risk_collective) {
                const double work_start = MPI_Wtime();
                try {
                    capacity_independent_work();
                    work_completed = true;
                } catch (...) {
                    work_failure = std::current_exception();
                }
                risk_overlap_work_seconds_ += MPI_Wtime() - work_start;
            }
            const double wait_start = MPI_Wtime();
            check_mpi(MPI_Wait(&request, MPI_STATUS_IGNORE), wait_label);
            const double wait_elapsed = MPI_Wtime() - wait_start;
            collective_elapsed += wait_elapsed;
            if (config_.use_nonblocking_risk_collective) {
                risk_wait_after_overlap_seconds_ += wait_elapsed;
            }
            if (work_failure) std::rethrow_exception(work_failure);
        };
#if LOB_HAS_MPI_PERSISTENT_COLLECTIVES
        if (config_.use_persistent_risk_collective) {
            if (risk_request_ != MPI_REQUEST_NULL
                && persistent_risk_width_ != risk_width) {
                release_persistent_risk_collective();
            }
            const double post_start = MPI_Wtime();
            if (risk_request_ == MPI_REQUEST_NULL) {
                check_mpi(MPI_Allreduce_init(
                              risk_local_fixed_.data(), risk_global_fixed_.data(),
                              risk_width,
                              MPI_LONG_LONG, MPI_SUM, communicator_,
                              MPI_INFO_NULL, &risk_request_),
                          "MPI_Allreduce_init(shared risk)");
                persistent_risk_collective_active_ = true;
                persistent_risk_width_ = risk_width;
            }
            check_mpi(MPI_Start(&risk_request_),
                      "MPI_Start(shared risk)");
            collective_elapsed += MPI_Wtime() - post_start;
            overlap_or_wait(risk_request_, "MPI_Wait(shared risk)");
        } else
#endif
        {
            if (config_.use_nonblocking_risk_collective) {
                MPI_Request request = MPI_REQUEST_NULL;
                const double post_start = MPI_Wtime();
                check_mpi(MPI_Iallreduce(
                              risk_local_fixed_.data(), risk_global_fixed_.data(),
                              risk_width,
                              MPI_LONG_LONG, MPI_SUM, communicator_, &request),
                          "MPI_Iallreduce(shared risk)");
                collective_elapsed += MPI_Wtime() - post_start;
                overlap_or_wait(request, "MPI_Wait(shared risk)");
            } else {
                const double blocking_start = MPI_Wtime();
                check_mpi(MPI_Allreduce(
                              risk_local_fixed_.data(), risk_global_fixed_.data(),
                              risk_width,
                              MPI_LONG_LONG, MPI_SUM, communicator_),
                          "MPI_Allreduce(shared risk)");
                collective_elapsed += MPI_Wtime() - blocking_start;
            }
        }
        if (!work_completed) capacity_independent_work();
        communication_seconds_ += collective_elapsed;
        risk_collective_seconds_ += collective_elapsed;
        if (active_window_phase_ != nullptr) {
            active_window_phase_->risk_collective_seconds +=
                collective_elapsed;
        }
        if (!config_.boundary_arrival_csv.empty()) {
            boundary_arrivals_.back().collective_seconds = collective_elapsed;
        }
        last_risk_completion_seconds_ = MPI_Wtime();
        ++collective_calls_;
        ++risk_collective_calls_;
        shared_gross_exposure_ = static_cast<double>(risk_global_fixed_[0])
            / risk_fixed_point_scale;
        maximum_shared_gross_exposure_ = std::max(
            maximum_shared_gross_exposure_, shared_gross_exposure_);
        const double global_limit = config_.shared_global_risk_limit_per_asset
            * static_cast<double>(config_.asset_count);
        shared_utilization_ = shared_gross_exposure_ / global_limit;
        if (config_.enable_global_shared_capacity) {
            shared_quote_scale_ = detail::shared_capacity_quote_scale(
                shared_utilization_,
                config_.shared_capacity_threshold,
                config_.shared_minimum_quote_scale,
                true);
        } else {
            double local_scale_sum = 0.0;
            for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
                local_scale_sum += uncoupled_quote_scale(*asset);
            }
            double global_scale_sum = 0.0;
            const double scale_start = MPI_Wtime();
            check_mpi(MPI_Allreduce(
                          &local_scale_sum, &global_scale_sum, 1,
                          MPI_DOUBLE, MPI_SUM, communicator_),
                      "MPI_Allreduce(uncoupled quote-scale diagnostic)");
            const double scale_elapsed = MPI_Wtime() - scale_start;
            communication_seconds_ += scale_elapsed;
            observation_collective_seconds_ += scale_elapsed;
            ++collective_calls_;
            ++observation_collective_calls_;
            shared_quote_scale_ = global_scale_sum
                / static_cast<double>(config_.asset_count);
        }
        minimum_shared_quote_scale_ = std::min(
            minimum_shared_quote_scale_, shared_quote_scale_);
        if (shared_quote_scale_ < 0.5) ++withdrawal_windows_;

        if (config_.risk_lookahead_max_windows > 0U) {
            if (shared_quote_scale_ < 1.0) {
                ++risk_lookahead_active_capacity_streak_;
                if (risk_lookahead_active_capacity_streak_ >= 2U
                    && !risk_lookahead_permanently_disabled_) {
                    risk_lookahead_permanently_disabled_ = true;
                    risk_lookahead_disabled_after_boundary_ = risk_boundaries_;
                }
            } else {
                risk_lookahead_active_capacity_streak_ = 0U;
            }
        }

        risk_lookahead_remaining_ = 0U;
        if (risk_width == 2 && !terminal_boundary
            && shared_quote_scale_ == 1.0) {
            const long double threshold_raw =
                static_cast<long double>(global_limit)
                * static_cast<long double>(config_.shared_capacity_threshold)
                * static_cast<long double>(risk_fixed_point_scale);
            if (std::isfinite(threshold_raw)
                && threshold_raw > 0.0L
                && threshold_raw <= static_cast<long double>(
                    std::numeric_limits<std::uint64_t>::max())) {
                std::uint64_t maximum_span =
                    config_.risk_lookahead_max_windows;
                // The first boundary after the intervention must synchronize.
                // This makes the stress response directly auditable even when
                // the pre-shock headroom proof spans many ordinary windows.
                if (config_.enable_shock && time_ns <= config_.shock_time_ns) {
                    const auto before_shock = static_cast<std::uint64_t>(
                        (config_.shock_time_ns - time_ns)
                        / config_.decision_window_ns);
                    maximum_span = std::min(maximum_span, before_shock);
                }
                if (maximum_span > 0U) {
                    ++risk_lookahead_attempted_boundaries_;
                    const std::uint64_t current =
                        static_cast<std::uint64_t>(risk_global_fixed_[0]);
                    const std::uint64_t reachable_increase =
                        static_cast<std::uint64_t>(risk_global_fixed_[1]);
                    const std::uint64_t threshold =
                        static_cast<std::uint64_t>(std::floor(threshold_raw));
                    // Prove exactly one future boundary.  The envelope above
                    // covers all currently executable shared quotes and every
                    // quote that the phi=1 policy can request for the next
                    // window.  At the skipped boundary no second proof is
                    // attempted; the following boundary synchronizes.
                    if (current < threshold
                        && reachable_increase < threshold - current) {
                        risk_lookahead_remaining_ = 1U;
                    } else {
                        ++risk_lookahead_bound_rejections_;
                    }
                }
                if (risk_lookahead_remaining_ > 0U) {
                    ++risk_lookahead_batches_;
                    risk_lookahead_max_span_ = std::max(
                        risk_lookahead_max_span_, risk_lookahead_remaining_);
                }
            }
        }
    }

    void record_asset_moments(std::int64_t time_ns) {
        if (config_.asset_summary_csv.empty() || time_ns <= 0
            || time_ns % config_.asset_summary_interval_ns != 0) return;
        const auto record = [&](LocalAsset& asset) {
            const MarketState state = asset.book.lob.state(
                time_ns, asset.fundamental_value_ticks);
            asset.calibration_moments.observe(state, config_.tick_size);
        };
        if (config_.parallel_metric_scans) {
            for_each_local_asset(record);
        } else {
            for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
                record(*pointer);
            }
        }
    }

    void open_return_panel_output() {
        if (config_.return_panel_prefix.empty()) return;
        std::ostringstream name;
        name << config_.return_panel_prefix << ".rank"
             << std::setw(5) << std::setfill('0') << rank_ << ".csv";
        const std::filesystem::path path(name.str());
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        return_panel_output_.open(path);
        if (!return_panel_output_) {
            throw std::runtime_error(
                "cannot open rank-local return panel CSV: " + path.string());
        }
        return_panel_output_ << "time_seconds";
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            return_panel_output_ << ','
                << config_.asset_configs.at(
                       static_cast<std::size_t>(asset->asset_id)).symbol;
        }
        return_panel_output_ << '\n';
    }

    void record_return_panel(std::int64_t time_ns) {
        if (!return_panel_output_
            || time_ns % config_.return_panel_interval_ns != 0) return;
        return_panel_output_ << std::fixed << std::setprecision(9)
            << static_cast<double>(time_ns) / 1e9;
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            const MarketState state = asset->book.lob.state(
                time_ns, asset->fundamental_value_ticks);
            return_panel_output_ << ',';
            if (state.best_bid_ticks > 0
                && state.best_ask_ticks >= state.best_bid_ticks) {
                const std::int64_t twice_midpoint =
                    static_cast<std::int64_t>(state.best_bid_ticks)
                    + static_cast<std::int64_t>(state.best_ask_ticks);
                return_panel_output_ << twice_midpoint;
            }
        }
        return_panel_output_ << '\n';
    }

    AggregateMetricSums local_metrics(std::int64_t time_ns) {
        const auto accumulate_asset = [&](AggregateMetricSums& metrics,
                                          const LocalAsset& asset,
                                          std::array<double, 8>* cluster) {
            const double shared_bid_resting = static_cast<double>(
                asset.book.lob.owner_resting_depth(
                    shared_market_maker_owner, Side::Buy));
            const double shared_ask_resting = static_cast<double>(
                asset.book.lob.owner_resting_depth(
                    shared_market_maker_owner, Side::Sell));
            const double shared_resting = shared_bid_resting + shared_ask_resting;
            const double shared_reducing_resting = asset.shared_inventory > 0
                ? shared_ask_resting
                : (asset.shared_inventory < 0 ? shared_bid_resting : 0.0);
            const double shared_increasing_resting =
                shared_resting - shared_reducing_resting;
            const MarketState state = asset.book.lob.state(
                time_ns, asset.fundamental_value_ticks);
            double shared_best_bid = 0.0;
            double shared_best_ask = 0.0;
            for (const OwnerRestingQuote& quote
                    : asset.book.lob.owner_resting_quotes(
                        shared_market_maker_owner)) {
                if (quote.side == Side::Buy
                    && quote.price_ticks == state.best_bid_ticks) {
                    shared_best_bid += static_cast<double>(quote.quantity);
                }
                if (quote.side == Side::Sell
                    && quote.price_ticks == state.best_ask_ticks) {
                    shared_best_ask += static_cast<double>(quote.quantity);
                }
            }
            const bool two_sided = state.best_bid_ticks > 0
                && state.best_ask_ticks > state.best_bid_ticks;
            const double asset_spread = two_sided
                ? static_cast<double>(state.best_ask_ticks - state.best_bid_ticks)
                    / asset.config.fundamental_price_ticks * 10'000.0
                : 0.0;
            const double asset_depth = static_cast<double>(state.best_bid_depth)
                + static_cast<double>(state.best_ask_depth);
            metrics.top_depth_sum += asset_depth;
            if (two_sided) {
                metrics.spread_sum_bps += asset_spread;
                ++metrics.two_sided_book_count;
            }
            const bool affected = !two_sided
                || asset_spread > 2.0 * asset.baseline_mean_spread_bps
                || asset_depth < 0.5 * asset.baseline_top_depth;
            if (affected) ++metrics.affected_asset_count;

            const bool shocked = shock_mask_.at(
                static_cast<std::size_t>(asset.asset_id));
            if (shocked) {
                ++metrics.shocked_asset_count;
                metrics.shocked_top_depth_sum += asset_depth;
                metrics.shocked_shared_inventory_sum +=
                    static_cast<double>(asset.shared_inventory);
                metrics.shocked_bid_top_depth_sum +=
                    static_cast<double>(state.best_bid_depth);
                metrics.shocked_shared_bid_resting_depth_sum +=
                    shared_best_bid;
                metrics.shocked_shared_absolute_inventory_sum += std::abs(
                    static_cast<double>(asset.shared_inventory));
                if (affected) ++metrics.affected_shocked_asset_count;
                if (two_sided) {
                    metrics.shocked_spread_sum_bps += asset_spread;
                    ++metrics.shocked_two_sided_asset_count;
                }
            } else {
                ++metrics.unshocked_asset_count;
                metrics.unshocked_top_depth_sum += asset_depth;
                metrics.unshocked_shared_quote_depth_sum +=
                    asset.shared_requested_quote_depth;
                metrics.unshocked_shared_resting_depth_sum += shared_resting;
                if (affected) ++metrics.affected_unshocked_asset_count;
                if (two_sided) {
                    metrics.unshocked_spread_sum_bps += asset_spread;
                    ++metrics.unshocked_two_sided_asset_count;
                }
            }
            metrics.value_order_count +=
                static_cast<double>(asset.value_order_count);
            metrics.value_requested_quantity +=
                static_cast<double>(asset.value_requested_quantity);
            metrics.shared_requested_quote_depth_sum +=
                asset.shared_requested_quote_depth;
            metrics.shared_risk_reducing_quote_depth_sum +=
                asset.shared_risk_reducing_requested_quote_depth;
            metrics.shared_risk_increasing_quote_depth_sum +=
                asset.shared_risk_increasing_requested_quote_depth;
            metrics.shared_resting_quote_depth_sum += shared_resting;
            metrics.shared_best_bid_depth_sum += shared_best_bid;
            metrics.shared_best_ask_depth_sum += shared_best_ask;
            if (shared_best_bid > 0.0) {
                ++metrics.shared_at_best_bid_asset_count;
            }
            if (shared_best_ask > 0.0) {
                ++metrics.shared_at_best_ask_asset_count;
            }
            metrics.shared_risk_reducing_resting_depth_sum +=
                shared_reducing_resting;
            metrics.shared_risk_increasing_resting_depth_sum +=
                shared_increasing_resting;
            if (shared_resting > 0.0) {
                ++metrics.shared_active_asset_count;
            }
            if (shared_bid_resting > 0.0 && shared_ask_resting > 0.0) {
                ++metrics.shared_two_sided_active_asset_count;
            }
            if (asset.shared_inventory != 0) {
                ++metrics.shared_nonzero_inventory_asset_count;
            }
            if (asset.shared_requested_quote_depth > 0.0) {
                ++metrics.shared_requested_active_asset_count;
            }
            if (asset.shared_requested_bid_depth > 0.0
                && asset.shared_requested_ask_depth > 0.0) {
                ++metrics.shared_requested_two_sided_asset_count;
            }
            metrics.shared_absolute_inventory_sum += std::abs(
                static_cast<double>(asset.shared_inventory));
            if (cluster != nullptr && !shocked) {
                const int cluster_id = config_.shock_cluster_ids.at(
                    static_cast<std::size_t>(asset.asset_id));
                *cluster = {{
                    static_cast<double>(cluster_id),
                    1.0,
                    two_sided ? 1.0 : 0.0,
                    asset_spread,
                    asset_depth,
                    affected ? 1.0 : 0.0,
                    asset.shared_requested_quote_depth,
                    shared_resting}};
            }
        };

        AggregateMetricSums metrics;
        const bool fuse_clusters = config_.fuse_metric_cluster_scans
            && !config_.cluster_metrics_csv.empty();
        if (fuse_clusters) {
            fused_cluster_contributions_.assign(local_assets_.size(), {});
            fused_cluster_time_ns_ = time_ns;
        }
        if (!config_.parallel_metric_scans) {
            for (std::size_t index = 0; index < local_assets_.size(); ++index) {
                accumulate_asset(
                    metrics, *local_assets_[index],
                    fuse_clusters ? &fused_cluster_contributions_[index]
                                  : nullptr);
            }
            return metrics;
        }
        // Expensive book-state scans run concurrently.  The cheap final sum
        // remains in stable asset order, preserving byte-identical floating
        // output across rank/thread configurations.
        std::vector<AggregateMetricSums> contributions(local_assets_.size());
        for_each_local_index(local_assets_.size(), [&](std::size_t index) {
            accumulate_asset(
                contributions[index], *local_assets_[index],
                fuse_clusters ? &fused_cluster_contributions_[index] : nullptr);
        });
        const auto add = [](AggregateMetricSums& total,
                            const AggregateMetricSums& part) {
#define DLOB_ADD_METRIC(field) total.field += part.field
            DLOB_ADD_METRIC(spread_sum_bps);
            DLOB_ADD_METRIC(top_depth_sum);
            DLOB_ADD_METRIC(affected_asset_count);
            DLOB_ADD_METRIC(two_sided_book_count);
            DLOB_ADD_METRIC(shocked_asset_count);
            DLOB_ADD_METRIC(unshocked_asset_count);
            DLOB_ADD_METRIC(affected_shocked_asset_count);
            DLOB_ADD_METRIC(affected_unshocked_asset_count);
            DLOB_ADD_METRIC(shocked_spread_sum_bps);
            DLOB_ADD_METRIC(unshocked_spread_sum_bps);
            DLOB_ADD_METRIC(shocked_two_sided_asset_count);
            DLOB_ADD_METRIC(unshocked_two_sided_asset_count);
            DLOB_ADD_METRIC(shocked_top_depth_sum);
            DLOB_ADD_METRIC(unshocked_top_depth_sum);
            DLOB_ADD_METRIC(unshocked_shared_quote_depth_sum);
            DLOB_ADD_METRIC(shocked_shared_inventory_sum);
            DLOB_ADD_METRIC(value_order_count);
            DLOB_ADD_METRIC(value_requested_quantity);
            DLOB_ADD_METRIC(shared_requested_quote_depth_sum);
            DLOB_ADD_METRIC(shared_risk_reducing_quote_depth_sum);
            DLOB_ADD_METRIC(shared_risk_increasing_quote_depth_sum);
            DLOB_ADD_METRIC(shared_resting_quote_depth_sum);
            DLOB_ADD_METRIC(shared_risk_reducing_resting_depth_sum);
            DLOB_ADD_METRIC(shared_risk_increasing_resting_depth_sum);
            DLOB_ADD_METRIC(unshocked_shared_resting_depth_sum);
            DLOB_ADD_METRIC(shared_active_asset_count);
            DLOB_ADD_METRIC(shared_two_sided_active_asset_count);
            DLOB_ADD_METRIC(shocked_bid_top_depth_sum);
            DLOB_ADD_METRIC(shocked_shared_bid_resting_depth_sum);
            DLOB_ADD_METRIC(shared_best_bid_depth_sum);
            DLOB_ADD_METRIC(shared_best_ask_depth_sum);
            DLOB_ADD_METRIC(shared_at_best_bid_asset_count);
            DLOB_ADD_METRIC(shared_at_best_ask_asset_count);
            DLOB_ADD_METRIC(shared_nonzero_inventory_asset_count);
            DLOB_ADD_METRIC(shared_absolute_inventory_sum);
            DLOB_ADD_METRIC(shocked_shared_absolute_inventory_sum);
            DLOB_ADD_METRIC(shared_requested_active_asset_count);
            DLOB_ADD_METRIC(shared_requested_two_sided_asset_count);
#undef DLOB_ADD_METRIC
        };
        for (const AggregateMetricSums& contribution : contributions) {
            add(metrics, contribution);
        }
        return metrics;
    }

    void open_metrics_output() {
        if (rank_ != 0) return;
        if (!config_.metrics_csv.empty()) {
            const std::filesystem::path path(config_.metrics_csv);
            if (path.has_parent_path()) {
                std::filesystem::create_directories(path.parent_path());
            }
            metrics_output_.open(path);
            if (!metrics_output_) {
                throw std::runtime_error("cannot open distributed metrics CSV: "
                                         + path.string());
            }
            metrics_output_
            << "time_seconds,mean_spread_bps,mean_top_depth,"
               "affected_asset_fraction,two_sided_book_fraction,"
               "affected_shocked_fraction,"
               "affected_unshocked_fraction,shocked_mean_spread_bps,"
               "unshocked_mean_spread_bps,shocked_mean_top_depth,"
               "unshocked_mean_top_depth,shared_gross_exposure,"
               "shared_utilization,shared_quote_scale,"
               "unshocked_shared_requested_quote_depth,"
               "mean_shocked_shared_inventory,value_agent_order_count,"
               "value_agent_requested_quantity,shared_requested_quote_depth,"
               "shared_risk_reducing_requested_quote_depth,"
               "shared_risk_increasing_requested_quote_depth,"
               "shared_resting_quote_depth,"
               "shared_risk_reducing_resting_quote_depth,"
               "shared_risk_increasing_resting_quote_depth,"
               "unshocked_shared_resting_quote_depth,"
               "shared_active_asset_fraction,"
               "shared_two_sided_active_asset_fraction,"
               "shocked_bid_top_depth,"
               "shocked_shared_bid_resting_depth,"
               "shocked_shared_bid_participation,"
               "shared_nonzero_inventory_asset_fraction,"
               "mean_absolute_shared_inventory,"
               "mean_absolute_shocked_shared_inventory,"
               "shared_requested_active_asset_fraction,"
               "shared_requested_two_sided_asset_fraction,"
               "shared_best_bid_depth,shared_best_ask_depth,"
                   "shared_at_best_bid_asset_fraction,"
                   "shared_at_best_ask_asset_fraction,"
                   "shared_bbo_depth_participation\n";
        }
        if (!config_.cluster_metrics_csv.empty()) {
            const std::filesystem::path path(config_.cluster_metrics_csv);
            if (path.has_parent_path()) {
                std::filesystem::create_directories(path.parent_path());
            }
            cluster_metrics_output_.open(path);
            if (!cluster_metrics_output_) {
                throw std::runtime_error("cannot open cluster metrics CSV: "
                                         + path.string());
            }
            cluster_metrics_output_
                << "time_seconds,cluster_id,non_target_asset_count,"
                   "two_sided_book_fraction,mean_spread_bps,mean_top_depth,"
                   "affected_asset_fraction,shared_requested_quote_depth,"
                   "shared_resting_quote_depth\n";
        }
    }

    void observe_cluster_metrics(std::int64_t time_ns) {
        if (config_.cluster_metrics_csv.empty()) return;
        std::vector<double> local(
            static_cast<std::size_t>(cluster_count_)
                * cluster_metric_field_count,
            0.0);
        if (!config_.parallel_metric_scans
            && !config_.fuse_metric_cluster_scans) {
            // Preserve the baseline implementation exactly when neither
            // cluster optimization is selected.  This is the denominator of
            // the ablation matrix and must not pay for candidate data
            // structures.
            for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
                const LocalAsset& asset = *pointer;
                if (shock_mask_.at(
                        static_cast<std::size_t>(asset.asset_id))) {
                    continue;
                }
                const int cluster = config_.shock_cluster_ids.at(
                    static_cast<std::size_t>(asset.asset_id));
                const std::size_t offset =
                    static_cast<std::size_t>(cluster)
                    * cluster_metric_field_count;
                const MarketState state = asset.book.lob.state(
                    time_ns, asset.fundamental_value_ticks);
                const bool two_sided = state.best_bid_ticks > 0
                    && state.best_ask_ticks > state.best_bid_ticks;
                const double spread = two_sided
                    ? static_cast<double>(
                        state.best_ask_ticks - state.best_bid_ticks)
                        / asset.config.fundamental_price_ticks * 10'000.0
                    : 0.0;
                const double depth = static_cast<double>(state.best_bid_depth)
                    + static_cast<double>(state.best_ask_depth);
                const bool affected = !two_sided
                    || spread > 2.0 * asset.baseline_mean_spread_bps
                    || depth < 0.5 * asset.baseline_top_depth;
                local[offset] += 1.0;
                local[offset + 1U] += two_sided ? 1.0 : 0.0;
                local[offset + 2U] += spread;
                local[offset + 3U] += depth;
                local[offset + 4U] += affected ? 1.0 : 0.0;
                local[offset + 5U] += asset.shared_requested_quote_depth;
                local[offset + 6U] += static_cast<double>(
                    asset.book.lob.owner_resting_depth(
                        shared_market_maker_owner, Side::Buy)
                    + asset.book.lob.owner_resting_depth(
                        shared_market_maker_owner, Side::Sell));
            }
            finish_cluster_observation(time_ns, std::move(local));
            return;
        }
        struct ClusterContribution {
            int cluster = -1;
            std::array<double, cluster_metric_field_count> values{};
        };
        const auto scan_asset = [&](const LocalAsset& asset,
                                    ClusterContribution& contribution) {
            if (shock_mask_.at(static_cast<std::size_t>(asset.asset_id))) return;
            const int cluster = config_.shock_cluster_ids.at(
                static_cast<std::size_t>(asset.asset_id));
            const MarketState state = asset.book.lob.state(
                time_ns, asset.fundamental_value_ticks);
            const bool two_sided = state.best_bid_ticks > 0
                && state.best_ask_ticks > state.best_bid_ticks;
            const double spread = two_sided
                ? static_cast<double>(state.best_ask_ticks - state.best_bid_ticks)
                    / asset.config.fundamental_price_ticks * 10'000.0
                : 0.0;
            const double depth = static_cast<double>(state.best_bid_depth)
                + static_cast<double>(state.best_ask_depth);
            const bool affected = !two_sided
                || spread > 2.0 * asset.baseline_mean_spread_bps
                || depth < 0.5 * asset.baseline_top_depth;
            contribution.cluster = cluster;
            contribution.values[0] = 1.0;
            contribution.values[1] = two_sided ? 1.0 : 0.0;
            contribution.values[2] = spread;
            contribution.values[3] = depth;
            contribution.values[4] = affected ? 1.0 : 0.0;
            contribution.values[5] = asset.shared_requested_quote_depth;
            contribution.values[6] = static_cast<double>(
                asset.book.lob.owner_resting_depth(
                    shared_market_maker_owner, Side::Buy)
                + asset.book.lob.owner_resting_depth(
                    shared_market_maker_owner, Side::Sell));
        };
        std::vector<ClusterContribution> contributions(local_assets_.size());
        if (config_.fuse_metric_cluster_scans
            && fused_cluster_time_ns_ == time_ns
            && fused_cluster_contributions_.size() == local_assets_.size()) {
            for (std::size_t index = 0; index < contributions.size(); ++index) {
                const auto& cached = fused_cluster_contributions_[index];
                if (cached[1] == 0.0) continue;
                contributions[index].cluster = static_cast<int>(cached[0]);
                std::copy(
                    cached.begin() + 1, cached.end(),
                    contributions[index].values.begin());
            }
        } else {
            if (config_.parallel_metric_scans) {
                for_each_local_index(
                    local_assets_.size(), [&](std::size_t index) {
                        scan_asset(*local_assets_[index], contributions[index]);
                    });
            } else {
                for (std::size_t index = 0; index < local_assets_.size(); ++index) {
                    scan_asset(*local_assets_[index], contributions[index]);
                }
            }
        }
        for (const ClusterContribution& contribution : contributions) {
            if (contribution.cluster < 0) continue;
            const std::size_t offset =
                static_cast<std::size_t>(contribution.cluster)
                * cluster_metric_field_count;
            for (std::size_t field = 0;
                 field < cluster_metric_field_count; ++field) {
                local[offset + field] += contribution.values[field];
            }
        }
        finish_cluster_observation(time_ns, std::move(local));
    }

    void finish_cluster_observation(
        std::int64_t time_ns, std::vector<double> local) {
        if (config_.buffer_global_observations) {
            cluster_observations_.push_back(
                ClusterObservationFrame{time_ns, std::move(local)});
            return;
        }
        std::vector<double> global(local.size(), 0.0);
        const double start = MPI_Wtime();
        check_mpi(MPI_Allreduce(
                      local.data(), global.data(),
                      static_cast<int>(global.size()),
                      MPI_DOUBLE, MPI_SUM, communicator_),
                  "MPI_Allreduce(cluster metrics)");
        const double elapsed = MPI_Wtime() - start;
        communication_seconds_ += elapsed;
        observation_collective_seconds_ += elapsed;
        ++collective_calls_;
        ++observation_collective_calls_;
        write_cluster_observation(time_ns, global);
    }

    void write_cluster_observation(
        std::int64_t time_ns, const std::vector<double>& global) {
        if (rank_ != 0 || !cluster_metrics_output_) return;
        for (int cluster = 0; cluster < cluster_count_; ++cluster) {
            const std::size_t offset = static_cast<std::size_t>(cluster)
                * cluster_metric_field_count;
            const double count = global[offset];
            const double two_sided = global[offset + 1U];
            cluster_metrics_output_ << std::fixed << std::setprecision(9)
                << static_cast<double>(time_ns) / 1e9 << ','
                << cluster << ',' << count << ','
                << (count > 0.0 ? two_sided / count : 0.0) << ','
                << (two_sided > 0.0 ? global[offset + 2U] / two_sided : 0.0) << ','
                << (count > 0.0 ? global[offset + 3U] / count : 0.0) << ','
                << (count > 0.0 ? global[offset + 4U] / count : 0.0) << ','
                << global[offset + 5U] << ','
                << global[offset + 6U] << '\n';
        }
    }

    void observe_global_metrics(std::int64_t time_ns) {
        const double local_metrics_start = active_window_phase_ != nullptr
            ? MPI_Wtime() : 0.0;
        const AggregateMetricSums local = local_metrics(time_ns);
        if (active_window_phase_ != nullptr) {
            active_window_phase_->global_metrics_local_seconds +=
                MPI_Wtime() - local_metrics_start;
        }
        GlobalObservationFrame frame;
        frame.time_ns = time_ns;
        frame.local = {{
            local.spread_sum_bps,
            local.top_depth_sum,
            local.affected_asset_count,
            local.two_sided_book_count,
            local.shocked_asset_count,
            local.unshocked_asset_count,
            local.affected_shocked_asset_count,
            local.affected_unshocked_asset_count,
            local.shocked_spread_sum_bps,
            local.unshocked_spread_sum_bps,
            local.shocked_two_sided_asset_count,
            local.unshocked_two_sided_asset_count,
            local.shocked_top_depth_sum,
            local.unshocked_top_depth_sum,
            local.unshocked_shared_quote_depth_sum,
            local.shocked_shared_inventory_sum,
            local.value_order_count,
            local.value_requested_quantity,
            local.shared_requested_quote_depth_sum,
            local.shared_risk_reducing_quote_depth_sum,
            local.shared_risk_increasing_quote_depth_sum,
            local.shared_active_asset_count,
            local.shared_resting_quote_depth_sum,
            local.shared_risk_reducing_resting_depth_sum,
            local.shared_risk_increasing_resting_depth_sum,
            local.unshocked_shared_resting_depth_sum,
            local.shared_two_sided_active_asset_count,
            local.shocked_bid_top_depth_sum,
            local.shocked_shared_bid_resting_depth_sum,
            local.shared_nonzero_inventory_asset_count,
            local.shared_absolute_inventory_sum,
            local.shocked_shared_absolute_inventory_sum,
            local.shared_requested_active_asset_count,
            local.shared_requested_two_sided_asset_count,
            local.shared_best_bid_depth_sum,
            local.shared_best_ask_depth_sum,
            local.shared_at_best_bid_asset_count,
            local.shared_at_best_ask_asset_count}};
        frame.shared_gross_exposure = shared_gross_exposure_;
        frame.shared_utilization = shared_utilization_;
        frame.shared_quote_scale = shared_quote_scale_;
        if (config_.buffer_global_observations
            && config_.enable_shared_market_maker
            && (!config_.enable_global_shared_capacity
                || last_risk_boundary_was_skipped_)) {
            if (risk_observations_.empty()
                || risk_observations_.back().time_ns != time_ns) {
                throw std::logic_error(
                    "buffered observation is missing its risk frame");
            }
            frame.risk_frame_index = risk_observations_.size() - 1U;
        }
        if (config_.buffer_global_observations) {
            global_observations_.push_back(std::move(frame));
            observe_cluster_metrics(time_ns);
            return;
        }
        std::array<double, global_metric_field_count> global{};
        const double start = MPI_Wtime();
        check_mpi(MPI_Allreduce(frame.local.data(), global.data(),
                                static_cast<int>(global.size()),
                                MPI_DOUBLE, MPI_SUM, communicator_),
                  "MPI_Allreduce(distributed metrics)");
        const double elapsed = MPI_Wtime() - start;
        communication_seconds_ += elapsed;
        observation_collective_seconds_ += elapsed;
        if (active_window_phase_ != nullptr) {
            active_window_phase_->global_metrics_collective_seconds +=
                elapsed;
        }
        ++collective_calls_;
        ++observation_collective_calls_;
        consume_global_observation(frame, global);
        observe_cluster_metrics(time_ns);
    }

    void consume_global_observation(
        const GlobalObservationFrame& frame,
        const std::array<double, global_metric_field_count>& global) {
        AggregateMetrics metrics;
        metrics.mean_spread_bps = global[3] > 0.0 ? global[0] / global[3] : 0.0;
        const double observed_asset_count = global[4] + global[5];
        metrics.mean_top_depth = observed_asset_count > 0.0
            ? global[1] / observed_asset_count : 0.0;
        metrics.affected_fraction = global[2]
            / static_cast<double>(config_.asset_count);
        metrics.two_sided_book_fraction = global[3]
            / static_cast<double>(config_.asset_count);
        metrics.affected_shocked_fraction = global[4] > 0.0
            ? global[6] / global[4] : 0.0;
        metrics.affected_unshocked_fraction = global[5] > 0.0
            ? global[7] / global[5] : 0.0;
        metrics.shocked_mean_spread_bps = global[10] > 0.0
            ? global[8] / global[10] : 0.0;
        metrics.unshocked_mean_spread_bps = global[11] > 0.0
            ? global[9] / global[11] : 0.0;
        metrics.shocked_mean_top_depth = global[4] > 0.0
            ? global[12] / global[4] : 0.0;
        metrics.unshocked_mean_top_depth = global[5] > 0.0
            ? global[13] / global[5] : 0.0;
        last_metrics_ = metrics;
        peak_affected_fraction_ = std::max(
            peak_affected_fraction_, metrics.affected_fraction);
        peak_mean_spread_bps_ = std::max(
            peak_mean_spread_bps_, metrics.mean_spread_bps);
        peak_affected_unshocked_fraction_ = std::max(
            peak_affected_unshocked_fraction_, metrics.affected_unshocked_fraction);
        minimum_two_sided_book_fraction_ = std::min(
            minimum_two_sided_book_fraction_, metrics.two_sided_book_fraction);

        if (rank_ == 0 && metrics_output_) {
            metrics_output_ << std::fixed << std::setprecision(9)
                << static_cast<double>(frame.time_ns) / 1e9 << ','
                << metrics.mean_spread_bps << ','
                << metrics.mean_top_depth << ','
                << metrics.affected_fraction << ','
                << metrics.two_sided_book_fraction << ','
                << metrics.affected_shocked_fraction << ','
                << metrics.affected_unshocked_fraction << ','
                << metrics.shocked_mean_spread_bps << ','
                << metrics.unshocked_mean_spread_bps << ','
                << metrics.shocked_mean_top_depth << ','
                << metrics.unshocked_mean_top_depth << ','
                << frame.shared_gross_exposure << ','
                << frame.shared_utilization << ','
                << frame.shared_quote_scale << ','
                << global[14] << ','
                << (global[4] > 0.0 ? global[15] / global[4] : 0.0) << ','
                << global[16] << ','
                << global[17] << ','
                << global[18] << ','
                << global[19] << ','
                << global[20] << ','
                << global[22] << ','
                << global[23] << ','
                << global[24] << ','
                << global[25] << ','
                << global[21] / static_cast<double>(config_.asset_count)
                << ',' << global[26] / static_cast<double>(config_.asset_count)
                << ',' << global[27]
                << ',' << global[28]
                << ',' << (global[27] > 0.0 ? global[28] / global[27] : 0.0)
                << ',' << global[29] / static_cast<double>(config_.asset_count)
                << ',' << global[30] / static_cast<double>(config_.asset_count)
                << ',' << (global[4] > 0.0 ? global[31] / global[4] : 0.0)
                << ',' << global[32] / static_cast<double>(config_.asset_count)
                << ',' << global[33] / static_cast<double>(config_.asset_count)
                << ',' << global[34]
                << ',' << global[35]
                << ',' << global[36] / static_cast<double>(config_.asset_count)
                << ',' << global[37] / static_cast<double>(config_.asset_count)
                << ',' << (global[1] > 0.0
                    ? (global[34] + global[35]) / global[1] : 0.0)
                << '\n';
        }
    }

    void flush_buffered_observations() {
        if (!config_.buffer_global_observations) return;

        if (!risk_observations_.empty()) {
            const std::size_t count = risk_observations_.size();
            if (count > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
                throw std::overflow_error(
                    "buffered risk observations exceed MPI count limit");
            }
            std::vector<long long> local_fixed;
            std::vector<double> local_scale;
            local_fixed.reserve(count);
            local_scale.reserve(count);
            for (const RiskObservationFrame& frame : risk_observations_) {
                local_fixed.push_back(frame.local_fixed_exposure);
                local_scale.push_back(frame.local_uncoupled_scale_sum);
            }
            std::vector<long long> global_fixed(
                rank_ == 0 ? count : 0U, 0);
            std::vector<double> global_scale;
            if (!config_.enable_global_shared_capacity) {
                global_scale.assign(rank_ == 0 ? count : 0U, 0.0);
            }
            const double start = MPI_Wtime();
            check_mpi(MPI_Reduce(
                          local_fixed.data(),
                          rank_ == 0 ? global_fixed.data() : nullptr,
                          static_cast<int>(count), MPI_LONG_LONG, MPI_SUM, 0,
                          communicator_),
                      "MPI_Reduce(buffered uncoupled exposure)");
            if (!config_.enable_global_shared_capacity) {
                check_mpi(MPI_Reduce(
                              local_scale.data(),
                              rank_ == 0 ? global_scale.data() : nullptr,
                              static_cast<int>(count), MPI_DOUBLE, MPI_SUM, 0,
                              communicator_),
                          "MPI_Reduce(buffered uncoupled quote scale)");
            }
            const double elapsed = MPI_Wtime() - start;
            communication_seconds_ += elapsed;
            observation_collective_seconds_ += elapsed;
            const std::uint64_t reductions =
                config_.enable_global_shared_capacity ? 1U : 2U;
            collective_calls_ += reductions;
            observation_collective_calls_ += reductions;
            if (rank_ == 0) {
                const double global_limit =
                    config_.shared_global_risk_limit_per_asset
                    * static_cast<double>(config_.asset_count);
                std::vector<double> exposures(count, 0.0);
                std::vector<double> utilizations(count, 0.0);
                std::vector<double> scales(count, 1.0);
                if (!config_.enable_global_shared_capacity) {
                    minimum_shared_quote_scale_ = 1.0;
                    withdrawal_windows_ = 0;
                }
                for (std::size_t index = 0; index < count; ++index) {
                    exposures[index] = static_cast<double>(global_fixed[index])
                        / risk_fixed_point_scale;
                    utilizations[index] = exposures[index] / global_limit;
                    if (!config_.enable_global_shared_capacity) {
                        scales[index] = global_scale[index]
                            / static_cast<double>(config_.asset_count);
                        minimum_shared_quote_scale_ = std::min(
                            minimum_shared_quote_scale_, scales[index]);
                        if (scales[index] < 0.5) ++withdrawal_windows_;
                    }
                }
                if (!config_.enable_global_shared_capacity) {
                    shared_gross_exposure_ = exposures.back();
                    maximum_shared_gross_exposure_ = std::max(
                        maximum_shared_gross_exposure_,
                        shared_gross_exposure_);
                    shared_utilization_ = utilizations.back();
                    shared_quote_scale_ = scales.back();
                }
                for (GlobalObservationFrame& frame : global_observations_) {
                    if (frame.risk_frame_index
                        == std::numeric_limits<std::size_t>::max()) {
                        continue;
                    }
                    if (frame.risk_frame_index >= count) {
                        throw std::logic_error(
                            "buffered global observation has invalid risk frame");
                    }
                    frame.shared_gross_exposure =
                        exposures[frame.risk_frame_index];
                    frame.shared_utilization =
                        utilizations[frame.risk_frame_index];
                    frame.shared_quote_scale = scales[frame.risk_frame_index];
                }
            }
        }

        if (!global_observations_.empty()) {
            const std::size_t count = global_observations_.size()
                * global_metric_field_count;
            if (count > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
                throw std::overflow_error(
                    "buffered global observations exceed MPI count limit");
            }
            std::vector<double> local;
            local.reserve(count);
            for (const GlobalObservationFrame& frame : global_observations_) {
                local.insert(local.end(), frame.local.begin(), frame.local.end());
            }
            std::vector<double> global(
                rank_ == 0 ? count : 0U, 0.0);
            const double start = MPI_Wtime();
            check_mpi(MPI_Reduce(
                          local.data(), rank_ == 0 ? global.data() : nullptr,
                          static_cast<int>(count), MPI_DOUBLE, MPI_SUM, 0,
                          communicator_),
                      "MPI_Reduce(buffered global observations)");
            const double elapsed = MPI_Wtime() - start;
            communication_seconds_ += elapsed;
            observation_collective_seconds_ += elapsed;
            ++collective_calls_;
            ++observation_collective_calls_;
            if (rank_ == 0) {
                for (std::size_t frame_index = 0;
                     frame_index < global_observations_.size(); ++frame_index) {
                    std::array<double, global_metric_field_count> values{};
                    const auto begin = global.begin()
                        + static_cast<std::ptrdiff_t>(
                            frame_index * global_metric_field_count);
                    std::copy_n(begin, global_metric_field_count, values.begin());
                    consume_global_observation(
                        global_observations_[frame_index], values);
                }
            }
        }

        if (!cluster_observations_.empty()) {
            const std::size_t frame_width = static_cast<std::size_t>(
                cluster_count_) * cluster_metric_field_count;
            const std::size_t count = cluster_observations_.size() * frame_width;
            if (count > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
                throw std::overflow_error(
                    "buffered cluster observations exceed MPI count limit");
            }
            std::vector<double> local;
            local.reserve(count);
            for (const ClusterObservationFrame& frame : cluster_observations_) {
                if (frame.local.size() != frame_width) {
                    throw std::logic_error(
                        "inconsistent buffered cluster observation width");
                }
                local.insert(local.end(), frame.local.begin(), frame.local.end());
            }
            std::vector<double> global(
                rank_ == 0 ? count : 0U, 0.0);
            const double start = MPI_Wtime();
            check_mpi(MPI_Reduce(
                          local.data(), rank_ == 0 ? global.data() : nullptr,
                          static_cast<int>(count), MPI_DOUBLE, MPI_SUM, 0,
                          communicator_),
                      "MPI_Reduce(buffered cluster observations)");
            const double elapsed = MPI_Wtime() - start;
            communication_seconds_ += elapsed;
            observation_collective_seconds_ += elapsed;
            ++collective_calls_;
            ++observation_collective_calls_;
            if (rank_ == 0) {
                for (std::size_t frame_index = 0;
                     frame_index < cluster_observations_.size(); ++frame_index) {
                    const auto begin = global.begin()
                        + static_cast<std::ptrdiff_t>(frame_index * frame_width);
                    std::vector<double> values(begin, begin
                        + static_cast<std::ptrdiff_t>(frame_width));
                    write_cluster_observation(
                        cluster_observations_[frame_index].time_ns, values);
                }
            }
        }

        global_observations_.clear();
        cluster_observations_.clear();
        risk_observations_.clear();
    }

    template <typename Value>
    std::vector<Value> gather_values(const std::vector<Value>& local,
                                     const char* label) {
        static_assert(std::is_trivially_copyable_v<Value>);
        const int local_bytes = checked_bytes(local.size(), sizeof(Value), label);
        std::vector<int> counts(rank_ == 0
            ? static_cast<std::size_t>(world_size_) : 0U);
        const double start = MPI_Wtime();
        check_mpi(MPI_Gather(&local_bytes, 1, MPI_INT,
                             rank_ == 0 ? counts.data() : nullptr,
                             1, MPI_INT, 0, communicator_),
                  "MPI_Gather(distributed counts)");
        ++collective_calls_;
        ++terminal_collective_calls_;
        int total_bytes = 0;
        std::vector<int> displacements;
        if (rank_ == 0) {
            displacements.resize(static_cast<std::size_t>(world_size_));
            for (int index = 0; index < world_size_; ++index) {
                const int count = counts[static_cast<std::size_t>(index)];
                if (count < 0
                    || total_bytes > std::numeric_limits<int>::max() - count) {
                    throw std::overflow_error("distributed gather size overflow");
                }
                displacements[static_cast<std::size_t>(index)] = total_bytes;
                total_bytes += count;
            }
            if (total_bytes % static_cast<int>(sizeof(Value)) != 0) {
                throw std::logic_error("distributed gather contains partial values");
            }
        }
        std::vector<Value> gathered(rank_ == 0
            ? static_cast<std::size_t>(total_bytes)
                / static_cast<std::size_t>(sizeof(Value)) : 0U);
        check_mpi(MPI_Gatherv(local.empty() ? nullptr : local.data(),
                              local_bytes, MPI_BYTE,
                              rank_ == 0 && !gathered.empty()
                                  ? gathered.data() : nullptr,
                              rank_ == 0 ? counts.data() : nullptr,
                              rank_ == 0 ? displacements.data() : nullptr,
                              MPI_BYTE, 0, communicator_),
                  "MPI_Gatherv(distributed values)");
        ++collective_calls_;
        ++terminal_collective_calls_;
        const double elapsed = MPI_Wtime() - start;
        communication_seconds_ += elapsed;
        terminal_collective_seconds_ += elapsed;
        return gathered;
    }

    std::vector<BookResultWire> gather_book_results() {
        std::vector<BookResultWire> local;
        local.reserve(local_assets_.size());
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            const LocalBook& book = asset->book;
            BookResultWire wire;
            wire.state = book.lob.state(
                end_time_ns_, asset->fundamental_value_ticks);
            wire.processed_orders = book.processed_orders;
            wire.trade_count = book.trade_count;
            local.push_back(wire);
        }
        return gather_values(local, "distributed book result");
    }

    std::vector<AssetResultWire> gather_asset_results() {
        std::vector<AssetResultWire> local;
        local.reserve(local_assets_.size());
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            AssetResultWire wire;
            wire.asset_id = asset->asset_id;
            wire.shared_inventory = asset->shared_inventory;
            wire.shared_cash_ticks = asset->shared_cash_ticks;
            wire.shared_buy_quantity = asset->shared_buy_quantity;
            wire.shared_sell_quantity = asset->shared_sell_quantity;
            wire.shared_fill_count = asset->shared_fill_count;
            const MarketState external_state = asset->book.lob.state_excluding_owner(
                end_time_ns_, asset->fundamental_value_ticks,
                shared_market_maker_owner);
            wire.shared_mark_mid_ticks = external_state.mid_price_ticks > 0.0
                ? external_state.mid_price_ticks
                : asset->book.lob.mid_price();
            if (config_.enable_shared_financial_diagnostics) {
                const TerminalLiquidationPreview liquidation =
                    asset->book.lob.preview_terminal_liquidation(
                        asset->shared_inventory,
                        shared_market_maker_owner,
                        config_.shared_terminal_fallback_distance_ticks);
                wire.shared_liquidation_cash_change_ticks =
                    static_cast<double>(liquidation.signed_cash_change_ticks);
                wire.shared_liquidation_unliquidated_quantity =
                    static_cast<std::uint64_t>(
                        liquidation.unliquidated_quantity);
            }
            wire.value_inventory = asset->value_inventory;
            wire.shock_requested_quantity = asset->shock_requested_quantity;
            wire.shock_side_code = asset->shock_side_code;
            wire.shock_pre_shared_inventory =
                asset->shock_pre_shared_inventory;
            wire.fundamental_value_ticks = asset->fundamental_value_ticks;
            wire.fundamental_log_variance =
                asset->fundamental_log_variance;
            wire.value_recheck_due_ns = asset->value_recheck_due_ns;
            wire.remaining_value_rechecks = asset->remaining_value_rechecks;
            local.push_back(wire);
        }
        return gather_values(local, "distributed asset result");
    }

    std::vector<AssetMomentWire> gather_asset_moments() {
        if (config_.asset_summary_csv.empty()) return {};
        std::vector<AssetMomentWire> local;
        local.reserve(local_assets_.size());
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            local.push_back(AssetMomentWire{
                asset->asset_id,
                asset->calibration_moments.snapshots,
                asset->calibration_moments.invalid_snapshots,
                asset->hawkes.accepted_events(),
                asset->background_limit_requested_quantity,
                asset->background_limit_resting_quantity,
                asset->background_market_requested_quantity,
                asset->background_market_executed_quantity,
                asset->background_cancel_requested_quantity,
                asset->background_cancel_effective_quantity,
                asset->removal_boundary_truncation_events,
                asset->removal_boundary_truncated_quantity,
                asset->market_boundary_truncation_events,
                asset->market_boundary_truncated_quantity,
                asset->cancel_boundary_truncation_events,
                asset->cancel_boundary_truncated_quantity,
                asset->background_boundary_truncation_events,
                asset->background_boundary_truncated_quantity,
                asset->value_order_count,
                asset->value_requested_quantity,
                asset->value_boundary_truncation_events,
                asset->value_boundary_truncated_quantity,
                asset->other_boundary_truncation_events,
                asset->other_boundary_truncated_quantity,
                asset->local_quote_revision_requested_quantity,
                asset->local_quote_revision_moved_quantity,
                asset->local_quote_revision_no_donor_events,
                asset->calibration_moments.finalize()});
        }
        return gather_values(local, "distributed asset moment");
    }

    std::vector<AssetWorkWire> gather_asset_work() {
        if (config_.asset_work_csv.empty()) return {};
        std::vector<AssetWorkWire> local;
        local.reserve(local_assets_.size());
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            local.push_back(AssetWorkWire{
                asset->asset_id,
                static_cast<std::int32_t>(rank_),
                asset->measured_processed_orders,
                asset->hawkes.accepted_events(),
                asset->measured_processing_nanoseconds});
        }
        return gather_values(local, "distributed asset work");
    }

    void write_asset_work(std::vector<AssetWorkWire> work) const {
        if (rank_ != 0 || config_.asset_work_csv.empty()) return;
        if (work.size() != static_cast<std::size_t>(config_.asset_count)) {
            throw std::logic_error("incomplete per-asset work profile");
        }
        std::sort(work.begin(), work.end(),
                  [](const AssetWorkWire& left, const AssetWorkWire& right) {
                      return left.asset_id < right.asset_id;
                  });
        const std::filesystem::path path(config_.asset_work_csv);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error(
                "cannot open per-asset work CSV: " + path.string());
        }
        output << "asset_id,symbol,cluster_id,is_shock_target,owner_rank,"
                  "processed_orders,background_events,processing_nanoseconds,"
                  "processing_seconds,nanoseconds_per_order\n";
        output << std::setprecision(17);
        for (const AssetWorkWire& row : work) {
            const std::size_t index = static_cast<std::size_t>(row.asset_id);
            if (index >= config_.asset_configs.size()) {
                throw std::logic_error("invalid asset in per-asset work profile");
            }
            output << row.asset_id << ','
                   << config_.asset_configs[index].symbol << ','
                   << (config_.shock_cluster_ids.empty()
                       ? -1 : config_.shock_cluster_ids[index]) << ','
                   << (shock_mask_[index] ? 1 : 0) << ','
                   << row.owner_rank << ','
                   << row.processed_orders << ','
                   << row.background_events << ','
                   << row.processing_nanoseconds << ','
                   << static_cast<double>(row.processing_nanoseconds) / 1e9
                   << ','
                   << (row.processed_orders > 0
                       ? static_cast<double>(row.processing_nanoseconds)
                           / static_cast<double>(row.processed_orders)
                       : 0.0)
                   << '\n';
        }
    }

    std::vector<BoundaryArrivalWire> gather_boundary_arrivals() {
        if (config_.boundary_arrival_csv.empty()) return {};
        return gather_values(boundary_arrivals_, "risk-boundary arrivals");
    }

    void write_boundary_arrivals(
        std::vector<BoundaryArrivalWire> arrivals) const {
        if (rank_ != 0 || config_.boundary_arrival_csv.empty()) return;
        std::sort(arrivals.begin(), arrivals.end(),
                  [](const BoundaryArrivalWire& left,
                     const BoundaryArrivalWire& right) {
                      if (left.boundary_index != right.boundary_index) {
                          return left.boundary_index < right.boundary_index;
                      }
                      return left.rank < right.rank;
                  });
        const std::filesystem::path path(config_.boundary_arrival_csv);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error(
                "cannot open boundary-arrival CSV: " + path.string());
        }
        output << "boundary_index,time_seconds,rank,arrival_seconds,"
                  "work_interval_seconds,work_interval_spread_seconds,"
                  "collective_seconds\n";
        output << std::setprecision(17);
        std::size_t begin = 0;
        while (begin < arrivals.size()) {
            std::size_t end = begin + 1U;
            while (end < arrivals.size()
                   && arrivals[end].boundary_index
                       == arrivals[begin].boundary_index) {
                ++end;
            }
            double minimum = arrivals[begin].work_interval_seconds;
            double maximum = minimum;
            for (std::size_t index = begin + 1U; index < end; ++index) {
                minimum = std::min(
                    minimum, arrivals[index].work_interval_seconds);
                maximum = std::max(
                    maximum, arrivals[index].work_interval_seconds);
            }
            for (std::size_t index = begin; index < end; ++index) {
                const BoundaryArrivalWire& row = arrivals[index];
                output << row.boundary_index << ','
                       << static_cast<double>(row.time_ns) / 1e9 << ','
                       << row.rank << ',' << row.arrival_seconds << ','
                       << row.work_interval_seconds << ','
                       << maximum - minimum << ','
                       << row.collective_seconds << '\n';
            }
            begin = end;
        }
    }

    std::vector<WindowPhaseWire> gather_window_phase_profiles() {
        if (config_.window_phase_profile_csv.empty()) return {};
        return gather_values(window_phase_profiles_, "window phase profiles");
    }

    void write_window_phase_profiles(
        std::vector<WindowPhaseWire> rows) const {
        if (rank_ != 0 || config_.window_phase_profile_csv.empty()) return;
        std::sort(rows.begin(), rows.end(),
                  [](const WindowPhaseWire& left,
                     const WindowPhaseWire& right) {
                      if (left.window_index != right.window_index) {
                          return left.window_index < right.window_index;
                      }
                      return left.rank < right.rank;
                  });
        if (rows.empty()
            || rows.size() % static_cast<std::size_t>(world_size_) != 0U) {
            throw std::logic_error("incomplete window phase profile");
        }
        for (std::size_t begin = 0; begin < rows.size();
             begin += static_cast<std::size_t>(world_size_)) {
            const WindowPhaseWire& first = rows[begin];
            for (int rank = 0; rank < world_size_; ++rank) {
                const WindowPhaseWire& row = rows[
                    begin + static_cast<std::size_t>(rank)];
                if (row.window_index != first.window_index
                    || row.start_time_ns != first.start_time_ns
                    || row.end_time_ns != first.end_time_ns
                    || row.rank != rank) {
                    throw std::logic_error(
                        "inconsistent rank coverage in window phase profile");
                }
            }
        }
        const std::filesystem::path path(config_.window_phase_profile_csv);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error(
                "cannot open window phase profile CSV: " + path.string());
        }
        output << "window_index,start_time_seconds,end_time_seconds,rank,"
                  "event_processing_seconds,risk_local_seconds,"
                  "risk_collective_seconds,asset_moments_seconds,"
                  "return_panel_seconds,local_market_maker_seconds,"
                  "risk_finalize_seconds,global_metrics_local_seconds,"
                  "global_metrics_collective_seconds,"
                  "global_metrics_write_seconds,fundamental_seconds,"
                  "shared_market_maker_seconds,news_value_agent_seconds,"
                  "periodic_value_agent_seconds,other_seconds,"
                  "total_window_seconds\n";
        output << std::setprecision(17);
        for (const WindowPhaseWire& row : rows) {
            output << row.window_index << ','
                   << static_cast<double>(row.start_time_ns) / 1e9 << ','
                   << static_cast<double>(row.end_time_ns) / 1e9 << ','
                   << row.rank << ','
                   << row.event_processing_seconds << ','
                   << row.risk_local_seconds << ','
                   << row.risk_collective_seconds << ','
                   << row.asset_moments_seconds << ','
                   << row.return_panel_seconds << ','
                   << row.local_market_maker_seconds << ','
                   << row.risk_finalize_seconds << ','
                   << row.global_metrics_local_seconds << ','
                   << row.global_metrics_collective_seconds << ','
                   << row.global_metrics_write_seconds << ','
                   << row.fundamental_seconds << ','
                   << row.shared_market_maker_seconds << ','
                   << row.news_value_agent_seconds << ','
                   << row.periodic_value_agent_seconds << ','
                   << row.other_seconds << ','
                   << row.total_window_seconds << '\n';
        }
    }

    void write_asset_summary(std::vector<AssetMomentWire> moments) const {
        if (rank_ != 0 || config_.asset_summary_csv.empty()) return;
        if (moments.size() != static_cast<std::size_t>(config_.asset_count)) {
            throw std::logic_error("incomplete distributed per-asset moment summary");
        }
        std::sort(moments.begin(), moments.end(),
                  [](const AssetMomentWire& left, const AssetMomentWire& right) {
                      return left.asset_id < right.asset_id;
                  });
        const std::filesystem::path path(config_.asset_summary_csv);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error(
                "cannot open distributed per-asset summary CSV: " + path.string());
        }
        const std::uint64_t expected_samples = static_cast<std::uint64_t>(
            end_time_ns_ / config_.asset_summary_interval_ns);
        output << "asset_id,symbol,sample_count,expected_sample_count,"
                  "invalid_sample_count,two_sided_sample_fraction,structurally_valid,"
                  "background_event_count,background_event_rate,"
                  "background_limit_requested_quantity,"
                  "background_limit_resting_quantity,"
                  "background_limit_resting_fraction,"
                  "background_market_requested_quantity,"
                  "background_market_executed_quantity,"
                  "background_market_execution_fraction,"
                  "background_cancel_requested_quantity,"
                  "background_cancel_effective_quantity,"
                  "background_cancel_effective_fraction,"
                  "removal_boundary_truncation_events,"
                  "removal_boundary_truncated_quantity,"
                  "market_boundary_truncation_events,"
                  "market_boundary_truncated_quantity,"
                  "cancel_boundary_truncation_events,"
                  "cancel_boundary_truncated_quantity,"
                  "background_boundary_truncation_events,"
                  "background_boundary_truncated_quantity,"
                  "value_order_count,value_requested_quantity,"
                  "value_boundary_truncation_events,"
                  "value_boundary_truncated_quantity,"
                  "other_boundary_truncation_events,"
                  "other_boundary_truncated_quantity,"
                  "local_quote_revision_requested_quantity,"
                  "local_quote_revision_moved_quantity,"
                  "local_quote_revision_moved_fraction,"
                  "local_quote_revision_no_donor_events,"
                  "mean_spread_ticks,mean_bid_depth,mean_ask_depth,"
                  "mid_move_rate,return_variance,return_kurtosis,"
                  "absolute_return_acf1\n";
        output << std::setprecision(17);
        for (const AssetMomentWire& moment : moments) {
            if (moment.asset_id >= config_.asset_configs.size()) {
                throw std::logic_error("per-asset moment has an invalid asset id");
            }
            const std::uint64_t observed_samples = moment.sample_count
                + moment.invalid_sample_count;
            const double valid_fraction = expected_samples > 0
                ? static_cast<double>(moment.sample_count)
                    / static_cast<double>(expected_samples)
                : 0.0;
            // The certified calibration protocol requires a complete
            // two-sided fixed clock.  Keep the valid/invalid counts for
            // diagnosis, but do not label a one-sided execution path
            // structurally valid merely because its accounting sums.
            const bool structurally_valid = expected_samples > 0
                && observed_samples == expected_samples
                && moment.invalid_sample_count == 0;
            output << moment.asset_id << ','
                   << config_.asset_configs[
                       static_cast<std::size_t>(moment.asset_id)].symbol << ','
                   << moment.sample_count << ','
                   << expected_samples << ','
                   << moment.invalid_sample_count << ','
                   << valid_fraction << ','
                   << (structurally_valid ? 1 : 0) << ','
                   << moment.background_event_count << ','
                   << static_cast<double>(moment.background_event_count)
                        / static_cast<double>(config_.duration_seconds) << ','
                   << moment.background_limit_requested_quantity << ','
                   << moment.background_limit_resting_quantity << ','
                   << (moment.background_limit_requested_quantity > 0
                       ? static_cast<double>(
                           moment.background_limit_resting_quantity)
                           / static_cast<double>(
                               moment.background_limit_requested_quantity)
                       : 0.0) << ','
                   << moment.background_market_requested_quantity << ','
                   << moment.background_market_executed_quantity << ','
                   << (moment.background_market_requested_quantity > 0
                       ? static_cast<double>(
                           moment.background_market_executed_quantity)
                           / static_cast<double>(
                               moment.background_market_requested_quantity)
                       : 0.0) << ','
                   << moment.background_cancel_requested_quantity << ','
                   << moment.background_cancel_effective_quantity << ','
                   << (moment.background_cancel_requested_quantity > 0
                       ? static_cast<double>(
                           moment.background_cancel_effective_quantity)
                           / static_cast<double>(
                               moment.background_cancel_requested_quantity)
                       : 0.0) << ','
                   << moment.removal_boundary_truncation_events << ','
                   << moment.removal_boundary_truncated_quantity << ','
                   << moment.market_boundary_truncation_events << ','
                   << moment.market_boundary_truncated_quantity << ','
                   << moment.cancel_boundary_truncation_events << ','
                   << moment.cancel_boundary_truncated_quantity << ','
                   << moment.background_boundary_truncation_events << ','
                   << moment.background_boundary_truncated_quantity << ','
                   << moment.value_order_count << ','
                   << moment.value_requested_quantity << ','
                   << moment.value_boundary_truncation_events << ','
                   << moment.value_boundary_truncated_quantity << ','
                   << moment.other_boundary_truncation_events << ','
                   << moment.other_boundary_truncated_quantity << ','
                   << moment.local_quote_revision_requested_quantity << ','
                   << moment.local_quote_revision_moved_quantity << ','
                   << (moment.local_quote_revision_requested_quantity > 0
                       ? static_cast<double>(
                           moment.local_quote_revision_moved_quantity)
                           / static_cast<double>(
                               moment.local_quote_revision_requested_quantity)
                       : 0.0) << ','
                   << moment.local_quote_revision_no_donor_events;
            for (const double value : moment.values) output << ',' << value;
            output << '\n';
        }
    }

    void compute_shared_financial_results(
        std::vector<BookResultWire> books,
        std::vector<AssetResultWire> assets) {
        if (rank_ != 0) return;
        std::sort(books.begin(), books.end(),
                  [](const BookResultWire& left, const BookResultWire& right) {
                      return left.state.book_id < right.state.book_id;
                  });
        std::sort(assets.begin(), assets.end(),
                  [](const AssetResultWire& left, const AssetResultWire& right) {
                      return left.asset_id < right.asset_id;
                  });
        if (books.size() != static_cast<std::size_t>(config_.asset_count)
            || assets.size() != static_cast<std::size_t>(config_.asset_count)) {
            throw std::logic_error("incomplete final market result");
        }

        long double mark_to_mid_ticks = 0.0L;
        long double liquidation_ticks = 0.0L;
        unsigned __int128 terminal_absolute_inventory = 0;
        unsigned __int128 unliquidated_quantity = 0;
        unsigned __int128 buy_quantity = 0;
        unsigned __int128 sell_quantity = 0;
        unsigned __int128 fill_count = 0;
        for (std::size_t index = 0; index < assets.size(); ++index) {
            if (books[index].state.book_id != assets[index].asset_id) {
                throw std::logic_error("book/asset mismatch in final result");
            }
            const AssetResultWire& asset = assets[index];
            mark_to_mid_ticks += static_cast<long double>(asset.shared_cash_ticks)
                + static_cast<long double>(asset.shared_inventory)
                    * static_cast<long double>(asset.shared_mark_mid_ticks);
            liquidation_ticks += static_cast<long double>(asset.shared_cash_ticks)
                + static_cast<long double>(
                    asset.shared_liquidation_cash_change_ticks);
            const __int128 inventory = static_cast<__int128>(
                asset.shared_inventory);
            terminal_absolute_inventory += static_cast<unsigned __int128>(
                inventory >= 0 ? inventory : -inventory);
            unliquidated_quantity += static_cast<unsigned __int128>(
                asset.shared_liquidation_unliquidated_quantity);
            buy_quantity += static_cast<unsigned __int128>(
                asset.shared_buy_quantity);
            sell_quantity += static_cast<unsigned __int128>(
                asset.shared_sell_quantity);
            fill_count += static_cast<unsigned __int128>(
                asset.shared_fill_count);
        }

        const long double unit = static_cast<long double>(
            config_.shared_price_unit_usd);
        shared_signed_mark_to_mid_pnl_usd_ = static_cast<double>(
            mark_to_mid_ticks * unit);
        shared_signed_liquidation_pnl_usd_ = static_cast<double>(
            liquidation_ticks * unit);
        shared_terminal_liquidation_cost_usd_ = std::max(
            0.0,
            shared_signed_mark_to_mid_pnl_usd_
                - shared_signed_liquidation_pnl_usd_);
        shared_terminal_absolute_inventory_ = checked_uint64(
            terminal_absolute_inventory, "terminal absolute inventory");
        shared_unliquidated_terminal_quantity_ = checked_uint64(
            unliquidated_quantity, "unliquidated terminal quantity");
        shared_buy_quantity_ = checked_uint64(
            buy_quantity, "shared buy quantity");
        shared_sell_quantity_ = checked_uint64(
            sell_quantity, "shared sell quantity");
        shared_fill_count_ = checked_uint64(
            fill_count, "shared fill count");
    }

    SimulationResult reduce_result(double local_wall) {
        RankResultWire local;
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            local.counts[0] += asset->book.processed_orders;
            local.counts[1] += asset->book.trade_count;
            local.counts[2] += asset->shock_executed_quantity;
            local.counts[3] += asset->shock_shared_mm_quantity;
            local.counts[4] += asset->shock_requested_quantity;
            local.counts[5] += asset->shock_local_mm_quantity;
            local.counts[6] += asset->shock_value_agent_quantity;
            local.counts[7] += asset->shock_background_quantity;
            local.counts[8] += asset->shock_other_quantity;
        }
        local.books = static_cast<unsigned long long>(local_assets_.size());
        local.wall_seconds = local_wall;
        local.initialization_seconds = initialization_seconds_;
        local.compute_seconds = compute_seconds_;
        local.communication_seconds = communication_seconds_;
        local.risk_collective_seconds = risk_collective_seconds_;
        local.observation_collective_seconds = observation_collective_seconds_;
        local.terminal_collective_seconds = terminal_collective_seconds_;
        local.boundary_wait_seconds = boundary_wait_seconds_;
        local.risk_overlap_work_seconds = risk_overlap_work_seconds_;
        local.risk_wait_after_overlap_seconds =
            risk_wait_after_overlap_seconds_;

        std::vector<RankResultWire> ranks(rank_ == 0
            ? static_cast<std::size_t>(world_size_) : 0U);
        std::array<unsigned long long, 9> global_counts{};
        const double reduction_start = MPI_Wtime();
        check_mpi(MPI_Gather(
                      &local, static_cast<int>(sizeof(RankResultWire)), MPI_BYTE,
                      rank_ == 0 ? ranks.data() : nullptr,
                      static_cast<int>(sizeof(RankResultWire)), MPI_BYTE,
                      0, communicator_),
                  "MPI_Gather(distributed rank results)");
        ++collective_calls_;
        ++terminal_collective_calls_;
        std::array<double, 10> global_times{};
        double min_compute = 0.0;
        double sum_compute = 0.0;
        const double reduction_elapsed = MPI_Wtime() - reduction_start;
        communication_seconds_ += reduction_elapsed;
        terminal_collective_seconds_ += reduction_elapsed;
        std::array<unsigned long long, 3> order_balance{};
        std::array<unsigned long long, 3> book_balance{};
        if (rank_ == 0) {
            min_compute = std::numeric_limits<double>::infinity();
            order_balance[0] = std::numeric_limits<unsigned long long>::max();
            book_balance[0] = std::numeric_limits<unsigned long long>::max();
            for (const RankResultWire& wire : ranks) {
                for (std::size_t index = 0; index < global_counts.size(); ++index) {
                    global_counts[index] += wire.counts[index];
                }
                global_times[0] = std::max(global_times[0], wire.wall_seconds);
                global_times[1] = std::max(
                    global_times[1], wire.initialization_seconds);
                global_times[2] = std::max(
                    global_times[2], wire.compute_seconds);
                global_times[3] = std::max(
                    global_times[3], wire.communication_seconds);
                global_times[4] = std::max(
                    global_times[4], wire.risk_collective_seconds);
                global_times[5] = std::max(
                    global_times[5], wire.observation_collective_seconds);
                global_times[6] = std::max(
                    global_times[6], wire.terminal_collective_seconds);
                global_times[7] = std::max(
                    global_times[7], wire.boundary_wait_seconds);
                global_times[8] = std::max(
                    global_times[8], wire.risk_overlap_work_seconds);
                global_times[9] = std::max(
                    global_times[9], wire.risk_wait_after_overlap_seconds);
                min_compute = std::min(min_compute, wire.compute_seconds);
                sum_compute += wire.compute_seconds;
                order_balance[0] = std::min(
                    order_balance[0], wire.counts[0]);
                order_balance[1] += wire.counts[0];
                order_balance[2] = std::max(
                    order_balance[2], wire.counts[0]);
                book_balance[0] = std::min(book_balance[0], wire.books);
                book_balance[1] += wire.books;
                book_balance[2] = std::max(book_balance[2], wire.books);
            }
        }

        SimulationResult result;
        result.world_size = world_size_;
        result.asset_count = config_.asset_count;
        result.lob_count = static_cast<std::uint64_t>(config_.asset_count);
        result.windows = window_count_;
        result.local_mm_refresh_boundaries = local_mm_refresh_boundaries_;
        result.collective_calls = collective_calls_;
        result.risk_collective_calls = risk_collective_calls_;
        result.risk_boundaries = risk_boundaries_;
        result.risk_lookahead_skipped_boundaries =
            risk_lookahead_skipped_boundaries_;
        result.risk_lookahead_batches = risk_lookahead_batches_;
        result.risk_lookahead_max_span = risk_lookahead_max_span_;
        result.risk_lookahead_attempted_boundaries =
            risk_lookahead_attempted_boundaries_;
        result.risk_lookahead_bound_rejections =
            risk_lookahead_bound_rejections_;
        result.risk_lookahead_bound_evaluations =
            risk_lookahead_bound_evaluations_;
        result.risk_lookahead_disabled_after_boundary =
            risk_lookahead_disabled_after_boundary_;
        result.observation_collective_calls = observation_collective_calls_;
        result.terminal_collective_calls = terminal_collective_calls_;
        result.boundary_wait_calls = boundary_wait_calls_;
        result.buffered_observations = config_.buffer_global_observations;
        result.persistent_risk_collective =
            persistent_risk_collective_active_;
        result.nonblocking_risk_collective =
            config_.use_nonblocking_risk_collective;
        result.weighted_partition = weighted_partition_active_;
        result.realized_cost_partition = realized_cost_partition_active_;
#if LOB_HAS_OPENMP
        result.openmp_enabled = true;
#else
        result.openmp_enabled = false;
#endif
        result.worker_threads = config_.worker_threads;
        result.openmp_window_only = config_.openmp_window_only;
        result.persistent_openmp_team = config_.persistent_openmp_team;
        result.parallel_asset_initialization =
            config_.parallel_asset_initialization;
        result.parallel_boundary_reductions =
            config_.parallel_boundary_reductions;
        result.parallel_metric_scans = config_.parallel_metric_scans;
        result.fuse_metric_cluster_scans =
            config_.fuse_metric_cluster_scans;
        result.predicted_partition_imbalance =
            predicted_partition_imbalance_;
        result.shock_target_assets = shock_asset_count_;
        result.shock_assets = shock_asset_count_;
        result.withdrawal_windows = withdrawal_windows_;
        result.final_shared_gross_exposure = shared_gross_exposure_;
        result.maximum_shared_gross_exposure = maximum_shared_gross_exposure_;
        result.final_shared_utilization = shared_utilization_;
        result.minimum_shared_quote_scale = minimum_shared_quote_scale_;
        result.shared_signed_mark_to_mid_pnl_usd =
            shared_signed_mark_to_mid_pnl_usd_;
        result.shared_signed_liquidation_pnl_usd =
            shared_signed_liquidation_pnl_usd_;
        result.shared_terminal_liquidation_cost_usd =
            shared_terminal_liquidation_cost_usd_;
        result.shared_unliquidated_terminal_quantity =
            shared_unliquidated_terminal_quantity_;
        result.shared_terminal_absolute_inventory =
            shared_terminal_absolute_inventory_;
        result.shared_buy_quantity = shared_buy_quantity_;
        result.shared_sell_quantity = shared_sell_quantity_;
        result.shared_fill_count = shared_fill_count_;
        result.peak_affected_fraction = peak_affected_fraction_;
        result.peak_mean_spread_bps = peak_mean_spread_bps_;
        result.final_mean_spread_bps = last_metrics_.mean_spread_bps;
        result.final_mean_top_depth = last_metrics_.mean_top_depth;
        result.final_affected_shocked_fraction =
            last_metrics_.affected_shocked_fraction;
        result.final_affected_unshocked_fraction =
            last_metrics_.affected_unshocked_fraction;
        result.peak_affected_unshocked_fraction =
            peak_affected_unshocked_fraction_;
        result.minimum_two_sided_book_fraction =
            minimum_two_sided_book_fraction_;
        if (rank_ == 0) {
            result.processed_orders = global_counts[0];
            result.trades = global_counts[1];
            result.shock_executed_quantity = global_counts[2];
            result.shock_shared_mm_quantity = global_counts[3];
            result.shock_requested_quantity = global_counts[4];
            result.shock_local_mm_quantity = global_counts[5];
            result.shock_value_agent_quantity = global_counts[6];
            result.shock_background_quantity = global_counts[7];
            result.shock_other_quantity = global_counts[8];
            if (result.shock_executed_quantity
                    != result.shock_shared_mm_quantity
                     + result.shock_local_mm_quantity
                     + result.shock_value_agent_quantity
                     + result.shock_background_quantity
                     + result.shock_other_quantity) {
                throw std::logic_error(
                    "shock fill ownership does not sum to executed quantity: "
                    + std::to_string(result.shock_executed_quantity) + " != "
                    + std::to_string(result.shock_shared_mm_quantity) + "+"
                    + std::to_string(result.shock_local_mm_quantity) + "+"
                    + std::to_string(result.shock_value_agent_quantity) + "+"
                    + std::to_string(result.shock_background_quantity) + "+"
                    + std::to_string(result.shock_other_quantity));
            }
            result.wall_seconds = global_times[0];
            result.max_initialization_seconds = global_times[1];
            result.max_compute_seconds = global_times[2];
            result.min_compute_seconds = min_compute;
            result.mean_compute_seconds = sum_compute
                / static_cast<double>(world_size_);
            result.compute_imbalance = result.mean_compute_seconds > 0.0
                ? result.max_compute_seconds / result.mean_compute_seconds : 1.0;
            result.min_orders_per_rank = order_balance[0];
            result.mean_orders_per_rank = static_cast<double>(order_balance[1])
                / static_cast<double>(world_size_);
            result.max_orders_per_rank = order_balance[2];
            result.min_books_per_rank = book_balance[0];
            result.mean_books_per_rank = static_cast<double>(book_balance[1])
                / static_cast<double>(world_size_);
            result.max_books_per_rank = book_balance[2];
            result.max_communication_seconds = global_times[3];
            result.communication_fraction = result.wall_seconds > 0.0
                ? result.max_communication_seconds / result.wall_seconds : 0.0;
            result.max_risk_collective_seconds = global_times[4];
            result.max_observation_collective_seconds = global_times[5];
            result.max_terminal_collective_seconds = global_times[6];
            result.max_boundary_wait_seconds = global_times[7];
            result.max_risk_overlap_work_seconds = global_times[8];
            result.max_risk_wait_after_overlap_seconds = global_times[9];
        }
        return result;
    }

    MPI_Comm communicator_ = MPI_COMM_WORLD;
    SimulationConfig config_;
    std::int64_t end_time_ns_ = 0;
    int rank_ = 0;
    int world_size_ = 1;
    std::vector<bool> shock_mask_;
    std::vector<int> asset_owner_ranks_;
    std::vector<std::unique_ptr<LocalAsset>> local_assets_;
    ValueAgentPolicy default_value_agent_policy_;
    std::ofstream metrics_output_;
    std::ofstream cluster_metrics_output_;
    std::ofstream return_panel_output_;
    std::vector<GlobalObservationFrame> global_observations_;
    std::vector<ClusterObservationFrame> cluster_observations_;
    std::vector<RiskObservationFrame> risk_observations_;
    std::vector<BoundaryArrivalWire> boundary_arrivals_;
    std::vector<WindowPhaseWire> window_phase_profiles_;
    WindowPhaseWire* active_window_phase_ = nullptr;
    double active_risk_total_seconds_ = 0.0;
    double active_global_metrics_total_seconds_ = 0.0;
    // Reused at every boundary; avoids 23,401 heap allocations in the
    // parallel exact-integer exposure ablation.
    std::vector<long long> fixed_exposure_contributions_;
    std::vector<std::vector<std::size_t>> local_thread_buckets_;
    std::vector<std::array<double, 8>> fused_cluster_contributions_;
    std::int64_t fused_cluster_time_ns_ = -1;
    int cluster_count_ = 0;

    double compute_seconds_ = 0.0;
    double initialization_seconds_ = 0.0;
    double communication_seconds_ = 0.0;
    double risk_collective_seconds_ = 0.0;
    double observation_collective_seconds_ = 0.0;
    double terminal_collective_seconds_ = 0.0;
    double boundary_wait_seconds_ = 0.0;
    double risk_overlap_work_seconds_ = 0.0;
    double risk_wait_after_overlap_seconds_ = 0.0;
    std::array<long long, 2> risk_local_fixed_{};
    std::array<long long, 2> risk_global_fixed_{};
    std::uint64_t risk_lookahead_remaining_ = 0;
    bool last_risk_boundary_was_skipped_ = false;
#if LOB_HAS_MPI_PERSISTENT_COLLECTIVES
    MPI_Request risk_request_ = MPI_REQUEST_NULL;
#endif
    int persistent_risk_width_ = 0;
    bool persistent_risk_collective_active_ = false;
    bool weighted_partition_active_ = false;
    bool realized_cost_partition_active_ = false;
    double predicted_partition_imbalance_ = 1.0;
    double wall_start_seconds_ = 0.0;
    double last_risk_completion_seconds_ = 0.0;
    double shared_gross_exposure_ = 0.0;
    double maximum_shared_gross_exposure_ = 0.0;
    double shared_utilization_ = 0.0;
    double shared_quote_scale_ = 1.0;
    double minimum_shared_quote_scale_ = 1.0;
    double shared_signed_mark_to_mid_pnl_usd_ = 0.0;
    double shared_signed_liquidation_pnl_usd_ = 0.0;
    double shared_terminal_liquidation_cost_usd_ = 0.0;
    std::uint64_t shared_unliquidated_terminal_quantity_ = 0;
    std::uint64_t shared_terminal_absolute_inventory_ = 0;
    std::uint64_t shared_buy_quantity_ = 0;
    std::uint64_t shared_sell_quantity_ = 0;
    std::uint64_t shared_fill_count_ = 0;
    std::vector<double> shared_capacity_weights_;
    double peak_affected_fraction_ = 0.0;
    double peak_affected_unshocked_fraction_ = 0.0;
    double peak_mean_spread_bps_ = 0.0;
    double minimum_two_sided_book_fraction_ = 1.0;
    AggregateMetrics last_metrics_;
    std::uint64_t window_count_ = 0;
    std::uint64_t local_mm_refresh_boundaries_ = 0;
    std::uint64_t collective_calls_ = 0;
    std::uint64_t risk_collective_calls_ = 0;
    std::uint64_t risk_boundaries_ = 0;
    std::uint64_t risk_lookahead_skipped_boundaries_ = 0;
    std::uint64_t risk_lookahead_batches_ = 0;
    std::uint64_t risk_lookahead_max_span_ = 0;
    std::uint64_t risk_lookahead_attempted_boundaries_ = 0;
    std::uint64_t risk_lookahead_bound_rejections_ = 0;
    std::uint64_t risk_lookahead_bound_evaluations_ = 0;
    std::uint64_t risk_lookahead_active_capacity_streak_ = 0;
    std::uint64_t risk_lookahead_disabled_after_boundary_ = 0;
    bool risk_lookahead_permanently_disabled_ = false;
    std::uint64_t observation_collective_calls_ = 0;
    std::uint64_t terminal_collective_calls_ = 0;
    std::uint64_t boundary_wait_calls_ = 0;
    std::uint64_t shock_asset_count_ = 0;
    std::uint64_t withdrawal_windows_ = 0;
};

DistributedMarketSimulator::DistributedMarketSimulator(
    MPI_Comm communicator,
    SimulationConfig config)
    : impl_(new Impl(communicator, std::move(config))) {}

DistributedMarketSimulator::~DistributedMarketSimulator() { delete impl_; }

DistributedMarketSimulator::DistributedMarketSimulator(
    DistributedMarketSimulator&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

DistributedMarketSimulator& DistributedMarketSimulator::operator=(
    DistributedMarketSimulator&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = std::exchange(other.impl_, nullptr);
    }
    return *this;
}

SimulationResult DistributedMarketSimulator::run() {
    if (impl_ == nullptr) throw std::logic_error("moved-from distributed simulator");
    return impl_->run();
}

} // namespace dlob
