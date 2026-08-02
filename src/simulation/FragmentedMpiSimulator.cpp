#include "simulation/FragmentedMpiSimulator.hpp"

#include "common/AgentUtilities.hpp"
#include "common/TradeTapeHasher.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/DistributedLimitOrderBook.hpp"
#include "simulation/AssetMomentAccumulator.hpp"
#include "simulation/DeterministicFundamentalProcess.hpp"
#include "simulation/FragmentedQuotePlacement.hpp"
#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dlob {
namespace {

constexpr std::int64_t nanoseconds_per_second = 1'000'000'000LL;
constexpr std::int64_t fundamental_news_interval_ns = nanoseconds_per_second;
constexpr std::int64_t agent_latency_ns = 5'000;
constexpr std::int32_t shared_market_maker_owner = 900'001;
constexpr StableEntityId local_market_maker_entity_base = 0x0008'0000ULL;
constexpr StableEntityId fragmented_shared_maker_entity = 0x0009'0000ULL;
constexpr StableEntityId fragmented_value_entity_base = 0x000a'0000ULL;
constexpr double risk_fixed_point_scale = 1'000'000.0;

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

template <typename Integer>
void hash_integer(std::uint64_t& hash, Integer value) {
    static_assert(std::is_integral_v<Integer>);
    using Unsigned = std::make_unsigned_t<Integer>;
    const Unsigned bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
        const std::size_t shift = (sizeof(Unsigned) - index - 1U) * 8U;
        hash ^= static_cast<std::uint8_t>(bits >> shift);
        hash *= TradeTapeHasher::prime;
    }
}

void hash_double(std::uint64_t& hash, double value) {
    hash_integer(hash, std::bit_cast<std::uint64_t>(value));
}

struct LocalBook {
    BookId book_id = 0;
    DistributedLimitOrderBook lob;
    TradeTapeHasher trade_hasher;
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
    std::int64_t value_inventory = 0;
    std::uint64_t shock_executed_quantity = 0;
    std::uint64_t shock_shared_mm_quantity = 0;
    std::uint64_t shock_requested_quantity = 0;
    bool shock_injected = false;
    double shared_requested_quote_depth = 0.0;
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
    std::uint64_t trade_hash = TradeTapeHasher::offset_basis;
};

struct AssetResultWire {
    BookId asset_id = 0;
    std::int64_t shared_inventory = 0;
    std::int64_t value_inventory = 0;
    std::uint64_t shock_requested_quantity = 0;
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

static_assert(std::is_trivially_copyable_v<BookResultWire>);
static_assert(std::is_trivially_copyable_v<AssetResultWire>);
static_assert(std::is_trivially_copyable_v<AssetMomentWire>);

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

class FragmentedMpiSimulator::Impl {
public:
    Impl(MPI_Comm communicator, FragmentedMpiConfig config)
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
        default_value_agent_policy_.enabled = true;
        default_value_agent_policy_.threshold_bps = config_.value_threshold_bps;
        default_value_agent_policy_.depth_participation =
            config_.value_depth_participation;
        default_value_agent_policy_.order_quantity = config_.value_order_quantity;
        select_shock_assets();
    }

    FragmentedMpiResult run() {
        check_mpi(MPI_Barrier(communicator_), "MPI_Barrier(start)");
        ++collective_calls_;
        const double wall_start = MPI_Wtime();

        const double initialization_start = MPI_Wtime();
        initialize_local_assets();
        initialization_seconds_ = MPI_Wtime() - initialization_start;
        open_metrics_output();
        update_shared_risk();
        record_asset_moments(0);
        observe_global_metrics(0);
        schedule_local_market_makers(0, 0);
        schedule_shared_market_makers(0, 0);
        schedule_value_agents(0, 0);
        if (config_.enable_local_market_makers) {
            ++local_mm_refresh_boundaries_;
        }

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
        while (current_ns < end_time_ns_) {
            const std::int64_t end_ns = std::min({
                end_time_ns_, next_global_boundary_ns,
                next_local_refresh_ns, next_fundamental_news_ns,
                next_value_decision_ns});
            const double compute_start = MPI_Wtime();
            for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
                process_window(*asset, current_ns, end_ns);
            }
            compute_seconds_ += MPI_Wtime() - compute_start;

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
                update_shared_risk();
                record_asset_moments(end_ns);
                if (terminal_boundary
                    || end_ns % config_.global_metrics_interval_ns == 0) {
                    observe_global_metrics(end_ns);
                }
            }
            if (!terminal_boundary) {
                // At a coincident one-second wake time this is exactly the
                // historical order: local maker, fundamental news, shared
                // maker, then value agent.  Separate indices keep every
                // entity's stochastic stream stable when only the MPI window
                // is changed.
                if (local_refresh) {
                    schedule_local_market_makers(end_ns, local_refresh_index);
                    ++local_mm_refresh_boundaries_;
                    next_local_refresh_ns = checked_add_time(
                        next_local_refresh_ns, config_.local_mm_interval_ns);
                    ++local_refresh_index;
                }
                if (fundamental_news) {
                    advance_fundamental_news(fundamental_news_index);
                    next_fundamental_news_ns = checked_add_time(
                        next_fundamental_news_ns,
                        fundamental_news_interval_ns);
                    ++fundamental_news_index;
                }
                if (global_boundary) {
                    schedule_shared_market_makers(
                        end_ns, global_boundary_index);
                }
                if (fundamental_news) {
                    schedule_news_impulse_value_agents(
                        end_ns, fundamental_news_index - 1U);
                }
                if (value_decision) {
                    schedule_value_agents(end_ns, value_decision_index);
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
            current_ns = end_ns;
        }

        const std::vector<BookResultWire> books = gather_book_results();
        const std::vector<AssetResultWire> assets = gather_asset_results();
        const std::vector<AssetMomentWire> moments = gather_asset_moments();
        write_asset_summary(moments);
        write_shock_targets(assets);
        const std::uint64_t state_hash = compute_state_hash(books, assets);

        check_mpi(MPI_Barrier(communicator_), "MPI_Barrier(finish)");
        ++collective_calls_;
        const double local_wall = MPI_Wtime() - wall_start;
        return reduce_result(local_wall, state_hash);
    }

private:
    void validate_config() const {
        if (world_size_ <= 0 || config_.asset_count <= 0
            || config_.decision_window_ns <= 0
            || config_.decision_window_ns > end_time_ns_
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
            || !std::isfinite(config_.shock_asset_fraction)
            || config_.shock_asset_fraction <= 0.0
            || config_.shock_asset_fraction > 1.0
            || config_.shock_target_count < 0
            || config_.shock_target_count > config_.asset_count
            || config_.shock_quantity_per_asset <= 0
            || !std::isfinite(config_.shock_top_depth_multiple)
            || config_.shock_top_depth_multiple < 0.0) {
            throw std::invalid_argument("invalid fragmented MPI configuration");
        }
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
        for (const FragmentedValueAgentPolicy& policy : config_.value_agent_policies) {
            if (!std::isfinite(policy.threshold_bps) || policy.threshold_bps < 0.0
                || !std::isfinite(policy.depth_participation)
                || policy.depth_participation < 0.0
                || policy.depth_participation > 1.0
                || (policy.depth_participation == 0.0
                    && policy.order_quantity <= 0)
                || policy.maximum_news_rechecks < 0
                || policy.maximum_news_rechecks > 16
                || (policy.trigger_mode
                        == FragmentedValueTriggerMode::PeriodicGap
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
                || (config_.shared_quote_relative_to_asset
                    && book.market_maker_quote_quantity <= 0)) {
                throw std::invalid_argument("invalid fragmented asset template");
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
            // within-cluster hash ranking.  This keeps the principal mask
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

    void write_shock_targets(const std::vector<AssetResultWire>& assets) const {
        if (rank_ != 0 || config_.shock_targets_csv.empty()) return;
        std::vector<std::uint64_t> requested(
            static_cast<std::size_t>(config_.asset_count), 0U);
        for (const AssetResultWire& asset : assets) {
            if (asset.asset_id >= static_cast<BookId>(config_.asset_count)) {
                throw std::logic_error("invalid asset in shock target gather");
            }
            requested[static_cast<std::size_t>(asset.asset_id)] =
                asset.shock_requested_quantity;
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
                  "requested_sell_quantity,mask_seed\n";
        for (std::size_t index = 0; index < config_.asset_configs.size(); ++index) {
            output << index << ','
                   << config_.asset_configs[index].symbol << ','
                   << (config_.shock_cluster_ids.empty()
                       ? -1 : config_.shock_cluster_ids[index]) << ','
                   << (shock_mask_[index] ? 1 : 0) << ','
                   << (config_.enable_shock ? 1 : 0) << ','
                   << (shock_mask_[index] && config_.enable_shock
                       ? requested[index] : 0)
                   << ',' << config_.shock_target_seed
                   << '\n';
        }
    }

    int owner_rank_for_asset(BookId asset_id) const {
        return static_cast<int>(asset_id % static_cast<BookId>(world_size_));
    }

    void initialize_local_assets() {
        local_assets_.clear();
        for (int asset_index = 0; asset_index < config_.asset_count; ++asset_index) {
            const BookId asset_id = static_cast<BookId>(asset_index);
            if (owner_rank_for_asset(asset_id) != rank_) continue;
            const MultiAssetBookConfig& source =
                config_.asset_configs[static_cast<std::size_t>(asset_index)];

            BackgroundHawkesConfig background;
            if (config_.background_configs.empty()) {
                SequentialMultiAssetConfig bridge;
                bridge.seed = config_.seed;
                bridge.tick_size = config_.tick_size;
                background = make_multi_asset_background_config(
                    bridge, source, asset_id);
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
                background.stochastic_baseline_normalization_bins =
                    static_cast<std::uint64_t>(
                        (end_time_ns_ + fundamental_news_interval_ns - 1)
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
                && config_.shock_top_depth_multiple <= 0.0) {
                const int shock_quantity = config_.shock_quantity_per_asset;
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
            local_assets_.push_back(std::move(asset));
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
            book.trade_hasher.add(trade);
            if (trade.buyer_owner_id == shared_market_maker_owner) {
                asset.shared_inventory += trade.quantity;
            }
            if (trade.seller_owner_id == shared_market_maker_owner) {
                asset.shared_inventory -= trade.quantity;
            }
            const bool shock_seller =
                trade.seller_owner_id == liquidity_shock_owner_id;
            const bool shock_buyer =
                trade.buyer_owner_id == liquidity_shock_owner_id;
            if (shock_seller || shock_buyer) {
                asset.shock_executed_quantity +=
                    static_cast<std::uint64_t>(trade.quantity);
                if ((shock_seller
                     && trade.buyer_owner_id == shared_market_maker_owner)
                    || (shock_buyer
                        && trade.seller_owner_id == shared_market_maker_owner)) {
                    asset.shock_shared_mm_quantity +=
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
                && config_.shock_top_depth_multiple > 0.0
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
                const int contemporaneous_bid_depth = std::max(
                    1, asset.book.lob.best_bid_depth());
                const int quantity = bounded_positive_quantity(
                    config_.shock_top_depth_multiple
                        * static_cast<double>(contemporaneous_bid_depth),
                    "contemporaneous-depth sell-side stress quantity");
                asset.shock_requested_quantity =
                    static_cast<std::uint64_t>(quantity);
                asset.shock_injected = true;
                apply_order(asset, make_market_order(
                    asset.asset_id, config_.shock_time_ns,
                    liquidity_shock_owner_id, AgentKind::Institutional,
                    Side::Sell, quantity,
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

    void append_quotes(LocalAsset& asset,
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
                       double bid_scale,
                       double ask_scale) {
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
                && !quote_only_when_repairing) {
                make_order(OrderAction::CancelOwner, Side::Buy, 0, 0);
            }
            return;
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
        if (!detail::fragmented_quote_required(
                quote_only_when_repairing,
                one_sided,
                shallow_top,
                wide_spread,
                improve_wide_spread)) {
            return;
        }
        if (!conserve_empirical_liquidity) {
            make_order(OrderAction::CancelOwner, Side::Buy, 0, 0);
        }
        const detail::FragmentedQuotePrices prices =
            detail::fragmented_quote_prices(
                state, asset.fundamental_value_ticks,
                config_.tick_size, asset.config.target_spread_ticks,
                improve_wide_spread);
        const std::int64_t bid = prices.bid;
        const std::int64_t ask = prices.ask;

        for (int level = 0; level < levels; ++level) {
            const std::int64_t level_bid = bid - static_cast<std::int64_t>(level) * tick;
            const std::int64_t level_ask = ask + static_cast<std::int64_t>(level) * tick;
            // Inventory skew can enlarge one side of a shared quote.  Route
            // it through the same bounded conversion used for empirical quote
            // multipliers rather than narrowing an out-of-range double to int.
            const int bid_quantity = bid_scale > 0.0
                ? bounded_positive_quantity(
                    static_cast<double>(base_quantity) * bid_scale,
                    "bid quote quantity")
                : 0;
            const int ask_quantity = ask_scale > 0.0
                ? bounded_positive_quantity(
                    static_cast<double>(base_quantity) * ask_scale,
                    "ask quote quantity")
                : 0;
            if (level_bid > 0 && bid_quantity > 0
                && level_bid <= std::numeric_limits<std::int32_t>::max()) {
                make_order(conserve_empirical_liquidity
                               ? OrderAction::ConservedLimit
                               : OrderAction::Limit,
                           Side::Buy, bid_quantity,
                           static_cast<int>(level_bid));
            }
            if (level_ask > level_bid && ask_quantity > 0
                && level_ask <= std::numeric_limits<std::int32_t>::max()) {
                make_order(conserve_empirical_liquidity
                               ? OrderAction::ConservedLimit
                               : OrderAction::Limit,
                           Side::Sell, ask_quantity,
                           static_cast<int>(level_ask));
            }
        }
    }

    void schedule_local_market_makers(std::int64_t decision_time_ns,
                                      std::uint64_t refresh_index) {
        if (!config_.enable_local_market_makers) return;
        const double compute_start = MPI_Wtime();
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            LocalAsset& asset = *pointer;
            LocalBook& book = asset.book;
            const int local_quantity = bounded_positive_quantity(
                config_.local_mm_quantity_multiplier * static_cast<double>(
                    asset.config.market_maker_quote_quantity),
                "local market-maker quote quantity");
            append_quotes(
                asset, book, decision_time_ns, refresh_index,
                local_market_maker_owner_id(asset.asset_id),
                local_market_maker_entity_base, local_quantity, 1,
                config_.local_mm_improvement_probability,
                config_.local_mm_spread_elasticity,
                config_.local_mm_max_improvement_probability,
                true,
                false,
                1.0, 1.0);
        }
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    void advance_fundamental_news(std::uint64_t news_index) {
        const double compute_start = MPI_Wtime();
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            LocalAsset& asset = *pointer;
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
        }
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    void schedule_shared_market_makers(std::int64_t decision_time_ns,
                                       std::uint64_t boundary_index) {
        if (!config_.enable_shared_market_maker) return;
        const double compute_start = MPI_Wtime();
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            LocalAsset& asset = *pointer;
            LocalBook& book = asset.book;
            const double inventory_ratio = std::min(
                0.75,
                std::abs(static_cast<double>(asset.shared_inventory))
                    / std::max(1.0, config_.shared_local_inventory_scale));
            const double bid_inventory_scale = asset.shared_inventory > 0
                ? 1.0 - inventory_ratio : 1.0 + inventory_ratio;
            const double ask_inventory_scale = asset.shared_inventory < 0
                ? 1.0 - inventory_ratio : 1.0 + inventory_ratio;
            const double shared_base_quantity =
                config_.shared_quote_relative_to_asset
                ? config_.shared_quote_multiplier * static_cast<double>(
                    asset.config.market_maker_quote_quantity)
                : static_cast<double>(config_.shared_quote_quantity);
            const int shared_book_quantity = bounded_positive_quantity(
                shared_base_quantity, "shared market-maker quote quantity");
            const double bid_scale = shared_quote_scale_ * bid_inventory_scale;
            const double ask_scale = shared_quote_scale_ * ask_inventory_scale;
            asset.shared_requested_quote_depth =
                static_cast<double>(shared_book_quantity)
                * static_cast<double>(config_.shared_quote_levels)
                * (bid_scale + ask_scale);
            append_quotes(
                asset, book, decision_time_ns, boundary_index,
                shared_market_maker_owner,
                fragmented_shared_maker_entity,
                shared_book_quantity,
                config_.shared_quote_levels,
                0.0,
                0.0,
                1.0,
                false,
                false,
                bid_scale, ask_scale);
        }
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    void schedule_value_agents(std::int64_t decision_time_ns,
                               std::uint64_t decision_index) {
        if (!config_.enable_value_agents) return;
        const double compute_start = MPI_Wtime();
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            const FragmentedValueAgentPolicy& policy =
                value_agent_policy(pointer->asset_id);
            if (policy.trigger_mode
                == FragmentedValueTriggerMode::PeriodicGap) {
                schedule_value_agent(
                    *pointer, decision_time_ns, decision_index);
            } else if (policy.trigger_mode
                           == FragmentedValueTriggerMode::NewsImpulse
                       && pointer->remaining_value_rechecks > 0
                       && decision_time_ns >= pointer->value_recheck_due_ns) {
                schedule_value_agent(
                    *pointer, decision_time_ns, decision_index);
                --pointer->remaining_value_rechecks;
                if (pointer->remaining_value_rechecks > 0) {
                    pointer->value_recheck_due_ns = checked_add_time(
                        pointer->value_recheck_due_ns,
                        config_.value_agent_interval_ns);
                } else {
                    pointer->value_recheck_due_ns = 0;
                }
            }
        }
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    void schedule_news_impulse_value_agents(
        std::int64_t decision_time_ns, std::uint64_t news_index) {
        const double compute_start = MPI_Wtime();
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            LocalAsset& asset = *pointer;
            const FragmentedValueAgentPolicy& policy =
                value_agent_policy(asset.asset_id);
            if (config_.enable_value_agents && asset.fresh_fundamental_news) {
                if (policy.trigger_mode
                    == FragmentedValueTriggerMode::NewsImpulse) {
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
        }
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    [[nodiscard]] const FragmentedValueAgentPolicy& value_agent_policy(
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
        const FragmentedValueAgentPolicy& policy =
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
                fragmented_value_entity_base
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
            fragmented_value_entity_base
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

    void update_shared_risk() {
        if (!config_.enable_shared_market_maker) {
            shared_gross_exposure_ = 0.0;
            shared_utilization_ = 0.0;
            shared_quote_scale_ = 1.0;
            return;
        }
        long long local_fixed = 0;
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            const double exposure = std::abs(
                asset->config.beta * static_cast<double>(asset->shared_inventory));
            const long long fixed = std::llround(
                exposure * risk_fixed_point_scale);
            if (fixed > std::numeric_limits<long long>::max() - local_fixed) {
                throw std::overflow_error("shared market-maker exposure overflow");
            }
            local_fixed += fixed;
        }
        long long global_fixed = 0;
        const double start = MPI_Wtime();
        check_mpi(MPI_Allreduce(&local_fixed, &global_fixed, 1,
                                MPI_LONG_LONG, MPI_SUM, communicator_),
                  "MPI_Allreduce(shared risk)");
        communication_seconds_ += MPI_Wtime() - start;
        ++collective_calls_;
        shared_gross_exposure_ = static_cast<double>(global_fixed)
            / risk_fixed_point_scale;
        const double global_limit = config_.shared_global_risk_limit_per_asset
            * static_cast<double>(config_.asset_count);
        shared_utilization_ = shared_gross_exposure_ / global_limit;
        const double threshold = config_.shared_capacity_threshold;
        shared_quote_scale_ = !config_.enable_global_shared_capacity
            || shared_utilization_ <= threshold
            ? 1.0
            : std::max(0.0, (1.0 - shared_utilization_) / (1.0 - threshold));
        minimum_shared_quote_scale_ = std::min(
            minimum_shared_quote_scale_, shared_quote_scale_);
        if (shared_quote_scale_ < 0.5) ++withdrawal_windows_;
    }

    void record_asset_moments(std::int64_t time_ns) {
        if (config_.asset_summary_csv.empty() || time_ns <= 0
            || time_ns % config_.asset_summary_interval_ns != 0) return;
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            LocalAsset& asset = *pointer;
            const MarketState state = asset.book.lob.state(
                time_ns, asset.fundamental_value_ticks);
            asset.calibration_moments.observe(state, config_.tick_size);
        }
    }

    AggregateMetricSums local_metrics(std::int64_t time_ns) const {
        AggregateMetricSums metrics;
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            const LocalAsset& asset = *pointer;
            const MarketState state = asset.book.lob.state(
                time_ns, asset.fundamental_value_ticks);
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
        }
        return metrics;
    }

    void open_metrics_output() {
        if (rank_ != 0 || config_.metrics_csv.empty()) return;
        const std::filesystem::path path(config_.metrics_csv);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        metrics_output_.open(path);
        if (!metrics_output_) {
            throw std::runtime_error("cannot open fragmented metrics CSV: "
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
               "value_agent_requested_quantity\n";
    }

    void observe_global_metrics(std::int64_t time_ns) {
        const AggregateMetricSums local = local_metrics(time_ns);
        const std::array<double, 18> send{{
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
            local.value_requested_quantity}};
        std::array<double, 18> global{};
        const double start = MPI_Wtime();
        check_mpi(MPI_Allreduce(send.data(), global.data(),
                                static_cast<int>(global.size()),
                                MPI_DOUBLE, MPI_SUM, communicator_),
                  "MPI_Allreduce(fragmented metrics)");
        communication_seconds_ += MPI_Wtime() - start;
        ++collective_calls_;

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
                << static_cast<double>(time_ns) / 1e9 << ','
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
                << shared_gross_exposure_ << ','
                << shared_utilization_ << ','
                << shared_quote_scale_ << ','
                << global[14] << ','
                << (global[4] > 0.0 ? global[15] / global[4] : 0.0) << ','
                << global[16] << ','
                << global[17] << '\n';
        }
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
                  "MPI_Gather(fragmented counts)");
        ++collective_calls_;
        int total_bytes = 0;
        std::vector<int> displacements;
        if (rank_ == 0) {
            displacements.resize(static_cast<std::size_t>(world_size_));
            for (int index = 0; index < world_size_; ++index) {
                const int count = counts[static_cast<std::size_t>(index)];
                if (count < 0
                    || total_bytes > std::numeric_limits<int>::max() - count) {
                    throw std::overflow_error("fragmented gather size overflow");
                }
                displacements[static_cast<std::size_t>(index)] = total_bytes;
                total_bytes += count;
            }
            if (total_bytes % static_cast<int>(sizeof(Value)) != 0) {
                throw std::logic_error("fragmented gather contains partial values");
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
                  "MPI_Gatherv(fragmented values)");
        ++collective_calls_;
        communication_seconds_ += MPI_Wtime() - start;
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
            wire.trade_count = book.trade_hasher.trade_count();
            wire.trade_hash = book.trade_hasher.digest();
            local.push_back(wire);
        }
        return gather_values(local, "fragmented book result");
    }

    std::vector<AssetResultWire> gather_asset_results() {
        std::vector<AssetResultWire> local;
        local.reserve(local_assets_.size());
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            local.push_back(AssetResultWire{
                asset->asset_id, asset->shared_inventory, asset->value_inventory,
                asset->shock_requested_quantity,
                asset->fundamental_value_ticks,
                asset->fundamental_log_variance,
                asset->value_recheck_due_ns,
                asset->remaining_value_rechecks});
        }
        return gather_values(local, "fragmented asset result");
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
        return gather_values(local, "fragmented asset moment");
    }

    void write_asset_summary(std::vector<AssetMomentWire> moments) const {
        if (rank_ != 0 || config_.asset_summary_csv.empty()) return;
        if (moments.size() != static_cast<std::size_t>(config_.asset_count)) {
            throw std::logic_error("incomplete fragmented per-asset moment summary");
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
                "cannot open fragmented per-asset summary CSV: " + path.string());
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

    std::uint64_t compute_state_hash(std::vector<BookResultWire> books,
                                     std::vector<AssetResultWire> assets) const {
        if (rank_ != 0) return 0;
        std::sort(books.begin(), books.end(),
                  [](const BookResultWire& left, const BookResultWire& right) {
                      return left.state.book_id < right.state.book_id;
                  });
        std::sort(assets.begin(), assets.end(),
                  [](const AssetResultWire& left, const AssetResultWire& right) {
                      return left.asset_id < right.asset_id;
                  });
        const std::size_t expected_books = static_cast<std::size_t>(config_.asset_count);
        if (books.size() != expected_books
            || assets.size() != static_cast<std::size_t>(config_.asset_count)) {
            throw std::logic_error("incomplete fragmented final state");
        }
        std::uint64_t hash = TradeTapeHasher::offset_basis;
        for (const BookResultWire& wire : books) {
            const MarketState& state = wire.state;
            hash_integer(hash, state.book_id);
            hash_integer(hash, state.exchange_time_ns);
            hash_integer(hash, state.best_bid_ticks);
            hash_integer(hash, state.best_ask_ticks);
            hash_integer(hash, state.best_bid_depth);
            hash_integer(hash, state.best_ask_depth);
            hash_integer(hash, state.background_best_bid_depth);
            hash_integer(hash, state.background_best_ask_depth);
            hash_integer(hash, state.total_background_bid_depth);
            hash_integer(hash, state.total_background_ask_depth);
            hash_integer(hash, state.last_trade_price_ticks);
            hash_double(hash, state.mid_price_ticks);
            hash_double(hash, state.fundamental_value_ticks);
            hash_integer(hash, state.cumulative_aggressive_buy);
            hash_integer(hash, state.cumulative_aggressive_sell);
            hash_integer(hash, wire.processed_orders);
            hash_integer(hash, wire.trade_count);
            hash_integer(hash, wire.trade_hash);
        }
        for (const AssetResultWire& wire : assets) {
            hash_integer(hash, wire.asset_id);
            hash_integer(hash, wire.shared_inventory);
            hash_integer(hash, wire.value_inventory);
            hash_integer(hash, wire.shock_requested_quantity);
            hash_double(hash, wire.fundamental_value_ticks);
            const MultiAssetBookConfig& book_config =
                config_.asset_configs[static_cast<std::size_t>(wire.asset_id)];
            // Preserve historical hashes for configurations where stochastic
            // volatility is absent.  When active, both parameters and the
            // terminal hidden state are part of the canonical state digest.
            if (book_config.fundamental_log_variance_std > 0.0) {
                hash_double(
                    hash, book_config.fundamental_log_variance_persistence);
                hash_double(hash, book_config.fundamental_log_variance_std);
                if (book_config.fundamental_order_flow_coupling > 0.0) {
                    hash_double(
                        hash, book_config.fundamental_order_flow_coupling);
                }
                hash_double(hash, wire.fundamental_log_variance);
            }
            const FragmentedValueAgentPolicy& policy =
                value_agent_policy(wire.asset_id);
            if (config_.enable_value_agents
                && policy.trigger_mode
                    == FragmentedValueTriggerMode::NewsImpulse
                && policy.maximum_news_rechecks > 0) {
                hash_integer(hash, wire.remaining_value_rechecks);
                hash_integer(hash, wire.value_recheck_due_ns);
            }
        }
        return hash;
    }

    FragmentedMpiResult reduce_result(double local_wall,
                                      std::uint64_t state_hash) {
        std::array<unsigned long long, 5> local_counts{};
        for (const std::unique_ptr<LocalAsset>& asset : local_assets_) {
            local_counts[0] += asset->book.processed_orders;
            local_counts[1] += asset->book.trade_hasher.trade_count();
            local_counts[2] += asset->shock_executed_quantity;
            local_counts[3] += asset->shock_shared_mm_quantity;
            local_counts[4] += asset->shock_requested_quantity;
        }
        std::array<unsigned long long, 5> global_counts{};
        const double reduction_start = MPI_Wtime();
        check_mpi(MPI_Reduce(local_counts.data(), global_counts.data(),
                             static_cast<int>(local_counts.size()),
                             MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator_),
                  "MPI_Reduce(fragmented counts)");
        ++collective_calls_;
        std::array<double, 4> local_times{{
            local_wall, initialization_seconds_, compute_seconds_,
            communication_seconds_}};
        std::array<double, 4> global_times{};
        check_mpi(MPI_Reduce(local_times.data(), global_times.data(),
                             static_cast<int>(local_times.size()),
                             MPI_DOUBLE, MPI_MAX, 0, communicator_),
                  "MPI_Reduce(fragmented timings)");
        ++collective_calls_;
        double min_compute = 0.0;
        double sum_compute = 0.0;
        check_mpi(MPI_Reduce(&compute_seconds_, &min_compute, 1,
                             MPI_DOUBLE, MPI_MIN, 0, communicator_),
                  "MPI_Reduce(min fragmented compute)");
        check_mpi(MPI_Reduce(&compute_seconds_, &sum_compute, 1,
                             MPI_DOUBLE, MPI_SUM, 0, communicator_),
                  "MPI_Reduce(sum fragmented compute)");
        collective_calls_ += 2;
        const unsigned long long local_orders = local_counts[0];
        const unsigned long long local_books =
            static_cast<unsigned long long>(local_assets_.size());
        std::array<unsigned long long, 3> order_balance{};
        std::array<unsigned long long, 3> book_balance{};
        check_mpi(MPI_Reduce(&local_orders, &order_balance[0], 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, communicator_),
                  "MPI_Reduce(min orders per rank)");
        check_mpi(MPI_Reduce(&local_orders, &order_balance[1], 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator_),
                  "MPI_Reduce(sum orders per rank)");
        check_mpi(MPI_Reduce(&local_orders, &order_balance[2], 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, communicator_),
                  "MPI_Reduce(max orders per rank)");
        check_mpi(MPI_Reduce(&local_books, &book_balance[0], 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, communicator_),
                  "MPI_Reduce(min books per rank)");
        check_mpi(MPI_Reduce(&local_books, &book_balance[1], 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator_),
                  "MPI_Reduce(sum books per rank)");
        check_mpi(MPI_Reduce(&local_books, &book_balance[2], 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, communicator_),
                  "MPI_Reduce(max books per rank)");
        collective_calls_ += 6;
        communication_seconds_ += MPI_Wtime() - reduction_start;

        FragmentedMpiResult result;
        result.world_size = world_size_;
        result.asset_count = config_.asset_count;
        result.lob_count = static_cast<std::uint64_t>(config_.asset_count);
        result.windows = window_count_;
        result.local_mm_refresh_boundaries = local_mm_refresh_boundaries_;
        result.collective_calls = collective_calls_;
        result.shock_target_assets = shock_asset_count_;
        result.shock_assets = shock_asset_count_;
        result.withdrawal_windows = withdrawal_windows_;
        result.final_shared_gross_exposure = shared_gross_exposure_;
        result.final_shared_utilization = shared_utilization_;
        result.minimum_shared_quote_scale = minimum_shared_quote_scale_;
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
        result.state_hash = state_hash;
        if (rank_ == 0) {
            result.processed_orders = global_counts[0];
            result.trades = global_counts[1];
            result.shock_executed_quantity = global_counts[2];
            result.shock_shared_mm_quantity = global_counts[3];
            result.shock_requested_quantity = global_counts[4];
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
        }
        return result;
    }

    MPI_Comm communicator_ = MPI_COMM_WORLD;
    FragmentedMpiConfig config_;
    std::int64_t end_time_ns_ = 0;
    int rank_ = 0;
    int world_size_ = 1;
    std::vector<bool> shock_mask_;
    std::vector<std::unique_ptr<LocalAsset>> local_assets_;
    FragmentedValueAgentPolicy default_value_agent_policy_;
    std::ofstream metrics_output_;

    double compute_seconds_ = 0.0;
    double initialization_seconds_ = 0.0;
    double communication_seconds_ = 0.0;
    double shared_gross_exposure_ = 0.0;
    double shared_utilization_ = 0.0;
    double shared_quote_scale_ = 1.0;
    double minimum_shared_quote_scale_ = 1.0;
    double peak_affected_fraction_ = 0.0;
    double peak_affected_unshocked_fraction_ = 0.0;
    double peak_mean_spread_bps_ = 0.0;
    double minimum_two_sided_book_fraction_ = 1.0;
    AggregateMetrics last_metrics_;
    std::uint64_t window_count_ = 0;
    std::uint64_t local_mm_refresh_boundaries_ = 0;
    std::uint64_t collective_calls_ = 0;
    std::uint64_t shock_asset_count_ = 0;
    std::uint64_t withdrawal_windows_ = 0;
};

FragmentedMpiSimulator::FragmentedMpiSimulator(
    MPI_Comm communicator,
    FragmentedMpiConfig config)
    : impl_(new Impl(communicator, std::move(config))) {}

FragmentedMpiSimulator::~FragmentedMpiSimulator() { delete impl_; }

FragmentedMpiSimulator::FragmentedMpiSimulator(
    FragmentedMpiSimulator&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

FragmentedMpiSimulator& FragmentedMpiSimulator::operator=(
    FragmentedMpiSimulator&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = std::exchange(other.impl_, nullptr);
    }
    return *this;
}

FragmentedMpiResult FragmentedMpiSimulator::run() {
    if (impl_ == nullptr) throw std::logic_error("moved-from fragmented simulator");
    return impl_->run();
}

} // namespace dlob
