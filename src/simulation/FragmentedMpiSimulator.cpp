#include "simulation/FragmentedMpiSimulator.hpp"

#include "common/AgentUtilities.hpp"
#include "common/TradeTapeHasher.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/DistributedLimitOrderBook.hpp"
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
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dlob {
namespace {

constexpr std::int64_t nanoseconds_per_second = 1'000'000'000LL;
constexpr std::int64_t agent_latency_ns = 5'000;
constexpr std::int32_t shared_market_maker_owner = 900'001;
constexpr std::int32_t local_market_maker_owner_base = 100'001;
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
        case OrderAction::Limit: return 1;
        case OrderAction::Market: return 2;
        case OrderAction::CancelAtDistance: return 3;
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

class StreamingHawkes {
public:
    explicit StreamingHawkes(BackgroundHawkesConfig config)
        : config_(std::move(config)), rng_(config_.seed) {}

    const HawkesEvent& peek() {
        if (!pending_.has_value()) pending_ = generate_next();
        return *pending_;
    }

    HawkesEvent pop() {
        const HawkesEvent event = peek();
        pending_.reset();
        ++accepted_events_;
        return event;
    }

    [[nodiscard]] std::uint64_t accepted_events() const noexcept {
        return accepted_events_;
    }

private:
    HawkesEvent generate_next() {
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
            if (upper_sum <= 1e-12) {
                return HawkesEvent{std::numeric_limits<std::int64_t>::max(),
                                   HawkesEventType::LimitBuy};
            }

            const double wait_seconds = -std::log(rng_.uniform01()) / upper_sum;
            time_seconds_ += wait_seconds;
            const double decay = std::exp(
                -std::max(1e-6, config_.beta) * wait_seconds);
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
            const double time_ns = time_seconds_ * 1e9;
            const auto timestamp = time_ns >= static_cast<double>(
                    std::numeric_limits<std::int64_t>::max())
                ? std::numeric_limits<std::int64_t>::max()
                : static_cast<std::int64_t>(std::llround(time_ns));
            return HawkesEvent{
                timestamp, static_cast<HawkesEventType>(event_index)};
        }
    }

    BackgroundHawkesConfig config_;
    FastRng rng_;
    double time_seconds_ = 0.0;
    std::array<double, 6> excitation_{};
    std::optional<HawkesEvent> pending_;
    std::uint64_t accepted_events_ = 0;
};

struct LocalBook {
    BookId book_id = 0;
    DistributedLimitOrderBook lob;
    TradeTapeHasher trade_hasher;
    std::uint64_t processed_orders = 0;

    LocalBook(BookId id, int tick_size) : book_id(id), lob(tick_size, id) {}
};

// A fixed-memory version of the moment recorder used by the sequential
// calibration reference.  Fragmented MPI needs only final per-asset moments,
// not a 23,400-row state trace for every book in a large universe.
struct AssetMomentAccumulator {
    std::uint64_t snapshots = 0;
    std::uint64_t invalid_snapshots = 0;
    double spread_sum = 0.0;
    double bid_depth_sum = 0.0;
    double ask_depth_sum = 0.0;
    double previous_mid = 0.0;
    std::uint64_t mid_moves = 0;
    std::uint64_t return_count = 0;
    double return_sum = 0.0;
    double return_sum2 = 0.0;
    double return_sum4 = 0.0;
    double abs_return_sum = 0.0;
    double abs_return_sum2 = 0.0;
    double abs_pair_product_sum = 0.0;
    std::uint64_t abs_pair_count = 0;
    double previous_abs_return = 0.0;
    bool have_previous_abs_return = false;

    void observe(const MarketState& state, int tick_size) {
        if (state.best_bid_ticks <= 0
            || state.best_ask_ticks <= state.best_bid_ticks
            || tick_size <= 0) {
            ++invalid_snapshots;
            return;
        }
        const double mid = state.mid_price_ticks > 0.0
            ? state.mid_price_ticks
            : 0.5 * static_cast<double>(
                state.best_bid_ticks + state.best_ask_ticks);
        if (!(mid > 0.0) || !std::isfinite(mid)) {
            ++invalid_snapshots;
            return;
        }
        ++snapshots;
        spread_sum += static_cast<double>(
            state.best_ask_ticks - state.best_bid_ticks)
            / static_cast<double>(tick_size);
        bid_depth_sum += static_cast<double>(std::max(0, state.best_bid_depth));
        ask_depth_sum += static_cast<double>(std::max(0, state.best_ask_depth));
        if (previous_mid > 0.0) {
            if (mid != previous_mid) ++mid_moves;
            const double value = std::log(mid / previous_mid);
            if (std::isfinite(value)) {
                ++return_count;
                return_sum += value;
                const double value2 = value * value;
                return_sum2 += value2;
                return_sum4 += value2 * value2;
                const double absolute = std::abs(value);
                abs_return_sum += absolute;
                abs_return_sum2 += absolute * absolute;
                if (have_previous_abs_return) {
                    abs_pair_product_sum += absolute * previous_abs_return;
                    ++abs_pair_count;
                }
                previous_abs_return = absolute;
                have_previous_abs_return = true;
            }
        }
        previous_mid = mid;
    }

    [[nodiscard]] std::array<double, 7> finalize() const {
        std::array<double, 7> values{};
        if (snapshots > 0) {
            const double count = static_cast<double>(snapshots);
            values[0] = spread_sum / count;
            values[1] = bid_depth_sum / count;
            values[2] = ask_depth_sum / count;
            if (snapshots > 1) {
                values[3] = static_cast<double>(mid_moves)
                    / static_cast<double>(snapshots - 1U);
            }
        }
        if (return_count > 0) {
            const double count = static_cast<double>(return_count);
            const double mean = return_sum / count;
            const double variance = std::max(
                0.0, return_sum2 / count - mean * mean);
            values[4] = variance;
            if (variance > std::numeric_limits<double>::epsilon()) {
                values[5] = (return_sum4 / count) / (variance * variance);
            }
        }
        if (abs_pair_count > 0 && return_count > 0) {
            const double count = static_cast<double>(return_count);
            const double mean = abs_return_sum / count;
            const double variance = std::max(
                0.0, abs_return_sum2 / count - mean * mean);
            if (variance > std::numeric_limits<double>::epsilon()) {
                const double cross = abs_pair_product_sum
                    / static_cast<double>(abs_pair_count);
                values[6] = (cross - mean * mean) / variance;
            }
        }
        return values;
    }
};

struct LocalAsset {
    BookId asset_id = 0;
    MultiAssetBookConfig config;
    BackgroundHawkesAgent background;
    StreamingHawkes hawkes;
    LocalBook book;
    std::vector<OrderMessage> pending_orders;
    std::int64_t shared_inventory = 0;
    std::int64_t value_inventory = 0;
    std::uint64_t shock_executed_quantity = 0;
    std::uint64_t shock_shared_mm_quantity = 0;
    std::uint64_t shock_requested_quantity = 0;
    double baseline_mean_spread_bps = 0.0;
    double baseline_top_depth = 0.0;
    AssetMomentAccumulator calibration_moments;

    LocalAsset(BookId id,
               MultiAssetBookConfig asset_config,
               const BackgroundHawkesConfig& background_config,
               int tick_size)
        : asset_id(id),
          config(std::move(asset_config)),
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
};

struct AssetMomentWire {
    BookId asset_id = 0;
    std::uint64_t sample_count = 0;
    std::uint64_t invalid_sample_count = 0;
    std::uint64_t background_event_count = 0;
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
        validate_config();
        default_value_agent_policy_.enabled = true;
        default_value_agent_policy_.threshold_bps = config_.value_threshold_bps;
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
        write_shock_targets();
        open_metrics_output();
        update_shared_risk();
        observe_boundary(0);
        schedule_local_market_makers(0, 0);
        schedule_global_agents(0, 0);
        if (config_.enable_local_market_makers) {
            ++local_mm_refresh_boundaries_;
        }

        // Global boundaries are the only points at which ranks must agree on
        // the shared firm's inventory.  Local market-makers can wake on a
        // finer (or coarser) deterministic clock without an MPI collective.
        // Process the books only over the union of both clocks, so no agent
        // observes a future state or gets an implementation-order advantage.
        std::int64_t current_ns = 0;
        std::int64_t next_global_boundary_ns = config_.decision_window_ns;
        std::int64_t next_local_refresh_ns = config_.enable_local_market_makers
            ? config_.local_mm_interval_ns
            : std::numeric_limits<std::int64_t>::max();
        std::uint64_t global_boundary_index = 1;
        std::uint64_t local_refresh_index = 1;
        while (current_ns < end_time_ns_) {
            const std::int64_t end_ns = std::min(
                end_time_ns_, std::min(next_global_boundary_ns,
                                       next_local_refresh_ns));
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
            if (global_boundary) {
                update_shared_risk();
                observe_boundary(end_ns);
            }
            if (!terminal_boundary) {
                // At coincident wake times local liquidity is placed first,
                // then the globally dependent agents, matching the legacy
                // same-boundary order while keeping sequence IDs stable.
                if (local_refresh) {
                    schedule_local_market_makers(end_ns, local_refresh_index);
                    ++local_mm_refresh_boundaries_;
                    next_local_refresh_ns = checked_add_time(
                        next_local_refresh_ns, config_.local_mm_interval_ns);
                    ++local_refresh_index;
                }
                if (global_boundary) {
                    schedule_global_agents(end_ns, global_boundary_index);
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
            || config_.tick_size <= 0
            || !std::isfinite(config_.initial_depth_scale)
            || config_.initial_depth_scale <= 0.0
            || config_.asset_configs.size()
                != static_cast<std::size_t>(config_.asset_count)
            || !std::isfinite(config_.value_threshold_bps)
            || config_.value_threshold_bps < 0.0
            || config_.value_order_quantity <= 0
            || !std::isfinite(config_.hawkes_activity_scale)
            || config_.hawkes_activity_scale <= 0.0
            || !std::isfinite(config_.local_mm_quantity_multiplier)
            || config_.local_mm_quantity_multiplier <= 0.0
            || config_.shared_quote_quantity <= 0
            || config_.shared_quote_levels <= 0
            || config_.shared_quote_levels > 16
            || !std::isfinite(config_.shared_quote_multiplier)
            || config_.shared_quote_multiplier <= 0.0
            || !std::isfinite(config_.shared_risk_limit_per_asset)
            || config_.shared_risk_limit_per_asset <= 0.0
            || !std::isfinite(config_.shock_asset_fraction)
            || config_.shock_asset_fraction <= 0.0
            || config_.shock_asset_fraction > 1.0
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
        for (const FragmentedValueAgentPolicy& policy : config_.value_agent_policies) {
            if (!std::isfinite(policy.threshold_bps) || policy.threshold_bps < 0.0
                || policy.order_quantity <= 0) {
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
        const auto count = static_cast<std::size_t>(std::max<long long>(
            1LL,
            std::llround(config_.shock_asset_fraction
                         * static_cast<double>(config_.asset_count))));
        std::vector<std::pair<std::uint64_t, std::size_t>> ranked;
        ranked.reserve(static_cast<std::size_t>(config_.asset_count));
        for (std::size_t asset = 0;
             asset < static_cast<std::size_t>(config_.asset_count); ++asset) {
            ranked.emplace_back(
                stable_sequence(liquidity_shock_entity(static_cast<BookId>(asset)),
                                config_.seed),
                asset);
        }
        std::sort(ranked.begin(), ranked.end());
        for (std::size_t index = 0; index < count; ++index) {
            shock_mask_[ranked[index].second] = true;
        }
        shock_asset_count_ = static_cast<std::uint64_t>(count);
    }

    void write_shock_targets() const {
        if (rank_ != 0 || config_.shock_targets_csv.empty()) return;
        const std::filesystem::path path(config_.shock_targets_csv);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("cannot open shock target CSV: "
                                     + path.string());
        }
        output << "asset_id,symbol,is_shock_target,shock_enabled,"
                  "counterfactual_sell_quantity\n";
        for (std::size_t index = 0; index < config_.asset_configs.size(); ++index) {
            output << index << ','
                   << config_.asset_configs[index].symbol << ','
                   << (shock_mask_[index] ? 1 : 0) << ','
                   << (config_.enable_shock ? 1 : 0) << ','
                   << (shock_mask_[index]
                       ? shock_quantity_for_asset(config_.asset_configs[index]) : 0)
                   << '\n';
        }
    }

    [[nodiscard]] double opening_bid_depth_for_asset(
        const MultiAssetBookConfig& source) const {
        // Match seed_calibrated_book's top-level rounding exactly so the
        // recorded target quantity is the quantity that will be submitted.
        return static_cast<double>(std::max(1, static_cast<int>(
            std::llround(config_.initial_depth_scale
                         * static_cast<double>(source.initial_best_bid_depth)))));
    }

    [[nodiscard]] int shock_quantity_for_asset(
        const MultiAssetBookConfig& source) const {
        if (config_.shock_top_depth_multiple > 0.0) {
            return bounded_positive_quantity(
                config_.shock_top_depth_multiple
                    * opening_bid_depth_for_asset(source),
                "normalised liquidity-shock quantity");
        }
        return config_.shock_quantity_per_asset;
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

            SequentialMultiAssetConfig bridge;
            bridge.seed = config_.seed;
            bridge.tick_size = config_.tick_size;
            BackgroundHawkesConfig background =
                make_multi_asset_background_config(bridge, source, asset_id);
            background.activity_scale = config_.hawkes_activity_scale;
            auto asset = std::make_unique<LocalAsset>(
                asset_id, source, background, config_.tick_size);
            asset->book.lob.seed_calibrated_book(
                source.initial_best_bid_ticks,
                source.initial_best_ask_ticks,
                source.initial_best_bid_depth,
                source.initial_best_ask_depth,
                config_.initial_depth_scale);
            compute_baseline(*asset);
            if (config_.enable_shock
                && shock_mask_[static_cast<std::size_t>(asset_index)]) {
                const int shock_quantity = shock_quantity_for_asset(source);
                asset->shock_requested_quantity =
                    static_cast<std::uint64_t>(shock_quantity);
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

    int apply_to_book(LocalAsset& asset,
                      const OrderMessage& source,
                      int quantity) {
        LocalBook& book = asset.book;
        OrderMessage order = source;
        order.book_id = book.book_id;
        order.quantity = quantity;
        const ApplyResult result = book.lob.apply(order);
        ++book.processed_orders;
        account_trades(asset, book);
        return result.executed_quantity;
    }

    void apply_order(LocalAsset& asset, const OrderMessage& order) {
        if (order.book_id != asset.book.book_id) {
            throw std::logic_error("order targets a book outside its asset shard");
        }
        (void)apply_to_book(asset, order, std::max(0, order.quantity));
    }

    void process_window(LocalAsset& asset,
                        std::int64_t start_ns,
                        std::int64_t end_ns) {
        std::sort(asset.pending_orders.begin(), asset.pending_orders.end(), order_less);
        std::size_t pending_index = 0;
        while (true) {
            const bool has_pending = pending_index < asset.pending_orders.size()
                && asset.pending_orders[pending_index].arrival_time_ns < end_ns;
            const HawkesEvent& next_background = asset.hawkes.peek();
            const bool has_background = next_background.time_ns < end_ns;
            if (!has_pending && !has_background) break;

            const std::int64_t pending_time = has_pending
                ? asset.pending_orders[pending_index].arrival_time_ns
                : std::numeric_limits<std::int64_t>::max();
            const std::int64_t background_time = has_background
                ? next_background.time_ns : std::numeric_limits<std::int64_t>::max();
            if (pending_time <= background_time) {
                OrderMessage order = asset.pending_orders[pending_index++];
                order.arrival_time_ns = std::max(start_ns, order.arrival_time_ns);
                apply_order(asset, order);
                continue;
            }

            const HawkesEvent event = asset.hawkes.pop();
            const std::uint64_t event_sequence = asset.hawkes.accepted_events();
            LocalBook& book = asset.book;
            OrderMessage order = asset.background.make_order(
                event,
                book.lob.state(event.time_ns,
                               asset.config.fundamental_price_ticks),
                stable_sequence(background_entity(asset.asset_id), event_sequence));
            order.book_id = book.book_id;
            apply_order(asset, order);
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
                       double bid_scale,
                       double ask_scale) {
        const std::int64_t arrival_time_ns = checked_add_time(
            decision_time_ns, agent_latency_ns);
        const MarketState state = book.lob.state(
            decision_time_ns, asset.config.fundamental_price_ticks);
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
            order.owner_id = owner_id;
            order.agent_kind = AgentKind::MarketMaker;
            order.action = action;
            order.side = side;
            order.quantity = quantity;
            order.price_ticks = price;
            asset.pending_orders.push_back(order);
        };

        make_order(OrderAction::CancelOwner, Side::Buy, 0, 0);
        if (base_quantity <= 0
            || (bid_scale <= 0.0 && ask_scale <= 0.0)) return;

        const std::int64_t tick = config_.tick_size;
        const std::int64_t target_spread = std::max<std::int64_t>(
            tick,
            tick * static_cast<std::int64_t>(asset.config.target_spread_ticks));
        std::int64_t bid = state.best_bid_ticks;
        std::int64_t ask = state.best_ask_ticks;
        if (bid <= 0 || ask <= bid) {
            const auto center = static_cast<std::int64_t>(std::llround(
                asset.config.fundamental_price_ticks / static_cast<double>(tick))) * tick;
            bid = center - target_spread / 2;
            ask = bid + target_spread;
        } else if (ask - bid > target_spread) {
            double reference = state.last_trade_price_ticks > 0
                ? static_cast<double>(state.last_trade_price_ticks)
                : state.mid_price_ticks;
            if (!std::isfinite(reference) || reference <= 0.0) {
                reference = asset.config.fundamental_price_ticks;
            }
            const std::int64_t preferred_bid = static_cast<std::int64_t>(
                std::llround((reference - static_cast<double>(target_spread) / 2.0)
                             / static_cast<double>(tick))) * tick;
            bid = std::clamp<std::int64_t>(
                preferred_bid, state.best_bid_ticks,
                static_cast<std::int64_t>(state.best_ask_ticks) - target_spread);
            ask = bid + target_spread;
        }

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
                make_order(OrderAction::Limit, Side::Buy, bid_quantity,
                           static_cast<int>(level_bid));
            }
            if (level_ask > level_bid && ask_quantity > 0
                && level_ask <= std::numeric_limits<std::int32_t>::max()) {
                make_order(OrderAction::Limit, Side::Sell, ask_quantity,
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
                local_market_maker_owner_base
                    + static_cast<std::int32_t>(book.book_id),
                local_market_maker_entity_base, local_quantity, 1, 1.0, 1.0);
        }
        compute_seconds_ += MPI_Wtime() - compute_start;
    }

    void schedule_global_agents(std::int64_t decision_time_ns,
                                std::uint64_t boundary_index) {
        const double compute_start = MPI_Wtime();
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            LocalAsset& asset = *pointer;
            LocalBook& book = asset.book;
            if (config_.enable_shared_market_maker) {
                const double inventory_ratio = std::min(
                    0.75,
                    std::abs(static_cast<double>(asset.shared_inventory))
                        / std::max(1.0, config_.shared_risk_limit_per_asset));
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
                append_quotes(
                    asset, book, decision_time_ns, boundary_index,
                    shared_market_maker_owner,
                    fragmented_shared_maker_entity,
                    shared_book_quantity,
                    config_.shared_quote_levels,
                    shared_quote_scale_ * bid_inventory_scale,
                    shared_quote_scale_ * ask_inventory_scale);
            }
            schedule_value_agent(asset, decision_time_ns, boundary_index);
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
        const double mid = 0.5 * static_cast<double>(
            book.lob.best_ask() + book.lob.best_bid());
        const double deviation_bps = (mid - asset.config.fundamental_price_ticks)
            / asset.config.fundamental_price_ticks * 10'000.0;
        if (std::abs(deviation_bps) < policy.threshold_bps) return;
        const Side side = deviation_bps > 0.0 ? Side::Sell : Side::Buy;
        const std::int64_t arrival_time = checked_add_time(
            decision_time_ns, agent_latency_ns);
        asset.pending_orders.push_back(make_market_order(
            asset.asset_id,
            arrival_time,
            fundamental_value_owner_id(asset.asset_id),
            AgentKind::Value,
            side,
            policy.order_quantity,
            stable_sequence(fragmented_value_entity_base
                                + static_cast<StableEntityId>(asset.asset_id),
                            boundary_index + 1U)));
    }

    void update_shared_risk() {
        if (!config_.enable_shared_market_maker) {
            shared_gross_exposure_ = 0.0;
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
        const double global_limit = config_.shared_risk_limit_per_asset
            * static_cast<double>(config_.asset_count);
        const double utilization = shared_gross_exposure_ / global_limit;
        shared_quote_scale_ = utilization <= 0.5
            ? 1.0 : std::max(0.0, 2.0 * (1.0 - utilization));
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
                time_ns, asset.config.fundamental_price_ticks);
            asset.calibration_moments.observe(state, config_.tick_size);
        }
    }

    AggregateMetricSums local_metrics(std::int64_t time_ns) const {
        AggregateMetricSums metrics;
        for (const std::unique_ptr<LocalAsset>& pointer : local_assets_) {
            const LocalAsset& asset = *pointer;
            const MarketState state = asset.book.lob.state(
                time_ns, asset.config.fundamental_price_ticks);
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
                if (affected) ++metrics.affected_shocked_asset_count;
                if (two_sided) {
                    metrics.shocked_spread_sum_bps += asset_spread;
                    ++metrics.shocked_two_sided_asset_count;
                }
            } else {
                ++metrics.unshocked_asset_count;
                metrics.unshocked_top_depth_sum += asset_depth;
                if (affected) ++metrics.affected_unshocked_asset_count;
                if (two_sided) {
                    metrics.unshocked_spread_sum_bps += asset_spread;
                    ++metrics.unshocked_two_sided_asset_count;
                }
            }
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
               "shared_quote_scale\n";
    }

    void observe_boundary(std::int64_t time_ns) {
        record_asset_moments(time_ns);
        const AggregateMetricSums local = local_metrics(time_ns);
        const std::array<double, 14> send{{
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
            local.unshocked_top_depth_sum}};
        std::array<double, 14> global{};
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
                << shared_quote_scale_ << '\n';
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
                end_time_ns_, asset->config.fundamental_price_ticks);
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
                asset->asset_id, asset->shared_inventory, asset->value_inventory});
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
                  "invalid_sample_count,structurally_valid,"
                  "background_event_count,background_event_rate,"
                  "mean_spread_ticks,mean_bid_depth,mean_ask_depth,"
                  "mid_move_rate,return_variance,return_kurtosis,"
                  "absolute_return_acf1\n";
        output << std::setprecision(17);
        for (const AssetMomentWire& moment : moments) {
            if (moment.asset_id >= config_.asset_configs.size()) {
                throw std::logic_error("per-asset moment has an invalid asset id");
            }
            const bool structurally_valid = moment.sample_count == expected_samples
                && moment.invalid_sample_count == 0;
            output << moment.asset_id << ','
                   << config_.asset_configs[
                       static_cast<std::size_t>(moment.asset_id)].symbol << ','
                   << moment.sample_count << ','
                   << expected_samples << ','
                   << moment.invalid_sample_count << ','
                   << (structurally_valid ? 1 : 0) << ','
                   << moment.background_event_count << ','
                   << static_cast<double>(moment.background_event_count)
                        / static_cast<double>(config_.duration_seconds);
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
            hash_integer(hash, state.last_trade_price_ticks);
            hash_double(hash, state.mid_price_ticks);
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
