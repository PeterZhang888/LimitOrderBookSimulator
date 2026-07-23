#include "simulation/ExactMpiMultiAssetSimulator.hpp"

#include "agents/EtfArbitrageAgent.hpp"
#include "agents/FundamentalValueAgent.hpp"
#include "agents/SharedMarketMakerAgent.hpp"
#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dlob {
namespace {

constexpr std::int64_t nanoseconds_per_second = 1'000'000'000LL;
constexpr std::size_t error_message_bytes = 512;

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}

int checked_byte_count(std::size_t count, std::size_t width, const char* label) {
    if (width != 0U && count > static_cast<std::size_t>(
            std::numeric_limits<int>::max()) / width) {
        throw std::overflow_error(std::string(label) + " exceeds the MPI byte-count range");
    }
    return static_cast<int>(count * width);
}

std::string current_exception() {
    try {
        throw;
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return "unknown non-standard exception";
    }
}

std::int64_t checked_duration_ns(int duration_seconds) {
    if (duration_seconds <= 0) {
        throw std::invalid_argument("multi-asset duration must be positive");
    }
    constexpr auto maximum_seconds = std::numeric_limits<std::int64_t>::max()
        / nanoseconds_per_second;
    if (static_cast<std::int64_t>(duration_seconds) > maximum_seconds) {
        throw std::invalid_argument("multi-asset duration is too large");
    }
    return static_cast<std::int64_t>(duration_seconds) * nanoseconds_per_second;
}

std::int64_t checked_add(std::int64_t time_ns,
                         std::int64_t delay_ns,
                         const char* label) {
    if (delay_ns <= 0) {
        throw std::invalid_argument(std::string(label) + " must be positive");
    }
    if (time_ns > std::numeric_limits<std::int64_t>::max() - delay_ns) {
        throw std::overflow_error(std::string(label) + " overflows event time");
    }
    return time_ns + delay_ns;
}

std::uint64_t report_origin_sequence(const TradeExecution& trade) {
    return stable_sequence(background_entity(trade.book_id), trade.trade_sequence);
}

bool order_equal(const OrderMessage& left, const OrderMessage& right) {
    return left.generated_time_ns == right.generated_time_ns
        && left.arrival_time_ns == right.arrival_time_ns
        && left.sequence == right.sequence
        && left.tie_breaker == right.tie_breaker
        && left.source_rank == right.source_rank
        && left.owner_id == right.owner_id
        && left.agent_kind == right.agent_kind
        && left.action == right.action
        && left.side == right.side
        && left.quantity == right.quantity
        && left.price_ticks == right.price_ticks
        && left.distance_ticks == right.distance_ticks
        && left.book_id == right.book_id;
}

bool report_equal(const AgentReport& left, const AgentReport& right) {
    return left.timestamp_ns == right.timestamp_ns
        && left.owner_id == right.owner_id
        && left.order_sequence == right.order_sequence
        && left.kind == right.kind
        && left.action == right.action
        && left.side == right.side
        && left.requested_quantity == right.requested_quantity
        && left.executed_quantity == right.executed_quantity
        && left.resting_quantity == right.resting_quantity
        && left.cancelled_quantity == right.cancelled_quantity
        && left.fill_quantity == right.fill_quantity
        && left.fill_price_ticks == right.fill_price_ticks
        && left.book_id == right.book_id;
}

bool trade_equal(const TradeExecution& left, const TradeExecution& right) {
    return left.book_id == right.book_id
        && left.timestamp_ns == right.timestamp_ns
        && left.trade_sequence == right.trade_sequence
        && left.price_ticks == right.price_ticks
        && left.quantity == right.quantity
        && left.buyer_owner_id == right.buyer_owner_id
        && left.seller_owner_id == right.seller_owner_id
        && left.buyer_order_sequence == right.buyer_order_sequence
        && left.seller_order_sequence == right.seller_order_sequence
        && left.aggressor_side == right.aggressor_side
        && left.aggressor_action == right.aggressor_action;
}

bool event_equal(const MultiAssetEvent& left, const MultiAssetEvent& right) {
    return left.key == right.key
        && left.kind == right.kind
        && left.source_book_id == right.source_book_id
        && left.hawkes.time_ns == right.hawkes.time_ns
        && left.hawkes.type == right.hawkes.type
        && order_equal(left.order, right.order)
        && report_equal(left.report, right.report)
        && trade_equal(left.trade, right.trade)
        && left.may_trigger_cross_book_reaction
            == right.may_trigger_cross_book_reaction;
}

bool event_vectors_equal(const std::vector<MultiAssetEvent>& left,
                         const std::vector<MultiAssetEvent>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (!event_equal(left[index], right[index])) return false;
    }
    return true;
}

class ConfigHasher {
public:
    template <typename Integer>
    void add_integer(Integer value) {
        static_assert(std::is_integral_v<Integer>);
        using Unsigned = std::make_unsigned_t<Integer>;
        const Unsigned bits = static_cast<Unsigned>(value);
        for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
            const std::size_t shift = (sizeof(Unsigned) - index - 1U) * 8U;
            add_byte(static_cast<std::uint8_t>(bits >> shift));
        }
    }

    void add_double(double value) {
        add_integer(std::bit_cast<std::uint64_t>(value));
    }

    void add_string(const std::string& value) {
        add_integer(static_cast<std::uint64_t>(value.size()));
        for (char character : value) {
            add_byte(static_cast<std::uint8_t>(
                static_cast<unsigned char>(character)));
        }
    }

    [[nodiscard]] std::uint64_t digest() const noexcept { return hash_; }

private:
    void add_byte(std::uint8_t byte) {
        hash_ ^= byte;
        hash_ *= 1099511628211ULL;
    }

    std::uint64_t hash_ = 14695981039346656037ULL;
};

std::uint64_t config_digest(const SequentialMultiAssetConfig& config) {
    ConfigHasher hash;
    hash.add_integer(config.duration_seconds);
    hash.add_integer(config.sample_interval_ns);
    hash.add_integer(config.book_count);
    hash.add_integer(config.seed);
    hash.add_string(config.data_dir);
    hash.add_string(config.hawkes_rates_file);
    hash.add_integer(static_cast<std::uint64_t>(config.book_configs.size()));
    for (const MultiAssetBookConfig& book : config.book_configs) {
        hash.add_string(book.symbol);
        hash.add_string(book.data_dir);
        hash.add_string(book.hawkes_rates_file);
        hash.add_double(book.fundamental_price_ticks);
        hash.add_integer(book.initial_best_bid_ticks);
        hash.add_integer(book.initial_best_ask_ticks);
        hash.add_integer(book.initial_best_bid_depth);
        hash.add_integer(book.initial_best_ask_depth);
        hash.add_double(book.beta);
        hash.add_double(book.basket_weight);
        hash.add_integer(book.market_maker_quote_quantity);
        hash.add_integer(book.target_spread_ticks);
        hash.add_double(book.quote_improvement_probability);
    }
    // output_dir intentionally does not affect the simulated model.
    hash.add_integer(config.tick_size);
    hash.add_double(config.initial_depth_scale);
    hash.add_double(config.fundamental_price_ticks);
    hash.add_integer(config.market_maker_order_quantity);
    hash.add_integer(config.market_maker_quote_levels);
    hash.add_integer(config.market_maker_quote_quantity_growth);
    hash.add_integer(config.market_maker_quote_interval_ns);
    hash.add_integer(config.market_maker_order_latency_ns);
    hash.add_integer(config.report_latency_ns);
    hash.add_integer(config.cross_book_reaction_latency_ns);
    hash.add_integer(config.hedge_order_latency_ns);
    hash.add_double(config.market_maker_exposure_threshold);
    hash.add_integer(config.enable_shared_market_maker_hedging ? 1U : 0U);
    hash.add_integer(config.hedge_lot_size);
    hash.add_integer(config.max_hedge_quantity);
    hash.add_integer(config.liquidity_shock.has_value() ? 1U : 0U);
    if (config.liquidity_shock.has_value()) {
        const LiquidityShockConfig& shock = *config.liquidity_shock;
        hash.add_integer(shock.time_ns);
        hash.add_integer(shock.book_id);
        hash.add_integer(static_cast<std::int32_t>(shock.side));
        hash.add_integer(shock.quantity);
    }
    hash.add_integer(config.etf_arbitrage.enabled ? 1U : 0U);
    hash.add_integer(config.etf_arbitrage.etf_book_id);
    hash.add_double(config.etf_arbitrage.trigger_bps);
    hash.add_double(config.etf_arbitrage.release_bps);
    hash.add_integer(config.etf_arbitrage.etf_order_quantity);
    hash.add_integer(config.etf_arbitrage.max_component_quantity);
    hash.add_integer(config.etf_arbitrage.decision_interval_ns);
    hash.add_integer(config.etf_arbitrage.order_latency_ns);
    hash.add_integer(config.fundamental_value.enabled ? 1U : 0U);
    hash.add_double(config.fundamental_value.threshold_bps);
    hash.add_double(config.fundamental_value.response_step_bps);
    hash.add_integer(config.fundamental_value.base_order_quantity);
    hash.add_integer(config.fundamental_value.max_order_quantity);
    hash.add_integer(config.fundamental_value.max_abs_inventory);
    hash.add_double(
        config.fundamental_value.fundamental_volatility_bps_sqrt_second);
    hash.add_integer(config.fundamental_value.decision_interval_ns);
    hash.add_integer(config.fundamental_value.order_latency_ns);
    return hash.digest();
}

struct BookSummaryWire {
    BookId book_id = 0;
    MarketState final_state{};
    std::int64_t market_maker_inventory = 0;
    std::int64_t market_maker_cash_ticks = 0;
    std::int64_t arbitrage_inventory = 0;
    std::int64_t arbitrage_cash_ticks = 0;
    std::int64_t value_agent_inventory = 0;
    std::int64_t value_agent_cash_ticks = 0;
    double final_fundamental_value_ticks = 0.0;
    std::uint64_t processed_events = 0;
    std::uint64_t submitted_orders = 0;
    std::uint64_t trade_count = 0;
    std::uint64_t trade_hash = TradeTapeHasher::offset_basis;
    std::array<std::uint64_t, calibration::empirical_event_bucket_count> event_counts{};
    std::uint64_t owner_cancel_messages = 0;
    calibration::MarketFeatureSummary market{};
};

struct ReplicaWire {
    std::uint64_t combined_trade_count = 0;
    std::uint64_t combined_trade_hash = TradeTapeHasher::offset_basis;
    std::uint64_t processed_events = 0;
    std::uint64_t cross_book_reaction_events = 0;
    std::uint64_t hedge_order_events = 0;
    std::uint64_t liquidity_shock_events = 0;
    std::uint64_t arbitrage_decision_events = 0;
    std::uint64_t arbitrage_order_events = 0;
    std::uint64_t value_decision_events = 0;
    std::uint64_t value_order_events = 0;
    std::int64_t market_maker_cash_ticks = 0;
    std::int64_t arbitrage_cash_ticks = 0;
};

static_assert(std::is_trivially_copyable_v<MultiAssetEvent>);
static_assert(std::is_trivially_copyable_v<BookSummaryWire>);
static_assert(std::is_trivially_copyable_v<ReplicaWire>);

} // namespace

class ExactMpiMultiAssetSimulator::Impl {
public:
    Impl(MPI_Comm communicator, SequentialMultiAssetConfig config)
        : communicator_(communicator), config_(std::move(config)) {
        check_mpi(MPI_Comm_rank(communicator_, &rank_),
                  "MPI_Comm_rank(exact multi-asset)");
        check_mpi(MPI_Comm_size(communicator_, &world_size_),
                  "MPI_Comm_size(exact multi-asset)");
        if (world_size_ <= 0 || rank_ < 0 || rank_ >= world_size_) {
            throw std::runtime_error("invalid MPI rank metadata");
        }
    }

    ExactMpiMultiAssetResult run() {
        if (completed_) {
            throw std::logic_error("an ExactMpiMultiAssetSimulator instance can only run once");
        }

        check_mpi(MPI_Barrier(communicator_),
                  "MPI_Barrier(exact multi-asset start)");
        const double wall_start = MPI_Wtime();
        initialize();

        while (true) {
            const Selection selection = select_global_minimum();
            if (selection.stop) break;
            MultiAssetEvent event = broadcast_winning_event(selection);
            validate_global_event(event, selection);
            last_processed_key_ = event.key;
            process_global_event(event);
        }

        check_mpi(MPI_Barrier(communicator_),
                  "MPI_Barrier(exact multi-asset end)");
        const double local_wall_seconds = MPI_Wtime() - wall_start;
        double wall_seconds = 0.0;
        check_mpi(MPI_Reduce(&local_wall_seconds, &wall_seconds, 1, MPI_DOUBLE,
                             MPI_MAX, 0, communicator_),
                  "MPI_Reduce(exact wall time)");
        check_mpi(MPI_Bcast(&wall_seconds, 1, MPI_DOUBLE, 0, communicator_),
                  "MPI_Bcast(exact wall time)");

        ExactMpiMultiAssetResult result = finish(wall_seconds);
        completed_ = true;
        return result;
    }

private:
    using EventQueue = std::priority_queue<MultiAssetEvent,
                                           std::vector<MultiAssetEvent>,
                                           MultiAssetEventLater>;

    struct Selection {
        bool stop = false;
        int winner_rank = -1;
        MultiAssetEventKey key{};
    };

    void validate_config() {
        if (config_.book_count < 1) {
            throw std::invalid_argument("--books must be at least 1");
        }
        if (static_cast<std::uint64_t>(config_.book_count)
            > static_cast<std::uint64_t>(std::numeric_limits<BookId>::max())) {
            throw std::invalid_argument("--books exceeds the BookId range");
        }
        if (config_.fundamental_value.enabled && config_.book_count > 100'000) {
            throw std::invalid_argument(
                "value-agent owner identifiers support at most 100000 books");
        }
        if (config_.tick_size <= 0 || config_.initial_depth_scale <= 0.0
            || !std::isfinite(config_.initial_depth_scale)
            || config_.fundamental_price_ticks <= 0.0
            || !std::isfinite(config_.fundamental_price_ticks)
            || config_.sample_interval_ns <= 0
            || config_.market_maker_order_quantity <= 0
            || config_.market_maker_quote_levels <= 0
            || config_.market_maker_quote_quantity_growth <= 0
            || config_.market_maker_quote_interval_ns <= 0
            || config_.market_maker_order_latency_ns <= 0
            || config_.report_latency_ns <= 0
            || config_.cross_book_reaction_latency_ns <= 0
            || config_.hedge_order_latency_ns <= 0
            || config_.market_maker_exposure_threshold < 0.0
            || !std::isfinite(config_.market_maker_exposure_threshold)
            || config_.hedge_lot_size <= 0 || config_.max_hedge_quantity <= 0) {
            throw std::invalid_argument("invalid exact MPI multi-asset configuration");
        }
        end_time_ns_ = checked_duration_ns(config_.duration_seconds);
        if (config_.sample_interval_ns > end_time_ns_) {
            throw std::invalid_argument("sample interval exceeds simulated duration");
        }
        if (config_.liquidity_shock.has_value()) {
            const LiquidityShockConfig& shock = *config_.liquidity_shock;
            if (shock.time_ns < 0 || shock.time_ns > end_time_ns_
                || shock.book_id >= static_cast<BookId>(config_.book_count)
                || (shock.side != Side::Buy && shock.side != Side::Sell)
                || shock.quantity <= 0) {
                throw std::invalid_argument(
                    "invalid exact MPI liquidity-shock configuration");
            }
        }
    }

    void synchronize_error(const std::string& local_error, const char* context) const {
        const int failed = local_error.empty() ? 0 : 1;
        std::array<char, error_message_bytes> local_message{};
        if (failed != 0) {
            const std::size_t count = std::min(local_error.size(),
                                               local_message.size() - 1U);
            std::memcpy(local_message.data(), local_error.data(), count);
        }

        std::vector<int> failures;
        std::vector<char> messages;
        if (rank_ == 0) {
            failures.resize(static_cast<std::size_t>(world_size_));
            messages.resize(static_cast<std::size_t>(world_size_)
                            * error_message_bytes);
        }
        check_mpi(MPI_Gather(&failed, 1, MPI_INT,
                             rank_ == 0 ? failures.data() : nullptr,
                             1, MPI_INT, 0, communicator_),
                  "MPI_Gather(collective error flags)");
        check_mpi(MPI_Gather(local_message.data(),
                             static_cast<int>(local_message.size()), MPI_BYTE,
                             rank_ == 0 ? messages.data() : nullptr,
                             static_cast<int>(local_message.size()), MPI_BYTE,
                             0, communicator_),
                  "MPI_Gather(collective error messages)");

        int failing_rank = -1;
        std::array<char, error_message_bytes> chosen{};
        if (rank_ == 0) {
            for (int candidate = 0; candidate < world_size_; ++candidate) {
                if (failures[static_cast<std::size_t>(candidate)] == 0) continue;
                failing_rank = candidate;
                std::memcpy(chosen.data(),
                            messages.data() + static_cast<std::size_t>(candidate)
                                * error_message_bytes,
                            error_message_bytes);
                break;
            }
        }
        check_mpi(MPI_Bcast(&failing_rank, 1, MPI_INT, 0, communicator_),
                  "MPI_Bcast(collective error rank)");
        check_mpi(MPI_Bcast(chosen.data(), static_cast<int>(chosen.size()),
                            MPI_BYTE, 0, communicator_),
                  "MPI_Bcast(collective error message)");
        if (failing_rank >= 0) {
            throw std::runtime_error(std::string(context) + " failed on rank "
                                     + std::to_string(failing_rank) + ": "
                                     + chosen.data());
        }
    }

    void verify_config_consistency() const {
        const std::uint64_t local = config_digest(config_);
        std::vector<std::uint64_t> digests;
        if (rank_ == 0) digests.resize(static_cast<std::size_t>(world_size_));
        check_mpi(MPI_Gather(&local, 1, MPI_UNSIGNED_LONG_LONG,
                             rank_ == 0 ? digests.data() : nullptr,
                             1, MPI_UNSIGNED_LONG_LONG, 0, communicator_),
                  "MPI_Gather(exact model config digest)");
        std::string error;
        if (rank_ == 0) {
            for (int candidate = 1; candidate < world_size_; ++candidate) {
                if (digests[static_cast<std::size_t>(candidate)] != digests[0]) {
                    error = "model configuration differs across MPI ranks";
                    break;
                }
            }
        }
        synchronize_error(error, "configuration consistency check");
    }

    void initialize() {
        std::string error;
        try {
            validate_config();
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "configuration validation");
        verify_config_consistency();

        error.clear();
        try {
            resolved_book_configs_ = resolve_multi_asset_book_configs(config_);
            shared_market_maker_ = std::make_unique<SharedMarketMakerAgent>(
                make_multi_asset_market_maker_config(
                    config_, resolved_book_configs_));
            etf_arbitrage_agent_ = std::make_unique<EtfArbitrageAgent>(
                config_.etf_arbitrage, resolved_book_configs_);
            value_agents_.reserve(static_cast<std::size_t>(config_.book_count));
            for (int index = 0; index < config_.book_count; ++index) {
                const BookId id = static_cast<BookId>(index);
                value_agents_.push_back(std::make_unique<FundamentalValueAgent>(
                    config_.fundamental_value, id,
                    resolved_book_configs_[static_cast<std::size_t>(index)]
                        .fundamental_price_ticks,
                    config_.seed));
            }

            for (int index = 0; index < config_.book_count; ++index) {
                const BookId id = static_cast<BookId>(index);
                if (owner_for(id) != rank_) continue;
                const MultiAssetBookConfig& book_config =
                    resolved_book_configs_[static_cast<std::size_t>(index)];
                BackgroundHawkesConfig background =
                    make_multi_asset_background_config(config_, book_config, id);
                auto runtime = std::make_unique<BookRuntime>(
                    id, book_config.symbol,
                    book_config.fundamental_price_ticks, background,
                    config_.tick_size,
                    stable_sequence(sampler_entity(id), config_.seed));
                if (book_config.initial_best_bid_ticks > 0) {
                    runtime->lob.seed_calibrated_book(
                        book_config.initial_best_bid_ticks,
                        book_config.initial_best_ask_ticks,
                        book_config.initial_best_bid_depth,
                        book_config.initial_best_ask_depth,
                        config_.initial_depth_scale);
                } else {
                    runtime->lob.seed_default_book(config_.initial_depth_scale);
                }
                const auto [position, inserted] = local_books_.emplace(
                    id, std::move(runtime));
                (void)position;
                if (!inserted) throw std::logic_error("duplicate local book id");
            }

            for (auto& [id, runtime] : local_books_) {
                (void)id;
                schedule_background_events(*runtime);
                schedule_samples(runtime->book_id);
                schedule_next_quote(runtime->book_id, 0, 1);
                if (config_.fundamental_value.enabled) {
                    schedule_next_value_decision(
                        runtime->book_id,
                        config_.fundamental_value.decision_interval_ns, 1);
                }
            }
            schedule_liquidity_shock();
            if (config_.etf_arbitrage.enabled) {
                schedule_next_arbitrage(
                    config_.etf_arbitrage.decision_interval_ns, 1);
            }
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "local model initialization");
        initialized_ = true;
    }

    int owner_for(BookId book_id) const {
        return ExactMpiMultiAssetSimulator::owner_rank(book_id, world_size_);
    }

    BookRuntime& local_book(BookId book_id) {
        auto found = local_books_.find(book_id);
        if (found == local_books_.end()) {
            throw std::out_of_range("rank does not own requested book");
        }
        return *found->second;
    }

    void schedule_local(MultiAssetEvent event) {
        if (event.key.timestamp_ns < 0) {
            throw std::logic_error("cannot schedule an event before simulation start");
        }
        if (event.key.book_id >= static_cast<BookId>(config_.book_count)) {
            throw std::logic_error("event targets an unknown book");
        }
        if (owner_for(event.key.book_id) != rank_) {
            throw std::logic_error("event was enqueued on a non-owning MPI rank");
        }
        if ((event.kind == MultiAssetEventKind::OrderArrival
             || event.kind == MultiAssetEventKind::HedgeOrderArrival
             || event.kind == MultiAssetEventKind::LiquidityShockOrderArrival
             || event.kind == MultiAssetEventKind::ArbitrageOrderArrival
             || event.kind == MultiAssetEventKind::FundamentalValueOrderArrival)
            && event.order.book_id != event.key.book_id) {
            throw std::logic_error("order payload and event key target different books");
        }
        if (last_processed_key_.has_value() && event.key <= *last_processed_key_) {
            throw std::logic_error("generated event violates positive global causality");
        }
        const auto [position, inserted] = queued_keys_.insert(event.key);
        (void)position;
        if (!inserted) {
            throw std::logic_error("duplicate MultiAssetEventKey on local queue");
        }
        events_.push(std::move(event));
    }

    void schedule_background_events(BookRuntime& runtime) {
        const std::vector<HawkesEvent> generated = runtime.background.simulate(
            0, end_time_ns_);
        std::uint64_t sequence = 1;
        for (const HawkesEvent& hawkes : generated) {
            MultiAssetEvent event;
            event.key = MultiAssetEventKey{
                hawkes.time_ns,
                MultiAssetEventPhase::ExogenousWake,
                runtime.book_id,
                background_entity(runtime.book_id),
                sequence,
                0};
            event.kind = MultiAssetEventKind::BackgroundWake;
            event.hawkes = hawkes;
            schedule_local(std::move(event));
            ++sequence;
        }
    }

    void schedule_samples(BookId book_id) {
        std::int64_t timestamp_ns = config_.sample_interval_ns;
        std::uint64_t sequence = 1;
        while (timestamp_ns <= end_time_ns_) {
            MultiAssetEvent event;
            event.key = MultiAssetEventKey{
                timestamp_ns,
                MultiAssetEventPhase::Observation,
                book_id,
                sampler_entity(book_id),
                sequence,
                0};
            event.kind = MultiAssetEventKind::SampleState;
            schedule_local(std::move(event));
            ++sequence;
            if (end_time_ns_ - timestamp_ns < config_.sample_interval_ns) break;
            timestamp_ns += config_.sample_interval_ns;
        }
    }

    void schedule_liquidity_shock() {
        if (!config_.liquidity_shock.has_value()) return;
        const LiquidityShockConfig& shock = *config_.liquidity_shock;
        if (owner_for(shock.book_id) != rank_) return;
        const StableEntityId entity = liquidity_shock_entity(shock.book_id);

        MultiAssetEvent event;
        event.key = MultiAssetEventKey{
            shock.time_ns,
            MultiAssetEventPhase::OrderArrival,
            shock.book_id,
            entity,
            1,
            0};
        event.kind = MultiAssetEventKind::LiquidityShockOrderArrival;
        event.source_book_id = shock.book_id;
        event.order.generated_time_ns = shock.time_ns;
        event.order.arrival_time_ns = shock.time_ns;
        event.order.sequence = stable_sequence(entity, 1);
        event.order.tie_breaker = stable_sequence(entity, 1, 1);
        event.order.source_rank = 0;
        event.order.owner_id = liquidity_shock_owner_id;
        event.order.agent_kind = AgentKind::Institutional;
        event.order.action = OrderAction::Market;
        event.order.side = shock.side;
        event.order.quantity = shock.quantity;
        event.order.book_id = shock.book_id;
        schedule_local(std::move(event));
    }

    void schedule_next_quote(BookId book_id,
                             std::int64_t timestamp_ns,
                             std::uint64_t quote_sequence) {
        if (timestamp_ns > end_time_ns_) return;
        MultiAssetEvent event;
        event.key = MultiAssetEventKey{
            timestamp_ns,
            MultiAssetEventPhase::AgentDecision,
            book_id,
            shared_market_maker_entity,
            quote_sequence,
            0};
        event.kind = MultiAssetEventKind::MarketMakerQuoteWake;
        schedule_local(std::move(event));
    }

    void schedule_next_arbitrage(std::int64_t timestamp_ns,
                                 std::uint64_t decision_sequence) {
        if (!config_.etf_arbitrage.enabled || timestamp_ns > end_time_ns_
            || owner_for(config_.etf_arbitrage.etf_book_id) != rank_) {
            return;
        }
        MultiAssetEvent event;
        event.key = MultiAssetEventKey{
            timestamp_ns,
            MultiAssetEventPhase::AgentDecision,
            config_.etf_arbitrage.etf_book_id,
            etf_arbitrage_entity,
            decision_sequence,
            0};
        event.kind = MultiAssetEventKind::EtfArbitrageWake;
        schedule_local(std::move(event));
    }

    void schedule_next_value_decision(BookId book_id,
                                      std::int64_t timestamp_ns,
                                      std::uint64_t decision_sequence) {
        if (!config_.fundamental_value.enabled || timestamp_ns > end_time_ns_
            || owner_for(book_id) != rank_) {
            return;
        }
        MultiAssetEvent event;
        event.key = MultiAssetEventKey{
            timestamp_ns,
            MultiAssetEventPhase::AgentDecision,
            book_id,
            fundamental_value_entity(book_id),
            decision_sequence,
            0};
        event.kind = MultiAssetEventKind::FundamentalValueWake;
        schedule_local(std::move(event));
    }

    static MultiAssetEventKey decode_key(const std::uint64_t* values) {
        MultiAssetEventKey key;
        key.timestamp_ns = static_cast<std::int64_t>(values[0]);
        key.phase = static_cast<MultiAssetEventPhase>(values[1]);
        key.book_id = static_cast<BookId>(values[2]);
        key.origin_entity = values[3];
        key.origin_local_sequence = values[4];
        key.child_index = static_cast<std::uint32_t>(values[5]);
        return key;
    }

    Selection select_global_minimum() const {
        constexpr std::size_t candidate_width = 8;
        std::array<std::uint64_t, candidate_width> candidate{};
        if (!events_.empty()) {
            const MultiAssetEventKey& key = events_.top().key;
            candidate[0] = 1;
            candidate[1] = static_cast<std::uint64_t>(key.timestamp_ns);
            candidate[2] = static_cast<std::uint64_t>(key.phase);
            candidate[3] = key.book_id;
            candidate[4] = key.origin_entity;
            candidate[5] = key.origin_local_sequence;
            candidate[6] = key.child_index;
            candidate[7] = static_cast<std::uint64_t>(rank_);
        }

        std::vector<std::uint64_t> gathered;
        if (rank_ == 0) {
            gathered.resize(static_cast<std::size_t>(world_size_) * candidate_width);
        }
        check_mpi(MPI_Gather(candidate.data(), static_cast<int>(candidate.size()),
                             MPI_UNSIGNED_LONG_LONG,
                             rank_ == 0 ? gathered.data() : nullptr,
                             static_cast<int>(candidate.size()),
                             MPI_UNSIGNED_LONG_LONG, 0, communicator_),
                  "MPI_Gather(global minimum candidates)");

        // action: 0=terminal, 1=winner, 2=collective scheduling error.
        std::array<std::uint64_t, 9> control{};
        if (rank_ == 0) {
            std::optional<MultiAssetEventKey> minimum;
            int winner = -1;
            bool invalid = false;
            std::vector<MultiAssetEventKey> seen;
            for (int source = 0; source < world_size_; ++source) {
                const auto offset = static_cast<std::size_t>(source) * candidate_width;
                const std::uint64_t* values = gathered.data() + offset;
                if (values[0] == 0) continue;
                if (values[7] != static_cast<std::uint64_t>(source)
                    || values[1] > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max())
                    || values[2] > static_cast<std::uint64_t>(
                        MultiAssetEventPhase::Observation)
                    || values[3] >= static_cast<std::uint64_t>(config_.book_count)) {
                    invalid = true;
                    continue;
                }
                const MultiAssetEventKey key = decode_key(values + 1);
                if (owner_for(key.book_id) != source) invalid = true;
                if (std::find(seen.begin(), seen.end(), key) != seen.end()) invalid = true;
                seen.push_back(key);
                if (!minimum.has_value() || key < *minimum) {
                    minimum = key;
                    winner = source;
                }
            }
            if (invalid) {
                control[0] = 2;
            } else if (!minimum.has_value() || minimum->timestamp_ns > end_time_ns_) {
                control[0] = 0;
            } else {
                control[0] = 1;
                control[1] = static_cast<std::uint64_t>(winner);
                control[2] = static_cast<std::uint64_t>(minimum->timestamp_ns);
                control[3] = static_cast<std::uint64_t>(minimum->phase);
                control[4] = minimum->book_id;
                control[5] = minimum->origin_entity;
                control[6] = minimum->origin_local_sequence;
                control[7] = minimum->child_index;
            }
        }
        check_mpi(MPI_Bcast(control.data(), static_cast<int>(control.size()),
                            MPI_UNSIGNED_LONG_LONG, 0, communicator_),
                  "MPI_Bcast(global minimum winner)");
        if (control[0] == 2) {
            throw std::runtime_error(
                "global scheduler detected a duplicate, malformed, or misrouted candidate key");
        }
        if (control[0] == 0) return Selection{true, -1, {}};
        if (control[0] != 1
            || control[1] >= static_cast<std::uint64_t>(world_size_)) {
            throw std::runtime_error("global scheduler broadcast an invalid action");
        }
        return Selection{
            false,
            static_cast<int>(control[1]),
            decode_key(control.data() + 2)};
    }

    MultiAssetEvent broadcast_winning_event(const Selection& selection) {
        MultiAssetEvent event;
        std::string error;
        if (rank_ == selection.winner_rank) {
            try {
                if (events_.empty() || events_.top().key != selection.key) {
                    throw std::logic_error("winning queue head changed after candidate gather");
                }
                event = events_.top();
                events_.pop();
                if (queued_keys_.erase(event.key) != 1U) {
                    throw std::logic_error("winning event key missing from duplicate guard");
                }
            } catch (...) {
                error = current_exception();
            }
        }
        check_mpi(MPI_Bcast(&event, checked_byte_count(1, sizeof(event),
                                                   "winning event"),
                            MPI_BYTE, selection.winner_rank, communicator_),
                  "MPI_Bcast(winning event)");
        synchronize_error(error, "winning event removal");
        return event;
    }

    void validate_global_event(const MultiAssetEvent& event,
                               const Selection& selection) const {
        std::string error;
        try {
            if (event.key != selection.key) {
                throw std::logic_error("broadcast event does not match selected key");
            }
            if (owner_for(event.key.book_id) != selection.winner_rank) {
                throw std::logic_error("selected event winner does not own its book");
            }
            if (event.key.timestamp_ns < 0 || event.key.timestamp_ns > end_time_ns_) {
                throw std::logic_error("selected event lies outside the simulation horizon");
            }
            if (last_processed_key_.has_value()
                && event.key <= *last_processed_key_) {
                throw std::logic_error(
                    "MultiAssetEventKeys are not unique and globally increasing");
            }
            if (static_cast<std::uint64_t>(event.kind)
                > static_cast<std::uint64_t>(
                    MultiAssetEventKind::FundamentalValueOrderArrival)) {
                throw std::logic_error("selected event has an invalid kind");
            }
            if (event.kind == MultiAssetEventKind::LiquidityShockOrderArrival) {
                if (!config_.liquidity_shock.has_value()) {
                    throw std::logic_error(
                        "liquidity-shock event exists in a control run");
                }
                const LiquidityShockConfig& shock = *config_.liquidity_shock;
                const StableEntityId entity = liquidity_shock_entity(shock.book_id);
                if (event.key.timestamp_ns != shock.time_ns
                    || event.key.phase != MultiAssetEventPhase::OrderArrival
                    || event.key.book_id != shock.book_id
                    || event.key.origin_entity != entity
                    || event.key.origin_local_sequence != 1
                    || event.key.child_index != 0
                    || event.source_book_id != shock.book_id
                    || event.order.generated_time_ns != shock.time_ns
                    || event.order.arrival_time_ns != shock.time_ns
                    || event.order.sequence != stable_sequence(entity, 1)
                    || event.order.tie_breaker != stable_sequence(entity, 1, 1)
                    || event.order.source_rank != 0
                    || event.order.owner_id != liquidity_shock_owner_id
                    || event.order.agent_kind != AgentKind::Institutional
                    || event.order.action != OrderAction::Market
                    || event.order.side != shock.side
                    || event.order.quantity != shock.quantity
                    || event.order.price_ticks != 0
                    || event.order.distance_ticks != 0
                    || event.order.book_id != shock.book_id) {
                    throw std::logic_error(
                        "selected liquidity-shock event does not match its configuration");
                }
            } else if (event.key.origin_entity
                       == liquidity_shock_entity(event.key.book_id)) {
                throw std::logic_error(
                    "liquidity-shock entity emitted a non-shock event kind");
            }
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "global event validation");
    }

    template <typename Value>
    Value broadcast_value(Value value, int root, const char* label) const {
        static_assert(std::is_trivially_copyable_v<Value>);
        check_mpi(MPI_Bcast(&value, checked_byte_count(1, sizeof(Value), label),
                            MPI_BYTE, root, communicator_), label);
        return value;
    }

    template <typename Value>
    std::vector<Value> broadcast_vector(std::vector<Value> values,
                                        int root,
                                        const char* label) const {
        static_assert(std::is_trivially_copyable_v<Value>);
        std::uint64_t count = rank_ == root
            ? static_cast<std::uint64_t>(values.size()) : 0;
        check_mpi(MPI_Bcast(&count, 1, MPI_UNSIGNED_LONG_LONG, root, communicator_),
                  label);
        if (count > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(std::string(label) + " vector is too large");
        }
        if (rank_ != root) values.resize(static_cast<std::size_t>(count));
        const int bytes = checked_byte_count(static_cast<std::size_t>(count),
                                             sizeof(Value), label);
        check_mpi(MPI_Bcast(values.empty() ? nullptr : values.data(), bytes,
                            MPI_BYTE, root, communicator_), label);
        return values;
    }

    MarketState current_state(BookId book_id,
                              std::int64_t timestamp_ns,
                              const char* context) {
        const int owner = owner_for(book_id);
        MarketState state;
        std::string error;
        if (rank_ == owner) {
            try {
                BookRuntime& runtime = local_book(book_id);
                state = runtime.lob.state(timestamp_ns,
                                          runtime.fundamental_value_ticks);
            } catch (...) {
                error = current_exception();
            }
        }
        synchronize_error(error, context);
        state = broadcast_value(state, owner, "MPI_Bcast(current MarketState)");
        error.clear();
        if (state.book_id != book_id || state.exchange_time_ns != timestamp_ns) {
            error = "book owner broadcast a mismatched MarketState";
        }
        synchronize_error(error, "MarketState validation");
        return state;
    }

    void enqueue_broadcast_events(const std::vector<MultiAssetEvent>& generated,
                                  const char* context) {
        std::string error;
        try {
            for (const MultiAssetEvent& event : generated) {
                if (owner_for(event.key.book_id) == rank_) schedule_local(event);
            }
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, context);
    }

    std::vector<MultiAssetEvent> broadcast_generated(
        std::vector<MultiAssetEvent> owner_events,
        int owner,
        const char* context) {
        std::vector<MultiAssetEvent> generated = broadcast_vector(
            std::move(owner_events), owner, "MPI_Bcast(generated events)");
        enqueue_broadcast_events(generated, context);
        return generated;
    }

    void process_background(const MultiAssetEvent& event, int owner) {
        const MarketState state = current_state(
            event.key.book_id, event.key.timestamp_ns,
            "background pre-event MarketState");
        std::vector<MultiAssetEvent> generated;
        std::string error;
        if (rank_ == owner) {
            try {
                BookRuntime& runtime = local_book(event.key.book_id);
                OrderMessage order = runtime.background.make_order(
                    event.hawkes, state,
                    stable_sequence(event.key.origin_entity,
                                    event.key.origin_local_sequence));
                order.book_id = runtime.book_id;
                order.source_rank = 0;
                MultiAssetEvent arrival;
                arrival.key = MultiAssetEventKey{
                    order.arrival_time_ns,
                    MultiAssetEventPhase::OrderArrival,
                    runtime.book_id,
                    event.key.origin_entity,
                    event.key.origin_local_sequence,
                    0};
                arrival.kind = MultiAssetEventKind::OrderArrival;
                arrival.order = order;
                if (arrival.key.timestamp_ns <= end_time_ns_) {
                    generated.push_back(std::move(arrival));
                }
            } catch (...) {
                error = current_exception();
            }
        }
        synchronize_error(error, "background event transition");
        (void)broadcast_generated(std::move(generated), owner,
                                  "background generated-event enqueue");
    }

    std::vector<MultiAssetEvent> make_quote_events(const MultiAssetEvent& event,
                                                    const MarketState& state) {
        std::vector<MultiAssetEvent> generated;
        const std::vector<OrderMessage> quotes = shared_market_maker_->make_quotes(
            event.key.book_id, state, event.key.timestamp_ns);
        std::uint32_t child = 0;
        for (const OrderMessage& quote : quotes) {
            MultiAssetEvent arrival;
            arrival.key = MultiAssetEventKey{
                quote.arrival_time_ns,
                MultiAssetEventPhase::OrderArrival,
                event.key.book_id,
                event.key.origin_entity,
                event.key.origin_local_sequence,
                child++};
            arrival.kind = MultiAssetEventKind::OrderArrival;
            arrival.order = quote;
            if (arrival.key.timestamp_ns <= end_time_ns_) {
                generated.push_back(std::move(arrival));
            }
        }
        if (event.kind == MultiAssetEventKind::MarketMakerQuoteWake) {
            const std::int64_t next = checked_add(
                event.key.timestamp_ns,
                config_.market_maker_quote_interval_ns,
                "market-maker quote interval");
            if (next <= end_time_ns_) {
                MultiAssetEvent quote_wake;
                quote_wake.key = MultiAssetEventKey{
                    next,
                    MultiAssetEventPhase::AgentDecision,
                    event.key.book_id,
                    shared_market_maker_entity,
                    event.key.origin_local_sequence + 1,
                    0};
                quote_wake.kind = MultiAssetEventKind::MarketMakerQuoteWake;
                generated.push_back(std::move(quote_wake));
            }
        }
        return generated;
    }

    void process_quote(const MultiAssetEvent& event, int owner) {
        const MarketState state = current_state(
            event.key.book_id, event.key.timestamp_ns,
            "quote-decision MarketState");
        std::vector<MultiAssetEvent> replica_events;
        std::string error;
        try {
            replica_events = make_quote_events(event, state);
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "replicated market-maker quote transition");

        std::vector<MultiAssetEvent> authoritative = broadcast_vector(
            rank_ == owner ? replica_events : std::vector<MultiAssetEvent>{},
            owner, "MPI_Bcast(market-maker quote events)");
        error.clear();
        if (!event_vectors_equal(replica_events, authoritative)) {
            error = "replicated market-maker quote state diverged";
        }
        synchronize_error(error, "market-maker quote replication check");
        enqueue_broadcast_events(authoritative, "quote generated-event enqueue");
    }

    std::vector<MultiAssetEvent> make_arbitrage_events(
        const MultiAssetEvent& event,
        const std::vector<MarketState>& states) {
        std::vector<MultiAssetEvent> generated;
        const std::vector<OrderMessage> orders =
            etf_arbitrage_agent_->make_orders(
                states, event.key.timestamp_ns,
                event.key.origin_local_sequence);
        generated.reserve(orders.size() + 1U);
        std::uint32_t child = 0;
        for (const OrderMessage& order : orders) {
            MultiAssetEvent arrival;
            arrival.key = MultiAssetEventKey{
                order.arrival_time_ns,
                MultiAssetEventPhase::OrderArrival,
                order.book_id,
                etf_arbitrage_entity,
                event.key.origin_local_sequence,
                child++};
            arrival.kind = MultiAssetEventKind::ArbitrageOrderArrival;
            arrival.source_book_id = config_.etf_arbitrage.etf_book_id;
            arrival.order = order;
            if (arrival.key.timestamp_ns <= end_time_ns_) {
                generated.push_back(std::move(arrival));
            }
        }
        const std::int64_t next = checked_add(
            event.key.timestamp_ns,
            config_.etf_arbitrage.decision_interval_ns,
            "ETF-arbitrage decision interval");
        if (next <= end_time_ns_) {
            MultiAssetEvent wake;
            wake.key = MultiAssetEventKey{
                next,
                MultiAssetEventPhase::AgentDecision,
                config_.etf_arbitrage.etf_book_id,
                etf_arbitrage_entity,
                event.key.origin_local_sequence + 1,
                0};
            wake.kind = MultiAssetEventKind::EtfArbitrageWake;
            generated.push_back(std::move(wake));
        }
        return generated;
    }

    void process_arbitrage(const MultiAssetEvent& event, int owner) {
        std::vector<MarketState> states;
        states.reserve(resolved_book_configs_.size());
        for (std::size_t index = 0;
             index < resolved_book_configs_.size(); ++index) {
            states.push_back(current_state(
                static_cast<BookId>(index), event.key.timestamp_ns,
                "ETF-arbitrage MarketState"));
        }

        std::vector<MultiAssetEvent> replica_events;
        std::string error;
        try {
            replica_events = make_arbitrage_events(event, states);
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "replicated ETF-arbitrage transition");
        std::vector<MultiAssetEvent> authoritative = broadcast_vector(
            rank_ == owner ? replica_events : std::vector<MultiAssetEvent>{},
            owner, "MPI_Bcast(ETF-arbitrage events)");
        error.clear();
        if (!event_vectors_equal(replica_events, authoritative)) {
            error = "replicated ETF-arbitrage state diverged";
        }
        synchronize_error(error, "ETF-arbitrage replication check");
        enqueue_broadcast_events(authoritative,
                                 "ETF-arbitrage generated-event enqueue");
    }

    std::vector<MultiAssetEvent> make_value_events(
        const MultiAssetEvent& event,
        const MarketState& state) {
        FundamentalValueAgent& agent =
            *value_agents_[static_cast<std::size_t>(event.key.book_id)];
        const std::optional<OrderMessage> order = agent.make_order(
            state, event.key.timestamp_ns,
            event.key.origin_local_sequence);
        std::vector<MultiAssetEvent> generated;
        generated.reserve(2);
        if (order.has_value() && order->arrival_time_ns <= end_time_ns_) {
            MultiAssetEvent arrival;
            arrival.key = MultiAssetEventKey{
                order->arrival_time_ns,
                MultiAssetEventPhase::OrderArrival,
                event.key.book_id,
                fundamental_value_entity(event.key.book_id),
                event.key.origin_local_sequence,
                0};
            arrival.kind = MultiAssetEventKind::FundamentalValueOrderArrival;
            arrival.source_book_id = event.key.book_id;
            arrival.order = *order;
            generated.push_back(std::move(arrival));
        }
        const std::int64_t next = checked_add(
            event.key.timestamp_ns,
            config_.fundamental_value.decision_interval_ns,
            "fundamental-value decision interval");
        if (next <= end_time_ns_) {
            MultiAssetEvent wake;
            wake.key = MultiAssetEventKey{
                next,
                MultiAssetEventPhase::AgentDecision,
                event.key.book_id,
                fundamental_value_entity(event.key.book_id),
                event.key.origin_local_sequence + 1,
                0};
            wake.kind = MultiAssetEventKind::FundamentalValueWake;
            generated.push_back(std::move(wake));
        }
        return generated;
    }

    void process_value(const MultiAssetEvent& event, int owner) {
        const MarketState state = current_state(
            event.key.book_id, event.key.timestamp_ns,
            "fundamental-value decision MarketState");
        std::vector<MultiAssetEvent> replica_events;
        std::string error;
        try {
            replica_events = make_value_events(event, state);
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "replicated fundamental-value transition");

        error.clear();
        if (rank_ == owner) {
            try {
                local_book(event.key.book_id).fundamental_value_ticks =
                    value_agents_[static_cast<std::size_t>(event.key.book_id)]
                        ->fundamental_value_ticks();
            } catch (...) {
                error = current_exception();
            }
        }
        synchronize_error(error, "fundamental-value owner-state update");

        std::vector<MultiAssetEvent> authoritative = broadcast_vector(
            rank_ == owner ? replica_events : std::vector<MultiAssetEvent>{},
            owner, "MPI_Bcast(fundamental-value events)");
        error.clear();
        if (!event_vectors_equal(replica_events, authoritative)) {
            error = "replicated fundamental-value state diverged";
        }
        synchronize_error(error, "fundamental-value replication check");
        enqueue_broadcast_events(authoritative,
                                 "fundamental-value generated-event enqueue");
    }

    void process_order(const MultiAssetEvent& event,
                       int owner,
                       bool is_hedge) {
        std::vector<TradeExecution> local_trades;
        MarketState state;
        std::string error;
        if (rank_ == owner) {
            try {
                BookRuntime& runtime = local_book(event.order.book_id);
                runtime.recorder.observe_order(event.order);
                (void)runtime.lob.apply(event.order);
                ++runtime.submitted_orders;
                local_trades = runtime.lob.take_trades();
                (void)runtime.lob.take_reports();
                state = runtime.lob.state(event.key.timestamp_ns,
                                          runtime.fundamental_value_ticks);
            } catch (...) {
                error = current_exception();
            }
        }
        synchronize_error(error, "owning-rank LOB transition");
        state = broadcast_value(state, owner, "MPI_Bcast(post-order MarketState)");
        error.clear();
        if (state.book_id != event.key.book_id
            || state.exchange_time_ns != event.key.timestamp_ns) {
            error = "post-order MarketState does not match the processed event";
        }
        synchronize_error(error, "post-order MarketState validation");

        std::vector<TradeExecution> trades = broadcast_vector(
            std::move(local_trades), owner, "MPI_Bcast(canonical trade batch)");
        error.clear();
        try {
            BookRuntime* runtime = rank_ == owner
                ? &local_book(event.key.book_id) : nullptr;
            for (const TradeExecution& trade : trades) {
                if (trade.book_id != event.key.book_id
                    || trade.timestamp_ns != event.key.timestamp_ns
                    || trade.quantity <= 0 || trade.price_ticks <= 0) {
                    throw std::logic_error("owner broadcast a malformed trade execution");
                }
                combined_trade_hasher_.add(trade);
                if (runtime != nullptr) runtime->trade_hasher.add(trade);
                etf_arbitrage_agent_->on_trade(trade);
                value_agents_[static_cast<std::size_t>(trade.book_id)]
                    ->on_trade(trade);
            }
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "canonical trade stream update");

        std::vector<MultiAssetEvent> generated;
        if (rank_ == owner) {
            try {
                const int maker_owner = shared_market_maker_->logical_owner_id();
                std::uint32_t report_child = 0;
                std::optional<std::uint64_t> first_maker_trade_sequence;
                for (const TradeExecution& trade : trades) {
                    if (trade.buyer_owner_id != maker_owner
                        && trade.seller_owner_id != maker_owner) {
                        continue;
                    }
                    if (!first_maker_trade_sequence.has_value()) {
                        first_maker_trade_sequence = trade.trade_sequence;
                    }
                    MultiAssetEvent report;
                    report.key = MultiAssetEventKey{
                        checked_add(trade.timestamp_ns, config_.report_latency_ns,
                                    "report latency"),
                        MultiAssetEventPhase::ReportDelivery,
                        trade.book_id,
                        shared_market_maker_entity,
                        report_origin_sequence(trade),
                        report_child++};
                    report.kind = MultiAssetEventKind::ReportDelivery;
                    report.source_book_id = event.key.book_id;
                    report.trade = trade;
                    report.may_trigger_cross_book_reaction = !is_hedge;
                    generated.push_back(std::move(report));
                }
                if (first_maker_trade_sequence.has_value()) {
                    MultiAssetEvent repair;
                    repair.key = MultiAssetEventKey{
                        checked_add(event.key.timestamp_ns, config_.report_latency_ns,
                                    "market-maker repair latency"),
                        MultiAssetEventPhase::AgentDecision,
                        event.key.book_id,
                        market_maker_repair_entity(event.key.book_id),
                        *first_maker_trade_sequence,
                        0};
                    repair.kind = MultiAssetEventKind::MarketMakerRepairWake;
                    if (repair.key.timestamp_ns <= end_time_ns_) {
                        generated.push_back(std::move(repair));
                    }
                } else if (event.order.agent_kind != AgentKind::MarketMaker
                           && (state.best_bid_ticks <= 0
                               || state.best_ask_ticks
                                   <= state.best_bid_ticks)) {
                    MultiAssetEvent repair;
                    repair.key = MultiAssetEventKey{
                        checked_add(event.key.timestamp_ns,
                                    config_.report_latency_ns,
                                    "market-maker depleted-book repair latency"),
                        MultiAssetEventPhase::AgentDecision,
                        event.key.book_id,
                        market_maker_repair_entity(event.key.book_id),
                        stable_sequence(event.key.origin_entity,
                                        event.key.origin_local_sequence,
                                        event.key.child_index),
                        0};
                    repair.kind = MultiAssetEventKind::MarketMakerRepairWake;
                    if (repair.key.timestamp_ns <= end_time_ns_) {
                        generated.push_back(std::move(repair));
                    }
                }
                if (is_hedge) {
                    MultiAssetEvent completion;
                    completion.key = MultiAssetEventKey{
                        checked_add(event.key.timestamp_ns, config_.report_latency_ns,
                                    "hedge completion report latency"),
                        MultiAssetEventPhase::OrderCompletion,
                        event.order.book_id,
                        shared_market_maker_entity,
                        event.order.sequence,
                        0};
                    completion.kind = MultiAssetEventKind::HedgeOrderCompletion;
                    completion.source_book_id = event.source_book_id;
                    completion.order = event.order;
                    generated.push_back(std::move(completion));
                }
            } catch (...) {
                error = current_exception();
            }
        }
        synchronize_error(error, "trade report generation");
        (void)broadcast_generated(std::move(generated), owner,
                                  "trade report/completion enqueue");
    }

    std::vector<MultiAssetEvent> make_report_events(const MultiAssetEvent& event) {
        const std::int64_t expected_report_time = checked_add(
            event.trade.timestamp_ns, config_.report_latency_ns, "report latency");
        if (event.key.timestamp_ns != expected_report_time) {
            throw std::logic_error("trade report was delivered at the wrong causal time");
        }
        const std::vector<OrderMessage> hedges = shared_market_maker_->on_trade(
            event.trade, event.may_trigger_cross_book_reaction);
        std::vector<MultiAssetEvent> generated;
        generated.reserve(hedges.size());
        for (const OrderMessage& hedge : hedges) {
            MultiAssetEvent reaction;
            reaction.key = MultiAssetEventKey{
                hedge.generated_time_ns,
                MultiAssetEventPhase::CrossBookReaction,
                hedge.book_id,
                shared_market_maker_entity,
                hedge.sequence,
                0};
            reaction.kind = MultiAssetEventKind::CrossBookReaction;
            reaction.source_book_id = event.trade.book_id;
            reaction.order = hedge;
            if (reaction.key.timestamp_ns <= end_time_ns_) {
                generated.push_back(std::move(reaction));
            }
        }
        return generated;
    }

    void process_report(const MultiAssetEvent& event, int owner) {
        std::vector<MultiAssetEvent> replica_events;
        std::string error;
        try {
            replica_events = make_report_events(event);
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "replicated market-maker report transition");
        std::vector<MultiAssetEvent> authoritative = broadcast_vector(
            rank_ == owner ? replica_events : std::vector<MultiAssetEvent>{},
            owner, "MPI_Bcast(cross-book reaction events)");
        error.clear();
        if (!event_vectors_equal(replica_events, authoritative)) {
            error = "replicated market-maker report state diverged";
        }
        synchronize_error(error, "market-maker report replication check");
        enqueue_broadcast_events(authoritative, "cross-book reaction enqueue");
    }

    std::vector<MultiAssetEvent> make_reaction_events(
        const MultiAssetEvent& event) const {
        if (event.source_book_id == event.order.book_id) {
            throw std::logic_error("a cross-book reaction cannot target its source book");
        }
        if (event.order.generated_time_ns != event.key.timestamp_ns
            || event.order.arrival_time_ns <= event.order.generated_time_ns) {
            throw std::logic_error("cross-book hedge violates positive causal latency");
        }
        std::vector<MultiAssetEvent> generated;
        MultiAssetEvent arrival;
        arrival.key = MultiAssetEventKey{
            event.order.arrival_time_ns,
            MultiAssetEventPhase::OrderArrival,
            event.order.book_id,
            shared_market_maker_entity,
            event.order.sequence,
            0};
        arrival.kind = MultiAssetEventKind::HedgeOrderArrival;
        arrival.source_book_id = event.source_book_id;
        arrival.order = event.order;
        if (arrival.key.timestamp_ns <= end_time_ns_) {
            generated.push_back(std::move(arrival));
        }
        return generated;
    }

    void process_reaction(const MultiAssetEvent& event, int owner) {
        std::vector<MultiAssetEvent> replica_events;
        std::string error;
        try {
            replica_events = make_reaction_events(event);
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "replicated cross-book reaction transition");
        std::vector<MultiAssetEvent> authoritative = broadcast_vector(
            rank_ == owner ? replica_events : std::vector<MultiAssetEvent>{},
            owner, "MPI_Bcast(hedge arrival events)");
        error.clear();
        if (!event_vectors_equal(replica_events, authoritative)) {
            error = "replicated cross-book reaction state diverged";
        }
        synchronize_error(error, "cross-book reaction replication check");
        enqueue_broadcast_events(authoritative, "hedge arrival enqueue");
    }

    void process_completion(const MultiAssetEvent& event, int owner) {
        int replica_result = 0;
        std::string error;
        try {
            replica_result = shared_market_maker_->complete_order(
                event.order.sequence) ? 1 : 0;
        } catch (...) {
            error = current_exception();
        }
        synchronize_error(error, "replicated hedge completion transition");
        int authoritative = rank_ == owner ? replica_result : 0;
        check_mpi(MPI_Bcast(&authoritative, 1, MPI_INT, owner, communicator_),
                  "MPI_Bcast(hedge completion result)");
        error.clear();
        if (replica_result != authoritative) {
            error = "replicated hedge completion state diverged";
        }
        synchronize_error(error, "hedge completion replication check");
    }

    void process_sample(const MultiAssetEvent& event, int owner) {
        MarketState state;
        std::string error;
        if (rank_ == owner) {
            try {
                BookRuntime& runtime = local_book(event.key.book_id);
                state = runtime.lob.state(event.key.timestamp_ns,
                                          runtime.fundamental_value_ticks);
                runtime.recorder.observe_state(state);
            } catch (...) {
                error = current_exception();
            }
        }
        synchronize_error(error, "owning-rank state observation");
        state = broadcast_value(state, owner, "MPI_Bcast(observed MarketState)");
        error.clear();
        if (state.book_id != event.key.book_id
            || state.exchange_time_ns != event.key.timestamp_ns) {
            error = "observed MarketState does not match the sample event";
        }
        synchronize_error(error, "observed MarketState validation");
    }

    void process_global_event(const MultiAssetEvent& event) {
        if (!initialized_) {
            throw std::logic_error("exact MPI simulator was not initialized");
        }
        const int owner = owner_for(event.key.book_id);
        ++processed_events_;
        if (rank_ == owner) ++local_book(event.key.book_id).processed_events;

        switch (event.kind) {
            case MultiAssetEventKind::BackgroundWake:
                process_background(event, owner);
                break;
            case MultiAssetEventKind::MarketMakerQuoteWake:
            case MultiAssetEventKind::MarketMakerRepairWake:
                process_quote(event, owner);
                break;
            case MultiAssetEventKind::OrderArrival:
                process_order(event, owner, false);
                break;
            case MultiAssetEventKind::HedgeOrderArrival:
                ++hedge_order_events_;
                process_order(event, owner, true);
                break;
            case MultiAssetEventKind::LiquidityShockOrderArrival:
                ++liquidity_shock_events_;
                process_order(event, owner, false);
                break;
            case MultiAssetEventKind::EtfArbitrageWake:
                ++arbitrage_decision_events_;
                process_arbitrage(event, owner);
                break;
            case MultiAssetEventKind::ArbitrageOrderArrival:
                ++arbitrage_order_events_;
                process_order(event, owner, false);
                break;
            case MultiAssetEventKind::FundamentalValueWake:
                ++value_decision_events_;
                process_value(event, owner);
                break;
            case MultiAssetEventKind::FundamentalValueOrderArrival:
                ++value_order_events_;
                process_order(event, owner, false);
                break;
            case MultiAssetEventKind::ReportDelivery:
                process_report(event, owner);
                break;
            case MultiAssetEventKind::CrossBookReaction:
                ++cross_book_reaction_events_;
                process_reaction(event, owner);
                break;
            case MultiAssetEventKind::HedgeOrderCompletion:
                process_completion(event, owner);
                break;
            case MultiAssetEventKind::SampleState:
                process_sample(event, owner);
                break;
        }
    }

    void verify_replicated_state() const {
        ReplicaWire local;
        local.combined_trade_count = combined_trade_hasher_.trade_count();
        local.combined_trade_hash = combined_trade_hasher_.digest();
        local.processed_events = processed_events_;
        local.cross_book_reaction_events = cross_book_reaction_events_;
        local.hedge_order_events = hedge_order_events_;
        local.liquidity_shock_events = liquidity_shock_events_;
        local.arbitrage_decision_events = arbitrage_decision_events_;
        local.arbitrage_order_events = arbitrage_order_events_;
        local.value_decision_events = value_decision_events_;
        local.value_order_events = value_order_events_;
        local.market_maker_cash_ticks = shared_market_maker_->total_cash_ticks();
        local.arbitrage_cash_ticks = etf_arbitrage_agent_->total_cash_ticks();

        std::vector<ReplicaWire> replicas;
        if (rank_ == 0) replicas.resize(static_cast<std::size_t>(world_size_));
        check_mpi(MPI_Gather(&local,
                             checked_byte_count(1, sizeof(local), "replica state"),
                             MPI_BYTE,
                             rank_ == 0 ? replicas.data() : nullptr,
                             checked_byte_count(1, sizeof(local), "replica state"),
                             MPI_BYTE, 0, communicator_),
                  "MPI_Gather(replicated model state)");
        std::string error;
        if (rank_ == 0) {
            for (int candidate = 1; candidate < world_size_; ++candidate) {
                const ReplicaWire& other = replicas[static_cast<std::size_t>(candidate)];
                if (other.combined_trade_count != replicas[0].combined_trade_count
                    || other.combined_trade_hash != replicas[0].combined_trade_hash
                    || other.processed_events != replicas[0].processed_events
                    || other.cross_book_reaction_events
                        != replicas[0].cross_book_reaction_events
                    || other.hedge_order_events != replicas[0].hedge_order_events
                    || other.liquidity_shock_events
                        != replicas[0].liquidity_shock_events
                    || other.arbitrage_decision_events
                        != replicas[0].arbitrage_decision_events
                    || other.arbitrage_order_events
                        != replicas[0].arbitrage_order_events
                    || other.value_decision_events
                        != replicas[0].value_decision_events
                    || other.value_order_events
                        != replicas[0].value_order_events
                    || other.market_maker_cash_ticks
                        != replicas[0].market_maker_cash_ticks
                    || other.arbitrage_cash_ticks
                        != replicas[0].arbitrage_cash_ticks) {
                    error = "replicated global hash/counter/cash state differs";
                    break;
                }
            }
        }
        synchronize_error(error, "replicated global-state check");

        for (int index = 0; index < config_.book_count; ++index) {
            const BookId book_id = static_cast<BookId>(index);
            const std::array<std::int64_t, 2> maker_state{{
                shared_market_maker_->inventory(book_id),
                shared_market_maker_->cash_ticks(book_id)}};
            std::vector<std::int64_t> gathered;
            if (rank_ == 0) {
                gathered.resize(static_cast<std::size_t>(world_size_) * 2U);
            }
            check_mpi(MPI_Gather(maker_state.data(), 2, MPI_LONG_LONG,
                                 rank_ == 0 ? gathered.data() : nullptr,
                                 2, MPI_LONG_LONG, 0, communicator_),
                      "MPI_Gather(replicated maker book state)");
            error.clear();
            if (rank_ == 0) {
                for (int candidate = 1; candidate < world_size_; ++candidate) {
                    const auto offset = static_cast<std::size_t>(candidate) * 2U;
                    if (gathered[offset] != gathered[0]
                        || gathered[offset + 1U] != gathered[1]) {
                        error = "replicated market-maker inventory/cash differs for book "
                            + std::to_string(book_id);
                        break;
                    }
                }
            }
            synchronize_error(error, "replicated market-maker state check");

            const std::array<std::int64_t, 2> arbitrage_state{{
                etf_arbitrage_agent_->inventory(book_id),
                etf_arbitrage_agent_->cash_ticks(book_id)}};
            gathered.clear();
            if (rank_ == 0) {
                gathered.resize(static_cast<std::size_t>(world_size_) * 2U);
            }
            check_mpi(MPI_Gather(arbitrage_state.data(), 2, MPI_LONG_LONG,
                                 rank_ == 0 ? gathered.data() : nullptr,
                                 2, MPI_LONG_LONG, 0, communicator_),
                      "MPI_Gather(replicated arbitrage book state)");
            error.clear();
            if (rank_ == 0) {
                for (int candidate = 1; candidate < world_size_; ++candidate) {
                    const auto offset = static_cast<std::size_t>(candidate) * 2U;
                    if (gathered[offset] != gathered[0]
                        || gathered[offset + 1U] != gathered[1]) {
                        error = "replicated arbitrage inventory/cash differs for book "
                            + std::to_string(book_id);
                        break;
                    }
                }
            }
            synchronize_error(error, "replicated arbitrage state check");

            const FundamentalValueAgent& value_agent =
                *value_agents_[static_cast<std::size_t>(book_id)];
            const std::array<std::uint64_t, 3> value_state{{
                static_cast<std::uint64_t>(value_agent.inventory()),
                static_cast<std::uint64_t>(value_agent.cash_ticks()),
                std::bit_cast<std::uint64_t>(
                    value_agent.fundamental_value_ticks())}};
            std::vector<std::uint64_t> gathered_value;
            if (rank_ == 0) {
                gathered_value.resize(static_cast<std::size_t>(world_size_) * 3U);
            }
            check_mpi(MPI_Gather(value_state.data(), 3, MPI_UNSIGNED_LONG_LONG,
                                 rank_ == 0 ? gathered_value.data() : nullptr,
                                 3, MPI_UNSIGNED_LONG_LONG, 0, communicator_),
                      "MPI_Gather(replicated value-agent state)");
            error.clear();
            if (rank_ == 0) {
                for (int candidate = 1; candidate < world_size_; ++candidate) {
                    const auto offset = static_cast<std::size_t>(candidate) * 3U;
                    if (gathered_value[offset] != gathered_value[0]
                        || gathered_value[offset + 1U] != gathered_value[1]
                        || gathered_value[offset + 2U] != gathered_value[2]) {
                        error = "replicated value-agent state differs for book "
                            + std::to_string(book_id);
                        break;
                    }
                }
            }
            synchronize_error(error, "replicated value-agent state check");
        }
    }

    SequentialMultiAssetResult gather_result(double wall_seconds) {
        verify_replicated_state();
        SequentialMultiAssetResult result;
        result.books.reserve(static_cast<std::size_t>(config_.book_count));

        for (int index = 0; index < config_.book_count; ++index) {
            const BookId book_id = static_cast<BookId>(index);
            const int owner = owner_for(book_id);
            BookSummaryWire wire;
            calibration::SimulationRecord owner_record;
            std::string error;
            if (rank_ == owner) {
                try {
                    const BookRuntime& runtime = local_book(book_id);
                    owner_record = runtime.recorder.finalize();
                    wire.book_id = book_id;
                    wire.final_state = runtime.lob.state(
                        end_time_ns_, runtime.fundamental_value_ticks);
                    wire.market_maker_inventory = shared_market_maker_->inventory(book_id);
                    wire.market_maker_cash_ticks = shared_market_maker_->cash_ticks(book_id);
                    wire.arbitrage_inventory = etf_arbitrage_agent_->inventory(book_id);
                    wire.arbitrage_cash_ticks = etf_arbitrage_agent_->cash_ticks(book_id);
                    const FundamentalValueAgent& value_agent =
                        *value_agents_[static_cast<std::size_t>(book_id)];
                    wire.value_agent_inventory = value_agent.inventory();
                    wire.value_agent_cash_ticks = value_agent.cash_ticks();
                    wire.final_fundamental_value_ticks =
                        value_agent.fundamental_value_ticks();
                    wire.processed_events = runtime.processed_events;
                    wire.submitted_orders = runtime.submitted_orders;
                    wire.trade_count = runtime.trade_hasher.trade_count();
                    wire.trade_hash = runtime.trade_hasher.digest();
                    wire.event_counts = owner_record.event_counts;
                    wire.owner_cancel_messages = owner_record.owner_cancel_messages;
                    wire.market = owner_record.market;
                } catch (...) {
                    error = current_exception();
                }
            }
            synchronize_error(error, "per-book result finalization");
            wire = broadcast_value(wire, owner, "MPI_Bcast(per-book result)");
            if (wire.book_id != book_id || wire.final_state.book_id != book_id
                || wire.final_state.exchange_time_ns != end_time_ns_) {
                error = "owner broadcast a mismatched final book summary";
            }
            synchronize_error(error, "per-book result validation");

            MultiAssetBookSummary summary;
            summary.book_id = book_id;
            summary.symbol = resolved_book_configs_[
                static_cast<std::size_t>(book_id)].symbol;
            summary.final_state = wire.final_state;
            summary.market_maker_inventory = wire.market_maker_inventory;
            summary.market_maker_cash_ticks = static_cast<double>(
                wire.market_maker_cash_ticks);
            summary.arbitrage_inventory = wire.arbitrage_inventory;
            summary.arbitrage_cash_ticks = wire.arbitrage_cash_ticks;
            summary.value_agent_inventory = wire.value_agent_inventory;
            summary.value_agent_cash_ticks = static_cast<double>(
                wire.value_agent_cash_ticks);
            summary.final_fundamental_value_ticks =
                wire.final_fundamental_value_ticks;
            summary.processed_events = wire.processed_events;
            summary.submitted_orders = wire.submitted_orders;
            summary.trade_count = wire.trade_count;
            summary.trade_hash = wire.trade_hash;
            summary.expected_sample_count = static_cast<std::uint64_t>(
                end_time_ns_ / config_.sample_interval_ns);
            summary.calibration_record.event_counts = wire.event_counts;
            summary.calibration_record.owner_cancel_messages = wire.owner_cancel_messages;
            summary.calibration_record.market = wire.market;

            for (std::size_t bucket = 0;
                 bucket < calibration::empirical_event_bucket_count; ++bucket) {
                std::vector<int> samples = rank_ == owner
                    ? owner_record.quantity_samples[bucket] : std::vector<int>{};
                samples = broadcast_vector(std::move(samples), owner,
                                           "MPI_Bcast(recorder reservoir)");
                summary.calibration_record.quantity_samples[bucket] = std::move(samples);
            }
            std::vector<MarketState> state_trace = rank_ == owner
                ? owner_record.state_trace : std::vector<MarketState>{};
            summary.calibration_record.state_trace = broadcast_vector(
                std::move(state_trace), owner,
                "MPI_Bcast(recorder state trace)");
            summary.structurally_valid =
                summary.calibration_record.market.snapshots
                    == summary.expected_sample_count
                && summary.final_state.best_bid_ticks > 0
                && summary.final_state.best_ask_ticks
                    > summary.final_state.best_bid_ticks;
            result.books.push_back(std::move(summary));
        }

        result.structurally_valid = std::all_of(
            result.books.begin(), result.books.end(),
            [](const MultiAssetBookSummary& summary) {
                return summary.structurally_valid;
            });

        result.combined_trade_count = combined_trade_hasher_.trade_count();
        result.combined_trade_hash = combined_trade_hasher_.digest();
        result.processed_events = processed_events_;
        result.cross_book_reaction_events = cross_book_reaction_events_;
        result.hedge_order_events = hedge_order_events_;
        result.liquidity_shock_events = liquidity_shock_events_;
        result.arbitrage_decision_events = arbitrage_decision_events_;
        result.arbitrage_order_events = arbitrage_order_events_;
        result.value_decision_events = value_decision_events_;
        result.value_order_events = value_order_events_;
        result.market_maker_cash_ticks = static_cast<double>(
            shared_market_maker_->total_cash_ticks());
        result.arbitrage_cash_ticks = etf_arbitrage_agent_->total_cash_ticks();
        result.wall_seconds = wall_seconds;
        result.summary_csv = (std::filesystem::path(config_.output_dir)
                              / "exact_mpi_multi_asset_summary.csv").string();
        return result;
    }

    void write_summary_csv(const SequentialMultiAssetResult& result) const {
        const std::filesystem::path output_path(result.summary_csv);
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream output(output_path);
        if (!output) {
            throw std::runtime_error("cannot write exact MPI multi-asset summary: "
                                     + output_path.string());
        }
        output << "book_id,symbol,owner_rank,processed_events,submitted_orders,trade_count,trade_hash,"
                  "best_bid_ticks,best_ask_ticks,best_bid_depth,best_ask_depth,"
                  "last_trade_price_ticks,market_maker_inventory,market_maker_cash_ticks,"
                  "arbitrage_inventory,arbitrage_cash_ticks,"
                  "value_agent_inventory,value_agent_cash_ticks,final_fundamental_value_ticks,"
                  "sample_count,expected_sample_count,structurally_valid,"
                  "mean_spread_ticks,mean_bid_depth,mean_ask_depth,"
                  "mid_move_rate,return_variance,return_kurtosis,absolute_return_acf1,"
                  "limit_buy_events,limit_sell_events,market_buy_events,market_sell_events,"
                  "cancel_bid_events,cancel_ask_events,owner_cancel_messages,"
                  "combined_trade_count,combined_trade_hash,total_processed_events,"
                  "cross_book_reaction_events,hedge_order_events,liquidity_shock_events,"
                  "arbitrage_decision_events,arbitrage_order_events,"
                  "value_decision_events,value_order_events,"
                  "mpi_ranks,wall_seconds\n";
        output << std::setprecision(17);
        for (const MultiAssetBookSummary& book : result.books) {
            const MarketState& state = book.final_state;
            const calibration::SimulationRecord& record = book.calibration_record;
            const calibration::MarketFeatureSummary& market = record.market;
            output << book.book_id << ','
                   << book.symbol << ','
                   << owner_for(book.book_id) << ','
                   << book.processed_events << ','
                   << book.submitted_orders << ','
                   << book.trade_count << ','
                   << book.trade_hash << ','
                   << state.best_bid_ticks << ','
                   << state.best_ask_ticks << ','
                   << state.best_bid_depth << ','
                   << state.best_ask_depth << ','
                   << state.last_trade_price_ticks << ','
                   << book.market_maker_inventory << ','
                   << book.market_maker_cash_ticks << ','
                   << book.arbitrage_inventory << ','
                   << book.arbitrage_cash_ticks << ','
                   << book.value_agent_inventory << ','
                   << book.value_agent_cash_ticks << ','
                   << book.final_fundamental_value_ticks << ','
                   << market.snapshots << ','
                   << book.expected_sample_count << ','
                   << (book.structurally_valid ? 1 : 0) << ','
                   << market.mean_spread_ticks << ','
                   << market.mean_bid_depth << ','
                   << market.mean_ask_depth << ','
                   << market.mid_move_rate << ','
                   << market.return_variance << ','
                   << market.return_kurtosis << ','
                   << market.absolute_return_acf1 << ',';
            for (std::uint64_t count : record.event_counts) output << count << ',';
            output << record.owner_cancel_messages << ','
                   << result.combined_trade_count << ','
                   << result.combined_trade_hash << ','
                   << result.processed_events << ','
                   << result.cross_book_reaction_events << ','
                   << result.hedge_order_events << ','
                   << result.liquidity_shock_events << ','
                   << result.arbitrage_decision_events << ','
                   << result.arbitrage_order_events << ','
                   << result.value_decision_events << ','
                   << result.value_order_events << ','
                   << world_size_ << ','
                   << result.wall_seconds << '\n';
        }

        const std::filesystem::path trace_path = output_path.parent_path()
            / "exact_mpi_multi_asset_state_trace.csv";
        std::ofstream trace(trace_path);
        if (!trace) {
            throw std::runtime_error("cannot write exact MPI state trace: "
                                     + trace_path.string());
        }
        trace << "book_id,symbol,exchange_time_ns,best_bid_ticks,best_ask_ticks,"
                 "best_bid_depth,best_ask_depth,last_trade_price_ticks,mid_price_ticks,"
                 "fundamental_value_ticks,cumulative_aggressive_buy,"
                 "cumulative_aggressive_sell\n";
        trace << std::setprecision(17);
        for (const MultiAssetBookSummary& book : result.books) {
            for (const MarketState& state : book.calibration_record.state_trace) {
                trace << book.book_id << ',' << book.symbol << ','
                      << state.exchange_time_ns << ','
                      << state.best_bid_ticks << ',' << state.best_ask_ticks << ','
                      << state.best_bid_depth << ',' << state.best_ask_depth << ','
                      << state.last_trade_price_ticks << ',' << state.mid_price_ticks << ','
                      << state.fundamental_value_ticks << ','
                      << state.cumulative_aggressive_buy << ','
                      << state.cumulative_aggressive_sell << '\n';
            }
        }
    }

    ExactMpiMultiAssetResult finish(double wall_seconds) {
        ExactMpiMultiAssetResult output;
        output.rank = rank_;
        output.world_size = world_size_;
        output.local_book_count = static_cast<int>(local_books_.size());
        output.model = gather_result(wall_seconds);

        std::string error;
        if (rank_ == 0) {
            try {
                write_summary_csv(output.model);
            } catch (...) {
                error = current_exception();
            }
        }
        synchronize_error(error, "exact MPI summary output");
        return output;
    }

    MPI_Comm communicator_;
    SequentialMultiAssetConfig config_;
    int rank_ = 0;
    int world_size_ = 1;
    std::int64_t end_time_ns_ = 0;
    bool initialized_ = false;
    bool completed_ = false;
    std::map<BookId, std::unique_ptr<BookRuntime>> local_books_;
    std::vector<MultiAssetBookConfig> resolved_book_configs_;
    std::unique_ptr<SharedMarketMakerAgent> shared_market_maker_;
    std::unique_ptr<EtfArbitrageAgent> etf_arbitrage_agent_;
    std::vector<std::unique_ptr<FundamentalValueAgent>> value_agents_;
    EventQueue events_;
    std::set<MultiAssetEventKey> queued_keys_;
    TradeTapeHasher combined_trade_hasher_;
    std::uint64_t processed_events_ = 0;
    std::uint64_t cross_book_reaction_events_ = 0;
    std::uint64_t hedge_order_events_ = 0;
    std::uint64_t liquidity_shock_events_ = 0;
    std::uint64_t arbitrage_decision_events_ = 0;
    std::uint64_t arbitrage_order_events_ = 0;
    std::uint64_t value_decision_events_ = 0;
    std::uint64_t value_order_events_ = 0;
    std::optional<MultiAssetEventKey> last_processed_key_;
};

ExactMpiMultiAssetSimulator::ExactMpiMultiAssetSimulator(
    MPI_Comm communicator,
    SequentialMultiAssetConfig config)
    : impl_(std::make_unique<Impl>(communicator, std::move(config))) {}

ExactMpiMultiAssetSimulator::~ExactMpiMultiAssetSimulator() = default;

ExactMpiMultiAssetResult ExactMpiMultiAssetSimulator::run() {
    return impl_->run();
}

int ExactMpiMultiAssetSimulator::owner_rank(BookId book_id, int world_size) {
    if (world_size <= 0) {
        throw std::invalid_argument("MPI world size must be positive");
    }
    return static_cast<int>(book_id % static_cast<BookId>(world_size));
}

} // namespace dlob
