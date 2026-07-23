#include "simulation/BatchedMpiMultiAssetSimulator.hpp"

#include "agents/EtfArbitrageAgent.hpp"
#include "agents/SharedMarketMakerAgent.hpp"
#include "common/TradeTapeHasher.hpp"
#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dlob {
namespace {

constexpr std::int64_t nanoseconds_per_second = 1'000'000'000LL;

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

std::int64_t duration_ns(int seconds) {
    if (seconds <= 0
        || static_cast<std::int64_t>(seconds)
            > std::numeric_limits<std::int64_t>::max() / nanoseconds_per_second) {
        throw std::invalid_argument("batched duration must be positive and representable");
    }
    return static_cast<std::int64_t>(seconds) * nanoseconds_per_second;
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

bool pending_order_less(const OrderMessage& left, const OrderMessage& right) {
    if (left.arrival_time_ns != right.arrival_time_ns) {
        return left.arrival_time_ns < right.arrival_time_ns;
    }
    if (action_priority(left.action) != action_priority(right.action)) {
        return action_priority(left.action) < action_priority(right.action);
    }
    if (left.agent_kind != right.agent_kind) return left.agent_kind < right.agent_kind;
    if (left.sequence != right.sequence) return left.sequence < right.sequence;
    return left.tie_breaker < right.tie_breaker;
}

bool trade_less(const TradeExecution& left, const TradeExecution& right) {
    if (left.timestamp_ns != right.timestamp_ns) {
        return left.timestamp_ns < right.timestamp_ns;
    }
    if (left.book_id != right.book_id) return left.book_id < right.book_id;
    return left.trade_sequence < right.trade_sequence;
}

bool valid_two_sided_state(const MarketState& state) {
    return state.best_bid_ticks > 0
        && state.best_ask_ticks > state.best_bid_ticks
        && std::isfinite(state.mid_price_ticks)
        && state.mid_price_ticks > 0.0;
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

struct BookResultWire {
    MarketState state{};
    std::uint64_t processed_orders = 0;
    std::uint64_t trade_count = 0;
    std::uint64_t trade_hash = TradeTapeHasher::offset_basis;
};

static_assert(std::is_trivially_copyable_v<BookResultWire>);

struct LocalBook {
    BookId id = 0;
    MultiAssetBookConfig config;
    BackgroundHawkesAgent background;
    DistributedLimitOrderBook lob;
    std::vector<HawkesEvent> background_events;
    std::size_t next_background = 0;
    std::vector<OrderMessage> pending_orders;
    TradeTapeHasher trade_hasher;
    std::uint64_t processed_orders = 0;

    LocalBook(BookId book_id,
              MultiAssetBookConfig book_config,
              const BackgroundHawkesConfig& background_config,
              int tick_size)
        : id(book_id),
          config(std::move(book_config)),
          background(background_config),
          lob(tick_size, book_id) {}
};

} // namespace

class BatchedMpiMultiAssetSimulator::Impl {
public:
    Impl(MPI_Comm communicator,
         SequentialMultiAssetConfig config,
         std::int64_t window_ns)
        : communicator_(communicator),
          config_(std::move(config)),
          books_(resolve_multi_asset_book_configs(config_)),
          window_ns_(window_ns),
          end_time_ns_(duration_ns(config_.duration_seconds)) {
        check_mpi(MPI_Comm_rank(communicator_, &rank_), "MPI_Comm_rank");
        check_mpi(MPI_Comm_size(communicator_, &world_size_), "MPI_Comm_size");
        if (world_size_ <= 0 || window_ns_ <= 0 || window_ns_ > end_time_ns_) {
            throw std::invalid_argument("invalid batched MPI rank count or window");
        }
        if (books_.size() != static_cast<std::size_t>(config_.book_count)) {
            throw std::logic_error("resolved batched book count differs from configuration");
        }
        local_index_by_book_.assign(books_.size(), -1);
        if (rank_ == 0) {
            SharedMarketMakerConfig maker_config =
                make_multi_asset_market_maker_config(config_, books_);
            // This benchmark batches observations and quote refreshes.  It does
            // not schedule sub-window reactive hedges, which would require the
            // conservative event protocol being benchmarked separately.
            maker_config.enable_cross_book_hedging = false;
            market_maker_ = std::make_unique<SharedMarketMakerAgent>(
                std::move(maker_config));
            arbitrage_ = std::make_unique<EtfArbitrageAgent>(
                config_.etf_arbitrage, books_);
        }
    }

    BatchedMpiMultiAssetResult run() {
        check_mpi(MPI_Barrier(communicator_), "MPI_Barrier(start)");
        const double total_start = MPI_Wtime();

        const double initialization_start = MPI_Wtime();
        initialize_local_books();
        const double local_initialization = MPI_Wtime() - initialization_start;

        // Initial quotes are based on the calibrated opening state.  They are
        // timestamped just after t=0 and processed with the first local window.
        boundary_exchange(0, true, 1);

        std::uint64_t boundary_sequence = 2;
        for (std::int64_t start = 0; start < end_time_ns_;) {
            const std::int64_t end = std::min(end_time_ns_, start + window_ns_);
            const double compute_start = MPI_Wtime();
            process_window(start, end);
            compute_seconds_ += MPI_Wtime() - compute_start;
            const bool create_orders = end < end_time_ns_;
            boundary_exchange(end, create_orders, boundary_sequence++);
            ++window_count_;
            start = end;
        }

        const std::vector<BookResultWire> gathered_results = gather_book_results();
        const std::uint64_t state_hash = compute_state_hash(gathered_results);
        check_mpi(MPI_Barrier(communicator_), "MPI_Barrier(finish)");
        const double local_wall = MPI_Wtime() - total_start;
        return reduce_result(local_wall, local_initialization, state_hash);
    }

private:
    void initialize_local_books() {
        local_books_.clear();
        for (std::size_t index = 0; index < books_.size(); ++index) {
            const BookId id = static_cast<BookId>(index);
            if (owner_rank(id, world_size_) != rank_) continue;
            BackgroundHawkesConfig background_config =
                make_multi_asset_background_config(config_, books_[index], id);
            auto local = std::make_unique<LocalBook>(
                id, books_[index], background_config, config_.tick_size);
            if (local->config.initial_best_bid_ticks > 0) {
                local->lob.seed_calibrated_book(
                    local->config.initial_best_bid_ticks,
                    local->config.initial_best_ask_ticks,
                    local->config.initial_best_bid_depth,
                    local->config.initial_best_ask_depth,
                    config_.initial_depth_scale);
            } else {
                local->lob.seed_default_book(config_.initial_depth_scale);
            }
            local->background_events = local->background.simulate(0, end_time_ns_);
            local_index_by_book_[index] = static_cast<int>(local_books_.size());
            local_books_.push_back(std::move(local));
        }
    }

    void process_window(std::int64_t start_ns, std::int64_t end_ns) {
        for (const std::unique_ptr<LocalBook>& pointer : local_books_) {
            LocalBook& book = *pointer;
            std::sort(book.pending_orders.begin(), book.pending_orders.end(),
                      pending_order_less);
            std::size_t pending_index = 0;
            while (true) {
                const bool has_pending = pending_index < book.pending_orders.size()
                    && book.pending_orders[pending_index].arrival_time_ns < end_ns;
                const bool has_background =
                    book.next_background < book.background_events.size()
                    && book.background_events[book.next_background].time_ns < end_ns;
                if (!has_pending && !has_background) break;

                const std::int64_t pending_time = has_pending
                    ? book.pending_orders[pending_index].arrival_time_ns
                    : std::numeric_limits<std::int64_t>::max();
                const std::int64_t background_time = has_background
                    ? book.background_events[book.next_background].time_ns
                    : std::numeric_limits<std::int64_t>::max();

                OrderMessage order;
                if (pending_time <= background_time) {
                    order = book.pending_orders[pending_index++];
                } else {
                    const HawkesEvent event =
                        book.background_events[book.next_background];
                    const std::uint64_t local_sequence =
                        static_cast<std::uint64_t>(book.next_background) + 1U;
                    ++book.next_background;
                    order = book.background.make_order(
                        event,
                        book.lob.state(event.time_ns,
                                       book.config.fundamental_price_ticks),
                        stable_sequence(background_entity(book.id), local_sequence));
                    order.book_id = book.id;
                }
                if (order.arrival_time_ns < start_ns) {
                    order.arrival_time_ns = start_ns;
                }
                book.lob.apply(order);
                ++book.processed_orders;
            }
            if (pending_index > 0) {
                book.pending_orders.erase(
                    book.pending_orders.begin(),
                    book.pending_orders.begin()
                        + static_cast<std::ptrdiff_t>(pending_index));
            }

            std::vector<TradeExecution> trades = book.lob.take_trades();
            for (const TradeExecution& trade : trades) book.trade_hasher.add(trade);
            local_trades_.insert(local_trades_.end(), trades.begin(), trades.end());
            (void)book.lob.take_reports();
        }
    }

    template <typename Value>
    std::vector<Value> gather_variable(const std::vector<Value>& local,
                                       const char* label) {
        static_assert(std::is_trivially_copyable_v<Value>);
        const int local_bytes = checked_bytes(local.size(), sizeof(Value), label);
        std::vector<int> counts(rank_ == 0
                                    ? static_cast<std::size_t>(world_size_) : 0U);
        const double start = MPI_Wtime();
        check_mpi(MPI_Gather(&local_bytes, 1, MPI_INT,
                             rank_ == 0 ? counts.data() : nullptr,
                             1, MPI_INT, 0, communicator_),
                  "MPI_Gather(variable byte counts)");

        std::vector<int> displacements;
        int total_bytes = 0;
        if (rank_ == 0) {
            displacements.resize(static_cast<std::size_t>(world_size_));
            for (int index = 0; index < world_size_; ++index) {
                if (counts[static_cast<std::size_t>(index)] < 0
                    || total_bytes > std::numeric_limits<int>::max()
                        - counts[static_cast<std::size_t>(index)]) {
                    throw std::overflow_error("gathered byte count overflows int");
                }
                displacements[static_cast<std::size_t>(index)] = total_bytes;
                total_bytes += counts[static_cast<std::size_t>(index)];
            }
            if (total_bytes % static_cast<int>(sizeof(Value)) != 0) {
                throw std::logic_error("gathered bytes do not form complete values");
            }
        }
        std::vector<Value> gathered(
            rank_ == 0
                ? static_cast<std::size_t>(total_bytes)
                    / static_cast<std::size_t>(sizeof(Value))
                : 0U);
        check_mpi(MPI_Gatherv(local.empty() ? nullptr : local.data(),
                              local_bytes, MPI_BYTE,
                              rank_ == 0 && !gathered.empty()
                                  ? gathered.data() : nullptr,
                              rank_ == 0 ? counts.data() : nullptr,
                              rank_ == 0 ? displacements.data() : nullptr,
                              MPI_BYTE, 0, communicator_),
                  "MPI_Gatherv(variable values)");
        communication_seconds_ += MPI_Wtime() - start;
        return gathered;
    }

    std::vector<MarketState> gather_states(std::int64_t time_ns) {
        std::vector<MarketState> local;
        local.reserve(local_books_.size());
        for (const std::unique_ptr<LocalBook>& book : local_books_) {
            local.push_back(book->lob.state(time_ns,
                                            book->config.fundamental_price_ticks));
        }
        const std::vector<MarketState> gathered = gather_variable(
            local, "market-state batch");
        if (rank_ != 0) return {};
        std::vector<MarketState> states(books_.size());
        std::vector<bool> seen(books_.size(), false);
        for (const MarketState& state : gathered) {
            if (state.book_id >= books_.size()
                || seen[static_cast<std::size_t>(state.book_id)]) {
                throw std::logic_error("invalid or duplicate gathered book state");
            }
            states[static_cast<std::size_t>(state.book_id)] = state;
            seen[static_cast<std::size_t>(state.book_id)] = true;
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
            throw std::logic_error("a gathered book state is missing");
        }
        return states;
    }

    std::vector<OrderMessage> make_controller_orders(
        const std::vector<MarketState>& states,
        const std::vector<TradeExecution>& trades,
        std::int64_t decision_time_ns,
        bool create_orders,
        std::uint64_t decision_sequence) {
        if (rank_ != 0) return {};
        const double start = MPI_Wtime();
        for (const TradeExecution& trade : trades) {
            (void)market_maker_->on_trade(trade, false);
            arbitrage_->on_trade(trade);
        }
        std::vector<OrderMessage> orders;
        if (create_orders) {
            for (const MarketState& state : states) {
                std::vector<OrderMessage> quotes = market_maker_->make_quotes(
                    state.book_id, state, decision_time_ns);
                market_maker_order_count_ += quotes.size();
                orders.insert(orders.end(), quotes.begin(), quotes.end());
            }
            if (last_valid_states_.empty()) last_valid_states_ = states;
            std::vector<MarketState> arbitrage_states = states;
            for (std::size_t index = 0; index < states.size(); ++index) {
                if (valid_two_sided_state(states[index])) {
                    last_valid_states_[index] = states[index];
                } else {
                    if (!valid_two_sided_state(last_valid_states_[index])) {
                        throw std::logic_error(
                            "ETF basket has no valid initial state for one book");
                    }
                    arbitrage_states[index] = last_valid_states_[index];
                    ++stale_snapshot_uses_;
                }
            }
            std::vector<OrderMessage> arbitrage_orders = arbitrage_->make_orders(
                arbitrage_states, decision_time_ns, decision_sequence);
            arbitrage_order_count_ += arbitrage_orders.size();
            orders.insert(orders.end(), arbitrage_orders.begin(),
                          arbitrage_orders.end());
        }
        controller_seconds_ += MPI_Wtime() - start;
        return orders;
    }

    void broadcast_orders(std::vector<OrderMessage>& orders) {
        std::uint64_t count = rank_ == 0
            ? static_cast<std::uint64_t>(orders.size()) : 0U;
        const double start = MPI_Wtime();
        check_mpi(MPI_Bcast(&count, 1, MPI_UNSIGNED_LONG_LONG,
                            0, communicator_),
                  "MPI_Bcast(order count)");
        if (count > static_cast<std::uint64_t>(
                std::numeric_limits<int>::max() / sizeof(OrderMessage))) {
            throw std::overflow_error("batched order broadcast is too large");
        }
        if (rank_ != 0) orders.resize(static_cast<std::size_t>(count));
        const int bytes = checked_bytes(orders.size(), sizeof(OrderMessage),
                                        "order batch");
        check_mpi(MPI_Bcast(orders.empty() ? nullptr : orders.data(),
                            bytes, MPI_BYTE, 0, communicator_),
                  "MPI_Bcast(order batch)");
        communication_seconds_ += MPI_Wtime() - start;
    }

    void assign_orders(const std::vector<OrderMessage>& orders) {
        for (const OrderMessage& order : orders) {
            if (order.book_id >= books_.size()) {
                throw std::logic_error("controller generated an invalid target book");
            }
            const int local_index =
                local_index_by_book_[static_cast<std::size_t>(order.book_id)];
            if (local_index >= 0) {
                local_books_[static_cast<std::size_t>(local_index)]
                    ->pending_orders.push_back(order);
            }
        }
    }

    void boundary_exchange(std::int64_t time_ns,
                           bool create_orders,
                           std::uint64_t decision_sequence) {
        std::vector<MarketState> states = gather_states(time_ns);
        std::vector<TradeExecution> trades = gather_variable(
            local_trades_, "trade batch");
        local_trades_.clear();
        if (rank_ == 0) {
            std::sort(trades.begin(), trades.end(), trade_less);
        }
        std::vector<OrderMessage> orders = make_controller_orders(
            states, trades, time_ns, create_orders, decision_sequence);
        if (create_orders) {
            broadcast_orders(orders);
            assign_orders(orders);
        }
        if (rank_ == 0) last_states_ = std::move(states);
    }

    std::vector<BookResultWire> gather_book_results() {
        std::vector<BookResultWire> local;
        local.reserve(local_books_.size());
        for (const std::unique_ptr<LocalBook>& book : local_books_) {
            BookResultWire wire;
            wire.state = book->lob.state(end_time_ns_,
                                         book->config.fundamental_price_ticks);
            wire.processed_orders = book->processed_orders;
            wire.trade_count = book->trade_hasher.trade_count();
            wire.trade_hash = book->trade_hasher.digest();
            local.push_back(wire);
        }
        return gather_variable(local, "book-result batch");
    }

    std::uint64_t compute_state_hash(std::vector<BookResultWire> results) const {
        if (rank_ != 0) return 0;
        std::sort(results.begin(), results.end(),
                  [](const BookResultWire& left, const BookResultWire& right) {
                      return left.state.book_id < right.state.book_id;
                  });
        if (results.size() != books_.size()) {
            throw std::logic_error("final book-result batch is incomplete");
        }
        std::uint64_t hash = TradeTapeHasher::offset_basis;
        for (const BookResultWire& result : results) {
            const MarketState& state = result.state;
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
            hash_integer(hash, result.processed_orders);
            hash_integer(hash, result.trade_count);
            hash_integer(hash, result.trade_hash);
            hash_integer(hash, market_maker_->inventory(state.book_id));
            hash_integer(hash, market_maker_->cash_ticks(state.book_id));
            hash_integer(hash, arbitrage_->inventory(state.book_id));
            hash_integer(hash, arbitrage_->cash_ticks(state.book_id));
        }
        return hash;
    }

    BatchedMpiMultiAssetResult reduce_result(double local_wall,
                                              double local_initialization,
                                              std::uint64_t state_hash) {
        std::uint64_t local_orders = 0;
        std::uint64_t local_trades = 0;
        for (const std::unique_ptr<LocalBook>& book : local_books_) {
            local_orders += book->processed_orders;
            local_trades += book->trade_hasher.trade_count();
        }
        std::uint64_t total_orders = 0;
        std::uint64_t total_trades = 0;
        double max_wall = 0.0;
        double max_initialization = 0.0;
        double max_compute = 0.0;
        double max_communication = 0.0;
        check_mpi(MPI_Reduce(&local_orders, &total_orders, 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator_),
                  "MPI_Reduce(processed orders)");
        check_mpi(MPI_Reduce(&local_trades, &total_trades, 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator_),
                  "MPI_Reduce(trades)");
        check_mpi(MPI_Reduce(&local_wall, &max_wall, 1, MPI_DOUBLE,
                             MPI_MAX, 0, communicator_),
                  "MPI_Reduce(wall time)");
        check_mpi(MPI_Reduce(&local_initialization, &max_initialization, 1,
                             MPI_DOUBLE, MPI_MAX, 0, communicator_),
                  "MPI_Reduce(initialization time)");
        check_mpi(MPI_Reduce(&compute_seconds_, &max_compute, 1, MPI_DOUBLE,
                             MPI_MAX, 0, communicator_),
                  "MPI_Reduce(compute time)");
        check_mpi(MPI_Reduce(&communication_seconds_, &max_communication, 1,
                             MPI_DOUBLE, MPI_MAX, 0, communicator_),
                  "MPI_Reduce(communication time)");

        const double local_load = static_cast<double>(local_orders);
        double total_load = 0.0;
        double maximum_load = 0.0;
        check_mpi(MPI_Reduce(&local_load, &total_load, 1, MPI_DOUBLE,
                             MPI_SUM, 0, communicator_),
                  "MPI_Reduce(total load)");
        check_mpi(MPI_Reduce(&local_load, &maximum_load, 1, MPI_DOUBLE,
                             MPI_MAX, 0, communicator_),
                  "MPI_Reduce(maximum load)");

        BatchedMpiMultiAssetResult result;
        result.rank = rank_;
        result.world_size = world_size_;
        result.book_count = static_cast<int>(books_.size());
        result.local_book_count = static_cast<int>(local_books_.size());
        result.windows = window_count_;
        if (rank_ == 0) {
            result.processed_orders = total_orders;
            result.trades = total_trades;
            result.market_maker_orders = market_maker_order_count_;
            result.arbitrage_orders = arbitrage_order_count_;
            result.stale_snapshot_uses = stale_snapshot_uses_;
            result.state_hash = state_hash;
            result.wall_seconds = max_wall;
            result.initialization_seconds = max_initialization;
            result.max_compute_seconds = max_compute;
            result.max_communication_seconds = max_communication;
            result.controller_seconds = controller_seconds_;
            const double average_load = total_load / static_cast<double>(world_size_);
            result.load_imbalance = average_load > 0.0
                ? maximum_load / average_load : 1.0;
        }
        return result;
    }

    MPI_Comm communicator_ = MPI_COMM_NULL;
    SequentialMultiAssetConfig config_;
    std::vector<MultiAssetBookConfig> books_;
    std::int64_t window_ns_ = 0;
    std::int64_t end_time_ns_ = 0;
    int rank_ = 0;
    int world_size_ = 1;
    std::vector<std::unique_ptr<LocalBook>> local_books_;
    std::vector<int> local_index_by_book_;
    std::vector<TradeExecution> local_trades_;
    std::vector<MarketState> last_states_;
    std::vector<MarketState> last_valid_states_;
    std::unique_ptr<SharedMarketMakerAgent> market_maker_;
    std::unique_ptr<EtfArbitrageAgent> arbitrage_;
    std::uint64_t window_count_ = 0;
    std::uint64_t market_maker_order_count_ = 0;
    std::uint64_t arbitrage_order_count_ = 0;
    std::uint64_t stale_snapshot_uses_ = 0;
    double compute_seconds_ = 0.0;
    double communication_seconds_ = 0.0;
    double controller_seconds_ = 0.0;
};

BatchedMpiMultiAssetSimulator::BatchedMpiMultiAssetSimulator(
    MPI_Comm communicator,
    SequentialMultiAssetConfig config,
    std::int64_t window_ns)
    : impl_(std::make_unique<Impl>(communicator, std::move(config), window_ns)) {}

BatchedMpiMultiAssetSimulator::~BatchedMpiMultiAssetSimulator() = default;

BatchedMpiMultiAssetResult BatchedMpiMultiAssetSimulator::run() {
    return impl_->run();
}

int BatchedMpiMultiAssetSimulator::owner_rank(BookId book_id, int world_size) {
    if (world_size <= 0) throw std::invalid_argument("world_size must be positive");
    return static_cast<int>(book_id % static_cast<BookId>(world_size));
}

} // namespace dlob
