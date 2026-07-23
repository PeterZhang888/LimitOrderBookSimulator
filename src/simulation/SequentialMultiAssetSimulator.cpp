#include "simulation/SequentialMultiAssetSimulator.hpp"

#include "agents/EtfArbitrageAgent.hpp"
#include "agents/FundamentalValueAgent.hpp"
#include "agents/SharedMarketMakerAgent.hpp"
#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace dlob {
namespace {

constexpr std::int64_t nanoseconds_per_second = 1'000'000'000LL;

[[nodiscard]] std::int64_t checked_duration_ns(int duration_seconds) {
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

[[nodiscard]] std::int64_t checked_add(std::int64_t time_ns,
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

[[nodiscard]] std::uint64_t report_origin_sequence(const TradeExecution& trade) {
    return stable_sequence(background_entity(trade.book_id), trade.trade_sequence);
}

} // namespace

SequentialMultiAssetSimulator::SequentialMultiAssetSimulator(
    SequentialMultiAssetConfig config)
    : config_(std::move(config)) {}

SequentialMultiAssetSimulator::~SequentialMultiAssetSimulator() = default;

void SequentialMultiAssetSimulator::initialize() {
    if (initialized_) return;
    if (config_.book_count < 1) {
        throw std::invalid_argument("--books must be at least 1");
    }
    if (config_.fundamental_value.enabled && config_.book_count > 100'000) {
        throw std::invalid_argument(
            "value-agent owner identifiers support at most 100000 books");
    }
    if (static_cast<std::uint64_t>(config_.book_count)
        > static_cast<std::uint64_t>(std::numeric_limits<BookId>::max())) {
        throw std::invalid_argument("--books exceeds the BookId range");
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
        throw std::invalid_argument("invalid sequential multi-asset configuration");
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
                "invalid sequential liquidity-shock configuration");
        }
    }
    resolved_book_configs_ = resolve_multi_asset_book_configs(config_);
    books_.reserve(static_cast<std::size_t>(config_.book_count));
    value_agents_.reserve(static_cast<std::size_t>(config_.book_count));
    for (int index = 0; index < config_.book_count; ++index) {
        const BookId id = static_cast<BookId>(index);
        const MultiAssetBookConfig& book_config =
            resolved_book_configs_[static_cast<std::size_t>(index)];
        BackgroundHawkesConfig background = make_multi_asset_background_config(
            config_, book_config, id);
        books_.emplace_back(id, book_config.symbol,
                            book_config.fundamental_price_ticks,
                            background, config_.tick_size,
                            stable_sequence(sampler_entity(id), config_.seed));
        if (book_config.initial_best_bid_ticks > 0) {
            books_.back().lob.seed_calibrated_book(
                book_config.initial_best_bid_ticks,
                book_config.initial_best_ask_ticks,
                book_config.initial_best_bid_depth,
                book_config.initial_best_ask_depth,
                config_.initial_depth_scale);
        } else {
            books_.back().lob.seed_default_book(config_.initial_depth_scale);
        }
        value_agents_.push_back(std::make_unique<FundamentalValueAgent>(
            config_.fundamental_value, id,
            book_config.fundamental_price_ticks, config_.seed));
    }

    shared_market_maker_ = std::make_unique<SharedMarketMakerAgent>(
        make_multi_asset_market_maker_config(config_, resolved_book_configs_));
    etf_arbitrage_agent_ = std::make_unique<EtfArbitrageAgent>(
        config_.etf_arbitrage, resolved_book_configs_);
    initialized_ = true;

    for (BookRuntime& runtime : books_) {
        schedule_background_events(runtime);
        schedule_samples(runtime.book_id);
        schedule_next_quote(runtime.book_id, 0, 1);
        if (config_.fundamental_value.enabled) {
            schedule_next_value_decision(
                runtime.book_id,
                config_.fundamental_value.decision_interval_ns, 1);
        }
    }
    schedule_liquidity_shock();
    if (config_.etf_arbitrage.enabled) {
        schedule_next_arbitrage(config_.etf_arbitrage.decision_interval_ns, 1);
    }
}

BookRuntime& SequentialMultiAssetSimulator::book(BookId book_id) {
    const auto index = static_cast<std::size_t>(book_id);
    if (index >= books_.size() || books_[index].book_id != book_id) {
        throw std::out_of_range("event targets an unknown book");
    }
    return books_[index];
}

const BookRuntime& SequentialMultiAssetSimulator::book(BookId book_id) const {
    const auto index = static_cast<std::size_t>(book_id);
    if (index >= books_.size() || books_[index].book_id != book_id) {
        throw std::out_of_range("event targets an unknown book");
    }
    return books_[index];
}

void SequentialMultiAssetSimulator::schedule(MultiAssetEvent event) {
    if (event.key.timestamp_ns < 0) {
        throw std::logic_error("cannot schedule an event before simulation start");
    }
    (void)book(event.key.book_id);
    if ((event.kind == MultiAssetEventKind::OrderArrival
         || event.kind == MultiAssetEventKind::HedgeOrderArrival
         || event.kind == MultiAssetEventKind::LiquidityShockOrderArrival
         || event.kind == MultiAssetEventKind::ArbitrageOrderArrival
         || event.kind == MultiAssetEventKind::FundamentalValueOrderArrival)
        && event.order.book_id != event.key.book_id) {
        throw std::logic_error("order payload and event key target different books");
    }
    events_.push(std::move(event));
}

void SequentialMultiAssetSimulator::schedule_next_value_decision(
    BookId book_id,
    std::int64_t timestamp_ns,
    std::uint64_t decision_sequence) {
    if (!config_.fundamental_value.enabled || timestamp_ns > end_time_ns_) return;
    MultiAssetEvent event;
    event.key = MultiAssetEventKey{
        timestamp_ns,
        MultiAssetEventPhase::AgentDecision,
        book_id,
        fundamental_value_entity(book_id),
        decision_sequence,
        0};
    event.kind = MultiAssetEventKind::FundamentalValueWake;
    schedule(std::move(event));
}

void SequentialMultiAssetSimulator::schedule_next_arbitrage(
    std::int64_t timestamp_ns,
    std::uint64_t decision_sequence) {
    if (!config_.etf_arbitrage.enabled || timestamp_ns > end_time_ns_) return;
    MultiAssetEvent event;
    event.key = MultiAssetEventKey{
        timestamp_ns,
        MultiAssetEventPhase::AgentDecision,
        config_.etf_arbitrage.etf_book_id,
        etf_arbitrage_entity,
        decision_sequence,
        0};
    event.kind = MultiAssetEventKind::EtfArbitrageWake;
    schedule(std::move(event));
}

void SequentialMultiAssetSimulator::schedule_background_events(BookRuntime& runtime) {
    runtime.background_events = runtime.background.simulate(0, end_time_ns_);
    runtime.next_background_event_index = 0;
    schedule_next_background_event(runtime);
}

void SequentialMultiAssetSimulator::schedule_next_background_event(
    BookRuntime& runtime) {
    if (runtime.next_background_event_index >= runtime.background_events.size()) return;
    const std::size_t index = runtime.next_background_event_index++;
    const HawkesEvent& hawkes = runtime.background_events[index];
    MultiAssetEvent event;
    event.key = MultiAssetEventKey{
        hawkes.time_ns,
        MultiAssetEventPhase::ExogenousWake,
        runtime.book_id,
        background_entity(runtime.book_id),
        static_cast<std::uint64_t>(index) + 1U,
        0};
    event.kind = MultiAssetEventKind::BackgroundWake;
    event.hawkes = hawkes;
    schedule(std::move(event));
}

void SequentialMultiAssetSimulator::schedule_samples(BookId book_id) {
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
        schedule(std::move(event));
        ++sequence;
        if (end_time_ns_ - timestamp_ns < config_.sample_interval_ns) break;
        timestamp_ns += config_.sample_interval_ns;
    }
}

void SequentialMultiAssetSimulator::schedule_liquidity_shock() {
    if (!config_.liquidity_shock.has_value()) return;
    const LiquidityShockConfig& shock = *config_.liquidity_shock;
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
    schedule(std::move(event));
}

void SequentialMultiAssetSimulator::schedule_next_quote(
    BookId book_id,
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
    schedule(std::move(event));
}

void SequentialMultiAssetSimulator::capture_trades(
    BookRuntime& runtime,
    const MultiAssetEvent& cause,
    bool may_trigger_cross_book_reaction) {
    std::vector<TradeExecution> trades = runtime.lob.take_trades();
    const int maker_owner = shared_market_maker_->logical_owner_id();
    std::uint32_t report_child = 0;
    std::optional<std::uint64_t> first_maker_trade_sequence;
    for (const TradeExecution& trade : trades) {
        runtime.trade_hasher.add(trade);
        combined_trade_hasher_.add(trade);
        etf_arbitrage_agent_->on_trade(trade);
        value_agents_[static_cast<std::size_t>(runtime.book_id)]->on_trade(trade);

        if (trade.buyer_owner_id != maker_owner && trade.seller_owner_id != maker_owner) {
            continue;
        }
        if (!first_maker_trade_sequence.has_value()) {
            first_maker_trade_sequence = trade.trade_sequence;
        }
        MultiAssetEvent report;
        report.key = MultiAssetEventKey{
            checked_add(trade.timestamp_ns, config_.report_latency_ns, "report latency"),
            MultiAssetEventPhase::ReportDelivery,
            trade.book_id,
            shared_market_maker_entity,
            report_origin_sequence(trade),
            report_child++};
        report.kind = MultiAssetEventKind::ReportDelivery;
        report.source_book_id = cause.key.book_id;
        report.trade = trade;
        report.may_trigger_cross_book_reaction = may_trigger_cross_book_reaction;
        schedule(std::move(report));
    }

    if (first_maker_trade_sequence.has_value()) {
        MultiAssetEvent repair;
        repair.key = MultiAssetEventKey{
            checked_add(cause.key.timestamp_ns, config_.report_latency_ns,
                        "market-maker repair latency"),
            MultiAssetEventPhase::AgentDecision,
            runtime.book_id,
            market_maker_repair_entity(runtime.book_id),
            *first_maker_trade_sequence,
            0};
        repair.kind = MultiAssetEventKind::MarketMakerRepairWake;
        if (repair.key.timestamp_ns <= end_time_ns_) schedule(std::move(repair));
    } else if (cause.order.agent_kind != AgentKind::MarketMaker) {
        const MarketState state = runtime.lob.state(
            cause.key.timestamp_ns, runtime.fundamental_value_ticks);
        if (state.best_bid_ticks <= 0
            || state.best_ask_ticks <= state.best_bid_ticks) {
            MultiAssetEvent repair;
            repair.key = MultiAssetEventKey{
                checked_add(cause.key.timestamp_ns, config_.report_latency_ns,
                            "market-maker depleted-book repair latency"),
                MultiAssetEventPhase::AgentDecision,
                runtime.book_id,
                market_maker_repair_entity(runtime.book_id),
                stable_sequence(cause.key.origin_entity,
                                cause.key.origin_local_sequence,
                                cause.key.child_index),
                0};
            repair.kind = MultiAssetEventKind::MarketMakerRepairWake;
            if (repair.key.timestamp_ns <= end_time_ns_) {
                schedule(std::move(repair));
            }
        }
    }

    // AgentReport remains the compatibility channel for the original local
    // agents.  The shared market maker consumes canonical per-match executions,
    // so drain these reports after each atomic order transition.
    (void)runtime.lob.take_reports();
}

void SequentialMultiAssetSimulator::apply_order(const MultiAssetEvent& event,
                                                 bool is_hedge) {
    BookRuntime& runtime = book(event.order.book_id);
    runtime.recorder.observe_order(event.order);
    (void)runtime.lob.apply(event.order);
    ++runtime.submitted_orders;
    capture_trades(runtime, event, !is_hedge);
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
        schedule(std::move(completion));
    }
}

void SequentialMultiAssetSimulator::process_event(const MultiAssetEvent& event) {
    if (!initialized_) {
        throw std::logic_error("initialize the multi-asset simulator before processing events");
    }
    if (event.key.timestamp_ns > end_time_ns_) {
        throw std::logic_error("attempted to process an event beyond the configured horizon");
    }
    if (last_processed_key_.has_value() && event.key <= *last_processed_key_) {
        throw std::logic_error("multi-asset event keys must be unique and globally increasing");
    }
    last_processed_key_ = event.key;

    BookRuntime& runtime = book(event.key.book_id);
    ++processed_events_;
    ++runtime.processed_events;

    switch (event.kind) {
        case MultiAssetEventKind::BackgroundWake: {
            const MarketState state = runtime.lob.state(
                event.key.timestamp_ns, runtime.fundamental_value_ticks);
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
            if (arrival.key.timestamp_ns <= end_time_ns_) schedule(std::move(arrival));
            schedule_next_background_event(runtime);
            break;
        }

        case MultiAssetEventKind::MarketMakerQuoteWake:
        case MultiAssetEventKind::MarketMakerRepairWake: {
            const MarketState state = runtime.lob.state(
                event.key.timestamp_ns, runtime.fundamental_value_ticks);
            const std::vector<OrderMessage> quotes = shared_market_maker_->make_quotes(
                runtime.book_id, state, event.key.timestamp_ns);
            std::uint32_t child = 0;
            for (const OrderMessage& quote : quotes) {
                MultiAssetEvent arrival;
                arrival.key = MultiAssetEventKey{
                    quote.arrival_time_ns,
                    MultiAssetEventPhase::OrderArrival,
                    runtime.book_id,
                    event.key.origin_entity,
                    event.key.origin_local_sequence,
                    child++};
                arrival.kind = MultiAssetEventKind::OrderArrival;
                arrival.order = quote;
                if (arrival.key.timestamp_ns <= end_time_ns_) schedule(std::move(arrival));
            }
            if (event.kind == MultiAssetEventKind::MarketMakerQuoteWake) {
                const std::int64_t next = checked_add(
                    event.key.timestamp_ns,
                    config_.market_maker_quote_interval_ns,
                    "market-maker quote interval");
                schedule_next_quote(runtime.book_id, next,
                                    event.key.origin_local_sequence + 1);
            }
            break;
        }

        case MultiAssetEventKind::OrderArrival:
            apply_order(event, false);
            break;

        case MultiAssetEventKind::HedgeOrderArrival:
            ++hedge_order_events_;
            apply_order(event, true);
            break;

        case MultiAssetEventKind::LiquidityShockOrderArrival:
            ++liquidity_shock_events_;
            apply_order(event, false);
            break;

        case MultiAssetEventKind::EtfArbitrageWake: {
            ++arbitrage_decision_events_;
            std::vector<MarketState> states;
            states.reserve(books_.size());
            for (const BookRuntime& current : books_) {
                states.push_back(current.lob.state(
                    event.key.timestamp_ns, current.fundamental_value_ticks));
            }
            const std::vector<OrderMessage> orders =
                etf_arbitrage_agent_->make_orders(
                    states, event.key.timestamp_ns,
                    event.key.origin_local_sequence);
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
                    schedule(std::move(arrival));
                }
            }
            const std::int64_t next = checked_add(
                event.key.timestamp_ns,
                config_.etf_arbitrage.decision_interval_ns,
                "ETF-arbitrage decision interval");
            schedule_next_arbitrage(next,
                                    event.key.origin_local_sequence + 1);
            break;
        }

        case MultiAssetEventKind::ArbitrageOrderArrival:
            ++arbitrage_order_events_;
            apply_order(event, false);
            break;

        case MultiAssetEventKind::FundamentalValueWake: {
            ++value_decision_events_;
            FundamentalValueAgent& agent =
                *value_agents_[static_cast<std::size_t>(runtime.book_id)];
            const MarketState state = runtime.lob.state(
                event.key.timestamp_ns, runtime.fundamental_value_ticks);
            const std::optional<OrderMessage> order = agent.make_order(
                state, event.key.timestamp_ns,
                event.key.origin_local_sequence);
            runtime.fundamental_value_ticks = agent.fundamental_value_ticks();
            if (order.has_value()) {
                MultiAssetEvent arrival;
                arrival.key = MultiAssetEventKey{
                    order->arrival_time_ns,
                    MultiAssetEventPhase::OrderArrival,
                    runtime.book_id,
                    fundamental_value_entity(runtime.book_id),
                    event.key.origin_local_sequence,
                    0};
                arrival.kind = MultiAssetEventKind::FundamentalValueOrderArrival;
                arrival.source_book_id = runtime.book_id;
                arrival.order = *order;
                if (arrival.key.timestamp_ns <= end_time_ns_) {
                    schedule(std::move(arrival));
                }
            }
            const std::int64_t next = checked_add(
                event.key.timestamp_ns,
                config_.fundamental_value.decision_interval_ns,
                "fundamental-value decision interval");
            schedule_next_value_decision(
                runtime.book_id, next,
                event.key.origin_local_sequence + 1);
            break;
        }

        case MultiAssetEventKind::FundamentalValueOrderArrival:
            ++value_order_events_;
            apply_order(event, false);
            break;

        case MultiAssetEventKind::ReportDelivery: {
            const std::int64_t expected_report_time = checked_add(
                event.trade.timestamp_ns, config_.report_latency_ns, "report latency");
            if (event.key.timestamp_ns != expected_report_time) {
                throw std::logic_error("trade report was delivered at the wrong causal time");
            }
            const std::vector<OrderMessage> hedges = shared_market_maker_->on_trade(
                event.trade, event.may_trigger_cross_book_reaction);
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
                    schedule(std::move(reaction));
                }
            }
            break;
        }

        case MultiAssetEventKind::CrossBookReaction: {
            ++cross_book_reaction_events_;
            if (event.source_book_id == event.order.book_id) {
                throw std::logic_error("a cross-book reaction cannot target its source book");
            }
            if (event.order.generated_time_ns != event.key.timestamp_ns
                || event.order.arrival_time_ns <= event.order.generated_time_ns) {
                throw std::logic_error("cross-book hedge violates positive causal latency");
            }
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
            if (arrival.key.timestamp_ns <= end_time_ns_) schedule(std::move(arrival));
            break;
        }

        case MultiAssetEventKind::HedgeOrderCompletion:
            (void)shared_market_maker_->complete_order(event.order.sequence);
            break;

        case MultiAssetEventKind::SampleState:
            runtime.recorder.observe_state(runtime.lob.state(
                event.key.timestamp_ns, runtime.fundamental_value_ticks));
            break;
    }
}

SequentialMultiAssetResult SequentialMultiAssetSimulator::run() {
    if (completed_) {
        throw std::logic_error("a SequentialMultiAssetSimulator instance can only run once");
    }
    const auto wall_start = std::chrono::steady_clock::now();
    initialize();

    while (!events_.empty() && events_.top().key.timestamp_ns <= end_time_ns_) {
        MultiAssetEvent event = events_.top();
        events_.pop();
        process_event(event);
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    completed_ = true;
    return finish(wall_seconds);
}

SequentialMultiAssetResult SequentialMultiAssetSimulator::finish(double wall_seconds) {
    SequentialMultiAssetResult result;
    result.books.reserve(books_.size());
    for (const BookRuntime& runtime : books_) {
        MultiAssetBookSummary summary;
        summary.book_id = runtime.book_id;
        summary.symbol = runtime.symbol;
        summary.final_state = runtime.lob.state(
            end_time_ns_, runtime.fundamental_value_ticks);
        summary.market_maker_inventory = shared_market_maker_->inventory(runtime.book_id);
        summary.market_maker_cash_ticks = static_cast<double>(
            shared_market_maker_->cash_ticks(runtime.book_id));
        summary.arbitrage_inventory = etf_arbitrage_agent_->inventory(
            runtime.book_id);
        summary.arbitrage_cash_ticks = etf_arbitrage_agent_->cash_ticks(
            runtime.book_id);
        const FundamentalValueAgent& value_agent =
            *value_agents_[static_cast<std::size_t>(runtime.book_id)];
        summary.value_agent_inventory = value_agent.inventory();
        summary.value_agent_cash_ticks = static_cast<double>(
            value_agent.cash_ticks());
        summary.final_fundamental_value_ticks =
            value_agent.fundamental_value_ticks();
        summary.processed_events = runtime.processed_events;
        summary.submitted_orders = runtime.submitted_orders;
        summary.trade_count = runtime.trade_hasher.trade_count();
        summary.trade_hash = runtime.trade_hasher.digest();
        summary.calibration_record = runtime.recorder.finalize();
        summary.expected_sample_count = static_cast<std::uint64_t>(
            end_time_ns_ / config_.sample_interval_ns);
        summary.structurally_valid =
            summary.calibration_record.market.snapshots
                == summary.expected_sample_count
            && summary.final_state.best_bid_ticks > 0
            && summary.final_state.best_ask_ticks
                > summary.final_state.best_bid_ticks;
        result.books.push_back(summary);
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
                          / "sequential_multi_asset_summary.csv").string();
    write_summary_csv(result);
    return result;
}

void SequentialMultiAssetSimulator::write_summary_csv(
    const SequentialMultiAssetResult& result) const {
    const std::filesystem::path output_path(result.summary_csv);
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("cannot write multi-asset summary: "
                                 + output_path.string());
    }
    output << "book_id,symbol,processed_events,submitted_orders,trade_count,trade_hash,"
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
              "wall_seconds\n";
    output << std::setprecision(17);
    for (const MultiAssetBookSummary& book_summary : result.books) {
        const MarketState& state = book_summary.final_state;
        const calibration::SimulationRecord& record = book_summary.calibration_record;
        const calibration::MarketFeatureSummary& market = record.market;
        output << book_summary.book_id << ','
               << book_summary.symbol << ','
               << book_summary.processed_events << ','
               << book_summary.submitted_orders << ','
               << book_summary.trade_count << ','
               << book_summary.trade_hash << ','
               << state.best_bid_ticks << ','
               << state.best_ask_ticks << ','
               << state.best_bid_depth << ','
               << state.best_ask_depth << ','
               << state.last_trade_price_ticks << ','
               << book_summary.market_maker_inventory << ','
               << book_summary.market_maker_cash_ticks << ','
               << book_summary.arbitrage_inventory << ','
               << book_summary.arbitrage_cash_ticks << ','
               << book_summary.value_agent_inventory << ','
               << book_summary.value_agent_cash_ticks << ','
               << book_summary.final_fundamental_value_ticks << ','
               << market.snapshots << ','
               << book_summary.expected_sample_count << ','
               << (book_summary.structurally_valid ? 1 : 0) << ','
               << market.mean_spread_ticks << ','
               << market.mean_bid_depth << ','
               << market.mean_ask_depth << ','
               << market.mid_move_rate << ','
               << market.return_variance << ','
               << market.return_kurtosis << ','
               << market.absolute_return_acf1 << ',';
        for (const std::uint64_t count : record.event_counts) output << count << ',';
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
               << result.wall_seconds << '\n';
    }

    const std::filesystem::path trace_path = output_path.parent_path()
        / "sequential_multi_asset_state_trace.csv";
    std::ofstream trace(trace_path);
    if (!trace) {
        throw std::runtime_error("cannot write multi-asset state trace: "
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

} // namespace dlob
