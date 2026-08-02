#pragma once

#include "simulation/MultiAssetTypes.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <queue>
#include <vector>

namespace dlob {

class SharedMarketMakerAgent;
class EtfArbitrageAgent;
class FundamentalValueAgent;

class SequentialMultiAssetSimulator {
public:
    explicit SequentialMultiAssetSimulator(SequentialMultiAssetConfig config);
    ~SequentialMultiAssetSimulator();

    SequentialMultiAssetSimulator(const SequentialMultiAssetSimulator&) = delete;
    SequentialMultiAssetSimulator& operator=(const SequentialMultiAssetSimulator&) = delete;
    SequentialMultiAssetSimulator(SequentialMultiAssetSimulator&&) = delete;
    SequentialMultiAssetSimulator& operator=(SequentialMultiAssetSimulator&&) = delete;

    // initialize() is public so an MPI executor can construct exactly the same
    // model state before assigning books to ranks.  run() calls it automatically.
    void initialize();

    // The one authoritative state-transition function.  A future exact MPI
    // implementation should select the same minimum key and invoke this core.
    void process_event(const MultiAssetEvent& event);

    [[nodiscard]] SequentialMultiAssetResult run();
    [[nodiscard]] const std::vector<BookRuntime>& books() const noexcept { return books_; }
    [[nodiscard]] std::size_t pending_event_count() const noexcept { return events_.size(); }

private:
    using EventQueue = std::priority_queue<MultiAssetEvent,
                                           std::vector<MultiAssetEvent>,
                                           MultiAssetEventLater>;

    [[nodiscard]] BookRuntime& book(BookId book_id);
    [[nodiscard]] const BookRuntime& book(BookId book_id) const;
    void schedule(MultiAssetEvent event);
    void schedule_background_events(BookRuntime& runtime);
    void schedule_next_background_event(BookRuntime& runtime);
    void schedule_samples(BookId book_id);
    void schedule_liquidity_shock();
    void schedule_next_arbitrage(std::int64_t timestamp_ns,
                                 std::uint64_t decision_sequence);
    void schedule_next_value_decision(BookId book_id,
                                      std::int64_t timestamp_ns,
                                      std::uint64_t decision_sequence);
    void schedule_next_quote(BookId book_id,
                             std::int64_t timestamp_ns,
                             std::uint64_t quote_sequence);
    void apply_order(const MultiAssetEvent& event, bool is_hedge);
    void capture_trades(BookRuntime& runtime,
                        const MultiAssetEvent& cause,
                        bool may_trigger_cross_book_reaction);
    [[nodiscard]] SequentialMultiAssetResult finish(double wall_seconds);
    void write_summary_csv(const SequentialMultiAssetResult& result) const;

    SequentialMultiAssetConfig config_;
    std::int64_t end_time_ns_ = 0;
    bool initialized_ = false;
    bool completed_ = false;
    std::vector<BookRuntime> books_;
    std::vector<MultiAssetBookConfig> resolved_book_configs_;
    std::unique_ptr<SharedMarketMakerAgent> shared_market_maker_;
    std::unique_ptr<EtfArbitrageAgent> etf_arbitrage_agent_;
    std::vector<std::unique_ptr<FundamentalValueAgent>> value_agents_;
    EventQueue events_;
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

} // namespace dlob
