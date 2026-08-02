// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "calibration/SimulationRecorder.hpp"
#include "common/DistributedTypes.hpp"
#include "common/TradeTapeHasher.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/DistributedLimitOrderBook.hpp"

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dlob {

// These identifiers are properties of the simulated model.  They are never
// derived from an MPI rank, so repartitioning books cannot change event order.
using StableEntityId = std::uint64_t;

// A symbol is the persistent identity of an empirical book.  Configuration
// subsets must reindex routing book_ids contiguously for the simulator, so a
// stochastic stream keyed by book_id changes whenever the same stock is
// evaluated in a different calibration subset.  Use a fully specified FNV-1a
// digest instead of std::hash so the stream identity is stable across
// processes, platforms, MPI decompositions, and subset sizes.
[[nodiscard]] inline StableEntityId stable_symbol_stream_id(
    std::string_view symbol) noexcept {
    StableEntityId hash = 14'695'981'039'346'656'037ULL;
    for (const char character : symbol) {
        const auto byte = static_cast<unsigned char>(character);
        hash ^= static_cast<StableEntityId>(byte);
        hash *= 1'099'511'628'211ULL;
    }
    // Keep this domain distinct from the small fixed entity identifiers.
    return hash ^ 0xa076'1d64'78bd'642fULL;
}

inline constexpr StableEntityId background_entity_base = 0x0001'0000ULL;
inline constexpr StableEntityId shared_market_maker_entity = 0x0002'0000ULL;
inline constexpr StableEntityId sampler_entity_base = 0x0003'0000ULL;
inline constexpr StableEntityId market_maker_repair_entity_base = 0x0004'0000ULL;
inline constexpr StableEntityId liquidity_shock_entity_base = 0x0005'0000ULL;
inline constexpr StableEntityId etf_arbitrage_entity = 0x0006'0000ULL;
inline constexpr StableEntityId fundamental_value_entity_base = 0x0007'0000ULL;
inline constexpr std::int32_t liquidity_shock_owner_id = 800'001;
inline constexpr std::int32_t etf_arbitrage_owner_id = 700'001;
inline constexpr std::int32_t fundamental_value_owner_id_base = 600'001;
inline constexpr std::int32_t local_market_maker_owner_id_base = 500'001;

[[nodiscard]] constexpr StableEntityId background_entity(BookId book_id) noexcept {
    return background_entity_base + static_cast<StableEntityId>(book_id);
}

[[nodiscard]] constexpr StableEntityId sampler_entity(BookId book_id) noexcept {
    return sampler_entity_base + static_cast<StableEntityId>(book_id);
}

[[nodiscard]] constexpr StableEntityId market_maker_repair_entity(
    BookId book_id) noexcept {
    return market_maker_repair_entity_base + static_cast<StableEntityId>(book_id);
}

[[nodiscard]] constexpr StableEntityId liquidity_shock_entity(
    BookId book_id) noexcept {
    return liquidity_shock_entity_base + static_cast<StableEntityId>(book_id);
}

[[nodiscard]] constexpr StableEntityId fundamental_value_entity(
    BookId book_id) noexcept {
    return fundamental_value_entity_base + static_cast<StableEntityId>(book_id);
}

[[nodiscard]] constexpr std::int32_t fundamental_value_owner_id(
    BookId book_id) noexcept {
    return fundamental_value_owner_id_base + static_cast<std::int32_t>(book_id);
}

[[nodiscard]] constexpr std::int32_t local_market_maker_owner_id(
    BookId book_id) noexcept {
    return local_market_maker_owner_id_base + static_cast<std::int32_t>(book_id);
}

// An event phase is only consulted after timestamp.  It makes simultaneous
// causal stages explicit and avoids relying on priority_queue insertion order.
enum class MultiAssetEventPhase : std::uint8_t {
    ExogenousWake = 0,
    AgentDecision = 1,
    OrderArrival = 2,
    ReportDelivery = 3,
    OrderCompletion = 4,
    CrossBookReaction = 5,
    Observation = 6
};

enum class MultiAssetEventKind : std::uint8_t {
    BackgroundWake = 0,
    MarketMakerQuoteWake = 1,
    OrderArrival = 2,
    HedgeOrderArrival = 3,
    ReportDelivery = 4,
    CrossBookReaction = 5,
    HedgeOrderCompletion = 6,
    SampleState = 7,
    MarketMakerRepairWake = 8,
    LiquidityShockOrderArrival = 9,
    EtfArbitrageWake = 10,
    ArbitrageOrderArrival = 11,
    FundamentalValueWake = 12,
    FundamentalValueOrderArrival = 13
};

// This is the complete, rank-independent global priority key.  child_index
// distinguishes several children (cancel, bid, ask, reports, hedges) created by
// one entity decision without inventing a process-local sequence number.
struct MultiAssetEventKey {
    std::int64_t timestamp_ns = 0;
    MultiAssetEventPhase phase = MultiAssetEventPhase::ExogenousWake;
    BookId book_id = 0;
    StableEntityId origin_entity = 0;
    std::uint64_t origin_local_sequence = 0;
    std::uint32_t child_index = 0;

    [[nodiscard]] friend constexpr bool operator==(const MultiAssetEventKey&,
                                                   const MultiAssetEventKey&) = default;

    [[nodiscard]] friend constexpr std::strong_ordering
    operator<=>(const MultiAssetEventKey& left, const MultiAssetEventKey& right) noexcept {
        if (const auto value = left.timestamp_ns <=> right.timestamp_ns; value != 0) return value;
        if (const auto value = left.phase <=> right.phase; value != 0) return value;
        if (const auto value = left.book_id <=> right.book_id; value != 0) return value;
        if (const auto value = left.origin_entity <=> right.origin_entity; value != 0) return value;
        if (const auto value = left.origin_local_sequence <=> right.origin_local_sequence;
            value != 0) return value;
        return left.child_index <=> right.child_index;
    }
};

// std::priority_queue is a max heap.  Reversing the exact key produces a global
// minimum queue and nothing else is permitted to affect scheduling order.
struct MultiAssetEventKeyLater {
    [[nodiscard]] constexpr bool operator()(const MultiAssetEventKey& left,
                                            const MultiAssetEventKey& right) const noexcept {
        return left > right;
    }
};

struct MultiAssetEvent {
    MultiAssetEventKey key{};
    MultiAssetEventKind kind = MultiAssetEventKind::BackgroundWake;
    BookId source_book_id = 0;
    HawkesEvent hawkes{};
    OrderMessage order{};
    AgentReport report{};
    TradeExecution trade{};

    // Reports caused by a hedge update capital and inventory but do not create
    // another hedge.  This prevents an artificial cross-book feedback loop.
    bool may_trigger_cross_book_reaction = false;
};

struct MultiAssetEventLater {
    [[nodiscard]] constexpr bool operator()(const MultiAssetEvent& left,
                                            const MultiAssetEvent& right) const noexcept {
        return left.key > right.key;
    }
};

// An optional, deterministic intervention for paired control/shock runs.  It
// inserts one exogenous market order without consuming any background-agent
// random numbers, so the background Hawkes streams remain common random
// numbers between the two scenarios.
struct LiquidityShockConfig {
    std::int64_t time_ns = 0;
    BookId book_id = 0;
    Side side = Side::Sell;
    std::int32_t quantity = 0;
};

// A calibrated empirical input for one logical book.  The legacy CLI leaves
// this vector empty and receives the original replicated-QQQ configuration.
struct MultiAssetBookConfig {
    std::string symbol;
    std::string data_dir;
    std::string hawkes_rates_file;
    double fundamental_price_ticks = 0.0;
    // Per-square-root-second volatility of the latent fundamental in basis
    // points. Certified empirical configurations derive this directly from
    // the five training days' fixed-clock return variance. A zero value
    // preserves the legacy static-fundamental behaviour.
    double fundamental_volatility_bps_sqrt_second = 0.0;
    // Probability that the latent reference receives a news innovation in a
    // one-second interval. Together with the unconditional volatility above,
    // this represents the intermittent timing of empirical price changes.
    // Certified configurations estimate it from training data only.
    double fundamental_move_probability_per_second = 1.0;
    // Kurtosis of the latent innovation conditional on a news move.  The
    // pooled training fixed-clock kurtosis K and move probability p imply
    // this value as K*p.  A value of three recovers the former conditional
    // Gaussian process; values in [1, infinity) admit both thinner- and
    // heavier-tailed empirical innovations without changing their variance.
    double fundamental_conditional_kurtosis = 3.0;
    // Persistence of an optional stationary AR(1) log-variance state.  The
    // state is keyed by symbol, model seed and decision boundary, so changing
    // MPI ownership cannot change the volatility path.  A value in [0, 1) is
    // required.
    double fundamental_log_variance_persistence = 0.0;
    // Stationary standard deviation of the log-variance state.  The
    // corresponding volatility multiplier is normalized so that its squared
    // expectation is one.  Zero disables stochastic volatility exactly and
    // preserves the legacy latent-value path.
    double fundamental_log_variance_std = 0.0;
    // Loading of the persistent activity state into Hawkes immigration.  The
    // effective stationary log-baseline standard deviation is
    // fundamental_log_variance_std times this value.  A lognormal mean
    // correction preserves fitted unconditional immigration in expectation,
    // while the event clock acquires persistent active and quiet intervals.
    // The field name is retained for CSV compatibility.  Zero is exact legacy
    // behaviour.
    double fundamental_order_flow_coupling = 0.0;
    std::int32_t initial_best_bid_ticks = 0;
    std::int32_t initial_best_ask_ticks = 0;
    std::int32_t initial_best_bid_depth = 0;
    std::int32_t initial_best_ask_depth = 0;
    // Frozen five-day pooled-training top-queue means.  The same per-symbol
    // values must be used in every training runtime and held-out validation;
    // using the simulated day's target would leak validation information.
    // Zero disables the response for legacy configurations.
    double target_mean_bid_depth = 0.0;
    double target_mean_ask_depth = 0.0;
    double beta = 1.0;
    double basket_weight = 0.0;
    std::int32_t market_maker_quote_quantity = 0;
    std::int32_t target_spread_ticks = 1;
    // Shared probability that an empirically sampled distance-zero limit mark
    // is labelled as inside-spread when the pre-add spread is at least two
    // ticks.  This maximum-symmetry compact-data mapping is distinct from the
    // local maker's mean-spread target above.
    double quote_improvement_probability = 0.05;
};

struct EtfArbitrageConfig {
    bool enabled = false;
    BookId etf_book_id = 0;
    double trigger_bps = 5.0;
    double release_bps = 2.5;
    std::int32_t etf_order_quantity = 100;
    std::int32_t max_component_quantity = 100'000;
    std::int64_t decision_interval_ns = 100'000'000;
    std::int64_t order_latency_ns = 5'000;
};

struct FundamentalValueConfig {
    bool enabled = false;
    double threshold_bps = 10.0;
    double response_step_bps = 5.0;
    std::int32_t base_order_quantity = 25;
    std::int32_t max_order_quantity = 500;
    std::int64_t max_abs_inventory = 2'000'000;
    double fundamental_volatility_bps_sqrt_second = 0.0;
    std::int64_t decision_interval_ns = 1'000'000'000;
    std::int64_t order_latency_ns = 5'000;
};

struct SequentialMultiAssetConfig {
    int duration_seconds = 10;
    int book_count = 2;
    std::uint64_t seed = 12345;
    std::string data_dir = "data";
    std::string hawkes_rates_file;
    std::string output_dir = "results/sequential_multi_asset";
    std::vector<MultiAssetBookConfig> book_configs;

    int tick_size = 100;
    double initial_depth_scale = 1.0;
    double fundamental_price_ticks = 2'203'550.0;
    std::int64_t sample_interval_ns = 1'000'000'000;

    int market_maker_order_quantity = 100;
    int market_maker_quote_levels = 7;
    int market_maker_quote_quantity_growth = 2;
    std::int64_t market_maker_quote_interval_ns = 100'000'000;
    std::int64_t market_maker_order_latency_ns = 5'000;
    std::int64_t report_latency_ns = 5'000;
    std::int64_t cross_book_reaction_latency_ns = 50'000;
    std::int64_t hedge_order_latency_ns = 5'000;
    double market_maker_exposure_threshold = 500.0;
    bool enable_shared_market_maker_hedging = false;
    int hedge_lot_size = 1;
    int max_hedge_quantity = 1'000;

    // std::nullopt is the control experiment.  When present, exactly one
    // LiquidityShockOrderArrival is scheduled on the target book owner.
    std::optional<LiquidityShockConfig> liquidity_shock;
    EtfArbitrageConfig etf_arbitrage;
    FundamentalValueConfig fundamental_value;
};

struct BookRuntime {
    BookId book_id = 0;
    std::string symbol;
    double fundamental_value_ticks = 0.0;
    BackgroundHawkesAgent background;
    DistributedLimitOrderBook lob;
    TradeTapeHasher trade_hasher;
    calibration::SimulationRecorder recorder;
    // Keep the compact Hawkes path outside the generic event heap.  Only the
    // next wake is queued, which makes a 23,400-second run practical without
    // changing the exact global event order.
    std::vector<HawkesEvent> background_events;
    std::size_t next_background_event_index = 0;
    std::uint64_t processed_events = 0;
    std::uint64_t submitted_orders = 0;

    BookRuntime(BookId id,
                std::string symbol_name,
                double fundamental,
                const BackgroundHawkesConfig& background_config,
                int tick_size,
                std::uint64_t recorder_seed)
        : book_id(id),
          symbol(std::move(symbol_name)),
          fundamental_value_ticks(fundamental),
          background(background_config),
          lob(tick_size, id),
          recorder(recorder_seed, 8192, tick_size) {}
};

struct MultiAssetBookSummary {
    BookId book_id = 0;
    std::string symbol;
    MarketState final_state{};
    std::int64_t market_maker_inventory = 0;
    double market_maker_cash_ticks = 0.0;
    std::int64_t arbitrage_inventory = 0;
    std::int64_t arbitrage_cash_ticks = 0;
    std::int64_t value_agent_inventory = 0;
    double value_agent_cash_ticks = 0.0;
    double final_fundamental_value_ticks = 0.0;
    std::uint64_t processed_events = 0;
    std::uint64_t submitted_orders = 0;
    std::uint64_t trade_count = 0;
    std::uint64_t trade_hash = TradeTapeHasher::offset_basis;
    std::uint64_t expected_sample_count = 0;
    bool structurally_valid = false;
    calibration::SimulationRecord calibration_record{};
};

struct SequentialMultiAssetResult {
    std::vector<MultiAssetBookSummary> books;
    bool structurally_valid = false;
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
    double market_maker_cash_ticks = 0.0;
    std::int64_t arbitrage_cash_ticks = 0;
    double wall_seconds = 0.0;
    std::string summary_csv;
};

[[nodiscard]] constexpr std::uint64_t stable_sequence(StableEntityId entity,
                                                       std::uint64_t local_sequence,
                                                       std::uint32_t child_index = 0) noexcept {
    // SplitMix64-style finalization makes all key fields visible while remaining
    // fully specified and independent of standard-library hash implementations.
    std::uint64_t value = entity;
    value ^= local_sequence + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
    value ^= static_cast<std::uint64_t>(child_index) * 0xd1b54a32d192ed03ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

} // namespace dlob
