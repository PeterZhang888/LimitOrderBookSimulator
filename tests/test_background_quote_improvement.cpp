#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/LimitOrderBook.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using namespace dlob;

    BackgroundHawkesConfig config;
    // Python derives immigration rates against this exact runtime matrix.
    // Compact ITCH artifacts identify marginal type rates but not lagged
    // cross-type kernels, so only the documented diagonal self excitation is
    // admissible in the certified workflow.
    for (std::size_t row = 0; row < config.alpha.size(); ++row) {
        for (std::size_t column = 0; column < config.alpha[row].size(); ++column) {
            const double expected = row == column ? 0.20 : 0.0;
            assert(config.alpha[row][column] == expected);
        }
    }
    config.seed = 42;
    config.tick_size = 100;
    config.quote_improvement_probability = 1.0;
    const std::filesystem::path zero_distance_file =
        std::filesystem::temp_directory_path()
        / "dlob_background_zero_distance_distribution.csv";
    const std::filesystem::path one_distance_file =
        std::filesystem::temp_directory_path()
        / "dlob_background_one_distance_distribution.csv";
    const std::filesystem::path tail_distance_file =
        std::filesystem::temp_directory_path()
        / "dlob_background_tail_distance_distribution.csv";
    const std::filesystem::path quantity_file =
        std::filesystem::temp_directory_path()
        / "dlob_background_quantity_distribution.csv";
    {
        std::ofstream output(zero_distance_file);
        assert(output);
        output << "distance_ticks,count\n0,10\n";
    }
    {
        std::ofstream output(one_distance_file);
        assert(output);
        output << "distance_ticks,count\n1,10\n";
    }
    {
        std::ofstream output(tail_distance_file);
        assert(output);
        output << "distance_ticks,count\n10,10\n";
    }
    {
        std::ofstream output(quantity_file);
        assert(output);
        output << "quantity,count\n100,10\n";
    }
    config.limit_buy_quantity_file = quantity_file.string();
    config.limit_sell_quantity_file = quantity_file.string();
    config.market_buy_quantity_file = quantity_file.string();
    config.market_sell_quantity_file = quantity_file.string();
    config.cancel_bid_quantity_file = quantity_file.string();
    config.cancel_ask_quantity_file = quantity_file.string();
    config.limit_buy_distance_file = zero_distance_file.string();
    config.limit_sell_distance_file = zero_distance_file.string();
    config.cancel_bid_distance_file = zero_distance_file.string();
    config.cancel_ask_distance_file = zero_distance_file.string();

    MarketState state;
    state.best_bid_ticks = 10'000;
    state.best_ask_ticks = 10'200;
    state.mid_price_ticks = 10'100.0;

    BackgroundHawkesAgent buy_agent(config);
    const OrderMessage buy = buy_agent.make_order(
        HawkesEvent{1, HawkesEventType::LimitBuy}, state, 1);
    assert(buy.price_ticks == 10'100);
    assert(buy.price_ticks > state.best_bid_ticks);
    assert(buy.price_ticks < state.best_ask_ticks);

    BackgroundHawkesAgent sell_agent(config);
    const OrderMessage sell = sell_agent.make_order(
        HawkesEvent{1, HawkesEventType::LimitSell}, state, 1);
    assert(sell.price_ticks == 10'100);
    assert(sell.price_ticks > state.best_bid_ticks);
    assert(sell.price_ticks < state.best_ask_ticks);

    // The compact-data probability labels sampled zero-distance marks.  A
    // probability-one label must not overwrite positive-distance marks; this
    // preserves both side-specific empirical distance marginals exactly.
    BackgroundHawkesConfig positive_distance = config;
    positive_distance.limit_buy_distance_file = one_distance_file.string();
    positive_distance.limit_sell_distance_file = one_distance_file.string();
    BackgroundHawkesAgent positive_distance_buy_agent(positive_distance);
    const OrderMessage positive_distance_buy =
        positive_distance_buy_agent.make_order(
            HawkesEvent{1, HawkesEventType::LimitBuy}, state, 1);
    assert(positive_distance_buy.distance_ticks == 1);
    assert(positive_distance_buy.price_ticks == 9'900);
    BackgroundHawkesAgent positive_distance_sell_agent(positive_distance);
    const OrderMessage positive_distance_sell =
        positive_distance_sell_agent.make_order(
            HawkesEvent{1, HawkesEventType::LimitSell}, state, 1);
    assert(positive_distance_sell.distance_ticks == 1);
    assert(positive_distance_sell.price_ticks == 10'300);

    // The extractor defines eligibility geometrically at exactly two ticks.
    // The per-symbol rounded mean-spread target is unrelated and cannot move
    // this gate, even when it is much wider than the contemporaneous spread.
    BackgroundHawkesConfig empirical_gate = config;
    empirical_gate.target_spread_ticks = 99;
    MarketState exactly_two_ticks = state;
    exactly_two_ticks.best_ask_ticks = 10'200;
    BackgroundHawkesAgent exact_gate_buy_agent(empirical_gate);
    const OrderMessage exact_gate_buy = exact_gate_buy_agent.make_order(
        HawkesEvent{1, HawkesEventType::LimitBuy}, exactly_two_ticks, 1);
    assert(exact_gate_buy.distance_ticks == 0);
    assert(exact_gate_buy.price_ticks == 10'100);
    BackgroundHawkesAgent exact_gate_sell_agent(empirical_gate);
    const OrderMessage exact_gate_sell = exact_gate_sell_agent.make_order(
        HawkesEvent{1, HawkesEventType::LimitSell}, exactly_two_ticks, 1);
    assert(exact_gate_sell.distance_ticks == 0);
    assert(exact_gate_sell.price_ticks == 10'100);

    // With a one-tick spread there is no legal inside price, so the same
    // probability-one label leaves a zero-distance mark at the same-side BBO.
    MarketState one_tick_spread = state;
    one_tick_spread.best_ask_ticks = 10'100;
    BackgroundHawkesAgent ineligible_buy_agent(empirical_gate);
    const OrderMessage ineligible_buy = ineligible_buy_agent.make_order(
        HawkesEvent{1, HawkesEventType::LimitBuy}, one_tick_spread, 1);
    assert(ineligible_buy.distance_ticks == 0);
    assert(ineligible_buy.price_ticks == 10'000);
    BackgroundHawkesAgent ineligible_sell_agent(empirical_gate);
    const OrderMessage ineligible_sell = ineligible_sell_agent.make_order(
        HawkesEvent{1, HawkesEventType::LimitSell}, one_tick_spread, 1);
    assert(ineligible_sell.distance_ticks == 0);
    assert(ineligible_sell.price_ticks == 10'100);

    config.quote_improvement_probability = 0.0;
    MarketState missing_bid;
    missing_bid.best_ask_ticks = 10'200;
    missing_bid.fundamental_value_ticks = 50'000.0;
    BackgroundHawkesAgent missing_bid_agent(config);
    const OrderMessage repaired_bid = missing_bid_agent.make_order(
        HawkesEvent{2, HawkesEventType::LimitBuy}, missing_bid, 2);
    assert(repaired_bid.price_ticks == 10'100);
    assert(repaired_bid.price_ticks < missing_bid.best_ask_ticks);

    MarketState missing_ask;
    missing_ask.best_bid_ticks = 10'000;
    missing_ask.fundamental_value_ticks = 50'000.0;
    BackgroundHawkesAgent missing_ask_agent(config);
    const OrderMessage repaired_ask = missing_ask_agent.make_order(
        HawkesEvent{2, HawkesEventType::LimitSell}, missing_ask, 2);
    assert(repaired_ask.price_ticks == 10'100);
    assert(repaired_ask.price_ticks > missing_ask.best_bid_ticks);

    // The empirical loader must retain odd lots.  Before the regression fix,
    // the fallback floor of 25 silently discarded this one-share row.
    const std::filesystem::path odd_lot_file =
        std::filesystem::temp_directory_path()
        / "dlob_background_odd_lot_distribution.csv";
    {
        std::ofstream output(odd_lot_file);
        assert(output);
        output << "quantity,count\n1,10\n";
    }
    BackgroundHawkesConfig odd_lot_config = config;
    odd_lot_config.limit_buy_quantity_file = odd_lot_file.string();
    BackgroundHawkesAgent odd_lot_agent(odd_lot_config);
    const OrderMessage odd_lot = odd_lot_agent.make_order(
        HawkesEvent{3, HawkesEventType::LimitBuy}, state, 3);
    assert(odd_lot.quantity == 1);

    BackgroundHawkesConfig reactive = config;
    reactive.target_mean_bid_depth = 100.0;
    reactive.target_mean_ask_depth = 200.0;
    MarketState shallow = state;
    shallow.best_bid_depth = 1'025;
    shallow.best_ask_depth = 1'050;
    shallow.background_best_bid_depth = 25;
    shallow.background_best_ask_depth = 50;
    MarketState target = state;
    target.best_bid_depth = 1'100;
    target.best_ask_depth = 1'200;
    target.background_best_bid_depth = 100;
    target.background_best_ask_depth = 200;
    MarketState deep = state;
    deep.best_bid_depth = 1'800;
    deep.best_ask_depth = 2'600;
    deep.background_best_bid_depth = 800;
    deep.background_best_ask_depth = 1600;
    BackgroundHawkesAgent shallow_agent(reactive);
    BackgroundHawkesAgent target_agent(reactive);
    BackgroundHawkesAgent deep_agent(reactive);
    assert(shallow_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelBid}, shallow, 4).quantity == 25);
    assert(target_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelBid}, target, 4).quantity == 100);
    assert(deep_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelBid}, deep, 4).quantity == 400);
    BackgroundHawkesAgent shallow_ask_agent(reactive);
    assert(shallow_ask_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelAsk}, shallow, 4).quantity == 25);

    // The target is a top-queue statistic used as a reduced-form feedback for
    // every cancellation mark inside the represented ten-level band.  A
    // positive-distance retained mark therefore receives the same multiplier
    // as a distance-zero mark.
    BackgroundHawkesConfig non_top_reactive = reactive;
    non_top_reactive.cancel_bid_distance_file = one_distance_file.string();
    BackgroundHawkesAgent non_top_agent(non_top_reactive);
    const OrderMessage non_top_cancel = non_top_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelBid}, deep, 4);
    assert(non_top_cancel.distance_ticks == 1);
    assert(non_top_cancel.quantity == 400);
    BackgroundHawkesConfig non_top_ask_reactive = reactive;
    non_top_ask_reactive.cancel_ask_distance_file = one_distance_file.string();
    BackgroundHawkesAgent non_top_ask_agent(non_top_ask_reactive);
    const OrderMessage non_top_ask_cancel = non_top_ask_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelAsk}, deep, 4);
    assert(non_top_ask_cancel.distance_ticks == 1);
    assert(non_top_ask_cancel.quantity == 400);

    // Marks outside the retained band are filtered by the reduced book and do
    // not receive a queue multiplier before that no-op.  Keeping the sampled
    // quantity here makes the support boundary explicit and testable.
    BackgroundHawkesConfig tail_reactive = reactive;
    tail_reactive.cancel_bid_distance_file = tail_distance_file.string();
    BackgroundHawkesAgent tail_agent(tail_reactive);
    const OrderMessage tail_cancel = tail_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelBid}, deep, 4);
    assert(tail_cancel.distance_ticks == reduced_background_depth_levels);
    assert(tail_cancel.quantity == 100);

    // Additive shared-market-maker depth at the BBO must not feed back into
    // the anonymous background generator.  The total queue changes while the
    // owner-zero component, and hence the generated cancellation, does not.
    LimitOrderBook shared_quote_book(100);
    shared_quote_book.seed_calibrated_book(10'000, 10'200, 100, 200, 1.0);
    const MarketState before_shared = shared_quote_book.state(0, 10'100.0);
    OrderMessage shared_quote;
    shared_quote.sequence = 91;
    shared_quote.owner_id = 1'000'001;
    shared_quote.agent_kind = AgentKind::MarketMaker;
    shared_quote.action = OrderAction::Limit;
    shared_quote.side = Side::Buy;
    shared_quote.quantity = 500;
    shared_quote.price_ticks = shared_quote_book.best_bid();
    shared_quote_book.apply(shared_quote);
    const MarketState after_shared = shared_quote_book.state(1, 10'100.0);
    assert(after_shared.best_bid_depth == before_shared.best_bid_depth + 500);
    assert(after_shared.background_best_bid_depth
           == before_shared.background_best_bid_depth);
    assert(after_shared.total_background_bid_depth
           == before_shared.total_background_bid_depth);
    assert(after_shared.total_background_ask_depth
           == before_shared.total_background_ask_depth);
    BackgroundHawkesAgent before_shared_agent(reactive);
    BackgroundHawkesAgent after_shared_agent(reactive);
    const OrderMessage before_shared_cancel = before_shared_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelBid}, before_shared, 4);
    const OrderMessage after_shared_cancel = after_shared_agent.make_order(
        HawkesEvent{4, HawkesEventType::CancelBid}, after_shared, 4);
    assert(before_shared_cancel.quantity == 100);
    assert(after_shared_cancel.quantity == before_shared_cancel.quantity);

    // A maker may lead the market by one tick while anonymous liquidity
    // remains immediately behind.  Queue-reactive background cancellation
    // must measure owner zero's own best queue, not return zero merely because
    // the market BBO is maker-owned.  The generated distance-zero mark is then
    // mapped to and removes quantity from the nearest anonymous level.
    BackgroundHawkesConfig inside_reactive = reactive;
    inside_reactive.target_mean_bid_depth = 200.0;
    inside_reactive.target_mean_ask_depth = 400.0;

    LimitOrderBook inside_bid_book(100);
    inside_bid_book.seed_calibrated_book(10'000, 10'300, 100, 200, 1.0);
    OrderMessage inside_bid_quote;
    inside_bid_quote.sequence = 92;
    inside_bid_quote.owner_id = 1'000'002;
    inside_bid_quote.agent_kind = AgentKind::MarketMaker;
    inside_bid_quote.action = OrderAction::Limit;
    inside_bid_quote.side = Side::Buy;
    inside_bid_quote.quantity = 500;
    inside_bid_quote.price_ticks = 10'100;
    assert(inside_bid_book.apply(inside_bid_quote).resting_quantity == 500);
    const MarketState inside_bid_state = inside_bid_book.state(2, 10'150.0);
    assert(inside_bid_state.best_bid_ticks == 10'100);
    assert(inside_bid_state.best_bid_depth == 500);
    assert(inside_bid_state.background_best_bid_depth == 100);
    BackgroundHawkesAgent inside_bid_agent(inside_reactive);
    const OrderMessage inside_bid_cancel = inside_bid_agent.make_order(
        HawkesEvent{5, HawkesEventType::CancelBid}, inside_bid_state, 5);
    assert(inside_bid_cancel.distance_ticks == 0);
    assert(inside_bid_cancel.quantity == 50);
    const std::int64_t bid_background_before =
        inside_bid_book.total_background_bid_depth();
    const ApplyResult inside_bid_result = inside_bid_book.apply(inside_bid_cancel);
    assert(inside_bid_result.cancelled_quantity == 50);
    assert(inside_bid_book.total_background_bid_depth()
           == bid_background_before - 50);
    assert(inside_bid_book.background_best_bid_depth() == 50);
    assert(inside_bid_book.best_bid() == 10'100);

    LimitOrderBook inside_ask_book(100);
    inside_ask_book.seed_calibrated_book(10'000, 10'300, 100, 200, 1.0);
    OrderMessage inside_ask_quote;
    inside_ask_quote.sequence = 93;
    inside_ask_quote.owner_id = 1'000'003;
    inside_ask_quote.agent_kind = AgentKind::MarketMaker;
    inside_ask_quote.action = OrderAction::Limit;
    inside_ask_quote.side = Side::Sell;
    inside_ask_quote.quantity = 500;
    inside_ask_quote.price_ticks = 10'200;
    assert(inside_ask_book.apply(inside_ask_quote).resting_quantity == 500);
    const MarketState inside_ask_state = inside_ask_book.state(3, 10'150.0);
    assert(inside_ask_state.best_ask_ticks == 10'200);
    assert(inside_ask_state.best_ask_depth == 500);
    assert(inside_ask_state.background_best_ask_depth == 200);
    BackgroundHawkesAgent inside_ask_agent(inside_reactive);
    const OrderMessage inside_ask_cancel = inside_ask_agent.make_order(
        HawkesEvent{6, HawkesEventType::CancelAsk}, inside_ask_state, 6);
    assert(inside_ask_cancel.distance_ticks == 0);
    assert(inside_ask_cancel.quantity == 50);
    const std::int64_t ask_background_before =
        inside_ask_book.total_background_ask_depth();
    const ApplyResult inside_ask_result = inside_ask_book.apply(inside_ask_cancel);
    assert(inside_ask_result.cancelled_quantity == 50);
    assert(inside_ask_book.total_background_ask_depth()
           == ask_background_before - 50);
    assert(inside_ask_book.background_best_ask_depth() == 150);
    assert(inside_ask_book.best_ask() == 10'200);

    BackgroundHawkesConfig half_configured = config;
    half_configured.target_mean_bid_depth = 100.0;
    bool rejected_half_configured_targets = false;
    try {
        BackgroundHawkesAgent invalid(half_configured);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected_half_configured_targets = true;
    }
    assert(rejected_half_configured_targets);

    for (const double invalid_probability : {
             -0.01, 1.01, std::numeric_limits<double>::quiet_NaN()}) {
        BackgroundHawkesConfig invalid_probability_config = config;
        invalid_probability_config.quote_improvement_probability =
            invalid_probability;
        bool rejected_invalid_probability = false;
        try {
            BackgroundHawkesAgent invalid(invalid_probability_config);
            (void)invalid;
        } catch (const std::invalid_argument&) {
            rejected_invalid_probability = true;
        }
        assert(rejected_invalid_probability);
    }

    BackgroundHawkesConfig missing_marks = config;
    missing_marks.quote_improvement_probability = 0.0;
    missing_marks.limit_buy_quantity_file =
        "this_empirical_distribution_must_not_exist.csv";
    bool rejected_missing_marks = false;
    try {
        BackgroundHawkesAgent invalid(missing_marks);
        (void)invalid;
    } catch (const std::runtime_error&) {
        rejected_missing_marks = true;
    }
    assert(rejected_missing_marks);
    std::filesystem::remove(odd_lot_file);
    std::filesystem::remove(zero_distance_file);
    std::filesystem::remove(one_distance_file);
    std::filesystem::remove(tail_distance_file);
    std::filesystem::remove(quantity_file);

    std::cout << "background quote placement and mark tests passed\n";
    return 0;
}
