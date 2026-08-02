#include "exchange/BackgroundHawkesAgent.hpp"
#include "simulation/MultiAssetConfiguration.hpp"
#include "simulation/QueueReactiveBackgroundPolicy.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using dlob::BackgroundHawkesAgent;
using dlob::BackgroundHawkesConfig;
using dlob::HawkesEvent;
using dlob::HawkesEventType;
using dlob::MarketState;
using dlob::MultiAssetBookConfig;

constexpr std::array<const char*, 6> event_names{{
    "limit_buy", "limit_sell", "market_buy", "market_sell",
    "cancel_bid", "cancel_ask"}};
constexpr std::array<const char*, 4> feature_names{{
    "log_spread_ratio", "log_bid_depth_ratio", "log_ask_depth_ratio",
    "queue_imbalance"}};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto nonce = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
            / ("dlob_queue_policy_test_" + std::to_string(nonce));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    assert(output);
    output << text;
    assert(output.good());
}

void write_mark_files(const std::filesystem::path& directory) {
    for (const char* event : event_names) {
        const bool distance_event = std::string(event).starts_with("limit_")
            || std::string(event).starts_with("cancel_");
        write_text(directory / (std::string(event) + "_quantity_distribution.txt"),
                   "quantity,count\n100,1\n");
        if (distance_event) {
            write_text(directory / (std::string(event) + "_distance_distribution.txt"),
                       "distance_ticks,count\n0,1\n");
        }
    }
}

void write_rates(const std::filesystem::path& path,
                 const std::array<double, 6>& targets) {
    std::ofstream output(path);
    assert(output);
    output << "event_type,configured_mu,stationary_target_rate\n";
    for (std::size_t index = 0; index < targets.size(); ++index) {
        output << event_names[index] << ',' << targets[index] / 0.3
               << ',' << targets[index] << '\n';
    }
}

struct PolicyOptions {
    bool omit_last_state_coefficient = false;
    bool unstable = false;
    bool infeasible_for_symbol_target = false;
    int cluster_id = 4;
    double declared_spectral_radius = 0.11;
};

void write_policy(const std::filesystem::path& path,
                  const PolicyOptions options = {}) {
    std::ofstream output(path);
    assert(output);
    output << "kind,target,source,bin,value\n"
           << "meta,schema_version,,,1\n"
           << "meta,cluster_id,,," << options.cluster_id << "\n"
           << "meta,activity_scale,,,0.3\n"
           << "meta,fast_beta,,,2\n"
           << "meta,slow_beta,,,0.5\n"
           << "meta,state_log_multiplier_bound,,,3\n"
           << "meta,intraday_origin_ns,,,0\n"
           << "meta,intraday_bin_width_ns,,,1800000000000\n"
           << "meta,spectral_radius,,,"
           << options.declared_spectral_radius << "\n"
           << "meta,matrix_orientation,,,response_rows_trigger_columns\n"
           << "meta,stationary_target_scope,,,descriptive_cluster_member_mean\n";

    for (std::size_t target = 0; target < event_names.size(); ++target) {
        for (std::size_t source = 0; source < event_names.size(); ++source) {
            double fast = target == source ? 0.10 : 0.0;
            double slow = target == source ? 0.025 : 0.0;
            // One sparse replenishment channel makes the stationary equation
            // genuinely multivariate rather than six independent identities.
            if ((target == 0U && source == 2U)
                || (target == 2U && source == 0U)) {
                fast = 0.02;
            }
            if (options.unstable && target == source) fast = 1.50;
            // Spectral radius remains below 0.95, but the high-rate source can
            // still make one low-rate target's nonnegative immigration
            // infeasible.  The symbol-specific loader must reject it.
            if (options.infeasible_for_symbol_target
                && target == 0U && source == 5U) {
                fast = 1.70;
            }
            output << "fast_alpha," << event_names[target] << ','
                   << event_names[source] << ",," << fast << '\n';
            output << "slow_alpha," << event_names[target] << ','
                   << event_names[source] << ",," << slow << '\n';
        }
    }
    for (std::size_t target = 0; target < event_names.size(); ++target) {
        for (std::size_t feature = 0; feature < feature_names.size(); ++feature) {
            if (options.omit_last_state_coefficient
                && target + 1U == event_names.size()
                && feature + 1U == feature_names.size()) {
                continue;
            }
            const double coefficient = target == 0U && feature == 0U ? 0.2 : 0.0;
            output << "state_coefficient," << event_names[target] << ','
                   << feature_names[feature] << ",," << coefficient << '\n';
        }
    }
    for (int bin = 0; bin < 2; ++bin) {
        for (std::size_t target = 0; target < event_names.size(); ++target) {
            const double factor = bin == 0 ? 0.5 : 1.5;
            output << "intraday_factor," << event_names[target] << ",,"
                   << bin << ',' << factor << '\n';
        }
    }
}

void write_mapping(const std::filesystem::path& path,
                   const std::vector<std::string>& rows) {
    std::ofstream output(path);
    assert(output);
    output << "symbol,cluster_id,policy_file,limit_buy_improvement_file,"
              "limit_sell_improvement_file\n";
    for (const std::string& row : rows) output << row << '\n';
}

MultiAssetBookConfig make_asset(const std::string& symbol,
                                const std::filesystem::path& data,
                                const std::filesystem::path& rates) {
    MultiAssetBookConfig asset;
    asset.symbol = symbol;
    asset.data_dir = data.string();
    asset.hawkes_rates_file = rates.string();
    asset.fundamental_price_ticks = 10'000.0;
    asset.initial_best_bid_ticks = 9'800;
    asset.initial_best_ask_ticks = 10'200;
    asset.initial_best_bid_depth = 100;
    asset.initial_best_ask_depth = 100;
    asset.target_mean_bid_depth = 100.0;
    asset.target_mean_ask_depth = 100.0;
    asset.beta = 1.0;
    asset.market_maker_quote_quantity = 100;
    asset.target_spread_ticks = 4;
    asset.quote_improvement_probability = 1.0;
    return asset;
}

bool throws(const std::function<void()>& action) {
    try {
        action();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

double reconstructed_rate(const BackgroundHawkesConfig& config,
                          std::size_t target) {
    double result = config.activity_scale * config.mu[target];
    for (std::size_t source = 0; source < config.mu.size(); ++source) {
        result += (config.alpha[target][source] / config.beta
                   + config.slow_alpha[target][source] / config.slow_beta)
            * config.stationary_target_rates[source];
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    // Optional integration-driver mode lets the Python fitter's exact output
    // be round-tripped through the production C++ loader without duplicating
    // either parser in the test harness.
    if (argc == 3) {
        const auto assets = dlob::load_multi_asset_book_configs(argv[1]);
        const auto bundle = dlob::load_queue_reactive_background_bundle(
            argv[2], assets, 42U, 100);
        assert(bundle.configs.size() == assets.size());
        for (const auto& config : bundle.configs) {
            const BackgroundHawkesAgent agent(config);
            (void)agent;
        }
        return 0;
    }
    assert(argc == 1);
    TemporaryDirectory temporary;
    const auto root = temporary.path();
    const auto data_a = root / "data_a";
    const auto data_b = root / "data_b";
    write_mark_files(data_a);
    write_mark_files(data_b);

    const std::array<double, 6> targets_a{{10.0, 8.0, 3.0, 4.0, 12.0, 11.0}};
    const std::array<double, 6> targets_b{{6.0, 7.0, 2.0, 2.5, 9.0, 8.0}};
    write_rates(root / "rates_a.csv", targets_a);
    write_rates(root / "rates_b.csv", targets_b);
    write_policy(root / "policy.csv");
    write_text(root / "buy_improvement.csv",
               "improvement_ticks,improvement_price_units,count\n9,900,1\n");
    write_text(root / "sell_improvement.csv",
               "improvement_ticks,improvement_price_units,count\n8,800,1\n");
    write_mapping(root / "mapping.csv", {
        "AAA,4,policy.csv,buy_improvement.csv,sell_improvement.csv",
        "BBB,4,policy.csv,buy_improvement.csv,sell_improvement.csv",
    });

    const std::vector<MultiAssetBookConfig> assets{
        make_asset("AAA", data_a, root / "rates_a.csv"),
        make_asset("BBB", data_b, root / "rates_b.csv"),
    };
    const auto bundle = dlob::load_queue_reactive_background_bundle(
        root / "mapping.csv", assets, 42U, 100);
    assert(bundle.configs.size() == 2U);
    assert(bundle.cluster_ids == std::vector<int>({4, 4}));

    // The historical batch API has no contemporaneous book state.  It must
    // reject this policy instead of silently dropping its slow, intraday and
    // queue-response terms.
    assert(throws([&] {
        BackgroundHawkesAgent agent(bundle.configs[0]);
        (void)agent.simulate(0, 1'000'000'000LL);
    }));

    // Cluster dynamics are shared, but stationary targets and immigration
    // remain symbol specific and reconstruct every declared marginal rate.
    assert(bundle.configs[0].mu != bundle.configs[1].mu);
    for (std::size_t symbol = 0; symbol < bundle.configs.size(); ++symbol) {
        const auto& config = bundle.configs[symbol];
        assert(config.validate_stationary_target);
        assert(config.cancellation_quantity_depth_scaling);
        assert(config.intraday_factors.size() == 2U);
        for (std::size_t event = 0; event < event_names.size(); ++event) {
            const double expected = symbol == 0U ? targets_a[event] : targets_b[event];
            assert(std::abs(config.stationary_target_rates[event] - expected) < 1e-12);
            assert(std::abs(reconstructed_rate(config, event) - expected) < 1e-10);
        }
    }

    MarketState live;
    live.best_bid_ticks = 9'800;
    live.best_ask_ticks = 10'200;
    live.mid_price_ticks = 10'000.0;
    live.background_best_bid_depth = 1'000;
    live.background_best_ask_depth = 1'000;

    // Empirical improvements can span several ticks, but must remain strictly
    // inside the contemporaneous spread.  Both samples exceed the live gap
    // and are therefore clamped to three ticks.
    BackgroundHawkesAgent buy_agent(bundle.configs[0]);
    const auto buy = buy_agent.make_order(
        HawkesEvent{1, HawkesEventType::LimitBuy}, live, 1);
    assert(buy.distance_ticks == 0);
    assert(buy.price_ticks == 10'100);
    assert(buy.price_ticks > live.best_bid_ticks
           && buy.price_ticks < live.best_ask_ticks);
    BackgroundHawkesAgent sell_agent(bundle.configs[0]);
    const auto sell = sell_agent.make_order(
        HawkesEvent{1, HawkesEventType::LimitSell}, live, 1);
    assert(sell.distance_ticks == 0);
    assert(sell.price_ticks == 9'900);
    assert(sell.price_ticks > live.best_bid_ticks
           && sell.price_ticks < live.best_ask_ticks);

    // Event-type intensity and cancellation quantity are distinct margins.
    // The queue-reactive policy controls the former; the bounded mark response
    // removes more shares when the live anonymous queue greatly exceeds its
    // empirical target.  A tenfold ratio is capped at four by the production
    // agent, so the 100-share empirical mark becomes 400 shares.
    BackgroundHawkesAgent cancel_agent(bundle.configs[0]);
    const auto cancel = cancel_agent.make_order(
        HawkesEvent{2, HawkesEventType::CancelBid}, live, 2);
    assert(cancel.quantity == 400);

    // Mapping is a bijection over the configured asset universe.
    write_mapping(root / "missing_symbol.csv", {
        "AAA,4,policy.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "missing_symbol.csv", assets, 42U, 100);
    }));
    write_mapping(root / "wrong_symbol.csv", {
        "AAA,4,policy.csv,buy_improvement.csv,sell_improvement.csv",
        "CCC,4,policy.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "wrong_symbol.csv", assets, 42U, 100);
    }));
    write_mapping(root / "duplicate_symbol.csv", {
        "AAA,4,policy.csv,buy_improvement.csv,sell_improvement.csv",
        "AAA,4,policy.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "duplicate_symbol.csv", assets, 42U, 100);
    }));
    write_policy(root / "wrong_cluster.csv", PolicyOptions{.cluster_id = 5});
    write_mapping(root / "wrong_cluster_mapping.csv", {
        "AAA,4,wrong_cluster.csv,buy_improvement.csv,sell_improvement.csv",
        "BBB,4,wrong_cluster.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "wrong_cluster_mapping.csv", assets, 42U, 100);
    }));

    // Missing, incomplete and dynamically unstable policy artifacts all stop
    // construction before the simulator can schedule its first event.
    write_mapping(root / "missing_policy_mapping.csv", {
        "AAA,4,absent.csv,buy_improvement.csv,sell_improvement.csv",
        "BBB,4,absent.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "missing_policy_mapping.csv", assets, 42U, 100);
    }));
    write_policy(root / "incomplete.csv", PolicyOptions{.omit_last_state_coefficient = true});
    write_mapping(root / "incomplete_mapping.csv", {
        "AAA,4,incomplete.csv,buy_improvement.csv,sell_improvement.csv",
        "BBB,4,incomplete.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "incomplete_mapping.csv", assets, 42U, 100);
    }));
    write_policy(root / "unstable.csv", PolicyOptions{.unstable = true});
    write_mapping(root / "unstable_mapping.csv", {
        "AAA,4,unstable.csv,buy_improvement.csv,sell_improvement.csv",
        "BBB,4,unstable.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "unstable_mapping.csv", assets, 42U, 100);
    }));
    write_policy(root / "false_radius.csv",
                 PolicyOptions{.declared_spectral_radius = 0.2});
    write_mapping(root / "false_radius_mapping.csv", {
        "AAA,4,false_radius.csv,buy_improvement.csv,sell_improvement.csv",
        "BBB,4,false_radius.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "false_radius_mapping.csv", assets, 42U, 100);
    }));
    write_policy(root / "infeasible.csv", PolicyOptions{.infeasible_for_symbol_target = true});
    write_mapping(root / "infeasible_mapping.csv", {
        "AAA,4,infeasible.csv,buy_improvement.csv,sell_improvement.csv",
        "BBB,4,infeasible.csv,buy_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "infeasible_mapping.csv", assets, 42U, 100);
    }));

    // Missing empirical improvement support is fail-closed in the policy
    // loader before the simulator can construct a mark sampler.
    write_mapping(root / "missing_mark_mapping.csv", {
        "AAA,4,policy.csv,absent_buy.csv,absent_sell.csv",
        "BBB,4,policy.csv,absent_buy.csv,absent_sell.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "missing_mark_mapping.csv", assets, 42U, 100);
    }));

    // Price-unit and tick columns are redundant by design; disagreement or a
    // sub-tick observation must not be rounded silently.
    write_text(root / "bad_improvement.csv",
               "improvement_ticks,improvement_price_units,count\n1,99,1\n");
    write_mapping(root / "bad_mark_mapping.csv", {
        "AAA,4,policy.csv,bad_improvement.csv,sell_improvement.csv",
        "BBB,4,policy.csv,bad_improvement.csv,sell_improvement.csv",
    });
    assert(throws([&] {
        (void)dlob::load_queue_reactive_background_bundle(
            root / "bad_mark_mapping.csv", assets, 42U, 100);
    }));

    return 0;
}
