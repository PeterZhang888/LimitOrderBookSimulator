#include "agents/AgentPopulation.hpp"

#include "exchange/EventOrdering.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dlob {

int AgentPopulation::local_count(int global_count, int worker_index, int worker_count) {
    if (worker_count <= 0 || worker_index < 0 || global_count <= 0) return 0;
    return global_count / worker_count + (worker_index < global_count % worker_count ? 1 : 0);
}

WorkerRole AgentPopulation::role_for_rank(int rank, int world_size) {
    if (world_size <= 1) return WorkerRole::AllLocal;
    if (rank == 0) return WorkerRole::Exchange;
    if (world_size < 5) return WorkerRole::Mixed;
    switch ((rank - 1) % 4) {
        case 0: return WorkerRole::MarketMaker;
        case 1: return WorkerRole::Momentum;
        case 2: return WorkerRole::Informed;
        case 3: return WorkerRole::Institutional;
        default: break;
    }
    return WorkerRole::Mixed;
}

AgentKind AgentPopulation::role_agent_kind(WorkerRole role) {
    switch (role) {
        case WorkerRole::MarketMaker: return AgentKind::MarketMaker;
        case WorkerRole::Momentum: return AgentKind::Momentum;
        case WorkerRole::Informed: return AgentKind::Informed;
        case WorkerRole::Institutional: return AgentKind::Institutional;
        case WorkerRole::Exchange:
        case WorkerRole::Mixed:
        case WorkerRole::AllLocal:
            break;
    }
    return AgentKind::Background;
}

int AgentPopulation::role_worker_count(WorkerRole role) const {
    if (world_size_ <= 1) return 1;
    int count = 0;
    for (int worker = 1; worker < world_size_; ++worker) {
        if (role_for_rank(worker, world_size_) == role) ++count;
    }
    return count;
}

int AgentPopulation::role_worker_index(WorkerRole role) const {
    if (world_size_ <= 1) return 0;
    int index = 0;
    for (int worker = 1; worker < rank_; ++worker) {
        if (role_for_rank(worker, world_size_) == role) ++index;
    }
    return role_for_rank(rank_, world_size_) == role ? index : -1;
}

bool AgentPopulation::owns_kind(AgentKind kind) const {
    if (!is_worker_) return false;
    if (role_ == WorkerRole::AllLocal || role_ == WorkerRole::Mixed) return true;
    return role_agent_kind(role_) == kind;
}

AgentPopulation::AgentPopulation(int mpi_rank, int world_size, const PopulationConfig& config)
    : rank_(mpi_rank),
      world_size_(world_size),
      is_worker_(world_size == 1 || mpi_rank > 0),
      role_(role_for_rank(mpi_rank, world_size)),
      config_(config),
      message_builder_(mpi_rank) {
    const int scale = std::max(1, config_.population_scale);
    global_summary_.market_makers = std::max(0, config_.market_makers);
    global_summary_.momentum = std::max(0, config_.momentum_traders) * scale;
    global_summary_.informed = std::max(0, config_.informed_traders) * scale;
    global_summary_.institutional = std::max(0, config_.institutional_traders) * scale;

    if (!is_worker_) return;

    const auto count_for = [this](AgentKind kind, int global_count) {
        if (!owns_kind(kind)) return 0;
        const WorkerRole partition_role =
            (role_ == WorkerRole::AllLocal || role_ == WorkerRole::Mixed) ? role_ : role_;
        const int worker_count = role_ == WorkerRole::AllLocal
            ? 1
            : role_worker_count(partition_role);
        const int worker_index = role_ == WorkerRole::AllLocal
            ? 0
            : role_worker_index(partition_role);
        return local_count(global_count, worker_index, worker_count);
    };

    local_summary_.market_makers = count_for(AgentKind::MarketMaker, global_summary_.market_makers);
    local_summary_.momentum = count_for(AgentKind::Momentum, global_summary_.momentum);
    local_summary_.informed = count_for(AgentKind::Informed, global_summary_.informed);
    local_summary_.institutional = count_for(AgentKind::Institutional, global_summary_.institutional);

    const double activity_divisor = static_cast<double>(scale);
    int local_id = 0;

    momentum_.reserve(static_cast<std::size_t>(local_summary_.momentum));
    for (int i = 0; i < local_summary_.momentum; ++i) {
        const int owner = make_owner_id(rank_, local_id++);
        FastRng parameter_rng(config_.seed ^ static_cast<std::uint64_t>(owner) ^ 0x1001ULL);
        MomentumAgentConfig agent_config;
        agent_config.tick_size = config_.tick_size;
        agent_config.wake_rate_per_second = config_.momentum_rate_per_second / activity_divisor;
        const double heterogeneity = 0.75 + 0.50 * parameter_rng.uniform01();
        agent_config.threshold_ticks = std::max(0.001, config_.momentum_threshold_ticks * heterogeneity);
        agent_config.order_quantity = std::max(1, static_cast<int>(std::llround(
            static_cast<double>(config_.momentum_order_quantity)
                * (0.75 + 0.50 * parameter_rng.uniform01()))));
        momentum_.emplace_back(owner, agent_config, config_.simulation_start_ns,
                               config_.seed ^ static_cast<std::uint64_t>(owner));
        register_agent(owner, AgentKind::Momentum, static_cast<int>(momentum_.size()) - 1);
    }

    informed_.reserve(static_cast<std::size_t>(local_summary_.informed));
    for (int i = 0; i < local_summary_.informed; ++i) {
        const int owner = make_owner_id(rank_, local_id++);
        FastRng parameter_rng(config_.seed ^ static_cast<std::uint64_t>(owner) ^ 0x2002ULL);
        InformedTraderConfig agent_config;
        agent_config.tick_size = config_.tick_size;
        agent_config.wake_rate_per_second = config_.informed_rate_per_second / activity_divisor;
        agent_config.signal_noise_ticks = std::max(
            0.001, config_.informed_signal_noise_ticks
                * (0.75 + 0.50 * parameter_rng.uniform01()));
        agent_config.trade_threshold_ticks = std::max(
            0.001, config_.informed_trade_threshold_ticks
                * (0.75 + 0.50 * parameter_rng.uniform01()));
        agent_config.base_quantity = std::max(1, static_cast<int>(std::llround(
            static_cast<double>(config_.informed_base_quantity)
                * (0.75 + 0.50 * parameter_rng.uniform01()))));
        informed_.emplace_back(owner, agent_config, config_.simulation_start_ns,
                               config_.seed ^ static_cast<std::uint64_t>(owner));
        register_agent(owner, AgentKind::Informed, static_cast<int>(informed_.size()) - 1);
    }

    institutional_.reserve(static_cast<std::size_t>(local_summary_.institutional));
    for (int i = 0; i < local_summary_.institutional; ++i) {
        const int owner = make_owner_id(rank_, local_id++);
        FastRng parameter_rng(config_.seed ^ static_cast<std::uint64_t>(owner) ^ 0x3003ULL);
        LargeInstitutionalConfig agent_config;
        agent_config.side = parameter_rng.uniform01() < 0.5 ? Side::Buy : Side::Sell;
        agent_config.parent_quantity = 10'000 + parameter_rng.uniform_int(0, 90'000);
        agent_config.child_quantity = 100 + parameter_rng.uniform_int(0, 900);
        agent_config.participation_cap = std::clamp(
            config_.institutional_participation_cap
                * (0.75 + 0.50 * parameter_rng.uniform01()),
            0.001, 0.95);
        agent_config.wake_rate_per_second = config_.institutional_rate_per_second / activity_divisor;

        const std::int64_t total_duration = config_.simulation_end_ns - config_.simulation_start_ns;
        const double duration_fraction = 0.20 + 0.60 * parameter_rng.uniform01();
        const std::int64_t active_duration = static_cast<std::int64_t>(
            static_cast<long double>(duration_fraction)
                * static_cast<long double>(total_duration));
        const std::int64_t latest_start = std::max<std::int64_t>(0, total_duration - active_duration);
        agent_config.start_time_ns = config_.simulation_start_ns
            + static_cast<std::int64_t>(static_cast<long double>(parameter_rng.uniform01())
                                        * static_cast<long double>(latest_start));
        agent_config.end_time_ns = std::min(config_.simulation_end_ns,
                                             agent_config.start_time_ns + active_duration);

        institutional_.emplace_back(owner, agent_config,
                                    config_.seed ^ static_cast<std::uint64_t>(owner));
        register_agent(owner, AgentKind::Institutional, static_cast<int>(institutional_.size()) - 1);
    }

    market_makers_.reserve(static_cast<std::size_t>(local_summary_.market_makers));
    for (int i = 0; i < local_summary_.market_makers; ++i) {
        const int owner = make_owner_id(rank_, local_id++);
        PassiveMarketMakerConfig agent_config;
        agent_config.tick_size = config_.tick_size;
        agent_config.order_quantity = config_.market_maker_order_quantity;
        agent_config.min_spread_ticks = config_.market_maker_min_spread_ticks;
        agent_config.quote_interval_ns = config_.market_maker_interval_ns;
        agent_config.quote_skip_probability = config_.market_maker_quote_skip_probability;
        agent_config.recovery_quote_skip_probability = std::min(
            config_.market_maker_quote_skip_probability, 0.10);
        market_makers_.emplace_back(owner, agent_config, config_.simulation_start_ns,
                                    config_.seed ^ static_cast<std::uint64_t>(owner));
        register_agent(owner, AgentKind::MarketMaker, static_cast<int>(market_makers_.size()) - 1);
    }

    rebuild_wake_queue();
}

void AgentPopulation::register_agent(int owner_id, AgentKind kind, int vector_index) {
    const int local_index = owner_local_index(owner_id);
    if (local_index < 0) return;
    if (owner_table_.size() <= static_cast<std::size_t>(local_index)) {
        owner_table_.resize(static_cast<std::size_t>(local_index) + 1);
    }
    owner_table_[static_cast<std::size_t>(local_index)] = AgentRef{kind, vector_index};
}

void AgentPopulation::observe_market(const MarketState& state) {
    if (!is_worker_) return;
    current_state_ = state;
    history_.push_back(state);
    const std::int64_t keep_after = state.exchange_time_ns - 5'000'000'000LL;
    while (!history_.empty() && history_.front().exchange_time_ns < keep_after) {
        history_.pop_front();
    }
}

const MarketState* AgentPopulation::past_state(std::int64_t target_ns) const {
    const MarketState* result = nullptr;
    for (const MarketState& state : history_) {
        if (state.exchange_time_ns <= target_ns) result = &state;
        else break;
    }
    return result;
}

std::int64_t AgentPopulation::agent_next_wake(AgentKind kind, int index) const {
    if (index < 0) return no_wake_time;
    const auto i = static_cast<std::size_t>(index);
    switch (kind) {
        case AgentKind::MarketMaker:
            return i < market_makers_.size() ? market_makers_[i].next_wake_time() : no_wake_time;
        case AgentKind::Momentum:
            return i < momentum_.size() ? momentum_[i].next_wake_time() : no_wake_time;
        case AgentKind::Informed:
            return i < informed_.size() ? informed_[i].next_wake_time() : no_wake_time;
        case AgentKind::Institutional:
            return i < institutional_.size() ? institutional_[i].next_wake_time() : no_wake_time;
        case AgentKind::Arbitrage:
        case AgentKind::Value:
        case AgentKind::Background:
            break;
    }
    return no_wake_time;
}

void AgentPopulation::push_wake(AgentKind kind, int index) {
    const std::int64_t time = agent_next_wake(kind, index);
    if (time != no_wake_time) wake_queue_.push(WakeEntry{time, kind, index});
}

void AgentPopulation::rebuild_wake_queue() {
    wake_queue_ = {};
    for (std::size_t i = 0; i < market_makers_.size(); ++i) {
        push_wake(AgentKind::MarketMaker, static_cast<int>(i));
    }
    for (std::size_t i = 0; i < momentum_.size(); ++i) {
        push_wake(AgentKind::Momentum, static_cast<int>(i));
    }
    for (std::size_t i = 0; i < informed_.size(); ++i) {
        push_wake(AgentKind::Informed, static_cast<int>(i));
    }
    for (std::size_t i = 0; i < institutional_.size(); ++i) {
        push_wake(AgentKind::Institutional, static_cast<int>(i));
    }
}

std::int64_t AgentPopulation::next_wake_time() const {
    return wake_queue_.empty() ? no_wake_time : wake_queue_.top().time_ns;
}

std::int64_t AgentPopulation::batch_horizon_ns() const {
    switch (role_) {
        case WorkerRole::MarketMaker:
            return std::max<std::int64_t>(0, config_.market_maker_batch_horizon_ns);
        case WorkerRole::Momentum:
            return std::max<std::int64_t>(0, config_.momentum_batch_horizon_ns);
        case WorkerRole::Informed:
            return std::max<std::int64_t>(0, config_.informed_batch_horizon_ns);
        case WorkerRole::Institutional:
            return std::max<std::int64_t>(0, config_.institutional_batch_horizon_ns);
        case WorkerRole::Mixed:
        case WorkerRole::AllLocal:
            return std::max<std::int64_t>(0, std::min({
                config_.market_maker_batch_horizon_ns,
                config_.momentum_batch_horizon_ns,
                config_.informed_batch_horizon_ns,
                config_.institutional_batch_horizon_ns}));
        case WorkerRole::Exchange:
            break;
    }
    return 0;
}

void AgentPopulation::generate_one_due(const WakeEntry& entry,
                                       std::int64_t activation_time_ns,
                                       std::int64_t cutoff_ns,
                                       std::vector<OrderMessage>& output) {
    const std::int64_t exclusive_end = cutoff_ns == std::numeric_limits<std::int64_t>::max()
        ? cutoff_ns
        : cutoff_ns + 1;
    const auto index = static_cast<std::size_t>(entry.index);
    switch (entry.kind) {
        case AgentKind::MarketMaker:
            market_makers_[index].generate_orders(current_state_, activation_time_ns,
                                                   exclusive_end, message_builder_, output);
            break;
        case AgentKind::Momentum: {
            const MarketState* past = past_state(entry.time_ns - 1'000'000'000LL);
            momentum_[index].generate_orders(current_state_, past, activation_time_ns,
                                              exclusive_end, message_builder_, output);
            break;
        }
        case AgentKind::Informed:
            informed_[index].generate_orders(current_state_, activation_time_ns,
                                              exclusive_end, message_builder_, output);
            break;
        case AgentKind::Institutional:
            institutional_[index].generate_orders(current_state_, activation_time_ns,
                                                   exclusive_end, message_builder_, output);
            break;
        case AgentKind::Arbitrage:
        case AgentKind::Value:
        case AgentKind::Background:
            break;
    }
}

std::vector<OrderMessage> AgentPopulation::generate_due_orders(
    std::int64_t activation_time_ns,
    std::int64_t cutoff_ns) {
    std::vector<OrderMessage> output;
    if (!is_worker_ || cutoff_ns < activation_time_ns) return output;

    while (!wake_queue_.empty() && wake_queue_.top().time_ns <= cutoff_ns) {
        const WakeEntry entry = wake_queue_.top();
        wake_queue_.pop();
        generate_one_due(entry, activation_time_ns, cutoff_ns, output);
        push_wake(entry.kind, entry.index);
    }
    std::sort(output.begin(), output.end(), order_before);
    return output;
}

std::vector<OrderMessage> AgentPopulation::generate_orders(std::int64_t window_start_ns,
                                                           std::int64_t window_end_ns) {
    std::vector<OrderMessage> output;
    if (!is_worker_) return output;

    const MarketState* momentum_past = past_state(current_state_.exchange_time_ns - 1'000'000'000LL);
    for (MomentumAgent& agent : momentum_) {
        agent.generate_orders(current_state_, momentum_past, window_start_ns, window_end_ns,
                              message_builder_, output);
    }
    for (InformedTraderAgent& agent : informed_) {
        agent.generate_orders(current_state_, window_start_ns, window_end_ns,
                              message_builder_, output);
    }
    for (LargeInstitutionalAgent& agent : institutional_) {
        agent.generate_orders(current_state_, window_start_ns, window_end_ns,
                              message_builder_, output);
    }
    for (PassiveMarketMakerAgent& agent : market_makers_) {
        agent.generate_orders(current_state_, window_start_ns, window_end_ns,
                              message_builder_, output);
    }
    rebuild_wake_queue();
    std::sort(output.begin(), output.end(), order_before);
    return output;
}

void AgentPopulation::apply_reports(const std::vector<AgentReport>& reports) {
    if (!is_worker_) return;
    for (const AgentReport& report : reports) {
        if (owner_rank(report.owner_id) != rank_) continue;
        const int local_index = owner_local_index(report.owner_id);
        if (local_index < 0 || static_cast<std::size_t>(local_index) >= owner_table_.size()) continue;
        const AgentRef ref = owner_table_[static_cast<std::size_t>(local_index)];
        switch (ref.kind) {
            case AgentKind::Momentum:
                momentum_[static_cast<std::size_t>(ref.index)].apply_report(report);
                break;
            case AgentKind::Informed:
                informed_[static_cast<std::size_t>(ref.index)].apply_report(report);
                break;
            case AgentKind::Institutional:
                institutional_[static_cast<std::size_t>(ref.index)].apply_report(report);
                break;
            case AgentKind::MarketMaker:
                market_makers_[static_cast<std::size_t>(ref.index)].apply_report(report);
                break;
            case AgentKind::Arbitrage:
            case AgentKind::Value:
            case AgentKind::Background:
                break;
        }
    }
}

} // namespace dlob
