#include "agents/AgentPopulation.hpp"
#include <algorithm>
#include <cmath>
namespace dlob {
int AgentPopulation::local_count(int global_count, int worker_index, int worker_count) {
    if (worker_count <= 0 || worker_index < 0 || global_count <= 0) return 0;
    return global_count / worker_count + (worker_index < global_count % worker_count ? 1 : 0);
}
AgentPopulation::AgentPopulation(int mpi_rank, int world_size, const PopulationConfig& config)
    : rank_(mpi_rank), world_size_(world_size), config_(config), message_builder_(mpi_rank) {
    // Single-process mode is only a correctness fallback. Under real MPI, rank 0
    // is exchange-only and ranks 1..N-1 are workers.
    is_worker_ = world_size_ == 1 || rank_ > 0;
    worker_count_ = world_size_ == 1 ? 1 : std::max(0, world_size_ - 1);
    worker_index_ = world_size_ == 1 ? 0 : rank_ - 1;
    const int scale = std::max(1, config_.population_scale);
    global_summary_.market_makers = std::max(0, config_.market_makers);
    global_summary_.momentum = std::max(0, config_.momentum_traders) * scale;
    global_summary_.informed = std::max(0, config_.informed_traders) * scale;
    global_summary_.institutional = std::max(0, config_.institutional_traders) * scale;
    if (!is_worker_) return;
    local_summary_.market_makers = local_count(global_summary_.market_makers, worker_index_, worker_count_);
    local_summary_.momentum = local_count(global_summary_.momentum, worker_index_, worker_count_);
    local_summary_.informed = local_count(global_summary_.informed, worker_index_, worker_count_);
    local_summary_.institutional = local_count(global_summary_.institutional, worker_index_, worker_count_);
    const double activity_divisor = static_cast<double>(scale);
    int local_id = 0;
    momentum_.reserve(static_cast<std::size_t>(local_summary_.momentum));
    for (int i = 0; i < local_summary_.momentum; ++i) {
        const int owner = make_owner_id(rank_, local_id++);
        FastRng parameter_rng(config_.seed ^ static_cast<std::uint64_t>(owner) ^ 0x1001ULL);
        MomentumAgentConfig agent_config;
        agent_config.tick_size = config_.tick_size;
        agent_config.wake_rate_per_second = config_.momentum_rate_per_second / activity_divisor;
        agent_config.threshold_ticks = 0.15 + 0.35 * parameter_rng.uniform01();
        agent_config.order_quantity = 50 + parameter_rng.uniform_int(0, 150);
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
        agent_config.signal_noise_ticks = 0.5 + 2.5 * parameter_rng.uniform01();
        agent_config.trade_threshold_ticks = 0.5 + 1.5 * parameter_rng.uniform01();
        agent_config.base_quantity = 25 + parameter_rng.uniform_int(0, 175);
        informed_.emplace_back(owner, agent_config, config_.simulation_start_ns,config_.seed ^ static_cast<std::uint64_t>(owner));
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
        agent_config.participation_cap = 0.05 + 0.10 * parameter_rng.uniform01();
        agent_config.wake_rate_per_second = config_.institutional_rate_per_second / activity_divisor;

        const std::int64_t total_duration = config_.simulation_end_ns - config_.simulation_start_ns;
        const double duration_fraction = 0.20 + 0.60 * parameter_rng.uniform01();
        const std::int64_t active_duration = static_cast<std::int64_t>(
            static_cast<long double>(duration_fraction) * static_cast<long double>(total_duration));
        const std::int64_t latest_start = std::max<std::int64_t>(0, total_duration - active_duration);
        agent_config.start_time_ns = config_.simulation_start_ns
            + static_cast<std::int64_t>(static_cast<long double>(parameter_rng.uniform01()) * static_cast<long double>(latest_start));
        agent_config.end_time_ns = std::min(config_.simulation_end_ns,agent_config.start_time_ns + active_duration);
        institutional_.emplace_back(owner, agent_config,config_.seed ^ static_cast<std::uint64_t>(owner));
        register_agent(owner, AgentKind::Institutional, static_cast<int>(institutional_.size()) - 1);
    }
    market_makers_.reserve(static_cast<std::size_t>(local_summary_.market_makers));
    for (int i = 0; i < local_summary_.market_makers; ++i) {
        const int owner = make_owner_id(rank_, local_id++);
        PassiveMarketMakerConfig agent_config;
        agent_config.tick_size = config_.tick_size;
        agent_config.quote_interval_ns = config_.market_maker_interval_ns;
        market_makers_.emplace_back(owner, agent_config, config_.simulation_start_ns, config_.seed ^ static_cast<std::uint64_t>(owner));
        register_agent(owner, AgentKind::MarketMaker, static_cast<int>(market_makers_.size()) - 1);
    }
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
    while (!history_.empty() && history_.front().exchange_time_ns < keep_after) history_.pop_front();
}
const MarketState* AgentPopulation::past_state(std::int64_t target_ns) const {
    const MarketState* result = nullptr;
    for (const MarketState& state : history_) {
        if (state.exchange_time_ns <= target_ns) result = &state;
        else break;
    }
    return result;
}
std::vector<OrderMessage> AgentPopulation::generate_orders(std::int64_t window_start_ns,std::int64_t window_end_ns) {
    std::vector<OrderMessage> output;
    if (!is_worker_) return output;

    const MarketState* momentum_past = past_state(current_state_.exchange_time_ns - 1'000'000'000LL);
    for (MomentumAgent& agent : momentum_) {
        agent.generate_orders(current_state_, momentum_past, window_start_ns, window_end_ns,message_builder_, output);
    }
    for (InformedTraderAgent& agent : informed_) {
        agent.generate_orders(current_state_, window_start_ns, window_end_ns,message_builder_, output);
    }
    for (LargeInstitutionalAgent& agent : institutional_) {
        agent.generate_orders(current_state_, window_start_ns, window_end_ns, message_builder_, output);
    }
    for (PassiveMarketMakerAgent& agent : market_makers_) {
        agent.generate_orders(current_state_, window_start_ns, window_end_ns,message_builder_, output);
    }
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
            case AgentKind::Background:
                break;
        }
    }
}

} 
