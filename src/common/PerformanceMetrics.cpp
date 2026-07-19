#include "common/PerformanceMetrics.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace dlob {
namespace {

std::size_t index_of(TimingStage stage) {
    return static_cast<std::size_t>(stage);
}

const char* role_name(const RankProfile& profile) {
    if (profile.is_exchange && profile.is_worker) return "exchange+worker";
    if (profile.is_exchange) return "exchange";
    if (profile.is_worker) return "worker";
    return "idle";
}

void ensure_output_dir(const std::filesystem::path& output_dir) {
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    if (error) {
        throw std::runtime_error("Could not create output directory: " + output_dir.string()
                                 + " (" + error.message() + ")");
    }
}

} // namespace

const char* timing_stage_name(TimingStage stage) {
    switch (stage) {
        case TimingStage::Initialization: return "initialization";
        case TimingStage::StartBarrier: return "start_barrier";
        case TimingStage::HawkesGeneration: return "hawkes_generation";
        case TimingStage::MarketStateBuild: return "market_state_build";
        case TimingStage::BroadcastMarketState: return "broadcast_market_state";
        case TimingStage::AgentObserveAndGenerate: return "agent_observe_and_generate";
        case TimingStage::GatherOrderCounts: return "gather_order_counts";
        case TimingStage::GatherOrderPayload: return "gather_order_payload";
        case TimingStage::QueuePartition: return "queue_partition";
        case TimingStage::EventSort: return "event_sort";
        case TimingStage::MatchingEngine: return "matching_engine";
        case TimingStage::ReportPacking: return "report_packing";
        case TimingStage::ScatterReportCounts: return "scatter_report_counts";
        case TimingStage::ScatterReportPayload: return "scatter_report_payload";
        case TimingStage::ApplyReports: return "apply_reports";
        case TimingStage::EndBarrier: return "end_barrier";
        case TimingStage::TotalWall: return "total_wall";
        case TimingStage::Count: break;
    }
    return "unknown";
}

PerformanceMetrics::PerformanceMetrics(int rank, int world_size, bool is_exchange, bool is_worker) {
    profile_.rank = rank;
    profile_.world_size = world_size;
    profile_.is_exchange = is_exchange ? 1 : 0;
    profile_.is_worker = is_worker ? 1 : 0;
    char name[MPI_MAX_PROCESSOR_NAME]{};
    int length = 0;
    if (MPI_Get_processor_name(name, &length) == MPI_SUCCESS) {
        const std::size_t copy_length = std::min<std::size_t>(
            profile_.processor_name.size() - 1,
            length > 0 ? static_cast<std::size_t>(length) : 0U);
        std::copy_n(name, copy_length, profile_.processor_name.begin());
        profile_.processor_name[copy_length] = '\0';
    }
}

void PerformanceMetrics::set_local_population(int market_makers,
                                              int momentum,
                                              int informed,
                                              int institutional) {
    profile_.local_market_makers = market_makers;
    profile_.local_momentum = momentum;
    profile_.local_informed = informed;
    profile_.local_institutional = institutional;
}

void PerformanceMetrics::add(TimingStage stage, double seconds) {
    const std::size_t index = index_of(stage);
    if (index >= timing_stage_count) return;
    profile_.seconds[index] += std::max(0.0, seconds);
    ++profile_.calls[index];
}

void PerformanceMetrics::set(TimingStage stage, double seconds) {
    const std::size_t index = index_of(stage);
    if (index >= timing_stage_count) return;
    profile_.seconds[index] = std::max(0.0, seconds);
    profile_.calls[index] = 1;
}

void PerformanceMetrics::increment_windows() {
    ++profile_.counters.windows;
}

void PerformanceMetrics::write_rank_csv(const std::filesystem::path& output_dir) const {
    ensure_output_dir(output_dir);
    const std::filesystem::path file = output_dir
        / ("timing_rank_" + std::to_string(profile_.rank) + ".csv");
    std::ofstream output(file);
    if (!output) throw std::runtime_error("Could not open timing file: " + file.string());

    output << "rank,host,role,stage,total_seconds,calls,mean_seconds\n";
    output << std::setprecision(12);
    for (std::size_t i = 0; i < timing_stage_count; ++i) {
        const double mean = profile_.calls[i] > 0
            ? profile_.seconds[i] / static_cast<double>(profile_.calls[i]) : 0.0;
        output << profile_.rank << ',' << profile_.processor_name.data() << ','
               << role_name(profile_) << ',' << timing_stage_name(static_cast<TimingStage>(i)) << ','
               << profile_.seconds[i] << ',' << profile_.calls[i] << ',' << mean << '\n';
    }
}

std::vector<RankProfile> gather_rank_profiles(const RankProfile& local,
                                              int rank,
                                              int world_size) {
    static_assert(std::is_trivially_copyable_v<RankProfile>);
    if (sizeof(RankProfile) > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("RankProfile is too large for MPI_Gather byte count");
    }

    std::vector<RankProfile> profiles;
    if (rank == 0) profiles.resize(static_cast<std::size_t>(world_size));
    const int bytes = static_cast<int>(sizeof(RankProfile));
    const int result = MPI_Gather(&local, bytes, MPI_BYTE,
                                  rank == 0 ? profiles.data() : nullptr,
                                  bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS) throw std::runtime_error("MPI_Gather failed for rank profiles");
    return profiles;
}

void write_combined_timing_csv(const std::filesystem::path& output_dir,
                               const std::vector<RankProfile>& profiles) {
    ensure_output_dir(output_dir);
    const std::filesystem::path file = output_dir / "timing_all_ranks.csv";
    std::ofstream output(file);
    if (!output) throw std::runtime_error("Could not open combined timing file: " + file.string());

    output << "rank,host,role,stage,total_seconds,calls,mean_seconds\n";
    output << std::setprecision(12);
    for (const RankProfile& profile : profiles) {
        for (std::size_t i = 0; i < timing_stage_count; ++i) {
            const double mean = profile.calls[i] > 0
                ? profile.seconds[i] / static_cast<double>(profile.calls[i]) : 0.0;
            output << profile.rank << ',' << profile.processor_name.data() << ','
                   << role_name(profile) << ',' << timing_stage_name(static_cast<TimingStage>(i)) << ','
                   << profile.seconds[i] << ',' << profile.calls[i] << ',' << mean << '\n';
        }
    }
}

void write_timing_summary_csv(const std::filesystem::path& output_dir,
                              const std::vector<RankProfile>& profiles) {
    ensure_output_dir(output_dir);
    const std::filesystem::path file = output_dir / "timing_summary.csv";
    std::ofstream output(file);
    if (!output) throw std::runtime_error("Could not open timing summary file: " + file.string());

    output << "stage,min_seconds,mean_seconds,max_seconds,sum_seconds,max_rank\n";
    output << std::setprecision(12);
    for (std::size_t i = 0; i < timing_stage_count; ++i) {
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        int max_rank = -1;
        for (const RankProfile& profile : profiles) {
            const double value = profile.seconds[i];
            minimum = std::min(minimum, value);
            if (value > maximum) {
                maximum = value;
                max_rank = profile.rank;
            }
            sum += value;
        }
        if (profiles.empty()) {
            minimum = 0.0;
            maximum = 0.0;
        }
        const double mean = profiles.empty() ? 0.0 : sum / static_cast<double>(profiles.size());
        output << timing_stage_name(static_cast<TimingStage>(i)) << ','
               << minimum << ',' << mean << ',' << maximum << ',' << sum << ',' << max_rank << '\n';
    }
}

void write_rank_counters_csv(const std::filesystem::path& output_dir,
                             const std::vector<RankProfile>& profiles) {
    ensure_output_dir(output_dir);
    const std::filesystem::path file = output_dir / "rank_counters.csv";
    std::ofstream output(file);
    if (!output) throw std::runtime_error("Could not open rank counter file: " + file.string());

    output << "rank,host,role,local_market_makers,local_momentum,local_informed,local_institutional,"
              "local_total_agents,windows,strategic_orders_generated,strategic_orders_received_exchange,"
              "strategic_orders_processed,background_orders_processed,reports_created,reports_received,"
              "order_bytes_sent,order_bytes_received_exchange,report_bytes_sent_exchange,"
              "report_bytes_received,fill_reports,order_result_reports,peak_pending_orders\n";
    for (const RankProfile& p : profiles) {
        const int total_agents = p.local_market_makers + p.local_momentum
            + p.local_informed + p.local_institutional;
        const RankCounters& c = p.counters;
        output << p.rank << ',' << p.processor_name.data() << ',' << role_name(p) << ','
               << p.local_market_makers << ',' << p.local_momentum << ','
               << p.local_informed << ',' << p.local_institutional << ',' << total_agents << ','
               << c.windows << ',' << c.strategic_orders_generated << ','
               << c.strategic_orders_received_exchange << ',' << c.strategic_orders_processed << ','
               << c.background_orders_processed << ',' << c.reports_created << ','
               << c.reports_received << ',' << c.order_bytes_sent << ','
               << c.order_bytes_received_exchange << ',' << c.report_bytes_sent_exchange << ','
               << c.report_bytes_received << ',' << c.fill_reports << ','
               << c.order_result_reports << ',' << c.peak_pending_orders << '\n';
    }
}

} // namespace dlob
