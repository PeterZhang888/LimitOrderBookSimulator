#pragma once

#include "mpi/MpiCompat.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dlob {

enum class TimingStage : std::size_t {
    Initialization = 0,
    StartBarrier,
    HawkesGeneration,
    MarketStateBuild,
    BroadcastMarketState,
    AgentObserveAndGenerate,
    GatherOrderCounts,
    GatherOrderPayload,
    QueuePartition,
    EventSort,
    MatchingEngine,
    ReportPacking,
    ScatterReportCounts,
    ScatterReportPayload,
    ApplyReports,
    EndBarrier,
    TotalWall,
    Count
};

inline constexpr std::size_t timing_stage_count = static_cast<std::size_t>(TimingStage::Count);

const char* timing_stage_name(TimingStage stage);

struct RankCounters {
    std::uint64_t windows = 0;
    std::uint64_t strategic_orders_generated = 0;
    std::uint64_t strategic_orders_received_exchange = 0;
    std::uint64_t strategic_orders_processed = 0;
    std::uint64_t background_orders_processed = 0;
    std::uint64_t reports_created = 0;
    std::uint64_t reports_received = 0;
    std::uint64_t order_bytes_sent = 0;
    std::uint64_t order_bytes_received_exchange = 0;
    std::uint64_t report_bytes_sent_exchange = 0;
    std::uint64_t report_bytes_received = 0;
    std::uint64_t fill_reports = 0;
    std::uint64_t order_result_reports = 0;
    std::uint64_t peak_pending_orders = 0;
};

struct RankProfile {
    std::int32_t rank = 0;
    std::int32_t world_size = 1;
    std::int32_t is_exchange = 0;
    std::int32_t is_worker = 0;
    std::int32_t local_market_makers = 0;
    std::int32_t local_momentum = 0;
    std::int32_t local_informed = 0;
    std::int32_t local_institutional = 0;
    std::array<char, 128> processor_name{};
    std::array<double, timing_stage_count> seconds{};
    std::array<std::uint64_t, timing_stage_count> calls{};
    RankCounters counters{};
};

class PerformanceMetrics {
public:
    PerformanceMetrics(int rank, int world_size, bool is_exchange, bool is_worker);

    void set_local_population(int market_makers, int momentum, int informed, int institutional);
    void add(TimingStage stage, double seconds);
    void set(TimingStage stage, double seconds);
    void increment_windows();
    RankCounters& counters() { return profile_.counters; }
    const RankCounters& counters() const { return profile_.counters; }
    const RankProfile& profile() const { return profile_; }

    void write_rank_csv(const std::filesystem::path& output_dir) const;

private:
    RankProfile profile_{};
};

class ScopedStageTimer {
public:
    ScopedStageTimer(PerformanceMetrics& metrics, TimingStage stage)
        : metrics_(metrics), stage_(stage), start_(MPI_Wtime()) {}

    ~ScopedStageTimer() { metrics_.add(stage_, MPI_Wtime() - start_); }

    ScopedStageTimer(const ScopedStageTimer&) = delete;
    ScopedStageTimer& operator=(const ScopedStageTimer&) = delete;

private:
    PerformanceMetrics& metrics_;
    TimingStage stage_;
    double start_ = 0.0;
};

std::vector<RankProfile> gather_rank_profiles(const RankProfile& local,
                                              int rank,
                                              int world_size);

void write_combined_timing_csv(const std::filesystem::path& output_dir,
                               const std::vector<RankProfile>& profiles);
void write_timing_summary_csv(const std::filesystem::path& output_dir,
                              const std::vector<RankProfile>& profiles);
void write_rank_counters_csv(const std::filesystem::path& output_dir,
                             const std::vector<RankProfile>& profiles);

} // namespace dlob
