#include "mpi/AsyncExchangeLoop.hpp"

#include "calibration/SimulationRecorder.hpp"
#include "common/PerformanceMetrics.hpp"
#include "exchange/EventOrdering.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace dlob {
namespace {

constexpr int kMarketStateTag = 4101;
constexpr int kOrderBatchTag = 4102;
constexpr int kReportBatchTag = 4103;
constexpr std::uint64_t kMarketStateMagic = 0x444C4F4253544154ULL;
constexpr std::uint64_t kOrderBatchMagic = 0x444C4F424F524452ULL;
constexpr std::uint64_t kReportBatchMagic = 0x444C4F4252505254ULL;

struct StatePacket {
    std::uint64_t magic = kMarketStateMagic;
    std::uint64_t window_index = 0;
    MarketState state{};
};

struct BatchHeader {
    std::uint64_t magic = 0;
    std::uint64_t window_index = 0;
    std::uint64_t item_count = 0;
};

static_assert(std::is_trivially_copyable_v<StatePacket>);
static_assert(std::is_trivially_copyable_v<BatchHeader>);
static_assert(std::is_trivially_copyable_v<OrderMessage>);
static_assert(std::is_trivially_copyable_v<AgentReport>);

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

int checked_mpi_byte_count(std::size_t bytes, const char* label) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds MPI int byte-count limit");
    }
    return static_cast<int>(bytes);
}

template <typename T>
std::vector<unsigned char> make_batch_packet(std::uint64_t magic,
                                             std::uint64_t window_index,
                                             const std::vector<T>& items) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (items.size() >
        (std::numeric_limits<std::size_t>::max() - sizeof(BatchHeader)) / sizeof(T)) {
        throw std::runtime_error("MPI batch size overflow");
    }
    BatchHeader header{magic, window_index, static_cast<std::uint64_t>(items.size())};
    const std::size_t payload_bytes = items.size() * sizeof(T);
    std::vector<unsigned char> packet(sizeof(BatchHeader) + payload_bytes);
    std::memcpy(packet.data(), &header, sizeof(BatchHeader));
    if (payload_bytes > 0) {
        std::memcpy(packet.data() + sizeof(BatchHeader), items.data(), payload_bytes);
    }
    return packet;
}

template <typename T>
std::vector<T> decode_batch_packet(const std::vector<unsigned char>& packet,
                                   std::uint64_t expected_magic,
                                   std::uint64_t expected_window_index,
                                   const char* label) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (packet.size() < sizeof(BatchHeader)) {
        throw std::runtime_error(std::string(label) + " is smaller than its header");
    }
    BatchHeader header{};
    std::memcpy(&header, packet.data(), sizeof(BatchHeader));
    if (header.magic != expected_magic) {
        throw std::runtime_error(std::string(label) + " has invalid magic");
    }
    if (header.window_index != expected_window_index) {
        throw std::runtime_error(std::string(label) + " belongs to window "
            + std::to_string(header.window_index) + " but expected "
            + std::to_string(expected_window_index));
    }
    if (header.item_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string(label) + " item count is too large");
    }
    const std::size_t count = static_cast<std::size_t>(header.item_count);
    if (count > (std::numeric_limits<std::size_t>::max() - sizeof(BatchHeader)) / sizeof(T)) {
        throw std::runtime_error(std::string(label) + " payload size overflow");
    }
    const std::size_t expected_bytes = sizeof(BatchHeader) + count * sizeof(T);
    if (packet.size() != expected_bytes) {
        throw std::runtime_error(std::string(label) + " byte count does not match header");
    }
    std::vector<T> items(count);
    if (count > 0) {
        std::memcpy(items.data(), packet.data() + sizeof(BatchHeader), count * sizeof(T));
    }
    return items;
}

} // namespace

AsyncExchangeLoop::AsyncExchangeLoop(
    MPI_Comm communicator,
    int rank,
    int world_size,
    DistributedLimitOrderBook& book,
    BackgroundHawkesAgent& background,
    const std::vector<HawkesEvent>& hawkes_events,
    PerformanceMetrics& metrics,
    calibration::SimulationRecorder* recorder)
    : communicator_(communicator),
      rank_(rank),
      world_size_(world_size),
      book_(book),
      background_(background),
      hawkes_events_(hawkes_events),
      metrics_(metrics),
      recorder_(recorder) {
    if (world_size_ <= 0) throw std::invalid_argument("AsyncExchangeLoop requires world_size > 0");
    if (rank_ < 0 || rank_ >= world_size_) throw std::invalid_argument("Invalid rank");
    pending_orders_.reserve(100'000);
}

ExchangeWindowResult AsyncExchangeLoop::run_exchange_window(
    std::uint64_t window_index,
    const MarketState& opening_state,
    std::int64_t window_end_ns,
    double fundamental_value,
    const std::vector<OrderMessage>& local_orders) {
    if (rank_ != 0) throw std::logic_error("run_exchange_window only valid on rank 0");

    std::vector<StatePacket> state_packets;
    std::vector<MPI_Request> state_requests;
    if (world_size_ > 1) {
        state_packets.resize(static_cast<std::size_t>(world_size_ - 1));
        state_requests.resize(static_cast<std::size_t>(world_size_ - 1), MPI_REQUEST_NULL);
        const double start = MPI_Wtime();
        for (int worker = 1; worker < world_size_; ++worker) {
            StatePacket& packet = state_packets[static_cast<std::size_t>(worker - 1)];
            packet.magic = kMarketStateMagic;
            packet.window_index = window_index;
            packet.state = opening_state;
            check_mpi(MPI_Isend(&packet,
                                checked_mpi_byte_count(sizeof(StatePacket), "state packet"),
                                MPI_BYTE, worker, kMarketStateTag, communicator_,
                                &state_requests[static_cast<std::size_t>(worker - 1)]),
                      "MPI_Isend(state)");
        }
        metrics_.add(TimingStage::BroadcastMarketState, MPI_Wtime() - start);
    }

    std::vector<OrderMessage> orders = receive_order_batches(window_index, local_orders, state_requests);
    if (!state_requests.empty()) {
        const double start = MPI_Wtime();
        check_mpi(MPI_Waitall(static_cast<int>(state_requests.size()),
                              state_requests.data(), MPI_STATUSES_IGNORE),
                  "MPI_Waitall(states)");
        metrics_.add(TimingStage::BroadcastMarketState, MPI_Wtime() - start);
    }

    ExchangeWindowResult result = process_exchange_orders(
        window_end_ns, fundamental_value, std::move(orders));

    std::vector<AgentReport> reports = book_.take_reports();
    metrics_.counters().reports_created += reports.size();
    for (const AgentReport& report : reports) {
        if (report.kind == ReportKind::Fill) ++metrics_.counters().fill_reports;
        else ++metrics_.counters().order_result_reports;
    }
    send_report_batches(window_index, reports, result.local_reports);
    result.closing_state = book_.state(window_end_ns, fundamental_value);
    return result;
}

MarketState AsyncExchangeLoop::receive_market_state(std::uint64_t expected_window_index) {
    if (rank_ == 0 || world_size_ == 1) {
        throw std::logic_error("receive_market_state only valid on worker ranks");
    }
    StatePacket packet{};
    MPI_Request request = MPI_REQUEST_NULL;
    const double start = MPI_Wtime();
    check_mpi(MPI_Irecv(&packet,
                        checked_mpi_byte_count(sizeof(StatePacket), "state packet"),
                        MPI_BYTE, 0, kMarketStateTag, communicator_, &request),
              "MPI_Irecv(state)");
    int complete = 0;
    while (!complete) check_mpi(MPI_Test(&request, &complete, MPI_STATUS_IGNORE), "MPI_Test(state)");
    metrics_.add(TimingStage::BroadcastMarketState, MPI_Wtime() - start);
    if (packet.magic != kMarketStateMagic || packet.window_index != expected_window_index) {
        throw std::runtime_error("Invalid or out-of-window market-state packet");
    }
    return packet.state;
}

void AsyncExchangeLoop::post_order_batch(std::uint64_t window_index,
                                         const std::vector<OrderMessage>& orders) {
    if (rank_ == 0 || world_size_ == 1) {
        throw std::logic_error("post_order_batch only valid on worker ranks");
    }
    if (worker_order_send_active_) {
        check_mpi(MPI_Wait(&worker_order_send_request_, MPI_STATUS_IGNORE),
                  "MPI_Wait(previous orders)");
        worker_order_send_active_ = false;
    }
    worker_order_send_buffer_ = make_batch_packet(kOrderBatchMagic, window_index, orders);
    const double start = MPI_Wtime();
    check_mpi(MPI_Isend(worker_order_send_buffer_.data(),
                        checked_mpi_byte_count(worker_order_send_buffer_.size(), "order packet"),
                        MPI_BYTE, 0, kOrderBatchTag, communicator_, &worker_order_send_request_),
              "MPI_Isend(orders)");
    metrics_.add(TimingStage::GatherOrderPayload, MPI_Wtime() - start);
    worker_order_send_active_ = true;
    metrics_.counters().order_bytes_sent += orders.size() * sizeof(OrderMessage);
}

std::vector<AgentReport> AsyncExchangeLoop::receive_report_batch(
    std::uint64_t expected_window_index) {
    if (rank_ == 0 || world_size_ == 1 || !worker_order_send_active_) {
        throw std::logic_error("receive_report_batch called in invalid state");
    }
    const double start = MPI_Wtime();
    MPI_Status status{};
    int available = 0;
    while (!available) {
        check_mpi(MPI_Iprobe(0, kReportBatchTag, communicator_, &available, &status),
                  "MPI_Iprobe(reports)");
        int send_complete = 0;
        check_mpi(MPI_Test(&worker_order_send_request_, &send_complete, MPI_STATUS_IGNORE),
                  "MPI_Test(order send)");
        if (send_complete) worker_order_send_active_ = false;
    }
    int bytes = 0;
    check_mpi(MPI_Get_count(&status, MPI_BYTE, &bytes), "MPI_Get_count(reports)");
    if (bytes < 0) throw std::runtime_error("Negative report byte count");
    std::vector<unsigned char> packet(static_cast<std::size_t>(bytes));
    check_mpi(MPI_Recv(packet.empty() ? nullptr : packet.data(), bytes, MPI_BYTE,
                       0, kReportBatchTag, communicator_, MPI_STATUS_IGNORE),
              "MPI_Recv(reports)");
    if (worker_order_send_active_) {
        check_mpi(MPI_Wait(&worker_order_send_request_, MPI_STATUS_IGNORE),
                  "MPI_Wait(order send)");
        worker_order_send_active_ = false;
    }
    metrics_.add(TimingStage::ScatterReportPayload, MPI_Wtime() - start);
    std::vector<AgentReport> reports = decode_batch_packet<AgentReport>(
        packet, kReportBatchMagic, expected_window_index, "report packet");
    metrics_.counters().report_bytes_received += reports.size() * sizeof(AgentReport);
    metrics_.counters().reports_received += reports.size();
    return reports;
}

std::size_t AsyncExchangeLoop::pending_order_count() const noexcept { return pending_orders_.size(); }
std::size_t AsyncExchangeLoop::peak_pending_order_count() const noexcept { return peak_pending_orders_; }

std::vector<OrderMessage> AsyncExchangeLoop::receive_order_batches(
    std::uint64_t expected_window_index,
    const std::vector<OrderMessage>& local_orders,
    std::vector<MPI_Request>& state_send_requests) {
    std::vector<OrderMessage> received = local_orders;
    if (world_size_ == 1) {
        metrics_.counters().strategic_orders_received_exchange += received.size();
        return received;
    }
    std::vector<unsigned char> got(static_cast<std::size_t>(world_size_), 0);
    got[0] = 1;
    int remaining = world_size_ - 1;
    const double start = MPI_Wtime();
    while (remaining > 0) {
        MPI_Status status{};
        int available = 0;
        check_mpi(MPI_Iprobe(MPI_ANY_SOURCE, kOrderBatchTag, communicator_, &available, &status),
                  "MPI_Iprobe(orders)");
        if (!available) {
            for (MPI_Request& request : state_send_requests) {
                if (request == MPI_REQUEST_NULL) continue;
                int complete = 0;
                check_mpi(MPI_Test(&request, &complete, MPI_STATUS_IGNORE),
                          "MPI_Test(state send)");
            }
            continue;
        }
        const int source = status.MPI_SOURCE;
        if (source <= 0 || source >= world_size_ || got[static_cast<std::size_t>(source)] != 0) {
            throw std::runtime_error("Invalid or duplicate worker order packet");
        }
        int bytes = 0;
        check_mpi(MPI_Get_count(&status, MPI_BYTE, &bytes), "MPI_Get_count(orders)");
        if (bytes < 0) throw std::runtime_error("Negative order byte count");
        std::vector<unsigned char> packet(static_cast<std::size_t>(bytes));
        check_mpi(MPI_Recv(packet.empty() ? nullptr : packet.data(), bytes, MPI_BYTE,
                           source, kOrderBatchTag, communicator_, MPI_STATUS_IGNORE),
                  "MPI_Recv(orders)");
        std::vector<OrderMessage> orders = decode_batch_packet<OrderMessage>(
            packet, kOrderBatchMagic, expected_window_index, "order packet");
        metrics_.counters().order_bytes_received_exchange += orders.size() * sizeof(OrderMessage);
        metrics_.counters().strategic_orders_received_exchange += orders.size();
        received.insert(received.end(), orders.begin(), orders.end());
        got[static_cast<std::size_t>(source)] = 1;
        --remaining;
    }
    metrics_.add(TimingStage::GatherOrderPayload, MPI_Wtime() - start);
    return received;
}

ExchangeWindowResult AsyncExchangeLoop::process_exchange_orders(
    std::int64_t window_end_ns,
    double fundamental_value,
    std::vector<OrderMessage> received_orders) {
    ExchangeWindowResult result;
    {
        ScopedStageTimer timer(metrics_, TimingStage::QueuePartition);
        pending_orders_.insert(pending_orders_.end(), received_orders.begin(), received_orders.end());
        peak_pending_orders_ = std::max(peak_pending_orders_, pending_orders_.size());
        metrics_.counters().peak_pending_orders = std::max<std::uint64_t>(
            metrics_.counters().peak_pending_orders,
            static_cast<std::uint64_t>(pending_orders_.size()));
    }

    std::vector<OrderMessage> due;
    {
        ScopedStageTimer timer(metrics_, TimingStage::QueuePartition);
        std::vector<OrderMessage> future;
        due.reserve(pending_orders_.size());
        future.reserve(pending_orders_.size());
        for (const OrderMessage& message : pending_orders_) {
            if (message.arrival_time_ns < window_end_ns) due.push_back(message);
            else future.push_back(message);
        }
        pending_orders_.swap(future);
    }
    {
        ScopedStageTimer timer(metrics_, TimingStage::EventSort);
        std::sort(due.begin(), due.end(), order_before);
    }
    {
        ScopedStageTimer timer(metrics_, TimingStage::MatchingEngine);
        std::size_t order_index = 0;
        while (order_index < due.size()
               || cached_background_message_.has_value()
               || (next_hawkes_ < hawkes_events_.size()
                   && hawkes_events_[next_hawkes_].time_ns < window_end_ns)) {
            const bool have_order = order_index < due.size();
            const bool have_hawkes = next_hawkes_ < hawkes_events_.size()
                && hawkes_events_[next_hawkes_].time_ns < window_end_ns;

            if (!cached_background_message_.has_value() && have_hawkes
                && (!have_order || hawkes_events_[next_hawkes_].time_ns
                    <= due[order_index].arrival_time_ns)) {
                const HawkesEvent& event = hawkes_events_[next_hawkes_];
                cached_background_message_ = background_.make_order(
                    event, book_.state(event.time_ns, fundamental_value), background_sequence_++);
            }
            const bool background_due = cached_background_message_.has_value();
            if (background_due
                && (!have_order || order_before(*cached_background_message_, due[order_index]))) {
                if (recorder_ != nullptr) recorder_->observe_order(*cached_background_message_);
                book_.apply(*cached_background_message_);
                cached_background_message_.reset();
                ++next_hawkes_;
                ++metrics_.counters().background_orders_processed;
            } else if (have_order) {
                if (recorder_ != nullptr) recorder_->observe_order(due[order_index]);
                book_.apply(due[order_index++]);
                ++metrics_.counters().strategic_orders_processed;
            } else if (have_hawkes) {
                const HawkesEvent& event = hawkes_events_[next_hawkes_];
                cached_background_message_ = background_.make_order(
                    event, book_.state(event.time_ns, fundamental_value), background_sequence_++);
            } else {
                break;
            }
        }
    }
    return result;
}

void AsyncExchangeLoop::send_report_batches(
    std::uint64_t window_index,
    const std::vector<AgentReport>& reports,
    std::vector<AgentReport>& local_reports) {
    std::vector<std::vector<AgentReport>> by_rank(static_cast<std::size_t>(world_size_));
    {
        ScopedStageTimer timer(metrics_, TimingStage::ReportPacking);
        for (const AgentReport& report : reports) {
            const int destination = owner_rank(report.owner_id);
            if (world_size_ == 1 && destination == 0) by_rank[0].push_back(report);
            else if (destination > 0 && destination < world_size_) {
                by_rank[static_cast<std::size_t>(destination)].push_back(report);
            }
        }
    }
    local_reports = std::move(by_rank[0]);
    if (world_size_ == 1) return;

    std::vector<std::vector<unsigned char>> packets(static_cast<std::size_t>(world_size_));
    std::vector<MPI_Request> requests(static_cast<std::size_t>(world_size_ - 1), MPI_REQUEST_NULL);
    const double start = MPI_Wtime();
    for (int worker = 1; worker < world_size_; ++worker) {
        const auto& worker_reports = by_rank[static_cast<std::size_t>(worker)];
        packets[static_cast<std::size_t>(worker)] = make_batch_packet(
            kReportBatchMagic, window_index, worker_reports);
        metrics_.counters().report_bytes_sent_exchange += worker_reports.size() * sizeof(AgentReport);
        auto& packet = packets[static_cast<std::size_t>(worker)];
        check_mpi(MPI_Isend(packet.data(), checked_mpi_byte_count(packet.size(), "report packet"),
                            MPI_BYTE, worker, kReportBatchTag, communicator_,
                            &requests[static_cast<std::size_t>(worker - 1)]),
                  "MPI_Isend(reports)");
    }
    check_mpi(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE),
              "MPI_Waitall(reports)");
    metrics_.add(TimingStage::ScatterReportPayload, MPI_Wtime() - start);
}

} // namespace dlob
