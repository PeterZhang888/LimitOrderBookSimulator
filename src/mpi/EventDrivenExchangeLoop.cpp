// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "mpi/EventDrivenExchangeLoop.hpp"

#include "common/PerformanceMetrics.hpp"
#include "exchange/EventOrdering.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace dlob {
namespace {

constexpr int kActivationTag = 5101;
constexpr int kWorkerBatchTag = 5102;
constexpr std::uint64_t kActivationMagic = 0x455644524143544EULL;
constexpr std::uint64_t kWorkerBatchMagic = 0x4556445257424154ULL;

enum class ControlKind : std::int32_t { Activate = 1, Stop = 2 };

struct ActivationHeader {
    std::uint64_t magic = kActivationMagic;
    std::uint64_t activation_id = 0;
    ControlKind kind = ControlKind::Activate;
    std::int32_t state_inline = 1;
    std::int64_t activation_time_ns = 0;
    std::int64_t cutoff_time_ns = 0;
    std::uint64_t snapshot_version = 0;
    std::uint64_t report_count = 0;
};

struct WorkerBatchHeader {
    std::uint64_t magic = kWorkerBatchMagic;
    std::uint64_t activation_id = 0;
    std::int64_t next_wake_ns = no_wake_time;
    std::int64_t batch_horizon_ns = 0;
    std::uint64_t order_count = 0;
};

static_assert(std::is_trivially_copyable_v<ActivationHeader>);
static_assert(std::is_trivially_copyable_v<WorkerBatchHeader>);

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

int checked_count(std::size_t bytes, const char* label) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds MPI int count");
    }
    return static_cast<int>(bytes);
}

template <typename Header, typename Item>
std::vector<unsigned char> make_packet(const Header& header,
                                       const std::vector<Item>& items) {
    static_assert(std::is_trivially_copyable_v<Header>);
    static_assert(std::is_trivially_copyable_v<Item>);
    const std::size_t bytes = sizeof(Header) + items.size() * sizeof(Item);
    std::vector<unsigned char> packet(bytes);
    std::memcpy(packet.data(), &header, sizeof(Header));
    if (!items.empty()) {
        std::memcpy(packet.data() + sizeof(Header), items.data(), items.size() * sizeof(Item));
    }
    return packet;
}

template <typename Header, typename Item>
std::pair<Header, std::vector<Item>> decode_packet(const std::vector<unsigned char>& packet,
                                                   std::uint64_t expected_magic,
                                                   std::uint64_t item_count,
                                                   const char* label) {
    if (packet.size() < sizeof(Header)) {
        throw std::runtime_error(std::string(label) + " is smaller than its header");
    }
    Header header{};
    std::memcpy(&header, packet.data(), sizeof(Header));
    if (header.magic != expected_magic) {
        throw std::runtime_error(std::string(label) + " has invalid magic");
    }
    if (item_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string(label) + " item count overflow");
    }
    const std::size_t count = static_cast<std::size_t>(item_count);
    const std::size_t expected = sizeof(Header) + count * sizeof(Item);
    if (packet.size() != expected) {
        throw std::runtime_error(std::string(label) + " byte count mismatch");
    }
    std::vector<Item> items(count);
    if (count > 0) {
        std::memcpy(items.data(), packet.data() + sizeof(Header), count * sizeof(Item));
    }
    return {header, std::move(items)};
}

struct DecodedActivation {
    ActivationHeader header{};
    MarketState state{};
    std::vector<AgentReport> reports;
};

std::vector<unsigned char> make_activation_packet(
    const ActivationHeader& header,
    const MarketState& state,
    const std::vector<AgentReport>& reports) {
    const std::size_t state_bytes = header.state_inline != 0 ? sizeof(MarketState) : 0;
    const std::size_t bytes = sizeof(ActivationHeader)
        + state_bytes + reports.size() * sizeof(AgentReport);
    std::vector<unsigned char> packet(bytes);
    std::size_t offset = 0;
    std::memcpy(packet.data() + offset, &header, sizeof(header));
    offset += sizeof(header);
    if (state_bytes > 0) {
        std::memcpy(packet.data() + offset, &state, sizeof(state));
        offset += sizeof(state);
    }
    if (!reports.empty()) {
        std::memcpy(packet.data() + offset, reports.data(),
                    reports.size() * sizeof(AgentReport));
    }
    return packet;
}

DecodedActivation decode_activation_packet(
    const std::vector<unsigned char>& packet,
    const char* label) {
    if (packet.size() < sizeof(ActivationHeader)) {
        throw std::runtime_error(std::string(label) + " is smaller than its header");
    }
    DecodedActivation decoded;
    std::memcpy(&decoded.header, packet.data(), sizeof(decoded.header));
    if (decoded.header.magic != kActivationMagic) {
        throw std::runtime_error(std::string(label) + " has invalid magic");
    }
    if (decoded.header.report_count
        > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string(label) + " report count overflow");
    }
    const std::size_t report_count =
        static_cast<std::size_t>(decoded.header.report_count);
    const std::size_t state_bytes =
        decoded.header.state_inline != 0 ? sizeof(MarketState) : 0;
    const std::size_t expected = sizeof(ActivationHeader)
        + state_bytes + report_count * sizeof(AgentReport);
    if (packet.size() != expected) {
        throw std::runtime_error(std::string(label) + " byte count mismatch");
    }
    std::size_t offset = sizeof(ActivationHeader);
    if (state_bytes > 0) {
        std::memcpy(&decoded.state, packet.data() + offset, sizeof(MarketState));
        offset += sizeof(MarketState);
    }
    decoded.reports.resize(report_count);
    if (report_count > 0) {
        std::memcpy(decoded.reports.data(), packet.data() + offset,
                    report_count * sizeof(AgentReport));
    }
    return decoded;
}

std::vector<unsigned char> receive_packet(MPI_Comm communicator, int source, int tag) {
    MPI_Status status{};
    check_mpi(MPI_Probe(source, tag, communicator, &status), "MPI_Probe");
    int bytes = 0;
    check_mpi(MPI_Get_count(&status, MPI_BYTE, &bytes), "MPI_Get_count");
    if (bytes < 0) throw std::runtime_error("Negative MPI packet byte count");
    std::vector<unsigned char> packet(static_cast<std::size_t>(bytes));
    check_mpi(MPI_Recv(packet.empty() ? nullptr : packet.data(), bytes, MPI_BYTE,
                       source, tag, communicator, MPI_STATUS_IGNORE),
              "MPI_Recv(packet)");
    return packet;
}

void send_packet(MPI_Comm communicator,
                 int destination,
                 int tag,
                 const std::vector<unsigned char>& packet) {
    check_mpi(MPI_Send(packet.empty() ? nullptr : packet.data(),
                       checked_count(packet.size(), "MPI packet"),
                       MPI_BYTE, destination, tag, communicator),
              "MPI_Send(packet)");
}

std::int64_t safe_cutoff(std::int64_t start,
                         std::int64_t horizon,
                         std::int64_t end_time) {
    if (start >= end_time || horizon <= 0) return std::min(start, end_time);
    const std::int64_t remaining = end_time - start;
    return horizon >= remaining ? end_time : start + horizon;
}

} // namespace

bool EventDrivenExchangeLoop::OrderLater::operator()(const OrderMessage& left,
                                                     const OrderMessage& right) const {
    if (order_before(left, right)) return false;
    if (order_before(right, left)) return true;
    return false;
}

EventDrivenExchangeLoop::EventDrivenExchangeLoop(
    MPI_Comm communicator,
    int rank,
    int world_size,
    DistributedLimitOrderBook& book,
    BackgroundHawkesAgent& background,
    const std::vector<HawkesEvent>& hawkes_events,
    AgentPopulation& population,
    PerformanceMetrics& metrics,
    calibration::SimulationRecorder* recorder,
    SharedMarketSnapshot& shared_snapshot,
    std::int64_t end_time_ns,
    int tick_size,
    std::uint64_t seed,
    double max_wall_seconds)
    : communicator_(communicator),
      rank_(rank),
      world_size_(world_size),
      book_(book),
      background_(background),
      hawkes_events_(hawkes_events),
      population_(population),
      metrics_(metrics),
      recorder_(recorder),
      shared_snapshot_(shared_snapshot),
      end_time_ns_(end_time_ns),
      tick_size_(std::max(1, tick_size)),
      fundamental_rng_(seed + 0xABCDEFULL),
      max_wall_seconds_(std::max(0.0, max_wall_seconds)) {}

EventDrivenRunResult EventDrivenExchangeLoop::run() {
    if (world_size_ == 1) return run_single_process();
    return rank_ == 0 ? run_exchange() : run_worker();
}

void EventDrivenExchangeLoop::advance_fundamental(std::int64_t target_time_ns) {
    if (target_time_ns <= fundamental_time_ns_) return;
    const double duration_ms = static_cast<double>(target_time_ns - fundamental_time_ns_) / 1e6;
    fundamental_value_ += fundamental_shock_(fundamental_rng_)
        * std::sqrt(std::max(0.0, duration_ms)) * tick_size_;
    fundamental_time_ns_ = target_time_ns;
}

void EventDrivenExchangeLoop::accumulate_reports(
    std::vector<std::vector<AgentReport>>& pending_reports) {
    std::vector<AgentReport> reports = book_.take_reports();
    metrics_.counters().reports_created += reports.size();
    for (const AgentReport& report : reports) {
        if (report.kind == ReportKind::Fill) ++metrics_.counters().fill_reports;
        else ++metrics_.counters().order_result_reports;
        const int destination = owner_rank(report.owner_id);
        if (destination >= 0 && destination < static_cast<int>(pending_reports.size())) {
            pending_reports[static_cast<std::size_t>(destination)].push_back(report);
        }
    }
}

void EventDrivenExchangeLoop::apply_market_event(
    const OrderMessage& message,
    std::vector<std::vector<AgentReport>>& pending_reports) {
    if (recorder_ != nullptr) recorder_->observe_order(message);
    {
        ScopedStageTimer timer(metrics_, TimingStage::MatchingEngine);
        book_.apply(message);
    }
    if (message.agent_kind == AgentKind::Background) {
        ++metrics_.counters().background_orders_processed;
    } else {
        ++metrics_.counters().strategic_orders_processed;
    }
    accumulate_reports(pending_reports);
}

EventDrivenRunResult EventDrivenExchangeLoop::run_worker() {
    WorkerBatchHeader initial;
    initial.activation_id = 0;
    initial.next_wake_ns = population_.next_wake_time();
    initial.batch_horizon_ns = population_.batch_horizon_ns();
    initial.order_count = 0;
    send_packet(communicator_, 0, kWorkerBatchTag,
                make_packet<WorkerBatchHeader, OrderMessage>(initial, {}));
    ++metrics_.counters().order_batches_sent;

    EventDrivenRunResult result;
    for (;;) {
        const double receive_start = MPI_Wtime();
        const std::vector<unsigned char> packet = receive_packet(communicator_, 0, kActivationTag);
        metrics_.add(TimingStage::ActivationReceive, MPI_Wtime() - receive_start);

        DecodedActivation decoded = decode_activation_packet(packet, "activation packet");
        const ActivationHeader& header = decoded.header;
        std::vector<AgentReport>& reports = decoded.reports;
        metrics_.counters().activation_bytes_received += packet.size();
        metrics_.counters().reports_received += reports.size();
        metrics_.counters().report_bytes_received += reports.size() * sizeof(AgentReport);
        if (!reports.empty()) {
            ScopedStageTimer timer(metrics_, TimingStage::ApplyReports);
            population_.apply_reports(reports);
        }

        if (header.kind == ControlKind::Stop) {
            result.closing_state = decoded.state;
            break;
        }

        MarketState state{};
        if (header.state_inline != 0) {
            state = decoded.state;
        } else {
            const double start = MPI_Wtime();
            state = shared_snapshot_.read(header.snapshot_version);
            metrics_.add(TimingStage::SharedSnapshotRead, MPI_Wtime() - start);
            ++metrics_.counters().shared_snapshot_reads;
        }

        std::vector<OrderMessage> orders;
        {
            ScopedStageTimer timer(metrics_, TimingStage::AgentObserveAndGenerate);
            population_.observe_market(state);
            orders = population_.generate_due_orders(header.activation_time_ns,
                                                      header.cutoff_time_ns);
        }
        metrics_.counters().strategic_orders_generated += orders.size();
        ++metrics_.counters().worker_activations_received;
        ++result.activations;

        WorkerBatchHeader response;
        response.activation_id = header.activation_id;
        response.next_wake_ns = population_.next_wake_time();
        response.batch_horizon_ns = population_.batch_horizon_ns();
        response.order_count = orders.size();
        const std::vector<unsigned char> response_packet =
            make_packet<WorkerBatchHeader, OrderMessage>(response, orders);
        const double send_start = MPI_Wtime();
        send_packet(communicator_, 0, kWorkerBatchTag, response_packet);
        metrics_.add(TimingStage::OrderBatchSend, MPI_Wtime() - send_start);
        metrics_.counters().order_bytes_sent += orders.size() * sizeof(OrderMessage);
        metrics_.counters().order_batches_sent++;
        if (orders.empty()) ++metrics_.counters().empty_order_batches;
    }
    return result;
}

EventDrivenRunResult EventDrivenExchangeLoop::run_exchange() {
    std::vector<std::int64_t> next_wake(static_cast<std::size_t>(world_size_), no_wake_time);
    std::vector<std::int64_t> batch_horizon(static_cast<std::size_t>(world_size_), 0);
    std::vector<std::vector<AgentReport>> pending_reports(static_cast<std::size_t>(world_size_));

    for (int worker = 1; worker < world_size_; ++worker) {
        const std::vector<unsigned char> packet = receive_packet(communicator_, worker, kWorkerBatchTag);
        WorkerBatchHeader raw{};
        if (packet.size() < sizeof(raw)) throw std::runtime_error("Initial worker packet too small");
        std::memcpy(&raw, packet.data(), sizeof(raw));
        auto [header, orders] = decode_packet<WorkerBatchHeader, OrderMessage>(
            packet, kWorkerBatchMagic, raw.order_count, "initial worker packet");
        if (header.activation_id != 0 || !orders.empty()) {
            throw std::runtime_error("Invalid initial worker schedule packet");
        }
        next_wake[static_cast<std::size_t>(worker)] = header.next_wake_ns;
        batch_horizon[static_cast<std::size_t>(worker)] = std::max<std::int64_t>(0, header.batch_horizon_ns);
        ++metrics_.counters().order_batches_received_exchange;
    }

    std::int64_t next_sample_ns = 1'000'000'000LL;
    std::int64_t current_time_ns = 0;
    const double run_wall_start = MPI_Wtime();
    EventDrivenRunResult result;

    auto next_strategic_time = [this]() {
        return pending_orders_.empty() ? no_wake_time : pending_orders_.top().arrival_time_ns;
    };
    auto next_hawkes_time = [this]() {
        return next_hawkes_ < hawkes_events_.size()
            ? hawkes_events_[next_hawkes_].time_ns : no_wake_time;
    };
    auto next_worker_time = [&next_wake, this]() {
        std::int64_t value = no_wake_time;
        for (int worker = 1; worker < world_size_; ++worker) {
            value = std::min(value, next_wake[static_cast<std::size_t>(worker)]);
        }
        return value;
    };

    auto process_market_events_at = [&](std::int64_t time_ns) {
        for (;;) {
            const bool strategic_due = !pending_orders_.empty()
                && pending_orders_.top().arrival_time_ns == time_ns;
            const bool hawkes_due = next_hawkes_ < hawkes_events_.size()
                && hawkes_events_[next_hawkes_].time_ns == time_ns;
            if (!strategic_due && !hawkes_due) break;

            if (hawkes_due && !cached_background_.has_value()) {
                cached_background_ = background_.make_order(
                    hawkes_events_[next_hawkes_],
                    book_.state(time_ns, fundamental_value_),
                    background_sequence_++);
            }

            if (hawkes_due && (!strategic_due
                || order_before(*cached_background_, pending_orders_.top()))) {
                const OrderMessage background_message = *cached_background_;
                cached_background_.reset();
                ++next_hawkes_;
                apply_market_event(background_message, pending_reports);
            } else {
                const OrderMessage strategic = pending_orders_.top();
                pending_orders_.pop();
                apply_market_event(strategic, pending_reports);
            }
        }
    };

    while (true) {
        if (max_wall_seconds_ > 0.0
            && MPI_Wtime() - run_wall_start >= max_wall_seconds_) {
            result.terminated_early = true;
            result.termination_reason = "wall_time_limit";
            break;
        }

        const std::int64_t market_time = std::min(next_strategic_time(), next_hawkes_time());
        const std::int64_t worker_time = next_worker_time();
        const std::int64_t sample_time = next_sample_ns <= end_time_ns_
            ? next_sample_ns : no_wake_time;
        std::int64_t next_time = std::min({market_time, worker_time, sample_time, end_time_ns_});
        if (next_time == no_wake_time) break;
        if (next_time > end_time_ns_) next_time = end_time_ns_;

        current_time_ns = next_time;
        advance_fundamental(next_time);
        process_market_events_at(next_time);

        std::vector<int> active_workers;
        for (int worker = 1; worker < world_size_; ++worker) {
            if (next_wake[static_cast<std::size_t>(worker)] == next_time
                && next_time <= end_time_ns_) {
                active_workers.push_back(worker);
            }
        }

        if (!active_workers.empty()) {
            const MarketState state = book_.state(next_time, fundamental_value_);
            std::uint64_t snapshot_version = 0;
            if (shared_snapshot_.enabled()) {
                const double start = MPI_Wtime();
                snapshot_version = shared_snapshot_.publish(state);
                metrics_.add(TimingStage::SharedSnapshotPublish, MPI_Wtime() - start);
                ++metrics_.counters().shared_snapshot_publishes;
            }

            const std::uint64_t activation_id = activation_sequence_++;
            for (int worker : active_workers) {
                ActivationHeader header;
                header.activation_id = activation_id;
                header.kind = ControlKind::Activate;
                header.state_inline = shared_snapshot_.enabled() ? 0 : 1;
                header.activation_time_ns = next_time;
                header.cutoff_time_ns = safe_cutoff(
                    next_time,
                    batch_horizon[static_cast<std::size_t>(worker)],
                    end_time_ns_);
                header.snapshot_version = snapshot_version;
                header.report_count = pending_reports[static_cast<std::size_t>(worker)].size();
                const auto packet = make_activation_packet(
                    header, state, pending_reports[static_cast<std::size_t>(worker)]);
                metrics_.counters().report_bytes_sent_exchange +=
                    pending_reports[static_cast<std::size_t>(worker)].size() * sizeof(AgentReport);
                pending_reports[static_cast<std::size_t>(worker)].clear();
                const double start = MPI_Wtime();
                send_packet(communicator_, worker, kActivationTag, packet);
                metrics_.add(TimingStage::ActivationSend, MPI_Wtime() - start);
                metrics_.counters().activation_bytes_sent_exchange += packet.size();
                ++metrics_.counters().worker_activations_sent;
            }

            for (int worker : active_workers) {
                const double start = MPI_Wtime();
                const std::vector<unsigned char> packet =
                    receive_packet(communicator_, worker, kWorkerBatchTag);
                metrics_.add(TimingStage::OrderBatchReceive, MPI_Wtime() - start);
                WorkerBatchHeader raw{};
                if (packet.size() < sizeof(raw)) throw std::runtime_error("Worker batch too small");
                std::memcpy(&raw, packet.data(), sizeof(raw));
                auto [header, orders] = decode_packet<WorkerBatchHeader, OrderMessage>(
                    packet, kWorkerBatchMagic, raw.order_count, "worker batch");
                if (header.activation_id != activation_id) {
                    throw std::runtime_error("Worker batch activation id mismatch");
                }
                next_wake[static_cast<std::size_t>(worker)] = header.next_wake_ns;
                batch_horizon[static_cast<std::size_t>(worker)] =
                    std::max<std::int64_t>(0, header.batch_horizon_ns);
                metrics_.counters().order_bytes_received_exchange +=
                    orders.size() * sizeof(OrderMessage);
                metrics_.counters().strategic_orders_received_exchange += orders.size();
                ++metrics_.counters().order_batches_received_exchange;
                if (orders.empty()) ++metrics_.counters().empty_order_batches;
                for (const OrderMessage& order : orders) pending_orders_.push(order);
                peak_pending_orders_ = std::max(peak_pending_orders_, pending_orders_.size());
                metrics_.counters().peak_pending_orders = std::max<std::uint64_t>(
                    metrics_.counters().peak_pending_orders,
                    static_cast<std::uint64_t>(pending_orders_.size()));
            }
            result.activations += active_workers.size();
            metrics_.counters().windows += active_workers.size();

            // Latency can be configured as zero, so process any newly received
            // orders whose arrival time is exactly the activation time.
            process_market_events_at(next_time);
        }

        if (next_sample_ns == next_time && next_sample_ns <= end_time_ns_) {
            if (recorder_ != nullptr) {
                recorder_->observe_state(book_.state(next_time, fundamental_value_));
            }
            next_sample_ns += 1'000'000'000LL;
        }

        if (next_time == end_time_ns_) break;
    }

    const std::int64_t final_time_ns = result.terminated_early
        ? current_time_ns : end_time_ns_;
    advance_fundamental(final_time_ns);
    const MarketState closing = book_.state(final_time_ns, fundamental_value_);
    for (int worker = 1; worker < world_size_; ++worker) {
        ActivationHeader header;
        header.activation_id = activation_sequence_++;
        header.kind = ControlKind::Stop;
        header.state_inline = 1;
        header.activation_time_ns = final_time_ns;
        header.cutoff_time_ns = final_time_ns;
        header.report_count = pending_reports[static_cast<std::size_t>(worker)].size();
        const auto packet = make_activation_packet(
            header, closing, pending_reports[static_cast<std::size_t>(worker)]);
        metrics_.counters().report_bytes_sent_exchange +=
            pending_reports[static_cast<std::size_t>(worker)].size() * sizeof(AgentReport);
        send_packet(communicator_, worker, kActivationTag, packet);
        metrics_.counters().activation_bytes_sent_exchange += packet.size();
    }

    result.closing_state = closing;
    result.final_time_ns = final_time_ns;
    result.pending_orders = pending_orders_.size();
    result.peak_pending_orders = peak_pending_orders_;
    return result;
}

EventDrivenRunResult EventDrivenExchangeLoop::run_single_process() {
    std::vector<std::vector<AgentReport>> pending_reports(1);
    std::int64_t next_sample_ns = 1'000'000'000LL;
    std::int64_t current_time_ns = 0;
    const double run_wall_start = MPI_Wtime();
    EventDrivenRunResult result;

    auto process_market_events_at = [&](std::int64_t time_ns) {
        for (;;) {
            const bool strategic_due = !pending_orders_.empty()
                && pending_orders_.top().arrival_time_ns == time_ns;
            const bool hawkes_due = next_hawkes_ < hawkes_events_.size()
                && hawkes_events_[next_hawkes_].time_ns == time_ns;
            if (!strategic_due && !hawkes_due) break;
            if (hawkes_due && !cached_background_.has_value()) {
                cached_background_ = background_.make_order(
                    hawkes_events_[next_hawkes_],
                    book_.state(time_ns, fundamental_value_),
                    background_sequence_++);
            }
            if (hawkes_due && (!strategic_due
                || order_before(*cached_background_, pending_orders_.top()))) {
                const OrderMessage message = *cached_background_;
                cached_background_.reset();
                ++next_hawkes_;
                apply_market_event(message, pending_reports);
            } else {
                const OrderMessage message = pending_orders_.top();
                pending_orders_.pop();
                apply_market_event(message, pending_reports);
            }
        }
    };

    while (true) {
        if (max_wall_seconds_ > 0.0
            && MPI_Wtime() - run_wall_start >= max_wall_seconds_) {
            result.terminated_early = true;
            result.termination_reason = "wall_time_limit";
            break;
        }

        const std::int64_t strategic_time = pending_orders_.empty()
            ? no_wake_time : pending_orders_.top().arrival_time_ns;
        const std::int64_t hawkes_time = next_hawkes_ < hawkes_events_.size()
            ? hawkes_events_[next_hawkes_].time_ns : no_wake_time;
        const std::int64_t worker_time = population_.next_wake_time();
        const std::int64_t sample_time = next_sample_ns <= end_time_ns_
            ? next_sample_ns : no_wake_time;
        std::int64_t next_time = std::min({strategic_time, hawkes_time, worker_time,
                                           sample_time, end_time_ns_});
        current_time_ns = next_time;
        advance_fundamental(next_time);
        process_market_events_at(next_time);

        if (worker_time == next_time && next_time <= end_time_ns_) {
            if (!pending_reports[0].empty()) {
                population_.apply_reports(pending_reports[0]);
                metrics_.counters().reports_received += pending_reports[0].size();
                pending_reports[0].clear();
            }
            const MarketState state = book_.state(next_time, fundamental_value_);
            population_.observe_market(state);
            const std::int64_t cutoff = safe_cutoff(
                next_time, population_.batch_horizon_ns(), end_time_ns_);
            std::vector<OrderMessage> orders = population_.generate_due_orders(next_time, cutoff);
            metrics_.counters().strategic_orders_generated += orders.size();
            metrics_.counters().strategic_orders_received_exchange += orders.size();
            for (const OrderMessage& order : orders) pending_orders_.push(order);
            peak_pending_orders_ = std::max(peak_pending_orders_, pending_orders_.size());
            ++result.activations;
            ++metrics_.counters().worker_activations_sent;
            process_market_events_at(next_time);
        }

        if (next_sample_ns == next_time && next_sample_ns <= end_time_ns_) {
            if (recorder_ != nullptr) {
                recorder_->observe_state(book_.state(next_time, fundamental_value_));
            }
            next_sample_ns += 1'000'000'000LL;
        }
        if (next_time == end_time_ns_) break;
    }

    if (!pending_reports[0].empty()) population_.apply_reports(pending_reports[0]);
    const std::int64_t final_time_ns = result.terminated_early
        ? current_time_ns : end_time_ns_;
    result.final_time_ns = final_time_ns;
    result.closing_state = book_.state(final_time_ns, fundamental_value_);
    result.pending_orders = pending_orders_.size();
    result.peak_pending_orders = peak_pending_orders_;
    return result;
}

} // namespace dlob
