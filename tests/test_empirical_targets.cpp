// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "calibration/EmpiricalTargets.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>

int main() {
    const auto root = std::filesystem::temp_directory_path() / "dlob_empirical_target_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const char* names[] = {
        "limit_buy_quantity_distribution.txt", "limit_sell_quantity_distribution.txt",
        "market_buy_quantity_distribution.txt", "market_sell_quantity_distribution.txt",
        "cancel_bid_quantity_distribution.txt", "cancel_ask_quantity_distribution.txt"
    };
    for (const char* name : names) {
        std::ofstream out(root / name);
        out << "quantity,count,probability,cumulative_probability\n"
            << "1,1,0.25,0.25\n"
            << "2,3,0.75,1.0\n";
    }
    const auto targets = dlob::calibration::EmpiricalTargets::load(root);
    dlob::calibration::SimulationRecord record;
    for (std::size_t i = 0; i < dlob::calibration::empirical_event_bucket_count; ++i) {
        record.quantity_samples[i] = {1, 2, 2, 2};
        record.event_counts[i] = 10;
    }
    record.market.snapshots = 1;
    const auto distance = targets.distance(record);
    assert(distance.total >= 0.0);
    assert(distance.total < 0.01);
    std::filesystem::remove_all(root);
    return 0;
}
