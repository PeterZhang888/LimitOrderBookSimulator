#include "SimulationConfig.hpp"
#include "SimulationRunner.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

struct MCRow {
    int path_id = 0;
    unsigned long seed = 0;

    double total_loss = 0.0;
    double mean_spread = 0.0;
    double median_spread = 0.0;
    double p90_spread = 0.0;
    double p95_spread = 0.0;

    double mean_mid_change = 0.0;
    double std_mid_change = 0.0;
    double zero_mid_change_ratio = 0.0;
    double mid_move_rate = 0.0;
    double final_norm_mid = 0.0;
};

static double mean_value(
    const std::vector<double>& values
) {
    if (values.empty()) {
        return 0.0;
    }

    double total = 0.0;

    for (double value : values) {
        total += value;
    }

    return total / static_cast<double>(values.size());
}

static double standard_deviation(
    const std::vector<double>& values
) {
    if (values.size() < 2) {
        return 0.0;
    }

    const double average = mean_value(values);

    double squared_sum = 0.0;

    for (double value : values) {
        const double difference = value - average;
        squared_sum += difference * difference;
    }

    return std::sqrt(
        squared_sum / static_cast<double>(values.size() - 1)
    );
}

static double minimum_value(
    const std::vector<double>& values
) {
    if (values.empty()) {
        return 0.0;
    }

    return *std::min_element(values.begin(), values.end());
}

static double maximum_value(
    const std::vector<double>& values
) {
    if (values.empty()) {
        return 0.0;
    }

    return *std::max_element(values.begin(), values.end());
}

static SimulationConfig make_full_day_config(
    int path_id,
    unsigned long seed
) {
    SimulationConfig config;

    config.calibration_id = path_id;

    config.calibration_name =
        "mixed_side_quote_improvement_path_"
        + std::to_string(path_id);

    config.seed = seed;

    config.start_time_ns = 0;
    config.end_time_ns = 23400LL * 1000000000LL;
    config.sample_interval_ns = 100000000LL;

    config.tick_size = 100;

    config.hawkes_beta = 5.0;
    config.limit_branching = 0.30;
    config.market_branching = 0.45;
    config.cancel_branching = 0.08;

    config.num_market_maker_agents = 5;

    config.market_maker_order_quantity = 250;
    config.market_maker_min_spread_ticks = 1;
    config.market_maker_num_levels = 5;
    config.market_maker_level_spacing_ticks = 1;

    config.market_maker_interval_ns = 750LL * 1000000LL;

    config.heterogeneous_market_makers = true;

    config.market_maker_num_levels_step = -1;
    config.market_maker_order_quantity_step = -25;
    config.market_maker_min_spread_ticks_step = 1;

    config.market_maker_min_num_levels = 1;
    config.market_maker_min_order_quantity = 25;

    config.num_momentum_agents = 3;
    config.momentum_order_quantity = 300;
    config.momentum_threshold_ticks = 0.5;

    config.momentum_interval_ns = 1LL * 1000000000LL;
    config.momentum_base_lookback_ns = 2LL * 1000000000LL;
    config.momentum_lookback_step_ns = 3LL * 1000000000LL;

    config.heterogeneous_momentum_agents = true;
    config.momentum_order_quantity_step = 0;
    config.momentum_threshold_step_ticks = 0.0;

    config.use_mixed_institutional_sides = true;

    config.num_buy_institutional_agents = 4;
    config.num_sell_institutional_agents = 2;
    config.num_institutional_agents = 6;

    config.buy_institutional_parent_quantity = 330000;
    config.sell_institutional_parent_quantity = 350000;

    config.institutional_child_quantity = 1200;

    config.institutional_interval_ns = 1LL * 1000000000LL;
    config.institutional_base_start_time_ns = 0LL;
    config.institutional_start_spacing_ns = 150LL * 1000000000LL;
    config.institutional_duration_ns = 500LL * 1000000000LL;

    config.heterogeneous_institutional_agents = true;

    config.institutional_parent_quantity_step = 0;
    config.institutional_child_quantity_step = 0;

    config.limit_distance_shift_ticks = 0;
    config.market_order_quantity_scale = 1.0;

    config.quote_improvement_probability = 0.10;

    config.write_detailed_outputs = false;

    return config;
}

static void write_result_row(
    std::ofstream& output,
    int path_id,
    unsigned long seed,
    const CalibrationResult& result
) {
    output
        << path_id << ","
        << seed << ","
        << result.total_loss << ","
        << result.mean_spread << ","
        << result.median_spread << ","
        << result.p90_spread << ","
        << result.p95_spread << ","
        << result.mean_mid_change << ","
        << result.std_mid_change << ","
        << result.zero_mid_change_ratio << ","
        << result.mid_move_rate << ","
        << result.final_norm_mid << "\n";
}

static bool parse_row(
    const std::string& line,
    MCRow& row
) {
    if (
        line.empty() ||
        line.find("path_id") != std::string::npos
    ) {
        return false;
    }

    std::stringstream stream(line);
    std::string value;
    std::vector<std::string> fields;

    while (std::getline(stream, value, ',')) {
        fields.push_back(value);
    }

    if (fields.size() != 12) {
        return false;
    }

    row.path_id = std::stoi(fields[0]);
    row.seed = std::stoul(fields[1]);
    row.total_loss = std::stod(fields[2]);
    row.mean_spread = std::stod(fields[3]);
    row.median_spread = std::stod(fields[4]);
    row.p90_spread = std::stod(fields[5]);
    row.p95_spread = std::stod(fields[6]);
    row.mean_mid_change = std::stod(fields[7]);
    row.std_mid_change = std::stod(fields[8]);
    row.zero_mid_change_ratio = std::stod(fields[9]);
    row.mid_move_rate = std::stod(fields[10]);
    row.final_norm_mid = std::stod(fields[11]);

    return true;
}

static void write_summary(
    const std::vector<MCRow>& rows,
    const std::string& filename
) {
    std::vector<double> total_loss;
    std::vector<double> mean_spread;
    std::vector<double> median_spread;
    std::vector<double> p90_spread;
    std::vector<double> p95_spread;
    std::vector<double> mean_mid_change;
    std::vector<double> std_mid_change;
    std::vector<double> zero_mid_change_ratio;
    std::vector<double> mid_move_rate;
    std::vector<double> final_norm_mid;

    for (const MCRow& row : rows) {
        total_loss.push_back(row.total_loss);
        mean_spread.push_back(row.mean_spread);
        median_spread.push_back(row.median_spread);
        p90_spread.push_back(row.p90_spread);
        p95_spread.push_back(row.p95_spread);
        mean_mid_change.push_back(row.mean_mid_change);
        std_mid_change.push_back(row.std_mid_change);
        zero_mid_change_ratio.push_back(row.zero_mid_change_ratio);
        mid_move_rate.push_back(row.mid_move_rate);
        final_norm_mid.push_back(row.final_norm_mid);
    }

    std::ofstream output(filename);

    output << "metric,mean,std,min,max\n";

    auto write_metric = [&](
        const std::string& name,
        const std::vector<double>& values
    ) {
        output
            << name << ","
            << mean_value(values) << ","
            << standard_deviation(values) << ","
            << minimum_value(values) << ","
            << maximum_value(values) << "\n";
    };

    write_metric("total_loss", total_loss);
    write_metric("mean_spread", mean_spread);
    write_metric("median_spread", median_spread);
    write_metric("p90_spread", p90_spread);
    write_metric("p95_spread", p95_spread);
    write_metric("mean_mid_change", mean_mid_change);
    write_metric("std_mid_change", std_mid_change);
    write_metric("zero_mid_change_ratio", zero_mid_change_ratio);
    write_metric("mid_move_rate", mid_move_rate);
    write_metric("final_norm_mid", final_norm_mid);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int number_of_paths = 10;
    unsigned long base_seed = 12345;

    if (argc >= 2) {
        number_of_paths = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        base_seed = std::stoul(argv[2]);
    }

    if (rank == 0) {
        mkdir("mc_full_day_results", 0777);

        std::cout
            << "Mixed-side full-day Monte Carlo started.\n"
            << "Number of MPI ranks: "
            << world_size << "\n"
            << "Number of paths: "
            << number_of_paths << "\n"
            << "Base seed: "
            << base_seed << "\n"
            << "Market makers: 5\n"
            << "Market-maker quantity: 250\n"
            << "Market-maker interval: 750 ms\n"
            << "Institutional pattern: "
            << "Buy, Sell, Buy, Buy, Sell, Buy\n"
            << "Buy parent quantity: 330000\n"
            << "Sell parent quantity: 350000\n"
            << "Institutional child quantity: 1200\n"
            << "Net institutional parent volume: 620000\n"
            << "Quote improvement probability: 0.10\n"
            << "Full-day seconds: 23400\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const std::string temporary_filename =
        "mc_full_day_results/tmp_rank_"
        + std::to_string(rank)
        + ".csv";

    std::ofstream local_output(temporary_filename);

    local_output
        << "path_id,seed,total_loss,"
        << "mean_spread,median_spread,"
        << "p90_spread,p95_spread,"
        << "mean_mid_change,std_mid_change,"
        << "zero_mid_change_ratio,"
        << "mid_move_rate,final_norm_mid\n";

    for (
        int path_id = rank;
        path_id < number_of_paths;
        path_id += world_size
    ) {
        const unsigned long seed =
            base_seed
            + static_cast<unsigned long>(path_id);

        const SimulationConfig config =
            make_full_day_config(path_id, seed);

        const CalibrationResult result =
            run_single_simulation(config);

        write_result_row(
            local_output,
            path_id,
            seed,
            result
        );

        std::cout
            << "Rank " << rank
            << " finished path " << path_id
            << " seed " << seed
            << " loss " << result.total_loss
            << " final_norm_mid "
            << result.final_norm_mid
            << "\n";
    }

    local_output.close();

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        const std::string results_filename =
            "mc_full_day_results/mc_full_day_results.csv";

        const std::string summary_filename =
            "mc_full_day_results/mc_full_day_summary.csv";

        std::vector<MCRow> rows;

        for (
            int source_rank = 0;
            source_rank < world_size;
            ++source_rank
        ) {
            const std::string filename =
                "mc_full_day_results/tmp_rank_"
                + std::to_string(source_rank)
                + ".csv";

            std::ifstream input(filename);
            std::string line;

            while (std::getline(input, line)) {
                MCRow row;

                if (parse_row(line, row)) {
                    rows.push_back(row);
                }
            }

            input.close();
            std::remove(filename.c_str());
        }

        std::sort(
            rows.begin(),
            rows.end(),
            [](const MCRow& left, const MCRow& right) {
                return left.path_id < right.path_id;
            }
        );

        std::ofstream results_output(results_filename);

        results_output
            << "path_id,seed,total_loss,"
            << "mean_spread,median_spread,"
            << "p90_spread,p95_spread,"
            << "mean_mid_change,std_mid_change,"
            << "zero_mid_change_ratio,"
            << "mid_move_rate,final_norm_mid\n";

        for (const MCRow& row : rows) {
            results_output
                << row.path_id << ","
                << row.seed << ","
                << row.total_loss << ","
                << row.mean_spread << ","
                << row.median_spread << ","
                << row.p90_spread << ","
                << row.p95_spread << ","
                << row.mean_mid_change << ","
                << row.std_mid_change << ","
                << row.zero_mid_change_ratio << ","
                << row.mid_move_rate << ","
                << row.final_norm_mid << "\n";
        }

        results_output.close();

        write_summary(rows, summary_filename);

        std::cout
            << "\nFull-day Monte Carlo finished.\n"
            << "Results: "
            << results_filename << "\n"
            << "Summary: "
            << summary_filename << "\n";
    }

    MPI_Finalize();

    return 0;
}
