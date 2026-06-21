#include "LimitOrderBook.hpp"
#include "EmpiricalDistribution.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int K = 6;

constexpr int INITIAL_BEST_BID = 2203400;
constexpr int INITIAL_BEST_ASK = 2203700;
constexpr int DEFAULT_TICK_SIZE = 100;

// Real regular-hours event counts from hawkes_events_QQQ_clean.csv.
// Used for microstructure validation reporting.
const std::array<std::uint64_t, K> REAL_EVENT_COUNTS = {
    1100666ULL, // limit_buy
    1107318ULL, // limit_sell
    24919ULL,   // market_buy
    20901ULL,   // market_sell
    1075932ULL, // cancel_bid
    1091082ULL  // cancel_ask
};

const std::array<std::string, K> EVENT_NAMES = {
    "limit_buy",
    "limit_sell",
    "market_buy",
    "market_sell",
    "cancel_bid",
    "cancel_ask"
};

using Vec6 = std::array<double, K>;
using Mat6 = std::array<std::array<double, K>, K>;

struct Params {
    double beta = 10.0;
    double T = 23400.0;
    double start_clock_seconds = 34200.0;

    Vec6 mu{};
    Mat6 N{};
    Mat6 alpha{};
};

struct Summary {
    std::array<std::uint64_t, K> generated{};
    std::array<std::uint64_t, K> applied{};

    std::array<std::uint64_t, K> requested_volume{};
    std::array<std::uint64_t, K> applied_requested_volume{};
    std::array<std::uint64_t, K> executed_market_volume{};

    std::uint64_t total_snapshots = 0;
    std::uint64_t valid_snapshots = 0;
    double valid_snapshot_ratio = 0.0;

    double mean_spread = 0.0;
    double median_spread = 0.0;
    double p90_spread = 0.0;
    double p95_spread = 0.0;
    double max_spread = 0.0;

    double mean_mid_change = 0.0;
    double std_mid_change = 0.0;
    double mean_abs_mid_change = 0.0;
    double p95_abs_mid_change = 0.0;
    double zero_mid_change_ratio = 0.0;
    double mid_move_rate = 0.0;
    double final_norm_mid = 0.0;

    double mean_best_bid_quantity = 0.0;
    double mean_best_ask_quantity = 0.0;
    double mean_best_total_depth = 0.0;
    double mean_abs_depth_imbalance = 0.0;

    double final_best_bid_quantity = 0.0;
    double final_best_ask_quantity = 0.0;
    double max_best_bid_quantity = 0.0;
    double max_best_ask_quantity = 0.0;

    std::uint64_t best_bid_change_count = 0;
    std::uint64_t best_ask_change_count = 0;

    std::array<double, K> mean_distance{};
    std::array<double, K> median_distance{};
    std::array<double, K> p05_distance{};
    std::array<double, K> p95_distance{};
    std::array<double, K> negative_distance_ratio{};
    std::array<double, K> zero_distance_ratio{};

    bool hit_max_events = false;
};

std::string trim(std::string s) {
    if (s.size() >= 3 &&
        static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB &&
        static_cast<unsigned char>(s[2]) == 0xBF) {
        s.erase(0, 3);
    }

    while (!s.empty() &&
           (s.back() == '\r' || s.back() == '\n' ||
            s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }

    std::size_t start = 0;
    while (start < s.size() &&
           (s[start] == ' ' || s[start] == '\t')) {
        ++start;
    }

    if (start > 0) {
        s.erase(0, start);
    }

    return s;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ',')) {
        out.push_back(trim(item));
    }

    return out;
}

std::uint64_t array_sum(const std::array<std::uint64_t, K>& x) {
    std::uint64_t s = 0;
    for (auto v : x) {
        s += v;
    }
    return s;
}

std::string rank_dir_name(int rank) {
    std::ostringstream oss;
    oss << "path_" << std::setw(4) << std::setfill('0') << rank;
    return oss.str();
}

bool looks_large_number(const std::string& s) {
    try {
        return std::stod(s) > 100000.0;
    } catch (...) {
        return false;
    }
}

Params read_params_flat(const std::string& filename) {
    std::ifstream input(filename);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open parameter file: " + filename);
    }

    std::string header_line;
    std::string value_line;

    if (!std::getline(input, header_line) || !std::getline(input, value_line)) {
        throw std::runtime_error("Parameter file must contain header row and value row.");
    }

    const auto headers = split_csv_line(header_line);
    const auto values = split_csv_line(value_line);

    if (headers.size() != values.size()) {
        throw std::runtime_error("Parameter CSV header/value column count mismatch.");
    }

    std::unordered_map<std::string, double> m;

    for (std::size_t i = 0; i < headers.size(); ++i) {
        if (!headers[i].empty() && !values[i].empty()) {
            m[headers[i]] = std::stod(values[i]);
        }
    }

    Params p;

    p.beta = m.count("beta") ? m.at("beta") : 10.0;
    p.T = m.count("T") ? m.at("T") : 23400.0;

    p.start_clock_seconds = m.count("start_time_seconds_original")
        ? m.at("start_time_seconds_original")
        : 34200.0;

    for (int i = 0; i < K; ++i) {
        p.mu[i] = m.at("mu_" + std::to_string(i));
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            const std::string key = "N_" + std::to_string(i) + std::to_string(j);
            p.N[i][j] = m.at(key);
            p.alpha[i][j] = p.beta * p.N[i][j];
        }
    }

    return p;
}

double mean_value(const std::vector<double>& x) {
    if (x.empty()) {
        return 0.0;
    }

    return std::accumulate(x.begin(), x.end(), 0.0) /
           static_cast<double>(x.size());
}

double stdev_value(const std::vector<double>& x) {
    if (x.size() < 2) {
        return 0.0;
    }

    const double avg = mean_value(x);
    double s2 = 0.0;

    for (double v : x) {
        const double d = v - avg;
        s2 += d * d;
    }

    return std::sqrt(s2 / static_cast<double>(x.size() - 1));
}

double percentile(std::vector<double> x, double p) {
    if (x.empty()) {
        return 0.0;
    }

    std::sort(x.begin(), x.end());

    const double pos = p * static_cast<double>(x.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));

    if (lo == hi) {
        return x[lo];
    }

    const double w = pos - static_cast<double>(lo);
    return x[lo] * (1.0 - w) + x[hi] * w;
}

int scaled_quantity(int q, double scale) {
    return std::max(1, static_cast<int>(
        std::round(static_cast<double>(q) * scale)
    ));
}

bool valid_price(long long p) {
    return p > 0 && p <= static_cast<long long>(std::numeric_limits<int>::max());
}

void create_initial_book(LimitOrderBook& book, double depth_scale) {
    book.add_limit_order(Order{1, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(1623, depth_scale), 2203400});
    book.add_limit_order(Order{2, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(1723, depth_scale), 2203300});
    book.add_limit_order(Order{3, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(2100, depth_scale), 2203200});
    book.add_limit_order(Order{4, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(1100, depth_scale), 2203100});
    book.add_limit_order(Order{5, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(1200, depth_scale), 2203000});
    book.add_limit_order(Order{6, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(200, depth_scale), 2202900});
    book.add_limit_order(Order{7, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(564, depth_scale), 2202800});
    book.add_limit_order(Order{8, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(500, depth_scale), 2202600});
    book.add_limit_order(Order{9, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(700, depth_scale), 2202500});
    book.add_limit_order(Order{10, 0, OrderType::Limit, Side::Buy,
        scaled_quantity(200, depth_scale), 2202400});

    book.add_limit_order(Order{11, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(823, depth_scale), 2203700});
    book.add_limit_order(Order{12, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(823, depth_scale), 2203800});
    book.add_limit_order(Order{13, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(1823, depth_scale), 2203900});
    book.add_limit_order(Order{14, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(1923, depth_scale), 2204000});
    book.add_limit_order(Order{15, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(1923, depth_scale), 2204100});
    book.add_limit_order(Order{16, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(1223, depth_scale), 2204200});
    book.add_limit_order(Order{17, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(823, depth_scale), 2204300});
    book.add_limit_order(Order{18, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(200, depth_scale), 2204400});
    book.add_limit_order(Order{19, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(823, depth_scale), 2204500});
    book.add_limit_order(Order{20, 0, OrderType::Limit, Side::Sell,
        scaled_quantity(823, depth_scale), 2204600});
}

Vec6 compute_intensity(const Params& p, const Vec6& R) {
    Vec6 lambda{};

    for (int i = 0; i < K; ++i) {
        double v = p.mu[i];

        for (int j = 0; j < K; ++j) {
            v += p.alpha[i][j] * R[j];
        }

        lambda[i] = std::max(v, 0.0);
    }

    return lambda;
}

double sum_vec(const Vec6& x) {
    double s = 0.0;

    for (double v : x) {
        s += v;
    }

    return s;
}

int sample_event_type(const Vec6& lambda, double total, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif(0.0, total);
    const double u = unif(rng);

    double c = 0.0;
    for (int i = 0; i < K; ++i) {
        c += lambda[i];
        if (u <= c) {
            return i;
        }
    }

    return K - 1;
}

bool record_snapshot(
    const LimitOrderBook& book,
    double t_seconds,
    double start_clock_seconds,
    std::vector<double>& spreads,
    std::vector<double>& mids,
    std::vector<double>& best_bid_quantities,
    std::vector<double>& best_ask_quantities,
    std::vector<double>& best_total_depths,
    std::vector<double>& abs_depth_imbalances,
    std::vector<int>& best_bids,
    std::vector<int>& best_asks,
    std::ofstream& out
) {
    const double clock_seconds = start_clock_seconds + t_seconds;
    const std::int64_t time_ns =
        static_cast<std::int64_t>(std::llround(t_seconds * 1e9));

    if (book.has_bid() && book.has_ask()) {
        const int bid = book.best_bid();
        const int ask = book.best_ask();
        const int spread = ask - bid;

        if (spread < 0) {
            return false;
        }

        const double mid = 0.5 * static_cast<double>(bid + ask);

        const int bid_qty = book.quantity_at_best_bid();
        const int ask_qty = book.quantity_at_best_ask();
        const int total_depth = bid_qty + ask_qty;

        double abs_imbalance = 0.0;
        if (total_depth > 0) {
            abs_imbalance =
                std::abs(static_cast<double>(bid_qty - ask_qty)) /
                static_cast<double>(total_depth);
        }

        spreads.push_back(static_cast<double>(spread));
        mids.push_back(mid);
        best_bid_quantities.push_back(static_cast<double>(bid_qty));
        best_ask_quantities.push_back(static_cast<double>(ask_qty));
        best_total_depths.push_back(static_cast<double>(total_depth));
        abs_depth_imbalances.push_back(abs_imbalance);
        best_bids.push_back(bid);
        best_asks.push_back(ask);

        out << time_ns << ','
            << std::setprecision(12) << t_seconds << ','
            << clock_seconds << ','
            << bid << ','
            << ask << ','
            << spread << ','
            << mid << ','
            << bid_qty << ','
            << ask_qty << ','
            << total_depth << ','
            << abs_imbalance << '\n';

        return true;
    }

    out << time_ns << ','
        << std::setprecision(12) << t_seconds << ','
        << clock_seconds << ",,,,,,,,\n";

    return false;
}

void write_path_summary(
    const std::string& filename,
    const Params& params,
    const Summary& s,
    int rank,
    int mpi_size,
    std::uint64_t seed,
    std::uint64_t max_events,
    double runtime_seconds
) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open path summary output file.");
    }

    const std::uint64_t total_generated = array_sum(s.generated);
    const std::uint64_t total_applied = array_sum(s.applied);
    const std::uint64_t real_total = array_sum(REAL_EVENT_COUNTS);

    const double event_volume_ratio =
        real_total > 0
            ? static_cast<double>(total_generated) / static_cast<double>(real_total)
            : 0.0;

    const double event_volume_relative_error =
        real_total > 0
            ? std::abs(static_cast<double>(total_generated) -
                       static_cast<double>(real_total)) /
              static_cast<double>(real_total)
            : 0.0;

    double event_mix_l1_loss = 0.0;
    if (total_generated > 0 && real_total > 0) {
        for (int i = 0; i < K; ++i) {
            const double sim_share =
                static_cast<double>(s.generated[i]) /
                static_cast<double>(total_generated);

            const double real_share =
                static_cast<double>(REAL_EVENT_COUNTS[i]) /
                static_cast<double>(real_total);

            event_mix_l1_loss += std::abs(sim_share - real_share);
        }

        event_mix_l1_loss *= 0.5;
    }

    const std::uint64_t total_limit_requested_volume =
        s.requested_volume[0] + s.requested_volume[1];

    const std::uint64_t total_market_requested_volume =
        s.requested_volume[2] + s.requested_volume[3];

    const std::uint64_t total_cancel_requested_volume =
        s.requested_volume[4] + s.requested_volume[5];

    const std::uint64_t total_market_executed_volume =
        s.executed_market_volume[2] + s.executed_market_volume[3];

    out << std::setprecision(15);
    out << "metric,value\n";

    out << "model,M0_fitted_full_6x6_Hawkes_only_old_signed_distance_path\n";
    out << "rank," << rank << "\n";
    out << "mpi_size," << mpi_size << "\n";
    out << "seed," << seed << "\n";
    out << "runtime_seconds," << runtime_seconds << "\n";
    out << "beta," << params.beta << "\n";
    out << "T," << params.T << "\n";
    out << "start_clock_seconds," << params.start_clock_seconds << "\n";
    out << "initial_best_bid," << INITIAL_BEST_BID << "\n";
    out << "initial_best_ask," << INITIAL_BEST_ASK << "\n";
    out << "tick_size," << DEFAULT_TICK_SIZE << "\n";
    out << "max_events," << max_events << "\n";
    out << "hit_max_events," << (s.hit_max_events ? 1 : 0) << "\n";

    out << "real_total_events," << real_total << "\n";
    out << "generated_total," << total_generated << "\n";
    out << "applied_total," << total_applied << "\n";
    out << "event_volume_ratio_sim_over_real," << event_volume_ratio << "\n";
    out << "event_volume_relative_error," << event_volume_relative_error << "\n";
    out << "event_mix_l1_loss," << event_mix_l1_loss << "\n";

    for (int i = 0; i < K; ++i) {
        out << "real_" << EVENT_NAMES[i] << ',' << REAL_EVENT_COUNTS[i] << "\n";
    }

    for (int i = 0; i < K; ++i) {
        out << "generated_" << EVENT_NAMES[i] << ',' << s.generated[i] << "\n";
    }

    for (int i = 0; i < K; ++i) {
        out << "applied_" << EVENT_NAMES[i] << ',' << s.applied[i] << "\n";
    }

    for (int i = 0; i < K; ++i) {
        const double sim_share =
            total_generated > 0
                ? static_cast<double>(s.generated[i]) /
                  static_cast<double>(total_generated)
                : 0.0;

        const double real_share =
            real_total > 0
                ? static_cast<double>(REAL_EVENT_COUNTS[i]) /
                  static_cast<double>(real_total)
                : 0.0;

        out << "generated_share_" << EVENT_NAMES[i] << ',' << sim_share << "\n";
        out << "real_share_" << EVENT_NAMES[i] << ',' << real_share << "\n";
    }

    for (int i = 0; i < K; ++i) {
        out << "requested_volume_" << EVENT_NAMES[i] << ','
            << s.requested_volume[i] << "\n";
    }

    for (int i = 0; i < K; ++i) {
        out << "applied_requested_volume_" << EVENT_NAMES[i] << ','
            << s.applied_requested_volume[i] << "\n";
    }

    for (int i = 0; i < K; ++i) {
        out << "executed_market_volume_" << EVENT_NAMES[i] << ','
            << s.executed_market_volume[i] << "\n";
    }

    out << "total_limit_requested_volume," << total_limit_requested_volume << "\n";
    out << "total_market_requested_volume," << total_market_requested_volume << "\n";
    out << "total_cancel_requested_volume," << total_cancel_requested_volume << "\n";
    out << "total_market_executed_volume," << total_market_executed_volume << "\n";
    out << "market_execution_ratio,"
        << (total_market_requested_volume > 0
                ? static_cast<double>(total_market_executed_volume) /
                  static_cast<double>(total_market_requested_volume)
                : 0.0)
        << "\n";

    out << "total_snapshots," << s.total_snapshots << "\n";
    out << "valid_snapshots," << s.valid_snapshots << "\n";
    out << "valid_snapshot_ratio," << s.valid_snapshot_ratio << "\n";

    out << "mean_spread," << s.mean_spread << "\n";
    out << "median_spread," << s.median_spread << "\n";
    out << "p90_spread," << s.p90_spread << "\n";
    out << "p95_spread," << s.p95_spread << "\n";
    out << "max_spread," << s.max_spread << "\n";

    out << "mean_mid_change," << s.mean_mid_change << "\n";
    out << "std_mid_change," << s.std_mid_change << "\n";
    out << "mean_abs_mid_change," << s.mean_abs_mid_change << "\n";
    out << "p95_abs_mid_change," << s.p95_abs_mid_change << "\n";
    out << "zero_mid_change_ratio," << s.zero_mid_change_ratio << "\n";
    out << "mid_move_rate," << s.mid_move_rate << "\n";
    out << "final_norm_mid," << s.final_norm_mid << "\n";

    out << "mean_best_bid_quantity," << s.mean_best_bid_quantity << "\n";
    out << "mean_best_ask_quantity," << s.mean_best_ask_quantity << "\n";
    out << "mean_best_total_depth," << s.mean_best_total_depth << "\n";
    out << "mean_abs_depth_imbalance," << s.mean_abs_depth_imbalance << "\n";
    out << "final_best_bid_quantity," << s.final_best_bid_quantity << "\n";
    out << "final_best_ask_quantity," << s.final_best_ask_quantity << "\n";
    out << "max_best_bid_quantity," << s.max_best_bid_quantity << "\n";
    out << "max_best_ask_quantity," << s.max_best_ask_quantity << "\n";

    for (int i : {0, 1, 4, 5}) {
        out << "mean_distance_" << EVENT_NAMES[i] << ','
            << s.mean_distance[i] << "\n";

        out << "median_distance_" << EVENT_NAMES[i] << ','
            << s.median_distance[i] << "\n";

        out << "p05_distance_" << EVENT_NAMES[i] << ','
            << s.p05_distance[i] << "\n";

        out << "p95_distance_" << EVENT_NAMES[i] << ','
            << s.p95_distance[i] << "\n";

        out << "negative_distance_ratio_" << EVENT_NAMES[i] << ','
            << s.negative_distance_ratio[i] << "\n";

        out << "zero_distance_ratio_" << EVENT_NAMES[i] << ','
            << s.zero_distance_ratio[i] << "\n";
    }

    out << "best_bid_change_count," << s.best_bid_change_count << "\n";
    out << "best_ask_change_count," << s.best_ask_change_count << "\n";
}

std::vector<std::string> metric_names() {
    std::vector<std::string> names = {
        "seed",
        "runtime_seconds",
        "generated_total",
        "applied_total",
        "event_volume_ratio_sim_over_real",
        "event_volume_relative_error",
        "event_mix_l1_loss",

        "mean_spread",
        "median_spread",
        "p90_spread",
        "p95_spread",
        "max_spread",

        "mean_mid_change",
        "std_mid_change",
        "mean_abs_mid_change",
        "p95_abs_mid_change",
        "zero_mid_change_ratio",
        "mid_move_rate",
        "final_norm_mid",

        "mean_best_bid_quantity",
        "mean_best_ask_quantity",
        "mean_best_total_depth",
        "mean_abs_depth_imbalance",
        "final_best_bid_quantity",
        "final_best_ask_quantity",
        "max_best_bid_quantity",
        "max_best_ask_quantity",
        "best_bid_change_count",
        "best_ask_change_count"
    };

    for (int i = 0; i < K; ++i) {
        names.push_back("generated_" + EVENT_NAMES[i]);
    }

    for (int i = 0; i < K; ++i) {
        names.push_back("applied_" + EVENT_NAMES[i]);
    }

    for (int i = 0; i < K; ++i) {
        names.push_back("requested_volume_" + EVENT_NAMES[i]);
    }

    for (int i = 0; i < K; ++i) {
        names.push_back("executed_market_volume_" + EVENT_NAMES[i]);
    }

    for (int i : {0, 1, 4, 5}) {
        names.push_back("mean_distance_" + EVENT_NAMES[i]);
        names.push_back("median_distance_" + EVENT_NAMES[i]);
        names.push_back("p05_distance_" + EVENT_NAMES[i]);
        names.push_back("p95_distance_" + EVENT_NAMES[i]);
        names.push_back("negative_distance_ratio_" + EVENT_NAMES[i]);
        names.push_back("zero_distance_ratio_" + EVENT_NAMES[i]);
    }

    return names;
}

std::vector<double> metric_values(
    const Summary& s,
    std::uint64_t seed,
    double runtime_seconds
) {
    const std::uint64_t total_generated = array_sum(s.generated);
    const std::uint64_t total_applied = array_sum(s.applied);
    const std::uint64_t real_total = array_sum(REAL_EVENT_COUNTS);

    const double event_volume_ratio =
        real_total > 0
            ? static_cast<double>(total_generated) / static_cast<double>(real_total)
            : 0.0;

    const double event_volume_relative_error =
        real_total > 0
            ? std::abs(static_cast<double>(total_generated) -
                       static_cast<double>(real_total)) /
              static_cast<double>(real_total)
            : 0.0;

    double event_mix_l1_loss = 0.0;
    if (total_generated > 0 && real_total > 0) {
        for (int i = 0; i < K; ++i) {
            const double sim_share =
                static_cast<double>(s.generated[i]) /
                static_cast<double>(total_generated);

            const double real_share =
                static_cast<double>(REAL_EVENT_COUNTS[i]) /
                static_cast<double>(real_total);

            event_mix_l1_loss += std::abs(sim_share - real_share);
        }

        event_mix_l1_loss *= 0.5;
    }

    std::vector<double> values = {
        static_cast<double>(seed),
        runtime_seconds,
        static_cast<double>(total_generated),
        static_cast<double>(total_applied),
        event_volume_ratio,
        event_volume_relative_error,
        event_mix_l1_loss,

        s.mean_spread,
        s.median_spread,
        s.p90_spread,
        s.p95_spread,
        s.max_spread,

        s.mean_mid_change,
        s.std_mid_change,
        s.mean_abs_mid_change,
        s.p95_abs_mid_change,
        s.zero_mid_change_ratio,
        s.mid_move_rate,
        s.final_norm_mid,

        s.mean_best_bid_quantity,
        s.mean_best_ask_quantity,
        s.mean_best_total_depth,
        s.mean_abs_depth_imbalance,
        s.final_best_bid_quantity,
        s.final_best_ask_quantity,
        s.max_best_bid_quantity,
        s.max_best_ask_quantity,
        static_cast<double>(s.best_bid_change_count),
        static_cast<double>(s.best_ask_change_count)
    };

    for (int i = 0; i < K; ++i) {
        values.push_back(static_cast<double>(s.generated[i]));
    }

    for (int i = 0; i < K; ++i) {
        values.push_back(static_cast<double>(s.applied[i]));
    }

    for (int i = 0; i < K; ++i) {
        values.push_back(static_cast<double>(s.requested_volume[i]));
    }

    for (int i = 0; i < K; ++i) {
        values.push_back(static_cast<double>(s.executed_market_volume[i]));
    }

    for (int i : {0, 1, 4, 5}) {
        values.push_back(s.mean_distance[i]);
        values.push_back(s.median_distance[i]);
        values.push_back(s.p05_distance[i]);
        values.push_back(s.p95_distance[i]);
        values.push_back(s.negative_distance_ratio[i]);
        values.push_back(s.zero_distance_ratio[i]);
    }

    return values;
}

void write_aggregate_outputs(
    const std::string& output_dir,
    const std::vector<std::string>& names,
    const std::vector<double>& all_values,
    int mpi_size
) {
    const int metric_count = static_cast<int>(names.size());

    std::ofstream path_out(output_dir + "/m0_path_metrics.csv");
    path_out << "path_id";

    for (const auto& name : names) {
        path_out << ',' << name;
    }

    path_out << "\n";

    for (int r = 0; r < mpi_size; ++r) {
        path_out << r;

        for (int j = 0; j < metric_count; ++j) {
            path_out << ',' << std::setprecision(15)
                     << all_values[r * metric_count + j];
        }

        path_out << "\n";
    }

    std::ofstream agg_out(output_dir + "/m0_aggregate_summary.csv");
    agg_out << "metric,mean,std,min,max\n";

    for (int j = 0; j < metric_count; ++j) {
        std::vector<double> vals;
        vals.reserve(mpi_size);

        for (int r = 0; r < mpi_size; ++r) {
            vals.push_back(all_values[r * metric_count + j]);
        }

        const double mean = mean_value(vals);
        const double sd = stdev_value(vals);
        const double mn = *std::min_element(vals.begin(), vals.end());
        const double mx = *std::max_element(vals.begin(), vals.end());

        agg_out << names[j] << ','
                << std::setprecision(15) << mean << ','
                << sd << ','
                << mn << ','
                << mx << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int mpi_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    try {
        const std::string params_file = argc > 1
            ? argv[1]
            : "hawkes_mle_fixed_beta_lbfgsb_eta010/fixed_beta_mle_params_flat.csv";

        const std::string output_root = argc > 2
            ? argv[2]
            : "m0_fitted_hawkes_mpi_paths";

        std::uint64_t max_events = 20000000ULL;
        double sample_interval_seconds = 1.0;
        std::uint64_t base_seed = 12345ULL;

        std::string quantity_column = "quantity";
        std::string distance_column = "distance_ticks";

        // Supports both interfaces:
        //
        // New MPI style:
        //   mpirun -np 10 ./m0_fitted_hawkes params.csv output_dir max_events sample_interval [base_seed]
        //
        // Old single-path style:
        //   ./m0_fitted_hawkes params.csv output_dir seed max_events sample_interval
        if (argc >= 6 && looks_large_number(argv[4])) {
            // Old style.
            base_seed = static_cast<std::uint64_t>(std::stoull(argv[3]));
            max_events = static_cast<std::uint64_t>(std::stoull(argv[4]));
            sample_interval_seconds = std::stod(argv[5]);

            if (argc > 6) {
                quantity_column = argv[6];
            }
            if (argc > 7) {
                distance_column = argv[7];
            }
        } else {
            // New MPI style.
            if (argc > 3) {
                max_events = static_cast<std::uint64_t>(std::stoull(argv[3]));
            }
            if (argc > 4) {
                sample_interval_seconds = std::stod(argv[4]);
            }
            if (argc > 5) {
                base_seed = static_cast<std::uint64_t>(std::stoull(argv[5]));
            }
            if (argc > 6) {
                quantity_column = argv[6];
            }
            if (argc > 7) {
                distance_column = argv[7];
            }
        }

        const std::uint64_t path_seed =
            base_seed + static_cast<std::uint64_t>(rank);

        const Params params = read_params_flat(params_file);

        if (rank == 0) {
            std::filesystem::create_directories(output_root);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        const std::string path_dir =
            mpi_size == 1
                ? output_root
                : output_root + "/" + rank_dir_name(rank);

        std::filesystem::create_directories(path_dir);

        const double t0_wall = MPI_Wtime();

        LimitOrderBook book;
        create_initial_book(book, 1.0);

        std::uint64_t next_order_id = 21;

        EmpiricalDistribution limit_buy_qty;
        EmpiricalDistribution limit_sell_qty;
        EmpiricalDistribution market_buy_qty;
        EmpiricalDistribution market_sell_qty;
        EmpiricalDistribution cancel_bid_qty;
        EmpiricalDistribution cancel_ask_qty;

        limit_buy_qty.load_from_csv("limit_buy_quantity_distribution.txt", quantity_column);
        limit_sell_qty.load_from_csv("limit_sell_quantity_distribution.txt", quantity_column);
        market_buy_qty.load_from_csv("market_buy_quantity_distribution.txt", quantity_column);
        market_sell_qty.load_from_csv("market_sell_quantity_distribution.txt", quantity_column);
        cancel_bid_qty.load_from_csv("cancel_bid_quantity_distribution.txt", quantity_column);
        cancel_ask_qty.load_from_csv("cancel_ask_quantity_distribution.txt", quantity_column);

        EmpiricalDistribution limit_buy_dist;
        EmpiricalDistribution limit_sell_dist;
        EmpiricalDistribution cancel_bid_dist;
        EmpiricalDistribution cancel_ask_dist;

        limit_buy_dist.load_from_csv("limit_buy_distance_distribution.txt", distance_column);
        limit_sell_dist.load_from_csv("limit_sell_distance_distribution.txt", distance_column);
        cancel_bid_dist.load_from_csv("cancel_bid_distance_distribution.txt", distance_column);
        cancel_ask_dist.load_from_csv("cancel_ask_distance_distribution.txt", distance_column);

        std::ofstream ts_out(path_dir + "/m0_fitted_timeseries.csv");
        ts_out << "time_ns,time_seconds_session,time_seconds_clock,"
               << "best_bid,best_ask,spread,mid_price,"
               << "best_bid_quantity,best_ask_quantity,"
               << "best_total_depth,abs_depth_imbalance\n";

        std::ofstream ev_out(path_dir + "/m0_fitted_events.csv");
        ev_out << "time_ns,time_seconds_session,time_seconds_clock,"
               << "event_type,event_label,quantity,distance_ticks,applied,executed_quantity\n";

        std::mt19937_64 rng(path_seed);
        std::uniform_real_distribution<double> unif01(0.0, 1.0);

        Vec6 R{};
        double t = 0.0;
        double next_sample_t = 0.0;

        Summary summary;

        std::vector<double> spreads;
        std::vector<double> mids;
        std::vector<double> best_bid_quantities;
        std::vector<double> best_ask_quantities;
        std::vector<double> best_total_depths;
        std::vector<double> abs_depth_imbalances;
        std::vector<int> best_bids;
        std::vector<int> best_asks;

        std::array<std::vector<double>, K> distance_samples;

        while (t < params.T) {
            const Vec6 lambda = compute_intensity(params, R);
            const double lambda_sum = sum_vec(lambda);

            if (lambda_sum <= 0.0) {
                break;
            }

            std::exponential_distribution<double> exp_dist(lambda_sum);
            const double dt = exp_dist(rng);
            const double candidate_t = t + dt;

            const double decay = std::exp(-params.beta * dt);
            for (int j = 0; j < K; ++j) {
                R[j] *= decay;
            }

            t = candidate_t;

            while (next_sample_t <= t && next_sample_t <= params.T) {
                ++summary.total_snapshots;

                if (record_snapshot(
                        book,
                        next_sample_t,
                        params.start_clock_seconds,
                        spreads,
                        mids,
                        best_bid_quantities,
                        best_ask_quantities,
                        best_total_depths,
                        abs_depth_imbalances,
                        best_bids,
                        best_asks,
                        ts_out)) {
                    ++summary.valid_snapshots;
                }

                next_sample_t += sample_interval_seconds;
            }

            if (t > params.T) {
                break;
            }

            const Vec6 lambda_new = compute_intensity(params, R);
            const double lambda_new_sum = sum_vec(lambda_new);

            // Ogata thinning acceptance step.
            if (unif01(rng) * lambda_sum > lambda_new_sum) {
                continue;
            }

            const int type = sample_event_type(lambda_new, lambda_new_sum, rng);

            R[type] += 1.0;
            ++summary.generated[type];

            bool applied = false;
            int quantity = 0;
            int executed_quantity = 0;
            int distance_ticks = 0;
            bool has_distance = false;

            const std::int64_t event_time_ns =
                static_cast<std::int64_t>(std::llround(t * 1e9));
            const double clock_seconds = params.start_clock_seconds + t;

            if (type == 0) { // limit_buy
                if (book.has_bid()) {
                    quantity = std::max(1, limit_buy_qty.sample(rng));
                    distance_ticks = limit_buy_dist.sample(rng);
                    has_distance = true;

                    // Old signed-distance path:
                    // price = best_bid - signed_distance * tick_size.
                    const long long price =
                        static_cast<long long>(book.best_bid()) -
                        static_cast<long long>(distance_ticks) *
                        static_cast<long long>(DEFAULT_TICK_SIZE);

                    if (valid_price(price)) {
                        book.add_limit_order(Order{
                            next_order_id++,
                            event_time_ns,
                            OrderType::Limit,
                            Side::Buy,
                            quantity,
                            static_cast<int>(price)
                        });

                        applied = true;
                    }
                }
            } else if (type == 1) { // limit_sell
                if (book.has_ask()) {
                    quantity = std::max(1, limit_sell_qty.sample(rng));
                    distance_ticks = limit_sell_dist.sample(rng);
                    has_distance = true;

                    // Old signed-distance path:
                    // price = best_ask + signed_distance * tick_size.
                    const long long price =
                        static_cast<long long>(book.best_ask()) +
                        static_cast<long long>(distance_ticks) *
                        static_cast<long long>(DEFAULT_TICK_SIZE);

                    if (valid_price(price)) {
                        book.add_limit_order(Order{
                            next_order_id++,
                            event_time_ns,
                            OrderType::Limit,
                            Side::Sell,
                            quantity,
                            static_cast<int>(price)
                        });

                        applied = true;
                    }
                }
            } else if (type == 2) { // market_buy
                quantity = std::max(1, market_buy_qty.sample(rng));

                if (book.has_ask()) {
                    executed_quantity = book.submit_market_order(Side::Buy, quantity);
                    applied = executed_quantity > 0;
                }
            } else if (type == 3) { // market_sell
                quantity = std::max(1, market_sell_qty.sample(rng));

                if (book.has_bid()) {
                    executed_quantity = book.submit_market_order(Side::Sell, quantity);
                    applied = executed_quantity > 0;
                }
            } else if (type == 4) { // cancel_bid
                quantity = std::max(1, cancel_bid_qty.sample(rng));
                distance_ticks = cancel_bid_dist.sample(rng);
                has_distance = true;

                if (book.has_bid()) {
                    book.cancel_at_distance(
                        Side::Buy,
                        distance_ticks,
                        quantity,
                        DEFAULT_TICK_SIZE
                    );

                    applied = true;
                }
            } else if (type == 5) { // cancel_ask
                quantity = std::max(1, cancel_ask_qty.sample(rng));
                distance_ticks = cancel_ask_dist.sample(rng);
                has_distance = true;

                if (book.has_ask()) {
                    book.cancel_at_distance(
                        Side::Sell,
                        distance_ticks,
                        quantity,
                        DEFAULT_TICK_SIZE
                    );

                    applied = true;
                }
            }

            if (has_distance) {
                distance_samples[type].push_back(static_cast<double>(distance_ticks));
            }

            if (quantity > 0) {
                summary.requested_volume[type] +=
                    static_cast<std::uint64_t>(quantity);
            }

            if (applied) {
                ++summary.applied[type];

                if (quantity > 0) {
                    summary.applied_requested_volume[type] +=
                        static_cast<std::uint64_t>(quantity);
                }
            }

            if ((type == 2 || type == 3) && executed_quantity > 0) {
                summary.executed_market_volume[type] +=
                    static_cast<std::uint64_t>(executed_quantity);
            }

            ev_out << event_time_ns << ','
                   << std::setprecision(12) << t << ','
                   << clock_seconds << ','
                   << type << ','
                   << EVENT_NAMES[type] << ','
                   << quantity << ',';

            if (has_distance) {
                ev_out << distance_ticks;
            }

            ev_out << ','
                   << (applied ? 1 : 0) << ','
                   << executed_quantity << '\n';

            if (array_sum(summary.generated) >= max_events) {
                summary.hit_max_events = true;
                break;
            }
        }

        while (next_sample_t <= params.T) {
            ++summary.total_snapshots;

            if (record_snapshot(
                    book,
                    next_sample_t,
                    params.start_clock_seconds,
                    spreads,
                    mids,
                    best_bid_quantities,
                    best_ask_quantities,
                    best_total_depths,
                    abs_depth_imbalances,
                    best_bids,
                    best_asks,
                    ts_out)) {
                ++summary.valid_snapshots;
            }

            next_sample_t += sample_interval_seconds;
        }

        summary.valid_snapshot_ratio =
            summary.total_snapshots > 0
                ? static_cast<double>(summary.valid_snapshots) /
                  static_cast<double>(summary.total_snapshots)
                : 0.0;

        summary.mean_spread = mean_value(spreads);
        summary.median_spread = percentile(spreads, 0.50);
        summary.p90_spread = percentile(spreads, 0.90);
        summary.p95_spread = percentile(spreads, 0.95);

        summary.max_spread =
            spreads.empty()
                ? 0.0
                : *std::max_element(spreads.begin(), spreads.end());

        std::vector<double> mid_changes;
        std::vector<double> abs_mid_changes;

        if (mids.size() >= 2) {
            mid_changes.reserve(mids.size() - 1);
            abs_mid_changes.reserve(mids.size() - 1);
        }

        for (std::size_t i = 1; i < mids.size(); ++i) {
            const double change = mids[i] - mids[i - 1];
            mid_changes.push_back(change);
            abs_mid_changes.push_back(std::abs(change));
        }

        summary.mean_mid_change = mean_value(mid_changes);
        summary.std_mid_change = stdev_value(mid_changes);
        summary.mean_abs_mid_change = mean_value(abs_mid_changes);
        summary.p95_abs_mid_change = percentile(abs_mid_changes, 0.95);

        if (!mid_changes.empty()) {
            std::uint64_t zero = 0;

            for (double x : mid_changes) {
                if (std::abs(x) < 1e-12) {
                    ++zero;
                }
            }

            summary.zero_mid_change_ratio =
                static_cast<double>(zero) /
                static_cast<double>(mid_changes.size());

            summary.mid_move_rate = 1.0 - summary.zero_mid_change_ratio;
        }

        if (mids.size() >= 2) {
            summary.final_norm_mid = mids.back() - mids.front();
        }

        for (int i : {0, 1, 4, 5}) {
            const auto& d = distance_samples[i];

            summary.mean_distance[i] = mean_value(d);
            summary.median_distance[i] = percentile(d, 0.50);
            summary.p05_distance[i] = percentile(d, 0.05);
            summary.p95_distance[i] = percentile(d, 0.95);

            if (!d.empty()) {
                std::uint64_t negative_count = 0;
                std::uint64_t zero_count = 0;

                for (double x : d) {
                    if (x < 0.0) {
                        ++negative_count;
                    }

                    if (std::abs(x) < 1e-12) {
                        ++zero_count;
                    }
                }

                summary.negative_distance_ratio[i] =
                    static_cast<double>(negative_count) /
                    static_cast<double>(d.size());

                summary.zero_distance_ratio[i] =
                    static_cast<double>(zero_count) /
                    static_cast<double>(d.size());
            }
        }

        summary.mean_best_bid_quantity = mean_value(best_bid_quantities);
        summary.mean_best_ask_quantity = mean_value(best_ask_quantities);
        summary.mean_best_total_depth = mean_value(best_total_depths);
        summary.mean_abs_depth_imbalance = mean_value(abs_depth_imbalances);

        if (!best_bid_quantities.empty()) {
            summary.final_best_bid_quantity = best_bid_quantities.back();
            summary.max_best_bid_quantity =
                *std::max_element(
                    best_bid_quantities.begin(),
                    best_bid_quantities.end()
                );
        }

        if (!best_ask_quantities.empty()) {
            summary.final_best_ask_quantity = best_ask_quantities.back();
            summary.max_best_ask_quantity =
                *std::max_element(
                    best_ask_quantities.begin(),
                    best_ask_quantities.end()
                );
        }

        for (std::size_t i = 1; i < best_bids.size(); ++i) {
            if (best_bids[i] != best_bids[i - 1]) {
                ++summary.best_bid_change_count;
            }
        }

        for (std::size_t i = 1; i < best_asks.size(); ++i) {
            if (best_asks[i] != best_asks[i - 1]) {
                ++summary.best_ask_change_count;
            }
        }

        const double runtime_seconds = MPI_Wtime() - t0_wall;

        write_path_summary(
            path_dir + "/m0_fitted_summary.csv",
            params,
            summary,
            rank,
            mpi_size,
            path_seed,
            max_events,
            runtime_seconds
        );

        const auto names = metric_names();
        const auto values = metric_values(summary, path_seed, runtime_seconds);
        const int metric_count = static_cast<int>(values.size());

        std::vector<double> all_values;
        if (rank == 0) {
            all_values.resize(static_cast<std::size_t>(mpi_size) *
                              static_cast<std::size_t>(metric_count));
        }

        MPI_Gather(
            values.data(),
            metric_count,
            MPI_DOUBLE,
            rank == 0 ? all_values.data() : nullptr,
            metric_count,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            write_aggregate_outputs(output_root, names, all_values, mpi_size);
        }

        const std::uint64_t total_generated = array_sum(summary.generated);

        std::cout << "[rank " << rank << "] "
                  << "seed=" << path_seed
                  << " generated=" << total_generated
                  << " mean_spread=" << summary.mean_spread
                  << " p95_spread=" << summary.p95_spread
                  << " runtime=" << runtime_seconds
                  << "s\n";

        MPI_Finalize();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
