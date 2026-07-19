#include "LossFunction.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {
std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    std::size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) ++start;
    if (start > 0) s.erase(0, start);
    return s;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) fields.push_back(trim(cell));
    return fields;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

int find_column(const std::vector<std::string>& header, const std::vector<std::string>& candidates) {
    for (std::size_t i = 0; i < header.size(); ++i) {
        const std::string h = lower(trim(header[i]));
        for (const std::string& c : candidates) {
            if (h == lower(c)) return static_cast<int>(i);
        }
    }
    return -1;
}

double default_weight(const std::string& name) {
    static const std::unordered_map<std::string, double> w = {
        {"return_mean", 0.5}, {"return_variance", 3.0}, {"return_skewness", 0.5}, {"return_kurtosis", 1.5},
        {"return_acf_lag1", 2.0}, {"abs_return_acf_lag1", 1.5}, {"abs_return_acf_lag2", 1.0},
        {"abs_return_acf_lag5", 1.0}, {"abs_return_acf_lag10", 1.0}, {"abs_return_acf_lag25", 0.75},
        {"abs_return_acf_lag50", 0.75}, {"abs_return_acf_lag100", 0.5},
        {"mean_spread", 2.5}, {"spread_p90", 1.5}, {"spread_p95", 1.0},
        {"volume_volatility_corr", 1.0}, {"price_diffusion_10s", 3.0},
        {"mean_best_bid_depth", 2.0}, {"mean_best_ask_depth", 2.0},
        {"mid_move_rate", 3.0}, {"zero_mid_change_ratio", 2.0},
        {"market_order_best_removal_rate", 2.0}, {"cancel_best_removal_rate", 2.0},
        {"limit_order_rate", 1.0}, {"limit_buy_rate", 0.5}, {"limit_sell_rate", 0.5},
        {"market_order_rate", 1.0}, {"market_buy_rate", 0.5}, {"market_sell_rate", 0.5},
        {"cancel_rate", 1.0}, {"cancel_bid_rate", 0.5}, {"cancel_ask_rate", 0.5}
    };
    auto it = w.find(name);
    return it == w.end() ? 1.0 : it->second;
}

double default_floor(const std::string& name) {
    static const std::unordered_map<std::string, double> f = {
        {"return_mean", 1e-6}, {"return_variance", 1e-8}, {"return_skewness", 0.05}, {"return_kurtosis", 1.0},
        {"return_acf_lag1", 0.01}, {"abs_return_acf_lag1", 0.02}, {"abs_return_acf_lag2", 0.02},
        {"abs_return_acf_lag5", 0.02}, {"abs_return_acf_lag10", 0.02}, {"abs_return_acf_lag25", 0.02},
        {"abs_return_acf_lag50", 0.02}, {"abs_return_acf_lag100", 0.02},
        {"mean_spread", 100.0}, {"spread_p90", 100.0}, {"spread_p95", 100.0},
        {"volume_volatility_corr", 0.05}, {"price_diffusion_10s", 1000.0},
        {"mean_best_bid_depth", 100.0}, {"mean_best_ask_depth", 100.0},
        {"mid_move_rate", 0.02}, {"zero_mid_change_ratio", 0.02},
        {"market_order_best_removal_rate", 0.002}, {"cancel_best_removal_rate", 0.002},
        {"limit_order_rate", 1.0}, {"limit_buy_rate", 0.5}, {"limit_sell_rate", 0.5},
        {"market_order_rate", 0.5}, {"market_buy_rate", 0.25}, {"market_sell_rate", 0.25},
        {"cancel_rate", 1.0}, {"cancel_bid_rate", 0.5}, {"cancel_ask_rate", 0.5}
    };
    auto it = f.find(name);
    return it == f.end() ? 1e-6 : it->second;
}

std::string canonical_moment_name(std::string name) {
    name = lower(trim(name));
    static const std::unordered_map<std::string, std::string> aliases = {
        {"moment", ""},
        {"mean_return", "return_mean"},
        {"var_return", "return_variance"},
        {"variance_return", "return_variance"},
        {"skew_return", "return_skewness"},
        {"skewness_return", "return_skewness"},
        {"kurt_return", "return_kurtosis"},
        {"kurtosis_return", "return_kurtosis"},
        {"acf_return_lag1", "return_acf_lag1"},
        {"acf_abs_return_lag1", "abs_return_acf_lag1"},
        {"acf_abs_return_lag2", "abs_return_acf_lag2"},
        {"acf_abs_return_lag5", "abs_return_acf_lag5"},
        {"acf_abs_return_lag10", "abs_return_acf_lag10"},
        {"acf_abs_return_lag25", "abs_return_acf_lag25"},
        {"acf_abs_return_lag50", "abs_return_acf_lag50"},
        {"acf_abs_return_lag100", "abs_return_acf_lag100"},
        {"diffusion_10s", "price_diffusion_10s"},
        {"price_diffusion_coefficient", "price_diffusion_10s"},
        {"average_bid_ask_spread", "mean_spread"},
        {"avg_spread", "mean_spread"},
        {"p90_spread", "spread_p90"},
        {"p95_spread", "spread_p95"}
    };
    auto it = aliases.find(name);
    if (it != aliases.end() && !it->second.empty()) return it->second;
    return name;
}
}

std::vector<LossTarget> load_loss_targets(const std::string& filename) {
    std::ifstream input(filename);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open empirical moment target file: " + filename);
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("Empty empirical moment target file: " + filename);
    }

    std::vector<std::string> header = split_csv(header_line);
    const int name_col = find_column(header, {"moment_name", "moment", "name", "metric"});
    const int target_col = find_column(header, {"target", "value", "empirical", "empirical_value"});
    const int weight_col = find_column(header, {"weight", "loss_weight"});
    const int floor_col = find_column(header, {"scale_floor", "floor", "denom_floor", "denominator_floor"});

    if (name_col < 0 || target_col < 0) {
        throw std::runtime_error(
            "Empirical target CSV must contain moment_name,target columns, or moment,value columns: " + filename
        );
    }

    std::vector<LossTarget> targets;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f = split_csv(line);
        if (static_cast<int>(f.size()) <= std::max(name_col, target_col)) continue;
        try {
            LossTarget target;
            target.name = canonical_moment_name(f[static_cast<std::size_t>(name_col)]);
            target.target = std::stod(f[static_cast<std::size_t>(target_col)]);
            target.weight = default_weight(target.name);
            target.scale_floor = default_floor(target.name);
            if (weight_col >= 0 && static_cast<int>(f.size()) > weight_col && !f[static_cast<std::size_t>(weight_col)].empty()) {
                target.weight = std::stod(f[static_cast<std::size_t>(weight_col)]);
            }
            if (floor_col >= 0 && static_cast<int>(f.size()) > floor_col && !f[static_cast<std::size_t>(floor_col)].empty()) {
                target.scale_floor = std::stod(f[static_cast<std::size_t>(floor_col)]);
            }
            if (!target.name.empty() && target.weight > 0.0) targets.push_back(target);
        } catch (...) {
            continue;
        }
    }
    if (targets.empty()) {
        throw std::runtime_error("No valid loss targets found in: " + filename);
    }
    return targets;
}

LossBreakdown compute_loss(const std::vector<NamedMoment>& simulated, const std::vector<LossTarget>& targets) {
    LossBreakdown b;
    b.simulated_moments = simulated;
    double weighted = 0.0;
    double total_weight = 0.0;
    for (const LossTarget& t : targets) {
        const double sim = get_moment_value(simulated, t.name, 0.0);
        const double denom = std::max(std::abs(t.target), std::max(t.scale_floor, 1e-12));
        const double rel = (sim - t.target) / denom;
        const double term = t.weight * rel * rel;
        b.loss_terms.push_back(NamedMoment{"loss_" + t.name, term});
        weighted += term;
        total_weight += t.weight;
    }
    b.total_loss = total_weight > 0.0 ? weighted / total_weight : weighted;
    return b;
}
