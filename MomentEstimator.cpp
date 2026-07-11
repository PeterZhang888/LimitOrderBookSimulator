#include "MomentEstimator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace {
double mean(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

double variance(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    const double m = mean(x);
    double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return s / static_cast<double>(x.size() - 1);
}

double percentile(std::vector<double> x, double p) {
    if (x.empty()) return 0.0;
    std::sort(x.begin(), x.end());
    const double position = p * static_cast<double>(x.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(position));
    if (lo == hi) return x[lo];
    const double w = position - static_cast<double>(lo);
    return x[lo] * (1.0 - w) + x[hi] * w;
}

double skewness(const std::vector<double>& x) {
    if (x.size() < 3) return 0.0;
    const double m = mean(x);
    const double var = variance(x);
    if (var <= 1e-30) return 0.0;
    const double sd = std::sqrt(var);
    double s3 = 0.0;
    for (double v : x) s3 += std::pow((v - m) / sd, 3.0);
    return s3 / static_cast<double>(x.size());
}

double kurtosis(const std::vector<double>& x) {
    if (x.size() < 4) return 0.0;
    const double m = mean(x);
    const double var = variance(x);
    if (var <= 1e-30) return 0.0;
    const double sd = std::sqrt(var);
    double s4 = 0.0;
    for (double v : x) s4 += std::pow((v - m) / sd, 4.0);
    return s4 / static_cast<double>(x.size()); // non-excess kurtosis.
}

double autocorr(const std::vector<double>& x, int lag) {
    if (lag <= 0 || static_cast<std::size_t>(lag) >= x.size()) return 0.0;
    const double m = mean(x);
    double numerator = 0.0;
    double denominator = 0.0;
    for (double v : x) denominator += (v - m) * (v - m);
    if (denominator <= 1e-30) return 0.0;
    for (std::size_t i = static_cast<std::size_t>(lag); i < x.size(); ++i) {
        numerator += (x[i] - m) * (x[i - static_cast<std::size_t>(lag)] - m);
    }
    return numerator / denominator;
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = std::min(x.size(), y.size());
    if (n < 2) return 0.0;
    std::vector<double> xx(x.begin(), x.begin() + static_cast<long>(n));
    std::vector<double> yy(y.begin(), y.begin() + static_cast<long>(n));
    const double mx = mean(xx);
    const double my = mean(yy);
    double num = 0.0, vx = 0.0, vy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = xx[i] - mx;
        const double dy = yy[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if (vx <= 1e-30 || vy <= 1e-30) return 0.0;
    return num / std::sqrt(vx * vy);
}

std::vector<double> log_returns(const std::vector<double>& mid) {
    std::vector<double> r;
    if (mid.size() < 2) return r;
    r.reserve(mid.size() - 1);
    for (std::size_t i = 1; i < mid.size(); ++i) {
        if (mid[i] > 0.0 && mid[i - 1] > 0.0) {
            r.push_back(std::log(mid[i]) - std::log(mid[i - 1]));
        }
    }
    return r;
}

std::vector<double> abs_values(const std::vector<double>& x) {
    std::vector<double> y;
    y.reserve(x.size());
    for (double v : x) y.push_back(std::abs(v));
    return y;
}

std::vector<double> price_changes_at_lag(const std::vector<double>& mid, int lag) {
    std::vector<double> changes;
    if (lag <= 0 || static_cast<std::size_t>(lag) >= mid.size()) return changes;
    changes.reserve(mid.size() - static_cast<std::size_t>(lag));
    for (std::size_t i = static_cast<std::size_t>(lag); i < mid.size(); ++i) {
        changes.push_back(mid[i] - mid[i - static_cast<std::size_t>(lag)]);
    }
    return changes;
}
}

const std::vector<std::string>& standard_moment_names() {
    static const std::vector<std::string> names = {
        "return_mean",
        "return_variance",
        "return_skewness",
        "return_kurtosis",
        "return_acf_lag1",
        "abs_return_acf_lag1",
        "abs_return_acf_lag2",
        "abs_return_acf_lag5",
        "abs_return_acf_lag10",
        "abs_return_acf_lag25",
        "abs_return_acf_lag50",
        "abs_return_acf_lag100",
        "mean_spread",
        "spread_p90",
        "spread_p95",
        "volume_volatility_corr",
        "price_diffusion_10s",
        "mean_best_bid_depth",
        "mean_best_ask_depth",
        "mid_move_rate",
        "zero_mid_change_ratio",
        "market_order_best_removal_rate",
        "cancel_best_removal_rate",
        "limit_order_rate",
        "limit_buy_rate",
        "limit_sell_rate",
        "market_order_rate",
        "market_buy_rate",
        "market_sell_rate",
        "cancel_rate",
        "cancel_bid_rate",
        "cancel_ask_rate"
    };
    return names;
}

std::vector<NamedMoment> estimate_moments(const SimulationMetrics& metrics) {
    const std::vector<double> r = log_returns(metrics.mid_prices);
    const std::vector<double> ar = abs_values(r);

    std::vector<double> mid_changes;
    if (metrics.mid_prices.size() >= 2) {
        mid_changes.reserve(metrics.mid_prices.size() - 1);
        for (std::size_t i = 1; i < metrics.mid_prices.size(); ++i) {
            mid_changes.push_back(metrics.mid_prices[i] - metrics.mid_prices[i - 1]);
        }
    }

    double zero_mid_change_ratio = 0.0;
    double mid_move_rate = 0.0;
    if (!mid_changes.empty()) {
        std::size_t zeros = 0;
        for (double x : mid_changes) {
            if (std::abs(x) < 1e-12) ++zeros;
        }
        zero_mid_change_ratio = static_cast<double>(zeros) / static_cast<double>(mid_changes.size());
        mid_move_rate = 1.0 - zero_mid_change_ratio;
    }

    int diffusion_lag = 10;
    if (metrics.sample_interval_seconds > 1e-12) {
        diffusion_lag = std::max(1, static_cast<int>(std::llround(10.0 / metrics.sample_interval_seconds)));
    }
    const std::vector<double> changes_10s = price_changes_at_lag(metrics.mid_prices, diffusion_lag);

    std::vector<double> volumes = metrics.aggressive_volume_by_sample;
    if (!volumes.empty() && !ar.empty()) {
        // Align volume intervals with return intervals: the first snapshot volume has no preceding return.
        if (volumes.size() > ar.size()) volumes.erase(volumes.begin());
    }

    const double duration = std::max(1e-9, metrics.duration_seconds);

    std::vector<NamedMoment> moments;
    auto add = [&](const std::string& name, double value) {
        moments.push_back(NamedMoment{name, std::isfinite(value) ? value : 0.0});
    };

    add("return_mean", mean(r));
    add("return_variance", variance(r));
    add("return_skewness", skewness(r));
    add("return_kurtosis", kurtosis(r));
    add("return_acf_lag1", autocorr(r, 1));
    add("abs_return_acf_lag1", autocorr(ar, 1));
    add("abs_return_acf_lag2", autocorr(ar, 2));
    add("abs_return_acf_lag5", autocorr(ar, 5));
    add("abs_return_acf_lag10", autocorr(ar, 10));
    add("abs_return_acf_lag25", autocorr(ar, 25));
    add("abs_return_acf_lag50", autocorr(ar, 50));
    add("abs_return_acf_lag100", autocorr(ar, 100));
    add("mean_spread", mean(metrics.spreads));
    add("spread_p90", percentile(metrics.spreads, 0.90));
    add("spread_p95", percentile(metrics.spreads, 0.95));
    add("volume_volatility_corr", correlation(volumes, ar));
    add("price_diffusion_10s", variance(changes_10s));
    add("mean_best_bid_depth", mean(metrics.best_bid_depths));
    add("mean_best_ask_depth", mean(metrics.best_ask_depths));
    add("mid_move_rate", mid_move_rate);
    add("zero_mid_change_ratio", zero_mid_change_ratio);
    add("market_order_best_removal_rate", metrics.market_order_best_removal_rate);
    add("cancel_best_removal_rate", metrics.cancel_best_removal_rate);
    add("limit_order_rate", metrics.limit_order_count / duration);
    add("limit_buy_rate", metrics.limit_buy_count / duration);
    add("limit_sell_rate", metrics.limit_sell_count / duration);
    add("market_order_rate", metrics.market_order_count / duration);
    add("market_buy_rate", metrics.market_buy_count / duration);
    add("market_sell_rate", metrics.market_sell_count / duration);
    add("cancel_rate", metrics.cancel_count / duration);
    add("cancel_bid_rate", metrics.cancel_bid_count / duration);
    add("cancel_ask_rate", metrics.cancel_ask_count / duration);
    return moments;
}

double get_moment_value(const std::vector<NamedMoment>& moments, const std::string& name, double fallback) {
    for (const auto& m : moments) {
        if (m.name == name) return m.value;
    }
    return fallback;
}
