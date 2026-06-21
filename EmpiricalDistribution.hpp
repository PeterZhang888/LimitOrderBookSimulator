#ifndef EMPIRICAL_DISTRIBUTION_HPP
#define EMPIRICAL_DISTRIBUTION_HPP

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

class EmpiricalDistribution {
public:
    EmpiricalDistribution();

    void load_from_csv(
        const std::string& file_path,
        const std::string& value_column
    );

    int sample(std::mt19937_64& rng) const;

    std::size_t size() const;

private:
    std::vector<int> values_;
    std::vector<double> cumulative_probabilities_;
};

#endif