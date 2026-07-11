#pragma once

#include <random>
#include <string>
#include <vector>

class EmpiricalDistribution {
public:
    bool load_from_csv(const std::string& filename, const std::string& column_name);
    int sample(std::mt19937_64& rng) const;
    bool empty() const { return values_.empty(); }
    void set_fallback(int lower, int upper);

private:
    std::vector<int> values_;
    int fallback_lower_ = 1;
    int fallback_upper_ = 100;
};
