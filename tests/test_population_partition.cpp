#include "agents/AgentPopulation.hpp"

#include <cassert>
#include <iostream>

int main() {
    using namespace dlob;

    PopulationConfig config;
    config.market_makers = 3;
    config.momentum_traders = 600;
    config.informed_traders = 290;
    config.institutional_traders = 10;
    config.population_scale = 1;
    config.simulation_end_ns = 1'000'000'000LL;

    PopulationSummary summed;
    for (int rank = 1; rank < 32; ++rank) {
        AgentPopulation population(rank, 32, config);
        const PopulationSummary local = population.local_summary();
        summed.market_makers += local.market_makers;
        summed.momentum += local.momentum;
        summed.informed += local.informed;
        summed.institutional += local.institutional;
    }

    assert(summed.market_makers == 3);
    assert(summed.momentum == 600);
    assert(summed.informed == 290);
    assert(summed.institutional == 10);
    assert(summed.total() == 903);

    AgentPopulation exchange(0, 32, config);
    assert(!exchange.is_worker());
    assert(exchange.local_summary().total() == 0);

    std::cout << "population partition tests passed\n";
    return 0;
}
