CXX = mpicxx
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra -I./src

COMMON_SRC = \
    src/LimitOrderBook.cpp \
    src/EmpiricalDistribution.cpp \
    src/BackgroundHawkesAgent.cpp \
    src/PassiveMarketMakerAgent.cpp \
    src/MomentumAgent.cpp \
    src/LargeInstitutionalAgent.cpp \
    src/ExperimentalStrategy.cpp \
    src/ExperimentRunner.cpp

.PHONY: all clean check

all: strategy_1 strategy_2 strategy_3 tuner validator tester hawkes_full_mpi

strategy_1: $(COMMON_SRC) src/strategy_1_inventory.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

strategy_2: $(COMMON_SRC) src/strategy_2_queue.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

strategy_3: $(COMMON_SRC) src/strategy_3_hawkes.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

tuner: $(COMMON_SRC) src/hyperparameter_tuner.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

validator: $(COMMON_SRC) src/hyperparameter_validator.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

tester: $(COMMON_SRC) src/final_tester.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

hawkes_full_mpi: \
    src/LimitOrderBook.cpp \
    src/EmpiricalDistribution.cpp \
    src/BackgroundHawkesAgent.cpp \
    src/PassiveMarketMakerAgent.cpp \
    src/MomentumAgent.cpp \
    src/LargeInstitutionalAgent.cpp \
    src/FullSimulator.cpp \
    src/MomentEstimator.cpp \
    src/LossFunction.cpp \
    src/calibration_main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

check:
	@echo "CXX=$(CXX)"
	@which $(CXX)
	@$(CXX) --version | head -3

clean:
	rm -f strategy_1 strategy_2 strategy_3 tuner validator tester hawkes_full_mpi
