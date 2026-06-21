CXX = mpicxx

CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

TARGET = calibrate

SRCS = main.cpp \
       SimulationRunner.cpp \
       LimitOrderBook.cpp \
       HawkesProcess.cpp \
       EmpiricalDistribution.cpp \
       MarketMakerAgent.cpp \
       MomentumAgent.cpp \
       LargeInstitutionalAgent.cpp\
       AutonomousMarketMakerAgent.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -lstdc++fs

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

clean_outputs:
	rm -rf calibration_results outputs

.PHONY: all clean clean_outputs
