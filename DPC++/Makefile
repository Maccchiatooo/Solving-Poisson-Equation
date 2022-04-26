CXX = clang++
CXXFLAGS ?= -std=c++17 -mfma -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wno-unknown-cuda-version 
#CXXFLAGS ?= -std=c++17 -fsycl -fsycl-targets=nvptx64-unknown-nvidiacl -Wno-unknown-cuda-version 

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))
RUN :=  $(addprefix run_, $(PRGS))

.PHONY: run_all
run_all: $(sort $(RUN))

.PHONY: all
all: $(PRGS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

run_%:%
	./$^

.PHONY: clean
clean:
	rm -f $(PRGS)
