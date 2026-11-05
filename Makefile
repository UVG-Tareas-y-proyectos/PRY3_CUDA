NVCC ?= nvcc
HOST_CXX ?= g++
CUDA_ARCH ?= 86
NVCCFLAGS ?= -O2 -std=c++17 --fmad=false -Xcompiler=-Wall,-Wextra,-ffp-contract=off
EXTRA_NVCCFLAGS ?=
PYTHON ?= python3
RUNS ?= 30
INPUT ?= data/runway.pgm
OUTPUT ?= results/runway

TARGET := build/hough
SOURCES := src/hough.cu common/pgm.cpp common/output.cpp

.PHONY: all clean run benchmark test sanitize

all: $(TARGET)

$(TARGET): $(SOURCES) common/pgm.h common/output.h
	mkdir -p build
	$(NVCC) -ccbin $(HOST_CXX) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -I. $(SOURCES) -o $@

run: $(TARGET)
	./$(TARGET) $(INPUT) --output $(OUTPUT)

benchmark: $(TARGET)
	$(PYTHON) scripts/benchmark.py --binary ./$(TARGET) --image $(INPUT) --output $(OUTPUT) --runs $(RUNS)

test: $(TARGET)
	$(PYTHON) tests/test_integration.py --binary ./$(TARGET)

sanitize: $(TARGET)
	bash scripts/sanitize.sh ./$(TARGET)

clean:
	rm -rf build
