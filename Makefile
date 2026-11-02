NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17 --fmad=false -Xcompiler=-Wall,-Wextra,-ffp-contract=off

TARGET := build/hough
SOURCES := src/hough.cu common/pgm.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) common/pgm.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -I. $(SOURCES) -o $@

run: $(TARGET)
	./$(TARGET) data/runway.pgm

clean:
	rm -rf build
