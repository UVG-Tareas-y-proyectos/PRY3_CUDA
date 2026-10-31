CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic

TARGET := build/hough
SOURCES := src/hough.cpp common/pgm.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) common/pgm.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) -I. $(SOURCES) -o $@

run: $(TARGET)
	./$(TARGET) data/runway.pgm

clean:
	rm -rf build
