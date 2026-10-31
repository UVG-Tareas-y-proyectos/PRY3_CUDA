#include "common/pgm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kDegreeIncrement = 2;
constexpr int kDegreeBins = 180 / kDegreeIncrement;
constexpr int kRBins = 100;
constexpr double kPi = 3.14159265358979323846;

std::vector<int> houghCpu(const PGMImage &image) {
  const float rMax = static_cast<float>(
      std::sqrt(static_cast<double>(image.width) * image.width +
                static_cast<double>(image.height) * image.height) / 2.0);
  const float rScale = 2.0f * rMax / kRBins;
  const int xCenter = image.width / 2;
  const int yCenter = image.height / 2;
  std::vector<int> accumulator(kDegreeBins * kRBins, 0);
  std::array<float, kDegreeBins> cosTable{};
  std::array<float, kDegreeBins> sinTable{};
  for (int t = 0; t < kDegreeBins; ++t) {
    const float theta = static_cast<float>(t * kDegreeIncrement * kPi / 180.0);
    cosTable[t] = std::cos(theta);
    sinTable[t] = std::sin(theta);
  }

  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      if (image.pixels[static_cast<std::size_t>(y) * image.width + x] == 0) continue;
      const int xCoord = x - xCenter;
      const int yCoord = yCenter - y;
      for (int t = 0; t < kDegreeBins; ++t) {
        const float radius = xCoord * cosTable[t] + yCoord * sinTable[t];
        const int rIndex = std::clamp(static_cast<int>((radius + rMax) / rScale),
                                      0, kRBins - 1);
        ++accumulator[rIndex * kDegreeBins + t];
      }
    }
  }
  return accumulator;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Uso: " << argv[0] << " <imagen.pgm>\n";
    return 2;
  }

  try {
    const PGMImage image(argv[1]);
    const std::vector<int> accumulator = houghCpu(image);
    const int maxVotes = *std::max_element(accumulator.begin(), accumulator.end());
    std::cout << "Referencia CPU lista: " << image.width << "x" << image.height
              << ", maximo de votos=" << maxVotes << '\n';
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
