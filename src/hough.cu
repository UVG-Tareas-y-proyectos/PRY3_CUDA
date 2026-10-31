#include "common/pgm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {
constexpr int kDegreeIncrement = 2;
constexpr int kDegreeBins = 180 / kDegreeIncrement;
constexpr int kRBins = 100;
constexpr int kThreadsPerBlock = 256;
constexpr double kPi = 3.14159265358979323846;

void checkCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void makeTrigTables(std::array<float, kDegreeBins> &cosTable,
                    std::array<float, kDegreeBins> &sinTable) {
  for (int t = 0; t < kDegreeBins; ++t) {
    const float theta = static_cast<float>(t * kDegreeIncrement * kPi / 180.0);
    cosTable[t] = std::cos(theta);
    sinTable[t] = std::sin(theta);
  }
}

std::vector<int> houghCpu(const PGMImage &image,
                          const std::array<float, kDegreeBins> &cosTable,
                          const std::array<float, kDegreeBins> &sinTable,
                          float rMax, float rScale) {
  const int xCenter = image.width / 2;
  const int yCenter = image.height / 2;
  std::vector<int> accumulator(kDegreeBins * kRBins, 0);

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

__global__ void houghGlobalKernel(const unsigned char *image, int width, int height,
                                  int *accumulator, float rMax, float rScale,
                                  const float *cosTable, const float *sinTable) {
  const int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (globalId >= pixelCount || image[globalId] == 0) return;

  const int x = globalId % width;
  const int y = globalId / width;
  const int xCoord = x - width / 2;
  const int yCoord = height / 2 - y;

  for (int t = 0; t < kDegreeBins; ++t) {
    const float radius = xCoord * cosTable[t] + yCoord * sinTable[t];
    int rIndex = static_cast<int>((radius + rMax) / rScale);
    rIndex = max(0, min(kRBins - 1, rIndex));
    atomicAdd(&accumulator[rIndex * kDegreeBins + t], 1);
  }
}

std::vector<int> houghGpuGlobal(
    const PGMImage &image, const std::array<float, kDegreeBins> &cosTable,
    const std::array<float, kDegreeBins> &sinTable, float rMax, float rScale) {
  const std::size_t pixelBytes = image.pixels.size() * sizeof(unsigned char);
  const std::size_t accumulatorBytes = kDegreeBins * kRBins * sizeof(int);
  unsigned char *deviceImage = nullptr;
  float *deviceCos = nullptr;
  float *deviceSin = nullptr;
  int *deviceAccumulator = nullptr;

  checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceImage), pixelBytes), "cudaMalloc imagen");
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceCos), sizeof(float) * kDegreeBins),
            "cudaMalloc cosenos");
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceSin), sizeof(float) * kDegreeBins),
            "cudaMalloc senos");
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceAccumulator), accumulatorBytes),
            "cudaMalloc acumulador");

  checkCuda(cudaMemcpy(deviceImage, image.pixels.data(), pixelBytes, cudaMemcpyHostToDevice),
            "cudaMemcpy imagen");
  checkCuda(cudaMemcpy(deviceCos, cosTable.data(), sizeof(float) * kDegreeBins,
                       cudaMemcpyHostToDevice), "cudaMemcpy cosenos");
  checkCuda(cudaMemcpy(deviceSin, sinTable.data(), sizeof(float) * kDegreeBins,
                       cudaMemcpyHostToDevice), "cudaMemcpy senos");
  checkCuda(cudaMemset(deviceAccumulator, 0, accumulatorBytes), "cudaMemset acumulador");

  const int pixelCount = image.width * image.height;
  const int blocks = (pixelCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
  houghGlobalKernel<<<blocks, kThreadsPerBlock>>>(
      deviceImage, image.width, image.height, deviceAccumulator, rMax, rScale,
      deviceCos, deviceSin);
  checkCuda(cudaGetLastError(), "lanzamiento del kernel global");
  checkCuda(cudaDeviceSynchronize(), "ejecucion del kernel global");

  std::vector<int> accumulator(kDegreeBins * kRBins);
  checkCuda(cudaMemcpy(accumulator.data(), deviceAccumulator, accumulatorBytes,
                       cudaMemcpyDeviceToHost), "cudaMemcpy acumulador");

  checkCuda(cudaFree(deviceAccumulator), "cudaFree acumulador");
  checkCuda(cudaFree(deviceSin), "cudaFree senos");
  checkCuda(cudaFree(deviceCos), "cudaFree cosenos");
  checkCuda(cudaFree(deviceImage), "cudaFree imagen");
  return accumulator;
}

bool reportComparison(const std::vector<int> &cpu, const std::vector<int> &gpu) {
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    if (cpu[i] != gpu[i]) {
      if (mismatches == 0)
        std::cerr << "Primera diferencia en bin " << i << ": CPU=" << cpu[i]
                  << ", GPU=" << gpu[i] << '\n';
      ++mismatches;
    }
  }
  if (mismatches == 0) {
    std::cout << "Comparacion CPU/GPU: todos los bins coinciden.\n";
    return true;
  }
  std::cerr << "Comparacion CPU/GPU: " << mismatches << " bins diferentes.\n";
  return false;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Uso: " << argv[0] << " <imagen.pgm>\n";
    return 2;
  }

  try {
    const PGMImage image(argv[1]);
    std::array<float, kDegreeBins> cosTable{};
    std::array<float, kDegreeBins> sinTable{};
    makeTrigTables(cosTable, sinTable);

    const float rMax = static_cast<float>(
        std::sqrt(static_cast<double>(image.width) * image.width +
                  static_cast<double>(image.height) * image.height) / 2.0);
    const float rScale = 2.0f * rMax / kRBins;
    const std::vector<int> cpu = houghCpu(image, cosTable, sinTable, rMax, rScale);
    const std::vector<int> gpu = houghGpuGlobal(image, cosTable, sinTable, rMax, rScale);
    std::cout << "CUDA global lista: " << image.width << "x" << image.height << '\n';
    return reportComparison(cpu, gpu) ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
