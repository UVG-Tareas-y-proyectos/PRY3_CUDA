#include "common/pgm.h"
#include "common/output.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
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
constexpr int kTimingRuns = 10;
constexpr double kPi = 3.14159265358979323846;

__constant__ float constantCos[kDegreeBins];
__constant__ float constantSin[kDegreeBins];

struct TimedResult {
  std::vector<int> accumulator;
  std::array<float, kTimingRuns> measurementsMilliseconds{};
  float averageMilliseconds = 0.0f;
};

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

__global__ void houghConstantKernel(const unsigned char *image, int width, int height,
                                    int *accumulator, float rMax, float rScale) {
  const int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (globalId >= pixelCount || image[globalId] == 0) return;

  const int x = globalId % width;
  const int y = globalId / width;
  const int xCoord = x - width / 2;
  const int yCoord = height / 2 - y;

  for (int t = 0; t < kDegreeBins; ++t) {
    const float radius = xCoord * constantCos[t] + yCoord * constantSin[t];
    int rIndex = static_cast<int>((radius + rMax) / rScale);
    rIndex = max(0, min(kRBins - 1, rIndex));
    atomicAdd(&accumulator[rIndex * kDegreeBins + t], 1);
  }
}

// Un acumulador de 9000 enteros (36000 bytes) pertenece a cada bloque.
// Incluso los hilos del ultimo bloque que exceden la imagen deben alcanzar
// AMBAS barreras. Salir anticipadamente aqui produciria una sincronizacion invalida.
__global__ void houghSharedKernel(const unsigned char *image, int width, int height,
                                  int *accumulator, float rMax, float rScale) {
  __shared__ int localAcc[kDegreeBins * kRBins];
  const int locID = threadIdx.x;
  const int globalId = blockIdx.x * blockDim.x + locID;
  for (int i = locID; i < kDegreeBins * kRBins; i += blockDim.x)
    localAcc[i] = 0;
  __syncthreads();  // Nadie vota antes de terminar la inicializacion cooperativa.

  if (globalId < width * height && image[globalId] != 0) {
    const int xCoord = globalId % width - width / 2;
    const int yCoord = height / 2 - globalId / width;
    for (int t = 0; t < kDegreeBins; ++t) {
      const float radius = xCoord * constantCos[t] + yCoord * constantSin[t];
      int rIndex = static_cast<int>((radius + rMax) / rScale);
      rIndex = max(0, min(kRBins - 1, rIndex));
      // Diferentes pixeles pueden votar por la misma recta, aun dentro del bloque.
      atomicAdd(&localAcc[rIndex * kDegreeBins + t], 1);
    }
  }
  __syncthreads();  // La fusion solo lee votos completamente actualizados.
  for (int i = locID; i < kDegreeBins * kRBins; i += blockDim.x) {
    const int votes = localAcc[i];
    if (votes != 0) atomicAdd(&accumulator[i], votes);
  }
}

template <typename LaunchKernel>
TimedResult measureKernel(LaunchKernel launch, int *deviceAccumulator,
                          std::size_t accumulatorBytes) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate inicio");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate fin");

  checkCuda(cudaMemset(deviceAccumulator, 0, accumulatorBytes), "cudaMemset warmup");
  launch();
  checkCuda(cudaGetLastError(), "lanzamiento warmup");
  checkCuda(cudaDeviceSynchronize(), "warmup del kernel");

  TimedResult result;
  float totalMilliseconds = 0.0f;
  for (int run = 0; run < kTimingRuns; ++run) {
    checkCuda(cudaMemset(deviceAccumulator, 0, accumulatorBytes),
              "cudaMemset antes de medir");
    checkCuda(cudaEventRecord(start), "cudaEventRecord inicio");
    launch();
    checkCuda(cudaGetLastError(), "lanzamiento del kernel medido");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord fin");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize fin");
    float elapsed = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    result.measurementsMilliseconds[run] = elapsed;
    totalMilliseconds += elapsed;
  }

  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy fin");
  checkCuda(cudaEventDestroy(start), "cudaEventDestroy inicio");

  result.accumulator.resize(kDegreeBins * kRBins);
  checkCuda(cudaMemcpy(result.accumulator.data(), deviceAccumulator,
                       accumulatorBytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy acumulador medido");
  result.averageMilliseconds = totalMilliseconds / kTimingRuns;
  return result;
}

TimedResult houghGpuGlobal(
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
  const TimedResult result = measureKernel(
      [&]() {
        houghGlobalKernel<<<blocks, kThreadsPerBlock>>>(
            deviceImage, image.width, image.height, deviceAccumulator, rMax,
            rScale, deviceCos, deviceSin);
      },
      deviceAccumulator, accumulatorBytes);

  checkCuda(cudaFree(deviceAccumulator), "cudaFree acumulador");
  checkCuda(cudaFree(deviceSin), "cudaFree senos");
  checkCuda(cudaFree(deviceCos), "cudaFree cosenos");
  checkCuda(cudaFree(deviceImage), "cudaFree imagen");
  return result;
}

TimedResult houghGpuConstant(const PGMImage &image,
                             const std::array<float, kDegreeBins> &cosTable,
                             const std::array<float, kDegreeBins> &sinTable,
                             float rMax, float rScale, bool useShared = false) {
  const std::size_t pixelBytes = image.pixels.size() * sizeof(unsigned char);
  const std::size_t accumulatorBytes = kDegreeBins * kRBins * sizeof(int);
  unsigned char *deviceImage = nullptr;
  int *deviceAccumulator = nullptr;

  checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceImage), pixelBytes),
            "cudaMalloc imagen constante");
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceAccumulator), accumulatorBytes),
            "cudaMalloc acumulador constante");
  checkCuda(cudaMemcpy(deviceImage, image.pixels.data(), pixelBytes,
                       cudaMemcpyHostToDevice), "cudaMemcpy imagen constante");
  checkCuda(cudaMemcpyToSymbol(constantCos, cosTable.data(), sizeof(float) * kDegreeBins),
            "cudaMemcpyToSymbol cosenos");
  checkCuda(cudaMemcpyToSymbol(constantSin, sinTable.data(), sizeof(float) * kDegreeBins),
            "cudaMemcpyToSymbol senos");

  const int pixelCount = image.width * image.height;
  const int blocks = (pixelCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const TimedResult result = measureKernel(
      [&]() {
        if (useShared)
          houghSharedKernel<<<blocks, kThreadsPerBlock>>>(
              deviceImage, image.width, image.height, deviceAccumulator, rMax, rScale);
        else houghConstantKernel<<<blocks, kThreadsPerBlock>>>(
            deviceImage, image.width, image.height, deviceAccumulator, rMax,
            rScale);
      },
      deviceAccumulator, accumulatorBytes);

  checkCuda(cudaFree(deviceAccumulator), "cudaFree acumulador constante");
  checkCuda(cudaFree(deviceImage), "cudaFree imagen constante");
  return result;
}

bool reportComparison(const char *variant, const std::vector<int> &cpu,
                      const std::vector<int> &gpu) {
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
    std::cout << "Comparacion CPU/" << variant << ": todos los bins coinciden.\n";
    return true;
  }
  std::cerr << "Comparacion CPU/" << variant << ": " << mismatches
            << " bins diferentes.\n";
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
    TimedResult global = houghGpuGlobal(image, cosTable, sinTable, rMax, rScale);
    TimedResult constant = houghGpuConstant(image, cosTable, sinTable, rMax, rScale);
    TimedResult shared = houghGpuConstant(image, cosTable, sinTable, rMax, rScale, true);

    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    std::cout << std::fixed << std::setprecision(4)
              << "GPU: " << properties.name << "\n"
              << "Imagen: " << image.width << "x" << image.height << '\n'
              << "Promedio kernel global (" << kTimingRuns << " ejecuciones): "
              << global.averageMilliseconds << " ms\n"
              << "Tiempos global (ms):";
    for (float milliseconds : global.measurementsMilliseconds)
      std::cout << ' ' << milliseconds;
    std::cout << "\nPromedio kernel constante (" << kTimingRuns << " ejecuciones): "
              << constant.averageMilliseconds << " ms\n"
              << "Tiempos constante (ms):";
    for (float milliseconds : constant.measurementsMilliseconds)
      std::cout << ' ' << milliseconds;
    std::cout << "\nPromedio kernel compartido: " << shared.averageMilliseconds
              << " ms\nTiempos compartido (ms):";
    for (float milliseconds : shared.measurementsMilliseconds)
      std::cout << ' ' << milliseconds;
    std::cout << '\n';
    const bool globalMatches = reportComparison("global", cpu, global.accumulator);
    const bool constantMatches = reportComparison("constante", cpu, constant.accumulator);
    const bool sharedMatches = reportComparison("compartida", cpu, shared.accumulator);
    if (globalMatches && constantMatches && sharedMatches) {
      const int threshold = static_cast<int>(*std::max_element(cpu.begin(), cpu.end()) * 0.55);
      const auto lines = selectLines(shared.accumulator, kDegreeBins, kRBins, threshold, 12);
      writeOverlay("runway_lines.png", image, lines, kDegreeIncrement, rMax, rScale);
      writeLines("runway_lines.csv", lines, kDegreeIncrement, rMax, rScale);
      std::cout << "Lineas detectadas: " << lines.size() << ", umbral estricto > " << threshold << '\n';
    }
    checkCuda(cudaDeviceReset(), "cudaDeviceReset");
    return globalMatches && constantMatches && sharedMatches ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
