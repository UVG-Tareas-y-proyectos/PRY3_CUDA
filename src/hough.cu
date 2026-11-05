#include "common/pgm.h"
#include "common/output.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
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
constexpr int kDefaultRuns = 10;
constexpr int kWarmupRuns = 5;
constexpr double kPi = 3.14159265358979323846;

__constant__ float constantCos[kDegreeBins];
__constant__ float constantSin[kDegreeBins];

struct TimedResult {
  std::vector<int> accumulator;
  std::vector<float> measurementsMilliseconds;
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


// Propiedad de memoria y eventos: tambien se liberan durante excepciones.
template <typename T> class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  T *get() const { return data_; }
 private:
  T *data_ = nullptr;
};

class Event {
 public:
  Event() { checkCuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() { if (event_) cudaEventDestroy(event_); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  cudaEvent_t get() const { return event_; }
 private:
  cudaEvent_t event_ = nullptr;
};

enum class Variant { Global, Constant, Shared };
const char *variantName(Variant variant) {
  switch (variant) {
    case Variant::Global: return "global";
    case Variant::Constant: return "constante";
    case Variant::Shared: return "compartida";
  }
  return "desconocida";
}

struct Options {
  std::string input;
  std::string output = "results/runway";
  int runs = kDefaultRuns;
  int threshold = -1;
  int maximumLines = 12;
};

int integerOption(const std::string &text, const char *name, int minimum, int maximum) {
  std::size_t end = 0;
  long value = 0;
  try { value = std::stol(text, &end); }
  catch (...) { throw std::runtime_error(std::string("Entero invalido para ") + name); }
  if (end != text.size() || value < minimum || value > maximum)
    throw std::runtime_error(std::string("Fuera de rango: ") + name);
  return static_cast<int>(value);
}

Options parseOptions(int argc, char **argv) {
  Options options;
  if (argc < 2) throw std::runtime_error("Falta imagen PGM. Use --help");
  options.input = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("Falta valor para " + key);
    const std::string value = argv[++i];
    if (key == "--runs") options.runs = integerOption(value, "--runs", 1, 10000);
    else if (key == "--threshold") options.threshold = integerOption(value, "--threshold", 0, 2147483647);
    else if (key == "--max-lines") options.maximumLines = integerOption(value, "--max-lines", 1, 1000);
    else if (key == "--output") options.output = value;
    else throw std::runtime_error("Opcion desconocida: " + key);
  }
  if (options.output.empty()) throw std::runtime_error("Prefijo de salida vacio");
  return options;
}

TimedResult houghGpu(const PGMImage &image,
                     const std::array<float, kDegreeBins> &cosTable,
                     const std::array<float, kDegreeBins> &sinTable,
                     float rMax, float rScale, Variant variant, int runs,
                     const std::vector<int> &reference) {
  const std::size_t accumulatorBytes = kDegreeBins * kRBins * sizeof(int);
  DeviceBuffer<unsigned char> deviceImage(image.pixels.size());
  DeviceBuffer<int> deviceAccumulator(kDegreeBins * kRBins);
  // Solo la version global usa estas tablas en memoria global.
  DeviceBuffer<float> deviceTrig(variant == Variant::Global ? 2 * kDegreeBins : 1);
  checkCuda(cudaMemcpy(deviceImage.get(), image.pixels.data(), image.pixels.size(),
                       cudaMemcpyHostToDevice), "cudaMemcpy imagen");
  if (variant == Variant::Global) {
    checkCuda(cudaMemcpy(deviceTrig.get(), cosTable.data(), sizeof(float) * kDegreeBins,
                         cudaMemcpyHostToDevice), "cudaMemcpy cosenos");
    checkCuda(cudaMemcpy(deviceTrig.get() + kDegreeBins, sinTable.data(), sizeof(float) * kDegreeBins,
                         cudaMemcpyHostToDevice), "cudaMemcpy senos");
  } else {
    checkCuda(cudaMemcpyToSymbol(constantCos, cosTable.data(), sizeof(float) * kDegreeBins),
              "cudaMemcpyToSymbol cosenos");
    checkCuda(cudaMemcpyToSymbol(constantSin, sinTable.data(), sizeof(float) * kDegreeBins),
              "cudaMemcpyToSymbol senos");
  }
  const int pixels = image.width * image.height;
  const int blocks = (pixels - 1) / kThreadsPerBlock + 1;
  auto launch = [&]() {
    if (variant == Variant::Global)
      houghGlobalKernel<<<blocks, kThreadsPerBlock>>>(deviceImage.get(), image.width, image.height,
          deviceAccumulator.get(), rMax, rScale, deviceTrig.get(), deviceTrig.get() + kDegreeBins);
    else if (variant == Variant::Constant)
      houghConstantKernel<<<blocks, kThreadsPerBlock>>>(deviceImage.get(), image.width, image.height,
          deviceAccumulator.get(), rMax, rScale);
    else
      houghSharedKernel<<<blocks, kThreadsPerBlock>>>(deviceImage.get(), image.width, image.height,
          deviceAccumulator.get(), rMax, rScale);
    checkCuda(cudaGetLastError(), "lanzamiento del kernel");
  };

  Event start, stop;
  TimedResult result;
  result.accumulator.resize(kDegreeBins * kRBins);
  result.measurementsMilliseconds.reserve(runs);
  for (int run = -kWarmupRuns; run < runs; ++run) {
    // Reinicio y transferencias quedan FUERA del intervalo medido.
    checkCuda(cudaMemset(deviceAccumulator.get(), 0, accumulatorBytes), "cudaMemset acumulador");
    if (run >= 0) checkCuda(cudaEventRecord(start.get()), "cudaEventRecord inicio");
    launch();
    if (run < 0) {
      checkCuda(cudaDeviceSynchronize(), "warmup");
      continue;
    }
    checkCuda(cudaEventRecord(stop.get()), "cudaEventRecord fin");
    checkCuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize");
    float elapsed = 0;
    checkCuda(cudaEventElapsedTime(&elapsed, start.get(), stop.get()), "cudaEventElapsedTime");
    result.measurementsMilliseconds.push_back(elapsed);
    result.averageMilliseconds += elapsed / runs;
    checkCuda(cudaMemcpy(result.accumulator.data(), deviceAccumulator.get(), accumulatorBytes,
                         cudaMemcpyDeviceToHost), "cudaMemcpy acumulador");
    // Cada repeticion medida debe coincidir exactamente en los 9000 bins.
    if (result.accumulator != reference)
      throw std::runtime_error(std::string("Diferencias CPU/GPU en ") + variantName(variant)
                               + ", repeticion " + std::to_string(run + 1));
  }
  return result;
}

void ensureParent(const std::string &path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
}

void writeTimings(const std::string &path, const std::array<TimedResult, 3> &results) {
  std::ofstream out(path);
  out << "run,global_ms,constant_ms,shared_ms\n" << std::fixed << std::setprecision(6);
  for (std::size_t run = 0; run < results[0].measurementsMilliseconds.size(); ++run) {
    out << run + 1;
    for (const auto &result : results) out << ',' << result.measurementsMilliseconds[run];
    out << '\n';
  }
  if (!out) throw std::runtime_error("No se pudo escribir tiempos: " + path);
}

std::string jsonString(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
    else out << c;
  }
  out << '"';
  return out.str();
}

void writeMetadata(const Options &options, const PGMImage &image,
                    const cudaDeviceProp &properties, double cpuMs, int threshold,
                    std::size_t lineCount, const std::array<TimedResult, 3> &results) {
  int driver = 0, runtime = 0;
  checkCuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
  checkCuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
  std::ofstream out(options.output + "_metadata.json");
  out << std::fixed << std::setprecision(6)
      << "{\n  \"input\": " << jsonString(options.input)
      << ",\n  \"gpu\": " << jsonString(properties.name)
      << ",\n  \"compute_capability\": \"" << properties.major << '.' << properties.minor << "\""
      << ",\n  \"cuda_runtime\": " << runtime << ",\n  \"cuda_driver_api\": " << driver
      << ",\n  \"width\": " << image.width << ",\n  \"height\": " << image.height
      << ",\n  \"active_pixels\": " << std::count_if(image.pixels.begin(), image.pixels.end(), [](auto v) { return v > 0; })
      << ",\n  \"runs_per_variant\": " << options.runs << ",\n  \"warmup_runs_per_variant\": " << kWarmupRuns
      << ",\n  \"threads_per_block\": " << kThreadsPerBlock
      << ",\n  \"blocks\": " << (image.pixels.size() - 1) / kThreadsPerBlock + 1
      << ",\n  \"degree_bins\": " << kDegreeBins << ",\n  \"radius_bins\": " << kRBins
      << ",\n  \"shared_bytes_per_block\": " << kDegreeBins * kRBins * sizeof(int)
      << ",\n  \"device_shared_limit_bytes\": " << properties.sharedMemPerBlock
      << ",\n  \"constant_table_bytes\": " << 2 * kDegreeBins * sizeof(float)
      << ",\n  \"cpu_ms\": " << cpuMs << ",\n  \"threshold_strict_greater_than\": " << threshold
      << ",\n  \"detected_lines\": " << lineCount << ",\n  \"all_measured_runs_match_cpu\": true"
      << ",\n  \"mean_global_ms\": " << results[0].averageMilliseconds
      << ",\n  \"mean_constant_ms\": " << results[1].averageMilliseconds
      << ",\n  \"mean_shared_ms\": " << results[2].averageMilliseconds << "\n}\n";
  if (!out) throw std::runtime_error("No se pudo escribir metadata JSON");
}

int execute(const Options &options) {
  const PGMImage image(options.input);
  int device = 0;
  cudaDeviceProp properties{};
  checkCuda(cudaGetDevice(&device), "cudaGetDevice");
  checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  if (properties.sharedMemPerBlock < kDegreeBins * kRBins * sizeof(int))
    throw std::runtime_error("La GPU requiere al menos 36000 bytes de memoria compartida por bloque");
  std::array<float, kDegreeBins> cosTable{}, sinTable{};
  makeTrigTables(cosTable, sinTable);
  const float rMax = static_cast<float>(std::hypot(static_cast<double>(image.width), image.height) / 2);
  const float rScale = 2.0f * rMax / kRBins;
  const auto begin = std::chrono::steady_clock::now();
  const auto cpu = houghCpu(image, cosTable, sinTable, rMax, rScale);
  const auto active = std::count_if(image.pixels.begin(), image.pixels.end(), [](auto value) { return value != 0; });
  if (std::accumulate(cpu.begin(), cpu.end(), 0LL) != active * static_cast<long long>(kDegreeBins))
    throw std::runtime_error("Fallo de conservacion de votos CPU");
  const double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  const std::array<Variant, 3> variants{Variant::Global, Variant::Constant, Variant::Shared};
  std::array<TimedResult, 3> results;
  std::cout << std::fixed << std::setprecision(6) << "GPU: " << properties.name
            << "\nImagen: " << image.width << 'x' << image.height << "\nCPU: " << cpuMs << " ms\n";
  for (std::size_t i = 0; i < variants.size(); ++i) {
    results[i] = houghGpu(image, cosTable, sinTable, rMax, rScale, variants[i], options.runs, cpu);
    std::cout << "Kernel " << variantName(variants[i]) << ": media=" << results[i].averageMilliseconds
              << " ms; " << options.runs << " repeticiones; CPU/GPU: 9000/9000 bins coinciden en cada repeticion.\n";
  }
  const int threshold = options.threshold >= 0 ? options.threshold
      : static_cast<int>(*std::max_element(cpu.begin(), cpu.end()) * 0.55);
  const auto lines = selectLines(results[2].accumulator, kDegreeBins, kRBins, threshold, options.maximumLines);
  ensureParent(options.output);
  writeOverlay(options.output + "_lines.png", image, lines, kDegreeIncrement, rMax, rScale);
  writeLines(options.output + "_lines.csv", lines, kDegreeIncrement, rMax, rScale);
  writeTimings(options.output + "_timings.csv", results);
  writeMetadata(options, image, properties, cpuMs, threshold, lines.size(), results);
  std::cout << "Lineas: " << lines.size() << "; umbral de votos > " << threshold
            << "\nSalidas: " << options.output << "_{lines.png,lines.csv,timings.csv,metadata.json}\n";
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "Uso: " << argv[0] << " imagen.pgm [--runs N] [--threshold VOTOS]"
                 " [--max-lines N] [--output PREFIJO]\n"
                 "Por defecto: 10 mediciones por kernel, 5 warmups, umbral 55% del pico y hasta 12 lineas.\n"
                 "PGM P2/P5 de 8 bits. PNG a color, CSV de tiempos/lineas y JSON de metadata.\n";
    return 0;
  }
  try {
    const auto options = parseOptions(argc, argv);
    const int status = execute(options);  // RAII libera buffers y eventos antes de reiniciar el contexto.
    checkCuda(cudaDeviceReset(), "cudaDeviceReset");
    return status;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    cudaDeviceReset();
    return 1;
  }
}
