#include "output.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {
constexpr double kPi = 3.14159265358979323846;
using Bytes = std::vector<unsigned char>;

void append32(Bytes &out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    out.push_back(static_cast<unsigned char>(value >> shift));
}

std::uint32_t crc32(const unsigned char *data, std::size_t size) {
  std::uint32_t crc = 0xffffffffu;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

void chunk(Bytes &png, const char *type, const Bytes &data) {
  append32(png, static_cast<std::uint32_t>(data.size()));
  const auto begin = png.size();
  png.insert(png.end(), type, type + 4);
  png.insert(png.end(), data.begin(), data.end());
  append32(png, crc32(png.data() + begin, data.size() + 4));
}

void writePng(const std::string &path, int width, int height, const Bytes &rgb) {
  // PNG: cada fila usa filtro 0. El flujo zlib usa bloques DEFLATE almacenados.
  Bytes raw;
  raw.reserve(static_cast<std::size_t>(height) * (3ULL * width + 1));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    const auto begin = rgb.begin() + static_cast<std::size_t>(y) * width * 3;
    raw.insert(raw.end(), begin, begin + static_cast<std::size_t>(width) * 3);
  }
  Bytes deflate{0x78, 0x01};
  for (std::size_t offset = 0; offset < raw.size();) {
    const auto count = static_cast<unsigned>(std::min<std::size_t>(65535, raw.size() - offset));
    deflate.push_back(offset + count == raw.size() ? 1 : 0);
    deflate.push_back(static_cast<unsigned char>(count));
    deflate.push_back(static_cast<unsigned char>(count >> 8));
    deflate.push_back(static_cast<unsigned char>(~count));
    deflate.push_back(static_cast<unsigned char>(~count >> 8));
    deflate.insert(deflate.end(), raw.begin() + offset, raw.begin() + offset + count);
    offset += count;
  }
  std::uint32_t a = 1, b = 0;
  for (unsigned char value : raw) {
    a = (a + value) % 65521;
    b = (b + a) % 65521;
  }
  append32(deflate, (b << 16) | a);
  Bytes png{137, 80, 78, 71, 13, 10, 26, 10};
  Bytes ihdr;
  append32(ihdr, width);
  append32(ihdr, height);
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
  chunk(png, "IHDR", ihdr);
  chunk(png, "IDAT", deflate);
  chunk(png, "IEND", {});
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(png.data()), static_cast<std::streamsize>(png.size()));
  if (!output) throw std::runtime_error("No se pudo escribir PNG: " + path);
}
}  // namespace

std::vector<DetectedLine> selectLines(const std::vector<int> &accumulator,
                                     int degreeBins, int radiusBins,
                                     int threshold, int maximumLines) {
  std::vector<DetectedLine> candidates;
  for (int r = 0; r < radiusBins; ++r)
    for (int t = 0; t < degreeBins; ++t)
      if (accumulator[r * degreeBins + t] > threshold)
        candidates.push_back({r, t, accumulator[r * degreeBins + t]});
  std::stable_sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
    return a.votes > b.votes;
  });
  std::vector<DetectedLine> selected;
  for (const auto &candidate : candidates) {
    bool near = false;
    for (const auto &line : selected) {
      const int dt = std::abs(candidate.thetaBin - line.thetaBin);
      // Al cruzar 0/180 grados tambien se invierte el signo de rho.
      near |= (dt <= 4 && std::abs(candidate.radiusBin - line.radiusBin) <= 2)
           || (degreeBins - dt <= 4
               && std::abs(candidate.radiusBin + line.radiusBin - (radiusBins - 1)) <= 2);
    }
    if (!near) selected.push_back(candidate);
    if (static_cast<int>(selected.size()) >= maximumLines) break;
  }
  return selected;
}

void writeOverlay(const std::string &path, const PGMImage &image,
                  const std::vector<DetectedLine> &lines,
                  int degreeIncrement, float rMax, float rScale) {
  Bytes rgb(image.pixels.size() * 3);
  for (std::size_t i = 0; i < image.pixels.size(); ++i)
    rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = image.pixels[i];
  constexpr std::array<std::array<unsigned char, 3>, 6> colors{{
      {{255, 64, 70}}, {{0, 210, 255}}, {{255, 210, 0}},
      {{70, 240, 130}}, {{235, 100, 255}}, {{255, 150, 50}}}};
  for (std::size_t i = 0; i < lines.size(); ++i) {
    const double theta = lines[i].thetaBin * degreeIncrement * kPi / 180;
    const double c = std::cos(theta), s = std::sin(theta);
    const double rho = (lines[i].radiusBin + 0.5) * rScale - rMax;
    // Distancia perpendicular a la recta central de cada bin. No divide por
    // sen(theta) ni cos(theta), por lo que admite verticales y horizontales.
    for (int y = 0; y < image.height; ++y)
      for (int x = 0; x < image.width; ++x) {
        const double distance = (x - image.width / 2) * c
                              + (image.height / 2 - y) * s - rho;
        if (std::abs(distance) <= 0.8) {
          const auto index = (static_cast<std::size_t>(y) * image.width + x) * 3;
          for (int channel = 0; channel < 3; ++channel)
            rgb[index + channel] = colors[i % colors.size()][channel];
        }
      }
  }
  writePng(path, image.width, image.height, rgb);
}

void writeLines(const std::string &path, const std::vector<DetectedLine> &lines,
                int degreeIncrement, float rMax, float rScale) {
  std::ofstream out(path);
  out << "rho_bin,theta_bin,rho_pixels,theta_degrees,votes\n" << std::fixed << std::setprecision(6);
  for (const auto &line : lines)
    out << line.radiusBin << ',' << line.thetaBin << ','
        << (line.radiusBin + 0.5f) * rScale - rMax << ','
        << line.thetaBin * degreeIncrement << ',' << line.votes << '\n';
  if (!out) throw std::runtime_error("No se pudo escribir CSV de lineas: " + path);
}
