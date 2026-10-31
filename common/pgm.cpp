#include "pgm.h"

#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
std::string readToken(std::istream &input) {
  while (true) {
    input >> std::ws;
    if (input.peek() != '#') break;
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  std::string token;
  if (!(input >> token)) throw std::runtime_error("Cabecera PGM incompleta");
  return token;
}

int parsePositive(const std::string &token, const char *field) {
  std::size_t parsed = 0;
  const long value = std::stol(token, &parsed);
  if (parsed != token.size() || value <= 0 || value > std::numeric_limits<int>::max())
    throw std::runtime_error(std::string("Valor invalido para ") + field);
  return static_cast<int>(value);
}

int parsePixel(const std::string &token) {
  std::size_t parsed = 0;
  const long value = std::stol(token, &parsed);
  if (parsed != token.size() || value < 0 || value > 255)
    throw std::runtime_error("Valor de pixel PGM invalido");
  return static_cast<int>(value);
}
}  // namespace

PGMImage::PGMImage(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("No se pudo abrir la imagen: " + path);

  const std::string magic = readToken(input);
  if (magic != "P2" && magic != "P5")
    throw std::runtime_error("Solo se admiten imagenes PGM P2 y P5");

  width = parsePositive(readToken(input), "ancho");
  height = parsePositive(readToken(input), "alto");
  const int maxValue = parsePositive(readToken(input), "maximo gris");
  if (maxValue > 255) throw std::runtime_error("El maximo gris debe ser <= 255");

  const std::size_t count = static_cast<std::size_t>(width) * height;
  if (count / static_cast<std::size_t>(width) != static_cast<std::size_t>(height))
    throw std::runtime_error("Dimensiones PGM demasiado grandes");
  pixels.resize(count);

  if (magic == "P5") {
    char separator = 0;
    if (!input.get(separator) || !std::isspace(static_cast<unsigned char>(separator)))
      throw std::runtime_error("Falta separador antes de los pixeles PGM");
    if (separator == '\r' && input.peek() == '\n') input.get();
    input.read(reinterpret_cast<char *>(pixels.data()),
               static_cast<std::streamsize>(pixels.size()));
    if (input.gcount() != static_cast<std::streamsize>(pixels.size()))
      throw std::runtime_error("Datos PGM incompletos");
    if (maxValue != 255) {
      for (unsigned char &pixel : pixels)
        pixel = static_cast<unsigned char>((static_cast<int>(pixel) * 255) / maxValue);
    }
  } else {
    for (unsigned char &pixel : pixels) {
      const int value = parsePixel(readToken(input));
      if (value > maxValue)
        throw std::runtime_error("Pixel PGM fuera del rango declarado");
      pixel = static_cast<unsigned char>((value * 255) / maxValue);
    }
  }
}
