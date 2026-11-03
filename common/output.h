#ifndef HOUGH_OUTPUT_H
#define HOUGH_OUTPUT_H

#include "pgm.h"
#include <string>
#include <vector>

struct DetectedLine {
  int radiusBin;
  int thetaBin;
  int votes;
};

// Seleccion descendente con supresion de vecinos para evitar lineas repetidas.
std::vector<DetectedLine> selectLines(const std::vector<int> &accumulator,
                                     int degreeBins, int radiusBins,
                                     int threshold, int maximumLines);
// PNG RGB de 8 bits, sin bibliotecas externas. CSV conserva los parametros.
void writeOverlay(const std::string &path, const PGMImage &image,
                  const std::vector<DetectedLine> &lines,
                  int degreeIncrement, float rMax, float rScale);
void writeLines(const std::string &path, const std::vector<DetectedLine> &lines,
                int degreeIncrement, float rMax, float rScale);

#endif
