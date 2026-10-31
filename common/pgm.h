#ifndef HOUGH_PGM_H
#define HOUGH_PGM_H

#include <string>
#include <vector>

struct PGMImage {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> pixels;

  explicit PGMImage(const std::string &path);
};

#endif
