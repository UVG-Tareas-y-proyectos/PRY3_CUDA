# Transformada de Hough en CUDA

Proyecto de Computación Paralela y Distribuida para detectar líneas en imágenes
con la Transformada lineal de Hough y comparar implementaciones CUDA.

## Requisitos

- Toolkit de NVIDIA con `nvcc`, una GPU NVIDIA y Make.

## Compilar y ejecutar

```sh
make
make run
```

El archivo de entrada incluido es `data/runway.pgm`. El kernel CUDA asigna un
hilo a cada pixel y usa `blockIdx.x * blockDim.x + threadIdx.x` como ID global.
Al final compara cada bin del acumulador de GPU contra la referencia CPU.
