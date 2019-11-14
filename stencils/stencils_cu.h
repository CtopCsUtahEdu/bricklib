//
// Created by Tuowen Zhao on 12/5/18.
//

#ifndef BRICK_STENCILS_CU_H
#define BRICK_STENCILS_CU_H

#include <brick-cuda.h>
#include "stencils.h"

#define CU_ITER 100

template<typename T>
double cutime_func(T func) {
  func(); // Warm up
  cudaEvent_t start, stop;
  float elapsed;
  cudaDeviceSynchronize();
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < CU_ITER; ++i)
    func();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  return elapsed / CU_ITER / 1000;
}

void d3pt7cu();

void d3condcu();

#endif //BRICK_STENCILS_CU_H
