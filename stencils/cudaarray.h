//
// Created by Tuowen Zhao on 6/3/19.
//

#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include <vector>
#include <cstdio>

template<typename T>
void copyToDevice(const std::vector<long> &list, T *&dst, T *src) {
  long size = 1;
  for (auto i: list)
    size *= i;
  size *= sizeof(T);

  if (dst == nullptr)
    cudaMalloc(&dst, size);

  cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

template<typename T>
void copyFromDevice(const std::vector<long> &list, T *dst, T *src) {
  long size = 1;
  for (auto i: list)
    size *= i;
  size *= sizeof(T);
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

#endif

