#ifndef GPU_ARRAY_H
#define GPU_ARRAY_H

#if !(defined(BRICK_BRICK_CUDA_H) || defined(BRICK_BRICK_HIP_H))
#error "Include either brick-cuda.h or brick-hip.h for generic GPU defines"
#endif

#include <vector>
#include <cstdio>

template<typename T>
void copyToDevice(const std::vector<long> &list, T *&dst, T *src) {
  long size = 1;
  for (auto i: list)
    size *= i;
  size *= sizeof(T);

  if (dst == nullptr)
    gpuMalloc(&dst, size);

  gpuMemcpy(dst, src, size, gpuMemcpyHostToDevice);
}

template<typename T>
void copyFromDevice(const std::vector<long> &list, T *dst, T *src) {
  long size = 1;
  for (auto i: list)
    size *= i;
  size *= sizeof(T);
  gpuMemcpy(dst, src, size, gpuMemcpyDeviceToHost);
}

#endif

