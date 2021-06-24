/**
 * @file
 * @brief For using bricks with CUDA
 */

#ifndef BRICK_BRICK_CUDA_H
#define BRICK_BRICK_CUDA_H

#include <cassert>
#include <brick.h>
#include <cuda_runtime.h>

/**
 * @brief Check the return of CUDA calls, do nothing during release build
 */
#ifndef NDEBUG
#define cudaCheck(x) x
#else

#include <cstdio>

#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)
#endif


/// Internal for #cudaCheck(x)
template<typename T>
void _cudaCheck(T e, const char *func, const char *call, const int line) {
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int) e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Moving BrickInfo to or from GPU (allocate new)
 * @tparam dims implicit when used with bInfo argument
 * @param bInfo BrickInfo to copy from host or GPU
 * @param kind Currently must be cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost
 * @return a new BrickInfo struct allocated on the destination
 */
template<unsigned dims>
BrickInfo<dims> movBrickInfo(BrickInfo<dims> &bInfo, cudaMemcpyKind kind) {
  assert(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost);

  // Make a copy
  BrickInfo<dims> ret = bInfo;
  size_t size = bInfo.nbricks * static_power<3, dims>::value * sizeof(unsigned);

  if (kind == cudaMemcpyHostToDevice) {
    cudaCheck(cudaMalloc(&ret.adj, size));
  } else {
    ret.adj = (unsigned (*)[static_power<3, dims>::value]) malloc(size);
  }
  cudaCheck(cudaMemcpy(ret.adj, bInfo.adj, size, kind));
  return ret;
}

/**
 * @brief Moving BrickStorage to or from GPU (allocate new)
 * @param bStorage BrickStorage to copy from
 * @param kind Currently must be either cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost
 * @return a new BrickStorage struct allocated on the destination
 */
inline BrickStorage movBrickStorage(BrickStorage &bStorage, cudaMemcpyKind kind) {
  assert(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost);

  bool isToDevice = (kind == cudaMemcpyHostToDevice);
  // Make a copy
  BrickStorage ret = bStorage;
  size_t size = bStorage.step * bStorage.chunks * sizeof(bElem);
  bElem *datptr;
  if (isToDevice) {
    cudaCheck(cudaMalloc(&datptr, size));
  } else {
    datptr = (bElem *) malloc(size);
  }
  cudaCheck(cudaMemcpy(datptr, bStorage.dat.get(), size, kind));
  if (isToDevice) {
    ret.dat = std::shared_ptr<bElem>(datptr, [](bElem *p) { cudaFree(p); });
  } else {
    ret.dat = std::shared_ptr<bElem>(datptr, [](bElem *p) { free(p); });
  }
  return ret;
}

#include "dev_shl.h"

#endif //BRICK_BRICK_CUDA_H
