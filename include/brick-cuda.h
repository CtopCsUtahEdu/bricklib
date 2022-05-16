/**
 * @file brick-cuda.h
 * @brief For using bricklib with CUDA. Directs the functions used in brick-gpu.h to corresponding
 * CUDA functions
 */

#ifndef BRICK_BRICK_CUDA_H
#define BRICK_BRICK_CUDA_H

#include <cuda_runtime.h>

#define gpuMalloc(p, s) cudaMalloc(p, s)
#define gpuMemcpy(d, p, s, k) cudaMemcpy(d, p, s, k)
#define gpuMemcpyKind cudaMemcpyKind
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuFree(p) cudaFree(p)
#define gpuGetErrorString(e) cudaGetErrorString(e)
#define gpuSuccess cudaSuccess
#define gpuDeviceSynchronize() cudaDeviceSynchronize()
#define gpuMemcpyToSymbol(p, d, s) cudaMemcpyToSymbol(p, d, s)
#define gpuDeviceSetCacheConfig(c) cudaDeviceSetCacheConfig(c)
#define gpuFuncCachePreferL1 cudaFuncCachePreferL1
#define gpuExecKernel(f, b, t, a...) f<<<b,t>>>(a)

#define blockIdx_x cudaBlockIdx.x
#define blockIdx_y cudaBlockIdx.y
#define blockIdx_z cudaBlockIdx.z

#define threadIdx_x cudaThreadIdx.x
#define threadIdx_y cudaThreadIdx.y
#define threadIdx_z cudaThreadIdx.z

#include "brick-gpu.h"

#endif // BRICK_BRICK_CUDA_H
