/**
 * @file brick-hip.h
 * @brief For using bricklib with HIP. Directs the functions used in brick-gpu.h to corresponding
 * HIP functions
 */

#ifndef BRICK_BRICK_HIP_H
#define BRICK_BRICK_HIP_H

#include <hip/hip_runtime.h>

#define gpuMalloc(p, s) hipMalloc(p, s)
#define gpuMemcpy(d, p, s, k) hipMemcpy(d, p, s, k)
#define gpuFree(p) hipFree(p)
#define gpuGetErrorString(e) hipGetErrorString(e)
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#define gpuMemcpyToSymbol(p, d, s) hipMemcpyToSymbol(p, d, s, hipMemcpyHostToDevice)
#define gpuDeviceSetCacheConfig(c) hipDeviceSetCacheConfig(c)

#define gpuFuncCachePreferL1 hipFuncCachePreferL1
#define gpuSuccess hipSuccess
#define gpuMemcpyKind hipMemcpyKind
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost

#define gpuExecKernel(f, b, t, a...) hipLaunchKernelGGL(f, b, t, 0, 0, a)

#define blockIdx_x hipBlockIdx_x
#define blockIdx_y hipBlockIdx_y
#define blockIdx_z hipBlockIdx_z

#define threadIdx_x hipThreadIdx_x
#define threadIdx_y hipThreadIdx_y
#define threadIdx_z hipThreadIdx_z

#include "brick-gpu.h"

#endif // BRICK_BRICK_HIP_H