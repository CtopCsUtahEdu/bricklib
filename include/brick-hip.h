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
#define gpuMemcpyKind hipMemcpyKind
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuFree(p) hipFree(p)
#define gpuGetErrorString(e) hipGetErrorString(e)
#define gpuSuccess hipSuccess

#include "brick-gpu.h"

#endif // BRICK_BRICK_HIP_H