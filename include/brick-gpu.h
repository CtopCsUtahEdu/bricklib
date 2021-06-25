/**
 * @file
 * @brief For using bricks with GPUs.
 */

#ifndef BRICK_GPU_FUNCS_H
#define BRICK_GPU_FUNCS_H

#include <cassert>
#include <brick.h>

#if defined(__CUDACC__)

#include <cuda_runtime.h>

#define gpuMalloc(p, s) cudaMalloc(p, s)
#define gpuMemcpy(d, p, s, k) cudaMemcpy(d, p, s, k)
#define gpuMemcpyKind cudaMemcpyKind
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuFree(p) cudaFree(p)
#define gpuGetErrorString(e) cudaGetErrorString(e)

#elif defined(__HIP__)

#include <hip/hip_runtime.h>

#define gpuMalloc(p, s) hipMalloc(p, s)
#define gpuMemcpy(d, p, s, k) hipMemcpy(d, p, s, k)
#define gpuMemcpyKind hipMemcpyKind
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuFree(p) hipFree(p)
#define gpuGetErrorString(e) hipGetErrorString(e)

#endif // __CUDACC__ and __HIP__ 

#ifndef NDEBUG
    #define gpuCheck(x) x
    #else

    #include <cstdio>
    #define gpuCheck(x) _gpuCheck(x, #x, __FILE__, __LINE__)
#endif

template<typename T>
void _gpuCheck(T e, const char *func, const char *call, const int line) {
    if (e != hipSuccess) {
        printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int) e, hipGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Moving BrickInfo to or from GPU 
 * @tparam dims implicit when used with bInfo argument
 * @param bInfo BrickInfo to copy to destination
 * @param kind Currently must be <hip|cuda>MempyHostToDevice or <hip|cuda>MemcpyDeviceToHost
 * @return a new BrickInfo struct allocated on the destination
 */
template<unsigned dims>
BrickInfo<dims> movBrickInfo(BrickInfo<dims> &bInfo, gpuMemcpyKind kind) {
    assert(kind == gpuMemcpyHostToDevice || kind == gpuMemcpyDeviceToHost);

    BrickInfo<dims> ret = bInfo;
    unsigned size = bInfo.nbricks * static_power<3, dims>::value * sizeof(unsigned);
    if (kind == gpuMemcpyHostToDevice) {
        gpuCheck(gpuMalloc(&ret.adj, size));
    } else {
        ret.adj = (unsigned (*)[(static_power<3, dims>::value)]) malloc(size);
    }
    gpuCheck(gpuMemcpy(ret.adj, bInfo.adj, size, kind));
    return ret;
}

/**
 * @brief Moving the full BrickInfo to or from GPU, including the adjacency list and other elements 
 * @tparam dims implicit when used with bInfo argument
 * @param bInfo BrickInfo to copy to destination
 * @param kind Currently must be <hip|cuda>MempyHostToDevice or <hip|cuda>MemcpyDeviceToHost
 * @return a new pointer to a BrickInfo allocated on the destination
 */
template<unsigned dims>
BrickInfo<dims> *movBrickInfoDeep(BrickInfo<dims> &bInfo, gpuMemcpyKind kind) {
    BrickInfo<dims> *ret;
    BrickInfo<dims> temp = movBrickInfo(bInfo, kind);
    unsigned size = sizeof(BrickInfo<dims>);
    if (kind == gpuMemcpyHostToDevice) {
        gpuMalloc(&ret, size);
    } else {
        ret = (BrickInfo<dims> *) malloc(size);
    }
    gpuMemcpy(ret, &temp, size, kind);
    return ret;
}

/**
 * @brief Moving BrickStorage to or from GPU (allocate new)
 * @param bStorage BrickStorage to copy from
 * @param kind Currently must be either <hip|cuda>MemcpyHostToDevice or <hip|cuda>MemcpyDeviceToHost
 * @return a new BrickStorage struct allocated on the destination
 */
inline BrickStorage movBrickStorage(BrickStorage &bStorage, gpuMemcpyKind kind) {
    assert(kind == gpuMemcpyHostToDevice || kind == gpuMemcpyDeviceToHost);

    bool isToDevice = (kind == gpuMemcpyHostToDevice);
    BrickStorage ret = bStorage;
    unsigned size = bStorage.step * bStorage.chunks * sizeof(bElem);
    bElem *datptr;
    if (isToDevice) {
        gpuCheck(gpuMalloc(&datptr, size));
    } else {
        datptr = (bElem *) malloc(size);
    }
    gpuCheck(gpuMemcpy(datptr, bStorage.dat.get(), size, kind));
    if (isToDevice) {
        ret.dat = std::shared_ptr<bElem>(datptr, [](bElem *p) { gpuFree(p); });
    } else {
        ret.dat = std::shared_ptr<bElem>(datptr, [](bElem *p) { free(p); });
    }
    return ret;
}

#include "dev_shl.h"

#endif // BRICK_GPU_FUNCS_H
