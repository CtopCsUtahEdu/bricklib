/**
 * @file brick-gpu.h
 * @brief This file should not be directly included. It defines instructions for using bricklib with a GPU, but either brick-hip.h or brick-cuda.h should be included for correct runtime support
 */

#ifndef BRICK_GPU_FUNCS_H
#define BRICK_GPU_FUNCS_H

#include <cassert>
#include <brick.h>

#ifndef NDEBUG

#define gpuCheck(x) x

#else // defined(NDEBUG)

#include <cstdio>
#define gpuCheck(x) _gpuCheck(x, #x, __FILE__, __LINE__)

#endif // NDEBUG

template<typename T>
void _gpuCheck(T e, const char *func, const char *call, const int line) {
    if (e != gpuSuccess) {
        printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int) e, gpuGetErrorString(e));
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
