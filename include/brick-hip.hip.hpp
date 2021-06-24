#ifndef BRICK_BRICK_HIP_H
#define BRICK_BRICK_HIP_H

#include <cassert>
#include <brick.h>
#include <hip/hip_runtime.h>

#ifndef NDEBUG
#define hipCheck(x) x
#else

#include <cstdio>
#define hipCheck(x) _hipCheck(x, #x, __FILE__, __LINE__)
#endif

template<typename T>
void _hipCheck(T e, const char *func, const char *call, const int line) {
    if (e != hipSuccess) {
        printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int) e, hipGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Moving BrickInfo to or from GPU 
 * @tparam dims implicit when used with bInfo argument
 * @param bInfo BrickInfo to copy to destination
 * @param kind Currently must be hipMempyHostToDevice or hipMemcpyDeviceToHost
 * @return a new BrickInfo struct allocated on the destination
 */
template<unsigned dims>
BrickInfo<dims> movBrickInfo(BrickInfo<dims> &bInfo, hipMemcpyKind kind) {
    assert(kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToHost);

    BrickInfo<dims> ret = bInfo;
    unsigned size = bInfo.nbricks * static_power<3, dims>::value * sizeof(unsigned);
    if (kind == hipMemcpyHostToDevice) {
        hipCheck(hipMalloc(&ret.adj, size));
    } else {
        ret.adj = (unsigned (*)[(static_power<3, dims>::value)]) malloc(size);
    }
    hipCheck(hipMemcpy(ret.adj, bInfo.adj, size, kind));
    return ret;
}

/**
 * @brief Moving the full BrickInfo to or from GPU, including the adjacency list and other elements 
 * @tparam dims implicit when used with bInfo argument
 * @param bInfo BrickInfo to copy to destination
 * @param kind Currently must be hipMempyHostToDevice or hipMemcpyDeviceToHost
 * @return a new pointer to a BrickInfo allocated on the destination
 */
template<unsigned dims>
BrickInfo<dims> *movBrickInfoDeep(BrickInfo<dims> &bInfo, hipMemcpyKind kind) {
    BrickInfo<dims> *ret;
    BrickInfo<dims> temp = movBrickInfo(bInfo, kind);
    unsigned size = sizeof(BrickInfo<dims>);
    if (kind == hipMemcpyHostToDevice) {
        hipMalloc(&ret, size);
    } else {
        ret = (BrickInfo<dims> *) malloc(size);
    }
    hipMemcpy(ret, &temp, size, kind);
    return ret;
}

/**
 * @brief Moving BrickStorage to or from GPU (allocate new)
 * @param bStorage BrickStorage to copy from
 * @param kind Currently must be either hipMemcpyHostToDevice or hipMemcpyDeviceToHost
 * @return a new BrickStorage struct allocated on the destination
 */
inline BrickStorage movBrickStorage(BrickStorage &bStorage, hipMemcpyKind kind) {
    assert(kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToHost);

    bool isToDevice = (kind == hipMemcpyHostToDevice);
    BrickStorage ret = bStorage;
    unsigned size = bStorage.step * bStorage.chunks * sizeof(bElem);
    bElem *datptr;
    if (isToDevice) {
        hipCheck(hipMalloc(&datptr, size));
    } else {
        datptr = (bElem *) malloc(size);
    }
    hipCheck(hipMemcpy(datptr, bStorage.dat.get(), size, kind));
    if (isToDevice) {
        ret.dat = std::shared_ptr<bElem>(datptr, [](bElem *p) { hipFree(p); });
    } else {
        ret.dat = std::shared_ptr<bElem>(datptr, [](bElem *p) { free(p); });
    }
    return ret;
}

#include "dev_shl.h"

#endif
