/**
 * @file
 * @brief Header necessary for OpenCL program/kernel to include
 */

#ifndef BRICK_BRICK_OPENCL_H
#define BRICK_BRICK_OPENCL_H

#include "vecscatter.h"
#include "dev_shl.h"

// Only support OpenCL on Intel graphics which only support single precision

#define OCL_VSVEC "OPENCL"
#define OCL_SUBGROUP 16
#define OCL_VFOLD 2,8
// C compatible and dumbed down version of all data structures for OpenCL+Codegen to use

#ifdef __OPENCL_VERSION__
// Device only code

#define reqsg32 __attribute__((intel_reqd_sub_group_size(16)))

struct oclbrick {
  __global bElem *dat;
  __global const unsigned *adj;
  unsigned step;
};

#else

struct syclbrick {
  bElem *dat;
  unsigned *adj;
  unsigned step;
};

#endif

#endif //BRICK_BRICK_OPENCL_H
