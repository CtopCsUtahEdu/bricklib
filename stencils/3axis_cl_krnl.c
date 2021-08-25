#include <brick-opencl.h>

#define VSVEC "OPENCL"
#define VFOLD 4, 8

__kernel void stencil(__global bElem *bDat, __global const bElem *coeff,
                      __global const unsigned *adj, __global const unsigned *bIdx, unsigned len)
    __attribute__((intel_reqd_sub_group_size(OCL_SUBGROUP))) {
  // Get the index of the current element to be processed
  int sglid = get_sub_group_local_id();
  int gps = get_num_groups(0);

  struct oclbrick bIn = {bDat, adj, 1024};
  struct oclbrick bOut = {bDat + 512, adj, 1024};

  // Do the operation
  for (int i = get_group_id(0); i < len; i += get_num_groups(0)) {
    unsigned b = bIdx[i];
    brick("7pt.py", OCL_VSVEC, (8, 8, 8), (OCL_VFOLD), b);
  }
}