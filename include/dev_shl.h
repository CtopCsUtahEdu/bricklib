//
// Created by Tuowen Zhao on 11/11/19.
//

#ifndef BRICK_DEV_SHL_H
#define BRICK_DEV_SHL_H

#ifdef CL_SYCL_LANGUAGE_VERSION

// template<typename T>
// inline void dev_shl(cl::sycl::intel::sub_group &SG, T &res, T l, T r, unsigned kn, unsigned cw, unsigned cid) {
#define dev_shl(res, l, r, kn, cw, cid) do { \
    auto l_tmp = SG.shuffle_down(l, cw - (kn)); \
    auto r_tmp = SG.shuffle_up(r, kn); \
    res = (cid) < kn? l_tmp : r_tmp; \
  } while(false)

#elif defined(__OPENCL_VERSION__)

#define TWOSHL

#ifdef TWOSHL

/*
 * These two shuffle implementations are in fact equivalent, however using one shuffle is not stable right now
 * that produces random errors during computation
 */
#define dev_shl(res, l, r, kn, cw, cid) do { \
    float l_tmp = intel_sub_group_shuffle_down(l, l, cw - (kn)); \
    float r_tmp = intel_sub_group_shuffle_up(r, r, kn); \
    res = (cid) < kn? l_tmp : r_tmp; \
  } while(false)

#else

#define dev_shl(res, l, r, kn, cw, cid) do { \
    float l_tmp = (cid) < cw - (kn)? r : l; \
    res = intel_sub_group_shuffle(l_tmp, (sglid & (OCL_SUBGROUP - cw)) ^ ((sglid + cw - (kn)) & (cw - 1))); \
  } while(false)

#endif

#elif defined(__CUDACC__) || defined(__HIP__)

// dev_shl works for both NVidia (CUDA) and AMD (HIP)
template<typename T>
__device__ __forceinline__ void dev_shl(T &res, T l, T r, int kn, int cw, int cid) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 9000)
  // CUDA 9.0+ uses *sync
  T l_tmp = __shfl_down_sync(0xffffffff, l, cw - (kn));
  T r_tmp = __shfl_up_sync(0xffffffff, r, kn);
#else
  // CUDA < 9.0 and HIP works with shfl
  T l_tmp = __shfl_down(l, cw - (kn));
  T r_tmp = __shfl_up(r, kn);
#endif
  res = (cid) < kn? l_tmp : r_tmp;
}
#endif

#endif //BRICK_DEV_SHL_H
