//
// Created by ztuowen on 6/15/19.
//

#ifndef BRICK_CPUVFOLD_H
#define BRICK_CPUVFOLD_H

#if defined(__AVX512F__) || defined(__AVX512__)

// Setting for X86 with at least AVX512 support
#include <immintrin.h>
#define VSVEC "AVX512"
#define VFOLD 8

#elif defined(__AVX2__)

// Setting for X86 with at least AVX2 support
#include <immintrin.h>
#define VSVEC "AVX2"
#define VFOLD 2,2

#elif defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>
#define VSVEC "SVE512"
#if __ARM_FEATURE_SVE_BITS != 512
#error "This is for SVE512"
#endif
#define VFOLD 8

#elif defined(__ARM_NEON)

#include<arm_neon.h>
#define VSVEC "ASIMD"
#define VFOLD 2

#else

#define VSVEC "Scalar"
#define VFOLD 1

#endif

#endif //BRICK_CPUVFOLD_H
