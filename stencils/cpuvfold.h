//
// Created by ztuowen on 6/15/19.
//

#ifndef BRICK_CPUVFOLD_H
#define BRICK_CPUVFOLD_H

#ifdef __AVX__

// Setting for X86 with at least AVX support
#include <immintrin.h>
#define VSVEC "AVX2"
#define VFOLD 2,2

#else

#define VSVEC "Scalar"
#define VFOLD 1

#endif

#endif //BRICK_CPUVFOLD_H
