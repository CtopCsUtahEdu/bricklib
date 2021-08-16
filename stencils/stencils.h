//
// Created by Tuowen Zhao on 12/4/18.
//

#ifndef BRICK_STENCILS_H
#define BRICK_STENCILS_H

#include <omp.h>
#include "brick.h"

#define N 256
#define TILE 8

#define GZ TILE
#define PADDING 8

// Stride for arrays is GHOSTZONE + PADDING on each side
#define STRIDE (N + 2 * (GZ + PADDING))
#define STRIDEG (N + 2 * GZ)

#define NB (N / TILE)
#define GB (GZ / TILE)

// Stride for bricks is GHOSTZONE on each side
#define STRIDEB ((N + 2 * GZ) / TILE)

#define BDIM TILE,TILE,TILE
#define TOT_TIME 5

#define _PARFOR _Pragma("omp parallel for collapse(2)")
#define _TILEFOR _Pragma("omp parallel for collapse(2)") \
for (long tk = PADDING; tk < PADDING + STRIDEG; tk += TILE) \
for (long tj = PADDING; tj < PADDING + STRIDEG; tj += TILE) \
for (long ti = PADDING; ti < PADDING + STRIDEG; ti += TILE) \
for (long k = tk; k < tk + TILE; ++k) \
for (long j = tj; j < tj + TILE; ++j) \
_Pragma("omp simd") \
for (long i = ti; i < ti + TILE; ++i)

template<typename T>
double time_func(T func) {
  int it = 1;
  func(); // Warm up
  double st = omp_get_wtime();
  double ed = st;
  while (ed < st + TOT_TIME) {
    for (int i = 0; i < it; ++i)
      func();
    it <<= 1;
    ed = omp_get_wtime();
  }
  return (ed - st) / (it - 1);
}

void copy();

void d3pt7();

void d3cond();

void d3pt13();

void d3pt19();

void d3pt25();

void d3pt27();

void d3pt125();

void d3iso();

void d3cns();

extern bElem* coeff;


#endif //BRICK_STENCILS_H
