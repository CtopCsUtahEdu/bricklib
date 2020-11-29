//
// Created by ztuowen on 2/17/19.
//

#ifndef BRICK_FAKE_H
#define BRICK_FAKE_H

#include "brick.h"
#include "brick-mpi.h"

#define MPI_BETA 0.1
#define MPI_ALPHA 0.4

#define MPI_A0 0.1
#define MPI_A1 0.06
#define MPI_A2 0.045
#define MPI_A3 0.03
#define MPI_A4 0.015

#define MPI_B0 0.4
#define MPI_B1 0.07
#define MPI_B2 0.03

#define MPI_C0 0.1
#define MPI_C1 0.04
#define MPI_C2 0.03
#define MPI_C3 0.01
#define MPI_C4 0.006
#define MPI_C5 0.004
#define MPI_C6 0.005
#define MPI_C7 0.002
#define MPI_C8 0.003
#define MPI_C9 0.001

#ifndef MPI_STENCIL
#define MPI_7PT
#endif

#ifdef MPI_49PT
#define ST_SCRTPT "../stencils/mpi49pt.py"
#define ST_ITER 2
#define ST_CPU nullptr
#elif defined(MPI_25PT)
#define ST_SCRTPT "../stencils/mpi25pt.py"
#define ST_ITER 2
#define ST_CPU arrOut[k][j][i] = (arrIn[k + 4][j][i] + arrIn[k - 4][j][i] + \
                                  arrIn[k][j + 4][i] + arrIn[k][j - 4][i] + \
                                  arrIn[k][j][i + 4] + arrIn[k][j][i - 4]) * MPI_A4 + \
                                 (arrIn[k + 3][j][i] + arrIn[k - 3][j][i] + \
                                  arrIn[k][j + 3][i] + arrIn[k][j - 3][i] + \
                                  arrIn[k][j][i + 3] + arrIn[k][j][i - 3]) * MPI_A3 + \
                                 (arrIn[k + 2][j][i] + arrIn[k - 2][j][i] + \
                                  arrIn[k][j + 2][i] + arrIn[k][j - 2][i] + \
                                  arrIn[k][j][i + 2] + arrIn[k][j][i - 2]) * MPI_A2 + \
                                 (arrIn[k + 1][j][i] + arrIn[k - 1][j][i] + \
                                  arrIn[k][j + 1][i] + arrIn[k][j - 1][i] + \
                                  arrIn[k][j][i + 1] + arrIn[k][j][i - 1]) * MPI_A1 + \
                                 arrIn[k][j][i] * MPI_A0
#define ST_GPU out_ptr[pos] = in_ptr[pos] * MPI_A0 + \
                              (in_ptr[pos + stride[2]] + in_ptr[pos - stride[2]] + \
                               in_ptr[pos + stride[1]] + in_ptr[pos - stride[1]] + \
                               in_ptr[pos + 1] + in_ptr[pos - 1]) * MPI_A1 + \
                              (in_ptr[pos + 2 * stride[2]] + in_ptr[pos - 2 * stride[2]] + \
                               in_ptr[pos + 2 * stride[1]] + in_ptr[pos - 2 * stride[1]] + \
                               in_ptr[pos + 2] + in_ptr[pos - 2]) * MPI_A2 + \
                              (in_ptr[pos + 3 * stride[2]] + in_ptr[pos - 3 * stride[2]] + \
                               in_ptr[pos + 3 * stride[1]] + in_ptr[pos - 3 * stride[1]] + \
                               in_ptr[pos + 3] + in_ptr[pos - 3]) * MPI_A3 + \
                              (in_ptr[pos + 4 * stride[2]] + in_ptr[pos - 4 * stride[2]] + \
                               in_ptr[pos + 4 * stride[1]] + in_ptr[pos - 4 * stride[1]] + \
                               in_ptr[pos + 4] + in_ptr[pos - 4]) * MPI_A4
#elif defined(MPI_13PT)
#define ST_SCRTPT "../stencils/mpi13pt.py"
#define ST_ITER 4
#define ST_CPU arrOut[k][j][i] = (arrIn[k + 2][j][i] + arrIn[k - 2][j][i] + \
                                  arrIn[k][j + 2][i] + arrIn[k][j - 2][i] + \
                                  arrIn[k][j][i + 2] + arrIn[k][j][i - 2]) * MPI_B2 + \
                                 (arrIn[k + 1][j][i] + arrIn[k - 1][j][i] + \
                                  arrIn[k][j + 1][i] + arrIn[k][j - 1][i] + \
                                  arrIn[k][j][i + 1] + arrIn[k][j][i - 1]) * MPI_B1 + \
                                 arrIn[k][j][i] * MPI_B0
#define ST_GPU out_ptr[pos] = in_ptr[pos] * MPI_B0 + \
                              (in_ptr[pos + stride[2]] + in_ptr[pos - stride[2]] + \
                               in_ptr[pos + stride[1]] + in_ptr[pos - stride[1]] + \
                               in_ptr[pos + 1] + in_ptr[pos - 1]) * MPI_B1 + \
                              (in_ptr[pos + 2 * stride[2]] + in_ptr[pos - 2 * stride[2]] + \
                               in_ptr[pos + 2 * stride[1]] + in_ptr[pos - 2 * stride[1]] + \
                               in_ptr[pos + 2] + in_ptr[pos - 2]) * MPI_B2
#elif defined(MPI_125PT)
#define ST_SCRTPT "../stencils/mpi125pt.py"
#define ST_ITER 4
#define ST_CPU arrOut[k][j][i] = ( \
       MPI_C0 * arrIn[k][j][i] + \
       MPI_C1 * (arrIn[k + 1][j][i] + \
                 arrIn[k - 1][j][i] + \
                 arrIn[k][j + 1][i] + \
                 arrIn[k][j - 1][i] + \
                 arrIn[k][j][i + 1] + \
                 arrIn[k][j][i - 1]) + \
       MPI_C2 * (arrIn[k + 2][j][i] + \
                 arrIn[k - 2][j][i] + \
                 arrIn[k][j + 2][i] + \
                 arrIn[k][j - 2][i] + \
                 arrIn[k][j][i + 2] + \
                 arrIn[k][j][i - 2]) + \
       MPI_C3 * (arrIn[k + 1][j + 1][i] + \
                 arrIn[k - 1][j + 1][i] + \
                 arrIn[k + 1][j - 1][i] + \
                 arrIn[k - 1][j - 1][i] + \
                 arrIn[k + 1][j][i + 1] + \
                 arrIn[k - 1][j][i + 1] + \
                 arrIn[k + 1][j][i - 1] + \
                 arrIn[k - 1][j][i - 1] + \
                 arrIn[k][j + 1][i + 1] + \
                 arrIn[k][j - 1][i + 1] + \
                 arrIn[k][j + 1][i - 1] + \
                 arrIn[k][j - 1][i - 1]) + \
       MPI_C4 * (arrIn[k + 1][j + 2][i] + \
                 arrIn[k - 1][j + 2][i] + \
                 arrIn[k + 1][j - 2][i] + \
                 arrIn[k - 1][j - 2][i] + \
                 arrIn[k + 1][j][i + 2] + \
                 arrIn[k - 1][j][i + 2] + \
                 arrIn[k + 1][j][i - 2] + \
                 arrIn[k - 1][j][i - 2] + \
                 arrIn[k][j + 1][i + 2] + \
                 arrIn[k][j - 1][i + 2] + \
                 arrIn[k][j + 1][i - 2] + \
                 arrIn[k][j - 1][i - 2] + \
                 arrIn[k + 2][j + 1][i] + \
                 arrIn[k - 2][j + 1][i] + \
                 arrIn[k + 2][j - 1][i] + \
                 arrIn[k - 2][j - 1][i] + \
                 arrIn[k + 2][j][i + 1] + \
                 arrIn[k - 2][j][i + 1] + \
                 arrIn[k + 2][j][i - 1] + \
                 arrIn[k - 2][j][i - 1] + \
                 arrIn[k][j + 2][i + 1] + \
                 arrIn[k][j - 2][i + 1] + \
                 arrIn[k][j + 2][i - 1] + \
                 arrIn[k][j - 2][i - 1]) + \
       MPI_C5 * (arrIn[k + 2][j + 2][i] + \
                 arrIn[k - 2][j + 2][i] + \
                 arrIn[k + 2][j - 2][i] + \
                 arrIn[k - 2][j - 2][i] + \
                 arrIn[k + 2][j][i + 2] + \
                 arrIn[k - 2][j][i + 2] + \
                 arrIn[k + 2][j][i - 2] + \
                 arrIn[k - 2][j][i - 2] + \
                 arrIn[k][j + 2][i + 2] + \
                 arrIn[k][j - 2][i + 2] + \
                 arrIn[k][j + 2][i - 2] + \
                 arrIn[k][j - 2][i - 2]) + \
       MPI_C6 * (arrIn[k + 1][j + 1][i + 1] + \
                 arrIn[k - 1][j + 1][i + 1] + \
                 arrIn[k + 1][j - 1][i + 1] + \
                 arrIn[k - 1][j - 1][i + 1] + \
                 arrIn[k + 1][j + 1][i - 1] + \
                 arrIn[k - 1][j + 1][i - 1] + \
                 arrIn[k + 1][j - 1][i - 1] + \
                 arrIn[k - 1][j - 1][i - 1]) + \
       MPI_C7 * (arrIn[k + 1][j + 1][i + 2] + \
                 arrIn[k - 1][j + 1][i + 2] + \
                 arrIn[k + 1][j - 1][i + 2] + \
                 arrIn[k - 1][j - 1][i + 2] + \
                 arrIn[k + 1][j + 1][i - 2] + \
                 arrIn[k - 1][j + 1][i - 2] + \
                 arrIn[k + 1][j - 1][i - 2] + \
                 arrIn[k - 1][j - 1][i - 2] + \
                 arrIn[k + 1][j + 2][i + 1] + \
                 arrIn[k - 1][j + 2][i + 1] + \
                 arrIn[k + 1][j - 2][i + 1] + \
                 arrIn[k - 1][j - 2][i + 1] + \
                 arrIn[k + 1][j + 2][i - 1] + \
                 arrIn[k - 1][j + 2][i - 1] + \
                 arrIn[k + 1][j - 2][i - 1] + \
                 arrIn[k - 1][j - 2][i - 1] + \
                 arrIn[k + 2][j + 1][i + 1] + \
                 arrIn[k - 2][j + 1][i + 1] + \
                 arrIn[k + 2][j - 1][i + 1] + \
                 arrIn[k - 2][j - 1][i + 1] + \
                 arrIn[k + 2][j + 1][i - 1] + \
                 arrIn[k - 2][j + 1][i - 1] + \
                 arrIn[k + 2][j - 1][i - 1] + \
                 arrIn[k - 2][j - 1][i - 1]) + \
       MPI_C8 * (arrIn[k + 2][j + 2][i + 1] + \
                 arrIn[k - 2][j + 2][i + 1] + \
                 arrIn[k + 2][j - 2][i + 1] + \
                 arrIn[k - 2][j - 2][i + 1] + \
                 arrIn[k + 2][j + 2][i - 1] + \
                 arrIn[k - 2][j + 2][i - 1] + \
                 arrIn[k + 2][j - 2][i - 1] + \
                 arrIn[k - 2][j - 2][i - 1] + \
                 arrIn[k + 2][j + 1][i + 2] + \
                 arrIn[k - 2][j + 1][i + 2] + \
                 arrIn[k + 2][j - 1][i + 2] + \
                 arrIn[k - 2][j - 1][i + 2] + \
                 arrIn[k + 2][j + 1][i - 2] + \
                 arrIn[k - 2][j + 1][i - 2] + \
                 arrIn[k + 2][j - 1][i - 2] + \
                 arrIn[k - 2][j - 1][i - 2] + \
                 arrIn[k + 1][j + 2][i + 2] + \
                 arrIn[k - 1][j + 2][i + 2] + \
                 arrIn[k + 1][j - 2][i + 2] + \
                 arrIn[k - 1][j - 2][i + 2] + \
                 arrIn[k + 1][j + 2][i - 2] + \
                 arrIn[k - 1][j + 2][i - 2] + \
                 arrIn[k + 1][j - 2][i - 2] + \
                 arrIn[k - 1][j - 2][i - 2]) + \
       MPI_C9 * (arrIn[k + 2][j + 2][i + 2] + \
                 arrIn[k - 2][j + 2][i + 2] + \
                 arrIn[k + 2][j - 2][i + 2] + \
                 arrIn[k - 2][j - 2][i + 2] + \
                 arrIn[k + 2][j + 2][i - 2] + \
                 arrIn[k - 2][j + 2][i - 2] + \
                 arrIn[k + 2][j - 2][i - 2] + \
                 arrIn[k - 2][j - 2][i - 2]) )
#define ST_GPU out_ptr[pos] = ( \
       MPI_C0 * in_ptr[pos] + \
       MPI_C1 * (in_ptr[pos + 1] + \
                 in_ptr[pos - 1] + \
                 in_ptr[pos + stride[1]] + \
                 in_ptr[pos - stride[1]] + \
                 in_ptr[pos + stride[2]] + \
                 in_ptr[pos - stride[2]]) + \
       MPI_C2 * (in_ptr[pos + 2] + \
                 in_ptr[pos - 2] + \
                 in_ptr[pos + 2*stride[1]] + \
                 in_ptr[pos - 2*stride[1]] + \
                 in_ptr[pos + 2*stride[2]] + \
                 in_ptr[pos - 2*stride[2]]) + \
       MPI_C3 * (in_ptr[pos + 1 + stride[1]] + \
                 in_ptr[pos - 1 + stride[1]] + \
                 in_ptr[pos + 1 - stride[1]] + \
                 in_ptr[pos - 1 - stride[1]] + \
                 in_ptr[pos + 1 + stride[2]] + \
                 in_ptr[pos - 1 + stride[2]] + \
                 in_ptr[pos + 1 - stride[2]] + \
                 in_ptr[pos - 1 - stride[2]] + \
                 in_ptr[pos + stride[1] + stride[2]] + \
                 in_ptr[pos - stride[1] + stride[2]] + \
                 in_ptr[pos + stride[1] - stride[2]] + \
                 in_ptr[pos - stride[1] - stride[2]]) + \
       MPI_C4 * (in_ptr[pos + 1 + 2*stride[1]] + \
                 in_ptr[pos - 1 + 2*stride[1]] + \
                 in_ptr[pos + 1 - 2*stride[1]] + \
                 in_ptr[pos - 1 - 2*stride[1]] + \
                 in_ptr[pos + 1 + 2*stride[2]] + \
                 in_ptr[pos - 1 + 2*stride[2]] + \
                 in_ptr[pos + 1 - 2*stride[2]] + \
                 in_ptr[pos - 1 - 2*stride[2]] + \
                 in_ptr[pos + stride[1] + 2*stride[2]] + \
                 in_ptr[pos - stride[1] + 2*stride[2]] + \
                 in_ptr[pos + stride[1] - 2*stride[2]] + \
                 in_ptr[pos - stride[1] - 2*stride[2]] + \
                 in_ptr[pos + 2 + stride[1]] + \
                 in_ptr[pos - 2 + stride[1]] + \
                 in_ptr[pos + 2 - stride[1]] + \
                 in_ptr[pos - 2 - stride[1]] + \
                 in_ptr[pos + 2 + stride[2]] + \
                 in_ptr[pos - 2 + stride[2]] + \
                 in_ptr[pos + 2 - stride[2]] + \
                 in_ptr[pos - 2 - stride[2]] + \
                 in_ptr[pos + 2*stride[1] + stride[2]] + \
                 in_ptr[pos - 2*stride[1] + stride[2]] + \
                 in_ptr[pos + 2*stride[1] - stride[2]] + \
                 in_ptr[pos - 2*stride[1] - stride[2]]) + \
       MPI_C5 * (in_ptr[pos + 2 + 2*stride[1]] + \
                 in_ptr[pos - 2 + 2*stride[1]] + \
                 in_ptr[pos + 2 - 2*stride[1]] + \
                 in_ptr[pos - 2 - 2*stride[1]] + \
                 in_ptr[pos + 2 + 2*stride[2]] + \
                 in_ptr[pos - 2 + 2*stride[2]] + \
                 in_ptr[pos + 2 - 2*stride[2]] + \
                 in_ptr[pos - 2 - 2*stride[2]] + \
                 in_ptr[pos + 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 2*stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 2*stride[1] - 2*stride[2]]) + \
       MPI_C6 * (in_ptr[pos + 1 + stride[1] + stride[2]] + \
                 in_ptr[pos - 1 + stride[1] + stride[2]] + \
                 in_ptr[pos + 1 - stride[1] + stride[2]] + \
                 in_ptr[pos - 1 - stride[1] + stride[2]] + \
                 in_ptr[pos + 1 + stride[1] - stride[2]] + \
                 in_ptr[pos - 1 + stride[1] - stride[2]] + \
                 in_ptr[pos + 1 - stride[1] - stride[2]] + \
                 in_ptr[pos - 1 - stride[1] - stride[2]]) + \
       MPI_C7 * (in_ptr[pos + 1 + stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 1 + stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 1 - stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 1 - stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 1 + stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 1 + stride[1] - 2*stride[2]] + \
                 in_ptr[pos + 1 - stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 1 - stride[1] - 2*stride[2]] + \
                 in_ptr[pos + 1 + 2*stride[1] + stride[2]] + \
                 in_ptr[pos - 1 + 2*stride[1] + stride[2]] + \
                 in_ptr[pos + 1 - 2*stride[1] + stride[2]] + \
                 in_ptr[pos - 1 - 2*stride[1] + stride[2]] + \
                 in_ptr[pos + 1 + 2*stride[1] - stride[2]] + \
                 in_ptr[pos - 1 + 2*stride[1] - stride[2]] + \
                 in_ptr[pos + 1 - 2*stride[1] - stride[2]] + \
                 in_ptr[pos - 1 - 2*stride[1] - stride[2]] + \
                 in_ptr[pos + 2 + stride[1] + stride[2]] + \
                 in_ptr[pos - 2 + stride[1] + stride[2]] + \
                 in_ptr[pos + 2 - stride[1] + stride[2]] + \
                 in_ptr[pos - 2 - stride[1] + stride[2]] + \
                 in_ptr[pos + 2 + stride[1] - stride[2]] + \
                 in_ptr[pos - 2 + stride[1] - stride[2]] + \
                 in_ptr[pos + 2 - stride[1] - stride[2]] + \
                 in_ptr[pos - 2 - stride[1] - stride[2]]) + \
       MPI_C8 * (in_ptr[pos + 2 + 2*stride[1] + stride[2]] + \
                 in_ptr[pos - 2 + 2*stride[1] + stride[2]] + \
                 in_ptr[pos + 2 - 2*stride[1] + stride[2]] + \
                 in_ptr[pos - 2 - 2*stride[1] + stride[2]] + \
                 in_ptr[pos + 2 + 2*stride[1] - stride[2]] + \
                 in_ptr[pos - 2 + 2*stride[1] - stride[2]] + \
                 in_ptr[pos + 2 - 2*stride[1] - stride[2]] + \
                 in_ptr[pos - 2 - 2*stride[1] - stride[2]] + \
                 in_ptr[pos + 2 + stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 2 + stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 2 - stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 2 - stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 2 + stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 2 + stride[1] - 2*stride[2]] + \
                 in_ptr[pos + 2 - stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 2 - stride[1] - 2*stride[2]] + \
                 in_ptr[pos + 1 + 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 1 + 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 1 - 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 1 - 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 1 + 2*stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 1 + 2*stride[1] - 2*stride[2]] + \
                 in_ptr[pos + 1 - 2*stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 1 - 2*stride[1] - 2*stride[2]]) + \
       MPI_C9 * (in_ptr[pos + 2 + 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 2 + 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 2 - 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos - 2 - 2*stride[1] + 2*stride[2]] + \
                 in_ptr[pos + 2 + 2*stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 2 + 2*stride[1] - 2*stride[2]] + \
                 in_ptr[pos + 2 - 2*stride[1] - 2*stride[2]] + \
                 in_ptr[pos - 2 - 2*stride[1] - 2*stride[2]]) )
#else
#define ST_SCRTPT "../stencils/mpi7pt.py"
#define ST_ITER 8
#define ST_CPU arrOut[k][j][i] = (arrIn[k + 1][j][i] + arrIn[k - 1][j][i] + \
                                  arrIn[k][j + 1][i] + arrIn[k][j - 1][i] + \
                                  arrIn[k][j][i + 1] + arrIn[k][j][i - 1]) * MPI_BETA + \
                                 arrIn[k][j][i] * MPI_ALPHA
#define ST_GPU out_ptr[pos] = in_ptr[pos] * MPI_ALPHA + \
                              (in_ptr[pos + stride[2]] + in_ptr[pos - stride[2]] + \
                               in_ptr[pos + stride[1]] + in_ptr[pos - stride[1]] + \
                               in_ptr[pos + 1] + in_ptr[pos - 1]) * MPI_BETA
#endif

template<unsigned n>
inline void add_brick(bElem *in, bElem *out) {
  out = (bElem *) __builtin_assume_aligned(out, 64);
  in = (bElem *) __builtin_assume_aligned(in, 64);
#pragma omp simd
  for (unsigned i = 0; i < n; ++i)
    out[i] += in[i];
}

template<unsigned...BDims, unsigned ...Folds>
inline void
fake_stencil(Brick<Dim<BDims...>, Dim<Folds...>> &in, Brick<Dim<BDims...>, Dim<Folds...>> &out, unsigned b) {
  bElem *out_ptr = &(out.dat[b * out.step]);
  out_ptr = (bElem *) __builtin_assume_aligned(out_ptr, 64);
  bElem *in_ptr = &(in.dat[b * in.step]);
  in_ptr = (bElem *) __builtin_assume_aligned(in_ptr, 64);
#pragma omp simd
  for (unsigned i = 0; i < cal_size<BDims...>::value; ++i)
    out_ptr[i] = in_ptr[i];
  unsigned mid = static_power<3, sizeof...(BDims)>::value / 2; // Mid element
  unsigned shift = static_power<3, sizeof...(BDims) - 1>::value;
  if (in.bInfo->adj[b][mid] != b)
    throw std::runtime_error("err");
  while (shift > 0) {
    in_ptr = &(in.dat[in.bInfo->adj[b][mid - shift] * in.step]);
    add_brick<cal_size<BDims...>::value>(in_ptr, out_ptr);
    in_ptr = &(in.dat[in.bInfo->adj[b][mid + shift] * in.step]);
    add_brick<cal_size<BDims...>::value>(in_ptr, out_ptr);
    shift = shift / 3;
  }
#pragma omp simd
  for (unsigned i = 0; i < cal_size<BDims...>::value; ++i)
    out_ptr[i] = out_ptr[i] / (sizeof...(BDims) * 2 + 1);
}

extern int MPI_ITER;

template<typename T, typename decomp>
double time_mpi(T func, int &cnt, decomp &bDecomp) {
  int it = MPI_ITER;
  cnt = 0;
  func(); // Warm up
  packtime = calltime = waittime = movetime = calctime = 0;
  double st = omp_get_wtime(), ed;
  for (int i = 0; i < MPI_ITER; ++i)
    func();
  ed = omp_get_wtime();
  cnt = it;
  return (ed - st) / it;
}

#endif //BRICK_FAKE_H
