//
// Created by Tuowen Zhao on 12/4/18.
//

#include "stencils.h"
#include <iostream>
#include "immintrin.h"
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "cpuvfold.h"

using std::max;

void d3pt7() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                                coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                                coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                                coeff[0] * arr_in[k][j][i];
  };

#define bIn(i, j, k) arr_in[k][j][i]
#define bOut(i, j, k) arr_out[k][j][i]
  auto arr_tile_func = [&arr_in, &arr_out]() -> void {
    #pragma omp parallel for
    for (long tk = GZ; tk < STRIDE - GZ; tk += TILE)
    for (long tj = GZ; tj < STRIDE - GZ; tj += TILE)
    for (long ti = GZ; ti < STRIDE - GZ; ti += TILE)
      tile("7pt.py", "FLEX", (BDIM), ("tk", "tj", "ti"), (1,1,4));
  };
#undef bIn
#undef bOut

  auto brick_func = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                bOut[b][k][j][i] = coeff[5] * bIn[b][k + 1][j][i] + coeff[6] * bIn[b][k - 1][j][i] +
                                   coeff[3] * bIn[b][k][j + 1][i] + coeff[4] * bIn[b][k][j - 1][i] +
                                   coeff[1] * bIn[b][k][j][i + 1] + coeff[2] * bIn[b][k][j][i - 1] +
                                   coeff[0] * bIn[b][k][j][i];
              }
        }
  };

  auto brick_func_trans = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          brick("7pt.py", VSVEC, (BDIM), (VFOLD), b);
        }
  };

  std::cout << "d3pt7" << std::endl;
  std::cout << "Arr: " << time_func(arr_func) << std::endl;
  std::cout << "Bri: " << time_func(brick_func) << std::endl;
  if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");
  std::cout << "Arr Scatter: " << time_func(arr_tile_func) << std::endl;
  std::cout << "Trans: " << time_func(brick_func_trans) << std::endl;
  if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bStorage.dat);
  free(bInfo.adj);
}

void d3cond() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR {
            bElem partial =
                coeff[5] * max(arr_in[k + 1][j][i], 0.0) + coeff[6] * max(arr_in[k - 1][j][i], 0.0) +
                coeff[3] * max(arr_in[k][j + 1][i], 0.0) + coeff[4] * max(arr_in[k][j - 1][i], 0.0) +
                coeff[1] * max(arr_in[k][j][i + 1], 0.0) + coeff[2] * max(arr_in[k][j][i - 1], 0.0) +
                coeff[0] * max(arr_in[k][j][i], 0.0);
            arr_out[k][j][i] = partial > 0 ? partial : -partial;
          }
  };

  auto brick_func = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                bElem partial =
                    coeff[5] * max(bIn[b][k + 1][j][i], 0.0) + coeff[6] * max(bIn[b][k - 1][j][i], 0.0) +
                    coeff[3] * max(bIn[b][k][j + 1][i], 0.0) + coeff[4] * max(bIn[b][k][j - 1][i], 0.0) +
                    coeff[1] * max(bIn[b][k][j][i + 1], 0.0) + coeff[2] * max(bIn[b][k][j][i - 1], 0.0) +
                    coeff[0] * max(bIn[b][k][j][i], 0.0);
                bOut[b][k][j][i] = partial > 0 ? partial : -partial;
              }
        }
  };

  auto brick_func_trans = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          brick("cond.py", VSVEC, (BDIM), (VFOLD), b);
        }
  };

  std::cout << "cond" << std::endl;
  std::cout << "Arr: " << time_func(arr_func) << std::endl;
  std::cout << "Bri: " << time_func(brick_func) << std::endl;
  std::cout << "Trans: " << time_func(brick_func_trans) << std::endl;
  if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bStorage.dat);
  free(bInfo.adj);
}

void d3pt27() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] =
                 coeff[0] * arr_in[k - 1][j - 1][i - 1] + coeff[1] * arr_in[k - 1][j - 1][i] +
                 coeff[2] * arr_in[k - 1][j - 1][i + 1] +
                 coeff[3] * arr_in[k - 1][j][i - 1] + coeff[4] * arr_in[k - 1][j][i] +
                 coeff[5] * arr_in[k - 1][j][i + 1] +
                 coeff[6] * arr_in[k - 1][j + 1][i - 1] + coeff[7] * arr_in[k - 1][j + 1][i] +
                 coeff[8] * arr_in[k - 1][j + 1][i + 1] +
                 coeff[9] * arr_in[k][j - 1][i - 1] + coeff[10] * arr_in[k][j - 1][i] +
                 coeff[11] * arr_in[k][j - 1][i + 1] +
                 coeff[12] * arr_in[k][j][i - 1] + coeff[13] * arr_in[k][j][i] +
                 coeff[14] * arr_in[k][j][i + 1] +
                 coeff[15] * arr_in[k][j + 1][i - 1] + coeff[16] * arr_in[k + 1][j + 1][i] +
                 coeff[17] * arr_in[k][j + 1][i + 1] +
                 coeff[18] * arr_in[k + 1][j - 1][i - 1] + coeff[19] * arr_in[k + 1][j - 1][i] +
                 coeff[20] * arr_in[k + 1][j - 1][i + 1] +
                 coeff[21] * arr_in[k + 1][j][i - 1] + coeff[22] * arr_in[k + 1][j][i] +
                 coeff[23] * arr_in[k + 1][j][i + 1] +
                 coeff[24] * arr_in[k + 1][j + 1][i - 1] + coeff[25] * arr_in[k + 1][j + 1][i] +
                 coeff[26] * arr_in[k + 1][j + 1][i + 1];
  };

  auto brick_func = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                bOut[b][k][j][i] =
                    coeff[0] * bIn[b][k - 1][j - 1][i - 1] + coeff[1] * bIn[b][k - 1][j - 1][i] +
                    coeff[2] * bIn[b][k - 1][j - 1][i + 1] +
                    coeff[3] * bIn[b][k - 1][j][i - 1] + coeff[4] * bIn[b][k - 1][j][i] +
                    coeff[5] * bIn[b][k - 1][j][i + 1] +
                    coeff[6] * bIn[b][k - 1][j + 1][i - 1] + coeff[7] * bIn[b][k - 1][j + 1][i] +
                    coeff[8] * bIn[b][k - 1][j + 1][i + 1] +
                    coeff[9] * bIn[b][k][j - 1][i - 1] + coeff[10] * bIn[b][k][j - 1][i] +
                    coeff[11] * bIn[b][k][j - 1][i + 1] +
                    coeff[12] * bIn[b][k][j][i - 1] + coeff[13] * bIn[b][k][j][i] +
                    coeff[14] * bIn[b][k][j][i + 1] +
                    coeff[15] * bIn[b][k][j + 1][i - 1] + coeff[16] * bIn[b][k + 1][j + 1][i] +
                    coeff[17] * bIn[b][k][j + 1][i + 1] +
                    coeff[18] * bIn[b][k + 1][j - 1][i - 1] + coeff[19] * bIn[b][k + 1][j - 1][i] +
                    coeff[20] * bIn[b][k + 1][j - 1][i + 1] +
                    coeff[21] * bIn[b][k + 1][j][i - 1] + coeff[22] * bIn[b][k + 1][j][i] +
                    coeff[23] * bIn[b][k + 1][j][i + 1] +
                    coeff[24] * bIn[b][k + 1][j + 1][i - 1] + coeff[25] * bIn[b][k + 1][j + 1][i] +
                    coeff[26] * bIn[b][k + 1][j + 1][i + 1];
              }
        }
  };

  std::cout << "d3pt27" << std::endl;
  std::cout << "Arr: " << time_func(arr_func) << std::endl;
  std::cout << "Bri: " << time_func(brick_func) << std::endl;
  if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bStorage.dat);
  free(bInfo.adj);
}
