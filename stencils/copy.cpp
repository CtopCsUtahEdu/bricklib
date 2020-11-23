//
// Created by Tuowen Zhao on 12/TILE/18.
//

#include "stencils.h"
#include <iostream>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "cpuvfold.h"

void copy() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto bSize = cal_size<BDIM>::value;

  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] = arr_in[k][j][i];
  };

  auto to_func = [&grid, &bIn, &arr_in]() -> void {
    _PARFOR
    for (long tk = 0; tk < STRIDEB; ++tk)
      for (long tj = 0; tj < STRIDEB; ++tj)
        for (long ti = 0; ti < STRIDEB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                bIn[b][k][j][i] = arr_in[PADDING + tk * TILE + k][PADDING + tj * TILE + j][PADDING + ti * TILE + i];
              }
        }
  };

  auto from_func = [&grid, &bIn, &arr_out]() -> void {
    _PARFOR
    for (long tk = 0; tk < STRIDEB; ++tk)
      for (long tj = 0; tj < STRIDEB; ++tj)
        for (long ti = 0; ti < STRIDEB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                arr_out[PADDING + tk * TILE + k][PADDING + tj * TILE + j][PADDING + ti * TILE + i] = bIn[b][k][j][i];
              }
        }
  };

  std::cout << "Copy" << std::endl;
  int cnt;
  std::cout << "Arr: " << time_func(arr_func) << std::endl;
  std::cout << "To: " << time_func(to_func) << std::endl;
  if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, in_ptr, grid_ptr, bIn))
    throw std::runtime_error("result mismatch!");
  std::cout << "From: " << time_func(from_func) << std::endl;
  if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bIn))
    throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bInfo.adj);
}
