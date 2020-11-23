//
// Created by Tuowen Zhao on 12/6/18.
//

#include <mpi.h>
#include <iostream>
#include <brick.h>
#include <brick-mpi.h>
#include <bricksetup.h>
#include <brickcompare.h>
#include <multiarray.h>
#include "stencils/stencils.h"
#include "stencils/fake.h"
#include "stencils/cpuvfold.h"

void regular(bElem *in_ptr, bElem *out_ptr) {
  unsigned *grid_ptr;
  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  auto arr_in = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  auto arr_out = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(NB * NB * NB + 1, bSize * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDE, STRIDE, STRIDE}, in_ptr, grid_ptr, bIn);

  auto brick_func = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          fake_stencil(bIn, bOut, grid[tk][tj][ti]);
        }
  };

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] = arr_in[k + TILE][j][i] + arr_in[k - TILE][j][i] +
                                arr_in[k][j + TILE][i] + arr_in[k][j - TILE][i] +
                                arr_in[k][j][i + TILE] + arr_in[k][j][i - TILE] +
                                arr_in[k][j][i];
  };

  std::cout << "d3pt7 regular decomp" << std::endl;
  std::cout << "Bri: " << time_func(brick_func) << std::endl;
  std::cout << "Arr: " << time_func(arr_func) << std::endl;

  if (!compareBrick<3>({STRIDE, STRIDE, STRIDE}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(grid_ptr);
  free(bInfo.adj);
}

void decomp(bElem *in_ptr, bElem *out_ptr) {
  std::vector<BitSet> skinlist = {
      { 1}, { 1,-3}, { 1, 2,-3}, { 1, 2}, { 1, 2, 3}, { 2, 3}, { 2},
      { 2,-3}, {-1, 2,-3}, {-1, 2}, {-1, 2, 3}, {-1, 3}, {-1},
      {-3}, {-1,-3}, {-1,-2,-3}, {-1,-2}, {-1,-2, 3}, {-2, 3}, {-2},
      {-2,-3}, { 1,-2,-3}, { 1,-2}, { 1,-2, 3}, { 1, 3}, { 3}
  };

  BrickDecomp<3, BDIM> bDecomp({512, 512, 512}, 8);

  auto bSize = cal_size<BDIM>::value;
  bDecomp.initialize(skinlist);
  BrickInfo<3> bInfo = bDecomp.getBrickInfo();
  auto bStorage = bInfo.allocate(bSize * 2);

  auto grid_ptr = (unsigned *)malloc(sizeof(unsigned) * STRIDEB * STRIDEB * STRIDEB);
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  for (long i = 0; i < STRIDEB; ++i)
    for (long j = 0; j < STRIDEB; ++j)
      for (long k = 0; k < STRIDEB; ++k)
        grid[i][j][k] = bDecomp[i][j][k];

  for (long i = 1; i < STRIDEB - 1; ++i)
    for (long j = 1; j < STRIDEB - 1; ++j)
      for (long k = 1; k < STRIDEB - 1; ++k) {
        auto l = grid[i][j][k];
        for (long id = 0; id < 27; ++id)
          if (bInfo.adj[bInfo.adj[l][id]][26-id] != l)
            throw std::runtime_error("err");
      }

  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDE, STRIDE, STRIDE}, in_ptr, grid_ptr, bIn);

  auto arr_in = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  auto arr_out = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto brick_func = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          fake_stencil(bIn, bOut, b);
        }
  };

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] = arr_in[k + TILE][j][i] + arr_in[k - TILE][j][i] +
                                arr_in[k][j + TILE][i] + arr_in[k][j - TILE][i] +
                                arr_in[k][j][i + TILE] + arr_in[k][j][i - TILE] +
                                arr_in[k][j][i];
  };

  std::cout << "d3pt7 MPI decomp" << std::endl;
  std::cout << "Bri: " << time_func(brick_func) << std::endl;

  if (!compareBrick<3>({STRIDE, STRIDE, STRIDE}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(bInfo.adj);
}

int main(int argc, char **argv) {
  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});

  regular(in_ptr, out_ptr);

  decomp(in_ptr, out_ptr);

  free(in_ptr);

  return 0;
}
