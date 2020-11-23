//
// Created by Tuowen Zhao on 12/5/18.
//

#include "stencils_cu.h"
#include <iostream>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "gpuvfold.h"

__global__ void
d3pt7_brick(unsigned (*grid)[STRIDEB][STRIDEB], Brick <Dim<BDIM>, Dim<VFOLD>> bIn, Brick <Dim<BDIM>, Dim<VFOLD>> bOut,
            bElem *coeff) {
  long tk = GB + blockIdx.z;
  long tj = GB + blockIdx.y;
  long ti = GB + blockIdx.x;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned b = grid[tk][tj][ti];
  bOut[b][k][j][i] = coeff[5] * bIn[b][k + 1][j][i] + coeff[6] * bIn[b][k - 1][j][i] +
                     coeff[3] * bIn[b][k][j + 1][i] + coeff[4] * bIn[b][k][j - 1][i] +
                     coeff[1] * bIn[b][k][j][i + 1] + coeff[2] * bIn[b][k][j][i - 1] +
                     coeff[0] * bIn[b][k][j][i];
}

__global__ void
d3pt7_brick_trans(unsigned (*grid)[STRIDEB][STRIDEB], Brick <Dim<BDIM>, Dim<VFOLD>> bIn,
                  Brick <Dim<BDIM>, Dim<VFOLD>> bOut,
                  bElem *coeff) {
  long tk = GB + blockIdx.z;
  long tj = GB + blockIdx.y;
  long ti = GB + blockIdx.x;
  unsigned b = grid[tk][tj][ti];
  brick("7pt.py", VSVEC, (BDIM), (VFOLD), b);
}

__global__ void
d3pt7_arr(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE], bElem *coeff) {
  long k = PADDING + GZ + blockIdx.z * TILE + threadIdx.z;
  long j = PADDING + GZ + blockIdx.y * TILE + threadIdx.y;
  long i = PADDING + GZ + blockIdx.x * TILE + threadIdx.x;
  arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                     coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                     coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                     coeff[0] * arr_in[k][j][i];
}

__global__ void
d3pt7_arr_warp(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE], bElem *coeff) {
  long tk = PADDING + GZ + blockIdx.z * TILE;
  long tj = PADDING + GZ + blockIdx.y * TILE;
  long i = PADDING + GZ + blockIdx.x * 32 + threadIdx.x;
  for (int k = tk; k < tk + TILE; ++k)
    for (int j = tj; j < tj + TILE; ++j)
      arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                         coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                         coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                         coeff[0] * arr_in[k][j][i];
}

#define bIn(i, j, k) arr_in[k][j][i]
#define bOut(i, j, k) arr_out[k][j][i]

__global__ void
d3pt7_arr_scatter(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE], bElem *coeff) {
  long k = GZ + blockIdx.z * TILE;
  long j = GZ + blockIdx.y * TILE;
  long i = GZ + blockIdx.x * 32;
  tile("7pt.py", VSVEC, (TILE, TILE, 32), ("k", "j", "i"), (1, 1, 32));
}

#undef bIn
#undef bOut

void d3pt7cu() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  unsigned *grid_dev;
  {
    unsigned size = (STRIDEB * STRIDEB * STRIDEB) * sizeof(unsigned);
    cudaMalloc(&grid_dev, size);
    cudaMemcpy(grid_dev, grid_ptr, size, cudaMemcpyHostToDevice);
  }
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_dev;
  BrickInfo<3> *bInfo_dev;
  BrickInfo<3> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  {
    unsigned size = sizeof(BrickInfo < 3 > );
    cudaMalloc(&bInfo_dev, size);
    cudaMemcpy(bInfo_dev, &_bInfo_dev, size, cudaMemcpyHostToDevice);
  }

  unsigned size = STRIDE * STRIDE * STRIDE * sizeof(bElem);
  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;
  bElem *coeff_dev;
  {
    unsigned size = 129 * sizeof(bElem);
    cudaMalloc(&coeff_dev, size);
    cudaMemcpy(coeff_dev, coeff, size, cudaMemcpyHostToDevice);
  }

  bElem *in_dev, *out_dev;
  {
    cudaMalloc(&in_dev, size);
    cudaMemcpy(in_dev, in_ptr, size, cudaMemcpyHostToDevice);
  }
  {
    cudaMalloc(&out_dev, size);
    cudaMemcpy(out_dev, out_ptr, size, cudaMemcpyHostToDevice);
  }

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick <Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick <Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

  BrickStorage bStorage_dev = movBrickStorage(bStorage, cudaMemcpyHostToDevice);

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                                coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                                coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                                coeff[0] * arr_in[k][j][i];
  };

  auto brick_func = [&grid, &bInfo_dev, &bStorage_dev, &coeff_dev]() -> void {
    auto bSize = cal_size<BDIM>::value;
    Brick <Dim<BDIM>, Dim<VFOLD>> bIn(bInfo_dev, bStorage_dev, 0);
    Brick <Dim<BDIM>, Dim<VFOLD>> bOut(bInfo_dev, bStorage_dev, bSize);
    dim3 block(NB, NB, NB), thread(BDIM);
    d3pt7_brick << < block, thread >> > (grid, bIn, bOut, coeff_dev);
  };

  auto cuarr_func = [&in_dev, &out_dev, &coeff_dev]() -> void {
    bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
    bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
    dim3 block(NB, NB, NB), thread(BDIM);
    d3pt7_arr << < block, thread >> > (arr_in, arr_out, coeff_dev);
  };

  auto cuarr_warp = [&in_dev, &out_dev, &coeff_dev]() -> void {
    bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
    bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
    dim3 block(N / 32, NB, NB), thread(32);
    d3pt7_arr_warp << < block, thread >> > (arr_in, arr_out, coeff_dev);
  };

  auto cuarr_scatter = [&in_dev, &out_dev, &coeff_dev]() -> void {
    bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
    bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
    dim3 block(N / 32, NB, NB), thread(32);
    d3pt7_arr_scatter << < block, thread >> > (arr_in, arr_out, coeff_dev);
  };

  auto brick_func_trans = [&grid, &bInfo_dev, &bStorage_dev, &coeff_dev]() -> void {
    auto bSize = cal_size<BDIM>::value;
    Brick <Dim<BDIM>, Dim<VFOLD>> bIn(bInfo_dev, bStorage_dev, 0);
    Brick <Dim<BDIM>, Dim<VFOLD>> bOut(bInfo_dev, bStorage_dev, bSize);
    dim3 block(NB, NB, NB), thread(32);
    d3pt7_brick_trans << < block, thread >> > (grid, bIn, bOut, coeff_dev);
  };

  std::cout << "d3pt7" << std::endl;
  arr_func();
  std::cout << "Arr: " << cutime_func(cuarr_func) << std::endl;
  std::cout << "Arr warp: " << cutime_func(cuarr_warp) << std::endl;
  std::cout << "Arr scatter: " << cutime_func(cuarr_scatter) << std::endl;
  std::cout << "Bri: " << cutime_func(brick_func) << std::endl;
  std::cout << "Trans: " << cutime_func(brick_func_trans) << std::endl;

  cudaMemcpy(bStorage.dat.get(), bStorage_dev.dat.get(), bStorage.chunks * bStorage.step * sizeof(bElem), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  cudaMemcpy(out_ptr, out_dev, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bInfo.adj);
  cudaFree(_bInfo_dev.adj);
  cudaFree(in_dev);
  cudaFree(out_dev);
}

__global__ void
d3cond_brick(unsigned (*grid)[STRIDEB][STRIDEB], Brick <Dim<BDIM>, Dim<VFOLD>> bIn, Brick <Dim<BDIM>, Dim<VFOLD>> bOut,
             bElem *coeff) {
  long tk = GB + blockIdx.z;
  long tj = GB + blockIdx.y;
  long ti = GB + blockIdx.x;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned b = grid[tk][tj][ti];
  bElem partial =
      coeff[5] * max(bIn[b][k + 1][j][i], 0.0) + coeff[6] * max(bIn[b][k - 1][j][i], 0.0) +
      coeff[3] * max(bIn[b][k][j + 1][i], 0.0) + coeff[4] * max(bIn[b][k][j - 1][i], 0.0) +
      coeff[1] * max(bIn[b][k][j][i + 1], 0.0) + coeff[2] * max(bIn[b][k][j][i - 1], 0.0) +
      coeff[0] * max(bIn[b][k][j][i], 0.0);
  bOut[b][k][j][i] = partial > 0 ? partial : -partial;
}

__global__ void
d3cond_brick_trans(unsigned (*grid)[STRIDEB][STRIDEB], Brick <Dim<BDIM>, Dim<VFOLD>> bIn,
                   Brick <Dim<BDIM>, Dim<VFOLD>> bOut,
                   bElem *coeff) {
  long tk = GB + blockIdx.z;
  long tj = GB + blockIdx.y;
  long ti = GB + blockIdx.x;
  unsigned b = grid[tk][tj][ti];
  brick("cond.py", VSVEC, (BDIM), (VFOLD), b);
}

__global__ void
d3cond_arr(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE], bElem *coeff) {
  long k = PADDING + GZ + blockIdx.z * TILE + threadIdx.z;
  long j = PADDING + GZ + blockIdx.y * TILE + threadIdx.y;
  long i = PADDING + GZ + blockIdx.x * TILE + threadIdx.x;
  bElem partial =
      coeff[5] * max(arr_in[k + 1][j][i], 0.0) + coeff[6] * max(arr_in[k - 1][j][i], 0.0) +
      coeff[3] * max(arr_in[k][j + 1][i], 0.0) + coeff[4] * max(arr_in[k][j - 1][i], 0.0) +
      coeff[1] * max(arr_in[k][j][i + 1], 0.0) + coeff[2] * max(arr_in[k][j][i - 1], 0.0) +
      coeff[0] * max(arr_in[k][j][i], 0.0);
  arr_out[k][j][i] = partial > 0 ? partial : -partial;
}

void d3condcu() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  unsigned *grid_dev;
  {
    unsigned size = (STRIDEB * STRIDEB * STRIDEB) * sizeof(unsigned);
    cudaMalloc(&grid_dev, size);
    cudaMemcpy(grid_dev, grid_ptr, size, cudaMemcpyHostToDevice);
  }
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_dev;
  BrickInfo<3> *bInfo_dev;
  BrickInfo<3> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  {
    unsigned size = sizeof(BrickInfo < 3 > );
    cudaMalloc(&bInfo_dev, size);
    cudaMemcpy(bInfo_dev, &_bInfo_dev, size, cudaMemcpyHostToDevice);
  }

  unsigned size = STRIDE * STRIDE * STRIDE * sizeof(bElem);
  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;
  bElem *coeff_dev;
  {
    unsigned size = 129 * sizeof(bElem);
    cudaMalloc(&coeff_dev, size);
    cudaMemcpy(coeff_dev, coeff, size, cudaMemcpyHostToDevice);
  }

  bElem *in_dev, *out_dev;
  {
    cudaMalloc(&in_dev, size);
    cudaMemcpy(in_dev, in_ptr, size, cudaMemcpyHostToDevice);
  }
  {
    cudaMalloc(&out_dev, size);
    cudaMemcpy(out_dev, out_ptr, size, cudaMemcpyHostToDevice);
  }

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick <Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick <Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

  BrickStorage bStorage_dev = movBrickStorage(bStorage, cudaMemcpyHostToDevice);

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

  auto brick_func = [&grid, &bInfo_dev, &bStorage_dev, &coeff_dev]() -> void {
    auto bSize = cal_size<BDIM>::value;
    Brick <Dim<BDIM>, Dim<VFOLD>> bIn(bInfo_dev, bStorage_dev, 0);
    Brick <Dim<BDIM>, Dim<VFOLD>> bOut(bInfo_dev, bStorage_dev, bSize);
    dim3 block(NB, NB, NB), thread(BDIM);
    d3cond_brick << < block, thread >> > (grid, bIn, bOut, coeff_dev);
  };

  auto cuarr_func = [&in_dev, &out_dev, &coeff_dev]() -> void {
    bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
    bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
    dim3 block(NB, NB, NB), thread(BDIM);
    d3cond_arr << < block, thread >> > (arr_in, arr_out, coeff_dev);
  };

  auto brick_func_trans = [&grid, &bInfo_dev, &bStorage_dev, &coeff_dev]() -> void {
    auto bSize = cal_size<BDIM>::value;
    Brick <Dim<BDIM>, Dim<VFOLD>> bIn(bInfo_dev, bStorage_dev, 0);
    Brick <Dim<BDIM>, Dim<VFOLD>> bOut(bInfo_dev, bStorage_dev, bSize);
    dim3 block(NB, NB, NB), thread(32);
    d3cond_brick_trans << < block, thread >> > (grid, bIn, bOut, coeff_dev);
  };

  std::cout << "d3cond" << std::endl;
  arr_func();
  std::cout << "Arr: " << cutime_func(cuarr_func) << std::endl;
  std::cout << "Bri: " << cutime_func(brick_func) << std::endl;
  std::cout << "Trans: " << cutime_func(brick_func_trans) << std::endl;

  cudaMemcpy(bStorage.dat.get(), bStorage_dev.dat.get(), bStorage.chunks * bStorage.step * sizeof(bElem), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  cudaMemcpy(out_ptr, out_dev, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bInfo.adj);
  cudaFree(_bInfo_dev.adj);
  cudaFree(in_dev);
  cudaFree(out_dev);
}
