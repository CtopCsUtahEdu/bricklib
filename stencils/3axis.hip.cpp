//
// Created by Samantha Hirsch on 6/22/21.
//

#include "stencils.hip.h"
#include <iostream>
#include <ctime>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "hipvfold.h"

#define COEFF_SIZE 129
#define WARPSIZE 64

__global__ void
d3pt7_brick(unsigned (*grid)[STRIDEB][STRIDEB], Brick <Dim<BDIM>, Dim<VFOLD>> bIn, Brick <Dim<BDIM>, Dim<VFOLD>> bOut,
            bElem *coeff) {
  long tj = GB + hipBlockIdx_y;
  long tk = GB + hipBlockIdx_z;
  long ti = GB + hipBlockIdx_x;
  long k = hipThreadIdx_z;
  long j = hipThreadIdx_y;
  long i = hipThreadIdx_x;
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
  long tk = GB + hipBlockIdx_z;
  long tj = GB + hipBlockIdx_y;
  long ti = GB + hipBlockIdx_x;
  unsigned b = grid[tk][tj][ti];
  brick("7pt.py", VSVEC, (BDIM), (VFOLD), b);
}

__global__ void
d3pt7_arr(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE], bElem *coeff) {
  long k = PADDING + GZ + hipBlockIdx_z * TILE + hipThreadIdx_z;
  long j = PADDING + GZ + hipBlockIdx_y * TILE + hipThreadIdx_y;
  long i = PADDING + GZ + hipBlockIdx_x * TILE + hipThreadIdx_x;
  arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                     coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                     coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                     coeff[0] * arr_in[k][j][i];
}

__global__ void
d3pt7_arr_warp(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE], bElem *coeff) {
  long tk = PADDING + GZ + hipBlockIdx_z * TILE;
  long tj = PADDING + GZ + hipBlockIdx_y * TILE;
  long i = PADDING + GZ + hipBlockIdx_x * WARPSIZE + hipThreadIdx_x;
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
  long k = GZ + hipBlockIdx_z * TILE;
  long j = GZ + hipBlockIdx_y * TILE;
  long i = GZ + hipBlockIdx_x * WARPSIZE;
  tile("7pt.py", VSVEC, (TILE, TILE, WARPSIZE), ("k", "j", "i"), (1, 1, WARPSIZE));
}

#undef bIn
#undef bOut

void d3pt7hip() {
    unsigned *grid_ptr;

    BrickInfo<3> bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
    unsigned *grid_dev;
    {
        unsigned size = (STRIDEB * STRIDEB * STRIDEB) * sizeof(unsigned);
        hipMalloc(&grid_dev, size);
        hipMemcpy(grid_dev, grid_ptr, size, hipMemcpyHostToDevice);
    }
    auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_dev;

    BrickInfo<3> *binfo_dev = movBrickInfoDeep(bInfo, hipMemcpyHostToDevice);

    // Create out data
    unsigned data_size = STRIDE * STRIDE * STRIDE * sizeof(bElem);
    bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
    bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
    bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
    bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

    // Copy over the coefficient array
    bElem *coeff_dev;
    {
        unsigned size = COEFF_SIZE * sizeof(bElem);
        hipMalloc(&coeff_dev, size);
        hipMemcpy(coeff_dev, coeff, size, hipMemcpyHostToDevice);
    }

    // Copy over the data array
    bElem *in_dev, *out_dev;
    {
        hipMalloc(&in_dev, data_size);
        hipMalloc(&out_dev, data_size);
        hipMemcpy(in_dev, in_ptr, data_size, hipMemcpyHostToDevice);
        hipMemcpy(out_dev, out_ptr, data_size, hipMemcpyHostToDevice);
    }

    // Create the bricks
    auto bsize = cal_size<BDIM>::value;
    auto bstorage = BrickStorage::allocate(bInfo.nbricks, bsize * 2);
    Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bstorage, 0);
    Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bstorage, bsize);

    // Copy data to the bricks
    copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);
    BrickStorage bstorage_dev = movBrickStorage(bstorage, hipMemcpyHostToDevice);

    auto arr_func = [&arr_in, &arr_out]() -> void {
        _TILEFOR arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                                    coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                                    coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                                    coeff[0] * arr_in[k][j][i];
    };
    auto brick_func = [&grid, &binfo_dev, &bstorage_dev, &coeff_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NB, NB, NB), thread(BDIM);
        hipLaunchKernelGGL(d3pt7_brick, block, thread, 0, 0, 
            grid, bIn, bOut, coeff_dev);
    };
    auto cuarr_func = [&in_dev, &out_dev, &coeff_dev]() -> void {
        bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
        bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
        dim3 block(NB, NB, NB), thread(BDIM);
        hipLaunchKernelGGL(d3pt7_arr, block, thread, 0, 0,
            arr_in, arr_out, coeff_dev);
    };
    auto cuarr_warp = [&in_dev, &out_dev, &coeff_dev]() -> void {
        bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
        bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
        dim3 block(N / WARPSIZE, NB, NB), thread(WARPSIZE);
        hipLaunchKernelGGL(d3pt7_arr_warp, block, thread, 0, 0,
            arr_in, arr_out, coeff_dev);
    };
    auto cuarr_scatter = [&in_dev, &out_dev, &coeff_dev]() -> void {
        bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
        bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
        dim3 block(N / WARPSIZE, NB, NB), thread(WARPSIZE);
        hipLaunchKernelGGL(d3pt7_arr_scatter, block, thread, 0, 0,
            arr_in, arr_out, coeff_dev);
    };
    auto brick_func_trans = [&grid, &binfo_dev, &bstorage_dev, &coeff_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NB, NB, NB), thread(WARPSIZE);
        hipLaunchKernelGGL(d3pt7_brick_trans, block, thread, 0, 0,
            grid, bIn, bOut, coeff_dev);
    };

    std::cout << "d3pt7" << std::endl;
    arr_func();
    std::cout << "Arr: " << hiptime_func(cuarr_func) << std::endl;
    std::cout << "Arr warp: " << hiptime_func(cuarr_warp) << std::endl;
    std::cout << "Arr scatter: " << hiptime_func(cuarr_scatter) << std::endl;
    std::cout << "Bri: " << hiptime_func(brick_func) << std::endl;
    std::cout << "Trans: " << hiptime_func(brick_func_trans) << std::endl;
    hipDeviceSynchronize();

    hipMemcpy(bstorage.dat.get(), bstorage_dev.dat.get(), bstorage.chunks * bstorage.step * sizeof(bElem), hipMemcpyDeviceToHost);

    if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    hipMemcpy(out_ptr, out_dev, bsize, hipMemcpyDeviceToHost);
    if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    free(in_ptr);
    free(out_ptr);
    free(grid_ptr);
    free(bInfo.adj);
    hipFree(in_dev);
    hipFree(out_dev);
}


__global__ void
d3cond_brick(unsigned (*grid)[STRIDEB][STRIDEB], Brick <Dim<BDIM>, Dim<VFOLD>> bIn, Brick <Dim<BDIM>, Dim<VFOLD>> bOut,
             bElem *coeff) {
  long tk = GB + hipBlockIdx_z;
  long tj = GB + hipBlockIdx_y;
  long ti = GB + hipBlockIdx_x;
  long k = hipThreadIdx_z;
  long j = hipThreadIdx_y;
  long i = hipThreadIdx_x;
  unsigned b = grid[tk][tj][ti];
  bElem partial =
      coeff[5] * fmaxf(bIn[b][k + 1][j][i], 0.0) + coeff[6] * fmaxf(bIn[b][k - 1][j][i], 0.0) +
      coeff[3] * fmaxf(bIn[b][k][j + 1][i], 0.0) + coeff[4] * fmaxf(bIn[b][k][j - 1][i], 0.0) +
      coeff[1] * fmaxf(bIn[b][k][j][i + 1], 0.0) + coeff[2] * fmaxf(bIn[b][k][j][i - 1], 0.0) +
      coeff[0] * fmaxf(bIn[b][k][j][i], 0.0);
  bOut[b][k][j][i] = partial > 0 ? partial : -partial;
}

__global__ void
d3cond_brick_trans(unsigned (*grid)[STRIDEB][STRIDEB], Brick <Dim<BDIM>, Dim<VFOLD>> bIn,
                   Brick <Dim<BDIM>, Dim<VFOLD>> bOut,
                   bElem *coeff) {
  long tk = GB + hipBlockIdx_z;
  long tj = GB + hipBlockIdx_y;
  long ti = GB + hipBlockIdx_x;
  unsigned b = grid[tk][tj][ti];
  brick("cond.py", VSVEC, (BDIM), (VFOLD), b);
}

__global__ void
d3cond_arr(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE], bElem *coeff) {
  long k = PADDING + GZ + hipBlockIdx_z * TILE + hipThreadIdx_z;
  long j = PADDING + GZ + hipBlockIdx_y * TILE + hipThreadIdx_y;
  long i = PADDING + GZ + hipBlockIdx_x * TILE + hipThreadIdx_x;
  bElem partial =
      coeff[5] * fmaxf(arr_in[k + 1][j][i], 0.0) + coeff[6] * fmaxf(arr_in[k - 1][j][i], 0.0) +
      coeff[3] * fmaxf(arr_in[k][j + 1][i], 0.0) + coeff[4] * fmaxf(arr_in[k][j - 1][i], 0.0) +
      coeff[1] * fmaxf(arr_in[k][j][i + 1], 0.0) + coeff[2] * fmaxf(arr_in[k][j][i - 1], 0.0) +
      coeff[0] * fmaxf(arr_in[k][j][i], 0.0);
  arr_out[k][j][i] = partial > 0 ? partial : -partial;
}

void d3condhip() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  unsigned *grid_dev;
  {
    unsigned size = (STRIDEB * STRIDEB * STRIDEB) * sizeof(unsigned);
    hipMalloc(&grid_dev, size);
    hipMemcpy(grid_dev, grid_ptr, size, hipMemcpyHostToDevice);
  }
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_dev;
  BrickInfo<3> *bInfo_dev;
  BrickInfo<3> _bInfo_dev = movBrickInfo(bInfo, hipMemcpyHostToDevice);
  {
    unsigned size = sizeof(BrickInfo < 3 > );
    hipMalloc(&bInfo_dev, size);
    hipMemcpy(bInfo_dev, &_bInfo_dev, size, hipMemcpyHostToDevice);
  }

  unsigned size = STRIDE * STRIDE * STRIDE * sizeof(bElem);
  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;
  bElem *coeff_dev;
  {
    unsigned size = 129 * sizeof(bElem);
    hipMalloc(&coeff_dev, size);
    hipMemcpy(coeff_dev, coeff, size, hipMemcpyHostToDevice);
  }

  bElem *in_dev, *out_dev;
  {
    hipMalloc(&in_dev, size);
    hipMemcpy(in_dev, in_ptr, size, hipMemcpyHostToDevice);
  }
  {
    hipMalloc(&out_dev, size);
    hipMemcpy(out_dev, out_ptr, size, hipMemcpyHostToDevice);
  }

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick <Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick <Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

  BrickStorage bStorage_dev = movBrickStorage(bStorage, hipMemcpyHostToDevice);

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR {
            bElem partial =
                coeff[5] * fmaxf(arr_in[k + 1][j][i], 0.0) + coeff[6] * fmaxf(arr_in[k - 1][j][i], 0.0) +
                coeff[3] * fmaxf(arr_in[k][j + 1][i], 0.0) + coeff[4] * fmaxf(arr_in[k][j - 1][i], 0.0) +
                coeff[1] * fmaxf(arr_in[k][j][i + 1], 0.0) + coeff[2] * fmaxf(arr_in[k][j][i - 1], 0.0) +
                coeff[0] * fmaxf(arr_in[k][j][i], 0.0);
            arr_out[k][j][i] = partial > 0 ? partial : -partial;
          }
  };

  auto brick_func = [&grid, &bInfo_dev, &bStorage_dev, &coeff_dev]() -> void {
    auto bSize = cal_size<BDIM>::value;
    Brick <Dim<BDIM>, Dim<VFOLD>> bIn(bInfo_dev, bStorage_dev, 0);
    Brick <Dim<BDIM>, Dim<VFOLD>> bOut(bInfo_dev, bStorage_dev, bSize);
    dim3 block(NB, NB, NB), thread(BDIM);
    hipLaunchKernelGGL(d3cond_brick, block, thread, 0, 0,
        grid, bIn, bOut, coeff_dev);
  };

  auto cuarr_func = [&in_dev, &out_dev, &coeff_dev]() -> void {
    bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
    bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
    dim3 block(NB, NB, NB), thread(BDIM);
    hipLaunchKernelGGL(d3cond_arr, block, thread, 0, 0,
        arr_in, arr_out, coeff_dev);
  };

  auto brick_func_trans = [&grid, &bInfo_dev, &bStorage_dev, &coeff_dev]() -> void {
    auto bSize = cal_size<BDIM>::value;
    Brick <Dim<BDIM>, Dim<VFOLD>> bIn(bInfo_dev, bStorage_dev, 0);
    Brick <Dim<BDIM>, Dim<VFOLD>> bOut(bInfo_dev, bStorage_dev, bSize);
    dim3 block(NB, NB, NB), thread(WARPSIZE);
    hipLaunchKernelGGL(d3cond_brick_trans, block, thread, 0, 0,
        grid, bIn, bOut, coeff_dev);
  };

  std::cout << "d3cond" << std::endl;
  arr_func();
  std::cout << "Arr: " << hiptime_func(cuarr_func) << std::endl;
  std::cout << "Bri: " << hiptime_func(brick_func) << std::endl;
  std::cout << "Trans: " << hiptime_func(brick_func_trans) << std::endl;

  hipMemcpy(bStorage.dat.get(), bStorage_dev.dat.get(), bStorage.chunks * bStorage.step * sizeof(bElem), hipMemcpyDeviceToHost);
  hipDeviceSynchronize();

  if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  hipMemcpy(out_ptr, out_dev, size, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();

  if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bInfo.adj);
  hipFree(_bInfo_dev.adj);
  hipFree(in_dev);
  hipFree(out_dev);
}
