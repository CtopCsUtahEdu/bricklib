//
// Created by Tuowen Zhao on 2/17/19.
// Deprecated: doesn't perform, might need fixes to work
//

#include <mpi.h>
#include <iostream>
#include <brick.h>
#include <brick-mpi.h>
#include <bricksetup.h>
#include <brick-cuda.h>
#include <cuda.h>
#include "stencils/stencils.h"
#include "stencils/fake.h"

#include "bitset.h"
#include "stencils/multiarray.h"
#include "stencils/brickcompare.h"
#include "stencils/cudaarray.h"
#include "stencils/gpuvfold.h"

#include <unistd.h>
#include <array-mpi.h>

int MPI_ITER = 100;

typedef Brick<Dim<BDIM>, Dim<VFOLD>> Brick3D;

__global__ void
arr_kernel(bElem *in_ptr, bElem *out_ptr) {
  auto in_arr = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  auto out_arr = (bElem (*)[STRIDE][STRIDE]) out_ptr;
  long k = PADDING + GZ + blockIdx.z * TILE + threadIdx.z;
  long j = PADDING + GZ + blockIdx.y * TILE + threadIdx.y;
  long i = PADDING + GZ + blockIdx.x * TILE + threadIdx.x;
  out_arr[k][j][i] =
      in_arr[k + 1][j][i] + in_arr[k - 1][j][i] +
      in_arr[k][j + 1][i] + in_arr[k][j - 1][i] +
      in_arr[k][j][i + 1] + in_arr[k][j][i - 1] +
      in_arr[k][j][i];
}

__global__ void
brick_kernel(unsigned (*grid)[STRIDEB][STRIDEB], Brick3D in, Brick3D out) {
  unsigned bk = GZ / TILE + blockIdx.z;
  unsigned bj = GZ / TILE + blockIdx.y;
  unsigned bi = GZ / TILE + blockIdx.x;

  unsigned b = grid[bk][bj][bi];
  unsigned k = threadIdx.z;
  unsigned j = threadIdx.y;
  unsigned i = threadIdx.x;

  out[b][k][j][i] =
      in[b][k + 1][j][i] + in[b][k - 1][j][i] +
      in[b][k][j + 1][i] + in[b][k][j - 1][i] +
      in[b][k][j][i + 1] + in[b][k][j][i - 1] +
      in[b][k][j][i];
}

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided != MPI_THREAD_SERIALIZED) {
    MPI_Finalize();
    return 1;
  }

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MEMFD::setup_prefix("mpi-main", rank);

  int dims[3];
  dims[0] = dims[1] = dims[2] = 0;
  MPI_Dims_create(size, 3, dims);
  if (rank == 0) {
    int page_size = sysconf(_SC_PAGESIZE);

    std::cout << "Running with pagesize " << page_size << std::endl;
    std::cout << "MPI Size " << size << ", dims: ";
    for (int i = 0; i < 3; ++i)
      std::cout << dims[i] << " ";
    std::cout << std::endl;
    int numthreads;
#pragma omp parallel
    numthreads = omp_get_num_threads();
    std::cout << "OpenMP threads " << numthreads << std::endl;
  }
  int prd[3] = {1, 1, 1};
  MPI_Comm cart;
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, prd, 0, &cart);

  if (cart != MPI_COMM_NULL) {
    MPI_Comm_rank(cart, &rank);
    int coo[3];
    MPI_Cart_get(cart, 3, dims, prd, coo);

    bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});

    std::vector<BitSet> skinlist = {
        {1},
        {1,  -3},
        {1,  2,  -3},
        {1,  2},
        {1,  2,  3},
        {2,  3},
        {2},
        {2,  -3},
        {-1, 2,  -3},
        {-1, 2},
        {-1, 2,  3},
        {-1, 3},
        {-1},
        {-3},
        {-1, -3},
        {-1, -2, -3},
        {-1, -2},
        {-1, -2, 3},
        {-2, 3},
        {-2},
        {-2, -3},
        {1,  -2, -3},
        {1,  -2},
        {1,  -2, 3},
        {1,  3},
        {3}
    };

    CUdevice device = 0;
    CUcontext pctx;
    cudaCheck((cudaError_t) cudaSetDevice(device));
    cudaCheck((cudaError_t) cuCtxCreate(&pctx, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device));

    BrickDecomp<3, BDIM> bDecomp({N, N, N}, GZ);
    bDecomp.comm = cart;
    populate(cart, bDecomp, 0, 1, coo);

    auto bSize = cal_size<BDIM>::value;
    bDecomp.initialize(skinlist);
    BrickInfo<3> bInfo = bDecomp.getBrickInfo();
    auto bStorage = bInfo.mmap_alloc(bSize);
    auto bStorageOut = bInfo.mmap_alloc(bSize);

    auto grid_ptr = (unsigned *) malloc(sizeof(unsigned) * STRIDEB * STRIDEB * STRIDEB);
    auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

    for (long i = 0; i < STRIDEB; ++i)
      for (long j = 0; j < STRIDEB; ++j)
        for (long k = 0; k < STRIDEB; ++k)
          grid[i][j][k] = bDecomp[i][j][k];

    Brick3D bIn(&bInfo, bStorage, 0);
    Brick3D bOut(&bInfo, bStorageOut, 0);

    copyToBrick<3>({STRIDE, STRIDE, STRIDE}, in_ptr, grid_ptr, bIn);

    bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});

    ExchangeView ev = bDecomp.exchangeView(bStorage);

    cudaCheck(cudaMemAdvise(bStorage.dat,
                            bStorage.step * bDecomp.sep_pos[0] * sizeof(bElem), cudaMemAdviseSetPreferredLocation,
                            device));

    cudaCheck(cudaMemAdvise(bStorage.dat + bStorage.step * bDecomp.sep_pos[0],
                            bStorage.step * (bDecomp.sep_pos[2] - bDecomp.sep_pos[0]) * sizeof(bElem),
                            cudaMemAdviseSetPreferredLocation,
                            cudaCpuDeviceId));

    cudaMemPrefetchAsync(bStorage.dat, bStorage.step * bDecomp.sep_pos[0] * sizeof(bElem), device);

    cudaCheck(cudaMemAdvise(bStorageOut.dat,
                            bStorage.step * bDecomp.sep_pos[0] * sizeof(bElem), cudaMemAdviseSetPreferredLocation,
                            device));

    cudaCheck(cudaMemAdvise(bStorageOut.dat + bStorage.step * bDecomp.sep_pos[0],
                            bStorage.step * (bDecomp.sep_pos[2] - bDecomp.sep_pos[0]) * sizeof(bElem),
                            cudaMemAdviseSetPreferredLocation,
                            cudaCpuDeviceId));

    cudaMemPrefetchAsync(bStorageOut.dat, bStorage.step * bDecomp.sep_pos[0] * sizeof(bElem), device);

    cudaMemPrefetchAsync(grid_ptr, STRIDEB * STRIDEB * STRIDEB * sizeof(unsigned), device);

    auto brick_func = [&]() -> void {
      // bDecomp.exchange(bStorage);

      ev.exchange();

      dim3 block(N / TILE, N / TILE, N / TILE), thread(TILE, TILE, TILE);

      brick_kernel <<< block, thread >>> (grid, bIn, bOut);
      cudaDeviceSynchronize();
    };

    bElem *in_ptr_dev = nullptr;
    bElem *out_ptr_dev = nullptr;

    copyToDevice({STRIDE, STRIDE, STRIDE}, in_ptr_dev, in_ptr);
    copyToDevice({STRIDE, STRIDE, STRIDE}, out_ptr_dev, out_ptr);

    auto arr_func = [&]() -> void {
      // Copy everything back from device
      copyFromDevice({STRIDE, STRIDE, STRIDE}, in_ptr, in_ptr_dev);
      exchangeArr<3>(in_ptr, cart, bDecomp.rank_map, {N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});
      copyToDevice({STRIDE, STRIDE, STRIDE}, in_ptr_dev, in_ptr);

      dim3 block(N / TILE, N / TILE, N / TILE), thread(TILE, TILE, TILE);

      arr_kernel <<< block, thread >>> (in_ptr_dev, out_ptr_dev);
    };

    if (rank == 0)
      std::cout << "d3pt7 MPI decomp" << std::endl;
    int cnt;
    double total;

    total = time_mpi(arr_func, cnt, bDecomp);

    // Copy back
    copyFromDevice({STRIDE, STRIDE, STRIDE}, out_ptr, out_ptr_dev);

    if (rank == 0) {
      std::cout << "Arr: " << total << std::endl;

      std::cout << "calc " << total - (packtime + calltime + waittime) / cnt << std::endl;
      std::cout << "pack " << packtime / cnt << std::endl;
      std::cout << "call " << calltime / cnt << std::endl;
      std::cout << "wait " << waittime / cnt << std::endl;
      double perf = N / 1000.0;
      perf = perf * perf * perf / total;
      std::cout << "perf " << perf << " GStencil/s" << std::endl;
      std::cout << std::endl;
    }

    total = time_mpi(brick_func, cnt, bDecomp);

    if (rank == 0) {
      std::cout << "Bri: " << total << std::endl;

      std::cout << "calc " << total - (packtime + calltime + waittime) / cnt << std::endl;
      std::cout << "call " << calltime / cnt << std::endl;
      std::cout << "wait " << waittime / cnt << std::endl;

      double perf = N / 1000.0;
      perf = perf * perf * perf / total;
      std::cout << "perf " << perf << " GStencil/s" << std::endl;
    }

    if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
      throw std::runtime_error("result mismatch!");

    free(bInfo.adj);
    free(out_ptr);
    free(in_ptr);

    ((MEMFD *) bStorage.mmap_info)->cleanup();
    ((MEMFD *) bStorageOut.mmap_info)->cleanup();
  }

  MPI_Finalize();
  return 0;
}