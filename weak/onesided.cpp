//
// Created by Tuowen Zhao on 2/17/19.
//

#include <mpi.h>
#include <iostream>
#include <brick.h>
#include <brick-mpi.h>
#include <array-mpi.h>
#include <bricksetup.h>
#include "stencils/stencils.h"
#include "stencils/fake.h"

#include "bitset.h"
#include <multiarray.h>
#include <brickcompare.h>

#include <unistd.h>

#include "stencils/cpuvfold.h"

int MPI_ITER = 25;

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

    for (long i = 1; i < STRIDEB - 1; ++i)
      for (long j = 1; j < STRIDEB - 1; ++j)
        for (long k = 1; k < STRIDEB - 1; ++k) {
          auto l = grid[i][j][k];
          for (long id = 0; id < 27; ++id)
            if (bInfo.adj[bInfo.adj[l][id]][26 - id] != l)
              throw std::runtime_error("err");
        }

    MPI_Win win;
    {
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info, "no_locks", "true");
      MPI_Win_create(bStorage.dat, bStorage.chunks * bStorage.step * sizeof(bElem), 1, info, cart, &win);
    }

    Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
    Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorageOut, 0);

    copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

    bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});

    auto arr_in = (bElem (*)[STRIDE][STRIDE]) in_ptr;
    auto arr_out = (bElem (*)[STRIDE][STRIDE]) out_ptr;

    auto brick_stencil = [&](Brick<Dim<BDIM>, Dim<VFOLD>> &out, Brick<Dim<BDIM>, Dim<VFOLD>> &in) -> void {
      _PARFOR
      for (long tk = 0; tk < STRIDEB; ++tk)
        for (long tj = 0; tj < STRIDEB; ++tj)
          for (long ti = 0; ti < STRIDEB; ++ti) {
            unsigned b = grid[tk][tj][ti];
            brick("../stencils/mpi7pt.py", VSVEC, (BDIM), (VFOLD), b);
          }
    };

    auto brick_func = [&]() -> void {
      bDecomp.exchange(bStorage, win);

      for (int i = 0; i < 4; ++i) {
        brick_stencil(bOut, bIn);
        brick_stencil(bIn, bOut);
      }
    };

    auto array_stencil = [&](bElem (*arrOut)[STRIDE][STRIDE], bElem (*arrIn)[STRIDE][STRIDE]) -> void {
      _TILEFOR arrOut[k][j][i] = (arrIn[k + 1][j][i] + arrIn[k - 1][j][i] +
                                  arrIn[k][j + 1][i] + arrIn[k][j - 1][i] +
                                  arrIn[k][j][i + 1] + arrIn[k][j][i - 1]) * MPI_BETA +
                                 arrIn[k][j][i] * MPI_ALPHA;
    };

    auto arr_func = [&]() -> void {
      exchangeArr<3>(in_ptr, cart, bDecomp.rank_map, {N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});
      for (int i = 0; i < 4; ++i) {
        array_stencil(arr_out, arr_in);
        array_stencil(arr_in, arr_out);
      }
    };

    if (rank == 0)
      std::cout << "d3pt7 MPI decomp" << std::endl;
    int cnt;
    double total;

    total = time_mpi(arr_func, cnt, bDecomp);

    if (rank == 0) {
      std::cout << "Arr: " << total << std::endl;

      std::cout << "calc " << total - (packtime + calltime + waittime) / cnt << std::endl;
      std::cout << "pack " << packtime / cnt << std::endl;
      std::cout << "call " << calltime / cnt << std::endl;
      std::cout << "wait " << waittime / cnt << std::endl;
      double perf = N / 1000.0;
      perf = perf * perf * perf * 8 / total;
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
      perf = perf * perf * perf * 8 / total;
      std::cout << "perf " << perf << " GStencil/s" << std::endl;
    }

    if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
      throw std::runtime_error("result mismatch!");

    MPI_Win_free(&win);

    free(bInfo.adj);
    free(out_ptr);
    free(in_ptr);

    ((MEMFD *) bStorage.mmap_info)->cleanup();
    ((MEMFD *) bStorageOut.mmap_info)->cleanup();
  }

  MPI_Finalize();
  return 0;
}
