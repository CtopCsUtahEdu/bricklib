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
#include "args.h"

typedef Brick<Dim<BDIM>, Dim<VFOLD>> Brick3D;
std::vector<long> stride(3), strideb(3), strideg(3);

void brick_stencil(Brick3D &out, Brick3D &in, unsigned *grid_ptr, long skip) {
  auto grid = (unsigned (*)[strideb[1]][strideb[0]]) grid_ptr;
  long s21 = strideb[2] - skip, s11 = strideb[1] - skip;
#pragma omp parallel for collapse(2)
  for (long tk = skip; tk < s21; ++tk)
    for (long tj = skip; tj < s11; ++tj)
      for (long ti = skip; ti < strideb[0] - skip; ++ti) {
        unsigned b = grid[tk][tj][ti];
        brick(ST_SCRTPT, VSVEC, (BDIM), (VFOLD), b);
      }
};

int main(int argc, char **argv) {
  MPI_ITER = 25;
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided != MPI_THREAD_SERIALIZED) {
    MPI_Finalize();
    return 1;
  }

  MPI_Comm cart = parseArgs(argc, argv, "cpu");

  if (cart != MPI_COMM_NULL) {
    int size, rank;

    MPI_Comm_size(cart, &size);
    MPI_Comm_rank(cart, &rank);

    MEMFD::setup_prefix("mpi-main", rank);

    int prd[3] = {1, 1, 1};
    int coo[3];
    MPI_Cart_get(cart, 3, (int *) dim_size.data(), prd, coo);

    for (int i = 0; i < 3; ++i) {
      stride[i] = dom_size[i] + 2 * TILE + 2 * GZ;
      strideg[i] = dom_size[i] + 2 * TILE;
      strideb[i] = strideg[i] / TILE;
    }

    bElem *in_ptr = randomArray(stride);

    BrickDecomp<3, BDIM> bDecomp(dom_size, GZ);
    bDecomp.comm = cart;
    populate(cart, bDecomp, 0, 1, coo);

    auto bSize = cal_size<BDIM>::value;
    bDecomp.initialize(skin3d_good);
    BrickInfo<3> bInfo = bDecomp.getBrickInfo();
    auto bStorage = bInfo.mmap_alloc(bSize);
    auto bStorageOut = bInfo.mmap_alloc(bSize);

    auto grid_ptr = (unsigned *) malloc(sizeof(unsigned) * strideb[2] * strideb[1] * strideb[0]);
    auto grid = (unsigned (*)[strideb[1]][strideb[0]]) grid_ptr;

    for (long k = 0; k < strideb[2]; ++k)
      for (long j = 0; j < strideb[1]; ++j)
        for (long i = 0; i < strideb[0]; ++i)
          grid[k][j][i] = bDecomp[k][j][i];

    for (long k = 1; k < strideb[2] - 1; ++k)
      for (long j = 1; j < strideb[1] - 1; ++j)
        for (long i = 1; i < strideb[0] - 1; ++i) {
          auto l = grid[k][j][i];
          for (long id = 0; id < 27; ++id)
            if (bInfo.adj[bInfo.adj[l][id]][26 - id] != l)
              throw std::runtime_error("err");
        }

    Brick3D bIn(&bInfo, bStorage, 0);
    Brick3D bOut(&bInfo, bStorageOut, 0);

    copyToBrick<3>(strideg, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

    bElem *out_ptr = zeroArray({stride[0], stride[1], stride[2]});

    auto arr_in = (bElem (*)[stride[1]][stride[0]]) in_ptr;
    auto arr_out = (bElem (*)[stride[1]][stride[0]]) out_ptr;

#ifndef DECOMP_PAGEUNALIGN
#ifdef MWAITS
    auto mwev = bDecomp.multiStageExchangeView(bStorage);
    auto mwev_out = bDecomp.multiStageExchangeView(bStorageOut);
#else
    ExchangeView ev = bDecomp.exchangeView(bStorage);
    auto ev_out = bDecomp.exchangeView(bStorageOut);
#endif
#endif

    auto array_stencil = [&](bElem *arrOut_ptr, bElem *arrIn_ptr, long skip) -> void {
      auto arrIn = (bElem (*)[stride[1]][stride[0]]) arrIn_ptr;
      auto arrOut = (bElem (*)[stride[1]][stride[0]]) arrOut_ptr;
      long skip2 = (skip / TILE) * TILE;
      long s20 = PADDING + skip, s21 = PADDING + strideg[2];
      long s10 = PADDING + skip, s11 = PADDING + strideg[1];
#pragma omp parallel for collapse(2)
      for (long tk = s20; tk < s21; tk += TILE)
        for (long tj = s10; tj < s11; tj += TILE)
          for (long ti = PADDING + skip2; ti < PADDING + strideg[0] - skip2; ti += TILE)
            for (long k = tk; k < tk + TILE; ++k)
              for (long j = tj; j < tj + TILE; ++j)
#pragma omp simd
                  for (long i = ti; i < ti + TILE; ++i)
                    ST_CPU;
    };

    std::unordered_map<uint64_t, MPI_Datatype> stypemap;
    std::unordered_map<uint64_t, MPI_Datatype> rtypemap;
    exchangeArrPrepareTypes<3>(stypemap, rtypemap, {dom_size[0], dom_size[1], dom_size[2]},
                        {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});
    auto arr_func = [&]() -> void {
#ifdef USE_TYPES
      exchangeArrTypes<3>(in_ptr, cart, bDecomp.rank_map, stypemap, rtypemap);
#else
      exchangeArr<3>(in_ptr, cart, bDecomp.rank_map, {dom_size[0], dom_size[1], dom_size[2]},
                     {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});
#endif
#ifdef MPI_49PT
      double t_a = omp_get_wtime();
      array_stencil(out_ptr, in_ptr, TILE);
      double t_b = omp_get_wtime();
      calctime += t_b - t_a;
#ifdef USE_TYPES
      exchangeArrTypes<3>(out_ptr, cart, bDecomp.rank_map, stypemap, rtypemap);
#else
      exchangeArr<3>(out_ptr, cart, bDecomp.rank_map, {dom_size[0], dom_size[1], dom_size[2]},
                     {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});
#endif
      t_a = omp_get_wtime();
      array_stencil(in_ptr, out_ptr, TILE);
      t_b = omp_get_wtime();
      calctime += t_b - t_a;
#else
      double t_a = omp_get_wtime();
      array_stencil(out_ptr, in_ptr, 0);
      for (int i = 0; i < ST_ITER / 2 - 1; ++i) {
        array_stencil(in_ptr, out_ptr, 0);
        array_stencil(out_ptr, in_ptr, 0);
      }
      array_stencil(in_ptr, out_ptr, TILE);
      double t_b = omp_get_wtime();
      calctime += t_b - t_a;
#endif
    };

    auto brick_func = [&]() -> void {
#ifndef DECOMP_PAGEUNALIGN
#ifdef MWAITS
      mwev.exchange();
#else
      ev.exchange();
#endif
#else
      bDecomp.exchange(bStorage);
#endif

#ifdef MPI_49PT
      double t_a = omp_get_wtime();
      brick_stencil(bOut, bIn, grid_ptr, 1);
      double t_b = omp_get_wtime();
      calctime += t_b - t_a;

#ifndef DECOMP_PAGEUNALIGN
#ifdef MWAITS
      mwev_out.exchange();
#else
      ev_out.exchange();
#endif
#else
      bDecomp.exchange(bStorageOut);
#endif
      t_a = omp_get_wtime();
      brick_stencil(bIn, bOut, grid_ptr, 1);
      t_b = omp_get_wtime();
      calctime += t_b - t_a;
#else
      double t_a = omp_get_wtime();
      brick_stencil(bOut, bIn, grid_ptr, 0);
      for (int i = 0; i < ST_ITER / 2 - 1; ++i) {
        brick_stencil(bIn, bOut, grid_ptr, 0);
        brick_stencil(bOut, bIn, grid_ptr, 0);
      }
      brick_stencil(bIn, bOut, grid_ptr, 1);
      double t_b = omp_get_wtime();
      calctime += t_b - t_a;
#endif
    };

    if (rank == 0)
      std::cout << "d3pt7 MPI decomp" << std::endl;
    int cnt;
    double total;

    size_t tsize = 0;
    for (auto g: bDecomp.ghost)
      tsize += g.len * bStorage.step * sizeof(bElem) * 2;

    {
      total = time_mpi(arr_func, cnt, bDecomp);
      cnt *= ST_ITER;
      mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
      mpi_stats pack_s = mpi_statistics(packtime / cnt, MPI_COMM_WORLD);
      mpi_stats pspd_s = mpi_statistics(tsize / 1.0e9 / packtime * cnt, MPI_COMM_WORLD);
      mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
      mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
      mpi_stats mspd_s = mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
      mpi_stats size_s = mpi_statistics((double) tsize * 1.0e-6, MPI_COMM_WORLD);
      total = calc_s.avg + wait_s.avg + call_s.avg + pack_s.avg;

      if (rank == 0) {
        std::cout << "Arr: " << total << std::endl;

        std::cout << "calc " << calc_s << std::endl;
        std::cout << "pack " << pack_s << std::endl;
        std::cout << "  | Pack speed (GB/s): " << pspd_s << std::endl;
        std::cout << "call " << call_s << std::endl;
        std::cout << "wait " << wait_s << std::endl;
        std::cout << "  | MPI size (MB): " << size_s << std::endl;
        std::cout << "  | MPI speed (GB/s): " << mspd_s << std::endl;

        double perf = (double) tot_elems * 1.0e-9;
        perf = perf / total;
        std::cout << "perf " << perf << " GStencil/s" << std::endl;
        std::cout << std::endl;
      }
    }

    {
      total = time_mpi(brick_func, cnt, bDecomp);
      cnt *= ST_ITER;
      mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
      mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
      mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
      mpi_stats mspd_s = mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
      mpi_stats size_s = mpi_statistics((double) tsize * 1.0e-6, MPI_COMM_WORLD);
      total = calc_s.avg + wait_s.avg + call_s.avg;

      if (rank == 0) {
        std::cout << "Bri: " << total << std::endl;

        std::cout << "calc " << calc_s << std::endl;
        std::cout << "call " << call_s << std::endl;
        std::cout << "wait " << wait_s << std::endl;
        std::cout << "  | MPI size (MB): " << size_s << std::endl;
        std::cout << "  | MPI speed (GB/s): " << mspd_s << std::endl;

        double perf = (double) tot_elems * 1.0e-9;
        perf = perf / total;
        std::cout << "perf " << perf << " GStencil/s" << std::endl;
        std::cout << "Total of " << bDecomp.ghost.size() << " parts" << std::endl;
      }
    }

    if (!compareBrick<3>({dom_size[0], dom_size[1], dom_size[2]}, {PADDING, PADDING, PADDING},
                         {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
      std::cout << "result mismatch!" << std::endl;

    free(bInfo.adj);
    free(out_ptr);
    free(in_ptr);

    ((MEMFD *) bStorage.mmap_info)->cleanup();
    ((MEMFD *) bStorageOut.mmap_info)->cleanup();
  }

  MPI_Finalize();
  return 0;
}
