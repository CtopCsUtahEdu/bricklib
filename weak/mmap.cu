//
// Created by Tuowen Zhao on 2/17/19.
// Experiments using unified memory (through ATS)
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
#include <multiarray.h>
#include <brickcompare.h>
#include "stencils/cudaarray.h"
#include "stencils/cudavfold.h"

#include <unistd.h>
#include <array-mpi.h>
#include "args.h"

typedef Brick<Dim<BDIM>, Dim<VFOLD>> Brick3D;

__global__ void
arr_kernel(bElem *in_ptr, bElem *out_ptr, unsigned *stride) {
  long k = PADDING + blockIdx.z * TILE + threadIdx.z;
  long j = PADDING + blockIdx.y * TILE + threadIdx.y;
  long i = PADDING + blockIdx.x * TILE + threadIdx.x;
  long pos = i + j * stride[1] + k * stride[2];
  ST_GPU;
}

__global__ void
brick_kernel(unsigned *grid, Brick3D in, Brick3D out, unsigned *stride) {
  unsigned bk = blockIdx.z;
  unsigned bj = blockIdx.y;
  unsigned bi = blockIdx.x;

  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];

  brick(ST_SCRTPT, VSVEC, (BDIM), (VFOLD), b);
}

int main(int argc, char **argv) {
  MPI_ITER = 100;
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided != MPI_THREAD_SERIALIZED) {
    MPI_Finalize();
    return 1;
  }

  MPI_Comm cart = parseArgs(argc, argv, "cuda-mmap");

  if (cart != MPI_COMM_NULL) {
    int rank;
    MPI_Comm_rank(cart, &rank);

    MEMFD::setup_prefix("mpi-main", rank);

    int prd[3] = {1, 1, 1};
    int coo[3];
    MPI_Cart_get(cart, 3, (int *) dim_size.data(), prd, coo);

    std::vector<long> stride(3), strideb(3), strideg(3);

    for (int i = 0; i < 3; ++i) {
      stride[i] = dom_size[i] + 2 * TILE + 2 * GZ;
      strideg[i] = dom_size[i] + 2 * TILE;
      strideb[i] = strideg[i] / TILE;
    }

    bElem *in_ptr = randomArray(stride);

    CUdevice device = 0;
    CUcontext pctx;
    gpuCheck((cudaError_t) cudaSetDevice(device));
    gpuCheck((cudaError_t) cuCtxCreate(&pctx, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device));

    BrickDecomp<3, BDIM> bDecomp(dom_size, GZ);
    bDecomp.comm = cart;
    populate(cart, bDecomp, 0, 1, coo);

    auto bSize = cal_size<BDIM>::value;
    bDecomp.initialize(skin3d_good);
    BrickInfo<3> bInfo = bDecomp.getBrickInfo();
    auto bStorage = bInfo.mmap_alloc(bSize);
    auto bStorageInt0 = bInfo.allocate(bSize);
    auto bStorageInt1 = bInfo.allocate(bSize);

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

    copyToBrick<3>(strideg, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

    bElem *out_ptr = zeroArray(stride);

    unsigned *arr_stride_dev = nullptr;
    {
      unsigned arr_stride_tmp[3];
      unsigned s = 1;
      for (int i = 0; i < 3; ++i) {
        arr_stride_tmp[i] = s;
        s *= stride[i];
      }
      copyToDevice({3}, arr_stride_dev, arr_stride_tmp);
    }

    bElem *out_ptr_dev = out_ptr;
    bElem *in_ptr_dev = in_ptr;

    size_t tsize = 0;
    for (int i = 0; i < bDecomp.ghost.size(); ++i)
      tsize += bDecomp.ghost[i].len * bStorage.step * sizeof(bElem) * 2;

    std::unordered_map<uint64_t, MPI_Datatype> stypemap;
    std::unordered_map<uint64_t, MPI_Datatype> rtypemap;
    exchangeArrPrepareTypes<3>(stypemap, rtypemap, {dom_size[0], dom_size[1], dom_size[2]},
                               {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});

    {
      long arr_size = stride[0] * stride[1] * stride[2] * sizeof(bElem);
      gpuCheck(cudaMemAdvise(in_ptr, arr_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      cudaMemPrefetchAsync(in_ptr, arr_size, device);
      gpuCheck(cudaMemAdvise(out_ptr, arr_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      cudaMemPrefetchAsync(out_ptr, arr_size, device);
    }

    auto arr_func = [&]() -> void {
      float elapsed;
      cudaEvent_t c_0, c_1;
      cudaEventCreate(&c_0);
      cudaEventCreate(&c_1);
#ifdef USE_TYPES
      exchangeArrTypes<3>(in_ptr_dev, cart, bDecomp.rank_map, stypemap, rtypemap);
#else
      exchangeArr<3>(in_ptr, cart, bDecomp.rank_map, {dom_size[0], dom_size[1], dom_size[2]},
                     {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});
#endif

      cudaEventRecord(c_0);
      dim3 block(strideb[0], strideb[1], strideb[2]), thread(TILE, TILE, TILE);
      for (int i = 0; i < ST_ITER / 2; ++i) {
        arr_kernel << < block, thread >> > (in_ptr_dev, out_ptr_dev, arr_stride_dev);
        arr_kernel << < block, thread >> > (out_ptr_dev, in_ptr_dev, arr_stride_dev);
      }
      cudaEventRecord(c_1);
      cudaEventSynchronize(c_1);
      cudaEventElapsedTime(&elapsed, c_0, c_1);
      calctime += elapsed / 1000.0;
    };

    if (rank == 0)
      std::cout << "d3pt7 MPI decomp" << std::endl;
    int cnt;
    double total;

    total = time_mpi(arr_func, cnt, bDecomp);
    cnt *= ST_ITER;

    {
      mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
      mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
      mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
      mpi_stats mspd_s = mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
      mpi_stats move_s = mpi_statistics(movetime / cnt, MPI_COMM_WORLD);
      mpi_stats pack_s = mpi_statistics(packtime / cnt, MPI_COMM_WORLD);
      mpi_stats size_s = mpi_statistics((double) tsize * 1.0e-6, MPI_COMM_WORLD);

      if (rank == 0) {
        total = calc_s.avg + call_s.avg + wait_s.avg + move_s.avg + pack_s.avg;

        std::cout << "Arr: " << total << std::endl;
        std::cout << "calc " << calc_s << std::endl;
        std::cout << "pack " << pack_s << std::endl;
        std::cout << "move " << move_s << std::endl;
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

    // setup brick on device
    BrickInfo<3> *bInfo_dev;
    auto _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
    {
      unsigned size = sizeof(BrickInfo<3>);
      cudaMalloc(&bInfo_dev, size);
      cudaMemcpy(bInfo_dev, &_bInfo_dev, size, cudaMemcpyHostToDevice);
    }

    BrickStorage bStorageInt0_dev = movBrickStorage(bStorageInt0, cudaMemcpyHostToDevice);
    BrickStorage bStorageInt1_dev = movBrickStorage(bStorageInt1, cudaMemcpyHostToDevice);

    Brick3D bIn_dev(bInfo_dev, bStorage, 0);
    Brick3D bInt0_dev(bInfo_dev, bStorageInt0_dev, 0);
    Brick3D bInt1_dev(bInfo_dev, bStorageInt1_dev, 0);

    unsigned *grid_dev_ptr = nullptr;
    copyToDevice(strideb, grid_dev_ptr, grid_ptr);

    unsigned *grid_stride_dev = nullptr;
    {
      unsigned grid_stride_tmp[3];
      for (int i = 0; i < 3; ++i)
        grid_stride_tmp[i] = strideb[i];
      copyToDevice({3}, grid_stride_dev, grid_stride_tmp);
    }

#ifndef DECOMP_PAGEUNALIGN
    ExchangeView ev = bDecomp.exchangeView(bStorage);
#endif

    gpuCheck(cudaMemAdvise(bStorage.dat.get(),
                            bStorage.step * bDecomp.sep_pos[2] * sizeof(bElem), cudaMemAdviseSetPreferredLocation,
                            device));

    cudaMemPrefetchAsync(bStorage.dat.get(), bStorage.step * bDecomp.sep_pos[2] * sizeof(bElem), device);

    cudaMemPrefetchAsync(grid_ptr, STRIDEB * STRIDEB * STRIDEB * sizeof(unsigned), device);

    auto brick_func = [&]() -> void {
      float elapsed;
      cudaEvent_t c_0, c_1;
      cudaEventCreate(&c_0);
      cudaEventCreate(&c_1);

#ifdef DECOMP_PAGEUNALIGN
      bDecomp.exchange(bStorage);
#else
      ev.exchange();
#endif

      dim3 block(strideb[0], strideb[1], strideb[2]), thread(32);
      cudaEventRecord(c_0);
      brick_kernel << < block, thread >> > (grid_dev_ptr, bIn_dev, bInt0_dev, grid_stride_dev);
      for (int i = 0; i < ST_ITER / 2 - 1; ++i) {
        brick_kernel << < block, thread >> > (grid_dev_ptr, bInt0_dev, bInt1_dev, grid_stride_dev);
        brick_kernel << < block, thread >> > (grid_dev_ptr, bInt1_dev, bInt0_dev, grid_stride_dev);
      }
      brick_kernel << < block, thread >> > (grid_dev_ptr, bInt0_dev, bIn_dev, grid_stride_dev);
      cudaEventRecord(c_1);
      cudaEventSynchronize(c_1);
      cudaEventElapsedTime(&elapsed, c_0, c_1);
      calctime += elapsed / 1000.0;
    };

    total = time_mpi(brick_func, cnt, bDecomp);
    cnt *= ST_ITER;

    {
      mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
      mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
      mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
      mpi_stats mspd_s = mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
      mpi_stats size_s = mpi_statistics((double) tsize * 1.0e-6, MPI_COMM_WORLD);
#ifndef DECOMP_PAGEUNALIGN
      size_t opt_size = 0;
      for (auto s: ev.seclen)
        opt_size += s * 2;
      mpi_stats opt_size_s = mpi_statistics((double) opt_size * 1.0e-6, MPI_COMM_WORLD);
#endif

      mpi_stats move_s = mpi_statistics(movetime / cnt, MPI_COMM_WORLD);

      if (rank == 0) {
        total = calc_s.avg + call_s.avg + wait_s.avg + move_s.avg;

        std::cout << "Bri: " << total << std::endl;
        std::cout << "calc " << calc_s << std::endl;
        std::cout << "move " << move_s << std::endl;
        std::cout << "call " << call_s << std::endl;
        std::cout << "wait " << wait_s << std::endl;
        std::cout << "  | MPI size (MB): " << size_s << std::endl;
#ifndef DECOMP_PAGEUNALIGN
        std::cout << "  | Opt MPI size (MB): " << opt_size_s << std::endl;
#endif
        std::cout << "  | MPI speed (GB/s): " << mspd_s << std::endl;

        double perf = (double) tot_elems * 1.0e-9;
        perf = perf / total;
        std::cout << "perf " << perf << " GStencil/s" << std::endl;
      }
    }

    if (!compareBrick<3>({dom_size[0], dom_size[1], dom_size[2]}, {PADDING, PADDING, PADDING},
                         {GZ, GZ, GZ}, in_ptr, grid_ptr, bIn))
      std::cout << "result mismatch!" << std::endl;

    free(bInfo.adj);
    free(out_ptr);
    free(in_ptr);

    ((MEMFD *) bStorage.mmap_info)->cleanup();
  }

  MPI_Finalize();
  return 0;
}
