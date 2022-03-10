#include "stencils/fake.h"
#include "stencils/stencils.h"
#include <brick-hip.h>
#include <brick-gpu.h>
#include <brick-mpi.h>
#include <brick.h>
#include <bricksetup.h>
#include <iostream>
#include <mpi.h>

#include "bitset.h"
#include "stencils/gpuarray.h"
#include "stencils/hipvfold.h"
#include <brickcompare.h>
#include <multiarray.h>

#include <array-mpi.h>
#include <unistd.h>

#include "args.h"

typedef Brick<Dim<BDIM>, Dim<VFOLD>> Brick3D;

__global__ void arr_kernel(bElem *in_ptr, bElem *out_ptr, unsigned *stride) {
  long k = PADDING + blockIdx.z * TILE + threadIdx.z;
  long j = PADDING + blockIdx.y * TILE + threadIdx.y;
  long i = PADDING + blockIdx.x * TILE + threadIdx.x;
  long pos = i + j * stride[1] + k * stride[2];
  ST_GPU;
}

__global__ void brick_kernel(unsigned *grid, Brick3D in, Brick3D out, unsigned *stride) {
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

  MPI_Comm cart = parseArgs(argc, argv, "hip");

  if (cart != MPI_COMM_NULL) {

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MEMFD::setup_prefix("mpi-main", rank);

    int prd[3] = {1, 1, 1};
    int coo[3];
    MPI_Cart_get(cart, 3, (int *)dim_size.data(), prd, coo);

    std::vector<long> stride(3), strideb(3), strideg(3);

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
#ifdef DECOMP_PAGEUNALIGN
    auto bStorage = bInfo.allocate(bSize);
    auto bStorageOut = bInfo.allocate(bSize);
#else
    auto bStorage = bInfo.mmap_alloc(bSize);
    auto bStorageOut = bInfo.mmap_alloc(bSize);
#endif

    auto grid_ptr = (unsigned *)malloc(sizeof(unsigned) * strideb[2] * strideb[1] * strideb[0]);
    auto grid = (unsigned(*)[strideb[1]][strideb[0]])grid_ptr;

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

    bElem *in_ptr_dev = nullptr;
    bElem *out_ptr_dev = nullptr;

    copyToDevice(stride, in_ptr_dev, in_ptr);
    copyToDevice(stride, out_ptr_dev, out_ptr);

    size_t tsize = 0;
    for (auto &g : bDecomp.ghost)
      tsize += g.len * bStorage.step * sizeof(bElem) * 2;

    std::unordered_map<uint64_t, MPI_Datatype> stypemap;
    std::unordered_map<uint64_t, MPI_Datatype> rtypemap;
    exchangeArrPrepareTypes<3>(stypemap, rtypemap, {dom_size[0], dom_size[1], dom_size[2]},
                               {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});
    auto arr_func = [&]() -> void {
      float elapsed;
      hipEvent_t c_0, c_1;
      hipEventCreate(&c_0);
      hipEventCreate(&c_1);
#if !defined(CUDA_AWARE) || !defined(USE_TYPES)
      // Copy everything back from device
      double st = omp_get_wtime();
      copyFromDevice(stride, in_ptr, in_ptr_dev);
      movetime += omp_get_wtime() - st;
      exchangeArr<3>(in_ptr, cart, bDecomp.rank_map, {dom_size[0], dom_size[1], dom_size[2]},
                     {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});
      st = omp_get_wtime();
      copyToDevice(stride, in_ptr_dev, in_ptr);
      movetime += omp_get_wtime() - st;
#else
      exchangeArrTypes<3>(in_ptr_dev, cart, bDecomp.rank_map, stypemap, rtypemap);
#endif
      hipEventRecord(c_0);
      dim3 block(strideb[0], strideb[1], strideb[2]), thread(TILE, TILE, TILE);
      for (int i = 0; i < ST_ITER / 2; ++i) {
        arr_kernel<<<block, thread>>>(in_ptr_dev, out_ptr_dev, arr_stride_dev);
        arr_kernel<<<block, thread>>>(out_ptr_dev, in_ptr_dev, arr_stride_dev);
      }
      hipEventRecord(c_1);
      hipEventSynchronize(c_1);
      hipEventElapsedTime(&elapsed, c_0, c_1);
      calctime += elapsed / 1000.0;
    };

    if (rank == 0)
      std::cout << "d3pt7 MPI decomp" << std::endl;
    int cnt;
    double total;

    total = time_mpi(arr_func, cnt, bDecomp);
    cnt *= ST_ITER;

    // Copy back
    copyFromDevice(stride, out_ptr, out_ptr_dev);

    hipFree(in_ptr_dev);
    hipFree(out_ptr_dev);
    hipFree(arr_stride_dev);
    {
      mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
      mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
      mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
      mpi_stats mspd_s =
          mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
      mpi_stats move_s = mpi_statistics(movetime / cnt, MPI_COMM_WORLD);
      mpi_stats pack_s = mpi_statistics(packtime / cnt, MPI_COMM_WORLD);
      mpi_stats size_s = mpi_statistics((double)tsize * 1.0e-6, MPI_COMM_WORLD);

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

        double perf = (double)tot_elems * 1.0e-9;
        perf = perf / total;
        std::cout << "perf " << perf << " GStencil/s" << std::endl;
        std::cout << std::endl;
      }
    }

    // setup brick on device
    BrickInfo<3> *bInfo_dev;
    auto _bInfo_dev = movBrickInfo(bInfo, hipMemcpyHostToDevice);
    {
      unsigned size = sizeof(BrickInfo<3>);
      hipMalloc(&bInfo_dev, size);
      hipMemcpy(bInfo_dev, &_bInfo_dev, size, hipMemcpyHostToDevice);
    }

    BrickStorage bStorage_dev = movBrickStorage(bStorage, hipMemcpyHostToDevice);
    BrickStorage bStorageOut_dev = movBrickStorage(bStorageOut, hipMemcpyHostToDevice);

    Brick3D bIn_dev(bInfo_dev, bStorage_dev, 0);
    Brick3D bOut_dev(bInfo_dev, bStorageOut_dev, 0);

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

    auto brick_func = [&]() -> void {
      float elapsed;
      hipEvent_t c_0, c_1;
      hipEventCreate(&c_0);
      hipEventCreate(&c_1);
#ifndef CUDA_AWARE
      {
        double t_a = omp_get_wtime();
        hipMemcpy(bStorage.dat.get() + bStorage.step * bDecomp.sep_pos[0],
                  bStorage_dev.dat.get() + bStorage.step * bDecomp.sep_pos[0],
                  bStorage.step * (bDecomp.sep_pos[1] - bDecomp.sep_pos[0]) * sizeof(bElem),
                  hipMemcpyDeviceToHost);
        double t_b = omp_get_wtime();
        movetime += t_b - t_a;
#ifdef DECOMP_PAGEUNALIGN
        bDecomp.exchange(bStorage);
#else
        ev.exchange();
#endif
        t_a = omp_get_wtime();
        hipMemcpy(bStorage_dev.dat.get() + bStorage.step * bDecomp.sep_pos[1],
                  bStorage.dat.get() + bStorage.step * bDecomp.sep_pos[1],
                  bStorage.step * (bDecomp.sep_pos[2] - bDecomp.sep_pos[1]) * sizeof(bElem),
                  hipMemcpyHostToDevice);
        t_b = omp_get_wtime();
        movetime += t_b - t_a;
      }
#else
      bDecomp.exchange(bStorage_dev);
#endif

      dim3 block(strideb[0], strideb[1], strideb[2]), thread(64);
      hipEventRecord(c_0);
      for (int i = 0; i < ST_ITER / 2; ++i) {
        brick_kernel<<<block, thread>>>(grid_dev_ptr, bIn_dev, bOut_dev, grid_stride_dev);
        brick_kernel<<<block, thread>>>(grid_dev_ptr, bOut_dev, bIn_dev, grid_stride_dev);
      }
      hipEventRecord(c_1);
      hipEventSynchronize(c_1);
      hipEventElapsedTime(&elapsed, c_0, c_1);
      calctime += elapsed / 1000.0;
    };

    total = time_mpi(brick_func, cnt, bDecomp);
    cnt *= ST_ITER;

    // Copy back
    hipMemcpy(bStorageOut.dat.get(), bStorageOut_dev.dat.get(),
              bStorageOut.step * bStorageOut.chunks * sizeof(bElem), hipMemcpyDeviceToHost);
    {
      mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
      mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
      mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
      mpi_stats mspd_s =
          mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
      mpi_stats size_s = mpi_statistics((double)tsize * 1.0e-6, MPI_COMM_WORLD);
      mpi_stats move_s = mpi_statistics(movetime / cnt, MPI_COMM_WORLD);

      if (rank == 0) {
        total = calc_s.avg + call_s.avg + wait_s.avg + move_s.avg;

        std::cout << "Bri: " << total << std::endl;
        std::cout << "calc " << calc_s << std::endl;
        std::cout << "move " << move_s << std::endl;
        std::cout << "call " << call_s << std::endl;
        std::cout << "wait " << wait_s << std::endl;
        std::cout << "  | MPI size (MB): " << size_s << std::endl;
        std::cout << "  | MPI speed (GB/s): " << mspd_s << std::endl;

        double perf = (double)tot_elems * 1.0e-9;
        perf = perf / total;
        std::cout << "perf " << perf << " GStencil/s" << std::endl;
      }
    }

    if (!compareBrick<3>({dom_size[0], dom_size[1], dom_size[2]}, {PADDING, PADDING, PADDING},
                         {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
      std::cout << "result mismatch!" << std::endl;

    free(bInfo.adj);

    free(out_ptr);
    free(in_ptr);

#ifndef DECOMP_PAGEUNALIGN
    ((MEMFD *)bStorage.mmap_info)->cleanup();
    ((MEMFD *)bStorageOut.mmap_info)->cleanup();
#endif
  }

  MPI_Finalize();
  return 0;
}
