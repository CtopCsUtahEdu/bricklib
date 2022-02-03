//
// Created by Tuowen Zhao on 2/17/19.
//

#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <brick.h>
#include <brick-mpi.h>
#include <bricksetup.h>
#include <brick-cuda.h>
#include <zmort.h>
#include "stencils/fake.h"

#include "bitset.h"
#include <multiarray.h>
#include <brickcompare.h>
#include "stencils/cudaarray.h"

#include <unistd.h>
#include <array-mpi.h>

#define GZ 8
#define TILE 8
#define BDIM 8,8,8

#include "stencils/cudavfold.h"
#include "args.h"

#define STRIDE (sdom_size + 2 * GZ)
#define STRIDEB (STRIDE / TILE)

typedef Brick<Dim<BDIM>, Dim<VFOLD>> Brick3D;

struct Subdomain {
  ZMORT zmort;
  std::vector<BrickStorage> storage;
  std::vector<BrickStorage> storage_dev;
  std::vector<Brick3D> brick;

  void cleanup() {
    for (auto bStorage: storage)
      if (bStorage.mmap_info != nullptr)
        ((MEMFD *) bStorage.mmap_info)->cleanup();
  }
};

struct RegionSpec {
  // This always specifies the sender's zmort/neighbor
  ZMORT zmort;
  unsigned neighbor;
  unsigned skin_st;
  BrickStorage *storage, *storage_dev;
  long pos; // Position within the subdomain
  long len; // Length of the regions
  long offset; // Offset in the send/recv buffer
  int rank;
};

struct RegionLink {
  // This always link two device memory together and uses DeviceToDevice Memcpy
  bElem *from, *to;
  size_t len;
};

struct ExView {
  // This specifies the regions to exchange that created on GPU
  bElem *reg;
#ifndef CUDA_AWARE
  bElem *host_reg;
#endif
  size_t len;
  int rank;
};

__global__ void cudaCopy(RegionLink *rl) {
  long i = blockIdx.x;
  long len = rl[i].len / sizeof(uint64_t);
  uint64_t *from = (uint64_t *) rl[i].from;
  uint64_t *to = (uint64_t *) rl[i].to;
  for (long x = threadIdx.x; x < len; x += blockDim.x)
    to[x] = from[x];
}

__global__ void
brick_kernel(unsigned *grid_ptr, unsigned strideb, Brick3D *barr, int outIdx, int inIdx) {
  unsigned s = blockIdx.x / strideb;

  unsigned bk = blockIdx.z;
  unsigned bj = blockIdx.y;
  unsigned bi = blockIdx.x % strideb;

  unsigned b = grid_ptr[bi + (bj + bk * strideb) * strideb];

  Brick3D &out = barr[s * 2 + outIdx];
  Brick3D &in = barr[s * 2 + inIdx];

  brick(ST_SCRTPT, VSVEC, (BDIM), (VFOLD), b);
}

// When it only contains a single domain remove the brick pointer to improve performance
__global__ void
brick_kernel_single_domain(unsigned *grid, Brick3D in, Brick3D out, unsigned strideb) {
  unsigned bk = blockIdx.z;
  unsigned bj = blockIdx.y;
  unsigned bi = blockIdx.x;

  unsigned b = grid[bi + (bj + bk * strideb) * strideb];

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

  parseArgs(argc, argv, "cuda");

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MEMFD::setup_prefix("mpi-main", rank);

  BrickDecomp<3, BDIM> bDecomp({sdom_size, sdom_size, sdom_size}, GZ);
  auto bSize = cal_size<BDIM>::value;
  bDecomp.initialize(skin3d_good);
  BrickInfo<3> bInfo = bDecomp.getBrickInfo();

  // Create subdomains
  // This requires the number of subdomains on each dimension is a perfect 2-power
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

  std::vector<Subdomain> subdomains(mysec_r - mysec_l);

  for (unsigned long sec_id = mysec_l; sec_id < mysec_r; ++sec_id) {
    ZMORT zid(sec_id, 3);
    unsigned long idx = sec_id - mysec_l;
    subdomains[idx].zmort = zid;
    // bIn
    auto bStorage = bInfo.allocate(bSize);
    subdomains[idx].brick.emplace_back(&bInfo, bStorage, 0);
    subdomains[idx].storage.push_back(bStorage);

    // Initialize with random numbers
    bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
    copyToBrick<3>({STRIDE, STRIDE, STRIDE}, {0, 0, 0}, {0, 0, 0}, in_ptr, grid_ptr, subdomains[idx].brick.back());
    subdomains[idx].storage_dev.push_back(movBrickStorage(bStorage, cudaMemcpyHostToDevice));
    free(in_ptr);

    // bOut
    bStorage = bInfo.allocate(bSize);
    subdomains[idx].brick.emplace_back(&bInfo, bStorage, 0);
    subdomains[idx].storage.push_back(bStorage);
    subdomains[idx].storage_dev.push_back(movBrickStorage(bStorage, cudaMemcpyHostToDevice));
  }

  std::vector<BitSet> neighbors;
  allneighbors(0, 1, 3, neighbors);

  // Starts linking all internal/external regions
  // Unordered map specified the dest/src rank and a bunch of regions
  std::unordered_map<int, std::vector<RegionSpec>> sendReg, recvReg;
  std::vector<RegionLink> links;

  for (auto &s: subdomains) {
    for (int n_idx = 0; n_idx < neighbors.size(); ++n_idx) {
      BitSet n = neighbors[n_idx];
      if (n.set == 0)
        continue;
      // get that neighbor's zmort
      ZMORT zmort = s.zmort;
      int dst, sub;
      getrank(n, zmort, dst, sub);
      // Process ghost/skins
      for (int i = 0; i < bDecomp.ghost.size(); ++i)
        if (n.set == bDecomp.ghost[i].neighbor.set) {
          size_t len = bDecomp.ghost[i].len * s.storage[0].step * sizeof(bElem);
          if (dst == rank) {
            // Internal - record the link
            RegionLink rlink;
            rlink.to = s.storage_dev[0].dat.get() + bDecomp.ghost[i].pos * s.storage[0].step;
            rlink.from = subdomains[sub].storage_dev[0].dat.get() + bDecomp.skin[i].pos * s.storage[0].step;
            rlink.len = len;
            links.push_back(rlink);
          } else {
            // External - record the ghost list
            auto recv = recvReg.find(dst);
            if (recv == recvReg.end())
              recv = recvReg.emplace(std::piecewise_construct, std::forward_as_tuple(dst),
                                     std::forward_as_tuple()).first;
            RegionSpec rreg;
            rreg.storage = &(s.storage[0]);
            rreg.storage_dev = &(s.storage_dev[0]);
            rreg.len = len;
            rreg.neighbor = i;
            rreg.skin_st = bDecomp.ghost[i].skin_st;
            rreg.zmort = zmort;
            rreg.pos = bDecomp.ghost[i].pos;
            rreg.rank = dst;
            recv->second.push_back(rreg);
          }
        } else if (n.set == bDecomp.skin[i].neighbor.set && dst != rank) {
          auto send = sendReg.find(dst);
          if (send == sendReg.end())
            send = sendReg.emplace(std::piecewise_construct, std::forward_as_tuple(dst),
                                   std::forward_as_tuple()).first;
          RegionSpec sreg;
          sreg.storage = &(s.storage[0]);
          sreg.storage_dev = &(s.storage_dev[0]);
          sreg.len = bDecomp.skin[i].len * s.storage[0].step * sizeof(bElem);
          sreg.neighbor = i;
          sreg.skin_st = bDecomp.skin[i].skin_st;
          sreg.zmort = s.zmort;
          sreg.pos = bDecomp.skin[i].pos;
          sreg.rank = dst;
          send->second.push_back(sreg);
        }
    }
  }

  std::vector<ExView> sendViews, recvViews;

  auto compareReg = [](const RegionSpec &a, const RegionSpec &b) {
    return (a.zmort.id < b.zmort.id) || ((a.zmort.id == b.zmort.id) && a.skin_st < b.skin_st);
  };

  {
    // sendReg needs to be coalesced
    std::unordered_map<int, std::vector<RegionSpec>> nsendReg;

    for (auto &sregs:sendReg) {
      std::sort(sregs.second.begin(), sregs.second.end(), compareReg);
      ExView sview;
      sview.rank = sregs.first;
      sview.len = 0;
      long lid = -1;
      long led = 0;
      auto send = nsendReg.emplace(std::piecewise_construct, std::forward_as_tuple(sregs.first),
                                   std::forward_as_tuple()).first;
      RegionSpec nsreg;
      // Calculate the total amount of memory required and change the offset list
      for (RegionSpec &sreg: sregs.second) {
        if (lid != sreg.zmort.id || sreg.skin_st >= led) {
          if (lid > 0)
            send->second.push_back(nsreg);
          nsreg.offset = sview.len;
          nsreg.storage = sreg.storage;
          nsreg.len = sreg.len;
          nsreg.storage_dev = sreg.storage_dev;
          nsreg.pos = sreg.pos;
          sreg.offset = sview.len;
          lid = sreg.zmort.id;
          led = bDecomp.skin[sreg.neighbor].skin_ed;
          sview.len += sreg.len;
        } else {
          // There is overlap, backtrace
          sreg.offset = sview.len;
          for (int i = sreg.skin_st; i < led; ++i)
            sreg.offset -= bDecomp.skin_size[i] * sreg.storage[0].step * sizeof(bElem);
          if (bDecomp.skin[sreg.neighbor].skin_ed > led) {
            led = bDecomp.skin[sreg.neighbor].skin_ed;
            sview.len = sreg.offset + sreg.len;
            nsreg.len = sview.len - nsreg.offset;
          }
        }
      }
      send->second.push_back(nsreg);
      // Allocate the region
      cudaMalloc(&sview.reg, sview.len);
#ifndef CUDA_AWARE
      sview.host_reg = (bElem*) malloc(sview.len);
#endif
      sendViews.push_back(sview);
    }
    sendReg = nsendReg;
  }

  for (auto &rregs:recvReg) {
    std::sort(rregs.second.begin(), rregs.second.end(), compareReg);
    ExView rview;
    rview.rank = rregs.first;
    rview.len = 0;
    long lid = -1;
    long led = 0;
    // Calculate the total amount of memory required and change the offset list
    for (RegionSpec &rreg: rregs.second) {
      if (lid != rreg.zmort.id || rreg.skin_st >= led) {
        rreg.offset = rview.len;
        lid = rreg.zmort.id;
        led = bDecomp.ghost[rreg.neighbor].skin_ed;
        rview.len += rreg.len;
      } else {
        // There is overlap, backtrace
        rreg.offset = rview.len;
        for (int i = rreg.skin_st; i < led; ++i)
          rreg.offset -= bDecomp.skin_size[i] * rreg.storage[0].step * sizeof(bElem);
        if (bDecomp.skin[rreg.neighbor].skin_ed > led) {
          led = bDecomp.skin[rreg.neighbor].skin_ed;
          rview.len = rreg.offset + rreg.len;
        }
      }
    }
    // Allocate the region
    cudaMalloc(&rview.reg, rview.len);
#ifndef CUDA_AWARE
    rview.host_reg = (bElem*) malloc(rview.len);
#endif
    recvViews.push_back(rview);
  }

  BrickInfo<3> *bInfo_dev;
  auto _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  {
    unsigned size = sizeof(BrickInfo<3>);
    cudaMalloc(&bInfo_dev, size);
    cudaMemcpy(bInfo_dev, &_bInfo_dev, size, cudaMemcpyHostToDevice);
  }

  unsigned *grid_dev_ptr = nullptr;
  copyToDevice({STRIDEB, STRIDEB, STRIDEB}, grid_dev_ptr, grid_ptr);

  // Create brick arrays for the device
  std::vector<Brick3D> bricks_dev_vec;

  for (unsigned long idx = 0; idx < mysec_r - mysec_l; ++idx) {
    bricks_dev_vec.emplace_back(bInfo_dev, subdomains[idx].storage_dev[0], 0);
    bricks_dev_vec.emplace_back(bInfo_dev, subdomains[idx].storage_dev[1], 0);
  }

  Brick3D *bricks_dev;
  {
    size_t size = sizeof(Brick3D) * bricks_dev_vec.size();
    cudaMalloc(&bricks_dev, size);
    cudaMemcpy(bricks_dev, bricks_dev_vec.data(), size, cudaMemcpyHostToDevice);
  }

  // Transfer links to GPU
  RegionLink *local_l_dev = nullptr;
  copyToDevice({(long) links.size()}, local_l_dev, links.data());

  RegionLink *pack_l_dev = nullptr;
  std::vector<RegionLink> pack_links;
  for (auto &sview: sendViews) {
    for (auto &sreg: sendReg[sview.rank]) {
      RegionLink rlink;
      rlink.from = sreg.storage_dev->dat.get() + sreg.storage->step * sreg.pos;
      rlink.to = (double *) (((uint8_t *) sview.reg) + sreg.offset);
      rlink.len = sreg.len;
      pack_links.push_back(rlink);
    }
  }
  copyToDevice({(long) pack_links.size()}, pack_l_dev, pack_links.data());

  RegionLink *unpack_l_dev = nullptr;
  std::vector<RegionLink> unpack_links;
  for (auto &rview: recvViews) {
    for (auto &rreg: recvReg[rview.rank]) {
      RegionLink rlink;
      rlink.from = (double *) (((uint8_t *) rview.reg) + rreg.offset);
      rlink.to = rreg.storage_dev->dat.get() + rreg.storage->step * rreg.pos;
      rlink.len = rreg.len;
      unpack_links.push_back(rlink);
    }
  }
  copyToDevice({(long) unpack_links.size()}, unpack_l_dev, unpack_links.data());

  auto brick_func = [&grid_dev_ptr, &sendViews, &sendReg, &recvViews, &recvReg, &bricks_dev_vec,
      &bricks_dev, &links, &local_l_dev, &pack_links, &pack_l_dev, &unpack_links, &unpack_l_dev]() -> void {
#ifdef BARRIER_TIMESTEP
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    float elapsed;
    double t_a = omp_get_wtime();
    cudaEvent_t c_0, c_1, c_2, c_3;
    cudaEventCreate(&c_0);
    cudaEventCreate(&c_1);
    cudaEventCreate(&c_2);
    cudaEventCreate(&c_3);
#ifndef CUDA_AWARE
    cudaEvent_t c_0m, c_3m;
    cudaEventCreate(&c_0m);
    cudaEventCreate(&c_3m);
    // Exchange
    std::vector <MPI_Request> requests(recvViews.size() + sendViews.size());
    // IRecv/ISend
    for (int i = 0; i < recvViews.size(); ++i)
      MPI_Irecv(recvViews[i].reg, recvViews[i].len, MPI_CHAR, recvViews[i].rank, 0, MPI_COMM_WORLD, &requests[i]);

    for (int i = 0; i < sendViews.size(); ++i)
      MPI_Isend(sendViews[i].reg, sendViews[i].len, MPI_CHAR, sendViews[i].rank, 0, MPI_COMM_WORLD,
                &requests[recvViews.size() + i]);
#else
    // Exchange
    std::vector<MPI_Request> requests(recvViews.size() + sendViews.size());
    // IRecv/ISend
    for (int i = 0; i < recvViews.size(); ++i)
      MPI_Irecv(recvViews[i].reg, recvViews[i].len, MPI_CHAR, recvViews[i].rank, 0, MPI_COMM_WORLD, &requests[i]);

    for (int i = 0; i < sendViews.size(); ++i)
      MPI_Isend(sendViews[i].reg, sendViews[i].len, MPI_CHAR, sendViews[i].rank, 0, MPI_COMM_WORLD,
                &requests[recvViews.size() + i]);
#endif
    double t_b = omp_get_wtime();
    calltime += t_b - t_a;

    {
      dim3 block(links.size()), thread(64);
      cudaCopy << < block, thread >> > (local_l_dev);
    }

    // Wait
    std::vector<MPI_Status> stats(requests.size());
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());
    t_a = omp_get_wtime();
    waittime += t_a - t_b;

    cudaEventRecord(c_0);
#ifndef CUDA_AWARE
    for (auto &rview: recvViews)
      cudaMemcpyAsync(rview.reg, rview.host_reg, rview.len, cudaMemcpyHostToDevice);

    cudaEventRecord(c_0m);
#endif
    // Unpack
    {
      dim3 block(unpack_links.size()), thread(64);
      cudaCopy << < block, thread >> > (unpack_l_dev);
    }
    cudaEventRecord(c_1);
    dim3 block(STRIDEB * (mysec_r - mysec_l), STRIDEB, STRIDEB), thread(32);
    if (mysec_r - mysec_l > 1)
      for (int i = 0; i < ST_ITER / 2; ++i) {
        brick_kernel << < block, thread >> > (grid_dev_ptr, STRIDEB, bricks_dev, 1, 0);
        brick_kernel << < block, thread >> > (grid_dev_ptr, STRIDEB, bricks_dev, 0, 1);
      }
    else
      for (int i = 0; i < ST_ITER / 2; ++i) {
        brick_kernel_single_domain << < block, thread >> >
                                               (grid_dev_ptr, bricks_dev_vec[0], bricks_dev_vec[1], STRIDEB);
        brick_kernel_single_domain << < block, thread >> >
                                               (grid_dev_ptr, bricks_dev_vec[1], bricks_dev_vec[0], STRIDEB);
      }
    cudaEventRecord(c_2);
    // Pack
    {
      dim3 block(pack_links.size()), thread(64);
      cudaCopy << < block, thread >> > (pack_l_dev);
    }
    cudaEventRecord(c_3);
#ifndef CUDA_AWARE
    // Move every skin back to host
    for (auto &sview: sendViews)
      cudaMemcpyAsync(sview.host_reg, sview.reg, sview.len, cudaMemcpyDeviceToHost);

    cudaEventRecord(c_3m);
    cudaEventSynchronize(c_3m);
    cudaEventElapsedTime(&elapsed, c_0, c_0m);
    movetime += elapsed / 1000.0;
    cudaEventElapsedTime(&elapsed, c_0m, c_1);
    packtime += elapsed / 1000.0;
    cudaEventElapsedTime(&elapsed, c_1, c_2);
    calctime += elapsed / 1000.0;
    cudaEventElapsedTime(&elapsed, c_2, c_3);
    packtime += elapsed / 1000.0;
    cudaEventElapsedTime(&elapsed, c_3, c_3m);
    movetime += elapsed / 1000.0;
#else
    cudaEventSynchronize(c_3);
    cudaEventElapsedTime(&elapsed, c_0, c_1);
    packtime += elapsed / 1000.0;
    cudaEventElapsedTime(&elapsed, c_1, c_2);
    calctime += elapsed / 1000.0;
    cudaEventElapsedTime(&elapsed, c_2, c_3);
    packtime += elapsed / 1000.0;
#endif
  };

  int cnt;

  {
    // Pack
    for (auto &sview: sendViews) {
      size_t pos = 0;
      for (auto &sreg: sendReg[sview.rank]) {
        cudaMemcpyAsync(((uint8_t *) sview.reg) + pos, sreg.storage_dev->dat.get() + sreg.storage->step * sreg.pos, sreg.len,
                        cudaMemcpyDeviceToDevice);
        pos += sreg.len;
      }
    }
#ifndef CUDA_AWARE
    // Move every skin back to host
    for (auto &sview: sendViews)
      cudaMemcpyAsync(sview.host_reg, sview.reg, sview.len, cudaMemcpyDeviceToHost);
#endif
    cudaDeviceSynchronize();
  }
  double tot = time_mpi(brick_func, cnt, bDecomp);
  cnt *= ST_ITER;
  size_t tsize = 0;
  for (auto &sview:sendViews)
    tsize += sview.len;
  for (auto &rview:sendViews)
    tsize += rview.len;

  mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
  mpi_stats pack_s = mpi_statistics(packtime / cnt, MPI_COMM_WORLD);
  mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
  mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
  mpi_stats move_s = mpi_statistics(movetime / cnt, MPI_COMM_WORLD);
  mpi_stats pspd_s = mpi_statistics(tsize / 1.0e9 / packtime * cnt, MPI_COMM_WORLD);
  mpi_stats mspd_s = mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
  mpi_stats size_s = mpi_statistics((double) tsize * 1.0e-6, MPI_COMM_WORLD);
  double total = calc_s.avg + call_s.avg + wait_s.avg + move_s.avg + pack_s.avg;

  if (rank == 0) {
    std::cout << "Bri: " << total << " : " << tot / ST_ITER << std::endl;

    std::cout << "calc : " << calc_s << std::endl;
    std::cout << "pack : " << pack_s << std::endl;
    std::cout << "  | pack speed (GB/s): " << pspd_s << std::endl;
    std::cout << "call : " << call_s << std::endl;
    std::cout << "wait : " << wait_s << std::endl;
    std::cout << "  | MPI size (MB): " << size_s << std::endl;
    std::cout << "  | MPI speed (GB/s): " << mspd_s << std::endl;
    std::cout << "move : " << move_s << std::endl;

    double perf = dom_size / 1000.0;
    perf = perf * perf * perf / total;
    std::cout << "perf " << perf << " GStencil/s" << std::endl;
    std::cout << "part " << sendViews.size() + recvViews.size() << std::endl;
  }

  for (auto &s: subdomains)
    s.cleanup();

  MPI_Finalize();

  return 0;
}