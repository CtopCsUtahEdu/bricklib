//
// Created by Tuowen Zhao on 2/17/19.
//

#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <brick.h>
#include <brick-mpi.h>
#include <array-mpi.h>
#include <bricksetup.h>
#include <zmort.h>
#include <map>
#include "stencils/fake.h"

#include "bitset.h"
#include <multiarray.h>
#include <brickcompare.h>

#include <unistd.h>
#include <memfd.h>

#define GZ 8
#define TILE 8
#define PADDING 8
#define BDIM 8,8,8

#include "stencils/cpuvfold.h"
#include "args.h"

#define STRIDE (sdom_size + 2 * GZ + 2 * PADDING)
#define STRIDEG (sdom_size + 2 * GZ)
#define STRIDEB ((sdom_size + 2 * GZ) / TILE)

typedef Brick<Dim<BDIM>, Dim<VFOLD>> Brick3D;

struct Subdomain {
  ZMORT zmort;
  std::vector<BrickStorage *> storage;
  std::vector<Brick3D> brick;
  std::vector<bElem *> array;
  std::unordered_map<uint64_t, int> rank_map;
  std::unordered_map<uint64_t, int> id_map;

  void cleanup() {
    for (auto bStorage: storage)
      if (bStorage->mmap_info != nullptr)
        ((MEMFD *) bStorage->mmap_info)->cleanup();
      else
        free(bStorage->dat);
  }
};

struct RegionSpec {
  // This always specifies the sender's zmort/neighbor
  ZMORT zmort;
  BitSet neighbor;
  BrickStorage *storage;
  long pos, len;
};

struct ExView {
  int rank, id;
  void *ptr;
  size_t len;
};

inline void brick_stencil_pencil(
    Brick3D &out, Brick3D &in, long skip, unsigned *grid_ptr) {
  for (long ti = skip; ti < STRIDEB - skip; ++ti) {
    unsigned b = grid_ptr[ti];
    brick(ST_SCRTPT, VSVEC, (BDIM), (VFOLD), b);
  }
}

int main(int argc, char **argv) {
  MPI_ITER = 25;
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided != MPI_THREAD_SERIALIZED) {
    MPI_Finalize();
    return 1;
  }

  parseArgs(argc, argv, "cpu");

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

  size_t sec_size = bSize * bInfo.nbricks * sizeof(bElem);
  MEMFD memfd(sec_size * (mysec_r - mysec_l));

  // Populate the list of all neighbors
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, 3, neighbors);

  for (unsigned long sec_id = mysec_l; sec_id < mysec_r; ++sec_id) {
    ZMORT zid(sec_id, 3);
    unsigned long idx = sec_id - mysec_l;
    subdomains[idx].zmort = zid;
    // bIn
    auto bStorage = new BrickStorage();
    *bStorage = bInfo.mmap_alloc(bSize, &memfd, sec_size * idx);
    subdomains[idx].brick.emplace_back(&bInfo, *bStorage, 0);
    subdomains[idx].storage.push_back(bStorage);

    // Initialize with random numbers
    bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
    copyToBrick<3>({sdom_size + 2 * GZ, sdom_size + 2 * GZ, sdom_size + 2 * GZ}, {PADDING, PADDING, PADDING}, {0, 0, 0},
                   in_ptr,
                   grid_ptr, subdomains[idx].brick.back());

    // bOut
    bStorage = new BrickStorage();
    *bStorage = bInfo.allocate(bSize);
    subdomains[idx].brick.emplace_back(&bInfo, *bStorage, 0);
    subdomains[idx].storage.push_back(bStorage);

    // internal bIn
    bStorage = new BrickStorage();
    *bStorage = bInfo.allocate(bSize);
    subdomains[idx].brick.emplace_back(&bInfo, *bStorage, 0);
    subdomains[idx].storage.push_back(bStorage);

    if (validate) {
      subdomains[idx].array.emplace_back(in_ptr);
      subdomains[idx].array.emplace_back(zeroArray({STRIDE, STRIDE, STRIDE}));
      // Populate rank map
      for (auto &n: neighbors) {
        int dst, sub;
        ZMORT zmort = zid;
        getrank(n, zmort, dst, sub);
        subdomains[idx].rank_map[n.set] = dst;
        subdomains[idx].id_map[n.set] = sub;
      }
    } else {
      free(in_ptr);
    }
  }

  // Starts linking all internal/external regions
  // This part assumes shared object so that the ghostzones on shared memory are reallocated to the correct page
  // Unordered map specified the dest/src rank and a bunch of regions
  std::unordered_map<int, std::vector<RegionSpec>> sendReg, recvReg;

  /* A ghost region can be in three states
   *   * Initial states (created) for communication
   *   * Linked with other region on the same node
   *   * Linked with other region's ghostzone
   * A recorder is needed for #3 to find out where is the other region's ghostzone located, which is referenced by
   * the targets (ID, inner parts no.), and need to produce:
   *   1. ID
   *   2. Location within (This is not reflective)
   */

  // Individual link instead of whole section link
  typedef struct {
    long i; // Index to other subdomain region
    size_t pos; // Position in that subdomain
    size_t len; // Length of memory region
  } region_instance;
  std::unordered_map<int, std::map<unsigned long, region_instance>> recv_parts, send_parts;

  for (long sidx = 0; sidx < subdomains.size(); ++sidx) {
    auto &s = subdomains[sidx];
    for (auto n: neighbors) {
      if (n.set == 0)
        continue;
      // get that neighbor's zmort
      int dst, sub;
      ZMORT zmort = s.zmort;
      getrank(n, zmort, dst, sub);
      // Process ghost/skins
      for (int i = 0; i < bDecomp.ghost.size(); ++i)
        if (n.set == bDecomp.ghost[i].neighbor.set) {
          size_t len = bDecomp.ghost[i].len * s.storage[0]->step * sizeof(bElem);
          if (dst == rank) {
            // Internal - link the ghost to the source's skin
            // un-map the ghost region
            bElem *st = s.storage[0]->dat + bDecomp.ghost[i].pos * s.storage[0]->step;
            int ret = munmap(st, len);
            if (ret != 0)
              throw std::runtime_error("Unmap failed");
            // map to the source - skin and ghost are mirrors
            ((MEMFD *) subdomains[sub].storage[0]->mmap_info)
                ->map_pointer(st, bDecomp.skin[i].pos * s.storage[0]->step * sizeof(bElem), len);
          } else {
            auto recv = recv_parts.find(dst);
            if (recv == recv_parts.end())
              recv = recv_parts.emplace(std::piecewise_construct, std::forward_as_tuple(dst),
                                        std::forward_as_tuple()).first;
            // Un-map the ghost region
            long spos = bDecomp.ghost[i].pos * s.storage[0]->step;
            int ret = munmap(s.storage[0]->dat + spos, len);
            if (ret != 0)
              throw std::runtime_error("Unmap failed");
            long last_mfd = -1;
            size_t last_pos = 0, last_size = 0;
            void *hint;
            // External - record the ghost list
            for (int j = bDecomp.ghost[i].skin_st; j < bDecomp.ghost[i].skin_ed; ++j) {
              // Create an id
              unsigned long id = (zmort.id << 5) + j;
              auto p = recv->second.find(id);
              if (p == recv->second.end()) {
                // Record a new link
                region_instance rins;
                rins.i = sidx;
                rins.pos = spos;
                rins.len = bDecomp.skin_size[j] * s.storage[0]->step * sizeof(bElem);
                p = recv->second.emplace(id, rins).first;
              }
              if (p->second.i == last_mfd && last_pos + last_size == p->second.pos * sizeof(bElem))
                last_size += p->second.len;
              else {
                if (last_mfd >= 0 && last_size > 0) {
                  // Map from somewhere else or remap it back
                  ((MEMFD *) subdomains[last_mfd].storage[0]->mmap_info)
                      ->map_pointer(hint, last_pos, last_size);
                }
                last_mfd = p->second.i;
                hint = s.storage[0]->dat + spos;
                last_pos = p->second.pos * sizeof(bElem);
                last_size = p->second.len;
              }
              spos += bDecomp.skin_size[j] * s.storage[0]->step;
            }
            if (last_mfd >= 0 && last_size > 0) {
              // Map from somewhere else or remap it back
              ((MEMFD *) subdomains[last_mfd].storage[0]->mmap_info)
                  ->map_pointer(hint, last_pos, last_size);
            }
          }
        } else if (n.set == bDecomp.skin[i].neighbor.set && dst != rank) {
          auto send = send_parts.find(dst);
          if (send == send_parts.end())
            send = send_parts.emplace(std::piecewise_construct, std::forward_as_tuple(dst),
                                      std::forward_as_tuple()).first;
          // Record the region
          long spos = bDecomp.skin[i].pos * s.storage[0]->step;
          for (int j = bDecomp.skin[i].skin_st; j < bDecomp.skin[i].skin_ed; ++j) {
            unsigned long id = ((sidx + mysec_l) << 5) + j;
            auto p = send->second.find(id);
            if (p == send->second.end()) {
              region_instance rins;
              rins.i = sidx;
              rins.pos = spos;
              rins.len = bDecomp.skin_size[j] * s.storage[0]->step * sizeof(bElem);
              send->second.emplace(id, rins);
            }
            spos += bDecomp.skin_size[j] * s.storage[0]->step;
          }
        }
    }
  }

  std::vector<ExView> sendViews, recvViews;

  // For send/recv sections we create one view of memory for each target

  for (auto &sregs: send_parts) {
    ExView sview;
    sview.ptr = nullptr;
    sview.len = 0;
    sview.rank = sregs.first;
    sview.id = (int) sregs.second.begin()->first;
    for (auto &sreg: sregs.second)
      if (sreg.second.len) {
        sview.ptr = ((MEMFD *) subdomains[sreg.second.i].storage[0]->mmap_info)->map_pointer(
            nullptr, sreg.second.pos * sizeof(bElem), sreg.second.len);
        sview.len += sreg.second.len;
      }
    sendViews.push_back(sview);
  }

  for (auto &rregs: recv_parts) {
    ExView rview;
    rview.ptr = nullptr;
    rview.len = 0;
    rview.rank = rregs.first;
    rview.id = (int) rregs.second.begin()->first;
    for (auto &rreg: rregs.second)
      if (rreg.second.len) {
        rview.ptr = ((MEMFD *) subdomains[rreg.second.i].storage[0]->mmap_info)->map_pointer(
            nullptr, rreg.second.pos * sizeof(bElem), rreg.second.len);
        rview.len += rreg.second.len;
      }
    recvViews.push_back(rview);
  }

  // Actual calculation
  auto brick_stencil = [&grid_ptr, &subdomains](int out, int in, long skip) -> void {
    auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;
    int size = subdomains.size();
#pragma omp parallel for collapse(3)
    for (int s = 0; s < size; ++s)
      for (long tk = skip; tk < STRIDEB - skip; ++tk)
        for (long tj = skip; tj < STRIDEB - skip; ++tj)
          brick_stencil_pencil(subdomains[s].brick[out], subdomains[s].brick[in], skip, &(grid[tk][tj][0]));
  };

  auto brick_func = [&]() -> void {
#ifdef BARRIER_TIMESTEP
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    double t_a = omp_get_wtime();

    std::vector<MPI_Request> requests(recvViews.size() + sendViews.size());

    for (int i = 0; i < recvViews.size(); ++i)
      MPI_Irecv(recvViews[i].ptr, recvViews[i].len, MPI_CHAR, recvViews[i].rank, recvViews[i].id, MPI_COMM_WORLD,
                &requests[i]);
    for (int i = 0; i < sendViews.size(); ++i)
      MPI_Isend(sendViews[i].ptr, sendViews[i].len, MPI_CHAR, sendViews[i].rank, sendViews[i].id, MPI_COMM_WORLD,
                &requests[i + recvViews.size()]);

    double t_b = omp_get_wtime();
    calltime += t_b - t_a;

    std::vector<MPI_Status> stats(requests.size());
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

    t_a = omp_get_wtime();
    waittime += t_a - t_b;

    // Iterate over all subdomains
    brick_stencil(1, 0, 0);
    for (int i = 0; i < ST_ITER / 2 - 1; ++i) {
      brick_stencil(2, 1, 0);
      brick_stencil(1, 2, 0);
    }
    brick_stencil(0, 1, 1);

    t_b = omp_get_wtime();
    calctime += t_b - t_a;
  };

  int cnt;
  double tot = time_mpi(brick_func, cnt, bDecomp);
  cnt *= ST_ITER;
  size_t tsize = 0;
  for (auto &sview:sendViews)
    tsize += sview.len;
  for (auto &rview:sendViews)
    tsize += rview.len;

  mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
  mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
  mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
  mpi_stats mspd_s = mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
  mpi_stats size_s = mpi_statistics((double) tsize * 1.0e-6, MPI_COMM_WORLD);
  double total = calc_s.avg + wait_s.avg + call_s.avg;

  if (rank == 0) {
    std::cout << "Bri: " << total << " : " << tot / ST_ITER << std::endl;

    std::cout << "calc : " << calc_s << std::endl;
    std::cout << "call : " << call_s << std::endl;
    std::cout << "wait : " << wait_s << std::endl;
    std::cout << "  | MPI size (MB): " << size_s << std::endl;
    std::cout << "  | MPI speed (GB/s): " << mspd_s << std::endl;
    std::cout << "parts " << recvViews.size() + sendViews.size() << std::endl;

    double perf = dom_size / 1000.0;
    perf = perf * perf * perf / total;
    std::cout << "perf " << perf << " GStencil/s" << std::endl;
  }

  if (validate) {
    if (rank == 0)
      std::cout << "Validation is enabled by \"-v\" on the commandline, validating" << std::endl;

    auto array_stencil = [&](bElem *arrOut_ptr, bElem *arrIn_ptr, long skip) -> void {
      auto arrIn = (bElem (*)[STRIDE][STRIDE]) arrIn_ptr;
      auto arrOut = (bElem (*)[STRIDE][STRIDE]) arrOut_ptr;
      long skip2 = (skip / TILE) * TILE;
#pragma omp parallel for collapse(2)
      for (long tk = PADDING + skip; tk < PADDING + STRIDEG - skip; tk += TILE)
        for (long tj = PADDING + skip; tj < PADDING + STRIDEG - skip; tj += TILE)
          for (long ti = PADDING + skip2; ti < PADDING + STRIDEG - skip2; ti += TILE)
            for (long k = tk; k < tk + TILE; ++k)
              for (long j = tj; j < tj + TILE; ++j)
#pragma omp simd
                  for (long i = ti; i < ti + TILE; ++i)
                    ST_CPU;
    };

    std::vector<ArrExPack> arrpack;
    for (auto &s: subdomains) {
      ArrExPack p;
      p.arr = s.array[0];
      p.id = s.zmort.id - mysec_l;
      p.rank_map = &s.rank_map;
      p.id_map = &s.id_map;
      arrpack.emplace_back(p);
    }
    auto arr_func = [&]() -> void {
      for (auto &s: subdomains)
        exchangeArrAll<3>(arrpack, MPI_COMM_WORLD, {sdom_size, sdom_size, sdom_size},
                          {PADDING, PADDING, PADDING}, {GZ, GZ, GZ});

      double t_a = omp_get_wtime();
      for (auto &s: subdomains)
        array_stencil(s.array[1], s.array[0], 0);
      for (int i = 0; i < ST_ITER / 2 - 1; ++i) {
        for (auto &s: subdomains)
          array_stencil(s.array[0], s.array[1], 0);
        for (auto &s: subdomains)
          array_stencil(s.array[1], s.array[0], 0);
      }
      for (auto &s: subdomains)
        array_stencil(s.array[0], s.array[1], TILE);
      double t_b = omp_get_wtime();
      calctime += t_b - t_a;
    };

    double total = time_mpi(arr_func, cnt, bDecomp);
    cnt *= ST_ITER;
    if (rank == 0)
      std::cout << "Checking result" << std::endl;

    bool correct = true, allcorrect;
    for (auto &s: subdomains)
      if (!compareBrick<3>({sdom_size, sdom_size, sdom_size}, {PADDING, PADDING, PADDING},
                           {GZ, GZ, GZ}, s.array[0], grid_ptr, s.brick[0])) {
        std::cout << "Rank " << rank << " result mismatch at subdomain " << s.zmort.id << std::endl;
        correct = false;
      }
    MPI_Reduce(&correct, &allcorrect, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
    if (rank == 0 && allcorrect)
      std::cout << "Validated" << std::endl;
    mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
    mpi_stats pack_s = mpi_statistics(packtime / cnt, MPI_COMM_WORLD);
    mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
    mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
    total = calc_s.avg + wait_s.avg + call_s.avg + pack_s.avg;
    std::cout << "Array: " << total << std::endl;

    std::cout << "calc : " << calc_s << std::endl;
    std::cout << "pack : " << pack_s << std::endl;
    std::cout << "call : " << call_s << std::endl;
    std::cout << "wait : " << wait_s << std::endl;
  }

  memfd.cleanup();

  MPI_Finalize();
  return 0;
}
