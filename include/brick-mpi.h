//
// Created by Tuowen Zhao on 12/10/18.
//

#ifndef BRICK_BRICK_MPI_H
#define BRICK_BRICK_MPI_H

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
#include "brick.h"
#include "bitset.h"
#include "memfd.h"

extern double packtime, calltime, waittime, movetime, calctime;

void allneighbors(BitSet cur, long idx, long dim, std::vector<BitSet> &neighbors);

// Grid accessor
template<typename T, unsigned dim, unsigned d>
struct grid_access;

template<typename T, unsigned dim>
struct grid_access<T, dim, 1> {
  T *self;
  unsigned ref;

  grid_access(T *self, unsigned ref) : self(self), ref(ref) {}

  inline unsigned operator[](int i) {
    return self->grid[ref + i];
  }
};

template<typename T, unsigned dim, unsigned d>
struct grid_access {
  T *self;
  unsigned ref;

  grid_access(T *self, unsigned ref) : self(self), ref(ref) {}

  inline grid_access<T, dim, d - 1> operator[](int i) {
    return grid_access<T, dim, d - 1>(self, ref + i * self->stride[d - 1]);
  }
};


struct ExchangeView {
  MPI_Comm comm;
  std::vector<size_t> seclen;
  typedef std::vector<std::pair<int, void *>> Dest;
  Dest send, recv;

  ExchangeView(MPI_Comm comm, std::vector<size_t> seclen, Dest send, Dest recv) :
      comm(comm), seclen(std::move(seclen)), send(std::move(send)), recv(std::move(recv)) {}

  void exchange() {
    std::vector<MPI_Request> requests(seclen.size() * 2);
    double st = omp_get_wtime(), ed;

    for (int i = 0; i < seclen.size(); ++i) {
      // receive to ghost[i]
      MPI_Irecv(recv[i].second, seclen[i], MPI_CHAR, recv[i].first, i, comm, &(requests[i << 1]));
      // send from skin[i]
      MPI_Isend(send[i].second, seclen[i], MPI_CHAR, send[i].first, i, comm, &(requests[(i << 1) + 1]));
    }

    ed = omp_get_wtime();
    calltime += ed - st;
    st = ed;

    std::vector<MPI_Status> stats(requests.size());
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

    ed = omp_get_wtime();
    waittime += ed - st;
  }
};

struct MultiStageExchangeView {
  MPI_Comm comm;
  typedef struct {
    int rank;
    size_t len;
    void *buf;
  } Package;
  typedef std::vector<Package> Stage;
  std::vector<Stage> send, recv;

  MultiStageExchangeView(MPI_Comm comm, std::vector<Stage> send, std::vector<Stage> recv) :
      comm(comm), send(std::move(send)), recv(std::move(recv)) {}

  void exchange() {
    double st = omp_get_wtime(), wtime = 0;
    for (long i = 0; i < send.size(); ++i) {
      std::vector<MPI_Request> requests(send[i].size() + recv[i].size());
      for (long j = 0; j < recv[i].size(); ++j)
        // TODO TAG might be a problem
        MPI_Irecv(recv[i][j].buf, recv[i][j].len, MPI_CHAR, recv[i][j].rank, j, comm, &(requests[j]));
      for (long j = 0; j < send[i].size(); ++j)
        MPI_Isend(send[i][j].buf, send[i][j].len, MPI_CHAR, send[i][j].rank, j, comm, &(requests[recv[i].size() + j]));
      std::vector<MPI_Status> stats(requests.size());
      double stime = omp_get_wtime();
      MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());
      wtime += omp_get_wtime() - stime;
    }
    calltime += omp_get_wtime() - st - wtime;
    waittime += wtime;
  }
};

/* This create a decomposition for MPI communication
 *
 * Decomposition is setup in steps:
 * 1. Reserve space of the inner-inner region
 * 2. Skin layout of inner region
 * 3. Exchange ghost region location (Segmenting info)
 *  * segment info
 * 4. Setup ghost region
 * 5. All extra ghost link to the end brick
 */
template<unsigned dim,
    unsigned ... BDims>
class BrickDecomp {
public:
  // We need to record the start of each of the ghostzones
  // To identify a ghost region use:
  // * The set for the neighbor
  // * The r for the region
  // To record the start and length of the region use A
  // Each neighbor can be identified using a map
  typedef struct {
    BitSet neighbor;
    unsigned skin_st, skin_ed;
    unsigned pos, len;
  } g_region;
  std::vector<g_region> ghost;
  std::vector<g_region> skin;
  unsigned sep_pos[3];
  std::vector<BitSet> skinlist; // The order of skin
  std::vector<long> skin_size; // The size of skin
private:
  typedef BrickDecomp<dim, BDims...> mytype;

  std::vector<unsigned> dims, t_dims;
  std::vector<unsigned> g_depth; // The depth of ghostzone
  std::vector<unsigned> stride;
  unsigned *grid;
  unsigned numfield;
  BrickInfo<dim> *bInfo;

  template<typename T, unsigned di, unsigned d>
  friend
  struct grid_access;


  /* Regions can be identified with:
   * Own/neighbor(ghost zone) + which region
   */
  void _populate(BitSet region, long ref, int d, unsigned &pos) {
    if (d == -1) {
      grid[ref] = pos++;
      return;
    }
    long sec = 0;
    long st = 0;
    if (region.get(d + 1)) {
      sec = 1;
      st = dims[d];
    }
    if (region.get(-d - 1)) {
      sec = -1;
      st = g_depth[d];
    }
    if (sec)
      for (unsigned i = 0; i < g_depth[d]; ++i)
        _populate(region, ref + (st + i) * stride[d], d - 1, pos);
    else
      for (unsigned i = 2 * g_depth[d]; i < dims[d]; ++i)
        _populate(region, ref + i * stride[d], d - 1, pos);
  }

  void populate(BitSet owner, BitSet region, unsigned &pos) {
    // For myself
    long ref = 0;
    // For neighbor owner
    for (long d = 0; d < dim; ++d) {
      if (owner.get(d + 1))
        // Shift up
        ref += dims[d] * stride[d];
      if (owner.get(-d - 1))
        // Shift down
        ref -= dims[d] * stride[d];
    }
    _populate(region, ref, dim - 1, pos);
  }

  void _adj_populate(long ref, unsigned d, unsigned idx, unsigned *adj) {
    long cur = ref;
    if (cur >= 0) {
      cur = ref / stride[d];
      cur = cur % (t_dims[d]);
    }
    idx *= 3;
    if (d == 0) {
      for (int i = 0; i < 3; ++i) {
        if (i + cur < 1 || i + cur > t_dims[d] || ref < 0)
          adj[idx + i] = 0;
        else
          adj[idx + i] = grid[ref + i - 1];
      }
    } else {
      for (int i = 0; i < 3; ++i)
        if (i + cur < 1 || i + cur > t_dims[d] || ref < 0)
          _adj_populate(-1, d - 1, idx + i, adj);
        else
          _adj_populate(ref + (i - 1) * (long) stride[d], d - 1, idx + i, adj);
    }
  }

  void adj_populate(unsigned i, unsigned *adj) {
    _adj_populate(i, dim - 1, 0, adj);
  }

public:
  MPI_Comm comm;

  std::unordered_map<uint64_t, int> rank_map; // Mapping from neighbor to each neighbor's rank

  BrickDecomp(const std::vector<unsigned> &dims, const unsigned depth, unsigned numfield = 1)
      : dims(dims), numfield(numfield), bInfo(nullptr), grid(nullptr) {
    assert(dims.size() == dim);
    std::vector<unsigned> bdims = {BDims ...};
    // Arrays needs to be kept contiguous first
    std::reverse(bdims.begin(), bdims.end());

    for (int i = 0; i < dim; ++i) {
      assert(depth % bdims[i] == 0);
      g_depth.emplace_back(depth / bdims[i]);
      this->dims[i] /= bdims[i];
    }
  };

  void initialize(const std::vector<BitSet> &skinlist) {
    calltime = waittime = 0;
    this->skinlist = skinlist;

    // All space is dimensions + ghostzone + 1
    unsigned grid_size = 1;
    stride.clear();
    stride.resize(dim);
    t_dims.clear();
    t_dims.resize(dim);

    for (int i = 0; i < dim; ++i) {
      stride[i] = grid_size;
      t_dims[i] = dims[i] + 2 * g_depth[i];
      grid_size *= t_dims[i];
    }

    // Global reference to grid Index
    grid = new unsigned[grid_size];

    int pagesize = sysconf(_SC_PAGESIZE);
    int bSize = cal_size<BDims ...>::value * sizeof(bElem) * numfield;

    if (std::max(bSize, pagesize) % std::min(bSize, pagesize) != 0)
      throw std::runtime_error("brick size must be a factor/multiple of pagesize.");

    int factor = 1;
    if (bSize < pagesize)
      factor = pagesize / bSize;

    // This assumes all domains are aligned and allocated with same "skinlist"
    unsigned pos = factor;

    auto mypop = [&pos, &factor, this](BitSet owner, BitSet region) {
      populate(owner, region, pos);
#ifndef DECOMP_PAGEUNALIGN
      pos = (pos + factor - 1) / factor * factor;
#endif
    };

    // Allocating inner region
    mypop(0, 0);

    std::vector<unsigned> st_pos;
    st_pos.emplace_back(pos);
    sep_pos[0] = pos;

    skin_size.clear();
    // Allocating skinlist
    for (auto i: skinlist) {
      long ppos = pos;
      if (i.set != 0)
        mypop(0, i);
      st_pos.emplace_back(pos);
      skin_size.emplace_back(pos - ppos);
    }
    sep_pos[1] = pos;

    /* Allocating ghost
     * A ghost regions contains all region that are superset of the "inverse"
     */
    std::vector<BitSet> neighbors;
    allneighbors(0, 1, dim, neighbors);
    ghost.clear();
    skin.clear();
    for (auto n: neighbors) {
      if (n.set == 0)
        continue;
      BitSet in = !n;
      g_region g, i;
      int last = -1;
      g.neighbor = n;
      i.neighbor = in;
      // Following the order of the skinlist
      // Record contiguousness
      for (int l = 0; l < skinlist.size(); ++l)
        if (in <= skinlist[l]) {
          if (last < 0) {
            last = l;
            i.skin_st = g.skin_st = (unsigned) last;
            g.pos = pos;
            i.pos = st_pos[last];
          }
          mypop(n, skinlist[l]);
        } else if (last >= 0) {
          g.skin_ed = (unsigned) l;
          g.len = pos - g.pos;
          ghost.emplace_back(g);
          i.len = st_pos[l] - i.pos;
          i.skin_ed = (unsigned) l;
          skin.emplace_back(i);
          last = -1;
        }
      if (last >= 0) {
        g.skin_ed = (unsigned) skinlist.size();
        g.len = pos - g.pos;
        ghost.emplace_back(g);
        i.len = st_pos[skinlist.size()] - i.pos;
        i.skin_ed = (unsigned) skinlist.size();
        skin.emplace_back(i);
      }
    }
    sep_pos[2] = pos;

    // Convert grid to adjlist
    int size = pos;
    if (bInfo == nullptr)
      bInfo = new BrickInfo<dim>(size);
    for (unsigned i = 0; i < grid_size; ++i)
      adj_populate(i, bInfo->adj[grid[i]]);
  }

  void exchange(BrickStorage &bStorage) {
    std::vector<MPI_Request> requests(ghost.size() * 2);
    double st = omp_get_wtime(), ed;

    for (int i = 0; i < ghost.size(); ++i) {
      // receive to ghost[i]
      MPI_Irecv(&(bStorage.dat[ghost[i].pos * bStorage.step]), ghost[i].len * bStorage.step * sizeof(bElem),
                MPI_CHAR, rank_map[ghost[i].neighbor.set], i, comm, &(requests[i << 1]));
      // send from skin[i]
      MPI_Isend(&(bStorage.dat[skin[i].pos * bStorage.step]), skin[i].len * bStorage.step * sizeof(bElem),
                MPI_CHAR, rank_map[skin[i].neighbor.set], i, comm, &(requests[(i << 1) + 1]));
    }

    ed = omp_get_wtime();
    calltime += ed - st;
    st = ed;

    std::vector<MPI_Status> stats(requests.size());
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

    ed = omp_get_wtime();
    waittime += ed - st;
  }


  grid_access<mytype, dim, dim - 1> operator[](int i) {
    auto ga = grid_access<mytype, dim, dim>(this, 0);
    return ga[i];
  }

  BrickInfo<dim> getBrickInfo() {
    return *bInfo;
  }

  ~BrickDecomp() {
    delete[] grid;
    delete bInfo;
  }

  ExchangeView exchangeView(BrickStorage bStorage) {
    // All Brick Storage are initialized with mmap for exchanging using views
    assert(bStorage.mmap_info != nullptr);

    // Generate all neighbor bitset
    std::vector<BitSet> neighbors;
    allneighbors(0, 1, dim, neighbors);

    // All each section is a pair of exchange with one neighbor
    std::vector<size_t> seclen;
    ExchangeView::Dest send;
    ExchangeView::Dest recv;

    // memfd that backs the canonical view
    auto memfd = (MEMFD *) bStorage.mmap_info;

    // Iterating over neighbors
    for (auto n: neighbors) {
      // Skip
      if (n.set == 0)
        continue;
      // Receive buffer + buffer length
      std::vector<size_t> packing;
      size_t len = 0;
      for (auto g: ghost) {
        if (g.neighbor.set == n.set) {
          packing.push_back(g.pos * bStorage.step * sizeof(bElem));
          size_t l = g.len * bStorage.step * sizeof(bElem);
          packing.push_back(l);
          len += l;
        }
      }
      recv.push_back(std::make_pair(rank_map[n.set], memfd->packed_pointer(packing)));
      seclen.push_back(len);
      packing.clear();
      // Send buffer
      BitSet in = !n;
      for (auto s: skin)
        if (s.neighbor.set == in.set) {
          packing.push_back(s.pos * bStorage.step * sizeof(bElem));
          packing.push_back(s.len * bStorage.step * sizeof(bElem));
        }
      send.push_back(std::make_pair(rank_map[in.set], memfd->packed_pointer(packing)));
    }

    return ExchangeView(comm, seclen, send, recv);
  }

  MultiStageExchangeView multiStageExchangeView(BrickStorage bStorage) {
    // All Brick Storage are initialized with mmap for exchanging using views
    assert(bStorage.mmap_info != nullptr);

    // Stages are exchanges along one axis.
    std::vector<MultiStageExchangeView::Stage> send, recv;

    // Generate all neighbor bitset
    // The skin_list is no longer important however it is useful to compose the messages.
    std::vector<BitSet> neighbors;
    allneighbors(0, 1, dim, neighbors);

    // memfd that backs the canonical view
    auto memfd = (MEMFD *) bStorage.mmap_info;

    BitSet exchanged = 0;

    for (int d = 1; d <= dim; ++d) {
      BitSet toexchange = exchanged;
      BitSet exchanging = 0;
      exchanging.flip(d);
      exchanging.flip(-d);
      toexchange = toexchange | exchanging;
      send.emplace_back();
      recv.emplace_back();
      // Exchanging in one direction at a time
      for (auto &n: neighbors)
        if (n <= exchanging && n.set != 0) {  // This neighbor is the one exchanging with
          MultiStageExchangeView::Package sendpkg, recvpkg;
          size_t len = 0;
          std::vector<size_t> rpacking;
          std::vector<size_t> spacking;
          // Receiving the stuff from the current neighbor
          for (auto g: ghost)
            if (g.neighbor.set == n.set) {
              rpacking.push_back(g.pos * bStorage.step * sizeof(bElem));
              size_t l = g.len * bStorage.step * sizeof(bElem);
              rpacking.push_back(l);
              len += l;
            }
          BitSet Sn = !n;
          for (auto s: skin)
            if (s.neighbor.set == Sn.set) {
              spacking.push_back(s.pos * bStorage.step * sizeof(bElem));
              spacking.push_back(s.len * bStorage.step * sizeof(bElem));
            }
          // Receiving from neighbor's neighbor, assuming neighbor is ordered much like our own.
          BitSet n2 = n | exchanged;
          for (auto &g: ghost) {
            if (g.neighbor <= n2 && !(g.neighbor <= exchanged) && !(g.neighbor.set == n.set)) {
              // this need to be exchanged
              rpacking.push_back(g.pos * bStorage.step * sizeof(bElem));
              size_t l = g.len * bStorage.step * sizeof(bElem);
              rpacking.push_back(l);
              len += l;
              // The corresponding us part also needs to be exchanged
              // Find the corresponding neighbor
              BitSet s = g.neighbor ^ n;
              for (int sec = g.skin_st; sec < g.skin_ed; ++sec)
                // looking for the corresponding skin part
                for (auto g2: ghost)
                  if (g2.neighbor.set == s.set)
                    if (sec >= g2.skin_st && sec < g2.skin_ed) {
                      auto pos = g2.pos;
                      for (int j = g2.skin_st; j < sec; ++j)
                        pos += skin_size[j];
                      int last = spacking.size();
                      size_t fst = pos * bStorage.step * sizeof(bElem);
                      size_t seclen = skin_size[sec] * bStorage.step * sizeof(bElem);
                      if (spacking[last - 2] + spacking[last - 1] == fst)
                        spacking[last - 1] += seclen;
                      else {
                        spacking.push_back(fst);
                        spacking.push_back(seclen);
                      }
                    }
            }
          }
          recvpkg.len = len;
          recvpkg.buf = memfd->packed_pointer(rpacking);
          recvpkg.rank = rank_map[n.set];
          recv.back().emplace_back(recvpkg);
          // Sending my stuff
          size_t ll = 0;
          for (int i = 0; i < spacking.size(); i += 2)
            ll += spacking[i + 1];
          sendpkg.len = len;
          sendpkg.buf = memfd->packed_pointer(spacking);
          sendpkg.rank = rank_map[Sn.set];
          send.back().emplace_back(sendpkg);
        }
      exchanged = toexchange;
    }
    return MultiStageExchangeView(comm, send, recv);
  }

  // This only works for same sized domain right now
  void exchange(BrickStorage bStorage, MPI_Win &win) {
    double st = omp_get_wtime(), ed;

    MPI_Win_fence(0, win);

    ed = omp_get_wtime();
    waittime += ed - st;
    st = ed;

    for (int i = 0; i < ghost.size(); ++i) {
      size_t len = ghost[i].len * bStorage.step * sizeof(bElem);
      // receive from remote
      MPI_Get(&(bStorage.dat[ghost[i].pos * bStorage.step]), len, MPI_CHAR, rank_map[ghost[i].neighbor.set],
              skin[i].pos * bStorage.step * sizeof(bElem), len, MPI_CHAR, win);
    }

    ed = omp_get_wtime();
    calltime += ed - st;
    st = ed;

    MPI_Win_fence(0, win);

    ed = omp_get_wtime();
    waittime += ed - st;
  }
};

template<unsigned dim, unsigned ...BDims>
void populate(MPI_Comm &comm, BrickDecomp<dim, BDims...> &bDecomp, BitSet neighbor, int d, int *coo) {
  if (d > dim) {
    int rank;
    MPI_Cart_rank(comm, coo, &rank);
    bDecomp.rank_map[neighbor.set] = rank;
    return;
  }

  int c = coo[d - 1];
  neighbor.flip(d);
  coo[d - 1] = c - 1;
  populate(comm, bDecomp, neighbor, d + 1, coo);
  neighbor.flip(d);
  // Not picked
  coo[d - 1] = c;
  populate(comm, bDecomp, neighbor, d + 1, coo);
  // Picked -
  neighbor.flip(-d);
  coo[d - 1] = c + 1;
  populate(comm, bDecomp, neighbor, d + 1, coo);
  coo[d - 1] = c;
}

typedef struct {
  double min, max, avg, sigma;
} mpi_stats;

inline mpi_stats mpi_statistics(double stats, MPI_Comm comm) {
  mpi_stats ret;
  int size;
  MPI_Comm_size(comm, &size);
  MPI_Reduce(&stats, &ret.avg, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  ret.avg = ret.avg;
  MPI_Reduce(&stats, &ret.max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats, &ret.min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  stats = stats * stats;
  MPI_Reduce(&stats, &ret.sigma, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  ret.sigma -= ret.avg * ret.avg / size;
  ret.avg /= size;
  ret.sigma /= std::max(size - 1, 1);
  ret.sigma = std::sqrt(ret.sigma);
  return ret;
}

inline std::ostream &operator<<(std::ostream &os, const mpi_stats &stats) {
  os << "[" << stats.min << ", " << stats.avg << ", " << stats.max << "]" << " (σ: " << stats.sigma << ")";
  return os;
}

extern std::vector<BitSet> skin3d_good, skin3d_normal, skin3d_bad;

#endif //BRICK_BRICK_MPI_H
