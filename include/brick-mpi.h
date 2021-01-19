/**
 * @file
 * @brief MPI stuff related to bricks
 */

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

/**
 * @brief Enumerate all neighbors
 * @param[in] cur usually 0
 * @param[in] idx current dimension, starts from 1
 * @param[in] dim total number of dimensions
 * @param[out] neighbors a list of neighbors
 */
void allneighbors(BitSet cur, long idx, long dim, std::vector<BitSet> &neighbors);

/**
 * @defgroup grid_access Accessing grid indices using []
 *
 * It can be fully unrolled and offers very little overhead.
 *
 * @{
 */


/**
 * @brief Generic base template for @ref grid_access
 * @tparam T type of the BrickDecomp
 * @tparam dim number of dimensions
 * @tparam d current dimension
 */
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
/**@}*/

/**
 * @brief PUT view of the ghost and surface region using mmap
 *
 * Created from BrickDecomp::exchangeView() for PUT exchange
 */
struct ExchangeView {
  MPI_Comm comm;
  std::vector<size_t> seclen;
  std::vector<size_t> first_pad;
  typedef std::vector<std::pair<int, void *>> Dest;
  Dest send, recv;

  ExchangeView(MPI_Comm comm, std::vector<size_t> seclen, std::vector<size_t> first_pad, Dest send, Dest recv) :
      comm(comm), seclen(std::move(seclen)), first_pad(std::move(first_pad)),
      send(std::move(send)), recv(std::move(recv)) {}

  /**
   * @brief Exchange all ghost zones
   */
  void exchange() {
    std::vector<MPI_Request> requests(seclen.size() * 2);
    std::vector<MPI_Status> stats(requests.size());

#ifdef BARRIER_TIMESTEP
    MPI_Barrier(comm);
#endif

    double st = omp_get_wtime(), ed;

    for (int i = 0; i < seclen.size(); ++i) {
      // receive to ghost[i]
      MPI_Irecv((uint8_t *) recv[i].second + first_pad[i], seclen[i], MPI_CHAR, recv[i].first, i, comm,
                &(requests[i << 1]));
      // send from skin[i]
      MPI_Isend((uint8_t *) send[i].second + first_pad[i], seclen[i], MPI_CHAR, send[i].first, i, comm,
                &(requests[(i << 1) + 1]));
    }

    ed = omp_get_wtime();
    calltime += ed - st;
    st = ed;

    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

    ed = omp_get_wtime();
    waittime += ed - st;
  }
};

/**
 * @brief SHIFT view of the ghost and surface region using mmap
 *
 * Created from BrickDecomp::multiStageExchangeView() for SHIFT exchange
 */
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

#ifdef BARRIER_TIMESTEP
    MPI_Barrier(comm);
#endif

    double st = omp_get_wtime(), wtime = 0;
    for (long i = 0; i < send.size(); ++i) {
      std::vector<MPI_Request> requests(send[i].size() + recv[i].size());
      for (long j = 0; j < recv[i].size(); ++j)
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

/**
 * @brief Decomposition for MPI communication
 * @tparam dim number of dimensions
 * @tparam BDims Brick dimensions
 *
 * Decomposition is setup in steps:
 * 1. Reserve space of the inner-inner region
 * 2. Surface layout of inner region
 * 3. Setup ghost region
 * 4. All extra ghost link to the end brick
 */
template<unsigned dim,
    unsigned ... BDims>
class BrickDecomp {
public:
  /**
   * @brief Record start and end of each region
   */
  typedef struct {
    BitSet neighbor;     ///< The set for the neighbor
    unsigned skin_st;    ///< starting elements in the skin list
    unsigned skin_ed;    ///< ending index in the skin list (not included)
    unsigned pos;        ///< starting from which brick
    unsigned len;        ///< ending at which brick (not included)
    unsigned first_pad;
    unsigned last_pad;
  } g_region;
  std::vector<g_region> ghost;      ///< ghost regions record
  std::vector<g_region> skin;       ///< surface regions record
  unsigned sep_pos[3];              ///< seperation points internal-surface-ghost
  std::vector<BitSet> skinlist;     ///< the order of skin
  std::vector<long> skin_size;      ///< the size of skin
private:
  typedef BrickDecomp<dim, BDims...> mytype;    ///< shorthand for type of this instance

  std::vector<unsigned> dims;       ///< dimension of internal in bricks
  std::vector<unsigned> t_dims;     ///< dimension including ghosts in bricks
  std::vector<unsigned> g_depth;    ///< The depth of ghostzone in bricks
  std::vector<unsigned> stride;     ///< stride in bricks
  unsigned *grid;                   ///< Grid indices
  unsigned numfield;                ///< Number of fields that are interleaved
  BrickInfo<dim> *bInfo;            ///< Associated BrickInfo

  template<typename T, unsigned di, unsigned d>
  friend
  struct grid_access;               ///< Need private access for @ref grid_access


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
    if (sec != 0)
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

  long get_region_size(BitSet region) {
    long ret = 1;
    for (long d = 1; d <= dim; ++d)
      if (region.get(d) || region.get(-d))
        ret *= g_depth[d - 1];
      else
        ret *= dims[d - 1] - 2 * g_depth[d - 1];
    return ret;
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
  MPI_Comm comm;        ///< MPI communicator it is attached to

  std::unordered_map<uint64_t, int> rank_map;        ///< Mapping from neighbor to each neighbor's rank

  /**
   * @brief MPI decomposition for bricks
   * @param dims the size of each dimension excluding the ghost (in elements)
   * @param depth the depths of ghost (in elements)
   * @param numfield number of interleaved fields
   */
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

  /**
   * @brief initialize the decomposition using skinlist
   * @param skinlist the layout of the surface area, recommended: skin3d_good
   */
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

    auto calc_pad = [&factor, this](const BitSet &region) -> long {
#ifdef DECOMP_PAGEUNALIGN
      return 0;
#else
      return factor - (get_region_size(region) + factor - 1) % factor - 1;
#endif
    };

    std::vector<unsigned> st_pos;  // The starting position of a segment
    std::vector<bool> pad_first;

    BitSet last = 0;
    for (long i = 0; i < skinlist.size(); ++i) {
      BitSet next = 0;
      if (i < skinlist.size() - 1)
        next = skinlist[i + 1];
      pad_first.push_back(((last & skinlist[i]).size() < (skinlist[i] & next).size()));
      last = skinlist[i];
    }

    // Allocating inner region
    mypop(0, 0);

    st_pos.emplace_back(pos);
    sep_pos[0] = pos;

    skin_size.clear();
    // Allocating skinlist
    for (long i = 0; i < skinlist.size(); ++i) {
      long ppos = pos;
      if (pad_first[i])
        pos += calc_pad(skinlist[i]);
      if (skinlist[i].set != 0)
        mypop(0, skinlist[i]);
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
            i.first_pad = g.first_pad = pad_first[last] ? calc_pad(skinlist[last]) : 0;
            g.pos = pos;
            i.pos = st_pos[last];
          }
          if (pad_first[l])
            pos += calc_pad(skinlist[l]);
          mypop(n, skinlist[l]);
        } else if (last >= 0) {
          last = l;
          i.last_pad = g.last_pad = pad_first[last - 1] ? 0 : calc_pad(skinlist[last - 1]);
          g.skin_ed = (unsigned) last;
          g.len = pos - g.pos;
          ghost.emplace_back(g);
          i.len = st_pos[last] - i.pos;
          i.skin_ed = (unsigned) last;
          skin.emplace_back(i);
          last = -1;
        }
      if (last >= 0) {
        last = skinlist.size();
        i.last_pad = g.last_pad = pad_first[last - 1] ? 0 : calc_pad(skinlist[last - 1]);
        g.skin_ed = (unsigned) last;
        g.len = pos - g.pos;
        ghost.emplace_back(g);
        i.len = st_pos[skinlist.size()] - i.pos;
        i.skin_ed = (unsigned) last;
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

  /**
   * @brief Minimal PUT exchange without mmap
   * @param bStorage a brick storage created using this decomposition
   */
  void exchange(BrickStorage &bStorage) {
    std::vector<MPI_Request> requests(ghost.size() * 2);
    std::vector<MPI_Status> stats(requests.size());

#ifdef BARRIER_TIMESTEP
    MPI_Barrier(comm);
#endif

    double st = omp_get_wtime(), ed;

    for (int i = 0; i < ghost.size(); ++i) {
      // receive to ghost[i]
      MPI_Irecv(&(bStorage.dat.get()[(ghost[i].pos + ghost[i].first_pad) * bStorage.step]),
                (ghost[i].len - ghost[i].first_pad - ghost[i].last_pad) * bStorage.step * sizeof(bElem),
                MPI_CHAR, rank_map[ghost[i].neighbor.set], i, comm, &(requests[i << 1]));
      // send from skin[i]
      MPI_Isend(&(bStorage.dat.get()[(skin[i].pos + skin[i].first_pad) * bStorage.step]),
                (skin[i].len - skin[i].first_pad - skin[i].last_pad) * bStorage.step * sizeof(bElem),
                MPI_CHAR, rank_map[skin[i].neighbor.set], i, comm, &(requests[(i << 1) + 1]));
    }

    ed = omp_get_wtime();
    calltime += ed - st;
    st = ed;

    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

    ed = omp_get_wtime();
    waittime += ed - st;
  }

  /**
   * @brief @ref grid_access
   * @param i include padded ghost regions
   * @return
   */
  grid_access<mytype, dim, dim - 1> operator[](int i) {
    auto ga = grid_access<mytype, dim, dim>(this, 0);
    return ga[i];
  }

  /**
   * @brief Access the associated metadata
   * @return
   */
  BrickInfo<dim> getBrickInfo() {
    return *bInfo;
  }

  ~BrickDecomp() {
    delete[] grid;
    delete bInfo;
  }

  /**
   * @brief Create a view for PUT exchange (mmap)
   * @param bStorage a brick storage created using this decomposition
   * @return all information needed for exchange
   */
  ExchangeView exchangeView(BrickStorage bStorage) {
    // All Brick Storage are initialized with mmap for exchanging using views
    assert(bStorage.mmap_info != nullptr);

    // Generate all neighbor bitset
    std::vector<BitSet> neighbors;
    allneighbors(0, 1, dim, neighbors);

    // All each section is a pair of exchange with one neighbor
    std::vector<size_t> seclen;
    std::vector<size_t> first_pad;
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
      long first_pad_v = -1;
      long last_pad = 0;
      for (auto g: ghost) {
        if (g.neighbor.set == n.set) {
          if (first_pad_v < 0)
            first_pad_v = g.first_pad * bStorage.step * sizeof(bElem);
          last_pad = g.last_pad * bStorage.step * sizeof(bElem);
          packing.push_back(g.pos * bStorage.step * sizeof(bElem));
          size_t l = g.len * bStorage.step * sizeof(bElem);
          packing.push_back(l);
          len += l;
        }
      }
      recv.push_back(std::make_pair(rank_map[n.set], memfd->packed_pointer(packing)));
      len -= first_pad_v;
      len -= last_pad;
      first_pad.emplace_back(first_pad_v);
      seclen.push_back(len);
      packing.clear();
      // Send buffer
      BitSet in = !n;
      for (auto s: skin)
        if (s.neighbor.set == in.set && s.len) {
          packing.push_back(s.pos * bStorage.step * sizeof(bElem));
          packing.push_back(s.len * bStorage.step * sizeof(bElem));
        }
      send.push_back(std::make_pair(rank_map[in.set], memfd->packed_pointer(packing)));
    }

    return ExchangeView(comm, seclen, first_pad, send, recv);
  }

  /**
   * @brief Create a view for SHIFT exchange (mmap)
   * @param bStorage a brick storage created using this decomposition
   * @return all information needed for exchange
   */
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
              BitSet s = g.neighbor ^n;
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

  /**
   * @brief Exchange using MPI_Win (don't use)
   * @param bStorage
   * @param win
   */
  void exchange(BrickStorage bStorage, MPI_Win &win) {
    double st = omp_get_wtime(), ed;

    MPI_Win_fence(0, win);

    ed = omp_get_wtime();
    waittime += ed - st;
    st = ed;

    for (int i = 0; i < ghost.size(); ++i) {
      size_t len = ghost[i].len * bStorage.step * sizeof(bElem);
      // receive from remote
      MPI_Get(&(bStorage.dat.get()[ghost[i].pos * bStorage.step]), len, MPI_CHAR, rank_map[ghost[i].neighbor.set],
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

/**
 * @brief Populate neighbor-rank map for BrickDecomp using MPI_Comm
 * @tparam dim number of dimensions
 * @tparam BDims Brick dimensions
 * @param comm
 * @param bDecomp
 * @param neighbor
 * @param d current dimension
 * @param coo This rank's coo
 *
 * Example:
 * @code{.cpp}
 * populate(cart, brickDecomp, 0, 1, coo);
 * @endcode
 */
template<unsigned dim, unsigned ...BDims>
void populate(MPI_Comm &comm, BrickDecomp<dim, BDims...> &bDecomp, BitSet neighbor, int d, int *coo) {
  if (d > dim) {
    int rank;
    MPI_Cart_rank(comm, coo, &rank);
    bDecomp.rank_map[neighbor.set] = rank;
    return;
  }

  int dd = dim - d;
  int c = coo[dd];
  neighbor.flip(d);
  coo[dd] = c - 1;
  populate(comm, bDecomp, neighbor, d + 1, coo);
  neighbor.flip(d);
  // Not picked
  coo[dd] = c;
  populate(comm, bDecomp, neighbor, d + 1, coo);
  // Picked -
  neighbor.flip(-d);
  coo[dd] = c + 1;
  populate(comm, bDecomp, neighbor, d + 1, coo);
  coo[dd] = c;
}

/**
 * @brief Statistics collection for MPI programs
 */
typedef struct {
  double min, max, avg, sigma;
} mpi_stats;

/**
 * Collect a stats within a certain communicator to its root
 * @param stats a double represent a stat
 * @param comm communicator
 * @return a stats object
 */
inline mpi_stats mpi_statistics(double stats, MPI_Comm comm) {
  mpi_stats ret = {0, 0, 0, 0};
  if (comm == MPI_COMM_NULL)
    return ret;
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

/**
 * @brief pretty print an mpi_stats object
 */
inline std::ostream &operator<<(std::ostream &os, const mpi_stats &stats) {
  os << "[" << stats.min << ", " << stats.avg << ", " << stats.max << "]" << " (σ: " << stats.sigma << ")";
  return os;
}

/**
 * @brief Optimized surface ordering for 3D
 *
 * @code{.cpp}
 * BrickDecomp<3, 8,8,8> bDecomp({128,128,128}, 8);
 * bDecomp.initialize(skin3d_good);
 * @endcode
 */
extern std::vector<BitSet> skin3d_good;
extern std::vector<BitSet> skin3d_normal, skin3d_bad;

#endif //BRICK_BRICK_MPI_H
