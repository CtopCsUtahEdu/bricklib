//
// Created by Tuowen Zhao on 6/14/19.
//

#ifndef BRICK_ARRAY_MPI_H
#define BRICK_ARRAY_MPI_H

#include "brick-mpi.h"
#include <mpi.h>

template<unsigned dim>
inline bElem *pack(bElem *arr, BitSet neighbor, bElem *buffer_out, const std::vector<unsigned long> &arrstride,
                   const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost) {
  // Inner region
  long sec = 0;
  long st = 0;
  int d = dim - 1;
  if (neighbor.get(dim)) {
    sec = 1;
    st = padding[d] + dimlist[d];
  } else if (neighbor.get(-(int) dim)) {
    sec = -1;
    st = padding[d] + ghost[d];
  }
  if (sec) {
    for (unsigned i = 0; i < ghost[d]; ++i)
      buffer_out = pack<dim - 1>(arr + arrstride[d] * (st + i), neighbor, buffer_out,
                                 arrstride, dimlist, padding, ghost);
  } else {
    for (unsigned i = 0; i < dimlist[d]; ++i)
      buffer_out = pack<dim - 1>(arr + arrstride[d] * (padding[d] + ghost[d] + i), neighbor, buffer_out,
                                 arrstride, dimlist, padding, ghost);
  }

  return buffer_out;
}

template<>
inline bElem *pack<1>(bElem *arr, BitSet neighbor, bElem *buffer_out, const std::vector<unsigned long> &arrstride,
                      const std::vector<long> &dimlist, const std::vector<long> &padding,
                      const std::vector<long> &ghost) {
  // Inner region
  long sec = 0;
  long st = 0;
  int d = 0;
  if (neighbor.get(1)) {
    sec = 1;
    st = padding[d] + dimlist[d];
  } else if (neighbor.get(-1)) {
    sec = -1;
    st = padding[d] + ghost[d];
  }
  if (sec != 0) {
    elemcpy(buffer_out, arr + st, ghost[d]);
    return buffer_out + ghost[d];
  } else {
    elemcpy(buffer_out, arr + padding[d] + ghost[d], dimlist[d]);
    return buffer_out + dimlist[d];
  }
}

template<unsigned dim>
inline bElem *unpack(bElem *arr, BitSet neighbor, bElem *buffer_recv, const std::vector<unsigned long> &arrstride,
                     const std::vector<long> &dimlist, const std::vector<long> &padding,
                     const std::vector<long> &ghost) {
  // Inner region
  long sec = 0;
  long st = 0;
  int d = (int) dim - 1;
  if (neighbor.get(dim)) {
    sec = 1;
    st = padding[d] + dimlist[d] + ghost[d];
  } else if (neighbor.get(-(int) dim)) {
    sec = -1;
    st = padding[d];
  }
  if (sec) {
    for (unsigned i = 0; i < ghost[d]; ++i)
      buffer_recv = unpack<dim - 1>(arr + arrstride[d] * (st + i), neighbor, buffer_recv,
                                    arrstride, dimlist, padding, ghost);
  } else {
    for (unsigned i = 0; i < dimlist[d]; ++i)
      buffer_recv = unpack<dim - 1>(arr + arrstride[d] * (padding[d] + ghost[d] + i), neighbor, buffer_recv,
                                    arrstride, dimlist, padding, ghost);
  }
  return buffer_recv;
}

template<>
inline bElem *unpack<1>(bElem *arr, BitSet neighbor, bElem *buffer_recv, const std::vector<unsigned long> &arrstride,
                        const std::vector<long> &dimlist, const std::vector<long> &padding,
                        const std::vector<long> &ghost) {
  // Inner region
  long sec = 0;
  long st = 0;
  int d = 0;
  if (neighbor.get(1)) {
    sec = 1;
    st = padding[d] + dimlist[d] + ghost[d];
  } else if (neighbor.get(-1)) {
    sec = -1;
    st = padding[d];
  }
  if (sec) {
    elemcpy(arr + st, buffer_recv, ghost[d]);
    return buffer_recv + ghost[d];
  } else {
    elemcpy(arr + padding[d] + ghost[d], buffer_recv, dimlist[d]);
    return buffer_recv + dimlist[d];
  }
}


inline unsigned
evalsize(BitSet region, const std::vector<long> &dimlist, const std::vector<long> &ghost, bool inner = true) {
  // Inner region
  unsigned size = 1;
  for (int i = 1; i <= (int) dimlist.size(); ++i)
    if (region.get(i) || region.get(-i))
      size = size * ghost[i - 1];
    else
      size = size * (dimlist[i - 1] - (inner ? 2 * ghost[i - 1] : 0));
  return size;
}

// ID is used to prevent message mismatch from messages with the same node, low performance only for validation testing.
template<unsigned dim>
void exchangeArr(bElem *arr, const MPI_Comm &comm, std::unordered_map<uint64_t, int> &rank_map,
                 const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost) {
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, dim, neighbors);
  neighbors.erase(neighbors.begin() + (neighbors.size() / 2));
  std::vector<bElem *> buffers_out(neighbors.size(), nullptr);
  std::vector<bElem *> buffers_recv(neighbors.size(), nullptr);
  std::vector<unsigned long> tot(neighbors.size());
  std::vector<MPI_Request> requests(neighbors.size() * 2);

  std::vector<unsigned long> arrstride(dimlist.size());
  unsigned long stri = 1;

  for (int i = 0; i < arrstride.size(); ++i) {
    arrstride[i] = stri;
    stri = stri * ((padding[i] + ghost[i]) * 2 + dimlist[i]);
  }

  double st = omp_get_wtime(), ed;

  for (int i = 0; i < (int) neighbors.size(); ++i) {
    tot[i] = (unsigned long) evalsize(neighbors[i], dimlist, ghost, false);
    buffers_recv[i] = new bElem[tot[i]];
    buffers_out[i] = new bElem[tot[i]];
    // receive to ghost[i]
    MPI_Irecv(buffers_recv[i], (int) (tot[i] * sizeof(bElem)), MPI_CHAR, rank_map[neighbors[i].set],
              (int) neighbors.size() - i - 1, comm, &(requests[i * 2]));
  }

  ed = omp_get_wtime();
  calltime += ed - st;
  st = ed;

  // Pack
#pragma omp parallel for
  for (int i = 0; i < (int) neighbors.size(); ++i)
    pack<dim>(arr, neighbors[i], buffers_out[i], arrstride, dimlist, padding, ghost);

  ed = omp_get_wtime();
  packtime += ed - st;
  st = ed;

  for (int i = 0; i < (int) neighbors.size(); ++i)
    MPI_Isend(buffers_out[i], (int) (tot[i] * sizeof(bElem)), MPI_CHAR, rank_map[neighbors[i].set], i, comm,
              &(requests[i * 2 + 1]));

  ed = omp_get_wtime();
  calltime += ed - st;
  st = ed;

  // Wait
  std::vector<MPI_Status> stats(requests.size());
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

  ed = omp_get_wtime();
  waittime += ed - st;
  st = ed;

  // Unpack
#pragma omp parallel for
  for (int i = 0; i < (int) neighbors.size(); ++i)
    unpack<dim>(arr, neighbors[i], buffers_recv[i], arrstride, dimlist, padding, ghost);

  ed = omp_get_wtime();
  packtime += ed - st;

  // Cleanup
  for (auto b: buffers_out)
    delete[] b;
  for (auto b: buffers_recv)
    delete[] b;
}

MPI_Datatype pack_type(BitSet neighbor, const std::vector<long> &dimlist, const std::vector<long> &padding,
                       const std::vector<long> &ghost) {
  std::vector<int> size(3), subsize(3), start(3);
  for (int dd = 0; dd < dimlist.size(); ++dd) {
    int d = dimlist.size() - dd - 1;
    size[d] = dimlist[d] + 2 * (padding[d] + ghost[d]);
    int dim = dd + 1;
    long sec = 0;
    if (neighbor.get(dim)) {
      sec = 1;
      start[d] = padding[d] + dimlist[d];
    } else if (neighbor.get(-(int) dim)) {
      sec = -1;
      start[d] = padding[d] + ghost[d];
    }
    if (sec) {
      subsize[d] = ghost[d];
    } else {
      subsize[d] = dimlist[d];
      start[d] = padding[d] + ghost[d];
    }
  }
  MPI_Datatype ret;
  MPI_Type_create_subarray(3, size.data(), subsize.data(), start.data(), MPI_ORDER_C, MPI_DOUBLE, &ret);
  return ret;
}

MPI_Datatype unpack_type(BitSet neighbor, const std::vector<long> &dimlist, const std::vector<long> &padding,
                         const std::vector<long> &ghost) {
  std::vector<int> size(3), subsize(3), start(3);
  for (int dd = 0; dd < dimlist.size(); ++dd) {
    int d = dimlist.size() - dd - 1;
    size[d] = dimlist[d] + 2 * (padding[d] + ghost[d]);
    int dim = dd + 1;
    long sec = 0;
    if (neighbor.get(dim)) {
      sec = 1;
      start[d] = padding[d] + dimlist[d] + ghost[d];
    } else if (neighbor.get(-(int) dim)) {
      sec = -1;
      start[d] = padding[d];
    }
    if (sec) {
      subsize[d] = ghost[d];
    } else {
      subsize[d] = dimlist[d];
      start[d] = padding[d] + ghost[d];
    }
  }
  MPI_Datatype ret;
  MPI_Type_create_subarray(3, size.data(), subsize.data(), start.data(), MPI_ORDER_C, MPI_DOUBLE, &ret);
  return ret;
}

template<unsigned dim>
void exchangeArrPrepareTypes(std::unordered_map<uint64_t, MPI_Datatype> &stypemap,
                          std::unordered_map<uint64_t, MPI_Datatype> &rtypemap,
                          const std::vector<long> &dimlist, const std::vector<long> &padding,
                          const std::vector<long> &ghost) {
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, dim, neighbors);
  neighbors.erase(neighbors.begin() + (neighbors.size() / 2));
  std::vector<MPI_Request> requests(neighbors.size() * 2);

  std::vector<unsigned long> arrstride(dimlist.size());
  unsigned long stri = 1;

  for (int i = 0; i < arrstride.size(); ++i) {
    arrstride[i] = stri;
    stri = stri * ((padding[i] + ghost[i]) * 2 + dimlist[i]);
  }

  for (int i = 0; i < (int) neighbors.size(); ++i) {
    MPI_Datatype MPI_rtype = unpack_type(neighbors[i], dimlist, padding, ghost);
    MPI_Type_commit(&MPI_rtype);
    rtypemap[neighbors[i].set] = MPI_rtype;
    MPI_Datatype MPI_stype = pack_type(neighbors[i], dimlist, padding, ghost);
    MPI_Type_commit(&MPI_stype);
    stypemap[neighbors[i].set] = MPI_stype;
  }
}

// Using data types
template<unsigned dim>
void exchangeArrTypes(bElem *arr, const MPI_Comm &comm, std::unordered_map<uint64_t, int> &rank_map,
                      std::unordered_map<uint64_t, MPI_Datatype> &stypemap,
                      std::unordered_map<uint64_t, MPI_Datatype> &rtypemap) {
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, dim, neighbors);
  neighbors.erase(neighbors.begin() + (neighbors.size() / 2));
  std::vector<MPI_Request> requests(neighbors.size() * 2);

  int rank;
  MPI_Comm_rank(comm, &rank);

  double st = omp_get_wtime(), ed;

  for (int i = 0; i < (int) neighbors.size(); ++i) {
    MPI_Irecv(arr, 1, rtypemap[neighbors[i].set], rank_map[neighbors[i].set],
              (int) neighbors.size() - i - 1, comm, &(requests[i * 2]));
    MPI_Isend(arr, 1, stypemap[neighbors[i].set], rank_map[neighbors[i].set], i, comm, &(requests[i * 2 + 1]));
  }

  ed = omp_get_wtime();
  calltime += ed - st;
  st = ed;

  // Wait
  std::vector<MPI_Status> stats(requests.size());
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

  ed = omp_get_wtime();
  waittime += ed - st;
}

typedef struct {
  bElem *arr;
  std::unordered_map<uint64_t, int> *rank_map;
  std::unordered_map<uint64_t, int> *id_map;
  int id;
} ArrExPack;

template<unsigned dim>
void exchangeArrAll(std::vector<ArrExPack> arr, const MPI_Comm &comm,
                    const std::vector<long> &dimlist, const std::vector<long> &padding,
                    const std::vector<long> &ghost) {
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, dim, neighbors);
  neighbors.erase(neighbors.begin() + (neighbors.size() / 2));
  std::vector<bElem *> buffers_out(arr.size() * neighbors.size(), nullptr);
  std::vector<bElem *> buffers_recv(arr.size() * neighbors.size(), nullptr);
  std::vector<unsigned long> tot(neighbors.size());
  std::vector<MPI_Request> requests(arr.size() * neighbors.size() * 2);

  std::vector<unsigned long> arrstride(dimlist.size());
  unsigned long stri = 1;

  for (int i = 0; i < arrstride.size(); ++i) {
    arrstride[i] = stri;
    stri = stri * ((padding[i] + ghost[i]) * 2 + dimlist[i]);
  }

  double st = omp_get_wtime(), ed;

  for (int i = 0; i < (int) neighbors.size(); ++i) {
    tot[i] = (unsigned long) evalsize(neighbors[i], dimlist, ghost, false);
    for (int s = 0; s < arr.size(); ++s) {
      buffers_recv[i + s * neighbors.size()] = new bElem[tot[i]];
      buffers_out[i + s * neighbors.size()] = new bElem[tot[i]];
      // receive to ghost[i]
      MPI_Irecv(buffers_recv[i + s * neighbors.size()], (int) (tot[i] * sizeof(bElem)), MPI_CHAR,
                arr[s].rank_map->at(neighbors[i].set),
                arr[s].id_map->at(neighbors[i].set) * 100 + (int) neighbors.size() - i - 1,
                comm, &(requests[i * 2 + s * neighbors.size() * 2]));
    }
  }

  ed = omp_get_wtime();
  calltime += ed - st;
  st = ed;

  // Pack
#pragma omp parallel for
  for (int i = 0; i < (int) neighbors.size(); ++i)
    for (int s = 0; s < arr.size(); ++s)
      pack<dim>(arr[s].arr, neighbors[i], buffers_out[i + s * neighbors.size()], arrstride, dimlist, padding, ghost);

  ed = omp_get_wtime();
  packtime += ed - st;
  st = ed;

  for (int i = 0; i < (int) neighbors.size(); ++i)
    for (int s = 0; s < arr.size(); ++s)
      MPI_Isend(buffers_out[i + s * neighbors.size()], (int) (tot[i] * sizeof(bElem)), MPI_CHAR,
                arr[s].rank_map->at(neighbors[i].set), arr[s].id * 100 + i, comm,
                &(requests[i * 2 + s * neighbors.size() * 2 + 1]));

  ed = omp_get_wtime();
  calltime += ed - st;
  st = ed;

  // Wait
  std::vector<MPI_Status> stats(requests.size());
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

  ed = omp_get_wtime();
  waittime += ed - st;
  st = ed;

  // Unpack
#pragma omp parallel for
  for (int i = 0; i < (int) neighbors.size(); ++i)
    for (int s = 0; s < arr.size(); ++s)
      unpack<dim>(arr[s].arr, neighbors[i], buffers_recv[i + s * neighbors.size()], arrstride, dimlist, padding, ghost);

  ed = omp_get_wtime();
  packtime += ed - st;

  // Cleanup
  for (auto b: buffers_out)
    delete[] b;
  for (auto b: buffers_recv)
    delete[] b;
}

#endif //BRICK_ARRAY_MPI_H
