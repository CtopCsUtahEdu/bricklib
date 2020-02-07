#include "multiarray.h"
#include "cmpconst.h"
#include <iostream>
#include <random>

namespace {
  std::mt19937_64 *mt = nullptr;
  std::uniform_real_distribution<bElem> *u = nullptr;

#pragma omp threadprivate(mt)
#pragma omp threadprivate(u)

  bElem randD() {
    if (mt == nullptr) {
#pragma omp critical
      {
        std::random_device r;
        mt = new std::mt19937_64(r());
        u = new std::uniform_real_distribution<bElem>(0, 1);
      }
    }
    return (*u)(*mt);
  }
}

bElem *uninitArray(const std::vector<long> &list, long &size) {
  size = 1;
  for (auto i: list)
    size *= i;
  return (bElem *) aligned_alloc(ALIGN, size * sizeof(bElem));
}

bElem *randomArray(const std::vector<long> &list) {
  long size;
  bElem *arr = uninitArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
    arr[l] = randD();
  return arr;
}

bElem *zeroArray(const std::vector<long> &list) {
  long size;
  bElem *arr = uninitArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
    arr[l] = 0.0;
  return arr;
}

bool compareArray(const std::vector<long> &list, bElem *arrA, bElem *arrB) {
  long size = 1;
  for (auto i: list)
    size *= i;
  bool same = true;
#pragma omp parallel for reduction(&&: same)
  for (long l = 0; l < size; ++l) {
    bElem diff = std::abs(arrA[l] - arrB[l]);
    bool r = (diff < BRICK_TOLERANCE) || (diff < (std::abs(arrA[l]) + std::abs(arrB[l])) * BRICK_TOLERANCE);
    same = same && r;
  }
  return same;
}
