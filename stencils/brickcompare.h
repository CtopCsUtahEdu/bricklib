//
// Created by Tuowen Zhao on 3/14/19.
//

#ifndef BRICK_BRICKCOMPARE_H
#define BRICK_BRICKCOMPARE_H

#include <iostream>
#include <cmath>
#include "bricksetup.h"
#include "cmpconst.h"

extern bool compareBrick_b;

#pragma omp threadprivate(compareBrick_b)

template<unsigned dims, typename T>
inline bool
compareBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost, bElem
*arr,
    unsigned *grid_ptr,
             T &brick) {
  bool ret = true;
  auto f = [&ret](bElem &brick, const bElem *arr) -> void {
    double diff = std::abs(brick - *arr);
    bool r = (diff < TOLERANCE) || (diff < (std::abs(brick) + std::abs(*arr)) * TOLERANCE);
    compareBrick_b = (compareBrick_b && r);
  };

#pragma omp parallel default(none)
  {
    compareBrick_b = true;
  }

  iter_grid<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, f);

#pragma omp parallel default(none) shared(ret)
  {
#pragma omp critical
    {
      ret = ret && compareBrick_b;
    }
  }

  return ret;
}

template<unsigned dims, typename T>
inline bool
compareBrick(const std::vector<long> &dimlist, bElem *arr, unsigned *grid_ptr,
             T &brick) {
  std::vector<long> padding(dimlist.size(), 0);
  std::vector<long> ghost(dimlist.size(), 0);

  return compareBrick<dims, T>(dimlist, padding, ghost, arr, grid_ptr, brick);
}

#endif //BRICK_BRICKCOMPARE_H
