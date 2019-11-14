//
// Created by ztuowen on 2/17/19.
//

#include "brick-mpi.h"

double packtime, calltime, waittime, movetime, calctime;

void allneighbors(BitSet cur, long idx, long dim, std::vector<BitSet> &neighbors) {
  if (idx > dim) {
    neighbors.emplace_back(cur);
    return;
  }
  // Picked +
  cur.flip(idx);
  allneighbors(cur, idx + 1, dim, neighbors);
  cur.flip(idx);
  // Not picked
  allneighbors(cur, idx + 1, dim, neighbors);
  // Picked -
  cur.flip(-idx);
  allneighbors(cur, idx + 1, dim, neighbors);
}

std::vector<BitSet> skin3d_good = {
    {1},
    {1,  -3},
    {1,  2,  -3},
    {1,  2},
    {1,  2,  3},
    {2,  3},
    {2},
    {2,  -3},
    {-1, 2,  -3},
    {-1, 2},
    {-1, 2,  3},
    {-1, 3},
    {-1},
    {-3},
    {-1, -3},
    {-1, -2, -3},
    {-1, -2},
    {-1, -2, 3},
    {-2, 3},
    {-2},
    {-2, -3},
    {1,  -2, -3},
    {1,  -2},
    {1,  -2, 3},
    {1,  3},
    {3}
};

std::vector<BitSet> skin3d_normal = {
    {-1, -2, -3},
    {-2, -3},
    {1,  -2, -3},
    {-1, -3},
    {-3},
    {1,  -3},
    {-1, 2,  -3},
    {2,  -3},
    {1,  2,  -3},
    {-1, -2},
    {-2},
    {1,  -2},
    {-1},
    {},
    {1},
    {-1, 2},
    {2},
    {1,  2},
    {-1, -2, 3},
    {-2, 3},
    {1,  -2, 3},
    {-1, 3},
    {3},
    {1,  3},
    {-1, 2,  3},
    {2,  3},
    {1,  2,  3}
};

std::vector<BitSet> skin3d_bad = {
    {-1, -2, -3},
    {},
    {-2, -3},
    {},
    {1,  -2, -3},
    {},
    {-1, -3},
    {},
    {-3},
    {},
    {1,  -3},
    {},
    {-1, 2,  -3},
    {},
    {2,  -3},
    {},
    {1,  2,  -3},
    {},
    {-1, -2},
    {},
    {-2},
    {},
    {1,  -2},
    {},
    {-1},
    {},
    {1},
    {},
    {-1, 2},
    {},
    {2},
    {},
    {1,  2},
    {},
    {-1, -2, 3},
    {},
    {-2, 3},
    {},
    {1,  -2, 3},
    {},
    {-1, 3},
    {},
    {3},
    {},
    {1,  3},
    {},
    {-1, 2,  3},
    {},
    {2,  3},
    {},
    {1,  2,  3}
};
