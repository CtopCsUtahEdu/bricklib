/**
 * The most basic brick definition used for JIT
 */

#ifndef BRICK_H
#define BRICK_H

#include <memory>

template<unsigned base, unsigned exp>
struct static_power {
  static constexpr unsigned value = base * static_power<base, exp - 1>::value;
};

template<unsigned base>
struct static_power<base, 0> {
  static constexpr unsigned value = 1;
};

struct BrickStorage {
  std::shared_ptr<bElem> dat;
  long chunks;
  long step;
  void *mmap_info = nullptr;
};

template<unsigned dims>
struct BrickInfo {
  typedef unsigned (*adjlist)[static_power<3, dims>::value];
  adjlist adj;
  unsigned nbricks;
};

template<unsigned ... Ds>
struct Dim {
};

template<typename...>
struct Brick;

template<
    unsigned ... BDims,
    unsigned ... Folds>
struct Brick<Dim<BDims...>, Dim<Folds...> > {
  typedef BrickInfo<sizeof...(BDims)> myBrickInfo;

  myBrickInfo *bInfo;
  unsigned step;
  bElem *dat;
  BrickStorage bStorage;
};

#endif //BRICK_H
