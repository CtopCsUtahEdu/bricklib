//
// Created by ztuowen on 2/8/19.
//

#ifndef BRICK_BITSET_H
#define BRICK_BITSET_H

#include <initializer_list>
#include <ostream>

typedef long axis;

const long zero = 31;

inline uint64_t to_set(long pos) {
  if (pos < 0)
    pos = zero - pos;
  return 1ul << (uint64_t)pos;
}

typedef struct BitSet {
  uint64_t set;

  BitSet() : set(0) {}

  BitSet(uint64_t s) : set(s) {
  }

  BitSet(std::initializer_list<int> l) {
    set = 0;
    for (auto p: l)
      set ^= to_set(p);
  }

  inline BitSet &flip(long pos) {
    set ^= to_set(pos);
    return *this;
  }

  inline long size() const {
    return __builtin_popcount(set);
  }

  inline bool get(long pos) const {
    return (set & to_set(pos)) > 0;
  }

  inline BitSet operator&(BitSet a) const {
    return set & a.set;
  }

  inline BitSet operator|(BitSet a) const {
    return set | a.set;
  }

  inline BitSet operator^(BitSet a) const {
    return set ^ a.set;
  }

  inline operator bool() const {
    return set != 0ul;
  }

  inline bool operator<=(BitSet a) const {
    return (set & a.set) == set;
  }

  inline bool operator>=(BitSet a) const {
    return (set & a.set) == a.set;
  }

  inline BitSet operator!() const {
    BitSet ret(0);

    uint64_t mask = (1ul << (uint64_t)(zero + 1)) - 1ul;

    ret.set = ((set & mask) << (uint64_t) zero) | (set >> (uint64_t) zero);
    return ret;
  }
} BitSet;


inline std::ostream &operator<<(std::ostream &os, const BitSet &bs) {
  os << "{";
  for (long i = 1; i < 32; ++i) {
    if (bs.get(i)) os << i << "+";
    if (bs.get(-i)) os << i << "-";
  }
  os << "}";
  return os;
}

inline std::istream &operator>>(std::istream &is, BitSet &bs) {
  std::string str;
  is >> str;
  if (str[0] != '{')
    throw std::runtime_error("Err");
  BitSet set;
  long pos = 1;
  while (str[pos] != '}') {
    long d = 0;
    while (isdigit(str[pos])) {
      d = d * 10 + str[pos] - '0';
      ++pos;
    }
    if (str[pos] == '-')
      d = -d;
    else if (str[pos] != '+')
      throw std::runtime_error("Err");
    ++pos;
    set.flip(d);
  }
  bs = set;
  return is;
}

#endif //BRICK_BITSET_H
