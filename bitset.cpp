//
// Created by joe on 12/30/18.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>

#include "bitset.h"

// axis 1 - N
// Positive denote the positive direction
// Negative denote the negative direction
long N;

void populate(std::vector<BitSet> &m, long cnt, BitSet cur, const BitSet &tot) {
  if (cnt == 0) {
    m.emplace_back(cur);
    return;
  }
  for (long i = 1; i <= N; ++i)
    if (cur.get(i) == cur.get(-i)) {
      if (tot.get(i)) {
        BitSet nxt = cur;
        populate(m, cnt - 1, nxt.flip(i), tot);
      }
      if (tot.get(-i)) {
        BitSet nxt = cur;
        populate(m, cnt - 1, nxt.flip(-i), tot);
      }
    } else
      break;
}

inline uint64_t fac(long n) {
  uint64_t r = 1;
  for (long i = 1; i <= n; ++i)
    r *= (uint64_t) i;
  return r;
}

inline uint64_t C(long n, long i) {
  return fac(n) / fac(i) / fac(n - i);
}

long gcd(long a, long b) {
  if (b == 0)
    return a;
  return gcd(b, a % b);
}

typedef std::vector<std::pair<std::vector<long>, long>> Segments;

int main() {
  BitSet bs;
  std::cout << "N: ";
  std::cin >> N;
  std::cout << "Edges to exchange: ";
  std::cin >> bs;
  std::cout << bs << std::endl;
  // std::cout << "Total: " << (long) std::pow(3l, N) - 1 << std::endl;

  Segments segments;

  for (long i = N; i > 0; --i) {
    std::vector<BitSet> m;
    populate(m, i, 0ul, bs);
    std::cout << C(N, i) * (1 << i) << ":" << m.size() << std::endl;
    for (auto s:m)
      std::cout << s;
    std::cout << std::endl;

    Segments new_segment;
    long n = m.size();

    long tot = 0;
    for (const auto &s:segments)
      tot += s.second;

    if (tot * 2 <= n) {
      // All iji + a few i
      for (auto s: segments) {
        new_segment.push_back({{}, s.second});
        new_segment.back().first.push_back(i);
        for (auto v: s.first)
          new_segment.back().first.push_back(v);
        new_segment.back().first.push_back(i);
      }

      if (n > tot * 2) {
        new_segment.push_back({{}, n - tot * 2});
        new_segment.back().first.push_back(i);
      }
    } else if (tot >= n) {
      // All ij + a few j
      new_segment.push_back({{}, 1});
      for (const auto &s: segments) {
        long l = std::min(s.second, n);
        for (long j = 0; j < l; ++j) {
          new_segment.back().first.push_back(i);
          for (auto v: s.first)
            new_segment.back().first.push_back(v);
          n = n - l;
        }
        l = s.second - l;
        for (long j = 0; j < l; ++j) {
          for (auto v: s.first)
            new_segment.back().first.push_back(v);
        }
      }
    } else {
      // Find gcd
      long c = gcd(tot, n);
      // n = cu
      // tot = cv
      // u > v
      // c(u - v - 1) * iji
      // c(v - (u - v - 1)) * ijiji..ji
      long u = n / c;
      long v = tot / c;
      long iji = c * (u - v - 1);
      long ijiji = c;
      long s = v - (u - v - 1);
      auto it = segments.begin();
      long l = 0;
      while (it != segments.end()) {
        l = std::min(it->second, iji);
        if (l > 0) {
          new_segment.push_back({{}, l});
          new_segment.back().first.push_back(i);
          for (auto v: it->first)
            new_segment.back().first.push_back(v);
          new_segment.back().first.push_back(i);
        }
        iji -= l;
        l = it->second - l;
        if (iji == 0)
          break;
        ++it;
      }
      auto getnext = [&it, &l]() -> Segments::iterator {
        if (l > 0) {
          --l;
          return it;
        }
        ++it;
        l = it->second - 1;
        return it;
      };
      while (ijiji > 0) {
        if (l == 0) {
          ++it;
          l = it->second;
        }
        if (l < s) {
          // This will split
          new_segment.push_back({{}, 1});
          for (long j = 0; j < s; ++j) {
            new_segment.back().first.push_back(i);
            auto n = getnext();
            for (auto v: n->first)
              new_segment.back().first.push_back(v);
          }
          new_segment.back().first.push_back(i);
          ijiji -= new_segment.back().second;
        } else {
          new_segment.push_back({{}, l / s});
          for (long j = 0; j < s; ++j) {
            new_segment.back().first.push_back(i);
            for (auto v: it->first)
              new_segment.back().first.push_back(v);
          }
          l = l % s;
          new_segment.back().first.push_back(i);
          ijiji -= new_segment.back().second;
        }
      }
    }

    segments = new_segment;
  }

  long calls = 0;
  long tot = 0;
  for (const auto &s: segments) {
    long t = 0;
    long l = 0;
    for (auto v: s.first) {
      long c = (l != v ? std::min(l, v) : (l - 1));
      t = t + (1l << v) - (1l << c);
      l = v;
      std::cout << v;
    }
    std::cout << ":" << s.second << std::endl;
    calls += t * s.second;
    tot += s.first.size() * s.second;
  }
  std::cout << "Calls/Neighbors: " << calls << "/" << tot << std::endl;
  return 0;
};
