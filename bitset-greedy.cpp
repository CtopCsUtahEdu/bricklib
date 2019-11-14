/*
 * Created by Tuowen Zhao on 8/1/19.
 * Testing the use of greedy algorithm for determining the skinlist.
 */

#include <iostream>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <tuple>
#include <algorithm>

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

void populate_seg(std::unordered_set<uint64_t> &segset, BitSet n, const std::vector<BitSet> &neighbors) {
  if (segset.find(n.set) == segset.end())
    segset.insert(n.set);
  else
    return;

  for (const auto i: neighbors) {
    if ((!(n <= i)) && ((!i) & n).set == 0ul)
      populate_seg(segset, n | i, neighbors);
  }
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

typedef std::unordered_set<uint64_t> UnorderedBSet;

typedef struct SkinSearch {
  std::vector<BitSet> bestSkin;
  std::vector<BitSet> neighbors;

  long num_link(BitSet left, BitSet right) {
    if ((!left) & right)
      return 0;

    BitSet common = left & right;
    long ret = 0;
    for (auto n: neighbors)
      if (n <= common)
        ++ret;
    return ret;
  }

  template<typename F>
  void permute(std::vector<BitSet> rem, long idx, long cur, BitSet last, BitSet right, F &update) {
    if (idx == rem.size()) {
      update(rem, cur + num_link(last, right));
      return;
    }

    for (long i = idx; i < rem.size(); ++i) {
      BitSet tmp = rem[idx];
      rem[idx] = rem[i];
      rem[i] = tmp;

      long ncur = cur + num_link(last, rem[idx]);
      permute(rem, idx + 1, ncur, rem[idx], right, update);

      tmp = rem[idx];
      rem[idx] = rem[i];
      rem[i] = tmp;
    }
  }

  long _divide(const UnorderedBSet &remaining, std::unordered_set<uint64_t> nset,
               std::vector<BitSet> &ret, BitSet left, BitSet right) {
    // Only search for CAP(N_i) = N_i
    std::unordered_set<uint64_t> frozen_set = nset;

    // Clean up the neighbor set
    BitSet bSet;
    long bSetv = 0;
    for (BitSet n: frozen_set) {
      BitSet cup;
      bool first = true;
      long cnt = 0;
      for (BitSet r: remaining)
        if (n <= r) {
          ++cnt;
          if (first) {
            first = false;
            cup = r;
          } else {
            cup = r & cup;
          }
        }

      if (cup.set != n.set || cnt == nset.size())
        nset.erase(n.set);
      else {
        if (cnt > bSetv) {
          bSetv = cnt;
          bSet = n;
        }
      }
    }

    // std::cout << "remaining: " << remaining.size() << " nset: " << nset.size() << std::endl;
    ret.clear();
    if (nset.empty()) {
      std::vector<BitSet> rem, stack;
      for (BitSet r: remaining)
        rem.emplace_back(r);
      long best = -1;
      auto update = [&ret, &best](const std::vector<BitSet> &seq, long v) -> void {
        if (v > best) {
          best = v;
          ret = seq;
        }
      };
      permute(rem, 0, 0, left, right, update);
      return best;
    }

    // re-freeze
    frozen_set = nset;
    UnorderedBSet split_left, split_right;
    std::vector<BitSet> best, div_left, div_right;
    long bestv = -1;
    for (BitSet n: frozen_set) {

      // Skip neighbors that are subset of another
      bool skip = false;
      for (BitSet ni: frozen_set)
        if (ni <= n && n.set != ni.set) {
          skip = true;
          break;
        }
      if (skip) continue;

      split_left.clear();
      split_right.clear();

      for (BitSet r: remaining)
        if (n <= r) {
          split_left.insert(r.set);
        } else
          split_right.insert(r.set);

      // if (split_left.size() < bSetv)
      //  continue;

      nset.erase(n.set);

      // take the fix from left/right doesn't matter much
      UnorderedBSet left_rem = split_left;
      for (auto fix: split_left) {
        left_rem.erase(fix);

        long r = _divide(left_rem, nset, div_left, fix, left) +
                 _divide(split_right, nset, div_right, fix, right);

        if (r > bestv) {
          bestv = r;

          std::reverse(div_left.begin(), div_left.end());
          best = div_left;
          best.emplace_back(fix);
          best.insert(best.end(), div_right.begin(), div_right.end());
        }

        left_rem.insert(fix);
      }

      nset.insert(n.set);
    }
    ret = best;
    return bestv;
  }

  long divide(const std::vector<BitSet> &segments, std::vector<BitSet> &best) {
    std::unordered_set<uint64_t> nset;
    for (auto n: neighbors)
      nset.insert(n.set);

    long tot = 0, meed = 0;
    BitSet ms = 0;
    for (auto s:segments) {
      int need = 0;
      for (auto n: neighbors)
        if (n <= s)
          ++need;
      if (need > meed) {
        meed = need;
        ms = s;
      }
      tot += need;
    }

    UnorderedBSet segset;
    for (auto s: segments)
      if (s.set != ms.set)
        segset.insert(s.set);
    segset.insert(0);

    std::vector<BitSet> temp;
    _divide(segset, nset, temp, ms, ms);
    // fix it by cutting at 0;
    best.clear();
    BitSet last = 0;
    long ret = 0;
    bool start = false;
    for (long i = 0; i < temp.size(); ++i) {
      if (start) {
        ret += num_link(last, temp[i]);
        last = temp[i];
        best.emplace_back(temp[i]);
      }
      if (temp[i].set == 0ul) {
        start = true;
      }
    }

    best.emplace_back(ms);
    ret += num_link(last, ms);
    last = ms;

    for (long i =0; i < temp.size(); ++i)
      if (temp[i].set == 0ul)
        break;
      else {
        ret += num_link(last, temp[i]);
        last = temp[i];
        best.emplace_back(temp[i]);
      }
    return tot - ret;
  }

  explicit SkinSearch(std::vector<BitSet> &neighbors) : neighbors(neighbors), bestSkin() {
  }
} SkinSearch;

int main() {
  BitSet bs = 0ul;
  std::cout << "N: " << 3 << std::endl;
  // std::cin >> N;
  N = 3;

  // Right now we simply list all exchanging neighbors

  std::vector<BitSet> neighbors;

  for (long i = 1; i <= N; ++i) {
    bs.flip(i);
    bs.flip(-i);
  }

  std::cout << bs << std::endl;
  // std::cout << "Total: " << (long) std::pow(3l, N) - 1 << std::endl;

  // Populate all neighbors
  for (long i = 1; i <= N; ++i)
    populate(neighbors, i, 0ul, bs);

  // Populate all segments
  std::vector<BitSet> segments;
  {
    std::unordered_set<uint64_t> segset;
    for (auto n: neighbors)
      populate_seg(segset, n, neighbors);
    for (auto k: segset)
      segments.emplace_back(k);
  }

  SkinSearch skinSearch(neighbors);

  std::vector<BitSet> skinlist;
  long best = skinSearch.divide(segments, skinlist);

  std::cout << best << std::endl;
  for (auto s: skinlist)
    std::cout << s << " ";
  std::cout << std::endl;
  return 0;
};
