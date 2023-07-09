/**
 * @file
 * @brief Set using bitfield
 */

#ifndef BRICK_BITSET_H
#define BRICK_BITSET_H

#include <initializer_list>
#include <cstdint>
#include <ostream>

/**
 * @brief Set using bitfield
 *
 * Numbers are translated into elements and then stored in the bitfield.
 * Can only held numbers in \f$[-32,31]\f$.
 */
typedef struct BitSet {
  static const long zero = 31; ///< negative zero start at 31 (1<<31).

  /**
   * @brief Turn number into corresponding element of set
   * @param pos Input number
   * @return an element of set
   */
  static inline uint64_t to_set(long pos) {
    if (pos < 0)
      pos = zero - pos;
    return 1ul << (uint64_t) pos;
  }

  /// The bitfield of this set
  uint64_t set;

  /// Default to empty set
  BitSet() : set(0) {}

  /// Initialize a set based on an unsigned bitfield
  BitSet(uint64_t s) : set(s) {
  }

  /**
   * @brief Initialize a set based on a list of numbers
   * @param l A list of numbers
   *
   * Example:
   * @code{.cpp}
   * BitSet bs = {1,-1,2};
   * @endcode
   */
  BitSet(std::initializer_list<int> l) {
    set = 0;
    for (auto p: l)
      set ^= to_set(p);
  }

  /**
   * @brief Flipping an element
   * @param pos the number
   * @return a new set
   *
   * Add the element if not exist, or remove it if it does.
   */
  inline BitSet &flip(long pos) {
    set ^= to_set(pos);
    return *this;
  }

  /// The number of elements currently stored in the set
  inline long size() const {
    return __builtin_popcountl(set);
  }

  /// Return whether a number is in the set
  inline bool get(long pos) const {
    return (set & to_set(pos)) > 0;
  }

  /// Intersection with another set
  inline BitSet operator&(BitSet a) const {
    return set & a.set;
  }

  /// Union with another set
  inline BitSet operator|(BitSet a) const {
    return set | a.set;
  }

  /// \f$A \cup B - A \cap B\f$
  inline BitSet operator^(BitSet a) const {
    return set ^ a.set;
  }

  /// Test emptiness, true if not empty
  inline operator bool() const {
    return set != 0ul;
  }

  /// True if \f$A \subseteq B\f$
  inline bool operator<=(BitSet a) const {
    return (set & a.set) == set;
  }

  /// True if \f$A \supseteq B\f$
  inline bool operator>=(BitSet a) const {
    return (set & a.set) == a.set;
  }

  /**
   * @brief Negate all elements in the set, not a set operation
   * @return a new set
   *
   * For example:
   * @code{.cpp}
   * BitSet bs = {1,-2,3};
   * BitSet nb = !bs; // {-1,2,3}
   * @endcode
   */
  inline BitSet operator!() const {
    BitSet ret(0);

    uint64_t mask = (1ul << (uint64_t) (zero + 1)) - 1ul;

    ret.set = ((set & mask) << (uint64_t) zero) | (set >> (uint64_t) zero);
    return ret;
  }
} BitSet;

/**
 * @brief Print a bit set
 * @relates BitSet
 *
 * Example:
 * @code{.cpp}
 * BitSet bs = {1,-2,3};
 * std::cout << bs; // {1+2-3+}
 * @endcode
 */
inline std::ostream &operator<<(std::ostream &os, const BitSet &bs) {
  os << "{";
  for (long i = 1; i < 32; ++i) {
    if (bs.get(i)) os << i << "+";
    if (bs.get(-i)) os << i << "-";
  }
  os << "}";
  return os;
}

/**
 * @brief Read a bit set
 * @relates BitSet
 *
 * Input is the same format as output. Numbers are separated by appended sign.
 */
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
