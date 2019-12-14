/**
 * @file
 * @brief Header for Z-Mort ordering
 */

#ifndef BRICK_ZMORT_H
#define BRICK_ZMORT_H

/**
 * @brief n-dimensional Z-Mort ordering
 *
 * Preliminary n-d Z-Mort implementation whose returning index is not compact. Only for perfect 2-exponentials will
 * return compact & contiguous index.
 *
 * This can be viewed as a single Z-Mort index or as an array of indices that represent the original n-dimensional
 * position. It can be constructed incrementally from 0-d Z-Mort.
 */
struct ZMORT {
  unsigned long id; ///< Z-Mort index of this struct
  unsigned long dim; ///< Number of dimensions

  /// Default to 0-d
  ZMORT() : id(0ul), dim(0ul) {}

  /// Initialize using z-mort id and the number of dimensions
  ZMORT(unsigned long id, unsigned long dim) : id(id), dim(dim) {};

  /**
   * @brief Continuously construct a Z-Mort index
   * @param p Position in the current dimension
   * @return ZMORT with one more dimension than before
   *
   * For example:
   * @code{.cpp}
   * ZMORT z = 0ul;
   * ZMORT n = z[5][6][7];
   * // n.dim = 3, n(0) = 7, n(1) = 6, n(2) = 5
   * @endcode
   */
  inline ZMORT operator[](unsigned long p) {
    unsigned long oid = id;

    ZMORT zmort;
    zmort.id = 0;
    zmort.dim = dim + 1ul;
    unsigned long omask = (1ul << dim) - 1ul;

    unsigned long i = 0ul;
    while (oid || p) {
      zmort.id |= (((oid & omask) << 1ul) + (p & 1ul)) << (i * zmort.dim);
      oid >>= dim;
      p >>= 1ul;
      ++i;
    }

    return zmort;
  }

  /**
   * @brief Get positions of a Z-Mort index on the d-th dimension
   * @param d The dimension to get index, 0 is the fastest varying dimension
   * @return The position
   *
   * For example, see ZMORT::operator[](unsigned long).
   */
  inline unsigned long operator()(unsigned long d) {
    unsigned long oid = id >> d;
    unsigned long pos = 0;

    unsigned long i = 0;
    while (oid) {
      pos |= (oid & 1ul) << i;
      oid >>= dim;
      ++i;
    }

    return pos;
  }

  /**
   * @brief Set positions of a Z-Mort index on the d-th dimension
   * @param d The dimension, 0 is the fastest varying dimension
   * @param p The position
   * @return A NEW ZMORT
   */
  inline ZMORT set(unsigned long d, unsigned long p) {
    unsigned long oid = id >> d;
    unsigned long omask = (1ul << dim) - 2ul;
    unsigned long i = d;

    ZMORT zmort(id & ((1ul << d) - 1), dim);
    while (oid || p) {
      zmort.id |= ((p & 1ul) + (oid & omask)) << i;
      oid >>= dim;
      p >>= 1ul;
      i += dim;
    }
    return zmort;
  }

  /// Implicit conversion to extract the Z-Mort index
  inline operator unsigned long() const {
    return id;
  }
};

#endif //BRICK_ZMORT_H
