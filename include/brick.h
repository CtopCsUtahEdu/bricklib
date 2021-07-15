/**
 * @file
 * @brief Main header for bricks
 */

#ifndef BRICK_H
#define BRICK_H

#include <stdlib.h>
#include <type_traits>
#include <memory>
#include "vecscatter.h"

/// BrickStorage allocation alignment
#define ALIGN 2048

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

/// Overloaded attributes for potentially GPU-usable functions (in place of __host__ __device__ etc.)
#if defined(__CUDACC__) || defined(__HIP__)
#define FORCUDA __host__ __device__
#else
#define FORCUDA
#endif

/**
 * @defgroup static_power Statically compute exponentials
 * @{
 */

/// Compute \f$base^{exp}\f$ @ref static_power
template<unsigned base, unsigned exp>
struct static_power {
  static constexpr unsigned value = base * static_power<base, exp - 1>::value;
};

/// Return 1 @ref static_power
template<unsigned base>
struct static_power<base, 0> {
  static constexpr unsigned value = 1;
};
/**@}*/

/**
 * @brief Initializing and holding the storage of bricks
 *
 * It requires knowing how many bricks to store before allocating.
 *
 * Built-in allocators are host-only.
 */
struct BrickStorage {
  /// Pointer holding brick data
  std::shared_ptr<bElem> dat;
  /**
   * @brief Number of chunks
   *
   * A chunk can contain multiple bricks from different sub-fields. Forming structure-of-array.
   */
  long chunks;
  /// Size of a chunk in number of elements
  size_t step;
  /// MMAP data structure when using mmap as allocator
  void *mmap_info = nullptr;

  /// Allocation using *alloc
  static BrickStorage allocate(long chunks, size_t step) {
    BrickStorage b;
    b.chunks = chunks;
    b.step = step;
    b.dat = std::shared_ptr<bElem>((bElem *)aligned_alloc(ALIGN, chunks * step * sizeof(bElem)),
                                   [](bElem *p) { free(p); });
    return b;
  }

  /// mmap allocator using default (new) file
  static BrickStorage mmap_alloc(long chunks, long step);

  /// mmap allocator using specified file starting from certain offset
  static BrickStorage mmap_alloc(long chunks, long step, void *mmap_fd, size_t offset);
};

/**
 * @brief Metadata related to bricks
 * @tparam dims
 *
 * It stores the adjacency list used by the computation. One of this data structure can be shared among multiple bricks.
 * In fact, for computation to succeed, it will require having the same adjacencies for all participating bricks.
 *
 * Each index of the adjacency list will indicate the memory location in the BrickStorage.
 *
 * Metadata can be used to allocate storage with minimal effort. It is recommended to build the metadata before creating
 * the storage.
 */
template<unsigned dims>
struct BrickInfo {
  /// Adjacency list type
  typedef unsigned (*adjlist)[static_power<3, dims>::value];
  /// Adjacency list
  adjlist adj;
  /// Number of bricks in this list
  unsigned nbricks;

  /**
   * @brief Creating an empty metadata consisting of the specified number of bricks
   * @param nbricks number of bricks
   */
  explicit BrickInfo(unsigned nbricks) : nbricks(nbricks) {
    adj = (adjlist) malloc(nbricks * static_power<3, dims>::value * sizeof(unsigned));
  }

  /// Allocate a new brick storage BrickStorage::allocate()
  BrickStorage allocate(long step) {
    return BrickStorage::allocate(nbricks, step);
  }

  /// Allocate a new brick storage BrickStorage::mmap_alloc(long, long)
  BrickStorage mmap_alloc(long step) {
    return BrickStorage::mmap_alloc(nbricks, step);
  }

  /// Allocate a new brick storage BrickStorage::mmap_alloc(long, long, void*, size_t)
  BrickStorage mmap_alloc(long step, void *mmap_fd, size_t offset) {
    return BrickStorage::mmap_alloc(nbricks, step, mmap_fd, offset);
  }
};

/// Empty template to specify an n-D list
template<unsigned ... Ds>
struct Dim {
};

/**
 * @defgroup cal_size Calculate the product of n numbers in a template
 * @{
 */
/**
 * @brief Generic base template for @ref cal_size
 * @tparam xs A list of numbers
 */
template<unsigned ... xs>
struct cal_size;

/**
 * @brief return x when only one number left @ref cal_size
 * @tparam x
 */
template<unsigned x>
struct cal_size<x> {
  static constexpr unsigned value = x;
};

/**
 * @brief Head of the list multiply by result from the rest of list @ref cal_size
 * @tparam x CAR
 * @tparam xs CDR
 */
template<unsigned x, unsigned ... xs>
struct cal_size<x, xs...> {
  static constexpr unsigned value = x * cal_size<xs ...>::value;
};
/**@}*/

/**
 * @defgroup cal_offs Calculating the offset within the adjacency list
 * @{
 */
/**
 * @brief Generic base template for @ref cal_offs
 * @tparam offs Numbers within [0,2]
 */
template<unsigned ... offs>
struct cal_offs;

/**
 * @brief Return offset when only one offset left @ref cal_offs
 * @tparam off
 */
template<unsigned off>
struct cal_offs<1, off> {
  static constexpr unsigned value = off;
};

/**
 * @brief Compute the offset @ref cal_offs
 * @tparam dim Current dimension
 * @tparam off CAR
 * @tparam offs CDR
 */
template<unsigned dim, unsigned off, unsigned ...offs>
struct cal_offs<dim, off, offs...> {
  static constexpr unsigned value = off * static_power<3, dim - 1>::value + cal_offs<dim - 1, offs...>::value;
};
/**@}*/

/**
 * @defgroup _BrickAccessor Accessing brick elements using []
 *
 * It can be fully unrolled and offers very little overhead. However, vectorization is tricky without using codegen.
 *
 * For example, the following code produces types:
 * @code{.cpp}
 * Brick<Dim<8,8,8>, Dim<2,4>> bIn(&bInfo, bStorage, 0);
 * // bIn[0]: _BrickAccessor<bElem, Dim<8,8,8>, Dim<2,4>, void>
 * // bIn[0][1]: _BrickAccessor<bElem, Dim<8,8>, Dim<2,4>, bool>
 * // bIn[0][1][1][1]: bElem
 * @endcode
 *
 * @{
 */

/// Generic base template for @ref _BrickAccessor
template<typename...>
struct _BrickAccessor;

/// Last dimension @ref _BrickAccessor
template<typename T,
    unsigned D,
    unsigned F>
struct _BrickAccessor<T, Dim<D>, Dim<F>, bool> {
  T *par;         ///< parent Brick data structure reference

  unsigned b;     ///< Reference (center) brick
  unsigned pos;   ///< Accumulative position within adjacency list
  unsigned nvec;  ///< Which vector
  unsigned wvec;  ///< Position within a vector

  FORCUDA
  _BrickAccessor(T *par, unsigned b, unsigned pos, unsigned nvec, unsigned wvec) :
      par(par), b(b), pos(pos), nvec(nvec), wvec(wvec) {
  }

  FORCUDA
  inline bElem &operator[](unsigned i) {
    // change pos
    unsigned dir = i + D;
    unsigned d = pos * 3 + dir / D;
    // new vec position
    unsigned l = dir % D;
    unsigned w = wvec * F + l % F;
    unsigned n = nvec * (D / F) + l / F;
    unsigned offset = n * par->VECLEN + w;

    return par->dat[par->bInfo->adj[b][d] * par->step + offset];
  }
};

/**
 * @brief When the number of Brick dimensions and Fold dimensions are the same @ref _BrickAccessor
 * @tparam T Element type
 * @tparam D CAR of brick dimension
 * @tparam BDims CDR of brick dimension
 * @tparam F CAR of vector folds
 * @tparam Folds CDR of vector folds
 */
template<typename T,
    unsigned D,
    unsigned F,
    unsigned ... BDims,
    unsigned ... Folds>
struct _BrickAccessor<T, Dim<D, BDims...>, Dim<F, Folds...>, bool> {
  T *par;         ///< parent Brick data structure reference

  unsigned b;     ///< Reference (center) brick
  unsigned pos;   ///< Accumulative position within adjacency list
  unsigned nvec;  ///< Which vector
  unsigned wvec;  ///< Position within a vector

  FORCUDA
  _BrickAccessor(T *par, unsigned b, unsigned pos, unsigned nvec, unsigned wvec) :
      par(par), b(b), pos(pos), nvec(nvec), wvec(wvec) {
  }

  FORCUDA
  inline _BrickAccessor<T, Dim<BDims...>, Dim<Folds...>, bool> operator[](unsigned i) {
    // change pos
    unsigned dir = i + D;
    unsigned d = pos * 3 + dir / D;
    // new vec position
    unsigned l = dir % D;
    unsigned w = wvec * F + l % F;
    unsigned n = nvec * (D / F) + l / F;
    return _BrickAccessor<T, Dim<BDims...>, Dim<Folds...>, bool>(par, b, d, n, w);
  }
};

/**
 * @brief When the number of Brick dimensions and Fold dimensions are not the same \f$1 + BDims > Folds\f$ @ref _BrickAccessor
 * @tparam T Element type
 * @tparam D CAR of brick dimension
 * @tparam BDims CDR of brick dimension
 * @tparam F CAR of vector folds
 * @tparam Folds CDR of vector folds
 */
template<typename T,
    unsigned D,
    unsigned ... BDims,
    unsigned ... Folds>
struct _BrickAccessor<T, Dim<D, BDims...>, Dim<Folds...>, void> {
  T *par;         ///< parent Brick data structure reference

  unsigned b;     ///< Reference (center) brick
  unsigned pos;   ///< Accumulative position within adjacency list
  unsigned nvec;  ///< Which vector
  unsigned wvec;  ///< Position within a vector

  FORCUDA
  _BrickAccessor(T *par, unsigned b, unsigned pos, unsigned nvec, unsigned wvec) :
      par(par), b(b), pos(pos), nvec(nvec), wvec(wvec) {
  }

  FORCUDA
  inline _BrickAccessor<T, Dim<BDims...>, Dim<Folds...>,
      typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type>
  operator[](unsigned i) {
    // change pos
    unsigned dir = i + D;
    unsigned d = pos * 3 + dir / D;
    // new vec position
    unsigned l = dir % D;
    unsigned w = wvec;
    unsigned n = nvec * D + l;
    return _BrickAccessor<T, Dim<BDims...>, Dim<Folds...>,
        typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type>(par, b, d, n, w);
  }
};
/**@}*/

/**
 * @defgroup Brick Brick data structure
 *
 * See <a href="structBrick_3_01Dim_3_01BDims_8_8_8_01_4_00_01Dim_3_01Folds_8_8_8_01_4_01_4.html">Brick< Dim< BDims... >, Dim< Folds... > ></a>
 *
 * @{
 */

/// Generic base template, see <a href="structBrick_3_01Dim_3_01BDims_8_8_8_01_4_00_01Dim_3_01Folds_8_8_8_01_4_01_4.html">Brick< Dim< BDims... >, Dim< Folds... > ></a>
template<typename...>
struct Brick;

/**
 * @brief Brick data structure
 * @tparam BDims The brick dimensions
 * @tparam Folds The fold dimensions
 *
 * Some example usage:
 * @code{.cpp}
 * Brick<Dim<8,8,8>, Dim<2,4>> bIn(&bInfo, bStorage, 0); // 8x8x8 bricks with 2x4 folding
 * bIn[1][0][0][0] = 2; // Setting the first element for the brick at index 1 to 2
 * @endcode
 */
template<
    unsigned ... BDims,
    unsigned ... Folds>
struct Brick<Dim<BDims...>, Dim<Folds...> > {
  typedef Brick<Dim<BDims...>, Dim<Folds...> > mytype;    ///< Shorthand for this struct's type
  typedef BrickInfo<sizeof...(BDims)> myBrickInfo;        ///< Shorthand for type of the metadata

  static constexpr unsigned VECLEN = cal_size<Folds...>::value;     ///< Vector length shorthand
  static constexpr unsigned BRICKSIZE = cal_size<BDims...>::value;  ///< Brick size shorthand

  myBrickInfo *bInfo;        ///< Pointer to (possibly shared) metadata
  size_t step;             ///< Spacing between bricks in unit of bElem (BrickStorage)
  bElem *dat;                ///< Offsetted memory (BrickStorage)
  BrickStorage bStorage;

  /// Indexing operator returns: @ref _BrickAccessor
  FORCUDA
  inline _BrickAccessor<mytype, Dim<BDims...>, Dim<Folds...>,
      typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type> operator[](unsigned b) {
    return _BrickAccessor<mytype, Dim<BDims...>, Dim<Folds...>,
        typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type>(this, b, 0, 0, 0);
  }

  /// Return the adjacency list of brick *b*
  template<unsigned ... Offsets>
  FORCUDA
  inline bElem *neighbor(unsigned b) {
    unsigned off = cal_offs<sizeof...(BDims), Offsets...>::value;
    return &dat[bInfo->adj[b][off] * step];
  }

  /**
   * @brief Initialize a brick data structure
   * @param bInfo Pointer to metadata
   * @param bStorage Brick storage (memory region)
   * @param offset Offset within the brick storage in number of elements, eg. is a multiple of 512 for 8x8x8 bricks
   */
  Brick(myBrickInfo *bInfo, const BrickStorage &brickStorage, unsigned offset) : bInfo(bInfo) {
    bStorage = brickStorage;
    dat = bStorage.dat.get() + offset;
    step = (unsigned) bStorage.step;
  }
};
/**@}*/

#endif //BRICK_H
