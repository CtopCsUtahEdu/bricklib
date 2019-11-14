//
// Created by Tuowen Zhao on 6/16/19.
//

#ifndef BRICK_ZMORT_H
#define BRICK_ZMORT_H

// This is a preliminary n-d zmort implementation
// The aim is to support 2 operations one is to convert from coordinate into index, another is from index into
// coordinate

// Note that zmort can be constructed incremental

struct ZMORT {
  unsigned long id; // index
  unsigned long dim; // number of dimensions

  ZMORT() : id(0ul), dim(0ul) {}

  ZMORT(unsigned long id, unsigned long dim) : id(id), dim(dim) {};

  // This add another dimension to current
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

  // This separate out one dimension
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

  inline operator unsigned long() const {
    return id;
  }
};

extern ZMORT zmort0;

#endif //BRICK_ZMORT_H
