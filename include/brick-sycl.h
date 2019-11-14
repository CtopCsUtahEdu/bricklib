//
// Created by Tuowen Zhao on 11/11/19.
//

#ifndef BRICK_BRICK_SYCL_H
#define BRICK_BRICK_SYCL_H

#include "vecscatter.h"
#include "dev_shl.h"

typedef struct oclbrick {
  bElem *dat;
  unsigned *adj;
  unsigned step;
} oclbrick;

#endif //BRICK_BRICK_SYCL_H
