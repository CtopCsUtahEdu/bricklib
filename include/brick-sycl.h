/**
 * @file
 * @brief Header necessary for SYCL program to include
 */

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
