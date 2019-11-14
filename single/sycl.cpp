//
// Created by Tuowen Zhao on 11/11/19.
//

#include <iostream>
#include <random>
#include "brick-sycl.h"
#include <stencils/stencils.h>

bElem *coeff;

void syclinit();

int main() {
  coeff = (bElem *) malloc(129 * sizeof(bElem));
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  for (int i = 0; i < 129; ++i)
    coeff[i] = u(mt);

  syclinit();
  d3pt7();
  return 0;
}
