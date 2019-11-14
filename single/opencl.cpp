//
// Created by Tuowen Zhao on 11/9/19.
//

#include <iostream>
#include <stencils/stencils.h>
#include <random>

void clinit();
void cldestroy();

bElem *coeff;

int main() {
  coeff = (bElem *) malloc(129 * sizeof(bElem));
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  for (int i = 0; i < 129; ++i)
    coeff[i] = u(mt);

  clinit();
  d3pt7();
  cldestroy();
  return 0;
}
