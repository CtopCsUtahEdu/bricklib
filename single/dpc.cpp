#define bElem float

#include <iostream>
#include "stencils/stencils_dpc.h"
#include <random>
#include "stencils/stencils.h"

bElem *coeff;

int main() {
  coeff = (bElem *) malloc(129 * sizeof(bElem));
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  for (int i = 0; i < 129; ++i)
    coeff[i] = u(mt);

  d3pt7dpc();
//   d3condcu();
  return 0;
}