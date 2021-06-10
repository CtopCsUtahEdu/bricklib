//
// Created by Tuowen Zhao on 12/5/18.
//

#include <iostream>
#include <random>
#include "brick.h"

bElem *coeff;

int main() {
  coeff = (bElem *) malloc(129 * sizeof(bElem));
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  for (int i = 0; i < 129; ++i)
    coeff[i] = u(mt);

  std::cout << "Hello world" << std::endl;
  return 0;
}