//
// Created by Tuowen Zhao on 6/16/19.
//

#include <zmort.h>
#include <iostream>

int main() {
  std::cout << zmort0[5][9][4] << ":" << zmort0[5][2][4].set(1,9) << std::endl;
  return 0;
}