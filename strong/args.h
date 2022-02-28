//
// Created by Tuowen Zhao on 8/23/19.
//

#ifndef BRICK_ARGS_H
#define BRICK_ARGS_H

#include "bitset.h"
#include "zmort.h"

void getrank(BitSet n, ZMORT &zmort, int &dst, int &sub);
void parseArgs(int argc, char **argv, const char *program);

extern unsigned dom_size, sdom_size;
extern unsigned long mysec_l, mysec_r;
extern bool validate;

#endif // BRICK_ARGS_H
