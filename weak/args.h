//
// Created by Tuowen Zhao on 9/8/19.
//

#ifndef BRICK_ARGS_H
#define BRICK_ARGS_H

#include <mpi.h>
#include <vector>

MPI_Comm parseArgs(int argc, char **argv, const char *program, int dims = 3);

extern std::vector<unsigned> dim_size, dom_size;
extern size_t tot_elems;

#endif // BRICK_ARGS_H
