//
// Created by ztuowen on 8/23/19.
//

#include "args.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>

#define SUBDIM (dom_size / sdom_size)

namespace {
const char *const shortopt = "d:s:I:hv";
const char *help = "Running MPI with %s\n\n"
                   "Program options\n"
                   "  -h: show help (this message)\n"
                   "  -d Int: set domain size to d^3 (default 512)\n"
                   "  -s Int: set subdomain size to s^3 (default 128)\n"
                   "  Benchmark control:\n"
                   "  -I: number of iterations, default %d\n"
                   "  -v: enable validation (CPU only)\n"
                   "Example usage:\n"
                   "  %s -d 2048 -s 64\n";

unsigned long sec_len, sec_split, sec_shift;
} // namespace

unsigned long mysec_l, mysec_r;
unsigned dom_size = 512, sdom_size = 128;
int MPI_ITER;
bool validate = false;

void getrank(BitSet n, ZMORT &zmort, int &dst, int &sub) {
  for (int d = 0; d < 3; ++d) {
    long offset = 0;
    if (n.get(d + 1))
      offset = 1;
    else if (n.get(-1 - d))
      offset = -1;
    if (offset) {
      long idx = zmort(d);
      idx = (idx + SUBDIM + offset) % SUBDIM;
      zmort = zmort.set(d, idx);
    }
  }
  if (zmort.id < sec_split) {
    dst = zmort.id / (sec_len + 1);
    sub = zmort.id % (sec_len + 1);
  } else {
    dst = (zmort.id - sec_shift) / sec_len;
    sub = (zmort.id - sec_shift) % sec_len;
  }
};

void parseArgs(int argc, char **argv, const char *program) {
  int c;
  while ((c = getopt(argc, argv, shortopt)) != -1) {
    switch (c) {
    case 'd':
      dom_size = std::stoi(optarg);
      break;
    case 's':
      sdom_size = std::stoi(optarg);
      break;
    case 'I':
      MPI_ITER = std::stoi(optarg);
      break;
    case 'v':
      validate = true;
      break;
    default:
      printf("Unknown options %c\n", c);
    case 'h':
      printf(help, program, MPI_ITER, argv[0]);
      MPI_Finalize();
      exit(0);
      break;
    }
  }

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int numthreads;
#pragma omp parallel shared(numthreads) default(none)
    numthreads = omp_get_num_threads();
    long page_size = sysconf(_SC_PAGESIZE);
    std::cout << "Pagesize " << page_size << "; MPI Size " << size << " * OpenMP threads "
              << numthreads << std::endl;
    std::cout << "Domain size of " << dom_size << "^3 decomposed into " << sdom_size
              << "^3 subdomains" << std::endl;
    long n = dom_size / sdom_size;
    n = n * n * n;
    std::cout << "Total of " << n << " subdomains, " << (n + size - 1) / size << " per rank"
              << std::endl;
  }

  // setting up the owning subdomains
  unsigned long allsubs = SUBDIM;
  allsubs = allsubs * allsubs * allsubs;

  sec_shift = allsubs % size;
  sec_len = allsubs / size + (sec_shift > rank ? 1 : 0);
  mysec_l = (unsigned long)rank * sec_len + (sec_shift > rank ? 0 : sec_shift);
  mysec_r = mysec_l + sec_len;
  sec_split = sec_shift * ((allsubs + size - 1) / size);
  sec_len = allsubs / size;
}
