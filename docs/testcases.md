# Test cases

Test cases are categorized as the folders.

* Single node (`\single`) for different compute model
    * `single-cpu`
    * `single-cuda`
    * `single-opencl`
    * `single-sycl`
    * `single-mpi` requires MPI for layout benchmark
* Weak scaling (`\weak`)
    * `weak-cpu`
    * `weak-cuda`
* Strong scaling (`\strong`)
    * `strong-cpu`
    * `strong-cuda`

Following is description of each of the case.

## Single node

All single node will compile into `<build_dir>/single/*` as executables without the prefix `single-`,
such as `single-cpu` will be built as `<build_dir>/single/cpu`.

Each single node experiment accept no commandline arguments. To change the stencils that it is
computing refer to corresponding file in `/stencils`. To change the domain size `N` so that the
total domain is $N^3$ modify the macro `#define N 64` in `/stencils/stencils.h`.

## Weak scaling

Weak scaling support fixed domain per node or fixed global domain decomposed into one subdomain per node.

Compiled into `<build_dir>/weak/*` as executables without the `weak-` prefix, such as `weak-cpu`
will be built as `<build_dir>/weak/cpu`.

Each weak scaling experiment supports the following commandline arguments:

* Changing the domain size (use one):
    * `-d Int,Int,Int` set global domain size
    * `-s Int,Int,Int` set per-rank subdomain size
* `-I Int` number of iterations to take average
* Other options
    * `-b` downsize number of ranks to perfect 2 exponential

For example `mpirun -np 4 <build_dir>/weak/cpu -d 512,512,512` will distribute domain of $512^3$ to
4 mpi ranks.

## Strong scaling

Strong scaling support 2-level decomposition where global domain is decomposed into fixed-sized
subdomains indexed using z-mort. These subdomains are then distributed to different MPI rank based
on index. Each rank thus may have more than one subdomains. It supports the following commandline
arguments:

* `-d Int` changing domain size to $Int^3$
* `-s Int` changing the subdomain size $Int^3$
* `-I Int` number of iterations to take average
* `-v` enable validation for `strong-cpu`
