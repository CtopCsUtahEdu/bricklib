# Brick Layout for C++

## Requirements

* *C++14* compatible compiler
* OpenMP
* MPI library
* CMake
* **Optional** backends
    * *CUDA*
    * *OpenCL*
    * *SYCL*
    * *HIP* **WIP**

## Building and running

1. Clone the repository
2. Create a build directory inside the source tree `mkdir build`
3. Create build configuration `cd build && cmake .. -DCMAKE_BUILD_TYPE=Release`
4. Build different test cases using `make <testname>`

For description of the test cases see [here](docs/testcases.md).

## Using the brick template

The brick template consists of 3 part:

* `Brick`: declare brick data structure
* `BrickInfo`: an adjacency list that describes the relations between bricks
* `BrickStorage`: a chunk of memory for storing bricks

The behavior of such templated data structures are as normal: they do not require the use of code generator to function;
provide a fallback way of writing code for compute & data movement.

## Stencil Expression Description

Stencil expression for code generator are specified using [Python library](docs/stencilExpr.md). Code generator provide 
optimization and vectorization support for different backend.

The code generation are carried out by CMake wrapper automatically. For details, see [Codegen Integration]().

# Dimension Ordering

Template arguments & code ordering is contiguous dimension last. Dimension arrays are contiguous at 0 (contiguous first).


## Directory & Files

* `include` and `src` contains the brick library headers and library files.
* `docs` various documents
* `cmake` CMake module file
* Included test cases are split into 4 folders:
    * `stencils` contains different stencils and related initialization code used by all tests as needed
    * `single` for single node (no MPI)
    * `weak` for weak scaling or strong scaling with one-level decomposition (one subdomain per rank)
    * `strong` for strong scaling with two-level decomposition (multiple fixed-sized subdomains per rank)

A large portion of the brick library is entirely based on templates and can be included as a header only library.

## Publications

@cite zhao2018 Zhao, Tuowen, Samuel Williams, Mary Hall, and Hans Johansen. "Delivering Performance-Portable Stencil Computations on CPUs and GPUs Using Bricks." In 2018 IEEE/ACM International Workshop on Performance, Portability and Productivity in HPC (P3HPC), pp. 59-70. IEEE, 2018. 

@cite zhao2019 Zhao, Tuowen, Protonu Basu, Samuel Williams, Mary Hall, and Hans Johansen. "Exploiting reuse and vectorization in blocked stencil computations on CPUs and GPUs." In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, p. 52. ACM, 2019.