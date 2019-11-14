# Brick Layout for C++

## Require

* *C++11* compatible compiler
* MPI library
* **optional** *CUDA*

## Directory & Files

* cpp files in `\ ` are main files for different tests.
* `include` contains the brick library.
* `stencils` contains different stencils and related initialization code.

This version of brick library relies on C++ template and is currently a header-only library that can be included however
one likes.

## Building and running

### To build with CMake

1. Clone the repository and checkout this branch
2. Create build library inside the source tree `mkdir build`
3. Create build configuration `cd build && cmake .. -DCMAKE_BUILD_TYPE=Release`
4. Build `make`

## Using the brick template

The brick template consists of 3 part:

* `Brick`: declare brick data structure
* `BrickInfo`: an adjacency list that describes the relations between bricks
* `BrickStorage`: a chunk of memory for storing bricks

# Dimension Ordering

Template arguments & code ordering is contiguous dimension last. Dimension arrays are contiguous at 0 (contiguous first).
