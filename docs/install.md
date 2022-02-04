# Installing this library

This guide includes how to install this library and use it as an external library in another CMake
project.

## Installation

* Add `-DCMAKE_INSTALL_PREFIX` during the CMake configuration phase indicating where this library is
to be installed.
* Build the `install` target will install the header files, code generator, and relavant
configuration script
* Pass the installation path of `-Dbrick_DIR=${CMAKE_INSTALL_PREFIX}/lib/brick/cmake` will enable
`find_package(brick)` to work in an external project

## find_package(brick)

`find_package` will expose the `BRICK_INCLUDE_DIR` as the header installation path and all library
targets exposed by the installation. Currently usable targets are as follows.

* brick (memory mapping)
* brick-mpi (mpi routines and constants etc.)
* brickhelper (compare and data copy between brick and arrays)

## Setup after importation

The vector scatter CMake functions will work out-of-box. What needs to be setup after `find_package`
includes the follows:

* Any used external dependencies: OpenMP, MPI, OpenCL, `rt` etc.
* Any compilation flags: `-march=native -O2`
* Any predefines that are used by the *headers*: `DECOMP_PAGEUNALIGN` and `BARRIER_TIMESTEP`

## Example

See `examples/external`