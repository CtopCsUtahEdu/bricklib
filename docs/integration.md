# Codegen Integration

Compilation have two phase when using the code generator.

1. The code generator is invoked to inject the generated code to the source.
2. Compile using host compiler as normal.

## Basic Integration Usage (*CMake*)

### Code

`include/vecscatter.h` defined two different code generation targets: a) for use with tiled code b) more importantly,
for use with bricked code. The detailed usage are shown with defintions `#tile(...)` and `#brick(...)`, correspondingly.

For examples see Line 34 in `weak/main.cpp` using the brick version. One constant expanded version of this call is as follows:

~~~cpp
brick("../stencils/mpi7pt.py", "AVX512", (8,8,8), (8), b);
~~~

### Code generation

The code generator, `codegen/vecscatter`, will run C/C++ preprocessor that can resolve all preprocessor defines and expanded into `vecscatter` pragma
by the preprocessor. The code generator will then scan the processed code and look for `#pragma vecscatter` to generate the code.

In CMake this code generation step is achieved by simply calling `VSTARGET`:

~~~cmake
VSTARGET(N3MPI main.cpp main-out.cpp)
~~~

This defines a code generation target with name `N3MPI`.

This CMake call uses information defined by `include_directories`, `CMAKE_CXX_FLAGS`, and any custom options in `${NAME}_COMPILE_OPTIONS`.

The preprocessor to use can be customized by setting variable `VS_PREPROCESSOR`, who uses `cpp` as the default.

### Compile

The `main-out.cpp` in the argument to `VSTARGET` is the code generated output. This is also captured by cmake variable `${VSTARGET_${NAME}_OUTPUT}`.

## Advanced Integration (*Make*)

To compile with *Make* is just as simple. The source code doesn't needs to be changed.

However, the work done with `VSTARGET` needs to be written as explicit targets in *Make*.

### Vecscatter target

This target calls the code generator by running `codegen/vecscatter`. The first two arguments are the input and output filename. The other options can be querried by running `codegen/vecscatter -h`.

~~~makefile
main-out.cpp:main.cpp
  $(CODEGEN) main.cpp main-out.cpp -- -march=native -O2 -I../include -D../stencils
~~~

`main-out.cpp` can thus be used as normal.
