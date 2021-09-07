# Vectorization in Bricks

One of main target of the code generation is to handle discontinuity using vectorization.
The process largely depends on which vectorization ISA is used.

## Vectorization Abstraction

Vector is data abstraction derived from single instruction, multiple data (SIMD) processing, where
one instruction can express operation applied to multiple elements in one or more vectors. Each
element of vector may also be referred to as a lane of the vector units. The vector abstraction we
used can be based on bits or fixed number of elements due to global definition of bElem.

Each individual brick is a collection of whole vectors. Each vector is logically multi-dimensional
even when physically the vector only have one dimension. The size of the vectors is specified with
the second template argument to the `Brick<Dim<...>,Dim<...>>` structure.

Note that the code generator can place some extra restriction on the vector dimensions due to ISA
support. For example, AVX2 support efficient suffling between 128 bits and within 128 bits. For
AVX2 it is recommended to view the physical vector as 2x128bits.

Due to the size of vectors, it may place some restrictions on the size of the brick. However,
because the dimensions of the vector is rather flexible from balancing between the dimensions,
this restriction on brick dimensions is generally irrelevant.

The underlying code generator assumes that vector ISA supports the following operations:

* Aligned load of vectors
* Element-wise computation between vectors
* Element-wise selection from two vectors
* Shuffling across lanes

## Using the vectorizing code generator

See [stencil expression](docs/stencilExpr.md) on expressing stencils for the code generator and
see [integration](docs/integration.md) on integrating the generated code in C++ program.

## Internal structure of the code generator

The code generator is located in `codegen`, note that the `vectorscatter` executable is used to
process C++ files which is usually what's of the most interests.

* `st/*.py` defines the AST elements. These elements are used to define the stencil expression.
* `st/codegen/backend/*` defines different backend ISA. See `asimd.py` for a basic example.
* `st/codegen/*.py` defines the code generation process.
* `st/codegen/base.py` is the entry point to the codegen. See `CodeGen::gencode`.
