# Using the Brick Library

**How to use the template library**

Brick library enables fine-grained data blocking for use in regular C++ program.
It should support initialization of bricks and copying between regular array layout.

## Overall design

The brick library uses C++ templates to handle dimensions and address calculation.
There are three main components/data structures.

* BrickStorage: contains pointer to a memory region for bricks
* BrickInfo: adjacency list of bricks
* Brick: BrickStorage + BrickInfo + accessor for address calculation

## Creating bricks

To create a brick data structure consider the code in `d3pt7` from `stencils/3axis.cpp`.

~~~cpp
BrickInfo bInfo = init_grid<3>(grid_ptr, {4, 4, 4});
BrickStorage bStorage = BrickStorage::allocate(bInfo.nbricks, 512 * 2);
Brick<Dim<8,8,8>, Dim<2,2>> bIn(&bInfo, bStorage, 0);
Brick<Dim<8,8,8>, Dim<2,2>> bOut(&bInfo, bStorage, 512);
~~~

The above code creates $4\times 4 \times 4$ of $8\times 8\times 8$ bricks with two brick accessors interleaved. The following explains each line:

1. BrickInfo struct is initialized by init_grid helper functions. It returns a three-dimensional
   array `grid_ptr`. The second argument is the size of the grid in number of bricks per dimension.
2. BrickStorage::allocate initialize a BrickStorage object with specified number of bricks and each
   "brick"'s size is specified in the second arguments. This accommodates for cases when multiple
   bricks from different "fields" are interleaved.
3. Also 4. The Brick accessor is created with bIn and bOut interleaved in `bStorage`. Each vector in
   brick is $2\times 2$.

## Using bricks

* Each brick can be treated as a multidimensional array.

## Cleaning up

Currently, the following requires manual cleanup.

* Any `grid_ptr` created by either the user or with `init_grid` methods.
* The `adj` field in brickInfo.
