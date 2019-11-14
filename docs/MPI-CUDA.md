# MPI with CUDA

## Implementations

Integrating MPI with CUDA depends on the process of shuffling data between host/device

Two process exists on Summit.

* Use explicit data movement (cudaMemCpy)
* Use unified memory (with system allocator)

## Array code

This is mostly todos

* [x] CPU packing/unpack
* [x] CPU MPI communication
* [ ] GPU packing/unpack
* [x] GPU MPI communication?

## Use explicit data movement

This is the most complex method of managing data. After each iteration the skins
needs to be moved to host and after MPI communication the ghost needs to be copied
to device.

## Use unified memory

This will work for NVidia with ATS (address translation service on Power9) and
HMM (heterogeneous memory management on x86). AMD should also work with HMM.

MMAP is tested with ATS but results in slow operation. Only one possible way to improve performance is to force pages to be resident on the GPU.

## Summit architecture

Each node have 6 gpus and 2 socket each with 21 cores. Each core uses 4 way multithreading.

## Preliminary timing

We can use this [launch schedule](https://jsrunvisualizer.olcf.ornl.gov/index.html?s4f1o01n6c7g1r11d1b1l0=)
for a naive mpi implementation.

### MMPI (mmpi-cuda.cu)

This is the mmap version of the MPI implementation using unified memory through ATS. The skin and ghost are mapped to
the host memory while the internals are stored on the GPU. One problem with this implementation is that the cudaMemAdvise
does not quite work and needs a mix of prefetch instructions. Wait here shouldn't take longer.

~~~
Running with pagesize 65536
MPI Size 24, dims: 4 3 2
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.258235
calc 0.2097
pack 0.0265242
call 0.000408364
wait 0.0216031
perf 0.519749 GStencil/s

Bri: 0.0306885
calc 0.0107308
call 3.28653e-05
wait 0.0199249
perf 4.37355 GStencil/s

Running with pagesize 65536
MPI Size 6, dims: 3 2 1
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.256968
calc 0.209753
pack 0.0266093
call 0.00038762
wait 0.0202175
perf 0.522314 GStencil/s

Bri: 0.0221514
calc 0.0108262
call 2.34756e-05
wait 0.0113018
perf 6.0591 GStencil/s
~~~

### MPI (mpi-cuda.cu)

This is the straight forward MPI implementation using explicit data movement. Calculation time in this case also includes
time taken for memcpy calls.

~~~
Running with pagesize 65536
MPI Size 6, dims: 3 2 1
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.250742
calc 0.20418
pack 0.0261503
call 0.000408387
wait 0.0200034
perf 0.535282 GStencil/s

Bri: 0.0315145
calc 0.0215939
call 2.17442e-05
wait 0.00989883
perf 4.25892 GStencil/s

Running with pagesize 65536
MPI Size 24, dims: 4 3 2
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.239981
calc 0.192468
pack 0.0264099
call 0.00041342
wait 0.0206896
perf 0.559286 GStencil/s

Bri: 0.0326538
calc 0.0211426
call 2.98414e-05
wait 0.0114814
perf 4.11032 GStencil/s
~~~

### CUDA aware MPI (modified mpi-cuda.cu)

This increases the message count but can leverage the nvlink.

~~~
Running with pagesize 65536
MPI Size 6, dims: 3 2 1
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.28319
calc 0.00176854
move 0.232428
pack 0.0301883
call 0.000397414
wait 0.0184083
perf 3.79159 GStencil/s

Bri: 0.0308001
calc 0.0284207
call 2.07698e-05
wait 0.00235863
perf 34.8616 GStencil/s

Running with pagesize 65536
MPI Size 24, dims: 4 3 2
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.284037
calc 0.00172242
move 0.232635
pack 0.0299339
call 0.000406787
wait 0.0193394
perf 3.78028 GStencil/s

Bri: 0.0341525
calc 0.0284069
call 2.54572e-05
wait 0.00572011
perf 31.4396 GStencil/s
~~~

*Transformed*

~~~

Running with pagesize 65536
MPI Size 6, dims: 3 2 1
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.277429
calc 0.00163977
move 0.225791
pack 0.0298127
call 0.000198327
wait 0.0199876
perf 3.87033 GStencil/s

Bri: 0.0307948
calc 0.0283145
call 2.11035e-05
wait 0.00245915
perf 34.8676 GStencil/s

Running with pagesize 65536
MPI Size 24, dims: 4 3 2
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.27983
calc 0.00170367
move 0.226384
pack 0.0297387
call 0.000204553
wait 0.0217988
perf 3.83712 GStencil/s

Bri: 0.033809
calc 0.0283709
call 2.74809e-05
wait 0.00541058
perf 31.7591 GStencil/s
~~~

### CUDA aware MPI + Unified memory (modified mmap-cuda.h)

This isn't performing well either with **ExchangeView** or not. Possible reason is that host buffer is created and takes
longer due to memory have to be copied back/forth to the GPU after host exchange.

The questions is how do we specify ranges of memory being pinned to the device and could device memory have the same
machanism to support shared paging.

OpenMPI use `cuPointerGetAttribute` to query the pointer attribute and decide if MPI calls should use device memory.

~~~
ExhangeView:
Running with pagesize 65536
MPI Size 6, dims: 3 2 1
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.246755
calc 0.200419
pack 0.0263704
call 0.000391275
wait 0.0195751
perf 0.54393 GStencil/s

Bri: 0.106167
calc 0.00537301
call 1.93445e-05
wait 0.100774
perf 1.26422 GStencil/s

Extra Messages:
Running with pagesize 65536
MPI Size 6, dims: 3 2 1
OpenMP threads 28
d3pt7 MPI decomp
Arr: 0.250882
calc 0.203823
pack 0.0265524
call 0.000385611
wait 0.020121
perf 0.534983 GStencil/s

Bri: 0.106433
calc 0.00537478
call 2.80967e-05
wait 0.10103
perf 1.26105 GStencil/s
~~~

## Analysis

CUDA aware MPI is the most performing variant. As 7pt stencil is bandwidth bound, performance uplift with transformation
is limited on V100.
