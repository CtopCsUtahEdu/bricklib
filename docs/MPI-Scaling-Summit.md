# MPI Scaling Experiments

## Weak scaling

Weak scaling experiments uses $512^3$ elements per mpi-process and scale all the way to 512 nodes where each node have
6 MPI process.

The result is as follows.

~~~
Sun Jun 16 23:18:30 EDT 2019
Running with pagesize 65536
MPI Size 6, dims: 3 2 1
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.0307753
calc 0.0283681
call 2.08609e-05
wait 0.00238634
perf 34.8897 GStencil/s
Running with pagesize 65536
MPI Size 12, dims: 3 2 2
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.0338175
calc 0.0284407
call 2.72357e-05
wait 0.00534956
perf 31.751 GStencil/s
Running with pagesize 65536
MPI Size 24, dims: 4 3 2
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.0338099
calc 0.0283407
call 2.66437e-05
wait 0.00544259
perf 31.7582 GStencil/s
Running with pagesize 65536
MPI Size 48, dims: 4 4 3
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.0374408
calc 0.028368
call 3.35222e-05
wait 0.00903928
perf 28.6784 GStencil/s
Running with pagesize 65536
MPI Size 96, dims: 6 4 4
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.049546
calc 0.0282925
call 3.28741e-05
wait 0.0212207
perf 21.6716 GStencil/s
Running with pagesize 65536
MPI Size 192, dims: 8 6 4
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.045825
calc 0.0283677
call 3.38506e-05
wait 0.0174234
perf 23.4313 GStencil/s
Running with pagesize 65536
MPI Size 384, dims: 8 8 6
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.045119
calc 0.0283946
call 3.25011e-05
wait 0.0166919
perf 23.798 GStencil/s
Running with pagesize 65536
MPI Size 768, dims: 12 8 8
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.0526365
calc 0.0283833
call 0.0038422
wait 0.020411
perf 20.3992 GStencil/s
Running with pagesize 65536
MPI Size 1536, dims: 16 12 8
OpenMP threads 28
d3pt7 MPI decomp
Bri: 0.0528424
calc 0.0283822
call 0.0037276
wait 0.0207326
perf 20.3197 GStencil/s

512 node failed with mpi library error "invalid device ordinal"
~~~

## Strong scaling

Each GPU have 16GB of memory. This allows for domain size slightly smaller than $1024^3$ for one GPU.

We used $4096^3$ overall domains with $128^3$ subdomains split over MPI processes.

We have 2 different implementations, one is not using any packing or unpacking or use some packing and unpacking.

Without packing
~~~
Running with pagesize 65536
MPI Size 192
OpenMP threads 28
Bri: 0.282318
calc 0.14142
move 0
call 0.00199384
wait 0.138905
perf 1947.29 GStencil/s
Running with pagesize 65536
MPI Size 384
OpenMP threads 28
Bri: 0.172403
calc 0.0706009
move 0
call 0.00117519
wait 0.100627
perf 3188.78 GStencil/s
Running with pagesize 65536
MPI Size 768
OpenMP threads 28
Bri: 0.0908419
calc 0.0327315
move 0
call 0.000680892
wait 0.0574295
perf 6051.79 GStencil/s
Running with pagesize 65536
MPI Size 1536
OpenMP threads 28
Bri: 0.0478411
calc 0.0141601
move 0
call 0.00147172
wait 0.0322093
perf 11491.3 GStencil/s
Running with pagesize 65536
MPI Size 3072
OpenMP threads 28
Bri: 0.0304894
calc 0.00680064
move 0
call 0.000447586
wait 0.0232411
perf 18031.1 GStencil/s
Running with pagesize 65536
MPI Size 6144
OpenMP threads 28
Bri: 0.0213373
calc 0.00346004
move 0
call 0.000264028
wait 0.0176132
perf 25765.1 GStencil/s
~~~

With packing
~~~
Running with pagesize 65536
MPI Size 192
OpenMP threads 28
Bri: 0.276796
calc 0.151059
call 3.19912e-05
wait 0.125705
perf 1986.14 GStencil/s
part 56
Running with pagesize 65536
MPI Size 384
OpenMP threads 28
Bri: 0.156261
calc 0.0739089
call 3.73282e-05
wait 0.0823147
perf 3518.19 GStencil/s
part 62
Running with pagesize 65536
MPI Size 768
OpenMP threads 28
Bri: 0.0874968
calc 0.0353746
call 3.52758e-05
wait 0.052087
perf 6283.15 GStencil/s
part 62
Running with pagesize 65536
MPI Size 1536
OpenMP threads 28
Bri: 0.0487554
calc 0.0157645
call 4.48647e-05
wait 0.032946
perf 11275.8 GStencil/s
part 60
Running with pagesize 65536
MPI Size 3072
OpenMP threads 28
Bri: 0.0268628
calc 0.00764753
call 3.38328e-05
wait 0.0191814
perf 20465.4 GStencil/s
part 58
Running with pagesize 65536
MPI Size 6144
OpenMP threads 28
Bri: 0.016089
calc 0.00394477
call 3.48566e-05
wait 0.0121094
perf 34169.6 GStencil/s
part 62
~~~

We can see that without packing there is a much longer wait time. Possibly due to higher number of messages in the network.
