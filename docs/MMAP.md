# Using MMAP for MPI/Cross GPU Ghostzone Communication

## Target

One of the major features of bricks is increasing memory locality.
Our previous work establishes brick to improve locality for vector/multicore computations.
This project is to explore the locality effect of brick in distributed settings.

## Tools

A few tools are needed for this.

* mem files to enable shared memory (kernel >= 3.17) or linux shm object (been forever)
* consecutive mmap calls to share certain parts of a continguous virtual memory space
* Using the communication region analysis to compact the regions

Linux have per process memory chunk of 128TB see, the mmap can start anywhere

[linux virtual memory mapping](https://www.kernel.org/doc/Documentation/x86/x86_64/mm.txt)

Using paging, however, changes the way memory regions are initialized.

1. Create a memfd that span **all** the data necessary for the computation.
2. Create views by specifying a sequence of specific ranges within the file.
3. Compute/communicate with views as if they are regular pointer structures.

## Analysis

Assuming a brick size of 4x4x4 doubles. Then we have 4x4x4x8=0.5kB. A normal 4kB pages to represent one brick have
some wastage. This price is necessary to allow no packing and no message increase. Also note that these waste are
$O(1)$ which won't result in large performance penalties.

This architecture works for both single stage communication or multi-stage communication.

## Programming interface

The major problem with programming interface is the two step allocation. One would have to first decide the maximum
amount of memory needed for **all** the data to create the memfd to prevent unnecessary explosion of opened files.
After one would also have to decide the views in one go. Those view cannot be expanded. Or we could implement some
form of heap management algorithm to manage the mmap regions.

We could have each brick regions being one memfd.

## Preliminary timing

We have choices of using either 1/2/6 MPI process per host. The best performing one appears to be using
[2](https://jsrunvisualizer.olcf.ornl.gov/index.html?s4f1o01n2c21g0r11d1b1l0=).

### MMAP

`ev.exchange()`

*Summit*

~~~
Running with pagesize 65536
MPI Size 8, dims: 2 2 2
OpenMP threads 84
d3pt7 MPI decomp
Arr: 0.0899363
calc 0.0419902
pack 0.0243037
call 0.00177483
wait 0.0218675
perf 1.49236 GStencil/s

Bri: 0.0454365
calc 0.0340519
call 0.000240905
wait 0.0111437
perf 2.95396 GStencil/s
~~~

*Cori*

~~~
Running with pagesize 4096
MPI Size 1, dims: 1 1 1
OpenMP threads 256
d3pt7 MPI decomp
Arr: 0.600676
calc 0.20492
pack 0.143562
call 0.252164
wait 3.07083e-05
perf 1.78756 GStencil/s

Bri: 0.135669
calc 0.105756
call 0.0298611
wait 5.1384e-05
perf 7.91445 GStencil/s

Running with pagesize 4096
MPI Size 8, dims: 2 2 2
OpenMP threads 256
d3pt7 MPI decomp
Arr: 0.269196
calc 0.137735
pack 0.0865624
call 0.0175454
wait 0.0273529
perf 3.9887 GStencil/s

Bri: 0.13402
calc 0.104867
call 0.00959902
wait 0.0195536
perf 8.0118 GStencil/s

Running with pagesize 4096
MPI Size 32, dims: 4 4 2
OpenMP threads 64
d3pt7 MPI decomp
Arr: 0.744692
calc 0.584394
pack 0.0885524
call 0.0146941
wait 0.0570516
perf 1.44186 GStencil/s

Bri: 0.533886
calc 0.399848
call 0.0102416
wait 0.123797
perf 2.01118 GStencil/s
~~~

### Extra messages

`bDecomp.exchange(bStorage)`

~~~
Running with pagesize 65536
MPI Size 8, dims: 2 2 2
OpenMP threads 84
d3pt7 MPI decomp
Arr: 0.0895492
calc 0.0418606
pack 0.0246195
call 0.00181641
wait 0.0212527
perf 1.49882 GStencil/s

Bri: 0.0457874
calc 0.0340434
call 0.000308851
wait 0.0114352
perf 2.93132 GStencil/s
~~~

*Cori*

~~~
Running with pagesize 4096
MPI Size 1, dims: 1 1 1
OpenMP threads 256
d3pt7 MPI decomp
Arr: 0.587701
calc 0.204579
pack 0.130091
call 0.253002
wait 2.86865e-05
perf 1.82702 GStencil/s

Bri: 0.136442
calc 0.106795
call 0.0295729
wait 7.35664e-05
perf 7.86961 GStencil/s

Running with pagesize 4096
MPI Size 8, dims: 2 2 2
OpenMP threads 256
d3pt7 MPI decomp
Arr: 0.267753
calc 0.138071
pack 0.0872588
call 0.0179323
wait 0.0244914
perf 4.01019 GStencil/s

Bri: 0.13762
calc 0.106492
call 0.00996408
wait 0.0211636
perf 7.80223 GStencil/s

Running with pagesize 4096
MPI Size 32, dims: 4 4 2
OpenMP threads 64
d3pt7 MPI decomp
Arr: 0.739128
calc 0.577006
pack 0.0902416
call 0.0154415
wait 0.0564393
perf 1.45271 GStencil/s

Bri: 0.477826
calc 0.397229
call 0.0105643
wait 0.0700328
perf 2.24714 GStencil/s
~~~

### Analysis

Extra messages make little to no difference on Summit.

On Cori KNL nodes, possibly due to larger impact from the paging system, extra messages are preferred over fewer
messages for 4 MPI process per node.

### YASK Comparison

$1024^3$ over 8 nodes.

~~~
──────────────────────────────────────────────────────────────────────
Running 3 performance trial(s) of 320 step(s) each...
──────────────────────────────────────────────────────────────────────
Trial number:                      1

Work stats:
 num-steps-done:                   320
 num-reads-per-step:               7.51619G
 num-writes-per-step:              1.07374G
 num-est-FP-ops-per-step:          7.51619G
 num-points-per-step:              1.07374G

Time stats:
 elapsed-time (sec):               7.65358
 Time breakdown by activity type:
  compute time (sec):                4.82962 (63.102673%)
  halo exchange time (sec):          2.80036 (36.588871%)
  other time (sec):                  23.6078m (0.308454%)
 Compute-time breakdown by halo area:
  rank-exterior compute (sec):       954.538m (19.764259%)
  rank-interior compute (sec):       3.87508 (80.235741%)
 Halo-time breakdown:
  MPI waits (sec):                   1.22469 (43.733398%)
  MPI tests (sec):                   9.87914m (0.352781%)
  packing, unpacking, etc. (sec):    1.56579 (55.913818%)

Rate stats:
 throughput (num-reads/sec):       314.256G
 throughput (num-writes/sec):      44.8937G
 throughput (est-FLOPS):           314.256G
 throughput (num-points/sec):      44.8937G
──────────────────────────────────────────────────────────────────────
Trial number:                      2

Work stats:
 num-steps-done:                   320
 num-reads-per-step:               7.51619G
 num-writes-per-step:              1.07374G
 num-est-FP-ops-per-step:          7.51619G
 num-points-per-step:              1.07374G

Time stats:
 elapsed-time (sec):               7.65605
 Time breakdown by activity type:
  compute time (sec):                4.83409 (63.140766%)
  halo exchange time (sec):          2.79807 (36.547241%)
  other time (sec):                  23.8863m (0.311993%)
 Compute-time breakdown by halo area:
  rank-exterior compute (sec):       957.532m (19.807932%)
  rank-interior compute (sec):       3.87655 (80.192070%)
 Halo-time breakdown:
  MPI waits (sec):                   1.17589 (42.025135%)
  MPI tests (sec):                   9.90637m (0.354042%)
  packing, unpacking, etc. (sec):    1.61227 (57.620823%)

Rate stats:
 throughput (num-reads/sec):       314.155G
 throughput (num-writes/sec):      44.8792G
 throughput (est-FLOPS):           314.155G
 throughput (num-points/sec):      44.8792G
──────────────────────────────────────────────────────────────────────
Trial number:                      3

Work stats:
 num-steps-done:                   320
 num-reads-per-step:               7.51619G
 num-writes-per-step:              1.07374G
 num-est-FP-ops-per-step:          7.51619G
 num-points-per-step:              1.07374G

Time stats:
 elapsed-time (sec):               7.65742
 Time breakdown by activity type:
  compute time (sec):                4.83166 (63.097801%)
  halo exchange time (sec):          2.80177 (36.589012%)
  other time (sec):                  23.982m (0.313186%)
 Compute-time breakdown by halo area:
  rank-exterior compute (sec):       953.795m (19.740511%)
  rank-interior compute (sec):       3.87787 (80.259491%)
 Halo-time breakdown:
  MPI waits (sec):                   1.22344 (43.666767%)
  MPI tests (sec):                   9.84617m (0.351426%)
  packing, unpacking, etc. (sec):    1.56848 (55.981808%)

Rate stats:
 throughput (num-reads/sec):       314.098G
 throughput (num-writes/sec):      44.8712G
 throughput (est-FLOPS):           314.098G
 throughput (num-points/sec):      44.8712G
──────────────────────────────────────────────────────────────────────
Performance stats of best trial:
 best-num-steps-done:              320
 best-elapsed-time (sec):          7.65358
 best-throughput (num-reads/sec):  314.256G
 best-throughput (num-writes/sec): 44.8937G
 best-throughput (est-FLOPS):      314.256G
 best-throughput (num-points/sec): 44.8937G
──────────────────────────────────────────────────────────────────────
Performance stats of 50th-percentile trial:
 mid-num-steps-done:               320
 mid-elapsed-time (sec):           7.65605
 mid-throughput (num-reads/sec):   314.155G
 mid-throughput (num-writes/sec):  44.8792G
 mid-throughput (est-FLOPS):       314.155G
 mid-throughput (num-points/sec):  44.8792G
──────────────────────────────────────────────────────────────────────
~~~
