# Improving Communication by Optimizing On-Node Data Movement with Data Layout

***Artifact Evaluation Package***

For the public version of our source code, which also includes more experiments, see [Bricklib on Github](https://github.com/CtopCsUtahEdu/bricklib).

For our code documentation generated from doxygen, see [Bricklib documentation](https://bricks.run/).

## Getting started

### Prerequisite

* A C++11 compatible compiler
    * GCC (>= 8)
    * Intel Compiler (gcc compatibility >= 8)
    * clang (Tested >= 9)
* CMake (>= 3.13)
* MPI
* OpenMP
* Python (>= 3.6)
* (Optional) CUDA Toolkit (>= 9)

### Quick build

Put the source code to *<srcdir>* and have all prerequisite loaded/installed.

~~~
# Prepare
cd <srcdir>
mkdir build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j`nproc`
~~~

#### Common build errors

The following are some of the common errors resulting from different software stacks present. This list is 
non-exhaustive.

When using a non-default compiler, please specify the actual compiler to use on during configuration. Such as when trying 
to use *g++-8 and gcc-8* while *cc -v* is gcc 4.8, add the following command-line option during configure:

`-DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_C_COMPILER=gcc-8`

Some MPI implementations may embed header location in the MPI compiler wrapper causing CMake unable to find them. One 
possible error resulted from this may be `mpi.h not found`. please find out where MPI headers are located and modify
*CMakeLists.txt* in the root folder of the source code.

~~~
 find_package(OpenMP REQUIRED)
 find_package(MPI)
+include_directories(/where/mpi/headers/are)
 find_package(OpenCL 2.0)
~~~

Our CMake script tries to optimize for the platform that builds the source code, this may result in non-optimal 
choices and conflicts when picking the vector folding parameters for the CPU platform. Such as, building the code on a 
Xeon platform without AVX512 and try to run the code on Xeon Phi. Worse still, if the compiler uses wrapper script and
hides the vectorization choice from CMake, which will result in the wrong code. In such cases, you can either modify the 
*CMakeLists.txt* to not use `--march=knl` rather than `--march=native`. Or modify *include/cpuvfold.h* and force any of 
the vectorization choices by deleting others.

#### Setting up on Mac OS

There are a few problems when on MacOS.

* openmp support
* undefined aligned_alloc()
* non-gcc-compliant preprocessor

##### OpenMP support & preprocessor selection

Install gcc from brew which comes with OpenMP support.

`brew install gcc@9`

When running CMake select the c++ compiler

`cmake .. -DCMAKE_CXX_COMPILER=g++-9 -DVS_PREPROCESSOR=cpp-9`

##### aligned_alloc undefined

Use c++17 instead of c++11 by modifying *CMakeLists.txt*.

~~~
-set(CMAKE_CXX_STANDARD 11)
+set(CMAKE_CXX_STANDARD 17)
 set(CMAKE_CXX_EXTENSIONS OFF)
~~~

### Quick runs

In the build directory *<srcdir>/build*, after `make`, *<srcdir>/build/weak* will contains executables used in our 
experiments. All executables share the same run time options, use `-h` to see the help messages. 

~~~
$> cd <srcdir>/build/weak
$> cpu -h
Running MPI with cpu

Program options
    -h: show help (this message)
  MPI downsizing:
    -b: MPI downsize to 2-exponential
  Domain size, pick either one, in array order contiguous first
    -d: comma separated Int[3], overall domain size
    -s: comma separated Int[3], per-process domain size
  Benchmark control:
    -I: number of iterations, default 25
Example usage:
  ./cpu -d 2048,2048,2048
$> # Running on a Ryzen 3700X desktop with AVX2
$> time ./cpu -d 512,512,512 -I 25
Pagesize 4096; MPI Size 1 * OpenMP threads 16
Domain size of 134217728 split among
A total of 1 processes 1x1x1
d3pt7 MPI decomp
Arr: 0.129265
calc [0.122527, 0.122527, 0.122527] (σ: 0)
pack [0, 0, 0] (σ: 0)
  | Pack speed (GB/s): [inf, inf, inf] (σ: -nan)
call [0.00380012, 0.00380012, 0.00380012] (σ: 0)
wait [0.00293775, 0.00293775, 0.00293775] (σ: 0)
  | MPI size (MB): [207.684, 207.684, 207.684] (σ: 0)
  | MPI speed (GB/s): [30.8233, 30.8233, 30.8233] (σ: 0)
perf 1.03832 GStencil/s

Bri: 0.121217
calc [0.119839, 0.119839, 0.119839] (σ: 0)
call [2.08451e-05, 2.08451e-05, 2.08451e-05] (σ: 0)
wait [0.00135744, 0.00135744, 0.00135744] (σ: 0)
  | MPI size (MB): [207.684, 207.684, 207.684] (σ: 0)
  | MPI speed (GB/s): [150.683, 150.683, 150.683] (σ: 0)
perf 1.10725 GStencil/s
Total of 42 parts
./cpu -d 512,512,512 -I 25  795.80s user 4.84s system 1505% cpu 53.195 total
~~~

The two sets of results represent running the code twice, once using arrays and MPI_Type denoted by *Arr*, the second 
time using bricks and our communication method denoted by *Bri*. The results are compared against each other after these 
two runs.

Individual timing and throughput for different steps of the computation are reported for each implementation. Statistics
are shown in the format of *[min, avg, max] (σ: Standard deviation)*. These statistics are computed on the reported 
per-node averages over the time steps, which is why in the previous example, there is no deviation due to using only one
node. The meaning of each of the individual timings:

* *calc* Time spent (in seconds per timestep) for computation
* *pack* Time spent doing packing and unpacking (not used for MPI_Types)
* *call* Time spent doing MPI calls (MPI_Isend/MPI_Irecv)
* *wait* Time spent in MPI_Waitall
* *perf* overall throughput based on the average of per iteration time

Running with MPI is dependent on the specific queuing system. The following example shows running on a Ryzen 3700X 
desktop using 8 mpi processes:

~~~
$> time mpiexec -n 8 ./cpu -d 512,512,512 -I 25
Pagesize 4096; MPI Size 8 * OpenMP threads 2
Domain size of 134217728 split among
A total of 8 processes 2x2x2
d3pt7 MPI decomp
Arr: 0.139424
calc [0.129223, 0.130791, 0.132963] (σ: 0.00135782)
pack [0, 0, 0] (σ: 0)
  | Pack speed (GB/s): [inf, inf, inf] (σ: -nan)
call [0.0020191, 0.00232987, 0.00279464] (σ: 0.000284769)
wait [0.00428886, 0.00630346, 0.00835792] (σ: 0.00131044)
  | MPI size (MB): [53.5429, 53.5429, 53.5429] (σ: 0)
  | MPI speed (GB/s): [5.15976, 6.36884, 8.4633] (σ: 1.15524)
perf 0.962659 GStencil/s

Bri: 0.134219
calc [0.123644, 0.127925, 0.132143] (σ: 0.00316686)
call [9.74655e-05, 0.000267072, 0.000625527] (σ: 0.000178157)
wait [0.00188402, 0.00602688, 0.00992449] (σ: 0.0030408)
  | MPI size (MB): [53.5429, 53.5429, 53.5429] (σ: 0)
  | MPI speed (GB/s): [5.07515, 12.2093, 26.9746] (σ: 9.00187)
perf 0.999991 GStencil/s
Total of 42 parts
mpiexec -n 8 ./cpu -d 512,512,512 -I 25  838.41s user 12.55s system 1467% cpu 57.973 total
~~~

## Experiments in paper

Our experiments based on different code paths in the source that may require rebuilding using different configuration 
options. First, we describe the options available during configuration. Second, we provide the build configurations we
used for the experiments described in our paper.

### Configuration options

The following options are available to be used during the configuration phase. You may see them using *ccmake* or 
*cmake-gui* command. They can be configured on the command-line using `-D<OPTION>=<VALUE>` during configuration.

* *USE_LAYOUT*
  * Disable memmap to use only layout optimization
  * Values: ON/OFF
  * Default: OFF
* *USE_MEMFD*
  * Use memfd instead of shm_open. This interface is supported on linux (>= 3.17) with kernel option 
*CONFIG_MEMFD_CREATE* enabled
  * Values: ON/OFF
  * Default: OFF
* *CUDA_AWARE*
  * Denote if the MPI implementation is CUDA-Aware
  * Values: ON/OFF
  * Default: ON
* *USE_TYPES*
  * Make the reference computation use MPI_TYPES
  * Values: ON/OFF
  * Default: ON
  * When *OFF* is selected, an optimized in-house packing routine will be used
* *BARRIER_TIMESTEPS*
  * Use barrier to reduce timing variation
  * Values: ON/OFF
  * Default: OFF
  * *OFF* during all our experiments
* *BRICK_BUILD_TEST*
  * Build the experiments
  * Must be *ON*
* *MPI_STENCIL*
  * Stencil used during the MPI experiments
  * Values: 7PT/13PT/45PT/125PT
  * Default: 7PT
  
### Configurations corresponding to the experiments in the paper

#### CPU experiment K1

* *MemMap*
  * Configuration: `-DUSE_LAYOUT=OFF -DMPI_STENCIL=7PT` (default)
  * Build target: `make weak-cpu`
  * Executable: `<srcdir>/build/weak/cpu`
  * Result reported: *Bri*
* *Layout*
  * Configuration: `-DUSE_LAYOUT=ON -DMPI_STENCIL=7PT`
  * Build target: `make weak-cpu`
  * Executable: `<srcdir>/build/weak/cpu`
  * Result reported: *Bri*
* *MPI_Types*
  * Configuration: `-DUSE_TYPES=ON -DMPI_STENCIL=7PT` (default)
  * Build target: `make weak-cpu`
  * Executable: `<srcdir>/build/weak/cpu`
  * Result reported: *Arr*
  
#### CPU experiment K2

* *MemMap - 7pt*
  * Configuration: `-DUSE_LAYOUT=OFF -DMPI_STENCIL=7PT` (default)
  * Build target: `make weak-cpu`
  * Executable: `<srcdir>/build/weak/cpu`
  * Result reported: *Bri*
* *MemMap - 125pt*
  * Configuration: `-DUSE_LAYOUT=OFF -DMPI_STENCIL=125PT` (default)
  * Build target: `make weak-cpu`
  * Executable: `<srcdir>/build/weak/cpu`
  * Result reported: *Bri*
  
#### GPU experiment V1

* *Layout<sup>CA</sup>*
  * Configuration: `-DUSE_LAYOUT=ON -DCUDA_AWARE=ON -DMPI_STENCIL=7PT`
  * Build target: `make weak-cuda`
  * Executable: `<srcdir>/build/weak/cuda`
  * Result reported: *Bri*
* *MemMap<sup>UM</sup>*
  * Configuration: `-DUSE_LAYOUT=OFF -DMPI_STENCIL=7PT` (default)
  * Build target: `make weak-cuda-mmap`
  * Executable: `<srcdir>/build/weak/cuda-mmap`
  * Result reported: *Bri*
* *MPI_Types<sup>UM</sup>*
  * Configuration: `-DUSE_LAYOUT=OFF -DMPI_STENCIL=7PT` (default)
  * Build target: `make weak-cuda-mmap`
  * Executable: `<srcdir>/build/weak/cuda-mmap`
  * Result reported: *Arr*

#### GPU experiment V2

* *Layout<sup>CA</sup>*
  * Configuration (7pt): `-DUSE_LAYOUT=ON -DCUDA_AWARE=ON -DMPI_STENCIL=7PT`
  * Configuration (125pt): `-DUSE_LAYOUT=ON -DCUDA_AWARE=ON -DMPI_STENCIL=125PT`
  * Build target: `make weak-cuda`
  * Executable: `<srcdir>/build/weak/cuda`
  * Result reported: *Bri*
* *MemMap<sup>UM</sup>*
  * Configuration (7pt): `-DUSE_LAYOUT=OFF -DMPI_STENCIL=7PT` (default)
  * Configuration (125pt): `-DUSE_LAYOUT=OFF -DMPI_STENCIL=125PT`
  * Build target: `make weak-cuda-mmap`
  * Executable: `<srcdir>/build/weak/cuda-mmap`
  * Result reported: *Bri*
* *MPI_Types<sup>UM</sup>*
  * Configuration (7pt): `-DUSE_LAYOUT=OFF -DMPI_STENCIL=7PT` (default)
  * Configuration (125pt): `-DUSE_LAYOUT=OFF -DMPI_STENCIL=125PT`
  * Build target: `make weak-cuda-mmap`
  * Executable: `<srcdir>/build/weak/cuda-mmap`
  * Result reported: *Arr*

#### Other results

These results are not the main contribution of the paper but rather used to illustrate a point. They may not be
streamlined in this artifact package.

*Proposed* in Figure 1 uses the same result as *MemMap* in experiment K1. 

*Layout* in Figure 4 uses the same result as *Layout* in experiment K1. *Basic* in Figure 4 uses the following 
configurations,

`-DUSE_LAYOUT=ON`

with the following modification of *include/brick-mpi.h*

~~~
 extern std::vector<BitSet> surface3d_good;
 extern std::vector<BitSet> surface3d_normal, surface3d_bad;
-#define surface3d surface3d_good
+#define surface3d surface3d_bad
~~~

Figure 16 is generated by modifying *include/brick-mpi.h* by making the library think that the page size is larger than
what is reported. For example, Theta is a system with 4KiB pages, the following is used for the 16KiB experiment. 
Configuration for all experiments in this figure is the same as *MemMap* in K1. *MPI_Types* uses the *MPI_Types* result 
from K1.

~~~
-    int pagesize = sysconf(_SC_PAGESIZE);
+    int pagesize = sysconf(_SC_PAGESIZE) * 4;
     int bSize = cal_size<BDims ...>::value * sizeof(bElem) * numfield;
 
     if (std::max(bSize, pagesize) % std::min(bSize, pagesize) != 0)
       throw std::runtime_error("brick size must be a factor/multiple of pagesize.");
~~~

### Loaded modules for experiments

#### Theta KNL

~~~
Currently Loaded Modulefiles:
  1) modules/3.2.11.3
  2) alps/6.6.43-6.0.7.1_5.51__ga796da32.ari
  3) udreg/2.3.2-6.0.7.1_5.15__g5196236.ari
  4) ugni/6.0.14.0-6.0.7.1_3.15__gea11d3d.ari
  5) gni-headers/5.0.12.0-6.0.7.1_3.13__g3b1768f.ari
  6) dmapp/7.1.1-6.0.7.1_6.6__g45d1b37.ari
  7) xpmem/2.2.15-6.0.7.1_5.13__g7549d06.ari
  8) llm/21.3.530-6.0.7.1_5.4__g3b4230e.ari
  9) nodehealth/5.6.14-6.0.7.1_8.50__gd6a82f3.ari
 10) system-config/3.5.2794-6.0.7.1_9.1__g651274ab.ari
 11) Base-opts/2.4.135-6.0.7.1_5.6__g718f891.ari
 12) intel/19.0.5.281
 13) craype-network-aries
 14) craype/2.6.1
 15) cray-libsci/19.06.1
 16) pmi/5.0.14
 17) atp/2.1.3
 18) rca/2.2.18-6.0.7.1_5.52__g2aa4f39.ari
 19) perftools-base/7.0.5
 20) PrgEnv-intel/6.0.5
 21) craype-mic-knl
 22) cray-mpich/7.7.10
 23) nompirun/nompirun
 24) darshan/3.1.8
 25) trackdeps
 26) xalt
~~~

#### Summit V100

~~~
Currently Loaded Modules:
  1) hsi/5.0.2.p5            6) cuda/10.1.243
  2) xalt/1.2.0              7) gcc/7.4.0
  3) lsf-tools/2.0           8) spectrum-mpi/10.3.1.2-20200121
  4) darshan-runtime/3.1.7   9) python/3.6.6-anaconda3-5.3.0
  5) DefApps                10) cmake/3.15.2
~~~

## YASK

YASK package is obtained from [Github](https://github.com/intel/yask) which is used in some of our comparisons.

The basic compilation usage of YASK for compiling for Intel KNL platform on 7pt stencil that we used as our comparison 
is shown below. 

`make arch=knl mpi=1 real_bytes=8 stencil=3axis radius=1`

125pt stencil:

`make arch=knl mpi=1 real_bytes=8 stencil=cube radius=2`

Stencils can be executed using:

`./bin/yask.sh -stencil 3axis -arch=knl -g 512`

The actual run command of the stencil code depends on the job system provided. We pick the best result out of the three 
trials reported when running the stencil.
