# Bricks

***Distributed Performance-portable Stencil Compuitation***

# What is bricks

Bricks is a data layout and code generation framework, enable performance-portable stencil across Intel CPUs, Intel 
Knights Landing, Intel GPUs, NVIDIA GPUs, AMD GPUs. Math kernel code and data manipulation code is shared across these 
platform, while achieving best-in-class performance on all platforms simultaneously. Especially, Brick layout is well 
suited to higher-order(bigger-wider) stencil computations.

---

![Performance portability of bricks](media/performance-portability.png)

**Achieve consistent 1.9x-4.9x speedup across different architectures including Skylake, Intel Knights Landing, and NVidia P100 GPUs**

---

Brick layout is flexible, allows flexible domain shapes and enables fast "ghost cell" data communication with MPI.

---

![Fast MPI-communication](media/fast-MPI-ghostzone.png)

**Efficient ghost zone exchange achieves up to 10x faster than state-of-the-art YASK and up to 600x compared to cray-mpich/7.7.10 MPI_Types.**

---

## Get the Code

Get code from [Github](https://github.com/CtopCsUtahEdu/brick), and start exploring the code documentation here.

## Acknowledgements

* This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy's Office of Science and National Nuclear Security Administration.
* This research used resources of the Oak Ridge Leadership Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.
* This research used resources of the Argonne Leadership Computing Facility at Argonne National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under contract DE-AC02-06CH11357.
* This research used resources in Lawrence Berkeley National Laboratory and the National Energy Research Scientific Computing Center, which are supported by the U.S. Department of Energy Office of Science’s Advanced Scientific Computing Research program under contract number DE-AC02-05CH11231.

## Publications

@cite zhao2018 Zhao, Tuowen, Samuel Williams, Mary Hall, and Hans Johansen. "Delivering Performance-Portable Stencil Computations on CPUs and GPUs Using Bricks." In 2018 IEEE/ACM International Workshop on Performance, Portability and Productivity in HPC (P3HPC), pp. 59-70. IEEE, 2018. 

@cite zhao2019 Zhao, Tuowen, Protonu Basu, Samuel Williams, Mary Hall, and Hans Johansen. "Exploiting reuse and vectorization in blocked stencil computations on CPUs and GPUs." In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, p. 52. ACM, 2019.
