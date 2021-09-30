#include "stencils_dpc.h"
#include <iostream>
#include "bricksetup.h"
#include "multiarray.h"
#include "brick-sycl.h"
#include "brickcompare.h"

#define COEFF_SIZE 129
#define VSVEC "DPCPP"
#define VFOLD 8, 8
#define DPC_ITER 100

typedef std::function<sycl::event()> timeable_func;

double dpctime_func(timeable_func func) {
  func(); // Warm up
  double elapsed = 0;
  for (int i = 0; i < DPC_ITER; i++) {
      sycl::event exec = func();
      exec.wait();
      // time here is recorded in ns
      auto end = exec.get_profiling_info<sycl::info::event_profiling::command_end>();
      auto start = exec.get_profiling_info<sycl::info::event_profiling::command_start>();
      elapsed += (end - start) * (1.0e-9);
  }
  return elapsed / DPC_ITER;
}

void d3pt7_brick(sycl::id<3> id, unsigned (*grid)[STRIDEB][STRIDEB], BrickInfo<3> *bInfo, sycl::accessor<bElem, 1, sycl::access::mode::read_write> dat_accessor,
            bElem *coeff) {
    bElem *bDat = (bElem *) dat_accessor.get_pointer();
    auto bSize = cal_size<BDIM>::value;
    DPCBrick<Dim<BDIM>, Dim<VFOLD>> bIn(bInfo, bDat, bSize * 2, 0);
    DPCBrick<Dim<BDIM>, Dim<VFOLD>> bOut(bInfo, bDat, bSize * 2, bSize);
    long tk = GB + (id[2] / TILE);
    long tj = GB + (id[1] / TILE);
    long ti = GB + (id[0] / TILE);
    long k = (id[2] % TILE);
    long j = (id[1] % TILE);
    long i = (id[0] % TILE);
    unsigned b = grid[tk][tj][ti];
    bOut[b][k][j][i] = coeff[5] * bIn[b][k + 1][j][i] + coeff[6] * bIn[b][k - 1][j][i] +
                        coeff[3] * bIn[b][k][j + 1][i] + coeff[4] * bIn[b][k][j - 1][i] +
                        coeff[1] * bIn[b][k][j][i + 1] + coeff[2] * bIn[b][k][j][i - 1] +
                        coeff[0] * bIn[b][k][j][i];
}

void d3pt7_brick_trans(sycl::id<3> id, unsigned (*grid)[STRIDEB][STRIDEB], BrickInfo<3> *bInfo, sycl::accessor<bElem, 1, sycl::access::mode::read_write> dat_accessor,
            bElem *coeff) {
    bElem *bDat = (bElem *) dat_accessor.get_pointer();
    auto bSize = cal_size<BDIM>::value;
    DPCBrick<Dim<BDIM>, Dim<VFOLD>> bIn(bInfo, bDat, bSize * 2, 0);
    DPCBrick<Dim<BDIM>, Dim<VFOLD>> bOut(bInfo, bDat, bSize * 2, bSize);
    long tk = GB + (id[2] / TILE);
    long tj = GB + (id[1] / TILE);
    long ti = GB + (id[0] / TILE);
    long sglid = (id[0] % TILE);
    unsigned b = grid[tk][tj][ti];
    brick("7pt.py", VSVEC, (BDIM), (VFOLD), b);
}

void d3pt7_arr(sycl::id<3> id, bElem (*in)[STRIDE][STRIDE], bElem (*out)[STRIDE][STRIDE], bElem *coeff) {
    long k = PADDING + GZ + id[2];
    long j = PADDING + GZ + id[1];
    long i = PADDING + GZ + id[0];
    out[k][j][i] = coeff[5] * in[k + 1][j][i] + coeff[6] * in[k - 1][j][i] +
                        coeff[3] * in[k][j + 1][i] + coeff[4] * in[k][j - 1][i] +
                        coeff[1] * in[k][j][i + 1] + coeff[2] * in[k][j][i - 1] +
                        coeff[0] * in[k][j][i];
}

#define bIn(i, j, k) arr_in[k][j][i]
#define bOut(i, j, k) arr_out[k][j][i]

void d3pt7_arr_scatter(sycl::id<3> id, bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE], bElem *coeff) {
  unsigned sglid = id[0];
  long k = GZ + id[2];
  long j = GZ + id[1];
  long i = GZ + id[0] * 64;
  tile("7pt.py", VSVEC, (TILE, TILE, 64), ("k", "j", "i"), (1, 1, 64));
}

#undef bIn
#undef bOut


void d3pt7dpc() {
    unsigned *grid_ptr;

    BrickInfo<3> bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
    unsigned *grid_dev;
    {
        unsigned size = (STRIDEB * STRIDEB * STRIDEB) * sizeof(unsigned);
        gpuMalloc(&grid_dev, size);
        gpuMemcpy(grid_dev, grid_ptr, size, gpuMemcpyHostToDevice);
    }
    auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_dev;

    BrickInfo<3> _bInfo_dev = movBrickInfo(bInfo, gpuMemcpyHostToDevice);
    BrickInfo<3> *bInfo_dev;
    {
        unsigned size = sizeof(BrickInfo < 3 > );
        gpuMalloc(&bInfo_dev, size);
        gpuMemcpy(bInfo_dev, &_bInfo_dev, size, gpuMemcpyHostToDevice);
    }

    unsigned data_size = STRIDE * STRIDE * STRIDE * sizeof(bElem);
    bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
    bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
    bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
    bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

    bElem *coeff_dev;
    {
        unsigned size = COEFF_SIZE * sizeof(bElem);
        gpuMalloc(&coeff_dev, size);
        gpuMemcpy(coeff_dev, coeff, size, gpuMemcpyHostToDevice);
    }

    bElem *in_dev, *out_dev;
    {
        gpuMalloc(&in_dev, data_size);
        gpuMalloc(&out_dev, data_size);
        gpuMemcpy(in_dev, in_ptr, data_size, gpuMemcpyHostToDevice);
        gpuMemcpy(out_dev, out_ptr, data_size, gpuMemcpyHostToDevice);
    }

    auto bSize = cal_size<BDIM>::value;
    auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
    Brick <Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
    Brick <Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

    copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

    std::cout << "CPU: " << cpu_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    auto arr_func = [&arr_in, &arr_out]() -> void {
        _TILEFOR arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                                    coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                                    coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                                    coeff[0] * arr_in[k][j][i];
    };
    
    auto brick_func = [&grid_dev, &bInfo_dev, &bStorage, &coeff_dev]() -> sycl::event {
        // using a buffer and accessor the data is returned to the host at the end of execution
        auto dat_buffer = sycl::buffer<bElem>(bStorage.dat.get(), sycl::range(bStorage.chunks * bStorage.step));
        dat_buffer.set_final_data(bStorage.dat.get());
        return gpu_queue.submit([&](sycl::handler& cgh) {
            auto dat_accessor = dat_buffer.get_access<sycl::access::mode::read_write>(cgh);
            cgh.parallel_for<class Brickd3pt7>(sycl::range(N, N, N), [=](sycl::item<3> item) {
                d3pt7_brick(item.get_id(), (unsigned (*)[STRIDEB][STRIDEB]) grid_dev, bInfo_dev, dat_accessor, coeff_dev);
            });
        });
    };
    auto dpcarr_func = [&in_dev, &out_dev, &coeff_dev]() -> sycl::event {
        return gpu_queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class Arrayd3pt7>(sycl::range(N, N, N), [=](sycl::item<3> item) {
                d3pt7_arr(item.get_id(), (bElem (*)[STRIDE][STRIDE]) in_dev, (bElem (*)[STRIDE][STRIDE]) out_dev, coeff_dev);
            });
        });
    };
    auto dpcarr_scatter = [&in_dev, &out_dev, &coeff_dev]() -> sycl::event {
        return gpu_queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class Arrayd3pt7Scatter>(sycl::range(N, N, N / 64), [=](sycl::item<3> item) {
                d3pt7_arr_scatter(item.get_id(), (bElem (*)[STRIDE][STRIDE]) in_dev, (bElem (*)[STRIDE][STRIDE]) out_dev, coeff_dev);
            });
        });
    };
    auto dcpbrick_scatter = [&grid_dev, &bInfo_dev, &bStorage, &coeff_dev]() -> sycl::event {
        auto dat_buffer = sycl::buffer<bElem>(bStorage.dat.get(), sycl::range(bStorage.chunks * bStorage.step));
        dat_buffer.set_final_data(bStorage.dat.get());
        return gpu_queue.submit([&](sycl::handler& cgh) {
            auto dat_accessor = dat_buffer.get_access<sycl::access::mode::read_write>(cgh);
            cgh.parallel_for<class Brickd4pt7Scatter>(sycl::range(N, N, N / 32), [=](sycl::item<3> item) {
                d3pt7_brick_trans(item.get_id(), (unsigned (*)[STRIDEB][STRIDEB]) grid_dev, bInfo_dev, dat_accessor, coeff_dev);
            });
        });
    };

    std::cout << "d3pt7" << std::endl;
    arr_func();
    std::cout << "Arr: " << dpctime_func(dpcarr_func) << std::endl;
    std::cout << "Arr scatter: " << dpctime_func(dpcarr_scatter) << std::endl;
    std::cout << "Bri: " << dpctime_func(brick_func) << std::endl;
    if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");
    std::cout << "Bri scatter: " << dpctime_func(dcpbrick_scatter) << std::endl;
    if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    free(in_ptr);
    free(out_ptr);
    free(bInfo.adj);
    gpuFree(bInfo_dev);
    gpuFree(in_dev);
    gpuFree(out_dev);
}