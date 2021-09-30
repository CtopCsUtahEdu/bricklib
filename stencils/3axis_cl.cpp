//
// Created by Tuowen Zhao on 11/9/19.
//

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 210

#include "brick-opencl.h"
#include "brickcompare.h"
#include "multiarray.h"
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <stencils/stencils.h>

cl::Device *device;
cl::Context *context;

void clinit() {
  device = new cl::Device;
  bool selected = false;
  cl::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (auto plat : platforms) {
    cl::vector<cl::Device> devices;
    plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (!devices.empty()) {
      std::cout << "Platform: " << plat.getInfo<CL_PLATFORM_NAME>() << std::endl;
      std::cout << "Vendor: " << plat.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

      for (auto dev : devices) {
        std::cout << "\tDevice: " << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "\tHardware version: " << dev.getInfo<CL_DEVICE_VERSION>() << std::endl;
        std::cout << "\tSoftware version: " << dev.getInfo<CL_DRIVER_VERSION>() << std::endl;
        std::cout << "\tC Lang version: " << dev.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
        std::cout << "\tParallel units: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                  << std::endl;
        bool sfl = dev.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_subgroup_shuffle_relative") !=
                   std::string::npos;
        std::cout << "\tShuffle: " << (sfl ? "Yes" : "No") << std::endl;

        if (!selected && dev.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU && sfl) {
          *device = dev;
          selected = true;
          std::cout << "\t[Selected]" << std::endl;
        }
      }
    }
  }
  if (!selected)
    throw std::runtime_error("No suitable device");
  context = new cl::Context(*device);
};

void cldestroy() {
  delete device;
  delete context;
}

void d3pt7() {
  unsigned *grid_ptr;

  std::ifstream fin("3axis_cl_krnl-out.c");
  std::string program_str((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
  cl::Program brickStencil_prog(program_str);

  try {
    brickStencil_prog.build(CL_KRNL_OPTIONS);
  } catch (...) {
    cl_int buildErr = CL_SUCCESS;
    auto buildInfo = brickStencil_prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
    for (auto &pair : buildInfo) {
      std::cerr << pair.second << std::endl << std::endl;
    }
  }

  auto brickStencil_cl =
      cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned>(brickStencil_prog,
                                                                                  "stencil");

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  auto grid = (unsigned(*)[STRIDEB][STRIDEB])grid_ptr;

  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem(*)[STRIDE][STRIDE])in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem(*)[STRIDE][STRIDE])out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick<Dim<BDIM>, Dim<OCL_VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<OCL_VFOLD>> bOut(&bInfo, bStorage, bSize);

  auto command_queue = cl::CommandQueue(*context, *device, cl::QueueProperties::Profiling);
  cl::CommandQueue::setDefault(command_queue);

  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr,
                 grid_ptr, bIn);
  // Setup bricks for opencl
  auto coeff_buf = cl::Buffer(*context, CL_MEM_READ_ONLY, 129 * sizeof(bElem));
  cl::enqueueWriteBuffer(coeff_buf, false, 0, 129 * sizeof(bElem), (void *)coeff);
  std::vector<unsigned> bIdx;

  for (long tk = GB; tk < STRIDEB - GB; ++tk)
    for (long tj = GB; tj < STRIDEB - GB; ++tj)
      for (long ti = GB; ti < STRIDEB - GB; ++ti)
        bIdx.push_back(grid[tk][tj][ti]);

  auto bIdx_buf = cl::Buffer(*context, CL_MEM_READ_ONLY, bIdx.size() * sizeof(unsigned));
  cl::enqueueWriteBuffer(bIdx_buf, false, 0, bIdx.size() * sizeof(unsigned), (void *)bIdx.data());

  size_t adj_size = bInfo.nbricks * 27 * sizeof(unsigned);
  auto adj_buf = cl::Buffer(*context, CL_MEM_READ_ONLY, adj_size);
  cl::enqueueWriteBuffer(adj_buf, false, 0, adj_size, (void *)bInfo.adj);

  size_t bDat_size = bStorage.chunks * bStorage.step * sizeof(bElem);
  auto bDat_buf = cl::Buffer(*context, CL_MEM_READ_WRITE, bDat_size);
  cl::enqueueWriteBuffer(bDat_buf, false, 0, bDat_size, (void *)bStorage.dat.get());

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                                coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                                coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                                coeff[0] * arr_in[k][j][i];
  };

  std::cout << "d3pt7" << std::endl;
  std::cout << "Arr: " << time_func(arr_func) << std::endl;

  int ocl_iter = 1000;
  cl::Event st_event =
      brickStencil_cl(cl::EnqueueArgs(cl::NDRange(OCL_SUBGROUP * 1024), cl::NDRange(OCL_SUBGROUP)),
                      bDat_buf, coeff_buf, adj_buf, bIdx_buf, bIdx.size());
  for (int i = 0; i < ocl_iter; ++i)
    brickStencil_cl(cl::EnqueueArgs(cl::NDRange(OCL_SUBGROUP * 1024), cl::NDRange(OCL_SUBGROUP)),
                    bDat_buf, coeff_buf, adj_buf, bIdx_buf, bIdx.size());
  cl::Event ed_event =
      brickStencil_cl(cl::EnqueueArgs(cl::NDRange(OCL_SUBGROUP * 1024), cl::NDRange(OCL_SUBGROUP)),
                      bDat_buf, coeff_buf, adj_buf, bIdx_buf, bIdx.size());
  ed_event.wait();
  auto btime = (double)(ed_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                        st_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) /
               1e9 / (ocl_iter + 2);

  std::cout << "OCL brick: " << btime << std::endl;
  cl::enqueueReadBuffer(bDat_buf, true, 0, bDat_size, bStorage.dat.get());
  command_queue.flush();
  command_queue.finish();

  if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr,
                       bOut)) {
    std::cout << "result mismatch!" << std::endl;
    // Identify mismatch
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          auto b = grid[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                auto aval = arr_out[tk * TILE + k + PADDING][tj * TILE + j + PADDING]
                                   [ti * TILE + i + PADDING];
                auto diff = abs(bOut[b][k][j][i] - aval);
                auto sum = abs(bOut[b][k][j][i]) + abs(aval);
                if (sum > 1e-6 && diff / sum > 1e-6)
                  std::cout << "mismatch at " << ti * TILE + i - TILE << " : "
                            << tj * TILE + j - TILE << " : " << tk * TILE + k - TILE << std::endl;
              }
        }
  }

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bInfo.adj);
}
