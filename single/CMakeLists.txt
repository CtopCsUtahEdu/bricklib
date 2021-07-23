VSTARGET(N3AXIS ../stencils/3axis.cpp ../stencils/3axis-out.cpp)
add_executable(single-cpu cpu.cpp ${VSTARGET_N3AXIS_OUTPUT} ../stencils/copy.cpp)
target_link_libraries(single-cpu brickhelper)
set_target_properties(single-cpu
        PROPERTIES OUTPUT_NAME cpu)

add_executable(single-mpi mpi.cpp ../src/brick-mpi.cpp)
target_link_libraries(single-mpi brick brickhelper ${MPI_CXX_LIBRARIES})
set_target_properties(single-mpi
        PROPERTIES OUTPUT_NAME mpi)

if (HAS_SYCL)
    set(SYCL3AXIS_INCLUDE_DIRS
            "${SYCL_DIR}/include"
            )
    set(SYCL3AXIS_COMPILE_OPTIONS
            "-fsycl"
            )
    VSTARGET(SYCL3AXIS ../stencils/3axis_sycl.cpp ${CMAKE_CURRENT_BINARY_DIR}/3axis_sycl-out.cpp)
    add_executable(single-sycl sycl.cpp ${VSTARGET_SYCL3AXIS_OUTPUT})
    target_link_libraries(single-sycl brick brickhelper stdc++ sycl OpenCL)
    target_include_directories(single-sycl PUBLIC "${SYCL_DIR}/include")
    target_link_directories(single-sycl PUBLIC "${SYCL_DIR}/lib")
    # sycl need option for both phase
    target_compile_options(single-sycl PUBLIC "-fsycl")
    target_link_options(single-sycl PUBLIC "-fsycl")
    set_target_properties(single-sycl
            PROPERTIES OUTPUT_NAME sycl)
endif ()

if (OpenCL_FOUND)
    add_executable(single-opencl opencl.cpp ../stencils/3axis_cl.cpp)
    target_compile_definitions(single-opencl PRIVATE "CL_KRNL_OPTIONS=\"-I${PROJECT_SOURCE_DIR}/include\"")
    VSTARGET(CL3AXIS ../stencils/3axis_cl_krnl.c ${CMAKE_CURRENT_BINARY_DIR}/3axis_cl_krnl-out.c)
    add_custom_target(cl3axis
            SOURCES ${VSTARGET_CL3AXIS_OUTPUT})
    target_link_libraries(single-opencl brick brickhelper ${OpenCL_LIBRARIES})
    target_include_directories(single-opencl PUBLIC "${OpenCL_INCLUDE_DIRS}")
    add_dependencies(single-opencl cl3axis)
    set_target_properties(single-opencl
            PROPERTIES OUTPUT_NAME opencl)
endif ()

if (CMAKE_CUDA_COMPILER)
    set(C3AXIS_INCLUDE_DIRS
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
    VSTARGET(C3AXIS ../stencils/3axis.cu ../stencils/3axis-out.cu)
    add_executable(single-cuda cuda.cpp ${VSTARGET_C3AXIS_OUTPUT})
    target_link_libraries(single-cuda brickhelper)
    set_target_properties(single-cuda
            PROPERTIES OUTPUT_NAME cuda)
    target_include_directories(single-cuda PUBLIC
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
endif ()

if (USE_HIP)
    VSTARGET(H3AXIS ../stencils/3axis.hip.cpp ../stencils/3axis-out.hip.cpp)
    set(HIP_SRCS hip.cpp ${VSTARGET_H3AXIS_OUTPUT})
    set_source_files_properties(${HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(single-hip ${HIP_SRCS} CLANG_OPTIONS ${CMAKE_CXX_FLAGS})
    target_link_libraries(single-hip brickhelper hip::host)
    set_target_properties(single-hip
            PROPERTIES OUTPUT_NAME hip)
endif ()

if (USE_DPC)
   VSTARGET(D3AXIS ../stencils/3axis_dpc.cpp ../stencils/3axis_dpc-out.cpp)
   set(CMAKE_CXX_COMPILER "dpcpp")
   set(CMAKE_LINKER "dpcpp") 
   set(CMAKE_AR "llvm-ar") 
   set(CMAKE_RANLIB "llvm-ranlib") 
   set(CMAKE_CXX_STANDARD 17) 
   add_executable(single-dpc dpc.cpp ${VSTARGET_D3AXIS_OUTPUT})
   target_link_libraries(single-dpc brickhelper)
   target_include_directories(single-dpc PUBLIC "$(which dpcpp)/../../sycl/")
   set_target_properties(single-dpc PROPERTIES OUTPUT_NAME dpc)
endif ()

