VSTARGET(N3MPI main.cpp main-out.cpp)
add_executable(weak-cpu ${VSTARGET_N3MPI_OUTPUT} args.cpp)
target_link_libraries(weak-cpu brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(weak-cpu
        PROPERTIES OUTPUT_NAME cpu)

VSTARGET(N4MPI main-4d.cpp main-4d-out.cpp)
add_executable(weak-cpu-4d ${VSTARGET_N4MPI_OUTPUT} args.cpp)
target_link_libraries(weak-cpu-4d brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(weak-cpu-4d
        PROPERTIES OUTPUT_NAME cpu-4d)

VSTARGET(N3MPI_ONESIDED onesided.cpp onesided-out.cpp)
add_executable(weak-onesided ${VSTARGET_N3MPI_ONESIDED_OUTPUT})
target_link_libraries(weak-onesided brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(weak-onesided
        PROPERTIES OUTPUT_NAME onesided)

VSTARGET(N3MPI_SHMEM shmem.cpp shmem-out.cpp)
add_executable(weak-shmem EXCLUDE_FROM_ALL ${VSTARGET_N3MPI_SHMEM_OUTPUT})
message(STATUS ${MPI_CXX_LIBRARIES})
target_link_libraries(weak-shmem brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} /usr/lib/openmpi/liboshmem.so ${LIBRT})
set_target_properties(weak-shmem
        PROPERTIES OUTPUT_NAME shmem)

if(CMAKE_CUDA_COMPILER)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    VSTARGET(C3MPI main.cu main-out.cu)
    add_executable(weak-cuda ${VSTARGET_C3MPI_OUTPUT} args.cpp)
    target_compile_definitions(weak-cuda PRIVATE DECOMP_PAGEUNALIGN)
    target_link_libraries(weak-cuda brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
    set_target_properties(weak-cuda
            PROPERTIES OUTPUT_NAME cuda)

    VSTARGET(C3MPIMMAP mmap.cu mmap-out.cu)
    add_executable(weak-cuda-mmap ${VSTARGET_C3MPIMMAP_OUTPUT} args.cpp)
    target_link_libraries(weak-cuda-mmap brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT} cuda)
    set_target_properties(weak-cuda-mmap
            PROPERTIES OUTPUT_NAME cuda-mmap)
endif ()

if (USE_HIP)
    set(H3MPI_INCLUDE_DIRS "${HIP_PATH}/include")
    VSTARGET(H3MPI main.hip.cpp main-out.hip.cpp)
    set(HIP_SRCS ${VSTARGET_H3MPI_OUTPUT} args.cpp)
    set_source_files_properties(${HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(weak-hip ${HIP_SRCS} CLANG_OPTIONS ${CMAKE_CXX_FLAGS})
    target_link_libraries(weak-hip brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT} hip::host)
    set_target_properties(weak-hip
            PROPERTIES OUTPUT_NAME hip)

    set(H3MPIMMAP_INCLUDE_DIRS "${HIP_PATH}/include")
    VSTARGET(H3MPIMMAP mmap.hip.cpp mmap-out.hip.cpp)
    set(HIP_SRCS ${VSTARGET_H3MPIMMAP_OUTPUT} args.cpp)
    set_source_files_properties(${HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(weak-hip-mmap ${HIP_SRCS} CLANG_OPTIONS ${CMAKE_CXX_FLAGS})
    target_link_libraries(weak-hip-mmap brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT} hip::host)
    set_target_properties(weak-hip-mmap
            PROPERTIES OUTPUT_NAME hip-mmap)
endif ()