VSTARGET(N3MPI main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/main-out.cpp)
add_executable(weak-cpu ${VSTARGET_N3MPI_OUTPUT} args.cpp)
target_link_libraries(weak-cpu brick brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(weak-cpu
        PROPERTIES OUTPUT_NAME cpu)

VSTARGET(N3MPI onesided.cpp ${CMAKE_CURRENT_SOURCE_DIR}/onesided-out.cpp)
add_executable(weak-onesided ${VSTARGET_N3MPI_OUTPUT})
target_link_libraries(weak-onesided brick brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(weak-onesided
        PROPERTIES OUTPUT_NAME onesided)

VSTARGET(N3MPI shmem.cpp ${CMAKE_CURRENT_SOURCE_DIR}/shmem-out.cpp)
add_executable(weak-shmem EXCLUDE_FROM_ALL ${VSTARGET_N3MPI_OUTPUT})
message(STATUS ${MPI_CXX_LIBRARIES})
target_link_libraries(weak-shmem brick brickhelper ${MPI_CXX_LIBRARIES} /usr/lib/openmpi/liboshmem.so ${LIBRT})
set_target_properties(weak-shmem
        PROPERTIES OUTPUT_NAME shmem)

if(CMAKE_CUDA_COMPILER)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    VSTARGET(C3MPI main.cu ${CMAKE_CURRENT_SOURCE_DIR}/main-out.cu)
    add_executable(weak-cuda ${VSTARGET_C3MPI_OUTPUT} args.cpp)
    target_compile_definitions(weak-cuda PRIVATE DECOMP_PAGEUNALIGN)
    target_link_libraries(weak-cuda brick brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
    set_target_properties(weak-cuda
            PROPERTIES OUTPUT_NAME cuda)

    VSTARGET(C3MPIMMAP mmap.cu ${CMAKE_CURRENT_SOURCE_DIR}/mmap-out.cu)
    add_executable(weak-cuda-mmap ${VSTARGET_C3MPIMMAP_OUTPUT} args.cpp)
    target_link_libraries(weak-cuda-mmap brick brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT} cuda)
    set_target_properties(weak-cuda-mmap
            PROPERTIES OUTPUT_NAME cuda-mmap)
endif ()