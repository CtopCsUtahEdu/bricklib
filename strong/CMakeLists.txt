VSTARGET(N3MPI main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/main-out.cpp)
add_executable(strong-cpu ${VSTARGET_N3MPI_OUTPUT} args.cpp)
target_link_libraries(strong-cpu brick brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(strong-cpu
        PROPERTIES OUTPUT_NAME cpu)

if(CMAKE_CUDA_COMPILER)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    VSTARGET(C3MPI main.cu ${CMAKE_CURRENT_SOURCE_DIR}/main-out.cu)
    add_executable(strong-cuda ${VSTARGET_C3MPI_OUTPUT} args.cpp)
    target_compile_definitions(strong-cuda PRIVATE DECOMP_PAGEUNALIGN)
    target_link_libraries(strong-cuda brick brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
    set_target_properties(strong-cuda
            PROPERTIES OUTPUT_NAME cuda)

    include_directories(
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    VSTARGET(C3MPIMMAP mmap.cu ${CMAKE_CURRENT_SOURCE_DIR}/mmap-out.cu)
    add_executable(strong-cuda-mmap ${VSTARGET_C3MPIMMAP_OUTPUT} args.cpp)
    target_compile_definitions(strong-cuda-mmap PRIVATE DECOMP_PAGEUNALIGN)
    target_link_libraries(strong-cuda-mmap brick brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT} cuda)
    set_target_properties(strong-cuda-mmap
            PROPERTIES OUTPUT_NAME cuda-mmap)
endif ()

