cmake_minimum_required(VERSION 3.12)
project(brick LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
endif()

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCL 2.0)

option(HAS_SYCL "Denote compiler that support sycl" ON)

if (HAS_SYCL)
    get_filename_component(bin_dir "${CMAKE_CXX_COMPILER}" PATH)
    get_filename_component(SYCL_DIR "${bin_dir}" PATH)

    message(STATUS "Setting include/lib path according to compiler path: ${CMAKE_CXX_COMPILER}")
    message(STATUS "SYCL compiler install dir: ${SYCL_DIR}")

    set(CMAKE_BUILD_RPATH "${SYCL_DIR}/lib")
endif()

include(VecScatter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CMAKE_CXX_FLAGS}\"")

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MPI_CXX_HEADER_DIR}
)

add_library(brickhelper stencils/brickcompare.cpp stencils/multiarray.cpp)

add_subdirectory(src)
add_subdirectory(single)
add_subdirectory(weak)
add_subdirectory(strong)

# Generating html doc from doxygen output
set(PROJECT_INC_DIRS "${PROJECT_SOURCE_DIR}/include")

find_package(Doxygen
        REQUIRED dot)

if(DOXYGEN_FOUND)
    configure_file(docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doxydoc
            Doxygen::doxygen ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
            )
endif()
