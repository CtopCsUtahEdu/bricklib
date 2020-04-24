cmake_minimum_required(VERSION 3.13)
project(brick
        VERSION 0.1
        LANGUAGES CXX)

cmake_policy(SET CMP0077 NEW)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
include(CheckCXXCompilerFlag)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
endif ()

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCL 2.0)

option(USE_MEMFD "Using memfd instead of shm_open, supported on Linux >= 3.17 with \"CONFIG_MEMFD_CREATE\"" OFF)
if (USE_MEMFD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MEMFD")
endif ()

option(CUDA_AWARE "Use CUDA_AWARE communication" ON)
if (CUDA_AWARE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_AWARE")
endif ()

option(USE_TYPES "Use MPI_TYPES" OFF)
if (USE_TYPES)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TYPES")
endif ()

option(BARRIER_TIMESTEP "Use barrier to reduce timing variation" OFF)
if (BARRIER_TIMESTEP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBARRIER_TIMESTEP")
endif ()

option(BRICK_BUILD_TEST "Include various tests as build target" ON)

option(HAS_HMM "Use ATS/HMM for CUDA (not-used)" OFF)

# MacOS doesn't have librt
# shm_open is recommended to link with -lrt (man page)
find_library(LIBRT rt)
if (LIBRT)
    message(STATUS "LIBRT ${LIBRT}")
else ()
    message(STATUS "LIBRT not found, assuming its a Mac")
    set(LIBRT "")
endif ()

check_cxx_compiler_flag(-fsycl HAS_SYCL)

if (HAS_SYCL)
    get_filename_component(bin_dir "${CMAKE_CXX_COMPILER}" PATH)
    get_filename_component(SYCL_DIR "${bin_dir}" PATH)

    message(STATUS "Setting include/lib path according to compiler path: ${CMAKE_CXX_COMPILER}")
    message(STATUS "SYCL compiler install dir: ${SYCL_DIR}")
    set(CMAKE_BUILD_RPATH "${SYCL_DIR}/lib")
endif ()

set(VS_PREPROCESSOR cpp CACHE STRING "Preprocessor for vector scatter")

# BRICK_PROJECT_PATH is set when brick is used as a submodule
# parent project should take care of VecScatter initialization
if (NOT BRICK_PROJECT_PATH)
    set(BRICK_PROJECT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
    include(cmake/VecScatter.cmake)
endif ()

# Avoid using O2
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

check_cxx_compiler_flag(-march=native HAS_MARCH)
if (HAS_MARCH)
    message(STATUS "Using march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else ()
    # Work around Summit only have mcpu flag
    check_cxx_compiler_flag(-mcpu=native HAS_MCPU)
    if (HAS_MCPU)
        message(STATUS "Using mcpu=native")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    endif ()
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CMAKE_CXX_FLAGS}\"")

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MPI_CXX_HEADER_DIR}
)

add_subdirectory(src)
if (BRICK_BUILD_TEST)
    add_subdirectory(single)
    add_subdirectory(weak)
    add_subdirectory(strong)
endif ()

# Install headers
install(DIRECTORY include/ DESTINATION include/${CMAKE_PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h")

# Install codegen
install(DIRECTORY codegen/ DESTINATION lib/${CMAKE_PROJECT_NAME}
        PATTERN "__pycache__" EXCLUDE)

# Generating html doc from doxygen output
set(PROJECT_INC_DIRS "${PROJECT_SOURCE_DIR}/include")

find_package(Doxygen
        COMPONENTS dot)

if (DOXYGEN_FOUND)
    configure_file(docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doxydoc
            Doxygen::doxygen ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
            )
endif ()
