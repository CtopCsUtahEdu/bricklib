# Basic example using brick library as an external project
cmake_minimum_required(VERSION 3.13)
project(brick_example)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(brick REQUIRED)
include_directories("${BRICK_INCLUDE_DIR}")

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

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

VSTARGET(B7PT main.cpp main-out.cpp)
add_executable(brick-7pt ${VSTARGET_B7PT_OUTPUT})
target_link_libraries(brick-7pt brickhelper)
set_target_properties(brick-7pt
        PROPERTIES OUTPUT_NAME example)