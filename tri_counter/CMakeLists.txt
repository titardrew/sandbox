cmake_minimum_required(VERSION 3.16)

project(MakeTri)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENMP "Whether to use OpenMP." OFF)

# -fsanitize=address
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
if(USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()


add_executable(CountTri source/main.cc source/scalar.cc source/scalar_unrolled.cc source/sse_v0.cc source/sse_v1.cc)
if(USE_OPENMP)
    target_compile_definitions(CountTri PUBLIC USE_OPENMP)
endif()
