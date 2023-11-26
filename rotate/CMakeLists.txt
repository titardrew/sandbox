cmake_minimum_required(VERSION 3.16)

project(Rotate)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENMP "Whether to use OpenMP." OFF)

# -fsanitize=address
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb -fno-omit-frame-pointer")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_executable(Rotate source/rotate.cc)
