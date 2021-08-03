cmake_minimum_required(VERSION 3.18.1)
project(CloudChamber)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(OPTIMIZE "Add optimization flags." OFF)
option(DBG_SYMBOLS "Compile with debugging symbols." ON)
option(ASAN "Build with address sanitizer enabled." ON)

if(${OPTIMIZE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
endif()

if(${DBG_SYMBOLS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -ggdb")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -g -ggdb")
endif()

if(${ASAN})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
endif()

# Adding module path for find_package routines.
#include(FindPkgConfig)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# FFMPEG
# @Todo(atyt): uncomment when video component is done.
#find_package(FFMPEG REQUIRED)
#message("")
#if (${FFMPEG_FOUND})
#    message( STATUS "FFMPEG_INCLUDE_DIRS:" "${FFMPEG_INCLUDE_DIRS}")
#    message( STATUS "FFMPEG_LIBRARIES: " "${FFMPEG_LIBRARIES}")
#else()
#    message( FATAL_ERROR "FFMPEG NOT FOUND" )
#endif()

# SDL2
#find_package(SDL2 REQUIRED)
#message("")
#if (${SDL2_FOUND})
#    message( STATUS "SDL2_INCLUDE_DIR:" ${SDL2_INCLUDE_DIR})
#    message( STATUS "SDL2_LIBRARY: " ${SDL2_LIBRARY})
#else()
#    message( FATAL_ERROR "SDL2 NOT FOUND" )
#endif()

add_subdirectory(external)

# ImGUI
# Using version with the docking feature.
# set(IMGUI_SOURCES
#    external/imgui/imgui.cpp
#    external/imgui/imgui_demo.cpp
#    external/imgui/imgui_draw.cpp
#    external/imgui/imgui_impl_opengl3.cpp
#    external/imgui/imgui_impl_sdl.cpp
#    external/imgui/imgui_tables.cpp
#    external/imgui/imgui_widgets.cpp
#)
#add_library(imgui ${IMGUI_SOURCES})
#target_include_directories(imgui PUBLIC ${GLAD_INCLUDE_DIR} ${SDL2_INCLUDE_DIR})
#target_link_libraries(imgui PUBLIC glad ${SDL2_LIBRARY})
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

# Main lib.
add_library(cloud_chamber_lib STATIC
    source/artifacts.cc
    source/shader.cc
)
set(CLOUD_CHAMBER_INCLUDE_DIR source) # external/imgui)
target_include_directories(cloud_chamber_lib PUBLIC
    ${OPENGL_INCLUDE_DIR}
    external
    external/glfw-3.1.2/include
    external/eigen
)
message( WARNING "GLEW_LIBS: " ${GLEW_LIBRARY})

target_link_libraries(cloud_chamber_lib PUBLIC
    ${OPENGL_gl_LIBRARY}
    ${GLFW_LIBRARIES}
    GLEW_1_13
    glm
    glfw
#    imgui
)

# Get the include directories for the target.
get_target_property(CLOUD_LIBS cloud_chamber_lib INCLUDE_DIRECTORIES)

# Construct the compiler string for the include directories.
foreach(dir ${CLOUD_LIBS})
    string(APPEND INCLUDE_COMPILER_STRING "/I${dir} ")
endforeach()
message( WARNING "CLOUD_LIBS:" ${INCLUDE_COMPILER_STRING})

# CloudChamber main binary.
add_executable(CloudChamber source/cloud_chamber.cc)
# @Refactor(atyt): do we need this?
target_include_directories(CloudChamber PUBLIC ${CLOUD_CHAMBER_INCLUDE_DIR})
target_link_libraries(CloudChamber PUBLIC cloud_chamber_lib)


# install sucks
add_custom_command(
    TARGET CloudChamber POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/source/shaders/
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/
)
