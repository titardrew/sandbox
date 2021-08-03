add_definitions(
    -DTW_STATIC
    -DTW_NO_LIB_PRAGMA
    -DTW_NO_DIRECT3D
    -DGLEW_STATIC
    -D_CRT_SECURE_NO_WARNINGS
)

### OpenGL ###
find_package(OpenGL REQUIRED)

### GLFW ###
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw-3.1.2)

### GLEW ###
# find_package(GLEW REQUIRED)  @Todo(aty): does not work.

add_library(GLEW_1_13 STATIC
    glew-1.13.0/src/glew.c
)
target_include_directories(GLEW_1_13 PUBLIC
    glew-1.13.0/include/
)

if(TARGET glfw)
    message( WARNING "GLFW IS OKAY")
elseif()
    message( WARNING "GLFW IS NOT OKAY")
endif()

### GLM ###
add_subdirectory(glm-0.9.7.1)
