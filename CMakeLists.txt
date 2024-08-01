cmake_minimum_required(VERSION 3.10)
project(OCRProject C)

# Compiler flags
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -fopenmp")

# Debug and Release flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -DDEBUG -fsanitize=address")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

# Find packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL REQUIRED sdl)
pkg_check_modules(GTK REQUIRED gtk+-3.0)

# Include directories
include_directories(${SDL_INCLUDE_DIRS} ${GTK_INCLUDE_DIRS})

# Source files
set(SRC
    main.c
    source/process/process.c
    source/sdl/our_sdl.c
    source/segmentation/segmentation.c
    source/network/tools.c
    source/network/OCR.c
    source/network/XOR.c
    source/network/early_stop.c
    source/network/pool.c
    source/network/fc.c
    source/network/conv.c
    source/network/volume.c
    source/network/adam.c
    source/GUI/gui.c
)

# Executable
add_executable(${PROJECT_NAME} ${SRC})
add_executable(${PROJECT_NAME}_debug ${SRC})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${SDL_LIBRARIES} ${GTK_LIBRARIES} SDL_image m)
target_link_libraries(${PROJECT_NAME}_debug ${SDL_LIBRARIES} ${GTK_LIBRARIES} SDL_image m)

# Set the output directory for executables
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..
)
set_target_properties(${PROJECT_NAME}_debug PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..
)

# Create necessary directories
add_custom_target(create_dirs
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/source/Xor
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/source/OCR
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_SOURCE_DIR}/source/Xor/xordata.txt
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_SOURCE_DIR}/source/Xor/xorwb.txt
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_SOURCE_DIR}/source/OCR/ocrwb.txt
)

# Ensure directories are created before building
add_dependencies(${PROJECT_NAME} create_dirs)
add_dependencies(${PROJECT_NAME}_debug create_dirs)

# Set debug target properties
set_target_properties(${PROJECT_NAME}_debug PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS_DEBUG}" LINK_FLAGS "${CMAKE_C_FLAGS_DEBUG}")

# Clean up custom target
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/../OCRProject
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/../OCRProject_debug
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/main
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/main_debug
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/*.bmp
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/img/temp
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/source/Xor
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/source/OCR
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/*.tst
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/img/training/maj/*.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/img/training/min/*.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/*.bin
)
