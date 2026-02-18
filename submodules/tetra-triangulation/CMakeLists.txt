cmake_minimum_required(VERSION 3.16.3)

# 设置C++标准支持
set(CMAKE_CXX_STANDARD 17) # xxadd 1023
set(CMAKE_CXX_STANDARD_REQUIRED ON) # xxadd 1023

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions") # xxadd 1203


set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc") # xxadd 1021

set(PYTHON_EXECUTABLE "/home/gb/anaconda3/envs/gaussian-avatars/bin/python") # xxadd 1021
find_package(Python REQUIRED Interpreter) # xxadd 1021

project(fast_traversal LANGUAGES CXX CUDA)
enable_language(CUDA)

include(FetchContent)
# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE "Release")

find_library(libcnpy cnpy ${CONDA_PREFIX}/lib)
include_directories(${CONDA_PREFIX}/include)

# TODO: fix this
find_package(Torch REQUIRED)

#  这个不加的话会有python_add_dic的错误 这个函数在development组件里
find_package (Python 3.10.9 EXACT COMPONENTS Interpreter Development REQUIRED) # xxadd 1128
# 1. 再tetra目录下手动下载了pybind
 add_subdirectory(pybind11) # xxadd 1022
 find_package(pybind11 REQUIRED) # xxadd 1022

# 2. 在conda环境里手动安装里的pybind
#find_package (Python 3.10.9 EXACT COMPONENTS Interpreter Development REQUIRED)
#set(PYBIND11_DIR "/home/gb/anaconda3/envs/gaussian-avatars/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
#find_package(pybind11 REQUIRED)

# 3. 远程下载
#include(FetchContent)
#FetchContent_Declare(
#    pybind11
#    GIT_REPOSITORY https://github.com/pybind/pybind11
#    GIT_TAG        v2.9.2
#)
#FetchContent_GetProperties(pybind11)
#if(NOT pybind11_POPULATED)
#    FetchContent_Populate(pybind11)
#    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
#endif()


find_path(GMP_INCLUDE_DIR NAMES gmp.h
        HINTS "/home/gb/anaconda3/envs/gaussian-avatars/include")
find_library(GMP_LIBRARY NAMES gmp
        HINTS "/home/gb/anaconda3/envs/gaussian-avatars/lib")
set(GMP_LIBRARIES "${GMP_LIBRARY}")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG GMP_LIBRARY GMP_INCLUDE_DIR)
message(STATUS "GMP_INCLUDE_DIR: ${GMP_INCLUDE_DIR}")
message(STATUS "GMP_LIBRARY: ${GMP_LIBRARY}")

if(GMP_FOUND)
    message(STATUS "GMP found: ${GMP_LIBRARY}")
    include_directories(${GMP_INCLUDE_DIR})
    link_libraries(${GMP_LIBRARY})
else()
    message(FATAL_ERROR "GMP not found")
endif()
include_directories(${GMP_INCLUDE_DIR})
link_directories(${GMP_LIBRARY})




find_path(MPFR_INCLUDE_DIR NAMES mpfr.h
        HINTS "/home/gb/anaconda3/envs/gaussian-avatars/include")
find_library(MPFR_LIBRARY NAMES mpfr
        HINTS "/home/gb/anaconda3/envs/gaussian-avatars/lib")
set(MPFR_LIBRARIES "${MPFR_LIBRARY}")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR DEFAULT_MSG MPFR_LIBRARY MPFR_INCLUDE_DIR)

if(MPFR_FOUND)
    message(STATUS "MPFR found: ${MPFR_LIBRARY}")
    include_directories(${MPFR_INCLUDE_DIR})
    link_libraries(${MPFR_LIBRARY})
else()
    message(FATAL_ERROR "MPFR not found")
endif()




set(CGAL_DIR "/home/gb/anaconda3/envs/gaussian-avatars/lib/cmake/CGAL") # xxadd 1021
find_package(CGAL REQUIRED)



set(TBB_DIR "/home/gb/anaconda3/envs/gaussian-avatars/lib/cmake/TBB") # xxadd 1021
find_package(TBB REQUIRED)
if (NOT TARGET TBB::TBB)
    find_library(TBB_LIB NAMES tbb HINTS ${TBB_DIR})
    find_path(TBB_INCLUDE_DIR tbb/tbb.h HINTS ${TBB_DIR}/../include)
    add_library(TBB::TBB UNKNOWN IMPORTED)
    set_target_properties(TBB::TBB PROPERTIES
        IMPORTED_LOCATION ${TBB_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIR}
    )
endif()
message(STATUS "TBB library: ${TBB_LIBRARIES}")




# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(BUILD_DIRECTORY "${CMAKE_BINARY_DIR}/build")
set(CUDA_GENERATED_OUTPUT_DIR "${BUILD_DIRECTORY}")


#------------------------------------------------------------------------------
# set environment
#------------------------------------------------------------------------------

# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})

# configure_file("${CMAKE_SOURCE_DIR}/path_util.py.in" "${CMAKE_SOURCE_DIR}/../examples/path_util.py")


#------------------------------------------------------------------------------
# main build
#------------------------------------------------------------------------------

# configure_file(${generated_files} file.ptx.h)
add_link_options(-flto=auto)  # disables some warning
pybind11_add_module(tetranerf_cpp_extension 
    src/triangulation.cpp
    src/py_binding.cpp)
target_link_libraries(tetranerf_cpp_extension PRIVATE TBB::TBB)
target_link_libraries(tetranerf_cpp_extension PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(tetranerf_cpp_extension PRIVATE CGAL::CGAL)
target_compile_features(tetranerf_cpp_extension PRIVATE cxx_std_17)
set_target_properties(tetranerf_cpp_extension
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tetranerf/utils/extension"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
# target_compile_definitions(tetranerf_cpp_extension PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})