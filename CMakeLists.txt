cmake_minimum_required(VERSION 3.22)
project(dbscan LANGUAGES CXX CUDA)

#setting flags
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2")
# change to your architecture for best performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 -rdc=true -g -G")
find_package(CUDAToolkit REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(dbscan dbscan.cu)
target_link_libraries(dbscan ${OpenCV_LIBS})

set_target_properties(dbscan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
