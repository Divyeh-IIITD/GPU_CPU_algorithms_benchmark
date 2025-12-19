cmake_minimum_required(VERSION 3.18)
project(GpuAlgoBenchmarks LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Windows/MSVC specific flags
# /O2 = Maximize Speed, /openmp = Enable OpenMP
set(CMAKE_CXX_FLAGS "/O2 /openmp")
set(CMAKE_CUDA_FLAGS "-O3 --use_fast_math")

include_directories(include)

file(GLOB CPU_SRCS "src/cpu/*.cpp")
file(GLOB GPU_SRCS "src/gpu/*.cu")

add_library(algo_lib ${CPU_SRCS} ${GPU_SRCS})

# Link OpenMP (Windows specific)
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(algo_lib PRIVATE OpenMP::OpenMP_CXX)

add_executable(reduction_bench benchmarks/reduction_bench.cpp)
target_link_libraries(reduction_bench algo_lib)

add_executable(matmul_bench benchmarks/matmul_bench.cpp)
target_link_libraries(matmul_bench algo_lib)

add_executable(scan_bench benchmarks/scan_bench.cpp)
target_link_libraries(scan_bench algo_lib)

add_executable(convolution_bench benchmarks/convolution_bench.cpp)
target_link_libraries(convolution_bench algo_lib)

