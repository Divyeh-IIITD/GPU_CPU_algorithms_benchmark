# GPU-Accelerated Algorithms Benchmark Suite

This project is a high-performance benchmarking suite comparing CPU (OpenMP) and GPU (CUDA) implementations of fundamental parallel algorithms. It demonstrates the impact of hardware-aware optimizations such as Shared Memory Tiling, Warp Shuffles, and Constant Memory caching on modern NVIDIA hardware.

## Benchmark Results

The following results were collected on an NVIDIA GPU vs. an Intel CPU (OpenMP enabled).

| Algorithm | Input Size | CPU Time (ms) | GPU Time (ms) | Speedup | Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fast Fourier Transform** | 1 Million (Complex) | 365.17 | 0.11 | **3364.3x** | Library Optimization (cuFFT) |
| **Matrix Multiplication** | 1024x1024 | 860.33 | 2.05 | **419.6x** | 1.05 TFLOPS |
| **2D Convolution** | 3840x2160 (4K) | 112.01 | 0.46 | **243.6x** | Spatial Locality |
| **Parallel Reduction** | 67 Million | 34.56 | 1.26 | **27.5x** | 214 GB/s Bandwidth |
| **Prefix Sum (Scan)** | 1 Million | 1.28 | 0.70 | **1.82x** | Latency Hiding |

### Analysis

1.  **Fast Fourier Transform (FFT):** Achieved a massive **3364x speedup** by leveraging the NVIDIA **cuFFT** library ($O(N \log N)$), demonstrating the advantage of using vendor-optimized libraries over raw C++ recursion.
2.  **Matrix Multiplication (Compute Bound):** Achieved over 1 TeraFLOP of throughput by implementing Shared Memory Tiling (32x32 blocks) to minimize Global Memory access latency.
3.  **2D Convolution (Memory Bound):** Achieved a 243x speedup by utilizing Constant Memory (`__constant__`) for the filter mask, enabling high-speed broadcast of read-only data to all threads.
4.  **Parallel Reduction (Bandwidth Bound):** Saturated the GPU memory bandwidth (~214 GB/s) using Warp Shuffles (`__shfl_down_sync`) to perform reductions at the register level.
5.  **Prefix Sum (Dependency Bound):** Implemented a Blelloch/Hillis-Steele hybrid approach to resolve strict data dependencies in parallel.

## Technical Implementations

### CPU Baselines
* **OpenMP:** utilized for multi-threaded parallelization (`#pragma omp parallel for`) and loop collapsing.
* **STL/Recursion:** Used `std::partial_sum` for scan and Cooley-Tukey recursion for FFT.

### GPU Optimizations (CUDA)
* **Tiling:** Breaks large matrices into small blocks that fit into L1/Shared Memory.
* **Warp Shuffles:** Allows threads within a warp (32 threads) to exchange data directly without using Shared Memory.
* **Constant Memory:** Uses dedicated GPU cache for small, read-only data structures (e.g., convolution masks).
* **Vendor Libraries:** Integrated cuFFT for spectral analysis.

## Build Instructions

### Prerequisites
* Windows 10/11
* Visual Studio 2022 (Desktop development with C++ workload)
* NVIDIA CUDA Toolkit (v12.x or v13.x)
* CMake (3.18 or newer)
* Git

### Compilation
1.  Open the "x64 Native Tools Command Prompt for VS 2022".
2.  Navigate to the project directory.
3.  Run the following commands:

```cmd
mkdir build
cd build
cmake ..
cmake --build . --config Release
