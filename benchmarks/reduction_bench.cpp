#include "algorithms.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

int main() {
    // 1. Setup: Create 67 million elements (approx 256MB)
    const int N = 1 << 26; 
    std::cout << "Running Benchmark with N = " << N << "...\n";

    std::vector<float> h_data(N, 1.0f); // Fill array with 1.0s

    // 2. Run CPU Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    float cpu_res = cpu_reduce_omp(h_data);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "CPU Time: " << cpu_ms << " ms\n";

    // 3. GPU Memory Setup
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, 1024 * sizeof(float));

    // Move data to GPU
    cudaMemcpy(d_in, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Run GPU Benchmark
    // Run once to warm up the GPU
    gpu_reduce_v3_shuffle(d_in, d_out, N); 
    // Run for real
    float gpu_ms = gpu_reduce_v3_shuffle(d_in, d_out, N);

    // Get result back
    float gpu_res;
    cudaMemcpy(&gpu_res, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "GPU Time: " << gpu_ms << " ms\n";
    std::cout << "Speedup:  " << std::fixed << std::setprecision(2) << cpu_ms / gpu_ms << "x\n";

    if (abs(cpu_res - gpu_res) < 1.0f) std::cout << "✅ Result Correct\n";
    else std::cout << "❌ Result Mismatch (" << cpu_res << " vs " << gpu_res << ")\n";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}