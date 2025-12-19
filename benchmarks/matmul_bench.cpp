#include "algorithms.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <cmath>
#include <chrono>

int main() {
    // 1024x1024 matrix = 1 million elements per matrix
    // This is large enough to show GPU benefits
    const int N = 1024; 
    size_t bytes = N * N * sizeof(float);

    std::cout << "Running Matrix Multiplication Benchmark (N = " << N << "x" << N << ")...\n\n";

    // Host Data
    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 2.0f);
    std::vector<float> h_C_cpu(N * N);
    std::vector<float> h_C_gpu(N * N);

    // 1. CPU Benchmark
    std::cout << "Running CPU OpenMP...\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_matmul_omp(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // 2. GPU Setup
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // 3. GPU Naive
    float naive_ms = gpu_matmul_naive(d_A, d_B, d_C, N);
    
    // 4. GPU Tiled
    float tiled_ms = gpu_matmul_tiled(d_A, d_B, d_C, N);

    // Verify Result (Check first element)
    cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    bool correct = (std::abs(h_C_gpu[0] - (N * 2.0f)) < 0.1f);

    // 5. Results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Time:      " << cpu_ms << " ms\n";
    std::cout << "GPU Naive:     " << naive_ms << " ms\n";
    std::cout << "GPU Tiled:     " << tiled_ms << " ms\n";
    std::cout << "Tiled Speedup: " << naive_ms / tiled_ms << "x over Naive\n";
    std::cout << "Correctness:   " << (correct ? "✅ PASS" : "❌ FAIL") << "\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}