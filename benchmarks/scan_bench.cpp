#include "algorithms.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <cmath>
#include <chrono>

int main() {
    // 1 Million elements
    const int N = 1 << 20; 
    std::cout << "Running Parallel Prefix Sum (Scan) Benchmark (N = " << N << ")...\n\n";

    std::vector<float> h_in(N, 1.0f); // All 1s. Result should be 1, 2, 3... N
    std::vector<float> h_out_cpu(N);
    std::vector<float> h_out_gpu(N);

    // 1. CPU Benchmark
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_scan_std(h_in.data(), h_out_cpu.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // 2. GPU Setup
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Run GPU Scan
    float gpu_ms = gpu_scan_blelloch(d_in, d_out, N);

    // 4. Verification
    cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify last element (should be N)
    bool correct = std::abs(h_out_gpu[N-1] - (float)N) < 1.0f;
    // Verify a random element in middle
    bool correct_mid = std::abs(h_out_gpu[N/2] - (float)(N/2 + 1)) < 1.0f;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Time: " << cpu_ms << " ms\n";
    std::cout << "GPU Time: " << gpu_ms << " ms\n";
    std::cout << "Speedup:  " << cpu_ms / gpu_ms << "x\n";
    std::cout << "Correctness: " << ((correct && correct_mid) ? "✅ PASS" : "❌ FAIL") << "\n";
    
    if (!correct) std::cout << "Expected " << N << " got " << h_out_gpu[N-1] << "\n";

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}