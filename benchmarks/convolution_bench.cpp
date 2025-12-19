#include "algorithms.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <cmath>
#include <chrono>

int main() {
    // 4K Image size
    const int W = 3840;
    const int H = 2160;
    const int MASK_WIDTH = 5; // 5x5 Blur Filter
    
    std::cout << "Running 2D Convolution Benchmark (4K Image: " << W << "x" << H << ")...\n\n";

    size_t bytes_img = W * H * sizeof(float);
    size_t bytes_mask = MASK_WIDTH * MASK_WIDTH * sizeof(float);

    // Host Data
    std::vector<float> h_img(W * H, 1.0f);
    std::vector<float> h_mask(MASK_WIDTH * MASK_WIDTH, 0.04f); // Simple box blur (1/25)
    std::vector<float> h_out_cpu(W * H);
    std::vector<float> h_out_gpu(W * H);

    // 1. CPU Benchmark
    std::cout << "Running CPU OpenMP...\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_convolution(h_img.data(), h_mask.data(), h_out_cpu.data(), W, H, MASK_WIDTH);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // 2. GPU Setup
    float *d_img, *d_mask, *d_out;
    cudaMalloc(&d_img, bytes_img);
    cudaMalloc(&d_mask, bytes_mask); // Note: We still malloc this to pass it to the helper
    cudaMalloc(&d_out, bytes_img);

    cudaMemcpy(d_img, h_img.data(), bytes_img, cudaMemcpyHostToDevice);
    // Note: We copy mask to global here, but inside the function we copy it to CONSTANT
    cudaMemcpy(d_mask, h_mask.data(), bytes_mask, cudaMemcpyHostToDevice); 

    // 3. Run GPU
    float gpu_ms = gpu_convolution(d_img, d_mask, d_out, W, H, MASK_WIDTH);

    // 4. Verify Center Pixel (Should be sum of 25 * 0.04 * 1.0 = 1.0)
    cudaMemcpy(h_out_gpu.data(), d_out, bytes_img, cudaMemcpyDeviceToHost);
    int center_idx = (H/2) * W + (W/2);
    bool correct = std::abs(h_out_gpu[center_idx] - 1.0f) < 0.01f;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Time: " << cpu_ms << " ms\n";
    std::cout << "GPU Time: " << gpu_ms << " ms\n";
    std::cout << "Speedup:  " << cpu_ms / gpu_ms << "x\n";
    std::cout << "Correctness: " << (correct ? "✅ PASS" : "❌ FAIL") << "\n";

    cudaFree(d_img); cudaFree(d_mask); cudaFree(d_out);
    return 0;
}