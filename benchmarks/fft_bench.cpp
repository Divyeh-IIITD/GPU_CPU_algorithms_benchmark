#include "algorithms.h"
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <chrono>

int main() {
    // Size must be power of 2 for standard Cooley-Tukey
    const int N = 1 << 20; // 1 Million points (2^20)
    std::cout << "Running FFT Benchmark (N = " << N << ")...\n\n";

    std::vector<std::complex<float>> h_in(N);
    std::vector<std::complex<float>> h_out_cpu(N);
    std::vector<std::complex<float>> h_out_gpu(N);

    // Initialize with a simple sine wave: sin(x) + sin(2x)
    for (int i = 0; i < N; ++i) {
        float t = (float)i / N;
        h_in[i] = std::complex<float>(sin(2 * 3.14159f * t) + 0.5f * sin(4 * 3.14159f * t), 0);
    }

    // 1. CPU Benchmark
    std::cout << "Running CPU FFT (Recursive)...\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_fft(h_in.data(), h_out_cpu.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // 2. GPU Benchmark
    std::cout << "Running GPU cuFFT...\n";
    float gpu_ms = gpu_fft_library(h_in.data(), h_out_gpu.data(), N);

    // 3. Verification
    std::cout << "\n--- Verification (First 5 Frequencies) ---\n";
    bool pass = true;
    for (int i = 0; i < 5; ++i) {
        float cpu_mag = std::abs(h_out_cpu[i]);
        float gpu_mag = std::abs(h_out_gpu[i]);
        float diff = std::abs(cpu_mag - gpu_mag);
        
        std::cout << "Idx " << i << " | CPU: " << cpu_mag << " | GPU: " << gpu_mag << " | Diff: " << diff << "\n";

        // Allow larger error for large magnitudes (N is 1 million!)
        if (diff > 10.0f) { 
            pass = false;
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Time: " << cpu_ms << " ms\n";
    std::cout << "GPU Time: " << gpu_ms << " ms\n";
    std::cout << "Speedup:  " << cpu_ms / gpu_ms << "x\n";
    std::cout << "Correctness: " << (pass ? "✅ PASS" : "❌ FAIL") << "\n";

    return 0;
}