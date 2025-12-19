#pragma once
#include <vector>

// CPU Function
float cpu_reduce_omp(const std::vector<float>& data);

// GPU Wrapper Function
float gpu_reduce_v3_shuffle(const float* d_in, float* d_out, int n);

// --- Matrix Multiplication ---
// CPU Baseline
void cpu_matmul_omp(const float* A, const float* B, float* C, int N);

// GPU Implementations
// Returns execution time in ms
float gpu_matmul_naive(const float* d_A, const float* d_B, float* d_C, int N);
float gpu_matmul_tiled(const float* d_A, const float* d_B, float* d_C, int N);

// --- Prefix Sum (Scan) ---
// CPU Baseline
void cpu_scan_std(const float* input, float* output, int n);

// GPU Implementation (Blelloch Work-Efficient)
float gpu_scan_blelloch(float* d_in, float* d_out, int n);