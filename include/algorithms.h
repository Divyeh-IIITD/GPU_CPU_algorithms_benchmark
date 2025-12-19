#pragma once
#include <vector>

// CPU Function
float cpu_reduce_omp(const std::vector<float>& data);

// GPU Wrapper Function
float gpu_reduce_v3_shuffle(const float* d_in, float* d_out, int n);