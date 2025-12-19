#include "algorithms.h"
#include <numeric>

void cpu_scan_std(const float* input, float* output, int n) {
    // std::partial_sum is the C++ equivalent of Prefix Sum
    std::partial_sum(input, input + n, output);
}