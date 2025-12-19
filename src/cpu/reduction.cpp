#include "algorithms.h"
#include <omp.h>

float cpu_reduce_omp(const std::vector<float>& data) {
    float sum = 0.0f;
    // Use OpenMP to use all CPU cores
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    return sum;
}