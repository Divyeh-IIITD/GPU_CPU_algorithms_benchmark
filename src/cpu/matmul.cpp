#include "algorithms.h"
#include <omp.h>

void cpu_matmul_omp(const float* A, const float* B, float* C, int N) {
    // Basic parallel 3-loop multiplication
    // "i" is the row of A, "j" is the col of B, "k" is the dot product
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}