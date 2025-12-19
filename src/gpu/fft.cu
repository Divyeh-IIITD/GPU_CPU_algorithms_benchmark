#include "algorithms.h"
#include <cuda_runtime.h>
#include <cufft.h> // NVIDIA's FFT Library header
#include <iostream>

float gpu_fft_library(const std::complex<float>* h_in, std::complex<float>* h_out, int n) {
    // Complex number size
    size_t bytes = n * sizeof(std::complex<float>);

    // 1. GPU Memory Allocation
    cufftComplex *d_data;
    cudaMalloc(&d_data, bytes);

    // Copy Input (Host -> Device)
    // We cast std::complex to cufftComplex (they are binary compatible)
    cudaMemcpy(d_data, h_in, bytes, cudaMemcpyHostToDevice);

    // 2. Create cuFFT Plan
    // This tells cuFFT "We want a 1D FFT of size N, Complex-to-Complex"
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);

    // 3. Execute
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // ExecC2C: Execute Complex to Complex
    // CUFFT_FORWARD: Transform from Time -> Frequency
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 4. Copy Output (Device -> Host)
    cudaMemcpy(h_out, d_data, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);

    return ms;
}