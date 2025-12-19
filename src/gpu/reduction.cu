#include "algorithms.h"
#include <cuda_runtime.h>
#include <algorithm> // for std::min

// A special helper function that runs on the GPU (device)
// It shifts data between threads without using memory (very fast)
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// The main GPU Kernel
__global__ void reduce_v3_kernel(const float* g_idata, float* g_odata, int n) {
    // Shared memory to store partial sums for one block
    static __shared__ float shared[32]; 

    int tid = threadIdx.x;
    int lane = tid % 32; // My index within the warp (0-31)
    int wid = tid / 32;  // My warp ID

    // Calculate initial sum for this thread
    float sum = 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    // Loop over the array (Grid-Stride Loop)
    while (i < n) {
        sum += g_idata[i];
        i += gridSize;
    }

    // 1. Sum within my warp
    sum = warpReduceSum(sum);

    // 2. First thread of each warp writes to shared memory
    if (lane == 0) shared[wid] = sum;
    __syncthreads(); // Wait for all warps

    // 3. The first warp sums the results from the other warps
    sum = (tid < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0) sum = warpReduceSum(sum);

    // 4. The very first thread writes the block's total to global memory
    if (tid == 0) g_odata[blockIdx.x] = sum;
}

// The function called from the CPU
float gpu_reduce_v3_shuffle(const float* d_in, float* d_out, int n) {
    int threads = 256;
    // Calculate how many blocks we need, cap at 1024
    int blocks = std::min((n + threads - 1) / threads, 1024); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Pass 1: Reduce huge array -> 1024 partial sums
    reduce_v3_kernel<<<blocks, threads>>>(d_in, d_out, n);

    // Pass 2: Reduce 1024 partial sums -> 1 final sum
    // We launch 1 block with 1024 threads (or fewer if blocks is small)
    reduce_v3_kernel<<<1, 1024>>>(d_out, d_out, blocks);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds;
}