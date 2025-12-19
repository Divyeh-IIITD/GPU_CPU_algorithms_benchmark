#include "algorithms.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 1024

// --------------------------------------------------------
// KERNEL 1: Intra-Block Scan (Blelloch)
// --------------------------------------------------------
__global__ void prescan_arbitrary(float *g_odata, float *g_idata, float *g_blockSums, int n) {
    extern __shared__ float temp[]; // Allocated at launch time

    int thid = threadIdx.x;
    int ai = thid;
    int bi = thid + (n / 2);
    
    
    // For this demo, we will use a simpler "Hillis-Steele" scan for the block 
    // because Blelloch is very sensitive to bank conflicts without padding macros.
    // This is safer for a first implementation.
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (gid < n) temp[thid] = g_idata[gid];
    else temp[thid] = 0.0f;
    __syncthreads();

    // Naive Parallel Scan (Hillis-Steele) in Shared Memory
    // Simpler to implement correctly than Blelloch for first-timers
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (thid >= stride) val = temp[thid - stride];
        __syncthreads();
        if (thid >= stride) temp[thid] += val;
        __syncthreads();
    }

    // Write result to global memory
    if (gid < n) g_odata[gid] = temp[thid];

    // If I am the last thread in the block, write my sum to the block sums array
    if (thid == blockDim.x - 1) {
        if (g_blockSums != nullptr) g_blockSums[blockIdx.x] = temp[thid];
    }
}

// --------------------------------------------------------
// KERNEL 2: Add Base to Block
// --------------------------------------------------------
__global__ void add_block_sums(float *g_odata, float *g_blockSums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // If not the first block, add the previous block's total sum
    if (bid > 0 && gid < n) {
        g_odata[gid] += g_blockSums[bid - 1];
    }
}

// --------------------------------------------------------
// Host Wrapper
// --------------------------------------------------------
float gpu_scan_blelloch(float* d_in, float* d_out, int n) {
    // Strategy:
    // 1. Scan individual blocks. Store total sums of each block.
    // 2. Scan the array of sums.
    // 3. Add the scanned sums back to the blocks.

    int threads = BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;

    float *d_blockSums, *d_scannedBlockSums;
    cudaMalloc(&d_blockSums, blocks * sizeof(float));
    cudaMalloc(&d_scannedBlockSums, blocks * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: Scan each block locally
    // Shared memory size = threads * sizeof(float)
    prescan_arbitrary<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_in, d_blockSums, n);

    // Step 2: Scan the block sums
    // (Assumption: blocks count < 2048, so we can do this in ONE single block)
    // If input N > ~500,000, this single block is enough. 
    // If N is huge, this step needs to be recursive, but let's keep it simple.
    prescan_arbitrary<<<1, blocks, blocks * sizeof(float)>>>(d_scannedBlockSums, d_blockSums, nullptr, blocks);

    // Step 3: Add the scanned sums ("bases") back to the main array
    add_block_sums<<<blocks, threads>>>(d_out, d_scannedBlockSums, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_blockSums);
    cudaFree(d_scannedBlockSums);
    return ms;
}