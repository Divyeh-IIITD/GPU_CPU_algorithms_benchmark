#include "algorithms.h"
#include <cuda_runtime.h>
#include <cstdio>

#define TILE_WIDTH 32

// --------------------------------------------------------
// KERNEL 1: Naive (Global Memory Heavy)
// --------------------------------------------------------
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            // Uncoalesced access on B if not careful, but mainly just too many reads
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// --------------------------------------------------------
// KERNEL 2: Tiled (Shared Memory Optimized)
// --------------------------------------------------------
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    // Shared memory for the sub-matrices (tiles)
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;

    // Loop over the tiles required to compute the dot product
    for (int p = 0; p < (N - 1) / TILE_WIDTH + 1; ++p) {
        
        // 1. Load data into shared memory
        // Check bounds to avoid crashing on edges
        if (Row < N && (p * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + p * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (p * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        // 2. Wait for all threads to load their piece
        __syncthreads();

        // 3. Compute partial dot product from shared memory
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }

        // 4. Wait before loading the next tile
        __syncthreads();
    }

    if (Row < N && Col < N) {
        C[Row * N + Col] = sum;
    }
}

// --------------------------------------------------------
// Host Wrappers
// --------------------------------------------------------
float run_kernel(void (*kernel)(const float*, const float*, float*, int), 
                 const float* d_A, const float* d_B, float* d_C, int N, 
                 dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

float gpu_matmul_naive(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    return run_kernel(matmul_naive_kernel, d_A, d_B, d_C, N, grid, block);
}

float gpu_matmul_tiled(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    return run_kernel(matmul_tiled_kernel, d_A, d_B, d_C, N, grid, block);
}