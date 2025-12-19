#include "algorithms.h"
#include <cuda_runtime.h>
#include <cstdio>

#define TILE_W 16
#define MASK_W 5
#define MASK_R (MASK_W / 2)

// SPECIAL MEMORY: "Constant Memory"
// This lives on the GPU but is cached for super-fast broadcast to all threads.
__constant__ float c_mask[MASK_W * MASK_W];

__global__ void convolution_tiled(const float* input, float* output, int width, int height) {
    // Shared memory needs to be bigger than the block to hold the "halo" (neighbors)
    __shared__ float tile[TILE_W + MASK_W - 1][TILE_W + MASK_W - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output coordinates
    int col_o = blockIdx.x * TILE_W + tx;
    int row_o = blockIdx.y * TILE_W + ty;

    // Input coordinates (shifted by radius to center the tile)
    int row_i = row_o - MASK_R;
    int col_i = col_o - MASK_R;

    // 1. Load Data into Shared Memory (Including Halos)
    // We break the tile loading into patches
    // This part is complex, so we use a simpler approach for the demo:
    // Every thread loads its corresponding pixel. 
    // Note: For a "perfect" tiled implementation, we need to load extra boundary pixels.
    // To keep this readable, we will check bounds inside the math loop (Global Memory fallback).
    // But we STILL use Constant Memory for the mask, which gives the biggest speedup.

    if (row_o < height && col_o < width) {
        float sum = 0.0f;
        
        // Loop over the mask (stored in Constant Memory)
        for (int ky = -MASK_R; ky <= MASK_R; ++ky) {
            for (int kx = -MASK_R; kx <= MASK_R; ++kx) {
                int dy = row_o + ky;
                int dx = col_o + kx;

                if (dy >= 0 && dy < height && dx >= 0 && dx < width) {
                    // Read Image from Global, Mask from Constant
                    sum += input[dy * width + dx] * c_mask[(ky + MASK_R) * MASK_W + (kx + MASK_R)];
                }
            }
        }
        output[row_o * width + col_o] = sum;
    }
}

float gpu_convolution(const float* d_input, const float* d_mask, float* d_output, int width, int height, int mask_width) {
    // 1. Copy the mask to the special Constant Memory symbol
    cudaMemcpyToSymbol(c_mask, d_mask, mask_width * mask_width * sizeof(float));

    dim3 block(TILE_W, TILE_W);
    dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_W - 1) / TILE_W);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution_tiled<<<grid, block>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}