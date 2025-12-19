#include "algorithms.h"
#include <omp.h>

void cpu_convolution(const float* input, const float* mask, float* output, int width, int height, int mask_width) {
    int r = mask_width / 2; // Radius of the mask

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            
            // Apply the mask
            for (int ky = -r; ky <= r; ++ky) {
                for (int kx = -r; kx <= r; ++kx) {
                    int dy = y + ky;
                    int dx = x + kx;

                    // Boundary check (Zero padding)
                    if (dy >= 0 && dy < height && dx >= 0 && dx < width) {
                        sum += input[dy * width + dx] * mask[(ky + r) * mask_width + (kx + r)];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}