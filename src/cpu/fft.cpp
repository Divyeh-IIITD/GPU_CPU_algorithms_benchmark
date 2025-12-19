#include "algorithms.h"
#include <cmath>
#include <vector>
#include <iostream>

const float PI = 3.14159265358979f;

// Helper: Recursive FFT
void fft_recursive(std::vector<std::complex<float>>& a) {
    int n = a.size();
    if (n <= 1) return;

    // Split into even and odd
    std::vector<std::complex<float>> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        even[i] = a[i * 2];
        odd[i] = a[i * 2 + 1];
    }

    // Recurse
    fft_recursive(even);
    fft_recursive(odd);

    // Combine (Butterfly)
    for (int k = 0; k < n / 2; ++k) {
        std::complex<float> t = std::polar(1.0f, -2 * PI * k / n) * odd[k];
        a[k] = even[k] + t;
        a[k + n / 2] = even[k] - t;
    }
}

void cpu_fft(const std::complex<float>* input, std::complex<float>* output, int n) {
    // Copy input to a vector for processing
    std::vector<std::complex<float>> data(input, input + n);
    
    // Run Recursive FFT
    fft_recursive(data);

    // Copy back to output
    for (int i = 0; i < n; ++i) {
        output[i] = data[i];
    }
}