#include <cassert>
#include <iostream>

#include "KernelMul.cuh"

int main() {
    int numElements = 1 << 28;
    size_t size = numElements * sizeof(float);

    float* hostX = (float*)malloc(size);
    float* hostY = (float*)malloc(size);
    float* hostResult = (float*)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        hostX[i] = 2.0f;
        hostY[i] = 3.0f;
    }

    float* deviceX;
    float* deviceY;
    float* deviceResult;

    cudaMalloc(&deviceX, size);
    cudaMalloc(&deviceY, size);
    cudaMalloc(&deviceResult, size);

    cudaMemcpy(deviceX, hostX, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceY, hostY, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    KernelMul<<<numBlocks, blockSize>>>(numElements, deviceX, deviceY, deviceResult);
    
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(hostResult, deviceResult, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElements; ++i) {
        assert(hostResult[i] == 6.0f);
    }

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << elapsedTime << '\n';

    cudaFree(deviceX);
    cudaFree(deviceY);
    cudaFree(deviceResult);

    free(hostX);
    free(hostY);
    free(hostResult);

    return 0;
}
