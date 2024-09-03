#include <iostream>
#include <cassert>

#include <MatrixMul.cuh>

void fillMatrix(float* matrix, int width, int height, float value) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i * width + j] = value;
        }
    }
}

int main() {
    int heightA = 1000, widthA = 1000, heightB = 1000, widthB = 1000;

    float* hostA = (float*)malloc(sizeof(float) * heightA * widthA);
    float* hostB = (float*)malloc(sizeof(float) * heightB * widthB);
    float* hostResult = (float*)malloc(sizeof(float) * heightA * widthB);

    fillMatrix(hostA, widthA, heightA, 1);
    fillMatrix(hostB, widthB, heightB, 1);

    float* deviceA;
    float* deviceB;
    float* deviceResult;

    cudaMalloc(&deviceA, sizeof(float) * widthA * heightA);
    cudaMalloc(&deviceB, sizeof(float) * widthB * heightB);
    cudaMalloc(&deviceResult, sizeof(float) * widthB * heightA);

    cudaMemcpy(deviceA, hostA, sizeof(float) * widthA * heightA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float) * widthB * heightB, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 numBlocks((heightA + blockSize.x - 1) / blockSize.x, (widthB + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    MatrixMul<<<numBlocks, blockSize>>>(heightA, widthA, widthB, deviceA, deviceB, deviceResult);
    
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(hostResult, deviceResult, sizeof(float) * heightA * widthB, cudaMemcpyDeviceToHost);

    for (int i = 0; i < heightA; ++i) {
        for (int j = 0; j < widthB; ++j) {
            assert(hostResult[i * widthB + j] == 1000.0f);
        }
    }

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << elapsedTime << '\n';

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceResult);

    free(hostA);
    free(hostB);
    free(hostResult);

    return 0;
}

