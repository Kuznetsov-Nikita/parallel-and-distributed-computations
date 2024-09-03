#include <cassert>
#include <iostream>

#include <KernelMatrixAdd.cuh>

void fillMatrix(float* matrix, int width, int height, float value) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i * width + j] = value;
        }
    }
}

int main() {
    int width = 1000, height = 1000;

    float* hostA = (float*)malloc(sizeof(float) * width * height);
    float* hostB = (float*)malloc(sizeof(float) * width * height);
    float* hostResult = (float*)malloc(sizeof(float) * width * height);

    fillMatrix(hostA, width, height, 1.0f);
    fillMatrix(hostB, width, height, 2.0f);

    float* deviceA;
    float* deviceB;
    float* deviceResult;

    size_t pitch;

    cudaMallocPitch(&deviceA, &pitch, sizeof(float) * width, height);
    cudaMallocPitch(&deviceB, &pitch, sizeof(float) * width, height);
    cudaMallocPitch(&deviceResult, &pitch, sizeof(float) * width, height);

    cudaMemcpy2D(deviceA, pitch, hostA, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(deviceB, pitch, hostB, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 numBlocks((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    KernelMatrixAdd<<<numBlocks, blockSize>>>(height, width, pitch, deviceA, deviceB, deviceResult);
    
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy2D(hostResult, sizeof(float) * width, deviceResult, pitch, sizeof(float) * width, height, cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            assert(hostResult[i * width + j] == 3.0f);
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

