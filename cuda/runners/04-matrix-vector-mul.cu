#include <cassert>
#include <iostream>

#include <MatrixVectorMul.cuh>

void fillMatrix(float* matrix, int width, int height, float value) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i * width + j] = value;
        }
    }
}

int main() {
    int width = 1000, height = 1000;

    float* hostMatrix = (float*)malloc(sizeof(float) * width * height);
    float* hostVector = (float*)malloc(sizeof(float) * width);
    float* hostResult = (float*)malloc(sizeof(float) * width);

    fillMatrix(hostMatrix, width, height, 2.0f);
    
    for (int i = 0; i < width; ++i) {
        hostVector[i] = 3.0f;
    }

    float* deviceMatrix;
    float* deviceVector;
    float* deviceResult;

    cudaMalloc(&deviceMatrix, sizeof(float) * width * height);
    cudaMalloc(&deviceVector, sizeof(float) * width);
    cudaMalloc(&deviceResult, sizeof(float) * width);

    cudaMemcpy(deviceMatrix, hostMatrix, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVector, hostVector, sizeof(float) * width, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 numBlocks((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    MatrixVectorMul<<<numBlocks, blockSize>>>(height, width, deviceMatrix, deviceVector, deviceResult);
    
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(hostResult, deviceResult, sizeof(float) * width, cudaMemcpyDeviceToHost);

    for (int i = 0; i < width; ++i) {
        assert(hostResult[i] == 6000.0f);
    }

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << elapsedTime << '\n';

    cudaFree(deviceMatrix);
    cudaFree(deviceVector);
    cudaFree(deviceResult);

    free(hostMatrix);
    free(hostVector);
    free(hostResult);

    return 0;

}

