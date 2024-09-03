#include <cassert>
#include <iostream>

#include <CosineVector.cuh>

int main() {
    int numElements = 1 << 20;
    int blockSize = 256;

    float* vector1 = (float*)malloc(sizeof(float) * numElements);
    float* vector2 = (float*)malloc(sizeof(float) * numElements);

    vector1[0] = 1.0f;
    vector2[0] = vector2[1] = 1.0f;

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float result = CosineVector(numElements, vector1, vector2, blockSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    assert(result - 0.707f < 0.0002f);

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << elapsedTime << '\n';

    free(vector1);
    free(vector2);

    return 0;
}

