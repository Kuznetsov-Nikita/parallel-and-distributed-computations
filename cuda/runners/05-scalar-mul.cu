#include <cassert>
#include <iostream>

#include <ScalarMulRunner.cuh>

void fillVector(float* vector, int numElements, float value) {
    for (int i = 0; i < numElements; ++i) {
        vector[i] = value;
    }
}

int main() {
    int numElements = 1 << 20;
    int blockSize = 1024;

    float* vector1 = (float*)malloc(sizeof(float) * numElements);
    float* vector2 = (float*)malloc(sizeof(float) * numElements);

    fillVector(vector1, numElements, 1.0f);
    fillVector(vector2, numElements, 2.0f);

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float result = ScalarMulSumPlusReduction(numElements, vector1, vector2, blockSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    assert(result == (float)(1 << 21));

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << elapsedTime << '\n';

    cudaEventRecord(start);

    result = ScalarMulTwoReductions(numElements, vector1, vector2, blockSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    assert(result == (float)(1 << 21));

    elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << elapsedTime << '\n';

    free(vector1);
    free(vector2);

    return 0;
}

