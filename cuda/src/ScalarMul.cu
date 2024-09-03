#include <ScalarMul.cuh>

/*
 * Calculates scalar multiplication for block
 */
__global__
void ScalarMulBlock(int numElements, float* vector1, float* vector2, float *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    for (int i = index; i < numElements; i += stride) {
        sum += vector1[i] * vector2[i];
    }

    atomicAdd(&result[blockIdx.x], sum);
}

