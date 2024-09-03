#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < height; i += stride) {
        float* rowMatrix = matrix + i * width;
        float sum = 0.0f;

        for (int j = 0; j < width; ++j) {
            sum += rowMatrix[j] * vector[j];
        }

        result[i] = sum;
    }
}

