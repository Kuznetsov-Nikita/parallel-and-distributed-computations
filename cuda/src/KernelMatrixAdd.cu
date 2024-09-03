#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, size_t pitch, float* A, float* B, float* result) {
    int indexRow = blockIdx.x * blockDim.x + threadIdx.x;
    int indexCol = blockIdx.y * blockDim.y + threadIdx.y;
    int strideRow = blockDim.x * gridDim.x;
    int strideCol = blockDim.y * gridDim.y;

    pitch *= sizeof(float);

    for (int i = indexRow; i < height; i += strideRow) {
        float* rowA = (float*)((char*)A + i * pitch);
        float* rowB = (float*)((char*)B + i * pitch);
        float* rowResult = (float*)((char*)result + i * pitch);

        for (int j = indexCol; j < width; j += strideCol) {
            rowResult[j] = rowA[j] + rowB[j];
        }
    }
}

