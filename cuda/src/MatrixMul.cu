#include <MatrixMul.cuh>

#define DIM 32

__global__
void MatrixMul(int heightA, int widthA, int widthB, float *matrixA, float *matrixB, float *matrixResult) {
    int heightB = widthA;
    int heightResult = heightA;
    int widthResult = widthB;
    
    int col = blockIdx.x * DIM + threadIdx.x;
    int row = blockIdx.y * DIM + threadIdx.y;

    __shared__ float sharedA[DIM][DIM];
    __shared__ float sharedB[DIM][DIM];

    float sum = 0.0f;

    for (int i = 0; i < (DIM + widthA - 1) / DIM; ++i) {
        if (i * DIM + threadIdx.x < widthA && row < heightA) {
            sharedA[threadIdx.y][threadIdx.x] = matrixA[row * widthA + i * DIM + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (i * DIM + threadIdx.y < heightB && col < widthB) {
            sharedB[threadIdx.y][threadIdx.x] = matrixB[(i * DIM + threadIdx.y) * widthB + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < DIM; ++j) {
            sum += sharedA[threadIdx.y][j] * sharedB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < heightResult && col < widthResult) {
        matrixResult[((blockIdx.y * blockDim.y + threadIdx.y) * widthResult) + (blockIdx.x * blockDim.x) + threadIdx.x] = sum;
    }
}

