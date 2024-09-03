#include <CommonKernels.cuh>
#include <ScalarMul.cuh>
#include <ScalarMulRunner.cuh>

__global__ void Reduce(float* inData, float* outData) {
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = inData[index];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
      outData[blockIdx.x] = sharedData[0];
    }
}

float ScalarMulTwoReductions(int numElements, float* vector1, float* vector2, int blockSize) {
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    int numBlocksReduce = (numBlocks + blockSize - 1) / blockSize;

    float* deviceVector1;
    float* deviceVector2;
    float* deviceMulResult;
    float* deviceReduceResult;
    float* deviceResult;

    cudaMalloc(&deviceVector1, sizeof(float) * numElements);
    cudaMalloc(&deviceVector2, sizeof(float) * numElements);
    cudaMalloc(&deviceMulResult, sizeof(float) * numElements);
    cudaMalloc(&deviceReduceResult, sizeof(float) * numBlocks);
    cudaMalloc(&deviceResult, sizeof(float) * numBlocksReduce);

    cudaMemcpy(deviceVector1, vector1, sizeof(float) * numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVector2, vector2, sizeof(float) * numElements, cudaMemcpyHostToDevice);

    KernelMul<<<numBlocks, blockSize>>>(numElements, deviceVector1, deviceVector2, deviceMulResult);
    cudaDeviceSynchronize();

    Reduce<<<numBlocks, blockSize, sizeof(float) * blockSize>>>(deviceMulResult, deviceReduceResult);
    cudaDeviceSynchronize();

    Reduce<<<numBlocksReduce, blockSize, sizeof(float) * blockSize>>>(deviceReduceResult, deviceResult);

    float* result = (float*)malloc(sizeof(float) * numBlocksReduce);
    cudaMemcpy(result, deviceResult, sizeof(float) * numBlocksReduce, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < numBlocksReduce; ++i) {
        sum += result[i];
    }

    cudaFree(deviceVector1);
    cudaFree(deviceVector2);
    cudaFree(deviceMulResult);
    cudaFree(deviceReduceResult);
    cudaFree(deviceResult);

    free(result);

    return sum;
}

float ScalarMulSumPlusReduction(int numElements, float* vector1, float* vector2, int blockSize) {
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    int numBlocksReduce = (numBlocks + blockSize - 1) / blockSize;

    float* deviceVector1;
    float* deviceVector2;
    float* deviceMulResult;
    float* deviceResult;

    cudaMalloc(&deviceVector1, sizeof(float) * numElements);
    cudaMalloc(&deviceVector2, sizeof(float) * numElements);
    cudaMalloc(&deviceMulResult, sizeof(float) * numBlocks);
    cudaMalloc(&deviceResult, sizeof(float) * numBlocksReduce);

    cudaMemcpy(deviceVector1, vector1, sizeof(float) * numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVector2, vector2, sizeof(float) * numElements, cudaMemcpyHostToDevice);

    ScalarMulBlock<<<numBlocks, blockSize>>>(numElements, deviceVector1, deviceVector2, deviceMulResult);
    cudaDeviceSynchronize();

    Reduce<<<numBlocksReduce, blockSize, sizeof(float) * blockSize>>>(deviceMulResult, deviceResult);

    float* result = (float*)malloc(sizeof(float) * numBlocksReduce);
    cudaMemcpy(result, deviceResult, sizeof(float) * numBlocksReduce, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < numBlocksReduce; ++i) {
        sum += result[i];
    }

    cudaFree(deviceVector1);
    cudaFree(deviceVector2);
    cudaFree(deviceMulResult);
    cudaFree(deviceResult);

    free(result);

    return sum;
}

