#include <cmath>

#include <CosineVector.cuh>
#include <ScalarMulRunner.cuh>

float CosineVector(int numElements, float* vector1, float* vector2, int blockSize) {
    float scalarMul = ScalarMulTwoReductions(numElements, vector1, vector2, blockSize);
    float vector1Module = sqrt(ScalarMulTwoReductions(numElements, vector1, vector1, blockSize));
    float vector2Module = sqrt(ScalarMulTwoReductions(numElements, vector2, vector2, blockSize));

    float result = scalarMul / (vector1Module * vector2Module);

    return result;
}

