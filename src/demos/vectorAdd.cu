#include<cuda.h>
#include<cuda_runtime_api.h>
#include<vector>
#include"errorHandle.h"
#include"vecUtils.h"

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void VecSet(float* v, float value)
{
    int i = threadIdx.x;
    v[i] = value;
}

__global__ void cuda_print(float* v)
{
    int i = threadIdx.x;
    printf("%d ", i);
}

int main()
{
    // kernel invocation with N threads
    constexpr int N = 10;
    float *A_dev, *B_dev, *C_dev;
    HANDLE_ERROR(cudaMalloc((void**)&A_dev, sizeof(float)*N));
    HANDLE_ERROR(cudaMalloc((void**)&B_dev, sizeof(float)*N));
    HANDLE_ERROR(cudaMalloc((void**)&C_dev, sizeof(float)*N));

    VecSet<<<1, N>>>(A_dev, 1);
    VecSet<<<1, N>>>(B_dev, 2);
    VecAdd<<<1, N>>>(A_dev, B_dev, C_dev);

    const auto A = ToVector(A_dev, N);
    const auto B = ToVector(B_dev, N);
    const auto C = ToVector(C_dev, N);
                             
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    print_array(C, N);
}
