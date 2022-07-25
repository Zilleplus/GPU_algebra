#include<cuda.h>
#include<cuda_runtime_api.h>
#include<vector>
#include"errorHandle.h"
#include"vecUtils.h"

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

// use N blocks -> N kernels
__global__ void VecAdd(float* A, float* B, float* C, int size)
{
    // int i = threadIdx.x;
    int i = blockIdx.x;
    if(i < size){
        C[i] = A[i] + B[i];
    }
}

// use N/2 blocks -> N/2 kernels
__global__ void VecAdd_multi(float* A, float* B, float* C, int size)
{
    // int i = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size){
        C[i] = A[i] + B[i];
    }
}

// use 1 block and N thread -> 1 kernel
__global__ void VecSet(float* v, float value, int size)
{
    int i = threadIdx.x;
    // int i = blockIdx.x;
    if(i < size){
        v[i] = value;
    }
}

__global__ void cuda_print(float* v)
{
    int i = threadIdx.x;
    printf("%d ", i);
}

int main()
{
    // kernel invocation with N threads
    int N = 10;
    float *A_dev, *B_dev, *C_dev, *C_multi_dev;
    HANDLE_ERROR(cudaMalloc((void**)&A_dev, sizeof(float)*N));
    HANDLE_ERROR(cudaMalloc((void**)&B_dev, sizeof(float)*N));
    HANDLE_ERROR(cudaMalloc((void**)&C_dev, sizeof(float)*N));
    HANDLE_ERROR(cudaMalloc((void**)&C_multi_dev, sizeof(float)*N));

    VecSet<<<1, N>>>(A_dev, 1, N);
    VecSet<<<1, N>>>(B_dev, 2, N);
    VecAdd<<<N, 1>>>(A_dev, B_dev, C_dev, N);
    VecAdd_multi<<<N/2, 2>>>(A_dev, B_dev, C_dev, N);

    const auto A = ToVector(A_dev, N);
    const auto B = ToVector(B_dev, N);
    const auto C = ToVector(C_dev, N);
    const auto C_multi = ToVector(C_multi_dev, N);
                             
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    cudaFree(C_multi_dev);

    print_array(A, N);
    print_array(B, N);
    std::cout << "+ -------" << std::endl;
    print_array(C, N);
    print_array(C_multi, N);
}
