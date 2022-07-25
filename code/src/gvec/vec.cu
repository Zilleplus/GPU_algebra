#include<gvec/utils.h>
#include<gvec/vec.h>
#include<cuda.h>
#include<assert.h>

__global__ void VecAddVec(const float* A, const float* B, float* C, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < size)
    {
        C[i] = A[i] + B[i];
        i = i + blockDim.x * gridDim.x;
    }
}

__global__ void VecSubVec(const float* A, const float* B, float* C, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < size)
    {
        C[i] = A[i] - B[i];
        i = i + blockDim.x * gridDim.x;
    }
}

__global__ void VecMulVec(const float* A, const float* B, float* C, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < size)
    {
        C[i] = A[i] * B[i];
        i = i + blockDim.x * gridDim.x;
    }
}

__global__ void VecDivVec(const float* A, const float* B, float* C, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < size)
    {
        C[i] = A[i] / B[i];
        i = i + blockDim.x * gridDim.x;
    }
}

__global__ void dotProduct(const float* A, const float* B, float* C, int size)
{
    __shared__ float cache[gvec::Vec::threadsPerBlock];

    int num_threads_per_block = blockDim.x;
    int num_blocks = gridDim.x;

    // do element wise multiplication
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0;
    while(tid<size){
        temp = temp + A[tid]*B[tid];
        tid = tid + num_threads_per_block*num_blocks;
    }
    cache[threadIdx.x] = temp;

    __syncthreads();

    // Reduce the cache to 1 value per block.
    int i = num_threads_per_block/2;
    while(i > 0){
        if(threadIdx.x < i){
            cache[threadIdx.x] = cache[threadIdx.x] + cache[threadIdx.x + i];
        }
        i = i / 2;
    }

    C[blockIdx.x] = cache[0];
}

namespace gvec{

    Vec::Vec(int size) : size_(size){
        HANDLE_ERROR( cudaMalloc((void**)&d_, sizeof(float)*size) );
    }

    Vec::Vec(const std::initializer_list<float>& elements)
    {
        size_ = std::size(elements);
        HANDLE_ERROR( cudaMalloc((void**)&d_, sizeof(float)*size_) );
        HANDLE_ERROR( 
            cudaMemcpy(
                d_,                  // dst
                std::data(elements), // src
                sizeof(float)*size_, // size to copy
                cudaMemcpyHostToDevice ) );
    }

    Vec::Vec(Vec&& other)
    {
        d_ = other.d_;
        // other.d_ to nullptr to avoid the cudaFree call.
        other.d_ = nullptr;
        other.d_ = 0;
        size_ = other.size_;
    }

    Vec::~Vec(){
        if(d_!=nullptr)
        {
			cudaFree(d_);
        }
        d_ = nullptr;
    }

    int Vec::size() const{
        return size_;
    }

    float Vec::getValue(int index) const{
        float out;
        HANDLE_ERROR( 
                cudaMemcpy(
                    &out,          // dst
                    d_+index,      // src
                    sizeof(float), // size to copy
                    cudaMemcpyDeviceToHost ) );

        return out;
    }

    void Vec::setValue(int index, float value){
        HANDLE_ERROR( 
                cudaMemcpy(
                    d_+index,       // dst
                    &value,         // src
                    sizeof(float),  // size to copy
                    cudaMemcpyHostToDevice ) );
    }

    Vec Vec::operator+(const Vec& other) const{
        assert(size() == other.size());
        Vec out(size());


        dim3 num_blocks(numBlocks());
        dim3 num_threads(threadsPerBlock);

        VecAddVec<<<num_blocks, num_threads>>>(d_, other.d_, out.d_, size_);

        return out;
    }

    Vec Vec::operator-(const Vec& other) const{
        assert(size() == other.size());
        Vec out(size());


        dim3 num_blocks(numBlocks());
        dim3 num_threads(threadsPerBlock);

        VecSubVec<<<num_blocks, num_threads>>>(d_, other.d_, out.d_, size_);

        return out;
    }

    Vec Vec::operator*(const Vec& other) const{
        assert(size() == other.size());
        Vec out(size());


        dim3 num_blocks(numBlocks());
        dim3 num_threads(threadsPerBlock);

        VecMulVec<<<num_blocks, num_threads>>>(d_, other.d_, out.d_, size_);

        return out;
    }

    Vec Vec::operator/(const Vec& other) const{
        assert(size() == other.size());
        Vec out(size());


        dim3 num_blocks(numBlocks());
        dim3 num_threads(threadsPerBlock);

        VecDivVec<<<num_blocks, num_threads>>>(d_, other.d_, out.d_, size_);

        return out;

    }

    float Vec::dot(const Vec& other) const {
        dim3 num_blocks(numBlocks());
        dim3 num_threads(threadsPerBlock);

        Vec out(numBlocks()); // we return 1 value per block

        dotProduct<<<num_blocks, num_threads>>>(d_, other.d_, out.d_, size_);

        float* to_reduce = (float*)malloc(sizeof(float)*numBlocks());
        HANDLE_ERROR( 
                cudaMemcpy(
                    to_reduce,    // dst
                    out.d_,      // src
                    sizeof(float)*numBlocks(), // size to copy
                    cudaMemcpyDeviceToHost ) );

        float dot_prod = 0;
        for(int i = 0; i < numBlocks(); ++i)
        {
            dot_prod = dot_prod + to_reduce[i];
        }


        return dot_prod;
    }
}
