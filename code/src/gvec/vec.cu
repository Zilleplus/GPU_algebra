#include<gvec/utils.h>
#include<gvec/vec.h>
#include<cuda.h>
#include<assert.h>

constexpr int N = 1;

__global__ void VecAdd(const float* A, const float* B, float* C, int size)
{
    int i = threadIdx.x;
    while(i < size)
    {
        C[i] = A[i] + B[i];
        // jumping like this is not so good for cach locality
        i = i + N;
    }
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

    Vec Vec::operator+(const Vec& other) const &{
        assert(size() == other.size());
        Vec out(size());

        // Use 1 unit with N threads
        // <<<blocks, threads>>>
        VecAdd<<<1, N>>>(d_, other.d_, out.d_, size_);

        return out;
    }
}
