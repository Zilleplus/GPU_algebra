#include<gvec/utils.h>
#include<gvec/vec.h>
#include<cuda.h>
#include<assert.h>

__global__ void VecAdd(const float* A, const float* B, float* C, int size)
{
    int i = threadIdx.x;
    if(i < size)
    {
        C[i] = A[i] + B[i];
    }
}

constexpr int N = 10;

namespace gvec{
    Vec::Vec(int size){
        HANDLE_ERROR( cudaMalloc((void**)&d_, sizeof(float)*size) );
    }

    Vec::Vec(const std::initializer_list<float>& elements)
    {
        auto s = std::size(elements);
        HANDLE_ERROR( cudaMalloc((void**)&d_, sizeof(float)*s) );
        int i = 0;
        for(const auto v : elements)
        {
            setValue(i, v);
            i = i + 1;
        }
    }

    Vec::Vec(Vec&& other)
    {
        d_ = other.d_;
        size_ = other.size_;
    }

    Vec::~Vec(){
        cudaFree(d_);
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

        VecAdd<<<N, 1>>>(d_, other.d_, out.d_, size_);

        return out;
    }
}
