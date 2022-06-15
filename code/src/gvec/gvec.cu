#include<gvec/utils.h>
#include<gvec/vec.h>
#include<cuda>

namespace gvec{
    Vec::Vec(int size){
        HANDLE_ERROR( cudaMalloc((void**)&d_, sizeof(float)*size) )
    }

    Vec::~Vec(){
        cudaFree(d_);
    }

    int size() const{
        return size_;
    }

    float getValue(int i) const{
        // TODO::copy from device
        return 0.0;
    }

    void setValue(int index, float value){
        // TODO::copy to device
    }

    Vec operator+(const Vec& other) const{
        Vec out(size());

        // TODO::add

        return out;
    }
}
