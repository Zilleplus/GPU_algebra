#include"vecUtils.h"

int main(){
    cudaDeviceProp prop;

    // First find out how many devices there are.
    int count;
    HANDLE_ERROR( cudaGetDeviceCount(&count) );
    for(int i=0; i < count; ++i){
        HANDLE_ERROR( cudaGetDeviceProperties(&prop, i ) );

        std::cout << "found device with name=" << prop.name << std::endl;
    }
}
