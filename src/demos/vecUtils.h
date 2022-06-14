#include<iostream>
#include<vector>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

std::vector<float> ToVector(float* cuda_v, int length)
{
    std::vector<float> out(length);
    HANDLE_ERROR( cudaMemcpy( out.data(), cuda_v, sizeof(float)*length, cudaMemcpyDeviceToHost ) );

    return out;
}

void print_array(const std::vector<float>& v, int length)
{
    for(int i = 0; i < length; ++i)
    {
        std::cout << " " << v[i];
    }
    std::cout << std::endl;
}
