#include <iostream>
#include <cuda_runtime.h>

#define CEIL (a+b-1)/b
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)

void __cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("cuda error(%s), in file(%s), on line(%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

void softmax_matrix_cpu()
{

}

__global__ void softmax_matrix_row(const float *din, float *dout, const int n)
{
    
}