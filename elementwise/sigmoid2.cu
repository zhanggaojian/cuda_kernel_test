#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a,b) (a+b-1)/b
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_r err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("cuda error %s in file %s on line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

int main()
{

}