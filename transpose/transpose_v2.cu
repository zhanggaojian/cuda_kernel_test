#include <iostream>
#include <cuda_runtime.h>

#define CEIL(a,b) (a+b-1)/b
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)
void __cudaCheck(cudaError_t err, const char *file, const int line)
{
    if (err != cudaSuccess) {
        printf("cuda error(%s), in file(%s), on line(%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

//shared memory : use padding to deal with bank conflict
template<int blockRows, int TILE_DIM>
__global__ void transpose_v2(const float *din, float *dout, const int m, const int n)
{
    __shared__ float shm[TILE_DIM][TILE_DIM+1];
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i+=blockRows) {
        int row = by + threadIdx.y + i;
        int col = bx + threadIdx.x;
        if (row < m && col < n) {
            shm[threadIdx.y + i][threadIdx.x] = din[row * n + col];
        }
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i+=blockRows) {
        int row = bx + threadIdx.y + i;
        int col = by + threadIdx.x;
        if (row < n && col < m) {
            dout[row * m + col] = shm[threadIdx.x][threadIdx.y + i];
        }
    }

}