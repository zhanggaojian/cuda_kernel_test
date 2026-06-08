#include <iostream>
#include <cuda_runtime.h>

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


//shared memory : use swizzling to deal with bank conflict
//1. swizzling特点 x1 ^ y != x2 ^ y, 当且仅当x1 != x2
//结合bank conflict，不同线程会访问同一个bank的不同地址，那么我们可以让不同的线程访问不同的bank，也就是shm的坐标y不一样
//那对于不同的线程threadIdx.x和threadIdx.y
template<int blockRows, int TILE_DIM>
__global__ void transpose_v3(const float *din, float *dout, const int m, const int n)
{
    __shared__ float shm[TILE_DIM][TILE_DIM];
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    for (int i = 0; i < TILE_DIM; i += blockRows) {
        int row = by + threadIdx.y + i;
        int col = bx + threadIdx.x;
        int ty = threadIdx.y + i;
        if (row < m && col < n) {
            shm[ty][threadIdx.x ^ ty] = din[row * n + col];
        }
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i+=blockRows) {
        int row = bx + threadIdx.y + i;
        int col = by + threadIdx.x;
        int ty = threadIdx.y + i;
        if (row < n && col < m) {
            dout[row * m + col] = shm[threadIdx.x][threadIdx.x ^ ty];
        }
    }
}