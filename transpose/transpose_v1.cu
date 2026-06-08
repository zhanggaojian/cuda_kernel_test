#include <iostream>
#include <cuda_runtime.h>

#define CEIL(a,b) (a+b-1)/b
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)
void __cudaCheck(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess) {
        printf("cuda error(%s), in file(%s), on line(%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

void transpose_cpu(const float *hin, float *hout, const int m, const int n)
{
    for (int i = 0 ;i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            hout[j * m + i] = hin[i * n + j];
        }
    }
}

//shared memory
//需要注意区分的是，block dim和矩阵的下标的对应关系
template<int blockRows, int TILE_DIM>
__global__ void transpose_v1(const float *din, float *dout, const int m, const int n)
{
    __shared__ float shm[TILE_DIM][TILE_DIM];
    const int bx = blockIdx.x * TILE_DIM; //在x轴上，是第几个block的起始位置，其实这里blockDim.x=TILE_DIM
    const int by = blockIdx.y * TILE_DIM; //在y轴上，是第几个block的起始位置

    //合并read to shm from din
    // shared memory是一个block共享的，通常是一个block处理一个tile，然后tile内部一个thread处理一个或者多个元素
    for (int i = threadIdx.y; i < TILE_DIM; i += blockRows) {
        int row = by + i;
        int col = bx + threadIdx.x; //row and col are global data index
        if (row < m && col < n)
            shm[i][threadIdx.x] = din[row * n + col]; //shm is block
    }
    __syncthreads();

    //合并write to dout from shm
    for (int i = threadIdx.y; i < TILE_DIM; i += blockRows) {
        int row = bx + i;
        int col = by + threadIdx.x;
        if (row < n && col < m)
            dout[row * m + col] = shm[threadIdx.x][i];
    }
}