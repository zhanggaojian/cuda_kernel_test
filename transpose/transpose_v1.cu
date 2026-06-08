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

bool check_result(const float *din, const float *din_cpu ,const int n, const int m)
{
    const float atol = 1e-6;
    const float rtol = 1e-6;
    for (int i = 0;i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int index = i * m + j;
            float abs_diff = std::fabs(din[index] - din_cpu[index]);
            float scale = std::max(std::fabs(din[index]), std::fabs(din_cpu[index]));
            if (abs_diff > atol + scale * rtol) {
                printf("res err, index = %d, din = %f, din_cpu = %f\n", index, din[index], din_cpu[index]);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    //matirx shape = [M, N];
    //M 是矩阵的行，在二维坐标系中对应到y轴
    //N 是矩阵的列，在二维坐标系中对应到x轴
    const int M = 1000;
    const int N = 1000;
    const int BLOCK_SIZE = 32;
    dim3 grid_size(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    dim3 block_size(BLOCK_SIZE, 8);

    float *hin, *hout;
    float *din, *dout;
    hin = (float*)malloc(M * N * sizeof(float));
    hout = (float*)malloc(N * M * sizeof(float));
    for (int i = 0;i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int index = i * N  + j;
            hin[index] = index + 1;
        }
    }

    cudaCheck(cudaMalloc((void**)&din, M * N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dout, N * M * sizeof(float)));

    cudaCheck(cudaMemcpy(din, hin, M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    transpose_v1<8, BLOCK_SIZE><<<grid_size, block_size>>>(din, dout, M, N);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMemcpy(hout, dout, N * M *sizeof(float), cudaMemcpyDeviceToHost));

    float *dout_cpu = (float*)malloc(N * M * sizeof(float));
    transpose_cpu(hin, dout_cpu, M, N);

    if (check_result(dout_cpu, hout, N, M)) {
        printf("tets passed!\n");
    } else {
        printf("test failed!\n");
    }

    cudaFree(din);
    cudaFree(dout);

    free(hin);
    free(hout);
    free(dout_cpu);

    return 0;
}