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

void transpose_cpu(const float *hin, float *hout, const int m, const int n)
{
    for (int i = 0;i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            hout[j * m + i] = hin[i * n + j];
        }
    }
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

bool check_result(const float *hin, const float *hout, const int m, const int n)
{
    const float atol = 1e-6;
    const float rtol = 1e-6;
    for (int i = 0;i < m; ++i) {
        for (int j = 0; j < n;++j) {
            int index = i * n + j;
            float abs_diff = std::fabs(hin[index] - hout[index]);
            float scale = std::max(std::fabs(hin[index]), std::fabs(hout[index]));
            if (abs_diff > atol + rtol * scale) {
                printf("res err, index = %d, hin = %f, hout = %f\n", index, hin[index], hout[index]);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    //matrix shape = [M,N]
    //M代表矩阵的行数，对应到二维坐标系中是y轴
    //N代表矩阵的列数，对应到二维坐标系中是x轴
    const int M = 1000;
    const int N =1000;
    const int BLOCK_SIZE = 32;
    dim3 grid_size(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    dim3 block_size(BLOCK_SIZE, 8);

    float *hin = (float*)malloc(M * N * sizeof(float));
    float *hout = (float*)malloc(N * M * sizeof(float));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int index = i * N + j;
            hin[index] = index + 1;
        }
    }

    float *din,*dout;
    cudaCheck(cudaMalloc((void**)&din, M * N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dout, N * M * sizeof(float)));
    cudaCheck(cudaMemcpy(din, hin, M * N * sizeof(float), cudaMemcpyHostToDevice));
    transpose_v2<8, BLOCK_SIZE><<<grid_size,block_size>>>(din, dout, M, N);
    cudaCheck(cudaGetLastError());

    float *dout_cpu = (float*)malloc(N * M * sizeof(float));
    cudaCheck(cudaMemcpy(dout_cpu, dout, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    transpose_cpu(hin, hout, M, N);

    if (check_result(hout, dout_cpu, N, M)) {
        printf("test passed!\n");
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