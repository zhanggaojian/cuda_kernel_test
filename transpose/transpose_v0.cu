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
    for (int i = 0;i < m; ++i) {
        for (int j = 0;j < n;++j) {
            hout[j * m + i] = hin[i * n + j];
        }
    }
}

//naive v0
//读的时候按行读取，合并读数据；读合并，写不合并
__global__ void transpose_v0(const float *din, float *dout, const int m, const int n)
{
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int stride_row = gridDim.y * blockDim.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride_col = gridDim.x * blockDim.x;

    for (int i = row; i < m; i+=stride_row) {
        for (int j = col; j < n; j+=stride_col) {
            dout[j * m + i] = din[i * n + j];
        }
    }
}

// //读的时候按列读取，写入的时候按行写; 写合并，读不合并
// __global__ void transpose_v0(const float *din, float *dout, const int m, const int n)
// {
//     const int row = blockDim.y * blockIdx.y + threadIdx.y;
//     const int stride_row = gridDim.y * blockDim.y;
//     const int col = blockDim.x * blockIdx.x + threadIdx.x;
//     const int stride_col = gridDim.x * blockDim.x;

//     for (int i = row; i < n; i+=stride_row) {
//         for (int j = col; j < m; j+=stride_col) {
//             dout[i * m + j] = din[j * n + i];
//         }
//     }
// }

bool check_result(const float *hin, const float *din, const int m, const int n)
{
    float atol = 1e-5;
    float rtol = 1e-6;
    for (int i = 0; i < n; ++i) {
        for (int j = 0;j < m;++j) {
            int index = i * m + j;
            float abs_diff = std::fabs(hin[index] - din[index]);
            float scale = std::max(std::fabs(hin[index]), std::fabs(din[index]));
            if (abs_diff > atol + rtol * scale) {
                printf("res err, index=%d, hin=%f, din=%f\n", index, hin[index], din[index]);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    const int M = 1000;
    const int N = 256;
    constexpr int BLOCK_SIZE = 32;


    //计算cpu的值，用于校验正确性
    float *hin, *hout;
    hin = (float*)malloc(M * N * sizeof(float));
    hout = (float*) malloc(N * M * sizeof(float));
    for(int i = 0;i <M ;++i) {
        for (int j = 0;j < N;++j) {
            hin[i * N + j] =  i * N + j;
        }
    }
    transpose_cpu(hin, hout, M, N);

    //计算kernel的值
    float *din, *dout;
    cudaCheck(cudaMalloc((void**)&din, M * N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dout, N * M * sizeof(float)));
    cudaCheck(cudaMemcpy(din, hin , M * N * sizeof(float), cudaMemcpyHostToDevice));
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    transpose_v0<<<gridSize, blockSize>>>(din, dout, M, N);
    cudaCheck(cudaGetLastError());

    float *dout_cpu = (float*)malloc(N * M * sizeof(float));
    cudaCheck(cudaMemcpy(dout_cpu, dout, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    if (check_result(hout, dout_cpu, M, N)) {
        printf("test pass!\n");
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