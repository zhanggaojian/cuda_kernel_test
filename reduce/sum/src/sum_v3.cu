#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>

#define CEIL(a,b) (a+b-1)/b

void sum_cpu(const float *hin, double *hout, const int n)
{
    double sum = 0.0;
    for (int i = 0;i < n; ++i) {
        sum += hin[i];
    }
    *hout = sum;
}

bool check_result(const double hin, const float *din, const int n)
{
    double sum = 0.0;
    for (int i = 0;i < n;++i) {
        sum += din[i];
    }
    double atol = 1e-6;
    double rtol = 1e-6;
    double abs_diff = std::fabs(hin - sum);
    double diff = atol + rtol * std::max(std::fabs(hin), std::fabs(sum));
    if (abs_diff > diff) {
        printf("result err, hin=%f, din=%f\n", hin, sum);
        return false;
    }
    return true;
}

template<int blockSize, int warpSize>
__global__ void sum_v3(const float *din, float *dout, const int n)
{
    constexpr int WARP_NUMS = blockSize / warpSize;
    __shared__ float shm[WARP_NUMS];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warpSize;  // warp id in block
    int laneId = tid % warpSize;   //  lane id in warp
    shm[warpId] = 0.0;
    float val = gtid < n ? din[gtid] : 0.0f;
    for (int i = warpSize >> 1;i > 0;i >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, i);
    }

    if (laneId == 0) {
        shm[warpId] = val; //把所有warp的结果，都放到一个shm里面，然后再做统一
    }
    __syncthreads();

    //现在所有的结果都在每个block的shm里面了，现在再把shm全部求和，因为shm大小是waprsize，因此只需要一个warp就能把所有的数据给归约
    if (warpId == 0) {//使用第一个warp做归约
        //给这个warp的所有lane的寄存器做初始化
        val = tid < (blockDim.x / warpSize) ? shm[laneId] : 0.0;
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, i);
        }
        if (laneId == 0)
            dout[blockIdx.x] = val;
    }
}

template<typename T>
float benchmark_kernel(T func, int repeats, int warmup = 1)
{
    float time = 0.0;
    if (repeats <= 0) {
        return time;
    }

    for (int i = 0;i < warmup;++i) {
        func();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0;i < repeats;++i) {
        func();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time / repeats;
}

int main()
{
    constexpr int N = 10000000;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    dim3 grid_size(GRID_SIZE);
    dim3 block_size(BLOCK_SIZE);
    float *hin;
    float *din, *dout;
    hin = (float*)malloc(N * sizeof(float));
    for (int i = 0;i < N;++i) {
        hin[i] = i % BLOCK_SIZE;
    }
    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMalloc((void**)&dout, GRID_SIZE * sizeof(float));
    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, GRID_SIZE * sizeof(float));
    double *sum_host = (double*)malloc(sizeof(double));
    memset(sum_host, 0, sizeof(double));
    sum_cpu(hin, sum_host, N);
    sum_v3<BLOCK_SIZE, WARP_SIZE><<<grid_size, block_size>>>(din, dout, N);
    float *dout_cpu = (float*)malloc(GRID_SIZE * sizeof(float));
    cudaMemcpy(dout_cpu, dout, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    if (check_result(*sum_host, dout_cpu, GRID_SIZE)) {
        printf("test passed!\n");
    } else {
        printf("test failed!\n");
    }

    auto sum_v3_kernel = [&](){sum_v3<BLOCK_SIZE, WARP_SIZE><<<grid_size, block_size>>>(din, dout, N);};
    float time = benchmark_kernel(sum_v3_kernel, 5, 3);
    printf("time=%f\n", time);
    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(dout_cpu);
    free(sum_host);
    return 0;
}