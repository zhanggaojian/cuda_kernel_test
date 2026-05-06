#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a,b) (a+b-1)/b

void sum_cpu(const float *hin, double *hout, const int n)
{
    double sum = 0.0f;
    for (int i = 0;i < n; ++i) {
        sum += hin[i];
    }
    *hout = sum;
}

bool check_result(const double *h_sum, const float *d_sum, int n)
{
    double sum_d = 0.0;
    for (int i = 0;i < n;++i) {
        sum_d += d_sum[i];
    }
    const double atol = 1e-12;
    const double rtol = 1e-12;
    const double abs_diff = std::fabs(*h_sum - sum_d);
    const double scale = std::max(std::fabs(*h_sum), std::abs(sum_d));
    const double diff = atol + rtol * scale;
    if (abs_diff <= diff) {
        printf("result ok, h_sum=%f, d_sum=%f\n", *h_sum, sum_d);
        return true;
    } else {
        printf("result err, h_sum=%f, d_sum=%f\n", *h_sum, sum_d);
        return false;
    }
}

template<int blockSize, int warpSize>
__global__ void sum_v4(float *din, float *dout, int n)
{
    constexpr int WARP_NUMS = blockSize / warpSize;
    __shared__ float shm[WARP_NUMS];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    
    float val = gtid < n ? din[gtid] : 0.0f; //current val in current warp
#pragma unroll
    for (int i = warpSize / 2; i > 0; i >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }

    if (laneId == 0) {
        shm[warpId] = val; //shm中每个值，存储的是warp的第一个lane的值
    }
    __syncthreads(); //shm需要syncthreads

    if (warpId == 0) {
        val = tid < WARP_NUMS ? shm[tid] : 0.0;
        for (int i = warpSize / 2; i > 0; i >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, i);
        }
        if (laneId == 0) {
            dout[blockIdx.x] = val;
        }
    }
}

template<typename T>
float bench_mark(T func, int repeats, int warmup = 1)
{
    float time = 0.0;
    if (repeats <= 0) {
        printf("repeats=%d is err\n", repeats);
        return time;
    }

    for (int i = 0;i < warmup; ++i) {
        func();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0;i < repeats; ++i) {
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
    const int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    float *hin, *din, *dout, *dout_cpu;
    hin = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        hin[i] = i % BLOCK_SIZE;
    }
    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);
    double *hout;
    hout = (double*)malloc(sizeof(double));
    memset(hout, 0, sizeof(double));
    cudaMalloc((void**)&dout, GRID_SIZE * sizeof(float));
    cudaMemset(dout, 0, GRID_SIZE * sizeof(float));
    sum_v4<BLOCK_SIZE, WARP_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(din, dout, N);
    dout_cpu = (float*)malloc(GRID_SIZE * sizeof(float));
    cudaMemcpy(dout_cpu, dout, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    sum_cpu(hin, hout, N);
    if (check_result(hout, dout_cpu, GRID_SIZE)) {
        printf("test passed\n");
    } else {
        printf("test failed\n");
    }

    auto sum_v4_kernel = [&](){sum_v4<BLOCK_SIZE, WARP_SIZE><<<GRID_SIZE,BLOCK_SIZE>>>(din, dout, N);};
    float time = bench_mark(sum_v4_kernel, 5, 3);
    printf("time=%f\n", time);
    return 0;
}