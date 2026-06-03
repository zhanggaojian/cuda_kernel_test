#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <cfloat>

#define CEIL(a,b) (a+b-1) / b

void max_cpu(const float *hin, float *hout, const int n)
{
    if (n <= 0) {
        printf("input is error, n=%d\n", n);
        return;
    }
    float max_res = hin[0];
    for (int i = 1; i < n; ++i) {
        max_res = max_res > hin[i] ? max_res : hin[i];
    }
    *hout = max_res;
}

bool check_result(const float *hin, const float *din) {
    if (*hin == *din) {
        printf("result is ok, hin=%f, din=%f\n", *hin, *din);
        return true;
    } else {
        printf("result is err, hin=%f, din=%f\n", *hin, *din);
        return false;
    }
}

template<int blockSize>
__global__ void max_v1(const float *din, float *dout, const int n)
{
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    //每个block内都会有这样一个shm，大小就是block中线程数量的大小，每个block内部的所有线程都可以访问到。
    //blocksize指的是block内部线程数量的多少，并不是block本身的数量
    __shared__ float shm[blockSize];
    shm[tid] = gtid < n ? din[gtid] : -FLT_MAX;
    __syncthreads(); //每次读写shared memory都要同步，同步是针对block中的所有线程

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm[tid] = fmaxf(shm[tid], shm[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        dout[blockIdx.x] = shm[tid];
    }
}

template<typename T>
float benchmark_kernel(T func, int repeats, int warmup = 1)
{
    float time = 0.0f;
    if (repeats <= 0) return time;

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
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    float *hin, *hout, *hout_cpu;
    float *din, *dout;
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(GRID_SIZE * sizeof(float));
    hout_cpu = (float*)malloc(sizeof(float));
    for (int i = 0; i < N; ++i) {
        hin[i] = i % BLOCK_SIZE;
    }
    memset(hout_cpu, 0, sizeof(float));

    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMalloc((void**)&dout, GRID_SIZE * sizeof(float));
    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);

    max_v1<BLOCK_SIZE><<<GRID_SIZE,BLOCK_SIZE>>>(din, dout, N);
    cudaMemcpy(hout, dout, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    float max_val = hout[0];
    for (int i = 0; i < GRID_SIZE; ++i) {
        max_val = max_val > hout[i] ? max_val : hout[i];
    }
    max_cpu(hin, hout_cpu, N);
    if (check_result(hout_cpu, &max_val)) {
        printf("test passed!\n");
    } else {
        printf("test failed, hout_cpu = %f, dout = %f\n", *hout_cpu, max_val);
    }
    auto max_v1_kernel = [&](){max_v1<BLOCK_SIZE><<<GRID_SIZE,BLOCK_SIZE>>>(din, dout, N);};
    float time = benchmark_kernel(max_v1_kernel, 5, 3);
    printf("time=%f\n", time);
    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(hout_cpu);
    return 0;
}