#include <cuda_runtime.h>
#include <iostream>
#define CEIL(a,b) (a + b - 1) / b

void sum_cpu(float *hin, float *hout, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += hin[i];
    }
    *hout = sum;
}

bool check_result(float *hin, float *din)
{
    if (abs(*hin - *din) > 1e-6) {
        printf("result err, hin=%f, din=%f\n", hin, din);
        return false;
    }
    return true;
}

__global__ void sum_v0(float *din, float *dout, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += din[i];
    }
    *dout = sum;
}

template<typename T>
float benchmark_kernel(T func, int repeats, int warpup = 1)
{
    float time = 0.0;
    if (repeats <= 0) return time;
    for (int i = 0; i < warmup; ++i) {
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
    constexpr int BLOCK_SIZE = 1;
    constexpr int GRID_SIZE = 1;
    float *hin, *hout;
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(sizeof(float));
    memset(hout, 0, sizeof(float));
    for (int i = 0; i < N; ++i) {
        hin[i] = i % 256;
    }
    float *din, *dout;
    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMalloc((void**)&dout, sizeof(float));
    cudaMemset(dout, 0, sizeof(float));
    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);
    sum_v0<<<BLOCK_SIZE, GRID_SIZE>>>(din, dout, N);
    sum_cpu(hin, hout, N);
    float *dout_cpu = (float*)malloc(sizeof(float));
    memset(dout_cpu, 0, sizeof(float));
    cudaMemcpy(dout_cpu, dout, sizeof(float), cudaMemcpyDeviceToHost);
    if (check_result(hout, dout_cpu)) {
        printf("test passed!\n");
    } else {
        printf("test failed, hout = %f, dout_cpu = %f\n", hout, dout_cpu);
    }
    auto sum_v1_kernel = [&](){sum_v1<<<BLOCK_SIZE, GRID_SIZE>>>(din, dout, N);};
    float time = benchmark_kernel(sum_v1_kernel, 5, 2);
    printf("time = %f\n", time);
    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(dout_cpu);
    return 0;
}