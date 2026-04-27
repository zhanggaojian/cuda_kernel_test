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

__global__ void sum_v1(float *din, float *dout, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += din[i];
    }
    *dout = sum;
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sum_v1<<<BLOCK_SIZE, GRID_SIZE>>>(din, dout, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time = %f\n", time);
    sum_cpu(hin, hout, N);
    float *dout_cpu = (float*)malloc(sizeof(float));
    memset(dout_cpu, 0, sizeof(float));
    cudaMemcpy(dout_cpu, dout, sizeof(float), cudaMemcpyDeviceToHost);
    if (check_result(hout, dout_cpu)) {
        printf("test passed!\n");
    } else {
        printf("test failed, hout = %f, dout_cpu = %f\n", hout, dout_cpu);
    }
    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(dout_cpu);
    return 0;
}