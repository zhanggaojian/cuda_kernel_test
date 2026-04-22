#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a, b) (a + b - 1) / b
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("cuda error in file:%s, line:%d, error:%s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return;
}

void relu_cpu(float *ha, float *hr, int n)
{
    for (int i = 0; i < n; ++i) {
        hr[i] = ha[i] > 0 ? ha[i] : 0;
    }
}

__global__ void relu_v1(float *da, float *dr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dr[i] = da[i] > 0 ? da[i] : 0;
    }
}

int main()
{
    constexpr int N = 10000000;
    float *ha, *hr;
    float *da, *dr;
    ha = (float*)malloc(N * sizeof(float));
    hr = (float*)malloc(N * sizeof(float));
    cudaCheck(cudaMalloc((void**)&da, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dr, N * sizeof(float)));

    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = CEIL(N, BLOCK_SIZE);

    for(int i = 0; i < N; ++i) {
        ha[i] = i;
    }
    relu_cpu(ha, hr, N);
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    relu_v1<<<GRID_SIZE, BLOCK_SIZE>>>(da, dr, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time: %f ms\n", time);
    float *dr_cpu = (float*)malloc(N * sizeof(float));
    cudaMemcpy(dr_cpu, dr, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        if (abs(dr_cpu[i] - hr[i]) > 1e-6) {
            printf("error at index %d: %f != %f\n", i, dr_cpu[i], hr[i]);
            passed = false;
            break;
        }
    }
    if (passed) {
        printf("test passed\n");
    } else {
        printf("test failed\n");
    }
    free(ha);
    free(hr);
    cudaFree(da);
    cudaFree(dr);
    free(dr_cpu);
    return 0;
}