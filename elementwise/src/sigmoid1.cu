#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a,b) (a + b - 1) / b

#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("cuda error %s in file %s on line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

void sigmoid_cpu(float* ha, float* hs, int n)
{
    for (int i = 0;i < n; ++i) {
        hs[i] = 1.0f / (1.0f + exp(-ha[i]));
    }
}

__global__ void sigmoid1(float* da, float* ds, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        ds[i] = 1.0f / (1.0f + expf(-da[i]));
    }
}


int main()
{
    constexpr int N = 10000000;
    float *ha, *hs_cpu;
    ha = (float*)malloc(N * sizeof(float));
    hs_cpu = (float*)malloc(N * sizeof(float));

    float *da, *ds;
    cudaCheck(cudaMalloc((void**)&da, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&ds, N * sizeof(float)));
    cudaCheck(cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    sigmoid_cpu(ha, hs_cpu, N);
    // printf("sigmoid_cpu done\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sigmoid1<<<GRID_SIZE, BLOCK_SIZE>>>(da, ds, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time: %f ms\n", time);
    // printf("sigmoid1 done\n");

    float *ds_cpu = (float*)malloc(N * sizeof(float));
    cudaMemcpy(ds_cpu, ds, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        if (abs(hs_cpu[i] - ds_cpu[i]) > 1e-6) {
            printf("error at index %d\n, hs_cpu[i]: %f, ds_cpu[i]: %f\n", i, hs_cpu[i], ds_cpu[i]);
            break;
        }
    }
    printf("test passed\n");
    free(ha);
    free(hs_cpu);
    cudaFree(da);
    cudaFree(ds);
}