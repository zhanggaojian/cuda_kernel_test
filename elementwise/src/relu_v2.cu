#include <cuda_runtime.h>
#include <iostream>
#define CEIL(a,b) (a+b-1)/b

void relu_cpu(float* ha, float *hb, int n)
{
    for (int i = 0; i < n; ++i) {
        hb[i] = ha[i] > 0 ? ha[i] : 0;
    }
}

__global__ void relu_v2(float* da, float *db, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n/4; i += blockDim.x * gridDim.x) {
        float4 ta = reinterpret_cast<float4*>(da)[i];
        float4 tb;
        tb.x = ta.x > 0 ? ta.x : 0;
        tb.y = ta.y > 0 ? ta.y : 0;
        tb.z = ta.z > 0 ? ta.z : 0;
        tb.w = ta.w > 0 ? ta.w : 0;
        reinterpret_cast<float4*>(db)[i] = tb;
    }
}

int main()
{
    constexpr int N = 10000000;
    const int BLOCK_SIZE = 256;
    int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    float *ha, *hb, *da, *db;
    ha = (float*)malloc(N * sizeof(float));
    hb = (float*)malloc(N * sizeof(float));
    cudaMalloc((void**)&da, N * sizeof(float));
    cudaMalloc((void**)&db, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        ha[i] = i;
    }
    relu_cpu(ha, hb, N);
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    relu_v2<<<GRID_SIZE, BLOCK_SIZE>>>(da, db, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time: %f ms\n", time);
    float* db_cpu = (float*)malloc(N * sizeof(float));
    cudaMemcpy(db_cpu, db, N*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        if (abs(hb[i] - db_cpu[i]) > 1e-6) {
            printf("test failed at index %d\n, hb[%d] = %f, db_cpu[%d] = %f\n", i, i, hb[i], i, db_cpu[i]);
            return 0;
        }
    }
    printf("test passed\n");
    return 0;
}