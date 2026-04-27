#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a,b) (a + b - 1) / b

void sum_cpu(float *ha, float *hc, int n)
{
    double sum = 0;
    for(int i = 0;i < n; ++i){
        sum += static_cast<double>(ha[i]);
    }
    *hc = static_cast<float>(sum);
}

bool check_result(float *hc, float *dc)
{
    if (abs(*hc - *dc) > 1e-5) {
        printf("result is not matched, hc is %f, dc is %f\n", *hc, *dc);
        return false;
    }
    return true;
}

__global__ void sum_err(float *da, float *dc, int n)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gtid; i < n ; i += blockDim.x * gridDim.x) {
        atomicAdd(dc, da[i]);
    }
}

int main()
{
    constexpr int N = 10000000;
    constexpr int BLOCK_SIZE = 256;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    float *ha, *hc;
    float *da, *dc;
    ha = (float*)malloc(N * sizeof(float));
    hc = (float*)malloc(sizeof(float));
    cudaMalloc((void**)&da, N * sizeof(float));
    cudaMalloc((void**)&dc, 1 * sizeof(float));
    cudaMemset(dc, 0, sizeof(float));
    for (int i = 0; i < N; ++i) {
        ha[i] = i % BLOCK_SIZE;
    }
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sum_err<<<GRID_SIZE, BLOCK_SIZE>>>(da, dc, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time %f\n", time);
    cudaMemcpy(hc, dc, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    float *hc_cpu = (float*)malloc(sizeof(float));
    memset(hc_cpu, 0, sizeof(float));
    sum_cpu(ha, hc_cpu, N);
    if (check_result(hc_cpu, hc)) {
        printf("test passed\n");
    } else {
        printf("test failed, hc_cpu=%f, hc=%f\n", *hc_cpu, *hc);
    }
    cudaFree(da);
    cudaFree(dc);
    free(ha);
    free(hc);
    free(hc_cpu);
    return 0;
}