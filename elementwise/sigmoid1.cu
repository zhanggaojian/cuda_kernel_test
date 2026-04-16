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
        ds[i] = 1.0f / (1.0f + exp(-da[i]));
    }
}


int main()
{
    constexpr int N = 10000000;
    float *ha, *hb, *hs_cpu;
    ha = (float*)malloc(N * sizeof(float));
    hb = (float*)malloc(N * sizeof(float));
    hs_cpu = (float*)malloc(N * sizeof(float));

    float *da, *ds;
    cudaCheck(cudaMalloc((void**)&da, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&ds, N * sizeof(float)));
    cudaCheck(cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    sigmoid_cpu(ha, hs_cpu, N);
    sigmoid1<<<GRID_SIZE, BLOCK_SIZE>>>(da, ds, N);
}