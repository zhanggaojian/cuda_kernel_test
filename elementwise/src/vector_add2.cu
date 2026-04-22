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

__global__ void vector_add_float4(float* da, float* db, float* dc, int n)
{
    //global tid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < n/4; i += gridDim.x * blockDim.x) {
        float4 ta = reinterpret_cast<float4*>(da)[i];
        float4 tb = reinterpret_cast<float4*>(db)[i];
        float4 tc;
        tc.x = ta.x + tb.x;
        tc.y = ta.y + tb.y;
        tc.z = ta.z + tb.z;
        tc.w = ta.w + tb.w;
        reinterpret_cast<float4*>(dc)[i] = tc;
    }
}

int main()
{
    constexpr int N = 10000000;
    float* ha, *hb, *hc;
    float* da, *db, *dc;
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    ha = (float*)malloc(N * sizeof(float));
    hb = (float*)malloc(N * sizeof(float));
    hc = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        ha[i] = i;
        hb [i] = i + 1;
        hc[i] = ha[i] + hb[i];
    }

    cudaCheck(cudaMalloc((void**)&da, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&db, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dc, N * sizeof(float)));
    cudaCheck(cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_add_float4<<<GRID_SIZE, BLOCK_SIZE>>>(da, db, dc, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time: %f ms\n", time);

    float* hc_gpu = (float*)malloc(N * sizeof(float));
    cudaCheck(cudaMemcpy(hc_gpu, dc, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        if (abs(hc_gpu[i] - hc[i]) > 1e-6) {
            printf("error at index %d\n", i);
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
    free(hb);
    free(hc);
    free(hc_gpu);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}