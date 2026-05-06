#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a,b) (a + b - 1) / b

void sum_cpu(float *ha, double *hc, int n)
{
    double sum = 0;
    for(int i = 0;i < n; ++i){
        sum += static_cast<double>(ha[i]);
    }
    *hc = sum;
}

bool check_result(const double *hc, const double *dc)
{
    const double atol = 1e-12;
    const double rtol = 1e-10;
    double abs_diff = std::fabs(*hc - *dc);
    double scale = std::max(std::fabs(*hc), std::fabs(*dc));
    double diff = atol + rtol * scale;
    if (abs_diff <= diff) {
        printf("test passed, hc=%f, dc=%f\n", *hc, *dc);
        return true;
    } else {
        printf("test failed, hc=%f, dc=%f\n", *hc, *dc);
        return false;
    }
}

__global__ void sum_v0(float *da, double *dc, int n)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gtid; i < n ; i += blockDim.x * gridDim.x) {
        atomicAdd(dc, da[i]);
    }
}

template<typename T>
float benchmark_kernel(T func,int repeats, int warmup = 3)
{
    float time = 0.0;
    for (int i = 0;i < warmup; ++i) {
        func();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
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
    float *ha;
    float *da;
    double *dc, *hc;
    ha = (float*)malloc(N * sizeof(float));
    hc = (double*)malloc(sizeof(double));
    cudaMalloc((void**)&da, N * sizeof(float));
    cudaMalloc((void**)&dc, 1 * sizeof(double));
    cudaMemset(dc, 0, sizeof(double));
    for (int i = 0; i < N; ++i) {
        ha[i] = i % BLOCK_SIZE;
    }
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    sum_v0<<<GRID_SIZE, BLOCK_SIZE>>>(da, dc, N);
    cudaMemcpy(hc, dc, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    double *res_cpu= (double*)malloc(sizeof(double));
    sum_cpu(ha, res_cpu, N);
    if (check_result(res_cpu, hc)) {
        printf("test passed\n");
    } else {
        printf("test failed, hc_cpu=%f, hc=%f\n", *res_cpu, *hc);
    }

    auto sum_v0_kernel = [&](){sum_v0<<<GRID_SIZE, BLOCK_SIZE>>>(da, dc, N);};
    float time = benchmark_kernel(sum_v0_kernel, 5, 3);
    printf("time=%f\n", time);
    cudaFree(da);
    cudaFree(dc);
    free(ha);
    free(hc);
    free(res_cpu);
    return 0;
}