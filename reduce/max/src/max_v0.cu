#include <cuda_runtime.h>
#include <iostream>
#include<limits>

#define CEIL(a,b) (a+b-1)/b

void max_cpu(float *hin, float *hout, int n)
{
    float max_res = hin[0];
    for (int i = 1; i < n; ++i) {
        max_res = max_res > hin[i] ? max_res : hin[i];
    }
    *hout = max_res;
}

template<typename T>
bool check_result(T a, T b)
{
    if (a == b) {
        printf("test passed\n");
        return true;
    } else {
        printf("test failed\n");
        return false;
    }
}

__device__ float atomicMax(float* address, float val)
{
    int *address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i; //作为旧值
    int expected;
    do {
        expected = old;
        old = atomicCAS(address_as_i, expected, __float_as_int(fmaxf(val, __int_as_float(expected))));  //最终address_as_i中就是最大的值
    } while(expected != old);

    return __int_as_float(old);
}

__global__ void max_v0(float *din, float *dout, int n)
{
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = gtid;i < n; i += gridDim.x * blockDim.x) {
        atomicMax(dout, din[i]);
    }
}

template<typename T>
float benchmark_kernel(T func, int repeats, int warmup = 1)
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
    cudaEventElapsedTime(&time, start ,stop);
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
    float *hin, *hout;
    float *din, *dout;
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(sizeof(float));
    memset(hout, 0, sizeof(float));
    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMalloc((void**)&dout, sizeof(float));
    float init = std::numeric_limits<float>::lowest();
    cudaMemcpy(dout, &init, sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < N; ++i) {
        hin[i] = i % BLOCK_SIZE;
    }
    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);
    max_v0<<<GRID_SIZE, BLOCK_SIZE>>>(din, dout, N);
    cudaMemcpy(hout, dout, sizeof(float), cudaMemcpyDeviceToHost);
    float *hout_cpu = (float*)malloc(sizeof(float));
    memset(hout_cpu, 0, sizeof(float));
    max_cpu(hin, hout_cpu, N);
    printf("hout_cpu=%f\n", *hout_cpu);
    if (check_result<float>(*hout_cpu, *hout)) {
        printf("test passed, hout_cpu = %f, hout = %f\n", *hout_cpu, *hout);
    } else {
        printf("test failed, hout_cpu = %f, hout = %f\n", *hout_cpu, *hout);
    }
    auto max_v0_kernel = [&](){max_v0<<<GRID_SIZE, BLOCK_SIZE>>>(din, dout, N);};
    constexpr int repeats = 5;
    float time = benchmark_kernel(max_v0_kernel, repeats);
    printf("time=%f\n", time);
    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(hout_cpu);
    return 0;
}
