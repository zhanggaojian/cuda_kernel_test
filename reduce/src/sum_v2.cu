#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a,b) (a+b-1)/b

void sum_cpu(float *hin, float *hout, int n)
{
    double sum = 0.0;
    for (int i = 0;i < n; ++i) {
        sum += hin[i];
    }
    *hout = sum;
}

bool check_result(float hin, float din)
{
    if (abs(hin - din) > 1e-6) {
        printf("result err, hin = %f, din = %f\n", hin, din);
        return false;
    }
    return true;
}

template<int blockSize>
__global__ sum_v2(float *din, float *dout, int n)
{
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shm[blockSize];
    shm[tid] = gtid < n ? din[gtid] : 0.0;
    __syncthreads();
    for (int i = blockDim.x >> 1; i > 0; i >> 1) {
        if (tid < i) { //这里判断，左边的线程id < i，把计算结果放置到数组左半边., 减少取模运算，降低SM利用率
            shm[tid] += shm[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dout[blockIdx.x] = shm[tid]; //每个block的tid=0的线程，替代当前block的结果
    }
}

template<typename T>
float benchmark_kernel(T func, int repeats, int warmup = 3)
{
    float time = 0.0;
    if (repeats <= 0) return time;

    for (int i = 0;i < warmup; ++i) {
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
    constexpr int BLOCK_SIZE = 256;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&devicePeop, 0);
    int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    float *hin, *hout;
    float *din, *dout;
    dim3 grid_size(GRID_SIZE);
    dim3 block_size(BLOCK_SIZE);
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(1 * sizeof(float));
    for (int i = 0; i < N; ++i) {
        hin[i] = i % BLOCK_SIZE;
    }
    memset(hout, 0, sizeof(float));
    cudaMalloc((void**)&hin, N * sizeof(float));
    cudaMalloc((void**)&hout, GRID_SIZE * sizeof(float));
    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, GRID_SIZE * sizeof(float));
    sum_cpu(hin, hout, N);
    sum_v2<BLOCK_SIZE><<<grid_size, block_size>>>(din, dout, N);
    float* dout_cpu = (float*)malloc(GRID_SIZE * sizeof(float));
    cudaMemcpy(dout_cpu, dout, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    double sum_d = 0.0;
    for (int i = 0; i < GRID_SIZE; ++i) {
        sum_d += dout_cpu[i];
    }
    if (check_result(*hout, sum_d)) {
        printf("test passed\n");
    } else {
        printf("test failed\n");
    }

    auto sum_v2_kernel = [&](){sum_v2<BLOCK_SIZE><<<grid_size, block_size>>>(din, dout, N);};
    float time = benchmark_kernel(sum_v2_kernel, 5, 3);
    printf("time = %f\n", time);
    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(dout_cpu);
    return 0;
}