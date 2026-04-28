#include <cuda_runtime.h>
#include <iostream>
#define CEIL(a, b) (a+b-1)/b

void sum_cpu(float *hin, float *hout, int n)
{
    double sum = 0.0f;
    for (int i = 0;i < n; ++i) {
        sum += hin[i];
    }
    *hout = sum;
}

bool check_result(float hin, float din) {
    if (abs(hin - din) > 1e-6) {
        printf("result err, hin=%f, din=%f\n", hin, din);
        return false;
    }
    return true;
}

template<int blockSize>
__global__ void sum_v1(float *din, float *dout, int n)
{
    int gtid = blockDim.x * blockIdx.x + threadIdx.x; //gtid in global
    int tid = threadIdx.x; //tid in current block
    __shared__ float shm[blockSize]; //shm in every block
     //shm[tid] = din[gtid]; //gtid element global index
    shm[tid] = gtid < n ? din[tid] : 0.0f;
    __syncthreads(); //all threads in one block
    //这里是在block内对所有元素做归约，也是两两求和，每次求和
    //问题来了，每次求和的两个元素的index之间的offset是多少呢
    for (int i = 1; i < blockDim.x; i *= 2) {
        if (tid % (i * 2) == 0)
            shm[tid] += shm[tid + i]; //tid and tid + i reduce
        __syncthreads();
    }

    //write back to global memory
    if (tid == 0) {
        dout[blockIdx.x] = shm[tid]; //reduce in cpu
    }
}

template<typename T>
float benchmark_kernel(T func, int repeats, int warmup = 1)
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
    return time /repeats;
} 

int main()
{
    constexpr int N = 10000000;
    constexpr int BLOCK_SIZE = 256;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    dim3 grid_size(GRID_SIZE);
    dim3 block_size(BLOCK_SIZE);
    float *hin, *hout;
    float *din, *dout;
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(1 * sizeof(float));
    memset(hout, 0, sizeof(float));
    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMalloc((void**)&dout, GRID_SIZE * sizeof(float));
    cudaMemset(dout, 0, sizeof(float));
    for (int i = 0; i < N ;++i) {
        hin[i] = i % 256;
    }
    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);
    sum_v1<BLOCK_SIZE><<<grid_size, block_size>>>(din, dout, N);
    float *dout_cpu = (float*)malloc(GRID_SIZE * sizeof(float));
    memset(dout_cpu, 0, sizeof(float));
    sum_cpu(hin, hout, N);
    cudaMemcpy(dout_cpu, dout, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    float sum_cpu;
    for (int i = 0;i < GRID_SIZE; ++i) {
        sum_cpu += dout_cpu[i];
    }
    if (check_result(*hout, sum_cpu)) {
        printf("test passed\n");
    } else {
        printf("test failed, hout=%f, dout_cpu=%f\n", *hout, sum_cpu);
    }
    auto sum_v1_kernel = [&](){sum_v1<BLOCK_SIZE><<<grid_size, block_size>>>(din, dout, N);};
    float time = benchmark_kernel(sum_v1_kernel, 5, 2);
    printf("time=%f\n", time);
    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(dout_cpu);
    return 0;
}