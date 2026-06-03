#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>

#define CEIL(a,b) (a+b-1) / b

void max_cpu(const float *hin, float *hout, int n)
{
    float maxval = hin[0];
    for (int i = 1; i < n; ++i) {
        maxval = maxval > hin[i] ? maxval : hin[i];
    }
    *hout = maxval;
}

bool check_result(const float hin, const float din)
{
    if (hin == din) {
        printf("result is ok\n");
        return true; 
    } else {
        printf("result is err, hin=%f, din=%f\n", hin, din);
        return false;
    }
}

template<int blockSize, int warpSize>
__global__ void max_v2(const float *din, float *dout, int n)
{
    constexpr int WARP_NUMS = blockSize / warpSize;
    __shared__ float shm[WARP_NUMS];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    float val = gtid < n ? din[gtid] : -FLT_MAX;
    for (int i = warpSize / 2; i > 0; i >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, i));
    }

    if (laneId == 0) {
        shm[warpId] = val; 
    }
    __syncthreads();

    if (warpId == 0) {
        val = laneId < WARP_NUMS ? shm[laneId] : -FLT_MAX;
        for (int i = warpSize / 2; i > 0; i >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, i));
        }
        if (laneId == 0) {
            dout[blockIdx.x] = val;
        }
    }
}

template<typename T>
float benchmark_kernel(T func, int repeats, int warmup = 1)
{
    float time = 0.0f;
    if (repeats <= 0) return time;
    for (int i = 0; i < warmup; ++i) {
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
    return time/ repeats;
}

int main()
{
    constexpr int N = 10000000;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    dim3 grid_size(GRID_SIZE);
    dim3 block_size(BLOCK_SIZE);

    float *hin, *hout, *hout_cpu;
    float *din, *dout;
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(GRID_SIZE * sizeof(float));
    hout_cpu = (float*)malloc(sizeof(float));

    for (int i = 0; i < N; ++i) {
        hin[i] = i % BLOCK_SIZE + 1;
    }
    memset(hout_cpu, 0, sizeof(float));
    
    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMalloc((void**)&dout, GRID_SIZE * sizeof(float));

    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);

    max_v2<BLOCK_SIZE, WARP_SIZE><<<grid_size, block_size>>>(din, dout, N);
    cudaMemcpy(hout, dout, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float max_val = hout[0];
    for (int i = 1; i < GRID_SIZE; ++i) {
        max_val = max_val > hout[i] ? max_val : hout[i];
    }

    max_cpu(hin, hout_cpu, N);

    if (check_result(*hout_cpu, max_val)) {
        printf("test passed\n");
    } else {
        printf("test failed, hout_cpu = %f, max_val = %f\n",*hout_cpu, max_val);
    }

    auto max_v2_kernel = [&](){max_v2<BLOCK_SIZE, WARP_SIZE><<<grid_size, block_size>>>(din, dout, N);};
    float time = benchmark_kernel(max_v2_kernel, 5, 3);
    printf("time=%f\n", time);
    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(hout_cpu);

    return 0;
}