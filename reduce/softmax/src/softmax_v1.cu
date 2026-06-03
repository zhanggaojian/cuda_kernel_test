#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cfloat>

#define CEIL(a,b) (a+b-1)/b
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)

void __cudaCheck(cudaError_t err,const char *file, int line)
{   
    if (err != cudaSuccess) {
        printf("cuda error(%s), in file(%s), on line(%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

//减去最大值
void softmax_cpu(const float*hin, float* hout, const int n)
{
    //max val
    if (n <= 0) return;
    float max_val = hin[0];
    for (int i = 1; i < n; ++i) {
        max_val = max_val > hin[i] ? max_val : hin[i];
    }

    double sum = 0.0;
    for (int i = 0;i < n; ++i) {
        hout[i] = expf(hin[i] - max_val);
        sum += hout[i];
    }

    for (int i = 0;i < n; ++i) {
        hout[i] /= sum;
    }
}

bool check_result(const float *hin, const float *din, const int n)
{
    float atol = 1e-6;
    float rtol = 1e-6;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(hin[i]) || !std::isfinite(din[i])) {
            printf("input is nan, i=%d, hin[i]=%f, din[i]=%f\n", i, hin[i], din[i]);
            return false;
        }
        float abs_dff = std::fabs(hin[i] - din[i]);
        float diff = atol + rtol * std::max(std::fabs(hin[i]), std::fabs(din[i]));
        if (abs_dff > diff) {
            printf("result is err, i=%d, hin[i]=%f, din[i]=%f\n", i, hin[i], din[i]);
            return false;
        }
    }
    return true;
}

__device__ float atomicMax(float *addr, float val)
{
    int *addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int expected = 0;
    do {
        expected = old;
        old = atomicCAS(addr_as_int, expected, __float_as_int(fmaxf(val, __int_as_float(expected))));
    } while(expected != old);
    return __int_as_float(old);
}


template<int blockSize, int warpSize>
__global__ void max_kernel(const float* din, float *dout, const int n)
{
    constexpr int warpNum = blockSize / warpSize;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    __shared__ float shm[warpNum];
    float val = -FLT_MAX;
    for (int i = gtid;i < n; i += gridDim.x * blockDim.x) {
        val = fmaxf(val , din[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if (laneId == 0) {
        shm[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        val = laneId < warpNum ? shm[laneId] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }

        if (laneId == 0) {
            atomicMax(dout, val);
        }
    }
}

template<int blockSize, int warpSize>
__global__ void sum_kernel(const float *din, double *dout, const int n, const float *dmax)
{
    constexpr int warpNum = blockSize / warpSize;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int laneId = tid % warpSize;
    int warpId = tid / warpSize;

    __shared__ float shm[warpNum];
    float val = 0.0f;
    for (int i = gtid;i < n;i+=gridDim.x * blockDim.x) {
        val += expf(din[i] - *dmax);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (laneId == 0) {
        shm[warpId] = val;
    }

    __syncthreads();
    if (warpId == 0) {
        val = laneId < warpNum ? shm[laneId] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (laneId == 0) {
            atomicAdd(dout, val);
        }
    }
}

__global__ void softmax_v1(const float *din, const double *dsum, const float *dmax, float *dout, const int n)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // int tid = threadIdx.x;
    for (int offset = gtid; offset < n; offset += blockDim.x * gridDim.x) {
        dout[offset] = expf(din[offset] - *dmax) / *dsum;
    }
}

void call_softmax_v1()
{
    constexpr int N = 1000000;
    constexpr int BLOCK_SIZE = 256;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    constexpr int WARP_SIZE = 32;
    const int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    float *hin;
    float *din, *dout;
    hin = (float*)malloc(N * sizeof(float));
    cudaCheck(cudaMalloc((void**)&din, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dout, N * sizeof(float)));
    for (int i = 0; i < N; ++i) {
        hin[i] = i % BLOCK_SIZE;
    }
    cudaCheck(cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice));

    float *dmax;
    cudaCheck(cudaMalloc((void**)&dmax, sizeof(float)));
    float init_max = -FLT_MAX;
    cudaCheck(cudaMemcpy(dmax, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    
    double *dsum;
    cudaCheck(cudaMalloc((void**)&dsum, sizeof(double)));
    cudaCheck(cudaMemset(dsum, 0, sizeof(double)));
    max_kernel<BLOCK_SIZE, WARP_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(din, dmax, N);
    cudaCheck(cudaGetLastError());
    sum_kernel<BLOCK_SIZE, WARP_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(din, dsum, N, dmax);
    cudaCheck(cudaGetLastError());
    softmax_v1<<<GRID_SIZE, BLOCK_SIZE>>>(din, dsum, dmax, dout, N);
    cudaCheck(cudaGetLastError());

    float *dout_cpu = (float*)malloc(N *sizeof(float));
    cudaCheck(cudaMemcpy(dout_cpu, dout, N * sizeof(float), cudaMemcpyDeviceToHost));

    float *hout_cpu = (float*)malloc(N * sizeof(float));
    softmax_cpu(hin, hout_cpu, N);
    if (check_result(hout_cpu, dout_cpu, N)) {
        printf("test passed!\n");
    } else {
        printf("test failed!\n");
    }

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dmax);
    cudaFree(dsum);

    free(hin);
    free(hout_cpu);
    free(dout_cpu);
}

int main()
{
    call_softmax_v1();
    return 0;
}