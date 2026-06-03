#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>

#define CEIL(a,b) (a+b-1)/b
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)

void __cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("cuda error(%s), in file(%s), on line(%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

void softmax_cpu(const float *hin, float *hout, const int n)
{
    //max
    float max_val = hin[0];
    for (int i = 1; i < n;++i) {
        max_val = max_val > hin[i] ? max_val : hin[i];
    }

    //exp
    double sum = 0.0;
    for (int i = 0; i < n; i ++) {
        float temp = exp(hin[i] - max_val);
        hout[i] = temp;
        sum += temp;
    }

    for (int i = 0; i < n; ++i) {
        hout[i] /= sum;
    }
}

__device__ float atomicMax(float *addr, const float val)
{
    int *addr_as_int = reinterpret_cast<int*>(addr);
    int old_val = *addr_as_int;
    int expected = 0;
    do {
        expected = old_val;
        old_val = atomicCAS(addr_as_int, expected, __float_as_int(fmaxf(val, __int_as_float(expected))));
    } while(expected != old_val);
    return __int_as_float(old_val);
}

template<int blockSize,int warpSize>
__global__ void max_kernel(const float* din, float *dout, const int n)
{
    constexpr int WARP_NUM = blockSize / warpSize;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    __shared__ float shm[WARP_NUM];

    float val = -FLT_MAX;
    for (int i = gtid;i < n;i+=gridDim.x * blockDim.x) {
        val = fmaxf(val, din[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (laneId == 0) { // 每个warp的第一个线程
        shm[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) { //第一个warp
        val = laneId < WARP_NUM ? shm[laneId] : -FLT_MAX;
        for (int offset = WARP_NUM; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val , offset));
        }
        if (laneId == 0) {
            atomicMax(dout, val);
        }
    }   
}


template<int blockSize, int warpSize>
__global__ void sum_kernel(const float* din, float *dout, const int n, const float *dmax,double* dsum)
{
    constexpr int WARP_NUM = blockSize / warpSize;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    __shared__ float shm[WARP_NUM];
    float val = 0.0f;
    for (int i = gtid; i < n; i+=gridDim.x * blockDim.x) {
        float temp = expf(din[i] - *dmax);
        dout[i] = temp;
        val += temp;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (laneId == 0) {
        shm[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        val = laneId < WARP_NUM ? shm[laneId] : 0.0f;
        for (int offset = WARP_NUM; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (laneId == 0) {
            atomicAdd(dsum, val);
        }
    }
}

__global__ void softmax_v2(float* dout, const double *dsum, const int n)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    float inv_sum = 1.0f / static_cast<float>(*dsum);
    for (int i = gtid; i < n; i += gridDim.x * blockDim.x) {
        dout[i] *= inv_sum;
    }
}

bool check_result(const float *hin, const float *hout, const int n)
{
    constexpr float rtol = 1e-5;
    constexpr float atol = 1e-5;
    for (int i = 0; i < n; ++i) {
        float abs_diff = std::fabs(hin[i] - hout[i]);
        float scale = std::max(std::fabs(hin[i]), std::fabs(hout[i]));
        if (abs_diff > atol + rtol * scale) {
            printf("res error, i = %d, hin[i]=%f, hout[i]=%f\n", i, hin[i], hout[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int N = 1000000;
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE = 256;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    float *hin, *hout;
    float *din, *dout, *dmax;
    double *dsum;
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(N * sizeof(float));
    cudaCheck(cudaMalloc((void**)&din, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dout, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dmax, sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dsum, sizeof(double)));
    cudaCheck(cudaMemset(dsum, 0, sizeof(double)));
    float init_max = -FLT_MAX;
    cudaCheck(cudaMemcpy(dmax, &init_max, sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i< N;++i) {
        hin[i] = i % BLOCK_SIZE;
    }

    cudaCheck(cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice));

    max_kernel<BLOCK_SIZE, WARP_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(din, dmax, N);
    cudaCheck(cudaGetLastError());
    sum_kernel<BLOCK_SIZE, WARP_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(din, dout, N, dmax, dsum);
    cudaCheck(cudaGetLastError());
    softmax_v2<<<GRID_SIZE, BLOCK_SIZE>>>(dout, dsum, N);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMemcpy(hout, dout, N *sizeof(float), cudaMemcpyDeviceToHost));

    float *hout_cpu = (float*)malloc(N * sizeof(float));
    memset(hout_cpu, 0, N * sizeof(float));

    softmax_cpu(hin, hout_cpu, N);

    if (check_result(hout_cpu, hout, N)) {
        printf("test passed!\n");
    } else {
        printf("test failed\n");
    }

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dmax);
    cudaFree(dsum);
    free(hin);
    free(hout);
    free(hout_cpu);
    return 0;
}