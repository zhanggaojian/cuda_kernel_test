#include<cuda_runtime.h>
#include<iostream>
#include <cmath>

#define CEIL(a,b) (a+b-1)/b
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)

void __cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("cuda error(%s) in file(%s) on line(%d)", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

// void softmax_cpu(const float *hin, float *hout, const int n)
// {
//     if (n <= 0) return;
//     // find max
//     float maxval = hin[0];

//     for (int i = 1; i < n; ++i) {
//         maxval = maxval > hin[i] ? maxval : hin[i];
//     }

//     // // 减去最大值
//     // for (int i = 0;i < n; ++i) {
//     //     hin[i] -= maxval;
//     // }

//     //求和e^i
//     float sum = 0.0f;
//     for (int i = 0; i < n; ++i) {
//         hout[i] = expf(hin[i] - maxval);
//         sum += hout[i];
//     }

//     //最终结果
//     for (int i = 0; i < n; ++i) {
//         hout[i] /= sum;
//     }
// }

//不减去最大值
void softmax_cpu(const float *hin, float *hout ,const int n)
{
    if (n <= 0) return;
    double sum = 0.0;
    for (int i = 0;i < n;++i) {
        hout[i] = expf(hin[i]);
        sum += hout[i];
    }

    for (int i = 0;i < n;++i)
        hout[i] /= sum;
}

bool check_result(const float *hin, const float *din, const int n)
{
    float atol = 1e-6;
    float rtol = 1e-6;
    for (int i = 0;i < n; ++i) {
        if (!std::isfinite(hin[i]) || !std::isfinite(din[i])) {
            printf("input is nan, %d, hin[i]=%f, din[i]=%f\n", i, hin[i], din[i]);
            return false;
        }
        float abs_diff = fabsf(hin[i] - din[i]);
        //rtol乘的是scale，也就是找出两个比较的值的绝对值的最大值，因为比较的是量级
        float abs_diff_tol = atol + rtol * std::max(fabsf(hin[i]), fabsf(din[i]));
        if (abs_diff >= abs_diff_tol) {
            printf("result err, i=%d, hin[i]=%f, din[i]=%f\n", i, hin[i], din[i]);
            return false;
        }
    }
    return true;
}

template<typename T>
float benchmark_kernel(T func, int repeats, int warmup=1)
{
    float time = 0.0f;
    if (repeats <= 0) return time;
    for (int i = 0;i < warmup;++i) {
        func();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0;i < repeats;++i) {
        func();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time / repeats;
}

template<int blockSize, int warpSize>
__global__ void softmax_sum(const float* hin, double* hout, const int n)
{
    constexpr int warpNum = blockSize / warpSize;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    __shared__ float shm[warpNum];
    float val = gtid < n ? expf(hin[gtid]) : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (laneId == 0) {
        shm[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        val = tid < warpNum ? shm[tid] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (laneId == 0) {
            atomicAdd(hout, val);
        }
    }
}

__global__ void softmax_v0(const float* din, const double *sum, float *dout, const int n)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // int tid = threadIdx.x;
    for (int i = gtid; i < n; i+=gridDim.x * blockDim.x) {
        dout[i] = expf(din[i]) / *sum;
    }
}

//不减去最大值
template<int blockSize, int warpSize>
void call_softmax_v0(const float *hin, float* hout, const int n, const int gridSize)
{
    float *din, *dout;
    double *dsum;
    cudaCheck(cudaMalloc((void**)&din, n * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dout, n * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dsum, 1 * sizeof(double)));
    cudaCheck(cudaMemcpy(din, hin, n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dsum, 0, sizeof(double)));

    softmax_sum<blockSize, warpSize><<<gridSize, blockSize>>>(din, dsum, n);
    cudaCheck(cudaGetLastError());
    softmax_v0<<<gridSize, blockSize>>>(din, dsum, dout, n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(hout, dout, n * sizeof(float), cudaMemcpyDeviceToHost));

    float *dout_cpu = (float*)malloc(n * sizeof(float));
    softmax_cpu(hin, dout_cpu, n);
    if (check_result(dout_cpu, hout, n)) {
        printf("test passed!\n");
    } else {
        printf("test failed!\n");
    }

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dsum);
    free(dout_cpu);

}

int main()
{
    constexpr int N = 1000000;
    constexpr int BLOCK_SIZE = 256;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    constexpr int WARP_SIZE = 32;
    const int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);

    float *hin, *hout;
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(N * sizeof(float));
    for (int i = 0;i < N; ++i) {
        hin[i] = i % BLOCK_SIZE;
    }


    call_softmax_v0<BLOCK_SIZE, WARP_SIZE>(hin, hout, N, GRID_SIZE);

    free(hin);
    free(hout);
    return 0;
}