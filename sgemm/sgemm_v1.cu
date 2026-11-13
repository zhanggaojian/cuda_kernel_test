#include <iostream>
#include <cuda_runtime.h>
#include <limits.h>

#define CEIL(a,b) (a+b-1)/b
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)
void __cudaCheck(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess) {
        printf("cuda error(%s), in file(%s), on line(%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

bool check_result(const float* cpu, const float* gpu, const int size)
{
    constexpr float eps = 1e-3f;
    for (int i = 0;i < size; ++i) {
        float diff = std::fabs(cpu[i] - gpu[i]);
        if (diff > eps) {
            printf("error");
            return false;
        }
    }
    return true;
}

//cpu
void gemm_cpu(const float* A, const float* B, float* C, const int m, const int n, const int k)
{

    for(int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float temp = 0.0f;
            for(int t = 0;t < k; ++t) {
                temp += A[i * k + t] * B[t * n + j];
            }
            C[i * n + j] = temp;
        }
    }
}

//A * B = C
//A[m,k]
//B[k,n]
//C[m,n]
__global__ void sgemm_v1(const float *A, const float *B, float *C, const int m, const int n, const int k)
{
    //C[M,N]
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float temp = 0.0;
    for (int i = 0;i < k; ++i) {
        temp += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = temp;
}

template<typename T>
float benchmark_kernel(T func, int repeat, int warmup = 1)
{
    float time = 0.0f;
    if (repeat <= 0) {
        return time;
    }

    for (int i = 0 ;i < warmup; ++i) {
        func();
    }

    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        func();
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time/repeat;
}

int main()
{
    constexpr int m = 1024;
    constexpr int n = 1024;
    constexpr int k = 1024;
    constexpr int blockSize = 16;
    cudaSetDevice(0);


    //host memory
    float* ha;
    float* hb;
    float* hc;
    float* hc_cpu;

    ha = (float*)malloc(m * k *sizeof(float));
    hb = (float*)malloc(k * n * sizeof(float));
    hc = (float*)malloc(m * n * sizeof(float));
    hc_cpu = (float*)malloc(m * n * sizeof(float));

    for (int i = 0 ; i < m * k; ++i) {
        ha[i] = i % 10;
    }

    for (int j = 0; j< k*n;++j) {
        hb[j] = j % 10;
    }

    //gpu memory
    float* da;
    float* db;
    float* dc;

    cudaCheck(cudaMalloc((void**)&da, m * k * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&db, k * n * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dc, m * n * sizeof(float)));
    cudaCheck(cudaMemcpy(da, ha, m * k * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(db, hb, k * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(blockSize, blockSize);
    dim3 grid(CEIL(n, blockSize), CEIL(m, blockSize));
    sgemm_v1<<<grid, block>>>(da, db, dc, m, n, k);
    cudaDeviceSynchronize();
    cudaMemcpy(hc, dc, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    gemm_cpu(ha, hb, hc_cpu, m, n, k);


    if (check_result(hc_cpu, hc, m * n)) {
        printf("test passed!\n");
    } else {
        printf("test failed!\n");
    }
    auto gemm_v1_kernel = [&](){
        sgemm_v1<<<grid, block>>>(da, db, dc, m, n, k);
    };

    float time = benchmark_kernel(gemm_v1_kernel, 20, 5);

    //性能计算
    // gemm :
    //每个元素需要k次乘法和k次加法
    // flops = 2 * m * n * k
    //GFLOPS = flops / time;
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);

    double gflops = flops / time;
    printf("M = %d, N = %d, K = %d\n",
        m, n, k);

    printf("block = (%d, %d)\n",
            blockSize,
            blockSize);

    printf("grid = (%d, %d)\n",
            grid.x,
            grid.y);

    printf("time = %f ms\n", time);

    printf("performance = %f GFLOPS\n",
            gflops);


        // --------------------------------------------------------
    // free
    // --------------------------------------------------------

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(ha);
    free(hb);
    free(hc);
    free(hc_cpu);
}