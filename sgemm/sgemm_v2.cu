#include <iostream>
#include <cuda_runtime.h>

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

void gemm_cpu(const float* ha, const float* hb, float* hc, const int m, const int n, const int k)
{
    if (!ha || !hb || !hc) return;
    for (int i = 0;i < m;++i) {
        for (int j = 0;j < n;++j) {
            float temp = 0.0f;
            for (int t = 0; t < k;++t) {
                temp += ha[i * k + t] * hb[t * n + j];
            }
            hc[i * n + j] = temp;
        }
    }
}

//matrix shape = [M, N]
// M代表的是矩阵的行，对应到二维坐标系中的y轴
// N代表的是矩阵的列，对应到二维坐标系中的x轴
template<int blockSize>
__global__ void gemm_v2(const float *da, const float *db, float *dc, const int m, const int n, const int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float AS[blockSize][blockSize];
    __shared__ float BS[blockSize][blockSize];

    float sum = 0.0f;

    int num_tile = CEIL(k, blockSize); //这里是除法，CELI(a,b)=a/b，这里计算的是tile的个数
    for (int i = 0; i < num_tile; ++i) {
        int a_col = i * blockSize + threadIdx.x;
        //da[row][a_col] = AS
        if (row < m && a_col < k) {
            AS[threadIdx.y][threadIdx.x] = da[row * k + a_col];
        } else {
            AS[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = i * blockSize + threadIdx.y;
        if (b_row < k && col < n) {
            BS[threadIdx.y][threadIdx.x] = db[b_row * n + col];
        } else {
            BS[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int j = 0;j < blockSize; ++j) {
            sum += AS[threadIdx.y][j] * BS[j][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < m && col < n)
        dc[row * n + col] = sum;
}

bool check_result(const float* cpu, const float* gpu, const int size)
{
    const double eps = 1e-3f;
    for (int i = 0; i < size; ++i) {
        double diff = std::fabs(cpu[i] - gpu[i]);
        if (diff > eps) {
            printf("error, cpu[%d] = %f, gpu[%d] = %f\n", i, cpu[i], i, gpu[i]);
            return false;
        }
    }
    printf("pass!\n");
    return true;
}

template<typename T>
float benchmark_kernel(T func, const int repeat, const int warmup=1)
{
    float time = 0.0;
    if (repeat <= 0) return time;
    for (int i = 0 ;i < warmup; ++i) {
        func();
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0;i < repeat; ++i) {
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

    const int blockSize = 16;

    float *ha, *hb, *hc, *hc_cpu;
    float *da, *db, *dc;
    ha = (float*)malloc(m * k * sizeof(float));
    hb = (float*)malloc(k * n * sizeof(float));
    hc = (float*)malloc(m * n * sizeof(float));
    hc_cpu = (float*)malloc(m * n * sizeof(float));

    cudaMalloc((void**)&da, m * k * sizeof(float));
    cudaMalloc((void**)&db, k * n * sizeof(float));
    cudaMalloc((void**)&dc, m * n * sizeof(float));

    for (int i = 0; i < m * k ;++i) {
        ha[i] = i % 10;
    }

    for (int i = 0;i < k*n ; ++i) {
        hb[i] = i % 10;
    }

    cudaMemcpy(da, ha, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, k * n * sizeof(float), cudaMemcpyHostToDevice);

    gemm_cpu(ha,hb,hc_cpu,m,n,k);

    dim3 block(blockSize, blockSize);
    dim3 grid(CEIL(n, blockSize), CEIL(m, blockSize));
    gemm_v2<blockSize><<<grid,block>>>(da,db,dc,m,n,k);
    cudaMemcpy(hc, dc, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    if (check_result(hc_cpu, hc, m * n)) {
        printf("test passed!\n");
    } else {
        printf("test failed!\n");
    }

    auto gemm_v2_kernel = [&]() {
        gemm_v2<blockSize><<<grid,block>>>(da,db, dc,m,n,k);
    };

    float time = benchmark_kernel(gemm_v2_kernel, 20, 5);

    double flops = 2.0 * static_cast<double>(m) * k * n;

    double gflops = flops/time;

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
    return 0;
}