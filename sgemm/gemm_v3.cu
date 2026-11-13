#include <iostream>
#include <cuda_runtime.h>

#define CEIL(a,b) ((a)+(b)-1)/(b)

void gemm_cpu(const float* ha, const float* hb, float* hc, const int m, const int n,const int k)
{
    if (!ha || !hb || !hc) return;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += ha[i * k + t] * hb[t * n + j];
            }
            hc[i * n + j] = sum;
        }
    }
}

template<int blockSize,int thread_tile>
__global__ void gemm_v3(const float* da, const float* db, float* dc, const int m, const int n, const int k)
{
    constexpr int block_n = blockSize * thread_tile;
    int col = blockIdx.x * block_n + threadIdx.x * thread_tile;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float AS[blockSize][blockSize];
    __shared__ float BS[blockSize][block_n];

    float sum[thread_tile] = {0.0f};
    int num_tile = CEIL(k, blockSize);

    
    
    for (int i = 0;i < num_tile; ++i) {
        int a_col = i * blockSize + threadIdx.x;
        int b_row = i * blockSize + threadIdx.y;

        if (a_col < k && row < m) {
            AS[threadIdx.y][threadIdx.x] = da[row * k + a_col];
        } else {
            AS[threadIdx.y][threadIdx.x] = 0.0f;
        }

        #pragma unroll
        for (int j = 0;j < thread_tile; ++j) {
            int local_col = threadIdx.x * thread_tile + j;
            int global_col = blockIdx.x * block_n + local_col;
            if (b_row < k && global_col < n) {
                BS[threadIdx.y][local_col] = db[b_row * n + global_col];
            } else {
                BS[threadIdx.y][local_col] = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int p = 0; p < blockSize; ++p) {
            float a = AS[threadIdx.y][p];

            #pragma unroll
            for (int t = 0; t < thread_tile; ++t) {
                int b_col = threadIdx.x * thread_tile + t;
                sum[t] += a * BS[p][b_col];
            }
        }
        __syncthreads();
    }

    if (row < m) {
        #pragma unroll
        for (int yy = 0; yy < thread_tile; ++yy) {
            int c_col = col + yy;
            if (c_col < n) {
                dc[row * n + c_col] = sum[yy];
            }
        }
    }
}

bool check_result(const float* cpu, const float* gpu, const int size)
{
    float eps = 1e-3f;
    for(int i = 0; i < size; ++i) {
        double diff = std::fabs(cpu[i] - gpu[i]);
        if (diff > eps) {
            printf("error, i = %d, cpu[i] = %f, gpu[i]=%f\n", i, cpu[i], gpu[i]);
            return false;
        }
    }
    printf("test passed!\n");
    return true;
}

template<typename T>
float benchmark_kernel(T func, const int repeat=10, const int warmup=1)
{

    float time = 0.0f;
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0;i < repeat; ++i) {
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
    constexpr int thread_tile = 4;

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
    dim3 grid(CEIL(n, blockSize * thread_tile), CEIL(m, blockSize));
    gemm_v3<blockSize, thread_tile><<<grid,block>>>(da,db,dc,m,n,k);
    cudaMemcpy(hc, dc, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    if (check_result(hc_cpu, hc, m * n)) {
        printf("test passed!\n");
    } else {
        printf("test failed!\n");
    }

    auto gemm_v3_kernel = [&]() {
        gemm_v3<blockSize, thread_tile><<<grid,block>>>(da,db, dc,m,n,k);
    };

    float time = benchmark_kernel(gemm_v3_kernel, 20, 5);

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