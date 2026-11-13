#include <iostream>
#include <cuda_runtime.h>

#define CEIL(a,b) ((a) + (b) - 1) / (b)

void gemm_cpu(const float* ha, const float* hb, float* hc, const int m, const int n, const int k)
{
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for(int t = 0;t < k;++t) {
                sum += ha[i * k + t] * hb[t * n + j];
            }
            hc[i * n + j] = sum;
        }
    }
}

__device__ __forceinline__
void cp_async_4(void* dst_shared, const void* src_shared)
{
    unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(dst_shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                :
                : "r"(dst), "l"(src_shared)
    );
}

__device__ __forceinline__
void cp_async_16(void* dst_shared, const void* src_shared)
{
    unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(dst_shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :
                : "r"(dst),"l"(src_shared)
    );
}

__device__ __forceinline__
void cp_async_commit()
{
    asm volatile(
        "cp.async.commit_group;\n"
        :
        :
    );
}

__device__ __forceinline__
void cp_async_wait_all()
{
    asm volatile(
        "cp.async.wait_all;\n"
        :
        :
    );
}

template<int blockSize>
__global__ void gemm_v6_async(const float* da, const float* db, float* dc, const int m, const int n, const int k)
{
    constexpr int thread_tile = 4;
    constexpr int block_m = blockSize;
    constexpr int block_k = blockSize;
    constexpr int block_n = blockSize * thread_tile;

    int col_base = blockIdx.x * block_n + threadIdx.x * thread_tile;
    int row_base = blockIdx.y * block_m + threadIdx.y;

    __shared__ float AS[2][block_m][block_k];
    __shared__ float4 BS[2][block_k][block_k];

    int num_tile = CEIL(k, block_k);

    float sum[4] = {0.0f};

    //1. first tile0-buffer0
    int first_col_a = threadIdx.x;
    int first_row_b = threadIdx.y;

    //da[0]->AS[0]
    cp_async_4(&AS[0][threadIdx.y][threadIdx.x], &da[row_base * k + first_col_a]);
    //db[0]->BS[0]
    size_t first_b_offset = static_cast<size_t>(first_row_b) * n + col_base;
    cp_async_16(&BS[0][threadIdx.y][threadIdx.x], db + first_b_offset);
    cp_async_commit(); //A+B归为一组

    cp_async_wait_all();
    __syncthreads();

    //2. tile循环
    for (int tile = 0; tile < num_tile; ++tile) {
        int read_index = tile & 1;
        int write_index = read_index ^ 1;

        //2.1 先cp async
        int next_tile = tile + 1;
        if (next_tile < num_tile) {
            //next A
            int next_col_a = next_tile * block_k + threadIdx.x;
            cp_async_4(&AS[write_index][threadIdx.y][threadIdx.x], &da[row_base * k + next_col_a]);

            //next B

            int next_row_b = next_tile * block_k + threadIdx.y;
            size_t next_b_offset = static_cast<size_t>(next_row_b) * n + col_base;
            cp_async_16(&BS[write_index][threadIdx.y][threadIdx.x], db + next_b_offset);

            cp_async_commit();
        }

        //2.2 compute buffer 0
        #pragma unroll
        for (int p = 0; p < block_k; ++p) {
            float a = AS[read_index][threadIdx.y][p];

            float4 b = BS[read_index][p][threadIdx.x];

            sum[0] += a * b.x;
            sum[1] += a * b.y;
            sum[2] += a * b.z;
            sum[3] += a * b.w;
        }

        //2.3 当前tile算完，下一轮要用next tile的，所以要wait
        if (next_tile < num_tile) {
            cp_async_wait_all();// 等当前thread提交的异步copy完成
            // 确保整个block所有thread
            // 都已经完成next tile搬运
            __syncthreads();
        }
    }

    //3 write to dc
    if (row_base < m && col_base < n) {
        size_t offset = static_cast<size_t>(row_base) * n + col_base;
        *reinterpret_cast<float4*>(dc + offset) = make_float4(sum[0], sum[1], sum[2], sum[3]);
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
    constexpr int m = 4096;
    constexpr int n = 4096;
    constexpr int k = 4096;
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
    gemm_v6_async<blockSize><<<grid,block>>>(da,db,dc,m,n,k);
    cudaMemcpy(hc, dc, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    if (check_result(hc_cpu, hc, m * n)) {
        printf("test passed!\n");
    } else {
        printf("test failed!\n");
    }

    auto gemm_v6_kernel = [&]() {
        gemm_v6_async<blockSize><<<grid,block>>>(da,db, dc,m,n,k);
    };

    float time = benchmark_kernel(gemm_v6_kernel, 20, 5);

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