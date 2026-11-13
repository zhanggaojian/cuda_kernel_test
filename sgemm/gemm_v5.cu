#include <iostream>
#include <cuda_runtime.h>

#define CEIL(a,b) ((a) + (b) - 1) /(b)

void gemm_cpu(const float* ha, const float* hb, float* hc, const int m,const int n, const int k)
{
    for(int i = 0;i < m;++i) {
        for(int j = 0;j < n;++j) {
            float sum = 0.0f;
            for(int t = 0;t < k;++t) {
                sum += ha[i * k + t] * hb[t * n + j];
            }
            hc[i * n + j] = sum;
        }
    }
}

template<int blockSize>
__global__ void gemm_v5(const float* da, const float* db, float* dc, const int m, const int n, const int k)
{
    constexpr int thread_tile = 4;
    constexpr int block_m = blockSize;
    constexpr int block_k = blockSize;
    constexpr int block_n = blockSize * thread_tile;

    int col_base = blockIdx.x * block_n + threadIdx.x * thread_tile;
    int row_base = blockIdx.y * block_m + threadIdx.y;

    __shared__ float AS[2][block_m][block_k];
    __shared__ float4 BS[2][block_k][block_k]; //因为每个读4个float，所以这里是block_k

    float sum[4] = {0.0f};

    const bool aligned = (n % 4) == 0;

    int num_tile = CEIL(k, block_k);

    //AS[0],BS[0] - buffer 0
    //AS[1],BS[1] - buffer 1
    
    //preload tile0->buffer0
    int first_col_a = threadIdx.x;
    int first_row_b = threadIdx.y;

    if (row_base < m && first_col_a < k) {
        AS[0][threadIdx.y][threadIdx.x] = da[row_base * k + first_col_a];
    } else {
        AS[0][threadIdx.y][threadIdx.x] = 0.0f;
    }

    float4 first_b4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (first_row_b < k && col_base < n) {
        const size_t offset = static_cast<size_t>(first_row_b) * n + col_base;
        if (aligned && col_base + 3 < n) {
            first_b4 = *reinterpret_cast<const float4*>(db + offset);
        } else {
            first_b4.x = db[offset];
            if (col_base + 1 < n) first_b4.y = db[offset + 1];
            if (col_base + 2 < n) first_b4.z = db[offset + 2];
            if (col_base + 3 < n) first_b4.w = db[offset + 3];
        }
    }
    BS[0][threadIdx.y][threadIdx.x] = first_b4;
    __syncthreads();

    //2. k tile循环
    for (int tile = 0; tile < num_tile; ++tile) {
        int read_index = tile & 1; //当前从哪个buffer读数据
        int write_index = read_index ^ 1; //当前写数据到哪个buffer
        
        //当前tile计算
        #pragma unroll
        for (int p = 0; p < block_k; ++p) {
            float a = AS[read_index][threadIdx.y][p];
            float4 temp_b = BS[read_index][p][threadIdx.x];

            sum[0] += a * temp_b.x;
            sum[1] += a * temp_b.y;
            sum[2] += a * temp_b.z;
            sum[3] += a * temp_b.w;
        }

        //准备下一个tile
        int next_tile = tile + 1;
        if (next_tile < num_tile) {
            //A next tile
            int next_col_a = next_tile * block_k + threadIdx.x;
            if (row_base < m && next_col_a < k) {
                AS[write_index][threadIdx.y][threadIdx.x] = da[row_base * k + next_col_a];
            } else {
                AS[write_index][threadIdx.y][threadIdx.x] = 0.0f;
            }

            //B next tile
            int next_row_b = next_tile * block_k + threadIdx.y;
            float4 next_b4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (next_row_b < k && col_base < n) {
                const size_t offset = static_cast<size_t>(next_row_b) * n + col_base;
                if (aligned && col_base + 3 < n) {
                    next_b4 = *reinterpret_cast<const float4*>(db + offset);
                } else {
                    next_b4.x = db[offset];
                    if (col_base + 1 < n) next_b4.y = db[offset + 1];
                    if (col_base + 2 < n) next_b4.z = db[offset + 2];
                    if (col_base + 3 < n) next_b4.w = db[offset + 3];
                }
            }
            BS[write_index][threadIdx.y][threadIdx.x] = next_b4;
            __syncthreads();
        }
    }

    //写回到dc
    if (row_base < m && col_base < n) {
        size_t offset = static_cast<size_t>(row_base) * n + col_base;
        if (aligned && col_base + 3 < n) {
            *reinterpret_cast<float4*>(dc + offset) = make_float4(sum[0], sum[1], sum[2], sum[3]);
        } else {
            dc[offset] = sum[0];
            if (col_base + 1 < n) dc[offset + 1] = sum[1];
            if (col_base + 2 < n) dc[offset + 2] = sum[2];
            if (col_base + 3 < n) dc[offset + 3] = sum[3];
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
    gemm_v5<blockSize><<<grid,block>>>(da,db,dc,m,n,k);
    cudaMemcpy(hc, dc, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    if (check_result(hc_cpu, hc, m * n)) {
        printf("test passed!\n");
    } else {
        printf("test failed!\n");
    }

    auto gemm_v5_kernel = [&]() {
        gemm_v5<blockSize><<<grid,block>>>(da,db, dc,m,n,k);
    };

    float time = benchmark_kernel(gemm_v5_kernel, 20, 5);

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