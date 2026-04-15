#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a, b) (a + b -1) / b

#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__) //定义成宏的目的是每次宏展开的时候，会把__FILE__和__LINE__替换成实际的文件名和行号
void _cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("cuda error %s in file %s on line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}

__global__ void vector_add(float* a, float* b, float* c,  int n)
{
    //全局thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add_cpu(float* a, float* b, float* c,  int n)
{
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    float *ha, *hb, *hc;
    float *da, *db, *dc;
    constexpr int N = 10000000; //总共有N个元素值
    constexpr int BLOCK_SIZE = 256; //thread nums every block
    constexpr int GRID_SIZE = CEIL(N, BLOCK_SIZE); // block nums

    //给host分配内存
    ha = (float*)malloc(N * sizeof(float));
    hb = (float*)malloc(N * sizeof(float));
    hc = (float*)malloc(N * sizeof(float));

    //给device分配内存
    cudaCheck(cudaMalloc((void**)&da, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&db, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dc, N * sizeof(float)));

    //初始化host数据
    for (int i = 0; i < N; ++i) {
        ha[i] = i;
        hb[i] = i + 1;
    }

    //将host数据拷贝给device数据
    cudaCheck(cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice));
    //调用kernel函数
    vector_add<<<GRID_SIZE, BLOCK_SIZE>>>(da, db, dc, N);

    //最终把device的结果拷贝回host，然后在host上比较数据大小
    cudaCheck(cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost));

    float* hc_cpu = (float*)malloc(N * sizeof(float));

    vector_add_cpu(ha, hb, hc_cpu, N);

    for (int i = 0; i < N; ++i) {
        if (abs(hc_cpu[i] - hc[i]) > 1e-6) {
            printf("error at index %d\n", i);
            break;
        }
    }

    printf("test passed\n");

    free(ha);
    free(hb);
    free(hc);
    free(hc_cpu);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}