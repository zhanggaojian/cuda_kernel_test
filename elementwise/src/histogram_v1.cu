/*直方图中有若干个bin，纵坐标代表每个bin出现的频率，横坐标代表每个bin的值
这个算子就是给定一个数组，数组的每个值就代表bin，统计每个bin出现的频率*/
#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a,b) (a+b-1)/b

void histogram_cpu(int *ha, int *hb, int n)
{
    for (int i=0;i<n;++i)
    {
        hb[ha[i]]++; //ha[i]是数组的值，也就是bin的值; hb[ha[i]]是统计每个bin出现的频率
    }
}

__global__ void histogram_v1(int *da, int *db, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&db[da[i]], 1);
    }

}

bool check_result(int *hb, int *db_cpu, int n)
{
    for (int i = 0; i < n; ++i) {
        if (hb[i] != db_cpu[i]) {
            printf("check result failed at index %d, hb[%d] = %d, db_cpu[%d] = %d\n", i, i, hb[i], i, db_cpu[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int N = 1000000;
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    int *ha = (int*)malloc(N * sizeof(int));
    int *hb = (int*)malloc(N * sizeof(int));
    int *da;
    cudaMalloc((void**)&da, N * sizeof(int));
    int *db;
    cudaMalloc((void**)&db, N *sizeof(int));
    //init data
    for (int i = 0; i < N; ++i) {
        ha[i] = rand() % N;
    }
    cudaMemcpy(da, ha, N *sizeof(int), cudaMemcpyHostToDevice);
    histogram_cpu(ha, hb, N);
    histogram_v1<<<GRID_SIZE, BLOCK_SIZE>>>(da, db, N);
    int *db_cpu = (int*)malloc(N * sizeof(int));
    cudaMemcpy(db_cpu, db, N * sizeof(int), cudaMemcpyDeviceToHost);
    if(check_result(hb, db_cpu, N)) {
        printf("check result success\n");
    } else {
        printf("check result failed\n");
    }
    cudaFree(da);
    cudaFree(db);
    free(ha);
    free(hb);
    free(db_cpu);
    return 0;
}