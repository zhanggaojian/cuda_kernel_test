#include<cuda_runtime.h>
#include <iostream>

#define CEIL(a,b) (a+b-1) / b

void histogram_cpu(int *ha, int *hb, int n)
{
    for (int i = 0;i < n; ++i) {
        hb[ha[i]]++;
    }
}

template<int blockSize>
__global__ void histogram_v2(int *da, int *db, int n)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x; //只是当前block内的线程id
    __shared__ int shm[blockSize]; //每个block分配一个shared memory
    for (int i = tid; i < blockSize; i += blockDim.x) {
        shm[i] = 0;
    }
    shm[tid] = 0; // 只是针对当前block内的线程，去初始化对应的shared memory
    __syncthreads(); //同步当前block内的所有线程
    for (int i = gtid; i < n; i += blockDim.x * gridDim.x) {
        int val = da[i]; // 从global memory中取出当前线程gtid对应的元素值
        atomicAdd(&shm[val], 1); //shm[tid]是当前block内的线程对应的元素，给它加1
    }
    __syncthreads(); //同步当前block内的所有线程的计算结果

    //需要把结果写回到global memory
    //为什么是写回到db[tid]，因为最终每个bin就对应到每个block中的tid里面去
    //不同block中相同tid对应的bin值也是一样的
    //因此需要把不同的tid对应的bin值得结果加起来，存入到显存中去
    for (int i = tid; i < blockSize; i += blockDim.x) {
        atomicAdd(&db[i], shm[i]);
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
    constexpr int N = 10000000;
    constexpr int BLOCK_SIZE = 256;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int GRID_SIZE = std::min(CEIL(N, BLOCK_SIZE), deviceProp.maxGridSize[0]);
    int *ha, *hb;
    int *da, *db;
    ha = (int*)malloc(N * sizeof(int));
    hb = (int*)malloc(N * sizeof(int));
    cudaMalloc((void**)&da, N * sizeof(int));
    cudaMalloc((void**)&db, N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        ha[i] = rand() % BLOCK_SIZE;
    }
    histogram_cpu(ha, hb, N);
    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histogram_v2<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(da, db, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time %f\n", time);
    int *db_cpu = (int *)malloc(N * sizeof(int));
    cudaMemcpy(db_cpu, db, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (check_result(hb, db_cpu, N)) {
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