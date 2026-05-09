#include<cuda_runtime.h>
#include<iostream>
#include <cmath>

#define CEIL(a,b) (a+b-1)/b

void softmax_cpu(const float *hin, float *hout, int n)
{
    if (n <= 0) return;
    // find max
    float maxval = hin[0];

    for (int i = 1; i < n; ++i) {
        maxval = maxval > hin[i] ? maxval : hin[i];
    }

    // // 减去最大值
    // for (int i = 0;i < n; ++i) {
    //     hin[i] -= maxval;
    // }

    //求和e^i
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        hout[i] = expf(hin[i] - maxval);
        sum += hout[i];
    }

    //最终结果
    for (int i = 0; i < n; ++i) {
        hout[i] /= sum;
    }
}

bool check_result(const float *hin, const float *din, int n)
{
    float atol = 1e-6;
    float rtol = 1e-6;
    for (int i = 0;i < n; ++i) {
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

//不减去最大值
void call_softmax_v0()
{
    
}

int main()
{
    return 0;
}