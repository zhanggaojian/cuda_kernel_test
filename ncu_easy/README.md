# Nsight Compute Profiling 使用说明

## Python 版本

```bash
chmod +x ncu_profile.py
./ncu_profile.py ./my_cuda_app -k myKernel -o my_report
```

- `./my_cuda_app`：待分析的 CUDA 可执行程序
- `myKernel`：要分析的 kernel 名称
- `my_report`：输出报告名前缀

## Bash 版本

```bash
chmod +x ncu_profile.sh
./ncu_profile.sh ./my_cuda_app myKernel my_report
```

- 第 1 个参数：待分析的 CUDA 可执行程序
- 第 2 个参数：要分析的 kernel 名称
- 第 3 个参数：输出报告名前缀