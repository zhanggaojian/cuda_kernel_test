# cuda_kernel_test

## compile

```bash
nvcc test.cu -o test
```

## nsys easy

```bash
nsys_easy -t cuda,osrt -s none -c none -o nsys_easy -r cuda_gpu_sum ./test
```

## ncu

### 基本使用

```bash
ncu ./test
```

### 常用选项

```bash
# 只显示 summary（不显示每个 kernel 的详细 trace）
ncu --print-summary per-gpu ./test

# 指定输出文件（生成 ncu_report.ncu-rep，可用 Nsight Compute GUI 打开）
ncu -o ncu_report ./test

# 收集指定指标（如内存带宽、计算吞吐量）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./test

# 只分析特定内核（通过 kernel 名称过滤）
cu --kernel-name regex:kernel_name ./test

# 限制分析的 kernel 实例数量（如只分析前 5 个）
ncu --launch-count 5 ./test

# 收集完整性能指标（默认是精简模式，full 模式收集数百个详细指标）
ncu --set full -o full_report ./test
```

### 常用指标参考

| 指标 | 说明 |
|------|------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 计算单元利用率 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 显存带宽利用率 |
| `l1tex__t_bytes.avg.pct_of_peak_sustained_elapsed` | L1/Tex 缓存带宽利用率 |
| `smsp__sass_thread_inst_executed_per_inst_executed` | 线程发散程度 |

### 查看报告

```bash
# 命令行方式查看 .ncu-rep 文件
ncu --import ncu_report.ncu-rep --print-summary

# 或直接用 Nsight Compute GUI 打开
ncu-ui ncu_report.ncu-rep
```
