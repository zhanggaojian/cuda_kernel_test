#!/bin/bash
# ncu_profile.sh - 简化版 NCU 性能分析脚本

EXECUTABLE=$1
KERNEL_NAME=${2:-""}
OUTPUT=${3:-"profile_report"}

if [ -z "$EXECUTABLE" ]; then
    echo "用法: $0 <可执行文件> [kernel名称] [输出文件名]"
    exit 1
fi

echo "=========================================="
echo "NVIDIA Nsight Compute 性能分析"
echo "=========================================="

# 执行 ncu 分析
if [ -n "$KERNEL_NAME" ]; then
    ncu --set full \
        --kernel-name "$KERNEL_NAME" \
        --export "$OUTPUT" \
        --force-overwrite \
        "$EXECUTABLE"
else
    ncu --set full \
        --export "$OUTPUT" \
        --force-overwrite \
        "$EXECUTABLE"
fi

echo ""
echo "=========================================="
echo "关键性能指标"
echo "=========================================="

# 显示汇总信息
ncu --import "${OUTPUT}.ncu-rep" --page summary

echo ""
echo "=========================================="
echo "优化建议"
echo "=========================================="
echo "1. 使用 'ncu-ui ${OUTPUT}.ncu-rep' 查看详细图形报告"
echo "2. 关注以下关键指标:"
echo "   - SM Throughput (计算利用率)"
echo "   - Memory Throughput (内存带宽)"
echo "   - Warp Occupancy (占用率)"
echo "   - Stall Reasons (停顿原因)"
echo "3. 根据瓶颈进行针对性优化"
